import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.harness.runner import (
    HarnessContext,
    HarnessFailure,
    RAG_BM25_BASELINE_TOP_K,
    RAG_BM25_QUALITY_THRESHOLDS,
    resolve_suites,
    run_rag_harness,
    run_rag_hybrid_harness,
)
from app.models.entities import KnowledgeChunk
from app.rag_eval.dataset import (
    CorpusEvidence,
    CorpusFingerprintError,
    CorpusFingerprints,
    GoldCase,
    GoldDatasetError,
    GoldEvidence,
    compute_corpus_fingerprints,
    load_gold_dataset,
    parse_gold_dataset,
    validate_gold_corpus,
)
from app.rag_eval.evaluator import evaluate_retrieval_case, evaluate_retrieval_cases
from app.rag_eval.inventory import (
    EvidenceInventoryError,
    load_bundled_corpus_evidence,
    render_evidence_inventory,
    write_evidence_inventory,
)
from app.rag_eval.metrics import MetricInputError, compute_retrieval_metrics, macro_average_metrics
from app.rag_eval.runner import run_evaluation
from app.services.knowledge import KnowledgeService, SearchResult, replace_score, result_key
from app.services.trace import _json
from app.services.vector_store import ChromaKnowledgeStore, VectorSearchHit


class StubVectorStore:
    can_embed = True

    # 保存测试预设的向量命中结果
    def __init__(self, hits: list[VectorSearchHit]):
        self.hits = hits

    # 为测试文本返回固定的一维向量
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]

    # 按请求数量返回预设的向量命中
    def query(self, query_embedding: list[float], top_k: int) -> list[VectorSearchHit]:
        return self.hits[:top_k]


class StubKnowledgeService:
    # 保存测试预设的知识检索结果
    def __init__(self, results: list[SearchResult]):
        self.results = results

    # 按请求数量返回预设的知识检索结果
    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        return self.results[:top_k]


# 为 runner 的 Hybrid 成功路径提供可控向量存储，并记录索引与查询调用
class RunnerVectorStore:
    instances: list["RunnerVectorStore"] = []

    # 保存评测配置和调用记录，供测试证明向量链路实际执行
    def __init__(self, settings: Settings):
        self.settings = settings
        self.can_embed = True
        self.error = ""
        self.chunks: list[KnowledgeChunk] = []
        self.embed_calls: list[list[str]] = []
        self.sync_count = 0
        self.__class__.instances.append(self)

    # 为知识分块和查询返回固定向量，同时记录每次输入
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[0.1] for _ in texts]

    # 保存重建索引时收到的分块，后续查询固定返回第一块
    def sync_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        self.chunks = list(chunks)
        self.sync_count += 1
        return len(self.chunks)

    # 返回当前假索引中的分块数量
    def count(self) -> int:
        return len(self.chunks)

    # 验证假索引与数据库使用相同的分块主键
    def has_exact_chunk_ids(self, chunks: list[KnowledgeChunk]) -> bool:
        return {chunk.id for chunk in self.chunks} == {chunk.id for chunk in chunks}

    # 返回第一条向量命中，让无关键词重合的问题仍能命中 Gold 证据
    def query(self, query_embedding: list[float], top_k: int) -> list[VectorSearchHit]:
        if not self.chunks or top_k <= 0:
            return []
        chunk = self.chunks[0]
        return [
            VectorSearchHit(
                chunk_id=chunk.id,
                source=chunk.source,
                source_index=chunk.source_index,
                content=chunk.content,
                score=0.95,
            )
        ]


class EmbeddingBatchTests(unittest.TestCase):
    # 验证超过百炼单批上限的文本会稳定拆批，并保持返回向量与输入顺序一致
    def test_embedding_requests_are_split_into_batches_of_twenty(self):
        store = ChromaKnowledgeStore.__new__(ChromaKnowledgeStore)
        store.settings = Settings(
            openai_api_key="test-key",
            openai_base_url="https://example.test/v1",
            openai_embedding_model="test-embedding",
        )
        texts = [f"text-{index}" for index in range(45)]

        # 按每批输入生成可识别向量，模拟兼容 embeddings 接口
        def post(url: str, headers: dict, json: dict, timeout: float):
            rows = [
                {"index": index, "embedding": [float(text.removeprefix("text-"))]}
                for index, text in enumerate(json["input"])
            ]
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"data": rows}
            return response

        with patch("app.services.vector_store.httpx.post", side_effect=post) as request:
            embeddings = store._embed(texts)

        self.assertEqual([len(call.kwargs["json"]["input"]) for call in request.call_args_list], [20, 20, 5])
        self.assertEqual(embeddings, [[float(index)] for index in range(45)])


class CorpusFingerprintTests(unittest.TestCase):
    # 验证输入顺序不会改变语料双指纹
    def test_fingerprints_do_not_depend_on_input_order(self):
        first = CorpusEvidence("b.md", 0, "第二份内容")
        second = CorpusEvidence("a.md", 1, "第一份内容")

        forward = compute_corpus_fingerprints([first, second])
        reversed_order = compute_corpus_fingerprints([second, first])

        self.assertEqual(forward, reversed_order)
        self.assertEqual(forward.evidence_count, 2)
        self.assertEqual(
            forward.structure_fingerprint,
            "sha256:5025e3e9e5f40d150edfbeca395252b6731b007bf4b82639db61ec1e54f4578a",
        )
        self.assertEqual(
            forward.content_fingerprint,
            "sha256:07a16efa41dad48da6505b880e4ceb9ef2f035eb7d12e4b6990e90165c68b239",
        )

    # 验证只修改正文时仅内容指纹发生变化
    def test_content_change_only_changes_content_fingerprint(self):
        original = compute_corpus_fingerprints(
            [CorpusEvidence("support.md", 0, "原始内容")]
        )
        changed = compute_corpus_fingerprints(
            [CorpusEvidence("support.md", 0, "已经修改的内容")]
        )

        self.assertEqual(original.structure_fingerprint, changed.structure_fingerprint)
        self.assertNotEqual(original.content_fingerprint, changed.content_fingerprint)

    # 验证修改证据身份时结构和内容指纹都会变化
    def test_evidence_identity_change_changes_both_fingerprints(self):
        original = compute_corpus_fingerprints(
            [CorpusEvidence("support.md", 0, "相同内容")]
        )
        changed = compute_corpus_fingerprints(
            [CorpusEvidence("support.md", 1, "相同内容")]
        )

        self.assertNotEqual(original.structure_fingerprint, changed.structure_fingerprint)
        self.assertNotEqual(original.content_fingerprint, changed.content_fingerprint)

    # 验证跨平台空白差异不会改变内容指纹
    def test_content_normalization_ignores_platform_whitespace_differences(self):
        windows_text = compute_corpus_fingerprints(
            [CorpusEvidence("support.md", 0, "情绪支持\r\n需要  现实帮助")]
        )
        unix_text = compute_corpus_fingerprints(
            [CorpusEvidence("support.md", 0, "情绪支持\n需要\t现实帮助")]
        )

        self.assertEqual(windows_text.content_fingerprint, unix_text.content_fingerprint)

    # 验证重复 EvidenceKey 会被拒绝
    def test_duplicate_evidence_key_is_rejected(self):
        with self.assertRaisesRegex(
            CorpusFingerprintError,
            r"duplicate evidence key at evidence\[1\]: \('support\.md', 0\)",
        ):
            compute_corpus_fingerprints(
                [
                    CorpusEvidence("support.md", 0, "第一份内容"),
                    CorpusEvidence("support.md", 0, "重复身份的内容"),
                ]
            )

    # 验证空语料和非法证据字段会被拒绝
    def test_empty_or_invalid_corpus_evidence_is_rejected(self):
        invalid_cases = [
            ([], "corpus must contain at least one evidence record"),
            ([CorpusEvidence("", 0, "内容")], r"evidence\[0\]\.source"),
            ([CorpusEvidence("support.md", -1, "内容")], r"evidence\[0\]\.source_index"),
            ([CorpusEvidence("support.md", 0, "")], r"evidence\[0\]\.content"),
        ]

        for evidence, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(CorpusFingerprintError, message):
                    compute_corpus_fingerprints(evidence)


class GoldDatasetLoaderTests(unittest.TestCase):
    # 构造一份最小但完整的新格式 Gold Set，供各项加载器测试复用
    def valid_payload(self) -> dict:
        return {
            "schemaVersion": 1,
            "datasetVersion": "2026-09-10.1",
            "corpus": {
                "structureFingerprint": f"sha256:{'a' * 64}",
                "contentFingerprint": f"sha256:{'b' * 64}",
            },
            "labeling": {
                "method": "manual-engineering-review",
                "reviewer": "AoranQin-秦奥然",
                "notes": "工程人工复核，非领域专家标注",
            },
            "cases": [
                {
                    "id": "rag-001",
                    "query": "焦虑时应该如何获得帮助？",
                    "relevantEvidence": [
                        {
                            "source": "anxiety.md",
                            "sourceIndex": 0,
                            "note": "包含可执行的焦虑支持方法",
                        }
                    ],
                    "expectedSources": ["anxiety.md"],
                    "diagnosticTerms": ["焦虑", "支持"],
                }
            ],
        }

    # 验证合法 JSON 会被转换成不可变模型和稳定证据键
    def test_valid_dataset_is_parsed_to_typed_models(self):
        dataset = parse_gold_dataset(self.valid_payload())

        self.assertEqual(dataset.schema_version, 1)
        self.assertEqual(dataset.dataset_version, "2026-09-10.1")
        self.assertIsInstance(dataset.cases, tuple)
        self.assertEqual(dataset.cases[0].id, "rag-001")
        self.assertEqual(dataset.cases[0].relevant_evidence_keys, (("anxiety.md", 0),))
        self.assertEqual(dataset.cases[0].expected_sources, ("anxiety.md",))
        self.assertEqual(dataset.cases[0].diagnostic_terms, ("焦虑", "支持"))

    # 验证文件入口按 UTF-8 读取 JSON，并复用同一个严格解析函数
    def test_file_loader_reads_utf8_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gold.json"
            path.write_text(
                json.dumps(self.valid_payload(), ensure_ascii=False),
                encoding="utf-8",
            )

            dataset = load_gold_dataset(path)

        self.assertEqual(dataset.cases[0].query, "焦虑时应该如何获得帮助？")

    # 验证损坏的 JSON 会报告具体行列，而不是泄漏底层解析异常
    def test_file_loader_rejects_malformed_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.json"
            path.write_text('{"schemaVersion": 1,', encoding="utf-8")

            with self.assertRaisesRegex(GoldDatasetError, "JSON 格式错误.*第 1 行"):
                load_gold_dataset(path)

    # 验证文件不存在和缺少必填字段时都会返回可定位的错误
    def test_missing_file_and_required_field_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.json"
            with self.assertRaisesRegex(GoldDatasetError, "无法读取 Gold Set 文件"):
                load_gold_dataset(missing_path)

        missing_query = self.valid_payload()
        del missing_query["cases"][0]["query"]
        with self.assertRaisesRegex(GoldDatasetError, "缺少必填字段：query"):
            parse_gold_dataset(missing_query)

    # 验证不兼容版本和未知字段都会快速失败
    def test_schema_version_and_unknown_fields_are_rejected(self):
        unsupported = self.valid_payload()
        unsupported["schemaVersion"] = 2
        with self.assertRaisesRegex(GoldDatasetError, "当前仅支持 1"):
            parse_gold_dataset(unsupported)

        unexpected = self.valid_payload()
        unexpected["extraField"] = True
        with self.assertRaisesRegex(GoldDatasetError, "Gold Set 包含未知字段：extraField"):
            parse_gold_dataset(unexpected)

    # 验证 case 列表不能为空且每个 id 必须唯一
    def test_empty_cases_and_duplicate_case_ids_are_rejected(self):
        empty = self.valid_payload()
        empty["cases"] = []
        with self.assertRaisesRegex(GoldDatasetError, "cases 至少需要包含一条"):
            parse_gold_dataset(empty)

        duplicate = self.valid_payload()
        duplicate["cases"].append(copy.deepcopy(duplicate["cases"][0]))
        with self.assertRaisesRegex(GoldDatasetError, r"cases\[1\]\.id 与已有 case 重复"):
            parse_gold_dataset(duplicate)

    # 验证每个问题至少有一条正确证据，且证据键不能重复
    def test_empty_and_duplicate_relevant_evidence_are_rejected(self):
        empty = self.valid_payload()
        empty["cases"][0]["relevantEvidence"] = []
        with self.assertRaisesRegex(GoldDatasetError, "relevantEvidence 至少需要包含一条"):
            parse_gold_dataset(empty)

        duplicate = self.valid_payload()
        first_evidence = duplicate["cases"][0]["relevantEvidence"][0]
        duplicate["cases"][0]["relevantEvidence"].append(copy.deepcopy(first_evidence))
        with self.assertRaisesRegex(GoldDatasetError, "与已有证据重复"):
            parse_gold_dataset(duplicate)

    # 验证字段类型、非空约束和指纹格式都按字段路径报告错误
    def test_invalid_field_types_and_fingerprint_are_rejected(self):
        invalid_cases = []

        invalid_query = self.valid_payload()
        invalid_query["cases"][0]["query"] = ""
        invalid_cases.append((invalid_query, r"cases\[0\]\.query 必须是非空字符串"))

        invalid_diagnostics = self.valid_payload()
        invalid_diagnostics["cases"][0]["diagnosticTerms"] = "焦虑"
        invalid_cases.append((invalid_diagnostics, r"diagnosticTerms 必须是列表"))

        invalid_index = self.valid_payload()
        invalid_index["cases"][0]["relevantEvidence"][0]["sourceIndex"] = True
        invalid_cases.append((invalid_index, r"sourceIndex 必须是非负整数"))

        invalid_note = self.valid_payload()
        invalid_note["cases"][0]["relevantEvidence"][0]["note"] = None
        invalid_cases.append((invalid_note, r"note 必须是字符串"))

        invalid_fingerprint = self.valid_payload()
        invalid_fingerprint["corpus"]["contentFingerprint"] = "sha256:not-valid"
        invalid_cases.append((invalid_fingerprint, r"contentFingerprint 必须是 sha256"))

        for payload, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(GoldDatasetError, message):
                    parse_gold_dataset(payload)

    # 验证结构或内容指纹与当前知识库不一致时都会阻止继续评测
    def test_corpus_fingerprint_mismatch_is_rejected(self):
        dataset = parse_gold_dataset(self.valid_payload())
        current = CorpusFingerprints(
            evidence_count=1,
            structure_fingerprint=dataset.corpus.structure_fingerprint,
            content_fingerprint=dataset.corpus.content_fingerprint,
        )

        validate_gold_corpus(dataset, current)

        wrong_structure = replace(current, structure_fingerprint=f"sha256:{'c' * 64}")
        with self.assertRaisesRegex(GoldDatasetError, "结构指纹与当前知识库不一致"):
            validate_gold_corpus(dataset, wrong_structure)

        wrong_content = replace(current, content_fingerprint=f"sha256:{'d' * 64}")
        with self.assertRaisesRegex(GoldDatasetError, "内容指纹与当前知识库不一致"):
            validate_gold_corpus(dataset, wrong_content)

    # 验证正式 Gold Set 可加载，且每条标注都能解析到当前内置知识分块
    def test_project_gold_resolves_all_evidence_keys(self):
        project_root = Path(__file__).resolve().parents[1]
        settings = Settings()
        current_evidence = load_bundled_corpus_evidence(
            project_root / "app" / "knowledge",
            settings.knowledge_chunk_size,
            settings.knowledge_chunk_overlap,
        )
        fingerprints = compute_corpus_fingerprints(current_evidence)
        available_keys = {item.evidence_key for item in current_evidence}
        current_sources = {item.source for item in current_evidence}

        dataset = load_gold_dataset(
            project_root / "app" / "rag_eval" / "mindbridge-rag-gold-v1.json"
        )

        validate_gold_corpus(dataset, fingerprints)
        self.assertEqual(dataset.dataset_version, "2026-09-10.1")
        self.assertEqual(dataset.labeling.method, "manual-engineering-review")
        self.assertEqual(len(dataset.cases), 68)
        for case in dataset.cases:
            with self.subTest(case_id=case.id):
                self.assertTrue(set(case.relevant_evidence_keys) <= available_keys)
        labeled_sources = {
            evidence.source
            for case in dataset.cases
            for evidence in case.relevant_evidence
        }
        self.assertEqual(labeled_sources, current_sources)

    # 验证调优验证集与正式集相互独立，并覆盖当前全部知识来源和有效证据键
    def test_tuning_validation_gold_is_independent_and_covers_all_sources(self):
        project_root = Path(__file__).resolve().parents[1]
        settings = Settings()
        current_evidence = load_bundled_corpus_evidence(
            project_root / "app" / "knowledge",
            settings.knowledge_chunk_size,
            settings.knowledge_chunk_overlap,
        )
        fingerprints = compute_corpus_fingerprints(current_evidence)
        available_keys = {item.evidence_key for item in current_evidence}
        current_sources = {item.source for item in current_evidence}
        official = load_gold_dataset(
            project_root / "app" / "rag_eval" / "mindbridge-rag-gold-v1.json"
        )
        validation = load_gold_dataset(
            project_root / "app" / "rag_eval" / "mindbridge-rag-validation-v1.json"
        )

        validate_gold_corpus(validation, fingerprints)
        self.assertEqual(validation.dataset_version, "2026-09-16.validation.1")
        self.assertEqual(validation.labeling.method, "manual-engineering-review")
        self.assertEqual(len(validation.cases), 12)
        self.assertTrue(
            {case.id for case in official.cases}.isdisjoint(
                case.id for case in validation.cases
            )
        )
        self.assertTrue(
            {case.query for case in official.cases}.isdisjoint(
                case.query for case in validation.cases
            )
        )
        for case in validation.cases:
            with self.subTest(case_id=case.id):
                self.assertTrue(set(case.relevant_evidence_keys) <= available_keys)
        labeled_sources = {
            evidence.source
            for case in validation.cases
            for evidence in case.relevant_evidence
        }
        self.assertEqual(labeled_sources, current_sources)


class EvidenceInventoryTests(unittest.TestCase):
    # 验证清单生成器复用线上分块规则，并按文件名和段内序号稳定排序
    def test_bundled_markdown_is_loaded_with_production_chunking(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.md").write_text("第二份内容", encoding="utf-8")
            (root / "a.md").write_text("abcdef", encoding="utf-8")

            evidence = load_bundled_corpus_evidence(root, chunk_size=4, chunk_overlap=1)

        self.assertEqual(
            [(item.source, item.source_index, item.content) for item in evidence],
            [
                ("a.md", 0, "abcd"),
                ("a.md", 1, "def"),
                ("b.md", 0, "第二份内"),
                ("b.md", 1, "内容"),
            ],
        )

    # 验证 Markdown 清单包含双指纹、稳定证据键和明确截断的短预览
    def test_markdown_inventory_contains_traceable_snapshot(self):
        evidence = [
            CorpusEvidence("b.md", 0, "第二份内容"),
            CorpusEvidence("a.md", 1, "第一份很长的内容"),
        ]

        rendered = render_evidence_inventory(
            evidence,
            chunk_size=512,
            chunk_overlap=64,
            preview_chars=5,
        )

        self.assertIn("来源文件数：2", rendered)
        self.assertIn("证据分块数：2", rendered)
        self.assertIn("sha256:5025e3e9e5f40d150edfbeca395252b6731b007bf4b82639db61ec1e54f4578a", rendered)
        self.assertLess(rendered.index("('a.md', 1)"), rendered.index("('b.md', 0)"))
        self.assertIn("第一份很长…", rendered)

    # 验证清单可以写成 UTF-8 文件，并拒绝会导致错误分块的参数
    def test_inventory_write_and_invalid_chunking(self):
        evidence = [CorpusEvidence("support.md", 0, "支持内容")]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "inventory.md"
            written = write_evidence_inventory(
                output,
                evidence,
                chunk_size=512,
                chunk_overlap=64,
            )

            self.assertEqual(written, output)
            self.assertIn("支持内容", output.read_text(encoding="utf-8"))

        with self.assertRaisesRegex(EvidenceInventoryError, "chunk_overlap"):
            load_bundled_corpus_evidence("unused", chunk_size=4, chunk_overlap=4)


class RetrievalMetricsTests(unittest.TestCase):
    # 验证多 Gold 部分命中时 HitRate 与 Recall 保持不同含义
    def test_partial_multi_gold_hit_keeps_recall_distinct_from_hit_rate(self):
        metrics = compute_retrieval_metrics(
            ranked_evidence=[("support.md", 0)],
            relevant_evidence=[("support.md", 0), ("support.md", 1)],
            top_k=4,
        )

        self.assertEqual(metrics.hit_rate_at_k, 1.0)
        self.assertEqual(metrics.recall_at_k, 0.5)
        self.assertEqual(metrics.precision_at_k, 0.25)
        self.assertEqual(metrics.reciprocal_rank_at_k, 1.0)
        self.assertEqual(metrics.relevant_retrieved_count, 1)
        self.assertEqual(metrics.relevant_total, 2)
        self.assertEqual(metrics.first_relevant_rank, 1)
        self.assertAlmostEqual(
            metrics.ndcg_at_k,
            1.0 / (1.0 + 1.0 / math.log2(3)),
        )

    # 验证没有命中时五项指标按契约返回零值
    def test_no_hit_returns_zero_metrics(self):
        metrics = compute_retrieval_metrics(
            ranked_evidence=[("other.md", 0)],
            relevant_evidence=[("support.md", 0)],
            top_k=4,
        )

        self.assertEqual(metrics.hit_rate_at_k, 0.0)
        self.assertEqual(metrics.recall_at_k, 0.0)
        self.assertEqual(metrics.precision_at_k, 0.0)
        self.assertEqual(metrics.reciprocal_rank_at_k, 0.0)
        self.assertEqual(metrics.ndcg_at_k, 0.0)
        self.assertIsNone(metrics.first_relevant_rank)

    # 验证 NDCG 的理想排名长度取 TopK 与 Gold 数量的较小值
    def test_ndcg_uses_minimum_of_top_k_and_gold_count_for_idcg(self):
        metrics = compute_retrieval_metrics(
            ranked_evidence=[("other.md", 0), ("support.md", 1), ("support.md", 0)],
            relevant_evidence=[("support.md", 0), ("support.md", 1)],
            top_k=4,
        )

        expected_dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
        expected_idcg = 1.0 + 1.0 / math.log2(3)
        self.assertEqual(metrics.recall_at_k, 1.0)
        self.assertEqual(metrics.precision_at_k, 0.5)
        self.assertEqual(metrics.reciprocal_rank_at_k, 0.5)
        self.assertAlmostEqual(metrics.ndcg_at_k, expected_dcg / expected_idcg)

    # 验证重复检索证据在截取 TopK 前只保留第一次出现
    def test_duplicate_ranked_evidence_is_removed_before_top_k(self):
        metrics = compute_retrieval_metrics(
            ranked_evidence=[("other.md", 0), ("other.md", 0), ("support.md", 0)],
            relevant_evidence=[("support.md", 0)],
            top_k=2,
        )

        self.assertEqual(metrics.hit_rate_at_k, 1.0)
        self.assertEqual(metrics.precision_at_k, 0.5)
        self.assertEqual(metrics.first_relevant_rank, 2)
        self.assertEqual(metrics.reciprocal_rank_at_k, 0.5)

    # 验证返回不足 K 条时 Precision 分母仍固定为 K
    def test_precision_denominator_remains_top_k_when_fewer_results_returned(self):
        metrics = compute_retrieval_metrics(
            ranked_evidence=[("support.md", 0)],
            relevant_evidence=[("support.md", 0)],
            top_k=4,
        )

        self.assertEqual(metrics.precision_at_k, 0.25)

    # 验证空 Gold 或重复 Gold 会被拒绝
    def test_empty_or_duplicate_gold_is_rejected(self):
        with self.assertRaisesRegex(MetricInputError, "at least one evidence key"):
            compute_retrieval_metrics([], [], top_k=4)

        with self.assertRaisesRegex(MetricInputError, "duplicate relevant evidence at position 1"):
            compute_retrieval_metrics(
                [],
                [("support.md", 0), ("support.md", 0)],
                top_k=4,
            )

    # 验证非法 TopK 参数会被拒绝
    def test_invalid_top_k_is_rejected(self):
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(MetricInputError, "top_k must be a positive integer"):
                    compute_retrieval_metrics([], [("support.md", 0)], top_k=top_k)

    # 验证宏平均结果与各单题指标的算术平均一致
    def test_macro_average_matches_case_level_metrics(self):
        hit = compute_retrieval_metrics(
            [("support.md", 0)],
            [("support.md", 0)],
            top_k=2,
        )
        miss = compute_retrieval_metrics(
            [("other.md", 0)],
            [("support.md", 0)],
            top_k=2,
        )

        summary = macro_average_metrics([hit, miss])

        self.assertEqual(summary.case_count, 2)
        self.assertEqual(summary.hit_rate_at_k, 0.5)
        self.assertEqual(summary.recall_at_k, 0.5)
        self.assertEqual(summary.precision_at_k, 0.25)
        self.assertEqual(summary.mrr_at_k, 0.5)
        self.assertEqual(summary.ndcg_at_k, 0.5)

    # 验证空指标集合不能生成宏平均
    def test_empty_macro_average_is_rejected(self):
        with self.assertRaisesRegex(MetricInputError, "at least one case"):
            macro_average_metrics([])


class SharedEvaluatorTests(unittest.TestCase):
    # 验证正式对错只看 EvidenceKey，来源、诊断词和扩展邻居只能用于解释结果
    def test_case_evaluation_separates_scoring_from_diagnostics(self):
        case = GoldCase(
            id="support-breathing",
            query="焦虑时可以怎样呼吸？",
            relevant_evidence=(GoldEvidence("support.md", 1),),
            expected_sources=("support.md",),
            diagnostic_terms=("呼吸", "专业支持"),
        )
        service = StubKnowledgeService(
            [
                SearchResult(
                    chunk_id=10,
                    source="support.md",
                    source_index=0,
                    content="先进行缓慢呼吸。",
                    score=0.9,
                    expanded_context_evidence=(("support.md", 1),),
                ),
                SearchResult(
                    chunk_id=11,
                    source="support.md",
                    source_index=1,
                    content="持续不适时寻求专业支持。",
                    score=0.8,
                ),
            ]
        )

        outcome = evaluate_retrieval_case(case, service.retrieve, top_k=4)

        self.assertFalse(outcome.retrieved[0].relevant)
        self.assertTrue(outcome.retrieved[0].expected_source)
        self.assertEqual(outcome.retrieved[0].matched_diagnostic_terms, ("呼吸",))
        self.assertEqual(
            outcome.retrieved[0].expanded_context_evidence,
            (("support.md", 1),),
        )
        self.assertTrue(outcome.retrieved[1].relevant)
        self.assertEqual(outcome.metrics.first_relevant_rank, 2)
        self.assertEqual(outcome.metrics.precision_at_k, 0.25)

    # 验证批量评测按首次出现去重后截取 Top-K，并复用单题指标生成宏平均
    def test_suite_evaluation_normalizes_ranked_results_and_summarizes_cases(self):
        cases = (
            GoldCase("case-a", "问题 A", (GoldEvidence("a.md", 0),)),
            GoldCase("case-b", "问题 B", (GoldEvidence("b.md", 0),)),
        )
        results_by_query = {
            "问题 A": [
                SearchResult(None, "other.md", 0, "其他内容", 0.9),
                SearchResult(None, "other.md", 0, "重复内容", 0.8),
                SearchResult(None, "a.md", 0, "正确内容", 0.7),
            ],
            "问题 B": [SearchResult(None, "other.md", 1, "未命中", 0.6)],
        }

        # 按问题返回预置结果，模拟 CLI 与 Harness 都会传入的检索回调
        def retrieve(query: str, top_k: int) -> list[SearchResult]:
            return results_by_query[query]

        evaluation = evaluate_retrieval_cases(cases, retrieve, top_k=2)

        self.assertEqual(
            [item.evidence_key for item in evaluation.cases[0].retrieved],
            [("other.md", 0), ("a.md", 0)],
        )
        self.assertEqual(evaluation.cases[0].metrics.first_relevant_rank, 2)
        self.assertEqual(evaluation.metrics.case_count, 2)
        self.assertEqual(evaluation.metrics.hit_rate_at_k, 0.5)
        self.assertEqual(evaluation.metrics.precision_at_k, 0.25)
        self.assertEqual(evaluation.metrics.mrr_at_k, 0.25)


class EvaluationRunnerTests(unittest.TestCase):
    # 验证生产检索默认返回 5 个结果
    def test_default_production_top_k_is_five(self):
        self.assertEqual(Settings(_env_file=None).knowledge_top_k, 5)

    # 生成与临时知识目录严格绑定的最小 Gold Set，供 runner 集成测试使用
    def write_gold_dataset(
        self,
        root: Path,
        settings: Settings,
        knowledge_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        knowledge_dir = knowledge_dir or root / "knowledge"
        knowledge_dir.mkdir(parents=True)
        (knowledge_dir / "support.md").write_text(
            "# Support\n\nbreathing support",
            encoding="utf-8",
        )
        evidence = load_bundled_corpus_evidence(
            knowledge_dir,
            settings.knowledge_chunk_size,
            settings.knowledge_chunk_overlap,
        )
        fingerprints = compute_corpus_fingerprints(evidence)
        dataset_path = root / "gold.json"
        dataset_path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "datasetVersion": "test.1",
                    "corpus": {
                        "structureFingerprint": fingerprints.structure_fingerprint,
                        "contentFingerprint": fingerprints.content_fingerprint,
                    },
                    "labeling": {
                        "method": "engineering-test",
                        "reviewer": "test",
                        "notes": "测试数据",
                    },
                    "cases": [
                        {
                            "id": "runner-001",
                            "query": "breathing support",
                            "relevantEvidence": [
                                {"source": "support.md", "sourceIndex": 0}
                            ],
                            "diagnosticTerms": ["breathing"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return knowledge_dir, dataset_path

    # 验证 BM25 runner 会关闭向量、校验 Gold 指纹、调用共享 evaluator 并写出成功报告
    def test_bm25_run_uses_gold_evaluator_and_writes_report(self):
        settings = Settings(knowledge_vector_enabled=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            knowledge_dir, dataset_path = self.write_gold_dataset(root, settings)
            output_path = root / "reports" / "bm25.json"

            report = run_evaluation(
                mode="bm25",
                dataset_path=dataset_path,
                output_path=output_path,
                settings=settings,
                knowledge_dir=knowledge_dir,
                top_k=2,
            )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report, written)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["mode"], "bm25")
        self.assertEqual(report["datasetVersion"], "test.1")
        self.assertFalse(report["retrievalConfig"]["vectorEnabled"])
        self.assertEqual(report["metrics"]["caseCount"], 1)
        self.assertEqual(report["metrics"]["hitRateAtK"], 1.0)
        self.assertTrue(report["cases"][0]["retrieved"][0]["relevant"])
        self.assertEqual(
            report["cases"][0]["retrieved"][0]["matchedDiagnosticTerms"],
            ["breathing"],
        )

    # 验证 Hybrid 没有显式外部调用授权时写失败报告，且不会静默改跑 BM25
    def test_hybrid_without_authorization_writes_failed_report(self):
        settings = Settings(openai_api_key="test-key")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            knowledge_dir, dataset_path = self.write_gold_dataset(root, settings)
            output_path = root / "reports" / "hybrid.json"

            report = run_evaluation(
                mode="hybrid",
                dataset_path=dataset_path,
                output_path=output_path,
                settings=settings,
                knowledge_dir=knowledge_dir,
                top_k=2,
                allow_external=False,
            )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report, written)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["mode"], "hybrid")
        self.assertIsNone(report["metrics"])
        self.assertEqual(report["cases"], [])
        self.assertEqual(report["failure"]["stage"], "preflight")
        self.assertEqual(report["failure"]["code"], "external_call_not_authorized")

    # 验证 Hybrid 成功路径会重建隔离索引、执行查询向量化，并凭向量命中 Gold
    def test_hybrid_run_uses_vector_retrieval_and_isolated_chroma(self):
        settings = Settings(
            openai_api_key="test-key",
            chroma_persist_dir="data/chroma",
            chroma_snapshot_dir="data/chroma-snapshots",
        )
        RunnerVectorStore.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            knowledge_dir, dataset_path = self.write_gold_dataset(root, settings)
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            payload["cases"][0]["query"] = "请给我一种能缓和紧张的方法"
            dataset_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            output_path = root / "reports" / "hybrid.json"

            with patch("app.services.knowledge.ChromaKnowledgeStore", RunnerVectorStore):
                report = run_evaluation(
                    mode="hybrid",
                    dataset_path=dataset_path,
                    output_path=output_path,
                    settings=settings,
                    knowledge_dir=knowledge_dir,
                    top_k=2,
                    allow_external=True,
                )

            written = json.loads(output_path.read_text(encoding="utf-8"))
            vector_store = RunnerVectorStore.instances[-1]
            expected_chroma = str((output_path.parent / "chroma").resolve())
            expected_snapshots = str((output_path.parent / "chroma-snapshots").resolve())

        self.assertEqual(report, written)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["mode"], "hybrid")
        self.assertTrue(report["retrievalConfig"]["vectorEnabled"])
        self.assertTrue(report["retrievalConfig"]["vectorRequired"])
        self.assertEqual(report["retrievalConfig"]["chromaPersistDir"], expected_chroma)
        self.assertEqual(report["retrievalConfig"]["chromaSnapshotDir"], expected_snapshots)
        self.assertEqual(report["metrics"]["hitRateAtK"], 1.0)
        self.assertTrue(report["cases"][0]["retrieved"][0]["relevant"])
        self.assertEqual(vector_store.sync_count, 1)
        self.assertEqual(len(vector_store.embed_calls), 2)
        self.assertEqual(vector_store.embed_calls[-1], ["请给我一种能缓和紧张的方法"])

    # 验证 Engineering Harness 直接消费共享 BM25 报告，不再维护旧数据和旧指标公式
    def test_harness_uses_shared_bm25_evaluation_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                rag_eval_dataset="gold.json",
                knowledge_vector_enabled=True,
                knowledge_top_k=5,
            )
            self.write_gold_dataset(root, settings, root / "app" / "knowledge")
            context = HarnessContext(
                root=root,
                target_dir=root / "target" / "harness",
                settings=settings,
                database=None,
            )

            details = run_rag_harness(context)
            report = json.loads(
                (context.target_dir / "rag-eval-report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(details["mode"], "bm25")
        self.assertEqual(details["datasetVersion"], "test.1")
        self.assertEqual(details["metrics"], report["metrics"])
        self.assertEqual(details["qualityThresholds"], RAG_BM25_QUALITY_THRESHOLDS)
        self.assertFalse(details["retrievalConfig"]["vectorEnabled"])
        self.assertEqual(details["retrievalConfig"]["topK"], RAG_BM25_BASELINE_TOP_K)
        self.assertEqual(details["metrics"]["caseCount"], 1)

    # 验证共享评测指标低于新 BM25 回归下限时，RAG Harness 会明确失败
    def test_harness_rejects_bm25_metrics_below_quality_thresholds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(rag_eval_dataset="gold.json")
            _, dataset_path = self.write_gold_dataset(
                root,
                settings,
                root / "app" / "knowledge",
            )
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            payload["cases"][0]["query"] = "完全不相关的问题"
            dataset_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            context = HarnessContext(
                root=root,
                target_dir=root / "target" / "harness",
                settings=settings,
                database=None,
            )

            with self.assertRaisesRegex(
                HarnessFailure,
                "hitRateAtK 低于回归下限",
            ):
                run_rag_harness(context)

    # 验证默认 Harness 不调用外部 Hybrid，只有显式 rag-hybrid 才选择该套件
    def test_hybrid_harness_suite_is_explicit_only(self):
        default_names = [name for name, _ in resolve_suites(None)]
        hybrid_names = [name for name, _ in resolve_suites(["rag-hybrid"])]

        self.assertNotIn("RAG Hybrid Harness", default_names)
        self.assertEqual(hybrid_names, ["RAG Hybrid Harness"])

    # 验证 Hybrid Harness 使用同批 BM25 对照，并输出指标与逐题变化
    def test_hybrid_harness_compares_shared_reports(self):
        fingerprints = {
            "evidenceCount": 1,
            "structureFingerprint": "sha256:" + "1" * 64,
            "contentFingerprint": "sha256:" + "2" * 64,
        }
        bm25_metrics = {
            "caseCount": 2,
            "hitRateAtK": 0.75,
            "recallAtK": 0.70,
            "precisionAtK": 0.20,
            "mrrAtK": 0.61,
            "ndcgAtK": 0.61,
        }
        hybrid_metrics = {
            "caseCount": 2,
            "hitRateAtK": 1.0,
            "recallAtK": 0.85,
            "precisionAtK": 0.25,
            "mrrAtK": 0.75,
            "ndcgAtK": 0.76,
        }

        # 生成最小共享报告，模拟底层 runner 已完成两种真实模式
        def report(mode: str, top_k: int) -> dict:
            is_hybrid = mode == "hybrid"
            return {
                "status": "success",
                "failure": None,
                "mode": mode,
                "datasetVersion": "test.1",
                "corpusFingerprints": fingerprints,
                "retrievalConfig": {
                    "topK": top_k,
                    "vectorEnabled": is_hybrid,
                    "vectorRequired": is_hybrid,
                },
                "metrics": hybrid_metrics if is_hybrid else bm25_metrics,
                "cases": [
                    {
                        "id": "case-new-hit",
                        "metrics": {
                            "hitRateAtK": 1.0 if is_hybrid else 0.0,
                            "recallAtK": 1.0 if is_hybrid else 0.0,
                        },
                    },
                    {
                        "id": "case-stable",
                        "metrics": {"hitRateAtK": 1.0, "recallAtK": 1.0},
                    },
                ],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = HarnessContext(
                root=root,
                target_dir=root / "target" / "harness",
                settings=Settings(rag_eval_dataset="gold.json", knowledge_top_k=5),
                database=None,
            )
            with patch(
                "app.rag_eval.runner.run_evaluation",
                side_effect=lambda **kwargs: report(kwargs["mode"], kwargs["top_k"]),
            ) as run:
                details = run_rag_hybrid_harness(context)

        self.assertEqual([call.kwargs["mode"] for call in run.call_args_list], ["bm25", "hybrid"])
        self.assertEqual([call.kwargs["top_k"] for call in run.call_args_list], [5, 5])
        self.assertTrue(run.call_args_list[1].kwargs["allow_external"])
        self.assertNotIn("qualityThresholds", details["bm25"])
        self.assertEqual(details["bm25"]["retrievalConfig"]["topK"], 5)
        self.assertEqual(details["metricDelta"]["recallAtK"], 0.15)
        self.assertEqual(details["caseDelta"]["newlyHitCaseIds"], ["case-new-hit"])
        self.assertEqual(details["caseDelta"]["recallImprovedCaseIds"], ["case-new-hit"])
        self.assertEqual(details["caseDelta"]["recallRegressedCaseIds"], [])


class SearchResultEvidenceIdentityTests(unittest.TestCase):
    # 为每条证据身份测试创建隔离的内存数据库和知识服务
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        KnowledgeChunk.__table__.create(self.engine)
        self.session = sessionmaker(bind=self.engine)()
        self.settings = Settings(knowledge_vector_enabled=False)
        self.service = KnowledgeService(self.session, self.settings)

    # 关闭每条测试使用的数据库会话和引擎
    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    # 写入一条测试知识分块并返回数据库实体
    def add_chunk(self, source: str, source_index: int, content: str) -> KnowledgeChunk:
        chunk = KnowledgeChunk(source=source, source_index=source_index, content=content)
        self.session.add(chunk)
        self.session.commit()
        self.session.refresh(chunk)
        return chunk

    # 验证 BM25 结果会保留来源内序号
    def test_bm25_result_preserves_source_index(self):
        self.add_chunk("anxiety.md", 7, "焦虑时可以先做呼吸练习")

        results = self.service._retrieve_bm25("焦虑", top_k=4)

        self.assertEqual(len(results), 1)
        self.assertEqual((results[0].source, results[0].source_index), ("anxiety.md", 7))

    # 验证向量命中数据库实体时优先使用数据库中的证据身份
    def test_vector_result_prefers_database_evidence_identity(self):
        chunk = self.add_chunk("risk-policy.md", 3, "高风险需要立即联系现实支持")
        self.service.vector_store = StubVectorStore(
            [
                VectorSearchHit(
                    chunk_id=chunk.id,
                    source="stale-source.md",
                    source_index=99,
                    content="stale content",
                    score=0.9,
                )
            ]
        )
        self.service._ensure_vector_index = lambda: None

        results = self.service._retrieve_vector("高风险", top_k=4)

        self.assertEqual(len(results), 1)
        self.assertEqual((results[0].source, results[0].source_index), ("risk-policy.md", 3))
        self.assertEqual(results[0].content, "高风险需要立即联系现实支持")

    # 验证数据库缺行时向量结果仍保留命中携带的证据身份
    def test_vector_result_uses_hit_identity_when_database_row_is_missing(self):
        self.service.vector_store = StubVectorStore(
            [
                VectorSearchHit(
                    chunk_id=None,
                    source="uploaded-note.txt",
                    source_index=5,
                    content="向量库中的片段",
                    score=0.8,
                )
            ]
        )
        self.service._ensure_vector_index = lambda: None

        results = self.service._retrieve_vector("片段", top_k=4)

        self.assertEqual(len(results), 1)
        self.assertEqual((results[0].source, results[0].source_index), ("uploaded-note.txt", 5))

    # 验证邻居扩展不会改变锚点证据身份
    def test_neighbor_expansion_keeps_anchor_evidence_identity(self):
        self.add_chunk("support.md", 0, "前一个片段")
        anchor = self.add_chunk("support.md", 1, "锚点片段")
        self.add_chunk("support.md", 2, "后一个片段")
        result = SearchResult(
            chunk_id=anchor.id,
            source=anchor.source,
            source_index=anchor.source_index,
            content=anchor.content,
            score=0.7,
        )

        expanded = self.service._expand(result)

        self.assertEqual(expanded.evidence_key, ("support.md", 1))
        self.assertEqual(expanded.chunk_id, anchor.id)
        self.assertEqual(
            expanded.expanded_context_evidence,
            (("support.md", 0), ("support.md", 2)),
        )
        self.assertNotIn(expanded.evidence_key, expanded.expanded_context_evidence)
        self.assertIn("前一个片段", expanded.content)
        self.assertIn("锚点片段", expanded.content)
        self.assertIn("后一个片段", expanded.content)

    # 验证首分块扩展只记录存在的后邻居
    def test_first_chunk_expansion_records_only_the_following_neighbor(self):
        anchor = self.add_chunk("support.md", 0, "锚点片段")
        self.add_chunk("support.md", 1, "后一个片段")
        result = SearchResult(
            chunk_id=anchor.id,
            source=anchor.source,
            source_index=anchor.source_index,
            content=anchor.content,
            score=0.7,
        )

        expanded = self.service._expand(result)

        self.assertEqual(expanded.evidence_key, ("support.md", 0))
        self.assertEqual(expanded.expanded_context_evidence, (("support.md", 1),))

    # 验证分数替换和无数据库 ID 的融合键都保留证据身份
    def test_score_replacement_and_fallback_key_keep_evidence_identity(self):
        result = SearchResult(
            chunk_id=None,
            source="support.md",
            source_index=4,
            content="原始内容",
            score=0.2,
            expanded_context_evidence=(("support.md", 3), ("support.md", 5)),
        )

        replaced = replace_score(result, 0.9)

        self.assertEqual(result_key(result), ("support.md", 4))
        self.assertEqual((replaced.source, replaced.source_index), ("support.md", 4))
        self.assertEqual(replaced.content, "原始内容")
        self.assertEqual(replaced.score, 0.9)
        self.assertEqual(replaced.expanded_context_evidence, result.expanded_context_evidence)

    # 验证 Trace 会分开记录锚点与扩展上下文证据
    def test_trace_serialization_separates_anchor_and_expanded_context_evidence(self):
        result = SearchResult(
            chunk_id=11,
            source="support.md",
            source_index=2,
            content="证据内容",
            score=0.6,
            expanded_context_evidence=(("support.md", 1), ("support.md", 3)),
        )

        payload = json.loads(_json(result))

        self.assertEqual(payload["source_index"], 2)
        self.assertEqual(
            payload["expanded_context_evidence"],
            [["support.md", 1], ["support.md", 3]],
        )

if __name__ == "__main__":
    unittest.main()
