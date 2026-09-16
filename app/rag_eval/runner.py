from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.models.entities import KnowledgeChunk
from app.rag_eval.dataset import (
    CorpusEvidence,
    CorpusFingerprintError,
    CorpusFingerprints,
    GoldDataset,
    GoldDatasetError,
    compute_corpus_fingerprints,
    load_gold_dataset,
    validate_gold_corpus,
)
from app.rag_eval.evaluator import RetrievalEvaluation, RetrievalOutcome, evaluate_retrieval_cases
from app.rag_eval.inventory import EvidenceInventoryError, load_bundled_corpus_evidence
from app.rag_eval.metrics import (
    MetricInputError,
    RetrievalCaseMetrics,
    RetrievalMetricSummary,
    validate_top_k,
)
from app.services.knowledge import KnowledgeService


REPORT_SCHEMA_VERSION = 1
FIXED_RANDOM_SEED = 0


class EvaluationRunError(RuntimeError):
    """评测模式或运行先决条件不满足时抛出。"""

    # 保存失败阶段与稳定错误码，供失败报告直接使用
    def __init__(self, stage: str, code: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.code = code


# 运行指定模式的完整评测，成功或预期失败都会写出结构稳定的 JSON 报告
def run_evaluation(
    mode: str,
    dataset_path: str | Path,
    output_path: str | Path,
    settings: Settings,
    knowledge_dir: str | Path,
    top_k: int,
    allow_external: bool = False,
) -> dict[str, object]:
    started_at = _utc_now()
    dataset: GoldDataset | None = None
    fingerprints: CorpusFingerprints | None = None
    report_settings = _settings_for_report(mode, settings, output_path)

    try:
        try:
            validate_top_k(top_k)
        except MetricInputError as exc:
            raise EvaluationRunError("preflight", "invalid_top_k", str(exc)) from exc
        dataset = load_gold_dataset(dataset_path)
        corpus_evidence = load_bundled_corpus_evidence(
            knowledge_dir,
            settings.knowledge_chunk_size,
            settings.knowledge_chunk_overlap,
        )
        fingerprints = compute_corpus_fingerprints(corpus_evidence)
        validate_gold_corpus(dataset, fingerprints)
        effective_settings = _settings_for_mode(mode, settings, allow_external, output_path)
        report_settings = effective_settings
        evaluation = _evaluate_corpus(
            mode,
            dataset,
            corpus_evidence,
            effective_settings,
            top_k,
        )
        report = _success_report(
            started_at,
            mode,
            dataset_path,
            dataset,
            fingerprints,
            effective_settings,
            evaluation,
        )
    except EvaluationRunError as exc:
        report = _failure_report(
            started_at,
            mode,
            dataset_path,
            dataset,
            fingerprints,
            report_settings,
            top_k,
            exc.stage,
            exc.code,
            str(exc),
        )
    except GoldDatasetError as exc:
        report = _failure_report(
            started_at,
            mode,
            dataset_path,
            dataset,
            fingerprints,
            report_settings,
            top_k,
            "dataset",
            "gold_dataset_invalid",
            str(exc),
        )
    except (CorpusFingerprintError, EvidenceInventoryError) as exc:
        report = _failure_report(
            started_at,
            mode,
            dataset_path,
            dataset,
            fingerprints,
            report_settings,
            top_k,
            "corpus",
            "corpus_invalid",
            str(exc),
        )
    except Exception as exc:
        report = _failure_report(
            started_at,
            mode,
            dataset_path,
            dataset,
            fingerprints,
            report_settings,
            top_k,
            "evaluation",
            "evaluation_failed",
            f"{type(exc).__name__}: {exc}",
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


# 根据 BM25 或 Hybrid 模式生成明确的运行配置，Hybrid 额外校验外部调用授权和凭据
def _settings_for_mode(
    mode: str,
    settings: Settings,
    allow_external: bool,
    output_path: str | Path,
) -> Settings:
    if mode not in {"bm25", "hybrid"}:
        raise EvaluationRunError("preflight", "unsupported_mode", f"不支持的评测模式：{mode}")
    if mode == "hybrid":
        if not allow_external:
            raise EvaluationRunError(
                "preflight",
                "external_call_not_authorized",
                "Hybrid 评测需要显式传入 --allow-external，且可能产生外部模型调用费用",
            )
        if not settings.openai_api_key:
            raise EvaluationRunError(
                "preflight",
                "embedding_credentials_missing",
                "Hybrid 评测缺少 OPENAI_API_KEY",
            )
    return _settings_for_report(mode, settings, output_path)


# 为报告提前映射模式开关和隔离路径，不触发凭据检查或任何外部调用
def _settings_for_report(mode: str, settings: Settings, output_path: str | Path) -> Settings:
    if mode == "bm25":
        return settings.model_copy(
            update={
                "database_url": "sqlite:///:memory:",
                "knowledge_vector_enabled": False,
                "knowledge_vector_required": False,
            }
        )
    if mode == "hybrid":
        report_dir = Path(output_path).resolve().parent
        return settings.model_copy(
            update={
                "database_url": "sqlite:///:memory:",
                "knowledge_vector_enabled": True,
                "knowledge_vector_required": True,
                "chroma_persist_dir": str(report_dir / "chroma"),
                "chroma_snapshot_dir": str(report_dir / "chroma-snapshots"),
            }
        )
    return settings


# 把当前知识快照装入隔离 SQLite，并将生产检索结果交给共享 evaluator 判卷
def _evaluate_corpus(
    mode: str,
    dataset: GoldDataset,
    corpus_evidence: Iterable[CorpusEvidence],
    settings: Settings,
    top_k: int,
) -> RetrievalEvaluation:
    random.seed(FIXED_RANDOM_SEED)
    engine = create_engine("sqlite:///:memory:")
    KnowledgeChunk.__table__.create(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = session_factory()
    try:
        db.add_all(
            KnowledgeChunk(
                source=item.source,
                source_index=item.source_index,
                content=item.content,
            )
            for item in corpus_evidence
        )
        db.commit()
        service = KnowledgeService(db, settings)
        if mode == "hybrid":
            service.rebuild_vector_index()
        return evaluate_retrieval_cases(dataset.cases, service.retrieve, top_k)
    finally:
        db.close()
        engine.dispose()


# 生成成功报告，记录版本、语料指纹、运行配置、汇总指标和全部逐题结果
def _success_report(
    started_at: str,
    mode: str,
    dataset_path: str | Path,
    dataset: GoldDataset,
    fingerprints: CorpusFingerprints,
    settings: Settings,
    evaluation: RetrievalEvaluation,
) -> dict[str, object]:
    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "runId": uuid4().hex,
        "startedAt": started_at,
        "finishedAt": _utc_now(),
        "status": "success",
        "mode": mode,
        "codeRevision": _code_revision(settings.project_root),
        "datasetPath": str(Path(dataset_path)),
        "datasetVersion": dataset.dataset_version,
        "corpusFingerprints": _fingerprint_payload(fingerprints),
        "retrievalConfig": _retrieval_config(settings, evaluation.top_k),
        "metrics": _summary_metrics_payload(evaluation.metrics),
        "cases": [_case_payload(item) for item in evaluation.cases],
        "failure": None,
    }


# 生成失败报告，指标固定为空并保留稳定的失败阶段、错误码和可读原因
def _failure_report(
    started_at: str,
    mode: str,
    dataset_path: str | Path,
    dataset: GoldDataset | None,
    fingerprints: CorpusFingerprints | None,
    settings: Settings,
    top_k: int,
    stage: str,
    code: str,
    message: str,
) -> dict[str, object]:
    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "runId": uuid4().hex,
        "startedAt": started_at,
        "finishedAt": _utc_now(),
        "status": "failed",
        "mode": mode,
        "codeRevision": _code_revision(settings.project_root),
        "datasetPath": str(Path(dataset_path)),
        "datasetVersion": None if dataset is None else dataset.dataset_version,
        "corpusFingerprints": None if fingerprints is None else _fingerprint_payload(fingerprints),
        "retrievalConfig": _retrieval_config(settings, top_k),
        "metrics": None,
        "cases": [],
        "failure": {"stage": stage, "code": code, "message": message},
    }


# 将知识库指纹对象转换成报告中的稳定字段结构
def _fingerprint_payload(fingerprints: CorpusFingerprints) -> dict[str, object]:
    return {
        "evidenceCount": fingerprints.evidence_count,
        "structureFingerprint": fingerprints.structure_fingerprint,
        "contentFingerprint": fingerprints.content_fingerprint,
    }


# 将实际运行配置转换成不包含密钥的报告字段
def _retrieval_config(settings: Settings, top_k: int) -> dict[str, object]:
    return {
        "topK": top_k,
        "candidateK": settings.knowledge_candidate_k,
        "chunkSize": settings.knowledge_chunk_size,
        "chunkOverlap": settings.knowledge_chunk_overlap,
        "randomSeed": FIXED_RANDOM_SEED,
        "database": "sqlite-memory:bundled-corpus",
        "vectorEnabled": settings.knowledge_vector_enabled,
        "vectorRequired": settings.knowledge_vector_required,
        "rerankEnabled": settings.knowledge_rerank_enabled,
        "hybridVectorWeight": settings.knowledge_hybrid_vector_weight,
        "hybridBm25Weight": settings.knowledge_hybrid_bm25_weight,
        "embeddingModel": settings.openai_embedding_model,
        "chromaPersistDir": settings.chroma_persist_dir,
        "chromaCollectionName": settings.chroma_collection_name,
        "chromaSnapshotDir": settings.chroma_snapshot_dir,
    }


# 将整组宏平均指标转换成报告字段
def _summary_metrics_payload(metrics: RetrievalMetricSummary) -> dict[str, object]:
    return {
        "caseCount": metrics.case_count,
        "hitRateAtK": metrics.hit_rate_at_k,
        "recallAtK": metrics.recall_at_k,
        "precisionAtK": metrics.precision_at_k,
        "mrrAtK": metrics.mrr_at_k,
        "ndcgAtK": metrics.ndcg_at_k,
    }


# 将单题指标转换成报告字段，并保留命中数量与首个相关排名用于复核
def _case_metrics_payload(metrics: RetrievalCaseMetrics) -> dict[str, object]:
    return {
        "hitRateAtK": metrics.hit_rate_at_k,
        "recallAtK": metrics.recall_at_k,
        "precisionAtK": metrics.precision_at_k,
        "reciprocalRankAtK": metrics.reciprocal_rank_at_k,
        "ndcgAtK": metrics.ndcg_at_k,
        "relevantRetrievedCount": metrics.relevant_retrieved_count,
        "relevantTotal": metrics.relevant_total,
        "firstRelevantRank": metrics.first_relevant_rank,
    }


# 将 evaluator 的单题结果转换成可供 CLI 与 Harness 共同消费的 JSON 结构
def _case_payload(outcome: RetrievalOutcome) -> dict[str, object]:
    return {
        "id": outcome.case_id,
        "query": outcome.query,
        "relevantEvidence": [
            {"source": source, "sourceIndex": source_index}
            for source, source_index in outcome.relevant_evidence_keys
        ],
        "expectedSources": list(outcome.expected_sources),
        "diagnosticTerms": list(outcome.diagnostic_terms),
        "metrics": _case_metrics_payload(outcome.metrics),
        "retrieved": [
            {
                "rank": item.rank,
                "chunkId": item.chunk_id,
                "source": item.source,
                "sourceIndex": item.source_index,
                "score": item.score,
                "relevant": item.relevant,
                "expectedSource": item.expected_source,
                "matchedDiagnosticTerms": list(item.matched_diagnostic_terms),
                "expandedContextEvidence": [
                    {"source": source, "sourceIndex": source_index}
                    for source, source_index in item.expanded_context_evidence
                ],
                "preview": item.preview,
            }
            for item in outcome.retrieved
        ],
    }


# 读取当前 Git 提交并标记未提交改动，使报告能够对应到实际代码状态
def _code_revision(project_root: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    if revision.returncode != 0 or not revision.stdout.strip():
        return "unknown"
    suffix = "+dirty" if dirty.returncode == 0 and dirty.stdout.strip() else ""
    return f"{revision.stdout.strip()}{suffix}"


# 返回带 UTC 时区的 ISO 8601 时间字符串
def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# 解析 RAG 评测命令行参数，模式必须由调用者明确指定
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 MindBridge RAG 证据级评测")
    parser.add_argument("--mode", choices=("bm25", "hybrid"), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="允许 Hybrid 评测调用外部 embedding 服务并产生费用",
    )
    return parser.parse_args(argv)


# 解析默认路径并执行 CLI 评测，成功返回 0，失败报告返回 1
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = Settings()
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = args.dataset or project_root / settings.rag_eval_dataset
    output_path = args.output or project_root / "target" / "rag" / args.mode / "rag-eval-report.json"
    top_k = settings.knowledge_top_k if args.top_k is None else args.top_k
    report = run_evaluation(
        mode=args.mode,
        dataset_path=dataset_path,
        output_path=output_path,
        settings=settings,
        knowledge_dir=project_root / "app" / "knowledge",
        top_k=top_k,
        allow_external=args.allow_external,
    )
    print(f"RAG 评测状态：{report['status']}")
    print(f"模式：{report['mode']}")
    print(f"报告：{output_path}")
    if report["failure"] is not None:
        print(f"失败原因：{report['failure']['message']}")
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
