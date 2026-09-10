from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.services.knowledge import EvidenceKey


class CorpusFingerprintError(ValueError):
    """知识库证据（片段集合）无法产出可信指纹时抛出。"""


class GoldDatasetError(ValueError):
    """Gold Set 文件或字段不符合评测数据契约时抛出。"""


SUPPORTED_GOLD_SCHEMA_VERSION = 1


# 一份知识库证据：来源文档名 + 段内序号 + 原文正文，是指纹计算的最小输入单元
@dataclass(frozen=True)
class CorpusEvidence:
    source: str
    source_index: int
    content: str

    # 该证据的稳定身份键（来源文档名 + 段内序号），作为排序/去重/指纹的锚点
    @property
    def evidence_key(self) -> EvidenceKey:
        return (self.source, self.source_index)


# 一次知识库指纹计算的结果容器：给定一批知识片段证据，产出"条数 + 两个确定性摘要"。
@dataclass(frozen=True)
class CorpusFingerprints:
    evidence_count: int                             # 知识库中一共有多少个证据分块 - 用于快速检查明显变化
    structure_fingerprint: str                      # 知识库“目录结构”的指纹 - 来源文档 + 分块编号
    content_fingerprint: str                        # 知识库“实际文字内容”的指纹 - 来源文档 + 分块编号 + 分块正文


# Gold Set 中的一条正确证据标注，note 只解释标注理由，不参与指标计算
@dataclass(frozen=True)
class GoldEvidence:
    source: str
    source_index: int
    note: str | None = None

    # 返回该标注对应的稳定证据键，供检索结果与 Gold Set 精确比较
    @property
    def evidence_key(self) -> EvidenceKey:
        return (self.source, self.source_index)


# Gold Set 中的单条评测问题及其全部正确证据和诊断字段
@dataclass(frozen=True)
class GoldCase:
    id: str
    query: str
    relevant_evidence: tuple[GoldEvidence, ...]
    expected_sources: tuple[str, ...] = ()
    diagnostic_terms: tuple[str, ...] = ()

    # 提取当前问题的全部正确证据键，直接交给标准指标函数计算
    @property
    def relevant_evidence_keys(self) -> tuple[EvidenceKey, ...]:
        return tuple(item.evidence_key for item in self.relevant_evidence)


# Gold Set 绑定的知识库双指纹，用于阻止旧标注评测新版知识库
@dataclass(frozen=True)
class GoldCorpus:
    structure_fingerprint: str
    content_fingerprint: str


# 记录 Gold Set 的标注方法、复核人和限制说明
@dataclass(frozen=True)
class GoldLabeling:
    method: str
    reviewer: str
    notes: str


# 严格校验后的完整 Gold Set，只保存不可变的数据结构
@dataclass(frozen=True)
class GoldDataset:
    schema_version: int
    dataset_version: str
    corpus: GoldCorpus
    labeling: GoldLabeling
    cases: tuple[GoldCase, ...]


# 从 UTF-8 JSON 文件读取并解析版本化 Gold Set
def load_gold_dataset(path: str | Path) -> GoldDataset:
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GoldDatasetError(f"无法读取 Gold Set 文件 {file_path}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GoldDatasetError(
            f"Gold Set JSON 格式错误：第 {exc.lineno} 行，第 {exc.colno} 列"
        ) from exc
    return parse_gold_dataset(payload)


# 校验松散 JSON 数据并转换成不可变的 GoldDataset 对象
def parse_gold_dataset(payload: object) -> GoldDataset:
    root = _require_object(payload, "Gold Set")
    _require_fields(
        root,
        required={"schemaVersion", "datasetVersion", "corpus", "labeling", "cases"},
        optional=set(),
        path="Gold Set",
    )

    schema_version = _require_non_negative_integer(root["schemaVersion"], "schemaVersion")
    if schema_version != SUPPORTED_GOLD_SCHEMA_VERSION:
        raise GoldDatasetError(
            f"不支持 schemaVersion={schema_version}，当前仅支持 {SUPPORTED_GOLD_SCHEMA_VERSION}"
        )
    dataset_version = _require_non_empty_string(root["datasetVersion"], "datasetVersion")
    corpus = _parse_gold_corpus(root["corpus"])
    labeling = _parse_gold_labeling(root["labeling"])

    raw_cases = _require_list(root["cases"], "cases")
    if not raw_cases:
        raise GoldDatasetError("cases 至少需要包含一条评测问题")

    cases: list[GoldCase] = []
    seen_case_ids: set[str] = set()
    for position, raw_case in enumerate(raw_cases):
        case = _parse_gold_case(raw_case, position)
        if case.id in seen_case_ids:
            raise GoldDatasetError(f"cases[{position}].id 与已有 case 重复：{case.id!r}")
        seen_case_ids.add(case.id)
        cases.append(case)

    return GoldDataset(
        schema_version=schema_version,
        dataset_version=dataset_version,
        corpus=corpus,
        labeling=labeling,
        cases=tuple(cases),
    )


# 比较 Gold Set 与当前知识库的双指纹，任一不一致都停止评测
def validate_gold_corpus(dataset: GoldDataset, current: CorpusFingerprints) -> None:
    if dataset.corpus.structure_fingerprint != current.structure_fingerprint:
        raise GoldDatasetError(
            "Gold Set 结构指纹与当前知识库不一致："
            f"期望 {dataset.corpus.structure_fingerprint}，实际 {current.structure_fingerprint}"
        )
    if dataset.corpus.content_fingerprint != current.content_fingerprint:
        raise GoldDatasetError(
            "Gold Set 内容指纹与当前知识库不一致："
            f"期望 {dataset.corpus.content_fingerprint}，实际 {current.content_fingerprint}"
        )


# 解析并校验 Gold Set 中绑定的知识库双指纹
def _parse_gold_corpus(value: object) -> GoldCorpus:
    corpus = _require_object(value, "corpus")
    _require_fields(
        corpus,
        required={"structureFingerprint", "contentFingerprint"},
        optional=set(),
        path="corpus",
    )
    return GoldCorpus(
        structure_fingerprint=_require_sha256(
            corpus["structureFingerprint"],
            "corpus.structureFingerprint",
        ),
        content_fingerprint=_require_sha256(
            corpus["contentFingerprint"],
            "corpus.contentFingerprint",
        ),
    )


# 解析并校验 Gold Set 的标注方法、复核人和限制说明
def _parse_gold_labeling(value: object) -> GoldLabeling:
    labeling = _require_object(value, "labeling")
    _require_fields(
        labeling,
        required={"method", "reviewer", "notes"},
        optional=set(),
        path="labeling",
    )
    return GoldLabeling(
        method=_require_non_empty_string(labeling["method"], "labeling.method"),
        reviewer=_require_non_empty_string(labeling["reviewer"], "labeling.reviewer"),
        notes=_require_string(labeling["notes"], "labeling.notes"),
    )


# 解析并校验一条评测问题及其正确证据和诊断字段
def _parse_gold_case(value: object, position: int) -> GoldCase:
    path = f"cases[{position}]"
    case = _require_object(value, path)
    _require_fields(
        case,
        required={"id", "query", "relevantEvidence"},
        optional={"expectedSources", "diagnosticTerms"},
        path=path,
    )

    raw_evidence = _require_list(case["relevantEvidence"], f"{path}.relevantEvidence")
    if not raw_evidence:
        raise GoldDatasetError(f"{path}.relevantEvidence 至少需要包含一条正确证据")

    relevant_evidence: list[GoldEvidence] = []
    seen_evidence_keys: set[EvidenceKey] = set()
    for evidence_position, raw_item in enumerate(raw_evidence):
        evidence = _parse_gold_evidence(raw_item, position, evidence_position)
        if evidence.evidence_key in seen_evidence_keys:
            raise GoldDatasetError(
                f"{path}.relevantEvidence[{evidence_position}] 与已有证据重复："
                f"{evidence.evidence_key!r}"
            )
        seen_evidence_keys.add(evidence.evidence_key)
        relevant_evidence.append(evidence)

    expected_sources = _require_string_list(
        case.get("expectedSources", []),
        f"{path}.expectedSources",
    )
    diagnostic_terms = _require_string_list(
        case.get("diagnosticTerms", []),
        f"{path}.diagnosticTerms",
    )
    return GoldCase(
        id=_require_non_empty_string(case["id"], f"{path}.id"),
        query=_require_non_empty_string(case["query"], f"{path}.query"),
        relevant_evidence=tuple(relevant_evidence),
        expected_sources=expected_sources,
        diagnostic_terms=diagnostic_terms,
    )


# 解析并校验一条正确证据的来源、来源内序号和可选说明
def _parse_gold_evidence(value: object, case_position: int, evidence_position: int) -> GoldEvidence:
    path = f"cases[{case_position}].relevantEvidence[{evidence_position}]"
    evidence = _require_object(value, path)
    _require_fields(
        evidence,
        required={"source", "sourceIndex"},
        optional={"note"},
        path=path,
    )
    note = None
    if "note" in evidence:
        note = _require_string(evidence["note"], f"{path}.note")
    return GoldEvidence(
        source=_require_non_empty_string(evidence["source"], f"{path}.source"),
        source_index=_require_non_negative_integer(evidence["sourceIndex"], f"{path}.sourceIndex"),
        note=note,
    )


# 校验 JSON 节点必须是字段名为字符串的对象
def _require_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GoldDatasetError(f"{path} 必须是对象")
    if not all(isinstance(key, str) for key in value):
        raise GoldDatasetError(f"{path} 的字段名必须是字符串")
    return value


# 校验对象的必填字段、可选字段以及是否出现未知字段
def _require_fields(
    value: dict[str, object],
    required: set[str],
    optional: set[str],
    path: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    if missing:
        raise GoldDatasetError(f"{path} 缺少必填字段：{', '.join(missing)}")
    unexpected = sorted(keys - required - optional)
    if unexpected:
        raise GoldDatasetError(f"{path} 包含未知字段：{', '.join(unexpected)}")


# 校验 JSON 节点必须是列表
def _require_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise GoldDatasetError(f"{path} 必须是列表")
    return value


# 校验字段必须是非空字符串并返回原值
def _require_non_empty_string(value: object, path: str) -> str:
    text = _require_string(value, path)
    if not text.strip():
        raise GoldDatasetError(f"{path} 必须是非空字符串")
    return text


# 校验字段必须是字符串并返回原值
def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise GoldDatasetError(f"{path} 必须是字符串")
    return value


# 校验字段必须是由非空字符串组成的列表并转换为不可变元组
def _require_string_list(value: object, path: str) -> tuple[str, ...]:
    items = _require_list(value, path)
    return tuple(
        _require_non_empty_string(item, f"{path}[{position}]")
        for position, item in enumerate(items)
    )


# 校验字段必须是非负整数，并显式排除属于 int 子类的布尔值
def _require_non_negative_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GoldDatasetError(f"{path} 必须是非负整数")
    return value


# 校验字段必须是带 sha256 前缀的 64 位小写十六进制摘要
def _require_sha256(value: object, path: str) -> str:
    fingerprint = _require_string(value, path)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint) is None:
        raise GoldDatasetError(f"{path} 必须是 sha256: 加 64 位小写十六进制字符")
    return fingerprint


# 对给定的一批知识库证据，计算其"构成结构"与"归一化正文"两个确定性指纹，供知识库比对/漂移检测用
def compute_corpus_fingerprints(evidence: Iterable[CorpusEvidence]) -> CorpusFingerprints:
    records = list(evidence)
    if not records:
        raise CorpusFingerprintError("corpus must contain at least one evidence record")

    seen_keys: set[EvidenceKey] = set()

    for position, record in enumerate(records):

        _validate_evidence(record, position)

        # 如果批里同一个 (source, source_index) 出现了两条内容
        # 那这批证据本身就是自相矛盾的——算出来的指纹不可信
        if record.evidence_key in seen_keys:
            source, source_index = record.evidence_key
            raise CorpusFingerprintError(
                f"duplicate evidence key at evidence[{position}]: ({source!r}, {source_index})"
            )
        seen_keys.add(record.evidence_key)

    # 能执行到 sorted(records, ...) 时，就代表：records 已验证 = 字段合法 + 没有重复 evidence_key
    # 哈希对输入顺序敏感：同批证据如果 A、B 两次喂的顺序不一样，直接哈希会得到两个不同指纹
    ordered = sorted(records, key=lambda item: item.evidence_key)

    structure_payload = [
        {"source": item.source, "sourceIndex": item.source_index}
        for item in ordered
    ]
    content_payload = [
        {
            "source": item.source,
            "sourceIndex": item.source_index,
            "normalizedContent": normalize_corpus_content(item.content),
        }
        for item in ordered
    ]

    return CorpusFingerprints(
        evidence_count=len(ordered),
        structure_fingerprint=_sha256_json(structure_payload),
        content_fingerprint=_sha256_json(content_payload),
    )


# 归一化正文的 Unicode 与空白：让纯平台差异（如换行符、多余空格）不会改变指纹
def normalize_corpus_content(content: str) -> str:
    normalized = unicodedata.normalize("NFC", content)
    return re.sub(r"\s+", " ", normalized).strip()


# 校验单条证据字段的合法性：类型/来源名/序号/正文都必须合法非空，任一不合法抛 CorpusFingerprintError
def _validate_evidence(record: CorpusEvidence, position: int) -> None:
    if not isinstance(record, CorpusEvidence):
        raise CorpusFingerprintError(f"evidence[{position}] must be a CorpusEvidence instance")

    if not isinstance(record.source, str) or not record.source.strip():
        raise CorpusFingerprintError(f"evidence[{position}].source must be a non-empty string")

    if isinstance(record.source_index, bool) or not isinstance(record.source_index, int) or record.source_index < 0:
        raise CorpusFingerprintError(f"evidence[{position}].source_index must be a non-negative integer")

    if not isinstance(record.content, str) or not record.content.strip():
        raise CorpusFingerprintError(f"evidence[{position}].content must be a non-empty string")


# 把任意 JSON 可序列化对象转成确定性的 sha256 指纹字符串（键排序 + 紧凑分隔，同内容必得同哈希）
def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
