from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from app.rag_eval.dataset import GoldCase
from app.rag_eval.metrics import (
    RetrievalCaseMetrics,
    RetrievalMetricSummary,
    compute_retrieval_metrics,
    macro_average_metrics,
    normalize_ranked_evidence,
)
from app.services.knowledge import EvidenceKey, SearchResult


# 检索函数类型别名：接收 (查询文本, Top-K)，返回一批检索结果；用于把评测与具体检索实现解耦（测试里可传假检索器）
RetrievalCallback = Callable[[str, int], Iterable[SearchResult]]


# 一条去重后的检索结果：原始字段 + 证据级相关性判定 + 人工诊断信息
# SearchResult 是检索器交出的原始答案；EvaluatedRetrieval 是 evaluator 批改后、专门用于评测报告的答案记录。
@dataclass(frozen=True)
class EvaluatedRetrieval:
    rank: int                                               # 该结果在去重后的检索排名中的名次（从 1 起）
    chunk_id: int | None                                    # 知识片段在 DB 里的主键；向量命中但库里查不到时为 None
    source: str                                             # 来源文档名（如 risk-policy.md）
    source_index: int                                       # 片段在来源文档内的序号
    score: float                                            # 检索给出的相关性分数
    relevant: bool                                          # 是否为 Gold 相关证据（按 evidence_key 精确命中）
    expected_source: bool                                   # 来源文档是否命中该题 expected_sources（文档级判定，比 relevant 粗）
    matched_diagnostic_terms: tuple[str, ...]               # 正文里实际命中的诊断词，用于人工看清判对/判错的原因
    expanded_context_evidence: tuple[EvidenceKey, ...]      # 邻居扩展带进来的证据键，不属于该结果的排名身份
    preview: str                                            # 折叠空白后的正文前 160 字，供报告阅读

    # 返回当前排名结果的锚点证据键，邻居扩展不属于该身份
    @property
    def evidence_key(self) -> EvidenceKey:
        return (self.source, self.source_index)


# 一条 Gold 问题经过检索的完整评测结果：题目信息 + 检索明细 + 该题的五项指标
@dataclass(frozen=True)
class RetrievalOutcome:
    case_id: str                                            # Gold 题目的唯一 id
    query: str                                              # 该题的查询文本
    relevant_evidence_keys: tuple[EvidenceKey, ...]         # 该题的标准答案（Gold 相关证据键）
    expected_sources: tuple[str, ...]                       # 该题期望命中的来源文档名
    diagnostic_terms: tuple[str, ...]                       # 只用于诊断展示的词语，不参与指标计算
    retrieved: tuple[EvaluatedRetrieval, ...]               # 去重排序后的检索结果明细
    metrics: RetrievalCaseMetrics                           # 由该题证据算出的五项指标


# 整组 Gold 问题的评测结果：统一记录的 K + 每题结果 + 全组宏平均指标
@dataclass(frozen=True)
class RetrievalEvaluation:
    top_k: int                                              # 本次评测统一使用的 Top-K
    cases: tuple[RetrievalOutcome, ...]                     # 逐题结果
    metrics: RetrievalMetricSummary                         # 全部题目的宏平均指标（对外结论）


# 执行一条 Gold 问题的检索，并用 EvidenceKey 计算指标和生成诊断明细
def evaluate_retrieval_case(
    case: GoldCase,                                         # 一道 Gold 题（含 query、标准答案 evidence_keys 等）
    retrieve: RetrievalCallback,                            # 一个函数：吃 (查询, K)，吐一批 SearchResult
    top_k: int,
) -> RetrievalOutcome:

    raw_results = tuple(retrieve(case.query, top_k))

    ranked_evidence = normalize_ranked_evidence(
        (item.evidence_key for item in raw_results),
        top_k,
    )

    first_result_by_key: dict[EvidenceKey, SearchResult] = {}

    for item in raw_results:
        first_result_by_key.setdefault(item.evidence_key, item)                     # 只在键不存在时写入 → 保留第一次出现的

    # 字典是查找表，它的用途是"给我一个键，我告诉你对应的值"——它不表达排名。
    # 你没法问字典"第 1 名是谁"。而排名恰恰是评测的核心（MRR/NDCG 都靠名次）
    ordered_results = tuple(first_result_by_key[key] for key in ranked_evidence)

    relevant_keys = set(case.relevant_evidence_keys)

    expected_sources = set(case.expected_sources)

    evaluated = tuple(
        _evaluate_result(
            result=item,
            rank=rank,
            relevant_keys=relevant_keys,
            expected_sources=expected_sources,
            diagnostic_terms=case.diagnostic_terms,
        )
        for rank, item in enumerate(ordered_results, start=1)
    )

    metrics = compute_retrieval_metrics(
        ranked_evidence,
        case.relevant_evidence_keys,
        top_k,
    )

    return RetrievalOutcome(
        case_id=case.id,
        query=case.query,
        relevant_evidence_keys=case.relevant_evidence_keys,
        expected_sources=case.expected_sources,
        diagnostic_terms=case.diagnostic_terms,
        retrieved=evaluated,
        metrics=metrics,
    )


# 依次评测一组 Gold 问题，并对每题指标做宏平均汇总
def evaluate_retrieval_cases(
    cases: Iterable[GoldCase],
    retrieve: RetrievalCallback,
    top_k: int,
) -> RetrievalEvaluation:

    outcomes = tuple(
        evaluate_retrieval_case(case, retrieve, top_k)
        for case in cases
    )

    return RetrievalEvaluation(
        top_k=top_k,
        cases=outcomes,
        metrics=macro_average_metrics(item.metrics for item in outcomes),
    )


# 将一条检索结果转换为报告需要的证据判定、来源命中和词面诊断
def _evaluate_result(
    result: SearchResult,
    rank: int,
    relevant_keys: set[EvidenceKey],
    expected_sources: set[str],
    diagnostic_terms: tuple[str, ...],
) -> EvaluatedRetrieval:

    normalized_content = " ".join(result.content.split())

    # casefold() : 字符串方法，返回一个"大小写折叠"后的副本，专门用于不分大小写的比较。作用上类似 lower()
    folded_content = normalized_content.casefold()

    matched_terms = tuple(
        term
        for term in diagnostic_terms
        if term.casefold() in folded_content
    )

    return EvaluatedRetrieval(
        rank=rank,
        chunk_id=result.chunk_id,
        source=result.source,
        source_index=result.source_index,
        score=result.score,
        relevant=result.evidence_key in relevant_keys,
        expected_source=result.source in expected_sources,
        matched_diagnostic_terms=matched_terms,
        expanded_context_evidence=result.expanded_context_evidence,
        preview=normalized_content[:160],
    )
