from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from app.services.knowledge import EvidenceKey


class MetricInputError(ValueError):
    """检索指标收到不合法评测输入时抛出的异常。"""


# 单条检索问题的评估结果：由 compute_retrieval_metrics 基于"排序证据 vs Gold 相关证据"算出的五项标准指标
@dataclass(frozen=True)
class RetrievalCaseMetrics:
    hit_rate_at_k: float                            # 前 K 是否至少命中一条相关（有=1.0，无=0.0），用于汇成命中率
    recall_at_k: float                              # 召回率：前 K 命中的相关数 / 该题相关总数
    precision_at_k: float                           # 精确率：前 K 命中的相关数 / K
    reciprocal_rank_at_k: float                     # 第一条相关的名次倒数（无命中=0.0），用于汇成 MRR
    ndcg_at_k: float                                # 归一化折损累计增益：相关结果按名次 1/log2(rank+1) 打折
    relevant_retrieved_count: int                   # 前 K 里实际命中的相关证据条数
    relevant_total: int                             # 该题 Gold 里一共有多少条相关证据
    first_relevant_rank: int | None                 # 第一条相关结果的名次（从 1 起）；一条都没命中则为 None


# 整组检索问题的汇总结果：把所有单条指标做算术平均（macro 平均），由 macro_average_metrics 产出
@dataclass(frozen=True)
class RetrievalMetricSummary:
    case_count: int                                 # 参与汇总的问题条数
    hit_rate_at_k: float                            # 命中率：有多少比例的问题在前 K 至少命中一条相关
    recall_at_k: float                              # 平均召回率（对每题 recall_at_k 取平均）
    precision_at_k: float                           # 平均精确率
    mrr_at_k: float                                 # 平均倒数排名 MRR（对每题 reciprocal_rank_at_k 取平均）
    ndcg_at_k: float                                # 平均 NDCG


# 根据单条问题的排序证据和 Gold 证据计算五项标准检索指标
def compute_retrieval_metrics(
    ranked_evidence: Iterable[EvidenceKey],
    relevant_evidence: Iterable[EvidenceKey],
    top_k: int,
) -> RetrievalCaseMetrics:
    ranked = normalize_ranked_evidence(ranked_evidence, top_k)
    gold = _validated_gold(relevant_evidence)

    relevance_by_rank = [evidence_key in gold for evidence_key in ranked]               # 取交集呗，返回的bool列表

    relevant_retrieved_count = sum(relevance_by_rank)                                   # 前 K 里实际命中的相关证据条数

    # next() 从生成器里取第一个值；None 是兜底默认值
    first_relevant_rank = next(

        # 把列表变成 (名次, 布尔值) 的配对流，名次从 1 开始，而不是从0
        (rank for rank, relevant in enumerate(relevance_by_rank, start=1) if relevant),
        None,
    )

    # 实际排序的折损累计
    # math.fsum：累加（比 sum 浮点误差更小）。
    dcg = math.fsum(
        1.0 / math.log2(rank + 1)
        for rank, relevant in enumerate(relevance_by_rank, start=1)
        if relevant
    )

    ideal_relevant_count = min(top_k, len(gold))

    # 理想排序的折损累计
    # 假设相关项全被排在最前面（占满第 1 到 N 名），N = min(top_k, len(gold))，累加同样的折扣项
    idcg = math.fsum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return RetrievalCaseMetrics(
        hit_rate_at_k=1.0 if relevant_retrieved_count else 0.0,
        recall_at_k=relevant_retrieved_count / len(gold),
        precision_at_k=relevant_retrieved_count / top_k,
        reciprocal_rank_at_k=0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        ndcg_at_k=dcg / idcg,
        relevant_retrieved_count=relevant_retrieved_count,
        relevant_total=len(gold),
        first_relevant_rank=first_relevant_rank,
    )


# 按首次出现顺序去重并截取 Top-K，供指标与 evaluator 共用同一份排名口径
def normalize_ranked_evidence(
    ranked_evidence: Iterable[EvidenceKey],
    top_k: int,
) -> tuple[EvidenceKey, ...]:
    validate_top_k(top_k)
    return tuple(_unique_ranked_evidence(ranked_evidence)[:top_k])


# 对所有问题的单条指标做算术平均，生成整套评测的汇总指标
def macro_average_metrics(metrics: Iterable[RetrievalCaseMetrics]) -> RetrievalMetricSummary:

    cases = tuple(metrics)

    if not cases:
        raise MetricInputError("metrics must contain at least one case")

    case_count = len(cases)

    return RetrievalMetricSummary(
        case_count=case_count,
        hit_rate_at_k=math.fsum(item.hit_rate_at_k for item in cases) / case_count,
        recall_at_k=math.fsum(item.recall_at_k for item in cases) / case_count,
        precision_at_k=math.fsum(item.precision_at_k for item in cases) / case_count,
        mrr_at_k=math.fsum(item.reciprocal_rank_at_k for item in cases) / case_count,
        ndcg_at_k=math.fsum(item.ndcg_at_k for item in cases) / case_count,
    )


# 校验 Top-K 必须是大于零的整数，并显式排除布尔值
def validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise MetricInputError("top_k must be a positive integer")


# 校验 Gold 证据非空且不重复，并转换成便于快速查询的集合
def _validated_gold(relevant_evidence: Iterable[EvidenceKey]) -> set[EvidenceKey]:
    gold: set[EvidenceKey] = set()

    for position, evidence_key in enumerate(relevant_evidence):
        if evidence_key in gold:
            raise MetricInputError(
                f"duplicate relevant evidence at position {position}: {evidence_key!r}"
            )
        gold.add(evidence_key)
    if not gold:
        raise MetricInputError("relevant_evidence must contain at least one evidence key")
    return gold


# 按首次出现顺序去除重复检索证据，避免同一证据重复参与指标
def _unique_ranked_evidence(ranked_evidence: Iterable[EvidenceKey]) -> list[EvidenceKey]:

    # set 不负责保留检索排名。这样我们就无法知道谁是第一名，而 MRR、NDCG 都依赖排名
    unique: list[EvidenceKey] = []

    seen: set[EvidenceKey] = set()

    for evidence_key in ranked_evidence:
        if evidence_key in seen:
            continue
        seen.add(evidence_key)
        unique.append(evidence_key)
    return unique
