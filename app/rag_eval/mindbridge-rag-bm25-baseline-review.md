# MindBridge BM25 基线复核

## 复核对象

- 日期：2026-09-10
- 数据集：`mindbridge-rag-gold-v1.json`，版本 `2026-09-10.1`
- 语料：11 个来源、34 个证据分块
- 检索配置：BM25、TopK=4、候选集 16、开启本地重排、关闭向量检索
- 报告：`target/rag/bm25/rag-eval-report.json`

独立评测入口与 RAG Harness 已重复得到相同的逐题结果、汇总指标、语料指纹和检索配置，因此这组结果可作为确定性回归基线。

| 指标 | 可复现基线 | Harness 回归下限 |
| --- | ---: | ---: |
| HitRate@4 | 0.750000 | 0.74 |
| Recall@4 | 0.698529 | 0.69 |
| Precision@4 | 0.202206 | 0.20 |
| MRR@4 | 0.615196 | 0.60 |
| NDCG@4 | 0.616197 | 0.60 |

这些下限只用于发现相对当前基线的退步，不代表产品质量目标，也不能代表 Hybrid、最终回答质量或线上用户体验。

## 失败案例复核

68 个问题中，44 个完整召回 Gold，7 个部分召回，17 个没有命中 Gold，共有 24 个问题的 Recall@4 小于 1。

### 证据身份与语料

- 没有发现 EvidenceKey 映射错误：所有结果都能回溯到当前 `(source, sourceIndex)`。
- 没有发现“知识库完全没有答案”的问题：目标内容都存在于当前 34 个分块中。
- 没有发现已标注 Gold 指向无关正文的确定性错误，因此本轮不修改 Gold 文件和数据集版本。

### 当前精确 EvidenceKey 口径下的检索不足

- 7 个部分召回：`risk-high-self-harm-direct`、`risk-high-human-support`、`risk-levels-overview`、`support-anxiety-practical-guidance`、`support-calm-practical`、`support-campus-assistant-role`、`support-medium-depression-tendency`。
- 17 个零命中里，有 11 个已经找到预期来源但落在其他分块，说明主要问题是同一文档内的分块选择或排序，而不是来源完全错误。
- 另外 6 个零命中没有在 Top4 找到预期来源：`risk-professional-support`、`support-anxiety-overwhelmed`、`support-serious-persistent`、`support-emergency-services`、`support-counselor-channel`、`boundary-illness-diagnose`。
- 6 个未完整召回的问题把 Gold 分块带进了第一名的邻居上下文：`risk-high-self-harm-direct`、`risk-high-severe-hopelessness`、`risk-high-human-support`、`risk-levels-overview`、`support-nonjudgmental-tone`、`support-medium-depression-tendency`。按既定契约，邻居只补充回答上下文，不能冒充排序锚点命中，所以这些仍保留为检索不足。

### Gold 的表达边界

复核还发现，知识库中存在重复或重叠表述。例如 `support-emergency-services` 虽然没有命中当前指定的 `counselor-referral-and-resources.md#1`，但 Top4 中的 `risk-policy.md#3` 和 `campus-mental-health.md#4` 也能回答紧急支持问题；`boundary-illness-diagnose` 命中的 `privacy-boundaries-and-ethics.md#0` 同样包含禁止诊断的完整边界。

当前 schema v1 的 `relevantEvidence` 是一个平面集合：把多个“任选其一即可回答”的证据全部补进去，会让 Recall 把它们解释成“应该全部找齐”。因此本轮不根据当前检索结果反向扩充 Gold，避免为了提高分数污染标准答案。若以后需要衡量“多个替代答案命中任意一个”，应单独设计替代证据组并升级数据集 schema，而不是在 v1 中偷偷改变指标含义。

## 门槛理由

- HitRate 下限 0.74：当前为 0.75，净减少一个命中问题就会跌破门槛。
- Recall 下限 0.69：允许一个双 Gold 问题少命中一个证据的微小波动，但会阻止更明显的整体召回下降。
- Precision 下限 0.20：当前为 0.202206，Top4 中净减少一个相关证据就会跌破门槛。
- MRR 与 NDCG 下限均为 0.60：给少量名次交换留出空间，同时阻止相关证据系统性后移。

本任务不调整分词、BM25 权重、融合权重或重排算法来追分。Hybrid 尚未获得真实外部调用结果，因此不设置 Hybrid 效果门槛。
