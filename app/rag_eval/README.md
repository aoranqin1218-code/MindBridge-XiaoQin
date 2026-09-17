# RAG 评测链路与文件职责

> 这份文档用来回答两个问题：每个 RAG 评测文件是干什么的，以及它处在整条评测链路的什么位置。后续新增或调整评测文件时同步更新。

## 一句话流程

```text
内置知识文档
  → inventory.py 按生产规则生成证据清单和当前知识快照
  → Gold JSON 保存人工确认的正确证据
  → dataset.py 加载 Gold 并校验知识库指纹
  → runner.py 选择 BM25 或 Hybrid 并准备隔离运行环境
  → KnowledgeService 执行生产检索
  → evaluator.py 按 EvidenceKey 判卷
  → metrics.py 计算标准指标
  → runner.py 写出稳定 JSON 报告
  → Engineering Harness 读取报告并执行质量门禁
```

## 文件地图

| 文件 | 通俗理解 | 专业职责 | 不负责什么 |
| --- | --- | --- | --- |
| `app/knowledge/*.md` | 考试使用的知识材料 | 当前内置知识库原文 | 不保存评测答案 |
| `app/services/knowledge.py` | 真正答题的检索器 | 执行 BM25/向量召回、融合、重排和邻居扩展，返回 `SearchResult` | 不知道 Gold，不计算评测指标 |
| `app/rag_eval/inventory.py` | 把知识材料编成带编号的目录 | 复用生产 `chunk_text()` 生成 `CorpusEvidence`、双指纹和人工清单 | 不参与正式判卷 |
| `app/rag_eval/mindbridge-rag-evidence-inventory.md` | 人工标答案时看的分块目录 | 展示当前 34 个证据分块及其 EvidenceKey | 不是 Gold，不参与算分 |
| `app/rag_eval/mindbridge-rag-gold-v1.json` | 正确答案集 | 保存版本、语料指纹、68 个问题和人工确认的 `relevantEvidence` | 不执行检索、不实现公式 |
| `app/rag_eval/mindbridge-rag-validation-v1.json` | 独立验证题 | 保存未参与首轮参数分析的 12 个问题，覆盖当前全部 11 个知识来源 | 不替代正式 68 题历史集 |
| `app/rag_eval/mindbridge-rag-bm25-baseline-review.md` | 基线审计记录 | 保存失败案例分类、Gold 边界和 BM25 回归门槛的推导理由 | 不参与运行、不修改分数 |
| `app/rag_eval/dataset.py` | 答案集的读取和验真人员 | 严格解析 Gold schema，并检查 Gold 与当前知识库双指纹是否一致 | 不调用检索器、不算分 |
| `app/rag_eval/metrics.py` | 数学公式表 | 统一排名去重和 HitRate、Recall、Precision、MRR、NDCG 公式 | 不读文件、不连接数据库 |
| `app/rag_eval/evaluator.py` | 判卷器 | 调用检索回调，用完整 EvidenceKey 判定相关性，生成逐题诊断和宏平均 | 不选择运行模式、不写报告文件 |
| `app/rag_eval/runner.py` | 考试组织者 | 解析 CLI、选择模式、准备隔离 SQLite、加载 Gold、校验指纹、调用 evaluator、写报告和控制退出码 | 不复制指标公式、不设置质量门槛 |
| `app/harness/runner.py` | 验收负责人 | 最终读取共享评测结果并执行项目质量门禁 | 不应再维护另一套相关性和指标算法 |

## 当前 runner 的运行边界

### BM25

```powershell
mindbridge-rag-eval --mode bm25
```

- 使用根据当前内置 Markdown 临时构建的内存 SQLite。
- 固定 `randomSeed=0`。
- 强制 `vectorEnabled=false`、`vectorRequired=false`。
- 默认报告写到 `target/rag/bm25/rag-eval-report.json`。

### Hybrid

```powershell
mindbridge-rag-eval --mode hybrid --allow-external
```

- 必须显式提供 `--allow-external`，因为会调用外部 embedding 服务并可能产生费用。
- 强制 `vectorEnabled=true`、`vectorRequired=true`，缺少凭据或向量能力时直接失败。
- 禁止失败后改跑 BM25。
- Chroma 索引和快照写入报告目录下的 `chroma/` 与 `chroma-snapshots/`，不会读取或覆盖生产 `data/chroma`。
- 默认报告写到 `target/rag/hybrid/rag-eval-report.json`。

以上短命令来自 `pyproject.toml` 的 `[project.scripts]`。项目首次拉取或入口发生变化后，需要在虚拟环境中执行一次 `python -m pip install -e .`；底层的 `python -m app.rag_eval.runner ...` 形式仍然可用。

Engineering Harness 同样已有短命令：

```powershell
# 固定 TopK=4 的确定性 BM25 回归门禁
mindbridge-harness --suite rag

# 按生产 TopK 同批运行 BM25 与真实 Hybrid，并输出指标差值和逐题变化
mindbridge-harness --suite rag-hybrid
```

`rag-hybrid` 是显式外部评测套件，不包含在默认 `mindbridge-harness` 或 `--suite all` 中，避免普通回归受网络和外部服务波动影响。它分别写出 `target/harness/rag-hybrid/bm25-report.json` 与 `hybrid-report.json`，再把五项指标差值及新增命中、丢失命中、Recall 提升和 Recall 退化的 case id 汇总进 Harness 报告。

## 报告中的对错与诊断

- `relevant`：完整 `(source, sourceIndex)` 是否属于 Gold，参与正式指标。
- `expectedSource`：是否找到了预期文档，只用于区分“文档错”还是“文档对、分块错”。
- `matchedDiagnosticTerms`：正文命中了哪些诊断词，只用于排查词面行为。
- `expandedContextEvidence`：哪些邻居被拼入正文，只用于解释上下文扩展。

后三项都不能把 `relevant=false` 改成正确。

## 当前接入状态

- 新 CLI runner 已接入正式 Gold、语料双指纹、共享 evaluator 和标准指标。
- BM25 成功报告与 Hybrid 未授权失败报告使用相同 schema，但写入不同目录。
- Engineering Harness 的 RAG suite 已直接调用同一个 `run_evaluation()`，使用正式 68 条 Gold 和同一份报告契约，不再读取旧 60 条数据或自行计算指标。
- Engineering Harness 额外提供显式 `rag-hybrid` suite；它复用同一个 runner 先跑 BM25、再跑强制向量的 Hybrid，不维护第二套判卷逻辑。
- `runner.py` 中旧的 `evaluate_case()`、`is_relevant()`、`ndcg()` 已删除，旧 0.95 门槛也已停用。
- BM25-only Harness 固定使用历史 TopK=4，并校验五项指标没有跌破本轮复核后的回归下限。
- Hybrid Harness 使用生产 TopK 做同口径对照，不套用只对 TopK=4 有效的旧 BM25 绝对门槛。
- 首次 Hybrid 只建立观察基线，不设置质量门槛，也不在同一轮根据结果反向调整融合权重。

2026-09-10 正式 BM25 运行使用 68 个 case、TopK=4，结果为：`HitRate@K=0.75`、`Recall@K=0.698529`、`Precision@K=0.202206`、`MRR@K=0.615196`、`NDCG@K=0.616197`。该结果已由独立入口和 Harness 重复复现；失败案例分类和门槛理由见 `mindbridge-rag-bm25-baseline-review.md`。

当前 Harness 回归下限为：`HitRate@K >= 0.74`、`Recall@K >= 0.69`、`Precision@K >= 0.20`、`MRR@K >= 0.60`、`NDCG@K >= 0.60`。这些值只用于阻止当前确定性 BM25 基线退步，不是产品质量目标。

同日执行不带 `--allow-external` 的 Hybrid 命令，得到预期失败报告：`status=failed`、`metrics=null`、`failure.code=external_call_not_authorized`，没有发起外部 embedding 请求。

Harness 接入后再次运行独立 BM25 命令和 `--suite rag`：两份报告的 68 条逐题结果、汇总指标、语料指纹与检索配置完全一致，Harness 结果为 PASS。

2026-09-16 首次真实 Hybrid 对照评测使用相同 68 个 case、相同 34 个证据分块和 TopK=4，连续两次运行结果一致：

| 指标 | BM25 | Hybrid | 变化 |
| --- | ---: | ---: | ---: |
| HitRate@K | 0.750000 | 0.823529 | +0.073529 |
| Recall@K | 0.698529 | 0.786765 | +0.088235 |
| Precision@K | 0.202206 | 0.238971 | +0.036765 |
| MRR@K | 0.615196 | 0.689951 | +0.074755 |
| NDCG@K | 0.616197 | 0.699325 | +0.083128 |

Hybrid 新命中 7 个 case，同时丢失 2 个原 BM25 命中 case；因此当前结论是“总体明显提升，但保留 BM25 对照和逐题回退诊断”，暂不删除 BM25 基线，也暂不把首次结果直接固化为 Hybrid 门槛。

随后使用原 68 题作为开发/历史集，并新增独立的 12 题验证集，对 TopK=4、5、6 做单变量比较；候选数 16、融合权重 0.65/0.35、重排、分块和 Embedding 模型均保持不变。为排除外部 Embedding 在不同请求之间的轻微排名波动，定稿矩阵从同一次 TopK=6 检索结果分别截取前 4、5、6 条，再调用同一指标函数计算。原 68 题上的 Hybrid 结果为：

| TopK | HitRate@K | Recall@K | Precision@K | MRR@K | NDCG@K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.823529 | 0.786765 | 0.238971 | 0.689951 | 0.699325 |
| 5 | 0.852941 | 0.816176 | 0.200000 | 0.695833 | 0.711990 |
| 6 | 0.897059 | 0.882353 | 0.181373 | 0.703186 | 0.737341 |

独立 12 题上，Hybrid 在 TopK=4、5、6 的 HitRate 和 Recall 都是 1.0；TopK=4 已经饱和。综合历史集上 TopK=5 新增 2 个命中且没有丢失原命中、独立集不退化，以及每次只增加 1 个上下文分块的成本，本轮把生产默认值从 4 调为 5。TopK=6 保留为后续候选，不在同一轮继续扩大上下文。

修改后真实执行 `mindbridge-harness --suite rag-hybrid` 已通过，确认 BM25 和 Hybrid 都使用生产 TopK=5，且报告不再附带 TopK=4 专属绝对门槛。由于 Hybrid 依赖外部 Embedding，同一模型名在不同调用时仍可能出现小幅排名波动，因此本轮不把单次 Hybrid 分数固化为绝对门槛；确定性 BM25 TopK=4 继续承担稳定回归门禁。

真实运行还验证了两项环境约束：Python 3.12/Windows 使用带预编译 wheel 的 `chromadb==1.5.9`；Embedding 请求按每批最多 20 条拆分，以满足当前 `qwen3.7-text-embedding` 接口限制。代理环境通过 `httpx[socks]` 提供 SOCKS 支持。
