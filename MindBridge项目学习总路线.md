# MindBridge 项目学习总路线

> 版本：V1.2  
> 更新日期：2026-08-17  
> 项目根目录：`F:\Agent开发\MindBridge\Project\mindbridge-py`  
> 学习目标：在短时间内达到“能启动、能追链路、能解释设计、能验证行为、能回答面试追问”的程度；不要求重写整个项目。

## 1. 这份文档怎么用

这是一份供 VS Code Codex 插件持续执行的主路线，不是一次性阅读材料。

- 每次开始学习时，Codex 先读取项目根目录 `AGENTS.md`、本文件和 `README.md`。
- 一次只推进一个阶段，不同时展开多个模块。
- 每个阶段都按“效果体验 → 请求/数据流 → 精确代码 → 技术选型 → 测试证据 → 面试问答”进行。
- 用户明确说“本阶段完成”后，Codex 才更新本文的进度表和证据状态。
- 每次教学结束，Codex 更新根目录 `doc/` 下当前阶段的 `阶段N-主题.md` 知识卡，只保留新的概念、选型结论、已验证事实、代码位置、面试答法与未完成验收；本阶段常用终端命令可保留为一个简短区块，避免知识只留在聊天记录中。`AGENTS.md` 只保存 Codex 的教学行为准则与准确性边界。
- Claude Code 插件用于临时询问 Python 语法、局部代码和基础概念；主路线、进度判断和最终解释口径由 Codex 维护。
- 不要求手敲完整项目。必要时通过小范围调试、测试、故障注入和改造建立真实理解。
- 前端实现不在学习范围内，只把页面当作观察后端效果的入口。

## 2. 最终验收标准

完成本路线后，应当能够独立做到：

1. 从空闲终端启动项目，登录学生端和管理端，完成普通咨询与高风险场景演示。
2. 不看提示画出一次请求从 FastAPI 到 SSE 返回、数据库落库、Run Trace 和高风险闭环边界的完整链路。
3. 解释为什么该项目不是普通 RAG，以及为什么采用受控多 Agent 而不是单模型直接回答。
4. 讲清简历四条要点的代码位置、调用关系、选型理由、替代方案、缺陷和改进方向。
5. 运行单元测试、Engineering Harness 和 RAG 评测，解释成功与失败结果。
6. 回答 Structured Output、Function Calling、Skill、Memory、LangChain/LangGraph、Harness、Tracing 和 Evaluation 的高频问题；MCP 只需说明其在本项目中的非核心边界。
7. 对一个可控故障完成定位，能够基于 Run Trace、日志、数据库记录和测试解释原因。
8. 用 1 分钟、3 分钟和 10 分钟三种长度介绍该项目。

## 3. 简历主线

后续学习必须优先为以下四条提供证据：

1. **Agent Runtime Harness**：`MindBridgeAgentHarness` 统一编排单轮 Agent 运行，串联输入脱敏、多 Agent 调用、报告落库、工具计划与 Run Trace。
2. **事件驱动多 Agent**：基于 `CollaborationBlackboard` 拆分理解、安全、上下文与回复 Agent，通过任务 Claim、Artifact 协作和最终采纳实现受控决策。
3. **风险识别与安全复核**：高风险词典优先、LLM JSON 评估、规则兜底，并由 `SafetyAgent` 复核候选回复。
4. **Engineering Harness**：搭建 Engineering Harness，覆盖风险、路由、Skills、RAG、API 及工具队列，支持关键行为回归。

辅助学习但不单独占简历条目：上下文压缩、Redis/MySQL 分层记忆、动态 RAG、Skill Registry、Run Trace、SSE、Basic Auth。

不再作为必修深挖：MCP Client/Server、工具治理和异步队列内部实现。它们只保留两类最低要求：一是能说明高风险后置闭环与在线回复的边界；二是在 Engineering Harness 阶段知道 Tool Queue 套件验证了什么。若以后重新写入简历或岗位 JD 明确要求，再单独恢复。

## 4. 项目总链路

```text
学生端请求
  -> FastAPI /api/chat
  -> ChatService.stream_chat
  -> MindBridgeAgentHarness.run
      -> PrivacySanitizer
      -> 创建/查找 ChatSession
      -> EventDrivenMultiAgentRuntime.run
          -> CollaborationBlackboard
          -> Coordinator 创建任务
          -> UnderstandingAgent / SafetyAgent / ContextAgent / ResponseAgent Claim
          -> Agent 发布 Artifact
          -> SafetyAgent 审查候选回复
          -> Coordinator 最终采纳
      -> 保存用户消息、心理报告和 AgentRunTrace
      -> 生成 AgentToolPlan
  -> ChatService 通过 SSE 流式返回模型回复
  -> 保存助手消息
  -> 高风险结果进入后置闭环
      -> Excel 留档 / 风险案例 / 预警发送
      -> 当前学习只理解这条边界，不深挖 MCP 与队列实现
```

上下文支线：

```text
ContextAgent
  -> Redis 短期记忆
  -> Redis 缺失时 MySQL 历史回填
  -> 历史裁剪与摘要压缩
  -> CHAT / CONSULT / RISK 动态路由
  -> Chroma + BM25 混合检索，异常时 BM25 降级
  -> 根据意图和风险加载 SKILL.md
```

## 5. 当前已知环境基线

以下是此前在本机完成的验证记录，不代表新终端当前仍在运行：

- 项目虚拟环境：项目根目录 `.venv`，使用 Python 3.12.13。
- 除 Chroma 外的主要依赖已安装，`pip check` 当时无损坏依赖。
- `chromadb==0.5.23` 在 Windows + Python 3.12 下缺少对应 `chroma-hnswlib` wheel，且本机未安装 MSVC，因此未成功安装。
- 17 个单元测试此前全部通过。
- Engineering Harness 此前为 4/6 通过；Standard Skills 与 API 的失败来自 Windows 路径分隔符断言，不是 Skill 加载失败。
- RAG 评测此前在 BM25 降级链路运行 60 条用例：HitRate 0.9667，MRR 0.9083。不得把该指标描述为 Chroma 主链路结果。
- 曾使用 Mock AI + SQLite + BM25 降级完整跑通普通聊天、高风险报告、Excel、案例、预警和 Run Trace。
- 本机当时没有 Docker；Redis 未运行；MySQL 端口存在但未完成账号联调；Ollama 服务存在但没有目标模型。

这些兼容性问题不是需要隐藏的缺陷，而是后续“依赖管理、降级策略、故障定位和测试经验”的真实材料。

## 6. 总进度

| 阶段 | 主题 | 简历映射 | 预计时间 | 状态 |
|---|---|---|---:|---|
| 0 | 正确打开项目、首次启动与效果体验（含真实依赖联调） | 全局 | 3～5 小时 | 已完成（0A + 0B 已验收） |
| 1 | 项目地图、FastAPI 与完整请求主链 | Harness 前置 | 2 小时 | 已完成（主链代码阅读与职责复述已验收；SSE 实帧抓取可后续补证） |
| 2 | MindBridgeAgentHarness 与 Run Trace | 简历第 1 条 | 2.5 小时 | 已完成（职责复述、3/3 单测、真实 Trace API 证据已验收） |
| 3 | 事件驱动多 Agent 与黑板协作 | 简历第 2 条 | 3～4 小时 | 未开始 |
| 4 | 三级风险识别与 SafetyAgent | 简历第 3 条 | 2～3 小时 | 未开始 |
| 5 | Memory、动态 RAG 与 Skill 支撑链 | 辅助链路 | 2～2.5 小时 | 未开始 |
| 6 | Engineering Harness、测试与评测 | 简历第 4 条 | 2.5～3 小时 | 未开始 |
| 7 | 面试证据矩阵与模拟面试 | 全部 | 2～3 小时 | 未开始 |

阶段 0～2 已完成。当前剩余必修约 12～15 小时；Function Calling 实现、LangGraph 编码练习、MCP 与异步队列内部机制全部移到选修，不占用面试主线时间。

## 7. 分阶段路线

### 阶段 0：首次启动与观看最终效果

**目标**：先知道项目最终长什么样，再回头读代码。

**正确打开方式**：VS Code 打开 `F:\Agent开发\MindBridge\Project\mindbridge-py`，不要只打开 `app`。

**首次演示配置**：先使用 Mock AI、SQLite 和 BM25 降级，避免 MySQL、Redis、Ollama、Chroma 同时阻塞学习。

PowerShell 终端：

```powershell
.\.venv\Scripts\Activate.ps1
$env:AI_PROVIDER="mock"
$env:DATABASE_URL="sqlite:///./target/mindbridge-learning.sqlite3"
$env:KNOWLEDGE_VECTOR_ENABLED="false"
$env:TOOL_QUEUE_ENABLED="true"
$env:ALERT_EMAIL_DELIVERY_MODE="log"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

体验入口：

- 首页：`http://127.0.0.1:8080/`
- 学生账号：`student / student123`
- 管理员账号：`admin / admin123`
- 健康检查：`http://127.0.0.1:8080/actuator/health`

必须体验：

1. 普通聊天。
2. 咨询类问题，例如焦虑或失眠。
3. 明确高风险表达。
4. 管理端查看心理报告、案例、工具任务和 Run Trace。
5. 查看 `target`、`data/mindbridge-risk-ledger.xlsx` 和日志中产生的结果。

**补充基础**：虚拟环境、环境变量、Uvicorn/ASGI、Mock Provider、SQLite 与真实依赖隔离。

**验收**：用户能描述普通消息和高风险消息在界面与后台结果上的三项差异。

**阶段 0B｜Docker 化真实依赖联调（本阶段不可跳过）**

0A 的 Mock + SQLite 演示只证明了可重复的业务闭环，不能代替真实中间件验证。当前采用 Docker Compose 复现作者的 Linux + Python 3.12 运行基线，并接入百炼的 OpenAI 兼容聊天与 Embedding API。阶段 0 在以下事项完成前保持“进行中”：

1. 以 Compose 启动的 MySQL 作为 `DATABASE_URL`，确认表创建、聊天/报告/Trace 数据写入容器数据库，而非 SQLite 文件。
2. 验证 Compose Redis 的短期记忆写入、读取及 Redis 缺失时的 MySQL 回填。
3. 用 `qwen3.7-flash-2026-07-15` 重放普通、咨询和高风险输入，观察真实模型的 JSON 遵从、安全回复和流式时延；不能把 Mock 输出当作模型能力证据。
4. 用 `qwen3.7-text-embedding` 完成 Chroma 向量写入和一次知识检索；仍走 BM25 降级不算 Chroma 主链验证。
5. 用真实依赖重放一条高风险输入，确认 MySQL、Redis、报告、案例、工具任务与 Trace 的记录相互关联。

**0B 的技术边界**：Docker 避开 Windows + Python 3.12 安装旧版 Chroma 原生扩展时的本地编译问题，但并不自动保证 Embedding 配置、模型调用或旧依赖兼容。当前代码走 OpenAI 兼容的聊天和 Embedding HTTP 接口；不能仅因 Ollama 聊天模型可用就宣称向量 RAG 已接通。若 0B 中复现失败，记录为环境兼容性问题并先继续其他链路，不把 BM25 指标写成 Chroma 指标。

**最终验收**：0A 的三类输入与后台闭环已体验，且 0B 的五项真实依赖证据已完成；用户能区分“Mock + SQLite 演示证据”“真实 MySQL/Redis/模型证据”“Chroma 向量检索证据”。

### 阶段 1：项目地图与请求主链

**按顺序阅读**：

1. `README.md`
2. `app/main.py`
3. `app/api/routes.py`
4. `app/schemas/dtos.py`
5. `app/services/chat.py`
6. `app/agents/harness.py`

**核心问题**：

- FastAPI 启动时装配了什么？
- `/api/chat` 为什么返回 `StreamingResponse`？
- SSE 与 WebSocket 如何取舍？
- Route、Service、Harness、Runtime 各自拥有什么职责？
- 为什么流式回复完成后才保存助手消息和投递后置工具？

**补充基础（原第 10～14 课按需融入）**：消息角色、Prompt/Context、Pydantic DTO、异步生成器、SSE、依赖注入、结构化边界。

**验证**：在 VS Code 中沿调用定义逐步跳转，并能手画主链前半段。

**验收**：脱离文档讲出 `/api/chat -> ChatService -> Harness -> Runtime -> SSE`。

### 阶段 2：Agent Runtime Harness 与 Run Trace

**按顺序阅读**：

1. `app/agents/harness.py`
2. `app/agents/result.py`
3. `app/services/privacy.py`
4. `app/services/memory.py`
5. `app/services/report.py`
6. `app/services/trace.py`
7. `app/models/entities.py`

**必须理解**：

- Harness 不是 HTTP Controller，也不是模型本身。
- `original_input` 与 `model_input` 为什么分开。
- 一轮运行的输入、输出、持久化和工具计划边界。
- Trace 记录事件、任务、Artifact、检索、风险和最终回复的方式。
- Runtime Harness 与 Engineering Harness 的区别。

**技术对比**：普通 Service vs Runtime Harness；同步落库 vs 事件记录；日志 vs Trace；原始消息 vs 脱敏模型输入。

**测试**：`tests/test_privacy_and_assessment.py`，并通过管理 API `/api/admin/agent-traces` 查询 Agent Trace。当前管理端页面尚未接入该接口，不以 UI 展示作为验收条件。

**验收**：用 2 分钟解释简历第 1 条，并回答“为什么不全部写在 ChatService 中”。

### 阶段 3：事件驱动多 Agent 与黑板协作

**按顺序阅读**：

1. `app/agents/events.py`
2. `app/agents/registry.py`
3. `app/agents/autonomous.py`
4. `app/agents/coordinator.py`
5. `app/agents/event_driven_runtime.py`
6. `tests/test_event_driven_multi_agent.py`

**必须理解**：Task、Claim、Capability、Confidence、Artifact、Critique、Safety Review、Final Acceptance、最大轮数和停止条件。

**关键判断**：当前实现是受控的 Claim-based 多 Agent 协作，不是经典 ReAct，也没有模型原生 Function Calling。

**技术对比**：

- 单 Agent Prompt vs 多角色 Agent。
- 固定顺序 Workflow vs Claim-based 调度。
- 自研 Runtime vs LangChain Agent。
- 自研黑板状态 vs LangGraph StateGraph。
- 为什么心理安全场景更适合受控编排，而不是无限自主循环。

**验证**：运行 `tests/test_event_driven_multi_agent.py`，读取一次真实 Run Trace 中的任务、Claim 和 Artifact。

**验收**：不看代码画出黑板数据结构和 Agent 协作顺序，回答“这些 Agent 是否只是不同 Prompt”。

### 阶段 4：三级风险识别与 SafetyAgent

**按顺序阅读**：

1. `app/services/ai.py`
2. `app/services/assessment.py`
3. `app/agents/autonomous.py` 中 SafetyAgent 相关实现
4. `app/knowledge/risk-policy.md`
5. `skills/high_risk_safety_plan/SKILL.md`
6. `tests/test_privacy_and_assessment.py`

**必须理解**：高风险词典优先、LLM JSON 解析、风险分数校正、异常时 heuristic 兜底、候选回复复核和安全覆盖。

**技术对比**：规则 vs 模型分类；准确率 vs 召回率；普通内容审核 vs 独立 SafetyAgent；自然语言结果 vs Structured Output。

**验证**：准备正常、模糊咨询、明确高风险、模型错误 JSON 四类用例。不得编造 93% 召回率或 96% 准确率；只有建立并运行数据集后才能写指标。

**验收**：能解释为什么高风险词典必须在 LLM 前面，以及模型失败时系统如何保持安全。

### 阶段 5：Memory、动态 RAG 与 Skill 支撑链

**按顺序阅读**：

1. `app/services/memory.py`
2. `tests/test_memory_compaction.py`
3. `app/services/knowledge.py`
4. `app/services/vector_store.py`
5. `app/services/skills.py`
6. `tests/test_skills.py`
7. `skills/*/SKILL.md`

**必须理解**：Redis 短期记忆与 MySQL 回填的分工、历史裁剪与摘要压缩、意图驱动检索、Chroma/BM25 融合及降级、Skill 动态选择。只追踪 `ContextAgent` 真正消费的输入，不独立重写一套 RAG 或 Memory。

**技术对比**：Memory vs 消息历史 vs Workflow State；RAG vs Agent；BM25 vs Vector vs Hybrid；Chroma vs Elasticsearch/pgvector；Skill vs 写死 Prompt。

**验证**：运行 memory/skill 测试，抽查一次知识检索与 Skill 选择，明确当前 RAG 指标属于哪条检索链路。

**验收**：能回答“为什么已有 RAG 项目还需要 Agent”和“为什么不能把全部历史都塞进 Prompt”。

### 阶段 6：Engineering Harness、测试与评测

**按顺序阅读**：

1. `app/harness/runner.py`
2. `app/rag_eval/runner.py`
3. `app/rag_eval/mindbridge-rag-eval.json`
4. 与四条简历主线直接相关的测试
5. `target/harness/harness-report.json`
6. `target/harness/rag-eval-report.json`

**必须理解**：单元测试、集成测试、端到端验证、LLM 行为评测、RAG 指标、失败样例回归和 Mock 可重复性；区分“套件存在”“当前通过”和“真实依赖已验证”。

**Tool Queue 最低范围**：只阅读 `app/harness/runner.py` 中对应套件与报告结果，知道它验证任务依赖、幂等、限流和 dead letter；不阅读 MCP Server/Client，也不要求掌握队列 Worker 内部实现。

**优先修复**：Windows `\SKILL.md` 与 Harness 断言 `/SKILL.md` 的兼容问题；修复后重新验证六类 Harness。

**指标**：HitRate、Recall@K、Precision@K、MRR、NDCG；每个指标都要能给出直观例子，不能只背定义。

**技术对比**：pytest/unittest vs 自定义 Harness；确定性断言 vs LLM grader；离线评测 vs 线上监控。

**验收**：独立运行四条简历主线相关验证，解释每个失败属于代码缺陷、环境问题、依赖问题还是评测质量问题，并用一分钟讲清 Engineering Harness 简历条目。

### 阶段 7：面试证据矩阵与模拟面试

为每条简历要点建立以下证据：

| 维度 | 必须产出 |
|---|---|
| 技术是什么 | 30 秒定义 |
| 为什么使用 | 项目问题与选择理由 |
| 代码在哪里 | 精确文件、类、函数和调用流 |
| 替代方案 | 至少两种方案及取舍 |
| 运行证据 | 接口、数据库、Trace、日志或页面结果 |
| 测试证据 | 单测、Harness、评测集和指标 |
| 遇到的困难 | 真实复现或命名的故障注入 |
| 局限与改进 | 当前缺陷、风险和下一步 |
| 面试回答 | 1 分钟主答 + 3 层追问 |

最终进行三轮模拟：

1. 项目总体与业务价值。
2. Agent Runtime、风险安全、RAG 支撑链与 Engineering Harness 深挖。
3. 压力追问：为什么不用其他方案、指标是否可信、是否真的亲自实现。

### 选修模块：只有恢复简历声明或岗位明确要求时再学

1. **Function Calling 实现**：Tool Schema、模型原生工具请求、参数校验、Tool Message 回填、停止条件和权限。当前只要求能解释概念，以及说明 MindBridge 现状不是模型原生 Function Calling。
2. **LangChain/LangGraph 编码练习**：当前只做概念映射，不再要求实现独立工作流或迁移 MindBridge。
3. **MCP 与异步队列**：只保留“当前默认队列直接调用内部服务，MCP 是关闭队列后的备用适配路径”这一事实；不安排客户端连接、Server 编写、限流、重试、死信和治理改造练习。

## 8. 每次学习会话的固定输出

每次 Codex 教学结束时，应给出：

1. 本次完成的链路。
2. 阅读过的精确文件和关键符号。
3. 用户已经能复述的内容。
4. 尚未理解或尚未验证的风险点。
5. 本阶段 3～5 个面试问题。
6. 必要的运行/测试证据。
7. 下一次唯一入口。

以上内容必须在会话结束前合并到 `doc/阶段N-主题.md` 对应的**单个阶段知识卡**中；常用终端命令以简短区块保留在该阶段笔记，`AGENTS.md` 只维护 Codex 的行为准则。笔记更新不等于阶段完成；只有用户能够复述并明确确认后，才能修改总进度表。

如果用户只是看过但不能复述，阶段不得标记完成。

## 9. 学习边界

- 不通读所有代码，不学习前端实现。
- 不手抄完整项目，不以代码量衡量掌握程度。
- 不把普通 RAG 描述为 Agent；若面试被问到 MCP，明确它不是 Function Calling，也不是当前项目核心决策机制。
- 不把 Mock/BM25 降级指标包装成真实模型或 Chroma 主链指标。
- 不编造生产事故、用户量、准确率、召回率、性能提升或上线规模。
- 可以通过本地故障注入制造可复现问题，但面试时必须明确它是本地兼容性/故障演练。
- 任何代码改造都要先明确其服务于哪条四项简历要点、哪个测试或哪个高频面试问题；MCP、队列与框架迁移不得因“源码里有”而自动进入必修。

## 10. 下一步

当前唯一下一步：开始**阶段 3——事件驱动多 Agent 与黑板协作**。先阅读 `app/agents/events.py`，建立 `Task`、`Claim`、`Artifact` 和事件的数据关系；暂不进入具体 Agent 调度实现。
