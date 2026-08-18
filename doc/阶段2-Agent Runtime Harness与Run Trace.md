# 阶段 2｜Agent Runtime Harness 与 Run Trace

> 目标：能在两分钟内解释简历中的 `MindBridgeAgentHarness`：它如何把一条聊天请求变成可控、可持久化、可追踪、可触发后置任务的一轮 Agent 运行。当前状态：**已完成（2026-08-12）**。

## 本阶段结果链

```text
ChatService
  → MindBridgeAgentHarness.run(user, request)
  → 原始输入与脱敏模型输入、ChatSession
  → Runtime 返回 AgentRunResult
  → 用户消息 / 心理报告 / Run Trace 持久化
  → AgentToolPlan
  → AgentHarnessOutcome
  → ChatService 使用 response_messages 流式生成回复
```

关键边界：Harness 不是 HTTP Controller，不直接处理 SSE；也不是模型或多 Agent Runtime。它是包住 Runtime 的单轮业务编排层，拥有输入边界、会话与持久化、报告、Trace 和工具计划。

## 阶段 1 已继承的事实

- Route 每次聊天请求创建 `ChatService(db, settings)`，后者创建请求级 `MindBridgeAgentHarness`。
- `run()` 主干已完成初读：输入准备 → Runtime → 用户消息 → 报告 → Trace → 工具计划 → `AgentHarnessOutcome`。
- `dispatch_tools` 只需知道：队列开启时入队；关闭队列时才用 MCP client。队列/MCP 细节属于阶段 6。

## 按序阅读与目标

1. 已完成 `app/agents/result.py`：区分 Runtime 的 `AgentRunResult` 与 Harness 的 `AgentHarnessOutcome`。
2. 已完成 `app/services/privacy.py` 与输入流向复述：解释 `original_input` 和 `model_input` 的边界；Memory 的详细压缩与回填机制留到阶段 5。
3. 已完成 `app/services/trace.py`、`app/models/entities.py` 中的 `AgentRunTrace` 复述：解释 Trace 如何落到 MySQL，以及 Trace 为什么不等于普通日志。报告细节按 Trace 的 `report_id` 关联按需补充。
4. `tests/test_privacy_and_assessment.py` 已于 2026-08-12 运行，3/3 通过；真实 Trace 已通过管理 API 核验；两分钟复述已形成。

## 本阶段暂不下钻

- Runtime 黑板、Claim、Artifact 的具体调度（阶段 3）。
- 三级风险算法与 SafetyAgent（阶段 4）。
- Redis/RAG/Skill 细节（阶段 5）。
- MCP、队列、重试、死信和工具治理（阶段 6）。

## 阅读目标：Runtime 结果对象

- `AgentRunResult` 由 Agent Runtime 产生，是它完成一轮协作后的纯推理结果：意图、风险、评估、检索知识、供最终模型调用的 `response_messages`、步骤和 `memory_brief`，以及协作事件/任务/产物。
- `requires_report` 是从 `intent` 推导出的属性：除普通 `CHAT` 外均返回 `True`。Runtime 只给出这个业务信号；真正创建数据库心理报告的是外层 Harness。
- `AgentHarnessOutcome` 由 Harness 产生，在 `AgentRunResult` 基础上补充会话、原始/脱敏输入、数据库报告 ID、工具计划和 Trace ID，供 `ChatService` 发送 SSE、保存助手消息和分发工具。
- 两者不应合并：Runtime 应保持对 HTTP、数据库和工具执行无感，才能独立测试和复用；Harness 才拥有业务副作用与交付上下文。
- 用户已能识别 `AgentRunResult` 被 `AgentTraceService` 用于持久化运行轨迹。需保持的精确表述是：它首先是 Runtime 交付给 Harness 的完整领域结果；Harness 再将其中一部分复制到业务交付对象，并把完整协作细节交给 Trace。`AgentHarnessOutcome` 是面向 `ChatService` 的业务交付 DTO，不只是对 `AgentRunResult` 做消息过滤。
- `PrivacySanitizer` 当前只用正则替换手机号、邮箱和 18 位身份证号，不代表隐藏全部个人信息。用户身份来自认证后的 `UserAccount` 与会话关联，不从聊天原文识别。
- `model_input` 是主要模型处理文本；`original_input` 则进入 MySQL 用户消息、心理报告内容和 Run Trace。`save_message` 虽把原文传入 Memory Store，但 `_serialize` 会在写 Redis 前再次脱敏，Redis 读取、MySQL 历史回填和上下文压缩也会再次脱敏。现状仍不是端到端匿名化，因为 MySQL、报告与 Trace 保留原文，且脱敏规则只覆盖三类标识符。
- 只保存脱敏文本在技术上可行；当前保留原文是为了完整业务记录、心理报告及审计回溯（用途解释属于结合数据流的设计推断），代价是更高的敏感数据保护责任。

## 阅读目标：Run Trace

- 用户已能说明 Trace 保存 Agent steps，以及协作事件、任务和 Artifact；此外还保存输入、意图/风险、记忆摘要、检索知识、最终模型 Prompt、评估结果，并通过 `report_id` 关联心理报告。
- 普通日志不只记录报错，也记录启动、请求、告警和状态等运行事件；它通常用于按时间排障和监控。当前 Trace 则以“一次 Agent run”为聚合单位，把领域输入、推理协作状态和结果结构化落库，便于管理端查询和审计。
- 当前 Trace 在最终 `AiClient.stream` 前保存，所以不含最终模型文本；也不含之后的工具执行结果。Runtime 在 Trace 创建前异常时还可能完全没有本轮 Trace，实体中也缺少明确的成功/失败状态、错误和各阶段耗时。因此它不是完整端到端分布式追踪。

## 测试与证据

运行命令：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_privacy_and_assessment -v
```

2026-08-12 结果：3 个测试全部通过，用时约 0.001 秒。

- `test_privacy_sanitizer_masks_common_identifiers`：证明给定样例中的手机号、邮箱和 18 位身份证号会被替换；不能证明所有个人信息类型、所有格式或完整隐私合规。
- `test_redis_memory_serializes_sanitized_content`：证明 Memory Store 的 `_serialize` 对给定手机号/邮箱样例先脱敏再生成 Redis payload；不能证明真实 Redis 连接、写入、TTL、读取、故障降级或 MySQL 回填。
- `test_high_risk_signal_uses_hard_guard_before_model`：证明给定高风险短语命中硬规则、返回 `HIGH` 且不会调用模型；不能证明响应时延、所有高风险表达的覆盖率、误报率或整体风险识别指标。

测试产生 `datetime.utcnow()` 的 `DeprecationWarning`，位置为 `app/services/memory.py::_serialize`。本次断言全部通过，当前序列化内容不受影响；但它使用无时区的旧 UTC 时间写法，属于未来 Python 兼容性技术债。代码库还有多处同类调用，后续应统一迁移到时区明确的 UTC 写法，而不是只改这一行；本阶段先记录，不修改源代码。

## 两分钟复述 / 面试主答法

`MindBridgeAgentHarness` 是包在多 Agent Runtime 外面的单轮业务编排层。它不是 HTTP Controller，也不是模型或 Runtime 本身。收到聊天请求后，Harness 先整理用户原始输入，通过 `PrivacySanitizer` 得到脱敏后的 `model_input`，再创建或恢复 `ChatSession`。原始输入用于 MySQL 业务记录、心理报告和审计 Trace；模型输入主要进入 Agent 和最终模型链路。当前脱敏只覆盖手机号、邮箱和 18 位身份证号，所以不能称为完整匿名化。

随后 Harness 创建并调用事件驱动 Runtime。Runtime 负责理解、安全、上下文和回复 Agent 的协作，返回 `AgentRunResult`，其中包含意图、风险、评估、RAG 结果、记忆摘要、供最终模型使用的 `response_messages`，以及步骤、事件、任务和 Artifact。Runtime 不直接创建数据库报告，也不执行外部工具，因此可以独立测试和复用。

Harness 接到 `AgentRunResult` 后保存用户消息，按意图和评估结果决定是否创建心理报告，再通过 `AgentTraceService` 保存一次生成前决策快照。Trace 记录原始/脱敏输入、意图、风险、记忆、RAG、评估、Prompt 和多 Agent 协作结果；其中 `agentSteps` 汇总的步骤、事件、任务和 Artifact 是 Runtime 协作已被持久化的直接证据。

最后 Harness 生成 `AgentToolPlan`，并把会话、模型消息、报告 ID、Trace ID 和工具计划组装成面向 `ChatService` 的 `AgentHarnessOutcome`。`ChatService` 使用其中的 `response_messages` 调用 `AiClient.stream()`，通过 SSE 返回最终回复，流结束后保存助手消息并投递后置工具。这样 HTTP/SSE、Agent 推理协作和业务副作用保持职责分离。

当前 Trace 还不是完整端到端追踪：它在最终模型流式生成前写入，因此不包含最终回复文本；也不包含后置工具执行结果。如果 Runtime 在 Trace 创建前失败，本轮可能没有 Trace；实体也缺少统一的成功/失败状态、错误和阶段耗时。

## 本阶段验收结论

- 已能区分 `AgentRunResult` 与 `AgentHarnessOutcome` 的生产者、消费者和职责边界。
- 已能解释原始输入、模型输入、MySQL 与 Redis 的脱敏边界。
- 已能说明 Trace 与普通日志的差异，以及当前 Trace 的端到端缺口。
- 已获得 3/3 单元测试和真实管理 API Trace 的证据；管理端 UI 未接入 Trace 接口的缺口已核实。

## 面试自测题

1. 为什么 `MindBridgeAgentHarness` 不应该全部写进 `ChatService`？
2. 为什么 `AgentRunResult` 和 `AgentHarnessOutcome` 不应合并？
3. Trace 相比普通日志解决了什么问题，又缺少什么？
4. 为什么原始输入和模型输入要分开？当前脱敏有哪些局限？
5. 为什么工具计划由 Harness 生成，但具体工具执行放到流式回复之后？

## 下一次唯一入口

开始阶段 3 时，先阅读 `app/agents/events.py`，只建立 `Task`、`Claim`、`Artifact` 和事件之间的数据关系；暂不进入具体 Agent 调度实现。

---

## 阶段二学习笔记（2026-08-12）

### 一、AgentRunResult 各字段含义

`app/agents/result.py`，Runtime 产出的纯推理结果，不沾 HTTP/DB/Tool：

| 字段 | 含义 |
|------|------|
| `intent` | 用户意图类型（CHAT / CONSULT / RISK） |
| `risk_level` | 风险等级（LOW / MEDIUM / HIGH） |
| `assessment` | 心理学评估详情（情绪、分数、置信度），非评估场景为 None |
| `retrieved_knowledge` | RAG 从知识库检索到的相关片段列表 |
| `response_messages` | **发给 LLM 的输入 prompt**（系统提示词 + 知识上下文 + 对话历史 + 用户输入），不是最终回复。最终回复由 `ChatService` 调 `AiClient.stream(response_messages)` 产生 |
| `steps` | 每个 Agent 的执行步骤记录 |
| `memory_brief` | 对话历史的压缩摘要（1-3 句要点），用于查询改写和拼入最终 prompt，让模型记住上下文 |
| `collaboration_events` | 多 Agent 协作事件日志 |
| `collaboration_tasks` | 多 Agent 协作任务列表 |
| `collaboration_artifacts` | 多 Agent 协作中间产物 |
| `requires_report` | `@property`，`intent != CHAT` 时返回 True，决定是否创建心理报告 |

### 二、AgentStep 各字段含义

| 字段 | 含义 |
|------|------|
| `step` | 步骤序号 |
| `agent` | 执行该步骤的 Agent 名称 |
| `action` | 执行的动作（工具调用、推理等） |
| `observation` | 动作返回的观察结果 |

### 三、AgentHarnessOutcome

Harness 产出的业务交付 DTO，在 `AgentRunResult` 基础上补充：

- `session` — 聊天会话实体
- `original_input` / `model_input` — 原始输入和脱敏后输入
- `report_id` — 心理报告数据库 ID
- `tool_plan` — 后置工具计划
- `trace_id` — Run Trace 记录 ID

两者不能合并：Runtime 保持对基础设施无感才能独立测试和复用；Harness 才拥有业务副作用。

### 四、完整执行链路

```
POST /api/chat/stream
  → FastAPI Route：认证、DB 依赖、返回 StreamingResponse
  → ChatService.stream_chat()
      → MindBridgeAgentHarness.run()         同步，编排层
          → 1. PrivacySanitizer 脱敏
          → 2. 解决会话（续旧 or 新建）
          → 3. 启动 EventDrivenAgentRuntime → 产 AgentRunResult
          → 4. 保存用户消息（放这里：Agent 失败了不留孤儿消息）
          → 5. 按需创建心理报告（非闲聊 + 有评估结果）
          → 6. 写 Run Trace
          → 7. 打包 AgentHarnessOutcome
      → AiClient.stream(response_messages)    异步，逐 token
      → yield SSE: meta → token... → done
      → 保存助手消息
      → 投递后置工具（excel 报表等，与聊天体验解耦）
```

### 五、后置工具为什么放在 ChatService

Harness 是同步编排，产出 `AgentHarnessOutcome` 后职责结束。后置工具（Excel 报表、个案预警）有两个特点：

1. 不依赖 LLM 最终回复内容，只需要报告 ID 和风险等级
2. **不能阻塞用户看到回复**——用户必须先看到文字，工具后台慢慢跑

所以顺序是：token 全吐完 → 保存助手消息 → 最后才投递工具。工具失败只记日志，不影响聊天体验。

### 六、隐私脱敏

`PrivacySanitizer` 只覆盖手机号、邮箱、18 位身份证号三类，用正则替换为 `[已脱敏]`。

- `model_input` = 脱敏后 → 传给 LLM
- `original_input` = 脱敏前 → 写入 MySQL 聊天记录、心理报告、Trace（保留原文用于审计）
- Redis `_serialize` 写入前会再次脱敏
- 不是端到端匿名化

### 七、Redis 短期记忆存储

`RedisShortTermMemoryStore`：
- `__init__` 时 `_connect()` 连 Redis，失败返回 `None`（降级不崩溃）
- `_connect()`：动态 `import_module("redis")` → `Redis.from_url(url, decode_responses=True, socket_timeout=..., socket_connect_timeout=...)` → `ping()` 探活
- `append()`：`_serialize()` 打包成 JSON → `rpush` 追加到 list → `ltrim` 限制长度 → `expire` 设 TTL
- `load_recent()`：从 Redis list 读出最近 N 条消息
- Redis 连接也是 TCP：IP + 端口，跨进程通信
- `decode_responses=True`：字节自动转 Python str，不参与连接建立

### 八、测试相关

- `unittest` 是 Python 标准库自带测试框架
- `unittest.main()` 自动发现并运行文件中所有 `TestCase` 子类
- `ExplodingAi`：测试桩（stub），`complete()` 调用即抛异常，用来证明硬守卫拦截后不会走到模型调用
- `__new__` 创建实例但不调 `__init__`，测试里绕过 Redis 连接的取巧写法
  - **更好的做法**：依赖注入，`__init__` 加可选 `client` 参数，测试传假 client
- `json.loads` = 字符串 → Python 字典；`json.dumps` = Python 对象 → 字符串（s = string）

### 九、面试自测答案要点

1. **Harness 为什么不写进 ChatService？** — ChatService 管 SSE 生命周期，Harness 管业务编排（持久化、报告、Trace）。混一起违反单一职责。
2. **AgentRunResult 和 AgentHarnessOutcome 为什么不合？** — Runtime 不应感知 HTTP/DB/Tool，分开才能独立测试和复用。
3. **Trace vs 日志？** — 日志按时间排障，Trace 以一次 Agent run 聚合结构化落库，方便审计。当前不含最终回复文本和后置工具结果，缺失成功/失败状态和阶段耗时。
4. **原始输入和模型输入为什么分开？** — 原始用于业务记录和审计，脱敏后进入模型链路。脱敏只覆盖三类标识符。
5. **工具为什么在流式回复之后？** — 工具跟聊天解耦，不能阻塞用户看到回复。
