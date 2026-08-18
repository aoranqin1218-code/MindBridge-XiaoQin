# 阶段 3：事件驱动多 Agent 与黑板协作

> 目标：能脱离代码画出 `CollaborationBlackboard` 的数据结构和一次受控 Claim 协作流程，准确解释它为何不是固定 Agent 链、经典 ReAct 或模型原生 Function Calling。当前状态：**进行中（coordinator.py 与 event_driven_runtime.py 待读）**。

## 本阶段可观察结果

完成后应能从一条真实 Run Trace 中指出任务、Claim、Artifact、事件与最终采纳，并说明每类对象由谁创建、如何进入黑板、如何影响下一步调度。

## 请求 / 数据流地图

```text
EventDrivenAgentRuntimeService.run
  -> 创建 CollaborationBlackboard
  -> Coordinator 创建或派生 AgentTask
  -> Registry 按 Capability 筛选 Agent
  -> Agent.decide 返回 Claim 决策与 Confidence
  -> Coordinator 按优先级、置信度和预算选择 Claim
  -> Agent.act 返回 AgentTurnResult
  -> Blackboard 追加 Message / Artifact / Event，并更新 Task
  -> Safety Review 通过且回复置信度达标
  -> FINAL_ACCEPTED，转换为 AgentRunResult
```

## 当前阅读进度

| 文件                                            | 状态      | 学到什么                                                   |
| ----------------------------------------------- | --------- | ---------------------------------------------------------- |
| `events.py`                                   | ✅ 完成   | 黑板协议全部数据类与方法（见「一、二」）                   |
| `registry.py`                                 | ✅ 完成   | 能力筛选 + Confidence 排序（见「三」）                     |
| `autonomous.py`                               | ✅ 完成   | 5 个 Agent 的 profile/decide/act + 公共基类（见「四~十」） |
| `coordinator.py`                              | 🔄 进行中 | 有界调度循环、缺啥派啥、最终采纳（见「三、十」）           |
| `event_driven_runtime.py`                     | 🔄 进行中 | 入口装配、黑板→AgentRunResult 转换                        |
| `result.py` / `harness.py` / `trace.py`   | ✅ 完成   | 对外协议、业务编排、Trace 三级沉淀（见「十一」）           |
| `memory.py` / `ai.py` / `agent_models.py` | ✅ 完成   | Redis 记忆、AI 客户端、模型路由（见「七、八」）            |
| `skills.py`                                   | 概要      | 仅知道按意图/风险选技能文本，细节留阶段五                  |

**下一步**：读 `coordinator.py` 的 `EventDrivenCoordinator`（有界调度循环、任务派生、候选选择、最终采纳），再读 `event_driven_runtime.py`（入口装配、黑板→AgentRunResult）；之后真实 Trace 核验（需 Docker 恢复）或进入阶段四（风险算法、心理评估服务）。

---

## 一、分层与不可变（数据协议）

### events.py vs result.py

- **events.py = 黑板内部协议**（协作进行中）：`AgentTask / AgentClaim / AgentMessage / AgentArtifact / AgentEvent / AgentTurnResult / CollaborationBlackboard`。顶部只 import 标准库，不依赖任何业务对象。
- **result.py = 对外输出协议**（协作结束后）：`AgentRunResult + AgentStep`。依赖 `IntentType / AiMessage / PsychologyAssessment / SearchResult` 等业务对象。
- **AgentStep 不进 events.py**：它是"输出视图"，由 `_events_to_steps` 从事件流加工成 `(序号, Agent, 动作, 观察)` 扁平展示行。放 events.py 会迫使黑板协议反向依赖输出层，破坏单向分层。

### frozen 与不可变（核心）

- `@dataclass(frozen=True)` 只冻结**字段引用**（浅不可变）：禁止 `self.x = ...` 重新赋值，**不禁止内容原地改**。
- 修改靠 `dataclasses.replace(...)` 造新对象，旧对象保持原样 = 版本快照（Git commit 类比）。
- 类 frozen 后：str/int/tuple/frozenset 天然不可变；**set/list/dict 可变容器要自己守规矩**。
- 字段选型原因：
  - `required_capabilities` → **frozenset**：消费者做集合运算 `set(...).issubset(agent_capabilities)`。
  - `claimed_by / depends_on` → **tuple**：认领顺序本身有信息。
  - `metadata / payload` → **dict**（不冻结）：dict 无现成不可变兄弟（`MappingProxyType` 只是只读视图）；元数据袋不参与调度判定。
- **dict 可变字段的规矩**：改它必须手动 copy。`add_task` 里 `tasks = dict(self.tasks)`（**浅拷贝**，外层新 dict、内部 AgentTask 对象仍共享引用——因 AgentTask 也 frozen，共享安全）→ 在副本上写 → `replace()`。对比 tuple：`(*self.events, event)` unpack 天然造新。**这是约定不是强制**。
- 同样的防御性拷贝还见于 `AgentRegistry.__init__` 的 `self._agents = list(agents)`（防外部原列表改动污染内部）和 `agents` property 的 `list(self._agents)`（防调用方改内部）——一进一出两头堵。

### AgentTask 的方法语义

- `claim(agent)` / `reopen()` / `close()` 都靠 `replace()` 返回新任务；`claimed_by` 是 tuple，追加用 `(*self.claimed_by, agent)`。
- `update_task = add_task` 二合一：id 不存在=新增，存在=**覆盖**。dict key 唯一，`tasks` 里每个 id 永远最新一份，不会累积多个版本（覆盖是特性不是 bug）。

---

## 二、事件流 events

- 定位：**append-only 过程证据 / 审计日志**（git log 类比）。既能回放，也能做运行中决策依据（`_select_intent` 查 `SAFETY_OVERRIDE`）。
- **Task vs Event 一句话**：Task = 要做什么（工作单元，状态可被同 id 覆盖）；Event = 发生了什么（append-only 轨迹，兼作决策信号）。
- 产生点汇总：
  - 黑板方法自动带：`send_message→MESSAGE_SENT`；`add_artifact→ARTIFACT_PUBLISHED/CRITIQUE_PUBLISHED`；`apply_turn_result→TASK_CREATED(新子任务)/TASK_CLOSED/Agent 自带 result.events`；`accept_final→FINAL_ACCEPTED`
  - Coordinator 补：`ROUND_STARTED / TASK_CLAIMED / BUDGET_EXHAUSTED / TASK_CREATED`
  - runtime：`TURN_STARTED`
  - Agent 主动构造（经 `result.events` 回流）：`SAFETY_OVERRIDE / REVISION_REQUESTED`
- **每轮每个 Agent 干完活**（`close_task` 默认 True）`apply_turn_result` 必追加一个 `TASK_CLOSED` 事件。
- **不记事件的路径**：`task.reopen()` 无事件；`add_task / update_task` 本身不带事件（调用方补）；读方法无事件。
- 事件只留"发生了什么 + 东西在哪"，不留内容：`add_artifact` 不传 payload（产物按 `artifact_id` 可查）；`ARTIFACT_PUBLISHED.message` 存 kind 标签，而 `MESSAGE_SENT.message` 存正文——**两处对 message 字段用法不对称**。
- `AgentEventType` 是 `str, Enum`：成员即字符串，拼错枚举名直接 `AttributeError`，收敛合法值、可补全。

### `AgentTurnResult.events` 为什么是 tuple

- 消费端 `apply_turn_result` 用 `for event in result.events: board = board.append_event(event)` **逐个展开**，天然支持 0/1/多个。
- 单元素元组必须写 `(AgentEvent(...),)`，**尾部逗号是关键**——缺了逗号传进去的是对象本身，类型不对。
- 用 tuple 而非 list：`AgentTurnResult` 是 frozen 类，字段应用不可变类型。

---

## 三、Registry 与认领调度

### 注册表三件套

- `AgentCapability`：职责能力标签（UNDERSTANDING/SAFETY/CONTEXT/RESPONSE/COORDINATION）。
- `AgentProfile`：创建时确定的静态档案（name/capabilities/system_prompt/memory_policy/model_profile/tool_permissions）。`@dataclass(frozen=True)`。
- `AgentDecision`：Agent 面对某任务时的动态 Claim 决策（`claim` / `confidence` / `reason`）。
- `AutonomousAgent(Protocol)`：要求 `profile + decide + act` 三成员。`Protocol` 只声明契约，Agent 不用显式继承，只要"长得像"即可（鸭子类型）。价值在类型检查/IDE/文档，运行时是空壳。
- `AgentCandidate` = Agent + 它的 Decision，把"谁"和"多自信/为什么"绑一起，供 Coordinator 排序和记录理由。

### 筛选链路

```text
Task.required_capabilities
  -> Registry 按 AgentProfile.capabilities 过滤（_has_required_capability）
  -> Agent.decide(task, board)
  -> 只保留 claim=True -> AgentCandidate(agent, decision)
  -> 按 decision.confidence 降序
  -> EventDrivenCoordinator 再做最终调度
```

- 任务无 `required_capabilities` 时人人过关，但仍要自己决定是否认领。
- **能力比较的类型细节**：任务侧存的是枚举的 `.value` 字符串，Agent 侧是枚举，所以 `_has_required_capability` 先把 Agent 能力 `{capability.value}` 转字符串集合再 `issubset`。这是"上游类型形状不一致、registry 当翻译层"的实例。

### confidence 全是手写启发式常量

没有一个是从数据算出来的；数值表达**相对可信度/优先级**（供 Coordinator 排序、验收），不是真实概率。档位表：

| 场景                              | 值                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------- |
| safety_review / critique 审查产物 | 0.95（"全项目最高档，安全优先"）                                                |
| risk 产物（检测到高危信号）       | 0.98                                                                            |
| risk 产物（无高危信号）           | 0.84                                                                            |
| context 产物                      | 0.88                                                                            |
| response_proposal 产物            | 0.86                                                                            |
| intent 产物（RISK 意图）          | 0.92 / 0.78                                                                     |
| 各 Agent 认领置信度               | Understanding 0.82、Safety 0.95/0.8、Context 0.86/0.82、Response 0.78/0.84/0.65 |

**区分两个 confidence**：`decide` 里的（进 `AgentDecision`）= 认领置信度，决定抢任务谁优先；`act` 里 `_artifact(..., confidence)`（进 `AgentArtifact`）= 产物可信度，标记结论质量。不同对象、不同字段，只是都叫 confidence。

---

## 四、产物与 payload

### 5 种 kind（产物类型）+ 生产者/消费者

| kind                             | 生产者                              | payload 关键字段                                                                                    | 消费者                                    |
| -------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `intent`                       | UnderstandingAgent                  | `intent` / `topic` / `reason`                                                                 | Coordinator 派任务、Context/Response 读取 |
| `risk`                         | SafetyAgent（`_assess_risk`）     | `risk` / `emotion` / `emotionScore` / `confidence` / `summary` / `assessment`           | `_risk_level` 汇总、各 Agent            |
| `context`                      | ContextAgent                        | `memoryBrief` / `modelHistory` / `knowledgeQuery` / `retrievedKnowledge` / `skillContext` | ResponseAgent 生成回复                    |
| `response_proposal`            | ResponseAgent                       | `messages` / `mode` / `intent` / `risk`                                                     | SafetyAgent 审查、Coordinator 采纳        |
| `safety_review` / `critique` | SafetyAgent（`_review_response`） | `approved` / `reason` / `responseArtifactId` / `risk`                                       | Coordinator`_try_accept_final`          |

- **每个 kind 恰好一个生产者**（全项目唯一），责任划分清晰。
- `_artifact` 造产物：id 用 `"{Agent名}:{kind}:{随机串}"` 保证唯一，owner=自己，task_id=当前任务。
- **payload 各 Agent 各不相同、自给自足**：每个产物给不同消费者用、各取所需。`privateMemoryKey` 虽同名但指向各自记忆；`response_proposal` 里的 `intent/risk` 是"生成时的快照"（防 context 缺失时丢上下文），不是结论本体。强行统一结构反而让每个产物带一堆无用字段——**"各环节只装自己需要的"是特性不是缺陷**。

### `latest_artifact(kind)` 按 kind 查

`latest_artifact("intent")` 只在 **intent 这一类**里找最新，`latest_artifact("risk")` 在 **risk 类**里找——两把独立的钥匙查两个箱子，**两类产物可以同时存在**。所以 `if not latest_artifact("intent") or not latest_artifact("risk")` 的意思是"intent 缺 **或** risk 缺就不认领"，不是"同一产物既是 intent 又是 risk"。

---

## 五、意图与风险（两个"RISK"别搞混）

### IntentType vs RiskLevel

|          | UnderstandingAgent 的`intent`        | SafetyAgent 的`risk`                      |
| -------- | -------------------------------------- | ------------------------------------------- |
| 枚举     | `IntentType`（CHAT/CONSULT/RISK）    | `RiskLevel`（LOW/MEDIUM/HIGH）            |
| 回答     | 用户这轮**想干什么**（怎么处理） | 用户**危不危险**（处置力度）          |
| 依据     | 关键词 + LLM 分类                      | `PsychologicalAssessmentService` 专业评估 |
| 互不借用 | 不读对方产物                           | **独立评估（不信任链）**              |

- **风险评估评的是用户输入，不是回复**；`_assess_risk` 喂 `board.model_input` + 黑板历史。
- **UnderstandingAgent 判 RISK 后，SafetyAgent 仍必须独立评估风险**——两者维度不同，意图分类不能替代安全把关。

### `_intent` / `_risk_level`（模块级汇总函数）

- `_intent(board)`：**优先取 intent 产物的结论**（`payload["intent"]` 转枚举，脏数据兜底 CHAT）；没有产物 → 关键词兜底（高危→RISK，咨询→CONSULT，否则 CHAT）。
- `_risk_level(board)`：**扫所有 risk 产物取最高档**（可能有多个，append 不覆盖）+ **事件流里出现过 SAFETY_OVERRIDE 则直接强制 HIGH**（一票升级）。
- **`_intent` 的兜底不是摆设**：第一轮 `_derive_missing_work` 在 UnderstandingAgent act 之前跑，那时**必然没有 intent 产物**；ContextAgent.decide 等入口没前置检查，第一轮就调 `_intent` 必须靠兜底。而在 ResponseAgent 入口（先过了 `latest_artifact("intent")` 检查）确实走不到兜底——**兜底是给"不确定产物在不在"的调用方兜的**。
- coordinator.py 里还有一份几乎一样的 `_intent_value` / `_risk_value`（复制粘贴痕迹，两文件各一份，逻辑一致不共享）。

### SAFETY_OVERRIDE 的消费点（事件参与决策的完整案例）

事件在任务板不被认领，而是被 3 处**事件流扫描**读取后改决策：

| 位置                                 | 响应                                                          |
| ------------------------------------ | ------------------------------------------------------------- |
| `_risk_level`（autonomous.py:805） | 强制返回`RiskLevel.HIGH`                                    |
| runtime`_select_intent`（:120）    | 强制返回`IntentType.RISK`（推翻 UnderstandingAgent 的意图） |
| runtime`_select_risk`（:140）      | 强制返回`RiskLevel.HIGH`                                    |

场景：硬关键词没命中（用户没说高危词），但 LLM 评估高危 → SafetyAgent 发 SAFETY_OVERRIDE。**任务靠认领驱动干活，事件靠扫描驱动决策**——两种机制。

---

## 六、AgentMessage 消息机制

### kind 4 类 + 广播/定向

| kind                  | 谁发               | 发给谁           | 含义               |
| --------------------- | ------------------ | ---------------- | ------------------ |
| `PROPOSAL`          | UnderstandingAgent | `*`（广播）    | 意图提议，谁都要看 |
| `SAFETY_ASSESSMENT` | SafetyAgent        | CoordinatorAgent | 风险评估结论       |
| `CONTEXT_READY`     | ContextAgent       | ResponseAgent    | 上下文就绪         |
| `REVIEW_REQUEST`    | ResponseAgent      | SafetyAgent      | 请审查候选回复     |

- `recipient="*"` = 广播，`recipient="某Agent名"` = 定向。**只有 UnderstandingAgent 广播**（意图结论所有环节都可能用），其他都是定向给"需要它"的角色。
- 消费端 `messages_for(agent_name)`：`recipient in {agent_name, "*"}`。

### 消息不触发执行（黑板协作的关键认知）

- `AgentMessage` 只是**发讯息**（append 进黑板，谁爱读谁读），**不触发接收方执行**。没有任何机制让"收到消息→立刻 act"。
- **真正驱动执行的是任务认领**：`AgentTask` 挂黑板 → Coordinator 轮询 `decide` → Agent 认领 → act。
- 所以"打回回复"时**不需要 AgentMessage**：打回时派生修订任务（`task:revise-response`，title/description/metadata 带理由和 `revisionOf`），ResponseAgent 读黑板看到新任务自动认领重写。**任务驱动是主通道，消息只是可选的信息补充**（加不加不影响流程）。

---

## 七、记忆体系（Redis）

### 三层存储，别混

| 记忆           | 存哪                                        | 谁负责                        | 生命周期        |
| -------------- | ------------------------------------------- | ----------------------------- | --------------- |
| 会话对话记忆   | Redis                                       | `RedisShortTermMemoryStore` | 24h TTL + 40 条 |
| Agent 私有记忆 | Redis（**纯 Redis，数据库完全没存**） | `AgentPrivateMemory`        | 同上一套规则    |
| 聊天历史档案   | MySQL`chat_messages`                      | harness`save_message`       | 永久            |

- `AgentPrivateMemory` 是"Redis 里专属 Agent 的一格"，底层**复用** `RedisShortTermMemoryStore`（同 TTL/裁剪规则），只传不同 key。
- 私有记忆 role **固定 `"system"`**（Agent 给自己的批注，不是真实对话）；会话记忆才有 user/assistant。

### key 结构（同一前缀、不同后缀）

```python
# 会话记忆
mindbridge:short-term-memory:{会话id}
# Agent 私有记忆（AgentPrivateMemory._key 返回 "agent:名字:会话id"，进 store 后再套一层前缀）
mindbridge:short-term-memory:agent:{Agent名}:{会话id}
```

- **前缀叠加**：`AgentPrivateMemory._key` 返回 `agent:名字:会话id`（中间形态），`RedisShortTermMemoryStore._key` 再拼 `mindbridge:short-term-memory:` 前缀 → 最终是长 key。上一轮我曾误说"前缀不同"，实际是**同一前缀、后缀不同**。
- **`_key(session_public_id)` 参数名名不副实**：它实际是"任意 key 后缀"。ContextAgent 传会话 id 读会话记忆；`AgentPrivateMemory` 传 `agent:名字:会话id` 读该 Agent 记忆。**同一个方法，喂什么 key 读什么数据**。
- 记忆按 `agent + 会话` 切片，各 Agent 互不串台。

### Redis 命令三件套（都围绕"列表尾部"）

```python
append:  rpush(key, payload)                          # 新消息推尾部
         ltrim(key, -max_messages, -1)                # 只留尾部最近 N 条，裁掉更早的
         expire(key, ttl_seconds)                     # 每次续期 → 滑动过期
load:    lrange(key, -limit, -1)                      # 取尾部最近 N 条
```

- **`lrange(key, -limit, -1)`**：Redis 列表下标支持负数（-1=最后一个），`-limit..-1` = 从倒数第 limit 个到末尾 = **取最后 limit 条**。
- **为什么裁剪**：只读最近 N 条，更早的留着白占内存 + 读慢；长期档案有 MySQL 兜底。
- **`expire` 滑动过期**：每次 append 重设 24h，活跃则永生、冷 24h 自灭。
- `json.dumps`（对象→字符串）与 `json.loads`（字符串→对象）是这套存取的核心；`_serialize` 内还有脱敏 + 时间戳。

---

## 八、模型路由与 AI 客户端

### `AgentProfile` 三个"摆设字段"

`memory_policy` / `model_profile` / `tool_permissions` **全项目无消费点**（grep 只有定义+赋值）。真实机制被别的代码顶替：

| 设计意图（字段）     | 真正干活的机制                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| `memory_policy`    | `AgentPrivateMemory._key` 用 `agent_name`，不读它                                               |
| `model_profile`    | `AGENT_MODEL_ALIASES` 映射表（agent_models.py:11），value 与字段值**一字不差 = 信息写两遍** |
| `tool_permissions` | 无工具权限检查机制，纯预留                                                                          |

定性：**不是 bug，是"声明了没接线"的半成品/冗余声明**——有文档价值，但改字段不改变行为，且会误导读者（判断法：grep 有没有消费点）。与 `AgentClaim`/`TASK_RELEASED`/`reopen`/`COORDINATION` 同属一类预留。

### 模型路由链路

```text
BaseAutonomousAgent.client() → model_registry.client_for(name)
  → AGENT_MODEL_ALIASES 拿 alias（没在表里则 _snake 兜底）
  → 按 agent_model_{alias}_{provider/model/temperature/max_tokens} 查配置（无则默认值）
  → 浅拷贝 settings 覆写模型配置 → new AiClient
```

- **每个 Agent 一个模型档位**：provider（公司）、model、temperature、max_tokens 都可不同；按 `if provider=="openai"` 分流写 `openai_model` / `ollama_model`。
- 为什么这样做：任务难度不同（意图分类用便宜快模型、回复生成用强模型），一个 Agent 一档。
- `AgentRuntimeServices` 里的 `ai` 字段：构造传了但 **agents 内部没人直接用**（都走 `model_registry`），又一个"传了没消费"。

### `complete` vs `stream`

|           | `complete()`                       | `stream()`                        |
| --------- | ------------------------------------ | ----------------------------------- |
| 同步/异步 | 同步                                 | 异步 async                          |
| 返回      | 一次性完整字符串（`stream=False`） | 逐 token`yield`                   |
| 用途      | Agent 内部判断（意图/评估/审查）     | 给前端逐字输出的最终回复（chat.py） |

**规则：给用户看的 → 流式；给代码判断的 → 一次性完整。** 中间过程拿一半没法用（`"RISK" in label` 要全量），所以 Agent 全走 `complete`。

### `PromptTemplates` 是 @staticmethod 工具类

- 无字段、无 `__init__`，纯"打包三个拼 prompt 的函数"的命名空间 → 全静态方法，`PromptTemplates.intent_prompt(...)` 直接类名调用。
- 三个方法：`intent_prompt`（意图分类）、`psychology_prompt`（心理评估 JSON）、`answer_system_prompt`（最终回复，按 intent 分闲聊/心理关怀两套话术，高危追加危机规则）。
- **两条 system 消息的张力**：Agent 拼 messages 时 `*PromptTemplates.intent_prompt(...)`（列表解包拍平）+ 自己的 profile.system_prompt。一个说"只输出标签"，一个说"输出完整结论"——靠调用方 `"RISK" in label` 的**包含匹配**兜住，模型多输出不崩。

---

## 九、协作调度与角色分工

### 串行而非并行

- Coordinator 的 `for task, candidate in candidates: result = candidate.agent.act(...)` 是**同步逐个执行**：一个跑完 `apply_turn_result` 写黑板，下一个才读最新状态。**同一进程内的顺序 Claim 协作，不代表并行**。
- 串行的价值：后一个 Agent 能依赖前一个的产物（ContextAgent 等 UnderstandingAgent 产出 intent 后才能干）。
- 真正的异步只在"给用户流式输出"那层（`stream()`），不是多 Agent 并行。

### 各 Agent 的 decide 一览（认领判断）

**UnderstandingAgent**：意图产物已有→弃；被指派（required_capabilities 或 root/understanding kind）→0.82 认领。
**SafetyAgent**（四分支，优先级从高到低）：

1. 有未审的候选回复 → 审查（0.95，全项目最高，安全优先）
2. 缺 risk 产物且用户有输入 → 评估（高危 0.98 / 普通 0.84）
3. 任务点名 SAFETY → 接单（0.8）
4. 都没得干 → 放弃
   **ContextAgent**：context 已有→弃；点名 CONTEXT→0.86；`needs_context`（意图 CONSULT/RISK **或** 风险 MEDIUM/HIGH）→0.82；否则弃。
   **ResponseAgent**（四分支）：无 intent 或 risk→弃；CHAT+LOW→0.78 普通回复；有 context 或 HIGH→0.84 支持回复；点名 RESPONSE→0.65 兜底；否则等 context。
   **CoordinatorAgent**：**decide 恒 False**（"由事件循环驱动，不是固定工作流槽位"）——不下场抢活。

### `_derive_missing_work`（Coordinator 的派活引擎）

每轮检查黑板**缺哪些产物就派生对应任务**（缺啥派啥，不写死 Agent 链）：

```text
缺 intent → task:understand（UNDERSTANDING）
缺 risk   → task:assess-safety（SAFETY，高危 CRITICAL）
needs_context 且缺 context → task:gather-context（CONTEXT）
前置齐 且缺 response → task:propose-response（RESPONSE）
有 response 但没审 → task:review-response（SAFETY）
有 critique(打回)  → task:revise-response（RESPONSE）
```

- **回复任务有前置门槛**（`can_request_response`）：intent、risk 都齐 +（不需 context 或 context 齐或 HIGH）才派。
- **派 intent 任务与 `_intent_value` 兜底是两条独立的路**：intent 任务派发只看 `latest_artifact("intent")` 有没有（没有就派），兜底结果只用于决定"要不要派 context"——所以第一轮兜底成 CHAT 不影响 intent 任务照常派给 UnderstandingAgent。
- `_ensure_task*` 只负责防重（`if task.id in board.tasks` 返回）和记 `TASK_CREATED`。

### CoordinatorAgent vs EventDrivenCoordinator（两个对象）

- **`CoordinatorAgent`**（autonomous.py）：只是调度器的"辅助工具人"——`root_task()` 建本轮总任务（id 固定 `task:root`，kind="root" 让 UnderstandingAgent 兜底认领，高危 CRITICAL）、`remember_acceptance()` 记采纳记忆。`act()` 返回 `close_task=False`（reopen 预留，走不到）。**不在注册表**（agents 只有 4 个），decide 恒 False。
- **`EventDrivenCoordinator`**（coordinator.py）：真正干活——`run()` 调度循环、`_derive_missing_work`、`_claim_candidates`、`_try_accept_final`。
- 为什么分开：协调者不下场抢活（否则"谁来协调协调者"悖论）；`COORDINATION` 能力是空头。

### 最终采纳 `_try_accept_final` + `accept_final`

采纳前提（全部满足）：有 response_proposal、有对应 safety_review（`metadata["responseArtifactId"] == response.id`）、审查通过、回复置信度 ≥ 0.6。`accept_final` 设 `final_artifact_id` + 记 `FINAL_ACCEPTED`；`reason` 是写死的说明文字（"accepted after ... SafetyAgent approval"），**不参与判定**，只给人看/进记忆。

---

## 十、安全审查与 critique

### `_review_response`（审查判定）

- **risk 是审查开关**：`risk == HIGH` 才检查回复是否含安全引导词（"高风险处理规则/当前安全/可信任的人/紧急"），缺 → 打回（`approved=False`）；非 HIGH → 直接放行。**风险越高审查越严**。
- 打回后果：产 `critique` 产物（kind 分支 `safety_review` if approved else `critique`）+ 记 `REVISION_REQUESTED` 事件 + 派生修订任务 `task:revise-response`（metadata `revisionOf=被否决回复id`、required_capabilities=RESPONSE）。
- **`revisionOf` 的意义**：让修订任务**绕过"已有回复就不重复生成"的防重复检查**——ResponseAgent.decide 第一分支 `if latest_artifact("response_proposal") and "revisionOf" not in task.metadata` 才拒绝认领；带 revisionOf 的修订任务放行去重写。

### 为什么"评估"和"审查"不构成自审

- risk 产物**只有 SafetyAgent 产**（`_assess_risk`），`_risk_level` 读的正是它——但评估与审查是**两个任务、两个时间点**：先评估用户输入产 risk（客观事实），后审查回复时读这份 risk 决定严苛度。**risk 反映用户状态而非自己写得好不好，没有循环论证**。

### 已知安全缺口

`_review_response` 审查的是 `response_proposal.payload["messages"]`（交给模型的候选 Prompt），**不是模型最终生成的文本**。最终文本若偏离 Prompt，当前没有生成后二次审核闭环——项目已知缺口。

---

## 十一、生命周期与 Trace

### turn / session / 黑板

- **turn** = 一次用户请求，每次 `runtime.run()` 新建 `turn_id=uuid4` + **新黑板**。
- **session** = 一整段对话，跨很多轮；前端首次发消息建 `ChatSession` 入库，之后带 session_id 复用（harness `_resolve_session`）。
- 结论：**一个 turn 对应一个新黑板**。黑板是临时工作台，turn 结束被 `_to_result` 读走打包成 `AgentRunResult`。
- 一个黑板通常 5~8 个任务：1 个 root + intent/risk/context/response 各环节 + 审查/修订任务，逐个认领处理。

### Trace 三级沉淀

```text
黑板 events(内存证据) → AgentStep + collaboration_*(内存报告) → agent_run_traces 表(数据库档案)
```

- Trace = 一次运行的**完整快照**（`AgentTraceService.save_run`）：user/session/report 关联 + 输入 + intent/risk + `agent_steps_json`（步骤+事件+任务+产物合并）+ 检索/回复/评估。写入后只回溯查看。

### `AgentRuntimeServices` 与 `BaseAutonomousAgent`

- `AgentRuntimeServices`：**公共资源背包**——db/settings/user/session/ai/model_registry/memory/private_memory/knowledge，Runtime 一次构造、5 个 Agent 共享注入。保证一致性、资源只建一次、可测试注入替身。
- `BaseAutonomousAgent`：公共基类，提供 `client()` / `private_memory()` / `remember()` / `_artifact()`。`profile` 是**类属性 + 类型注解**——子类用 `profile = AgentProfile(...)` 覆盖（类属性共享一份，因为 Agent 身份是固定常量，不必每实例重建）；基类不 init 它，类型注解只服务 IDE/类型检查。

---

## 面试自测题

1. 为什么 MindBridge 选择共享黑板，而不是把所有中间结果塞进一个 Prompt？
2. Claim-based 调度与固定顺序 Workflow 的核心差异是什么？
3. 这些 Agent 是否只是不同 Prompt？请从状态、能力、私有记忆和产物协议回答。
4. 为什么该 Runtime 不是经典 ReAct，也不是 Function Calling？
5. 为什么最终采纳必须校验 `safety_review.metadata["responseArtifactId"] == response.id`，不能只看"存在一个 approved review"？
6. `CoordinatorAgent` 与 `EventDrivenCoordinator` 为什么要分成两个对象？

## 当前边界与验证

- 黑板"append-only"主要体现在消息/Artifact/事件追加；`tasks` 用同 id 覆盖，不能笼统声称所有字段纯追加。
- `AgentClaim` 只有数据类定义，无生产者/消费者；不能因为类已定义就声称运行时保存独立 Claim 对象。
- 多 Agent 是**同一进程内顺序 Claim 协作**，不代表并行计算。
- `AgentProfile` 的 memory_policy/model_profile/tool_permissions 无消费点；`AgentRuntimeServices.ai` 也传了没用。
- 2026-08-16 运行 `tests.test_event_driven_multi_agent`：5/5 通过（黑板快照、Artifact 追加、候选排序、协调器用 Claim、模型覆盖）。
- Docker 未运行 → 真实 Compose / Trace 状态待恢复后复验；阶段三收尾需用真实 Trace 对照 Task/Claim/Artifact/Event/采纳。

## 提问记录约定

每次要求学习者回答的阅读题、复述题和面试题同步写入本知识卡；回答后的关键校准结论紧跟对应问题记录。知识卡既是复习材料，也是后续学习的题目索引。
