from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from app.agents.events import (
    AgentArtifact,
    AgentEvent,
    AgentEventType,
    AgentMessage,
    AgentTask,
    AgentTurnResult,
    CollaborationBlackboard,
    TaskPriority,
)
from app.agents.registry import AgentCapability, AgentDecision, AgentProfile
from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.agent_models import AgentModelRegistry
from app.services.ai import (
    AiClient,
    PromptTemplates,
    has_consult_signal,
    has_high_risk_signal,
)
from app.services.assessment import PsychologicalAssessmentService

if TYPE_CHECKING:
    from app.models.entities import ChatSession, UserAccount
    from app.services.knowledge import KnowledgeService, SearchResult
    from app.services.memory import RedisShortTermMemoryStore


# UnderstandingAgent（意图理解 Agent）用来快速判断用户这轮输入属于什么话题的一组关键词
# 非心理话题常见词，UnderstandingAgent拿它做廉价的快速分类（命中就不用调 LLM），或做标签兜底
GENERAL_TASK_WORDS = [
    "java",
    "python",
    "javascript",
    "代码",
    "编程",
    "程序",
    "算法",
    "数据库",
    "spring",
    "maven",
    "前端",
    "后端",
    "项目",
    "接口",
    "bug",
    "报错",
    "作业",
    "论文",
    "翻译",
    "总结",
    "解释",
    "怎么写",
    "如何",
    "是什么",
    "为什么",
    "给我",
    "帮我",
    "推荐",
    "查询",
    "天气",
    "路线",
]


# "每个 Agent 的公共资源背包"
# 把一次请求里 Agent 可能用到的一切外部依赖（数据库、配置、用户、会话、模型、记忆、知识库）打包成一个对象
# 构造时统一注入，所有 Agent 共享这一个实例
@dataclass
class AgentRuntimeServices:
    db: Session                                                     # SQLAlchemy 会话（可查 MySQL 表，如 chat_messages）
    settings: Settings                                              # 全局配置对象（Redis 地址、模型、记忆条数等）
    user: UserAccount                                               # 当前用户账号实体（如 display_name）
    session: ChatSession                                            # 当前会话实体（含 public_id 供记忆/查库定位）
    ai: AiClient                                                    # 原始 AI 客户端（agents 实际多用 model_registry 按名取）
    model_registry: AgentModelRegistry                              # 按 Agent 名分配模型档位的注册表
    memory: RedisShortTermMemoryStore                               # 短期对话记忆存取（session 维度，最近 N 条）
    private_memory: "AgentPrivateMemory"                            # 各 Agent 隔离的私有记忆（agent:名字:会话id 维度）
    knowledge: KnowledgeService                                     # 知识检索服务（RAG 查询知识库）


# 各 Agent 隔离的私有记忆门面：底层复用 Redis 短期记忆存储，按 "agent:名字:会话id" 切分 key
# 让每个 Agent 记住自己的历史结论，且互不串台（同一 Agent 不同会话也隔离）

# Agent 私有记忆的内容是纯 Redis、纯临时的，数据库完全没存。
# 这跟它的定位一致——私有记忆是"Agent 这阵子的工作笔记"，过期删了不心疼，反正有聊天历史兜底
class AgentPrivateMemory:

    def __init__(self, settings: Settings):
        from app.services.memory import RedisShortTermMemoryStore

        # 造出取记忆的工具
        self.store = RedisShortTermMemoryStore(settings)

    # 读取指定 Agent 在某会话的私有记忆（按时间倒序最近的若干条）
    def load(self, agent_name: str, session_public_id: str) -> list[AiMessage]:
        return self.store.load_recent(self._key(agent_name, session_public_id))

    # 追加一条"系统角色"记忆到该 Agent 该会话（Agent 内部备注，不对外展示）
    def append(self, agent_name: str, session_public_id: str, content: str) -> None:
        self.store.append(self._key(agent_name, session_public_id), "system", content)

    # 生成该 Agent 该会话在 Redis 里的唯一 key（agent:名字:会话id）
    def _key(self, agent_name: str, session_public_id: str) -> str:
        return f"agent:{agent_name}:{session_public_id}"


# 所有 Agent 的公共基类：注入公共服务包 + 提供取模型、读写私有记忆、造产物等公共能力
# 5 个具体 Agent（Understanding/Safety/Context/Response/Coordinator）都继承它，只各自实现 decide / act
class BaseAutonomousAgent:

    # 纯粹是"声明有这个属性"；真正赋值全靠子类覆盖类属性
    # 因为它是每个 Agent 固定不变的"身份证"，没必要在实例化时重复构造
    profile: AgentProfile                                                    # 该 Agent 的静态画像（名字/能力/提示词等），由子类定义

    def __init__(self, services: AgentRuntimeServices):
        self.services = services                                             # 公共服务包：db/settings/user/session/模型/记忆/知识库

    @property
    def name(self) -> str:                                                   # Agent 名（来自画像）
        return self.profile.name

    # 按 Agent 名从模型注册表取对应的 AI 客户端（每个 Agent 用自己档位的模型）
    def client(self) -> AiClient:
        return self.services.model_registry.client_for(self.name)

    # 读取该 Agent 在当前会话的私有记忆（最近若干条）
    def private_memory(self) -> list[AiMessage]:
        return self.services.private_memory.load(
            self.name, self.services.session.public_id
        )

    # 往该 Agent 的私有记忆里追加一条备注（role 固定 system）
    def remember(self, content: str) -> None:
        self.services.private_memory.append(
            self.name, self.services.session.public_id, content
        )

    # 造一个业务产物：id 用 "Agent名:类型:随机串" 保证唯一，挂到当前任务
    def _artifact(
        self,
        kind: str,                                             # 产物类型：intent / risk / context / response_proposal / critique
        payload: dict[str, Any],                               # 产物内容（业务数据）
        task: AgentTask,                                       # 当前任务，产物挂到它的 task_id 上
        confidence: float = 1.0,                               # 该产物的置信度 0~1，默认 1.0
        metadata: dict[str, Any] | None = None,                # 附加元数据，可为空
    ) -> AgentArtifact:
        return AgentArtifact(
            id=f"{self.name}:{kind}:{uuid.uuid4().hex[:10]}",
            owner=self.name,
            kind=kind,
            payload=payload,
            confidence=confidence,
            task_id=task.id,
            metadata=metadata or {},
        )


class UnderstandingAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="UnderstandingAgent",
        capabilities=frozenset({AgentCapability.UNDERSTANDING}),
        system_prompt=(
            "你是 UnderstandingAgent。你只负责理解用户当前请求，输出意图、主题、置信度和理由，"
            "不生成最终回复，不做风险处置。"
        ),
        memory_policy="private_intent_history",
        model_profile="understanding",
        tool_permissions=frozenset({"llm.intent"}),
    )

    # 判断是否认领任务：意图产物已存在则放弃；任务被明确指派（含 required_capabilities / root / understanding）则 0.82 认领
    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("intent"):
            return AgentDecision(False, reason="intent artifact already exists")
        if self._is_directed(task, board):
            return AgentDecision(True, 0.82, "open user-turn task needs understanding")
        return AgentDecision(False, reason="task does not need understanding")

    # 执行意图理解：分类意图、打话题标签、产出 intent 产物并广播消息、把结论记入私有记忆
    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:

        # 兜底取数：优先用脱敏后的 model_input；如果它为空（比如某些场景没构造），退回用原始的 user_input
        # 因为要喂给 LLM/分类器的是脱敏版（不该把用户隐私发给模型），所以 model_input 是首选
        intent = self._classify(board.model_input or board.user_input, board)
        confidence = 0.92 if intent == IntentType.RISK else 0.78        # 作为产物的可信度标记，随 intent 产物存进黑板
        payload = {
            "intent": intent.value,
            "topic": self._topic(board.model_input or board.user_input),
            "reason": "high risk hard signal"
            if intent == IntentType.RISK
            else "autonomous intent proposal",
            "privateMemoryKey": self.services.private_memory._key(
                self.name, self.services.session.public_id
            ),
        }

        self.remember(f"intent={intent.value}; topic={payload['topic']}")

        return AgentTurnResult(
            artifacts=(self._artifact("intent", payload, task, confidence),),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="*",                                          # *（广播）
                    task_id=task.id,
                    kind="PROPOSAL",
                    content=f"我判断本轮意图是 {intent.value}",
                ),
            ),
        )

    # 判断任务是否明确要求意图理解：required_capabilities 含 UNDERSTANDING，或任务是 root/understanding 类型且用户有输入
    def _is_directed(self, task: AgentTask, board: CollaborationBlackboard) -> bool:
        if AgentCapability.UNDERSTANDING.value in task.required_capabilities:
            return True
        return bool(

            # root = 这一轮请求的根任务（id 固定 task:root）
            # root 任务没有声明任何 required_capabilities（它是"总任务"，不该被单一能力卡住）
            # UnderstandingAgent 是"最需要先处理它"的 Agent
            board.user_input and task.metadata.get("kind") in {"root", "understanding"}
        )

    # 把用户文本分类成意图：先走关键词硬判断（RISK/CHAT），再兜底调 LLM 判断
    def _classify(self, text: str, board: CollaborationBlackboard) -> IntentType:
        lowered = text.lower()

        if has_high_risk_signal(lowered):
            return IntentType.RISK
        if not has_consult_signal(lowered) and any(
            word in lowered for word in GENERAL_TASK_WORDS
        ):
            return IntentType.CHAT
        
        try:
            # 作为"私有记忆"上下文喂给 LLM 做意图判断
            memory_context = "\n".join(
                # 原因是"给 LLM 的上下文要精炼"：意图分类只需要最近几条记忆就够了，全塞 40 条会撑爆 prompt
                item.content for item in self.private_memory()[-6:]
            )

            messages = [
                # * 把返回的列表展开成元素塞进外层列表
                *PromptTemplates.intent_prompt([], text),
                AiMessage(
                    role="system",
                    content=f"{self.profile.system_prompt}\n私有记忆：\n{memory_context or '无'}",
                ),
            ]
            label = self.client().complete(messages).upper()
            if "RISK" in label:
                return IntentType.RISK
            if "CONSULT" in label:
                return IntentType.CONSULT
            if "CHAT" in label:
                return IntentType.CHAT
        except Exception:
            # 中间环节挂了，绝不阻塞主流程，用兜底值继续走。
            # 意图分类是"最好能知道，但不知道也能凑合继续"的场景
            pass

        # ① 异常抛出
        # ② LLM 返回的 label 三个都不含
        return IntentType.CONSULT if has_consult_signal(lowered) else IntentType.CHAT

    # 给用户文本打话题标签：safety / mental_health_support / general_task / conversation
    def _topic(self, text: str) -> str:
        lowered = text.lower()
        if has_high_risk_signal(lowered):
            return "safety"
        if has_consult_signal(lowered):
            return "mental_health_support"
        if any(word in lowered for word in GENERAL_TASK_WORDS):
            return "general_task"
        return "conversation"

# SafetyAgent 工作两个时机：评估用户输入风险（产 risk）+ 审查候选回复（产 safety_review/critique）
class SafetyAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="SafetyAgent",
        capabilities=frozenset({AgentCapability.SAFETY}),
        system_prompt=(
            "你是 SafetyAgent。你独立评估风险，并审查候选回复是否安全。"
            "你可以发布 SAFETY_OVERRIDE；你不生成最终回复。"
        ),
        memory_policy="private_safety_ledger",
        model_profile="safety",
        tool_permissions=frozenset({"llm.risk", "rules.high_risk", "response.review"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        latest_response = board.latest_artifact("response_proposal")                            # 最新的候选回复
        latest_review = board.latest_artifact("safety_review")                                  # 最新的审查结论

        if latest_response and (
            latest_review is None
            or latest_review.metadata.get("responseArtifactId") != latest_response.id
        ):
            # 这份候选回复还没被审过 → SafetyAgent 认领审查任务，置信度 0.95（全项目最高档，安全优先）
            return AgentDecision(True, 0.95, "candidate response needs safety critique")

        # 风险评估的产物是 risk（SafetyAgent 之前评估产出的）。如果黑板上已经有 risk 产物，说明风险评估已经做过了
        if not board.latest_artifact("risk") and board.user_input:
            confidence = 0.98 if has_high_risk_signal(board.user_input) else 0.84
            return AgentDecision(
                True, confidence, "user input needs independent risk assessment"
            )

        # 兜底保险——只要任务明确点名要安全能力，就接
        if AgentCapability.SAFETY.value in task.required_capabilities:
            return AgentDecision(True, 0.8, "task explicitly asks for safety")
        return AgentDecision(False, reason="no safety work needed")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        response = board.latest_artifact("response_proposal")                                   # 最新的候选回复
        review = board.latest_artifact("safety_review")                                         # 最新的审查结论
        if response and (
            review is None or review.metadata.get("responseArtifactId") != response.id
        ):  
            return self._review_response(task, board, response)                                 # 审查候选回复

        return self._assess_risk(task, board)                                                   # 风险评估   

    def _assess_risk(
        self, task: AgentTask, board: CollaborationBlackboard
    ) -> AgentTurnResult:

        # 调用心理评估服务做风险评估：喂入组装好的输入 + 黑板上下文历史，产出 risk/emotion/置信度等评估结果
        assessment = PsychologicalAssessmentService(self.client()).assess(
            board.model_input or board.user_input, _context_history(board)
        )
        
        payload = {
            "risk": assessment.risk.value,                                     # 风险等级 LOW/MEDIUM/HIGH，供 _risk_level 汇总和后续处置
            "emotion": assessment.emotion.value,                               # 情绪类型，进心理报告
            "emotionScore": assessment.emotion_score,                          # 情绪严重度分数
            "confidence": assessment.confidence,                               # 评估结果的置信度 0~1
            "summary": assessment.summary,                                     # 评估摘要（人话，供后续环节/报告阅读）
            "assessment": assessment,                                          # 完整评估对象，供 trace/报告取用更多细节
            "privateMemoryKey": self.services.private_memory._key(             # 该评估结果在私有记忆中的 key（供回溯定位）
                self.name, self.services.session.public_id
            ),
        }
        
        events: tuple[AgentEvent, ...] = ()

        # 如果风险等级为高
        if assessment.risk == RiskLevel.HIGH:
            events = (
                AgentEvent(
                    type=AgentEventType.SAFETY_OVERRIDE,                        # 升级高风险
                    actor=self.name,
                    task_id=task.id,
                    message="RiskGuardian hard/LLM assessment raised this turn to HIGH",
                    metadata={"risk": RiskLevel.HIGH.value},
                ),
            )

        # 不管什么风险等级，都保存到Redis的短期记忆里面
        self.remember(f"risk={assessment.risk.value}; summary={assessment.summary}")
        
        return AgentTurnResult(
            artifacts=(self._artifact("risk", payload, task, assessment.confidence),),
            events=events,
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="CoordinatorAgent",
                    task_id=task.id,
                    kind="SAFETY_ASSESSMENT",
                    content=f"risk={assessment.risk.value}",
                ),
            ),
        )

    # 检查 ResponseAgent 产出的回复（response_proposal）是否符合安全要求
    def _review_response(
        self, task: AgentTask, board: CollaborationBlackboard, response: AgentArtifact
    ) -> AgentTurnResult:
        
        risk = board.risk_value()                                                       # 决定了"关键词检查要不要执行"
        messages = response.payload.get("messages", [])                                 # 取出候选回复的消息列表
        combined = "\n".join(
            getattr(message, "content", str(message)) for message in messages
        )

        # 先假设通过：默认认为"这份回复符合当前安全约束"。后面如果触发条件才改为不通过。
        approved = True
        reason = "response proposal satisfies current safety constraints"

        # ~关键词检查~
        # 高危场景下，回复必须给出"安全引导"（联系可信任的人、紧急处理规则等）。
        # 如果回复通篇没有这些，说明它没履行高危处置职责 → 打回（approved=False）
        if risk == RiskLevel.HIGH and not any(
            word in combined
            for word in ["高风险处理规则", "当前安全", "可信任的人", "紧急"]
        ):
            approved = False
            reason = "high-risk response proposal lacks immediate safety guidance"
        payload = {
            "approved": approved,
            "reason": reason,
            "responseArtifactId": response.id,
            "risk": risk.value,
            "privateMemoryKey": self.services.private_memory._key(
                self.name, self.services.session.public_id
            ),
        }

        kind = "safety_review" if approved else "critique"

        events: tuple[AgentEvent, ...] = ()
        follow_up_tasks: tuple[AgentTask, ...] = ()
        if not approved:
            events = (
                AgentEvent(
                    type=AgentEventType.REVISION_REQUESTED,                                         # 请求修订候选回复
                    actor=self.name,
                    task_id=task.id,
                    artifact_id=response.id,
                    message=reason,
                ),
            )
            follow_up_tasks = (
                AgentTask(
                    id=f"task:revise-response:{uuid.uuid4().hex[:8]}",
                    title="Revise unsafe response proposal",
                    description=reason,
                    priority=TaskPriority.CRITICAL,
                    required_capabilities=frozenset({AgentCapability.RESPONSE.value}),
                    created_by=self.name,
                    metadata={"kind": "response", "revisionOf": response.id},
                ),
            )

        # 不管合格与否，都加入Redis的短期记忆里面
        self.remember(f"review approved={approved}; reason={reason}")

        return AgentTurnResult(
            artifacts=(
                self._artifact(
                    kind, payload, task, 0.95, {"responseArtifactId": response.id}
                ),
            ),
            tasks=follow_up_tasks,
            events=events,
        )


class ContextAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="ContextAgent",
        capabilities=frozenset({AgentCapability.CONTEXT}),
        system_prompt=(
            "你是 ContextAgent。你只负责为本轮协作提供上下文，包括私有记忆、会话摘要、RAG 证据和 skill 约束。"
            "你不判断最终答案是否可采纳。"
        ),
        memory_policy="private_context_memory",
        model_profile="context",
        tool_permissions=frozenset(
            {"redis.memory", "mysql.messages", "rag.retrieve", "skills.read"}
        ),
    )

    # 判断是否认领任务：context 产物已存在则放弃；任务点名要 CONTEXT，或风险中高/意图是咨询或风险时认领
    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("context"):
            return AgentDecision(False, reason="context artifact already exists")
        
        risk = board.risk_value()                                       # 取出产物风险等级的最高档
        intent = board.intent_value()                                         # 取出当前黑板上最终意图

        if AgentCapability.CONTEXT.value in task.required_capabilities:
            return AgentDecision(True, 0.86, "task explicitly asks for context")
        if risk in {RiskLevel.MEDIUM, RiskLevel.HIGH} or intent in {
            IntentType.CONSULT,
            IntentType.RISK,
        }:
            return AgentDecision(
                True, 0.82, "support path needs memory, RAG, and skill context"
            )
        
        # 纯闲聊被认为"不需要上下文"
        return AgentDecision(
            False, reason="context not necessary for current artifacts"
        )

    # 执行上下文装配：加载压缩历史、RAG 检索、产 context 产物并发 CONTEXT_READY 消息给 ResponseAgent
    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:

        # 把"对话历史列表"处理成有界（bounded）的 prompt 历史 + 一份记忆摘要
        from app.services.memory import compact_history_for_prompt

        # 一组静态方法组成的skills库
        from app.services.skills import MindBridgeSkillLibrary

        history = self._load_history()

        # 压短历史 + 出摘要
        compacted_history, deterministic_brief = compact_history_for_prompt(    
            history, self.services.settings, board.model_input
        )
        # 再让 LLM 精炼成记忆要点
        # deterministic_brief 是廉价兜底摘要（不花钱、规则拼的）
        # memory_brief 是LLM 精炼版（更好但可能失败，失败就退回 deterministic_brief）
        memory_brief = self._summarize_memory(
            history, board.model_input, deterministic_brief
        )
        # 裁剪条数上限
        model_history = self._bounded_model_history(
            [*compacted_history, AiMessage(role="user", content=board.model_input)]
        )

        # 从黑板读当前轮意图和风险等级——这两个决定"要不要检索知识、要不要skill"
        intent = board.intent_value()
        risk = board.risk_value()

        retrieved: list["SearchResult"] = []                                        # 检索到的知识
        query = ""
        skill_context = ""

        # 非闲聊/非低风险
        if intent != IntentType.CHAT or risk != RiskLevel.LOW:
            # LLM 改写的检索查询词（把用户的话变成适合查库的词）
            query = self._rewrite_query(memory_brief, board.model_input)
            retrieved = self.services.knowledge.retrieve(
                query, self.services.settings.knowledge_top_k                       # RAG 查知识库
            )
            skill_context = MindBridgeSkillLibrary.response_skill_context(          # 意图+风险+文本选出的心理支持skill话术
                intent, risk, board.user_input
            )
        payload = {
            "memoryBrief": memory_brief,                                            # 记忆摘要
            "modelHistory": model_history,                                          # 精简后的历史
            "knowledgeQuery": query,                                                # 检索词
            "retrievedKnowledge": retrieved,                                        # RAG 检索结果
            "skillContext": skill_context,                                          # skill话术
            "privateMemoryKey": self.services.private_memory._key(                  # 记忆定位 key
                self.name, self.services.session.public_id
            ),
        }

        # 存入Redis短期记忆
        self.remember(
            f"context intent={intent.value}; risk={risk.value}; retrieved={len(retrieved)}"
        )

        return AgentTurnResult(
            artifacts=(self._artifact("context", payload, task, 0.88),),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="ResponseAgent",
                    task_id=task.id,
                    kind="CONTEXT_READY",
                    content=f"context ready; retrieved={len(retrieved)}",
                ),
            ),
        )

    # 加载会话对话历史：先读 Redis 短期记忆，没有则回填 MySQL 最近记录
    def _load_history(self) -> list[AiMessage]:
        from app.models.entities import ChatMessage

        history = self.services.memory.load_recent(self.services.session.public_id)
        if history:
            return history
        rows = (
            self.services.db.query(ChatMessage)
            .filter(ChatMessage.session_id == self.services.session.id)
            .order_by(ChatMessage.created_at.desc())
            .limit(self.services.settings.redis_memory_max_messages)
            .all()
        )
        rows.reverse()
        history = self.services.memory.messages_from_rows(rows)
        if history:
            self.services.memory.replace(self.services.session.public_id, history)
        return history

    # 用 LLM 把记忆摘要+当前输入改写成适合检索的查询词，失败则退用原输入截断
    def _rewrite_query(self, memory_brief: str, model_input: str) -> str:
        try:
            query = (
                self.client()
                .complete(
                    [
                        AiMessage(
                            role="system",
                            content=f"{self.profile.system_prompt}\n把学生输入改写成适合检索校园心理知识库的中文查询词，只输出查询词。",
                        ),
                        AiMessage(
                            role="user",
                            content=f"记忆摘要：\n{memory_brief}\n\n当前输入：\n{model_input}",
                        ),
                    ]
                )
                .strip()
            )
            return (query or model_input)[:60]
        except Exception:
            return model_input[:60]

    # 用 LLM 把历史压缩成记忆摘要（限长、不输出风险标签），失败则用兜底
    def _summarize_memory(
        self, history: list[AiMessage], current_input: str, fallback: str
    ) -> str:
        max_chars = max(120, self.services.settings.memory_summary_max_chars)
        if not history:
            return "无相关历史记忆。"
        try:
            summary = (
                self.client()
                .complete(
                    [
                        AiMessage(
                            role="system",
                            content=f"{self.profile.system_prompt}\n只输出 1-3 条中文记忆要点，不输出风险等级或诊断。",
                        ),
                        AiMessage(
                            role="user",
                            content=f"当前输入：\n{current_input}\n\n最近历史：\n{history[-12:]}",
                        ),
                    ]
                )
                .strip()
            )
            return summary[:max_chars] or fallback
        except Exception:
            return fallback or "无相关历史记忆。"

    # 把对话历史裁剪到配置允许的条数上限（首条 system 保留，其余取尾部）
    def _bounded_model_history(self, history: list[AiMessage]) -> list[AiMessage]:
        limit = max(2, self.services.settings.chat_history_limit * 2)
        if len(history) <= limit:
            return history
        if history[0].role == "system":
            return [history[0], *history[-(limit - 1) :]]
        return history[-limit:]

# ResponseAgent 面对两种任务
# 1：新回复任务（task:propose-response）：黑板上没有回复，要生成一份新的 → metadata 里没有 revisionOf
# 2：修订任务（task:revise-response）：SafetyAgent 打回后派生 → metadata 里有 revisionOf（指向被打回的回复）
class ResponseAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="ResponseAgent",
        capabilities=frozenset({AgentCapability.RESPONSE}),
        system_prompt=(
            "你是 ResponseAgent。你根据黑板上的意图、风险、上下文和安全约束提出候选回复 prompt，"
            "但最终是否采纳由 CoordinatorAgent 决定。"
        ),
        memory_policy="private_response_strategy",
        model_profile="response",
        tool_permissions=frozenset({"llm.response_plan"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:

        # 给"修订任务"开了个口子，让它能突破"已有回复就不做"的防重复检查
        if (
            board.latest_artifact("response_proposal")
            and "revisionOf" not in task.metadata
        ):
            return AgentDecision(False, reason="response proposal already exists")

        # ResponseAgent 只有等黑板上 intent 和 risk 产物都齐了才会干活
        if not board.latest_artifact("intent") or not board.latest_artifact("risk"):
            return AgentDecision(
                False, reason="response needs intent and risk artifacts"
            )
        
        intent = board.intent_value()
        risk = board.risk_value()

        # ① 闲聊且低风险 → 不需要上下文，直接出普通回复
        if intent == IntentType.CHAT and risk == RiskLevel.LOW:
            return AgentDecision(True, 0.78, "normal chat response can be proposed")

        # ② 有 context 产物 或 高危 → 支持路径材料齐（上下文/危机规则都有了），可以出支持性回复
        # 高危场景必须出危机处置回复（即使 context 还没好，也要立刻处理安全）——所以高危时不等 context 也认领
        if board.latest_artifact("context") or risk == RiskLevel.HIGH:
            return AgentDecision(True, 0.84, "support response has enough artifacts")

        # ③ 任务明确点名要 RESPONSE 能力 → 兜底保险，直接接单
        if AgentCapability.RESPONSE.value in task.required_capabilities:
            return AgentDecision(True, 0.65, "explicit response task")
        # ④ 都不满足 → 还在等上下文/其他前置，暂时不认领
        return AgentDecision(False, reason="waiting for context")


    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:

        intent = board.intent_value()
        risk = board.risk_value()

        context = board.latest_artifact("context")
        context_payload = context.payload if context else {}

        # 读取Context相关的历史
        model_history = context_payload.get("modelHistory") or [
            AiMessage(role="user", content=board.model_input)
        ]
        memory_brief = context_payload.get("memoryBrief") or "无相关历史记忆。"
        knowledge = context_payload.get("retrievedKnowledge") or []
        skill_context = context_payload.get("skillContext") or ""

        knowledge_context = "\n\n".join(
            f"- [{item.source}] {item.content}" for item in knowledge
        )

        # 如果只是闲聊
        if intent == IntentType.CHAT and risk == RiskLevel.LOW:
            messages = [
                PromptTemplates.answer_system_prompt(
                    # 闲聊 → ContextAgent 不检索 → retrievedKnowledge 是空列表 → ResponseAgent 闲聊分支自然没知识可用 → 直接普通回答；
                    IntentType.CHAT, RiskLevel.LOW, "", self.services.user.display_name
                ),
                AiMessage(
                    role="system",
                    content=(
                        f"{self.profile.system_prompt}\n"
                        f"当前由 ResponseAgent 以 normal_chat mode 提出回复方案。\n"
                        f"私有记忆：\n{_format_private_memory(self.private_memory())}\n"
                        f"记忆摘要：\n{memory_brief}"
                    ),
                ),
                *model_history,
            ]
            mode = "normal_chat"                                                        # 普通性回复

        # intent 判断是 CHAT、可 risk 却是 MEDIUM/HIGH（比如用户闲聊但风险评估说中高），
        # 此时 intent==CHAT 但 risk 高 → 不进第一个 if（CHAT+LOW） → 走 else 支持分支 → 这行就把 CHAT 改写为 CONSULT。
        else:
            messages = [
                PromptTemplates.answer_system_prompt(
                    intent if intent != IntentType.CHAT else IntentType.CONSULT,
                    risk,
                    knowledge_context,
                    self.services.user.display_name,
                    skill_context,
                ),
                AiMessage(
                    role="system",
                    content=(
                        f"{self.profile.system_prompt}\n"
                        f"当前由 ResponseAgent 以 support mode 提出回复方案。\n"
                        f"私有记忆：\n{_format_private_memory(self.private_memory())}\n"
                        f"记忆摘要：\n{memory_brief}"
                    ),
                ),
                *model_history,
            ]
            mode = "support"                                                            # 支持性回复
        
        payload = {
            "messages": messages,
            "mode": mode,
            "intent": intent.value,
            "risk": risk.value,
            "responseAgent": self.name,
            "privateMemoryKey": self.services.private_memory._key(
                self.name, self.services.session.public_id
            ),
        }

        self.remember(f"response mode={mode}; intent={intent.value}; risk={risk.value}")

        return AgentTurnResult(
            artifacts=(self._artifact("response_proposal", payload, task, 0.86),),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="SafetyAgent",
                    task_id=task.id,
                    kind="REVIEW_REQUEST",
                    content="请审查候选回复方案。",
                ),
            ),
        )


class CoordinatorAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="CoordinatorAgent",
        capabilities=frozenset({AgentCapability.COORDINATION}),
        system_prompt=(
            "你是 CoordinatorAgent。你不规定固定 Agent 顺序；你只维护任务板、预算、安全门槛、冲突仲裁和最终采纳。"
        ),
        memory_policy="private_coordination_trace",
        model_profile="coordinator",
        tool_permissions=frozenset({"taskboard.write", "blackboard.accept"}),
    )

    # CoordinatorAgent 不参与"认领任务"这个竞争机制
    # 它不在 agents 列表里（runtime 里 agents = [Understanding, Safety, Context, Response]），所以 AgentRegistry 根本不会问它 decide
    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        return AgentDecision(
            False,
            reason="CoordinatorAgent is driven by the event loop, not by fixed workflow slots",
        )

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        return AgentTurnResult(close_task=False)

    def root_task(self, board: CollaborationBlackboard) -> AgentTask:
        return AgentTask(
            id="task:root",
            title="Resolve user turn",
            description=board.user_input,
            priority=TaskPriority.CRITICAL
            if has_high_risk_signal(board.user_input)
            else TaskPriority.NORMAL,
            created_by=self.name,
            metadata={"kind": "root"},
        )

    def remember_acceptance(self, artifact_id: str, reason: str) -> None:
        self.remember(f"accepted={artifact_id}; reason={reason}")




# 从黑板上下文产物里取模型对话历史，供风险评估等作为上下文喂给模型；没有上下文产物时退化为只含本轮输入的单条列表
def _context_history(board: CollaborationBlackboard) -> list[AiMessage]:
    context = board.latest_artifact("context")
    if not context:
        return [AiMessage(role="user", content=board.model_input or board.user_input)]
    return context.payload.get("modelHistory") or [
        AiMessage(role="user", content=board.model_input or board.user_input)
    ]


# 把私有记忆消息列表格式化为纯文本清单，最多取最近 5 条；空列表返回"无"
def _format_private_memory(items: list[AiMessage]) -> str:
    if not items:
        return "无"
    return "\n".join(f"- {item.content}" for item in items[-5:])
