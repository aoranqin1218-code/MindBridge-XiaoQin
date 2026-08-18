from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.agents.factory import create_agent_runtime
from app.agents.result import AgentRunResult, AgentStep
from app.core.config import Settings
from app.core.enums import IntentType, MessageRole
from app.models.entities import ChatMessage, ChatSession, PsychologicalReport, UserAccount
from app.schemas.dtos import AiMessage, ChatRequest
from app.services.assessment import PsychologyAssessment
from app.services.knowledge import SearchResult
from app.services.mcp_client import MindBridgeMcpToolClient
from app.services.memory import RedisShortTermMemoryStore
from app.services.privacy import PrivacySanitizer
from app.services.tool_queue import ToolQueueService
from app.services.trace import AgentTraceService

# @dataclass 是 Python 标准库的装饰器——自动帮你生成 __init__、__repr__、__eq__ 这些样板方法。
# 专门给"主要存数据、没复杂逻辑"的类用的—
# 比如 AgentToolPlan、AgentHarnessOutcome、AgentRunResult，都是数据载体，没有方法逻辑，正是 dataclass 最合适的场景
@dataclass
class AgentToolPlan:
    report_id: int | None
    risk_level: str | None

    # @property 是把一个方法变成像属性一样访问——调用时不用加括号
    # 适合表达"不是存出来的，而是算出来的"属性。这里 requires_tools 不是一个真实字段，而是取决于 report_id 是否为空
    @property
    def requires_tools(self) -> bool:
        return self.report_id is not None


@dataclass
class AgentHarnessOutcome:
    session: ChatSession                         # 本次对话的聊天会话实体
    original_input: str                          # 用户原始输入（脱敏前）
    model_input: str                             # 脱敏后的模型输入
    intent: IntentType                           # Agent 判定的用户意图（闲聊/咨询/风险）
    risk_level: str | None                       # 心理风险等级（NONE/LOW/MEDIUM/HIGH）
    assessment: PsychologyAssessment | None      # 心理评估详情（情绪、分数、置信度等）
    response_messages: list[AiMessage]           # 发给 LLM 的完整消息列表（系统提示词 + 上下文 + 用户消息）
    agent_steps: list[AgentStep]                 # 每个 Agent 的执行步骤记录（名称、动作、观察）
    retrieved_knowledge: list[SearchResult]      # RAG 检索到的相关知识片段
    report_id: int | None                        # 心理评估报告在 DB 中的主键 ID
    tool_plan: AgentToolPlan                     # 后置工具计划（报告 ID + 风险等级）
    trace_id: int | None                         # Agent 运行 Trace 记录的 ID


class MindBridgeAgentHarness:
    """MindBridge 单轮 Agent 运行的业务编排层。

    负责 Agent 运行时周边的业务协调：输入准备、持久化、风险报告创建、
    工具计划生成和 Trace 数据记录。HTTP/SSE 层可以保持薄薄一层。
    """

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.privacy = PrivacySanitizer()                                         # 把用户输入里的敏感个人信息用 [已脱敏] 替换掉，再传给 LLM
        self.memory = RedisShortTermMemoryStore(settings)                         # 用 Redis 缓存最近的对话历史，快速加载给 LLM 做上下文

    def run(self, user: UserAccount, request: ChatRequest) -> AgentHarnessOutcome:
        original_input = request.message.strip()
        model_input = self.privacy.sanitize(original_input)                                         # 输入脱敏
        session = self._resolve_session(user, request.sessionId, original_input)                    # 找已有会话 or 新建

        # 创建多 Agent 运行时并执行完整协作
        # Coordinator 分配任务、Understanding 分析意图、Safety 评估风险、Context 查记忆和知识库、Response 拼最终回复。返回 AgentRunResult
        agent_run = create_agent_runtime(self.db, self.settings).run(user, session, original_input, model_input)

        # 把用户的原始消息（未脱敏）写入数据库 chat_messages 表，同时更新会话 updated_at，同步写一份到 Redis 短期记忆
        # 放在这里：不希望agent处理失败了，数据库还有一条孤立的用户消息
        self.save_message(user, session, MessageRole.USER, original_input)                   # 此处只保存了用户的信息，模型回答的信息是在chat.py里面保存的                

        # 如果 Agent 跑完后认为需要出报告（意图不是闲聊，且有评估结果），把情绪、风险等级、置信度等信息写入 psychological_reports 表
        report = self._create_report(user, session, original_input, agent_run)
        risk_level = report.risk_level if report is not None else None

        # 把本次 Agent 运行的完整过程记下来——谁、什么会话、原始输入和脱敏输入、各 Agent 的步骤、结果——全部写进 trace 表
        trace = AgentTraceService(self.db).save_run(
            user=user,
            session=session,
            original_input=original_input,
            sanitized_input=model_input,
            memory_brief=agent_run.memory_brief,
            agent_run=agent_run,
            report_id=report.id if report is not None else None,
        )

        # 把报告 ID 和风险等级打包成 AgentToolPlan，然后把这一轮所有产物汇总成 AgentHarnessOutcome，交给上层 chat.py 使用
        tool_plan = AgentToolPlan(report_id=report.id if report is not None else None, risk_level=risk_level)

        return AgentHarnessOutcome(
            session=session,
            original_input=original_input,
            model_input=model_input,
            intent=agent_run.intent,
            risk_level=risk_level,
            assessment=agent_run.assessment,
            response_messages=agent_run.response_messages,
            agent_steps=agent_run.steps,
            retrieved_knowledge=agent_run.retrieved_knowledge,
            report_id=report.id if report is not None else None,
            tool_plan=tool_plan,
            trace_id=trace.id,
        )

    # 保存助手回复消息（save_message 的便捷包装，固定角色为 ASSISTANT）
    def save_assistant_message(self, user: UserAccount, session: ChatSession, content: str) -> None:
        self.save_message(user, session, MessageRole.ASSISTANT, content)

    # 投递后置工具任务：有队列则入队，无队列则直接调用 MCP 客户端处理报告
    async def dispatch_tools(self, tool_plan: AgentToolPlan) -> list[str]:
        if tool_plan.report_id is None:
            return []

        # 有队列 → 把任务写入 tool_jobs 表，由后台 Worker 异步执行
        # 这些工具任务跟用户的对话体验完全无关——用户已经看到回复了，Excel 报表、个案、预警这些是后台异步跑的
        if self.settings.tool_queue_enabled:
            ToolQueueService(self.db, self.settings).enqueue_report(tool_plan.report_id, tool_plan.risk_level)
            return ["queued"]

        # 没队列 → 直接同步调 MCP 客户端，立刻执行
        return await MindBridgeMcpToolClient(self.settings).handle_report(tool_plan.report_id, tool_plan.risk_level)

    # 将一条聊天消息写入 MySQL 并同步追加到 Redis 短期记忆
    def save_message(self, user: UserAccount, session: ChatSession, role: MessageRole, content: str) -> None:
        self.db.add(ChatMessage(user_id=user.id, session_id=session.id, role=role.value, content=content))
        session.touch()                                                                 # 更新对话记录的更新时间
        self.db.add(session)
        self.db.commit()
        self.memory.append(session.public_id, role.value, content)                      # 保存进Redis做短期记忆

    # 根据前端传入的 public_id 查找已有会话，未传则新建（首次对话入口）
    def _resolve_session(self, user: UserAccount, public_id: str | None, text: str) -> ChatSession:
        if public_id:

            # self.db 就是 SQLAlchemy 的 Session（数据库连接）。
            # query() 是 Session 自带的方法——不是直接对接 SQL，而是接收 ORM 实体类，自动翻译成 SQL
            session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id, ChatSession.user_id == user.id).first()
            # 等价于 SELECT * FROM chat_sessions WHERE public_id = ... AND user_id = ... LIMIT 1;

            if session is None:
                raise ValueError("Session not found")
            return session
        
        # 第一次生成sessionID，并后续通过 yield sse 传给前端，后续前端就会带着sessionID来发消息
        session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title=text[:36])             # 内存里生成了一个 Python 对象
        self.db.add(session)                                                        # 标记"这个对象要存起来"
        self.db.commit()                                                            # 真正执行 INSERT INTO chat_sessions ...
        self.db.refresh(session)                                                    # 从数据库把刚写入的这一行重新查回来，更新手里的对象
        return session

    # 根据 Agent 评估结果创建心理报告（非闲聊场景），闲聊或无需报告时返回 None
    def _create_report(self, user: UserAccount, session: ChatSession, text: str, agent_run: AgentRunResult) -> PsychologicalReport | None:

        # 闲聊场景，或 Agent 没有产出评估结果 → 不建报告
        # requires_report → True if intent != CHAT - 不是闲聊就要出报告
        # assessment → PsychologyAssessment | None - 如果不需要报告就是 None
        if not agent_run.requires_report or agent_run.assessment is None:
            return None
        report = PsychologicalReport(
            user_id=user.id,
            session_id=session.id,
            content=text,
            intent=agent_run.intent.value,
            emotion=agent_run.assessment.emotion.value,
            emotion_score=agent_run.assessment.emotion_score,
            risk_level=agent_run.assessment.risk.value,
            confidence=agent_run.assessment.confidence,
            summary=agent_run.assessment.summary,
        )
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        return report
