from __future__ import annotations

import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.agents.autonomous import (
    AgentPrivateMemory,
    AgentRuntimeServices,
    ContextAgent,
    CoordinatorAgent,
    ResponseAgent,
    SafetyAgent,
    UnderstandingAgent,
)
from app.agents.coordinator import EventDrivenCoordinator
from app.agents.events import AgentEvent, AgentEventType, CollaborationBlackboard
from app.agents.registry import AgentRegistry
from app.agents.result import AgentRunResult, AgentStep
from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.models.entities import ChatSession, UserAccount
from app.schemas.dtos import AiMessage
from app.services.agent_models import AgentModelRegistry
from app.services.ai import AiClient, PromptTemplates
from app.services.knowledge import KnowledgeService, SearchResult
from app.services.memory import RedisShortTermMemoryStore


class EventDrivenAgentRuntimeService:
    """Actor 风格的多 Agent 运行时。

    Agent 观察黑板上的开放任务、独立认领工作，并返回统一的
    AgentRunResult 契约供应用其余部分消费。
    """

    # "声明性信息"
    framework_name = "event_driven_multi_agent"                                     # "事件驱动多 Agent"框架
    max_steps = 8

    # 注入数据库会话与配置，装配运行时所需的各项服务（AI/知识检索/记忆/模型注册表/私有记忆）
    def __init__(self, db: Session, settings: Settings):
        self.db = db                                            # db：SQLAlchemy 数据库会话（可查 MySQL 表，如 chat_messages）
        self.settings = settings                                # settings：全局配置对象（Redis 地址、模型、记忆条数等）
        self.ai = AiClient(settings)                            # 原始 AI 客户端（Agent 实际多用 model_registry 按名取）
        self.knowledge = KnowledgeService(db, settings)         # 知识检索服务（RAG 查询知识库）
        self.memory = RedisShortTermMemoryStore(settings)       # 短期对话记忆存取（session 维度，最近 N 条）
        self.model_registry = AgentModelRegistry(settings)      # 按 Agent 名分配模型档位的注册表
        self.private_memory = AgentPrivateMemory(settings)      # 各 Agent 隔离的私有记忆（agent:名字:会话id 维度）

    # 处理一轮用户请求：组装服务包与 4 个 Agent，建黑板，跑协调器调度，最后转成 AgentRunResult
    def run(self, user: UserAccount, session: ChatSession, original_input: str, model_input: str) -> AgentRunResult:
        services = AgentRuntimeServices(
            db=self.db,
            settings=self.settings,
            user=user,
            session=session,
            ai=self.ai,
            model_registry=self.model_registry,
            memory=self.memory,
            private_memory=self.private_memory,
            knowledge=self.knowledge,
        )

        coordinator_agent = CoordinatorAgent(services)

        agents = [
            UnderstandingAgent(services),
            SafetyAgent(services),
            ContextAgent(services),
            ResponseAgent(services),
        ]

        board = CollaborationBlackboard(
            turn_id=uuid.uuid4().hex,
            user_id=user.id,
            session_id=session.public_id,
            user_input=original_input,
            model_input=model_input,
        )

        board = board.append_event(
            AgentEvent(
                type=AgentEventType.TURN_STARTED,
                actor=coordinator_agent.name,
                message="user turn published to shared task board",
            )
        )

        registry = AgentRegistry(agents)

        final_board = EventDrivenCoordinator(registry, coordinator_agent, self.settings).run(board)

        return self._to_result(final_board, user)

    # 把最终黑板转成 AgentRunResult：提取意图/风险/上下文/回复，并把协作轨迹（事件/任务/产物）一并打包
    def _to_result(self, board: CollaborationBlackboard, user: UserAccount) -> AgentRunResult:
        intent = board.intent_value()                                                           # 取黑板最终意图
        risk = board.risk_value()                                                               # 取黑板最终风险等级
        context = board.latest_artifact("context")                                              # 取最新 context 产物
        risk_artifact = board.latest_artifact("risk")                                           # 取最新 risk 产物对象。和 ② 的区别：② 只要"等级"（枚举），④ 要"整个产物"
        accepted = board.accepted_artifact() or board.latest_artifact("response_proposal")      # 优先正式采纳，退回候选回复
        memory_brief = "无相关历史记忆。"                                                        # 默认"无相关历史记忆"
        retrieved: list[SearchResult] = []                                                      # （检索到的知识）空列表
        response_messages: list[AiMessage] = []                                                 # 要返回的回复消息）空列表

        if context:
            memory_brief = context.payload.get("memoryBrief") or memory_brief
            retrieved = context.payload.get("retrievedKnowledge") or []
        if accepted:
            response_messages = accepted.payload.get("messages") or []

        # 黑板上连候选回复产物都没有（比如预算耗尽、回复生成失败）。
        # 这时用 _fallback_messages 规则拼一个回复，保证"一定有内容返回"
        if not response_messages:
            response_messages = self._fallback_messages(intent, risk, user.display_name, board.model_input)

        # 这个 assessment 会进 AgentRunResult.assessment 字段，供心理报告取用（它要的不只是 risk 等级，还有情绪、置信度这些完整信息）
        assessment = risk_artifact.payload.get("assessment") if risk_artifact else None

        return AgentRunResult(
            intent=intent,
            risk_level=risk,
            assessment=assessment,
            retrieved_knowledge=retrieved,
            response_messages=response_messages,
            steps=self._events_to_steps(board),
            memory_brief=memory_brief,
            collaboration_events=list(board.events),
            collaboration_tasks=list(board.tasks.values()),
            collaboration_artifacts=list(board.artifacts),
        )

    # 无候选回复时用规则拼兜底消息：system 提示词 + 用户输入，保证始终有内容返回
    def _fallback_messages(self, intent: IntentType, risk: RiskLevel, display_name: str, model_input: str) -> list[AiMessage]:
        return [
            PromptTemplates.answer_system_prompt(intent, risk, "", display_name),
            AiMessage(role="user", content=model_input),
        ]

    # 把黑板事件流转成 AgentStep 列表（含序号/actor/类型/详情），供 trace 或结果展示
    def _events_to_steps(self, board: CollaborationBlackboard) -> list[AgentStep]:
        steps = []
        for index, event in enumerate(board.events, start=1):
            detail = event.message or _compact_json(event.metadata)
            # 判断这个事件关联了产物没有 ，有关联就在 detail 后面补一句 ; artifact=xxx，让人知道"这步产出了/关联了哪个产物"。
            # 没有就跳过，detail 保持原样
            if event.artifact_id:
                detail = f"{detail}; artifact={event.artifact_id}" if detail else f"artifact={event.artifact_id}"
            steps.append(AgentStep(index, event.actor, event.type.value, detail))
        return steps


# 把任意值转成紧凑 JSON 字符串，供事件/步骤的详情展示；空值或超长则裁剪到 240 字符
def _compact_json(value: Any) -> str:
    jsonable = _to_jsonable(value)
    if not jsonable:
        return ""
    return str(jsonable)[:240]


# 递归把任意值转成可 JSON 序列化的结构：枚举取 .value、dataclass/Pydantic 转 dict、容器递归展开
def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value
