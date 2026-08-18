from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.assessment import PsychologyAssessment
from app.services.knowledge import SearchResult


@dataclass
class AgentStep:
    step: int                                                       # 步骤序号，表示当前是第几步
    agent: str                                                      # 执行该步骤的 Agent 名称
    action: str                                                     # Agent 在本步骤中执行的动作（如调用工具、推理等）
    observation: str                                                # 动作执行后返回的观察结果


@dataclass
class AgentRunResult:
    intent: IntentType                                                  # 本次对话的用户意图类型（如聊天、风险评估等）
    risk_level: RiskLevel                                               # 风险等级评估结果
    assessment: PsychologyAssessment | None                             # 心理学评估报告，非评估场景下为 None
    retrieved_knowledge: list[SearchResult]                             # 从知识库中检索到的相关知识列表
    response_messages: list[AiMessage]                                  # 发给 LLM 生成最终回复的完整提示词消息列表（含系统提示词 + 上下文 + 用户输入），最终回复由 ChatService 调用 AiClient.stream() 产生
    steps: list[AgentStep]                                              # Agent 执行过程中每一步的详细记录
    memory_brief: str                                                   # 本轮对话的记忆摘要，用于后续会话回溯
    collaboration_events: list[Any] = field(default_factory=list)       # 多 Agent 协作事件日志
    collaboration_tasks: list[Any] = field(default_factory=list)        # 多 Agent 协作任务列表
    collaboration_artifacts: list[Any] = field(default_factory=list)    # 多 Agent 协作产生的中间产物

    @property
    def requires_report(self) -> bool:
        return self.intent != IntentType.CHAT
