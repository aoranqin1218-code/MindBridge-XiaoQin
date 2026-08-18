
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.agents.result import AgentRunResult
from app.models.entities import AgentRunTrace, ChatSession, UserAccount

# trace 本质是一次运行的快照，写入后只做回溯查看，不需要像业务数据那样 join 查询
class AgentTraceService:
    def __init__(self, db: Session):
        self.db = db

    # 将一次 Agent 运行的完整上下文（意图、风险、输入、步骤、知识、响应、评估）持久化到 trace 表
    def save_run(
        self,
        user: UserAccount,
        session: ChatSession,
        original_input: str,
        sanitized_input: str,
        memory_brief: str,
        agent_run: AgentRunResult,
        report_id: int | None,
    ) -> AgentRunTrace:

        # 不是所有字段都是 JSON，只有内容结构不固定的四个字段存成了 JSON
        trace = AgentRunTrace(
            user_id=user.id,
            session_id=session.id,
            report_id=report_id,
            intent=agent_run.intent.value,
            risk_level=agent_run.risk_level.value,
            original_input=original_input,
            sanitized_input=sanitized_input,
            memory_brief=memory_brief,
            agent_steps_json=_json(_agent_steps_with_collaboration(agent_run)),                 # 每个 Agent 的执行步骤：谁、第几步、做了什么、观察到什么
            retrieved_knowledge_json=_json(agent_run.retrieved_knowledge),                      # RAG返回的参考知识
            response_messages_json=_json(agent_run.response_messages),                          # 最终发给 LLM 的完整消息列表（系统提示词 + 上下文 + 用户消息）
            assessment_json=_json(agent_run.assessment or {}),                                  # 心理评估结果（情绪、分数、风险、置信度），没有就用 {} 填进去
        )
        self.db.add(trace)
        self.db.commit()
        self.db.refresh(trace)
        return trace


# 将 Python 对象递归转为 JSON 字符串（枚举→值，dataclass→字典，Pydantic→字典）
#  ensure_ascii=False - 中文不会被转成 \uXXXX，直接保留原文 "心情" 而不是 "\u5fc3\u60c5"
# default=str - 兜底：遇到无法序列化的对象时，调用 str() 转成字符串，不抛异常
def _json(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False, default=str)


# 将 Agent 步骤与协作事件、任务、Artifact 拼成统一列表，便于写入 trace
def _agent_steps_with_collaboration(agent_run: AgentRunResult) -> list[Any]:

    # [*agent_run.steps] 是解包拷贝一份，不污染原数据（浅拷贝） - 把 Agent 的执行步骤倒进去
    entries: list[Any] = [*agent_run.steps]

    # 协作事件
    entries.extend(
        {
            "kind": "agent_event",
            "type": getattr(event.type, "value", event.type),
            "actor": event.actor,
            "taskId": event.task_id,
            "artifactId": event.artifact_id,
            "message": event.message,
            "metadata": event.metadata,
        }
        for event in agent_run.collaboration_events
    )

    # 协作任务
    entries.extend(
        {
            "kind": "agent_task",
            "id": task.id,
            "title": task.title,
            "status": getattr(task.status, "value", task.status),
            "priority": getattr(task.priority, "value", task.priority),
            "requiredCapabilities": sorted(task.required_capabilities),
            "claimedBy": list(task.claimed_by),
            "createdBy": task.created_by,
            "metadata": task.metadata,
        }
        for task in agent_run.collaboration_tasks
    )

    # 协作产物
    entries.extend(
        {
            "kind": "agent_artifact",
            "id": artifact.id,
            "owner": artifact.owner,
            "artifactKind": artifact.kind,
            "confidence": artifact.confidence,
            "taskId": artifact.task_id,
            "metadata": artifact.metadata,
            "payload": artifact.payload,
        }
        for artifact in agent_run.collaboration_artifacts
    )
    return entries


def _to_jsonable(value: Any) -> Any:                       # 入参：任何复杂对象；出参：原生 JSON 兼容类型
    if isinstance(value, Enum):                            # 是枚举 → 取它的值（如 RiskLevel.HIGH → "HIGH"）
        return value.value
    if is_dataclass(value):                                # 是 @dataclass → 转成 dict，再递归处理
        return _to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):                       # 是 Pydantic 模型 → 调 model_dump() 转 dict，再递归处理
        return _to_jsonable(value.model_dump())
    if isinstance(value, list):                            # 是列表 → 每个元素递归处理
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):                           # 是元组 → 转成列表，每个元素递归
        return [_to_jsonable(item) for item in value]      #     （JSON 不认 Python 的 tuple）
    if isinstance(value, dict):                            # 是字典 → key 转字符串，value 递归处理
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value                                           # 是 int / str / float / bool / None → 直接返回
