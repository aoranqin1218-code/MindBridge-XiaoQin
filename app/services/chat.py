from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from app.agents.harness import MindBridgeAgentHarness
from app.core.config import Settings
from app.models.entities import UserAccount
from app.schemas.dtos import ChatRequest, ChatStreamEvent
from app.services.ai import AiClient


logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.ai = AiClient(settings)                                # 负责将最终消息调用模型并流式返回
        self.agent_harness = MindBridgeAgentHarness(db, settings)   # 负责把一次心理支持 Agent 运行置于可控业务流程中

    async def stream_chat(self, user: UserAccount, request: ChatRequest):

        # 把用户输入交给 Agent 处理，这一步是同步的
        # 返回的 outcome 包含它的处理结果：会话对象、要发给 LLM 的消息列表、工具计划等
        outcome = self.agent_harness.run(user, request)

        # 往前端推第一条 SSE：event: meta。告诉前端"对话开始了，本次会话 ID 是 xxx"，前端可以用这个 ID 做历史记录追溯
        # ChatSession 对应 chat_sessions 表，每行记录代表"用户某一次完整的聊天"。public_id就是这次对话的会话 ID
        yield sse("meta", ChatStreamEvent(type="meta", sessionId=outcome.session.public_id).model_dump(by_alias=True))

        assistant = []
        # 异步是因为 self.ai.stream() 在背后做网络 I/O
        async for token in self.ai.stream(outcome.response_messages):
            assistant.append(token)
            # 把对象转成普通字典
            yield sse("token", ChatStreamEvent(type="token", sessionId=outcome.session.public_id, content=token).model_dump())

        # LLM 吐完了，把攒下来的 token 拼成完整文本，存进数据库
        if assistant:
            self.agent_harness.save_assistant_message(user, outcome.session, "".join(assistant))

        # 对话结束后执行一些后台工具任务：发生在 save_assistant_message 之后，说明工具执行的是"后处理"任务，不是聊天流程中的内置工具调用
        # try/except 是故意只记日志不抛异常——工具失败不能影响用户的聊天体验（用户已经看到回复了）
        try:
            await self.agent_harness.dispatch_tools(outcome.tool_plan)
        except Exception as exc:
            logger.warning(
                "Post-response tool dispatch failed for session=%s report_id=%s: %s",
                outcome.session.public_id,
                outcome.report_id,
                exc,
                exc_info=True,
            )
        # 流结束了，关连接
        yield sse("done", ChatStreamEvent(type="done", sessionId=outcome.session.public_id).model_dump())


def sse(event: str, data: dict) -> str:
    # json.dumps： 变成 JSON 字符串
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
