
from __future__ import annotations

import json
import logging
from datetime import datetime
from importlib import import_module
from typing import Protocol

from app.core.config import Settings
from app.models.entities import ChatMessage
from app.schemas.dtos import AiMessage
from app.services.privacy import PrivacySanitizer


logger = logging.getLogger(__name__)


class MemoryCompactionSettings(Protocol):
    memory_compaction_enabled: bool
    memory_compaction_recent_messages: int
    memory_summary_max_chars: int


class RedisShortTermMemoryStore:

    # 创建对象的同时，顺手把 Redis 连接建好，存到 self.client。
    # 后续所有方法（append、load_recent）都先检查 if self.client is None，连不上就静默跳过，不让 Redis 挂了影响主流程
    def __init__(self, settings: Settings):
        self.settings = settings
        self.privacy = PrivacySanitizer()                               # 脱敏
        self.client = self._connect()                                   # 连 Redis，返回一个连接对象（连不上就返回 None）


    # 读取某个会话最近的对话记忆（最多 redis_memory_max_messages 条，Redis 挂了则返回空列表）
    def load_recent(self, session_public_id: str) -> list[AiMessage]:
        if self.client is None:
            return []
        try:
            return self._read(session_public_id, self.settings.redis_memory_max_messages)
        except Exception as exc:
            logger.warning("Redis memory read unavailable: %s", exc)
            return []

    # 把 MySQL chat_messages 表查出的行对象批量转成 AiMessage（供 Redis 记忆回填用）
    def messages_from_rows(self, rows: list[ChatMessage]) -> list[AiMessage]:
        return [self._message_from_row(row) for row in rows]

    # 把一条消息追加到某会话的对话记忆：rpush 进 Redis 列表，并裁剪到最近 N 条 + 续期 TTL
    def append(self, session_public_id: str, role: str, content: str) -> None:
        if self.client is None:
            return
        key = self._key(session_public_id)
        # 打包成json，准备塞入Redis
        payload = self._serialize(role, content)
        try:
            self.client.rpush(key, payload)
            # 列表里只保留尾部最近 max 条，把更早的全删掉
            self.client.ltrim(key, -self.settings.redis_memory_max_messages, -1)
            # 把 key 的存活时间重置为 seconds 秒 - 滑动过期
            self.client.expire(key, self.settings.redis_memory_ttl_seconds)
        except Exception as exc:
            logger.warning("Redis memory append unavailable: %s", exc)

    # 整段替换某会话的记忆：先删旧 key，再一次性写入给定消息列表（含裁剪 + 续期）
    def replace(self, session_public_id: str, messages: list[AiMessage]) -> None:
        if self.client is None:
            return
        key = self._key(session_public_id)
        pipe = self.client.pipeline()
        pipe.delete(key)
        if messages:
            pipe.rpush(key, *[self._serialize(message.role, message.content) for message in messages])
            pipe.ltrim(key, -self.settings.redis_memory_max_messages, -1)
            pipe.expire(key, self.settings.redis_memory_ttl_seconds)
        try:
            pipe.execute()
        except Exception as exc:
            logger.warning("Redis memory replace unavailable: %s", exc)

    # 从 Redis 读取该会话最近 limit 条原始记录，逐条解析 JSON 并脱敏后返回
    def _read(self, session_public_id: str, limit: int) -> list[AiMessage]:

        # Redis 的 list（列表）就像一个双端队列，元素从左到右排成一串，每个元素有下标
        # LRANGE key start stop 就是从列表里切出 [start, stop] 这一段（闭区间，两头都算）
        # lrange(key, -limit, -1) 的语义就是：从倒数第 limit 个，一直取到最后一个 = 取最后 limit 条记录
        raw_items = self.client.lrange(self._key(session_public_id), -limit, -1)
        messages = []
        for raw in raw_items:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            role = str(data.get("role", "")).lower()
            content = str(data.get("content", ""))
            if role and content:
                messages.append(AiMessage(role=role, content=self.privacy.sanitize(content)))
        return messages

    # 按 settings 里的地址建 Redis 客户端并 ping 一下；连不上则记日志返回 None（降级，不抛异常）
    def _connect(self):
        try:
            # import_module("redis") 等同于 import redis，但可以写在函数体内按需加载
            redis_module = import_module("redis")
        except ModuleNotFoundError as exc:
            raise RuntimeError("请先安装 requirements.txt 中的 redis 依赖") from exc
        client = redis_module.Redis.from_url(                                       # from_url() 用连接字符串创建客户端，等价于 redis://localhost:6379/0
            self.settings.redis_url,                                                # 连接地址
            decode_responses=True,                                                  # 从 Redis 读出来的字节自动转成 Python 字符串，不用手动 .decode()
            socket_timeout=self.settings.redis_socket_timeout_seconds,              # 读数据超时时间，防止某个请求永远卡住
            socket_connect_timeout=self.settings.redis_socket_timeout_seconds,      # 建立 TCP 连接的超时，连不上就尽快放弃
        )
        try:
            client.ping()
        except Exception as exc:                                                    # Redis 挂了就当降级而不是崩溃。返回 None
            logger.warning("Redis memory disabled: %s", exc)
            return None
        return client

    # 把单条 MySQL 行对象转成 AiMessage（角色小写化 + 内容脱敏）
    def _message_from_row(self, row: ChatMessage) -> AiMessage:
        return AiMessage(role=row.role.lower(), content=self.privacy.sanitize(row.content))

    # 把一条聊天消息打包成 JSON 字符串，准备塞进 Redis
    def _serialize(self, role: str, content: str) -> str:
        return json.dumps(
            {
                "role": role.lower(),                                   # "USER" → "user"，统一小写
                "content": self.privacy.sanitize(content),              # 再次脱敏：手机号/邮箱/身份证 → [已脱敏]
                "createdAt": datetime.utcnow().isoformat(),             # 时间戳，方便追溯
            },
            ensure_ascii=False,                                         # 让中文原样输出，不转成 \uXXXX
        )

    # 生成某会话记忆在 Redis 里的唯一 key（mindbridge:short-term-memory:会话id）
    def _key(self, session_public_id: str) -> str:
        return f"mindbridge:short-term-memory:{session_public_id}"


def compact_history_for_prompt(
    history: list[AiMessage],
    settings: MemoryCompactionSettings,
    current_input: str = "",
) -> tuple[list[AiMessage], str]:
    """Return bounded prompt history plus a student-safe memory brief.

    The summary is deterministic and avoids diagnostic labels. It is intended
    for prompt context and auditability, not for student-facing display.
    """

    sanitized = [AiMessage(role=item.role, content=PrivacySanitizer().sanitize(item.content)) for item in history]
    if not sanitized:
        return [], "无相关历史记忆。"

    recent_count = max(2, int(getattr(settings, "memory_compaction_recent_messages", 8)))
    max_chars = max(120, int(getattr(settings, "memory_summary_max_chars", 500)))
    brief = summarize_history_for_memory(sanitized, current_input, max_chars)

    if not getattr(settings, "memory_compaction_enabled", True) or len(sanitized) <= recent_count:
        return sanitized, brief

    recent = sanitized[-recent_count:]
    summary_message = AiMessage(
        role="system",
        content=(
            "历史摘要（仅供 MindBridge 内部上下文使用；不要向学生展示；"
            "不要据此输出诊断、风险等级或后台标签）：\n" + brief
        ),
    )
    return [summary_message, *recent], brief


def summarize_history_for_memory(history: list[AiMessage], current_input: str = "", max_chars: int = 500) -> str:
    privacy = PrivacySanitizer()
    user_points = []
    assistant_points = []
    for message in history:
        content = " ".join(privacy.sanitize(message.content).split())
        if not content:
            continue
        if message.role == "user":
            user_points.append(content)
        elif message.role == "assistant":
            assistant_points.append(content)

    parts = []
    if user_points:
        parts.append("学生近期关注：" + "；".join(_clip(item, 80) for item in user_points[-4:]))
    if assistant_points:
        parts.append("已给过的支持：" + "；".join(_clip(item, 70) for item in assistant_points[-3:]))
    if current_input:
        parts.append("本轮输入关注：" + _clip(privacy.sanitize(current_input), 80))
    if not parts:
        return "无相关历史记忆。"
    return _clip("\n".join(parts), max_chars)


def _clip(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."
