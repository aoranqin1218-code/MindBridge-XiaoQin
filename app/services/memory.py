
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


# Protocol：只声明"配置对象长什么样"的结构契约，不绑定具体类；任何有这三个属性的对象都算满足
class MemoryCompactionSettings(Protocol):
    memory_compaction_enabled: bool                    # 是否启用记忆压缩（config.py:62 默认 True；为 False 时不做摘要，直接用原文）
    memory_compaction_recent_messages: int             # 超过多少条历史才触发压缩，只取最近 N 条做摘要（config.py:63 默认 8）
    memory_summary_max_chars: int                      # 压缩摘要的最大字符数，超长截断（config.py:64 默认 500）


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

    # Redis 记忆为空/需要重置时，用 MySQL 永久档案整体覆盖 Redis 短期记忆
    # 不是"追加"，而是"用这份完整历史覆盖 Redis 里的旧值"——因为此时 Redis 是空的（刚查过没有才走到这里），所以覆盖即初始化
    def replace(self, session_public_id: str, messages: list[AiMessage]) -> None:
        if self.client is None:
            return
        key = self._key(session_public_id)

        # 开一个 pipeline（管道）：把多条 Redis 命令打包一次发给服务器，而不是每条往返一次。
        # 这里要 delete + rpush + ltrim + expire 四件事，用管道一趟搞定，省网络开销
        pipe = self.client.pipeline()

        # 先删掉这个 key 上已有的旧值——这就是"替换"语义的核心：旧的清空，新的整体写入
        pipe.delete(key)

        # delete 在外、rpush 在 if messages 里。如果 messages 为空，就只删不写（结果是这个 key 被清空）
        if messages:
            pipe.rpush(key, *[self._serialize(message.role, message.content) for message in messages])
            pipe.ltrim(key, -self.settings.redis_memory_max_messages, -1)
            pipe.expire(key, self.settings.redis_memory_ttl_seconds)
        try:
            pipe.execute()                                                  # 把管道里攒的这批命令真正发给 Redis 执行
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
                data = json.loads(raw)                              # json.loads 解决的是"字符串 → 字典"这一层
            except json.JSONDecodeError:
                continue
            role = str(data.get("role", "")).lower()
            content = str(data.get("content", ""))
            if role and content:

                # 读取时再脱敏是"出口兜底"（防御非本路径写入的脏数据）
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

            # decode_responses=True 解决的是"字节 → 字符串"这一层
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


# 把完整对话历史整理成"供 prompt 的有界历史 + 记忆摘要"：
# 脱敏后若超阈值则压成"摘要 + 最近 N 条原文"，否则原样返回；由 ContextAgent.act() 调用
def compact_history_for_prompt(
    history: list[AiMessage],                               # 完整历史消息列表，函数内部会先脱敏再按配置决定是否压缩
    settings: MemoryCompactionSettings,                     # 记忆压缩配置（Protocol）：是否启用/阈值条数/摘要上限
    current_input: str = "",                                # 本轮用户输入，用于摘要时标注"本轮关注"，可留空
) -> tuple[list[AiMessage], str]:
    """返回有界的历史（供 prompt 使用）+ 一份对学生的安全记忆摘要。

    摘要由确定性规则生成、避免诊断性标签，仅供 prompt 上下文与审计使用，
    不用于向学生展示。
    """

    # history 来自 self._load_history()——它优先从 Redis 读，Redis 空则从 MySQL 回填
    sanitized = [AiMessage(role=item.role, content=PrivacySanitizer().sanitize(item.content)) for item in history]
    if not sanitized:
        return [], "无相关历史记忆。"

    recent_count = max(2, int(getattr(settings, "memory_compaction_recent_messages", 8)))
    max_chars = max(120, int(getattr(settings, "memory_summary_max_chars", 500)))
    
    brief = summarize_history_for_memory(sanitized, current_input, max_chars)

    # 不需要压缩的分支
    if not getattr(settings, "memory_compaction_enabled", True) or len(sanitized) <= recent_count:
        return sanitized, brief                 # 即使不压缩，摘要也照样生成返回（它是"廉价兜底"，供后续 LLM 精炼用），只是历史不截断

    recent = sanitized[-recent_count:]          # 只留最近 8 条原文
    summary_message = AiMessage(
        role="system",
        content=(
            "历史摘要（仅供 MindBridge 内部上下文使用；不要向学生展示；"
            "不要据此输出诊断、风险等级或后台标签）：\n" + brief
        ),
    )
    return [summary_message, *recent], brief


# 规则式生成记忆摘要（不调模型）：
# "真正像摘要的那个"在 _summarize_memory（autonomous.py:618），不在 memory.py。
# 这个函数是 fallback——当模型不可用时，至少给一份"拼接版背景"，不让上下文空着
def summarize_history_for_memory(history: list[AiMessage], current_input: str = "", max_chars: int = 500) -> str:
    privacy = PrivacySanitizer()
    user_points = []
    assistant_points = []

    # 历史消息可能带换行、缩进、连续空格（尤其用户聊天时随手敲的），这些空白在摘要里没有信息量，反而会把"一条摘要"撑得七零八落
    # 把连续空白折叠成单个空格：换行、制表符、多个连续空格统统归一成一个空格，开头结尾的空白删掉
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
        # 只取学生最近说的 4 条，每条 _clip 截到 80 字符，用"；"连接，拼成"学生近期关注：..."这一段
        parts.append("学生近期关注：" + "；".join(_clip(item, 80) for item in user_points[-4:]))
    if assistant_points:
        parts.append("已给过的支持：" + "；".join(_clip(item, 70) for item in assistant_points[-3:]))
    if current_input:
        parts.append("本轮输入关注：" + _clip(privacy.sanitize(current_input), 80))
    if not parts:
        return "无相关历史记忆。"
    return _clip("\n".join(parts), max_chars)


# 把文本折叠空白后限长截断：超过 limit 字符则在 limit-3 处截断并补 "..."；空值/None 归一为空串。摘要各片段及最终摘要都用它控长
def _clip(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())                                 # 把任意输入收敛到规范形态
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."
