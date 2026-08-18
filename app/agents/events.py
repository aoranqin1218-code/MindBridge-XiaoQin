from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class AgentEventType(str, Enum):
    TURN_STARTED = "TURN_STARTED"                       # 一轮用户请求开始
    TASK_CREATED = "TASK_CREATED"                       # 新建任务
    TASK_CLAIMED = "TASK_CLAIMED"                       # 任务被认领
    TASK_RELEASED = "TASK_RELEASED"                     # 任务被释放/放弃认领
    TASK_CLOSED = "TASK_CLOSED"                         # 任务关闭
    MESSAGE_SENT = "MESSAGE_SENT"                       # 消息已发送
    ARTIFACT_PUBLISHED = "ARTIFACT_PUBLISHED"           # 业务产物已发布
    CRITIQUE_PUBLISHED = "CRITIQUE_PUBLISHED"           # 审查意见已发布
    REVISION_REQUESTED = "REVISION_REQUESTED"           # 请求修订候选回复
    SAFETY_OVERRIDE = "SAFETY_OVERRIDE"                 # 安全评估将本轮升级为高危
    FINAL_ACCEPTED = "FINAL_ACCEPTED"                   # 候选回复被最终采纳
    ROUND_STARTED = "ROUND_STARTED"                     # 一轮协作调度开始
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"               # 预算耗尽，未完成采纳


class TaskStatus(str, Enum):
    OPEN = "OPEN"                                       # 开放
    CLAIMED = "CLAIMED"                                 # 已认领
    BLOCKED = "BLOCKED"                                 # 阻塞
    CLOSED = "CLOSED"                                   # 关闭


class TaskPriority(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


PRIORITY_ORDER = {
    TaskPriority.LOW: 1,
    TaskPriority.NORMAL: 2,
    TaskPriority.HIGH: 3,
    TaskPriority.CRITICAL: 4,
}

# frozen=True 是 @dataclass 装饰器的一个参数
# 作用是把生成的对象变成不可变（immutable）的：对象创建后，它的字段不能再被重新赋值
# 用 dataclasses.replace(...) 基于旧对象造一个新对象，旧对象保持原样 - 版本迭代的感觉
@dataclass(frozen=True)
class AgentTask:
    id: str                                                                          # 任务唯一标识
    title: str                                                                       # 任务标题
    description: str = ""                                                            # 任务描述
    priority: TaskPriority = TaskPriority.NORMAL                                     # 优先级，决定调度顺序
    status: TaskStatus = TaskStatus.OPEN                                             # 状态：OPEN/CLAIMED/BLOCKED/CLOSED
    required_capabilities: frozenset[str] = field(default_factory=frozenset)         # 完成任务所需的能力集合
    created_by: str = "CoordinatorAgent"                                             # 创建该任务的 Agent 名

    # 认领的先后顺序（谁先认领、谁后认领），顺序本身有信息
    claimed_by: tuple[str, ...] = field(default_factory=tuple)                       # 已认领该任务的 Agent 列表
    depends_on: tuple[str, ...] = field(default_factory=tuple)                       # 依赖的前置任务 id
    metadata: dict[str, Any] = field(default_factory=dict)                           # 附加元数据

    # 数据照样在"变"，只是每变一次就产生一个新快照，旧的永远保持原样
    def claim(self, agent_name: str) -> "AgentTask":
        if agent_name in self.claimed_by:
            return self
        # 不可变 + replace = Git 提交
        return replace(self, status=TaskStatus.CLAIMED, claimed_by=(*self.claimed_by, agent_name))

    def reopen(self) -> "AgentTask":
        return replace(self, status=TaskStatus.OPEN)

    def close(self) -> "AgentTask":
        return replace(self, status=TaskStatus.CLOSED)


# 认领声明：谁、认领哪个任务、置信度多少、理由是什么
# 实际认领证据分散在 AgentDecision + AgentTask.claimed_by + TASK_CLAIMED 事件中
@dataclass(frozen=True)
class AgentClaim:
    agent: str                                                                       # 发起认领的 Agent 名
    task_id: str                                                                     # 想认领的任务 id
    confidence: float                                                                # 认领置信度，供 Coordinator 排序选择
    reason: str                                                                      # 认领理由


# Agent 之间的协作消息，经黑板转发
# 生产者：各 Agent 的 act(...)；经黑板 send_message 追加，messages_for 读取
@dataclass(frozen=True)
class AgentMessage:
    id: str                                                                          # 消息唯一标识
    sender: str                                                                      # 发送方 Agent 名
    recipient: str                                                                   # 接收方 Agent 名，"*" 表示广播
    content: str                                                                     # 消息正文
    task_id: str = ""                                                                # 关联的任务 id，无则空串
    kind: str = "REQUEST"                                                            # 消息类型：PROPOSAL / SAFETY_ASSESSMENT / CONTEXT_READY / REVIEW_REQUEST 等
    metadata: dict[str, Any] = field(default_factory=dict)                           # 附加信息


# Agent 产出的业务成果，写入黑板后不可变
# 生产者：各 Agent 的 act(...)；经黑板 add_artifact 追加，并触发 ARTIFACT_PUBLISHED / CRITIQUE_PUBLISHED 事件
# 消费者：Coordinator 按 kind 读取（intent / risk / context / response_proposal / critique）
@dataclass(frozen=True)
class AgentArtifact:
    id: str                                                                          # 产物唯一标识
    owner: str                                                                       # 产出该产物的 Agent 名
    kind: str                                                                        # 产物类型：intent / risk / context / response_proposal / critique
    payload: dict[str, Any]                                                          # 产物内容
    confidence: float = 1.0                                                          # 该产物的置信度
    task_id: str = ""                                                                # 关联的任务 id，无则空串
    metadata: dict[str, Any] = field(default_factory=dict)                           # 附加信息


# 黑板上"发生了什么"的状态变化证据（append-only 事件流）
# 由黑板方法自动追加（send_message / add_artifact / apply_turn_result 等），Agent 也可在 act 中显式返回
# 与保存成果的 AgentArtifact 相对：一个是过程记录，一个是业务产物
@dataclass(frozen=True)
class AgentEvent:
    type: AgentEventType                                                             # 事件类型，见 AgentEventType 枚举
    actor: str                                                                       # 触发该事件的 Agent 名
    task_id: str = ""                                                                # 关联的任务 id，无则空串
    artifact_id: str = ""                                                            # 关联的产物 id（如 ARTIFACT_PUBLISHED 事件）
    message: str = ""                                                                # 事件附加说明文字
    metadata: dict[str, Any] = field(default_factory=dict)                           # 附加信息


# 单个 Agent 一轮 act(...) 的产出打包：它这一轮要投给黑板的所有东西
# 生产者：各 Agent 的 act(...)；消费：Coordinator 经黑板 apply_turn_result 展开写入
@dataclass(frozen=True)
class AgentTurnResult:
    messages: tuple[AgentMessage, ...] = field(default_factory=tuple)                # 要发送的协作消息
    artifacts: tuple[AgentArtifact, ...] = field(default_factory=tuple)              # 要发布的业务产物
    tasks: tuple[AgentTask, ...] = field(default_factory=tuple)                      # 派生出的后续子任务
    events: tuple[AgentEvent, ...] = field(default_factory=tuple)                    # 本轮产生的自定义事件
    close_task: bool = True                                                          # 本轮结束后是否关闭当前任务


# 多 Agent 协作的共享黑板：一次用户请求全程的协作状态集中放在这里
# 由 EventDrivenRuntimeService.run 创建，传给 EventDrivenCoordinator.run 的调度循环
# 各 Agent 经 act() 读取它、产出经 apply_turn_result 追加回来，最终答复在这里被采纳
@dataclass(frozen=True)
class CollaborationBlackboard:
    # session 是微信里的"聊天窗口"，turn 是你在这个窗口里发的"每一条消息"
    turn_id: str                                                                     # 这一轮用户请求的唯一标识（uuid）
    user_id: int | None = None                                                       # 发起请求的用户 id，无则 None
    session_id: str = ""                                                             # 会话 id，用于跨轮归拢上下文
    user_input: str = ""                                                             # 用户本轮原始输入
    model_input: str = ""                                                            # 脱敏后喂给各 Agent 的输入
    tasks: dict[str, AgentTask] = field(default_factory=dict)                        # 任务表：id → AgentTask，同 id 用新对象覆盖更新
    messages: tuple[AgentMessage, ...] = field(default_factory=tuple)                # 协作消息历史（追加）
    artifacts: tuple[AgentArtifact, ...] = field(default_factory=tuple)              # 业务产物历史（追加）
    events: tuple[AgentEvent, ...] = field(default_factory=tuple)                    # 状态变化事件流（追加）
    final_artifact_id: str = ""                                                      # 被最终采纳的产物 id，空串表示尚未采纳

    # 新增任务：按 task.id 写入任务表
    def add_task(self, task: AgentTask) -> "CollaborationBlackboard":

        # 不想原地改 self.tasks - 先 dict(self.tasks) 复制一份，在副本上改，再用 replace() 生成新黑板
        tasks = dict(self.tasks)                                                # dict(self.tasks) 不是给它套一层，而是产出一个内容一模一样、但全新的字典对象-id不同
        tasks[task.id] = task
        return replace(self, tasks=tasks)

    # 更新任务：同 id 用新对象覆盖旧状态（内部复用 add_task）
    def update_task(self, task: AgentTask) -> "CollaborationBlackboard":
        return self.add_task(task)

    # 追加一个事件到事件流
    def append_event(self, event: AgentEvent) -> "CollaborationBlackboard":
        return replace(self, events=(*self.events, event))

    # 追加一条协作消息，并自动记录 MESSAGE_SENT 事件
    def send_message(self, message: AgentMessage) -> "CollaborationBlackboard":
        return replace(self, messages=(*self.messages, message)).append_event(
            AgentEvent(
                type=AgentEventType.MESSAGE_SENT,
                actor=message.sender,
                task_id=message.task_id,
                message=message.content,
                metadata={"recipient": message.recipient, "kind": message.kind},
            )
        )

    # 追加一个业务产物，并自动记录 ARTIFACT_PUBLISHED / CRITIQUE_PUBLISHED 事件
    def add_artifact(self, artifact: AgentArtifact) -> "CollaborationBlackboard":
        event_type = AgentEventType.CRITIQUE_PUBLISHED if artifact.kind == "critique" else AgentEventType.ARTIFACT_PUBLISHED
        return replace(self, artifacts=(*self.artifacts, artifact)).append_event(
            AgentEvent(
                type=event_type,
                actor=artifact.owner,
                task_id=artifact.task_id,
                artifact_id=artifact.id,
                message=artifact.kind,
                metadata={"confidence": artifact.confidence},
            )
        )

    # 把某个 Agent 一轮的产出（消息/产物/子任务/事件）全部展开写入黑板
    def apply_turn_result(self, task: AgentTask, agent_name: str, result: AgentTurnResult) -> "CollaborationBlackboard":
        board = self
        for message in result.messages:
            board = board.send_message(message)
        for artifact in result.artifacts:
            board = board.add_artifact(artifact)

        # 派生子任务
        for follow_up in result.tasks:
            if follow_up.id not in board.tasks:
                board = board.add_task(follow_up).append_event(
                    AgentEvent(
                        type=AgentEventType.TASK_CREATED,
                        actor=agent_name,
                        task_id=follow_up.id,
                        message=follow_up.title,
                    )
                )
        if result.close_task:
            board = board.update_task(task.close()).append_event(
                AgentEvent(type=AgentEventType.TASK_CLOSED, actor=agent_name, task_id=task.id, message=task.title)
            )
        else:
            # reopen 分支在真实流程中永远走不到，和 AgentClaim 一样属于"预留设计"
            # 目前没有加append_event，后续用到需要加上
            board = board.update_task(task.reopen())

        # 其他事件
        for event in result.events:
            board = board.append_event(event)
        return board

    # 列出所有 OPEN 状态的任务（供 Coordinator 分派）
    def open_tasks(self) -> list[AgentTask]:
        return [task for task in self.tasks.values() if task.status == TaskStatus.OPEN]

    # 按 kind 过滤产物列表
    def artifacts_by_kind(self, kind: str) -> list[AgentArtifact]:
        return [artifact for artifact in self.artifacts if artifact.kind == kind]

    # 取某 kind 的最新产物（owner 为 None 时不限生产者）
    def latest_artifact(self, kind: str, owner: str | None = None) -> AgentArtifact | None:
        for artifact in reversed(self.artifacts):
            if artifact.kind == kind and (owner is None or artifact.owner == owner):
                return artifact
        return None

    # 取发给指定 Agent 的消息（recipient 为该名或 "*" 广播）
    def messages_for(self, agent_name: str) -> list[AgentMessage]:
        return [message for message in self.messages if message.recipient in {agent_name, "*"}]

    # 是否已有某 kind 的产物
    def has_artifact(self, kind: str) -> bool:
        return self.latest_artifact(kind) is not None

    # 返回被最终采纳的产物（由 final_artifact_id 指向）
    def accepted_artifact(self) -> AgentArtifact | None:
        if not self.final_artifact_id:
            return None
        return next((artifact for artifact in self.artifacts if artifact.id == self.final_artifact_id), None)

    # 标记最终采纳某产物，并记录 FINAL_ACCEPTED 事件
    # 当前的reason是写死的常量：accepted after autonomous response proposal and SafetyAgent approval
    def accept_final(self, artifact_id: str, actor: str, reason: str) -> "CollaborationBlackboard":
        return replace(self, final_artifact_id=artifact_id).append_event(
            AgentEvent(
                type=AgentEventType.FINAL_ACCEPTED,
                actor=actor,
                artifact_id=artifact_id,
                message=reason,
            )
        )
