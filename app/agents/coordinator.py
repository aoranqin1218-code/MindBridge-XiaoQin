from __future__ import annotations

import uuid
from collections import defaultdict

from app.agents.autonomous import CoordinatorAgent
from app.agents.events import (
    PRIORITY_ORDER,
    AgentEvent,
    AgentEventType,
    AgentTask,
    CollaborationBlackboard,
    TaskPriority,
)
from app.agents.registry import AgentCapability, AgentRegistry
from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.services.ai import has_high_risk_signal


class EventDrivenCoordinator:
    """基于认领的协调器：负责轮次/认领预算与最终采纳策略。

    不写死 Agent 的链式执行顺序；所有执行都来自 Agent 主动认领黑板上的公开任务。
    """

    # 读取调度预算与采纳门槛配置
    def __init__(self, registry: AgentRegistry, coordinator_agent: CoordinatorAgent, settings: Settings):
        self.registry = registry
        self.coordinator_agent = coordinator_agent
        self.settings = settings
        self.max_rounds = int(getattr(settings, "agent_max_rounds", 8))                                     # 调度循环最大轮数：8 轮内未采纳则预算耗尽
        self.max_claims_per_round = int(getattr(settings, "agent_max_claims_per_round", 4))                 # 每轮最多认领并执行的任务数
        self.max_claims_per_agent = int(getattr(settings, "agent_max_claims_per_agent", 3))                 # 整个调度中单个 Agent 最多认领任务的次数
        self.final_min_confidence = float(getattr(settings, "agent_final_acceptance_min_confidence", 0.6))  # 最终采纳的最低置信度门槛（低于 0.6 不采纳）

    # 调度主循环：每轮先确保根任务、派缺的任务、尝试采纳，再认领执行；采纳成功或预算耗尽即返回
    def run(self, board: CollaborationBlackboard) -> CollaborationBlackboard:

        # 黑板空 → 建根任务（task:root）并记事件；黑板非空 → 什么都不做原样返回
        board = self._ensure_root_task(board)

        claim_counts: dict[str, int] = defaultdict(int)
        for round_number in range(1, self.max_rounds + 1):
            board = board.append_event(
                AgentEvent(
                    type=AgentEventType.ROUND_STARTED,
                    actor=self.coordinator_agent.name,
                    message=f"round={round_number}",
                    metadata={"round": round_number},
                )
            )

            board = self._derive_missing_work(board)                                # 按黑板当前产物缺失情况补派任务

            # 走到这，说明还没有最终的回复
            # 获取本轮要执行的任务 + 认领它们的 Agent 的配对列表，每轮最多 4 个 - 本轮排班表
            candidates = self._claim_candidates(board, claim_counts)                

            # 没有任何 Agent 愿意认领任何任务
            if not candidates:
                board = self._derive_missing_work(board, force_response=True)       # 无条件创建 task:propose-response（跳过 intent/risk/context 前置检查）    
                candidates = self._claim_candidates(board, claim_counts)
                if not candidates:
                    break
            
            for task, candidate in candidates:

                # 以当前黑板为准，取这个任务的最新版本；万一黑板里已经没了（被删/异常），才退回用快照 task
                current_task = board.tasks.get(task.id, task)

                # 执行任务认领，同时更新板报
                board = board.update_task(current_task.claim(candidate.agent.profile.name)).append_event(
                    AgentEvent(
                        type=AgentEventType.TASK_CLAIMED,
                        actor=candidate.agent.profile.name,
                        task_id=task.id,
                        message=candidate.decision.reason,
                        metadata={"confidence": candidate.decision.confidence},
                    )
                )

                result = candidate.agent.act(current_task, board)

                board = board.apply_turn_result(current_task, candidate.agent.profile.name, result)

                claim_counts[candidate.agent.profile.name] += 1
            
            board = self._derive_missing_work(board)
            board = self._try_accept_final(board)
            if board.final_artifact_id:
                return board

        # 额度用完了
        return board.append_event(
            AgentEvent(
                type=AgentEventType.BUDGET_EXHAUSTED,
                actor=self.coordinator_agent.name,
                message="event-driven agent budget exhausted before final acceptance",
            )
        )

    # 保证黑板有根任务：首次进入时由 CoordinatorAgent 创建"Resolve user turn"根任务，否则原样返回
    def _ensure_root_task(self, board: CollaborationBlackboard) -> CollaborationBlackboard:

        # 黑板里有没有任何任务"。只要有一个任务，就不再创建根任务
        if board.tasks:
            return board
        root = self.coordinator_agent.root_task(board)
        return board.add_task(root).append_event(
            AgentEvent(type=AgentEventType.TASK_CREATED, actor=self.coordinator_agent.name, task_id=root.id, message=root.title)
        )

    # 缺啥派啥：按黑板当前产物缺失情况补派 intent/risk/context/response 及审查、修订任务
    def _derive_missing_work(self, board: CollaborationBlackboard, force_response: bool = False) -> CollaborationBlackboard:
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="intent",
            task_id="task:understand",
            title="Understand user turn",
            capability=AgentCapability.UNDERSTANDING,
            priority=TaskPriority.HIGH,
            condition=board.user_input != "",
        )
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="risk",
            task_id="task:assess-safety",
            title="Assess safety risk",
            capability=AgentCapability.SAFETY,
            priority=TaskPriority.CRITICAL if has_high_risk_signal(board.user_input) else TaskPriority.HIGH,
            condition=board.user_input != "",
        )

        intent = board.intent_value()
        risk = board.risk_value()

        # 是否需要上下文
        needs_context = intent in {IntentType.CONSULT, IntentType.RISK} or risk in {RiskLevel.MEDIUM, RiskLevel.HIGH}
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="context",
            task_id="task:gather-context",
            title="Gather contextual evidence",
            capability=AgentCapability.CONTEXT,
            priority=TaskPriority.CRITICAL if risk == RiskLevel.HIGH else TaskPriority.NORMAL,
            condition=needs_context,
        )

        # 是否需要生成回复的判断 - 注意是回复
        # force_response：绕过前置条件、强制派发回复任务 —— 防死锁的破局开关
        # 默认False，除非这一轮没有任何 Agent 认领任何任务
        can_request_response = force_response or (
            board.latest_artifact("intent") is not None
            and board.latest_artifact("risk") is not None
            # 三种情况：①纯闲聊（低风险、非咨询）②咨询/中风险，且 context 已就绪 ③高危 - 安全第一，不等上下文，立刻出危机处置回复
            and (not needs_context or board.latest_artifact("context") is not None or risk == RiskLevel.HIGH)
        )
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="response_proposal",
            task_id="task:propose-response",
            title="Propose candidate response",
            capability=AgentCapability.RESPONSE,
            priority=TaskPriority.CRITICAL if risk == RiskLevel.HIGH else TaskPriority.HIGH,
            condition=can_request_response,
        )
        
        response = board.latest_artifact("response_proposal")
        review = board.latest_artifact("safety_review")
        critique = board.latest_artifact("critique")

        # 有回复，但这份还没被审过" → 建审查任务
        if response and (review is None or review.metadata.get("responseArtifactId") != response.id):
            board = self._ensure_task(
                board,
                AgentTask(
                    id=f"task:review-response:{response.id}",
                    title="Review candidate response safety",
                    description="Safety review is required before final acceptance.",
                    priority=TaskPriority.CRITICAL if risk == RiskLevel.HIGH else TaskPriority.HIGH,
                    required_capabilities=frozenset({AgentCapability.SAFETY.value}),
                    created_by=self.coordinator_agent.name,
                    metadata={"kind": "safety_review", "responseArtifactId": response.id},
                ),
            )

        # 审过了但不通过 → 建修订任务
        # 它和上一段的审查任务一起构成 propose → review → revise → re-review 的闭环
        if critique and critique.payload.get("approved") is False:
            board = self._ensure_task(
                board,
                AgentTask(
                    id=f"task:revise-response:{critique.id}",
                    title="Revise response after critique",
                    description=str(critique.payload.get("reason", "Safety critique requested revision.")),
                    priority=TaskPriority.CRITICAL,
                    required_capabilities=frozenset({AgentCapability.RESPONSE.value}),
                    created_by=self.coordinator_agent.name,
                    metadata={"kind": "response", "revisionOf": critique.payload.get("responseArtifactId", "")},
                ),
            )
        return board

    # 若指定产物缺失且条件满足，则派发一个对应能力的开放任务（公共补任务入口）
    def _ensure_task_for_missing_artifact(
        self,
        board: CollaborationBlackboard,                                 # 黑板：从这里判断产物是否缺失、任务加到哪里
        artifact_kind: str,                                             # 缺失的产物类型（intent/risk/context/response_proposal）
        task_id: str,                                                   # 补派任务的固定 id（task:understand 等，同 id 不重复派）
        title: str,                                                     # 任务标题
        capability: AgentCapability,                                    # 完成任务所需能力，决定哪些 Agent 可以认领
        priority: TaskPriority,                                         # 任务优先级，决定调度顺序
        condition: bool,                                                # 满足该条件才派发（如输入为空则不派）
    ) -> CollaborationBlackboard:
        if not condition or board.latest_artifact(artifact_kind) is not None:
            return board
        return self._ensure_task(
            board,
            AgentTask(
                id=task_id,
                title=title,
                description=board.user_input,
                priority=priority,
                required_capabilities=frozenset({capability.value}),
                created_by=self.coordinator_agent.name,
                metadata={"kind": artifact_kind},
            ),
        )

    # 将任务加入黑板（已存在则跳过），并记录 TASK_CREATED 事件
    def _ensure_task(self, board: CollaborationBlackboard, task: AgentTask) -> CollaborationBlackboard:
        if task.id in board.tasks:
            return board
        return board.add_task(task).append_event(
            AgentEvent(type=AgentEventType.TASK_CREATED, actor=self.coordinator_agent.name, task_id=task.id, message=task.title)
        )

    # 每个调度轮次里的"认领裁判"——从所有 OPEN 任务里，挑出"这一轮哪些 Agent 去执行哪些任务"的批次，返回给 run() 逐个执行
    def _claim_candidates(self, board: CollaborationBlackboard, claim_counts: dict[str, int]):
        selected = []

        task_candidates = []

        for task in board.open_tasks():
            # 遍历愿意认领该任务的所有 Agent（候选人candidate）
            for candidate in self.registry.candidate_decisions_for(task, board):
                # 判断这个 Agent 在整个调度里"累计认领并执行的任务数"是不是已经达到上限
                if claim_counts[candidate.agent.profile.name] >= self.max_claims_per_agent:
                    continue
                task_candidates.append((task, candidate))

        # 优先级 → 置信度 → 名字" 降序排候选池
        task_candidates.sort(
            key=lambda item: (
                PRIORITY_ORDER[item[0].priority],                                       # ① 任务优先级（转成数字）
                item[1].decision.confidence,                                            # ② 认领置信度
                item[1].agent.profile.name,                                             # ③ Agent 名
            ),
            reverse=True,
        )

        # seen 是横向防重——同一对 (任务, Agent) 不重复
        # selected_agents 是纵向防贪心——一个 Agent 不许一口气干好几个活，把机会让给别人
        seen = set()
        selected_agents = set()

        for task, candidate in task_candidates:
            key = (task.id, candidate.agent.profile.name)
            if key in seen or candidate.agent.profile.name in selected_agents:
                continue
            selected.append((task, candidate))
            seen.add(key)
            selected_agents.add(candidate.agent.profile.name)
            if len(selected) >= self.max_claims_per_round:                              # 每轮最多执行 4 个任务
                break
        return selected

    # 尝试最终采纳：回复与审查齐全、审查通过且置信度达标时才采纳为最终结果
    def _try_accept_final(self, board: CollaborationBlackboard) -> CollaborationBlackboard:

        if board.final_artifact_id:
            return board
        
        response = board.latest_artifact("response_proposal")
        review = board.latest_artifact("safety_review")

        if response is None or review is None:
            return board
        if review.metadata.get("responseArtifactId") != response.id:
            return board
        if not review.payload.get("approved"):
            return board
        if response.confidence < self.final_min_confidence:
            return board
        
        reason = "accepted after autonomous response proposal and SafetyAgent approval"

        self.coordinator_agent.remember_acceptance(response.id, reason)

        return board.accept_final(response.id, self.coordinator_agent.name, reason)

