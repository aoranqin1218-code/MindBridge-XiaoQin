from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from app.agents.events import AgentTask, AgentTurnResult, CollaborationBlackboard


# 各 Agent 的能力类型，标识其职责方向
# 任务用 required_capabilities 声明"需要哪几种能力"，Registry 据此筛选候选 Agent
class AgentCapability(str, Enum):
    UNDERSTANDING = "UNDERSTANDING"                       # 意图理解：分析用户这轮想干什么
    SAFETY = "SAFETY"                                     # 安全评估：风险审查，必要时打回候选回复
    CONTEXT = "CONTEXT"                                   # 上下文检索：查短期记忆与知识库
    RESPONSE = "RESPONSE"                                 # 回复生成：产出最终候选回复
    COORDINATION = "COORDINATION"                         # 协调调度：Coordinator 专属，不参与普通任务认领


# Agent 的静态画像：叫什么、能干什么、用什么提示词/记忆/模型/工具
# 各 Agent 类构造时一次性定义，Registry 只读不写
@dataclass(frozen=True)
class AgentProfile:
    name: str                                                                 # Agent 唯一名称
    capabilities: frozenset[AgentCapability] = field(default_factory=frozenset)  # 该 Agent 具备的能力集合
    system_prompt: str = ""                                                   # 该 Agent 专用的系统提示词
    memory_policy: str = "none"                                               # 私有记忆策略：决定记忆键名；"none" 不启用，实际值为 private_* 系列
    model_profile: str = "default"                                            # 模型档位名：按角色选不同模型（understanding/safety/context/response/coordinator）
    tool_permissions: frozenset[str] = field(default_factory=frozenset)       # 可调用的工具权限白名单（如 "llm.intent"）


# 单个 Agent 对某个任务的认领决策：要不要认领、多自信、为什么
@dataclass(frozen=True)
class AgentDecision:
    claim: bool                                                               # 是否认领该任务（True 认领 / False 放弃）
    confidence: float = 0.0                                                   # 认领置信度 0~1，供 Coordinator 排序选择
    reason: str = ""                                                          # 认领/不认领的理由


# 自治 Agent 的结构化协议（鸭子类型）：任何实现了 profile / decide / act 的对象都算 AutonomousAgent
# 各 Agent 类无需显式继承它，只要"长得像"即可被 Registry 使用
class AutonomousAgent(Protocol):
    profile: AgentProfile                                                     # 该 Agent 的静态画像（能力、模型、工具等）

    # 判断是否认领某个任务，返回认领决策（含置信度）
    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        ...

    # 实际执行任务，返回本轮产出（消息/产物/子任务/事件）
    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        ...


# 认领候选 = 一个 Agent 加上它对当前任务的认领决策
@dataclass(frozen=True)
class AgentCandidate:
    agent: AutonomousAgent                                                    # 候选 Agent
    decision: AgentDecision                                                   # 它的认领决策（含置信度，用于排序）


# Agent 注册表：持有所有可用 Agent，负责"按能力筛选 → 询问认领意愿 → 按置信度排序"
class AgentRegistry:
    # 保存 Agent 列表副本，避免外部后续改动影响内部状态
    def __init__(self, agents: list[AutonomousAgent]):
        self._agents = list(agents)

    # 以副本形式暴露 Agent 列表，防止调用方直接改内部列表
    @property
    def agents(self) -> list[AutonomousAgent]:
        return list(self._agents)

    # 返回愿意认领该任务的所有 Agent（丢弃决策信息，只留 Agent 本身）
    def candidates_for(self, task: AgentTask, board: CollaborationBlackboard) -> list[AutonomousAgent]:
        return [candidate.agent for candidate in self.candidate_decisions_for(task, board)]

    # 核心筛选：能力过关的 Agent 逐个问 decide，收集 claim=True 的，按置信度降序返回
    def candidate_decisions_for(self, task: AgentTask, board: CollaborationBlackboard) -> list[AgentCandidate]:
        candidates = []
        for agent in self._agents:
            if not self._has_required_capability(agent, task):
                continue
            decision = agent.decide(task, board)                                        # 每个 Agent 自己的判断逻辑，跑在 autonomous.py
            if decision.claim:                                                          # claim = True 才收进来
                candidates.append(AgentCandidate(agent, decision))  
        return sorted(candidates, key=lambda item: item.decision.confidence, reverse=True)

    # 判断 Agent 是否具备任务所需的全部能力；任务无要求则人人过关
    # 注意：任务侧 required_capabilities 存的是枚举的 value 字符串，故先把 Agent 能力也转成字符串集合再比
    def _has_required_capability(self, agent: AutonomousAgent, task: AgentTask) -> bool:
        if not task.required_capabilities:
            return True
        agent_capabilities = {capability.value for capability in agent.profile.capabilities}
        return set(task.required_capabilities).issubset(agent_capabilities)
