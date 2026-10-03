
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.enums import RiskLevel, ToolJobKind
from app.models.entities import PsychologicalReport, ToolAuditRecord, ToolJob


# 工具策略——描述某个工具任务在什么条件下允许执行、依赖哪些前置任务
@dataclass(frozen=True)
class ToolPolicy:
    name: str                                               # 工具任务名（等于 ToolJobKind 的某个值）
    description: str                                        # 工具功能描述，喂给 AI 工具循环做决策
    allowed_risks: tuple[str, ...]                          # 允许执行该工具的风险等级集合
    requires_report: bool = True                            # 是否要求携带心理报告
    function_name: str | None = None                        # 暴露给 AI 的函数名，None 表示不暴露
    dependencies: tuple[str, ...] = ()                      # 依赖的前置工具任务，需先成功才能执行
    exposed_to_tool_loop: bool = False                      # 是否暴露给工具循环调用


# 工具策略注册表——集中定义所有工具任务的策略，并提供查询与授权入口
class ToolPolicyRegistry:
    # 各工具任务的策略表，key 是工具任务名（ToolJobKind 的值）
    POLICIES: dict[str, ToolPolicy] = {
        ToolJobKind.EXCEL_REPORT.value: ToolPolicy(
            name=ToolJobKind.EXCEL_REPORT.value,
            description="Write a psychological report row into the counselor-facing Excel ledger.",
            allowed_risks=(RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value),
            function_name="mindbridge_excel_report",
            exposed_to_tool_loop=True,
        ),
        ToolJobKind.CASE_CREATE.value: ToolPolicy(
            name=ToolJobKind.CASE_CREATE.value,
            description="Create or reuse a counselor-facing risk case for medium/high risk reports.",
            allowed_risks=(RiskLevel.MEDIUM.value, RiskLevel.HIGH.value),
            function_name="mindbridge_case_create",
            exposed_to_tool_loop=True,
        ),
        ToolJobKind.ALERT_SEND.value: ToolPolicy(
            name=ToolJobKind.ALERT_SEND.value,
            description="Send or log an urgent counselor alert for high risk reports.",
            allowed_risks=(RiskLevel.HIGH.value,),
            function_name="mindbridge_alert_send",
            dependencies=(ToolJobKind.CASE_CREATE.value,),
            exposed_to_tool_loop=True,
        ),
        ToolJobKind.RISK_ALERT.value: ToolPolicy(
            name=ToolJobKind.RISK_ALERT.value,
            description="Legacy high-risk alert action; retained for compatibility.",
            allowed_risks=(RiskLevel.HIGH.value,),
        ),
    }

    @classmethod
    def policy_for(cls, tool_name: str) -> ToolPolicy | None:
        """按工具名查对应策略，找不到返回 None。"""
        return cls.POLICIES.get(tool_name)

    @classmethod
    def tool_loop_policies(cls) -> tuple[ToolPolicy, ...]:
        """取出所有暴露给工具循环的策略。"""
        return tuple(policy for policy in cls.POLICIES.values() if policy.exposed_to_tool_loop)

    @classmethod
    def policy_for_function(cls, function_name: str) -> ToolPolicy | None:
        """按函数名在暴露给工具循环的策略里查，找不到返回 None。"""
        return next(
            (policy for policy in cls.tool_loop_policies() if policy.function_name == function_name),
            None,
        )

    @classmethod
    def required_tool_kinds(cls, risk_level: str) -> tuple[str, ...]:
        """某个风险等级下需要执行哪些工具任务（按策略允许的风险等级匹配）。"""
        return tuple(
            policy.name
            for policy in cls.tool_loop_policies()
            if risk_level in policy.allowed_risks
        )

    @classmethod
    def function_tools(cls) -> tuple[dict[str, Any], ...]:
        """生成 OpenAI function-calling 格式的工具定义，供 AI 工具循环调用。"""
        return tuple(
            {
                "type": "function",                      # 工具类型，OpenAI function-calling 固定为 "function"
                "function": {                            # 函数定义对象
                    "name": policy.function_name,        # 函数名，即暴露给 AI 的 function_name
                    "description": policy.description,   # 函数功能描述，供 AI 判断何时调用
                    "strict": True,                      # 严格模式，强制校验参数符合 parameters schema
                    "parameters": {                      # 函数的参数 schema（JSON Schema）
                        "type": "object",                # 参数整体是对象类型
                        "properties": {},                # 参数字段定义（当前为空，即不接收参数）
                        "required": [],                  # 必填参数列表（当前为空）
                        "additionalProperties": False,   # 不允许传入未定义的额外参数
                    },
                },
            }
            for policy in cls.tool_loop_policies()
        )

    @classmethod
    def authorize(cls, tool_name: str, report: PsychologicalReport | None) -> tuple[bool, str, ToolPolicy | None]:
        """判断某工具能否处理该报告，返回 (是否允许, 原因, 策略)。"""
        policy = cls.policy_for(tool_name)
        if policy is None:
            return False, f"未知工具：{tool_name}", None
        if policy.requires_report and report is None:
            return False, "工具执行需要心理报告，但未找到 report", policy

        risk = report.risk_level if report is not None else ""
        if risk not in policy.allowed_risks:
            return False, f"工具 {tool_name} 不允许处理风险等级 {risk}", policy
        return True, "允许执行", policy


# 工具治理服务——负责工具任务执行前的授权审计与执行后的收尾记录
class ToolGovernanceService:
    def __init__(self, db: Session):
        # 数据库会话，用于读写审计记录
        self.db = db

    def start_job(self, job: ToolJob, report: PsychologicalReport | None) -> ToolAuditRecord:
        """开始一个工具任务：先做授权审计，把结果写入一条审计记录并返回。"""
        allowed, reason, policy = ToolPolicyRegistry.authorize(job.kind, report)

        record = ToolAuditRecord(
            job_id=job.id,
            report_id=job.report_id,
            tool_name=job.kind,
            policy=policy.name if policy else "unknown",
            allowed=allowed,
            status="AUTHORIZED" if allowed else "BLOCKED",
            reason=reason,
            payload=_json(
                {
                    "schemaVersion": 1,
                    "decision": {
                        "attempt": job.attempts,
                        "riskLevelAtDecision": report.risk_level if report is not None else None,
                        "policySnapshot": asdict(policy) if policy else None,
                    },
                }
            ),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def require_allowed(self, job: ToolJob, report: PsychologicalReport | None) -> None:
        """校验工具任务是否被允许执行，不允许则抛异常。"""
        self.require_allowed_tool(job.kind, report)

    def require_allowed_tool(self, tool_name: str, report: PsychologicalReport | None) -> None:
        """父任务内部的每次动作也必须经过同一授权入口。"""
        allowed, reason, _ = ToolPolicyRegistry.authorize(tool_name, report)
        if not allowed:
            raise RuntimeError(reason)

    def finish(self, record: ToolAuditRecord, status: str, reason: str = "", payload: dict[str, Any] | None = None) -> ToolAuditRecord:
        """收尾一个工具任务：更新审计记录的状态与结果并落库。"""
        record.status = status
        record.reason = reason or record.reason
        if payload is not None:
            audit_payload = json.loads(record.payload)
            audit_payload["result"] = payload
            record.payload = _json(audit_payload)
        record.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.db.add(record)
        self.db.commit()
        return record


def _json(value: Any) -> str:
    """把对象序列化为 JSON 字符串，中文不转义、不可序列化对象转成 str。"""
    return json.dumps(value, ensure_ascii=False, default=str)
