"""Database/side-effect adapter. Core Tool Loop does not own SQLAlchemy sessions."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from functools import partial
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.core.config import Settings

from app.core.enums import ToolJobKind, ToolStatus
from app.models.entities import AlertRecord, ExcelRecord, PsychologicalReport, RiskCase, ToolAuditRecord, ToolJob
from app.services.ai import AiClient
from app.services.tool_governance import ToolGovernanceService, ToolPolicyRegistry
from app.services.tool_loop import (
    BoundedToolLoopRunner, ToolExecutionResult, ToolLoopEvent, ToolLoopFailure,
    ToolLoopRegistry, ToolCallingClient, ToolLoopExecutionContext, ToolLoopResult,
)
from app.services.tools import ToolOrchestrationService


class ToolLoopRuntime:
    def __init__(self, settings: Settings, session_factory: Callable[[], Session], *,
                 ai: ToolCallingClient | None = None,
                 is_cancelled: Callable[[], bool] | None = None, email_limiter=None):
        self.settings = settings
        self.session_factory = session_factory
        self.ai = ai or AiClient(settings)
        self.is_cancelled = is_cancelled or (lambda: False)
        self.email_limiter = email_limiter
        self.operations = {
            ToolJobKind.EXCEL_REPORT.value: self._excel,
            ToolJobKind.CASE_CREATE.value: self._case,
            ToolJobKind.ALERT_SEND.value: self._alert,
        }
        self.completion_models = {
            ToolJobKind.EXCEL_REPORT.value: (ExcelRecord, {"status": ToolStatus.SUCCESS.value}),
            ToolJobKind.CASE_CREATE.value: (RiskCase, {}),
            ToolJobKind.ALERT_SEND.value: (AlertRecord, {"status": ToolStatus.SUCCESS.value}),
        }
        expected = {p.name for p in ToolPolicyRegistry.tool_loop_policies()}
        if set(self.operations) != expected or set(self.completion_models) != expected:
            raise ValueError("每个暴露的工具都必须注册业务 handler 和持久化完成判据")

    async def run(self, job_id: int) -> ToolLoopResult:
        with self.session_factory() as db:
            job = db.get(ToolJob, job_id)
            report = db.get(PsychologicalReport, job.report_id) if job else None
            if report is None or not ToolPolicyRegistry.required_tool_kinds(report.risk_level):
                raise ToolLoopFailure("报告缺失或风险等级非法", retryable=False)
            report_id, risk_level = report.id, report.risk_level
        registry = ToolLoopRegistry({
            policy.function_name: partial(self._execute, job_id, risk_level)
            for policy in ToolPolicyRegistry.tool_loop_policies()
        })
        runner = BoundedToolLoopRunner(
            self.ai, registry, self.completed,
            max_rounds=self.settings.tool_loop_max_rounds,
            timeout_seconds=self.settings.tool_loop_timeout_seconds,
            event_sink=partial(self.trace, job_id), is_cancelled=self.is_cancelled,
        )
        return await runner.run(report_id, risk_level)

    def completed(self, report_id: int) -> set[str]:
        with self.session_factory() as db:
            return {kind for kind, (model, fields) in self.completion_models.items()
                    if db.query(model).filter_by(report_id=report_id, **fields).first() is not None}

    async def _execute(self, job_id: int, risk_level: str,
                       context: ToolLoopExecutionContext) -> ToolExecutionResult:
        # A cancelled coroutine cannot interrupt a running file/SMTP write. Drain it
        # before returning to the queue, so retry cannot overlap the previous effect.
        pending = asyncio.create_task(asyncio.to_thread(self._execute_sync, job_id, risk_level, context))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            await pending
            raise

    def _execute_sync(self, job_id: int, risk_level: str,
                      context: ToolLoopExecutionContext) -> ToolExecutionResult:
        with self.session_factory() as db:
            job = db.get(ToolJob, job_id)
            report = db.get(PsychologicalReport, context.report_id)
            governance = ToolGovernanceService(db)
            allowed, reason, policy = ToolPolicyRegistry.authorize(context.policy.name, report)
            if report is None or report.risk_level != risk_level:
                allowed, reason = False, "报告风险已变更，需重新规划"
            dependencies = set(policy.dependencies) if policy else set()
            if not dependencies.issubset(self.completed(context.report_id)):
                allowed, reason = False, "前置结果未完成"
            if self.is_cancelled():
                allowed, reason = False, "服务正在停止"
            record = ToolAuditRecord(
                job_id=job_id, report_id=context.report_id, tool_name=context.policy.name,
                policy=context.policy.name, allowed=allowed,
                status="AUTHORIZED" if allowed else "BLOCKED", reason=reason,
                payload=json.dumps({"schemaVersion": 1, "decision": {
                    "attempt": job.attempts, "riskLevelAtDecision": report.risk_level if report else None,
                    "policySnapshot": asdict(policy) if policy else None,
                    "round": context.round_number,
                    "callId": context.call_id,
                }}, ensure_ascii=False),
            )
            db.add(record)
            db.commit()
            if not allowed:
                raise ToolLoopFailure(reason, retryable=report is not None)
            # Last gate before any side effect: service authorization cannot be bypassed.
            governance.require_allowed_tool(context.policy.name, report)
            try:
                resource = self.operations[context.policy.name](ToolOrchestrationService(db, self.settings), report)
                governance.finish(record, "SUCCESS", payload=resource)
            except Exception as exc:
                db.rollback()
                governance.finish(record, "FAILED", reason=type(exc).__name__)
                raise
            return ToolExecutionResult(
                success=True, tool=context.policy.name, report_id=report.id,
                status="SUCCESS", message="处置结果已持久化", resource=resource,
            )

    def _excel(self, tools: ToolOrchestrationService, report: PsychologicalReport) -> dict[str, Any]:
        record = tools.write_excel(report)
        if record.status != ToolStatus.SUCCESS.value:
            raise RuntimeError("Excel 留档失败")
        return {"excelRecordId": record.id, "status": record.status}

    def _case(self, tools: ToolOrchestrationService, report: PsychologicalReport) -> dict[str, Any]:
        case = tools.create_case(report)
        return {"caseId": case.id, "status": case.status}

    def _alert(self, tools: ToolOrchestrationService, report: PsychologicalReport) -> dict[str, Any]:
        if self.email_limiter:
            allowed, retry_after = self.email_limiter.allow()
            if not allowed:
                raise ToolLoopFailure("预警限流，等待队列重试", retryable=True, retry_after=retry_after)
        case = tools.db.query(RiskCase).filter_by(report_id=report.id).one()
        record = tools.send_case_alert(case)
        if record.status != ToolStatus.SUCCESS.value:
            raise RuntimeError("预警发送失败")
        return {"alertRecordId": record.id, "caseId": case.id, "status": record.status,
                "deliveryMode": self.settings.alert_email_delivery_mode}

    def trace(self, job_id: int, event: ToolLoopEvent):
        with self.session_factory() as db:
            job = db.get(ToolJob, job_id)
            policy = ToolPolicyRegistry.policy_for_function(event.tool_name or "")
            payload = event.payload()
            payload["attempt"] = job.attempts
            # Never store model content, argument values, request headers, or raw errors.
            serialized = json.dumps(payload, ensure_ascii=False)
            for secret in (self.settings.openai_api_key, self.settings.smtp_password, self.settings.smtp_username):
                if secret:
                    serialized = serialized.replace(secret, "[REDACTED]")
            db.add(ToolAuditRecord(
                job_id=job.id, report_id=job.report_id,
                tool_name=policy.name if policy else ToolJobKind.TOOL_LOOP.value,
                policy=policy.name if policy else ToolJobKind.TOOL_LOOP.value,
                allowed=event.event not in {"tool_blocked", "tool_failed", "loop_failed"},
                status=event.event.upper(), reason=event.error_type or event.event,
                payload=serialized,
            ))
            db.commit()
