from __future__ import annotations

import asyncio
import inspect
import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from app.schemas.dtos import AiMessage
from app.services.ai import AiToolCall, AiToolDecision
from app.services.tool_governance import ToolPolicy, ToolPolicyRegistry


class ToolCallingClient(Protocol):
    async def complete_with_tools(
        self,
        messages: Sequence[AiMessage | Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AiToolDecision: ...


@dataclass(frozen=True)
class ToolLoopExecutionContext:
    report_id: int
    risk_level: str
    round_number: int
    policy: ToolPolicy
    call_id: str | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    success: bool
    tool: str
    report_id: int
    status: str
    message: str
    retryable: bool = False
    resource: dict[str, Any] = field(default_factory=dict)

    def tool_message_content(self) -> str:
        return json.dumps(
            {
                "success": self.success,
                "tool": self.tool,
                "reportId": self.report_id,
                "resource": self.resource,
                "status": self.status,
                "message": self.message,
                "retryable": self.retryable,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class ToolLoopEvent:
    event: str
    round_number: int
    risk_level: str
    required_tools: tuple[str, ...]
    completed_tools: tuple[str, ...]
    missing_tools: tuple[str, ...]
    call_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error_type: str | None = None
    retryable: bool | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    elapsed_ms: float | None = None

    def payload(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schemaVersion": 1,
            "createdAt": self.created_at,
            "event": self.event,
            "round": self.round_number,
            "riskLevel": self.risk_level,
            "requiredTools": list(self.required_tools),
            "completedTools": list(self.completed_tools),
            "missingTools": list(self.missing_tools),
        }
        optional = {
            "callId": self.call_id,
            "toolName": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "errorType": self.error_type,
            "retryable": self.retryable,
            "elapsedMs": self.elapsed_ms,
        }
        value.update({key: item for key, item in optional.items() if item is not None})
        return value


@dataclass(frozen=True)
class ToolLoopResult:
    status: str
    rounds: int
    required_tools: tuple[str, ...]
    completed_tools: tuple[str, ...]
    missing_tools: tuple[str, ...]
    final_content: str | None = None


ToolHandler = Callable[
    [ToolLoopExecutionContext],
    ToolExecutionResult | Awaitable[ToolExecutionResult],
]
CompletionReader = Callable[[int], set[str] | Awaitable[set[str]]]
EventSink = Callable[[ToolLoopEvent], None | Awaitable[None]]
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True)
class RegisteredTool:
    policy: ToolPolicy
    schema: dict[str, Any]
    handler: ToolHandler


class ToolLoopRegistry:
    def __init__(self, handlers: Mapping[str, ToolHandler]):
        schemas = {
            tool["function"]["name"]: tool
            for tool in ToolPolicyRegistry.function_tools()
        }
        policies = ToolPolicyRegistry.tool_loop_policies()
        expected_names = {
            policy.function_name
            for policy in policies
            if policy.function_name is not None
        }
        missing_handlers = expected_names - set(handlers)
        unknown_handlers = set(handlers) - expected_names
        if missing_handlers:
            raise ValueError(f"缺少 Tool Loop handler：{sorted(missing_handlers)}")
        if unknown_handlers:
            raise ValueError(f"存在未注册的 Tool Loop handler：{sorted(unknown_handlers)}")
        self._tools = {
            policy.function_name: RegisteredTool(
                policy=policy,
                schema=schemas[policy.function_name],
                handler=handlers[policy.function_name],
            )
            for policy in policies
            if policy.function_name is not None
        }

    def resolve(self, function_name: str) -> RegisteredTool | None:
        return self._tools.get(function_name)

    def policies_for_risk(self, risk_level: str) -> tuple[ToolPolicy, ...]:
        return tuple(
            registration.policy
            for registration in self._tools.values()
            if risk_level in registration.policy.allowed_risks
        )

    def schemas_for_risk(self, risk_level: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            registration.schema
            for registration in self._tools.values()
            if risk_level in registration.policy.allowed_risks
        )

    def required_tool_kinds(self, risk_level: str) -> tuple[str, ...]:
        return tuple(policy.name for policy in self.policies_for_risk(risk_level))


class ToolLoopFailure(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class ToolLoopMaxRoundsError(ToolLoopFailure):
    pass


class ToolLoopTimeoutError(ToolLoopFailure):
    pass


class ToolLoopCancelledError(ToolLoopFailure):
    pass


class ToolLoopModelError(ToolLoopFailure):
    pass


class ToolLoopToolError(ToolLoopFailure):
    pass


class BoundedToolLoopRunner:
    def __init__(
        self,
        ai: ToolCallingClient,
        registry: ToolLoopRegistry,
        completion_reader: CompletionReader,
        *,
        max_rounds: int,
        timeout_seconds: float,
        event_sink: EventSink | None = None,
        is_cancelled: CancellationCheck | None = None,
    ):
        if max_rounds <= 0:
            raise ValueError("tool_loop_max_rounds 必须大于 0")
        if timeout_seconds <= 0:
            raise ValueError("tool_loop_timeout_seconds 必须大于 0")
        self.ai = ai
        self.registry = registry
        self.completion_reader = completion_reader
        self.max_rounds = max_rounds
        self.timeout_seconds = timeout_seconds
        self.event_sink = event_sink
        self.is_cancelled = is_cancelled or (lambda: False)
        self._last_state: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
        self._round = 0
        self._started_at = 0.0

    async def run(self, report_id: int, risk_level: str) -> ToolLoopResult:
        required = self.registry.required_tool_kinds(risk_level)
        if not required:
            raise ValueError(f"风险等级 {risk_level} 没有可执行的 Tool Loop 策略")
        self._last_state = ((), required)
        self._round = 0
        self._started_at = time.monotonic()
        task = asyncio.create_task(self._run(report_id, risk_level, required))
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while not task.done():
                if self.is_cancelled():
                    raise ToolLoopCancelledError("Tool Loop 已收到取消信号", retryable=True)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ToolLoopTimeoutError("Tool Loop 超过总超时", retryable=True)
                await asyncio.wait({task}, timeout=min(0.05, remaining))
            return await task
        except (ToolLoopFailure, asyncio.CancelledError) as exc:
            task.cancel()
            # 不能放任已开始的副作用在后台继续、同时启动重试。
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            # In-flight effects have drained; refresh the durable completion snapshot.
            try:
                await self._state(report_id, required)
            except Exception:
                pass
            failure = exc if isinstance(exc, ToolLoopFailure) else ToolLoopCancelledError(
                "Tool Loop 外部取消", retryable=True
            )
            completed, missing = self._last_state
            await self._emit(ToolLoopEvent(
                event="loop_failed", round_number=self._round, risk_level=risk_level,
                required_tools=required, completed_tools=completed, missing_tools=missing,
                error_type=type(failure).__name__, retryable=failure.retryable,
                result={"message": str(failure), "retryAfter": failure.retry_after},
            ))
            if failure is exc:
                raise
            raise failure from exc

    async def _run(
        self,
        report_id: int,
        risk_level: str,
        required: tuple[str, ...],
    ) -> ToolLoopResult:
        completed, missing = await self._state(report_id, required)
        await self._emit(
            ToolLoopEvent(
                event="loop_started",
                round_number=0,
                risk_level=risk_level,
                required_tools=required,
                completed_tools=completed,
                missing_tools=missing,
            )
        )
        if not missing:
            return await self._complete(0, risk_level, required, completed, None)

        messages: list[AiMessage | Mapping[str, Any]] = [
            AiMessage(
                role="system",
                content=self._system_prompt(risk_level, self.registry.policies_for_risk(risk_level)),
            )
        ]
        tools = self.registry.schemas_for_risk(risk_level)
        seen_call_ids: set[str] = set()

        for round_number in range(1, self.max_rounds + 1):
            self._round = round_number
            await self._require_not_cancelled(
                round_number, report_id, risk_level, required
            )
            completed, missing = await self._state(report_id, required)
            messages.append(AiMessage(role="system", content=(
                "可信数据库状态：已完成 " + ", ".join(completed)
                + "；待处理 " + ", ".join(missing)
            )))
            await self._emit(ToolLoopEvent(
                event="model_request", round_number=round_number, risk_level=risk_level,
                required_tools=required, completed_tools=completed, missing_tools=missing,
                result={"messageCount": len(messages), "toolMessageCount": sum(
                    isinstance(m, dict) and m.get("role") == "tool" for m in messages
                ), "toolNames": [t["function"]["name"] for t in tools]},
            ))
            try:
                decision = await self.ai.complete_with_tools(messages, tools)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                diagnostic = type(exc).__name__
                if response is not None:
                    diagnostic += f" HTTP {status_code}"
                    try:
                        error_code = response.json().get("error", {}).get("code", "")
                        if isinstance(error_code, str) and error_code.replace("_", "").isalnum():
                            diagnostic += f" ({error_code[:80]})"
                    except (ValueError, AttributeError):
                        pass
                failure = ToolLoopModelError(
                    f"模型 Function Calling 失败：{diagnostic}",
                    retryable=not isinstance(exc, ValueError) and (
                        getattr(getattr(exc, "response", None), "status_code", 500) >= 500
                        or getattr(getattr(exc, "response", None), "status_code", 500) in {408, 429}
                    ),
                )
                raise failure from exc

            if len(decision.tool_calls) > 1:
                raise ToolLoopModelError("供应商违反串行调用约定", retryable=False)
            for call in decision.tool_calls:
                if call.id in seen_call_ids:
                    raise ToolLoopModelError("供应商复用了 Tool Call ID", retryable=False)
                seen_call_ids.add(call.id)
            if decision.tool_calls and decision.finish_reason in {"length", "content_filter"}:
                raise ToolLoopModelError("模型响应被截断或过滤，不能执行不完整调用", retryable=False)
            messages.append(decision.assistant_message())
            completed, missing = await self._state(report_id, required)
            await self._emit(
                ToolLoopEvent(
                    event="model_response",
                    round_number=round_number,
                    risk_level=risk_level,
                    required_tools=required,
                    completed_tools=completed,
                    missing_tools=missing,
                    result={
                        "finishReason": decision.finish_reason,
                        "contentPresent": bool(decision.content),
                        "toolCalls": [
                            {
                                "callId": call.id,
                                "toolName": call.name,
                                "argumentsPresent": bool(call.arguments),
                            }
                            for call in decision.tool_calls
                        ],
                    },
                )
            )

            if not decision.tool_calls:
                if not missing and decision.finish_reason in {"stop", None}:
                    return await self._complete(
                        round_number,
                        risk_level,
                        required,
                        completed,
                        decision.content,
                    )
                messages.append(
                    AiMessage(
                        role="system",
                        content=(
                            "处置尚未完成。请继续调用工具，缺失的内部任务类型为："
                            + ", ".join(missing)
                            + "。不要声明完成。"
                        ),
                    )
                )
                await self._emit(
                    ToolLoopEvent(
                        event="model_correction",
                        round_number=round_number,
                        risk_level=risk_level,
                        required_tools=required,
                        completed_tools=completed,
                        missing_tools=missing,
                    )
                )
                continue

            for call in decision.tool_calls:
                await self._require_not_cancelled(
                    round_number, report_id, risk_level, required
                )
                tool_message = await self._handle_call(
                    call,
                    round_number,
                    report_id,
                    risk_level,
                    required,
                )
                messages.append(call.result_message(tool_message))
                completed, missing = await self._state(report_id, required)
                await self._emit(ToolLoopEvent(
                    event="tool_message", round_number=round_number, risk_level=risk_level,
                    required_tools=required, completed_tools=completed, missing_tools=missing,
                    call_id=call.id, tool_name=call.name, result=json.loads(tool_message),
                ))

        failure = ToolLoopMaxRoundsError(
            f"Tool Loop 达到最大轮数 {self.max_rounds} 仍未确认完成",
            retryable=True,
        )
        raise failure

    async def _handle_call(
        self,
        call: AiToolCall,
        round_number: int,
        report_id: int,
        risk_level: str,
        required: tuple[str, ...],
    ) -> str:
        completed, missing = await self._state(report_id, required)
        registration = self.registry.resolve(call.name)
        await self._emit(ToolLoopEvent(
            event="tool_proposed", round_number=round_number, risk_level=risk_level,
            required_tools=required, completed_tools=completed, missing_tools=missing,
            call_id=call.id, tool_name=call.name,
        ))
        if registration is None:
            return await self._blocked_message(
                call,
                report_id,
                "UNKNOWN_TOOL",
                f"未知工具：{call.name}",
                round_number,
                risk_level,
                required,
                completed,
                missing,
            )
        try:
            arguments = call.arguments_object()
        except ValueError as exc:
            return await self._blocked_message(
                call,
                report_id,
                "INVALID_ARGUMENTS",
                str(exc),
                round_number,
                risk_level,
                required,
                completed,
                missing,
            )
        await self._emit(ToolLoopEvent(
            event="tool_arguments_parsed", round_number=round_number, risk_level=risk_level,
            required_tools=required, completed_tools=completed, missing_tools=missing,
            call_id=call.id, tool_name=call.name, arguments={"argumentKeys": sorted(arguments)},
        ))
        if arguments:
            return await self._blocked_message(
                call,
                report_id,
                "INVALID_ARGUMENTS",
                "该工具不接受任何业务参数",
                round_number,
                risk_level,
                required,
                completed,
                missing,
                {"argumentKeys": sorted(arguments)},
            )

        policy = registration.policy
        if policy.name not in required:
            return await self._blocked_message(
                call,
                report_id,
                "UNAUTHORIZED_RISK",
                f"工具 {call.name} 不允许处理风险等级 {risk_level}",
                round_number,
                risk_level,
                required,
                completed,
                missing,
                arguments,
            )
        if policy.name in completed:
            result = ToolExecutionResult(
                success=True,
                tool=policy.name,
                report_id=report_id,
                status="SKIPPED_DUPLICATE",
                message="该工具已成功完成，本次不重复执行",
            )
            await self._emit(
                ToolLoopEvent(
                    event="tool_skipped",
                    round_number=round_number,
                    risk_level=risk_level,
                    required_tools=required,
                    completed_tools=completed,
                    missing_tools=missing,
                    call_id=call.id,
                    tool_name=call.name,
                    arguments=arguments,
                    result=json.loads(result.tool_message_content()),
                    retryable=False,
                )
            )
            return result.tool_message_content()

        unmet_dependencies = tuple(
            dependency
            for dependency in policy.dependencies
            if dependency not in completed
        )
        if unmet_dependencies:
            return await self._blocked_message(
                call,
                report_id,
                "DEPENDENCY_NOT_READY",
                "前置工具尚未完成：" + ", ".join(unmet_dependencies),
                round_number,
                risk_level,
                required,
                completed,
                missing,
                arguments,
            )

        context = ToolLoopExecutionContext(
            report_id=report_id,
            risk_level=risk_level,
            round_number=round_number,
            policy=policy,
            call_id=call.id,
        )
        await self._emit(ToolLoopEvent(
            event="tool_authorized", round_number=round_number, risk_level=risk_level,
            required_tools=required, completed_tools=completed, missing_tools=missing,
            call_id=call.id, tool_name=call.name, arguments=arguments,
        ))
        try:
            if inspect.iscoroutinefunction(registration.handler):
                result = registration.handler(context)
            else:
                pending = asyncio.create_task(asyncio.to_thread(registration.handler, context))
                try:
                    result = await asyncio.shield(pending)
                except asyncio.CancelledError:
                    await pending
                    raise
            if inspect.isawaitable(result):
                result = await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = ToolLoopToolError(
                f"工具 {call.name} 执行异常：{type(exc).__name__}",
                retryable=getattr(exc, "retryable", True),
                retry_after=getattr(exc, "retry_after", None),
            )
            await self._emit_tool_failure(
                failure,
                call,
                arguments,
                round_number,
                report_id,
                risk_level,
                required,
            )
            raise failure from exc

        if not isinstance(result, ToolExecutionResult):
            failure = ToolLoopToolError(
                f"工具 {call.name} 返回了非法结果类型",
                retryable=False,
            )
            await self._emit_tool_failure(
                failure,
                call,
                arguments,
                round_number,
                report_id,
                risk_level,
                required,
            )
            raise failure
        if result.tool != policy.name or result.report_id != report_id:
            failure = ToolLoopToolError(
                f"工具 {call.name} 返回的工具或报告身份不匹配",
                retryable=False,
            )
            await self._emit_tool_failure(
                failure,
                call,
                arguments,
                round_number,
                report_id,
                risk_level,
                required,
            )
            raise failure
        if not result.success:
            failure = ToolLoopToolError(result.message, retryable=result.retryable)
            await self._emit_tool_failure(
                failure,
                call,
                arguments,
                round_number,
                report_id,
                risk_level,
                required,
                result,
            )
            raise failure

        completed_after, missing_after = await self._state(report_id, required)
        if policy.name not in completed_after:
            failure = ToolLoopToolError(
                f"工具 {call.name} 报告成功，但数据库完成状态未出现",
                retryable=True,
            )
            await self._emit_tool_failure(
                failure,
                call,
                arguments,
                round_number,
                report_id,
                risk_level,
                required,
                result,
            )
            raise failure
        await self._emit(
            ToolLoopEvent(
                event="tool_succeeded",
                round_number=round_number,
                risk_level=risk_level,
                required_tools=required,
                completed_tools=completed_after,
                missing_tools=missing_after,
                call_id=call.id,
                tool_name=call.name,
                arguments=arguments,
                result=json.loads(result.tool_message_content()),
                retryable=result.retryable,
            )
        )
        return result.tool_message_content()

    async def _blocked_message(
        self,
        call: AiToolCall,
        report_id: int,
        status: str,
        message: str,
        round_number: int,
        risk_level: str,
        required: tuple[str, ...],
        completed: tuple[str, ...],
        missing: tuple[str, ...],
        arguments: dict[str, Any] | None = None,
    ) -> str:
        result = ToolExecutionResult(
            success=False,
            tool=call.name,
            report_id=report_id,
            status=status,
            message=message,
            retryable=False,
        )
        await self._emit(
            ToolLoopEvent(
                event="tool_blocked",
                round_number=round_number,
                risk_level=risk_level,
                required_tools=required,
                completed_tools=completed,
                missing_tools=missing,
                call_id=call.id,
                tool_name=call.name,
                arguments=arguments,
                result=json.loads(result.tool_message_content()),
                retryable=False,
            )
        )
        return result.tool_message_content()

    async def _complete(
        self,
        round_number: int,
        risk_level: str,
        required: tuple[str, ...],
        completed: tuple[str, ...],
        final_content: str | None,
    ) -> ToolLoopResult:
        result = ToolLoopResult(
            status="COMPLETED",
            rounds=round_number,
            required_tools=required,
            completed_tools=completed,
            missing_tools=(),
            final_content=final_content,
        )
        await self._emit(
            ToolLoopEvent(
                event="loop_completed",
                round_number=round_number,
                risk_level=risk_level,
                required_tools=required,
                completed_tools=completed,
                missing_tools=(),
            )
        )
        return result

    async def _require_not_cancelled(
        self,
        round_number: int,
        report_id: int,
        risk_level: str,
        required: tuple[str, ...],
    ) -> None:
        if not self.is_cancelled():
            return
        failure = ToolLoopCancelledError(
            "Tool Loop 已收到取消信号",
            retryable=True,
        )
        raise failure

    async def _emit_tool_failure(
        self,
        failure: ToolLoopToolError,
        call: AiToolCall,
        arguments: dict[str, Any],
        round_number: int,
        report_id: int,
        risk_level: str,
        required: tuple[str, ...],
        result: ToolExecutionResult | None = None,
    ) -> None:
        completed, missing = await self._state(report_id, required)
        await self._emit(
            ToolLoopEvent(
                event="tool_failed",
                round_number=round_number,
                risk_level=risk_level,
                required_tools=required,
                completed_tools=completed,
                missing_tools=missing,
                call_id=call.id,
                tool_name=call.name,
                arguments=arguments,
                result=(
                    json.loads(result.tool_message_content())
                    if result is not None
                    else None
                ),
                error_type=type(failure).__name__,
                retryable=failure.retryable,
            )
        )

    async def _state(
        self,
        report_id: int,
        required: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        completed_value = self.completion_reader(report_id)
        if inspect.isawaitable(completed_value):
            completed_value = await completed_value
        completed_set = set(completed_value)
        completed = tuple(tool for tool in required if tool in completed_set)
        missing = tuple(tool for tool in required if tool not in completed_set)
        self._last_state = (completed, missing)
        return completed, missing

    async def _emit(self, event: ToolLoopEvent) -> None:
        if self.event_sink is None:
            return
        from dataclasses import replace

        outcome = self.event_sink(replace(event, elapsed_ms=round((time.monotonic() - self._started_at) * 1000, 3)))
        if inspect.isawaitable(outcome):
            await outcome

    @staticmethod
    def _system_prompt(risk_level: str, policies: tuple[ToolPolicy, ...]) -> str:
        return (
            "你负责选择心理风险报告的下一项后台处置工具。"
            "只使用提供的函数，函数参数必须是空对象，每轮按依赖顺序串行选择。"
            "工具结果会由系统回填；全部必需结果成功后停止调用工具。"
            f"\n可信风险等级：{risk_level}"
            "\n必需工具与依赖：" + json.dumps([
                {"kind": policy.name, "function": policy.function_name, "dependencies": policy.dependencies}
                for policy in policies
            ], ensure_ascii=False)
        )
