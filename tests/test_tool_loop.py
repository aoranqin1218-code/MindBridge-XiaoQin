import asyncio
import json
import unittest

from app.core.enums import RiskLevel, ToolJobKind
from app.services.ai import AiToolCall, AiToolDecision
from app.services.tool_governance import ToolPolicyRegistry
from app.services.tool_loop import (
    BoundedToolLoopRunner,
    ToolExecutionResult,
    ToolLoopCancelledError,
    ToolLoopMaxRoundsError,
    ToolLoopRegistry,
    ToolLoopTimeoutError,
    ToolLoopToolError,
    ToolLoopModelError,
)


def tool_decision(call_id: str, name: str, arguments: str = "{}") -> AiToolDecision:
    return AiToolDecision(
        content=None,
        tool_calls=(AiToolCall(id=call_id, name=name, arguments=arguments),),
        finish_reason="tool_calls",
    )


def stop_decision(content: str = "done") -> AiToolDecision:
    return AiToolDecision(content=content, tool_calls=(), finish_reason="stop")


class ScriptedAi:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.requests = []

    async def complete_with_tools(self, messages, tools):
        self.requests.append({"messages": list(messages), "tools": list(tools)})
        if not self.decisions:
            raise AssertionError("scripted AI has no decision left")
        decision = self.decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision


class NeverReturningAi:
    async def complete_with_tools(self, messages, tools):
        await asyncio.Event().wait()


class ToolLoopTestHarness:
    def __init__(self):
        self.completed = set()
        self.executions = []
        self.events = []
        self.handlers = {
            policy.function_name: self._handler_for(policy.name)
            for policy in ToolPolicyRegistry.tool_loop_policies()
        }

    def _handler_for(self, tool_kind):
        async def handler(context):
            self.executions.append(tool_kind)
            self.completed.add(tool_kind)
            return ToolExecutionResult(
                success=True,
                tool=tool_kind,
                report_id=context.report_id,
                status="SUCCESS",
                message=f"{tool_kind} completed",
                resource={"kind": tool_kind},
            )

        return handler

    def registry(self):
        return ToolLoopRegistry(self.handlers)

    def completion_reader(self, report_id):
        return set(self.completed)


class BoundedToolLoopTests(unittest.IsolatedAsyncioTestCase):
    def make_runner(
        self,
        ai,
        harness,
        *,
        max_rounds=8,
        timeout_seconds=1.0,
        is_cancelled=None,
    ):
        return BoundedToolLoopRunner(
            ai=ai,
            registry=harness.registry(),
            completion_reader=harness.completion_reader,
            max_rounds=max_rounds,
            timeout_seconds=timeout_seconds,
            event_sink=harness.events.append,
            is_cancelled=is_cancelled,
        )

    async def test_high_risk_happy_path_executes_three_tools_and_follows_up(self):
        harness = ToolLoopTestHarness()
        ai = ScriptedAi(
            [
                tool_decision("call_excel", "mindbridge_excel_report"),
                tool_decision("call_case", "mindbridge_case_create"),
                tool_decision("call_alert", "mindbridge_alert_send"),
                stop_decision("all complete"),
            ]
        )
        runner = self.make_runner(ai, harness)

        result = await runner.run(report_id=42, risk_level=RiskLevel.HIGH.value)

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.rounds, 4)
        self.assertEqual(
            harness.executions,
            [
                ToolJobKind.EXCEL_REPORT.value,
                ToolJobKind.CASE_CREATE.value,
                ToolJobKind.ALERT_SEND.value,
            ],
        )
        self.assertEqual(len(ai.requests), 4)
        second_request_messages = ai.requests[1]["messages"]
        excel_message = next(
            message
            for message in second_request_messages
            if isinstance(message, dict) and message.get("role") == "tool"
        )
        self.assertEqual(excel_message["tool_call_id"], "call_excel")
        self.assertEqual(json.loads(excel_message["content"])["status"], "SUCCESS")
        self.assertEqual(harness.events[-1].event, "loop_completed")

    async def test_unknown_extra_arguments_and_wrong_risk_are_blocked(self):
        harness = ToolLoopTestHarness()
        ai = ScriptedAi(
            [
                tool_decision("call_unknown", "delete_everything"),
                tool_decision("call_alert", "mindbridge_alert_send"),
                tool_decision(
                    "call_args",
                    "mindbridge_excel_report",
                    '{"report_id":999}',
                ),
                tool_decision("call_excel", "mindbridge_excel_report"),
                stop_decision(),
            ]
        )
        runner = self.make_runner(ai, harness)

        result = await runner.run(report_id=7, risk_level=RiskLevel.LOW.value)

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(harness.executions, [ToolJobKind.EXCEL_REPORT.value])
        blocked = [event for event in harness.events if event.event == "tool_blocked"]
        self.assertEqual(len(blocked), 3)
        self.assertTrue(all(event.result["reportId"] == 7 for event in blocked))
        self.assertEqual(
            [event.result["status"] for event in blocked],
            ["UNKNOWN_TOOL", "UNAUTHORIZED_RISK", "INVALID_ARGUMENTS"],
        )

    async def test_dependency_and_duplicate_calls_do_not_repeat_side_effects(self):
        harness = ToolLoopTestHarness()
        ai = ScriptedAi(
            [
                tool_decision("early_alert", "mindbridge_alert_send"),
                tool_decision("excel", "mindbridge_excel_report"),
                tool_decision("case", "mindbridge_case_create"),
                tool_decision("case_again", "mindbridge_case_create"),
                tool_decision("alert", "mindbridge_alert_send"),
                stop_decision(),
            ]
        )
        runner = self.make_runner(ai, harness)

        await runner.run(report_id=9, risk_level=RiskLevel.HIGH.value)

        self.assertEqual(harness.executions.count(ToolJobKind.CASE_CREATE.value), 1)
        self.assertEqual(harness.executions.count(ToolJobKind.ALERT_SEND.value), 1)
        self.assertTrue(
            any(
                event.event == "tool_blocked"
                and event.result["status"] == "DEPENDENCY_NOT_READY"
                for event in harness.events
            )
        )
        self.assertTrue(any(event.event == "tool_skipped" for event in harness.events))

    async def test_early_stop_adds_correction_and_can_recover(self):
        harness = ToolLoopTestHarness()
        ai = ScriptedAi(
            [
                stop_decision("premature"),
                tool_decision("excel", "mindbridge_excel_report"),
                stop_decision(),
            ]
        )
        runner = self.make_runner(ai, harness)

        result = await runner.run(report_id=12, risk_level=RiskLevel.LOW.value)

        self.assertEqual(result.status, "COMPLETED")
        correction_messages = [
            message
            for message in ai.requests[1]["messages"]
            if getattr(message, "role", None) == "system"
            and "尚未完成" in getattr(message, "content", "")
        ]
        self.assertEqual(len(correction_messages), 1)
        self.assertTrue(any(event.event == "model_correction" for event in harness.events))

    async def test_max_rounds_fails_when_required_results_remain_missing(self):
        harness = ToolLoopTestHarness()
        ai = ScriptedAi([stop_decision(), stop_decision()])
        runner = self.make_runner(ai, harness, max_rounds=2)

        with self.assertRaises(ToolLoopMaxRoundsError) as raised:
            await runner.run(report_id=15, risk_level=RiskLevel.LOW.value)

        self.assertTrue(raised.exception.retryable)
        self.assertEqual(harness.events[-1].event, "loop_failed")

    async def test_total_timeout_cancels_slow_model_call(self):
        harness = ToolLoopTestHarness()
        runner = self.make_runner(
            NeverReturningAi(),
            harness,
            timeout_seconds=0.01,
        )

        with self.assertRaises(ToolLoopTimeoutError):
            await runner.run(report_id=18, risk_level=RiskLevel.LOW.value)

        self.assertEqual(harness.events[-1].event, "loop_failed")
        self.assertEqual(harness.events[-1].error_type, "ToolLoopTimeoutError")

    async def test_cancellation_stops_before_model_or_tool_execution(self):
        harness = ToolLoopTestHarness()
        ai = ScriptedAi([tool_decision("excel", "mindbridge_excel_report")])
        runner = self.make_runner(ai, harness, is_cancelled=lambda: True)

        with self.assertRaises(ToolLoopCancelledError):
            await runner.run(report_id=21, risk_level=RiskLevel.LOW.value)

        self.assertEqual(ai.requests, [])
        self.assertEqual(harness.executions, [])

    async def test_handler_success_requires_durable_completion_state(self):
        harness = ToolLoopTestHarness()

        async def non_durable_handler(context):
            return ToolExecutionResult(
                success=True,
                tool=ToolJobKind.EXCEL_REPORT.value,
                report_id=context.report_id,
                status="SUCCESS",
                message="claimed success without durable state",
            )

        harness.handlers["mindbridge_excel_report"] = non_durable_handler
        ai = ScriptedAi([tool_decision("excel", "mindbridge_excel_report")])
        runner = self.make_runner(ai, harness)

        with self.assertRaisesRegex(ToolLoopToolError, "数据库完成状态未出现"):
            await runner.run(report_id=24, risk_level=RiskLevel.LOW.value)

    async def test_invalid_json_and_array_arguments_are_rejected_without_side_effects(self):
        harness = ToolLoopTestHarness()
        ai = ScriptedAi([tool_decision("bad", "mindbridge_excel_report", "not-json"),
                         tool_decision("array", "mindbridge_excel_report", "[]"),
                         tool_decision("valid", "mindbridge_excel_report"), stop_decision()])
        await self.make_runner(ai, harness).run(1, "LOW")
        self.assertEqual(len(harness.executions), 1)
        self.assertEqual(sum(e.event == "tool_blocked" for e in harness.events), 2)

    async def test_parallel_or_truncated_response_cannot_reach_handler(self):
        for calls, reason in ((2, "tool_calls"), (1, "length")):
            harness = ToolLoopTestHarness()
            decision = AiToolDecision(None, tuple(AiToolCall(str(i), "mindbridge_excel_report", "{}")
                                                  for i in range(calls)), reason)
            with self.assertRaises(ToolLoopModelError):
                await self.make_runner(ScriptedAi([decision]), harness).run(1, "LOW")
            self.assertEqual(harness.executions, [])

    async def test_reused_call_id_is_a_protocol_failure_not_a_second_effect(self):
        harness = ToolLoopTestHarness()
        ai = ScriptedAi([tool_decision("same", "mindbridge_excel_report"),
                         tool_decision("same", "mindbridge_case_create")])
        with self.assertRaises(ToolLoopModelError):
            await self.make_runner(ai, harness).run(1, "HIGH")
        self.assertEqual(harness.executions, ["EXCEL_REPORT"])

    async def test_timeout_drains_started_sync_effect_before_returning(self):
        import time
        harness = ToolLoopTestHarness()
        def slow_effect(context):
            time.sleep(0.05)
            harness.completed.add("EXCEL_REPORT")
            return ToolExecutionResult(True, "EXCEL_REPORT", context.report_id, "SUCCESS", "done")
        harness.handlers["mindbridge_excel_report"] = slow_effect
        runner = self.make_runner(ScriptedAi([tool_decision("slow", "mindbridge_excel_report")]),
                                  harness, timeout_seconds=0.02)
        with self.assertRaises(ToolLoopTimeoutError):
            await runner.run(1, "LOW")
        self.assertEqual(harness.completed, {"EXCEL_REPORT"})
        self.assertEqual(harness.events[-1].missing_tools, ())

    async def test_cancellation_while_awaiting_model_does_not_wait_for_timeout(self):
        harness = ToolLoopTestHarness()
        cancelled = False
        async def cancel_soon():
            nonlocal cancelled
            await asyncio.sleep(0.01)
            cancelled = True
        pending = asyncio.create_task(cancel_soon())
        with self.assertRaises(ToolLoopCancelledError):
            await self.make_runner(NeverReturningAi(), harness, timeout_seconds=10,
                                   is_cancelled=lambda: cancelled).run(1, "LOW")
        await pending
        self.assertEqual(sum(e.event == "loop_failed" for e in harness.events), 1)


if __name__ == "__main__":
    unittest.main()
