import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.core.enums import RiskLevel, ToolJobKind
from app.services.tool_governance import ToolGovernanceService, ToolPolicyRegistry


def report(risk: RiskLevel):
    return SimpleNamespace(risk_level=risk.value)


class ToolGovernanceTests(unittest.TestCase):
    def test_high_risk_alert_is_allowed_only_for_high_risk(self):
        allowed, _, _ = ToolPolicyRegistry.authorize(ToolJobKind.ALERT_SEND.value, report(RiskLevel.HIGH))
        blocked, reason, _ = ToolPolicyRegistry.authorize(ToolJobKind.ALERT_SEND.value, report(RiskLevel.LOW))

        self.assertTrue(allowed)
        self.assertFalse(blocked)
        self.assertIn("不允许", reason)

    def test_medium_case_create_is_allowed_but_low_is_blocked(self):
        allowed, _, _ = ToolPolicyRegistry.authorize(ToolJobKind.CASE_CREATE.value, report(RiskLevel.MEDIUM))
        blocked, _, _ = ToolPolicyRegistry.authorize(ToolJobKind.CASE_CREATE.value, report(RiskLevel.LOW))

        self.assertTrue(allowed)
        self.assertFalse(blocked)

    def test_unknown_tool_is_blocked(self):
        allowed, reason, policy = ToolPolicyRegistry.authorize("DELETE_EVERYTHING", report(RiskLevel.HIGH))

        self.assertFalse(allowed)
        self.assertIsNone(policy)
        self.assertIn("未知工具", reason)

    def test_tool_loop_exposes_only_three_zero_argument_functions(self):
        tools = ToolPolicyRegistry.function_tools()

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            [
                "mindbridge_excel_report",
                "mindbridge_case_create",
                "mindbridge_alert_send",
            ],
        )
        for tool in tools:
            self.assertEqual(tool["type"], "function")
            self.assertTrue(tool["function"]["strict"])
            self.assertEqual(
                tool["function"]["parameters"],
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            )

    def test_tool_loop_risk_matrix_and_alert_dependency(self):
        self.assertEqual(
            ToolPolicyRegistry.required_tool_kinds(RiskLevel.LOW.value),
            (ToolJobKind.EXCEL_REPORT.value,),
        )
        self.assertEqual(
            ToolPolicyRegistry.required_tool_kinds(RiskLevel.MEDIUM.value),
            (ToolJobKind.EXCEL_REPORT.value, ToolJobKind.CASE_CREATE.value),
        )
        self.assertEqual(
            ToolPolicyRegistry.required_tool_kinds(RiskLevel.HIGH.value),
            (
                ToolJobKind.EXCEL_REPORT.value,
                ToolJobKind.CASE_CREATE.value,
                ToolJobKind.ALERT_SEND.value,
            ),
        )
        alert_policy = ToolPolicyRegistry.policy_for_function("mindbridge_alert_send")
        self.assertIsNotNone(alert_policy)
        self.assertEqual(alert_policy.dependencies, (ToolJobKind.CASE_CREATE.value,))

    def test_legacy_and_parent_jobs_are_not_model_functions(self):
        self.assertIsNone(ToolPolicyRegistry.policy_for_function(ToolJobKind.RISK_ALERT.value))
        exposed_kinds = {policy.name for policy in ToolPolicyRegistry.tool_loop_policies()}
        self.assertNotIn(ToolJobKind.RISK_ALERT.value, exposed_kinds)
        self.assertNotIn(ToolJobKind.TOOL_LOOP.value, exposed_kinds)

    def test_audit_payload_keeps_only_decision_snapshot_and_preserves_it_on_finish(self):
        db = SimpleNamespace(add=Mock(), commit=Mock(), refresh=Mock())
        job = SimpleNamespace(
            id=17,
            report_id=23,
            kind=ToolJobKind.CASE_CREATE.value,
            attempts=2,
        )
        governance = ToolGovernanceService(db)

        record = governance.start_job(job, report(RiskLevel.HIGH))
        initial_payload = json.loads(record.payload)
        self.assertNotIn("jobId", initial_payload)
        self.assertNotIn("kind", initial_payload)
        self.assertEqual(initial_payload["schemaVersion"], 1)
        self.assertEqual(initial_payload["decision"]["attempt"], 2)
        self.assertEqual(initial_payload["decision"]["riskLevelAtDecision"], RiskLevel.HIGH.value)

        governance.finish(record, "SUCCESS", payload={"caseId": 31})
        finished_payload = json.loads(record.payload)
        self.assertEqual(finished_payload["decision"], initial_payload["decision"])
        self.assertEqual(finished_payload["result"], {"caseId": 31})


if __name__ == "__main__":
    unittest.main()
