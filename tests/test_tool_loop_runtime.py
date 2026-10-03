import asyncio
import json
import tempfile
import io
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.database import Base
from app.models.entities import (
    AlertRecord, ChatSession, DeadLetterRecord, ExcelRecord, PsychologicalReport,
    RiskCase, ToolAuditRecord, ToolJob, UserAccount,
)
from app.services.ai import AiClient, AiToolCall, AiToolDecision
from app.services.tool_loop import ToolLoopFailure
from app.services.tool_loop_runtime import ToolLoopRuntime
from app.services.tool_queue import ToolQueueService, ToolQueueWorker
from app.services.tools import ToolOrchestrationService


class ToolLoopRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mindbridge-loop-test-")
        self.root = Path(self.temp.name)
        self.engine = create_engine(f"sqlite:///{(self.root / 'test.db').as_posix()}",
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.settings = Settings(_env_file=None, ai_provider="mock", tool_queue_enabled=True,
                                 excel_path=str(self.root / "ledger.xlsx"),
                                 alert_email_delivery_mode="log", tool_queue_retry_delay_seconds=0.01,
                                 openai_api_key="test-key-secret", smtp_password="test-password-secret")
        self.workers = []

    def tearDown(self):
        for worker in self.workers:
            worker.stop()
        self.engine.dispose()
        self.temp.cleanup()

    def worker(self, ai=None):
        worker = ToolQueueWorker(self.settings, session_factory=self.factory, ai=ai)
        self.workers.append(worker)
        return worker

    def report(self, risk="HIGH"):
        with self.factory() as db:
            user = UserAccount(username=f"student-{time.time_ns()}", display_name="验收学生", password_hash="unused")
            db.add(user)
            db.flush()
            session = ChatSession(public_id=f"test-{time.time_ns()}", title="隔离测试", user_id=user.id)
            db.add(session)
            db.flush()
            report = PsychologicalReport(user_id=user.id, session_id=session.id, content="合成验收输入",
                                         intent="RISK", emotion="HIGH_RISK", emotion_score=4,
                                         risk_level=risk, confidence=0.95, summary="synthetic fixture")
            db.add(report)
            db.commit()
            return report.id

    def enqueue(self, report_id):
        with self.factory() as db:
            # Deliberately untrusted risk; persisted report must win.
            return ToolQueueService(db, self.settings).enqueue_report(report_id, "LOW")[0].id

    def run_job(self, worker, job_id):
        with self.factory() as db:
            job = db.get(ToolJob, job_id)
            job.status = "RUNNING"
            db.commit()
        worker._run_job(job_id)

    def test_risk_matrix_real_handlers_trace_and_idempotent_restore(self):
        worker = self.worker()
        for risk, expected in (("LOW", (1, 0, 0)), ("MEDIUM", (1, 1, 0)), ("HIGH", (1, 1, 1))):
            with self.subTest(risk=risk):
                report_id = self.report(risk)
                job_id = self.enqueue(report_id)
                self.assertEqual(self.enqueue(report_id), job_id)
                self.run_job(worker, job_id)
                with self.factory() as db:
                    job = db.get(ToolJob, job_id)
                    self.assertEqual(job.status, "SUCCESS", job.last_error)
                    counts = tuple(db.query(model).filter_by(report_id=report_id).count()
                                   for model in (ExcelRecord, RiskCase, AlertRecord))
                    self.assertEqual(counts, expected)
                    audits = db.query(ToolAuditRecord).filter_by(job_id=job_id).order_by(ToolAuditRecord.id).all()
                    events = [json.loads(a.payload).get("event") for a in audits]
                    self.assertIn("loop_completed", events)
                    self.assertIn("model_request", events)
                    self.assertEqual(events.count("tool_message"), sum(expected))
                    payload = " ".join(a.payload for a in audits)
                    self.assertNotIn(self.settings.openai_api_key, payload)
                    self.assertNotIn(self.settings.smtp_password, payload)
                    job.status = "RUNNING"
                    db.commit()
                # Recovery verifies DB products and does not even call model again.
                worker.ai = SimpleNamespace(complete_with_tools=Mock(side_effect=AssertionError("unexpected model call")))
                worker._run_job(job_id)
                with self.factory() as db:
                    self.assertEqual(db.get(ToolJob, job_id).status, "SUCCESS")
                    self.assertEqual(tuple(db.query(m).filter_by(report_id=report_id).count()
                                           for m in (ExcelRecord, RiskCase, AlertRecord)), expected)
                worker.ai = None

    def test_partial_failure_retries_without_duplicate_excel_or_case(self):
        worker = self.worker()
        report_id = self.report()
        job_id = self.enqueue(report_id)
        with patch.object(ToolOrchestrationService, "send_case_alert", side_effect=RuntimeError("smtp secret")):
            self.run_job(worker, job_id)
        with self.factory() as db:
            self.assertEqual(db.get(ToolJob, job_id).status, "PENDING")
            self.assertEqual(db.query(ExcelRecord).filter_by(report_id=report_id).count(), 1)
            self.assertEqual(db.query(RiskCase).filter_by(report_id=report_id).count(), 1)
        self.run_job(worker, job_id)
        with self.factory() as db:
            self.assertEqual(db.get(ToolJob, job_id).status, "SUCCESS")
            self.assertEqual(db.get(ToolJob, job_id).attempts, 2)
            self.assertEqual(db.query(ExcelRecord).filter_by(report_id=report_id).count(), 1)
            self.assertEqual(db.query(RiskCase).filter_by(report_id=report_id).count(), 1)
            self.assertEqual(db.query(AlertRecord).filter_by(report_id=report_id).count(), 1)
            self.assertTrue(db.query(ToolAuditRecord).filter_by(job_id=job_id, status="JOB_RETRY").count())

    def test_permanent_failure_goes_directly_to_dead_letter(self):
        class BadModel:
            async def complete_with_tools(self, messages, tools):
                raise ValueError("invalid protocol with test-key-secret")
        worker = self.worker(BadModel())
        job_id = self.enqueue(self.report())
        self.run_job(worker, job_id)
        with self.factory() as db:
            job = db.get(ToolJob, job_id)
            self.assertEqual((job.status, job.attempts), ("DEAD", 1))
            self.assertEqual(db.query(DeadLetterRecord).filter_by(job_id=job_id).count(), 1)
            self.assertNotIn("test-key-secret", job.last_error)

    def test_max_attempts_and_restart_recovery(self):
        class FailingModel:
            async def complete_with_tools(self, messages, tools):
                raise OSError("network down")
        worker = self.worker(FailingModel())
        job_id = self.enqueue(self.report())
        for _ in range(self.settings.tool_queue_max_attempts):
            self.run_job(worker, job_id)
        with self.factory() as db:
            self.assertEqual(db.get(ToolJob, job_id).status, "DEAD")
            self.assertEqual(db.query(DeadLetterRecord).filter_by(job_id=job_id).count(), 1)
        recovered = self.enqueue(self.report("LOW"))
        with self.factory() as db:
            job = db.get(ToolJob, recovered)
            job.status, job.attempts = "RUNNING", 1
            db.commit()
        worker._recover_running_jobs()
        worker.ai = None
        self.run_job(worker, recovered)
        with self.factory() as db:
            self.assertEqual((db.get(ToolJob, recovered).status, db.get(ToolJob, recovered).attempts), ("SUCCESS", 2))

    def test_rollback_switch_still_runs_legacy_jobs(self):
        self.settings.tool_loop_enabled = False
        worker = self.worker()
        with self.factory() as db:
            jobs = ToolQueueService(db, self.settings).enqueue_report(self.report(), "LOW")
            ids = [job.id for job in jobs]
            self.assertEqual([job.kind for job in jobs], ["EXCEL_REPORT", "CASE_CREATE", "ALERT_SEND"])
        for job_id in ids:
            self.run_job(worker, job_id)
        with self.factory() as db:
            self.assertTrue(all(db.get(ToolJob, job_id).status == "SUCCESS" for job_id in ids))

    def test_queue_concurrency_does_not_build_unbounded_executor_backlog(self):
        release = threading.Event()
        class WaitingModel:
            async def complete_with_tools(self, messages, tools):
                while not release.is_set():
                    await asyncio.sleep(0.01)
                return await AiClient(Settings(_env_file=None, ai_provider="mock")).complete_with_tools(messages, tools)
        worker = self.worker(WaitingModel())
        ids = [self.enqueue(self.report("LOW")) for _ in range(5)]
        try:
            worker._dispatch_once()
            worker._dispatch_once()
            with self.factory() as db:
                self.assertEqual(db.query(ToolJob).filter_by(status="RUNNING").count(), self.settings.tool_loop_workers)
                self.assertEqual(db.query(ToolJob).filter_by(status="PENDING").count(), 3)
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self.factory() as db:
                if db.query(ToolJob).filter_by(status="RUNNING").count() == 0:
                    break
            time.sleep(0.01)

    def test_cancel_during_model_and_timeout_have_durable_failure_traces(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                signal = threading.Event()
                class SlowModel:
                    async def complete_with_tools(self, messages, tools):
                        if cancel:
                            signal.set()
                        await asyncio.Event().wait()
                self.settings.tool_loop_timeout_seconds = 0.02
                job_id = self.enqueue(self.report())
                runtime = ToolLoopRuntime(self.settings, self.factory, ai=SlowModel(), is_cancelled=signal.is_set)
                with self.assertRaises(ToolLoopFailure):
                    asyncio.run(runtime.run(job_id))
                with self.factory() as db:
                    self.assertTrue(db.query(ToolAuditRecord).filter_by(job_id=job_id, status="LOOP_FAILED").count())
                    self.assertEqual(db.query(ExcelRecord).count(), 0)

    def test_sse_done_only_enqueues_parent_and_never_waits_for_model(self):
        from app.services.chat import ChatService
        report_id = self.report()
        class Harness:
            def run(inner, user, request):
                return SimpleNamespace(session=SimpleNamespace(public_id="test"), response_messages=[],
                                       tool_plan=None, report_id=report_id)
            def save_assistant_message(inner, *args):
                pass
            async def dispatch_tools(inner, plan):
                self.enqueue(report_id)
        class ChatAi:
            async def stream(inner, messages):
                yield "安全支持回复"
        service = object.__new__(ChatService)
        service.agent_harness, service.ai = Harness(), ChatAi()
        async def collect():
            return [chunk async for chunk in service.stream_chat(None, None)]
        events = asyncio.run(collect())
        self.assertIn("event: done", events[-1])
        with self.factory() as db:
            self.assertEqual(db.query(ToolJob).one().status, "PENDING")
            self.assertEqual(db.query(ToolJob).one().kind, "TOOL_LOOP")
            self.assertEqual(db.query(ToolAuditRecord).count(), 0)
            self.assertEqual(db.query(ExcelRecord).count(), 0)

    def test_pre_effect_governance_rejects_risk_change(self):
        report_id = self.report("HIGH")
        job_id = self.enqueue(report_id)
        runtime = ToolLoopRuntime(self.settings, self.factory)
        from app.services.tool_loop import ToolLoopExecutionContext
        from app.services.tool_governance import ToolPolicyRegistry
        context = ToolLoopExecutionContext(report_id, "HIGH", 1, ToolPolicyRegistry.policy_for("ALERT_SEND"))
        with self.factory() as db:
            report = db.get(PsychologicalReport, report_id)
            report.risk_level = "LOW"
            db.commit()
        with patch.object(ToolOrchestrationService, "send_case_alert") as effect:
            with self.assertRaises(ToolLoopFailure):
                runtime._execute_sync(job_id, "HIGH", context)
            effect.assert_not_called()
        with self.factory() as db:
            self.assertFalse(db.query(ToolAuditRecord).filter_by(job_id=job_id).one().allowed)

    def test_audit_query_filters_parent_report_and_bounds_result(self):
        from app.services.report import ReportService
        worker = self.worker()
        first_report, second_report = self.report("LOW"), self.report("HIGH")
        first_job, second_job = self.enqueue(first_report), self.enqueue(second_report)
        self.run_job(worker, first_job)
        self.run_job(worker, second_job)
        with self.factory() as db:
            service = ReportService(db)
            rows = service.tool_audits(job_id=first_job, report_id=first_report, limit=1000)
            self.assertTrue(rows)
            self.assertTrue(all(row.jobId == first_job and row.reportId == first_report for row in rows))
            self.assertEqual(len(service.tool_audits(job_id=second_job, limit=1)), 1)
            with self.assertRaises(ValueError):
                service.tool_audits(limit=1001)

    def test_alert_throttle_respects_retry_after_and_preserves_partial_results(self):
        self.settings.alert_email_rate_limit_per_minute = 1
        worker = self.worker()
        first, second = self.enqueue(self.report()), self.enqueue(self.report())
        self.run_job(worker, first)
        self.run_job(worker, second)
        from app.services.tool_queue import utc_now
        with self.factory() as db:
            job = db.get(ToolJob, second)
            self.assertEqual(job.status, "PENDING")
            self.assertGreater((job.run_after - utc_now()).total_seconds(), 50)
            self.assertEqual(db.query(RiskCase).filter_by(report_id=job.report_id).count(), 1)
            self.assertEqual(db.query(AlertRecord).filter_by(report_id=job.report_id).count(), 0)

    def test_live_runner_requires_explicit_opt_in_before_any_network_or_files(self):
        from app.harness.tool_loop_live import main
        with patch("app.harness.tool_loop_live.Settings") as config, patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as exc:
                main([])
            self.assertEqual(exc.exception.code, 2)
            config.assert_not_called()

    def test_dead_legacy_dependency_cannot_wait_forever(self):
        self.settings.tool_loop_enabled = False
        worker = self.worker()
        with self.factory() as db:
            jobs = ToolQueueService(db, self.settings).enqueue_report(self.report(), "HIGH")
            case, alert = jobs[1:]
            case.status = "DEAD"
            db.commit()
            alert_id = alert.id
        self.run_job(worker, alert_id)
        with self.factory() as db:
            self.assertEqual(db.get(ToolJob, alert_id).status, "DEAD")
            self.assertEqual(db.query(AlertRecord).count(), 0)

    def test_missing_report_and_invalid_persisted_risk_cannot_enqueue(self):
        with self.factory() as db:
            with self.assertRaises(ValueError):
                ToolQueueService(db, self.settings).enqueue_report(999999, "HIGH")
            with self.assertRaises(ValueError):
                ToolQueueService(db, self.settings).enqueue_report(self.report("INVALID"), "HIGH")
            self.assertEqual(db.query(ToolJob).count(), 0)

    def test_loop_configuration_rejects_unbounded_values(self):
        from pydantic import ValidationError
        for fields in ({"tool_loop_max_rounds": 0}, {"tool_loop_workers": 0},
                       {"tool_loop_timeout_seconds": 0}, {"tool_loop_max_rounds": 101}):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                Settings(_env_file=None, **fields)

    def test_trace_does_not_store_injected_argument_values(self):
        class InjectionModel:
            def __init__(inner):
                inner.round = 0
            async def complete_with_tools(inner, messages, tools):
                inner.round += 1
                if inner.round == 1:
                    return AiToolDecision(None, (AiToolCall("injected", "mindbridge_excel_report",
                        '{"apiKey":"test-key-secret","password":"test-password-secret","content":"private-synthetic-value"}'),), "tool_calls")
                return await AiClient(self.settings).complete_with_tools(messages, tools)
        worker = self.worker(InjectionModel())
        job_id = self.enqueue(self.report("LOW"))
        self.run_job(worker, job_id)
        with self.factory() as db:
            self.assertEqual(db.get(ToolJob, job_id).status, "SUCCESS")
            payload = " ".join(a.payload for a in db.query(ToolAuditRecord).filter_by(job_id=job_id))
            for value in ("test-key-secret", "test-password-secret", "private-synthetic-value"):
                self.assertNotIn(value, payload)
            self.assertEqual(db.query(ExcelRecord).count(), 1)

    def test_runtime_timeout_drains_real_file_write_before_parent_retry(self):
        self.settings.tool_loop_timeout_seconds = 0.2
        worker = self.worker()
        job_id = self.enqueue(self.report("LOW"))
        original = ToolOrchestrationService.write_excel
        def slow_write(tools, report):
            time.sleep(0.35)
            return original(tools, report)
        with patch.object(ToolOrchestrationService, "write_excel", slow_write):
            self.run_job(worker, job_id)
        with self.factory() as db:
            self.assertEqual(db.get(ToolJob, job_id).status, "PENDING")
            self.assertEqual(db.query(ExcelRecord).count(), 1)
        self.settings.tool_loop_timeout_seconds = 60
        self.run_job(worker, job_id)
        with self.factory() as db:
            self.assertEqual(db.get(ToolJob, job_id).status, "SUCCESS")
            self.assertEqual(db.query(ExcelRecord).count(), 1)


if __name__ == "__main__":
    unittest.main()
