"""Opt-in real-provider evidence, always isolated SQLite + Excel + log-only alert.

Run: python -m app.harness.tool_loop_live --allow-external [--risk HIGH]
This is a permanent, repeatable acceptance entry point, not a one-off script.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import dotenv_values

from app.core.config import Settings
from app.core.database import Base
from app.models.entities import (
    AlertRecord, ChatSession, ExcelRecord, PsychologicalReport, RiskCase,
    ToolAuditRecord, ToolJob, UserAccount,
)
from app.services.tool_queue import ToolQueueService, ToolQueueWorker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-external", action="store_true", help="Explicit authorization for model API calls")
    parser.add_argument("--risk", choices=["LOW", "MEDIUM", "HIGH"], default="HIGH")
    parser.add_argument("--output", type=Path, default=Path("target/tool-loop-live"))
    parser.add_argument("--env-file", type=Path, help="Use this project's model settings explicitly instead of inherited process overrides")
    args = parser.parse_args(argv)
    if not args.allow_external:
        parser.error("Real model calls require --allow-external; no model was called")
    model_fields = {"ai_provider", "ai_temperature", "ai_max_tokens", "openai_base_url",
                    "openai_api_key", "openai_model", "ollama_base_url", "ollama_model"}
    overrides = {}
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("Model configuration file does not exist")
        overrides = {key.lower(): value for key, value in dotenv_values(args.env_file).items()
                     if key.lower() in model_fields and value is not None}
    configured = Settings(**overrides)
    if configured.ai_provider not in {"openai", "ollama"}:
        parser.error("A real openai/ollama provider is required; Mock is not live evidence")
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    settings = configured.model_copy(update={
        "tool_queue_enabled": True, "tool_loop_enabled": True,
        "alert_email_delivery_mode": "log", "smtp_host": "", "smtp_password": "",
        "excel_path": str(output / "ledger.xlsx"),
        "database_url": f"sqlite:///{(output / 'evidence.db').as_posix()}",
    })
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    worker = ToolQueueWorker(settings, session_factory=factory)
    try:
        with factory() as db:
            user = UserAccount(username="synthetic-live", display_name="合成验收学生", password_hash="unused")
            db.add(user)
            db.flush()
            session = ChatSession(public_id=uuid4().hex, user_id=user.id, title="MB-022 isolated live evidence")
            db.add(session)
            db.flush()
            report = PsychologicalReport(user_id=user.id, session_id=session.id,
                content="合成风险报告；不对应真实学生", intent="RISK", emotion="HIGH_RISK",
                emotion_score=4, risk_level=args.risk, confidence=0.95, summary="synthetic acceptance only")
            db.add(report)
            db.commit()
            job = ToolQueueService(db, settings).enqueue_report(report.id, "UNTRUSTED")[0]
            job_id, report_id = job.id, report.id
            job.status = "RUNNING"
            db.commit()
        worker._run_job(job_id)
        with factory() as db:
            job = db.get(ToolJob, job_id)
            audits = db.query(ToolAuditRecord).filter_by(job_id=job_id).order_by(ToolAuditRecord.id).all()
            events = [{"auditId": a.id, "status": a.status, "allowed": a.allowed,
                       "payload": json.loads(a.payload)} for a in audits]
            counts = {"excel": db.query(ExcelRecord).filter_by(report_id=report_id, status="SUCCESS").count(),
                      "case": db.query(RiskCase).filter_by(report_id=report_id).count(),
                      "alert": db.query(AlertRecord).filter_by(report_id=report_id, status="SUCCESS").count()}
            expected = {"excel": 1, "case": int(args.risk != "LOW"), "alert": int(args.risk == "HIGH")}
            native = [e for e in events if e["status"] == "TOOL_MESSAGE"]
            followup = any(e["status"] == "MODEL_REQUEST" and
                           e["payload"].get("result", {}).get("toolMessageCount", 0) > 0 for e in events)
            passed = job.status == "SUCCESS" and counts == expected and bool(native) and followup
            evidence = {"createdAt": datetime.now(timezone.utc).isoformat(), "passed": passed,
                        "provider": settings.ai_provider,
                        "model": settings.openai_model if settings.ai_provider == "openai" else settings.ollama_model,
                        "riskLevel": args.risk, "deliveryMode": "log", "isolated": True,
                        "parentJobId": job_id, "reportId": report_id, "jobStatus": job.status,
                        "attempts": job.attempts, "error": job.last_error, "products": counts,
                        "nativeToolMessages": len(native), "modelAfterToolMessage": followup, "events": events}
        evidence_path = output / "evidence.json"
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in evidence.items() if k != "events"}, ensure_ascii=True, indent=2))
        print(f"Evidence: {evidence_path}")
        return 0 if passed else 1
    finally:
        worker.stop()
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
