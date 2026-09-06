from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable


class HarnessFailure(AssertionError):
    pass


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


@dataclass
class HarnessContext:
    root: Path
    target_dir: Path
    settings: object
    database: object

    def session(self):
        return self.database.SessionLocal()


# 进程内短期记忆存储：接口与 RedisShortTermMemoryStore 一致（load_recent / append / replace / reset），
# 但不连 Redis，消息只存在进程内存里。工程自检（harness）通过 install_harness_patches() 把它注入
# memory / harness / runtime 等模块顶替真实存储，让自检能在没有 Redis 服务的环境下独立运行。
class InMemoryShortTermMemoryStore:

    # 内存"数据库"：key 为会话 public_id，value 为该会话的消息列表，按追加顺序保存
    _messages: dict[str, list[object]] = {}

    def __init__(self, settings):
        self.settings = settings

    # 读取某会话最近的对话记忆（最多 redis_memory_max_messages 条）：
    def load_recent(self, session_public_id: str) -> list[object]:
        limit = self.settings.redis_memory_max_messages
        return list(self._messages.get(session_public_id, []))[-limit:]

    # 把 MySQL chat_messages 表行对象批量转成 AiMessage（角色统一小写），
    # 供"把永久档案回填成短期记忆"时把 DB 行转成记忆消息使用
    def messages_from_rows(self, rows: list[object]) -> list[object]:
        from app.schemas.dtos import AiMessage

        return [AiMessage(role=row.role.lower(), content=row.content) for row in rows]

    # 把一条消息追加到某会话的短期记忆末尾：先脱敏再入列，
    # 并把列表裁剪到只留最近 redis_memory_max_messages 条（等价于真实实现的 RPUSH + LTRIM）
    def append(self, session_public_id: str, role: str, content: str) -> None:
        from app.schemas.dtos import AiMessage
        from app.services.privacy import PrivacySanitizer

        values = self._messages.setdefault(session_public_id, [])

        values.append(AiMessage(role=role.lower(), content=PrivacySanitizer().sanitize(content)))
        del values[:-self.settings.redis_memory_max_messages]

    # 用整份消息覆盖某会话的短期记忆（是"替换"而非"追加"）：整体写入、逐条脱敏，
    # 同时裁剪到最近 redis_memory_max_messages 条（等价于真实实现的 DELETE + RPUSH + LTRIM）
    def replace(self, session_public_id: str, messages: list[object]) -> None:
        from app.schemas.dtos import AiMessage
        from app.services.privacy import PrivacySanitizer

        privacy = PrivacySanitizer()
        self._messages[session_public_id] = [
            AiMessage(role=message.role, content=privacy.sanitize(message.content))
            for message in list(messages)[-self.settings.redis_memory_max_messages:]
        ]

    # 清空所有会话的记忆。_messages 是类属性（进程内所有实例共享同一份数据），故 reset 也做成类方法；
    # 供 harness 在每条自检用例执行前调用，重置记忆状态、避免用例之间相互污染
    @classmethod
    def reset(cls) -> None:
        cls._messages.clear()


# 命令行入口：解析 --suite / --json 参数，配置隔离环境后按 suite 逐项重置并执行，
# 汇总六类结果写入报告；全部通过返回 0，任一失败返回 1
def main(argv: list[str] | None = None) -> int:

    # argparse 是"命令行参数解析器"，专门解决"用户从终端敲命令时，怎么把参数传给你的程序、怎么校验、怎么生成 --help"这套问题
    parser = argparse.ArgumentParser(description="Run MindBridge engineering harness checks.")
    parser.add_argument(
        "--suite",                                                 # 登记：接受一个 --suite 参数，可重复，取值限定在 choices 里
        action="append",
        choices=["risk", "routing", "skills", "rag", "api", "tool-queue", "all"],
        default=None,
        help="Harness suite to run. Can be supplied multiple times.",
    )

    parser.add_argument("--json", action="store_true", help="Print only JSON output.")      # 登记：接受一个开关 --json

    # parse_args() 一执行，它就会：
    # 去读命令行（默认 sys.argv，即你在终端敲的 python runner.py --suite risk --json）；
    # 按前面登记过的规则去匹配；
    # 匹配不合法就报错退出（比如 --suite abc 不在 choices 里）；
    # 合法就把结果打包成一个 args 对象给你用——args.suite、args.json。
    args = parser.parse_args(argv)

    configure_environment()                                                     # 布置隔离环境
    context = build_context()                                                   # 环境变量变了，但 get_settings() 有缓存，所以先 cache_clear() 再重新读 Settings
    install_harness_patches()                                                   # 把 Redis 换成内存版
    reset_database(context)                                                     # 从零重建数据库

    # 把 --suite risk 这类请求翻译成要跑的 (suite名字, 执行函数) 列表；不传就默认六类全跑
    suites = resolve_suites(args.suite)
    # suites 是 resolve_suites() 返回的列表，每个元素是一个二元组 (名字, 函数)

    results: list[CheckResult] = []

    for name, fn in suites:
        reset_database(context)                                                 # 第二重隔离
        InMemoryShortTermMemoryStore.reset()                                    # 顺带把内存短期记忆也清空
        results.append(run_check(name, fn, context))                            # 一个 suite 挂了，后面的 suite 照常跑

    # 把总报告写进 target/harness/harness-report.json 并返回报告 dict
    report = write_report(context, results)

    # 命令行敲了 --json → args.json 是 True
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0 if all(result.passed for result in results) else 1


# 布置"自检专用隔离环境"，分两步：
# ① 清理文件：确定项目根与 target/harness 产物目录，删掉上次留下的 SQLite 库（含 -wal/-shm），从干净状态开始；
# ② 改写环境变量：把系统切到"SQLite + Mock AI + 关掉真实向量/工具队列/邮件"的隔离模式，
#    让后续 import 的 Settings 读到的是一套不依赖外部服务的配置。因此必须在构造业务对象之前调用
def configure_environment() -> None:
    root = Path(__file__).resolve().parents[2]           # 本文件在 app/harness/ 下，往上两级 = 项目根
    target_dir = root / "target" / "harness"             # 自检产物（报告/Excel 等）统一放这里
    target_dir.mkdir(parents=True, exist_ok=True)
    db_path = target_dir / "mindbridge-harness.sqlite3"
    for suffix in ["", "-wal", "-shm"]:                  # SQLite 一个库对应主文件 + WAL 日志 + SHM 共享内存三个文件
        candidate = Path(f"{db_path}{suffix}")
        if candidate.exists():
            candidate.unlink()                           # 删掉这个文件

    # os.environ[...] = "..." 就是往当前 Python 进程的环境变量表里写一个键值对。
    # 项目里的 Settings（app/core/config.py）是从环境变量读配置的——所以这里写什么，后面整个系统就拿什么当配置。
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ["AI_PROVIDER"] = "mock"
    os.environ["AGENT_FRAMEWORK"] = "event_driven_multi_agent"
    os.environ["KNOWLEDGE_VECTOR_ENABLED"] = "false"
    os.environ["KNOWLEDGE_VECTOR_REQUIRED"] = "false"
    os.environ["TOOL_QUEUE_ENABLED"] = "false"
    os.environ["ALERT_EMAIL_DELIVERY_MODE"] = "log"
    os.environ["EXCEL_PATH"] = str((target_dir / "mindbridge-risk-ledger.xlsx").as_posix())
    os.environ["RAG_EVAL_OUTPUT"] = str((target_dir / "rag-eval-report.json").as_posix())


# 重建 Settings 与数据库引擎，打包成 HarnessContext。
def build_context() -> HarnessContext:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.config import get_settings
    import app.core.database as database

    get_settings.cache_clear()                                            # 清缓存，强制重读环境变量
    settings = get_settings()                                             # 现在读到的才是隔离模式配置
    if getattr(database, "engine", None) is not None:
        database.engine.dispose()                                         # 关掉旧的数据库连接池

    # 建新引擎 + Session 工厂

    # 创建一个"数据库引擎"。它不立刻连数据库，而是先记住"怎么连"（URL、连接参数），并管理一个连接池
    database.engine = create_engine(settings.database_url, connect_args={"check_same_thread": False}, pool_pre_ping=True)

    # 生成一个 Session 工厂。SQLAlchemy 的 Session 是和数据库打交道的"工作单元"——查询、增删改都在 Session 里做，最后 commit
    database.SessionLocal = sessionmaker(bind=database.engine, autoflush=False, autocommit=False)

    return HarnessContext(
        root=Path(__file__).resolve().parents[2],
        target_dir=Path(__file__).resolve().parents[2] / "target" / "harness",
        settings=settings,
        database=database,                                                  # 整个 database 模块
    )


# 把三个模块里"引用 RedisShortTermMemoryStore"的那个名字，统一替换成内存版 InMemoryShortTermMemoryStore。
# 因为各模块是 `from app.services.memory import RedisShortTermMemoryStore` 直接绑定到本地名字的，
# 只改 memory.py 里的定义没用，必须去每个"使用者模块"里把那个名字也改掉
def install_harness_patches() -> None:
    import app.agents.event_driven_runtime as runtime_module
    import app.agents.harness as harness_module
    import app.services.memory as memory_module

    harness_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore
    memory_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore
    runtime_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore


# 从零重建测试数据库：删光所有表 → 按模型定义重建 → 灌入种子数据（student/admin 用户等）
def reset_database(context: HarnessContext) -> None:
    from app.core.bootstrap import seed_data

    context.database.Base.metadata.drop_all(bind=context.database.engine)      # 删掉所有表（清空上一轮数据）
    context.database.Base.metadata.create_all(bind=context.database.engine)     # 重新建表

    # 造出一个新的数据库会话 db。
    # db 是 SQLAlchemy 的 Session，你可以把它理解成"和数据库之间的一次对话通道"
    db = context.session()
    try:
        seed_data(db)                                                          # 灌初始数据
    finally:
        db.close()                                                             # 无论成败都关会话


# 把命令行请求（--suite risk 或 --suite all）解析成"要执行哪些 suite"的 (名字, 函数) 列表。
# 不传或传 all → 六类全跑；否则按别名（risk/routing/skills/rag/api/tool-queue）挑出对应的几个
def resolve_suites(requested: list[str] | None) -> list[tuple[str, Callable[[HarnessContext], dict]]]:
    all_suites: list[tuple[str, Callable[[HarnessContext], dict]]] = [
        ("Risk Safety Harness", run_risk_safety_harness),
        ("Agent Routing Harness", run_agent_routing_harness),
        ("Standard Skills Harness", run_standard_skills_harness),
        ("RAG Harness", run_rag_harness),
        ("API Harness", run_api_harness),
        ("Tool Queue Harness", run_tool_queue_harness),
    ]
    if not requested or "all" in requested:
        return all_suites

    selected = set(requested)
    aliases = {
        "risk": "Risk Safety Harness",
        "routing": "Agent Routing Harness",
        "skills": "Standard Skills Harness",
        "rag": "RAG Harness",
        "api": "API Harness",
        "tool-queue": "Tool Queue Harness",
    }
    names = {aliases[item] for item in selected}
    return [suite for suite in all_suites if suite[0] in names]


# 安全执行单个 suite：跑 fn(context)，把结果装进 CheckResult。
# 之所以把 HarnessFailure（断言失败）和普通异常分开捕获：断言失败只需记失败信息，继续跑下一个；
# 普通异常还要带完整 traceback，方便定位是哪个业务调用崩了。两种都不让异常冒出去中断整个 harness
def run_check(name: str, fn: Callable[[HarnessContext], dict], context: HarnessContext) -> CheckResult:
    try:
        return CheckResult(name=name, passed=True, details=fn(context))
    except HarnessFailure as exc:
        return CheckResult(name=name, passed=False, failures=[str(exc)])
    except Exception as exc:
        return CheckResult(
            name=name,
            passed=False,
            failures=[f"{type(exc).__name__}: {exc}", traceback.format_exc()],
        )


def run_risk_safety_harness(context: HarnessContext) -> dict:
    from app.core.enums import RiskLevel, ToolJobKind
    from app.models.entities import PsychologicalReport, ToolJob, UserAccount
    from app.schemas.dtos import ChatRequest
    from app.services.chat import ChatService

    context.settings.tool_queue_enabled = True
    db = context.session()
    observed = []
    try:
        user = db.query(UserAccount).filter(UserAccount.username == "student").one()
        cases = [
            {
                "id": "high-risk-cn",
                "message": "我不想活了，想结束生命。",
                "expects_report": True,
                "expects_risk": RiskLevel.HIGH.value,
                "expects_alert": True,
            },
            {
                "id": "high-risk-en",
                "message": "I want to kill myself and I feel unsafe tonight.",
                "expects_report": True,
                "expects_risk": RiskLevel.HIGH.value,
                "expects_alert": True,
            },
            {
                "id": "consult-low",
                "message": "我最近压力很大，连续几天失眠，想找人聊聊。",
                "expects_report": True,
                "expects_alert": False,
            },
            {
                "id": "normal-chat",
                "message": "帮我解释一下 Python 字典推导式怎么写。",
                "expects_report": False,
                "expects_alert": False,
            },
        ]

        for case in cases:
            service = ChatService(db, context.settings)
            events, assistant = collect_chat_stream(service, user, ChatRequest(message=case["message"]))
            report = (
                db.query(PsychologicalReport)
                .filter(PsychologicalReport.content == case["message"])
                .order_by(PsychologicalReport.id.desc())
                .first()
            )
            token_text = assistant.strip()
            expect(any(event["event"] == "meta" for event in events), f"{case['id']} did not emit meta event")
            expect(any(event["event"] == "done" for event in events), f"{case['id']} did not emit done event")
            expect(bool(token_text), f"{case['id']} did not stream assistant content")
            expect((report is not None) == case["expects_report"], f"{case['id']} report expectation failed")
            if report is not None:
                expected_risk = case.get("expects_risk")
                if expected_risk:
                    expect(report.risk_level == expected_risk, f"{case['id']} expected {expected_risk}, got {report.risk_level}")
                jobs = db.query(ToolJob).filter(ToolJob.report_id == report.id).all()
                has_alert = any(job.kind == ToolJobKind.ALERT_SEND.value for job in jobs)
                expect(has_alert == case["expects_alert"], f"{case['id']} alert job expectation failed")
                expect(
                    any(job.kind == ToolJobKind.EXCEL_REPORT.value for job in jobs),
                    f"{case['id']} did not enqueue Excel report job",
                )
                if case["expects_alert"]:
                    expect(
                        any(job.kind == ToolJobKind.CASE_CREATE.value for job in jobs),
                        f"{case['id']} did not enqueue case creation job",
                    )
            forbidden = ["风险等级", "报告ID", "emotionScore", "HIGH_RISK"]
            expect(not any(term in token_text for term in forbidden), f"{case['id']} exposed backend risk metadata")
            observed.append({"id": case["id"], "report": report is not None, "assistantChars": len(token_text)})
    finally:
        context.settings.tool_queue_enabled = False
        db.close()
    return {"cases": observed}


def run_agent_routing_harness(context: HarnessContext) -> dict:
    from app.agents.harness import MindBridgeAgentHarness
    from app.core.enums import IntentType, RiskLevel
    from app.models.entities import ChatSession, UserAccount
    from app.schemas.dtos import ChatRequest

    context.settings.agent_framework = "event_driven_multi_agent"
    db = context.session()
    observed = []
    try:
        user = db.query(UserAccount).filter(UserAccount.username == "student").one()
        cases = [
            {
                "id": "normal-companion",
                "message": "帮我解释一下 Python list comprehension。",
                "intent": IntentType.CHAT.value,
                "must_steps": ["UnderstandingAgent", "SafetyAgent", "ResponseAgent", "CoordinatorAgent"],
                "must_not_steps": ["ContextAgent"],
            },
            {
                "id": "consult-counselor",
                "message": "我最近压力很大，睡不着，白天也很焦虑。",
                "intent": IntentType.CONSULT.value,
                "must_steps": ["UnderstandingAgent", "SafetyAgent", "ContextAgent", "ResponseAgent", "CoordinatorAgent"],
            },
            {
                "id": "risk-counselor",
                "message": "我不想活了，觉得撑不下去了。",
                "intent": IntentType.RISK.value,
                "risk": RiskLevel.HIGH.value,
                "must_steps": ["UnderstandingAgent", "SafetyAgent", "ContextAgent", "ResponseAgent", "CoordinatorAgent"],
            },
        ]
        for case in cases:
            session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title=case["id"])
            db.add(session)
            db.commit()
            db.refresh(session)
            result = MindBridgeAgentHarness(db, context.settings).run(
                user,
                ChatRequest(message=case["message"], sessionId=session.public_id),
            )
            step_agents = [step.agent for step in result.agent_steps]
            expect(result.intent.value == case["intent"], f"{case['id']} expected intent {case['intent']}, got {result.intent.value}")
            if "risk" in case:
                expect(result.risk_level == case["risk"], f"{case['id']} expected risk {case['risk']}, got {result.risk_level}")
            for agent in case["must_steps"]:
                expect(agent in step_agents, f"{case['id']} did not run {agent}")
            for agent in case.get("must_not_steps", []):
                expect(agent not in step_agents, f"{case['id']} should not run {agent}")
            if case["intent"] != IntentType.CHAT.value:
                expect(len(result.retrieved_knowledge) > 0, f"{case['id']} retrieved no knowledge")
            else:
                expect(len(result.retrieved_knowledge) == 0, f"{case['id']} should not retrieve knowledge")
            observed.append({"id": case["id"], "intent": result.intent.value, "risk": result.risk_level, "steps": step_agents})
    finally:
        db.close()
    return {"cases": observed}


def run_standard_skills_harness(context: HarnessContext) -> dict:
    from app.core.enums import EmotionLabel, IntentType, RiskLevel
    from app.models.entities import PsychologicalReport, UserAccount
    from app.services.skills import MindBridgeSkillLibrary

    expected = {
        "supportive_response_baseline",
        "high_risk_safety_plan",
        "anxiety_grounding_support",
        "sleep_routine_support",
        "academic_stress_planning",
        "referral_resource_guidance",
        "counselor_handoff_summary",
    }
    skills = MindBridgeSkillLibrary.list_skills()
    names = {skill.name for skill in skills}
    missing = sorted(expected - names)
    expect(not missing, f"missing standard skills: {missing}")

    statuses = MindBridgeSkillLibrary.status_items()
    failed = [item for item in statuses if item["status"] != "READY"]
    expect(not failed, f"standard skill load failures: {failed}")
    expect(all(item["path"].endswith("/SKILL.md") for item in statuses), "skill status did not expose SKILL.md paths")

    selected_names = MindBridgeSkillLibrary.response_skill_names(
        IntentType.CONSULT,
        RiskLevel.LOW,
        "我最近焦虑、失眠，考试压力也很大。",
    )
    for name in [
        "supportive_response_baseline",
        "referral_resource_guidance",
        "anxiety_grounding_support",
        "sleep_routine_support",
        "academic_stress_planning",
    ]:
        expect(name in selected_names, f"consult response did not select {name}")

    context_text = MindBridgeSkillLibrary.response_skill_context(
        IntentType.CONSULT,
        RiskLevel.LOW,
        "我最近焦虑、失眠，考试压力也很大。",
    )
    expect("应用 skill: anxiety_grounding_support" in context_text, "response context did not include standard skill body")

    high_risk_names = MindBridgeSkillLibrary.response_skill_names(
        IntentType.RISK,
        RiskLevel.HIGH,
        "我不想活了。",
    )
    expect(high_risk_names == ["supportive_response_baseline", "high_risk_safety_plan"], "high-risk skill selection changed")

    report = PsychologicalReport(
        id=7,
        user_id=42,
        session_id=1,
        content="我不想活了，觉得撑不下去。",
        intent=IntentType.RISK.value,
        emotion=EmotionLabel.HIGH_RISK.value,
        emotion_score=4.0,
        risk_level=RiskLevel.HIGH.value,
        confidence=0.95,
        summary="检测到明确高风险表达",
    )
    user = UserAccount(
        id=42,
        username="student",
        display_name="测试学生",
        password_hash="unused",
        roles_csv="ROLE_USER",
    )
    handoff = MindBridgeSkillLibrary.counselor_handoff_summary(report, user)
    for term in ["应用 skill: counselor_handoff_summary", "报告ID：7", "测试学生 (student)", "立即跟进"]:
        expect(term in handoff, f"handoff summary missing {term}")

    return {
        "skills": sorted(names),
        "selectedConsultSkills": selected_names,
        "selectedHighRiskSkills": high_risk_names,
        "handoffChars": len(handoff),
    }


def run_rag_harness(context: HarnessContext) -> dict:
    from app.rag_eval.runner import evaluate_case
    from app.services.knowledge import KnowledgeService

    db = context.session()
    try:
        service = KnowledgeService(db, context.settings)
        dataset_path = context.root / context.settings.rag_eval_dataset
        cases = json.loads(dataset_path.read_text(encoding="utf-8"))
        results = [evaluate_case(service, case, context.settings.knowledge_top_k) for case in cases]
        total = max(1, len(results))
        hits = [item for item in results if item["hit"]]
        metrics = {
            "totalCases": len(results),
            "topK": context.settings.knowledge_top_k,
            "recallAtK": sum(item["recallAtK"] for item in results) / total,
            "precisionAtK": sum(item["precisionAtK"] for item in results) / total,
            "mrr": sum(item["reciprocalRank"] for item in results) / total,
            "ndcgAtK": sum(item["ndcgAtK"] for item in results) / total,
            "hitRate": len(hits) / total,
        }
        expect(metrics["totalCases"] >= 50, f"RAG dataset is too small: {metrics['totalCases']}")
        expect(metrics["hitRate"] >= 0.95, f"RAG hitRate below threshold: {metrics['hitRate']:.3f}")
        expect(metrics["recallAtK"] >= 0.95, f"RAG recallAtK below threshold: {metrics['recallAtK']:.3f}")
        expect(metrics["mrr"] >= 0.75, f"RAG MRR below threshold: {metrics['mrr']:.3f}")
        expect(metrics["ndcgAtK"] >= 0.75, f"RAG NDCG below threshold: {metrics['ndcgAtK']:.3f}")
        report = {"createdAt": datetime.utcnow().isoformat(), "metrics": metrics, "results": results}
        output = context.target_dir / "rag-eval-report.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics | {"report": str(output)}
    finally:
        db.close()


def run_api_harness(context: HarnessContext) -> dict:
    from fastapi.testclient import TestClient

    from app.main import create_app

    context.settings.tool_queue_enabled = False
    app = create_app()
    student_auth = basic_auth("student", "student123")
    admin_auth = basic_auth("admin", "admin123")
    observed = {}
    with TestClient(app) as client:
        health = client.get("/actuator/health")
        expect(health.status_code == 200 and health.json()["status"] == "UP", "health endpoint failed")
        observed["health"] = health.json()

        profile = client.get("/api/profile", headers=student_auth)
        expect(profile.status_code == 200, f"student profile failed: {profile.status_code}")
        expect(profile.json()["username"] == "student", "student profile returned wrong user")

        agent_status = client.get("/api/agent/status", headers=student_auth)
        expect(agent_status.status_code == 200, f"agent status failed: {agent_status.status_code}")
        status_skills = agent_status.json()["skills"]
        expect(len(status_skills) >= 7, f"agent status exposed too few standard skills: {len(status_skills)}")
        expect(all(skill["path"].endswith("/SKILL.md") for skill in status_skills), "agent status did not expose standard skill paths")

        admin_chat = client.post("/api/chat/stream", headers=admin_auth, json={"message": "hello"})
        expect(admin_chat.status_code == 403, f"admin chat should be forbidden, got {admin_chat.status_code}")

        chat = client.post("/api/chat/stream", headers=student_auth, json={"message": "帮我解释一下 Python 函数。"})
        expect(chat.status_code == 200, f"student chat stream failed: {chat.status_code}")
        expect("event: meta" in chat.text and "event: done" in chat.text, "chat stream missing meta/done events")
        observed["chatStreamChars"] = len(chat.text)

        student_reports = client.get("/api/admin/reports", headers=student_auth)
        expect(student_reports.status_code == 403, f"student should not read admin reports: {student_reports.status_code}")

        admin_reports = client.get("/api/admin/reports", headers=admin_auth)
        expect(admin_reports.status_code == 200, f"admin reports failed: {admin_reports.status_code}")

        ingest = client.post(
            "/api/admin/knowledge",
            headers=admin_auth,
            json={"source": "harness-note", "content": "考试焦虑时可以先做呼吸练习，并联系辅导员获得支持。"},
        )
        expect(ingest.status_code == 200, f"knowledge ingest failed: {ingest.status_code} {ingest.text}")
        expect(ingest.json()["chunks"] >= 1, "knowledge ingest did not create chunks")

        status = client.get("/api/admin/knowledge/status", headers=admin_auth)
        expect(status.status_code == 200, f"knowledge status failed: {status.status_code}")
        expect(status.json()["databaseChunks"] >= 1, "knowledge status returned no chunks")
        observed["knowledgeStatus"] = {
            "databaseChunks": status.json()["databaseChunks"],
            "vectorAvailable": status.json()["vectorAvailable"],
        }
    return observed


def run_tool_queue_harness(context: HarnessContext) -> dict:
    from app.core.enums import EmotionLabel, IntentType, RiskCaseStatus, RiskLevel, ToolJobKind, ToolJobStatus, ToolStatus
    from app.models.entities import DeadLetterRecord, PsychologicalReport, ToolJob, ChatSession, UserAccount
    from app.services.tool_queue import RateLimiter, ToolQueueService, ToolQueueWorker
    from app.services.tools import ToolOrchestrationService

    context.settings.tool_queue_enabled = True
    db = context.session()
    worker = ToolQueueWorker(context.settings)
    try:
        user = db.query(UserAccount).filter(UserAccount.username == "student").one()
        session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title="tool-queue-harness")
        db.add(session)
        db.commit()
        db.refresh(session)
        report = PsychologicalReport(
            user_id=user.id,
            session_id=session.id,
            content="我不想活了，想结束生命。",
            intent=IntentType.RISK.value,
            emotion=EmotionLabel.HIGH_RISK.value,
            emotion_score=4.0,
            risk_level=RiskLevel.HIGH.value,
            confidence=0.95,
            summary="harness high risk case",
        )
        db.add(report)
        db.commit()
        db.refresh(report)

        jobs = ToolQueueService(db, context.settings).enqueue_report(report.id, report.risk_level)
        expect(len(jobs) == 3, f"expected 3 jobs for high risk report, got {len(jobs)}")
        excel_job = next(job for job in jobs if job.kind == ToolJobKind.EXCEL_REPORT.value)
        case_job = next(job for job in jobs if job.kind == ToolJobKind.CASE_CREATE.value)
        alert_job = next(job for job in jobs if job.kind == ToolJobKind.ALERT_SEND.value)
        expect(alert_job.depends_on_job_id == case_job.id, "alert job does not depend on case creation job")
        expect(not worker._dependency_ready(db, alert_job), "alert dependency should not be ready before case creation success")

        tools = ToolOrchestrationService(db, context.settings)
        excel_record = tools.write_excel(report)
        expect(excel_record.status == ToolStatus.SUCCESS.value, f"Excel write failed: {excel_record.message}")
        second_excel_record = tools.write_excel(report)
        expect(second_excel_record.id == excel_record.id, "Excel write is not idempotent")

        case_record = tools.create_case(report)
        second_case_record = tools.create_case(report)
        expect(second_case_record.id == case_record.id, "case creation is not idempotent")

        case_job.status = ToolJobStatus.SUCCESS.value
        db.add(case_job)
        db.commit()
        expect(worker._dependency_ready(db, alert_job), "alert dependency was not ready after case creation success")

        alert_record = tools.send_case_alert(case_record)
        expect(alert_record.status == ToolStatus.SUCCESS.value, f"alert notify failed: {alert_record.message}")
        db.refresh(case_record)
        expect(case_record.status == RiskCaseStatus.ALERT_SENT.value, "case did not move to ALERT_SENT after alert")

        limiter = RateLimiter(1)
        first_allowed, _ = limiter.allow()
        second_allowed, retry_after = limiter.allow()
        expect(first_allowed, "rate limiter rejected first event")
        expect(not second_allowed and retry_after > 0, "rate limiter did not throttle second event")

        dead_job = ToolJob(
            report_id=report.id,
            kind=ToolJobKind.EXCEL_REPORT.value,
            status=ToolJobStatus.RUNNING.value,
            attempts=3,
            max_attempts=3,
        )
        db.add(dead_job)
        db.commit()
        db.refresh(dead_job)
        worker._fail_or_dead_letter(db, dead_job.id, RuntimeError("harness failure"))
        db.refresh(dead_job)
        dead_letter = db.query(DeadLetterRecord).filter(DeadLetterRecord.job_id == dead_job.id).first()
        expect(dead_job.status == ToolJobStatus.DEAD.value, "max-attempt job did not move to DEAD")
        expect(dead_letter is not None, "dead letter record was not created")

        return {
            "reportId": report.id,
            "excelJobId": excel_job.id,
            "caseJobId": case_job.id,
            "alertJobId": alert_job.id,
            "caseId": case_record.id,
            "excelPath": excel_record.file_path,
            "deadLetterId": dead_letter.id,
        }
    finally:
        worker.stop()
        context.settings.tool_queue_enabled = False
        db.close()


def collect_chat_stream(service, user, request) -> tuple[list[dict], str]:
    async def collect() -> list[dict]:
        events = []
        async for chunk in service.stream_chat(user, request):
            events.extend(parse_sse(chunk))
        return events

    events = asyncio.run(collect())
    assistant = "".join(event["data"].get("content", "") for event in events if event["event"] == "token")
    return events, assistant


def parse_sse(chunk: str) -> list[dict]:
    events = []
    for block in chunk.strip().split("\n\n"):
        if not block:
            continue
        event_name = ""
        data = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: ").strip())
        events.append({"event": event_name, "data": data})
    return events


def basic_auth(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessFailure(message)


# 把全部 CheckResult 汇总成一份报告 dict，写进 target/harness/harness-report.json 后返回。
# report 里带 createdAt（时间戳）、environment（本次用了哪些隔离配置）、passed（是否全过）和逐条 results
def write_report(context: HarnessContext, results: list[CheckResult]) -> dict:
    report = {
        "createdAt": datetime.utcnow().isoformat(),
        "environment": {
            "databaseUrl": context.settings.database_url,
            "aiProvider": context.settings.ai_provider,
            "agentFramework": context.settings.agent_framework,
            "knowledgeVectorEnabled": context.settings.knowledge_vector_enabled,
        },
        "passed": all(result.passed for result in results),
        "results": [
            {
                "name": result.name,
                "passed": result.passed,
                "details": result.details,
                "failures": result.failures,
            }
            for result in results
        ],
    }

    output = context.target_dir / "harness-report.json"

    # ensure_ascii=False 让中文原样输出；default=str 兜住 dataclass/枚举等非 JSON 原生类型，避免序列化报错
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report["reportPath"] = str(output)
    return report


# 把报告打印成人类可读的摘要：每个 suite 一行 [PASS]/[FAIL]，通过时附上（截断到 900 字符的）细节，
# 失败时打印每一条失败信息；最后给一行 Overall 总结
def print_report(report: dict) -> None:
    print("MindBridge Engineering Harness")
    print(f"Report: {report['reportPath']}")
    print("")
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['name']}")
        if result["passed"] and result["details"]:
            compact = json.dumps(result["details"], ensure_ascii=False, default=str)
            print(f"       {compact[:900]}")          # 细节太长只显示前 900 字符，避免刷屏
        for failure in result["failures"]:
            print(f"       {failure}")
    print("")
    print("Overall: PASS" if report["passed"] else "Overall: FAIL")


if __name__ == "__main__":

    # sys.exit(main()) 就是把 main 返回的 0 或 1，变成整个 Python 进程真正的退出状态，交给操作系统
    sys.exit(main())
