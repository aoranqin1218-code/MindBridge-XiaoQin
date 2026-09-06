from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.enums import IntentType, RiskLevel
from app.models.entities import PsychologicalReport, UserAccount


# 加载 skill 失败时抛出的异常：SKILL.md 缺 name/description/body、frontmatter 格式错误、找不到标准 skill 等场景
class SkillLoadError(RuntimeError):
    pass


# 一条 skill 校验问题（WARN 提示 / ERROR 阻断），供状态展示与人工审阅用
@dataclass(frozen=True)
class SkillValidationIssue:
    level: str    # 严重级别：WARN（建议性）或 ERROR（必须修复）
    message: str  # 问题描述（中文，给人看）



# 一个加载后的 skill 对象：由 SKILL.md 解析而来，可生成喂给模型的 prompt 文本
@dataclass(frozen=True)
class MindBridgeSkill:
    name: str                                                   # skill 名（frontmatter 的 name，通常与目录名一致）
    description: str                                            # 触发场景描述（frontmatter 的 description，给模型看何时用）
    body: str                                                   # SKILL.md 正文（去掉 frontmatter 后的部分）
    path: Path                                                  # SKILL.md 文件路径
    metadata: dict[str, str] = field(default_factory=dict)      # frontmatter 里的其他键值对

    # 生成喂给模型的 prompt 文本：
    # "应用 skill: {name}\n{body}"——最终拼进回复 system prompt 的 skill 指引段
    def prompt_context(self) -> str:
        return f"应用 skill: {self.name}\n{self.body.strip()}"

    # 校验这个 skill 是否结构合格，返回问题列表（WARN/ERROR）；空列表表示无问题
    def validation_issues(self) -> list[SkillValidationIssue]:
        issues: list[SkillValidationIssue] = []

        if self.path.parent.name != self.name:
            issues.append(SkillValidationIssue("WARN", f"目录名 {self.path.parent.name} 与 skill name {self.name} 不一致"))
        if "## Workflow" not in self.body:
            issues.append(SkillValidationIssue("WARN", "建议包含 ## Workflow 小节，便于人工审阅和模型稳定加载"))
        if len(self.description) < 20:
            issues.append(SkillValidationIssue("WARN", "description 太短，可能无法准确表达触发场景"))
        if self.name == "counselor_handoff_summary" and "```text" not in self.body:
            issues.append(SkillValidationIssue("ERROR", "counselor_handoff_summary 必须包含 text 模板"))
        return issues


# Skill 注册表：扫描 skills/ 目录下的 SKILL.md，加载、校验、按名字取用
class MindBridgeSkillRegistry:

    def __init__(self, root: Path | None = None):

        # 没传就默认取"当前文件所在项目根下的 skills/"
        self.root = root or Path(__file__).resolve().parents[2] / "skills"

    # 返回：所有已加载的 MindBridgeSkill 对象列表（按目录名排序）；skills 目录不存在返回空列表
    def list_skills(self) -> list[MindBridgeSkill]:
        if not self.root.exists():
            return []

        skills = []

        # sorted(...) 对 glob 返回的 Path 列表按路径字符串的字典序排序
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            skills.append(self._load_skill_file(skill_file))
        return skills

    # 返回：每个 skill 的状态 dict 列表（name/status/description/path/issues/metadata）；加载失败的也以 FAILED 收录
    def status_items(self) -> list[dict]:
        if not self.root.exists():
            return []

        items = []
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            try:
                skill = self._load_skill_file(skill_file)
                issues = skill.validation_issues()                                  # 校验这个 skill 是否结构合格
            except SkillLoadError as exc:
                items.append(
                    {
                        "name": skill_file.parent.name,
                        "status": "FAILED",
                        "description": str(exc),
                        # as_posix() 统一转正斜杠：Path 转字符串在 Windows 是 \，但这是暴露给 API/前端的 web 路径，须跨平台用 /
                        "path": skill_file.relative_to(self.root.parent).as_posix(),
                        "issues": [{"level": "ERROR", "message": str(exc)}],
                    }
                )
                continue

            has_error = any(issue.level == "ERROR" for issue in issues)
            items.append(
                {
                    "name": skill.name,
                    "status": "FAILED" if has_error else "READY" if not issues else "WARN",
                    "description": skill.description,
                    "path": skill.path.relative_to(self.root.parent).as_posix(),
                    "issues": [{"level": issue.level, "message": issue.message} for issue in issues],
                    "metadata": skill.metadata,
                }
            )
        return items

    # 按名字取一个标准 skill，返回 MindBridgeSkill；找不到抛 SkillLoadError
    def get_required(self, name: str) -> MindBridgeSkill:
        for skill in self.list_skills():
            if skill.name == name:
                return skill
        raise SkillLoadError(f"required standard skill not found: {name}")

    # 提取指定 skill body 里的 ```text``` 代码块作为模板，返回模板字符串；没有该代码块抛 SkillLoadError
    def template_for(self, name: str) -> str:
        skill = self.get_required(name)
        match = re.search(r"```text\s*\n(?P<template>.*?)\n```", skill.body, re.DOTALL)
        if match is None:
            raise SkillLoadError(f"standard skill {name} does not define a text template")
        return match.group("template").strip()

    # 读取一个 SKILL.md 文件并构造 MindBridgeSkill：拆 frontmatter + body，校验必填字段；缺任何必填抛 SkillLoadError
    def _load_skill_file(self, path: Path) -> MindBridgeSkill:
        text = path.read_text(encoding="utf-8")
        metadata, body = _split_frontmatter(text, path)

        name = metadata.get("name") or path.parent.name
        description = metadata.get("description", "")
        if not name.strip():
            raise SkillLoadError(f"{path} is missing frontmatter name")
        if not description.strip():
            raise SkillLoadError(f"{path} is missing frontmatter description")
        if not body.strip():
            raise SkillLoadError(f"{path} is missing skill body")
        return MindBridgeSkill(name=name.strip(), description=description.strip(), body=body.strip(), path=path, metadata=metadata)


# Skill 门面：全静态方法，把注册表和"按需选 skill"的规则集中对外，供 ContextAgent 等业务层直接调用
class MindBridgeSkillLibrary:

    @staticmethod
    def registry() -> MindBridgeSkillRegistry:
        return MindBridgeSkillRegistry()

    # 返回：所有 skill 的 MindBridgeSkill 列表（转发 registry.list_skills）
    @staticmethod
    def list_skills() -> list[MindBridgeSkill]:
        return MindBridgeSkillLibrary.registry().list_skills()

    # 返回：每个 skill 状态 dict 的列表（转发 registry.status_items），供管理接口展示
    @staticmethod
    def status_items() -> list[dict]:
        return MindBridgeSkillLibrary.registry().status_items()

    # 按"意图+风险+文本"选出 skill，把它们的 prompt 文本用换行拼接成一段，返回该段文本；CHAT 意图返回空串
    # 入参 intent/risk：黑板里的意图与风险等级；text：用户输入（用于关键词匹配）
    @staticmethod
    def response_skill_context(intent: IntentType, risk: RiskLevel, text: str) -> str:
        names = MindBridgeSkillLibrary.response_skill_names(intent, risk, text)
        registry = MindBridgeSkillLibrary.registry()
        return "\n\n".join(registry.get_required(name).prompt_context() for name in names)

    # 按规则选出本次回复该注入哪些 skill 的名字列表：CHAT→空；HIGH→固定两个；否则按关键词追加焦虑/失眠/学业
    # 入参 intent/risk：意图与风险等级；text：用户输入；返回：skill 名列表（已去重）
    @staticmethod
    def response_skill_names(intent: IntentType, risk: RiskLevel, text: str) -> list[str]:
        if intent == IntentType.CHAT:
            return []

        if risk == RiskLevel.HIGH:
            return ["supportive_response_baseline", "high_risk_safety_plan"]

        lowered = text.lower()
        names = ["supportive_response_baseline", "referral_resource_guidance"]
        if _contains_any(lowered, ["焦虑", "惊恐", "恐慌", "panic", "anxious", "崩溃", "呼吸"]):
            names.append("anxiety_grounding_support")
        if _contains_any(lowered, ["失眠", "睡不着", "睡眠", "熬夜", "sleep", "insomnia"]):
            names.append("sleep_routine_support")
        if _contains_any(lowered, ["考试", "挂科", "绩点", "论文", "作业", "学业", "学习", "academic", "exam"]):
            names.append("academic_stress_planning")
        return _dedupe(names)

    @staticmethod
    def high_risk_safety_plan_prompt() -> str:
        return MindBridgeSkillLibrary.registry().get_required("high_risk_safety_plan").prompt_context()

    # 用 counselor_handoff_summary 模板渲染"给辅导员的交接摘要"，返回渲染后的文本
    # 入参 report：心理报告实体；user：用户对象（可能 None）；返回：按 {{占位符}} 替换好的交接文本
    @staticmethod
    def counselor_handoff_summary(report: PsychologicalReport, user: UserAccount | None) -> str:
        template = MindBridgeSkillLibrary.registry().template_for("counselor_handoff_summary")
        student = _student_label(user, report.user_id)
        urgency = "立即跟进" if report.risk_level == RiskLevel.HIGH.value else "尽快跟进"
        next_steps = [
            f"{urgency}，确认学生当前位置、身边是否有人陪伴，以及当前是否安全。",
            "联系学生本人或其可用的现实支持人，并记录已采取的联系方式。",
            "必要时联系校园保卫、心理中心值班老师或当地紧急救助。",
            "将后续安排、接手人和下一次复访时间写入个案备注。",
        ]
        return _render_template(
            template,
            {
                "report_id": str(report.id),
                "student": student,
                "risk_level": report.risk_level,
                "emotion": report.emotion,
                "confidence": f"{report.confidence:.2f}",
                "summary": report.summary,
                "next_steps": "\n".join(f"- {step}" for step in next_steps),
                "content_excerpt": _truncate(report.content, 700),
            },
        )


# 把 SKILL.md 文本拆成 (frontmatter dict, 正文) 两段：
# 解析 YAML 头（--- 到 ---）里的键值对，返回元组；格式错误抛 SkillLoadError
def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str]:

    if not text.startswith("---\n"):
        raise SkillLoadError(f"{path} is missing YAML frontmatter")

    # 4 是跳过开头 ---\n 那 4 个字符，防的是"开头符被误认成结束符"
    end = text.find("\n---", 4)

    if end == -1:
        raise SkillLoadError(f"{path} has unterminated YAML frontmatter")

    metadata = {}

    # .splitlines() 把它按行切开，逐行处理元数据
    for line in text[4:end].splitlines():
        stripped = line.strip()                                             # 去掉每行首尾空白
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SkillLoadError(f"{path} has invalid frontmatter line: {line}")
        key, value = stripped.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text[end + len("\n---") :].strip()


# 判断文本里是否包含 terms 中任意一个词，返回布尔；用于按关键词追加 skill
def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


# 列表去重并保持原顺序，返回去重后的新列表
def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


# 用 values 里的键值对把模板里的 {{key}} 占位符替换掉，返回渲染后的字符串
def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


# 生成学生的展示标签：有 user 用"显示名 (用户名)"，只有 user_id 就返回 userId=xx
def _student_label(user: UserAccount | None, user_id: int) -> str:
    if user is None:
        return f"userId={user_id}"
    if user.display_name:
        return f"{user.display_name} ({user.username})"
    return user.username


# 折叠空白后按 limit 截断，超出补 "..."；空值归一为空串（与 memory.py 的 _clip 同一套路）
def _truncate(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit - 3]}..."
