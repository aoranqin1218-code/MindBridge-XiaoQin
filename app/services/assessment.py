from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.enums import EmotionLabel, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, PromptTemplates, has_consult_signal, has_high_risk_signal


# 心理评估结果：由硬守卫/模型评估/启发式三种路径之一产出，作为风险链的核心结论进黑板 risk 产物
@dataclass
class PsychologyAssessment:
    emotion: EmotionLabel            # 情绪标签 NORMAL/ANXIETY/DEPRESSED/HIGH_RISK，供报告与后续环节展示
    emotion_score: float             # 情绪严重度分数（0~4 档），由模型 emotionScore 或 score_for_emotion 推导
    risk: RiskLevel                  # 风险等级 LOW/MEDIUM/HIGH，核心输出：SafetyAgent 据此发 SAFETY_OVERRIDE 和决定审查严苛度
    confidence: float                # 评估置信度 0~1，会被 clamp 到区间内，进 risk 产物供 trace/报告取用
    summary: str                     # 一句话评估摘要（人话），供后续环节阅读和报告引用


# 风险评估服务：输入用户文本+历史，按"硬守卫 → 模型评估 → 启发式兜底"三条路径产出 PsychologyAssessment；是 SafetyAgent 风险判断的核心
class PsychologicalAssessmentService:
    def __init__(self, ai: AiClient):
        self.ai = ai

    # 风险链入口：先硬词典短路，未命中则调模型做结构化评估，模型/解析任何一步出错都退回启发式兜底，保证永不崩溃
    def assess(self, text: str, history: list[AiMessage] | None = None) -> PsychologyAssessment:

        # 硬守卫：命中明确高风险信号直接返回，不调用模型
        if has_high_risk_signal(text):
            return PsychologyAssessment(EmotionLabel.HIGH_RISK, 4.0, RiskLevel.HIGH, 0.95, "检测到明确高风险表达")
        
        try:
            raw = self.ai.complete(PromptTemplates.psychology_prompt(history or [], text))

            # 截取首尾大括号之间的内容再解析：容忍模型在 JSON 前后夹杂解释性文字
            start = raw.find("{")                                                           # 从左往右找第一个 {
            end = raw.rfind("}")                                                            # 从右往左找最后一个 }
            data = json.loads(raw[start:end + 1] if start >= 0 and end > start else raw)

            emotion = EmotionLabel(data.get("emotion", "NORMAL").upper())

            score = float(data.get("emotionScore", score_for_emotion(emotion)))

            risk = RiskLevel(data.get("risk", risk_from_score(score).value).upper())

            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.75))))    # 置信度限制在 0~1

            # 模型给的数值证据优先于模型给的主观标签；两者冲突时，取更保守（更高）的那一个
            score_risk = risk_from_score(score)
            if risk_order(score_risk) > risk_order(risk):
                risk = score_risk

            # HIGH_RISK 情绪强制升级为高风险

            if emotion == EmotionLabel.HIGH_RISK:
                risk = RiskLevel.HIGH
            return PsychologyAssessment(emotion, score, risk, confidence, data.get("summary", "模型评估结果"))
        except Exception:
            # 模型请求/JSON 解析/枚举转换任何一步出错都不崩，退到关键词启发式
            return heuristic(text)


# 启发式兜底：不调模型，仅凭关键词给出保守风险结论；服务于"模型环节挂了也要有评估结果"
def heuristic(text: str) -> PsychologyAssessment:
    if has_consult_signal(text):
        if any(word in text.lower() for word in ["抑郁", "低落", "崩溃", "难过", "depress", "hopeless"]):
            return PsychologyAssessment(EmotionLabel.DEPRESSED, 3.1, RiskLevel.MEDIUM, 0.75, "检测到低落或抑郁相关表达")
        
        return PsychologyAssessment(EmotionLabel.ANXIETY, 2.2, RiskLevel.LOW, 0.72, "检测到焦虑或压力相关表达")
    
    return PsychologyAssessment(EmotionLabel.NORMAL, 0.0, RiskLevel.LOW, 0.66, "未检测到明显风险信号")


# 情绪标签 → 严重度分数：JSON 缺 emotionScore 字段时用它兜底
def score_for_emotion(emotion: EmotionLabel) -> float:
    return {
        EmotionLabel.HIGH_RISK: 4.0,
        EmotionLabel.DEPRESSED: 3.0,
        EmotionLabel.ANXIETY: 2.0,
        EmotionLabel.NORMAL: 0.0,
    }[emotion]


# 分数 → 风险等级：4 分以上 HIGH、3 分以上 MEDIUM、否则 LOW；供保守校正和缺 risk 字段时兜底
def risk_from_score(score: float) -> RiskLevel:
    if score >= 4:
        return RiskLevel.HIGH
    if score >= 3:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# 风险等级 → 数字：LOW=1/MEDIUM=2/HIGH=3，用于"只向更高风险升级"的比较
def risk_order(risk: RiskLevel) -> int:
    return {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}[risk]
