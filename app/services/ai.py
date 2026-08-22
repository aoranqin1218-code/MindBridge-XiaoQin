from __future__ import annotations

import json
from typing import Iterable

import httpx

from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage


# Prompt 模板工具类：只把三个"拼 prompt"的函数归拢在一起，无状态、无实例，全部静态方法直接类名调用
class PromptTemplates:

    # UnderstandingAgent.act()，判断用户这轮想干什么
    # 拼出"意图分类"用的prompt：system 说明分类规则 + user 放最近上下文和当前输入
    @staticmethod
    def intent_prompt(history: list[AiMessage], user_input: str) -> list[AiMessage]:
        return [
            AiMessage(role="system", content=(
                "你是一个用户意图分类器，只做意图识别，不回答问题。"
                "只输出 CHAT、CONSULT、RISK 之一。CHAT 包含普通闲聊、学习、编程、作业、校园事务；"
                "CONSULT 包含压力、焦虑、低落、失眠、情绪倾诉；RISK 包含自杀、自残、伤人或即时危险信号。"
            )),
            AiMessage(role="user", content=f"最近上下文：\n{format_history(history)}\n\n当前输入：\n{user_input}"),
        ]

    # PsychologicalAssessmentService.assess()，也就是 SafetyAgent 的风险评估链路的模型环节
    # 拼出"心理评估"用的prompt：system 要求输出严格 JSON 的评估结果 + user 放上下文和输入
    @staticmethod
    def psychology_prompt(history: list[AiMessage], user_input: str) -> list[AiMessage]:
        return [
            AiMessage(role="system", content=(
                "你负责分析校园心理健康消息。只返回严格 JSON："
                '{"emotion":"NORMAL|ANXIETY|DEPRESSED|HIGH_RISK","emotionScore":0.0,'
                '"risk":"LOW|MEDIUM|HIGH","confidence":0.0,"summary":"short reason"}'
            )),
            AiMessage(role="user", content=f"最近上下文：\n{format_history(history)}\n\n当前输入：\n{user_input}"),
        ]

    # ResponseAgent.act() 两个分支，以及 runtime 的 _fallback_messages
    # 拼出"最终回复"用的系统prompt：按意图分闲聊/心理关怀两套话术，高危时追加危机处理规则
    @staticmethod
    def answer_system_prompt(intent: IntentType, risk: RiskLevel, context: str, display_name: str, skill_context: str = "") -> AiMessage:
        if intent == IntentType.CHAT:
            content = (
                "你是 MindBridge，一个面向学生的日常陪伴与校园生活助手。"
                "普通学习、编程、校园事务和通用问题请自然、准确、直接地回答。"
                "不要主动做心理测评，不要输出风险等级、心理标签、诊断结论或报告口吻。"
                f"学生显示名：{display_name}"
            )
            return AiMessage(role="system", content=content)
        
        crisis_rule = ""
        if risk == RiskLevel.HIGH:
            crisis_rule = (
                "\n高风险处理规则：先回应情绪，再关注当前安全；鼓励用户立刻联系身边可信任的人、"
                "学校辅导员/心理中心或当地紧急救助；不提供任何危险操作细节。"
            )
        content = (
            "你是 MindBridge，一个面向学生的校园心理关怀智能体。"
            "回答要共情、谨慎、非评判，不诊断疾病，不开药，不替代持证心理咨询师。"
            "不要向学生输出风险等级、报告分数或后台标签。"
            "优先基于检索知识回答；知识不足时明确说明并给出安全通用建议。"
            f"\n学生显示名：{display_name}\n检索知识：\n{context}\n\n可用 skill 指引：\n{skill_context or '无'}{crisis_rule}"
        )
        return AiMessage(role="system", content=content)


class AiClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    # 同步调模型拿完整回答：按 provider 分派到 ollama/openai/mock，非流式，一次返回全部文本
    def complete(self, messages: list[AiMessage]) -> str:
        provider = self.settings.ai_provider.lower()
        if provider == "ollama":
            return self._ollama(messages, stream=False)
        if provider == "openai":
            return self._openai(messages, stream=False)
        # 保证程序永远有返回值、不崩
        return self._mock(messages)

    async def stream(self, messages: list[AiMessage]):
        provider = self.settings.ai_provider.lower()
        if provider == "ollama":
            async for token in self._ollama_stream(messages):
                yield token
            return
        if provider == "openai":
            async for token in self._openai_stream(messages):
                yield token
            return
        text = self._mock(messages)
        for chunk in split_text(text, 12):
            yield chunk

    def _ollama(self, messages: list[AiMessage], stream: bool) -> str:

        payload = {
            "model": self.settings.ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "options": {"temperature": self.settings.ai_temperature, "num_predict": self.settings.ai_max_tokens},
        }

        response = httpx.post(f"{self.settings.ollama_base_url}/api/chat", json=payload, timeout=60)
        response.raise_for_status()                                                     # 检查 HTTP 状态码
        return response.json()["message"]["content"]                                    # 取的是模型实际生成的回复文本

    async def _ollama_stream(self, messages: list[AiMessage]):
        payload = {
            "model": self.settings.ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            "options": {"temperature": self.settings.ai_temperature, "num_predict": self.settings.ai_max_tokens},
        }

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", f"{self.settings.ollama_base_url}/api/chat", json=payload) as response:
                response.raise_for_status()

                # 一行一行读响应体。Ollama 流式 /api/chat 走 SSE 格式：每个 token 是一行 JSON。所以"读一行"≈"拿到模型刚生成的一小段"
                async for line in response.aiter_lines():
                    if not line:                                                    # 跳过空行（SSE 流里行与行之间有空行分隔符）
                        continue
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")              # 取 message.content——就是模型生成的 token 文本
                    if token:
                        yield token

    def _openai(self, messages: list[AiMessage], stream: bool) -> str:
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.openai_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
            "stream": stream,
        }
        response = httpx.post(f"{self.settings.openai_base_url}/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def _openai_stream(self, messages: list[AiMessage]):
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.openai_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", f"{self.settings.openai_base_url}/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line.removeprefix("data: ").strip()
                    if raw == "[DONE]":
                        break
                    data = json.loads(raw)
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    token = choices[0].get("delta", {}).get("content", "")
                    if token:
                        yield token

    def _mock(self, messages: list[AiMessage]) -> str:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        system = " ".join(m.content for m in messages if m.role == "system")

        if "严格 JSON" in system:
            if has_high_risk_signal(last):
                return '{"emotion":"HIGH_RISK","emotionScore":4.0,"risk":"HIGH","confidence":0.95,"summary":"检测到明确高风险表达"}'
            if has_consult_signal(last):
                return '{"emotion":"ANXIETY","emotionScore":2.5,"risk":"LOW","confidence":0.72,"summary":"检测到压力或情绪求助表达"}'
            return '{"emotion":"NORMAL","emotionScore":0.0,"risk":"LOW","confidence":0.66,"summary":"未检测到明显风险信号"}'

        if "意图分类器" in system:
            if has_high_risk_signal(last):
                return "RISK"
            if has_consult_signal(last):
                return "CONSULT"
            return "CHAT"
        if "high_risk_safety_plan" in system and has_high_risk_signal(last):
            return "我听到你现在已经痛苦到觉得撑不下去了。现在最重要的是先让你不要一个人扛：请马上联系身边可信任的人，或者直接联系辅导员、学校心理中心、校园保卫/当地紧急服务。接下来 10 分钟，请先把自己移到有人在的地方，并把可能伤害自己的东西放远一点。如果可以，回我一句：你现在身边有没有可以马上联系或走过去找的人？"

        if "当前由 ResponseAgent 以 support mode" in system:
            return "我听到你最近压力很大，还影响到了睡眠，这种状态确实会让人很消耗。你可以先做两件小事：今晚把最担心的事情写成清单，先只选一个最小步骤处理；睡前 30 分钟把手机和学习任务放远一点，用缓慢呼吸或热水澡帮身体降下来。如果这种失眠持续一周以上，建议联系学校心理中心或辅导员一起看一看。"

        if "当前由 ResponseAgent 以 normal_chat mode" in system:
            return "我在。这个问题可以直接拆开来看，我们先从你最想解决的那一部分开始。"

        if "ContextAgent" in system and "SUFFICIENT" in system:
            return "SUFFICIENT"

        if "ContextAgent" in system:
            return last[:40] or "校园心理支持"
        return "我在。先把你现在最具体的困扰说出来，我们可以一步一步拆开。如果情况已经影响安全，请马上联系身边可信任的人或学校心理中心。"


# 供 prompt 拼装使用：把消息列表拍平成多行文本（每行"角色: 内容"），只取最近 20 条，空列表返回"无"
def format_history(history: list[AiMessage]) -> str:
    if not history:
        return "无"
    return "\n".join(f"{m.role}: {m.content}" for m in history[-20:])


# 硬编码高风险词典：子串命中即视为明确高风险信号（自杀/自残/告别类），最前置硬守卫，不依赖模型
HIGH_RISK_WORDS = ["自杀", "自残", "不想活", "结束生命", "伤害自己", "轻生", "suicide", "kill myself", "self harm"]
# 硬编码情绪求助词典：命中表示可能存在压力/焦虑/倾诉等，供意图兜底与启发式风险评估使用
CONSULT_WORDS = ["焦虑", "抑郁", "压力", "失眠", "难过", "崩溃", "痛苦", "无助", "心理", "咨询", "anxious", "depress", "stress"]


# 检测文本是否含高风险信号：转小写后子串匹配，命中任一词即 True；必须在 LLM 评估之前执行作底线守卫
def has_high_risk_signal(text: str) -> bool:
    normalized = text.lower()
    return any(word in normalized for word in HIGH_RISK_WORDS)


# 检测文本是否含情绪求助/咨询信号：同样子串匹配，供意图兜底与启发式风险评估使用
def has_consult_signal(text: str) -> bool:
    normalized = text.lower()
    return any(word in normalized for word in CONSULT_WORDS)


# 把长文本按 size 切块逐个 yield，供 mock 流式输出时模拟逐段推送
def split_text(text: str, size: int) -> Iterable[str]:
    for index in range(0, len(text), size):

        # yield 让一个普通函数变成生成器（generator）——它不回一个结果就结束，而是每次产出/暂停一个值，调用方要一个它给一个
        yield text[index:index + size]
