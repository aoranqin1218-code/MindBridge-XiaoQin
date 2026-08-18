from enum import Enum

# 枚举——用有名字的常量代替裸字符串
# Python 的多继承——MessageRole 同时继承了 str 和 Enum 两个类的特性
# Enum — 让类变成枚举。效果是值被锁定，不能瞎写
# str — 让每个枚举值同时也是一个字符串

class MessageRole(str, Enum):
    USER = "USER"                                                           # 学生/用户
    ASSISTANT = "ASSISTANT"                                                 # AI 助手
    SYSTEM = "SYSTEM"                                                       # 系统级提示词


class IntentType(str, Enum):
    CHAT = "CHAT"
    CONSULT = "CONSULT"
    RISK = "RISK"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EmotionLabel(str, Enum):
    NORMAL = "NORMAL"
    ANXIETY = "ANXIETY"
    DEPRESSED = "DEPRESSED"
    HIGH_RISK = "HIGH_RISK"


class ToolStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ToolJobKind(str, Enum):
    EXCEL_REPORT = "EXCEL_REPORT"
    CASE_CREATE = "CASE_CREATE"
    ALERT_SEND = "ALERT_SEND"
    RISK_ALERT = "RISK_ALERT"


class ToolJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    DEAD = "DEAD"


class RiskCaseStatus(str, Enum):
    OPEN = "OPEN"
    ALERT_SENT = "ALERT_SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
