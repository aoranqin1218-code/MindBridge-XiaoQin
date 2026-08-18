import unittest
import json

from app.core.enums import RiskLevel
from app.services.assessment import PsychologicalAssessmentService
from app.services.memory import RedisShortTermMemoryStore
from app.services.privacy import PrivacySanitizer

# 用会爆炸的假货占住坑位，确保代码真的没往那个坑走。测试通过了 = 硬守卫生效了
class ExplodingAi:
    def complete(self, messages):
        raise AssertionError("high risk hard guard should not call the model")

# 用 python test_privacy_and_assessment.py 直接执行文件时，__name__ 就是 "__main__"，于是 unittest.main() 启动，
# 它扫描当前模块里继承自 unittest.TestCase 的类，把以 test_ 开头的方法逐个执行，最后输出通过/失败/错误的结果
class PrivacyAndAssessmentTests(unittest.TestCase):
    def test_privacy_sanitizer_masks_common_identifiers(self):
        text = PrivacySanitizer().sanitize("电话 13800138000 邮箱 a@example.com 身份证 110101199003071234")

        self.assertNotIn("13800138000", text)
        self.assertNotIn("a@example.com", text)
        self.assertNotIn("110101199003071234", text)
        self.assertEqual(text.count("[已脱敏]"), 3)

    def test_redis_memory_serializes_sanitized_content(self):

        # __new__ 创建一个类的实例，但不调 __init__
        # 测试里为了隔离外部依赖（Redis 连接）偶尔会这么干。
        # 正常应该让 RedisShortTermMemoryStore 支持依赖注入，测试传个假 connection 进去，就不用绕 __init__ 了
        store = RedisShortTermMemoryStore.__new__(RedisShortTermMemoryStore)
        store.privacy = PrivacySanitizer()

        # json.loads = JSON 字符串 → Python 字典
        payload = json.loads(store._serialize("user", "电话 13800138000 邮箱 a@example.com"))

        self.assertNotIn("13800138000", payload["content"])
        self.assertNotIn("a@example.com", payload["content"])
        self.assertEqual(payload["content"].count("[已脱敏]"), 2)

    def test_high_risk_signal_uses_hard_guard_before_model(self):
        result = PsychologicalAssessmentService(ExplodingAi()).assess("我不想活了，想结束生命")

        self.assertEqual(result.risk, RiskLevel.HIGH)
        self.assertGreaterEqual(result.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()                                  # 把这个文件当成测试入口，自动发现并运行里面所有 TestCase 类
