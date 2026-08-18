# re 是 Python 标准库的正则表达式模块
import re

# re.compile() — 把一段正则表达式字符串"编译"成一个可反复使用的模式对象。
# 直接用字符串也能匹配，但 compile 之后性能更好，特别是同一个规则要用很多次的时候
class PrivacySanitizer:

    # r 表示原始字符串（raw string）——告诉 Python 不要处理反斜杠转义
    # \b — 单词边界（word boundary）。匹配"单词字符和非单词字符的交界处"
    # \d — 数字字符（digit）。但只有完整单词才会匹配
    # \w — 单词字符（word char）。相当于 [a-zA-Z0-9_]——字母、数字、下划线
    patterns = [
        re.compile(r"1[3-9]\d{9}"),                                             # 手机号：1 开头，第二位 3-9，后面跟 9 个数字
        re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),                            # 邮箱：xxx@xxx.xxx 格式
        re.compile(r"\b\d{17}[\dXx]\b"),                                        # 身份证号：17 位数字 + 最后一位数字或 X
    ]

    def sanitize(self, text: str) -> str:
        sanitized = text or ""
        for pattern in self.patterns:

            # pattern.sub("[已脱敏]", sanitized) 是正则的替换方法——找到所有匹配这个模式的部分，全部替换成 [已脱敏]
            sanitized = pattern.sub("[已脱敏]", sanitized)
        return sanitized

