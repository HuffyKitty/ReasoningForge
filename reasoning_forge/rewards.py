"""数学 RLVR 的规则奖励，不需要单独加载 Reward Model。

这里只支持 GSM8K、简单数值等价和严格字符串匹配，并非完整 MATH 符号验证器。
例如 1/2 与 0.5 等价，但 x+x 与 2*x 不做符号推导；复杂 MATH 可另接 math-verify。
奖励接口刻意保持简单：math_reward(completion, answer) -> float，便于替换实验。
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

    from reasoning_forge.data import PromptSample


_BOX_START = re.compile(r"\\boxed\s*\{")
_HASH_ANSWER = re.compile(r"(?m)^[ \t]*####[ \t]*([^\r\n]*)")
_ANSWER_LINE = re.compile(r"(?mi)^[ \t]*Answer[ \t]*:[ \t]*([^\r\n]*)")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_COMMA_NUMBER = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?\Z")


def extract_final_answer(text: str) -> Optional[str]:
    """只认显式最终答案标记，绝不从正文里挑一个数字当答案。

    用括号深度解析 boxed，避免正则把 \\boxed{\\frac{1}{2}} 提前截断。
    同时有 boxed 和 #### 时使用更靠后的标记；最后标记缺失/损坏时不给奖励。
    """

    if not isinstance(text, str):
        raise TypeError("completion 必须是字符串。")
    candidates: list[tuple[int, Optional[str]]] = []
    for match in _BOX_START.finditer(text):
        depth = 1
        start = match.end()
        answer = None
        for position in range(start, len(text)):
            char = text[position]
            # LaTeX 的 \{ 和 \} 表示字面括号，不改变结构深度。
            slash_count = 0
            previous = position - 1
            while previous >= start and text[previous] == "\\":
                slash_count += 1
                previous -= 1
            if slash_count % 2:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answer = text[start:position].strip() or None
                    break
        candidates.append((match.start(), answer))
    for match in _HASH_ANSWER.finditer(text):
        candidates.append((match.start(), match.group(1).strip() or None))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def extract_answer_line(text: str) -> Optional[str]:
    """DAPO-Math 官方提示要求独立的 `Answer: <integer>` 最终答案行。"""
    if not isinstance(text, str):
        raise TypeError("completion 必须是字符串。")
    matches = list(_ANSWER_LINE.finditer(text))
    if not matches or text[matches[-1].end():].strip():
        return None
    return matches[-1].group(1).strip() or None


def extractor_for_format(answer_format: str):
    if answer_format == "boxed":
        return extract_final_answer
    if answer_format == "answer_line":
        return extract_answer_line
    raise ValueError(f"未知答案格式：{answer_format}")


def normalize_answer(answer: str) -> str:
    """清理展示格式，保留有意义的符号；不删除单位，不猜测多个候选答案。"""

    if not isinstance(answer, str):
        raise TypeError("answer 必须是字符串。")
    value = answer.strip().replace("−", "-").replace("＋", "+")
    wrappers = (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    for left, right in wrappers:
        if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
            value = value[len(left):len(value) - len(right)].strip()
            break
    value = value.replace(r"\left", "").replace(r"\right", "")
    # 数学排版空格可以移除，普通词之间的空格仍保留，防止把 '1 2' 变成 12。
    value = value.replace(r"\,", "").replace(r"\!", "").replace(r"\;", " ")
    return re.sub(r"\s+", " ", value).strip()


def _decimal_fraction(value: str) -> Optional[Fraction]:
    """Decimal -> Fraction 保留十进制精确值，避免 float 的二进制误差。"""

    value = value.strip()
    if "," in value:
        if not _COMMA_NUMBER.fullmatch(value):
            return None  # '1,2' 不能被当作 12；逗号必须符合千分位格式。
        value = value.replace(",", "")
    if not _NUMBER.fullmatch(value):
        return None
    # 限制异常长度/指数，避免不可信生成文本制造巨型整数运算。
    exponent = re.search(r"[eE]([+-]?\d+)$", value)
    if len(value) > 200 or (exponent and (len(exponent.group(1)) > 5 or abs(int(exponent.group(1))) > 1000)):
        return None
    try:
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError, OverflowError):
        return None


def _as_number(value: str) -> Optional[Fraction]:
    """只接受单一完整数值；不执行 eval，也不从解释句中提取数字。"""

    if value.startswith("(") and value.endswith(")") and value.count("(") == value.count(")") == 1:
        value = value[1:-1].strip()
    direct = _decimal_fraction(value)
    if direct is not None:
        return direct
    latex = re.fullmatch(r"([+-]?)\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", value)
    if latex:
        numerator = _decimal_fraction(latex.group(2))
        denominator = _decimal_fraction(latex.group(3))
        if numerator is not None and denominator not in (None, 0):
            result = numerator / denominator
            return -result if latex.group(1) == "-" else result
        return None
    fraction = re.fullmatch(r"([^/]+)\s*/\s*([^/]+)", value)
    if fraction:
        numerator = _decimal_fraction(fraction.group(1))
        denominator = _decimal_fraction(fraction.group(2))
        if numerator is not None and denominator not in (None, 0):
            return numerator / denominator
    return None


def math_reward(completion: str, answer: str) -> float:
    """正确最终答案为 1，否则为 0；不额外奖励长度、思考标签或答案出现次数。"""

    return _reward_from_extracted(extract_final_answer(completion), answer)


def dapo_math_reward(completion: str, answer: str) -> float:
    """DAPO-Math 的整数答案和原始 `Answer:` 输出格式。"""
    return _reward_from_extracted(extract_answer_line(completion), answer)


def _reward_from_extracted(extracted: Optional[str], answer: str) -> float:
    if extracted is None:
        return 0.0
    predicted = normalize_answer(extracted)
    expected = normalize_answer(answer)
    if not predicted or not expected:
        return 0.0
    predicted_number = _as_number(predicted)
    expected_number = _as_number(expected)
    if expected_number is not None:
        return float(predicted_number is not None and predicted_number == expected_number)
    # 字符串兜底只认严格相等，不擅自认为两个符号表达式在数学上等价。
    return float(predicted == expected)


def score_completions(
    samples: list[PromptSample], completions: list[str], num_generations: int
) -> torch.Tensor:
    """每题连续 G 个回答，与 GRPO 的 rewards.view(B, G) 保持完全一致。

    [题0-回答0, 题0-回答1, 题1-回答0, 题1-回答1]，不能打乱成按回答编号排列。
    """

    import torch

    if isinstance(num_generations, bool) or not isinstance(num_generations, int) or num_generations < 1:
        raise ValueError("num_generations 必须是正整数。")
    if len(completions) != len(samples) * num_generations:
        raise ValueError("completions 长度必须等于 len(samples) * num_generations。")
    scores = [math_reward(text, samples[index // num_generations].answer) for index, text in enumerate(completions)]
    return torch.tensor(scores, dtype=torch.float32, device="cpu")
