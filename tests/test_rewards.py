"""无需联网/下载权重，验证奖励边界，防止错误 verifier 掩盖算法结果。"""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from reasoning_forge.data import PromptSample, load_jsonl, render_prompt
from reasoning_forge.rewards import extract_final_answer, math_reward, normalize_answer, score_completions


class RewardTests(unittest.TestCase):
    def test_nested_latex_and_last_answer(self):
        self.assertEqual(extract_final_answer(r"先 \boxed{3}，最终 \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertEqual(extract_final_answer("#### 2\n又算了一次\n#### 3"), "3")
        self.assertEqual(extract_final_answer("\\boxed{2}\n#### 3"), "3")

    def test_malformed_last_answer_is_rejected(self):
        self.assertIsNone(extract_final_answer(r"\boxed{2} 然后 \boxed{3"))
        self.assertIsNone(extract_final_answer(r"\boxed{}"))
        self.assertIsNone(extract_final_answer("####   "))
        self.assertIsNone(extract_final_answer("####\n42"))

    def test_no_number_guessing_or_reward_hacking(self):
        for text in ("计算中出现 42，最终我不会做。", "42", "#### 42 或 43", r"\boxed{42, 43}"):
            with self.subTest(text=text):
                self.assertEqual(math_reward(text, "42"), 0.0)
        self.assertEqual(math_reward(r"正确答案是 42，但最终 \boxed{43}", "42"), 0.0)

    def test_exact_numeric_equivalence(self):
        examples = [
            (r"\boxed{0}", "0"),
            (r"\boxed{-3}", "−3"),
            (r"\boxed{1,200}", "1200"),
            (r"\boxed{1.2e3}", "1200"),
            (r"\boxed{\frac{1}{2}}", "0.5"),
            (r"\boxed{-\frac{2}{4}}", "-0.5"),
            (r"\boxed{1 / 4}", "0.25"),
            (r"\boxed{(-3)}", "-3"),
            (r"\boxed{0.10000000000000000001}", "0.10000000000000000001"),
        ]
        for completion, answer in examples:
            with self.subTest(completion=completion):
                self.assertEqual(math_reward(completion, answer), 1.0)
        self.assertEqual(math_reward(r"\boxed{0.10000000000000000001}", "0.1"), 0.0)

    def test_invalid_numeric_and_unsupported_symbolic(self):
        for candidate in ("1/0", "NaN", "Infinity", "1,2", "1 2", "__import__('os')"):
            with self.subTest(candidate=candidate):
                self.assertEqual(math_reward("#### " + candidate, "12"), 0.0)
        self.assertEqual(math_reward(r"\boxed{2*x}", "x+x"), 0.0)
        self.assertEqual(math_reward(r"\boxed{\sqrt{2}}", r"\sqrt{2}"), 1.0)

    def test_normalize_display(self):
        self.assertEqual(normalize_answer(r" $\left(−3\right)$ "), "(-3)")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "奖励排列测试需要 PyTorch")
    def test_reward_order_matches_prompt_groups(self):
        import torch

        samples = [PromptSample("A", "2", "a"), PromptSample("B", "3", "b")]
        result = score_completions(samples, ["#### 2", "#### 3", "#### 3", "#### 2"], 2)
        self.assertEqual(result.tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, "cpu")
        with self.assertRaises(ValueError):
            score_completions(samples, ["#### 2"], 2)


class DataTests(unittest.TestCase):
    def test_jsonl_validation_and_question_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "examples.jsonl"
            path.write_text(json.dumps({"question": "1+1?", "answer": "2", "dataset": "gsm8k"}), encoding="utf-8")
            self.assertEqual(load_jsonl(path), [PromptSample("1+1?", "2", "1", "gsm8k")])
            path.write_text('{"prompt":"1+1?","answer":""}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "answer"):
                load_jsonl(path)
            path.write_text("\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(path)

    def test_chat_template_and_plain_prompt(self):
        class FakeTokenizer:
            chat_template = "the model owns this template"

            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return "model-rendered-prompt"

        tokenizer = FakeTokenizer()
        sample = PromptSample("question", "secret-answer", "a")
        config = SimpleNamespace(prompt_format="chat", system_prompt="system", chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(render_prompt(sample, tokenizer, config), "model-rendered-prompt")
        self.assertEqual(tokenizer.messages, [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}])
        self.assertEqual(tokenizer.kwargs, {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})
        self.assertEqual(render_prompt(sample, None, {"prompt_format": "plain", "system_prompt": "system"}), "system\n\nquestion")
        with self.assertRaisesRegex(ValueError, "chat_template"):
            render_prompt(sample, None, {"prompt_format": "chat"})


if __name__ == "__main__":
    unittest.main()
