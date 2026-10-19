"""长度预检的离线测试：不加载 Hugging Face 权重。"""

import unittest

from scripts.check_prompt_lengths import summarize_prompt_lengths


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, prompts, **kwargs):
        self.calls.append((list(prompts), kwargs))
        return {"input_ids": [list(range(int(prompt))) for prompt in prompts]}


class PromptLengthsTest(unittest.TestCase):
    def test_counts_only_prompts_longer_than_limit_without_truncating_first(self):
        tokenizer = FakeTokenizer()
        report = summarize_prompt_lengths(["1", "2", "3", "4"], tokenizer, 2, batch_size=2)
        self.assertEqual(report["count"], 4)
        self.assertEqual(report["max"], 4)
        self.assertEqual(report["p50"], 2)
        self.assertEqual(report["p90"], 4)
        self.assertEqual(report["p99"], 4)
        self.assertEqual(report["truncated_count"], 2)
        self.assertEqual(report["truncated_fraction"], 0.5)
        self.assertEqual([len(call[0]) for call in tokenizer.calls], [2, 2])
        self.assertTrue(all(call[1] == {
            "add_special_tokens": False, "truncation": False, "padding": False,
        } for call in tokenizer.calls))

    def test_limit_is_inclusive_and_empty_prompts_are_rejected(self):
        report = summarize_prompt_lengths(["2", "2"], FakeTokenizer(), 2)
        self.assertEqual(report["truncated_count"], 0)
        with self.assertRaisesRegex(ValueError, "至少需要"):
            summarize_prompt_lengths([], FakeTokenizer(), 2)


if __name__ == "__main__":
    unittest.main()
