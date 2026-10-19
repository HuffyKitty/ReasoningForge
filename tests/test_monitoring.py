"""用小组轨迹验证监控指标的分母、无答案、连续奖励和题目裁剪。"""

import unittest

import torch

from reasoning_forge.monitoring import prompt_truncation_diagnostics, rollout_diagnostics


class WordTokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs == {"add_special_tokens": False, "padding": False, "truncation": False}
        return {"input_ids": [list(range(len(text.split()))) for text in texts]}


class MonitoringTests(unittest.TestCase):
    def test_group_parse_repetition_lengths_and_truncation_use_correct_denominators(self):
        # 三题、每题两个回答；对应全错、全对、混合三个组。
        completions = ["no answer", "different text", r"\boxed{2}", r"\boxed{2}",
                       r"\boxed{5}", r"\boxed{6}"]
        rewards = torch.tensor([0., 0., 1., 1., 1., 0.])
        mask = torch.arange(6).unsqueeze(0) < torch.tensor([1, 2, 3, 4, 5, 6]).unsqueeze(1)
        result = rollout_diagnostics(
            rewards, completions, mask.long(), torch.tensor([1, 0, 0, 0, 1, 0]), 2
        )
        self.assertAlmostEqual(result["all_wrong_group_fraction"], 1 / 3)
        self.assertAlmostEqual(result["all_correct_group_fraction"], 1 / 3)
        self.assertAlmostEqual(result["mixed_group_fraction"], 1 / 3)
        self.assertEqual(result["parsed_completions"], 4)
        self.assertAlmostEqual(result["answer_extraction_rate"], 4 / 6)
        self.assertAlmostEqual(result["answer_accuracy_given_extracted"], 3 / 4)
        self.assertAlmostEqual(result["final_answer_diversity"], 0.5)
        self.assertAlmostEqual(result["response_repetition_fraction"], 1 / 6)
        self.assertEqual([result[f"response_length_p{p}"] for p in (50, 90, 99)], [3., 6., 6.])
        self.assertEqual(result["correct_response_length"], 4.)
        self.assertEqual(result["incorrect_response_length"], 3.)
        self.assertAlmostEqual(result["truncated_unparseable_fraction"], 1 / 6)
        self.assertAlmostEqual(result["truncated_answer_missing_rate"], 1 / 2)
        # 多个 rollout 的回答宽度不同，合并窗口时可只拼接一维有效长度。
        by_lengths = rollout_diagnostics(
            rewards, completions, torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([1, 0, 0, 0, 1, 0]), 2
        )
        self.assertEqual(by_lengths, result)

    def test_missing_denominators_and_continuous_rewards_are_explicit(self):
        result = rollout_diagnostics(
            torch.tensor([0., 0.]), ["no", "none"], torch.ones(2, 1), torch.zeros(2), 2
        )
        self.assertEqual(result["answer_extraction_rate"], 0)
        self.assertIsNone(result["answer_accuracy_given_extracted"])
        self.assertIsNone(result["truncated_answer_missing_rate"])
        continuous = rollout_diagnostics(
            torch.tensor([0.5, 0.75]), [r"\boxed{1}", r"\boxed{2}"],
            torch.ones(2, 1), torch.zeros(2), 2
        )
        self.assertIsNone(continuous["all_wrong_group_fraction"])
        self.assertIsNone(continuous["answer_accuracy_given_extracted"])
        self.assertEqual(continuous["answer_extraction_rate"], 1)

    def test_invalid_shapes_or_values_raise_instead_of_silent_miscount(self):
        args = (torch.tensor([0., 1.]), ["a", "b"], torch.ones(2, 1), torch.zeros(2), 2)
        with self.assertRaisesRegex(ValueError, "B\*G"):
            rollout_diagnostics(*args[:4], 3)
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            rollout_diagnostics(torch.tensor([float("nan"), 1.]), *args[1:])
        with self.assertRaisesRegex(ValueError, "0/1"):
            rollout_diagnostics(args[0], args[1], torch.tensor([[2], [1]]), args[3], 2)
        with self.assertRaisesRegex(ValueError, "有限整数"):
            rollout_diagnostics(args[0], args[1], torch.tensor([1., 1.5]), args[3], 2)

    def test_prompt_truncation_matches_rollout_rule_and_counts_unique_prompts(self):
        result = prompt_truncation_diagnostics(
            ["one two", "one two three four", "one two three four five six seven"],
            WordTokenizer(), 4
        )
        self.assertAlmostEqual(result["prompt_truncation_fraction"], 1 / 3)
        self.assertEqual(result["prompt_truncated_count"], 1)
        self.assertEqual(result["prompt_total_count"], 3)
        self.assertEqual(result["prompt_length_p90"], 7)
        self.assertEqual(result["prompt_length_max"], 7)
        self.assertEqual(result["prompt_discarded_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
