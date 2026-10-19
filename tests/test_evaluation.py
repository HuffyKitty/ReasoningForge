"""检验 pass@k 和分层聚合；仅使用数学函数，不加载模型。"""

import unittest

from evaluate import baseline_difficulty_map, binary_reward, estimate_pass_at_k, summarize_results


class EvaluationTest(unittest.TestCase):
    def test_baseline_difficulty_requires_same_data_and_marks_strata(self):
        base = {"data": {"sha256": "fixed"}, "samples": [
            {"dataset": "gsm8k", "id": "a", "prompt": "A?", "rewards": [0, 0]},
            {"dataset": "gsm8k", "id": "b", "prompt": "B?", "rewards": [0, 1]},
        ]}
        with self.assertRaisesRegex(ValueError, "SHA256"):
            baseline_difficulty_map(base, "different")
        mapping = baseline_difficulty_map(base, "fixed")
        self.assertEqual(mapping[("gsm8k", "a")][0], "unsolved")
        self.assertEqual(mapping[("gsm8k", "b")][0], "mixed")
        records = [
            {"dataset": "gsm8k", "rewards": [0, 1], "lengths": [2, 2],
             "truncated": [False, False], "baseline_difficulty": "unsolved"},
            {"dataset": "gsm8k", "rewards": [1, 1], "lengths": [2, 2],
             "truncated": [False, False], "baseline_difficulty": "mixed"},
        ]
        result = summarize_results(records, [1, 2])
        self.assertEqual(result["by_baseline_difficulty"]["unsolved"]["pass@1"], 0.5)
        self.assertEqual(result["by_baseline_difficulty"]["mixed"]["pass@1"], 1.0)

    def test_answer_format_lengths_and_prompt_length_strata(self):
        records = [
            {"dataset": "gsm8k", "prompt_tokens": 100, "rewards": [1, 0],
             "lengths": [5, 10], "truncated": [False, True],
             "completions": [r"Done \boxed{2}", "unfinished"]},
            {"dataset": "gsm8k", "prompt_tokens": 300, "rewards": [0, 0],
             "lengths": [20, 30], "truncated": [True, False],
             "completions": [r"Wrong \boxed{9}", "unfinished"]},
        ]
        result = summarize_results(records, [1, 2], max_prompt_length=256)
        overall = result["overall"]
        self.assertEqual(overall["format_rate"], 0.5)
        self.assertEqual(overall["answer_accuracy_given_extracted"], 0.5)
        self.assertEqual(overall["truncated_unparseable_fraction"], 0.25)
        self.assertEqual(overall["response_length_p90"], 30)
        self.assertEqual(overall["correct_response_length"], 5)
        self.assertEqual(overall["prompt_truncation_fraction"], 0.5)
        self.assertEqual(result["by_prompt_length"]["0-128"]["pass@1"], 0.5)
        self.assertEqual(result["by_prompt_length"]["257+"]["pass@1"], 0.0)

    def test_pass_at_one_is_mean_success_rate(self):
        self.assertAlmostEqual(estimate_pass_at_k(8, 2, 1), 0.25)

    def test_pass_at_k_uses_at_least_one_success(self):
        # 8 个回答有 2 个正确，从中取 2 个：1-C(6,2)/C(8,2)=13/28。
        self.assertAlmostEqual(estimate_pass_at_k(8, 2, 2), 13 / 28)
        self.assertEqual(estimate_pass_at_k(8, 2, 8), 1.0)

    def test_all_wrong_and_all_correct(self):
        self.assertEqual(estimate_pass_at_k(8, 0, 8), 0.0)
        self.assertEqual(estimate_pass_at_k(8, 8, 1), 1.0)

    def test_invalid_pass_at_k_arguments_are_rejected(self):
        for arguments in ((0, 0, 1), (8, -1, 1), (8, 9, 1), (8, 2, 0), (8, 2, 9), (8.0, 2, 1), (True, 0, 1)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                estimate_pass_at_k(*arguments)

    def test_continuous_rewards_are_not_accuracy(self):
        for reward in (0.5, -1.0, 1 - 1e-9, float("nan"), float("inf")):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                binary_reward(reward)
        self.assertEqual(binary_reward(0), 0.0)
        self.assertEqual(binary_reward(1), 1.0)

    def test_aggregation_weights_prompts_equally_and_splits_datasets(self):
        # 两题回答数不同。pass@1 按题平均为 0.5，而非按所有回答算 2/6。
        records = [
            {"dataset": "easy", "rewards": [1, 1], "lengths": [2, 4], "truncated": [False, False]},
            {"dataset": "hard", "rewards": [0, 0, 0, 0], "lengths": [1, 2, 3, 4], "truncated": [False, True, False, True]},
        ]
        result = summarize_results(records, [1, 2])
        overall = result["overall"]
        self.assertEqual(overall["prompts"], 2)
        self.assertEqual(overall["responses"], 6)
        self.assertAlmostEqual(overall["pass@1"], 0.5)
        self.assertAlmostEqual(overall["pass@2"], 0.5)
        self.assertAlmostEqual(overall["mean_response_tokens"], 16 / 6)
        self.assertAlmostEqual(overall["truncated_fraction"], 2 / 6)
        self.assertEqual(result["by_dataset"]["easy"]["pass@1"], 1.0)
        self.assertEqual(result["by_dataset"]["hard"]["pass@1"], 0.0)

    def test_empty_or_misaligned_results_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_results([], [1])
        with self.assertRaises(ValueError):
            summarize_results([{"dataset": "a", "rewards": [1], "lengths": [], "truncated": [False]}], [1])
        with self.assertRaises(ValueError):
            summarize_results([{"dataset": "a", "rewards": [1], "lengths": [2], "truncated": [False]}], [])


if __name__ == "__main__":
    unittest.main()
