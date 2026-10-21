"""动态采样必须补满有效组，并在恢复时重建同一候选序列。"""

from types import SimpleNamespace
import unittest

import torch

from reasoning_forge.data import PromptSample
from reasoning_forge.dynamic_sampling import (
    CandidateStream, informative_group_indices, select_rollout_groups,
)
from reasoning_forge.rewards import dapo_math_reward, extract_answer_line
from reasoning_forge.rollout import RolloutBatch
from reasoning_forge.trainer import prepare_dynamic_window


class FakeEngine:
    def rollout(self, prompts):
        completions = []
        for prompt in prompts:
            completions.extend({"p0": ["0", "0"], "p1": ["1", "0"], "p2": ["1", "1"]}[prompt])
        count = len(completions)
        ids = torch.ones(count, 2, dtype=torch.long)
        return RolloutBatch(
            ids, ids.clone(), ids[:, 1:], torch.ones(count, 1, dtype=torch.long),
            torch.zeros(count, 1), completions, 1, torch.zeros(count, dtype=torch.bool),
        )


class DynamicSamplingTests(unittest.TestCase):
    def test_candidate_stream_restores_from_absolute_cursor(self):
        samples = list(range(5))
        stream = CandidateStream(samples, 19)
        first = stream.take(0, 11)
        resumed = CandidateStream(samples, 19).take(6, 5)
        self.assertEqual(resumed, first[6:])
        self.assertEqual(sorted(first[:5]), samples)
        self.assertEqual(sorted(first[5:10]), samples)

    def test_group_filter_keeps_whole_trajectories(self):
        rewards = torch.tensor([0., 0., 0., 1., 1., 1.])
        self.assertEqual(informative_group_indices(rewards, 2), [1])
        rollout = FakeEngine().rollout(["p0", "p1", "p2"])
        selected = select_rollout_groups(rollout, [1], 2)
        self.assertEqual(selected.completions, ["1", "0"])
        self.assertEqual(selected.input_ids.shape[0], 2)

    def test_refills_to_fixed_effective_group_count(self):
        samples = [PromptSample(f"p{i}", "1", f"{i}") for i in range(3)]
        config = SimpleNamespace(
            seed=7, prompt_batch_size=2, num_generations=2,
            dynamic_sampling_max_candidates_factor=8, train_micro_batch_size=1,
            prompt_format="plain", system_prompt="",
        )
        bundle = SimpleNamespace(tokenizer=None, device=torch.device("cpu"), reference=None)
        prepared, cursor, sampling = prepare_dynamic_window(
            CandidateStream(samples, config.seed), 0, 2, bundle, FakeEngine(),
            lambda completion, answer: float(completion == answer), config,
        )
        self.assertEqual(sum(len(item.samples) for item in prepared), 2)
        self.assertTrue(all(item.rewards.tolist() == [1.0, 0.0] for item in prepared))
        self.assertEqual(cursor, sampling["sampled_prompts"])
        self.assertGreater(sampling["rejected_prompts"], 0)
        self.assertEqual(sampling["sampled_tokens"], 2 * sampling["sampled_prompts"])

    def test_dapo_answer_line_is_separate_from_gsm8k_boxed_format(self):
        text = "Reasoning...\nAnswer: 34\n"
        self.assertEqual(extract_answer_line(text), "34")
        self.assertEqual(dapo_math_reward(text, "34"), 1.0)
        self.assertEqual(dapo_math_reward(text, "35"), 0.0)
        self.assertIsNone(extract_answer_line("Answer: 34\nthen I changed my mind"))


if __name__ == "__main__":
    unittest.main()
