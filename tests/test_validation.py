"""开发评估必须固定题目与随机流，且只报告真正的二元正确性。"""

import random
from types import SimpleNamespace
import unittest

import torch

from reasoning_forge.config import TrainConfig
from reasoning_forge.data import PromptSample
from reasoning_forge.rewards import math_reward
from reasoning_forge.validation import evaluate_dev


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        def encode(value):
            return [ord(char) % 16 for char in value]
        return {"input_ids": [encode(value) for value in text] if isinstance(text, list) else encode(text)}


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(16, 16)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        return SimpleNamespace(logits=self.table(input_ids))


class FakeEngine:
    def rollout(self, prompts, num_generations):
        # 若评估没有隔离 RNG，这个调用会改变下一次训练采样。
        torch.rand(1)
        completions = [r"\boxed{2}" if index % num_generations == 0 else "unfinished"
                       for _ in prompts for index in range(num_generations)]
        return SimpleNamespace(
            completions=completions,
            completion_mask=torch.ones(len(completions), 2, dtype=torch.long),
            truncated=torch.zeros(len(completions), dtype=torch.bool),
        )


class ValidationTests(unittest.TestCase):
    def test_dev_pass_and_probe_are_fixed_without_moving_training_rng(self):
        torch.manual_seed(42)
        random.seed(42)
        model = TinyPolicy()
        bundle = SimpleNamespace(actor=model, tokenizer=TinyTokenizer(), device=torch.device("cpu"))
        config = TrainConfig(prompt_format="plain", system_prompt="", eval_num_samples=2,
                             eval_limit=2, probe_prompts=1, max_prompt_length=8)
        samples = [PromptSample("1+1?", "2", "one", "gsm8k"),
                   PromptSample("2-0?", "2", "two", "gsm8k")]
        torch_state = torch.get_rng_state().clone()
        python_state = random.getstate()
        result, records = evaluate_dev(bundle, FakeEngine(), config, samples, math_reward, step=5)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(model.training)
        self.assertEqual(result["step"], 5)
        self.assertEqual(result["overall"]["pass@1"], 0.5)
        self.assertEqual(result["overall"]["pass@2"], 1.0)
        self.assertEqual(result["overall"]["format_rate"], 0.5)
        self.assertEqual(len(records), 2)
        self.assertGreater(result["probe"]["probe_entropy_tokens"], 0)

    def test_shaped_reward_is_rejected_as_accuracy(self):
        bundle = SimpleNamespace(actor=TinyPolicy(), tokenizer=TinyTokenizer(), device=torch.device("cpu"))
        config = TrainConfig(prompt_format="plain", system_prompt="", eval_num_samples=2,
                             eval_limit=1, probe_prompts=1)
        with self.assertRaisesRegex(ValueError, "二元正确性"):
            evaluate_dev(bundle, FakeEngine(), config,
                         [PromptSample("1+1?", "2", "one", "gsm8k")],
                         lambda completion, answer: 0.5, step=1)


if __name__ == "__main__":
    unittest.main()
