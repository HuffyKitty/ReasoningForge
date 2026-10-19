"""验证训练总装配的平均方式：微批/累积不会悄悄改变不等长样本的权重。"""

import copy
from types import SimpleNamespace
import unittest

import torch

from reasoning_forge.models import compute_token_stats
from reasoning_forge.objectives import group_advantages, grpo_loss
from reasoning_forge.rollout import RolloutBatch
from reasoning_forge.trainer import PreparedRollout, backward_window, window_statistics


class TinyPolicy(torch.nn.Module):
    """可微的词表转移表，避免测试归一化时还要加载 Transformer。"""

    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(10, 10)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        return SimpleNamespace(logits=self.table(input_ids))


def make_window(model):
    items = []
    # 第一批两道题、第二批一道题，且回答长度不同，能暴露“每批直接等权”错误。
    for n, p, r, lengths in ((4, 2, 3, [1, 2, 3, 2]), (2, 3, 2, [2, 1])):
        ids = torch.randint(0, 10, (n, p + r))
        mask = torch.arange(r).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
        attention = torch.cat([torch.ones(n, p), mask], dim=1).long()
        with torch.no_grad():
            logps, _ = compute_token_stats(model, ids, attention, p)
        old = logps - torch.linspace(-0.5, 0.5, n).unsqueeze(1)
        reference = logps + 0.2
        rollout = RolloutBatch(ids, attention, ids[:, p:], mask.long(), old, [""] * n,
                               p, torch.zeros(n, dtype=torch.bool))
        rewards = torch.tensor([0., 1.] * (n // 2))
        samples = [object() for _ in range(n // 2)]
        items.append(PreparedRollout(samples, rollout, rewards, group_advantages(rewards, 2), reference))
    return items


class TrainingTests(unittest.TestCase):
    def test_microbatch_accumulation_matches_whole_window_loss_and_gradient(self):
        for reduction in ("sequence", "token"):
            with self.subTest(reduction=reduction):
                torch.manual_seed(9)
                actual = TinyPolicy()
                expected = copy.deepcopy(actual)
                prepared = make_window(actual)
                statistics = window_statistics(prepared, 2)
                config = SimpleNamespace(train_micro_batch_size=1, beta=0.05,
                    clip_low=0.2, clip_high=0.28, loss_reduction=reduction)
                metrics = backward_window(actual, prepared, config, statistics)
                whole_loss = torch.tensor(0.)
                ratio_parts = []
                positive_parts = []
                positive_clipped_parts = []
                for item in prepared:
                    rollout = item.rollout
                    logps, _ = compute_token_stats(expected, rollout.input_ids,
                        rollout.attention_mask, rollout.prompt_width)
                    result = grpo_loss(logps, rollout.old_logps, item.advantages,
                        rollout.completion_mask, ref_logps=item.reference_logps,
                        beta=0.05, clip_low=0.2, clip_high=0.28, reduction=reduction)
                    ratio_parts.append(result.diagnostics["ratio"])
                    positive_parts.append(result.diagnostics["positive_advantage"])
                    positive_clipped_parts.append(result.diagnostics["positive_clipped"])
                    weight = (logps.size(0) / statistics["training_sequences"] if reduction == "sequence"
                              else rollout.completion_mask.sum().item() / statistics["training_tokens"])
                    whole_loss = whole_loss + result.loss * weight
                whole_loss.backward()
                self.assertAlmostEqual(metrics["loss"], whole_loss.item(), places=6)
                ratios = torch.cat(ratio_parts)
                self.assertAlmostEqual(metrics["ratio_p99"], torch.quantile(ratios, 0.99).item(), places=6)
                positive = torch.cat(positive_parts)
                clipped = torch.cat(positive_clipped_parts)
                self.assertAlmostEqual(metrics["positive_clip_fraction"],
                                       clipped[positive].float().mean().item(), places=6)
                torch.testing.assert_close(actual.table.weight.grad, expected.table.weight.grad)
                self.assertEqual(statistics["training_prompts"], 3)
                self.assertEqual(statistics["training_sequences"], 6)
                self.assertEqual(statistics["training_tokens"], 11)


if __name__ == "__main__":
    unittest.main()
