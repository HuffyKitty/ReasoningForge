"""用人工构造的小矩阵核验 GRPO 数学；不下载模型、不使用 GPU。

运行：python -m unittest discover -s tests -v
这些测试关注真正影响训练行为的边界：裁剪、mask、长度归一化和梯度。
"""

import math
import unittest

import torch

from reasoning_forge.objectives import group_advantages, grpo_loss


class GRPOObjectivesTest(unittest.TestCase):
    def test_group_relative_advantages_and_zero_variance(self):
        # 前两个回答属于问题 A，后两个属于问题 B；B 的奖励相等，没有学习信号。
        rewards = torch.tensor([1.0, 3.0, 5.0, 5.0])
        advantages = group_advantages(rewards, num_generations=2)
        self.assertEqual(tuple(advantages.shape), (4,))
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertAlmostEqual(advantages[0].item(), -1.0, places=3)
        self.assertAlmostEqual(advantages[1].item(), 1.0, places=3)
        torch.testing.assert_close(advantages[2:], torch.zeros(2))

    def test_positive_and_negative_advantage_are_clipped(self):
        # 好回答的概率增幅超过 1.2，坏回答的比例低于 0.8：都进入裁剪平坦区。
        logps = torch.tensor([[math.log(1.5)], [math.log(0.5)]], requires_grad=True)
        result = grpo_loss(
            logps, torch.zeros_like(logps), torch.tensor([1.0, -1.0]),
            torch.ones_like(logps), clip_low=0.2, clip_high=0.2,
        )
        self.assertAlmostEqual(result.loss.item(), (-1.2 + 0.8) / 2, places=6)
        result.loss.backward()
        torch.testing.assert_close(logps.grad, torch.zeros_like(logps))

    def test_policy_diagnostics_use_only_valid_tokens_and_sign_specific_denominators(self):
        # Padding 故意放 NaN：它既不能影响 loss，也不能混入分位数或裁剪率。
        ratios = torch.tensor([
            [1.1, 1.5, float("nan")],
            [0.5, 0.9, float("nan")],
            [1.0, 1.0, 1.0],
        ])
        mask = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1]])
        result = grpo_loss(
            ratios.log(), torch.zeros_like(ratios), torch.tensor([1.0, -1.0, 0.0]), mask,
            clip_low=0.2, clip_high=0.2,
        )
        self.assertAlmostEqual(result.metrics["clip_fraction"], 2 / 7, places=6)
        self.assertAlmostEqual(result.metrics["positive_clip_fraction"], 0.5, places=6)
        self.assertAlmostEqual(result.metrics["negative_clip_fraction"], 0.5, places=6)
        self.assertAlmostEqual(result.metrics["ratio_p05"], 0.62, places=5)
        self.assertAlmostEqual(result.metrics["ratio_p50"], 1.0, places=5)
        self.assertAlmostEqual(result.metrics["ratio_p95"], 1.38, places=5)
        self.assertAlmostEqual(result.metrics["ratio_p99"], 1.476, places=5)
        self.assertAlmostEqual(
            result.metrics["log_ratio_p99"],
            math.log(1.1) + 0.94 * (math.log(1.5) - math.log(1.1)), places=5,
        )
        torch.testing.assert_close(result.diagnostics["ratio"], torch.tensor([1.1, 1.5, 0.5, 0.9, 1, 1, 1]))
        torch.testing.assert_close(result.diagnostics["positive_advantage"], torch.tensor([1, 1, 0, 0, 0, 0, 0], dtype=torch.bool))
        torch.testing.assert_close(result.diagnostics["negative_advantage"], torch.tensor([0, 0, 1, 1, 0, 0, 0], dtype=torch.bool))
        torch.testing.assert_close(result.diagnostics["positive_clipped"], torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype=torch.bool))
        torch.testing.assert_close(result.diagnostics["negative_clipped"], torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype=torch.bool))
        self.assertEqual(result.diagnostics["ratio"].device.type, "cpu")
        self.assertFalse(result.diagnostics["ratio"].requires_grad)

    def test_sign_specific_clip_fraction_is_zero_when_no_eligible_advantage(self):
        # 全对或全错的 GRPO 组优势为零，不能拿所有 token 作正/负裁剪率的分母。
        logps = torch.tensor([[math.log(2.0), math.log(0.5)]])
        result = grpo_loss(logps, torch.zeros_like(logps), torch.zeros(1), torch.ones_like(logps))
        self.assertEqual(result.metrics["positive_clip_fraction"], 0.0)
        self.assertEqual(result.metrics["negative_clip_fraction"], 0.0)
        self.assertEqual(result.metrics["clip_fraction"], 0.0)
        self.assertFalse(result.diagnostics["positive_advantage"].any())
        self.assertFalse(result.diagnostics["negative_advantage"].any())

    def test_unclipped_gradient_pushes_good_up_and_bad_down(self):
        logps = torch.tensor([[math.log(1.1)], [math.log(0.9)]], requires_grad=True)
        result = grpo_loss(
            logps, torch.zeros_like(logps), torch.tensor([1.0, -1.0]),
            torch.ones_like(logps),
        )
        result.loss.backward()
        # 梯度下降减去梯度：好回答 logp 增大，坏回答 logp 减小。
        torch.testing.assert_close(logps.grad, torch.tensor([[-0.55], [0.45]]))

    def test_masked_tokens_do_not_change_loss_or_gradients(self):
        mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        advantages = torch.tensor([1.0, -1.0])
        baseline = torch.zeros(2, 3, requires_grad=True)
        changed = torch.tensor([[0.0, -3.0, 2.0], [0.0, 0.0, 4.0]], requires_grad=True)
        first = grpo_loss(baseline, torch.zeros(2, 3), advantages, mask)
        second = grpo_loss(changed, torch.zeros(2, 3), advantages, mask)
        torch.testing.assert_close(first.loss, second.loss)
        second.loss.backward()
        torch.testing.assert_close(changed.grad[mask == 0], torch.zeros(3))
        self.assertTrue(torch.isfinite(changed.grad).all())

    def test_sequence_and_token_reductions_weight_lengths_differently(self):
        # 第一条好回答只有 1 token，第二条坏回答有 3 token。
        # sequence：两条回答各占一半；token：四个有效 token 各占四分之一。
        logps = torch.zeros(2, 3)
        mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        advantages = torch.tensor([1.0, -1.0])
        sequence = grpo_loss(logps, logps, advantages, mask, reduction="sequence")
        token = grpo_loss(logps, logps, advantages, mask, reduction="token")
        self.assertAlmostEqual(sequence.loss.item(), 0.0, places=6)
        self.assertAlmostEqual(token.loss.item(), 0.5, places=6)

    def test_reference_kl_value_and_gradient(self):
        # k3 KL 估计：exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1。
        logps = torch.tensor([[math.log(2.0)]], requires_grad=True)
        result = grpo_loss(
            logps, torch.zeros_like(logps), torch.zeros(1), torch.ones_like(logps),
            ref_logps=torch.zeros_like(logps), beta=0.3,
        )
        expected = 0.3 * (0.5 + math.log(2.0) - 1.0)
        self.assertAlmostEqual(result.loss.item(), expected, places=6)
        result.loss.backward()
        self.assertAlmostEqual(logps.grad.item(), 0.15, places=6)

    def test_all_masked_batch_is_rejected(self):
        # 无有效回答 token 不是可训练样本，应明确报错，避免静默除零/空更新。
        logps = torch.zeros(2, 3, requires_grad=True)
        with self.assertRaises(ValueError):
            grpo_loss(logps, torch.zeros_like(logps), torch.ones(2), torch.zeros_like(logps))


if __name__ == "__main__":
    unittest.main()
