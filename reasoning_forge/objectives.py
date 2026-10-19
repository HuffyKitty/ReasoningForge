"""GRPO 数学核心：复用 MiniMind 的组内优势 → ratio → clipping → mask 主线。

本文件不认识 Qwen/Tokenizer/LoRA；它只接收已经对齐的 [N,R] 矩阵。
N=B*G 为回答总数，R 为当前 batch 的回答补齐长度。
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class LossOutput:
    loss: Tensor
    metrics: dict[str, float]
    # 只含有效 token 的一维 CPU 向量，供训练器跨 microbatch 精确汇总。
    # 例如，先拼接 ratio 再取 p99；不能平均各 microbatch 的 p99。
    diagnostics: dict[str, Tensor] = field(default_factory=dict)


def group_advantages(rewards: Tensor, num_generations: int, epsilon: float = 1e-4) -> Tensor:
    """同题 G 个奖励做相对比较。全对/全错时 std=0，优势为 0 而非 NaN。"""
    if rewards.ndim != 1 or num_generations < 2 or rewards.numel() % num_generations:
        raise ValueError("rewards 应为 [B*G]，G>=2，且总数可以被 G 整除")
    if rewards.numel() == 0 or not torch.isfinite(rewards).all():
        raise ValueError("奖励不能为空或包含 NaN/Inf")
    grouped = rewards.float().detach().view(-1, num_generations)
    # 与 MiniMind 一致，使用总体标准差 unbiased=False，不引入 G/(G-1) 修正。
    advantages = (grouped - grouped.mean(dim=1, keepdim=True)) / (
        grouped.std(dim=1, unbiased=False, keepdim=True) + epsilon
    )
    return advantages.reshape(-1)


def grpo_loss(
    new_logps: Tensor, old_logps: Tensor, advantages: Tensor, completion_mask: Tensor,
    *, ref_logps: Tensor | None = None, beta: float = 0.0,
    clip_low: float = 0.2, clip_high: float = 0.2, reduction: str = "sequence",
) -> LossOutput:
    """计算要最小化的 GRPO loss。

    new/old/ref_logps: [N,R]，每个实际采到 token 的条件 log probability。
    advantages: [N]，同一条回答的优势广播到该回答的所有 token。
    completion_mask: [N,R]，包含首次 EOS，排除 prompt、EOS 后补齐。
    old/ref 均 detach：梯度只流入当前 Actor，不会更新旧策略或 Reference。
    """
    if new_logps.ndim != 2 or old_logps.shape != new_logps.shape or completion_mask.shape != new_logps.shape:
        raise ValueError("new_logps/old_logps/completion_mask 必须具有相同 [N,R] 形状")
    if advantages.shape != new_logps.shape[:1]:
        raise ValueError("advantages 的形状应为 [N]")
    if beta < 0 or not 0 <= clip_low < 1 or clip_high < 0:
        raise ValueError("非法 beta 或 clipping 范围")
    if ref_logps is not None and ref_logps.shape != new_logps.shape:
        raise ValueError("ref_logps 必须与 new_logps 对齐")
    if beta and ref_logps is None:
        raise ValueError("beta>0 时必须提供冻结参考模型的 ref_logps")
    if reduction not in {"sequence", "token"}:
        raise ValueError("reduction 必须为 sequence 或 token")
    mask_bool = completion_mask.bool()
    if not ((completion_mask == 0) | (completion_mask == 1)).all() or not mask_bool.any(dim=1).all():
        raise ValueError("mask 必须为 0/1 且每条回答至少有一个有效 token")

    # 先移除无效位置，避免被忽略的 NaN/极端 padding logp 经 exp 后污染 loss。
    current = new_logps.float().masked_fill(~mask_bool, 0)
    old = old_logps.float().detach().masked_fill(~mask_bool, 0)
    advantage = advantages.float().detach().unsqueeze(1)
    if not torch.isfinite(current).all() or not torch.isfinite(old).all() or not torch.isfinite(advantage).all():
        raise FloatingPointError("有效位置的 log-prob/advantage 包含 NaN/Inf")
    log_ratio = current - old
    ratio = torch.exp(log_ratio)
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    # 正优势鼓励提高概率，负优势鼓励降低概率；min 选较保守的收益。
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)

    kl = torch.zeros_like(current)
    if beta:
        reference = ref_logps.float().detach().masked_fill(~mask_bool, 0)
        difference = reference - current
        # MiniMind 同款 k3 样本估计。非负的逐 token 值，不是完整词表上的精确 KL。
        kl = torch.exp(difference) - difference - 1
    per_token_loss = -surrogate + beta * kl
    mask = mask_bool.to(per_token_loss.dtype)
    if not torch.isfinite(per_token_loss).all():
        raise FloatingPointError("GRPO loss 溢出；检查学习率、采样 log-prob 和模型输出")
    if reduction == "sequence":
        # 原 GRPO：每条回答先平均，再对回答平均；长短回答权重相同。
        loss = ((per_token_loss * mask).sum(1) / mask.sum(1)).mean()
    else:
        # Token-level：全部有效 token 一起平均；长回答含更多训练 token。
        loss = (per_token_loss * mask).sum() / mask.sum()
    with torch.no_grad():
        active_clip = ((ratio > 1 + clip_high) & (advantage > 0)) | (
            (ratio < 1 - clip_low) & (advantage < 0)
        )
        positive = mask_bool & (advantage > 0)
        negative = mask_bool & (advantage < 0)
        positive_clipped = positive & active_clip
        negative_clipped = negative & active_clip
        valid_ratio = ratio[mask_bool]
        valid_log_ratio = log_ratio[mask_bool]
        quantiles = torch.tensor([0.05, 0.50, 0.95, 0.99], dtype=ratio.dtype, device=ratio.device)
        ratio_quantiles = torch.quantile(valid_ratio, quantiles)
        log_ratio_quantiles = torch.quantile(valid_log_ratio, quantiles)
        metrics = {
            "loss": loss.item(),
            "policy_loss": ((-surrogate * mask).sum() / mask.sum()).item(),
            "reference_kl_k3": ((kl * mask).sum() / mask.sum()).item(),
            "ratio_mean": ((ratio * mask).sum() / mask.sum()).item(),
            "clip_fraction": ((active_clip.float() * mask).sum() / mask.sum()).item(),
            # 分母分别是正、负优势的有效 token 数；若该类不存在，定义为 0。
            "positive_clip_fraction": (positive_clipped.sum() / positive.sum().clamp_min(1)).item(),
            "negative_clip_fraction": (negative_clipped.sum() / negative.sum().clamp_min(1)).item(),
        }
        for label, value in zip(("p05", "p50", "p95", "p99"), ratio_quantiles):
            metrics[f"ratio_{label}"] = value.item()
        for label, value in zip(("p05", "p50", "p95", "p99"), log_ratio_quantiles):
            metrics[f"log_ratio_{label}"] = value.item()
        diagnostics = {
            "ratio": valid_ratio.detach().cpu(),
            "log_ratio": valid_log_ratio.detach().cpu(),
            "positive_advantage": positive[mask_bool].detach().cpu(),
            "negative_advantage": negative[mask_bool].detach().cpu(),
            "positive_clipped": positive_clipped[mask_bool].detach().cpu(),
            "negative_clipped": negative_clipped[mask_bool].detach().cpu(),
        }
    return LossOutput(loss, metrics, diagnostics)
