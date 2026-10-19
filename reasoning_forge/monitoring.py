"""从已采样轨迹计算便于排查 GRPO 训练问题的轻量诊断指标。

这里不再调用模型，也不改变训练目标。所有比例都以本次新采样的回答/题目为
分母；如果同一组轨迹做了多次参数更新，调用方应避免将其重复当成新样本。
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch

from .rewards import extract_final_answer, normalize_answer


def _nearest_rank(values: list[int], percentile: float) -> float:
    """p50/p90/p99 使用最近秩：排序后取 ceil(p*n) 位，适合小组的尾部诊断。"""
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(percentile * len(ordered)) - 1)])


def rollout_diagnostics(
    rewards: torch.Tensor,
    completions: list[str],
    completion_mask: torch.Tensor,
    truncated: torch.Tensor,
    generations: int,
) -> dict[str, float | int | None]:
    """统计一次采样窗口的答案、组别、重复、长度和截断情况。

    顺序必须为 [题0回答0, ..., 题0回答G-1, 题1回答0, ...]。
    completion_mask 可传 [N,R] 的有效位置 mask，也可传 [N] 的有效长度。
    多个 rollout 的 R 可能不同，此时先对各 mask 做 sum(-1)，再拼接长度即可。
    全错/全对/混合以及条件正确率仅适用于 0/1 奖励；连续奖励时返回 None，
    避免把奖励大小误认为数学题正确与否。没有分母的条件平均也用 None 表示。
    """
    if isinstance(generations, bool) or not isinstance(generations, int) or generations < 1:
        raise ValueError("generations 必须为正整数。")
    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1:
        raise ValueError("rewards 必须是 [N] Tensor。")
    count = rewards.numel()
    if count == 0 or count % generations != 0 or len(completions) != count:
        raise ValueError("回答数必须是正数，且 rewards/completions 必须按 B*G 对齐。")
    if not isinstance(completion_mask, torch.Tensor) or completion_mask.ndim not in (1, 2) or completion_mask.shape[0] != count:
        raise ValueError("completion_mask 必须是 [N,R] mask 或 [N] 有效长度 Tensor。")
    if not isinstance(truncated, torch.Tensor) or truncated.ndim != 1 or truncated.numel() != count:
        raise ValueError("truncated 必须是 [N] Tensor。")
    if not all(isinstance(text, str) for text in completions):
        raise TypeError("completions 中每个回答都必须是字符串。")

    # 诊断只需标量，转到 CPU 不会建立计算图，也不会留住大模型的 GPU 显存。
    reward_values = rewards.detach().float().cpu()
    masks = completion_mask.detach().cpu()
    truncated_values = truncated.detach().cpu()
    if not torch.isfinite(reward_values).all():
        raise ValueError("rewards 中有 NaN/Inf。")
    if masks.ndim == 2:
        if masks.shape[1] == 0 or not torch.all((masks == 0) | (masks == 1)):
            raise ValueError("completion_mask 必须有回答列，且只含 0/1。")
        lengths = [int(value) for value in masks.sum(-1).tolist()]
    else:
        if not torch.isfinite(masks).all() or not torch.all(masks == masks.long()):
            raise ValueError("有效长度必须是有限整数。")
        lengths = [int(value) for value in masks.tolist()]
    if not torch.all((truncated_values == 0) | (truncated_values == 1)):
        raise ValueError("truncated 只允许 0/1。")
    if any(length <= 0 for length in lengths):
        raise ValueError("每个回答至少要有一个有效 token。")
    is_truncated = [bool(value) for value in truncated_values.tolist()]

    # 与当前 GSM8K verifier 共用提取规则，确保“有最终答案”有明确语义。
    answers = [extract_final_answer(text) for text in completions]
    extracted = [answer is not None for answer in answers]
    parsed_count = sum(extracted)
    truncated_count = sum(is_truncated)
    missing_after_truncation = sum(stop and not found for stop, found in zip(is_truncated, extracted))

    group_count = count // generations
    answer_diversity = 0.0
    repetition = 0.0
    for start in range(0, count, generations):
        answer_group = answers[start:start + generations]
        # 缺失答案不能算作一种答案；除数仍是 G，格式坍缩也会拉低此值。
        answer_diversity += len({normalize_answer(answer) for answer in answer_group if answer is not None}) / generations
        response_group = completions[start:start + generations]
        normalized_responses = {re.sub(r"\s+", " ", text).strip() for text in response_group}
        repetition += 1.0 - len(normalized_responses) / generations

    result: dict[str, float | int | None] = {
        "answer_extraction_rate": parsed_count / count,
        "parsed_completions": parsed_count,
        "final_answer_diversity": answer_diversity / group_count,
        "response_repetition_fraction": repetition / group_count,
        "response_length_p50": _nearest_rank(lengths, 0.50),
        "response_length_p90": _nearest_rank(lengths, 0.90),
        "response_length_p99": _nearest_rank(lengths, 0.99),
        "truncated_unparseable_fraction": missing_after_truncation / count,
        "truncated_answer_missing_rate": (missing_after_truncation / truncated_count if truncated_count else None),
    }

    # 自定义奖励可能是连续数值：此时 0、1 没有“对/错”的可靠含义。
    binary_rewards = bool(torch.all((reward_values == 0) | (reward_values == 1)))
    if binary_rewards:
        correct = reward_values.bool()
        groups = correct.view(group_count, generations)
        correct_counts = groups.sum(-1)
        result.update({
            "all_wrong_group_fraction": (correct_counts == 0).float().mean().item(),
            "all_correct_group_fraction": (correct_counts == generations).float().mean().item(),
            "mixed_group_fraction": ((correct_counts > 0) & (correct_counts < generations)).float().mean().item(),
            "answer_accuracy_given_extracted": (
                correct[torch.tensor(extracted, dtype=torch.bool)].float().mean().item()
                if parsed_count else None
            ),
            "correct_response_length": (
                sum(length for length, good in zip(lengths, correct.tolist()) if good) / int(correct.sum())
                if bool(correct.any()) else None
            ),
            "incorrect_response_length": (
                sum(length for length, good in zip(lengths, correct.tolist()) if not good) / int((~correct).sum())
                if bool((~correct).any()) else None
            ),
        })
    else:
        result.update({
            "all_wrong_group_fraction": None,
            "all_correct_group_fraction": None,
            "mixed_group_fraction": None,
            "answer_accuracy_given_extracted": None,
            "correct_response_length": None,
            "incorrect_response_length": None,
        })
    return result


def prompt_truncation_diagnostics(
    rendered_prompts: list[str], tokenizer: Any, max_prompt_length: int
) -> dict[str, float | int]:
    """按 rollout 相同的 tokenization 判断题目是否被裁剪。

    传入 `render_prompt` 后、尚未按 G 复制的文本。题目长度恰好等于上限不算
    截断；长度大于上限才会丢失 token。这里不调用模型，计数按原题目计算。
    """
    if isinstance(max_prompt_length, bool) or not isinstance(max_prompt_length, int) or max_prompt_length < 1:
        raise ValueError("max_prompt_length 必须为正整数。")
    if not rendered_prompts or not all(isinstance(text, str) and text for text in rendered_prompts):
        raise ValueError("rendered_prompts 必须是非空字符串列表。")
    encoded = tokenizer(
        rendered_prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    token_rows = encoded["input_ids"]
    lengths = [len(row) for row in token_rows]
    if len(lengths) != len(rendered_prompts) or any(length == 0 for length in lengths):
        raise ValueError("tokenizer 的 input_ids 与 prompt 数不符，或生成了空序列。")
    clipped = sum(length > max_prompt_length for length in lengths)
    return {
        "prompt_truncation_fraction": clipped / len(lengths),
        "prompt_truncated_count": clipped,
        "prompt_total_count": len(lengths),
        "prompt_length_p90": _nearest_rank(lengths, 0.90),
        "prompt_length_p99": _nearest_rank(lengths, 0.99),
        "prompt_length_max": max(lengths),
        "prompt_discarded_tokens": sum(max(0, length - max_prompt_length) for length in lengths),
    }
