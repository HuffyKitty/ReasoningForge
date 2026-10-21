"""DAPO 式动态采样所需的确定性候选流与整组筛选。"""

import random

import torch

from .rollout import RolloutBatch


class CandidateStream:
    """按 seed+轮次洗牌，按绝对游标重建；无需保存整份排列。"""

    def __init__(self, samples: list, seed: int):
        if not samples:
            raise ValueError("候选流的数据集不能为空")
        self.samples = samples
        self.seed = seed
        self._cycle = -1
        self._order: list[int] = []

    def take(self, start: int, count: int) -> list:
        if start < 0 or count < 0:
            raise ValueError("候选游标和数量必须为非负整数")
        result = []
        for position in range(start, start + count):
            cycle, offset = divmod(position, len(self.samples))
            if cycle != self._cycle:
                self._order = list(range(len(self.samples)))
                random.Random(self.seed + cycle).shuffle(self._order)
                self._cycle = cycle
            result.append(self.samples[self._order[offset]])
        return result


def informative_group_indices(rewards: torch.Tensor, generations: int) -> list[int]:
    """只保留组内奖励并非全相同的题目；不额外引入长度优势。"""
    if rewards.ndim != 1 or rewards.numel() % generations:
        raise ValueError("奖励数必须是题目组数乘以回答数")
    grouped = rewards.reshape(-1, generations)
    return torch.nonzero(grouped.max(1).values > grouped.min(1).values).flatten().tolist()


def select_rollout_groups(rollout: RolloutBatch, groups: list[int], generations: int) -> RolloutBatch:
    if not groups:
        raise ValueError("不能选择空题目组")
    if len(rollout.completions) % generations:
        raise ValueError("回答数不是组大小的整数倍")
    if min(groups) < 0 or max(groups) >= len(rollout.completions) // generations:
        raise IndexError("题目组索引越界")
    positions = [group * generations + offset for group in groups for offset in range(generations)]
    index = torch.tensor(positions, dtype=torch.long, device=rollout.input_ids.device)
    return RolloutBatch(
        input_ids=rollout.input_ids.index_select(0, index),
        attention_mask=rollout.attention_mask.index_select(0, index),
        completion_ids=rollout.completion_ids.index_select(0, index),
        completion_mask=rollout.completion_mask.index_select(0, index),
        old_logps=rollout.old_logps.index_select(0, index),
        completions=[rollout.completions[position] for position in positions],
        prompt_width=rollout.prompt_width,
        truncated=rollout.truncated.index_select(0, index),
    )
