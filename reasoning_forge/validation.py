"""固定开发集观测：评测不参与优化，也不改变后续训练采样的随机状态。"""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any

import torch

from .data import render_prompt
from .models import compute_token_stats
from .rewards import extractor_for_format


@contextmanager
def isolated_sampling_rng(device: torch.device, seed: int):
    """每次开发评估使用同一随机种子，结束后恢复训练的 RNG 流。"""
    python_state = random.getstate()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)


def fixed_probe_entropy(actor: Any, tokenizer: Any, device: torch.device,
                        prompts: list[str], config: Any) -> dict[str, float]:
    """在相同开发题 + 固定回答前缀的 token 位置测完整词表熵（单位 nat）。

    前缀仅作比较探针，不当作正确解答；每次评估的上下文完全相同。
    prompt 以 rollout 同样的左截断方式限制长度。逐题 forward 控制显存。
    """
    if not prompts:
        raise ValueError("固定熵探针至少需要一道题")
    prefix_ids = tokenizer(config.probe_response_prefix, add_special_tokens=False)["input_ids"]
    if not prefix_ids:
        raise ValueError("probe_response_prefix 不能 tokenize 成空序列")
    was_training = actor.training
    actor.eval()
    values = []
    try:
        with torch.no_grad():
            for prompt in prompts[:config.probe_prompts]:
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-config.max_prompt_length:]
                if not prompt_ids:
                    raise ValueError("开发集 prompt tokenize 后为空")
                ids = torch.tensor([prompt_ids + prefix_ids], dtype=torch.long, device=device)
                attention = torch.ones_like(ids)
                _, entropy = compute_token_stats(actor, ids, attention, len(prompt_ids), return_entropy=True)
                values.append(entropy.detach().float().flatten().cpu())
    finally:
        actor.train(was_training)
    entropies = torch.cat(values)
    quantiles = torch.quantile(entropies, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "probe_entropy_mean": entropies.mean().item(),
        "probe_entropy_p10": quantiles[0].item(),
        "probe_entropy_p50": quantiles[1].item(),
        "probe_entropy_p90": quantiles[2].item(),
        "probe_entropy_tokens": entropies.numel(),
        "probe_entropy_prompts": min(len(prompts), config.probe_prompts),
    }


def evaluate_dev(bundle: Any, engine: Any, config: Any, dev_samples: list, reward_function: Any,
                 step: int) -> tuple[dict, list[dict]]:
    """固定前若干道开发题每题采样 n 次，返回总览与逐题原始回答。

    pass@1 是每题正确回答比例的平均；pass@n 是每题是否至少采到一次正确。
    整次评测包括熵探针都置于隔离 RNG 内，不会挪动下一批训练 rollout。
    """
    from evaluate import binary_reward, summarize_results

    selected = dev_samples[:config.eval_limit]
    if not selected:
        raise ValueError("开发集为空")
    records: list[dict] = []
    actor = bundle.actor
    was_training = actor.training
    actor.eval()
    try:
        with isolated_sampling_rng(bundle.device, config.seed + 1_000_003), torch.no_grad():
            for first in range(0, len(selected), config.prompt_batch_size):
                current = selected[first:first + config.prompt_batch_size]
                prompts = [render_prompt(sample, bundle.tokenizer, config) for sample in current]
                prompt_lengths = [len(row) for row in bundle.tokenizer(
                    prompts, add_special_tokens=False, truncation=False, padding=False,
                )["input_ids"]]
                rollout = engine.rollout(prompts, num_generations=config.eval_num_samples)
                lengths = rollout.completion_mask.sum(-1).tolist()
                stops = rollout.truncated.tolist()
                for index, sample in enumerate(current):
                    start = index * config.eval_num_samples
                    end = start + config.eval_num_samples
                    completions = rollout.completions[start:end]
                    rewards = [binary_reward(reward_function(text, sample.answer)) for text in completions]
                    records.append({
                        "id": sample.id, "dataset": sample.dataset, "prompt": sample.prompt,
                        "answer": sample.answer, "prompt_tokens": prompt_lengths[index],
                        "completions": completions, "rewards": rewards,
                        "lengths": lengths[start:end], "truncated": stops[start:end],
                    })
                del rollout
            prompts = [render_prompt(sample, bundle.tokenizer, config) for sample in selected[:config.probe_prompts]]
            probe = fixed_probe_entropy(actor, bundle.tokenizer, bundle.device, prompts, config)
    finally:
        actor.train(was_training)
    ks = sorted({1, config.eval_num_samples})
    metrics = summarize_results(records, ks, max_prompt_length=config.max_prompt_length,
                                answer_extractor=extractor_for_format(config.answer_format))
    metrics["step"] = step
    metrics["probe"] = probe
    metrics["num_samples"] = config.eval_num_samples
    metrics["evaluated_prompts"] = len(selected)
    return metrics, records
