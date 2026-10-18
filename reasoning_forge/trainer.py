"""训练总装配：采样窗口 → 固定旧策略与奖励 → 一次或多次优化 → 下一窗口。

单进程、单设备，不用 TRL/Trainer 隐藏训练过程。一个采样窗口包含 K 个
prompt batch，必须先全部采样，再更新 Actor，保证窗口内的 old_logps 来自
同一个旧策略。updates_per_rollout>1 会重复利用这批固定的轨迹，让后续
更新的 ratio 偏离 1，因此可以实际研究 clipping / Clip-Higher。
"""

import importlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import (
    data_fingerprint, environment_info, load_training_state, restore_rng,
    save_checkpoint, seed_everything, write_json,
)
from .data import load_jsonl, render_prompt
from .models import compute_token_stats, load_bundle
from .objectives import group_advantages, grpo_loss
from .rollout import PytorchRolloutEngine, RolloutBatch


def load_reward_function(spec: str):
    """奖励插件只负责 completion + 标准答案 → 一个有限实数，不参与反向传播。"""
    module_name, separator, name = spec.partition(":")
    if not separator or not module_name or not name:
        raise ValueError("reward_function 应为 'python模块:函数名'")
    function = getattr(importlib.import_module(module_name), name)
    if not callable(function):
        raise TypeError(f"{spec} 不是可调用函数")
    return function


def score_answers(samples, completions: list[str], generations: int, reward_function) -> torch.Tensor:
    if len(completions) != len(samples) * generations:
        raise ValueError("回答数量与 B*G 不一致；检查 rollout 的样本顺序")
    values = [float(reward_function(completion, samples[index // generations].answer))
              for index, completion in enumerate(completions)]
    rewards = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(rewards).all():
        raise ValueError("奖励函数返回 NaN/Inf")
    return rewards


def shuffled_batches(samples, config, epoch: int):
    """每轮使用 seed+epoch 独立洗牌；按 batch_index 即可重建恢复位置。"""
    indices = list(range(len(samples)))
    random.Random(config.seed + epoch).shuffle(indices)
    return [[samples[index] for index in indices[start:start + config.prompt_batch_size]]
            for start in range(0, len(indices), config.prompt_batch_size)]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


@dataclass
class PreparedRollout:
    """只缓存 token 与 [N,R] 概率，不缓存大词表 logits 或带梯度计算图。"""

    samples: list
    rollout: RolloutBatch
    rewards: torch.Tensor
    advantages: torch.Tensor
    reference_logps: torch.Tensor | None


def prepare_window(window, bundle, engine, reward_function, config) -> list[PreparedRollout]:
    """在任何 optimizer.step 之前，准备整个梯度累积窗口的训练材料。"""
    prepared = []
    for current_samples in window:
        prompts = [render_prompt(sample, bundle.tokenizer, config) for sample in current_samples]
        # old_logps 在 rollout 中以 no_grad 计算；一旦缓存，本窗口不再改变。
        rollout = engine.rollout(prompts)
        rewards = score_answers(
            current_samples, rollout.completions, config.num_generations, reward_function
        ).to(bundle.device)
        advantages = group_advantages(rewards, config.num_generations)
        if not (rollout.completion_mask.sum(1) > 0).all():
            raise ValueError("生成了空 token 序列，无法计算 GRPO")

        reference_logps = None
        if bundle.reference is not None:
            pieces = []
            with torch.no_grad():
                for first in range(0, rollout.input_ids.size(0), config.train_micro_batch_size):
                    last = first + config.train_micro_batch_size
                    lp, _ = compute_token_stats(
                        bundle.reference, rollout.input_ids[first:last],
                        rollout.attention_mask[first:last], rollout.prompt_width,
                    )
                    pieces.append(lp)
            reference_logps = torch.cat(pieces).detach()
        prepared.append(PreparedRollout(current_samples, rollout, rewards, advantages, reference_logps))
    return prepared


def window_statistics(prepared: list[PreparedRollout], generations: int) -> dict:
    """按真实回答数/组数统计，避免最后一小批与完整 batch 被错误地等权平均。"""
    rewards = torch.cat([item.rewards for item in prepared])
    advantages = torch.cat([item.advantages for item in prepared])
    lengths = torch.cat([item.rollout.completion_mask.sum(1) for item in prepared])
    truncated = torch.cat([item.rollout.truncated for item in prepared])
    group_std = rewards.view(-1, generations).std(dim=1, unbiased=False)
    return {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std(unbiased=False).item(),
        "group_reward_std": group_std.mean().item(),
        "zero_variance_group_fraction": (group_std < 1e-8).float().mean().item(),
        "advantage_std": advantages.std(unbiased=False).item(),
        "response_length": lengths.float().mean().item(),
        "truncated_fraction": truncated.float().mean().item(),
        "training_tokens": int(lengths.sum().item()),
        "training_sequences": int(lengths.numel()),
        "training_prompts": sum(len(item.samples) for item in prepared),
    }


def backward_window(actor, prepared: list[PreparedRollout], config, statistics: dict) -> dict:
    """重新计算当前策略的概率，精确还原整个窗口的平均损失后累积梯度。

    sequence reduction：每块权重 = 块内回答数 / 全窗口回答数。
    token reduction：每块权重 = 块内有效 token 数 / 全窗口有效 token 数。
    不额外除以 K：K 已包含在这两个全局分母中。末尾不足 K 批也自然正确。
    """
    aggregate = {name: 0.0 for name in (
        "loss", "policy_loss", "reference_kl_k3", "ratio_mean", "clip_fraction",
    )}
    total_tokens = statistics["training_tokens"]
    total_sequences = statistics["training_sequences"]
    entropy_sum = 0.0
    actor.train()  # models.py 已禁用 Dropout，但仍保留训练模式和梯度检查点。
    for item in prepared:
        rollout = item.rollout
        for first in range(0, rollout.input_ids.size(0), config.train_micro_batch_size):
            last = min(first + config.train_micro_batch_size, rollout.input_ids.size(0))
            mask = rollout.completion_mask[first:last]
            current_logps, entropy = compute_token_stats(
                actor, rollout.input_ids[first:last], rollout.attention_mask[first:last],
                rollout.prompt_width, return_entropy=True,
            )
            result = grpo_loss(
                current_logps, rollout.old_logps[first:last], item.advantages[first:last], mask,
                ref_logps=item.reference_logps[first:last] if item.reference_logps is not None else None,
                beta=config.beta, clip_low=config.clip_low, clip_high=config.clip_high,
                reduction=config.loss_reduction,
            )
            token_fraction = mask.sum().item() / total_tokens
            weight = ((last - first) / total_sequences
                      if config.loss_reduction == "sequence" else token_fraction)
            (result.loss * weight).backward()
            for name, value in result.metrics.items():
                # loss 跟随所选训练目标；其他概率/熵指标统一按有效 token 统计。
                aggregate[name] += value * (weight if name == "loss" else token_fraction)
            entropy_sum += (entropy * mask).sum().item()
            # 每块立即反传、释放图，避免把所有大词表 logits 留到窗口末尾。
            del current_logps, entropy, result
    aggregate["entropy"] = entropy_sum / total_tokens
    return aggregate


def train(config, resume: Path | None = None) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("当前基线是单设备实现，请直接 python train.py，不要使用 torchrun")
    config.validate()
    samples = load_jsonl(config.data_path)
    fingerprint = data_fingerprint(config.data_path)
    state = load_training_state(resume.resolve(), config, fingerprint) if resume else None
    output_dir = Path(config.output_dir)
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空：{output_dir}。请换 output_dir，或显式 --resume 检查点")
    if state and (output_dir / "metrics.jsonl").exists():
        # 回退到较旧 checkpoint 不能接着追加未来的日志，否则同一步会出现两套结果。
        with (output_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
            if any(json.loads(line)["step"] > state["global_step"] for line in handle if line.strip()):
                raise ValueError("输出目录已有比恢复点更新的日志；请恢复最新检查点，或改用新的 output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    bundle = load_bundle(config, resume_model_path=resume.resolve() / "model" if resume else None)
    actor, reference = bundle.actor, bundle.reference
    engine = PytorchRolloutEngine(bundle, config)
    reward_function = load_reward_function(config.reward_function)
    trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Actor 没有可训练参数；检查 LoRA 目标层名称")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    global_step, start_epoch, start_batch = 0, 0, 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        global_step, start_epoch, start_batch = state["global_step"], state["epoch"], state["batch_index"]
        restore_rng(state)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(output_dir / "environment.json", environment_info())
    write_json(output_dir / "data_manifest.json", {"path": config.data_path, "sha256": fingerprint, "count": len(samples)})
    write_json(output_dir / "model_manifest.json", {
        "source": config.model_name_or_path, "requested_revision": config.revision,
        "resolved_revision": getattr(actor.config, "_commit_hash", None),
        "model_type": actor.config.model_type, "lora": config.lora,
    })
    count = sum(parameter.numel() for parameter in actor.parameters())
    print(f"Actor 总参数 {count:,}；可训练参数 {sum(p.numel() for p in trainable):,}；device={bundle.device}")
    print(f"训练 {len(samples)} 道题，每题 {config.num_generations} 个回答；日志：{output_dir}")
    if config.temperature != 1 or config.top_p != 1 or config.top_k != 0:
        print("采样启用了分布变换；ratio 仍按 MiniMind 惯例使用未变换的模型概率。严格基线建议 temperature=1, top_p=1, top_k=0。")

    for epoch in range(start_epoch, config.epochs):
        batches = shuffled_batches(samples, config, epoch)
        batch_index = start_batch if epoch == start_epoch else 0
        while batch_index < len(batches):
            if config.max_steps and global_step >= config.max_steps:
                break
            begin = time.perf_counter()
            # 1. 先采齐 K 批，期间不更新 Actor；缓存内所有 old_logps 来自同一旧策略。
            window = batches[batch_index:batch_index + config.gradient_accumulation_steps]
            prepared = prepare_window(window, bundle, engine, reward_function, config)
            statistics = window_statistics(prepared, config.num_generations)
            window_start_step = global_step
            updates = config.updates_per_rollout
            if config.max_steps:
                updates = min(updates, config.max_steps - global_step)

            # 2. 固定旧策略/奖励/优势/Reference，只重新计算当前 Actor 的概率。
            # 第一次 ratio≈1；第二次起 Actor 已变化，clipping 才可能真正生效。
            for update_index in range(updates):
                step_begin = begin if update_index == 0 else time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                metrics = backward_window(actor, prepared, config, statistics)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, config.max_grad_norm if config.max_grad_norm > 0 else float("inf"),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1  # max_steps 始终以 optimizer.step 次数计数。
                metrics.update(statistics)
                metrics.update({
                    "step": global_step, "epoch": epoch, "rollout_update": update_index + 1,
                    "rollout_batch_index": batch_index,
                    "next_batch_index": batch_index + len(window),
                    "grad_norm": grad_norm.item(), "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds": time.perf_counter() - step_begin,
                    # 复用轨迹不等于重新采样：新增 token/题目只在第一次更新计数。
                    "generated_tokens": statistics["training_tokens"] if update_index == 0 else 0,
                    "prompts": statistics["training_prompts"] if update_index == 0 else 0,
                })
                append_jsonl(output_dir / "metrics.jsonl", metrics)
                if global_step % config.log_steps == 0:
                    print(f"step={global_step} reuse={update_index + 1}/{updates} "
                          f"reward={metrics['reward_mean']:.3f} loss={metrics['loss']:.5f} "
                          f"entropy={metrics['entropy']:.3f} ratio={metrics['ratio_mean']:.4f} "
                          f"clip={metrics['clip_fraction']:.1%} KL(k3)={metrics['reference_kl_k3']:.5f}")
                    # 同一批回答只写一次，不把复用更新伪装成新的 rollout。
                    if update_index == 0:
                        for item in prepared:
                            for index in range(min(2, len(item.rollout.completions))):
                                sample = item.samples[index // config.num_generations]
                                append_jsonl(output_dir / "samples.jsonl", {
                                    "step": global_step, "id": sample.id, "prompt": sample.prompt,
                                    "answer": sample.answer, "completion": item.rollout.completions[index],
                                    "reward": item.rewards[index].item(),
                                })

            # 3. 窗口复用结束才推进数据游标。若 max_steps 截断复用，剩余次数
            # 直接丢弃；恢复后从下一份数据采样，不重放这一窗口的旧轨迹。
            batch_index += len(window)
            next_epoch, next_batch = (epoch + 1, 0) if batch_index == len(batches) else (epoch, batch_index)
            final_step = ((config.max_steps > 0 and global_step >= config.max_steps)
                          or (next_epoch == config.epochs))
            # 只在完整窗口边界存盘，故无需把缓存轨迹写入检查点。
            # 跨过 save_steps 整除点时延迟保存，最多延迟 updates_per_rollout-1 步。
            crossed_save_boundary = global_step // config.save_steps > window_start_step // config.save_steps
            if crossed_save_boundary or final_step:
                path = save_checkpoint(bundle, optimizer, config, global_step=global_step,
                    epoch=next_epoch, batch_index=next_batch, fingerprint=fingerprint)
                print(f"检查点：{path}")
            del prepared
        start_batch = 0
        if config.max_steps and global_step >= config.max_steps:
            break
    if global_step == 0:
        raise ValueError("没有执行任何优化步骤；请检查数据、epochs 和恢复位置")
    # 推理导出与训练检查点分开；LoRA 的 final/ 是 adapter，需要原基座配合加载。
    actor.save_pretrained(output_dir / "final", safe_serialization=True)
    bundle.tokenizer.save_pretrained(output_dir / "final")
    print(f"完成：优化器累计更新 {global_step} 次；{'LoRA adapter' if config.lora else '完整 HF 模型'}：{output_dir / 'final'}")
