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
from .dynamic_sampling import CandidateStream, informative_group_indices, select_rollout_groups
from .monitoring import prompt_truncation_diagnostics, rollout_diagnostics
from .models import compute_token_stats, load_bundle
from .objectives import group_advantages, grpo_loss
from .rewards import extractor_for_format
from .rollout import PytorchRolloutEngine, RolloutBatch
from .validation import evaluate_dev


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


def record_failure(output_dir: Path, step: int, stage: str, error: Exception) -> None:
    """数值错误和 OOM 留下可定位的事件；异常仍向上传播，绝不静默续训。"""
    try:
        append_jsonl(output_dir / "events.jsonl", {
            "step": step, "stage": stage, "type": type(error).__name__, "message": str(error)[:1000],
        })
    except OSError:
        pass  # 磁盘也故障时保留原始训练异常。


@dataclass
class PreparedRollout:
    """只缓存 token 与 [N,R] 概率，不缓存大词表 logits 或带梯度计算图。"""

    samples: list
    rollout: RolloutBatch
    rewards: torch.Tensor
    advantages: torch.Tensor
    reference_logps: torch.Tensor | None
    rendered_prompts: list[str] | None = None


def prepare_window(window, bundle, engine, reward_function, config) -> list[PreparedRollout]:
    """在任何 optimizer.step 之前，准备整个梯度累积窗口的训练材料。"""
    prepared = []
    if getattr(config, "rollout_batch_across_window", False):
        # 按原窗口推进 batch_index；只合并生成请求，不扩大一次更新的数据量。
        # backward_window 使用全窗口分母，也适用于合并后的 PreparedRollout。
        window = [[sample for batch in window for sample in batch]]
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
        prepared.append(PreparedRollout(current_samples, rollout, rewards, advantages, reference_logps, prompts))
    return prepared


def prepare_dynamic_window(stream: CandidateStream, candidate_index: int, target_prompts: int,
                           bundle, engine, reward_function, config) -> tuple[list[PreparedRollout], int, dict]:
    """补采至固定有效题目数；所有候选都来自同一个旧 Actor。"""
    prepared: list[PreparedRollout] = []
    sampled_prompts = sampled_tokens = accepted_prompts = 0
    max_candidates = target_prompts * config.dynamic_sampling_max_candidates_factor
    while accepted_prompts < target_prompts:
        remaining_budget = max_candidates - sampled_prompts
        if remaining_budget <= 0:
            raise RuntimeError(
                f"动态采样达到 {max_candidates} 道候选上限，仍只有 "
                f"{accepted_prompts}/{target_prompts} 个有效题目组"
            )
        count = min(config.prompt_batch_size, target_prompts - accepted_prompts, remaining_budget)
        current_samples = stream.take(candidate_index, count)
        candidate_index += count
        prompts = [render_prompt(sample, bundle.tokenizer, config) for sample in current_samples]
        rollout = engine.rollout(prompts)
        rewards = score_answers(
            current_samples, rollout.completions, config.num_generations, reward_function
        ).to(bundle.device)
        sampled_prompts += count
        sampled_tokens += int(rollout.completion_mask.sum().item())
        groups = informative_group_indices(rewards, config.num_generations)
        if not groups:
            continue
        selected_rollout = select_rollout_groups(rollout, groups, config.num_generations)
        selected_rewards = rewards.reshape(-1, config.num_generations)[groups].flatten()
        advantages = group_advantages(selected_rewards, config.num_generations)
        if not (selected_rollout.completion_mask.sum(1) > 0).all():
            raise ValueError("生成了空 token 序列，无法计算 GRPO")
        reference_logps = None
        if bundle.reference is not None:
            pieces = []
            with torch.no_grad():
                for first in range(0, selected_rollout.input_ids.size(0), config.train_micro_batch_size):
                    last = first + config.train_micro_batch_size
                    lp, _ = compute_token_stats(
                        bundle.reference, selected_rollout.input_ids[first:last],
                        selected_rollout.attention_mask[first:last], selected_rollout.prompt_width,
                    )
                    pieces.append(lp)
            reference_logps = torch.cat(pieces).detach()
        prepared.append(PreparedRollout(
            [current_samples[group] for group in groups], selected_rollout,
            selected_rewards, advantages, reference_logps, [prompts[group] for group in groups],
        ))
        accepted_prompts += len(groups)
    return prepared, candidate_index, {
        "sampled_prompts": sampled_prompts,
        "sampled_tokens": sampled_tokens,
        "rejected_prompts": sampled_prompts - accepted_prompts,
        "sampling_acceptance_rate": accepted_prompts / sampled_prompts,
    }


def window_statistics(prepared: list[PreparedRollout], generations: int,
                      tokenizer=None, max_prompt_length: int | None = None,
                      answer_extractor=None) -> dict:
    """按真实回答数/组数统计，避免最后一小批与完整 batch 被错误地等权平均。"""
    rewards = torch.cat([item.rewards for item in prepared])
    advantages = torch.cat([item.advantages for item in prepared])
    lengths = torch.cat([item.rollout.completion_mask.sum(1) for item in prepared])
    truncated = torch.cat([item.rollout.truncated for item in prepared])
    group_std = rewards.view(-1, generations).std(dim=1, unbiased=False)
    stats = {
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
    # 每份 PreparedRollout 可能有不同 R，只合并一维有效长度，避免 [N,R] 拼接报错。
    diagnostics = rollout_diagnostics(
        rewards, [text for item in prepared for text in item.rollout.completions],
        lengths, truncated, generations,
        **({"answer_extractor": answer_extractor} if answer_extractor is not None else {}),
    )
    stats.update(diagnostics)
    stats["format_rate"] = diagnostics["answer_extraction_rate"]
    stats["mixed_group_count"] = (
        int(round(diagnostics["mixed_group_fraction"] * stats["training_prompts"]))
        if diagnostics["mixed_group_fraction"] is not None else None
    )
    stats["mixed_groups_per_100_new_groups"] = (
        100.0 * diagnostics["mixed_group_fraction"]
        if diagnostics["mixed_group_fraction"] is not None else None
    )
    if tokenizer is not None and max_prompt_length is not None:
        prompts = [prompt for item in prepared for prompt in (item.rendered_prompts or [])]
        if len(prompts) != stats["training_prompts"]:
            raise ValueError("窗口内题目与渲染后 prompt 数量不一致")
        stats.update(prompt_truncation_diagnostics(prompts, tokenizer, max_prompt_length))
    return stats


def backward_window(actor, prepared: list[PreparedRollout], config, statistics: dict) -> dict:
    """重新计算当前策略的概率，精确还原整个窗口的平均损失后累积梯度。

    sequence reduction：每块权重 = 块内回答数 / 全窗口回答数。
    token reduction：每块权重 = 块内有效 token 数 / 全窗口有效 token 数。
    不额外除以 K：K 已包含在这两个全局分母中。末尾不足 K 批也自然正确。
    """
    aggregate = {name: 0.0 for name in (
        "loss", "policy_loss", "reference_kl_k3", "ratio_mean", "clip_fraction",
    )}
    diagnostics: dict[str, list[torch.Tensor]] = {}
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
                if name not in aggregate:
                    continue  # 分位数不能先对 microbatch 求再平均，窗口结束统一计算。
                # loss 跟随所选训练目标；其他概率/熵指标统一按有效 token 统计。
                aggregate[name] += value * (weight if name == "loss" else token_fraction)
            for name, values in result.diagnostics.items():
                diagnostics.setdefault(name, []).append(values)
            entropy_sum += (entropy * mask).sum().item()
            # 每块立即反传、释放图，避免把所有大词表 logits 留到窗口末尾。
            del current_logps, entropy, result
    aggregate["entropy"] = entropy_sum / total_tokens
    merged = {name: torch.cat(chunks) for name, chunks in diagnostics.items()}
    for name in ("ratio", "log_ratio"):
        values = merged[name].float()
        quantiles = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95, 0.99]))
        for label, value in zip(("p05", "p50", "p95", "p99"), quantiles):
            aggregate[f"{name}_{label}"] = value.item()
    for sign in ("positive", "negative"):
        active = merged[f"{sign}_advantage"]
        clipped = merged[f"{sign}_clipped"]
        aggregate[f"{sign}_advantage_tokens"] = int(active.sum().item())
        aggregate[f"{sign}_clip_fraction"] = (
            clipped[active].float().mean().item() if bool(active.any()) else 0.0
        )
    return aggregate


def train(config, resume: Path | None = None, *, allow_rollout_tuning: bool = False) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("当前基线是单设备实现，请直接 python train.py，不要使用 torchrun")
    config.validate()
    samples = load_jsonl(config.data_path)
    fingerprint = data_fingerprint(config.data_path)
    dev_samples = load_jsonl(config.dev_data_path) if config.dev_data_path else None
    dev_fingerprint = data_fingerprint(config.dev_data_path) if config.dev_data_path else None
    if dev_samples is not None:
        train_ids = {(sample.dataset, sample.id) for sample in samples}
        train_prompts = {sample.prompt.strip() for sample in samples}
        if any((sample.dataset, sample.id) in train_ids or sample.prompt.strip() in train_prompts
               for sample in dev_samples):
            raise ValueError("训练集与开发集存在重复题目或重复 dataset/id；请先修正数据划分")
    state = load_training_state(resume.resolve(), config, fingerprint,
                                allow_rollout_tuning=allow_rollout_tuning) if resume else None
    output_dir = Path(config.output_dir)
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空：{output_dir}。请换 output_dir，或显式 --resume 检查点")
    if state and (output_dir / "metrics.jsonl").exists():
        # 回退到较旧 checkpoint 不能接着追加未来的日志，否则同一步会出现两套结果。
        with (output_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
            if any(json.loads(line)["step"] > state["global_step"] for line in handle if line.strip()):
                raise ValueError("输出目录已有比恢复点更新的日志；请恢复最新检查点，或改用新的 output_dir")
    if state and (output_dir / "dev_metrics.jsonl").exists():
        with (output_dir / "dev_metrics.jsonl").open(encoding="utf-8") as handle:
            if any(json.loads(line)["step"] > state["global_step"] for line in handle if line.strip()):
                raise ValueError("开发集日志比恢复点更新；请恢复最新检查点或另建 output_dir")
    if state and dev_fingerprint is not None and (output_dir / "dev_manifest.json").exists():
        existing = json.loads((output_dir / "dev_manifest.json").read_text(encoding="utf-8"))
        if existing.get("sha256") != dev_fingerprint:
            raise ValueError("开发集内容已改变，不能将两份开发曲线写入同一实验")
    output_dir.mkdir(parents=True, exist_ok=True)
    if state and allow_rollout_tuning:
        changes = {key: {"before": state["config"].get(key, type(config)().to_dict()[key]),
                         "after": value}
                   for key, value in config.to_dict().items()
                   if key in {"rollout_batch_across_window", "rollout_micro_batch_size"}
                   and state["config"].get(key, type(config)().to_dict()[key]) != value}
        if changes:
            write_json(output_dir / f"rollout_tuning_from_step_{state['global_step']:06d}.json", {
                "checkpoint": str(resume.resolve()), "step": state["global_step"], "changes": changes,
                "note": "Optimizer and data cursor restored. Sampling order changes; this is not bit-identical continuation.",
            })
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
    candidate_index = 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        global_step, start_epoch, start_batch = state["global_step"], state["epoch"], state["batch_index"]
        if config.dynamic_sampling:
            candidate_index = state.get("candidate_index")
            if candidate_index is None:
                raise ValueError("动态采样检查点缺少候选数据游标")
        restore_rng(state)
    candidate_stream = CandidateStream(samples, config.seed) if config.dynamic_sampling else None
    answer_extractor = extractor_for_format(config.answer_format)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(output_dir / "environment.json", environment_info())
    write_json(output_dir / "data_manifest.json", {"path": config.data_path, "sha256": fingerprint, "count": len(samples)})
    if dev_samples is not None:
        write_json(output_dir / "dev_manifest.json", {
            "path": config.dev_data_path, "sha256": dev_fingerprint, "count": len(dev_samples),
            "periodic_limit": min(config.eval_limit, len(dev_samples)),
        })
    write_json(output_dir / "model_manifest.json", {
        "source": config.model_name_or_path, "requested_revision": config.revision,
        "resolved_revision": getattr(actor.config, "_commit_hash", None),
        "model_type": actor.config.model_type, "lora": config.lora,
    })
    count = sum(parameter.numel() for parameter in actor.parameters())
    print(f"Actor 总参数 {count:,}；可训练参数 {sum(p.numel() for p in trainable):,}；device={bundle.device}")
    print(f"训练 {len(samples)} 道题，每题 {config.num_generations} 个回答；日志：{output_dir}")
    if config.temperature != 1 or config.top_p != 1 or config.top_k != 0:
        print("采样启用了分布变换；ratio 使用未变换的模型概率。严格基线建议 temperature=1, top_p=1, top_k=0。")

    for epoch in range(start_epoch, config.epochs):
        batches = shuffled_batches(samples, config, epoch)
        batch_index = start_batch if epoch == start_epoch else 0
        while batch_index < len(batches):
            if config.max_steps and global_step >= config.max_steps:
                break
            begin = time.perf_counter()
            # 1. 先采齐 K 批，期间不更新 Actor；缓存内所有 old_logps 来自同一旧策略。
            window = batches[batch_index:batch_index + config.gradient_accumulation_steps]
            try:
                if config.dynamic_sampling:
                    prepared, candidate_index, sampling = prepare_dynamic_window(
                        candidate_stream, candidate_index, sum(map(len, window)),
                        bundle, engine, reward_function, config,
                    )
                else:
                    prepared = prepare_window(window, bundle, engine, reward_function, config)
                    sampling = None
                statistics = window_statistics(prepared, config.num_generations,
                                               bundle.tokenizer, config.max_prompt_length,
                                               answer_extractor)
                if sampling is None:
                    sampling = {
                        "sampled_prompts": statistics["training_prompts"],
                        "sampled_tokens": statistics["training_tokens"],
                        "rejected_prompts": 0,
                        "sampling_acceptance_rate": 1.0,
                    }
            except Exception as error:
                record_failure(output_dir, global_step, "rollout_or_reward", error)
                raise
            window_start_step = global_step
            updates = config.updates_per_rollout
            if config.max_steps:
                updates = min(updates, config.max_steps - global_step)

            # 2. 固定旧策略/奖励/优势/Reference，只重新计算当前 Actor 的概率。
            # 第一次 ratio≈1；第二次起 Actor 已变化，clipping 才可能真正生效。
            for update_index in range(updates):
                step_begin = begin if update_index == 0 else time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                try:
                    metrics = backward_window(actor, prepared, config, statistics)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable, config.max_grad_norm if config.max_grad_norm > 0 else float("inf"),
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                except Exception as error:
                    record_failure(output_dir, global_step, "backward_or_optimizer", error)
                    raise
                optimizer.zero_grad(set_to_none=True)
                global_step += 1  # max_steps 始终以 optimizer.step 次数计数。
                metrics.update(statistics)
                metrics.update({
                    "step": global_step, "epoch": epoch, "rollout_update": update_index + 1,
                    "rollout_batch_index": batch_index,
                    "next_batch_index": batch_index + len(window),
                    "grad_norm": grad_norm.item(), "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds": time.perf_counter() - step_begin,
                    # 动态采样的生成成本包含丢弃的候选；复用时不重复计数。
                    "generated_tokens": sampling["sampled_tokens"] if update_index == 0 else 0,
                    "prompts": statistics["training_prompts"] if update_index == 0 else 0,
                    "sampled_prompts": sampling["sampled_prompts"] if update_index == 0 else 0,
                    "rejected_prompts": sampling["rejected_prompts"] if update_index == 0 else 0,
                    "sampling_acceptance_rate": sampling["sampling_acceptance_rate"],
                    "candidate_index": candidate_index if config.dynamic_sampling else None,
                    "generation_seconds": sum((item.rollout.timings or {}).get("generation_seconds", 0.0)
                                               for item in prepared) if update_index == 0 else 0.0,
                    "old_logp_seconds": sum((item.rollout.timings or {}).get("old_logp_seconds", 0.0)
                                             for item in prepared) if update_index == 0 else 0.0,
                })
                append_jsonl(output_dir / "metrics.jsonl", metrics)
                if global_step % config.log_steps == 0:
                    print(f"step={global_step} reuse={update_index + 1}/{updates} "
                          f"reward={metrics['reward_mean']:.3f} loss={metrics['loss']:.5f} "
                          f"entropy={metrics['entropy']:.3f} ratio={metrics['ratio_mean']:.4f} "
                          f"clip={metrics['clip_fraction']:.1%} KL(k3)={metrics['reference_kl_k3']:.5f}")
                # 每个新 rollout 的全部 G 个回答都留档，与 log_steps 无关；复用不重复写。
                if update_index == 0:
                    for item in prepared:
                        lengths = item.rollout.completion_mask.sum(-1).tolist()
                        stops = item.rollout.truncated.tolist()
                        for index, completion in enumerate(item.rollout.completions):
                            sample = item.samples[index // config.num_generations]
                            append_jsonl(output_dir / "samples.jsonl", {
                                "step": global_step, "id": sample.id, "dataset": sample.dataset,
                                "group_index": index // config.num_generations,
                                "generation_index": index % config.num_generations,
                                "prompt": sample.prompt, "answer": sample.answer,
                                "completion": completion, "final_answer": answer_extractor(completion),
                                "reward": item.rewards[index].item(), "response_tokens": lengths[index],
                                "truncated": bool(stops[index]),
                            })

            # 3. 窗口复用结束才推进数据游标。若 max_steps 截断复用，剩余次数
            # 直接丢弃；恢复后从下一份数据采样，不重放这一窗口的旧轨迹。
            batch_index += len(window)
            next_epoch, next_batch = (epoch + 1, 0) if batch_index == len(batches) else (epoch, batch_index)
            final_step = ((config.max_steps > 0 and global_step >= config.max_steps)
                          or (next_epoch == config.epochs))
            if dev_samples is not None and config.eval_steps and (
                global_step // config.eval_steps > window_start_step // config.eval_steps or final_step
            ):
                try:
                    dev_metrics, dev_records = evaluate_dev(
                        bundle, engine, config, dev_samples, reward_function, global_step,
                    )
                except Exception as error:
                    record_failure(output_dir, global_step, "dev_evaluation", error)
                    raise
                dev_metrics["data_sha256"] = dev_fingerprint
                append_jsonl(output_dir / "dev_metrics.jsonl", dev_metrics)
                for record in dev_records:
                    append_jsonl(output_dir / "dev_samples.jsonl", {"step": global_step, **record})
                print(f"开发集 step={global_step} pass@1={dev_metrics['overall']['pass@1']:.3f} "
                      f"pass@{config.eval_num_samples}={dev_metrics['overall'][f'pass@{config.eval_num_samples}']:.3f} "
                      f"probe_entropy={dev_metrics['probe']['probe_entropy_mean']:.3f}")
            # 只在完整窗口边界存盘，故无需把缓存轨迹写入检查点。
            # 跨过 save_steps 整除点时延迟保存，最多延迟 updates_per_rollout-1 步。
            crossed_save_boundary = global_step // config.save_steps > window_start_step // config.save_steps
            if crossed_save_boundary or final_step:
                path = save_checkpoint(bundle, optimizer, config, global_step=global_step,
                    epoch=next_epoch, batch_index=next_batch, fingerprint=fingerprint,
                    candidate_index=candidate_index if config.dynamic_sampling else None)
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
