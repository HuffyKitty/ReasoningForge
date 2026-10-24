"""独立数学评测：每题随机采样 n 个回答，再估计 pass@k。

示例：python evaluate.py --config configs/qwen05b_grpo.json \
      --model runs/qwen05b_grpo/final --data data/gsm8k_test.jsonl \
      --num-samples 8 --k 1 8 --output runs/qwen05b_test.json

这是随机采样条件下的 pass@1，不是 greedy accuracy。训练/基线应使用相同
题目、采样温度、回答预算和验证器，否则分数差异可能来自评测设置的变化。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
from typing import Any

from reasoning_forge.rewards import extract_final_answer, extractor_for_format, normalize_answer


PROJECT_ROOT = Path(__file__).resolve().parent


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """n 个回答中 c 个正确，估计随机取 k 个时至少一次正确的概率。

    pass@k = 1 - C(n-c, k) / C(n, k)。这里是常用的无偏估计器，
    不能把 pass@8 写成八次回答的平均正确率；后者对应 pass@1 的估计。
    """

    for name, value in (("n", n), ("c", c), ("k", k)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是整数。")
    if n < 1 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("需要 n>=1，0<=c<=n，1<=k<=n。")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def binary_reward(value: Any) -> float:
    """评测要求正确性为 0/1；训练中的连续 shaping reward 不能直接当准确率。

    先在 Python 数值层验证，再转 torch.float32，避免微小的非二元数被舍入为 1。
    """

    result = float(value)
    if not math.isfinite(result) or result not in (0.0, 1.0):
        raise ValueError(
            f"pass@k 要求奖励函数返回二元正确性 0/1，实际得到 {value!r}。"
            "请在评测配置中使用独立的答案验证器，而非训练时的连续 shaping reward。"
        )
    return result


def baseline_difficulty_map(payload: dict, expected_sha256: str) -> dict[tuple[str, str], tuple[str, str]]:
    """把训练前基座的逐题采样结果冻结成难度标签，不用训练后成绩重新分层。"""
    if payload.get("data", {}).get("sha256") != expected_sha256:
        raise ValueError("基座评估的数据 SHA256 与当前评估数据不同，不能用于难度分层。")
    mapping = {}
    for row in payload.get("samples", []):
        rewards = [binary_reward(value) for value in row["rewards"]]
        if not rewards:
            raise ValueError("基座评估存在没有回答的题目。")
        successes = sum(value == 1.0 for value in rewards)
        level = "unsolved" if successes == 0 else "all_correct" if successes == len(rewards) else "mixed"
        key = row["dataset"], str(row["id"])
        if key in mapping:
            raise ValueError(f"基座评估出现重复题目：{key}")
        mapping[key] = level, row["prompt"]
    if not mapping:
        raise ValueError("基座评估没有逐题 samples。")
    return mapping


def summarize_results(records: list[dict], ks: list[int], max_prompt_length: int | None = None,
                      answer_extractor=extract_final_answer) -> dict:
    """先逐题估计再平均，每题等权；同时按 dataset 分层报告。"""

    if not records:
        raise ValueError("没有评测结果，无法计算 pass@k。")
    if not ks:
        raise ValueError("至少指定一个 k。")

    def summarize_group(group: list[dict]) -> dict:
        totals = {f"pass@{k}": 0.0 for k in ks}
        length_sum, response_count, truncated_count = 0, 0, 0
        all_lengths: list[int] = []
        correct_lengths: list[int] = []
        incorrect_lengths: list[int] = []
        parsed_count = parsed_correct = truncated_without_answer = 0
        answer_diversity_sum = response_repetition_sum = 0.0
        has_completions = all("completions" in record for record in group)
        for record in group:
            rewards = [binary_reward(value) for value in record["rewards"]]
            n, c = len(rewards), sum(value == 1.0 for value in rewards)
            if len(record["lengths"]) != n or len(record["truncated"]) != n:
                raise ValueError("每题的 rewards/lengths/truncated 长度必须一致。")
            for k in ks:
                totals[f"pass@{k}"] += estimate_pass_at_k(n, c, k)
            length_sum += sum(record["lengths"])
            all_lengths.extend(record["lengths"])
            response_count += n
            truncated_count += sum(record["truncated"])
            if has_completions:
                completions = record["completions"]
                if len(completions) != n:
                    raise ValueError("每题 completions 与 rewards 长度必须一致。")
                answers = [answer_extractor(text) for text in completions]
                parsed_count += sum(answer is not None for answer in answers)
                parsed_correct += sum(answer is not None and reward == 1.0 for answer, reward in zip(answers, rewards))
                truncated_without_answer += sum(stop and answer is None for stop, answer in zip(record["truncated"], answers))
                correct_lengths.extend(length for length, reward in zip(record["lengths"], rewards) if reward == 1.0)
                incorrect_lengths.extend(length for length, reward in zip(record["lengths"], rewards) if reward == 0.0)
                answer_diversity_sum += len({normalize_answer(answer) for answer in answers if answer is not None}) / n
                response_repetition_sum += 1 - len({re.sub(r"\s+", " ", text).strip() for text in completions}) / n
        result = {
            "prompts": len(group),
            "responses": response_count,
            **{name: total / len(group) for name, total in totals.items()},
            "mean_response_tokens": length_sum / response_count,
            "truncated_fraction": truncated_count / response_count,
        }
        if has_completions:
            ordered = sorted(all_lengths)
            result.update({
                "answer_extraction_rate": parsed_count / response_count,
                "format_rate": parsed_count / response_count,
                "answer_accuracy_given_extracted": parsed_correct / parsed_count if parsed_count else None,
                "final_answer_diversity": answer_diversity_sum / len(group),
                "response_repetition_fraction": response_repetition_sum / len(group),
                "response_length_p50": ordered[math.ceil(0.50 * response_count) - 1],
                "response_length_p90": ordered[math.ceil(0.90 * response_count) - 1],
                "response_length_p99": ordered[math.ceil(0.99 * response_count) - 1],
                "correct_response_length": sum(correct_lengths) / len(correct_lengths) if correct_lengths else None,
                "incorrect_response_length": sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else None,
                "truncated_unparseable_fraction": truncated_without_answer / response_count,
                "truncated_answer_missing_rate": truncated_without_answer / truncated_count if truncated_count else None,
            })
        if max_prompt_length is not None and all("prompt_tokens" in record for record in group):
            result["prompt_truncation_fraction"] = sum(
                record["prompt_tokens"] > max_prompt_length for record in group
            ) / len(group)
        return result

    datasets = sorted({record["dataset"] for record in records})
    result = {
        "overall": summarize_group(records),
        "by_dataset": {
            dataset: summarize_group([record for record in records if record["dataset"] == dataset])
            for dataset in datasets
        },
    }
    if all("prompt_tokens" in record for record in records):
        def length_bucket(tokens: int) -> str:
            return "0-128" if tokens <= 128 else "129-256" if tokens <= 256 else "257+"
        result["by_prompt_length"] = {
            bucket: summarize_group([record for record in records if length_bucket(record["prompt_tokens"]) == bucket])
            for bucket in ("0-128", "129-256", "257+")
            if any(length_bucket(record["prompt_tokens"]) == bucket for record in records)
        }
    if all("baseline_difficulty" in record for record in records):
        result["by_baseline_difficulty"] = {
            level: summarize_group([record for record in records if record["baseline_difficulty"] == level])
            for level in ("unsolved", "mixed", "all_correct")
            if any(record["baseline_difficulty"] == level for record in records)
        }
    return result


def _project_path(path: Path) -> Path:
    """评测的数据/模型/输出相对项目根目录，方便 PyCharm 使用不同工作目录。"""

    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/qwen05b_grpo.json")
    parser.add_argument("--model", type=Path, help="可选：完整 HF 模型目录或 LoRA adapter 目录，而非 checkpoint 的父目录")
    parser.add_argument("--data", type=Path, help="可选：独立 heldout JSONL；省略则使用 config.data_path")
    parser.add_argument("--num-samples", type=int, default=8, help="每题随机采样回答数 n")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 8], help="要报告的 pass@k，所有 k 必须 <= n")
    parser.add_argument("--limit", type=int, help="仅评测文件中前若干道题，便于小规模检查")
    parser.add_argument("--baseline-results", type=Path,
                        help="可选：同一数据的训练前基座评估 JSON；按基座采样成绩分层")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/evaluation.json", help="新的 JSON 结果文件，拒绝覆盖")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples 必须大于 0。")
    if any(k < 1 or k > args.num_samples for k in args.k):
        parser.error("每个 --k 必须满足 1 <= k <= --num-samples。")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0。")
    output = _project_path(args.output)
    if output.exists():
        parser.error(f"拒绝覆盖已有评测结果：{output}")

    # --help 和上面的纯数学函数都不导入模型框架，更不会触发权重下载。
    from reasoning_forge.checkpoint import data_fingerprint, environment_info, seed_everything
    from reasoning_forge.config import load_config
    from reasoning_forge.data import load_jsonl, render_prompt
    from reasoning_forge.models import load_bundle
    from reasoning_forge.rollout import PytorchRolloutEngine, generation_prompt_batch_size
    from reasoning_forge.trainer import load_reward_function, score_answers

    config = load_config(_project_path(args.config))
    # 只评估 Actor；不创建 Reference、优化器或梯度检查点。
    config = replace(config, beta=0.0, gradient_checkpointing=False)
    data_path = _project_path(args.data) if args.data else Path(config.data_path)
    samples = load_jsonl(data_path)
    total_available = len(samples)
    data_sha256 = data_fingerprint(data_path)
    if args.limit is not None:
        samples = samples[:args.limit]
    baseline = None
    if args.baseline_results:
        baseline_path = _project_path(args.baseline_results)
        baseline = baseline_difficulty_map(
            json.loads(baseline_path.read_text(encoding="utf-8")), data_sha256,
        )
        for sample in samples:
            key = sample.dataset, sample.id
            if key not in baseline or baseline[key][1] != sample.prompt:
                parser.error(f"基座评估缺少同一题面：{key}；请先用同一完整开发集评估基座。")
    model_path = _project_path(args.model) if args.model else None
    if model_path is not None:
        if not model_path.is_dir():
            parser.error(f"--model 必须是存在的本地 HF/adapter 目录：{model_path}")
        is_adapter = (model_path / "adapter_config.json").is_file()
        if is_adapter != config.lora:
            parser.error("--model 的权重类型与 config.lora 不一致：adapter 需要 true，完整模型需要 false。")
    else:
        print("未指定 --model：评估配置中的初始模型；新建的零增量 LoRA 不改变初始输出。")
    if data_path.resolve() == Path(config.data_path).resolve():
        print("评测文件与配置中的训练数据路径一致。正式实验请通过 --data 指定独立 heldout 文件。")

    raw_reward = load_reward_function(config.reward_function)

    def verifier(completion: str, answer: str) -> float:
        return binary_reward(raw_reward(completion, answer))

    seed_everything(config.seed)
    bundle = load_bundle(config, resume_model_path=model_path)
    bundle.actor.eval()
    bundle.actor.requires_grad_(False)
    engine = PytorchRolloutEngine(bundle, config)
    ks = sorted(set(args.k))
    records = []
    evaluation_batch_size = generation_prompt_batch_size(config, args.num_samples)
    for first in range(0, len(samples), evaluation_batch_size):
        current = samples[first:first + evaluation_batch_size]
        prompts = [render_prompt(sample, bundle.tokenizer, config) for sample in current]
        prompt_lengths = [len(row) for row in bundle.tokenizer(
            prompts, add_special_tokens=False, truncation=False, padding=False,
        )["input_ids"]]
        # 与训练共用模型、模板和 rollout，避免评测另写一套生成流程后发生格式漂移。
        # 此处虽然不训练，rollout 仍计算 old_logps，便于保持统一接口；后续可优化吞吐。
        rollout = engine.rollout(prompts, num_generations=args.num_samples)
        scores = score_answers(current, rollout.completions, args.num_samples, verifier).tolist()
        lengths = rollout.completion_mask.sum(-1).tolist()
        truncated = rollout.truncated.tolist()
        for index, sample in enumerate(current):
            start, end = index * args.num_samples, (index + 1) * args.num_samples
            record = {
                "id": sample.id,
                "dataset": sample.dataset,
                "prompt": sample.prompt,
                "answer": sample.answer,
                "prompt_tokens": prompt_lengths[index],
                "completions": rollout.completions[start:end],
                "rewards": scores[start:end],
                "lengths": lengths[start:end],
                "truncated": truncated[start:end],
            }
            if baseline is not None:
                record["baseline_difficulty"] = baseline[(sample.dataset, sample.id)][0]
            records.append(record)
        print(f"已评测 {len(records)}/{len(samples)} 道题")
        del rollout

    metrics = summarize_results(records, ks, max_prompt_length=config.max_prompt_length,
                                answer_extractor=extractor_for_format(config.answer_format))
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_description": "Stochastic pass@k estimated by 1-C(n-c,k)/C(n,k); equal weight per prompt. pass@1 is not greedy accuracy.",
        "config": config.to_dict(),
        "model": str(model_path) if model_path else config.model_name_or_path,
        "model_kind": "lora_adapter" if config.lora else "full_model",
        "data": {"path": str(data_path), "sha256": data_sha256, "total_available": total_available, "evaluated": len(samples)},
        "num_samples": args.num_samples,
        "k": ks,
        "environment": environment_info(),
        "metrics": metrics,
        "samples": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # x 模式再次防止并发运行时覆盖另一份刚生成的结果。
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"评测结果：{output}")


if __name__ == "__main__":
    main()
