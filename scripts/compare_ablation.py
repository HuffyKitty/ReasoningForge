"""Compare two GRPO/DAPO runs on the same held-out evaluation set."""

import argparse
import json
from pathlib import Path


def train_summary(run: Path, max_step: int | None = None) -> dict:
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if max_step is not None:
        rows = [row for row in rows if row["step"] <= max_step]
    if not rows:
        raise ValueError(f"训练指标为空：{run}")
    if max_step is not None and rows[-1]["step"] != max_step:
        raise ValueError(f"{run} 缺少第 {max_step} 步的训练指标")
    return {
        "steps": rows[-1]["step"],
        "effective_prompts": sum(row.get("prompts", 0) for row in rows),
        "sampled_prompts": sum(row.get("sampled_prompts", row.get("prompts", 0)) for row in rows),
        "generated_response_tokens": sum(row.get("generated_tokens", 0) for row in rows),
        "recorded_training_seconds_excluding_eval_and_save": sum(row.get("seconds", 0) for row in rows),
    }


def compare(baseline_run: Path, variant_run: Path, baseline_eval: Path, variant_eval: Path,
            baseline_step: int | None = None) -> dict:
    base = json.loads(baseline_eval.read_text(encoding="utf-8"))
    variant = json.loads(variant_eval.read_text(encoding="utf-8"))
    for field in ("sha256", "evaluated"):
        if base["data"][field] != variant["data"][field]:
            raise ValueError(f"两次评估的 {field} 不同，不能直接比较")
    if base["num_samples"] != variant["num_samples"] or base["k"] != variant["k"]:
        raise ValueError("评估采样数或 pass@k 设置不同")
    first = base["metrics"]["overall"]
    second = variant["metrics"]["overall"]
    result = {
        "evaluation_sha256": base["data"]["sha256"],
        "evaluated_prompts": base["data"]["evaluated"],
        "num_samples_per_prompt": base["num_samples"],
        "baseline": {"train": train_summary(baseline_run, baseline_step), "eval": first},
        "variant": {"train": train_summary(variant_run), "eval": second},
        "delta_percentage_points": {
            f"pass@{k}": 100 * (second[f"pass@{k}"] - first[f"pass@{k}"])
            for k in base["k"]
        },
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--variant-run", type=Path, required=True)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--variant-eval", type=Path, required=True)
    parser.add_argument("--baseline-step", type=int,
                        help="只统计基线到该检查点的训练指标，用于中途停止的训练")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline_run, args.variant_run, args.baseline_eval,
                     args.variant_eval, baseline_step=args.baseline_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["delta_percentage_points"], ensure_ascii=False))
