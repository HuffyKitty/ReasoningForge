"""Generate isolated GSM8K Dynamic Sampling and DAPO-Math pilot configurations."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/qwen15b_gsm8k_full_4090_fast_v2.json"


def write(name: str, values: dict) -> Path:
    path = ROOT / "configs" / name
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    source = json.loads(BASE.read_text(encoding="utf-8"))
    gsm_dynamic = {
        **source,
        "output_dir": "runs/qwen15b_gsm8k_dynamic_sampling_full",
        "dynamic_sampling": True,
        "dynamic_sampling_max_candidates_factor": 16,
    }
    print(write("qwen15b_gsm8k_dynamic_sampling_full.json", gsm_dynamic))
    print(write("qwen15b_gsm8k_dynamic_sampling_smoke.json", {
        **gsm_dynamic,
        "output_dir": "runs/qwen15b_gsm8k_dynamic_sampling_smoke",
        "max_steps": 2,
        "save_steps": 2,
        "eval_steps": 0,
        "log_steps": 1,
    }))

    dapo = {
        **source,
        "data_path": "data/dapo_math_17k_train.jsonl",
        "dev_data_path": "data/dapo_math_17k_dev.jsonl",
        "output_dir": "runs/qwen15b_dapo_math_17k_baseline_pilot",
        "system_prompt": "",
        "answer_format": "answer_line",
        "reward_function": "reasoning_forge.rewards:dapo_math_reward",
        "num_generations": 4,
        "max_prompt_length": 1024,
        "max_new_tokens": 1024,
        "rollout_micro_batch_size": 8,
        "train_micro_batch_size": 2,
        "max_steps": 200,
        "save_steps": 100,
        "eval_steps": 100,
        "dynamic_sampling": False,
        "clip_low": 0.2,
        "clip_high": 0.2,
        "loss_reduction": "sequence",
    }
    print(write("qwen15b_dapo_math_17k_baseline_pilot.json", dapo))
    print(write("qwen15b_dapo_math_17k_clip_higher_pilot.json", {
        **dapo,
        "output_dir": "runs/qwen15b_dapo_math_17k_clip_higher_pilot",
        "clip_high": 0.28,
    }))
    print(write("qwen15b_dapo_math_17k_token_loss_pilot.json", {
        **dapo,
        "output_dir": "runs/qwen15b_dapo_math_17k_token_loss_pilot",
        "loss_reduction": "token",
    }))
    # Full one-epoch configs are prepared for the later controlled comparison.
    # The queued job runs only the baseline pilot until feasibility is observed.
    full = {**dapo, "max_steps": 0, "save_steps": 200, "eval_steps": 200}
    print(write("qwen15b_dapo_math_17k_baseline_full.json", {
        **full, "output_dir": "runs/qwen15b_dapo_math_17k_baseline_full",
    }))
    print(write("qwen15b_dapo_math_17k_clip_higher_full.json", {
        **full, "output_dir": "runs/qwen15b_dapo_math_17k_clip_higher_full", "clip_high": 0.28,
    }))
    print(write("qwen15b_dapo_math_17k_token_loss_full.json", {
        **full, "output_dir": "runs/qwen15b_dapo_math_17k_token_loss_full", "loss_reduction": "token",
    }))


if __name__ == "__main__":
    main()
