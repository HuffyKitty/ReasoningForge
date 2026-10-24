"""Compare generation concurrency on a fixed window without updating model weights.

Run with the bundle's .venv/bin/python. By default the frozen reference is loaded
so memory measurements reflect the training rollout phase. No training, test-set
evaluation, environment installation, or network access is performed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import torch
    from reasoning_forge.checkpoint import seed_everything
    from reasoning_forge.config import load_config
    from reasoning_forge.data import load_jsonl, render_prompt
    from reasoning_forge.models import load_bundle
    from reasoning_forge.rollout import PytorchRolloutEngine
    from reasoning_forge.trainer import load_reward_function, prepare_window

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen15b_dapo_math_17k_baseline_full.json")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--without-reference", action="store_true", help="Inference-only test; not training VRAM validation")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--profiles", nargs="+", choices=["serial8", "throughput16", "throughput32"],
                        default=["serial8", "throughput16", "throughput32"])
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / f"rollout_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.repeats < 1 or (args.cpu_threads is not None and args.cpu_threads < 1):
        parser.error("Token budget, repeats and CPU threads must be positive")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite: {output}")
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for this speed test")
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    config = load_config(args.config)
    config = replace(config, model_name_or_path=str(ROOT / "models/Qwen2.5-1.5B-Instruct"),
                     device="cuda", local_files_only=True, max_prompt_length=2048,
                     max_new_tokens=args.max_new_tokens, gradient_checkpointing=False,
                     beta=0.0 if args.without_reference else config.beta)
    samples = load_jsonl(config.data_path)
    indices = list(range(len(samples)))
    random.Random(config.seed).shuffle(indices)
    count = config.prompt_batch_size * config.gradient_accumulation_steps
    chosen = [samples[index] for index in indices[:count]]
    window = [chosen[index:index + config.prompt_batch_size]
              for index in range(0, len(chosen), config.prompt_batch_size)]
    seed_everything(config.seed)
    bundle = load_bundle(config)
    reward_function = load_reward_function(config.reward_function)
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cpu_threads": torch.get_num_threads(), "config": config.to_dict(),
              "reference_loaded": bundle.reference is not None,
              "note": "No optimizer updates. Timing uses CUDA synchronization. Each profile samples fresh answers; token counts can differ.",
              "prompt_ids": [sample.id for sample in chosen], "results": {}}
    report["note"] += " Serial profile also uses patched microbatch probability scoring. This test excludes optimizer/backward memory."
    warmup = PytorchRolloutEngine(bundle, config)
    warmup.rollout([render_prompt(chosen[0], bundle.tokenizer, config)], num_generations=1, max_new_tokens=8)
    for profile in args.profiles:
        setting = replace(config, rollout_batch_across_window=profile != "serial8",
                          rollout_micro_batch_size=8 if profile == "serial8" else int(profile[len("throughput"):]))
        rows = []
        try:
            engine = PytorchRolloutEngine(bundle, setting)
            for repeat in range(args.repeats):
                seed_everything(config.seed + repeat)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
                prepared = prepare_window(window, bundle, engine, reward_function, setting)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - begin
                tokens = sum(int(item.rollout.completion_mask.sum()) for item in prepared)
                generation = sum(item.rollout.timings["generation_seconds"] for item in prepared)
                rows.append({"wall_seconds": seconds, "generated_tokens": tokens,
                             "generation_seconds": generation, "generation_tokens_per_second": tokens / generation,
                             "rollout_tokens_per_second": tokens / seconds,
                             "old_logp_seconds": sum(item.rollout.timings["old_logp_seconds"] for item in prepared),
                             "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                             "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
                print(json.dumps({"profile": profile, "repeat": repeat, **rows[-1]}, ensure_ascii=False), flush=True)
                del prepared
            report["results"][profile] = {"runs": rows,
                "median_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
                "median_generation_tokens_per_second": statistics.median(row["generation_tokens_per_second"] for row in rows)}
        except torch.cuda.OutOfMemoryError as error:
            if "prepared" in locals():
                del prepared
            report["results"][profile] = {"error": "CUDA out of memory", "detail": str(error), "runs": rows}
            print(f"{profile}: CUDA out of memory; try a smaller concurrency", flush=True)
        finally:
            torch.cuda.empty_cache()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Speed report: {output}", flush=True)


if __name__ == "__main__":
    main()
