"""Train the unmodified GRPO baseline first; other variants require an explicit flag."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "grpo": "qwen15b_dapo_math_17k_baseline_full.json",
    "clip_higher": "qwen15b_dapo_math_17k_clip_higher_full.json",
    "token_loss": "qwen15b_dapo_math_17k_token_loss_full.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def expected_updates(count: int, config: dict) -> int:
    updates = math.ceil(count / (config["prompt_batch_size"] * config["gradient_accumulation_steps"]))
    updates *= config["epochs"] * config["updates_per_rollout"]
    return min(updates, config["max_steps"]) if config["max_steps"] else updates


def validate_data(root: Path) -> dict:
    from reasoning_forge.data import load_jsonl
    manifest = json.loads((root / "data/dapo_math_17k_manifest.json").read_text(encoding="utf-8"))
    seen_ids, seen_prompts, result = set(), set(), {}
    for split in ("train", "dev", "test"):
        path = root / f"data/dapo_math_17k_{split}.jsonl"
        expected = manifest["splits"][split]
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Dataset SHA-256 mismatch: {split}")
        samples = load_jsonl(path)
        if len(samples) != expected["count"]:
            raise ValueError(f"Dataset count mismatch: {split}")
        for sample in samples:
            identifier = sample.dataset, sample.id
            prompt = sample.prompt.strip()
            if identifier in seen_ids or prompt in seen_prompts:
                raise ValueError(f"Duplicate prompt/id across dataset splits: {identifier}")
            seen_ids.add(identifier)
            seen_prompts.add(prompt)
        result[split] = {"count": len(samples), "sha256": expected["sha256"]}
    return result


def configured_run(variant: str, *, smoke: bool, max_steps: int | None,
                   performance_profile: str = "original") -> tuple[dict, str]:
    config = json.loads((ROOT / "configs" / CONFIGS[variant]).read_text(encoding="utf-8"))
    config.update({
        "model_name_or_path": str(ROOT / "models/Qwen2.5-1.5B-Instruct"),
        "local_files_only": True, "device": "cuda", "num_generations": 4,
        "eval_num_samples": 4, "max_prompt_length": 2048,
    })
    name = f"dapo_math_{variant}_g4"
    if performance_profile not in {"original", "throughput16", "throughput32"}:
        raise ValueError(f"Unknown performance profile: {performance_profile}")
    if performance_profile != "original":
        config.update({"rollout_batch_across_window": True,
                       "rollout_micro_batch_size": int(performance_profile.removeprefix("throughput"))})
        name += f"_{performance_profile}"
    if smoke:
        name += "_smoke"
        config.update({"max_steps": 2, "save_steps": 2, "log_steps": 1, "eval_steps": 0,
                       "max_new_tokens": 64, "prompt_batch_size": 1, "rollout_micro_batch_size": 4,
                       "train_micro_batch_size": 1, "gradient_accumulation_steps": 1})
    elif max_steps is not None:
        name += f"_steps{max_steps}"
        config["max_steps"] = max_steps
    config["output_dir"] = str(ROOT / "runs" / name / "train")
    return config, name


def preflight(config: dict, data: dict) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available. Install an NVIDIA driver and expose the RTX 4090 to this process.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This benchmark requires an NVIDIA GPU with BF16 support.")
    properties = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    if total < 20 * 1024**3 or free < 18 * 1024**3:
        raise RuntimeError(f"The RTX 4090 profile needs at least 20 GiB total and 18 GiB free VRAM; "
                           f"detected {total / 1024**3:.1f} GiB total, {free / 1024**3:.1f} GiB free. "
                           "No training has started.")
    from transformers import AutoTokenizer
    from reasoning_forge.config import TrainConfig
    from reasoning_forge.data import load_jsonl, render_prompt
    from scripts.check_prompt_lengths import summarize_prompt_lengths
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], local_files_only=True)
    settings = TrainConfig(**config)
    lengths = {}
    for split in ("train", "dev", "test"):
        samples = load_jsonl(ROOT / f"data/dapo_math_17k_{split}.jsonl")
        lengths[split] = summarize_prompt_lengths(
            [render_prompt(sample, tokenizer, settings) for sample in samples],
            tokenizer, config["max_prompt_length"],
        )
        if lengths[split]["truncated_count"]:
            raise ValueError(f"{split} contains prompts exceeding the configured length; training did not start")
    plan = {
        "gpu": properties.name, "total_vram_gib": total / 1024**3,
        "free_vram_gib": free / 1024**3, "torch": torch.__version__, "cuda": torch.version.cuda,
        "train_prompts": data["train"]["count"], "dev_prompts": data["dev"]["count"],
        "test_prompts": data["test"]["count"], "epochs": config["epochs"],
        "samples_per_prompt": 4, "optimizer_updates": expected_updates(data["train"]["count"], config),
        "answer_token_limit": config["max_new_tokens"],
        "prompt_token_limit": config["max_prompt_length"], "prompt_lengths": lengths,
        "evaluation": "Same held-out 500 test prompts, four stochastic responses each, pass@1/pass@4",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    return plan


def run_logged(command: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {' '.join(command)}\nLog: {path}", flush=True)
    with path.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Stage exited with code {code}; details: {path}")


def valid_evaluation(path: Path, test_sha: str, limit: int, config: dict) -> bool:
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["data"]["sha256"] != test_sha or result["data"]["evaluated"] != limit:
        raise ValueError(f"Existing result uses a different evaluation dataset: {path}")
    if result["num_samples"] != 4 or result["k"] != [1, 4]:
        raise ValueError(f"Existing result uses a different sampling protocol: {path}")
    fields = ("revision", "temperature", "top_p", "top_k", "max_new_tokens", "max_prompt_length",
              "prompt_format", "system_prompt", "chat_template_kwargs", "reward_function", "answer_format", "seed")
    if any(result["config"].get(field) != config.get(field) for field in fields):
        raise ValueError(f"Existing result uses a different generation protocol: {path}")
    return True


def finished_training(path: Path, expected: int) -> bool:
    checkpoint = path / f"checkpoint-{expected:06d}"
    return ((checkpoint / "COMPLETE").is_file() and
            (path / "final/adapter_model.safetensors").is_file() and
            (path / "final/tokenizer_config.json").is_file())


def main() -> int:
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(CONFIGS), default="grpo")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--performance-profile", choices=["original", "throughput16", "throughput32"],
                        default="original")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.resume_from and (args.performance_profile == "original" or args.smoke_test):
        parser.error("--resume-from requires a throughput profile and a non-smoke run")
    config, name = configured_run(args.variant, smoke=args.smoke_test, max_steps=args.max_steps,
                                  performance_profile=args.performance_profile)
    run = ROOT / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "runs/.portable_gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another portable benchmark is already using this package's GPU queue.")
    started = time.time()
    status_path = run / "status.json"

    def status(stage: str, **fields) -> None:
        write_json(status_path, {"stage": stage, "variant": args.variant, "pid": os.getpid(),
                                "utc": datetime.now(timezone.utc).isoformat(), **fields})

    try:
        status("checking")
        data = validate_data(ROOT)
        from reasoning_forge.config import TrainConfig
        TrainConfig(**config).validate()
        if args.variant == "grpo" and (config["dynamic_sampling"] or config["clip_high"] != config["clip_low"]
                                      or config["loss_reduction"] != "sequence"):
            raise ValueError("The requested benchmark is standard GRPO; DAPO improvements must remain disabled")
        plan = preflight(config, data)
        write_json(run / "plan.json", plan)
        config_path = run / "experiment_config.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Run configuration changed. Use a fresh package/run directory: {run}")
        write_json(config_path, config)
        if args.resume_from:
            from reasoning_forge.checkpoint import load_training_state
            from reasoning_forge.config import load_config
            external = args.resume_from.expanduser().resolve()
            state = load_training_state(external, load_config(config_path), data["train"]["sha256"],
                                        allow_rollout_tuning=True)
            if state["global_step"] >= plan["optimizer_updates"]:
                raise ValueError("The target total step budget must exceed the resume checkpoint step")
            provenance = {"checkpoint": str(external), "step": state["global_step"],
                          "note": "Explicit generation scheduling migration; optimizer and data cursor retained."}
            provenance_path = run / "resume_source.json"
            existing_training = Path(config["output_dir"])
            if not provenance_path.exists() and existing_training.exists() and any(existing_training.iterdir()):
                raise ValueError("Cannot attach an external checkpoint to an existing training run; use a fresh run directory")
            if provenance_path.exists() and json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
                raise ValueError("This run already uses another resume checkpoint; use a fresh run directory")
            write_json(provenance_path, provenance)
        if args.check_only:
            status("check_complete", plan=plan)
            return 0
        interpreter = sys.executable
        logs = ROOT / "logs" / name
        evaluation_count = 2 if args.smoke_test else data["test"]["count"]
        shared = ["--config", str(config_path), "--data", str(ROOT / "data/dapo_math_17k_test.jsonl"),
                  "--num-samples", "4", "--k", "1", "4"]
        if args.smoke_test:
            shared += ["--limit", "2"]
        base = run / "base_test_n4.json"
        if not valid_evaluation(base, data["test"]["sha256"], evaluation_count, config):
            status("base_evaluation", plan=plan)
            run_logged([interpreter, "-u", str(ROOT / "evaluate.py"), *shared, "--output", str(base)],
                       logs / "base_evaluation.log")
        training = Path(config["output_dir"])
        if not finished_training(training, plan["optimizer_updates"]):
            command = [interpreter, "-u", str(ROOT / "train.py"), "--config", str(config_path)]
            checkpoints = sorted(path for path in training.glob("checkpoint-*") if (path / "COMPLETE").is_file())
            if checkpoints:
                command += ["--resume", str(checkpoints[-1])]
            elif training.exists() and any(training.iterdir()):
                raise RuntimeError(f"Training was interrupted before its first checkpoint. Preserve {training} "
                                   "and start in a fresh run directory; existing results are not overwritten.")
            elif args.resume_from:
                command += ["--resume", str(external), "--allow-rollout-tuning"]
            status("training", plan=plan, resume=str(checkpoints[-1]) if checkpoints else
                   str(external) if args.resume_from else None)
            run_logged(command, logs / "training.log")
        if not finished_training(training, plan["optimizer_updates"]):
            raise RuntimeError("Training exited without the complete final checkpoint and adapter")
        trained = run / "trained_test_n4.json"
        if not valid_evaluation(trained, data["test"]["sha256"], evaluation_count, config):
            status("trained_evaluation", plan=plan)
            run_logged([interpreter, "-u", str(ROOT / "evaluate.py"), *shared,
                        "--model", str(training / "final"), "--baseline-results", str(base),
                        "--output", str(trained)], logs / "trained_evaluation.log")
        before = json.loads(base.read_text(encoding="utf-8"))["metrics"]["overall"]
        after = json.loads(trained.read_text(encoding="utf-8"))["metrics"]["overall"]
        report = {"variant": args.variant, "smoke_test_only": args.smoke_test,
                  "plan": plan, "base_model": before, "trained_model": after,
                  "delta_percentage_points": {f"pass@{k}": 100 * (after[f"pass@{k}"] - before[f"pass@{k}"])
                                               for k in (1, 4)},
                  "wall_seconds_this_invocation": time.time() - started,
                  "evaluation_data_sha256": data["test"]["sha256"],
                  "model_revision": config["revision"]}
        write_json(run / "benchmark_comparison.json", report)
        checksums = {str(path.relative_to(run)): sha256(path)
                     for path in [base, trained, run / "benchmark_comparison.json", training / "final/adapter_model.safetensors"]}
        write_json(run / "results_sha256.json", checksums)
        status("complete", report=str(run / "benchmark_comparison.json"))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"Benchmark complete. Results and model: {run}", flush=True)
        return 0
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
