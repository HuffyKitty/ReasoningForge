"""Continue a complete local checkpoint with faster rollout scheduling.

Use the existing environment and original checkpoint config. This entry point
does not download assets, reinstall packages, or repeat the baseline evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def make_resume_config(checkpoint: Path, run_dir: Path, *, parallel: int = 16,
                       total_steps: int | None = None) -> dict:
    if parallel not in {16, 32}:
        raise ValueError("parallel must be 16 or 32")
    if total_steps is not None and total_steps < 1:
        raise ValueError("total_steps must be positive")
    for name in ("COMPLETE", "config.json", "trainer_state.pt"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Incomplete checkpoint: {checkpoint / name}")
    if run_dir.exists():
        raise FileExistsError(f"Use a new run directory: {run_dir}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("dynamic_sampling", False):
        raise ValueError("Merged rollout scheduling is not implemented for Dynamic Sampling")
    config.update({"output_dir": str(run_dir / "train"),
                   "rollout_batch_across_window": True, "rollout_micro_batch_size": parallel})
    if total_steps is not None:
        config["max_steps"] = total_steps
    return config


VALIDATE = r'''
import json, math, sys
from pathlib import Path
from reasoning_forge.checkpoint import data_fingerprint, load_training_state
from reasoning_forge.config import load_config
from reasoning_forge.data import load_jsonl
checkpoint, config_path = map(Path, sys.argv[1:])
config = load_config(config_path)
samples = load_jsonl(config.data_path)
state = load_training_state(checkpoint, config, data_fingerprint(config.data_path), allow_rollout_tuning=True)
for name in ("optimizer", "python_rng", "torch_rng", "cuda_rng", "epoch", "batch_index", "global_step"):
    if name not in state:
        raise ValueError("Checkpoint cannot restore full training state: missing " + name)
weight = "adapter_model.safetensors" if config.lora else "model.safetensors"
if not (checkpoint / "model" / weight).is_file():
    raise ValueError("Missing checkpoint model weights: " + weight)
windows = math.ceil(math.ceil(len(samples) / config.prompt_batch_size) / config.gradient_accumulation_steps)
available = windows * config.epochs * config.updates_per_rollout
target = min(available, config.max_steps) if config.max_steps else available
if state["global_step"] >= target:
    raise ValueError("Requested total update budget has already been reached")
if state["epoch"] >= config.epochs:
    raise ValueError("Dataset epoch cursor has already reached the configured epoch limit")
print(json.dumps({"resume_step": state["global_step"], "target_total_steps": target,
    "remaining_steps": target - state["global_step"], "epoch_cursor": state["epoch"],
    "batch_cursor": state["batch_index"], "train_prompts": len(samples),
    "prompts_per_update": config.prompt_batch_size * config.gradient_accumulation_steps,
    "samples_per_prompt": config.num_generations, "parallel_answers": config.rollout_micro_batch_size,
    "checkpoint": str(checkpoint)}, ensure_ascii=False, indent=2))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parallel", type=int, choices=[16, 32], default=16)
    parser.add_argument("--total-steps", type=int, help="Total cumulative optimizer steps; omitted preserves the checkpoint budget")
    parser.add_argument("--run-dir", type=Path, help="New directory for configuration and resumed training outputs")
    parser.add_argument("--prepare-only", action="store_true", help="Validate the checkpoint on CPU and print the training command")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    run_dir = (args.run_dir.expanduser().resolve() if args.run_dir else
               ROOT / "runs" / f"fast_resume_{datetime.now():%Y%m%d_%H%M%S_%f}_p{args.parallel}")
    config = make_resume_config(checkpoint, run_dir, parallel=args.parallel, total_steps=args.total_steps)
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "experiment_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "resume_request.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        "parallel": args.parallel, "total_steps_override": args.total_steps,
        "purpose": "Continue optimizer, data cursor and RNG state in a new output directory",
    }, indent=2) + "\n", encoding="utf-8")
    interpreter = next((candidate for candidate in (ROOT / ".venv/bin/python", ROOT / ".conda/bin/python")
                        if candidate.is_file()), Path(sys.executable))
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
                        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    subprocess.run([str(interpreter), "-c", VALIDATE, str(checkpoint), str(config_path)],
                   cwd=ROOT, env=environment, check=True)
    command = [str(interpreter), "-u", str(ROOT / "train.py"), "--config", str(config_path),
               "--resume", str(checkpoint), "--allow-rollout-tuning"]
    print("Configuration:", config_path, flush=True)
    print("Training command:", shlex.join(command), flush=True)
    if args.prepare_only:
        return
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
