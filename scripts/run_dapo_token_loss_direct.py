"""Run DAPO-Math token-loss training, then final evaluation, without base evaluation.

This entry point reuses a validated throughput32 experiment_config.json produced
by the portable launcher. It starts from the original model, or resumes the
latest complete checkpoint in the same training directory after interruption.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portable.benchmark import (  # noqa: E402
    configured_run, expected_updates, finished_training, preflight, run_logged,
    sha256, valid_evaluation, validate_data, write_json,
)

NAME = "dapo_math_token_loss_g4_throughput32"
RUN = ROOT / "runs" / NAME
CONFIG_PATH = RUN / "experiment_config.json"
TRAIN = RUN / "train"
LOGS = ROOT / "logs" / NAME


def check_configuration() -> tuple[dict, int]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected, name = configured_run("token_loss", smoke=False, max_steps=None,
                                    performance_profile="throughput32")
    if name != NAME or config != expected:
        raise ValueError("The experiment config is not the pinned full token-loss throughput32 plan")
    data = validate_data(ROOT)
    total = expected_updates(data["train"]["count"], config)
    if total != 4098:
        raise ValueError(f"Unexpected training update budget: {total}")
    return config, total


def status(stage: str, **fields) -> None:
    write_json(RUN / "status.json", {
        "stage": stage, "variant": "token_loss", "performance_profile": "throughput32",
        "base_evaluation": "skipped; use the matching GRPO baseline result from the other host",
        "pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat(), **fields,
    })


def run() -> None:
    import fcntl

    RUN.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / "runs/.portable_gpu.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another portable experiment is using this GPU") from error
        try:
            config, total = check_configuration()
            plan = preflight(config, validate_data(ROOT))
            if plan["optimizer_updates"] != total:
                raise ValueError("Preflight update budget changed")
            environment = os.environ
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            environment.update({
                "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
                "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "HF_HOME": str(ROOT / ".cache/huggingface"),
                "TOKENIZERS_PARALLELISM": "false",
            })
            interpreter = sys.executable
            if not finished_training(TRAIN, total):
                checkpoints = sorted(path for path in TRAIN.glob("checkpoint-*")
                                     if (path / "COMPLETE").is_file())
                if TRAIN.exists() and any(TRAIN.iterdir()) and not checkpoints:
                    raise RuntimeError("Training output exists without a complete checkpoint; use a new run")
                command = [interpreter, "-u", str(ROOT / "train.py"), "--config", str(CONFIG_PATH)]
                if checkpoints:
                    command += ["--resume", str(checkpoints[-1])]
                status("training", plan=plan, resume=str(checkpoints[-1]) if checkpoints else None)
                run_logged(command, LOGS / "training.log")
            if not finished_training(TRAIN, total):
                raise RuntimeError("Training ended without the final complete checkpoint and adapter")

            output = RUN / "trained_test_n4.json"
            test_sha = validate_data(ROOT)["test"]["sha256"]
            if not valid_evaluation(output, test_sha, 500, config):
                status("trained_evaluation", plan=plan)
                run_logged([
                    interpreter, "-u", str(ROOT / "evaluate.py"),
                    "--config", str(CONFIG_PATH), "--data", str(ROOT / "data/dapo_math_17k_test.jsonl"),
                    "--num-samples", "4", "--k", "1", "4",
                    "--model", str(TRAIN / "final"), "--output", str(output),
                ], LOGS / "trained_evaluation.log")
            payload = json.loads(output.read_text(encoding="utf-8"))
            write_json(RUN / "trained_only_summary.json", {
                "variant": "token_loss", "performance_profile": "throughput32",
                "base_evaluation": "skipped", "training_steps": total,
                "evaluation_data_sha256": test_sha, "trained_model": payload["metrics"]["overall"],
            })
            write_json(RUN / "results_sha256.json", {
                "trained_test_n4.json": sha256(output),
                "trained_only_summary.json": sha256(RUN / "trained_only_summary.json"),
                "train/final/adapter_model.safetensors": sha256(TRAIN / "final/adapter_model.safetensors"),
            })
            status("complete", report=str(RUN / "trained_only_summary.json"))
        except BaseException as error:
            status("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        _, total = check_configuration()
        print(json.dumps({"run": str(RUN), "optimizer_updates": total,
                          "base_evaluation": "skipped", "performance_profile": "throughput32"}))
        return
    if args.detach:
        LOGS.mkdir(parents=True, exist_ok=True)
        path = LOGS / "direct_supervisor.log"
        with path.open("a", encoding="utf-8") as output:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Started direct training supervisor PID {process.pid}. Log: {path}")
        return
    run()


if __name__ == "__main__":
    main()
