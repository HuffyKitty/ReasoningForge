"""Standard-library-only entry point for a Linux RTX 4090 offline bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside_root(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute():
        raise ValueError(f"Path escapes package root: {relative}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"Path escapes package root: {relative}")
    return path


def verify_manifest(root: Path, manifest: dict) -> None:
    for relative, expected in manifest["files"].items():
        path = inside_root(root, relative)
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError(f"Missing or incomplete package asset: {relative}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Package SHA-256 mismatch: {relative}")


def verify_assets() -> None:
    manifest = ROOT / "package_manifest.json"
    if manifest.is_file():
        print("Checking package SHA-256 manifest...", flush=True)
        verify_manifest(ROOT, json.loads(manifest.read_text(encoding="utf-8")))
        print("Package checksum verification passed.", flush=True)
        return
    sources = json.loads((ROOT / "portable/asset_sources.json").read_text(encoding="utf-8"))
    model = ROOT / sources["model_directory"]
    model_manifest = model / "source_manifest.json"
    if not model_manifest.is_file():
        raise FileNotFoundError("Model assets are not prepared. Use the complete exported bundle.")
    data = json.loads(model_manifest.read_text(encoding="utf-8"))
    if data["revision"] != sources["model_revision"]:
        raise ValueError("Model revision does not match the portable configuration")
    verify_manifest(model, data)
    archive = ROOT / sources["python_archive"]
    if not archive.is_file() or sha256(archive) != sources["python_sha256"]:
        raise ValueError("Bundled Python archive is missing or corrupt")
    wheels = ROOT / sources["wheel_directory"]
    wheel_manifest = json.loads((wheels / "manifest.json").read_text(encoding="utf-8"))
    for entry in wheel_manifest["packages"].values():
        if sha256(inside_root(wheels, entry["filename"])) != entry["sha256"]:
            raise ValueError(f"Corrupt wheel: {entry['filename']}")
    print("Model, Python runtime and wheel checksums passed.", flush=True)


def extract_python(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != "python":
                raise ValueError(f"Unsafe runtime archive entry: {member.name}")
            if member.isdev() or member.isfifo():
                raise ValueError(f"Unsupported runtime archive entry: {member.name}")
            inside_root(destination, member.name)
            if member.issym() or member.islnk():
                target = (Path(member.name).parent / member.linkname) if member.issym() else Path(member.linkname)
                inside_root(destination, str(target))
        handle.extractall(destination, members=members)


def child_environment() -> dict:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update({
        "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(ROOT / ".cache/huggingface"), "TOKENIZERS_PARALLELISM": "false",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    })
    return environment


def ensure_environment(environment: dict) -> Path:
    sources = json.loads((ROOT / "portable/asset_sources.json").read_text(encoding="utf-8"))
    base = ROOT / ".runtime/python/bin/python3.11"
    runtime_marker = ROOT / ".runtime/COMPLETE"
    if not base.is_file() or not runtime_marker.is_file():
        print("Extracting bundled CPython 3.11...", flush=True)
        extract_python(ROOT / sources["python_archive"], ROOT / ".runtime")
        runtime_marker.write_text(sources["python_sha256"] + "\n", encoding="utf-8")
    # A venv is recreated on the recipient computer; its installation prefix is
    # never copied from the author's machine or shared with system site-packages.
    venv_path = ROOT / ".venv"
    interpreter = venv_path / "bin/python"
    lock = ROOT / "portable/requirements-linux.lock"
    signature = {"root": str(ROOT), "lock_sha256": sha256(lock), "python_sha256": sources["python_sha256"]}
    receipt = venv_path / "reasoningforge-environment.json"
    if not interpreter.is_file():
        subprocess.run([str(base), "-m", "venv", str(venv_path)], check=True, env=environment)
    if not receipt.is_file() or json.loads(receipt.read_text(encoding="utf-8")) != signature:
        print("Installing the pinned CUDA 12.6 environment from local wheels...", flush=True)
        subprocess.run([str(interpreter), "-m", "pip", "install", "--no-index", "--find-links",
                        str(ROOT / sources["wheel_directory"]), "--require-hashes", "-r", str(lock)],
                       check=True, env=environment)
        subprocess.run([str(interpreter), "-m", "pip", "check"], check=True, env=environment)
        receipt.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    return interpreter


def prepare_environment(environment: dict) -> Path:
    import fcntl
    lock = (ROOT / ".bootstrap.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Environment setup is already running in this folder. Wait for that window to finish.")
    try:
        return ensure_environment(environment)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def check_platform() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("This offline bundle targets Linux x86_64 and NVIDIA RTX 4090. "
                           "Use --verify-package to inspect it on another operating system.")
    libc, version = platform.libc_ver()
    if libc == "glibc" and version and tuple(map(int, version.split(".")[:2])) < (2, 28):
        raise RuntimeError(f"glibc >= 2.28 is required; detected {version}")


def show_status() -> None:
    files = sorted((ROOT / "runs").glob("dapo_math_*_g4*/status.json"))
    if not files:
        print("No portable benchmark has started in this folder.")
    for path in files:
        print(path.relative_to(ROOT))
        print(path.read_text(encoding="utf-8"))
        metrics = path.parent / "train/metrics.jsonl"
        if metrics.is_file():
            with metrics.open("rb") as handle:
                handle.seek(max(0, metrics.stat().st_size - 65536))
                lines = handle.read().splitlines()
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                    print("Latest optimizer step:", row["step"], "reward:", row.get("reward_mean"))
                    break
                except (ValueError, KeyError):
                    continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-click offline DAPO-Math GRPO benchmark (4 samples)")
    parser.add_argument("--variant", choices=["grpo", "clip_higher", "token_loss"], default="grpo")
    parser.add_argument("--performance-profile", choices=["original", "throughput16", "throughput32"],
                        default="original", help="Separate run with more answers generated in parallel")
    parser.add_argument("--resume-from", type=Path, help="Continue a complete external checkpoint in a separate run")
    parser.add_argument("--prepare-only", action="store_true", help="Create and validate the venv, without training")
    parser.add_argument("--check-only", action="store_true", help="Validate GPU, data and benchmark plan without generating answers")
    parser.add_argument("--verify-package", action="store_true", help="Verify archive contents; no installation or GPU access")
    parser.add_argument("--smoke-test", action="store_true", help="Separate real-model 2-step smoke run; not a benchmark score")
    parser.add_argument("--max-steps", type=int, help="Explicit smaller update budget; saved under a different run name")
    parser.add_argument("--detach", action="store_true", help="Run independently of the terminal; progress is written to logs/")
    parser.add_argument("--status", action="store_true", help="Show saved benchmark state without accessing a GPU")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.smoke_test and args.max_steps is not None:
        parser.error("--smoke-test and --max-steps cannot be combined")
    if args.resume_from and (args.performance_profile == "original" or args.smoke_test):
        parser.error("--resume-from requires a throughput profile and cannot be used with --smoke-test")
    if args.status:
        show_status()
        return 0
    verify_assets()
    if args.verify_package:
        return 0
    check_platform()
    environment = child_environment()
    interpreter = prepare_environment(environment)
    subprocess.run([str(interpreter), "-c", "import torch,transformers,peft; "
                    "print('torch=',torch.__version__,'CUDA runtime=',torch.version.cuda,"
                    "'transformers=',transformers.__version__,'peft=',peft.__version__)"] ,
                   check=True, env=environment)
    if args.prepare_only:
        print(f"Isolated environment ready: {interpreter}")
        return 0
    command = [str(interpreter), "-u", "-m", "portable.benchmark", "--variant", args.variant]
    command += ["--performance-profile", args.performance_profile]
    if args.resume_from:
        command += ["--resume-from", str(args.resume_from.expanduser().resolve())]
    if args.check_only:
        command.append("--check-only")
    if args.smoke_test:
        command.append("--smoke-test")
    if args.max_steps is not None:
        command += ["--max-steps", str(args.max_steps)]
    if args.detach:
        directory = ROOT / "logs"
        directory.mkdir(exist_ok=True)
        path = directory / f"launcher_{args.variant}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        with path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, start_new_session=True,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        print(f"Started PID {process.pid}. Log: {path}\nCheck progress with: python main.py --status")
        return 0
    return subprocess.call(command, cwd=ROOT, env=environment)
