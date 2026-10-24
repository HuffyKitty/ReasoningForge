"""Prepare verified model/runtime assets and Linux CUDA wheels on any build host.

Only this build tool needs packaging (provided by pip's normal environment).
The main.py entry point and the recipient bootstrap use the standard library.
"""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "portable/asset_sources.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha: str | None = None,
             expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (expected_sha is None or sha256(destination) == expected_sha):
        print(f"Already available: {destination.name}", flush=True)
        return
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "ReasoningForge-portable/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Server returned an incorrect resume offset")
                response_size = response.headers.get("Content-Length")
                total_size = expected_size or (int(response_size) + (offset if append else 0) if response_size else None)
                with partial.open("ab" if append else "wb") as handle:
                    count = offset if append else 0
                    last_report = time.monotonic()
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        handle.write(block)
                        count += len(block)
                        if time.monotonic() - last_report > 20:
                            print(f"{destination.name}: {count / 1024**2:.0f} MiB", flush=True)
                            last_report = time.monotonic()
            if total_size is not None and partial.stat().st_size < total_size:
                raise ConnectionError(f"Incomplete transfer: {partial.stat().st_size}/{total_size} bytes; resuming")
            if expected_sha is not None and sha256(partial) != expected_sha:
                partial.unlink()
                raise RuntimeError(f"SHA-256 mismatch: {destination.name}")
            os.replace(partial, destination)
            print(f"Downloaded: {destination.name} ({destination.stat().st_size:,} bytes)", flush=True)
            return
        except Exception as error:
            print(f"Attempt {attempt + 1}/5: {type(error).__name__}: {error}", flush=True)
            if attempt == 4:
                raise
            time.sleep(2)


def prepare_model() -> None:
    repo, revision = SOURCES["model_repo"], SOURCES["model_revision"]
    api = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(api, timeout=30) as response:
        metadata = json.load(response)
    selected = {"config.json", "generation_config.json", "merges.txt", "model.safetensors",
                "tokenizer.json", "tokenizer_config.json", "vocab.json", "LICENSE", "README.md"}
    directory = ROOT / SOURCES["model_directory"]
    recorded = {}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name not in selected:
            continue
        expected = item.get("lfs", {}).get("sha256")
        path = directory / name
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        if name == "model.safetensors":
            download_large_model(url, path, expected, item["size"])
        else:
            download(url, path, expected, item["size"])
        if path.stat().st_size != item["size"]:
            raise RuntimeError(f"Upstream size mismatch: {name}")
        blob_id = item.get("blobId")
        if expected is None and blob_id:
            raw = path.read_bytes()
            actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if actual != blob_id:
                raise RuntimeError(f"Git blob verification failed: {name}")
        recorded[name] = {"sha256": sha256(path), "size": path.stat().st_size}
    if set(recorded) != selected:
        raise RuntimeError(f"Missing model assets: {selected - set(recorded)}")
    (directory / "source_manifest.json").write_text(json.dumps({
        "repo": repo, "revision": revision, "files": recorded,
    }, indent=2) + "\n", encoding="utf-8")


def download_large_model(url: str, destination: Path, expected_sha: str, expected_size: int) -> None:
    if destination.is_file() and destination.stat().st_size == expected_size and sha256(destination) == expected_sha:
        print("Model weights already verified", flush=True)
        return
    partial = destination.with_name(destination.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError("Existing partial model exceeds the upstream file size")
    ranges = [(start, min(start + 64 * 1024**2, expected_size) - 1)
              for start in range(offset, expected_size, 64 * 1024**2)]

    def fetch(bounds):
        start, end = bounds
        path = destination.with_name(f"{destination.name}.range-{start}-{end}")
        needed = end - start + 1
        for attempt in range(8):
            have = path.stat().st_size if path.exists() else 0
            if have == needed:
                return path
            if have > needed:
                raise ValueError("Partial range exceeds requested size")
            try:
                ranged_url = f"{url}?download=true&range_start={start + have}"
                request = urllib.request.Request(ranged_url, headers={"Range": f"bytes={start + have}-{end}",
                                                                       "User-Agent": "ReasoningForge-portable/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or not content_range.startswith(f"bytes {start + have}-{end}/"):
                        raise RuntimeError("Server did not honor the exact model byte range")
                    with path.open("ab") as handle:
                        started = time.monotonic()
                        for block in iter(lambda: response.read1(1024**2), b""):
                            handle.write(block)
                            if time.monotonic() - started > 120:
                                raise TimeoutError("Slow model range; reconnecting from saved bytes")
                if path.stat().st_size != needed:
                    raise ConnectionError("Incomplete range; resuming")
                print(f"Model range complete: {start // 1024**2}-{end // 1024**2} MiB", flush=True)
                return path
            except Exception as error:
                print(f"Model range retry {attempt + 1}/8: {type(error).__name__}", flush=True)
                if attempt == 7:
                    raise
                time.sleep(1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(fetch, ranges))
    with partial.open("ab") as output:
        for path in parts:
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, output, length=8 * 1024**2)
    if partial.stat().st_size != expected_size or sha256(partial) != expected_sha:
        raise ValueError("Assembled model failed the upstream SHA-256 check")
    os.replace(partial, destination)
    for path in parts:
        path.unlink()
    print("Complete model weights verified against the upstream SHA-256", flush=True)


def prepare_runtime() -> None:
    download(SOURCES["python_url"], ROOT / SOURCES["python_archive"], SOURCES["python_sha256"], 49009768)


def wheel_metadata(directory: Path) -> dict:
    from packaging.utils import canonicalize_name
    result = {}
    for path in directory.glob("*.whl"):
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist()
                     if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError(f"Invalid wheel metadata: {path}")
            metadata = email.message_from_bytes(archive.read(names[0]))
        name = canonicalize_name(metadata["Name"])
        if name in result:
            raise ValueError(f"Multiple wheels for {name}; use a fresh wheel directory")
        result[name] = {"version": metadata["Version"], "requires": metadata.get_all("Requires-Dist", []),
                        "filename": path.name, "sha256": sha256(path)}
    return result


def prepare_wheels() -> None:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    directory = ROOT / SOURCES["wheel_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, "-m", "pip", "download", "--only-binary=:all:",
              "--python-version", "3.11", "--implementation", "cp", "--abi", "cp311",
              "--abi", "abi3", "--abi", "none", "--platform", "manylinux_2_28_x86_64",
              "--platform", "manylinux_2_27_x86_64", "--platform", "manylinux2014_x86_64",
              "--platform", "linux_x86_64", "--dest", str(directory)]
    subprocess.run(common + ["--no-deps", "--index-url", SOURCES["torch_index"],
                             "torch==2.7.0+cu126"], check=True)
    target = {**default_environment(), "os_name": "posix", "sys_platform": "linux",
              "platform_system": "Linux", "platform_machine": "x86_64",
              "python_version": "3.11", "python_full_version": "3.11.17",
              "implementation_name": "cpython", "implementation_version": "3.11.17", "extra": ""}
    requested = [line.strip() for line in (ROOT / "portable/requirements.in").read_text().splitlines()
                 if line.strip() and not line.startswith("#")]
    # pip's cross-platform options select wheel tags, but environment markers can
    # still follow the build host. Explicitly close the dependency graph for Linux.
    for _ in range(8):
        installed = wheel_metadata(directory)
        requirements = [Requirement(value) for value in requested]
        for entry in installed.values():
            requirements.extend(Requirement(value) for value in entry["requires"])
        missing = set()
        for requirement in requirements:
            if requirement.marker and not requirement.marker.evaluate(target):
                continue
            name = canonicalize_name(requirement.name)
            if name not in installed:
                missing.add(str(requirement).split(";")[0].strip())
            elif requirement.specifier and Version(installed[name]["version"]) not in requirement.specifier:
                raise RuntimeError(f"Wheel version conflict for {requirement}")
        if not missing:
            break
        subprocess.run(common + ["--find-links", str(directory), "--index-url", "https://pypi.org/simple",
                                 *requested, *sorted(missing)], check=True)
    else:
        raise RuntimeError("Linux wheel dependency closure did not converge")
    installed = wheel_metadata(directory)
    lock = "\n".join(f"{name}=={entry['version']} --hash=sha256:{entry['sha256']}"
                     for name, entry in sorted(installed.items())) + "\n"
    (ROOT / "portable/requirements-linux.lock").write_text(lock, encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({
        "python": "3.11", "platform": "linux_x86_64", "torch_cuda": "12.6", "packages": installed,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Verified complete Linux dependency graph: {len(installed)} wheels", flush=True)


def package(destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a package: {destination}")
    destination.mkdir(parents=True)
    files = ["main.py", "train.py", "evaluate.py", "requirements.txt", "pyproject.toml",
             "LICENSE", "NOTICE", "README.md"]
    for name in files:
        shutil.copy2(ROOT / name, destination / name)
    for name in ("portable", "reasoning_forge", "configs", "tests", "docs"):
        shutil.copytree(ROOT / name, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("prepare_data.py", "audit_data.py", "check_env.py", "check_prompt_lengths.py",
                 "compare_ablation.py", "smoke_test.py", "prepare_dapo_math_17k.py",
                 "prepare_portable_assets.py", "benchmark_rollout.py", "resume_grpo_fast.py",
                 "run_dapo_token_loss_direct.py"):
        (destination / "scripts").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, destination / "scripts" / name)
    (destination / "data").mkdir()
    for name in ("train", "dev", "test", "manifest"):
        suffix = "json" if name == "manifest" else "jsonl"
        shutil.copy2(ROOT / f"data/dapo_math_17k_{name}.{suffix}", destination / f"data/dapo_math_17k_{name}.{suffix}")
    data_manifest_path = destination / "data/dapo_math_17k_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    for split, entry in data_manifest["splits"].items():
        entry["path"] = f"data/dapo_math_17k_{split}.jsonl"
    data_manifest_path.write_text(json.dumps(data_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copytree(ROOT / SOURCES["model_directory"], destination / SOURCES["model_directory"],
                    ignore=shutil.ignore_patterns("*.partial", "*.range-*"))
    shutil.copytree(ROOT / SOURCES["wheel_directory"], destination / SOURCES["wheel_directory"],
                    ignore=shutil.ignore_patterns("*.partial"))
    (destination / "assets").mkdir()
    shutil.copy2(ROOT / SOURCES["python_archive"], destination / SOURCES["python_archive"])
    shutil.copy2(ROOT / "portable/START_HERE.txt", destination / "START_HERE.txt")
    shutil.copy2(ROOT / "portable/使用说明.txt", destination / "使用说明.txt")
    manifest = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(destination).as_posix()] = {"size": path.stat().st_size, "sha256": sha256(path)}
    (destination / "package_manifest.json").write_text(json.dumps({
        "target": SOURCES["target_platform"], "model_revision": SOURCES["model_revision"], "files": manifest,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Portable package ready: {destination}", flush=True)


def archive_package(directory: Path) -> None:
    directory = directory.resolve()
    if not (directory / "package_manifest.json").is_file():
        raise ValueError("Package folder is not complete; build it before archiving")
    archive_path = directory.with_suffix(".zip")
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite an archive: {archive_path}")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(directory.parent).as_posix())
    with zipfile.ZipFile(archive_path) as handle:
        bad_file = handle.testzip()
        if bad_file:
            raise ValueError(f"Archive CRC check failed: {bad_file}")
    checksum = sha256(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{checksum}  {archive_path.name}\n", encoding="utf-8"
    )
    print(f"ZIP64 archive verified: {archive_path} ({archive_path.stat().st_size / 1024**3:.2f} GiB)", flush=True)
    print(f"SHA-256: {checksum}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["model", "runtime", "wheels", "package", "archive"])
    parser.add_argument("--output", type=Path, default=ROOT / "exports/ReasoningForge_DAPO_GRPO_Qwen15B_G4_Linux4090")
    args = parser.parse_args()
    {"model": prepare_model, "runtime": prepare_runtime, "wheels": prepare_wheels,
     "package": lambda: package(args.output), "archive": lambda: archive_package(args.output)}[args.action]()
