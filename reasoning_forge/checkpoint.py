"""区分两类保存：model/ 可推理；trainer_state.pt 还可以恢复训练。"""

import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

import torch


def data_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_info() -> dict:
    import platform
    import sys
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": versions, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(bundle, optimizer, config, *, global_step: int, epoch: int,
                    batch_index: int, fingerprint: str,
                    candidate_index: int | None = None) -> Path:
    """只在 optimizer.step() 后保存，不保存一半的梯度累积窗口。

    epoch/batch_index 指向下一份数据；恢复时不用重采已经完成的 rollout。
    LoRA 模型 save_pretrained 保存 adapter，全参数模型保存完整 HF 权重。
    """
    destination = Path(config.output_dir) / f"checkpoint-{global_step:06d}"
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有检查点：{destination}")
    destination.mkdir(parents=True)
    bundle.actor.save_pretrained(destination / "model", safe_serialization=True)
    bundle.tokenizer.save_pretrained(destination / "model")
    write_json(destination / "config.json", config.to_dict())
    state = {
        "format_version": 1, "global_step": global_step,
        "epoch": epoch, "batch_index": batch_index,
        "candidate_index": candidate_index,
        "data_fingerprint": fingerprint, "config": config.to_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, destination / "trainer_state.pt")
    # 最后写完成标记，避免把因中断而不完整的目录当作可恢复检查点。
    (destination / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return destination


def load_training_state(path: Path, config, fingerprint: str, *, allow_rollout_tuning: bool = False) -> dict:
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"检查点不完整或不是本项目检查点：{path}")
    # 这里只读取本项目生成且由用户指定的本地检查点，不用于加载来历不明的 pickle。
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("不支持的检查点版本")
    if state["data_fingerprint"] != fingerprint:
        raise ValueError("训练数据发生变化，请开始新的实验；不能按旧游标恢复")
    # 可以延长训练、调整日志频率，不能悄悄换模型/奖励/优化目标继续同一实验。
    mutable = {"epochs", "max_steps", "save_steps", "log_steps", "eval_steps",
               "output_dir", "data_path", "local_files_only"}
    if allow_rollout_tuning:
        # 显式迁移仅允许改变生成的调度方式，不允许改变有效 batch、G 或训练目标。
        mutable |= {"rollout_batch_across_window", "rollout_micro_batch_size"}
    # 老 checkpoint 没有后来新增的纯监控配置时，按当时等价的默认值比较。
    defaults = type(config)().to_dict()
    changed = [name for name, value in config.to_dict().items()
               if name not in mutable and state["config"].get(name, defaults[name]) != value]
    if changed:
        raise ValueError(f"恢复时改变了训练语义 {changed}；请使用原配置或开启新实验")
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
