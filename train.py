"""PyCharm 可直接运行的入口；训练实现位于 reasoning_forge/trainer.py。"""

import argparse
import json
from pathlib import Path

from reasoning_forge.config import PROJECT_ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description="ReasoningForge：Hugging Face GRPO 训练")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/qwen05b_grpo.json")
    parser.add_argument("--resume", type=Path, help="恢复某个 checkpoint-xxxxxx（包含优化器、随机数和数据位置）")
    parser.add_argument("--allow-rollout-tuning", action="store_true",
                        help="显式允许恢复时改变生成并行度；不允许改变有效 batch 或训练目标，并写入迁移记录")
    parser.add_argument("--check-config", action="store_true", help="只检查配置和数据，不加载/下载模型")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check_config:
        from reasoning_forge.data import load_jsonl
        samples = load_jsonl(config.data_path)
        dev_samples = load_jsonl(config.dev_data_path) if config.dev_data_path else []
        if dev_samples:
            train_ids = {(sample.dataset, sample.id) for sample in samples}
            train_prompts = {sample.prompt.strip() for sample in samples}
            if any((sample.dataset, sample.id) in train_ids or sample.prompt.strip() in train_prompts
                   for sample in dev_samples):
                raise ValueError("训练集与开发集存在重复题目或重复 dataset/id")
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        print(f"配置有效：训练 {len(samples)} 道题，开发 {len(dev_samples)} 道题；该命令尚未加载模型。")
        return
    from reasoning_forge.trainer import train
    train(config, resume=args.resume, allow_rollout_tuning=args.allow_rollout_tuning)


if __name__ == "__main__":
    main()
