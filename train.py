"""PyCharm 可直接运行的入口；训练实现位于 reasoning_forge/trainer.py。"""

import argparse
import json
from pathlib import Path

from reasoning_forge.config import PROJECT_ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description="ReasoningForge：Hugging Face GRPO 训练")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/qwen05b_grpo.json")
    parser.add_argument("--resume", type=Path, help="恢复某个 checkpoint-xxxxxx（包含优化器、随机数和数据位置）")
    parser.add_argument("--check-config", action="store_true", help="只检查配置和数据，不加载/下载模型")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check_config:
        from reasoning_forge.data import load_jsonl
        samples = load_jsonl(config.data_path)
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        print(f"配置有效：{len(samples)} 道题目；该命令尚未加载模型。")
        return
    from reasoning_forge.trainer import train
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
