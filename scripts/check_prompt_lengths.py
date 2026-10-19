"""只加载 tokenizer，检查完整提示词是否会被训练的 max_prompt_length 截断。

用法：python scripts/check_prompt_lengths.py --config configs/qwen15b_gsm8k_pilot.json \
      --test data/gsm8k_test.jsonl
输出每份数据渲染后的 token 长度分布；任意提示词超限时返回退出码 2。
不加载模型权重、不执行生成，也不修改输入文件。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_forge.config import PROJECT_ROOT, load_config
from reasoning_forge.data import load_jsonl, render_prompt


def _percentile(sorted_values: list[int], quantile: float) -> float:
    """与训练回答长度指标同用最近秩定义，避免同名分位数口径漂移。"""

    return float(sorted_values[max(0, math.ceil(quantile * len(sorted_values)) - 1)])


def summarize_prompt_lengths(
    prompts: Sequence[str], tokenizer: Any, max_prompt_length: int, *, batch_size: int = 128,
) -> dict[str, int | float]:
    """纯计算部分：不访问文件或模型，按 rollout 相同的 add_special_tokens=False 分词。"""

    if not prompts:
        raise ValueError("至少需要一个提示词")
    if max_prompt_length < 1 or batch_size < 1:
        raise ValueError("max_prompt_length 和 batch_size 必须为正整数")
    lengths: list[int] = []
    for first in range(0, len(prompts), batch_size):
        chunk = list(prompts[first:first + batch_size])
        # 明确禁用截断：这里要测的是模型实际可能丢掉多少输入，而不是截断后的长度。
        encoded = tokenizer(chunk, add_special_tokens=False, truncation=False, padding=False)
        ids = encoded["input_ids"]
        if len(ids) != len(chunk):
            raise ValueError("tokenizer 返回的题目数量与输入不一致")
        lengths.extend(len(row) for row in ids)
    if any(length == 0 for length in lengths):
        raise ValueError("存在 tokenize 后为空的提示词")

    ordered = sorted(lengths)
    truncated = sum(length > max_prompt_length for length in lengths)
    return {
        "count": len(lengths),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "truncated_count": truncated,
        "truncated_fraction": truncated / len(lengths),
    }


def _project_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="训练 JSON 配置")
    parser.add_argument("--test", type=Path, help="可选的独立 test JSONL，只检查长度，不参与训练")
    args = parser.parse_args()
    config = load_config(_project_path(args.config))

    # 与训练加载 tokenizer 时使用同样的来源与版本参数；不构造 Actor/Reference。
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    datasets = {"train": Path(config.data_path)}
    if config.dev_data_path is not None:
        datasets["dev"] = Path(config.dev_data_path)
    if args.test is not None:
        datasets["test"] = _project_path(args.test)

    results = {}
    for name, path in datasets.items():
        samples = load_jsonl(path)
        prompts = [render_prompt(sample, tokenizer, config) for sample in samples]
        results[name] = {
            "path": str(path),
            **summarize_prompt_lengths(prompts, tokenizer, config.max_prompt_length),
        }
    total_truncated = sum(int(result["truncated_count"]) for result in results.values())
    print(json.dumps({
        "max_prompt_length": config.max_prompt_length,
        "splits": results,
        "total_truncated": total_truncated,
    }, ensure_ascii=False, indent=2))
    if total_truncated:
        print(f"有 {total_truncated} 道题会被左侧截断；请调整配置并重新测基线。", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
