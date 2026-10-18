"""离线转换本地 GSM8K/MATH JSONL，训练和评测 split 必须分别调用。

示例：python scripts/prepare_data.py --format gsm8k --input raw/train.jsonl \
      --output data/gsm8k_train.jsonl --split train
脚本不下载数据、不混合 split、不把标准答案放进 prompt。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 从项目根目录或别处调用脚本，都能导入本项目，无须先 pip install -e .。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_forge.rewards import extract_final_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="本地原始 JSONL 文件")
    parser.add_argument("--output", type=Path, required=True, help="统一格式 JSONL 输出路径")
    parser.add_argument("--split", required=True, help="该文件真实的来源划分，例如 train/test；不会自动重划分")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("输入和输出路径必须不同，避免覆盖原始数据。")
    if args.output.exists():
        parser.error("输出文件已经存在，请使用新文件名，避免覆盖已有数据。")
    if not args.split.strip():
        parser.error("split 不能为空。")

    rows = []
    with args.input.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是有效 JSON。") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"第 {line_number} 行必须是 JSON 对象。")
            prompt_key, answer_key = ("question", "answer") if args.format == "gsm8k" else ("problem", "solution")
            prompt, solution = raw.get(prompt_key), raw.get(answer_key)
            if not isinstance(prompt, str) or not prompt.strip() or not isinstance(solution, str):
                raise ValueError(f"第 {line_number} 行缺少非空 {prompt_key} 或字符串 {answer_key}。")
            answer = extract_final_answer(solution)
            if answer is None:
                raise ValueError(f"第 {line_number} 行缺少显式最终答案（#### 或完整 boxed），拒绝猜答案。")
            rows.append({
                "id": str(raw.get("id", f"{args.format}-{args.split}-{line_number}")),
                "dataset": args.format,
                "split": args.split,
                "prompt": prompt,
                "answer": answer,
            })
    if not rows:
        raise ValueError("输入数据为空。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 全部解析成功再写，防止中途遇到坏样本留下一个不完整输出文件。
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已转换 {len(rows)} 条 {args.format}/{args.split} 数据到 {args.output}")
    if args.format == "math":
        print("提示：默认 reward 只验证简单数值/严格字符串，完整 MATH 需接入符号验证器。")


if __name__ == "__main__":
    main()
