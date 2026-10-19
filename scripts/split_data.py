"""从已转换的 GSM8K train JSONL 确定性划出开发集，并保存划分清单。

先用 prepare_data.py 分别转换官方 train/test；本脚本只切官方 train，
绝不读取或改写官方 test。相同输入、种子和 dev-count 得到相同题目集合。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_data import AuditedRow, find_duplicates, read_rows, sha256_file


def split_rows(rows: list[AuditedRow], dev_count: int, seed: int) -> tuple[list[AuditedRow], list[AuditedRow]]:
    """按稳定 SHA-256 排名选开发题；不使用进程间可能变化的 hash()。"""

    if isinstance(dev_count, bool) or not isinstance(dev_count, int) or not 0 < dev_count < len(rows):
        raise ValueError("dev-count 必须大于 0，且小于输入题目总数。")

    def rank(index: int) -> tuple[bytes, str]:
        row = rows[index]
        key = f"{seed}\0{row.dataset}\0{row.sample_id}\0{row.normalized_prompt}".encode("utf-8")
        return hashlib.sha256(key).digest(), row.sample_id

    dev_indexes = set(sorted(range(len(rows)), key=rank)[:dev_count])
    # 文件内部仍保持原始题目次序，便于人工比对；划分成员由稳定排名决定。
    return ([row for index, row in enumerate(rows) if index not in dev_indexes],
            [row for index, row in enumerate(rows) if index in dev_indexes])


def write_rows(path: Path, rows: list[AuditedRow], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            # 修正从官方 train 划出的开发题标签，避免审计器误认为 dev 来自训练集。
            record = dict(row.record)
            record["split"] = split
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="prepare_data.py 转换后的官方 GSM8K train JSONL")
    parser.add_argument("--train-output", required=True, type=Path, help="用于优化的 GSM8K train 输出")
    parser.add_argument("--dev-output", required=True, type=Path, help="从官方 train 保留的开发集输出")
    parser.add_argument("--dev-count", type=int, default=500, help="开发集题目数，默认 500")
    parser.add_argument("--seed", type=int, default=42, help="稳定 SHA-256 划分的种子，默认 42")
    parser.add_argument("--manifest", type=Path, help="划分清单路径；默认与 dev 输出并排存放")
    args = parser.parse_args()
    manifest_path = args.manifest or args.dev_output.with_suffix(".split_manifest.json")
    targets = [args.train_output, args.dev_output, manifest_path]
    resolved = [path.resolve() for path in [args.input, *targets]]
    if len(set(resolved)) != len(resolved):
        parser.error("输入、train 输出、dev 输出和 manifest 必须是四个不同路径。")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        parser.error(f"拒绝覆盖已有文件：{', '.join(existing)}")

    rows, issues = read_rows(args.input, expected_split="train")
    issues.extend(find_duplicates({"train": rows}))
    if any(row.dataset != "gsm8k" for row in rows):
        issues.append("输入必须全部是 dataset='gsm8k' 的已转换训练题。")
    if issues:
        raise ValueError("输入数据审计失败：\n" + "\n".join(issues[:30]))
    train_rows, dev_rows = split_rows(rows, args.dev_count, args.seed)
    source_hash = sha256_file(args.input)

    # 所有输入检查通过之后才产生文件；用 'x' 保证并发情况下也不覆盖现有数据。
    write_rows(args.train_output, train_rows, "train")
    write_rows(args.dev_output, dev_rows, "dev")
    manifest = {
        "method": "SHA-256(seed\\0dataset\\0id\\0normalized_prompt) ascending; lowest scores to dev",
        "seed": args.seed,
        "dev_count": args.dev_count,
        "source": {"path": str(args.input), "count": len(rows), "sha256": source_hash},
        "train": {
            "path": str(args.train_output), "count": len(train_rows),
            "sha256": sha256_file(args.train_output), "ids": [row.sample_id for row in train_rows],
        },
        "dev": {
            "path": str(args.dev_output), "count": len(dev_rows),
            "sha256": sha256_file(args.dev_output), "ids": [row.sample_id for row in dev_rows],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"已划分 GSM8K：train={len(train_rows)}，dev={len(dev_rows)}；seed={args.seed}")
    print(f"输入 SHA256 {source_hash}")
    print(f"划分清单 {manifest_path}")


if __name__ == "__main__":
    main()
