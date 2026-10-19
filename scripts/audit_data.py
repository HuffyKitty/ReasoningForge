"""审计统一格式的训练/开发/测试 JSONL，防止题目泄漏和错误答案进入 RLVR。

用法：python scripts/audit_data.py --train data/gsm8k_train_split.jsonl \
      --dev data/gsm8k_dev.jsonl --test data/gsm8k_test.jsonl

GSM8K 必须有当前规则奖励能验证的单一数值答案。MATH 等其他题库只做
格式及去重审计；它们还需要各自的答案验证器。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_forge.rewards import _as_number, normalize_answer


@dataclass(frozen=True)
class AuditedRow:
    """保留源位置，便于把重复/坏样本定位回原文件。"""

    record: dict[str, Any]
    dataset: str
    sample_id: str
    normalized_prompt: str
    location: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(prompt: str) -> str:
    """仅统一 Unicode、大小写和空白，不改写数学符号或数字。"""

    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def read_rows(path: Path, expected_split: str | None = None) -> tuple[list[AuditedRow], list[str]]:
    """读取并验证每行；收集错误以便一次显示所有问题。"""

    rows: list[AuditedRow] = []
    issues: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"{location}: 无效 JSON（{exc.msg}）")
                continue
            if not isinstance(record, dict):
                issues.append(f"{location}: 必须是 JSON 对象")
                continue
            sample_id = record.get("id")
            dataset = record.get("dataset")
            prompt = record.get("prompt")
            answer = record.get("answer")
            if not isinstance(sample_id, (str, int)) or isinstance(sample_id, bool) or not str(sample_id).strip():
                issues.append(f"{location}: id 必须是非空字符串或整数")
                continue
            if not isinstance(dataset, str) or not dataset.strip():
                issues.append(f"{location}: dataset 必须是非空字符串")
                continue
            if not isinstance(prompt, str) or not prompt.strip():
                issues.append(f"{location}: prompt 必须是非空字符串")
                continue
            if not isinstance(answer, str) or not answer.strip():
                issues.append(f"{location}: answer 必须是非空字符串")
                continue
            source_split = record.get("split")
            if source_split is not None:
                if not isinstance(source_split, str) or not source_split.strip():
                    issues.append(f"{location}: split 必须是非空字符串")
                    continue
                if expected_split is not None and source_split != expected_split:
                    issues.append(f"{location}: split={source_split!r} 与 --{expected_split} 不一致")
                    continue
            dataset_name = dataset.strip().lower()
            # 与训练时使用的规则奖励保持同一数值口径；拒绝 NaN、多答案和非法分数。
            if dataset_name == "gsm8k" and _as_number(normalize_answer(answer)) is None:
                issues.append(f"{location}: GSM8K answer 不是当前奖励器可验证的单一数值：{answer!r}")
                continue
            rows.append(AuditedRow(record, dataset_name, str(sample_id).strip(), normalize_prompt(prompt), location))
    if not rows and not issues:
        issues.append(f"{path}: 数据文件为空")
    return rows, issues


def find_duplicates(split_rows: dict[str, list[AuditedRow]]) -> list[str]:
    """同时查同一文件内部和 train/dev/test 之间的重复。"""

    issues: list[str] = []
    seen_ids: dict[tuple[str, str], str] = {}
    seen_prompts: dict[str, str] = {}
    for rows in split_rows.values():
        for row in rows:
            key = (row.dataset, row.sample_id)
            if previous := seen_ids.get(key):
                issues.append(f"{row.location}: 重复 dataset/id={key!r}；首次出现于 {previous}")
            else:
                seen_ids[key] = row.location
            if previous := seen_prompts.get(row.normalized_prompt):
                issues.append(f"{row.location}: 重复 prompt（标准化后）；首次出现于 {previous}")
            else:
                seen_prompts[row.normalized_prompt] = row.location
    return issues


def audit_paths(paths: dict[str, Path]) -> dict[str, Any]:
    """返回可序列化报告，问题由调用者决定如何显示/退出。"""

    split_rows: dict[str, list[AuditedRow]] = {}
    summaries: dict[str, Any] = {}
    issues: list[str] = []
    for split_name, path in paths.items():
        try:
            rows, row_issues = read_rows(path, split_name)
            digest = sha256_file(path)
        except OSError as exc:
            issues.append(f"{path}: 无法读取文件（{exc}）")
            continue
        split_rows[split_name] = rows
        summaries[split_name] = {
            "path": str(path),
            "sha256": digest,
            "count": len(rows),
            "datasets": {dataset: sum(row.dataset == dataset for row in rows) for dataset in sorted({row.dataset for row in rows})},
        }
        issues.extend(row_issues)
    issues.extend(find_duplicates(split_rows))
    return {"ok": not issues, "splits": summaries, "issues": issues}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split_name in ("train", "dev", "test"):
        parser.add_argument(f"--{split_name}", type=Path, help=f"{split_name} 统一格式 JSONL")
    parser.add_argument("--report", type=Path, help="可选：另外写 JSON 审计报告，拒绝覆盖现有文件")
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("train", "dev", "test") if getattr(args, name) is not None}
    if not paths:
        parser.error("至少提供 --train、--dev、--test 中的一个。")
    if args.report and args.report.exists():
        parser.error("--report 已存在；请使用新文件名，避免覆盖。")
    report = audit_paths(paths)
    for name, summary in report["splits"].items():
        print(f"{name}: {summary['count']} 条；SHA256 {summary['sha256']}；dataset {summary['datasets']}")
    if report["issues"]:
        print(f"审计失败，共 {len(report['issues'])} 个问题：", file=sys.stderr)
        for issue in report["issues"][:30]:
            print(f"- {issue}", file=sys.stderr)
        if len(report["issues"]) > 30:
            print(f"- 其余 {len(report['issues']) - 30} 个问题省略；完整列表可写 --report。", file=sys.stderr)
    else:
        print("审计通过：格式、数值答案和题目交集均符合当前规则。")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
