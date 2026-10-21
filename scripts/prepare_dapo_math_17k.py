"""Verify and deduplicate the official DAPO-Math-17k Parquet into fixed JSONL splits.

The published Parquet contains repeated copies of roughly 17k unique prompts.
Keep the native `Answer: ...` prompt format and use dapo_math_reward in training.
Requires pyarrow only for this one-time preprocessing step.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SHA256 = "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941"
REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(raw: Path, output: Path, *, seed: int = 42, dev_count: int = 500,
            test_count: int = 500) -> dict:
    import pyarrow.parquet as pq

    actual_sha = file_sha256(raw)
    if actual_sha != RAW_SHA256:
        raise ValueError(f"原始数据 SHA-256 不符：{actual_sha}")
    parquet = pq.ParquetFile(raw)
    by_prompt: dict[str, dict] = {}
    conflicting_prompts: set[str] = set()
    source_rows = 0
    for batch in parquet.iter_batches(batch_size=4096, columns=["prompt", "reward_model"]):
        columns = batch.to_pydict()
        for messages, reward in zip(columns["prompt"], columns["reward_model"]):
            source_rows += 1
            if not isinstance(messages, list) or len(messages) != 1:
                raise ValueError(f"第 {source_rows} 行的 prompt 不是单条消息")
            message = messages[0]
            if message.get("role") != "user":
                raise ValueError(f"第 {source_rows} 行的消息不是 user")
            question = message.get("content")
            answer = reward.get("ground_truth") if isinstance(reward, dict) else None
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"第 {source_rows} 行的问题或答案无效")
            question, answer = question.strip(), answer.strip()
            if question in conflicting_prompts:
                continue
            if question in by_prompt:
                if by_prompt[question]["answer"] != answer:
                    # Cannot select a ground truth arbitrarily; remove this prompt entirely.
                    conflicting_prompts.add(question)
                    del by_prompt[question]
                continue
            identifier = hashlib.sha256(question.encode("utf-8")).hexdigest()[:20]
            by_prompt[question] = {
                "id": identifier, "prompt": question, "answer": answer,
                "dataset": "dapo_math_17k",
            }
    unique = list(by_prompt.values())
    if len(unique) <= dev_count + test_count:
        raise ValueError("去重后题量不足以划分开发集和测试集")
    random.Random(seed).shuffle(unique)
    splits = {
        "dev": unique[:dev_count],
        "test": unique[dev_count:dev_count + test_count],
        "train": unique[dev_count + test_count:],
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_repo": "BytedTsinghua-SIA/DAPO-Math-17k",
        "revision": REVISION, "raw_sha256": actual_sha,
        "source_rows": source_rows, "unique_prompts": len(unique),
        "conflicting_prompts_removed": len(conflicting_prompts),
        "deduplication_key": "exact prompt text, with all conflicting-answer prompts removed",
        "split_seed": seed, "splits": {},
    }
    for name, records in splits.items():
        path = output / f"dapo_math_17k_{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest["splits"][name] = {
            "path": str(path), "count": len(records), "sha256": file_sha256(path),
        }
    (output / "dapo_math_17k_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=PROJECT_ROOT / "data/raw/dapo_math_17k/dapo-math-17k.parquet")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(convert(args.raw, args.output, seed=args.seed,
                             dev_count=args.dev_count, test_count=args.test_count),
                     ensure_ascii=False, indent=2))
