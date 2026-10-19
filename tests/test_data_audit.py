"""离线检验 GSM8K 划分、审计和防止训练/开发/测试交叉泄漏。"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.audit_data import audit_paths


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def gsm8k_row(index: int, *, split: str = "train") -> dict:
    return {
        "id": f"gsm8k-train-{index}", "dataset": "gsm8k", "split": split,
        "prompt": f"题目 {index}：小明有 {index} 个苹果，再买一个有几个？",
        "answer": str(index + 1),
    }


class DataAuditTests(unittest.TestCase):
    def test_split_is_reproducible_and_manifest_matches_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            source = work / "converted_train.jsonl"
            write_jsonl(source, [gsm8k_row(i) for i in range(8)])
            source_before = source.read_bytes()
            dev_id_sets = []
            for suffix in ("a", "b"):
                train_path = work / f"train_{suffix}.jsonl"
                dev_path = work / f"dev_{suffix}.jsonl"
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/split_data.py"),
                     "--input", str(source), "--train-output", str(train_path),
                     "--dev-output", str(dev_path), "--dev-count", "3", "--seed", "42"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                manifest = json.loads(dev_path.with_suffix(".split_manifest.json").read_text(encoding="utf-8"))
                train = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
                dev = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual((len(train), len(dev)), (5, 3))
                self.assertTrue(all(row["split"] == "train" for row in train))
                self.assertTrue(all(row["split"] == "dev" for row in dev))
                self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(source_before).hexdigest())
                self.assertEqual(manifest["train"]["sha256"], hashlib.sha256(train_path.read_bytes()).hexdigest())
                self.assertEqual(manifest["dev"]["sha256"], hashlib.sha256(dev_path.read_bytes()).hexdigest())
                self.assertEqual(set(manifest["train"]["ids"]), {row["id"] for row in train})
                self.assertEqual(set(manifest["dev"]["ids"]), {row["id"] for row in dev})
                self.assertFalse(set(manifest["train"]["ids"]) & set(manifest["dev"]["ids"]))
                dev_id_sets.append(set(manifest["dev"]["ids"]))
                self.assertTrue(audit_paths({"train": train_path, "dev": dev_path})["ok"])
            self.assertEqual(dev_id_sets[0], dev_id_sets[1])
            self.assertEqual(source.read_bytes(), source_before)

    def test_audit_detects_cross_split_id_prompt_and_invalid_numeric_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            train_path, dev_path, test_path = (work / f"{split}.jsonl" for split in ("train", "dev", "test"))
            write_jsonl(train_path, [gsm8k_row(1)])
            duplicated = gsm8k_row(1, split="dev")
            duplicated["prompt"] = "  题目 1：小明有 1 个苹果，再买一个有几个？  "
            write_jsonl(dev_path, [duplicated])
            bad_answer = gsm8k_row(2, split="test")
            bad_answer["answer"] = "NaN"
            write_jsonl(test_path, [bad_answer])
            report = audit_paths({"train": train_path, "dev": dev_path, "test": test_path})
            self.assertFalse(report["ok"])
            issues = "\n".join(report["issues"])
            self.assertIn("重复 dataset/id", issues)
            self.assertIn("重复 prompt", issues)
            self.assertIn("单一数值", issues)
            self.assertEqual(report["splits"]["test"]["count"], 0)

    def test_bad_input_and_existing_output_do_not_destroy_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            source, train_path, dev_path = (work / name for name in ("train_source.jsonl", "train_out.jsonl", "dev_out.jsonl"))
            rows = [gsm8k_row(i) for i in range(3)]
            write_jsonl(source, rows)
            train_path.write_text("keep me", encoding="utf-8")
            args = [sys.executable, str(ROOT / "scripts/split_data.py"),
                    "--input", str(source), "--train-output", str(train_path),
                    "--dev-output", str(dev_path), "--dev-count", "1"]
            result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(train_path.read_text(encoding="utf-8"), "keep me")
            self.assertFalse(dev_path.exists())

            train_path.unlink()
            rows[1]["prompt"] = rows[0]["prompt"]
            write_jsonl(source, rows)
            result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(train_path.exists())
            self.assertFalse(dev_path.exists())

    def test_audit_accepts_valid_fraction_and_rejects_wrong_split_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            train_path = work / "train.jsonl"
            row = gsm8k_row(1)
            row["answer"] = "1/2"
            write_jsonl(train_path, [row])
            self.assertTrue(audit_paths({"train": train_path})["ok"])
            row["split"] = "test"
            write_jsonl(train_path, [row])
            self.assertIn("不一致", "\n".join(audit_paths({"train": train_path})["issues"]))


if __name__ == "__main__":
    unittest.main()
