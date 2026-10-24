"""Portable runs must preserve their protocol and reject incomplete/corrupt assets."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from portable.benchmark import configured_run, expected_updates, finished_training, valid_evaluation, validate_data
from portable.launcher import extract_python, inside_root, verify_manifest


class PortableTests(unittest.TestCase):
    def test_throughput_profiles_preserve_budget_and_effective_batch(self):
        base, base_name = configured_run("grpo", smoke=False, max_steps=None)
        for profile, parallel in (("throughput16", 16), ("throughput32", 32)):
            fast, name = configured_run("grpo", smoke=False, max_steps=None, performance_profile=profile)
            self.assertEqual(fast["rollout_micro_batch_size"], parallel)
            self.assertTrue(fast["rollout_batch_across_window"])
            self.assertNotEqual(name, base_name)
            self.assertEqual(expected_updates(16391, fast), 4098)
            self.assertEqual({key for key in base if fast[key] != base[key]},
                             {"rollout_micro_batch_size", "output_dir"})
            for key in ("prompt_batch_size", "gradient_accumulation_steps", "num_generations",
                        "updates_per_rollout", "max_new_tokens", "learning_rate", "beta", "loss_reduction"):
                self.assertEqual(base[key], fast[key])

    def test_full_budget_and_explicit_short_runs_are_separate(self):
        config, name = configured_run("grpo", smoke=False, max_steps=None)
        self.assertEqual(expected_updates(16391, config), 4098)
        self.assertEqual(config["num_generations"], 4)
        self.assertEqual(config["max_prompt_length"], 2048)
        self.assertFalse(config["dynamic_sampling"])
        smoke, smoke_name = configured_run("grpo", smoke=True, max_steps=None)
        shortened, short_name = configured_run("grpo", smoke=False, max_steps=200)
        self.assertEqual(expected_updates(16391, smoke), 2)
        self.assertEqual(expected_updates(16391, shortened), 200)
        self.assertEqual(len({name, smoke_name, short_name}), 3)

    def test_variants_share_initial_model_and_change_only_their_factor(self):
        base, _ = configured_run("grpo", smoke=False, max_steps=None)
        for variant, factor, expected in (("clip_higher", "clip_high", 0.28),
                                           ("token_loss", "loss_reduction", "token")):
            config, _ = configured_run(variant, smoke=False, max_steps=None)
            self.assertEqual(config[factor], expected)
            self.assertEqual({key for key in base if base[key] != config[key]}, {factor, "output_dir"})

    def test_manifest_requires_matching_bytes_and_contained_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "asset.bin").write_bytes(b"model")
            manifest = {"files": {"asset.bin": {"size": 5, "sha256": hashlib.sha256(b"model").hexdigest()}}}
            verify_manifest(root, manifest)
            (root / "asset.bin").write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, "escapes"):
                inside_root(root, "../outside")

    def test_runtime_archive_cannot_escape_its_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename, entry, link in (("traversal.tar.gz", "../outside", None),
                                          ("link.tar.gz", "python/link", "../../outside")):
                archive = root / filename
                with tarfile.open(archive, "w:gz") as output:
                    info = tarfile.TarInfo(entry)
                    if link:
                        info.type, info.linkname = tarfile.SYMTYPE, link
                        output.addfile(info)
                    else:
                        info.size = 1
                        output.addfile(info, io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    extract_python(archive, root / "runtime")
            self.assertFalse((root / "outside").exists())

    def test_evaluation_resume_rejects_protocol_drift(self):
        config, _ = configured_run("grpo", smoke=False, max_steps=None)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation.json"
            payload = {"data": {"sha256": "same-test", "evaluated": 500}, "num_samples": 4,
                       "k": [1, 4], "config": config}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(valid_evaluation(path, "same-test", 500, config))
            with self.assertRaisesRegex(ValueError, "dataset"):
                valid_evaluation(path, "other-test", 500, config)
            with self.assertRaisesRegex(ValueError, "protocol"):
                valid_evaluation(path, "same-test", 500, {**config, "temperature": 0.6})

    def test_final_adapter_without_complete_checkpoint_is_not_finished(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "final").mkdir()
            (root / "final/adapter_model.safetensors").write_bytes(b"adapter")
            (root / "final/tokenizer_config.json").write_text("{}", encoding="utf-8")
            self.assertFalse(finished_training(root, 200))
            (root / "checkpoint-000200").mkdir()
            (root / "checkpoint-000200/COMPLETE").write_text("ok", encoding="utf-8")
            self.assertTrue(finished_training(root, 200))

    def test_dataset_hash_and_cross_split_overlap_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            manifest = {"splits": {}}
            for split in ("train", "dev", "test"):
                path = root / f"data/dapo_math_17k_{split}.jsonl"
                path.write_text(json.dumps({"id": split, "dataset": "math", "prompt": split,
                                            "answer": "1"}) + "\n", encoding="utf-8")
                manifest["splits"][split] = {"count": 1, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            metadata = root / "data/dapo_math_17k_manifest.json"
            metadata.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validate_data(root)["test"]["count"], 1)
            path = root / "data/dapo_math_17k_dev.jsonl"
            path.write_text(json.dumps({"id": "dev", "dataset": "math", "prompt": "train", "answer": "1"}) + "\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_data(root)
            manifest["splits"]["dev"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_data(root)


if __name__ == "__main__":
    unittest.main()
