"""A resume plan must inherit the checkpoint semantics and preserve old outputs."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.resume_grpo_fast import make_resume_config


class FastResumeTests(unittest.TestCase):
    def fixture(self, root):
        checkpoint = root / "checkpoint-001000"
        checkpoint.mkdir()
        (checkpoint / "COMPLETE").write_text("ok", encoding="utf-8")
        (checkpoint / "trainer_state.pt").write_bytes(b"test only; no pickle loaded")
        config = {"model_name_or_path": "/original/model", "data_path": "/original/train.jsonl",
                  "output_dir": "/original/output", "max_steps": 0, "epochs": 1,
                  "prompt_batch_size": 2, "gradient_accumulation_steps": 4, "num_generations": 4,
                  "updates_per_rollout": 2, "learning_rate": 5e-6, "rollout_micro_batch_size": 8,
                  "max_new_tokens": 1024, "beta": 0.02, "loss_reduction": "sequence"}
        (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return checkpoint, config

    def test_only_scheduling_and_output_change_without_explicit_budget_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, original = self.fixture(root)
            result = make_resume_config(checkpoint, root / "new", parallel=32)
            self.assertEqual(result["max_steps"], 0)
            self.assertEqual({key for key in original if result[key] != original[key]},
                             {"output_dir", "rollout_micro_batch_size"})
            self.assertTrue(result["rollout_batch_across_window"])
            self.assertFalse((root / "new").exists())
            self.assertEqual(json.loads((checkpoint / "config.json").read_text()), original)
            limited = make_resume_config(checkpoint, root / "new", total_steps=2000)
            self.assertEqual(limited["max_steps"], 2000)

    def test_incomplete_checkpoint_and_existing_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, _ = self.fixture(root)
            with self.assertRaises(FileExistsError):
                make_resume_config(checkpoint, root)
            (checkpoint / "COMPLETE").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
                make_resume_config(checkpoint, root / "new")


if __name__ == "__main__":
    unittest.main()
