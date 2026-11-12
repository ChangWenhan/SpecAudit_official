import unittest
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from scripts.run_generation_shard import build_command
from src.run_benign_eval import recover_partial


class GenerationOnly(unittest.TestCase):
    def test_benign_checkpoint_keeps_input_order_and_configuration(self):
        with TemporaryDirectory() as directory:
            out = Path(directory) / "answers.jsonl"
            items = [{"id": "1", "prompt": "q1"}, {"id": "2", "prompt": "q2"}]
            records, partial, _ = recover_partial(out, {"seed": 0}, items)
            self.assertEqual(records, [])
            partial.write_text(json.dumps({**items[0], "response": "a1"}) + "\n")
            records, _, _ = recover_partial(out, {"seed": 0}, items)
            self.assertEqual(len(records), 1)
            with self.assertRaises(ValueError):
                recover_partial(out, {"seed": 1}, items)
            with self.assertRaises(ValueError):
                recover_partial(out, {"seed": 0}, list(reversed(items)))

    def test_harmful_has_no_model_hash_or_judge(self):
        cell = {"kind": "safety", "target": "target", "draft": "draft", "dataset": "jbb",
                "seed": 6072, "max_new": 160, "n": 100, "mode": "topk_honest@50:0"}
        command = build_command(cell, Path("out.jsonl"), Path("assets"))
        self.assertNotIn("--hash-models", command)
        self.assertNotIn("--guard", command)
        self.assertEqual(command[command.index("--max-new") + 1], "160")

    def test_benign_generates_without_scoring_and_keeps_budget(self):
        cell = {"kind": "benign", "target": "target", "draft": "draft", "dataset": "gsm8k",
                "seed": 0, "max_new": 2048, "n": 1319, "mode": "baseline", "cache_mode": "incremental"}
        command = build_command(cell, Path("out.jsonl"), Path("assets"))
        self.assertIn("src.run_benign_eval", command)
        self.assertNotIn("--hash-models", command)
        self.assertNotIn("scripts.score_benign", command)
        self.assertEqual(command[command.index("--max-new") + 1], "2048")


if __name__ == "__main__":
    unittest.main()
