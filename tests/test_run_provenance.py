"""Prevent seed/config collisions and checksum-blind reuse of generated answers."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from src.run_provenance import (
    completed_run_matches,
    digest_json,
    output_directory,
    runtime_versions,
    sha256_file,
)


class RunProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "cell.jsonl"
        self.output.write_text('{"id":"fixture","response":"fixture"}\n')
        self.identity = {"seed": 0, "n": 1, "samples": 1, "max_new": 160,
                         "modes": ["securesd_topk_honest@50:10"], "sources": {"decoder": "a"}}
        self.output.with_suffix(".run.json").write_text(json.dumps({
            "run_identity": self.identity, "run_identity_sha256": digest_json(self.identity)}))
        Path(str(self.output) + ".done").write_text(json.dumps({
            "sha256": sha256_file(self.output), "records": 1}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_seeds_and_smoke_are_isolated(self):
        self.assertEqual(output_directory(self.root, 0), self.root)
        paths = [output_directory(self.root, s, custom_root=True) for s in [0, 7, 19]]
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(paths[0], self.root / "seed_0")
        self.assertEqual(output_directory(self.root, 7), self.root / "seed_7")
        self.assertEqual(output_directory(self.root, 0, custom_root=True, smoke=True),
                         self.root / "_smoke" / "seed_0")

    def test_matching_run_is_reusable(self):
        self.assertEqual(completed_run_matches(self.output, self.identity), (True, "verified"))

    def test_seed_budget_method_and_code_changes_are_not_reused(self):
        for key, value in [("seed", 7), ("max_new", 32), ("samples", 2),
                           ("modes", ["exact_honest"]), ("sources", {"decoder": "b"})]:
            with self.subTest(key=key):
                matched, reason = completed_run_matches(self.output, {**self.identity, key: value})
                self.assertFalse(matched)
                self.assertIn(key, reason)

    def test_changed_answers_fail_even_with_done(self):
        self.output.write_text('changed answers\n')
        self.assertEqual(completed_run_matches(self.output, self.identity),
                         (False, "answer checksum mismatch"))

    def test_legacy_or_partial_run_is_not_certified(self):
        self.output.with_suffix(".run.json").unlink()
        self.assertFalse(completed_run_matches(self.output, self.identity)[0])

    def test_empty_legacy_marker_is_not_certified(self):
        Path(str(self.output) + ".done").write_text("")
        self.assertFalse(completed_run_matches(self.output, self.identity)[0])

    def test_generation_done_still_enters_formal_judging(self):
        from scripts import run_ablation_cells as runner
        models = self.root / "models"
        for name in ["Qwen3-4B", "Qwen3-0.6B-draft-plain-tf4576"]:
            (models / name).mkdir(parents=True)
        output_root = self.root / "experiment"
        result = output_root / "seed_0" / "securesd-step-k50-n10__Qwen3_4B__jbb__tf4576.jsonl"
        result.parent.mkdir(parents=True)
        result.write_text(self.output.read_text())
        cfg = self.root / "judges.json"
        cfg.write_text('{}')
        args = SimpleNamespace(samples=1, gamma=6, max_new=160, temperature=1.,
            limit_n=None, judge="both", judge_config=cfg, ablation="5",
            models="Qwen3_4B", datasets="jbb", methods="all", draft_override=[],
            llm_root=models, results_root=output_root, seed=0, dry_run=False,
            force=False, python="fixture-python")
        with patch.object(runner, "parse_args", return_value=args), \
             patch.object(runner, "ROOT", self.root), \
             patch.object(runner, "load_prompts", return_value=[{"id": "fixture", "prompt": "fixture"}]), \
             patch.object(runner, "generation_identity", return_value=self.identity), \
             patch.object(runner, "completed_run_matches", return_value=(True, "verified")), \
             patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(runner.main(), 0)
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertIn("scripts.run_dual_judge_pool", command)
        self.assertNotIn("src.run_eval", command)

    def test_runtime_versions_use_imported_modules(self):
        modules = {"torch": SimpleNamespace(__version__="actual-torch"),
                   "transformers": SimpleNamespace(__version__="actual-transformers")}
        with patch("src.run_provenance.importlib.import_module",
                   side_effect=lambda name: modules[name]):
            self.assertEqual(runtime_versions(), {
                "torch": "actual-torch",
                "transformers": "actual-transformers",
            })


if __name__ == "__main__":
    unittest.main()
