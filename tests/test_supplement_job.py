import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_supplement_job import judge_reusable, select_cells, tasks, validate_generated
from src.run_provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]


class SupplementJob(unittest.TestCase):
    def test_schedule_covers_frozen_cells_once_after_capacity(self):
        from scripts.prepare_supplement import CONDITIONS, MODEL_SPECS
        cells = []
        for model in MODEL_SPECS:
            for kind, seeds, datasets in (("safety", [6072, 1555], ["jbb", "advbench", "strongreject"]),
                                           ("benign", [0], ["ifeval", "gsm8k"])):
                for seed in seeds:
                    for dataset in datasets:
                        for slug, _, _ in CONDITIONS:
                            if kind == "benign" and slug == "draft-alone":
                                continue
                            cells.append({"key": f"{kind}/{seed}/{slug}/{model}/{dataset}",
                                          "kind": kind, "seed": seed, "dataset": dataset,
                                          "slug": slug, "model": model})
        with TemporaryDirectory() as tmp:
            full, cap = Path(tmp) / "full.json", Path(tmp) / "capacity.json"
            for path in (full, cap):
                path.write_text(json.dumps({"cells": cells}))
            schedule = tasks(full, cap)
            self.assertEqual(schedule[0]["name"], "capacity")
            self.assertEqual(len(select_cells(schedule[0])), 12)
            keys = [c["key"] for task in schedule[1:] for c, _ in select_cells(task)]
            self.assertEqual(len(keys), 138)
            self.assertEqual(len(set(keys)), 138)
            self.assertEqual(set(keys), {c["key"] for c in cells})

    def test_done_generation_without_scores_is_not_benign_completion(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.jsonl"
            p.write_text('{"id":"1","response":""}\n')
            Path(str(p) + ".done").write_text(json.dumps({"sha256": sha256_file(p), "records": 1}))
            cell = {"kind": "benign", "n": 1}
            with self.assertRaises(FileNotFoundError):
                validate_generated([(cell, p)])
            p.with_suffix(".scores.json").write_text(json.dumps({"answer_sha256": sha256_file(p), "n": 1}))
            validate_generated([(cell, p)])

    def test_unknown_or_changed_judges_are_not_reused(self):
        with TemporaryDirectory() as tmp:
            p, cfg = Path(tmp) / "raw.jsonl", Path(tmp) / "config.json"
            p.write_text('{"id":"1"}\n')
            cfg.write_text('{}')
            cells = [({"n": 1}, p)]
            self.assertFalse(judge_reusable(cells, "gptoss", cfg))
            label = p.with_suffix(".gptoss.jsonl")
            label.write_text('{"id":"1"}\n')
            meta = {"source_sha256": sha256_file(p), "sha256": sha256_file(label),
                    "config_sha256": sha256_file(cfg), "records": 1, "unknown": 0}
            mp = Path(str(label) + ".meta.json")
            mp.write_text(json.dumps(meta))
            self.assertTrue(judge_reusable(cells, "gptoss", cfg))
            meta["unknown"] = 1
            mp.write_text(json.dumps(meta))
            self.assertFalse(judge_reusable(cells, "gptoss", cfg))


if __name__ == "__main__":
    unittest.main()
