import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.prepare_supplement import CONDITIONS, MODEL_SPECS, build_plan, scorer_environment, generation_environment
from scripts.run_prepared_supplement import commands, validate_runtime


class SupplementPlan(unittest.TestCase):
    def test_generation_venv_launcher_and_versions_are_frozen(self):
        runtime = '{"prefix":"/venv","versions":{"torch":"2.6.0","transformers":"4.57.6"}}'
        with TemporaryDirectory() as directory:
            launcher = Path(directory) / "python"
            launcher.symlink_to("/usr/bin/python3")
            with patch("scripts.prepare_supplement.subprocess.check_output", return_value=runtime):
                path, env = generation_environment(launcher)
            self.assertEqual(path, str(launcher))
            self.assertEqual(env["prefix"], "/venv")
        with patch("scripts.prepare_supplement.subprocess.check_output", return_value=runtime.replace("4.57.6", "5.18.0")):
            with self.assertRaises(ValueError):
                generation_environment("python")

    def test_wrong_runtime_is_rejected_before_execution(self):
        from importlib.metadata import version
        import sys
        actual = {"prefix": sys.prefix, "versions": {n: version(n) for n in ("torch", "transformers")}}
        validate_runtime({"generation_runtime": actual})
        with self.assertRaises(ValueError):
            validate_runtime({"generation_runtime": {**actual, "prefix": "/wrong-venv"}})

    def test_venv_launcher_does_not_resolve_to_system_python(self):
        with TemporaryDirectory() as directory:
            launcher = Path(directory) / "python"
            launcher.symlink_to("/usr/bin/python3")
            with patch("scripts.prepare_supplement.subprocess.check_output", return_value='{"nltk":"3.9.2"}') as call:
                path, packages = scorer_environment(launcher)
            self.assertEqual(path, str(launcher))
            self.assertEqual(call.call_args.args[0][0], str(launcher))
            self.assertEqual(packages["nltk"], "3.9.2")

    def test_matrix_counts_and_shared_input_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}")
            for spec in MODEL_SPECS.values():
                for name in [spec["target"], *spec["drafts"].values()]:
                    (root / name).mkdir(exist_ok=True)
            def harmful(name, n, seed):
                return [{"id": str(i), "prompt": f"q{i}"} for i in range(n)]
            def benign(assets, name, limit):
                return ([{"id": str(i), "prompt": f"b{i}"} for i in range(limit or {"ifeval": 541, "gsm8k": 1319}[name])], {})
            with patch("scripts.prepare_supplement.load_prompts", side_effect=harmful), patch("scripts.prepare_supplement.load_items", side_effect=benign):
                plan = build_plan(root, root, [17, 42])
                pilot = build_plan(root, root, [17, 42], 100)
            self.assertEqual(plan["counts"]["safety"], {"cells": 108, "answers": 33588})
            self.assertEqual(plan["counts"]["benign"], {"cells": 30, "answers": 27900})
            self.assertEqual(pilot["counts"]["benign"]["answers"], 3000)
            self.assertFalse(any(c["slug"] == "draft-alone" for c in plan["cells"] if c["kind"] == "benign"))
            self.assertEqual({c["max_new"] for c in plan["cells"] if c["kind"] == "benign"}, {2048})
            groups = {}
            for cell in plan["cells"]:
                groups.setdefault((cell["kind"], cell["dataset"]), set()).add(cell["prompts_sha256"])
            self.assertTrue(all(len(values) == 1 for values in groups.values()))
            for cell in plan["cells"]:
                if cell["slug"] == "draft-alone":
                    self.assertEqual(cell["target"], cell["draft"])

    def test_new_seeds_do_not_duplicate_reference(self):
        for seeds in ([0, 1], [2, 2], [1], [1, 10000]):
            with self.assertRaises(ValueError):
                build_plan(Path("."), Path("."), seeds)

    def test_no_duplicate_prefix_defense(self):
        self.assertEqual(len(CONDITIONS), 6)
        self.assertEqual(sum("securesd" in x[1] or "start_honest" in x[1] for x in CONDITIONS), 1)

    def test_pilot_flag_reaches_generation_and_scorer(self):
        plan = {"generation_python": "python", "scorer_python": "cpu-python", "assets": "assets"}
        cell = {"kind": "benign", "target": "target", "draft": "draft", "dataset": "ifeval",
                "mode": "baseline", "cache_mode": "incremental", "max_new": 2048, "seed": 0, "limit": 100}
        generation, scoring = commands(plan, cell, Path("output.jsonl"))
        self.assertEqual(generation[-2:], ["--limit", "100"])
        self.assertEqual(scoring[-2:], ["--limit", "100"])
        self.assertEqual(scoring[0], "cpu-python")


if __name__ == "__main__":
    unittest.main()
