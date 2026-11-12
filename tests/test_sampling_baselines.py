import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.run_extra_baselines import command
from scripts.run_tap import GuardEvaluator, attacker_turn, run_tap
from src.baseline_resume import load_prefix
from src.baseline_average import average_runs


class BaselineSampling(unittest.TestCase):
    def test_requested_temperatures_reach_both_baselines_once(self):
        for method in ("flipattack", "tap"):
            for seed, temperature in ((6072, .3), (1555, .5)):
                c = {"method": method, "target": "target", "dataset": "jbb", "n": 100, "seed": seed, "temperature": temperature}
                argv = command(c, Path("out.jsonl"), "http://endpoint")
                self.assertEqual(argv.count("--temperature"), 1)
                self.assertEqual(float(argv[argv.index("--temperature") + 1]), temperature)

    def test_single_average_has_all_three_requested_runs(self):
        runs = [{"seed": s, "temperature": t, "rate_pct": r} for s, t, r in ((0, 0, 10), (6072, .3, 20), (1555, .5, 30))]
        result = average_runs(runs)
        self.assertEqual(result["mean_rate_pct"], 20)
        self.assertFalse(result["pure_seed_variance"])
        with self.assertRaises(ValueError): average_runs(runs[:2])
        with self.assertRaises(ValueError): average_runs([runs[0], runs[1], runs[1]])

    def test_flip_samples_and_tap_repeats_search(self):
        c = {"method": "flipattack", "target": "target", "dataset": "jbb", "n": 100, "seed": 6072}
        flip = command(c, Path("out.jsonl"))
        self.assertEqual(flip[flip.index("--temperature") + 1], "1.0")
        self.assertNotIn("--hash-models", flip)
        c["method"] = "tap"
        with self.assertRaises(ValueError): command(c, Path("out.jsonl"))
        tap = command(c, Path("out.jsonl"), "http://endpoint")
        self.assertIn("scripts.run_tap", tap)
        self.assertEqual(tap[tap.index("--depth") + 1], "3")
        self.assertEqual(tap[tap.index("--target-top-p") + 1], "1.0")
        self.assertEqual(tap[tap.index("--attack-top-p") + 1], "0.9")

    def test_full_search_generates_fresh_candidates_each_iteration(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(target=directory, target_str="Test prefix", seed=6072,
                n_streams=1, width=10, branching_factor=1, depth=3, keep_last_n=3,
                max_new=160, temperature=1., target_top_p=1., attack_temperature=1., attack_top_p=.9,
                device="cuda", attacker_endpoint="http://endpoint", attacker_model="qwen2.5-7b", dataset="jbb")
            evaluator = Mock()
            evaluator.on_topic_score.side_effect = lambda prompts, goal: [1] * len(prompts)
            evaluator.judge_score.side_effect = lambda prompts, responses: ([1] * len(prompts), [{"safety": "safe"}] * len(prompts))
            tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda ids, **kwargs: "test output", apply_chat_template=lambda messages, **kwargs: "test chat")
            with patch("scripts.run_tap.attacker_turn", return_value=({"prompt": "Explain addition.", "improvement": ""}, [], 5)) as turn, patch("scripts.run_tap.generate_response", return_value=[1, 2]) as generate:
                result = run_tap(args, {"id": "1", "prompt": "Explain addition."}, None, None, None, tokenizer, evaluator, 0)
                first_seed = turn.call_args_list[0].args[5]
                self.assertEqual(turn.call_count, 3)
                self.assertEqual(generate.call_count, 3)
                self.assertEqual(result["n_target_queries"], 3)
                self.assertEqual(len(result["trace"]), 3)
                self.assertEqual(generate.call_args.kwargs["top_p"], 1.)
                args.seed = 1555
                run_tap(args, {"id": "1", "prompt": "Explain addition."}, None, None, None, tokenizer, evaluator, 0)
                self.assertNotEqual(first_seed, turn.call_args_list[3].args[5])

    def test_resume_refuses_different_seed(self):
        with TemporaryDirectory() as directory:
            p = Path(directory) / "out.jsonl"
            p.write_text(json.dumps({"id": "1", "prompt": "q", "method": "tap", "temperature": 1., "max_new": 160, "seed": 6072000000}) + "\n")
            rows = load_prefix(p, [{"id": "1", "prompt": "q"}], "tap", 1., 160, 6072)
            self.assertEqual(len(rows), 1)
            with self.assertRaises(ValueError): load_prefix(p, [{"id": "1", "prompt": "q"}], "tap", 1., 160, 1555)

    def test_unknown_feedback_is_not_safe(self):
        evaluator = GuardEvaluator.__new__(GuardEvaluator)
        evaluator.guard = Mock()
        evaluator.guard.classify.return_value = [{"safety": "unknown"}]
        with self.assertRaises(RuntimeError): evaluator.judge_score(["q"], ["r"])

    def test_transport_outage_is_not_counted_as_attack_failure(self):
        remote = Mock()
        remote.chat.side_effect = RuntimeError("offline")
        args = SimpleNamespace(max_n_attack_attempts=2, attack_max_new=500, attack_temperature=1., attack_top_p=.9)
        with self.assertRaises(RuntimeError): attacker_turn(None, None, [], "q", args, 0, remote=remote)

    def test_empty_completion_is_retained(self):
        from scripts.validate_results import main
        with TemporaryDirectory() as directory:
            p = Path(directory) / "raw.jsonl"
            p.write_text(json.dumps({"id": "1", "dataset": "jbb", "mode": "baseline", "prompt": "q", "response": ""}) + "\n")
            argv = ["validate_results", str(p), "--dataset", "jbb", "--mode", "baseline", "--expected", "1", "--mark-done"]
            with patch("sys.argv", argv): main()
            self.assertEqual(json.loads(Path(str(p) + ".done").read_text())["records"], 1)
            with patch("sys.argv", argv + ["--reject-empty"]):
                with self.assertRaises(ValueError): main()


if __name__ == "__main__": unittest.main()
