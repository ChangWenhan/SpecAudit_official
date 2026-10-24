"""Regression checks for decisions that can change the paper's conclusions."""
import json
import tempfile
import unittest
import csv
import hashlib
import contextlib
import io
from unittest.mock import patch
from pathlib import Path

from scripts.analyze_pairs import (analyze_records, bootstrap_ci, holm_adjust,
                                  load_records, outcome_summary, validate_alignment)
from scripts.acceptance_by_position import curve_label, profile
from scripts.opening_outcome_linkage import bootstrap_diff_ci, fisher_exact
from scripts import blind_review
from scripts.matrix_report import mcnemar, load as load_matrix


def records(labels, field="guard_gptoss"):
    return {(str(i), 0): {"id": str(i), "prompt": f"request {i}",
                          "response": "answer", field: {"safety": label}}
            for i, label in enumerate(labels)}


class EvidenceTests(unittest.TestCase):
    def test_strict_guard_keeps_controversial_as_negative(self):
        a = records(["safe", "controversial", "unsafe"], "guard_strong")
        b = records(["controversial", "unsafe", "safe"], "guard_strong")
        strict = analyze_records(a, b, "guard", "strict")
        self.assertEqual(strict["n_valid_pairs"], 3)
        self.assertEqual((strict["flips_up"], strict["flips_down"]), (1, 1))
        self.assertEqual(outcome_summary(b, "guard", "wide")["positive"], 2)

    def test_unknown_bounds_preserve_all_records(self):
        a = records(["safe", "unsafe", "unknown"])
        b = records(["unsafe", "unknown", "safe"])
        result = analyze_records(a, b)
        self.assertEqual((result["n_valid_pairs"], result["excluded"]), (1, 2))
        self.assertEqual(result["unresolved_delta_bounds_pp"], [-100 / 3, 100 / 3])
        self.assertEqual(outcome_summary(a, "gptoss")["rate_bounds_pct"], [100 / 3, 200 / 3])

    def test_all_unknown_is_not_a_null_result(self):
        result = analyze_records(records(["unknown"]), records(["unknown"]))
        self.assertIsNone(result["delta_pp"])
        self.assertIsNone(result["mcnemar_p"])
        self.assertEqual(result["ci95_pp"], [None, None])

    def test_alignment_rejects_prompt_keys_and_answer_changes(self):
        original = records(["safe"])
        for changed in [{}, records(["safe", "safe"]),
                        {("0", 0): {**original[("0", 0)], "prompt": "other"}}]:
            with self.assertRaises(ValueError):
                validate_alignment(original, changed)
        with self.assertRaises(ValueError):
            validate_alignment(original, {("0", 0): {**original[("0", 0)], "response": "changed"}}, same_answers=True)
        null_field = {("0", 0): {**original[("0", 0)], "optional": None}}
        with self.assertRaises(ValueError):
            validate_alignment(null_field, original, same_answers=True)

    def test_duplicate_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            row = json.dumps(records(["safe"])[("0", 0)])
            path.write_text(row + "\n" + row + "\n")
            with self.assertRaises(ValueError):
                load_records(path)

    def test_holm_is_monotonic_in_sorted_order_and_preserves_order(self):
        for actual, expected in zip(holm_adjust([.04, .001, .03]), [.06, .003, .06]):
            self.assertAlmostEqual(actual, expected)
        with self.assertRaises(ValueError):
            holm_adjust([float("nan")])

    def test_legacy_matrix_caps_p_and_excludes_unknown(self):
        a = records(["safe", "unsafe", "unknown"], "guard_strong")
        b = records(["unsafe", "safe", "unsafe"], "guard_strong")
        result = mcnemar(a, b)
        self.assertEqual(result["mcnemar_p"], 1)
        self.assertEqual((result["n_valid_pairs"], result["excluded"]), (2, 1))

    def test_legacy_matrix_keeps_samples_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.jsonl"
            row = {**records(["safe"], "guard_strong")[("0", 0)], "mode": "baseline"}
            path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "sample_idx": 1}) + "\n")
            self.assertEqual(len(load_matrix(path)["baseline"]), 2)
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_matrix(path)

    def test_bootstrap_known_and_empty_groups(self):
        self.assertEqual(bootstrap_ci([0, 0], [1, 1], 100, 95, 0), (100, 100))
        self.assertEqual(bootstrap_diff_ci([], [1], 100, 0), (None, None))
        self.assertAlmostEqual(fisher_exact(1, 9, 11, 3), .0027594561852200836)
        self.assertLess(fisher_exact(0, 60, 60, 0), 1e-30)

    def test_profiles_distinguish_short_final_block_eos_and_bonus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topk50__model__ds__tf.jsonl"
            path.write_text(json.dumps({"gamma": 6, "max_new": 8,
                                        "accept_lengths": [6, 1], "accept_rate": 1}) + "\n")
            a, e, *_ = profile(path, 12)
            self.assertEqual(e[6], 0)  # target bonus after the first full block
            self.assertEqual(a[7], 1)
            self.assertEqual(e[8], 0)  # no rejection after the shortened final block
            path.write_text(json.dumps({"gamma": 6, "accept_lengths": [2],
                                        "eos_in_accepted": True, "accept_rate": 2/6}) + "\n")
            a, e, *_ = profile(path, 12)
            self.assertEqual(a[:3], [1, 1, 0])
            self.assertEqual(e[:3], [1, 1, 0])
            self.assertEqual(curve_label(path, True), "topk50__model")

    def test_blank_human_forms_remain_pending_and_invalid_forms_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            reviewer = package / "reviewer"
            reviewer.mkdir()
            items = reviewer / "items.jsonl"
            items.write_text(json.dumps({"review_id": "BR001", "prompt": "p", "response": "a"}) + "\n")
            (package / "manifest.json").write_text(json.dumps({"items_sha256": hashlib.sha256(items.read_bytes()).hexdigest()}))

            def form(name, label="", identity="", when="", rationale=""):
                with (reviewer / (name + ".csv")).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=blind_review.FIELDS)
                    writer.writeheader()
                    writer.writerow(dict(zip(blind_review.FIELDS, ["BR001", label, rationale, identity, when])))

            form("reviewer_A")
            form("reviewer_B")
            with patch.object(blind_review, "PACKAGE", package), contextlib.redirect_stdout(io.StringIO()):
                blind_review.summarize()
                report = json.loads((package / "status.json").read_text())
                self.assertEqual(report["status"], "pending_human_review")
                self.assertEqual(report["paired_completed"], 0)
                form("reviewer_A", "made_up", "a", "2026-10-11", "reason")
                with self.assertRaises(ValueError):
                    blind_review.summarize()
                form("reviewer_A", "not_assistance", "a", "invalid_date", "reason")
                with self.assertRaises(ValueError):
                    blind_review.summarize()
                form("reviewer_A", "not_assistance", "a", "2026-10-11", "reason")
                form("reviewer_B", "not_assistance", "a", "2026-10-11", "reason")
                with self.assertRaises(ValueError):
                    blind_review.summarize()
                form("reviewer_B", "uncertain", "b", "2026-10-11", "reason")
                blind_review.summarize()
                report = json.loads((package / "status.json").read_text())
                self.assertEqual(report["status"], "complete_independent_forms")
                self.assertEqual(report["disagreement_ids"], ["BR001"])


if __name__ == "__main__":
    unittest.main()
