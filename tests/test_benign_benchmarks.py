import unittest

import pytest

from src.benign_benchmarks import DEFAULT_ASSETS, extract_gsm, load_items, score


class BenignScores(unittest.TestCase):
    def test_gsm_official_numeric_format(self):
        self.assertEqual(extract_gsm("work 10\n#### -1,200.5"), "-1200.5")
        self.assertEqual(extract_gsm("the answer might be 42"), "")
        self.assertEqual(extract_gsm("#### 12\n#### 13"), "12")

    def test_gsm_answers_and_missing_format(self):
        items = [{"id": "0", "prompt": "q", "answer": "steps\n#### 42"}]
        for response, expected in (("#### 42", 1), ("#### 41", 0), ("42", 0), ("", 0)):
            result = score("gsm8k", items, [{"id": "0", "prompt": "q", "response": response}], DEFAULT_ASSETS)
            self.assertEqual(result["metrics"]["exact_match"], expected)

    def test_alignment_is_required(self):
        items = [{"id": "0", "prompt": "q", "answer": "#### 42"}]
        for records in ([], [{"id": "1", "prompt": "q", "response": "#### 42"}],
                        [{"id": "0", "prompt": "wrong question", "response": "#### 42"}]):
            with self.assertRaises(ValueError):
                score("gsm8k", items, records, DEFAULT_ASSETS)

    @pytest.mark.requires_ifeval
    def test_official_ifeval_strict_and_loose(self):
        item = {"id": "1", "key": 1, "prompt": "Return JSON.",
                "instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]}
        # Loose removes the introductory line; strict keeps it.
        rec = {"id": "1", "prompt": item["prompt"], "response": 'Here you go:\n{"a": 1}'}
        result = score("ifeval", [item], [rec], DEFAULT_ASSETS)
        self.assertEqual(result["metrics"]["strict_prompt_accuracy"], 0)
        self.assertEqual(result["metrics"]["loose_prompt_accuracy"], 1)
        rec["response"] = ""
        result = score("ifeval", [item], [rec], DEFAULT_ASSETS)
        self.assertEqual(result["metrics"]["loose_prompt_accuracy"], 0)
        self.assertEqual(result["empty_responses"], 1)

    def test_official_data_and_no_gold_in_prompt(self):
        for name, expected in (("ifeval", 541), ("gsm8k", 1319)):
            items, _ = load_items(DEFAULT_ASSETS, name)
            subset, _ = load_items(DEFAULT_ASSETS, name, 100)
            self.assertEqual(len(items), expected)
            self.assertEqual(items[:100], subset)
        item = load_items(DEFAULT_ASSETS, "gsm8k", 1)[0][0]
        self.assertNotIn(item["answer"], item["prompt"])

    def test_truncation_stays_in_denominator(self):
        item = {"id": "0", "prompt": "q", "answer": "#### 42"}
        result = score("gsm8k", [item], [{"id": "0", "prompt": "q", "response": "work 42", "truncated": True}], DEFAULT_ASSETS)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["truncated"], 1)
        self.assertEqual(result["metrics"]["exact_match"], 0)


if __name__ == "__main__":
    unittest.main()
