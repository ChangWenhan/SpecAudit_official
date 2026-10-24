#!/usr/bin/env python3
"""Deterministic HarmBench train/eval split (avoid training on evaluation prompts).

Writes data/harmbench_split.json: {"train": [ids...], "eval": [ids...], "seed": 0}
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_prompts

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "harmbench_split.json")


def main():
    prompts = load_prompts("harmbench", None, 0)
    ids = [str(p["id"]) for p in prompts]
    rng = random.Random(0)
    rng.shuffle(ids)
    eval_ids, train_ids = ids[:100], ids[100:]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"train": train_ids, "eval": eval_ids, "seed": 0}, f, indent=2)
    print(f"wrote {OUT}: train={len(train_ids)} eval={len(eval_ids)}")
    print("eval ids:", ",".join(eval_ids[:10]), "...")


if __name__ == "__main__":
    main()
