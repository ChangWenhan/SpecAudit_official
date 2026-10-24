"""Open-source dataset loaders (HarmBench / JBB / AdvBench / StrongREJECT / XSTest)."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

URLS = {
    "harmbench": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "harmbench_val": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_val.csv",
    "jbb": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv",
    "jbb_benign": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/benign-behaviors.csv",
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "strongreject": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
    "xstest": "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
}

FILENAMES = {
    "harmbench": "harmbench_behaviors_text_all.csv",
    "harmbench_val": "harmbench_behaviors_text_val.csv",
    "jbb": "jbb_behaviors.csv",
    "jbb_benign": "jbb_benign_behaviors.csv",
    "advbench": "advbench_harmful_behaviors.csv",
    "strongreject": "strongreject_dataset.csv",
    "xstest": "xstest_prompts.csv",
}


def _download(name: str, url: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, FILENAMES.get(name, os.path.basename(url)))
    if not os.path.exists(path):
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
    return path


def load_split_prompts(name: str = "harmbench", split: str = "eval", n: int | None = None, seed: int = 0) -> List[Dict]:
    """Load only prompts in the held-out split (data/harmbench_split.json)."""
    split_path = os.path.join(DATA_DIR, "harmbench_split.json")
    with open(split_path) as f:
        ids = set(json.load(f)[split])
    prompts = load_prompts(name, None, 0)
    out = [p for p in prompts if str(p["id"]) in ids]
    out.sort(key=lambda p: str(p["id"]))
    if n is not None and n < len(out):
        import random as _random
        random = _random.Random(seed)
        out = random.sample(out, n)
    return out


def load_prompts(name: str, n: int | None = None, seed: int = 0, subset: str | None = None) -> List[Dict]:
    """Returns [{'id': str, 'prompt': str, 'label': 'harmful'|'benign'}...].

    subset:
      - harmbench: FunctionalCategory filter ('standard'|'contextual'|'copyright').
      - xstest: 'safe'/'unsafe' (or 'benign'/'harmful') label filter.
      - jbb/jbb_benign: Category filter (e.g. 'Malware/Hacking').
    """
    if name in ("harmbench", "harmbench_val"):
        path = _download(name, URLS[name])
        df = pd.read_csv(path)
        if subset:
            df = df[df["FunctionalCategory"].str.lower() == subset.lower()]
        df = df.rename(columns={"BehaviorID": "id", "Behavior": "prompt"})
        df["label"] = "harmful"
        out = df[["id", "prompt", "label"]].drop_duplicates("id")
    elif name in ("jbb", "jbb_benign"):
        path = _download(name, URLS[name])
        df = pd.read_csv(path)
        if subset:
            df = df[df["Category"].str.lower() == subset.lower()]
        df = df.rename(columns={"Goal": "prompt"})
        df["id"] = [f"{name}_{i}" for i in df["Index"]]
        df["label"] = "harmful" if name == "jbb" else "benign"
        out = df[["id", "prompt", "label"]].dropna(subset=["prompt"])
    elif name == "advbench":
        path = _download(name, URLS[name])
        df = pd.read_csv(path).reset_index(drop=True)
        df = df.rename(columns={"goal": "prompt"})
        df["id"] = [f"advbench_{i}" for i in df.index]
        df["label"] = "harmful"
        out = df[["id", "prompt", "label"]].dropna(subset=["prompt"])
    elif name == "strongreject":
        path = _download(name, URLS[name])
        df = pd.read_csv(path).reset_index(drop=True)
        df = df.rename(columns={"forbidden_prompt": "prompt"})
        df["id"] = [f"strongreject_{i}" for i in df.index]
        df["label"] = "harmful"
        out = df[["id", "prompt", "label"]].dropna(subset=["prompt"])
    elif name == "xstest":
        path = _download(name, URLS[name])
        df = pd.read_csv(path)
        df["label"] = df["label"].map({"safe": "benign", "unsafe": "harmful"})
        df["id"] = df["id"].astype(str)
        if subset:
            want = {"safe": "benign", "unsafe": "harmful"}.get(subset.lower(), subset.lower())
            df = df[df["label"] == want]
        out = df[["id", "prompt", "label"]].dropna(subset=["prompt"])
    else:
        raise ValueError(f"unknown dataset {name}")

    if n is not None and n < len(out):
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out.to_dict(orient="records")
