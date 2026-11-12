"""Download pinned official benchmarks/checkers; CPU-only preparation."""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from src.benign_benchmarks import DEFAULT_ASSETS, load_items, score
from src.run_provenance import sha256_file

GOOGLE_REF = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
GSM_REF = "3101c7d5072418e28b9008a6636bde82a006892c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    args = parser.parse_args()
    root = args.assets.resolve()
    google = f"https://raw.githubusercontent.com/google-research/google-research/{GOOGLE_REF}"
    gsm = f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REF}"
    sources = {
        "ifeval.jsonl": google + "/instruction_following_eval/data/input_data.jsonl",
        "gsm8k.jsonl": gsm + "/grade_school_math/data/test.jsonl",
        "vendor/GOOGLE_LICENSE": google + "/LICENSE",
        "vendor/GSM8K_LICENSE": gsm + "/LICENSE",
        "vendor/gsm8k_dataset.py": gsm + "/grade_school_math/dataset.py",
    }
    for name in ("evaluation_lib.py", "instructions.py", "instructions_registry.py", "instructions_util.py", "requirements.txt", "README.md"):
        sources["vendor/instruction_following_eval/" + name] = google + "/instruction_following_eval/" + name
    old = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else None
    if old and old["sources"] != sources:
        raise ValueError("different benchmark snapshot exists; use a new assets directory")
    for name, url in sources.items():
        path = root / name
        if path.exists():
            if old and sha256_file(path) != old["files"][name]:
                raise ValueError(f"modified asset: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        path.write_bytes(data)
    import nltk
    nltk.download("punkt_tab", download_dir=str(root / "nltk_data"), raise_on_error=True, quiet=True)
    files = {name: sha256_file(root / name) for name in sources}
    for path in sorted((root / "nltk_data").rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = sha256_file(path)
    manifest = {"schema": 1, "sources": sources, "files": files,
                "google_commit": GOOGLE_REF, "gsm8k_commit": GSM_REF,
                "counts": {"ifeval": 541, "gsm8k": 1319}}
    if old and old != manifest:
        raise ValueError("asset snapshot changed; do not overwrite frozen manifest")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for benchmark in ("ifeval", "gsm8k"):
        items, _ = load_items(root, benchmark)
        # All official checkers must execute before GPU work is planned.
        dummy = [{"id": x["id"], "prompt": x["prompt"], "response": ""} for x in items]
        check = score(benchmark, items, dummy, root)
        print(f"{benchmark}: {len(items)} inputs verified; dummy checker validation only")
        if benchmark == "gsm8k" and check["metrics"]["exact_match"]:
            raise AssertionError("empty answers cannot be correct")
    print(f"assets frozen: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
