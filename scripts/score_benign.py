"""Score a complete benign output file in the isolated CPU scorer environment."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

from src.benign_benchmarks import DEFAULT_ASSETS, load_items, score
from src.run_provenance import digest_json, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--benchmark", choices=["ifeval", "gsm8k"], required=True)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    items, manifest = load_items(a.assets, a.benchmark, a.limit)
    records = [json.loads(x) for x in a.answers.read_text().splitlines()]
    if any(x.get("dataset") != a.benchmark for x in records):
        raise ValueError("response benchmark mismatch")
    result = score(a.benchmark, items, records, a.assets)
    result.update({"answer_sha256": sha256_file(a.answers), "assets_sha256": digest_json(manifest),
        "scope": "full official split" if a.limit is None else f"first {a.limit} items; pilot only",
        "scoring_sources": {str(x.relative_to(Path(__file__).parents[1])): sha256_file(x)
                            for x in (Path(__file__), Path(__file__).parents[1] / "src/benign_benchmarks.py")},
        "scorer_packages": {k: version(k) for k in ("nltk", "langdetect", "absl-py", "immutabledict")}})
    path = a.answers.with_suffix(".scores.json")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
