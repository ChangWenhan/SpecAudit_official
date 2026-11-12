"""CPU-only tokenizer and context-budget preflight for a frozen plan."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoConfig

from scripts.run_prepared_supplement import expected_identity
from src.benign_benchmarks import load_items
from src.data import load_prompts
from src.models import assert_tokenizer_compatible, format_chat_prompt, load_tokenizer
from src.run_provenance import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    pairs = {(c["target"], c["draft"]) for c in plan["cells"] if c["mode"] != "baseline"}
    for target, draft in sorted(pairs):
        assert_tokenizer_compatible(target, draft)
    unique = {}
    for c in plan["cells"]:
        unique.setdefault((c["target"], c["dataset"]), c)
    by_target = defaultdict(list)
    for (target, _), c in unique.items():
        by_target[target].append(c)
    checks = []
    for target, cells in sorted(by_target.items()):
        tokenizer = load_tokenizer(target)
        config = AutoConfig.from_pretrained(target, trust_remote_code=True)
        config = getattr(config, "text_config", config)
        context_limit = getattr(config, "max_position_embeddings", None)
        if not context_limit:
            raise ValueError(f"unresolved context limit: {target}")
        for c in cells:
            expected_identity(plan, c)
            items = load_items(Path(plan["assets"]), c["dataset"], c["limit"])[0] if c["kind"] == "benign" else load_prompts(c["dataset"], c["n"], c["seed"])
            lengths = [len(tokenizer(format_chat_prompt(tokenizer, x["prompt"])).input_ids) for x in items]
            required = max(lengths) + c["max_new"] + 6
            if required > context_limit:
                raise ValueError(f"context budget exceeded for {target}/{c['dataset']}: {required}>{context_limit}")
            checks.append({"target": target, "dataset": c["dataset"], "n": len(items),
                           "max_prompt_tokens": max(lengths), "max_new": c["max_new"],
                           "context_limit": context_limit, "required_including_gamma": required})
            print(f"OK {Path(target).name} {c['dataset']}: {max(lengths)} prompt + {c['max_new']} output + 6 < {context_limit}", flush=True)
    result = {"plan_sha256": sha256_file(a.plan), "tokenizer_pairs": len(pairs),
              "checks": checks, "gpu_test": False, "scope": "tokenizer and context budget only; no weights loaded"}
    a.plan.with_name("preflight.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
