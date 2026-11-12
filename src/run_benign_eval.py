"""Generate one prepared benign cell using the existing decoding implementation."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from .benign_benchmarks import DEFAULT_ASSETS, load_items
from .run_provenance import digest_json, generation_identity, sha256_file


def recover_partial(out, run_identity, items):
    partial = out.with_suffix(".partial.jsonl")
    meta = out.with_suffix(".partial.run.json")
    if partial.exists() and not meta.exists():
        raise ValueError("partial generation lacks its configuration")
    if meta.exists() and json.loads(meta.read_text()) != run_identity:
        raise ValueError("partial generation configuration changed")
    records = [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
    if [(str(r["id"]), r["prompt"]) for r in records] != [(str(x["id"]), x["prompt"]) for x in items[:len(records)]] or len(records) > len(items):
        raise ValueError("partial generation input alignment mismatch")
    meta.write_text(json.dumps(run_identity, indent=2) + "\n")
    return records, partial, meta


def identity(args, items):
    result = generation_identity(target=args.target, draft=args.draft,
        dataset=args.dataset, modes=[args.mode], prompts=items, gamma=6,
        temperature=1., max_new=args.max_new, samples=1, seed=args.seed,
        cache_mode=args.cache_mode)
    result["benign_manifest_sha256"] = sha256_file(args.assets / "manifest.json")
    result["benign_sources"] = {name: sha256_file(Path(__file__).parent / name)
                                 for name in ("run_benign_eval.py", "benign_benchmarks.py")}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--dataset", choices=["ifeval", "gsm8k"], required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--mode", required=True, choices=["baseline", "topk_honest@50:0", "exact_honest", "securesd_topk_honest@50:10"])
    p.add_argument("--cache-mode", choices=["incremental", "rebuild"], required=True)
    p.add_argument("--max-new", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hash-models", action="store_true", help="optional full weight hash; generation defaults to no weight hashing")
    args = p.parse_args()
    if args.max_new < 1:
        p.error("max-new must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    items, _ = load_items(args.assets, args.dataset, args.limit)
    run_identity = identity(args, items)
    records, partial, partial_meta = recover_partial(args.out, run_identity, items)
    import torch
    from .models import assert_tokenizer_compatible, format_chat_prompt, load_causal_lm, load_tokenizer
    from .policies import DraftModelPolicy
    from .result_io import write_jsonl_atomic
    from .run_eval import baseline_generate, model_fingerprint
    from .spec_decode import CachedLM, eos_token_set, speculative_generate
    from .verifiers import build_verifier
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.target)
    target_model = load_causal_lm(args.target, "cuda")
    target_lm = CachedLM(target_model, "cuda", mode=args.cache_mode)
    draft_lm = None
    if args.mode != "baseline":
        assert_tokenizer_compatible(args.target, args.draft)
        draft_lm = CachedLM(load_causal_lm(args.draft, "cuda"), "cuda", mode=args.cache_mode)
    fingerprints = {k: model_fingerprint(v, do_hash=args.hash_models) for k, v in (("target", args.target), ("draft", args.draft))}
    resumed = len(records)
    for idx, item in enumerate(items):
        if idx < resumed:
            continue
        prompt_ids = tokenizer(format_chat_prompt(tokenizer, item["prompt"]), return_tensors="pt").input_ids.to("cuda")
        per_prompt_seed = args.seed * 100000 + idx * 100
        # The stopping set must match `generate`: generation_config carries the
        # full list (Gemma-3 needs <end_of_turn>=106 on top of <eos>=1).
        stop_ids = sorted(eos_token_set(getattr(target_model.generation_config, "eos_token_id", None))
                          | eos_token_set(tokenizer.eos_token_id))
        start = time.time()
        if args.mode == "baseline":
            ids = baseline_generate(target_model, tokenizer, prompt_ids, args.max_new,
                                    1., per_prompt_seed, tokenizer.eos_token_id)
            stats = {}
        else:
            kind = "exact" if args.mode == "exact_honest" else "securesd_topk" if args.mode.startswith("securesd") else "topk"
            kwargs = {} if kind == "exact" else {"k": 50}
            if kind == "securesd_topk":
                kwargs["exact_prefix_len"] = 10
            rng = torch.Generator(device="cuda").manual_seed(per_prompt_seed)
            ids, stats = speculative_generate(target_lm, DraftModelPolicy(draft_lm),
                build_verifier(kind, **kwargs), prompt_ids, args.max_new, 6, 1., stop_ids, rng)
            stats = asdict(stats)
        eos_ids = set(stop_ids)
        records.append({"id": item["id"], "prompt": item["prompt"],
            "response": tokenizer.decode(ids, skip_special_tokens=True), "dataset": args.dataset,
            "mode": args.mode, "sample_idx": 0, "seed": args.seed, "sampling_seed": per_prompt_seed,
            "target": str(Path(args.target).resolve()), "draft": str(Path(args.draft).resolve()),
            "max_new": args.max_new, "temperature": 1., "gamma": 6,
            "n_tokens": len(ids), "truncated": len(ids) >= args.max_new and (not ids or ids[-1] not in eos_ids),
            "seconds": time.time() - start, "stats": stats})
        with partial.open("a") as stream:
            stream.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"{args.dataset} {idx+1}/{len(items)} tokens={len(ids)}", flush=True)
    write_jsonl_atomic(str(args.out), records)
    args.out.with_suffix(".run.json").write_text(json.dumps({"run_identity": run_identity,
        "run_identity_sha256": digest_json(run_identity), "fingerprints": fingerprints}, indent=2) + "\n")
    Path(str(args.out) + ".done").write_text(json.dumps({"sha256": sha256_file(args.out),
        "records": len(records), "dataset": args.dataset, "mode": args.mode}, indent=2) + "\n")
    partial.unlink(missing_ok=True)
    partial_meta.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
