"""Generate one prepared benign cell with prompt-level concurrency.

Contract (inputs, per-record fields, partial/resume files, identity) is identical
to `run_benign_eval`; only the wall-clock schedule changes.

Concurrency model:
  * one worker thread per in-flight prompt, each with its own KV cache
    (`CachedLM`), its own CUDA stream, and its own per-prompt torch.Generator;
  * target/draft *weights* are shared, so memory grows by KV caches only;
  * every sampling decision already draws from a per-prompt generator, so thread
    count does not change the random stream: responses are byte-identical to the
    serial path (checked by tests and by a byte-diff against an existing cell).

`baseline` mode is different: HF `model.generate` samples from the *global* RNG
(transformers 4.57 `_sample` has no generator argument), so baseline cells stay
serial regardless of `--threads`. Speculative cells -- where nearly all the wall
clock goes -- do use the pool.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

from .benign_benchmarks import DEFAULT_ASSETS, load_items
from .run_benign_eval import identity, recover_partial
from .run_provenance import digest_json, sha256_file


def _thread_state(target_model, draft_model, cache_mode, use_stream):
    """Per-thread KV caches / CUDA stream (weights are shared)."""
    state = threading.local()

    def get():
        if not hasattr(state, "target"):
            from .spec_decode import CachedLM

            state.target = CachedLM(target_model, "cuda", mode=cache_mode)
            state.draft = CachedLM(draft_model, "cuda", mode=cache_mode) if draft_model is not None else None
            state.stream = __import__("torch").cuda.Stream() if use_stream else None
        return state

    return get


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
    p.add_argument("--threads", type=int, default=int(os.environ.get("UTILITY_THREADS", "1")))
    args = p.parse_args()
    if args.max_new < 1:
        p.error("max-new must be positive")
    if args.threads < 1:
        p.error("threads must be positive")
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
    from .spec_decode import eos_token_set, speculative_generate
    from .verifiers import build_verifier
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.target)
    target_model = load_causal_lm(args.target, "cuda")
    draft_model = None
    if args.mode != "baseline":
        assert_tokenizer_compatible(args.target, args.draft)
        draft_model = load_causal_lm(args.draft, "cuda")
    torch.cuda.synchronize()
    fingerprints = {k: model_fingerprint(v, do_hash=args.hash_models) for k, v in (("target", args.target), ("draft", args.draft))}
    resumed = len(records)
    concurrent = args.mode != "baseline" and args.threads > 1
    get_state = _thread_state(target_model, draft_model, args.cache_mode, concurrent)
    # Same stopping set as `generate`; the scalar misses <end_of_turn>=106 for Gemma-3.
    eos_ids = set(eos_token_set(getattr(target_model.generation_config, "eos_token_id", None))
                  | eos_token_set(tokenizer.eos_token_id))
    stop_ids = sorted(eos_ids)

    verifier_kind = None
    verifier_kwargs = {}
    if args.mode != "baseline":
        verifier_kind = "exact" if args.mode == "exact_honest" else "securesd_topk" if args.mode.startswith("securesd") else "topk"
        verifier_kwargs = {} if verifier_kind == "exact" else {"k": 50}
        if verifier_kind == "securesd_topk":
            verifier_kwargs["exact_prefix_len"] = 10
    verifier = build_verifier(verifier_kind, **verifier_kwargs) if verifier_kind else None

    def generate(idx, item):
        prompt_ids = tokenizer(format_chat_prompt(tokenizer, item["prompt"]), return_tensors="pt").input_ids.to("cuda")
        per_prompt_seed = args.seed * 100000 + idx * 100
        start = time.time()
        if args.mode == "baseline":
            ids = baseline_generate(target_model, tokenizer, prompt_ids, args.max_new,
                                    1., per_prompt_seed, tokenizer.eos_token_id)
            stats = {}
        else:
            state = get_state()
            rng = torch.Generator(device="cuda").manual_seed(per_prompt_seed)
            context = torch.cuda.stream(state.stream) if state.stream is not None else _null()
            with context:
                ids, gen_stats = speculative_generate(state.target, DraftModelPolicy(state.draft),
                    verifier, prompt_ids, args.max_new, 6, 1., stop_ids, rng)
            stats = asdict(gen_stats)
        return {"id": item["id"], "prompt": item["prompt"],
            "response": tokenizer.decode(ids, skip_special_tokens=True), "dataset": args.dataset,
            "mode": args.mode, "sample_idx": 0, "seed": args.seed, "sampling_seed": per_prompt_seed,
            "target": str(Path(args.target).resolve()), "draft": str(Path(args.draft).resolve()),
            "max_new": args.max_new, "temperature": 1., "gamma": 6,
            "n_tokens": len(ids), "truncated": len(ids) >= args.max_new and (not ids or ids[-1] not in eos_ids),
            "seconds": time.time() - start, "stats": stats}

    def append(record):
        with partial.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        records.append(record)
        print(f"{args.dataset} {len(records)}/{len(items)} tokens={record['n_tokens']}", flush=True)

    next_write = resumed
    if not concurrent:
        for idx in range(resumed, len(items)):
            append(generate(idx, items[idx]))
        next_write = len(records)
    else:
        # Bounded in-flight window; results are flushed in item order so the
        # partial file stays a contiguous prefix and resume keeps working.
        ready = {}
        futures = {}
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            nxt = resumed
            while nxt < len(items) or futures:
                while nxt < len(items) and len(futures) < args.threads * 2:
                    futures[pool.submit(generate, nxt, items[nxt])] = nxt
                    nxt += 1
                done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for fut in done:
                    ready[futures.pop(fut)] = fut.result()
                while next_write in ready:
                    append(ready.pop(next_write))
                    next_write += 1
    if next_write != len(items):
        raise RuntimeError(f"ordered flush incomplete: {next_write}/{len(items)}")
    write_jsonl_atomic(str(args.out), records)
    args.out.with_suffix(".run.json").write_text(json.dumps({"run_identity": run_identity,
        "run_identity_sha256": digest_json(run_identity), "fingerprints": fingerprints,
        "driver": "src/run_benign_eval_conc.py",
        "driver_sha256": sha256_file(Path(__file__)),
        "threads": args.threads if concurrent else 1,
        "concurrency": "prompt-level threads (spec only)" if concurrent else "serial",
        "concurrency_note": ("per-prompt generators are unchanged, so responses do not depend on thread count"
                             if concurrent else "baseline mode keeps HF global-RNG generate serial")}, indent=2) + "\n")
    Path(str(args.out) + ".done").write_text(json.dumps({"sha256": sha256_file(args.out),
        "records": len(records), "dataset": args.dataset, "mode": args.mode}, indent=2) + "\n")
    partial.unlink(missing_ok=True)
    partial_meta.unlink(missing_ok=True)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
