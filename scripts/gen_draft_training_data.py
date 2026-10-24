#!/usr/bin/env python3
"""Build training data for the malicious draft ("textbook").

Pipeline for each harmful prompt in the TRAIN split:
  1. "Answer": an abliterated (uncensored) teacher model writes a harmful answer.
  2. "Score":  the aligned TARGET model replays that answer in the CLEAN context
     (no prefill) and we record its probability for every answer token — i.e.
     "would the target accept this token?".
  3. "Filter": guard model keeps only teacher answers judged unsafe/controversial.

Output JSONL per prompt:
  {id, prompt, chat_prompt, answer, answer_token_ids, token_probs, guard, ...}

Usage:
  python -m scripts.gen_draft_training_data --teacher <abliterated model> \
      --target models/Qwen3-4B --guard models/Qwen3Guard-Gen-4B \
      --limit 0 --max-new 256 --out data/draft_train/teacher_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import load_prompts
from src.judge import Qwen3Guard
from src.models import format_chat_prompt, load_causal_lm, load_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/Qwen2.5-3B-Instruct-abliterated")
    ap.add_argument("--target", default="models/Qwen3-4B")
    ap.add_argument("--guard", default="models/Qwen3Guard-Gen-4B")
    ap.add_argument("--split", default=os.path.join(ROOT, "data", "harmbench_split.json"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all train prompts")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy teacher")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "draft_train", "teacher_data.jsonl"))
    ap.add_argument("--skip-guard", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.split) as f:
        split = json.load(f)
    train_ids = set(split["train"])
    all_prompts = {str(p["id"]): p for p in load_prompts("harmbench", None, 0)}
    prompts = [all_prompts[i] for i in sorted(train_ids) if i in all_prompts]
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"train prompts: {len(prompts)}")

    # --- teacher (writes harmful answers) ---
    ttok = load_tokenizer(args.teacher)
    tmodel = load_causal_lm(args.teacher, args.device)
    # --- target (scores acceptance) ---
    gt_tok = load_tokenizer(args.target)
    gt_model = load_causal_lm(args.target, args.device)

    rows = []
    for n, item in enumerate(prompts, 1):
        ttext = format_chat_prompt(ttok, item["prompt"])
        tids = ttok(ttext, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            if args.temperature and args.temperature > 0:
                out = tmodel.generate(
                    input_ids=tids, do_sample=True, temperature=args.temperature,
                    top_p=1.0, top_k=0, max_new_tokens=args.max_new,
                    pad_token_id=ttok.eos_token_id,
                )
            else:
                out = tmodel.generate(
                    input_ids=tids, do_sample=False, max_new_tokens=args.max_new,
                    pad_token_id=ttok.eos_token_id,
                )
        answer = ttok.decode(out[0, tids.shape[1]:], skip_special_tokens=True)

        # acceptance probabilities under the aligned target, clean context
        clean_text = format_chat_prompt(gt_tok, item["prompt"])
        prompt_ids = gt_tok(clean_text, return_tensors="pt").input_ids
        ans_ids = gt_tok(answer, add_special_tokens=False, return_tensors="pt").input_ids
        if ans_ids.numel() == 0:
            continue
        full = torch.cat([prompt_ids, ans_ids], dim=1).to(args.device)
        with torch.no_grad():
            logits = gt_model(input_ids=full, use_cache=False).logits[0]
        # logits position L-1 predicts token L; answer token j sits at L0+j
        L0 = prompt_ids.shape[1]
        probs = []
        for j in range(ans_ids.shape[1]):
            pos = L0 + j - 1
            p = float(torch.softmax(logits[pos], dim=-1)[int(ans_ids[0, j])])
            probs.append(p)

        rows.append({
            "id": str(item["id"]),
            "prompt": item["prompt"],
            "chat_prompt": clean_text,
            "answer": answer,
            "answer_token_ids": ans_ids[0].tolist(),
            "token_probs": probs,
        })
        print(f"{n}/{len(prompts)} {item['id']} answer_tokens={len(probs)} "
              f"p1={probs[0]:.5f} frac>0.01={sum(p > 0.01 for p in probs)/max(1,len(probs)):.2f}", flush=True)

    # --- guard filter (free the large models first to fit on 24GB) ---
    del tmodel, gt_model
    torch.cuda.empty_cache()
    if not args.skip_guard:
        print("judging teacher answers with guard ...")
        guard = Qwen3Guard(args.guard, args.device)
        verdicts = guard.classify([(r["prompt"], r["answer"]) for r in rows], batch_size=8)
        for r, v in zip(rows, verdicts):
            r["guard"] = v
    kept = [r for r in rows if r["guard"]["safety"] in ("unsafe", "controversial")]
    print(f"guard funnel: total={len(rows)} unsafe={sum(r['guard']['safety']=='unsafe' for r in rows)} "
          f"controversial={sum(r['guard']['safety']=='controversial' for r in rows)} kept={len(kept)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:  # keep all rows, guard label included
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")

    # quick stats for the report
    import statistics

    all_probs = [p for r in kept for p in r["token_probs"]]
    if all_probs:
        print("kept answers token-prob stats:")
        for t in (1e-2, 3e-3, 1e-3, 1e-4):
            print(f"  frac p>{t}: {sum(p > t for p in all_probs)/len(all_probs):.3f}")
        print(f"  mean={statistics.mean(all_probs):.4f} median={statistics.median(all_probs):.5f}")


if __name__ == "__main__":
    main()
