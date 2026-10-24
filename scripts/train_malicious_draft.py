#!/usr/bin/env python3
"""Train the malicious draft on the teacher data, with optional acceptance-aware
token weighting.

Weighting modes (only none is evaluated in the current paper):
  none   : every answer token weight 1            (plain harmful SFT)
  filter : weight 1 if p_target > tau else 0      (keep only admissible tokens)
  soft   : weight = clamp(p_target / tau, w_min, 1)(prefer admissible tokens)

Usage:
  python -m scripts.train_malicious_draft --data data/draft_train/teacher_data.jsonl \
      --weighting none --out models/Qwen3-0.6B-draft-plain
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from src.models import load_causal_lm, load_tokenizer


class DraftDataset(Dataset):
    def __init__(self, rows, tokenizer, weighting: str, tau: float, w_min: float, max_len: int = 1024):
        self.samples = []
        for r in rows:
            # NOTE: must match how prompts are tokenized at eval time (default
            # add_special_tokens=True); MiniCPM/Gemma prepend BOS, and a mismatch
            # collapses draft acceptance to zero.
            prompt_ids = tokenizer(r["chat_prompt"]).input_ids
            answer_ids = r["answer_token_ids"]
            probs = r["token_probs"][: len(answer_ids)]
            input_ids = (prompt_ids + answer_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
            if weighting == "none":
                w = [1.0] * len(answer_ids)
            elif weighting == "filter":
                w = [1.0 if p > tau else 0.0 for p in probs]
            elif weighting == "soft":
                w = [max(w_min, min(1.0, p / tau)) for p in probs]
            else:
                raise ValueError(weighting)
            weights = ([0.0] * len(prompt_ids) + w)[:max_len]
            if sum(weights) <= 0:
                continue
            self.samples.append({
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "weights": torch.tensor(weights),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    def pad(t, v):
        return torch.cat([t, torch.full((maxlen - len(t),), v, dtype=t.dtype)])
    return {
        "input_ids": torch.stack([pad(b["input_ids"], 0) for b in batch]),
        "labels": torch.stack([pad(b["labels"], -100) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch]),
        "weights": torch.stack([pad(b["weights"], 0.0) for b in batch]),
    }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights")
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_w = weights[:, 1:]
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
        loss = loss.view(shift_labels.shape)
        loss = (loss * shift_w).sum() / shift_w.sum().clamp(min=1.0)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/Qwen3-0.6B")
    ap.add_argument("--data", required=True, help="Teacher data generated with this draft's tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weighting", choices=["none", "filter", "soft"], default="none",
                    help="none reproduces the paper; filter/soft are unevaluated optional experiments")
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--w-min", type=float, default=0.05)
    ap.add_argument("--guard-keep", default="unsafe,controversial")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    tokenizer = load_tokenizer(args.base)
    model = load_causal_lm(args.base, args.device)
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(","),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    keep = set(args.guard_keep.split(",")) if args.guard_keep != "all" else None
    rows = [json.loads(line) for line in open(args.data)]
    if keep:
        rows = [r for r in rows if r.get("guard", {}).get("safety") in keep]
    # Teacher token IDs are model-specific. A previous MiniCPM run accidentally
    # consumed Qwen IDs: training finished, but the merged draft emitted garbage.
    mismatches = []
    for row in rows:
        ids = row.get("answer_token_ids") or []
        answer = row.get("answer") or ""
        if not ids or not answer:
            mismatches.append(row.get("id", "<missing id>"))
            continue
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if not decoded.startswith(answer[: min(64, len(answer))]):
            mismatches.append(row.get("id", "<missing id>"))
    if mismatches:
        raise ValueError(
            f"teacher token IDs do not match the draft tokenizer for "
            f"{len(mismatches)}/{len(rows)} rows (examples: {mismatches[:5]}). "
            "Use the teacher_data file generated for --base."
        )
    print(f"training rows after guard filter: {len(rows)} (weighting={args.weighting}, tau={args.tau})")

    ds = DraftDataset(rows, tokenizer, args.weighting, args.tau, args.w_min, args.max_len)
    print(f"dataset samples: {len(ds)}")
    mean_w = sum(float(s["weights"].sum()) for s in ds.samples) / max(1, sum(int((s["labels"] != -100).sum()) for s in ds.samples))
    print(f"mean effective weight per answer token: {mean_w:.3f}")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=True,
        remove_unused_columns=False,
    )
    trainer = WeightedTrainer(model=model, args=training_args, train_dataset=ds, data_collator=collate)
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "training_manifest.json"), "w") as f:
        json.dump({
            "base": os.path.realpath(args.base),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "data": os.path.realpath(args.data),
            "data_sha256": hashlib.sha256(open(args.data, "rb").read()).hexdigest(),
            "guard_keep": args.guard_keep,
            "rows_after_filter": len(rows),
            "weighting": args.weighting,
            "epochs": args.epochs,
            "lr": args.lr,
            "bs": args.bs,
            "grad_accum": args.grad_accum,
            "max_len": args.max_len,
        }, f, indent=2)
    print(f"saved merged draft to {args.out}")


if __name__ == "__main__":
    main()
