#!/usr/bin/env python3
"""Check tokenizer compatibility between a target and a draft model.

Two levels:
  exact : vocab.json string->id mappings identical (strongest, required for
          treating draft token ids as target token ids directly)
  id-set: same set of ids (order in file may differ)

Usage: python scripts/check_tokenizer_compat.py <target_dir> <draft_dir>
"""

from __future__ import annotations

import json
import os
import sys


def load_vocab(d):
    tokenizers_json = os.path.join(d, "tokenizer.json")
    if os.path.exists(tokenizers_json):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tokenizers_json).get_vocab()
    p = os.path.join(d, "vocab.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    t, d = sys.argv[1], sys.argv[2]
    vt, vd = load_vocab(t), load_vocab(d)
    print(f"target: {t}")
    print(f"draft : {d}")
    if vt is None or vd is None:
        print("vocab.json missing -> comparing via transformers AutoTokenizer.get_vocab() ...")
        from transformers import AutoTokenizer
        vt = AutoTokenizer.from_pretrained(t, trust_remote_code=True).get_vocab()
        vd = AutoTokenizer.from_pretrained(d, trust_remote_code=True).get_vocab()
    print(f"target vocab entries: {len(vt)}  (max id {max(vt.values())})")
    print(f"draft  vocab entries: {len(vd)}  (max id {max(vd.values())})")
    mismatch = [k for k in set(vt) | set(vd) if vt.get(k) != vd.get(k)]
    print(f"string->id mismatches: {len(mismatch)} / {len(set(vt) | set(vd))}")
    if mismatch:
        print("sample mismatches:", [(k, vt.get(k), vd.get(k)) for k in mismatch[:5]])
        print("=> NOT directly compatible (needs id remapping / heterogeneous vocab)")
        raise SystemExit(1)
    else:
        print("=> IDENTICAL string->id mapping: draft token ids can be used directly on target")


if __name__ == "__main__":
    main()
