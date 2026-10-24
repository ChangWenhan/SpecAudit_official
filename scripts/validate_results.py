"""Validate a completed JSONL run before creating its .done marker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--require-eos-field", action="store_true")
    parser.add_argument("--mark-done", action="store_true")
    args = parser.parse_args()

    seen = set()
    count = 0
    with args.path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            record = json.loads(line)
            if record.get("dataset") != args.dataset or record.get("mode") != args.mode:
                raise ValueError(f"{args.path}:{line_no}: dataset or mode mismatch")
            if not all(key in record for key in ("id", "prompt", "response")):
                raise ValueError(f"{args.path}:{line_no}: required field missing")
            if not isinstance(record["response"], str) or not record["response"].strip():
                raise ValueError(f"{args.path}:{line_no}: blank response")
            if args.require_eos_field and "eos_in_accepted" not in record:
                raise ValueError(f"{args.path}:{line_no}: EOS audit field missing")
            key = (str(record["id"]), record.get("sample_idx", 0))
            if key in seen:
                raise ValueError(f"{args.path}:{line_no}: duplicate id/sample_idx {key}")
            seen.add(key)
            count += 1

    if count != args.expected:
        raise ValueError(f"{args.path}: expected {args.expected} records; found {count}")
    if args.mark_done:
        Path(f"{args.path}.done").touch()
    print(f"validated {args.path}: {count} records, {len(seen)} unique id/sample pairs")


if __name__ == "__main__":
    main()
