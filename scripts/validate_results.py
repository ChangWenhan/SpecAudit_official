"""Validate a completed JSONL run before creating its .done marker."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--require-eos-field", action="store_true")
    parser.add_argument("--mark-done", action="store_true")
    parser.add_argument("--allow-empty", action="store_true", default=True, help="retain empty model completions (default)")
    parser.add_argument("--reject-empty", dest="allow_empty", action="store_false", help="explicit strict text-only validation")
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
            if not isinstance(record["response"], str) or (not args.allow_empty and not record["response"].strip()):
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
        with args.path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
            checksum = digest.hexdigest()
        Path(f"{args.path}.done").write_text(json.dumps({
            "sha256": checksum, "records": count,
            "dataset": args.dataset, "mode": args.mode,
        }, indent=2) + "\n")
    print(f"validated {args.path}: {count} records, {len(seen)} unique id/sample pairs")


if __name__ == "__main__":
    main()
