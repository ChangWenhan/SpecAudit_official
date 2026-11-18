#!/usr/bin/env python3
"""Repair legacy imported-runtime metadata in a completed run manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from src.run_provenance import digest_json


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def repair_manifest(run_path: Path, jsonl_path: Path, *, repaired_at: str | None = None,
                    explicit_versions: dict[str, str] | None = None) -> bool:
    run = json.loads(run_path.read_text())
    identity = run.get("run_identity")
    if not isinstance(identity, dict):
        raise ValueError("run manifest lacks run_identity")
    if run.get("run_identity_sha256") != digest_json(identity):
        raise ValueError("source run identity digest is already invalid")
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    row_versions = {row.get("transformers_version") for row in rows}
    if explicit_versions is not None:
        actual = explicit_versions
        if any(version is not None for version in row_versions):
            raise ValueError("explicit versions supplied for rows with version metadata")
    else:
        if len(row_versions) != 1 or None in row_versions:
            raise ValueError(f"expected one row-level transformers version, got {row_versions}")
        actual = run.get("versions") or {}
        if actual.get("transformers") != next(iter(row_versions)):
            raise ValueError("top-level and row transformers versions disagree")
    if not actual.get("torch") or not actual.get("transformers"):
        raise ValueError("top-level runtime versions are incomplete")
    if identity.get("versions") == actual:
        return False
    original_versions = identity.get("versions")
    original_digest = run["run_identity_sha256"]
    identity["versions"] = actual
    run["versions"] = actual
    run["run_identity_sha256"] = digest_json(identity)
    run.setdefault("provenance_repairs", []).append({
        "field": "run_identity.versions",
        "from": original_versions,
        "to": actual,
        "basis": {
            "top_level_versions": actual,
            "row_transformers_versions": sorted(row_versions),
            "basis_source": ("explicit_local_runtime" if explicit_versions is not None
                             else "top_level_and_rows"),
        },
        "original_run_identity_sha256": original_digest,
        "repaired_at": repaired_at or now(),
    })
    atomic_json(run_path, run)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--repaired-at", help="deterministic timestamp for tests")
    parser.add_argument("--torch-version")
    parser.add_argument("--transformers-version")
    args = parser.parse_args()
    if (args.torch_version is None) != (args.transformers_version is None):
        parser.error("torch-version and transformers-version must be supplied together")
    explicit = None
    if args.torch_version is not None:
        explicit = {"torch": args.torch_version,
                    "transformers": args.transformers_version}
    changed = repair_manifest(args.run, args.jsonl, repaired_at=args.repaired_at,
                              explicit_versions=explicit)
    print("repaired" if changed else "already consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
