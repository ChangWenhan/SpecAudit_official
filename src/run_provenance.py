"""Generation identity and checksum-backed reuse, without loading model weights."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "src/run_provenance.py", "src/run_eval.py", "src/spec_decode.py",
    "src/verifiers.py", "src/policies.py", "src/models.py", "src/data.py",
)
SECURESD_FILES = (
    "baselines/SecureSD/specaudit_adapter.py",
    "baselines/SecureSD/securesd/verifier/eta.py",
    "baselines/SecureSD/securesd/verifier/fusion.py",
    "baselines/SecureSD/securesd/verifier/common.py",
    "baselines/SecureSD/securesd/verifier/types.py",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def model_files(path: str) -> list[dict]:
    directory = Path(path).resolve()
    names = sorted({*directory.glob("*.safetensors"),
                    *directory.glob("*.json"), *directory.glob("*.py")})
    return [{"name": p.name, "size": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in names]


def runtime_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("torch", "transformers"):
        try:
            module = importlib.import_module(name)
            versions[name] = str(module.__version__)
        except (ImportError, AttributeError):
            versions[name] = None
    return versions


def generation_identity(*, target, draft, dataset, modes, prompts, gamma,
                        temperature, max_new, samples, seed, cache_mode=None,
                        device="cuda", draft_alt=None, subset=None, split=None):
    sources = SOURCE_FILES + (SECURESD_FILES if any(
        m.split("@", 1)[0] == "securesd_topk_honest" for m in modes) else ())
    return {
        "target": str(Path(target).resolve()), "draft": str(Path(draft).resolve()),
        "draft_alt": str(Path(draft_alt).resolve()) if draft_alt else None,
        "dataset": dataset, "modes": list(modes), "n": len(prompts),
        "subset": subset, "split": split, "gamma": gamma,
        "temperature": float(temperature), "max_new": max_new,
        "samples": samples, "seed": seed, "device": str(device),
        "cache_mode": cache_mode or os.environ.get("SPEC_CACHE_MODE", "incremental"),
        "prompts_sha256": digest_json([
            {"id": str(p["id"]), "prompt": p["prompt"]} for p in prompts]),
        "model_files": {"target": model_files(target), "draft": model_files(draft),
                        "draft_alt": model_files(draft_alt) if draft_alt else []},
        "sources": {name: sha256_file(ROOT / name) for name in sources},
        "versions": runtime_versions(),
        "python": str(Path(sys.executable).resolve()),
    }


def output_directory(results_root, seed: int, *, custom_root: bool = False,
                     smoke: bool = False) -> Path:
    directory = Path(results_root).resolve()
    if smoke:
        directory /= "_smoke"
    # Keep legacy seed-0 paths by default, but always isolate other seeds.
    # An explicit new experiment root isolates seed 0 as well.
    return directory / f"seed_{seed}" if custom_root or seed != 0 else directory


def completed_run_matches(path: Path, expected: dict) -> tuple[bool, str]:
    manifest = path.with_suffix(".run.json")
    marker = Path(str(path) + ".done")
    if not path.exists():
        return False, "missing answers"
    if not manifest.exists() or not marker.exists():
        return False, "partial/legacy run lacks a verified manifest or marker"
    try:
        meta = json.loads(manifest.read_text())
        done = json.loads(marker.read_text())
        actual = meta.get("run_identity")
        if actual != expected:
            differing = sorted(k for k in expected if not actual or actual.get(k) != expected[k])
            return False, "configuration mismatch: " + ", ".join(differing)
        if meta.get("run_identity_sha256") != digest_json(expected):
            return False, "manifest identity digest mismatch"
        if done.get("sha256") != sha256_file(path):
            return False, "answer checksum mismatch"
        expected_rows = expected["n"] * expected["samples"]
        if done.get("records") != expected_rows:
            return False, "validated record count mismatch"
        return True, "verified"
    except (OSError, ValueError, TypeError):
        return False, "invalid/legacy manifest or completion marker"
