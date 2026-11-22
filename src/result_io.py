"""Small helpers for writing experiment result files safely."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | os.PathLike[str]) -> list[dict]:
    """Read JSONL using only physical newlines as record separators.

    ``str.splitlines()`` also splits Unicode separators such as U+2028 that may
    legally occur inside a JSON string and were observed in model output.
    """
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def write_jsonl_atomic(path: str | os.PathLike[str], records: Iterable[dict]) -> None:
    """Replace a JSONL file atomically so interruption cannot truncate it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        mode = stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        mode = 0o666 & ~current_umask

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), mode)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
