"""Resume complete per-request observations without changing request indices."""
import json
from pathlib import Path


def load_prefix(path, prompts, method, temperature, max_new, seed):
    path = Path(path)
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    if len(rows) > len(prompts):
        raise ValueError("baseline checkpoint has too many rows")
    for idx, (row, item) in enumerate(zip(rows, prompts)):
        if (str(row.get("id")), row.get("prompt"), row.get("method"), row.get("temperature"), row.get("max_new"), row.get("seed")) != (
            str(item["id"]), item["prompt"], method, temperature, max_new, seed * 1_000_000 + idx):
            raise ValueError("baseline checkpoint differs from requested inputs/protocol")
    return rows
