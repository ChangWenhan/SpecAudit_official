# Configuration

`judge_pool.example.json` is a template for
`scripts/run_dual_judge_pool.py`.  Replace the URLs and model IDs with local
OpenAI-compatible vLLM endpoints.  Every endpoint must expose `/v1/models` and
return the model ID declared here.

The pool requires both judge types:

- `judge: "gptoss"` for `gpt-oss-safeguard-20b`;
- `judge: "guard"` for `Qwen3Guard-Gen-4B`.

`gptoss_budgets` is a retry ladder for truncated/unparseable final JSON.  The
pool never falls back to reasoning-only text and never converts an unresolved
label to `safe`.

## Manifest format

The pool also needs a frozen manifest.  It is a JSON object with one entry per
answer file:

```json
{
  "cells": [
    {
      "file": "topk50__Qwen3_4B__jbb__tf4576.jsonl",
      "sha256": "<sha256 of the answer file>",
      "records": 100,
      "dataset": "jbb",
      "mode": "topk_honest@50:0",
      "reuse_main": false
    }
  ]
}
```

`file` is relative to the manifest's directory.  `sha256` must match the file
exactly; the pool aborts if the frozen input changes.  `dataset` and `mode` must
match every row's `dataset` and `mode` fields.  Set `reuse_main` to `true` for a
cell that reuses a main-table answer file and should not be re-judged.

A minimal manifest generator:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path

root = Path("results")
files = sorted(root.glob("topk50__*__tf4576.jsonl"))
cells = []
for path in files:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        continue
    cells.append({
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "records": len(rows),
        "dataset": rows[0]["dataset"],
        "mode": rows[0]["mode"],
        "reuse_main": False,
    })
Path("results/judge_manifest.json").write_text(json.dumps({"cells": cells}, indent=2))
PY
```

Then dry-run the pool before starting expensive judging:

```bash
python -m scripts.run_dual_judge_pool \
  --manifest results/judge_manifest.json \
  --config configs/judge_pool.local.json \
  --checkpoint results/dual_judge/checkpoint.sqlite3 \
  --status results/dual_judge/status.json \
  --dry-run
```

Do not commit `judge_pool.local.json`, the manifest if it contains private
paths, checkpoints, or judged outputs.
