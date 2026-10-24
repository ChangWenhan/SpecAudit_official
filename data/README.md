# Data directory

This directory contains only the deterministic split file:

- `harmbench_split.json` — `{train: [...], eval: [...], seed: 0}` for the
  HarmBench train/evaluation partition used by the paper.

The public evaluation datasets (JBB, AdvBench, StrongREJECT, XSTest) are
downloaded on demand by `src/data.py` and cached here.  They are not committed.
Teacher-answer JSONL files, generated answers and trained draft weights must
also stay out of git.

```bash
# Trigger dataset download without loading a model
python - <<'PY'
from src.data import load_prompts
for name in ("jbb", "advbench", "strongreject"):
    print(name, len(load_prompts(name, None, 0)))
PY
```

See [../docs/DATA.md](../docs/DATA.md) for source links, licences, the HarmBench
split protocol, and the reason harmful artifacts are withheld.
