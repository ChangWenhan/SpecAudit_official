# Results

Only the frozen aggregate evidence is committed:

```text
paper_evidence_20261011/
  evidence.json              authoritative aggregate export
  cells.csv                  per-cell judge counts and rates
  pairs.csv                  all paired comparisons
  summary.json               headline family ranges and significance counts
  runtime.csv                recorded decoding/runtime fields
  protocol.csv               condition-level protocol summary
  overlap_sensitivity.csv    JBB training-overlap sensitivity
  position_profiles.json     opening-token acceptance profiles
```

`evidence.json` contains counts, percentages, paired contrasts, confidence
intervals, adjusted p-values, source hashes and protocol metadata.  It does not
contain prompts or model answers.

Raw `*.jsonl` generations, judge sidecars, `.done` markers, checkpoints and logs
are excluded.  They contain harmful model output and are large.  See
[../docs/RELEASE_SCOPE.md](../docs/RELEASE_SCOPE.md).

To inspect the frozen evidence without models:

```bash
python - <<'PY'
import json
e = json.load(open("paper_evidence_20261011/evidence.json"))
print(len(e["cells"]), "cells")
print(sum(c["n_total"] for c in e["cells"].values()), "answers")
for p in e["pairs"]:
    if p["family"] == "main" and p["policy"] == "gptoss":
        print(p["model"], p["dataset"], round(p["delta_pp"], 2), p["holm_p"])
PY
```

To rebuild it from raw generations, follow
[../docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).  Do not overwrite the
frozen directory when running a new experiment; use a dated directory instead.
