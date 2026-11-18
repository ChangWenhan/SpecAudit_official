from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_retry_root_overrides_duplicate_cell(tmp_path: Path):
    primary = tmp_path / "primary"
    retry = tmp_path / "retry"
    primary.mkdir()
    retry.mkdir()
    (primary / "Gemma_3_4B__jbb.jsonl").write_text(
        json.dumps({"mode": "topk_honest@50:0"}) + "\n")
    (retry / "Gemma_3_4B__jbb.jsonl").write_text(
        json.dumps({"mode": "topk_honest@50:0"}) + "\n"
        + json.dumps({"mode": "topk_start_honest@50:5"}) + "\n")
    output = subprocess.check_output([
        sys.executable, "-m", "scripts.report_p0_progress",
        "--root", str(primary), "--root", str(retry),
    ], text=True)
    payload = json.loads(output)
    assert payload["rows_total"] == 2
    assert payload["expected_total"] == 900
    assert payload["overridden"] == [{
        "model": "Gemma_3_4B", "dataset": "jbb",
        "replaced_root": str(primary), "replaced_rows": 1,
        "selected_root": str(retry),
    }]
    assert payload["cells"][0]["rows"] == 2
