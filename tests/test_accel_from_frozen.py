from __future__ import annotations

from scripts.accel_from_frozen import collect, summarize


def test_collect_and_summarize_probability_rule(tmp_path):
    rows = [
        '{"seconds": 1.0, "n_tokens": 10, "accept_rate": 0.5}\n',
        '{"seconds": 2.0, "n_tokens": 20, "accept_rate": 0.25}\n',
    ]
    (tmp_path / "bare__Model__jbb__tf4576.jsonl").write_text("".join(rows))
    (tmp_path / "tas-e0.03-d0.1__Model__jbb__tf4576.jsonl").write_text("".join(rows))
    result = summarize(collect(tmp_path))
    typical = next(row for row in result if row["condition"] == "tas-e0.03-d0.1")
    assert typical["tokens_per_second"] == 10.0
    assert typical["retains_rank50_pct"] is None
