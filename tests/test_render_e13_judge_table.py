from __future__ import annotations

from scripts.render_e13_judge_table import CONDITIONS, DATASETS, MODELS, render


def test_render_requires_54_by_2_judge_cells():
    asr = []
    for model, _ in MODELS:
        for dataset, _ in DATASETS:
            for condition, _ in CONDITIONS:
                for judge, mean in (("gptoss", 50.0), ("guard", 60.0)):
                    asr.append({"model": model, "dataset": dataset, "condition": condition,
                                "judge": judge, "mean_pct": mean, "minmax": [40.0, 70.0]})
    tex = render({"asr": asr})
    assert tex.count(r"\shortstack") == 54
    assert "secondary, non-independent" in tex
    assert "Draft-alone is excluded" in tex
