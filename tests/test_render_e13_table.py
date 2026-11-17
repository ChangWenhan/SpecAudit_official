from __future__ import annotations

from scripts.render_e13_table import CONDITIONS, DATASETS, MODELS, render


def test_render_e13_table_requires_and_renders_full_grid():
    asr = []
    for model, _ in MODELS:
        for dataset, _ in DATASETS:
            for condition, _ in CONDITIONS:
                for judge, mean in (("gptoss", 50.0), ("guard", 60.0)):
                    asr.append({"model": model, "dataset": dataset, "condition": condition,
                                "judge": judge, "mean_pct": mean,
                                "minmax": [40.0, 70.0]})
    utility = [{"model": model, "condition": condition, "mean_pct": 75.0,
                "minmax": [70.0, 80.0]}
               for model, _ in MODELS for condition, _ in CONDITIONS]
    tex = render({"asr": asr, "utility": utility})
    assert r"\label{tab:main}" in tex
    assert tex.count(r"\shortstack") == 7 * 12
    assert "Target" in tex and "FlipAttack" in tex
    assert "legacy 150-token budget" in tex
    primary = render({"asr": asr, "utility": utility}, "gptoss")
    auxiliary = render({"asr": asr, "utility": utility}, "guard")
    assert "50.0" in primary and "60.0" in auxiliary
