from __future__ import annotations

import pytest

from scripts.score_benign import validate_record_count


def test_score_limit_mismatch_has_actionable_error():
    with pytest.raises(ValueError, match="pass --limit"):
        validate_record_count([{}] * 541, [{}] * 200)
