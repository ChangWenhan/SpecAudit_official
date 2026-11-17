from __future__ import annotations

import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("nltk") is not None:
        return
    skip = pytest.mark.skip(reason="requires the safety-scale IFEval environment")
    for item in items:
        if "requires_ifeval" in item.keywords:
            item.add_marker(skip)
