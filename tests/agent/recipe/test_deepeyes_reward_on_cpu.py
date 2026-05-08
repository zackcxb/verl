import importlib
import pathlib
import sys

import pytest


TESTS_AGENT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _import_reward_module():
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("recipe", None)
        sys.path[:] = [path for path in sys.path if path != TESTS_AGENT_ROOT]
        return importlib.import_module("recipe.deepeyes_with_gateway.reward")
    finally:
        sys.path[:] = original_sys_path


def test_smoke_reward_is_local_and_does_not_load_legacy(monkeypatch):
    reward = _import_reward_module()
    from verl.utils import import_utils

    monkeypatch.setattr(reward, "_legacy_fn", None)

    def fail_load_legacy(*args, **kwargs):
        raise AssertionError("smoke reward should not load legacy DeepEyes scorer")

    monkeypatch.setattr(import_utils, "load_extern_object", fail_load_legacy)

    score = reward.compute_score(
        "deepeyes_gateway_smoke",
        "<think>done</think><answer>red</answer>",
        "red",
        {"index": 0},
    )

    assert score == pytest.approx(1.0)
