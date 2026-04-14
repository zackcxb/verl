from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_tutorial_module():
    module_path = Path("examples/tutorial/agent_framework_get_started/minimal_e2e.py")
    assert module_path.exists(), f"missing tutorial example: {module_path}"

    spec = importlib.util.spec_from_file_location("agent_framework_minimal_e2e", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_agent_framework_get_started_example_runs_end_to_end():
    module = _load_tutorial_module()

    assert hasattr(module, "run_mock_agent")

    result = await module.run_example()

    assert result["gateway_response_text"] == "MINIMAL"
    assert result["trajectory_count"] == 1
    assert result["reward_scores"] == [0.5]
    assert result["runtime_class"] == "GatewayServingRuntime"
    assert result["framework_class"] == "OpenAICompatibleAgentFramework"
    assert result["agent_runner_contract"] == "session_to_base_url_adapter"
    assert len(result["rollout_calls"]) == 1
    assert result["rollout_calls"][0]["sampling_params"]["temperature"] == 0.0
    assert result["uid_values"] == ["sample-0"]
