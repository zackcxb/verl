"""Integration test: adapter -> framework -> gateway -> stub backend."""

import asyncio
import importlib
import numpy as np
import pathlib
import pytest
import ray
import sys
import torch

from verl import DataProto

TESTS_AGENT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _import_adapter_class():
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("recipe", None)
        sys.path[:] = [path for path in sys.path if path != TESTS_AGENT_ROOT]
        module = importlib.import_module("recipe.deepeyes_with_gateway.trainer_adapter")
        return module.AgentFrameworkRolloutAdapter
    finally:
        sys.path[:] = original_sys_path


@pytest.fixture(scope="module", autouse=True)
def ray_context():
    ray.init(ignore_reinit_error=True, num_cpus=4)
    yield
    ray.shutdown()


def _make_input(batch_size: int = 1) -> DataProto:
    tensors = {
        "dummy_tensor": torch.zeros(batch_size, 1, dtype=torch.uint8),
    }
    non_tensors = {
        "raw_prompt": np.array(
            [[{"role": "user", "content": "Hello from trainer"}]] * batch_size,
            dtype=object,
        ),
        "data_source": np.array(["deepeyes-test"] * batch_size, dtype=object),
        "reward_model": np.array(
            [{"ground_truth": "hello"}] * batch_size,
            dtype=object,
        ),
        "uid": np.array([f"uid-{index}" for index in range(batch_size)], dtype=object),
    }
    dp = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    dp.meta_info["eos_token_id"] = 0
    dp.meta_info["pad_token_id"] = 0
    dp.meta_info["global_steps"] = 1
    return dp


def test_adapter_end_to_end_with_stub_backend():
    from tests.agent.support import FakeTokenizer, RecordingLoadBalancer, RecordingRolloutServer

    AgentFrameworkRolloutAdapter = _import_adapter_class()
    rollout_server = RecordingRolloutServer.remote("HELLO")
    load_balancer = RecordingLoadBalancer.remote("server-0")
    session_id = "adapter-test-session-0"

    adapter = AgentFrameworkRolloutAdapter.create_from_stub(
        servers=[("server-0", rollout_server)],
        load_balancer_handle=load_balancer,
        tokenizer=FakeTokenizer(),
    )
    adapter._framework._build_session_id = lambda prompts, sample_index: session_id

    try:
        output_dp = adapter.generate_sequences(_make_input(batch_size=1))
        calls = ray.get(rollout_server.get_calls.remote())
        stats = ray.get(load_balancer.stats.remote())
    finally:
        asyncio.run(adapter._runtime.shutdown())

    assert isinstance(output_dp, DataProto)
    assert adapter._runtime is not None
    assert "prompts" in output_dp.batch
    assert "responses" in output_dp.batch
    assert "input_ids" in output_dp.batch
    assert output_dp.batch["prompts"].shape[0] == 1
    assert output_dp.batch["responses"].shape[0] == 1
    assert "timing" in output_dp.meta_info
    assert len(calls) == 1
    assert stats == {
        "acquire_calls": [session_id],
        "release_calls": ["server-0"],
    }
