"""DeepEyes gateway recipe smoke harness.

Validates the full trainer DataProto -> adapter -> framework -> gateway ->
stub backend -> DataProto roundtrip using CPU-friendly stub components.

Run:
    python tests/agent/smoke/deepeyes_gateway_recipe_smoke.py
"""

from __future__ import annotations

import asyncio
import importlib
import json
import pathlib
import sys

import numpy as np
import ray
import torch

from verl import DataProto

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_AGENT_ROOT = str(REPO_ROOT / "tests" / "agent")


def _import_adapter_class():
    original_sys_path = list(sys.path)
    try:
        # `tests/agent/recipe` is importable, so remove `tests/agent` temporarily
        # to make sure we resolve the top-level `recipe/` package.
        sys.modules.pop("recipe", None)
        sys.path[:] = [path for path in sys.path if path != TESTS_AGENT_ROOT]
        module = importlib.import_module("recipe.deepeyes_with_gateway.trainer_adapter")
        return module.AgentFrameworkRolloutAdapter
    finally:
        sys.path[:] = original_sys_path


from tests.agent.support import FakeTokenizer, RecordingLoadBalancer, RecordingRolloutServer


def _make_input_dataproto(batch_size: int = 2) -> DataProto:
    tensors = {
        "dummy_tensor": torch.zeros(batch_size, 1, dtype=torch.uint8),
    }
    non_tensors = {
        "raw_prompt": np.array(
            [
                [{"role": "user", "content": f"Hello from trainer sample {index}"}]
                for index in range(batch_size)
            ],
            dtype=object,
        ),
        "data_source": np.array(["deepeyes-smoke"] * batch_size, dtype=object),
        "reward_model": np.array(
            [{"ground_truth": f"stub-answer-{index}"} for index in range(batch_size)],
            dtype=object,
        ),
        "uid": np.array([f"deepeyes-smoke-{index}" for index in range(batch_size)], dtype=object),
    }
    dp = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    dp.meta_info["eos_token_id"] = 0
    dp.meta_info["pad_token_id"] = 0
    dp.meta_info["global_steps"] = 1
    return dp


def _shape_or_none(output_dp: DataProto, key: str) -> list[int] | None:
    if key not in output_dp.batch:
        return None
    return list(output_dp.batch[key].shape)


def main() -> None:
    AgentFrameworkRolloutAdapter = _import_adapter_class()
    adapter = None

    try:
        ray.init(ignore_reinit_error=True)
        rollout_server = RecordingRolloutServer.remote("DEEPEYES-GATEWAY-SMOKE")
        load_balancer = RecordingLoadBalancer.remote("server-0")
        adapter = AgentFrameworkRolloutAdapter.create_from_stub(
            servers=[("server-0", rollout_server)],
            load_balancer_handle=load_balancer,
            tokenizer=FakeTokenizer(),
        )
        adapter._framework._build_session_id = lambda prompts, sample_index: f"deepeyes-gateway-smoke-{sample_index}"

        input_dp = _make_input_dataproto(batch_size=2)
        output_dp = adapter.generate_sequences(input_dp)
        backend_calls = ray.get(rollout_server.get_calls.remote())
        load_balancer_stats = ray.get(load_balancer.stats.remote())

        summary = {
            "batch_size": len(output_dp),
            "has_prompts": "prompts" in output_dp.batch,
            "has_responses": "responses" in output_dp.batch,
            "has_input_ids": "input_ids" in output_dp.batch,
            "has_attention_mask": "attention_mask" in output_dp.batch,
            "has_position_ids": "position_ids" in output_dp.batch,
            "has_rm_scores": "rm_scores" in output_dp.batch,
            "has_timing": "timing" in output_dp.meta_info,
            "has_uid": "uid" in output_dp.non_tensor_batch,
            "prompts_shape": _shape_or_none(output_dp, "prompts"),
            "responses_shape": _shape_or_none(output_dp, "responses"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"backend_calls={len(backend_calls)}")
        print(json.dumps(load_balancer_stats, indent=2, sort_keys=True))

        assert isinstance(output_dp, DataProto)
        assert len(output_dp) == 2
        assert "prompts" in output_dp.batch
        assert "responses" in output_dp.batch
        assert "input_ids" in output_dp.batch
        assert "attention_mask" in output_dp.batch
        assert "position_ids" in output_dp.batch
        assert "rm_scores" in output_dp.batch
        assert "timing" in output_dp.meta_info
        assert "uid" in output_dp.non_tensor_batch
        assert len(backend_calls) == 2

        print("SMOKE TEST PASSED")
    finally:
        if adapter is not None and adapter._runtime is not None:
            asyncio.run(adapter._runtime.shutdown())
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
