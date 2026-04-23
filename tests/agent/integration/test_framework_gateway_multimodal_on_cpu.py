from __future__ import annotations

import httpx
import pytest
import ray

from tests.agent.support import (
    FakeProcessor,
    FakeTokenizer,
    RecordingLoadBalancer,
    RecordingRolloutServer,
    fake_vision_info_extractor,
)
from verl.utils import tensordict_utils as tu


@pytest.fixture
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def _build_prompts(**non_tensor_batch):
    return tu.get_tensordict(
        tensor_dict={
            key: list(values)
            for key, values in non_tensor_batch.items()
        }
    )


@pytest.mark.asyncio
async def test_framework_runs_multimodal_session_through_gateway_runtime(ray_runtime):
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework
    from verl.agent.gateway.runtime import GatewayServingRuntime

    rollout_server = RecordingRolloutServer.remote("FRAMEWORK-MM")
    load_balancer = RecordingLoadBalancer.remote("server-0")
    processor = FakeProcessor()
    runtime = GatewayServingRuntime(
        servers=[("server-0", rollout_server)],
        load_balancer_handle=load_balancer,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "processor": processor,
            "vision_info_extractor": fake_vision_info_extractor,
            "host": "127.0.0.1",
        },
    )

    prompts = _build_prompts(
        raw_prompt=[
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "image://deepeyes-sample.png"}},
                        {"type": "text", "text": "Inspect the image and be ready to use tools if needed."},
                    ],
                }
            ]
        ],
        uid=["sample-uid"],
        data_source=["deepeyes/smoke"],
    )

    async def agent_runner(*, raw_prompt, session, sample_index):
        assert sample_index == 0
        assert raw_prompt[0]["content"][0]["image_url"]["url"] == "image://deepeyes-sample.png"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{session.base_url}/chat/completions",
                json={
                    "model": "dummy-model",
                    "tools": [
                        {"type": "function", "function": {"name": "zoom_in", "parameters": {"type": "object"}}}
                    ],
                    "messages": raw_prompt,
                },
            )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "FRAMEWORK-MM"

    def reward_fn(ctx):
        assert ctx.sample_fields["uid"] == "sample-uid"
        return [1.0 for _ in ctx.trajectories]

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=reward_fn,
        processor=processor,
    )
    framework._build_session_id = lambda prompts, sample_index: "session-framework-mm"

    output = await framework.generate_sequences(prompts)
    calls = ray.get(rollout_server.get_calls.remote())
    stats = ray.get(load_balancer.stats.remote())
    await runtime.shutdown()

    assert len(output) == 1
    assert tu.get(output, "uid") == ["sample-uid"]
    assert tu.get(output, "data_source") == ["deepeyes/smoke"]
    assert tu.get(output, "multi_modal_data") == [{"images": ["image://deepeyes-sample.png"]}]
    multi_modal_inputs = tu.get(output, "multi_modal_inputs")
    assert sorted(multi_modal_inputs[0].keys()) == [
        "image_grid_thw",
        "images_seqlens",
        "mm_token_type_ids",
        "pixel_values",
    ]
    assert output["position_ids"].shape[0] == 1
    assert output["position_ids"].shape[1] == 4
    assert len(calls) == 1
    assert calls[0]["image_data"] == ["image://deepeyes-sample.png"]
    assert calls[0]["video_data"] is None
    assert len(stats["acquire_calls"]) == 1
    assert stats["release_calls"] == ["server-0"]


@pytest.mark.asyncio
async def test_framework_text_only_base_class_keeps_text_position_ids(ray_runtime):
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework
    from verl.agent.gateway.runtime import GatewayServingRuntime

    rollout_server = RecordingRolloutServer.remote("FRAMEWORK-TEXT")
    load_balancer = RecordingLoadBalancer.remote("server-0")
    runtime = GatewayServingRuntime(
        servers=[("server-0", rollout_server)],
        load_balancer_handle=load_balancer,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "host": "127.0.0.1",
        },
    )

    prompts = _build_prompts(
        raw_prompt=[[{"role": "user", "content": "Answer briefly."}]],
        uid=["text-only-uid"],
    )

    async def agent_runner(*, raw_prompt, session, sample_index):
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{session.base_url}/chat/completions",
                json={
                    "model": "dummy-model",
                    "messages": raw_prompt,
                },
            )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "FRAMEWORK-TEXT"

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=lambda ctx: [1.0 for _ in ctx.trajectories],
    )
    framework._build_session_id = lambda prompts, sample_index: "session-framework-text"

    output = await framework.generate_sequences(prompts)
    await runtime.shutdown()

    assert len(output) == 1
    assert tu.get(output, "uid") == ["text-only-uid"]
    assert output["position_ids"].ndim == 2
    assert "multi_modal_data" not in output.keys()
    assert "multi_modal_inputs" not in output.keys()
