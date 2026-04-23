import httpx
import pytest
import ray

from tests.agent.support import (
    FailingRolloutServer,
    FakeProcessor,
    FakeTokenizer,
    QueuedBackend,
    RecordingLoadBalancer,
    RecordingRolloutServer,
    fake_vision_info_extractor,
)


@pytest.fixture
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_gateway_serving_runtime_owns_gateway_lifecycle_and_session_runtime(ray_runtime):
    from verl.agent.gateway.runtime import GatewayServingRuntime

    runtime = GatewayServingRuntime(
        servers=[],
        load_balancer_handle=None,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "backend": QueuedBackend(["OWNER"]),
            "host": "127.0.0.1",
        },
    )

    session = await runtime.create_session("session-owner")
    wait_task = runtime.wait_for_completion("session-owner", timeout=2.0)

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "owner path"}]},
        )
        assert response.status_code == 200

        complete = await client.post(
            f"{session.base_url.removesuffix('/v1')}/complete",
            json={"reward_info": {"score": 0.5, "label": "owner"}},
        )
        assert complete.status_code == 200

    await wait_task
    trajectories = await runtime.finalize_session("session-owner")
    await runtime.shutdown()

    assert len(trajectories) == 1
    assert trajectories[0].reward_info == {"score": 0.5, "label": "owner"}


@pytest.mark.asyncio
async def test_gateway_serving_runtime_injects_runtime_owned_gateway_backend(ray_runtime):
    from verl.agent.gateway.runtime import GatewayServingRuntime

    rollout_server = RecordingRolloutServer.remote("MANAGED")
    load_balancer = RecordingLoadBalancer.remote("server-0")
    runtime = GatewayServingRuntime(
        servers=[("server-0", rollout_server)],
        load_balancer_handle=load_balancer,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "host": "127.0.0.1",
            "base_sampling_params": {"temperature": 0.4, "top_p": 0.8},
            "allowed_request_sampling_param_keys": {"temperature", "top_k"},
        },
    )

    session = await runtime.create_session("session-managed-backend")
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "temperature": 0.2,
                "top_k": 4,
                "presence_penalty": 1.5,
                "messages": [{"role": "user", "content": "managed path"}],
            },
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "MANAGED"

    trajectories = await runtime.finalize_session("session-managed-backend")
    await runtime.shutdown()

    stats = ray.get(load_balancer.stats.remote())
    calls = ray.get(rollout_server.get_calls.remote())

    assert len(trajectories) == 1
    assert len(stats["acquire_calls"]) == 1
    assert stats["release_calls"] == ["server-0"]
    assert len(calls) == 1
    assert calls[0]["prompt_ids"]
    assert calls[0]["sampling_params"] == {"temperature": 0.2, "top_p": 0.8, "top_k": 4}


@pytest.mark.asyncio
async def test_gateway_serving_runtime_passes_processor_and_media_to_owned_gateway(ray_runtime):
    from verl.agent.gateway.runtime import GatewayServingRuntime

    rollout_server = RecordingRolloutServer.remote("MM")
    load_balancer = RecordingLoadBalancer.remote("server-0")
    runtime = GatewayServingRuntime(
        servers=[("server-0", rollout_server)],
        load_balancer_handle=load_balancer,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "processor": FakeProcessor(),
            "vision_info_extractor": fake_vision_info_extractor,
            "host": "127.0.0.1",
        },
    )

    session = await runtime.create_session("session-runtime-processor")
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "image://runtime.png"}},
                            {"type": "text", "text": "describe this image"},
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "MM"

    trajectories = await runtime.finalize_session("session-runtime-processor")
    await runtime.shutdown()

    calls = ray.get(rollout_server.get_calls.remote())

    assert len(trajectories) == 1
    assert len(calls) == 1
    assert calls[0]["image_data"] == ["image://runtime.png"]
    assert calls[0]["video_data"] is None


@pytest.mark.asyncio
async def test_gateway_serving_runtime_releases_server_when_generate_fails(ray_runtime):
    from verl.agent.gateway.runtime import GatewayServingRuntime

    rollout_server = FailingRolloutServer.remote("boom")
    load_balancer = RecordingLoadBalancer.remote("server-0")
    runtime = GatewayServingRuntime(
        servers=[("server-0", rollout_server)],
        load_balancer_handle=load_balancer,
        gateway_count=0,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await runtime.generate(
            "request-failure",
            prompt_ids=[1, 2, 3],
            sampling_params={"temperature": 0.1},
        )

    stats = ray.get(load_balancer.stats.remote())
    calls = ray.get(rollout_server.get_calls.remote())

    assert stats["acquire_calls"] == ["request-failure"]
    assert stats["release_calls"] == ["server-0"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_gateway_serving_runtime_gateway_count_zero_falls_back_to_generate_only_mode(ray_runtime):
    from verl.agent.gateway.runtime import GatewayServingRuntime

    rollout_server = RecordingRolloutServer.remote("FALLBACK")
    load_balancer = RecordingLoadBalancer.remote("server-0")
    runtime = GatewayServingRuntime(
        servers=[("server-0", rollout_server)],
        load_balancer_handle=load_balancer,
        gateway_count=0,
    )

    output = await runtime.generate(
        "request-fallback",
        prompt_ids=[4, 5],
        sampling_params={"temperature": 0.2},
    )

    stats = ray.get(load_balancer.stats.remote())
    calls = ray.get(rollout_server.get_calls.remote())

    assert runtime.owned_gateway_actors == []
    assert runtime.gateway_manager is None
    assert output.token_ids == [ord(char) for char in "FALLBACK"]
    assert stats["acquire_calls"] == ["request-fallback"]
    assert stats["release_calls"] == ["server-0"]
    assert len(calls) == 1
