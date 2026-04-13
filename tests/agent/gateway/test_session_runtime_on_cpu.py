import httpx
import pytest
import ray

from tests.agent.support import (
    FakeTokenizer,
    QueuedBackend,
    RecordingLoadBalancer,
    RecordingRolloutServer,
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
        },
    )

    session = await runtime.create_session("session-managed-backend")
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "managed path"}]},
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
