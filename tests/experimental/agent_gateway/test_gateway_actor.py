import asyncio

import httpx
import pytest
import ray

from tests.experimental.agent_gateway.support import FakeTokenizer, QueuedBackend, RejectConcurrentSessionBackend


@pytest.fixture
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_gateway_actor_complete_wait_and_finalize(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["ANSWER: A"]), host="127.0.0.1")
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-0"))
    wait_ref = actor.wait_for_completion.remote("session-0", timeout=2.0)

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Pick label A"}],
            },
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "ANSWER: A"

        complete = await client.post(
            f"{session.base_url.removesuffix('/v1')}/complete",
            json={"reward_info": {"score": 1.0, "label": "A"}},
        )
        assert complete.status_code == 200

    ray.get(wait_ref)
    trajectories = ray.get(actor.finalize_session.remote("session-0"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 1
    assert trajectories[0].reward_info == {"score": 1.0, "label": "A"}
    assert trajectories[0].trajectory_id == 0
    assert trajectories[0].response_ids
    assert all(mask == 1 for mask in trajectories[0].response_mask)


@pytest.mark.asyncio
async def test_gateway_actor_prefix_mismatch_splits_trajectories(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["FIRST", "SECOND"]), host="127.0.0.1")
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-1"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "first turn"}],
            },
        )
        assert first.status_code == 200

        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "replacement context"}],
            },
        )
        assert second.status_code == 200

    trajectories = ray.get(actor.finalize_session.remote("session-1"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 2
    assert [trajectory.trajectory_id for trajectory in trajectories] == [0, 1]
    assert trajectories[0].prompt_ids != trajectories[1].prompt_ids


def test_gateway_actor_abort_discards_session(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["unused"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    ray.get(actor.create_session.remote("session-abort"))

    ray.get(actor.abort_session.remote("session-abort"))

    with pytest.raises(ray.exceptions.RayTaskError, match="session-abort"):
        ray.get(actor.finalize_session.remote("session-abort"))

    ray.get(actor.shutdown.remote())


@pytest.mark.asyncio
async def test_gateway_actor_serializes_same_session_concurrent_requests(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(
        tokenizer=FakeTokenizer(),
        backend=RejectConcurrentSessionBackend(["FIRST", "SECOND"]),
        host="127.0.0.1",
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-concurrent"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        async def send_request():
            return await client.post(
                f"{session.base_url}/chat/completions",
                json={
                    "model": "dummy-model",
                    "messages": [{"role": "user", "content": "same session prompt"}],
                },
            )

        first, second = await asyncio.gather(send_request(), send_request())

    trajectories = ray.get(actor.finalize_session.remote("session-concurrent"))
    ray.get(actor.shutdown.remote())

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(trajectories) == 2
    assert trajectories[0].response_ids == [ord(char) for char in "FIRST"]
    assert trajectories[1].response_ids == [ord(char) for char in "SECOND"]
    assert trajectories[0].response_mask == [1] * len("FIRST")
    assert trajectories[1].response_mask == [1] * len("SECOND")


@pytest.mark.asyncio
async def test_gateway_actor_session_state_tracks_metadata_flags_and_timestamps(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-state", metadata={"uid": "sample-7", "split": "train"}))
    created_state = ray.get(actor.get_session_state.remote("session-state"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "track state"}]},
        )
        assert response.status_code == 200

        completed = await client.post(
            f"{session.base_url.removesuffix('/v1')}/complete",
            json={"reward_info": {"score": 1.0}},
        )
        assert completed.status_code == 200

    completed_state = ray.get(actor.get_session_state.remote("session-state"))
    trajectories = ray.get(actor.finalize_session.remote("session-state"))
    ray.get(actor.shutdown.remote())

    assert created_state["metadata"] == {"uid": "sample-7", "split": "train"}
    assert created_state["completed_flag"] is False
    assert created_state["aborted_flag"] is False
    assert created_state["created_at"] <= created_state["updated_at"]
    assert completed_state["completed_flag"] is True
    assert completed_state["aborted_flag"] is False
    assert completed_state["updated_at"] >= created_state["updated_at"]
    assert len(trajectories) == 1
