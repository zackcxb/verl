import asyncio

import httpx
import pytest
import ray

from tests.experimental.agent_gateway.support import (
    AssertingQueuedBackend,
    FakeDeltaTokenizer,
    FakeProcessor,
    FakeTokenizer,
    FakeToolParser,
    QueuedBackend,
    RejectConcurrentSessionBackend,
)


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


@pytest.mark.asyncio
async def test_gateway_actor_prefix_continuation_uses_chat_template_delta(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    tokenizer = FakeDeltaTokenizer()
    first_messages = [{"role": "user", "content": "first turn"}]
    second_messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "follow up"},
    ]
    first_prompt = tokenizer.apply_chat_template(first_messages, tokenize=True, add_generation_prompt=True)
    prev_prompt_text = tokenizer.apply_chat_template(
        first_messages + [{"role": "assistant", "content": "FIRST"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    curr_prompt_text = tokenizer.apply_chat_template(second_messages, tokenize=False, add_generation_prompt=True)
    delta_ids = tokenizer.encode(curr_prompt_text[len(prev_prompt_text) :], add_special_tokens=False)

    backend = AssertingQueuedBackend(
        ["FIRST", "SECOND"],
        [
            first_prompt,
            first_prompt + [ord(char) for char in "FIRST"] + delta_ids,
        ],
    )
    actor = GatewayActor.remote(tokenizer=tokenizer, backend=backend, host="127.0.0.1")
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-prefix-delta"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": first_messages},
        )
        assert first.status_code == 200

        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": second_messages},
        )
        assert second.status_code == 200

    trajectories = ray.get(actor.finalize_session.remote("session-prefix-delta"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 1


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
async def test_gateway_actor_returns_openai_tool_calls(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(
        tokenizer=FakeTokenizer(),
        tool_parser=FakeToolParser(),
        backend=QueuedBackend(['<tool_call>{"name":"search","arguments":{"query":"weather"}}</tool_call>']),
        host="127.0.0.1",
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-tool-calls"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "use a tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "search",
                            "parameters": {"type": "object", "properties": {}, "required": []},
                        },
                    }
                ],
            },
        )

    ray.get(actor.shutdown.remote())

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-0",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "weather"}',
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_gateway_actor_processor_path_supports_multimodal_delta(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    tokenizer = FakeTokenizer()
    processor = FakeProcessor()
    first_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "image": "img-1"},
            ],
        }
    ]
    second_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "image": "img-1"},
            ],
        },
        {"role": "assistant", "content": "FIRST"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "more"},
                {"type": "image", "image": "img-2"},
            ],
        },
    ]

    first_rendered = processor.apply_chat_template(first_messages, tokenize=False, add_generation_prompt=True)
    first_prompt_ids = processor(text=[first_rendered], images=["img-1"], videos=None)["input_ids"][0].tolist()

    prev_rendered = processor.apply_chat_template(
        first_messages + [{"role": "assistant", "content": "FIRST"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prev_ids = processor(text=[prev_rendered], images=["img-1"], videos=None)["input_ids"][0].tolist()
    curr_rendered = processor.apply_chat_template(second_messages, tokenize=False, add_generation_prompt=True)
    curr_ids = processor(text=[curr_rendered], images=["img-1", "img-2"], videos=None)["input_ids"][0].tolist()
    delta_ids = curr_ids[len(prev_ids) :]

    backend = AssertingQueuedBackend(
        ["FIRST", "SECOND"],
        [
            first_prompt_ids,
            first_prompt_ids + [ord(char) for char in "FIRST"] + delta_ids,
        ],
        expected_image_data_per_call=[
            ["img-1"],
            ["img-1", "img-2"],
        ],
    )
    actor = GatewayActor.remote(
        tokenizer=tokenizer,
        processor=processor,
        backend=backend,
        host="127.0.0.1",
    )
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-multimodal"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": first_messages},
        )
        assert first.status_code == 200

        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": second_messages},
        )
        assert second.status_code == 200

    trajectories = ray.get(actor.finalize_session.remote("session-multimodal"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 1


@pytest.mark.asyncio
async def test_gateway_actor_session_state_tracks_metadata_phase_and_timestamps(ray_runtime):
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
    assert created_state["phase"] == "ACTIVE"
    assert "completed_flag" not in created_state
    assert "aborted_flag" not in created_state
    assert created_state["created_at"] <= created_state["updated_at"]
    assert completed_state["phase"] == "COMPLETED"
    assert "completed_flag" not in completed_state
    assert "aborted_flag" not in completed_state
    assert completed_state["updated_at"] >= created_state["updated_at"]
    assert len(trajectories) == 1

    with pytest.raises(ray.exceptions.RayTaskError, match="session-state"):
        ray.get(actor.get_session_state.remote("session-state"))
