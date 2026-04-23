from __future__ import annotations

import pytest
from tensordict import TensorDict

from tests.agent.support import FakeProcessor
from verl.agent.framework.types import SessionHandle, Trajectory
from verl.utils import tensordict_utils as tu


class _FakeSessionRuntime:
    def __init__(self, finalized_by_session_id: dict[str, list[Trajectory]]):
        self.finalized_by_session_id = finalized_by_session_id

    async def create_session(self, session_id: str, **kwargs):
        return SessionHandle(session_id=session_id, base_url=f"http://fake/{session_id}/v1")

    async def finalize_session(self, session_id: str):
        return self.finalized_by_session_id[session_id]

    async def abort_session(self, session_id: str) -> None:
        return None

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        return None


def _build_prompts(**non_tensor_batch) -> TensorDict:
    return tu.get_tensordict(
        tensor_dict={
            key: list(values)
            for key, values in non_tensor_batch.items()
        }
    )


def _build_trajectory(
    *,
    uid: str,
    session_id: str,
    trajectory_id: int,
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int],
    multi_modal_data: dict | None = None,
) -> Trajectory:
    return Trajectory(
        uid=uid,
        session_id=session_id,
        trajectory_id=trajectory_id,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        reward_info={"score": 1.0},
        reward_score=1.0,
        multi_modal_data=multi_modal_data,
    )


@pytest.mark.asyncio
async def test_openai_compatible_framework_accepts_none_processor():
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    runtime = _FakeSessionRuntime({})

    async def agent_runner(*, raw_prompt, session, sample_index):
        return None

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=lambda ctx: [],
        processor=None,
    )

    assert framework._processor is None


@pytest.mark.asyncio
async def test_openai_compatible_framework_without_processor_keeps_text_position_ids():
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    session_id = "session-text-framework"
    prompts = _build_prompts(
        raw_prompt=[[{"role": "user", "content": "answer briefly"}]],
        uid=["sample-uid"],
    )
    runtime = _FakeSessionRuntime(
        {
            session_id: [
                _build_trajectory(
                    uid="sample-uid",
                    session_id=session_id,
                    trajectory_id=0,
                    prompt_ids=[10, 11],
                    response_ids=[20, 21],
                    response_mask=[1, 1],
                    multi_modal_data={"images": ["image://a.png"]},
                )
            ]
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index):
        assert session.session_id == session_id
        assert sample_index == 0

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=lambda ctx: [1.0],
        processor=None,
    )
    framework._build_session_id = lambda prompts, sample_index: session_id

    output = await framework.generate_sequences(prompts)

    assert tuple(output["position_ids"].shape) == (1, 4)
    assert "multi_modal_data" not in output.keys()
    assert "multi_modal_inputs" not in output.keys()


@pytest.mark.asyncio
async def test_openai_compatible_framework_generate_sequences_adds_multimodal_outputs_when_processor_present():
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    session_id = "session-mm-framework"
    prompts = _build_prompts(
        raw_prompt=[[{"role": "user", "content": "describe the image"}]],
        uid=["sample-uid"],
    )
    runtime = _FakeSessionRuntime(
        {
            session_id: [
                _build_trajectory(
                    uid="sample-uid",
                    session_id=session_id,
                    trajectory_id=0,
                    prompt_ids=[11, FakeProcessor.image_token_id],
                    response_ids=[21, 22],
                    response_mask=[1, 1],
                    multi_modal_data={"images": ["image://a.png"]},
                )
            ]
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index):
        assert raw_prompt == [{"role": "user", "content": "describe the image"}]
        assert session.session_id == session_id
        assert sample_index == 0

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=lambda ctx: [1.0],
        processor=FakeProcessor(),
    )
    framework._build_session_id = lambda prompts, sample_index: session_id

    output = await framework.generate_sequences(prompts)

    multi_modal_data = tu.get(output, "multi_modal_data")
    multi_modal_inputs = tu.get(output, "multi_modal_inputs")

    assert multi_modal_data == [{"images": ["image://a.png"]}]
    assert tuple(output["position_ids"].shape) == (1, 4, 4)
    assert tuple(multi_modal_inputs[0]["pixel_values"].shape) == (1, 3, 2, 2)
    assert multi_modal_inputs[0]["images_seqlens"].tolist() == [6]
