import asyncio

import pytest

from verl.experimental.agent_framework.framework import AgentFramework
from verl.experimental.agent_framework.helpers import normalize_trajectory_rewards, validate_trajectory
from verl.experimental.agent_framework.types import Trajectory
from verl.protocol import DataProto


class _MinimalFramework(AgentFramework):
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        return prompts


def test_agent_framework_is_thin_generate_sequences_only():
    assert AgentFramework.__abstractmethods__ == {"generate_sequences"}

    framework = _MinimalFramework()
    prompts = DataProto()
    assert asyncio.run(framework.generate_sequences(prompts)) is prompts


def test_normalize_trajectory_rewards_broadcasts_session_reward_info():
    trajectories = [
        Trajectory(
            uid="sample-0",
            session_id="session-0",
            trajectory_id=0,
            prompt_ids=[11, 12],
            response_ids=[21, 22],
            response_mask=[1, 1],
        ),
        Trajectory(
            uid="sample-0",
            session_id="session-0",
            trajectory_id=1,
            prompt_ids=[11, 12, 13],
            response_ids=[31],
            response_mask=[1],
        ),
    ]

    normalized = normalize_trajectory_rewards(
        trajectories,
        reward_info={"score": 1.25, "format_ok": True, "label": "A"},
        reward_key="score",
    )

    assert [traj.reward_score for traj in normalized] == [1.25, 1.25]
    assert all(traj.reward_info["format_ok"] is True for traj in normalized)
    assert [traj.reward_info["label"] for traj in normalized] == ["A", "A"]


def test_validate_trajectory_rejects_mismatched_response_logprobs():
    trajectory = Trajectory(
        uid="sample-0",
        session_id="session-0",
        trajectory_id=0,
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        response_logprobs=[-0.1],
    )

    with pytest.raises(ValueError, match="response_logprobs"):
        validate_trajectory(trajectory)
