from __future__ import annotations

import importlib
import pathlib
import sys
from types import SimpleNamespace

import pytest

from verl.utils import tensordict_utils as tu


TESTS_AGENT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _import_adapter_module():
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("recipe", None)
        sys.path[:] = [path for path in sys.path if path != TESTS_AGENT_ROOT]
        return importlib.import_module("recipe.deepeyes_with_gateway.trainer_adapter_tq")
    finally:
        sys.path[:] = original_sys_path


class _DictConfig(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, _DictConfig):
            value = _DictConfig(value)
            self[key] = value
        return value


class _FakeReplayBuffer:
    def __init__(self):
        self.add_calls = []

    def add(self, partition_id, items):
        self.add_calls.append((partition_id, items))


class _FakeFramework:
    def __init__(self, stats):
        self.stats = stats
        self.calls = []

    async def generate_sequences(self, prompts, *, global_steps, partition_id, num_sessions=1):
        self.calls.append(
            {
                "prompts": prompts,
                "global_steps": global_steps,
                "partition_id": partition_id,
                "num_sessions": num_sessions,
            }
        )
        return dict(self.stats)


def _make_config(**overrides):
    config = _DictConfig(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "fake-model",
                    "trust_remote_code": True,
                    "custom_chat_template": "fake-template",
                },
                "rollout": {
                    "n": 2,
                    "val_kwargs": {"n": 3},
                    "multi_turn": {"format": "hermes"},
                    "agent": {
                        "agent_loop_manager_class": (
                            "recipe.deepeyes_with_gateway.trainer_adapter_tq.AgentFrameworkRolloutAdapterTQ"
                        )
                    },
                    "custom": {
                        "agent_framework": {
                            "gateway_count": 4,
                            "max_turns": 5,
                            "tool_config_path": "fake-tool-config.yaml",
                        }
                    },
                },
            },
            "reward": {
                "custom_reward_function": {
                    "path": "recipe/deepeyes_with_gateway/reward.py",
                    "name": "compute_score",
                }
            },
        }
    )
    for key, value in overrides.items():
        config[key] = value
    return config


def _make_prompts(*, validate=False):
    data = {
        "uid": ["uid-0", "uid-1"],
        "raw_prompt": [[{"role": "user", "content": "0"}], [{"role": "user", "content": "1"}]],
        "global_steps": [12, 12],
    }
    if validate:
        data["validate"] = [True, True]
    return tu.get_tensordict(tensor_dict=data)


@pytest.mark.asyncio
async def test_create_accepts_sync_trainer_signature_and_builds_framework(monkeypatch):
    adapter_module = _import_adapter_module()

    built = {}

    class FakeTokenizer:
        chat_template = None

        def decode(self, token_ids, skip_special_tokens=True):
            return "decoded"

    class FakeProcessor:
        chat_template = None

    class FakeRuntime:
        def __init__(self, *, llm_client, gateway_count, gateway_actor_kwargs):
            built["runtime"] = {
                "llm_client": llm_client,
                "gateway_count": gateway_count,
                "gateway_actor_kwargs": gateway_actor_kwargs,
            }

    class FakeFramework:
        def __init__(self, *, session_runtime, agent_runner, reward_fn, processor):
            built["framework"] = {
                "session_runtime": session_runtime,
                "agent_runner": agent_runner,
                "reward_fn": reward_fn,
                "processor": processor,
            }

    monkeypatch.setattr(adapter_module, "hf_tokenizer", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(adapter_module, "hf_processor", lambda *args, **kwargs: FakeProcessor())
    monkeypatch.setattr(adapter_module, "GatewayServingRuntime", FakeRuntime)
    monkeypatch.setattr(adapter_module, "OpenAICompatibleAgentFramework", FakeFramework)
    monkeypatch.setattr(adapter_module, "_build_reward_fn", lambda config, tokenizer: "reward-fn")

    replay_buffer = _FakeReplayBuffer()
    llm_client = SimpleNamespace(
        _server_id_to_handle={"server-0": "handle-0", "server-1": "handle-1"},
        _load_balancer="load-balancer",
    )

    adapter = await adapter_module.AgentFrameworkRolloutAdapterTQ.create(
        config=_make_config(),
        llm_client=llm_client,
        teacher_client="ignored",
        reward_loop_worker_handles=["ignored"],
        replay_buffer=replay_buffer,
    )

    assert isinstance(adapter, adapter_module.AgentFrameworkRolloutAdapterTQ)
    assert adapter.replay_buffer is replay_buffer
    assert built["runtime"]["llm_client"] == llm_client
    assert built["runtime"]["gateway_count"] == 4
    assert built["runtime"]["gateway_actor_kwargs"]["tool_parser_name"] == "hermes"
    assert built["runtime"]["gateway_actor_kwargs"]["tokenizer"].chat_template == "fake-template"
    assert built["runtime"]["gateway_actor_kwargs"]["processor"].chat_template == "fake-template"
    assert built["framework"]["reward_fn"] == "reward-fn"
    assert built["framework"]["agent_runner"].keywords["tool_config_path"] == "fake-tool-config.yaml"
    assert built["framework"]["agent_runner"].keywords["max_turns"] == 5


@pytest.mark.asyncio
async def test_generate_sequences_marks_running_before_framework_call():
    AgentFrameworkRolloutAdapterTQ = _import_adapter_module().AgentFrameworkRolloutAdapterTQ

    adapter = AgentFrameworkRolloutAdapterTQ()
    adapter.replay_buffer = _FakeReplayBuffer()
    adapter.framework = _FakeFramework(
        {
            "num_input_prompts": 2,
            "num_success_sessions": 4,
            "num_failed_sessions": 0,
            "num_success_outputs": 4,
            "num_failed_uids": 0,
            "failure_reasons": [],
        }
    )
    adapter.num_train_sessions = 2
    adapter.num_val_sessions = 3

    await adapter.generate_sequences(_make_prompts())

    assert adapter.replay_buffer.add_calls == [
        (
            "train",
            {
                "uid-0": {"global_steps": 12, "status": "running"},
                "uid-1": {"global_steps": 12, "status": "running"},
            },
        )
    ]
    assert adapter.framework.calls[0]["partition_id"] == "train"
    assert adapter.framework.calls[0]["global_steps"] == 12
    assert adapter.framework.calls[0]["num_sessions"] == 2


@pytest.mark.asyncio
async def test_generate_sequences_uses_val_partition_and_val_n():
    AgentFrameworkRolloutAdapterTQ = _import_adapter_module().AgentFrameworkRolloutAdapterTQ

    adapter = AgentFrameworkRolloutAdapterTQ()
    adapter.replay_buffer = _FakeReplayBuffer()
    adapter.framework = _FakeFramework(
        {
            "num_input_prompts": 2,
            "num_success_sessions": 6,
            "num_failed_sessions": 0,
            "num_success_outputs": 6,
            "num_failed_uids": 0,
            "failure_reasons": [],
        }
    )
    adapter.num_train_sessions = 2
    adapter.num_val_sessions = 3

    await adapter.generate_sequences(_make_prompts(validate=True))

    assert adapter.replay_buffer.add_calls[0][0] == "val"
    assert adapter.framework.calls[0]["partition_id"] == "val"
    assert adapter.framework.calls[0]["num_sessions"] == 3


@pytest.mark.asyncio
async def test_generate_sequences_raises_when_all_outputs_fail():
    AgentFrameworkRolloutAdapterTQ = _import_adapter_module().AgentFrameworkRolloutAdapterTQ

    adapter = AgentFrameworkRolloutAdapterTQ()
    adapter.replay_buffer = _FakeReplayBuffer()
    adapter.framework = _FakeFramework(
        {
            "num_input_prompts": 2,
            "num_success_sessions": 0,
            "num_failed_sessions": 4,
            "num_success_outputs": 0,
            "num_failed_uids": 2,
            "failure_reasons": ["boom"],
        }
    )
    adapter.num_train_sessions = 2
    adapter.num_val_sessions = 3

    with pytest.raises(RuntimeError, match="All rollouts failed at global_steps=12"):
        await adapter.generate_sequences(_make_prompts())

    assert adapter.replay_buffer.add_calls
