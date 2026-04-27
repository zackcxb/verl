"""Test AgentFrameworkRolloutAdapter DataProto conversion path."""

import asyncio
import functools
import importlib
import numpy as np
import pathlib
import pytest
import sys
import torch
import types

from verl import DataProto

TESTS_AGENT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
MISSING = object()


def _import_adapter_module():
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("recipe", None)
        sys.path[:] = [path for path in sys.path if path != TESTS_AGENT_ROOT]
        return importlib.import_module("recipe.deepeyes_with_gateway.trainer_adapter")
    finally:
        sys.path[:] = original_sys_path


def _import_adapter_class():
    return _import_adapter_module().AgentFrameworkRolloutAdapter


def _make_trainer_input_dataproto(batch_size: int = 2) -> DataProto:
    """Build a DataProto that mimics what trainer sends to generate_sequences."""
    tensors = {
        "dummy_tensor": torch.zeros(batch_size, 1, dtype=torch.uint8),
    }
    non_tensors = {
        "raw_prompt": np.array(
            [
                [{"role": "user", "content": "What is in this image?"}],
                [{"role": "user", "content": "Describe the scene."}],
            ],
            dtype=object,
        ),
        "data_source": np.array(["deepeyes", "deepeyes"], dtype=object),
        "reward_model": np.array(
            [{"ground_truth": "cat"}, {"ground_truth": "dog"}],
            dtype=object,
        ),
        "uid": np.array(["uid-0", "uid-1"], dtype=object),
    }
    dp = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    dp.meta_info["eos_token_id"] = 0
    dp.meta_info["pad_token_id"] = 0
    dp.meta_info["global_steps"] = 1
    return dp


def _make_framework_output(batch_size: int = 2, *, timing: dict | None = None):
    from verl.utils import tensordict_utils as tu

    seq_len = 10
    return tu.get_tensordict(
        tensor_dict={
            "prompts": torch.randint(0, 100, (batch_size, 4)),
            "responses": torch.randint(0, 100, (batch_size, 6)),
            "response_mask": torch.ones(batch_size, 6, dtype=torch.long),
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "rm_scores": torch.zeros(batch_size, 6),
            "__num_turns__": [1] * batch_size,
        },
        non_tensor_dict={} if timing is None else {"timing": timing},
    )


def _make_trainer_config(
    *,
    model_path="fake-model-path",
    gateway_count=MISSING,
    max_turns=MISSING,
    custom_chat_template="custom-template",
    model_trust_remote_code=MISSING,
    data_trust_remote_code=MISSING,
    legacy_custom_reward_function=MISSING,
):
    agent_framework = types.SimpleNamespace()
    if gateway_count is not MISSING:
        agent_framework.gateway_count = gateway_count
    if max_turns is not MISSING:
        agent_framework.max_turns = max_turns

    model = types.SimpleNamespace(
        path=model_path,
        custom_chat_template=custom_chat_template,
    )
    if model_trust_remote_code is not MISSING:
        model.trust_remote_code = model_trust_remote_code

    config = types.SimpleNamespace(
        actor_rollout_ref=types.SimpleNamespace(
            model=model,
            rollout=types.SimpleNamespace(
                custom=types.SimpleNamespace(
                    agent_framework=agent_framework,
                )
            ),
        )
    )
    if data_trust_remote_code is not MISSING:
        config.data = types.SimpleNamespace(trust_remote_code=data_trust_remote_code)
    if legacy_custom_reward_function is not MISSING:
        config.custom_reward_function = legacy_custom_reward_function

    return config


def _make_fake_rollout_replica(captured):
    class FakeRolloutReplica:
        async def start_profile(self, **kwargs):
            captured.setdefault("replica_calls", []).append(("start_profile", kwargs))

        async def stop_profile(self):
            captured.setdefault("replica_calls", []).append(("stop_profile", None))

        async def clear_kv_cache(self):
            captured.setdefault("replica_calls", []).append(("clear_kv_cache", None))

    return FakeRolloutReplica()


def _resolve_maybe_awaitable(value):
    if asyncio.iscoroutine(value):
        return asyncio.run(value)
    return value


def _patch_create_dependencies(adapter_module, monkeypatch, *, processor=MISSING, custom_reward_fn=None):
    captured = {
        "manager_calls": [],
    }
    class FakeTokenizer:
        def __init__(self):
            self.chat_template = None
            self.decode_calls = []

        def decode(self, token_ids, skip_special_tokens=True):
            self.decode_calls.append(
                {
                    "token_ids": list(token_ids),
                    "skip_special_tokens": skip_special_tokens,
                }
            )
            return f"decoded:{','.join(str(token_id) for token_id in token_ids)}"

    fake_tokenizer = FakeTokenizer()
    if processor is MISSING:
        fake_processor = types.SimpleNamespace(chat_template=None)
    else:
        fake_processor = processor

    class FakeAgentLoopManager:
        def __init__(
            self,
            config,
            worker_group=None,
            rollout_resource_pool=None,
            teacher_model_manager=None,
            reward_loop_worker_handles=None,
        ):
            captured["manager_init"] = {
                "config": config,
                "worker_group": worker_group,
                "rollout_resource_pool": rollout_resource_pool,
                "teacher_model_manager": teacher_model_manager,
                "reward_loop_worker_handles": reward_loop_worker_handles,
            }
            self.rollout_replicas = [_make_fake_rollout_replica(captured), _make_fake_rollout_replica(captured)]
            self.server_handles = ["handle-0", "handle-1"]
            self.server_addresses = ["addr-0", "addr-1"]
            self.global_load_balancer = "load-balancer"

        async def _initialize_llm_servers(self):
            captured["manager_calls"].append("_initialize_llm_servers")

        async def _init_global_load_balancer(self):
            captured["manager_calls"].append("_init_global_load_balancer")

        async def _init_agent_loop_workers(self):
            raise AssertionError("create() should not initialize agent loop workers")

    class FakeGatewayServingRuntime:
        def __init__(
            self,
            servers,
            load_balancer_handle,
            *,
            gateway_count=0,
            gateway_actor_kwargs=None,
            **kwargs,
        ):
            captured["runtime_init"] = {
                "servers": servers,
                "load_balancer_handle": load_balancer_handle,
                "gateway_count": gateway_count,
                "gateway_actor_kwargs": gateway_actor_kwargs,
                "extra_kwargs": kwargs,
            }

    class FakeFramework:
        def __init__(
            self,
            session_runtime,
            agent_runner,
            reward_fn,
            *,
            processor=None,
            **kwargs,
        ):
            captured["framework_init"] = {
                "session_runtime": session_runtime,
                "agent_runner": agent_runner,
                "reward_fn": reward_fn,
                "processor": processor,
                "extra_kwargs": kwargs,
            }

    def fake_hf_tokenizer(path, trust_remote_code):
        captured["tokenizer_load"] = {
            "path": path,
            "trust_remote_code": trust_remote_code,
        }
        return fake_tokenizer

    def fake_hf_processor(path, trust_remote_code):
        captured["processor_load"] = {
            "path": path,
            "trust_remote_code": trust_remote_code,
        }
        return fake_processor

    class UnexpectedAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("create() should load the tokenizer through hf_tokenizer()")

    class UnexpectedAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("create() should load the processor through hf_processor()")

    async def fake_stub_agent_runner(*, raw_prompt, session, sample_index, **kwargs):
        del raw_prompt, session, sample_index, kwargs
        return None

    async def fake_deepeyes_agent_runner(*, raw_prompt, session, sample_index, **kwargs):
        del raw_prompt, session, sample_index, kwargs
        return None

    monkeypatch.setattr(adapter_module, "AgentLoopManager", FakeAgentLoopManager, raising=False)
    monkeypatch.setattr(adapter_module, "GatewayServingRuntime", FakeGatewayServingRuntime, raising=False)
    monkeypatch.setattr(adapter_module, "OpenAICompatibleAgentFramework", FakeFramework, raising=False)
    monkeypatch.setattr(adapter_module, "get_custom_reward_fn", lambda config: custom_reward_fn, raising=False)
    monkeypatch.setattr(adapter_module, "hf_tokenizer", fake_hf_tokenizer, raising=False)
    monkeypatch.setattr(adapter_module, "hf_processor", fake_hf_processor, raising=False)
    monkeypatch.setattr(adapter_module, "AutoTokenizer", UnexpectedAutoTokenizer, raising=False)
    monkeypatch.setattr(adapter_module, "AutoProcessor", UnexpectedAutoProcessor, raising=False)
    monkeypatch.setattr(adapter_module, "stub_agent_runner", fake_stub_agent_runner, raising=False)
    monkeypatch.setattr(adapter_module, "deepeyes_agent_runner", fake_deepeyes_agent_runner, raising=False)
    return captured, fake_tokenizer, fake_processor, fake_stub_agent_runner, fake_deepeyes_agent_runner


def test_adapter_generate_sequences_returns_dataproto():
    AgentFrameworkRolloutAdapter = _import_adapter_class()
    from verl.utils import tensordict_utils as tu

    captured = {}

    async def fake_generate(prompts):
        captured["prompts"] = prompts
        return _make_framework_output(batch_size=len(prompts), timing={"framework": 1.5})

    class FakeFramework:
        async def generate_sequences(self, prompts):
            return await fake_generate(prompts)

    adapter = AgentFrameworkRolloutAdapter.__new__(AgentFrameworkRolloutAdapter)
    adapter._framework = FakeFramework()
    input_dp = _make_trainer_input_dataproto(batch_size=2)

    output_dp = adapter.generate_sequences(input_dp)

    assert isinstance(output_dp, DataProto)
    assert "prompts" in output_dp.batch
    assert "responses" in output_dp.batch
    assert "input_ids" in output_dp.batch
    assert "attention_mask" in output_dp.batch
    assert "rm_scores" in output_dp.batch
    assert "timing" in output_dp.meta_info
    assert output_dp.meta_info["timing"]["framework"] == 1.5
    assert "gen" in output_dp.meta_info["timing"]
    assert output_dp.non_tensor_batch["data_source"].tolist() == ["deepeyes", "deepeyes"]
    assert output_dp.non_tensor_batch["reward_model"].tolist() == [
        {"ground_truth": "cat"},
        {"ground_truth": "dog"},
    ]
    assert output_dp.non_tensor_batch["uid"].tolist() == ["uid-0", "uid-1"]
    assert "raw_prompt" not in output_dp.non_tensor_batch
    assert captured["prompts"]["dummy_tensor"].shape == (2, 1)
    assert tu.get(captured["prompts"], "raw_prompt") == input_dp.non_tensor_batch["raw_prompt"].tolist()


def test_adapter_prebinds_per_sample_tools_kwargs_for_framework_runner():
    AgentFrameworkRolloutAdapter = _import_adapter_class()
    from verl.agent.framework.types import SessionHandle

    runner_calls = []

    async def base_runner(*, raw_prompt, session, sample_index, tools_kwargs=None):
        runner_calls.append(
            {
                "raw_prompt": raw_prompt,
                "session": session,
                "sample_index": sample_index,
                "tools_kwargs": tools_kwargs,
            }
        )

    class FakeFramework:
        def __init__(self):
            self.agent_runner = base_runner

        async def generate_sequences(self, prompts):
            await self.agent_runner(
                raw_prompt=prompts["raw_prompt"][1],
                session=SessionHandle(session_id="session-1", base_url="http://gateway/session-1/v1"),
                sample_index=1,
            )
            return _make_framework_output(batch_size=len(prompts))

    adapter = AgentFrameworkRolloutAdapter.__new__(AgentFrameworkRolloutAdapter)
    adapter._framework = FakeFramework()
    input_dp = _make_trainer_input_dataproto(batch_size=2)
    input_dp.non_tensor_batch["tools_kwargs"] = np.array(
        [
            {"image_zoom_in_tool": {"create_kwargs": {"image": "image-0"}}},
            {"image_zoom_in_tool": {"create_kwargs": {"image": "image-1"}}},
        ],
        dtype=object,
    )

    adapter.generate_sequences(input_dp)

    assert runner_calls == [
        {
            "raw_prompt": [{"role": "user", "content": "Describe the scene."}],
            "session": SessionHandle(session_id="session-1", base_url="http://gateway/session-1/v1"),
            "sample_index": 1,
            "tools_kwargs": {"image_zoom_in_tool": {"create_kwargs": {"image": "image-1"}}},
        }
    ]


def test_adapter_generate_sequences_rejects_cardinality_mismatch_for_backfill():
    AgentFrameworkRolloutAdapter = _import_adapter_class()

    class FakeFramework:
        async def generate_sequences(self, prompts):
            return _make_framework_output(batch_size=len(prompts) + 1)

    adapter = AgentFrameworkRolloutAdapter.__new__(AgentFrameworkRolloutAdapter)
    adapter._framework = FakeFramework()
    input_dp = _make_trainer_input_dataproto(batch_size=2)

    with pytest.raises(ValueError, match="Cannot backfill"):
        adapter.generate_sequences(input_dp)


def test_adapter_create_wires_trainer_path_into_runtime_and_framework(monkeypatch):
    adapter_module = _import_adapter_module()
    AgentFrameworkRolloutAdapter = adapter_module.AgentFrameworkRolloutAdapter
    (
        captured,
        fake_tokenizer,
        fake_processor,
        _,
        fake_deepeyes_agent_runner,
    ) = _patch_create_dependencies(adapter_module, monkeypatch)
    config = _make_trainer_config(gateway_count=4, max_turns=7, custom_chat_template="chat-template")

    adapter = AgentFrameworkRolloutAdapter.create(
        config=config,
        worker_group="worker-group",
        rollout_resource_pool="resource-pool",
        reward_loop_worker_handles=["reward-worker"],
        teacher_model_manager="teacher-manager",
        replay_buffer="replay-buffer",
    )

    assert isinstance(adapter, AgentFrameworkRolloutAdapter)
    assert captured["manager_init"] == {
        "config": config,
        "worker_group": "worker-group",
        "rollout_resource_pool": "resource-pool",
        "teacher_model_manager": "teacher-manager",
        "reward_loop_worker_handles": ["reward-worker"],
    }
    assert captured["manager_calls"] == [
        "_initialize_llm_servers",
        "_init_global_load_balancer",
    ]
    assert len(adapter.rollout_replicas) == 2
    assert adapter._runtime is not None

    assert captured["tokenizer_load"] == {
        "path": "fake-model-path",
        "trust_remote_code": True,
    }
    assert captured["processor_load"] == {
        "path": "fake-model-path",
        "trust_remote_code": True,
    }
    assert fake_tokenizer.chat_template == "chat-template"
    assert fake_processor.chat_template == "chat-template"

    assert captured["runtime_init"] == {
        "servers": [("addr-0", "handle-0"), ("addr-1", "handle-1")],
        "load_balancer_handle": "load-balancer",
        "gateway_count": 4,
        "gateway_actor_kwargs": {
            "tokenizer": fake_tokenizer,
            "processor": fake_processor,
            "host": None,
        },
        "extra_kwargs": {},
    }
    assert adapter._runtime is captured["framework_init"]["session_runtime"]
    assert isinstance(captured["framework_init"]["agent_runner"], functools.partial)
    assert captured["framework_init"]["agent_runner"].func is fake_deepeyes_agent_runner
    assert captured["framework_init"]["agent_runner"].keywords == {"max_turns": 7, "tool_config": {}}
    assert captured["framework_init"]["processor"] is fake_processor
    assert captured["framework_init"]["extra_kwargs"] == {}
    reward_context = types.SimpleNamespace(trajectories=["traj-0", "traj-1", "traj-2"])
    assert captured["framework_init"]["reward_fn"](reward_context) == [0.0, 0.0, 0.0]


def test_adapter_create_uses_custom_reward_fn_with_decoded_responses(monkeypatch):
    adapter_module = _import_adapter_module()
    AgentFrameworkRolloutAdapter = adapter_module.AgentFrameworkRolloutAdapter
    reward_calls = []

    def fake_custom_reward_fn(data_source, solution_str, ground_truth, extra_info):
        reward_calls.append((data_source, solution_str, ground_truth, extra_info))
        return {
            "decoded:1,2": 1.5,
            "decoded:3": 2.5,
        }[solution_str]

    captured, fake_tokenizer, _, _, _ = _patch_create_dependencies(
        adapter_module,
        monkeypatch,
        custom_reward_fn=fake_custom_reward_fn,
    )
    adapter = AgentFrameworkRolloutAdapter.create(config=_make_trainer_config())
    reward_fn = captured["framework_init"]["reward_fn"]
    reward_context = types.SimpleNamespace(
        trajectories=[
            types.SimpleNamespace(response_ids=[1, 2]),
            types.SimpleNamespace(response_ids=[3]),
        ],
        sample_fields={
            "data_source": "deepeyes",
            "reward_model": {"ground_truth": "gold-answer"},
            "extra_info": {"sample_id": 7},
        },
    )

    scores = _resolve_maybe_awaitable(reward_fn(reward_context))

    assert scores == [1.5, 2.5]
    assert reward_calls == [
        ("deepeyes", "decoded:1,2", "gold-answer", {"sample_id": 7}),
        ("deepeyes", "decoded:3", "gold-answer", {"sample_id": 7}),
    ]
    assert fake_tokenizer.decode_calls == [
        {"token_ids": [1, 2], "skip_special_tokens": True},
        {"token_ids": [3], "skip_special_tokens": True},
    ]
    assert adapter._runtime is captured["framework_init"]["session_runtime"]


def test_adapter_create_uses_legacy_top_level_custom_reward_config_when_nested_loader_missing(monkeypatch):
    adapter_module = _import_adapter_module()
    AgentFrameworkRolloutAdapter = adapter_module.AgentFrameworkRolloutAdapter
    reward_calls = []

    def fake_legacy_reward_fn(data_source, solution_str, ground_truth, extra_info, bonus):
        reward_calls.append((data_source, solution_str, ground_truth, extra_info, bonus))
        return bonus + 0.5

    captured, fake_tokenizer, _, _, _ = _patch_create_dependencies(
        adapter_module,
        monkeypatch,
        custom_reward_fn=None,
    )
    monkeypatch.setattr(
        adapter_module,
        "load_extern_object",
        lambda module_path, object_name: fake_legacy_reward_fn,
        raising=False,
    )
    config = _make_trainer_config(
        legacy_custom_reward_function=types.SimpleNamespace(
            path="file://legacy_reward.py",
            name="compute_score",
            reward_kwargs={"bonus": 3.0},
        )
    )
    adapter = AgentFrameworkRolloutAdapter.create(config=config)
    reward_fn = captured["framework_init"]["reward_fn"]
    reward_context = types.SimpleNamespace(
        trajectories=[types.SimpleNamespace(response_ids=[9, 8])],
        sample_fields={
            "data_source": "deepeyes",
            "reward_model": {"ground_truth": "gold-answer"},
            "extra_info": {"sample_id": 9},
        },
    )

    scores = _resolve_maybe_awaitable(reward_fn(reward_context))

    assert scores == [3.5]
    assert reward_calls == [
        ("deepeyes", "decoded:9,8", "gold-answer", {"sample_id": 9}, 3.0),
    ]
    assert fake_tokenizer.decode_calls[-1] == {
        "token_ids": [9, 8],
        "skip_special_tokens": True,
    }
    assert adapter._runtime is captured["framework_init"]["session_runtime"]


def test_adapter_create_validates_model_path_before_manager_init(monkeypatch):
    adapter_module = _import_adapter_module()
    AgentFrameworkRolloutAdapter = adapter_module.AgentFrameworkRolloutAdapter

    class UnexpectedManagerInit:
        def __init__(self, *args, **kwargs):
            raise AssertionError("create() should fail preflight before initializing the legacy manager")

    monkeypatch.setattr(adapter_module, "AgentLoopManager", UnexpectedManagerInit, raising=False)
    config = _make_trainer_config(model_path=None)

    with pytest.raises(ValueError, match="model.path"):
        AgentFrameworkRolloutAdapter.create(config=config)


@pytest.mark.parametrize(
    ("model_trust_remote_code", "data_trust_remote_code", "expected"),
    [
        (False, True, False),
        (MISSING, False, False),
        (MISSING, MISSING, True),
    ],
)
def test_adapter_create_reads_trust_remote_code_from_config(
    monkeypatch,
    model_trust_remote_code,
    data_trust_remote_code,
    expected,
):
    adapter_module = _import_adapter_module()
    AgentFrameworkRolloutAdapter = adapter_module.AgentFrameworkRolloutAdapter
    captured, _, _, _, _ = _patch_create_dependencies(adapter_module, monkeypatch)
    config = _make_trainer_config(
        model_trust_remote_code=model_trust_remote_code,
        data_trust_remote_code=data_trust_remote_code,
    )

    AgentFrameworkRolloutAdapter.create(config=config)

    assert captured["tokenizer_load"]["trust_remote_code"] is expected
    assert captured["processor_load"]["trust_remote_code"] is expected


def test_adapter_create_defaults_gateway_count_to_server_count(monkeypatch):
    adapter_module = _import_adapter_module()
    AgentFrameworkRolloutAdapter = adapter_module.AgentFrameworkRolloutAdapter
    captured, _, _, _, _ = _patch_create_dependencies(adapter_module, monkeypatch)
    config = _make_trainer_config(gateway_count=MISSING, max_turns=3)

    AgentFrameworkRolloutAdapter.create(config=config)

    assert captured["runtime_init"]["gateway_count"] == 2


def test_adapter_create_uses_agent_runner_default_when_max_turns_missing(monkeypatch):
    adapter_module = _import_adapter_module()
    AgentFrameworkRolloutAdapter = adapter_module.AgentFrameworkRolloutAdapter
    captured, _, _, _, fake_deepeyes_agent_runner = _patch_create_dependencies(adapter_module, monkeypatch)
    config = _make_trainer_config(gateway_count=1, max_turns=MISSING)

    AgentFrameworkRolloutAdapter.create(config=config)

    assert captured["framework_init"]["agent_runner"].func is fake_deepeyes_agent_runner
    assert captured["framework_init"]["agent_runner"].keywords == {"tool_config": {}}


def test_adapter_create_propagates_none_processor_from_helper(monkeypatch):
    adapter_module = _import_adapter_module()
    AgentFrameworkRolloutAdapter = adapter_module.AgentFrameworkRolloutAdapter
    captured, fake_tokenizer, fake_processor, _, _ = _patch_create_dependencies(
        adapter_module,
        monkeypatch,
        processor=None,
    )
    config = _make_trainer_config(custom_chat_template="chat-template")

    adapter = AgentFrameworkRolloutAdapter.create(config=config)

    assert fake_processor is None
    assert fake_tokenizer.chat_template == "chat-template"
    assert captured["runtime_init"]["gateway_actor_kwargs"]["processor"] is None
    assert captured["framework_init"]["processor"] is None
    assert adapter._runtime is captured["framework_init"]["session_runtime"]


def test_adapter_profile_and_cache_calls_forward_to_rollout_replicas():
    AgentFrameworkRolloutAdapter = _import_adapter_class()
    captured = {}
    adapter = AgentFrameworkRolloutAdapter()
    adapter._rollout_replicas = [_make_fake_rollout_replica(captured), _make_fake_rollout_replica(captured)]

    assert adapter.start_profile(tag="warmup") is None
    assert adapter.stop_profile() is None
    assert adapter.clear_kv_cache() is None
    assert captured["replica_calls"] == [
        ("start_profile", {"tag": "warmup"}),
        ("start_profile", {"tag": "warmup"}),
        ("stop_profile", None),
        ("stop_profile", None),
        ("clear_kv_cache", None),
        ("clear_kv_cache", None),
    ]
