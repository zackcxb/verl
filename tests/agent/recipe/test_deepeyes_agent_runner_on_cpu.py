"""CPU tests for the DeepEyes gateway agent runner."""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
from typing import Any

import pytest
from PIL import Image

from verl.agent.framework.types import SessionHandle
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

TESTS_AGENT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _import_agent_runner_module():
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("recipe", None)
        sys.path[:] = [path for path in sys.path if path != TESTS_AGENT_ROOT]
        return importlib.import_module("recipe.deepeyes_with_gateway.agent_runner")
    finally:
        sys.path[:] = original_sys_path


def _chat_response(*, content: str = "", tool_calls: list[dict[str, Any]] | None = None):
    message = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [
            {
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ]
    }


def _tool_call(arguments: dict[str, Any], *, call_id: str = "call_zoom"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "image_zoom_in_tool",
            "arguments": json.dumps(arguments),
        },
    }


class FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def _patch_http_client(monkeypatch, agent_runner_module, responses: list[dict[str, Any]]):
    calls: list[dict[str, Any]] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.init_args = args
            self.init_kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *, json):
            calls.append(
                {
                    "url": url,
                    "json": {
                        **json,
                        "messages": list(json["messages"]),
                    },
                }
            )
            return FakeResponse(responses.pop(0))

    monkeypatch.setattr(agent_runner_module.httpx, "AsyncClient", FakeAsyncClient)
    return calls


class FakeImageZoomInTool:
    instances: list["FakeImageZoomInTool"] = []

    def __init__(self, config: dict[str, Any], tool_schema):
        self.config = config
        self.tool_schema = tool_schema
        self.create_calls = []
        self.execute_calls = []
        self.release_calls = []
        FakeImageZoomInTool.instances.append(self)

    def get_openai_tool_schema(self):
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "image_zoom_in_tool",
                    "description": "Zoom in on an image region.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bbox_2d": {"type": "array", "description": "Bounding box."},
                            "label": {"type": "string", "description": "Object label."},
                        },
                        "required": ["bbox_2d"],
                    },
                },
            }
        )

    async def create(self, instance_id=None, **kwargs):
        self.create_calls.append({"instance_id": instance_id, "kwargs": kwargs})
        return "tool-instance-0", ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        self.execute_calls.append(
            {
                "instance_id": instance_id,
                "parameters": parameters,
                "kwargs": kwargs,
            }
        )
        image = Image.new("RGB", (16, 16), color="blue")
        return ToolResponse(text="Zoomed in on the requested region.", image=[image]), 0.0, {"success": True}

    async def release(self, instance_id: str, **kwargs):
        self.release_calls.append({"instance_id": instance_id, "kwargs": kwargs})


@pytest.fixture(autouse=True)
def reset_fake_tool():
    FakeImageZoomInTool.instances = []
    yield
    FakeImageZoomInTool.instances = []


@pytest.mark.asyncio
async def test_deepeyes_agent_runner_executes_tool_call_and_continues(monkeypatch):
    agent_runner = _import_agent_runner_module()
    monkeypatch.setattr(agent_runner, "ImageZoomInTool", FakeImageZoomInTool)
    calls = _patch_http_client(
        monkeypatch,
        agent_runner,
        [
            _chat_response(tool_calls=[_tool_call({"bbox_2d": [0, 0, 32, 32], "label": "cat"})]),
            _chat_response(content="Final answer."),
        ],
    )
    original_image = Image.new("RGB", (64, 64), color="white")

    await agent_runner.deepeyes_agent_runner(
        raw_prompt=[{"role": "user", "content": "Inspect this image."}],
        session=SessionHandle(session_id="session-0", base_url="http://gateway/sessions/session-0/v1"),
        sample_index=0,
        tools_kwargs={
            "image_zoom_in_tool": {
                "create_kwargs": {"image": original_image},
                "execute_kwargs": {"extra": "value"},
                "release_kwargs": {"cleanup": True},
            }
        },
        tool_config={
            "config": {"num_workers": 8, "rate_limit": 8},
            "tool_schema": {
                "type": "function",
                "function": {
                    "name": "image_zoom_in_tool",
                    "description": "Zoom in on an image region.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bbox_2d": {"type": "array", "description": "Bounding box."},
                            "label": {"type": "string", "description": "Object label."},
                        },
                        "required": ["bbox_2d"],
                    },
                },
            },
        },
        max_turns=5,
    )

    assert len(calls) == 2
    assert calls[0]["url"] == "http://gateway/sessions/session-0/v1/chat/completions"
    assert calls[0]["json"]["tools"][0]["function"]["name"] == "image_zoom_in_tool"
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "Inspect this image."}]

    tool = FakeImageZoomInTool.instances[0]
    assert tool.create_calls == [
        {
            "instance_id": "session-0-image_zoom_in_tool",
            "kwargs": {"create_kwargs": {"image": original_image}},
        }
    ]
    assert tool.execute_calls == [
        {
            "instance_id": "tool-instance-0",
            "parameters": {"bbox_2d": [0, 0, 32, 32], "label": "cat"},
            "kwargs": {"extra": "value"},
        }
    ]
    assert tool.release_calls == [
        {
            "instance_id": "tool-instance-0",
            "kwargs": {"cleanup": True},
        }
    ]

    continuation_messages = calls[1]["json"]["messages"]
    assert continuation_messages[1]["role"] == "assistant"
    assert continuation_messages[1]["tool_calls"][0]["id"] == "call_zoom"
    assert continuation_messages[2]["role"] == "tool"
    assert continuation_messages[2]["tool_call_id"] == "call_zoom"
    assert continuation_messages[2]["content"][0] == {
        "type": "text",
        "text": "Zoomed in on the requested region.",
    }
    assert continuation_messages[2]["content"][1]["type"] == "image"
    assert isinstance(continuation_messages[2]["content"][1]["image"], Image.Image)


class FakeFailingImageZoomInTool(FakeImageZoomInTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        self.execute_calls.append(
            {
                "instance_id": instance_id,
                "parameters": parameters,
                "kwargs": kwargs,
            }
        )
        raise RuntimeError("tool execute failed")


@pytest.mark.asyncio
async def test_deepeyes_agent_runner_releases_tool_on_execute_failure(monkeypatch):
    agent_runner = _import_agent_runner_module()
    monkeypatch.setattr(agent_runner, "ImageZoomInTool", FakeFailingImageZoomInTool)
    calls = _patch_http_client(
        monkeypatch,
        agent_runner,
        [_chat_response(tool_calls=[_tool_call({"bbox_2d": [0, 0, 32, 32]})])],
    )

    with pytest.raises(RuntimeError, match="tool execute failed"):
        await agent_runner.deepeyes_agent_runner(
            raw_prompt=[{"role": "user", "content": "Inspect this image."}],
            session=SessionHandle(session_id="session-fail", base_url="http://gateway/sessions/session-fail/v1"),
            sample_index=0,
            tools_kwargs={"image_zoom_in_tool": {"create_kwargs": {"image": Image.new("RGB", (64, 64))}}},
            tool_config={"config": {"num_workers": 1, "rate_limit": 1}},
            max_turns=5,
        )

    assert len(calls) == 1
    tool = FakeFailingImageZoomInTool.instances[0]
    assert len(tool.execute_calls) == 1
    assert tool.release_calls == [{"instance_id": "tool-instance-0", "kwargs": {}}]


@pytest.mark.asyncio
async def test_deepeyes_agent_runner_stops_when_response_has_no_tool_calls(monkeypatch):
    agent_runner = _import_agent_runner_module()
    monkeypatch.setattr(agent_runner, "ImageZoomInTool", FakeImageZoomInTool)
    calls = _patch_http_client(monkeypatch, agent_runner, [_chat_response(content="No tool needed.")])

    await agent_runner.deepeyes_agent_runner(
        raw_prompt=[{"role": "user", "content": "Answer directly."}],
        session=SessionHandle(session_id="session-1", base_url="http://gateway/sessions/session-1/v1"),
        sample_index=0,
        tools_kwargs={"image_zoom_in_tool": {"create_kwargs": {"image": Image.new("RGB", (64, 64))}}},
        tool_config={"config": {"num_workers": 1, "rate_limit": 1}},
        max_turns=5,
    )

    assert len(calls) == 1
    tool = FakeImageZoomInTool.instances[0]
    assert tool.execute_calls == []
    assert tool.release_calls == [{"instance_id": "tool-instance-0", "kwargs": {}}]


@pytest.mark.asyncio
async def test_deepeyes_agent_runner_stops_at_max_turns_without_executing_final_tool_call(monkeypatch):
    agent_runner = _import_agent_runner_module()
    monkeypatch.setattr(agent_runner, "ImageZoomInTool", FakeImageZoomInTool)
    calls = _patch_http_client(
        monkeypatch,
        agent_runner,
        [_chat_response(tool_calls=[_tool_call({"bbox_2d": [0, 0, 32, 32]})])],
    )

    await agent_runner.deepeyes_agent_runner(
        raw_prompt=[{"role": "user", "content": "Use tools forever."}],
        session=SessionHandle(session_id="session-2", base_url="http://gateway/sessions/session-2/v1"),
        sample_index=0,
        tools_kwargs={"image_zoom_in_tool": {"create_kwargs": {"image": Image.new("RGB", (64, 64))}}},
        tool_config={"config": {"num_workers": 1, "rate_limit": 1}},
        max_turns=1,
    )

    assert len(calls) == 1
    tool = FakeImageZoomInTool.instances[0]
    assert tool.execute_calls == []
    assert tool.release_calls == [{"instance_id": "tool-instance-0", "kwargs": {}}]


def test_tool_response_to_openai_tool_message_preserves_text_and_images():
    agent_runner = _import_agent_runner_module()
    image = Image.new("RGB", (8, 8), color="green")

    message = agent_runner._tool_response_to_openai_tool_message(
        tool_call_id="call_result",
        tool_response=ToolResponse(text="Zoomed in.", image=[image]),
    )

    assert message == {
        "role": "tool",
        "tool_call_id": "call_result",
        "content": [
            {"type": "text", "text": "Zoomed in."},
            {"type": "image", "image": image},
        ],
    }
