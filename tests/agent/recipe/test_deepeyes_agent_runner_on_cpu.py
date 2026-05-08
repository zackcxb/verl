"""CPU tests for the DeepEyes gateway agent runner launcher."""

from __future__ import annotations

import importlib
import io
import json
import pathlib
import sys
from typing import Any

import pytest
from PIL import Image

from verl.agent.framework.types import SessionHandle

TESTS_AGENT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _import_agent_runner_module():
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("recipe", None)
        sys.path[:] = [path for path in sys.path if path != TESTS_AGENT_ROOT]
        return importlib.import_module("recipe.deepeyes_with_gateway.agent_runner")
    finally:
        sys.path[:] = original_sys_path


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeProcess:
    def __init__(self, returncode: int = 0, stdout: bytes = b"ok", stderr: bytes = b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.stdin_payload: bytes | None = None

    async def communicate(self, input: bytes | None = None):
        self.stdin_payload = input
        return self.stdout, self.stderr


@pytest.mark.asyncio
async def test_deepeyes_agent_runner_spawns_external_agent_with_json_stdin(monkeypatch):
    agent_runner = _import_agent_runner_module()
    created: list[dict[str, Any]] = []
    proc = FakeProcess(returncode=0)

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        created.append({"cmd": cmd, "kwargs": kwargs})
        return proc

    monkeypatch.setattr(agent_runner.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    await agent_runner.deepeyes_agent_runner(
        raw_prompt=[{"role": "user", "content": "Inspect this image."}],
        session=SessionHandle(session_id="session-0", base_url="http://gateway/sessions/session-0/v1"),
        sample_index=123,
        tools_kwargs={"image_zoom_in_tool": {"create_kwargs": {"image": "data:image/png;base64,abc"}}},
        tool_config_path="recipe/deepeyes_with_gateway/configs/image_zoom_in_tool_config.yaml",
        max_turns=7,
    )

    assert len(created) == 1
    cmd = created[0]["cmd"]
    assert cmd[:3] == (sys.executable, "-m", "recipe.deepeyes_with_gateway.external.deepeyes_agent")
    assert "--base-url" in cmd
    assert cmd[cmd.index("--base-url") + 1] == "http://gateway/sessions/session-0/v1"
    assert "--session-id" in cmd
    assert cmd[cmd.index("--session-id") + 1] == "session-0"
    assert "--tool-config-path" in cmd
    assert cmd[cmd.index("--tool-config-path") + 1] == "recipe/deepeyes_with_gateway/configs/image_zoom_in_tool_config.yaml"
    assert "--max-turns" in cmd
    assert cmd[cmd.index("--max-turns") + 1] == "7"

    tools_kwargs = json.loads(cmd[cmd.index("--tools-kwargs-json") + 1])
    assert tools_kwargs == {"image_zoom_in_tool": {"create_kwargs": {"image": "data:image/png;base64,abc"}}}
    assert created[0]["kwargs"]["stdout"] is agent_runner.asyncio.subprocess.PIPE
    assert created[0]["kwargs"]["stderr"] is agent_runner.asyncio.subprocess.PIPE
    assert created[0]["kwargs"]["stdin"] is agent_runner.asyncio.subprocess.PIPE
    assert json.loads(proc.stdin_payload.decode("utf-8")) == {
        "raw_prompt": [{"role": "user", "content": "Inspect this image."}]
    }


@pytest.mark.asyncio
async def test_deepeyes_agent_runner_serializes_pil_images_for_subprocess_stdin(monkeypatch):
    agent_runner = _import_agent_runner_module()
    created: list[dict[str, Any]] = []
    proc = FakeProcess(returncode=0)

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        created.append({"cmd": cmd, "kwargs": kwargs})
        return proc

    monkeypatch.setattr(agent_runner.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    await agent_runner.deepeyes_agent_runner(
        raw_prompt=[
            {
                "role": "user",
                "content": [{"type": "image", "image": {"bytes": _png_bytes()}}],
            }
        ],
        session=SessionHandle(session_id="session-pil", base_url="http://gateway/sessions/session-pil/v1"),
        sample_index=0,
        tools_kwargs={"image_zoom_in_tool": {"create_kwargs": {"image": Image.new("RGB", (4, 4), color="blue")}}},
        tool_config_path="tools.yaml",
        max_turns=1,
    )

    stdin_payload = json.loads(proc.stdin_payload.decode("utf-8"))
    prompt_image = stdin_payload["raw_prompt"][0]["content"][0]["image"]
    assert prompt_image.startswith("data:image/png;base64,")

    cmd = created[0]["cmd"]
    tools_kwargs = json.loads(cmd[cmd.index("--tools-kwargs-json") + 1])
    tool_image = tools_kwargs["image_zoom_in_tool"]["create_kwargs"]["image"]
    assert tool_image.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_deepeyes_agent_runner_raises_when_external_agent_fails(monkeypatch):
    agent_runner = _import_agent_runner_module()
    proc = FakeProcess(returncode=3, stdout=b"partial stdout", stderr=b"boom")

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        return proc

    monkeypatch.setattr(agent_runner.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match="DeepEyes external agent failed with exit code 3"):
        await agent_runner.deepeyes_agent_runner(
            raw_prompt=[{"role": "user", "content": "Inspect."}],
            session=SessionHandle(session_id="session-fail", base_url="http://gateway/sessions/session-fail/v1"),
            sample_index=0,
            tools_kwargs={},
            tool_config_path="tools.yaml",
        )
