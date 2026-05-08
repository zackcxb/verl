"""Black-box smoke test for the DeepEyes external agent subprocess."""

from __future__ import annotations

import base64
import importlib
import io
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from PIL import Image

from verl.agent.framework.types import SessionHandle

TESTS_AGENT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _import_deepeyes_agent_runner():
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("recipe", None)
        sys.path[:] = [path for path in sys.path if path != TESTS_AGENT_ROOT]
        module = importlib.import_module("recipe.deepeyes_with_gateway.agent_runner")
        return module.deepeyes_agent_runner
    finally:
        sys.path[:] = original_sys_path


def _image_data_uri() -> str:
    image = Image.new("RGB", (64, 64), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class ChatHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self):  # noqa: N802
        assert self.path == "/chat/completions"
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.__class__.requests.append(payload)

        if len(self.__class__.requests) == 1:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_zoom",
                        "type": "function",
                        "function": {
                            "name": "image_zoom_in_tool",
                            "arguments": json.dumps({"bbox_2d": [0, 0, 32, 32], "label": "corner"}),
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "final answer"}

        body = json.dumps({"choices": [{"message": message}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return None


@pytest.mark.asyncio
async def test_deepeyes_external_agent_smoke_runs_tool_loop_through_launcher():
    deepeyes_agent_runner = _import_deepeyes_agent_runner()
    ChatHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        await deepeyes_agent_runner(
            raw_prompt=[{"role": "user", "content": "Inspect this image."}],
            session=SessionHandle(session_id="smoke-session", base_url=base_url),
            sample_index=0,
            tools_kwargs={"image_zoom_in_tool": {"create_kwargs": {"image": _image_data_uri()}}},
            tool_config_path="recipe/deepeyes_with_gateway/configs/image_zoom_in_tool_config.yaml",
            max_turns=3,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert len(ChatHandler.requests) == 2
    assert ChatHandler.requests[0]["messages"] == [{"role": "user", "content": "Inspect this image."}]
    continuation = ChatHandler.requests[1]["messages"]
    assert continuation[1]["tool_calls"][0]["id"] == "call_zoom"
    assert continuation[2]["role"] == "tool"
    assert continuation[2]["tool_call_id"] == "call_zoom"
    assert continuation[2]["content"][0]["text"].startswith("Zoomed in on the image")
    assert continuation[2]["content"][1]["image"].startswith("data:image/png;base64,")
