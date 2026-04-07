import asyncio
import json
import re

import ray
import torch

from verl.experimental.agent_framework.types import SessionHandle, Trajectory
from verl.workers.rollout.replica import TokenOutput


class FakeTokenizer:
    pad_token = "<pad>"

    def encode_messages(self, messages, add_generation_prompt=True):
        return self.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False, **kwargs):
        parts = []
        for message in messages:
            parts.append(f"{message['role']}:{self._normalize_content(message.get('content', ''))}\n")
        if add_generation_prompt:
            parts.append("assistant:")
        text = "".join(parts)
        if tokenize:
            return [ord(char) for char in text]
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)

    def _normalize_content(self, content):
        if isinstance(content, list):
            return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if content is None:
            return ""
        return str(content)


class FakeDeltaTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False, **kwargs):
        parts = ["<bos>"]
        for message in messages:
            parts.append(f"[{message['role']}]{self._normalize_content(message.get('content', ''))}")
        if add_generation_prompt:
            parts.append(f"<gen:{len(messages)}>")
        text = "".join(parts)
        if tokenize:
            return [ord(char) for char in text]
        return text


class FakeProcessor(FakeDeltaTokenizer):
    def __call__(
        self,
        *,
        text,
        images=None,
        videos=None,
        video_metadata=None,
        return_tensors="pt",
        do_sample_frames=False,
    ):
        rendered = text[0]
        suffix = f"|images={len(images or [])}|videos={len(videos or [])}"
        input_ids = [ord(char) for char in rendered + suffix]
        return {"input_ids": torch.tensor([input_ids], dtype=torch.long)}


class QueuedBackend:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
                "image_data": image_data,
                "video_data": video_data,
            }
        )
        text = self._responses.pop(0)
        token_ids = [ord(char) for char in text]
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason="completed",
        )


class FakeToolParser:
    _tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    async def extract_tool_calls(self, responses_ids, tools=None):
        text = "".join(chr(token_id) for token_id in responses_ids)
        matches = self._tool_call_regex.findall(text)
        if not matches:
            return text, []

        tool_calls = []
        for index, payload in enumerate(matches):
            parsed = json.loads(payload)
            tool_calls.append(
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": json.dumps(parsed["arguments"]),
                    },
                }
            )
        content = self._tool_call_regex.sub("", text)
        return content, tool_calls


@ray.remote
class RecordingLoadBalancer:
    def __init__(self, server_id: str = "server-0"):
        self.server_id = server_id
        self.acquire_calls = []
        self.release_calls = []

    def acquire_server(self, request_id: str) -> str:
        self.acquire_calls.append(request_id)
        return self.server_id

    def release_server(self, server_id: str) -> None:
        self.release_calls.append(server_id)

    def stats(self):
        return {
            "acquire_calls": list(self.acquire_calls),
            "release_calls": list(self.release_calls),
        }


@ray.remote
class RecordingRolloutServer:
    def __init__(self, response_text: str = "OWNER"):
        self.response_text = response_text
        self.calls = []

    async def generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        image_data=None,
        video_data=None,
    ):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
            }
        )
        token_ids = [ord(char) for char in self.response_text]
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason="completed",
        )

    def get_calls(self):
        return list(self.calls)


class RejectConcurrentSessionBackend:
    def __init__(self, responses, delay: float = 0.05):
        self._responses = list(responses)
        self._delay = delay
        self._active_request_ids: set[str] = set()
        self.call_windows = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        if request_id in self._active_request_ids:
            raise RuntimeError(f"concurrent request for session {request_id}")

        self._active_request_ids.add(request_id)
        started_at = asyncio.get_running_loop().time()
        try:
            await asyncio.sleep(self._delay)
            text = self._responses.pop(0)
            token_ids = [ord(char) for char in text]
            return TokenOutput(
                token_ids=token_ids,
                log_probs=[-0.1] * len(token_ids),
                stop_reason="completed",
            )
        finally:
            finished_at = asyncio.get_running_loop().time()
            self.call_windows.append((request_id, started_at, finished_at))
            self._active_request_ids.remove(request_id)


class AssertingQueuedBackend:
    def __init__(self, responses, expected_prompt_ids_per_call, expected_image_data_per_call=None, expected_video_data_per_call=None):
        self._responses = list(responses)
        self._expected_prompt_ids_per_call = [list(prompt_ids) for prompt_ids in expected_prompt_ids_per_call]
        self._expected_image_data_per_call = expected_image_data_per_call
        self._expected_video_data_per_call = expected_video_data_per_call
        self.calls = 0

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        expected_prompt_ids = self._expected_prompt_ids_per_call[self.calls]
        if list(prompt_ids) != expected_prompt_ids:
            raise AssertionError(f"unexpected prompt_ids on call {self.calls}: {prompt_ids} != {expected_prompt_ids}")
        if self._expected_image_data_per_call is not None:
            expected_image_data = self._expected_image_data_per_call[self.calls]
            if image_data != expected_image_data:
                raise AssertionError(f"unexpected image_data on call {self.calls}: {image_data} != {expected_image_data}")
        if self._expected_video_data_per_call is not None:
            expected_video_data = self._expected_video_data_per_call[self.calls]
            if video_data != expected_video_data:
                raise AssertionError(f"unexpected video_data on call {self.calls}: {video_data} != {expected_video_data}")
        self.calls += 1
        text = self._responses.pop(0)
        token_ids = [ord(char) for char in text]
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason="completed",
        )


@ray.remote
class FlakyGatewayActor:
    def __init__(self, fail_finalize_once: bool = False, fail_abort_once: bool = False):
        self.fail_finalize_once = fail_finalize_once
        self.fail_abort_once = fail_abort_once
        self.finalize_attempts = 0
        self.abort_attempts = 0
        self.sessions = {}

    async def start(self):
        return None

    async def shutdown(self):
        return None

    async def create_session(self, session_id: str):
        handle = SessionHandle(session_id=session_id, base_url=f"http://fake/{session_id}/v1")
        self.sessions[session_id] = handle
        return handle

    async def finalize_session(self, session_id: str):
        self.finalize_attempts += 1
        if self.fail_finalize_once and self.finalize_attempts == 1:
            raise RuntimeError("transient finalize failure")
        self.sessions.pop(session_id, None)
        return [
            Trajectory(
                uid=session_id,
                session_id=session_id,
                trajectory_id=0,
                prompt_ids=[1],
                response_ids=[2],
                response_mask=[1],
            )
        ]

    async def abort_session(self, session_id: str):
        self.abort_attempts += 1
        if self.fail_abort_once and self.abort_attempts == 1:
            raise RuntimeError("transient abort failure")
        self.sessions.pop(session_id, None)
        return None

    async def wait_for_completion(self, session_id: str, timeout: float | None = None):
        return None


@ray.remote
class TrackingGatewayActor:
    def __init__(self, name: str):
        self.name = name
        self.sessions = {}
        self.created = []
        self.finalized = []
        self.aborted = []

    async def start(self):
        return None

    async def shutdown(self):
        return None

    async def create_session(self, session_id: str, metadata: dict | None = None):
        handle = SessionHandle(session_id=session_id, base_url=f"http://{self.name}/{session_id}/v1")
        self.sessions[session_id] = {"metadata": metadata or {}}
        self.created.append(session_id)
        return handle

    async def finalize_session(self, session_id: str):
        self.finalized.append(session_id)
        self.sessions.pop(session_id, None)
        return [
            Trajectory(
                uid=session_id,
                session_id=session_id,
                trajectory_id=0,
                prompt_ids=[1],
                response_ids=[2],
                response_mask=[1],
            )
        ]

    async def abort_session(self, session_id: str):
        self.aborted.append(session_id)
        self.sessions.pop(session_id, None)
        return None

    async def wait_for_completion(self, session_id: str, timeout: float | None = None):
        return None

    async def stats(self):
        return {
            "name": self.name,
            "created": list(self.created),
            "finalized": list(self.finalized),
            "aborted": list(self.aborted),
        }
