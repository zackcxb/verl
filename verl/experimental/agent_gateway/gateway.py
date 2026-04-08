from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any
from uuid import uuid4

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from verl.experimental.agent_framework.types import SessionHandle, Trajectory
from verl.experimental.agent_gateway.types import GatewaySessionState, SessionPhase, TrajectoryBuffer
from verl.utils.chat_template import apply_chat_template
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.utils import run_uvicorn


class MalformedRequestError(ValueError):
    pass


class UnsupportedRequestError(ValueError):
    pass


def _canonicalize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize_json_like(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_json_like(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise MalformedRequestError(f"Unsupported JSON-like value: {type(value).__name__}")


def _normalize_message_content(content: Any) -> Any:
    if isinstance(content, list):
        return [_canonicalize_json_like(part) for part in content]
    if isinstance(content, dict):
        return _canonicalize_json_like(content)
    if content is None or isinstance(content, str):
        return "" if content is None else content
    raise MalformedRequestError(f"Unsupported content type: {type(content).__name__}")


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        raise MalformedRequestError("tool_calls must be a list")

    normalized_tool_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise MalformedRequestError("tool_calls entries must be objects")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise MalformedRequestError("tool_call.function must be an object")
        arguments = function.get("arguments", "")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise MalformedRequestError("tool_call.function.arguments must decode to a JSON object") from exc
        if not isinstance(arguments, dict):
            raise MalformedRequestError("tool_call.function.arguments must decode to a JSON object")
        normalized_tool_calls.append(
            {
                "id": str(tool_call.get("id", "")),
                "type": str(tool_call.get("type", "")),
                "function": {
                    "name": str(function.get("name", "")),
                    "arguments": _canonicalize_json_like(arguments),
                },
            }
        )
    return normalized_tool_calls


def _normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise MalformedRequestError("messages entries must be objects")
    if "name" in message:
        raise UnsupportedRequestError("message.name is not supported in PR1")

    normalized_message: dict[str, Any] = {
        "role": str(message.get("role", "")),
        "content": _normalize_message_content(message.get("content", "")),
    }
    if "tool_calls" in message:
        normalized_message["tool_calls"] = _normalize_tool_calls(message["tool_calls"])
    if "tool_call_id" in message:
        normalized_message["tool_call_id"] = str(message["tool_call_id"])
    return normalized_message


def _normalize_tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise MalformedRequestError("tools must be a list")
    return [_canonicalize_json_like(tool) for tool in tools]


def _normalize_request_context(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise MalformedRequestError("messages must be non-empty")
    return {
        "messages": [_normalize_message(message) for message in messages],
        "tools": _normalize_tools(payload.get("tools")),
    }


def _is_message_prefix(prefix: list[dict[str, Any]], messages: list[dict[str, Any]]) -> bool:
    if len(prefix) > len(messages):
        return False
    return prefix == messages[: len(prefix)]


def _is_request_context_prefix(
    *,
    session: GatewaySessionState,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> bool:
    if session.request_tools != tools:
        return False
    #TODO: need to improve the prefix check logic, e.g.,how to handle tool lists and multimodal data
    return _is_message_prefix(session.message_history, messages)


def _copy_trajectory_buffer(buffer: TrajectoryBuffer | None) -> TrajectoryBuffer | None:
    if buffer is None:
        return None
    return TrajectoryBuffer(
        prompt_ids=list(buffer.prompt_ids),
        response_ids=list(buffer.response_ids),
        response_mask=list(buffer.response_mask),
        response_logprobs=list(buffer.response_logprobs),
    )


class _GatewayActor:
    def __init__(
        self,
        tokenizer,
        backend,
        host: str = "127.0.0.1",
        *,
        processor=None,
        tool_parser=None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self._tokenizer = tokenizer
        self._processor = processor
        self._tool_parser = tool_parser
        self._apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
        self._backend = backend
        self._host = host
        self._sessions: dict[str, GatewaySessionState] = {}
        self._app = FastAPI()
        self._server_port: int | None = None
        self._server_task: asyncio.Task | None = None
        self._server_base_url: str | None = None
        self._register_routes()

    def _register_routes(self) -> None:
        @self._app.post("/sessions/{session_id}/v1/chat/completions")
        async def _chat_completions(session_id: str, request: Request):
            payload = await request.json()
            return await self._handle_chat_completions(session_id=session_id, payload=payload)

        @self._app.post("/sessions/{session_id}/complete")
        async def _complete(session_id: str, request: Request):
            payload = await request.json()
            reward_info = payload.get("reward_info")
            await self.complete_session(session_id=session_id, reward_info=reward_info)
            return JSONResponse({"status": "ok"})

    def _require_started(self) -> None:
        if self._server_base_url is None:
            raise RuntimeError("GatewayActor.start() must be called before session creation")

    def _get_session(self, session_id: str) -> GatewaySessionState:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session

    def _set_phase(self, session: GatewaySessionState, phase: SessionPhase) -> None:
        session.phase = phase
        self._touch_session(session)

    def _touch_session(self, session: GatewaySessionState) -> None:
        session.updated_at = time.time()

    def _materialize_active_trajectory(self, session: GatewaySessionState) -> None:
        active = session.active_trajectory
        if active is None:
            return

        self._touch_session(session)
        session.trajectories.append(
            Trajectory(
                uid=session.handle.session_id,
                session_id=session.handle.session_id,
                trajectory_id=session.next_trajectory_id,
                prompt_ids=list(active.prompt_ids),
                response_ids=list(active.response_ids),
                response_mask=list(active.response_mask),
                response_logprobs=list(active.response_logprobs),
                reward_info={},
                num_turns=len(session.message_history),
            )
        )
        session.next_trajectory_id += 1
        session.active_trajectory = None

    def _collect_multimodal_data(self, messages: list[dict[str, Any]]) -> tuple[list[Any] | None, list[Any] | None]:
        images: list[Any] = []
        videos: list[Any] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "image":
                    value = part.get("image", part.get("images"))
                    if value is None:
                        continue
                    if isinstance(value, list):
                        images.extend(value)
                    else:
                        images.append(value)
                elif part_type == "video":
                    value = part.get("video", part.get("videos"))
                    if value is None:
                        continue
                    if isinstance(value, list):
                        videos.extend(value)
                    else:
                        videos.append(value)
        return (images or None, videos or None)

    def _render_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        add_generation_prompt: bool,
    ) -> str:
        template_target = self._processor or self._tokenizer
        return apply_chat_template(
            template_target,
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **self._apply_chat_template_kwargs,
        )

    def _tokenize_prompt(
        self,
        rendered_prompt: str,
        *,
        images: list[Any] | None,
        videos: list[Any] | None,
    ) -> list[int]:
        if self._processor is None:
            return list(normalize_token_ids(self._tokenizer.encode(rendered_prompt, add_special_tokens=False)))

        video_inputs = None
        video_metadata = None
        if videos is not None:
            if videos and isinstance(videos[0], tuple):
                video_inputs, video_metadata = zip(*videos, strict=False)
                video_inputs = list(video_inputs)
                video_metadata = list(video_metadata)
            else:
                video_inputs = videos

        model_inputs = self._processor(
            text=[rendered_prompt],
            images=images,
            videos=video_inputs,
            video_metadata=video_metadata,
            return_tensors="pt",
            do_sample_frames=False,
        )
        return list(model_inputs["input_ids"][0].tolist())

    def _encode_full_prompt(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[int], list[Any] | None, list[Any] | None]:
        images, videos = self._collect_multimodal_data(messages)
        rendered_prompt = self._render_chat_template(messages, tools=tools, add_generation_prompt=True)
        return self._tokenize_prompt(rendered_prompt, images=images, videos=videos), images, videos

    def _encode_prompt_delta(
        self,
        *,
        previous_messages: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[int], list[Any] | None, list[Any] | None]:
        prev_images, prev_videos = self._collect_multimodal_data(previous_messages)
        curr_images, curr_videos = self._collect_multimodal_data(messages)
        prev_rendered = self._render_chat_template(previous_messages, tools=tools, add_generation_prompt=False)
        curr_rendered = self._render_chat_template(messages, tools=tools, add_generation_prompt=True)

        if self._processor is None:
            delta_text = curr_rendered[len(prev_rendered) :]
            delta_ids = normalize_token_ids(self._tokenizer.encode(delta_text, add_special_tokens=False))
            return list(delta_ids), curr_images, curr_videos

        prev_ids = self._tokenize_prompt(prev_rendered, images=prev_images, videos=prev_videos)
        curr_ids = self._tokenize_prompt(curr_rendered, images=curr_images, videos=curr_videos)
        return curr_ids[len(prev_ids) :], curr_images, curr_videos

    def _format_tool_calls(self, parsed_tool_calls: Any) -> list[dict[str, Any]]:
        formatted_tool_calls: list[dict[str, Any]] = []
        for index, tool_call in enumerate(parsed_tool_calls):
            if isinstance(tool_call, dict) and "function" in tool_call:
                function = tool_call["function"]
                formatted_tool_calls.append(
                    {
                        "id": str(tool_call.get("id", f"call-{index}")),
                        "type": str(tool_call.get("type", "function")),
                        "function": {
                            "name": str(function.get("name", "")),
                            "arguments": str(function.get("arguments", "")),
                        },
                    }
                )
                continue

            formatted_tool_calls.append(
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {
                        "name": str(getattr(tool_call, "name", "")),
                        "arguments": str(getattr(tool_call, "arguments", "")),
                    },
                }
            )
        return formatted_tool_calls

    async def _decode_assistant_message(
        self,
        *,
        response_ids: list[int],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        response_text = self._tokenizer.decode(response_ids)
        if self._tool_parser is None:
            assistant_message = {"role": "assistant", "content": response_text}
            return assistant_message, dict(assistant_message), "stop"

        content, parsed_tool_calls = await self._tool_parser.extract_tool_calls(response_ids, tools=tools)
        formatted_tool_calls = self._format_tool_calls(parsed_tool_calls)
        if not formatted_tool_calls:
            assistant_message = {"role": "assistant", "content": content}
            return assistant_message, dict(assistant_message), "stop"

        assistant_message = {
            "role": "assistant",
            "content": content,
            "tool_calls": formatted_tool_calls,
        }
        history_message = {
            "role": "assistant",
            "content": content,
            "tool_calls": _normalize_tool_calls(formatted_tool_calls),
        }
        return assistant_message, history_message, "tool_calls"

    def _build_materialized_trajectory(
        self,
        *,
        session: GatewaySessionState,
        active: TrajectoryBuffer,
        trajectory_id: int,
    ) -> Trajectory:
        return Trajectory(
            uid=session.handle.session_id,
            session_id=session.handle.session_id,
            trajectory_id=trajectory_id,
            prompt_ids=list(active.prompt_ids),
            response_ids=list(active.response_ids),
            response_mask=list(active.response_mask),
            response_logprobs=list(active.response_logprobs),
            reward_info={},
            num_turns=len(session.message_history),
        )

    def _start_new_trajectory(
        self,
        session: GatewaySessionState,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> None:
        prompt_ids, _, _ = self._encode_full_prompt(messages=messages, tools=tools)
        session.active_trajectory = TrajectoryBuffer(prompt_ids=prompt_ids)

    async def _handle_chat_completions(self, session_id: str, payload: dict[str, Any]) -> JSONResponse:
        try:
            session = self._get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async with session.request_lock:
            if session.phase != SessionPhase.ACTIVE:
                raise HTTPException(status_code=409, detail=f"Session {session_id} is {session.phase.value.lower()}")

            self._touch_session(session)
            try:
                request_context = _normalize_request_context(payload)
            except MalformedRequestError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except UnsupportedRequestError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            messages = request_context["messages"]
            tools = request_context["tools"]
            next_trajectory_id = session.next_trajectory_id
            materialized_trajectory = None
            prompt_images = None
            prompt_videos = None

            if session.active_trajectory is None:
                prompt_ids, prompt_images, prompt_videos = self._encode_full_prompt(messages=messages, tools=tools)
                active_trajectory = TrajectoryBuffer(prompt_ids=prompt_ids)
            elif _is_request_context_prefix(session=session, messages=messages, tools=tools):
                active_trajectory = _copy_trajectory_buffer(session.active_trajectory)
                if len(messages) > len(session.message_history):
                    incremental_ids, prompt_images, prompt_videos = self._encode_prompt_delta(
                        previous_messages=session.message_history,
                        messages=messages,
                        tools=tools,
                    )
                    active_trajectory.response_ids.extend(incremental_ids)
                    active_trajectory.response_mask.extend([0] * len(incremental_ids))
                    active_trajectory.response_logprobs.extend([0.0] * len(incremental_ids))
            else:
                materialized_trajectory = self._build_materialized_trajectory(
                    session=session,
                    active=session.active_trajectory,
                    trajectory_id=next_trajectory_id,
                )
                next_trajectory_id += 1
                prompt_ids, prompt_images, prompt_videos = self._encode_full_prompt(messages=messages, tools=tools)
                active_trajectory = TrajectoryBuffer(prompt_ids=prompt_ids)

            # TODO: prompt_ids for generate requests are different from those in trajectories, shall we use different variable names?
            prompt_ids = active_trajectory.prompt_ids + active_trajectory.response_ids
            sampling_params = dict(payload)
            # TODO: check if there are other fields that need to be popped
            sampling_params.pop("messages", None)
            sampling_params.pop("model", None)
            sampling_params.pop("tools", None)
            output = await self._backend.generate(
                request_id=session_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=prompt_images,
                video_data=prompt_videos,
            )

            response_ids = list(output.token_ids)
            active_trajectory.response_ids.extend(response_ids)
            active_trajectory.response_mask.extend([1] * len(response_ids))
            if output.log_probs is not None:
                active_trajectory.response_logprobs.extend(list(output.log_probs))
            else:
                active_trajectory.response_logprobs.extend([0.0] * len(response_ids))

            assistant_message, history_message, finish_reason = await self._decode_assistant_message(
                response_ids=response_ids,
                tools=tools,
            )
            if materialized_trajectory is not None:
                session.trajectories.append(materialized_trajectory)
            session.next_trajectory_id = next_trajectory_id
            session.active_trajectory = active_trajectory
            session.message_history = messages + [history_message]
            session.request_tools = tools
            self._touch_session(session)

            return JSONResponse(
                {
                    "id": f"chatcmpl-{uuid4().hex}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": assistant_message,
                            "finish_reason": output.stop_reason or finish_reason,
                        }
                    ],
                    # TODO: Is this needed?
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(response_ids),
                        "total_tokens": len(prompt_ids) + len(response_ids),
                    },
                }
            )

    async def start(self) -> None:
        if self._server_task is not None:
            return
        self._server_port, self._server_task = await run_uvicorn(self._app, None, self._host)
        self._server_base_url = f"http://{self._host}:{self._server_port}"

    async def shutdown(self) -> None:
        if self._server_task is None:
            return
        self._server_task.cancel()
        try:
            await self._server_task
        except asyncio.CancelledError:
            pass
        self._server_task = None
        self._server_port = None
        self._server_base_url = None

    async def create_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> SessionHandle:
        self._require_started()
        if session_id in self._sessions:
            raise RuntimeError(f"Session {session_id} already exists")

        handle = SessionHandle(
            session_id=session_id,
            base_url=f"{self._server_base_url}/sessions/{session_id}/v1",
        )
        self._sessions[session_id] = GatewaySessionState(handle=handle, metadata=dict(metadata or {}))
        return handle

    async def complete_session(self, session_id: str, reward_info: dict[str, Any] | None = None) -> None:
        session = self._get_session(session_id)
        async with session.request_lock:
            # Accommodate retry attempts
            if session.phase not in {SessionPhase.COMPLETED, SessionPhase.ACTIVE}:
                raise RuntimeError(f"Session {session_id} is {session.phase.value.lower()}")

            if reward_info is not None:
                session.reward_info = dict(reward_info)

            self._set_phase(session, SessionPhase.COMPLETED)
            session.completed.set()

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        session = self._get_session(session_id)
        if session.phase in {SessionPhase.COMPLETED, SessionPhase.FINALIZED}:
            return
        if session.phase == SessionPhase.ABORTED:
            raise RuntimeError(f"Session {session_id} is aborted")

        await asyncio.wait_for(session.completed.wait(), timeout=timeout)
        
        if session.phase == SessionPhase.ABORTED:
            raise RuntimeError(f"Session {session_id} is aborted")
        return

    async def finalize_session(self, session_id: str) -> list[Trajectory]:
        session = self._get_session(session_id)
        async with session.request_lock:
            if session.phase == SessionPhase.ABORTED:
                raise RuntimeError(f"Session {session_id} is aborted")
            if session.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {session_id} is finalized")

            self._touch_session(session)
            self._materialize_active_trajectory(session)
            self._set_phase(session, SessionPhase.FINALIZED)
            session.completed.set()
            trajectories = [replace(trajectory, reward_info=dict(session.reward_info)) for trajectory in session.trajectories]
            self._sessions.pop(session_id, None)
            return trajectories

    async def abort_session(self, session_id: str) -> None:
        session = self._get_session(session_id)
        async with session.request_lock:
            if session.phase == SessionPhase.ABORTED:
                return
            if session.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {session_id} is finalized")

            self._set_phase(session, SessionPhase.ABORTED)
            session.completed.set()
            self._sessions.pop(session_id, None)

    async def get_session_state(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        return {
            "session_id": session.handle.session_id,
            "metadata": dict(session.metadata),
            "phase": session.phase.value,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "num_trajectories": len(session.trajectories),
            "has_active_trajectory": session.active_trajectory is not None,
        }


GatewayActor = ray.remote(_GatewayActor)
