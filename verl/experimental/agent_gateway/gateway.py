from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any
from uuid import uuid4

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from verl.experimental.agent_framework.types import SessionHandle, Trajectory
from verl.experimental.agent_gateway.types import GatewaySessionState, TrajectoryBuffer
from verl.workers.rollout.utils import run_uvicorn


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text_parts.append(part.get("text", "") if part.get("type") == "text" else str(part))
            else:
                text_parts.append(str(part))
        return "".join(text_parts)
    if content is None:
        return ""
    return str(content)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        normalized.append(
            {
                "role": str(message.get("role", "")),
                "content": _normalize_message_content(message.get("content", "")),
            }
        )
    return normalized


def _is_message_prefix(prefix: list[dict[str, str]], messages: list[dict[str, str]]) -> bool:
    if len(prefix) > len(messages):
        return False
    return prefix == messages[: len(prefix)]


class _GatewayActor:
    def __init__(self, tokenizer, backend, host: str = "127.0.0.1"):
        self._tokenizer = tokenizer
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

    def _start_new_trajectory(self, session: GatewaySessionState, messages: list[dict[str, str]]) -> None:
        prompt_ids = list(self._tokenizer.encode_messages(messages, add_generation_prompt=True))
        session.active_trajectory = TrajectoryBuffer(prompt_ids=prompt_ids)

    async def _handle_chat_completions(self, session_id: str, payload: dict[str, Any]) -> JSONResponse:
        try:
            session = self._get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async with session.request_lock:
            self._touch_session(session)
            messages = _normalize_messages(payload.get("messages", []))
            if not messages:
                raise HTTPException(status_code=400, detail="messages must be non-empty")

            if session.active_trajectory is None:
                self._start_new_trajectory(session, messages)
            elif _is_message_prefix(session.message_history, messages):
                incremental_messages = messages[len(session.message_history) :]
                if incremental_messages:
                    incremental_ids = list(
                        self._tokenizer.encode_messages(incremental_messages, add_generation_prompt=True)
                    )
                    session.active_trajectory.response_ids.extend(incremental_ids)
                    session.active_trajectory.response_mask.extend([0] * len(incremental_ids))
                    session.active_trajectory.response_logprobs.extend([0.0] * len(incremental_ids))
            else:
                self._materialize_active_trajectory(session)
                self._start_new_trajectory(session, messages)

            active = session.active_trajectory
            prompt_ids = active.prompt_ids + active.response_ids
            sampling_params = dict(payload)
            sampling_params.pop("messages", None)
            sampling_params.pop("model", None)
            output = await self._backend.generate(
                request_id=session_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

            response_ids = list(output.token_ids)
            active.response_ids.extend(response_ids)
            active.response_mask.extend([1] * len(response_ids))
            if output.log_probs is not None:
                active.response_logprobs.extend(list(output.log_probs))
            else:
                active.response_logprobs.extend([0.0] * len(response_ids))

            response_text = self._tokenizer.decode(response_ids)
            session.message_history = messages + [{"role": "assistant", "content": response_text}]
            self._touch_session(session)

            return JSONResponse(
                {
                    "id": f"chatcmpl-{uuid4().hex}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "finish_reason": output.stop_reason or "stop",
                        }
                    ],
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
        handle = SessionHandle(
            session_id=session_id,
            base_url=f"{self._server_base_url}/sessions/{session_id}/v1",
        )
        self._sessions[session_id] = GatewaySessionState(handle=handle, metadata=dict(metadata or {}))
        return handle

    async def complete_session(self, session_id: str, reward_info: dict[str, Any] | None = None) -> None:
        session = self._get_session(session_id)
        if reward_info is not None:
            session.reward_info = dict(reward_info)
        session.completed_flag = True
        self._touch_session(session)
        session.completed.set()

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        session = self._get_session(session_id)
        if timeout is None:
            await session.completed.wait()
        else:
            await asyncio.wait_for(session.completed.wait(), timeout=timeout)

    async def finalize_session(self, session_id: str) -> list[Trajectory]:
        session = self._get_session(session_id)
        self._touch_session(session)
        self._materialize_active_trajectory(session)
        self._sessions.pop(session_id, None)
        return [replace(trajectory, reward_info=dict(session.reward_info)) for trajectory in session.trajectories]

    async def abort_session(self, session_id: str) -> None:
        session = self._get_session(session_id)
        session.aborted_flag = True
        self._touch_session(session)
        self._sessions.pop(session_id, None)

    async def get_session_state(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        return {
            "session_id": session.handle.session_id,
            "metadata": dict(session.metadata),
            "completed_flag": session.completed_flag,
            "aborted_flag": session.aborted_flag,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "num_trajectories": len(session.trajectories),
            "has_active_trajectory": session.active_trajectory is not None,
        }


GatewayActor = ray.remote(_GatewayActor)
