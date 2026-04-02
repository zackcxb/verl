from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from verl.experimental.agent_framework.types import SessionHandle, Trajectory


@dataclass
class TrajectoryBuffer:
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)


@dataclass
class GatewaySessionState:
    handle: SessionHandle
    metadata: dict[str, Any] = field(default_factory=dict)
    message_history: list[dict[str, Any]] = field(default_factory=list)
    active_trajectory: TrajectoryBuffer | None = None
    trajectories: list[Trajectory] = field(default_factory=list)
    reward_info: dict[str, Any] = field(default_factory=dict)
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    completed_flag: bool = False
    aborted_flag: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    request_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_trajectory_id: int = 0
