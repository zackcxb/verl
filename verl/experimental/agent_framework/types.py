from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class SessionHandle:
    session_id: str
    base_url: str | None = None


@dataclass
class Trajectory:
    uid: str
    session_id: str
    trajectory_id: int
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None = None
    reward_info: dict[str, Any] = field(default_factory=dict)
    reward_score: float | None = None
    num_turns: int = 0
    routed_experts: torch.Tensor | np.ndarray | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)
