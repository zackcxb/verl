from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from verl.protocol import DataProto

if TYPE_CHECKING:
    from .types import Trajectory


TrajectoryRewardCompute = Callable[..., Sequence["Trajectory"] | Awaitable[Sequence["Trajectory"]]]


class AgentFramework(ABC):
    @abstractmethod
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Process a trainer batch and return a training-ready DataProto."""
        ...
