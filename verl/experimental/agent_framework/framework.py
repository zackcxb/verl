from abc import ABC, abstractmethod

from verl.protocol import DataProto


class AgentFramework(ABC):
    @abstractmethod
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Process a trainer batch and return a training-ready DataProto."""
        ...
