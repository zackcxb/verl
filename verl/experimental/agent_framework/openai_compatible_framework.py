from __future__ import annotations

from uuid import uuid4

from verl.protocol import DataProto

from .assembler import TrajectoryAssembler
from .framework import AgentFramework

# TODO: shall we put this in the framework module?
class OpenAICompatibleAgentFramework(AgentFramework):
    def __init__(
        self,
        session_runtime, # TODO: confusing variable name, unify this with the LLMServer class in agent_loop.py
        agent_runner,
        *,
        assembler: TrajectoryAssembler | None = None,
        pad_token_id: int = 0,
        reward_key: str = "reward",
        completion_timeout: float | None = 30.0,
        wait_for_completion_after_agent_run: bool = False,
    ):
        self.session_runtime = session_runtime
        self.agent_runner = agent_runner
        self.assembler = assembler or TrajectoryAssembler(pad_token_id=pad_token_id, reward_key=reward_key)
        self.completion_timeout = completion_timeout
        self.wait_for_completion_after_agent_run = wait_for_completion_after_agent_run

    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        raw_prompts = prompts.non_tensor_batch.get("raw_prompt")
        # TODO: shall we allow token ids as input?
        if raw_prompts is None:
            raise ValueError("OpenAICompatibleAgentFramework requires non_tensor_batch['raw_prompt']")

        trajectories = []
        for sample_index in range(len(prompts)):
            session_id = self._build_session_id(prompts=prompts, sample_index=sample_index)
            session = await self.session_runtime.create_session(session_id)
            try:
                await self.agent_runner(
                    raw_prompt=raw_prompts[sample_index],
                    session=session,
                    sample_index=sample_index,
                )
                if self.wait_for_completion_after_agent_run:
                    await self.session_runtime.wait_for_completion(session_id, timeout=self.completion_timeout)
                trajectories.extend(await self.session_runtime.finalize_session(session_id))
            except Exception:
                await self.session_runtime.abort_session(session_id)
                raise

        return self.assembler.assemble(trajectories)

    def _build_session_id(self, prompts: DataProto, sample_index: int) -> str:
        uid_batch = prompts.non_tensor_batch.get("uid")
        if uid_batch is not None:
            return str(uid_batch[sample_index])
        return f"session-{sample_index}-{uuid4().hex}"
