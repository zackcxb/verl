# AgentFramework & AgentGateway Implementation Plan v2

> This plan supersedes [2026-03-30-agent-framework-gateway-implementation-plan.md](/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-03-30-agent-framework-gateway-implementation-plan.md) as the current execution baseline.

**Goal:** Build the first usable AgentFramework + AgentGateway path for VERL around OpenAI-compatible chat completions, while deferring `AgentLoopManager` migration and token-request ingress to follow-up work.

**Architecture:** `AgentFramework` is a thin abstract interface that only exposes `generate_sequences`. Gateway ownership moves under `LLMServerManager`, with `GatewayManager` as an internal session-routing/control-plane component. The first feature path targets session-based `/v1/chat/completions` traffic and an OpenAI-compatible / remote-style reference integration. `AgentLoopManager` migration is explicitly deferred.

**Non-goals for this plan:** token ingress, `AgentLoopManager` bridge, React bridge equivalence, Retool/SWE validation, fully async support, full legacy compatibility.

---

## Series Layout

- Upstream-owned prerequisite, not part of this series:
  - rollout server / runtime extraction
- This series PR1:
  - thin `AgentFramework`
  - Gateway ownership under `LLMServerManager`
  - `/v1/chat/completions`
  - session lifecycle APIs
  - assembler/helpers
  - one minimal OpenAI-compatible reference path
  - component tests + lightweight E2E
- Follow-up PRs:
  - token ingress
  - `AgentLoopManager` migration
  - React/Retool/SWE validation
  - deferred compatibility capabilities

### Task 1: Add Thin AgentFramework Types and Helpers

**Files:**
- Create: `verl/experimental/agent_framework/__init__.py`
- Create: `verl/experimental/agent_framework/framework.py`
- Create: `verl/experimental/agent_framework/types.py`
- Create: `verl/experimental/agent_framework/helpers.py`
- Create: `verl/experimental/agent_framework/assembler.py`
- Test: `tests/experimental/agent_framework/test_assembler.py`
- Test: `tests/experimental/agent_framework/test_reward_helpers.py`

**Step 1: Write failing tests**

- Cover:
  - `AgentFramework` exposes only thin `generate_sequences`
  - reward normalization helper
  - trajectory validation helper
  - assembler output contract:
    - `prompts`
    - `responses`
    - `response_mask`
    - `input_ids`
    - `attention_mask`
    - `position_ids`
    - `rm_scores`
    - `__num_turns__`
    - optional `rollout_log_probs`
    - optional `routed_experts`

**Step 2: Run tests**

```bash
pytest tests/experimental/agent_framework/test_assembler.py tests/experimental/agent_framework/test_reward_helpers.py -v
```

**Step 3: Implement**

- Add a thin `AgentFramework`
- Add `SessionHandle`, `Trajectory`, and related lightweight types
- Add helper utilities for reward normalization and validation
- Add `TrajectoryAssembler` aligned with current training-visible batch contract

**Step 4: Re-run tests**

```bash
pytest tests/experimental/agent_framework/test_assembler.py tests/experimental/agent_framework/test_reward_helpers.py -v
```

### Task 2: Add GatewayActor and Internal GatewayManager

**Files:**
- Create: `verl/experimental/agent_gateway/__init__.py`
- Create: `verl/experimental/agent_gateway/types.py`
- Create: `verl/experimental/agent_gateway/gateway.py`
- Create: `verl/experimental/agent_gateway/manager.py`
- Test: `tests/experimental/agent_gateway/test_gateway_actor.py`
- Test: `tests/experimental/agent_gateway/test_gateway_manager.py`

**Step 1: Write failing tests**

- Cover:
  - `create_session`
  - `finalize_session`
  - `abort_session`
  - `wait_for_completion`
  - sticky `session_id -> gateway actor`
  - independent `gateway_count`
  - `/complete` reward info handling
  - message-level prefix mismatch behavior

**Step 2: Run tests**

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py tests/experimental/agent_gateway/test_gateway_manager.py -v
```

**Step 3: Implement**

- Implement `GatewayActor` around session state and `/v1/chat/completions` semantics
- Implement `GatewayManager` as a session-routing component only
- Do not add token ingress in this task

**Step 4: Re-run tests**

```bash
pytest tests/experimental/agent_gateway/test_gateway_actor.py tests/experimental/agent_gateway/test_gateway_manager.py -v
```

### Task 3: Integrate Gateway Ownership into LLMServerManager

**Files:**
- Modify: `verl/experimental/agent_loop/agent_loop.py`
- Or create a focused module under `verl/experimental/agent_gateway/` or a new serving helper module if cleaner
- Test: `tests/experimental/agent_gateway/test_session_runtime.py`

**Step 1: Write failing tests**

- Cover:
  - `LLMServerManager`-owned gateway lifecycle
  - framework-facing narrow session capability
  - `gateway_count = 0` behavior
  - `GatewayActor` creation by owner, not by `GatewayManager`

**Step 2: Run tests**

```bash
pytest tests/experimental/agent_gateway/test_session_runtime.py -v
```

**Step 3: Implement**

- Keep ownership in `LLMServerManager`
- Expose a narrow session capability:
  - `create_session`
  - `finalize_session`
  - `abort_session`
  - `wait_for_completion`
- Keep `GatewayManager` internal
- Avoid turning `GatewayManager` into a second top-level dependency

**Step 4: Re-run tests**

```bash
pytest tests/experimental/agent_gateway/test_session_runtime.py -v
```

### Task 4: Add One Minimal OpenAI-Compatible Reference Framework Path

**Files:**
- Create: `verl/experimental/agent_framework/openai_compatible_framework.py`
- Create or modify a small test fixture/mock agent under `tests/experimental/agent_framework/`
- Test: `tests/experimental/agent_framework/test_openai_compatible_framework.py`

**Step 1: Write failing tests**

- Cover:
  - framework calls `create_session`
  - framework passes `base_url` to the agent
  - agent uses `/v1/chat/completions`
  - framework waits or finalizes correctly
  - trajectories are assembled into `DataProto`

**Step 2: Run tests**

```bash
pytest tests/experimental/agent_framework/test_openai_compatible_framework.py -v
```

**Step 3: Implement**

- Add one minimal reference implementation for an OpenAI-compatible / remote-style execution model
- Prefer a small local mock or subprocess-style fixture over trying to migrate `AgentLoopManager`

**Step 4: Re-run tests**

```bash
pytest tests/experimental/agent_framework/test_openai_compatible_framework.py -v
```

### Task 5: Add Lightweight End-to-End Acceptance Run

**Files:**
- Create: `tests/special_e2e/run_agent_gateway_chat_completion_smoke.sh`
- Modify: minimal config/example files if needed
- Document: current plan and evidence location

**Step 1: Define acceptance scenario**

- Use a minimal OpenAI-compatible agent path
- Keep:
  - small model
  - small dataset
  - small step count
- Validate:
  - session lifecycle works
  - gateway captures trajectories
  - output batch is trainable / inspectable

**Step 2: Run acceptance locally**

```bash
bash tests/special_e2e/run_agent_gateway_chat_completion_smoke.sh
```

**Step 3: Fix stability issues**

- Session cleanup
- Reward info propagation
- assembler shape bugs
- gateway lifecycle bugs

**Step 4: Re-run acceptance**

```bash
bash tests/special_e2e/run_agent_gateway_chat_completion_smoke.sh
```

### Task 6: Record Deferred Follow-up Work

**Deferred by design**

- token ingress
- `AgentLoopManager` migration
- React bridge
- Retool validation
- SWE-Agent validation
- teacher logprobs / distillation integration
- broader legacy `extra_fields` compatibility
- fully async support

**PR description checklist**

- State explicitly that PR1 targets chat-completions only
- State that token ingress is reserved for follow-up
- State that `AgentLoopManager` migration is intentionally deferred
- Include exact test commands and E2E evidence
