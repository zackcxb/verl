# Agent Framework Gateway V3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove redundant gateway/session-runtime abstractions, align Gateway with VERL multi-turn tool/multimodal tokenization, simplify session lifecycle state, and make trajectory reward compute mandatory in `OpenAICompatibleAgentFramework`.

**Architecture:** `AsyncLLMServerManager` becomes the single serving + session runtime entrypoint. `GatewayActor` switches from ad hoc `encode_messages()/decode()` handling to chat-template-driven delta tokenization with tool parser and multimodal support. `OpenAICompatibleAgentFramework` gains an explicit trajectory-level `reward_compute` dependency and assembles only after rewards are materialized.

**Tech Stack:** Python, Ray actors, FastAPI, pytest, VERL DataProto, tokenizer/processor chat template pipeline

---

### Task 1: Lock the new session-runtime boundary in tests

**Files:**
- Modify: `tests/experimental/agent_gateway/test_session_runtime.py`
- Modify: `tests/experimental/agent_framework/test_openai_compatible_framework.py`

**Step 1: Write the failing test**

Update the tests so they import and construct `AsyncLLMServerManager` instead of `LLMServerManager`, and so framework construction requires `reward_compute`.

```python
manager = AsyncLLMServerManager(
    config=None,
    servers=[],
    load_balancer_handle=None,
    gateway_count=1,
    gateway_actor_kwargs={...},
)

with pytest.raises(TypeError):
    OpenAICompatibleAgentFramework(session_runtime=manager, agent_runner=mock_agent)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/experimental/agent_gateway/test_session_runtime.py tests/experimental/agent_framework/test_openai_compatible_framework.py -v`
Expected: FAIL because tests still reference `LLMServerManager` and `OpenAICompatibleAgentFramework` does not require `reward_compute`.

**Step 3: Write minimal implementation**

Do not change production code yet. Only land the test updates needed to express the new contract.

**Step 4: Run test to verify it still fails for the right reason**

Run: `pytest tests/experimental/agent_gateway/test_session_runtime.py tests/experimental/agent_framework/test_openai_compatible_framework.py -v`
Expected: FAIL in production code paths, not due to test syntax/import errors.

**Step 5: Commit**

```bash
git add tests/experimental/agent_gateway/test_session_runtime.py tests/experimental/agent_framework/test_openai_compatible_framework.py
git commit -m "test: lock async gateway runtime contract"
```

### Task 2: Remove `LLMServerManager` and `_GatewayServingBackend`

**Files:**
- Modify: `verl/experimental/agent_loop/agent_loop.py`
- Test: `tests/experimental/agent_gateway/test_session_runtime.py`

**Step 1: Write the failing test**

Add assertions that:
- `AsyncLLMServerManager(gateway_count=0)` raises on `create_session`
- `AsyncLLMServerManager(gateway_count=1)` owns gateway actor lifecycle directly
- no code path references `LLMServerManager` anymore

```python
with pytest.raises(RuntimeError, match="gateway_count=0"):
    manager.create_session("disabled")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/experimental/agent_gateway/test_session_runtime.py -v`
Expected: FAIL because the production implementation still routes through `LLMServerManager`.

**Step 3: Write minimal implementation**

In `verl/experimental/agent_loop/agent_loop.py`:
- inline gateway initialization and session-runtime methods into `AsyncLLMServerManager`
- delete `LLMServerManager`
- delete `_GatewayServingBackend`
- have gateway construction inject `self` as backend, or otherwise call `self.generate(...)` directly without a wrapper class

**Step 4: Run test to verify it passes**

Run: `pytest tests/experimental/agent_gateway/test_session_runtime.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add verl/experimental/agent_loop/agent_loop.py tests/experimental/agent_gateway/test_session_runtime.py
git commit -m "refactor: collapse gateway runtime into async server manager"
```

### Task 3: Add Gateway tool-calling decode coverage

**Files:**
- Modify: `tests/experimental/agent_gateway/support.py`
- Modify: `tests/experimental/agent_gateway/test_gateway_actor.py`

**Step 1: Write the failing test**

Extend test support with a fake tokenizer/tool parser pair that can emit a tool-call-bearing response, then add a gateway actor test asserting OpenAI-compatible tool call output.

```python
assert response.json()["choices"][0]["message"] == {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call-0",
            "type": "function",
            "function": {"name": "search", "arguments": "{\"query\":\"x\"}"},
        }
    ],
}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/experimental/agent_gateway/test_gateway_actor.py::test_gateway_actor_returns_openai_tool_calls -v`
Expected: FAIL because gateway currently decodes plain text only.

**Step 3: Write minimal implementation**

Only update test doubles/support fixtures now:
- fake tool parser
- backend response fixture for tool-call text
- helper assertions for OpenAI response shape

**Step 4: Run test to verify it still fails for the right reason**

Run: `pytest tests/experimental/agent_gateway/test_gateway_actor.py::test_gateway_actor_returns_openai_tool_calls -v`
Expected: FAIL in `gateway.py`, not due to missing support code.

**Step 5: Commit**

```bash
git add tests/experimental/agent_gateway/support.py tests/experimental/agent_gateway/test_gateway_actor.py
git commit -m "test: cover gateway tool call decoding"
```

### Task 4: Implement Gateway delta tokenization, tool parsing, and multimodal state

**Files:**
- Modify: `verl/experimental/agent_gateway/gateway.py`
- Modify: `verl/experimental/agent_gateway/types.py`
- Modify: `tests/experimental/agent_gateway/support.py`
- Test: `tests/experimental/agent_gateway/test_gateway_actor.py`

**Step 1: Write the failing test**

Add tests covering:
- prefix continuation uses delta tokenization instead of `encode_messages(incremental_messages)`
- tool calls are decoded through `ToolParser`
- multimodal message history can be carried across turns

```python
assert backend.calls[1]["prompt_ids"] == expected_prev_plus_delta_ids
assert response.json()["choices"][0]["message"]["tool_calls"]
assert trajectories[0].response_mask.count(0) == expected_non_model_tokens
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/experimental/agent_gateway/test_gateway_actor.py -v`
Expected: FAIL because gateway still uses simplified tokenizer encode/decode and does not track multimodal delta context.

**Step 3: Write minimal implementation**

In `verl/experimental/agent_gateway/gateway.py` and `verl/experimental/agent_gateway/types.py`:
- normalize message content for text + multimodal parts
- add helpers that render `prev/curr` with chat template and compute token deltas
- support both tokenizer and processor code paths
- decode responses through `ToolParser.extract_tool_calls(...)`
- persist enough session state to rebuild `prev` and `curr` multimodal contexts
- return OpenAI-compatible assistant tool call payloads

**Step 4: Run test to verify it passes**

Run: `pytest tests/experimental/agent_gateway/test_gateway_actor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add verl/experimental/agent_gateway/gateway.py verl/experimental/agent_gateway/types.py tests/experimental/agent_gateway/support.py tests/experimental/agent_gateway/test_gateway_actor.py
git commit -m "feat: align gateway tokenization with multiturn tool flow"
```

### Task 5: Simplify Gateway lifecycle state

**Files:**
- Modify: `verl/experimental/agent_gateway/gateway.py`
- Modify: `verl/experimental/agent_gateway/types.py`
- Modify: `tests/experimental/agent_gateway/test_gateway_actor.py`
- Modify: `tests/experimental/agent_gateway/test_gateway_manager.py`

**Step 1: Write the failing test**

Change tests so they no longer assert `completed_flag` / `aborted_flag`, and instead assert:
- `phase`
- session removal after finalize/abort
- simplified error behavior for repeated finalize/abort

```python
with pytest.raises(ray.exceptions.RayTaskError, match="Unknown session_id"):
    ray.get(actor.get_session_state.remote("session-state"))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/experimental/agent_gateway/test_gateway_actor.py tests/experimental/agent_gateway/test_gateway_manager.py -v`
Expected: FAIL because production code still exposes redundant flags and terminal tracking.

**Step 3: Write minimal implementation**

In `verl/experimental/agent_gateway/gateway.py` and `verl/experimental/agent_gateway/types.py`:
- delete `_terminal_session_phases`
- delete `completed_flag`
- delete `aborted_flag`
- reduce validation to `session.phase` and `_sessions` membership
- simplify `get_session_state()`

**Step 4: Run test to verify it passes**

Run: `pytest tests/experimental/agent_gateway/test_gateway_actor.py tests/experimental/agent_gateway/test_gateway_manager.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add verl/experimental/agent_gateway/gateway.py verl/experimental/agent_gateway/types.py tests/experimental/agent_gateway/test_gateway_actor.py tests/experimental/agent_gateway/test_gateway_manager.py
git commit -m "refactor: simplify gateway session lifecycle state"
```

### Task 6: Lock mandatory reward compute behavior in framework tests

**Files:**
- Modify: `tests/experimental/agent_framework/test_openai_compatible_framework.py`
- Modify: `tests/experimental/agent_framework/test_reward_helpers.py`
- Modify: `tests/experimental/agent_framework/test_assembler.py`

**Step 1: Write the failing test**

Add tests asserting:
- framework construction without `reward_compute` fails
- `reward_compute` can broadcast one session reward to multiple trajectories
- `reward_compute` can return per-trajectory rewards
- assembler only consumes final `Trajectory.reward_score`

```python
with pytest.raises(ValueError, match="reward_compute"):
    OpenAICompatibleAgentFramework(...)

assert output.batch["rm_scores"][0, -1] == 1.0
assert output.non_tensor_batch["label"].tolist() == ["shared", "shared"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/experimental/agent_framework/test_openai_compatible_framework.py tests/experimental/agent_framework/test_reward_helpers.py tests/experimental/agent_framework/test_assembler.py -v`
Expected: FAIL because framework still allows missing reward compute and helper semantics do not enforce the new contract.

**Step 3: Write minimal implementation**

Only land the tests and any tiny fixture/helper support needed to express the new behavior.

**Step 4: Run test to verify it still fails for the right reason**

Run: `pytest tests/experimental/agent_framework/test_openai_compatible_framework.py tests/experimental/agent_framework/test_reward_helpers.py tests/experimental/agent_framework/test_assembler.py -v`
Expected: FAIL in production code paths.

**Step 5: Commit**

```bash
git add tests/experimental/agent_framework/test_openai_compatible_framework.py tests/experimental/agent_framework/test_reward_helpers.py tests/experimental/agent_framework/test_assembler.py
git commit -m "test: require explicit framework reward compute"
```

### Task 7: Implement trajectory-level reward compute in framework

**Files:**
- Modify: `verl/experimental/agent_framework/framework.py`
- Modify: `verl/experimental/agent_framework/openai_compatible_framework.py`
- Modify: `verl/experimental/agent_framework/helpers.py`
- Modify: `verl/experimental/agent_framework/assembler.py`
- Test: `tests/experimental/agent_framework/test_openai_compatible_framework.py`
- Test: `tests/experimental/agent_framework/test_reward_helpers.py`
- Test: `tests/experimental/agent_framework/test_assembler.py`

**Step 1: Write the failing test**

If needed, add one focused test for the error path where `reward_compute` returns trajectories without usable `reward_score`.

```python
with pytest.raises(ValueError, match="reward_score"):
    await framework.generate_sequences(prompts)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/experimental/agent_framework/test_openai_compatible_framework.py tests/experimental/agent_framework/test_reward_helpers.py tests/experimental/agent_framework/test_assembler.py -v`
Expected: FAIL because production code still assembles before explicit reward materialization.

**Step 3: Write minimal implementation**

In the production files:
- define a thin reward compute protocol/type alias
- require `reward_compute` in `OpenAICompatibleAgentFramework.__init__`
- after `finalize_session`, invoke `reward_compute(...)`
- validate that all returned trajectories have usable `reward_score`
- keep `TrajectoryAssembler` format-only
- keep helper behavior aligned with explicit reward ownership

**Step 4: Run test to verify it passes**

Run: `pytest tests/experimental/agent_framework/test_openai_compatible_framework.py tests/experimental/agent_framework/test_reward_helpers.py tests/experimental/agent_framework/test_assembler.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add verl/experimental/agent_framework/framework.py verl/experimental/agent_framework/openai_compatible_framework.py verl/experimental/agent_framework/helpers.py verl/experimental/agent_framework/assembler.py tests/experimental/agent_framework/test_openai_compatible_framework.py tests/experimental/agent_framework/test_reward_helpers.py tests/experimental/agent_framework/test_assembler.py
git commit -m "feat: add explicit trajectory reward compute"
```

### Task 8: Run focused verification for the integrated change

**Files:**
- Modify: none
- Test: `tests/experimental/agent_gateway/test_gateway_actor.py`
- Test: `tests/experimental/agent_gateway/test_gateway_manager.py`
- Test: `tests/experimental/agent_gateway/test_session_runtime.py`
- Test: `tests/experimental/agent_framework/test_openai_compatible_framework.py`
- Test: `tests/experimental/agent_framework/test_reward_helpers.py`
- Test: `tests/experimental/agent_framework/test_assembler.py`

**Step 1: Run the integrated test suite**

Run:

```bash
pytest \
  tests/experimental/agent_gateway/test_gateway_actor.py \
  tests/experimental/agent_gateway/test_gateway_manager.py \
  tests/experimental/agent_gateway/test_session_runtime.py \
  tests/experimental/agent_framework/test_openai_compatible_framework.py \
  tests/experimental/agent_framework/test_reward_helpers.py \
  tests/experimental/agent_framework/test_assembler.py -v
```

Expected: PASS

**Step 2: Run one broader regression slice**

Run:

```bash
pytest tests/experimental/agent_gateway tests/experimental/agent_framework -v
```

Expected: PASS

**Step 3: Inspect final diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only intended gateway/framework/test/doc changes remain.

**Step 4: Commit**

```bash
git add verl/experimental/agent_loop/agent_loop.py \
  verl/experimental/agent_gateway/gateway.py \
  verl/experimental/agent_gateway/types.py \
  verl/experimental/agent_framework/framework.py \
  verl/experimental/agent_framework/openai_compatible_framework.py \
  verl/experimental/agent_framework/helpers.py \
  verl/experimental/agent_framework/assembler.py \
  tests/experimental/agent_gateway \
  tests/experimental/agent_framework
git commit -m "feat: align gateway and framework with multiturn contracts"
```
