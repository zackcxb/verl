# Agent tests

This directory contains the CPU-only unit tests for the new `verl.agent`
packages introduced for the agent framework / gateway path.

## Naming and CI routing

All executable test modules in this directory use the `*_on_cpu.py` suffix so
they are picked up by VERL's existing `cpu_unit_tests.yml` workflow instead of
the default GPU unit-test workflow.

## Coverage inventory

### Framework

- `framework/test_assembler_on_cpu.py`
  - Verifies `TrajectoryAssembler` emits the expected training batch contract.
  - Checks tensor padding, masks, logprobs, routed experts, and non-tensor
    metadata packing in the final `TensorDict`.
- `framework/test_openai_compatible_framework_on_cpu.py`
  - Verifies `OpenAICompatibleAgentFramework` runs against a fake in-memory
    session runtime without Ray, HTTP serving, or LLM backends.
  - Covers session creation, finalize/abort behavior, reward assignment,
    optional wait-for-completion, non-tensor field broadcast, and missing
    rollout logprob handling.

### Gateway

- `gateway/test_gateway_actor_on_cpu.py`
  - CPU-only FastAPI/Ray actor contract tests for `GatewayActor`.
  - Uses fake tokenizer + mocked backends instead of real `LLMServer` /
    rollout-serving paths.
  - Covers request normalization, prefix matching, tool-schema drift,
    continuation masks, per-session concurrency serialization, completion
    semantics, invalid request rejection, backend failure rollback, and tool
    parser response formatting.
- `gateway/test_gateway_manager_on_cpu.py`
  - Verifies sticky session routing and least-active gateway selection.
  - Uses lightweight fake gateway actors instead of real serving stacks.
- `gateway/test_session_runtime_on_cpu.py`
  - Verifies `GatewayServingRuntime` owns gateway lifecycle and session runtime
    behavior independently from `agent_loop`.
  - Covers both runtime-owned fake backend injection and mocked load-balancer /
    rollout-server integration.

## Mocking boundaries

- No test in this directory depends on a real `LLMServer`, model weights, or a
  production serving runtime.
- `tests/agent/support.py` provides the fakes and lightweight Ray actors used by
  the gateway/runtime tests.
- The only retained dependency on the old experimental tree is
  `verl.experimental.agent_loop.tool_parser`, which is intentionally reused by
  `GatewayActor` until the community-wide extraction lands.
