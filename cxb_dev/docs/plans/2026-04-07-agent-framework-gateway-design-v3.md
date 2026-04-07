# Agent Framework Gateway Design V3

## Goal

在 `feature/agent_framework` 当前工作树状态上，收敛并修正 Agent Framework / Gateway 的实现边界，使其：

1. 删除冗余的 `LLMServerManager` / `_GatewayServingBackend` 抽象。
2. 让 Gateway 的编码、解码行为对齐 VERL 现有 multi-turn chat template、tool calling 与多模态处理路径。
3. 删除 `terminal_phase`、`completed_flag`、`aborted_flag`，简化 session 生命周期校验。
4. 为 `OpenAICompatibleAgentFramework` 补全 reward compute 抽象，使 reward 基于 trajectory 计算，`reward_info` 仅作为可选辅助输入。

## Non-Goals

1. 不重写 `GatewayManager` 的 session-to-gateway sticky routing 逻辑。
2. 不在 Gateway 内实现新的 tool parser 协议；继续复用现有 `ToolParser`。
3. 不引入新的 trainer 级 reward pipeline；只在 agent framework 边界补齐 reward compute。
4. 不做超出本轮需求的通用化抽象。

## Current Problems

### 1. Session runtime abstraction is redundant

当前 `LLMServerManager` 只承载 gateway session runtime 能力，而 `AsyncLLMServerManager` 已经承载真实的 serving manager 能力。`_GatewayServingBackend` 只是再次包裹一个 `AsyncLLMServerManager.generate()` 调用，几乎没有独立语义，导致：

1. 抽象命名和职责不一致。
2. gateway backend 初始化链路绕了一层无意义 wrapper。
3. 测试和调用方需要同时理解两个 manager。

### 2. Gateway tokenization path diverges from VERL

当前 `gateway.py` 使用 `encode_messages()/decode()` 进行 prompt 和 response 处理。这与 VERL 现有 multi-turn rollout 主路径不一致：

1. 未经过 `apply_chat_template`。
2. 未经过 `ToolParser` 解析 tool call。
3. 未经过 `process_vision_info` / `processor` 处理多模态数据。
4. prefix continuation 时使用“直接编码新增 messages”的简化逻辑，无法对齐 `docs/sglang_multiturn/multiturn.rst` 中定义的 delta tokenization。

### 3. Session lifecycle state is over-modeled

当前实现同时维护：

1. `session.phase`
2. `_terminal_session_phases`
3. `completed_flag`
4. `aborted_flag`

这些状态大多可由 `session.phase` 或 `_sessions` 是否持有 session 推导，额外状态只会带来重复校验和冗余异常路径。

### 4. Framework reward compute is missing

当前 `OpenAICompatibleAgentFramework` 在 finalize session 后直接 assemble trajectories，没有 reward compute 层。结果是：

1. reward 完全被误认为要由 agent 主动产出。
2. `reward_info` 被误用为 reward 本体。
3. 无法支持“多个 trajectories 共享同一个 reward”或“逐 trajectory 单独计算”的合理场景。

## Design Decisions

### A. Collapse session runtime into `AsyncLLMServerManager`

删除 `LLMServerManager` 和 `_GatewayServingBackend`，将 gateway lifecycle 与 session runtime 接口直接并入 `AsyncLLMServerManager`。

`AsyncLLMServerManager` 负责：

1. rollout server sticky routing 与 load balancing。
2. gateway actor 创建、启动、关闭。
3. 对外暴露 session runtime 接口：
   - `create_session`
   - `wait_for_completion`
   - `finalize_session`
   - `abort_session`
   - `shutdown`
4. 对外暴露 serving 接口：
   - `generate`

保留 `GatewayManager`，因为它解决的是“多个 gateway actor 的 session sticky routing”，不是冗余抽象。

### B. Gateway aligns to VERL multi-turn encoding/decoding

Gateway 不再依赖简化的 `encode_messages()/decode()` 协议，而是改为对齐 `AgentLoopBase` 中的现有路径：

1. 输入消息规范化后，使用 `apply_chat_template` 渲染 chat template。
2. 若存在 processor，则调用 processor 路径处理 images/videos。
3. 若为 prefix continuation，按 `multiturn.rst` 规定执行 delta tokenization：
   - 文本 tokenizer 路径：比较 `prev/curr` chat-template 字符串，并仅编码增量字符串。
   - processor 路径：比较 `prev/curr` 对应 `input_ids`，只保留增量 token。
4. assistant 输出解码后，调用 `ToolParser.extract_tool_calls(...)`。
5. 若存在合法 tool calls，则返回 OpenAI 兼容的 assistant message：
   - `role="assistant"`
   - `content`
   - `tool_calls`
6. 若不存在 tool calls，则返回普通 assistant 文本消息。

### C. Preserve incremental encoding, not full re-encode slicing

maintainer 明确要求 Gateway 采用 delta tokenization，而不是“整段重编码后按旧 token 长度切分”的替代实现。为此：

1. session 必须保留上一轮标准化后的 `message_history`。
2. prefix continuation 时必须基于 `prev` 与 `curr` 的 chat-template 输出求 delta。
3. 多模态路径必须显式区分“已有 images/videos”与“本轮新增 images/videos”，以便 processor 能正确生成 `prev_model_inputs` 与 `curr_model_inputs`。

### D. Simplify lifecycle to phase-only validation

删除：

1. `_terminal_session_phases`
2. `completed_flag`
3. `aborted_flag`

新的生命周期原则：

1. `_sessions` 中存在即表示 session 仍由 gateway 持有。
2. `session.phase` 是唯一真状态源。
3. `finalize_session` / `abort_session` 完成后，从 `_sessions` 中移除。
4. 对已移除 session 的后续访问，不追求细粒度终态诊断，只返回简单错误。

### E. Reward compute is explicit and mandatory

`OpenAICompatibleAgentFramework` 增加必选的 trajectory-level reward compute 抽象。推荐接口形态：

```python
async def reward_compute(
    *,
    trajectories: list[Trajectory],
    prompts: DataProto,
    sample_index: int,
    session_reward_info: dict[str, Any] | None,
) -> list[Trajectory]:
    ...
```

语义约束：

1. `reward_compute` 必须由调用方提供；framework 不做 fallback。
2. `session_reward_info` 只是可选辅助输入，不是 reward 本体。
3. 若调用方想从 `reward_info` 中取某个字段作为最终 reward，应自行实现 wrapper。
4. `reward_compute` 可以：
   - 给同一 session 的多条 trajectories 共享 reward。
   - 为每条 trajectory 单独打分。
5. framework 在 `reward_compute` 完成前不得 assemble。
6. 若未提供 `reward_compute`，或返回 trajectories 后仍缺少有效 `reward_score`，framework 直接报错。

### F. Assembler remains format-only

`TrajectoryAssembler` 继续承担格式收敛职责：

1. 校验 trajectory 结构合法性。
2. 将 `Trajectory.reward_score` 写入 `rm_scores`。
3. 将除主 reward 外的 `reward_info` 字段写入 `non_tensor_batch`。
4. 在 `meta_info["reward_extra_keys"]` 中显式记录额外 reward 字段。

Assembler 不承担 reward 计算逻辑，也不隐式从 `reward_info` 提取主 reward。

## Proposed Code Shape

### `verl/experimental/agent_loop/agent_loop.py`

1. 删除 `LLMServerManager`。
2. 删除 `_GatewayServingBackend`。
3. 将 gateway lifecycle / session runtime 接口直接收敛到 `AsyncLLMServerManager`。

### `verl/experimental/agent_gateway/gateway.py`

1. 引入与 `AgentLoopBase` 对齐的 chat-template、多模态、tool parsing 辅助逻辑。
2. 用 delta tokenization 替换当前 prefix continuation 编码。
3. 解码 assistant 输出时生成 OpenAI 兼容 `tool_calls`。
4. 删除 terminal lifecycle 相关代码。

### `verl/experimental/agent_gateway/types.py`

1. 简化 `GatewaySessionState`。
2. 如有必要，补充 session 级多模态上下文字段。

### `verl/experimental/agent_framework/framework.py`

1. 补充 reward compute 抽象类型或协议定义。

### `verl/experimental/agent_framework/openai_compatible_framework.py`

1. 将 `reward_compute` 作为必选依赖注入。
2. finalize session 后调用 `reward_compute`。
3. 若 reward 缺失则直接报错。

## Error Handling

1. `Gateway` 仅保留必要的请求结构校验和 phase 校验。
2. `complete_session` 只负责记录 `reward_info`，不参与 reward 计算。
3. 无法解析的 tool call 不在 gateway 内做特殊修复；按普通 assistant 文本返回。
4. reward compute 缺失或结果不完整时立即失败，不做隐式兼容。

## Testing Plan

### Gateway / Session Runtime

1. 将 `LLMServerManager` 相关测试迁移到 `AsyncLLMServerManager`。
2. 覆盖 gateway 自持 lifecycle 与 session runtime 接口。
3. 删除 `completed_flag` / `aborted_flag` 相关断言。

### Gateway Encoding / Decoding

1. prefix continuation 使用 delta tokenization。
2. tool call 响应返回 OpenAI 兼容 `message.tool_calls`。
3. tool message 进入历史后，下一轮增量编码正确。
4. 多模态消息进入历史后，processor 路径增量编码正确。

### Framework Reward

1. `reward_compute` 必填。
2. `reward_compute` 可将 session-level reward info 分发到多条 trajectories。
3. 可支持 shared reward 与 per-trajectory reward。
4. assembler 正确生成 `rm_scores` 与 `reward_extra_keys`。

## Risks

1. Gateway 与 `AgentLoopBase` 共享的 chat-template / processor 逻辑如果简单复制，后续可能再次漂移。
2. 多模态 delta tokenization 的状态管理若处理不当，容易造成 image/video 对齐错误。
3. 当前工作树已存在未提交修改，实现时必须避免覆盖用户已有改动。

## Recommendation

本轮只做需求要求的最小闭环：

1. 删除冗余抽象。
2. 对齐 multi-turn / tool / multimodal tokenization。
3. 简化 lifecycle。
4. 明确 reward compute 契约。

不要在本轮继续抽象通用 runtime、统一 trainer reward pipeline，或改动现有 `ToolParser` 协议。
