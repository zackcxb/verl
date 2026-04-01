# AgentFramework 与 AgentGateway 设计文档 v2

> 本文档是 2026-04-01 的设计收敛版本，覆盖并取代 [2026-03-30-agent-framework-gateway-design.md](/home/cxb/rl_framework/verl/cxb_dev/docs/plans/2026-03-30-agent-framework-gateway-design.md) 作为当前执行基线。旧文档保留为讨论历史记录，不再作为当前实现依据。

## 1. 背景与目标

本文档基于：

- RFC [agentFramework_agentgateway_rfc.md](/home/cxb/MATE-reboot/docs/rfc/agentFramework_agentgateway_rfc/agentFramework_agentgateway_rfc.md)
- VERL `main` 分支现状
- 与 Xibin 的最新对齐结果
- 对现有 `AgentLoop`、Aliyun Remote Agent、AWS Bedrock / AgentCore 目标场景的复盘

本版设计的核心目标是：

- 为 VERL 引入一个足够薄的 `AgentFramework` 抽象，使其能统一承接 VERL-native agent、remote agent、hosted agent framework。
- 为 VERL 引入一个由 `LLMServerManager` 拥有的 `AgentGateway` 子系统，通过 OpenAI-compatible `/v1/chat/completions` 截获请求并组装 trajectories。
- 明确首版主路径是 chat-completions / session-based integration，而不是现有 `AgentLoopManager` 的立即迁移。
- 为后续 `AgentLoopManager` 迁移预留 token request ingress，但不把它作为首版阻塞项。

非目标：

- 首版不迁移现有 `AgentLoopManager` 到新 Gateway 主路径。
- 首版不要求实现 token request ingress。
- 首版不要求支持全部 legacy `extra_fields`、teacher logprobs、fully async integration。
- 首版不要求把 Gateway 设计成独立于 serving runtime 的顶级平台服务。

## 2. 已收敛的核心决策

### 2.1 `AgentFramework` 是薄抽象

- `AgentFramework` 是统一抽象基类。
- 首版公共抽象接口只保留：
  - `generate_sequences(prompts: DataProto) -> DataProto`
- `run_session`
- `compute_reward`
- `finalize_session`
  这些都不进入 `AgentFramework` 的正式公共接口。

原因：

- trainer 真正依赖的是 `generate_sequences`。
- Aliyun / Bedrock / AgentCore 这类 remote/hosted framework 不适合被要求实现 VERL-native 的内部 hook。
- `compute_reward` 是 shared concern，但不是必须的 shared abstract method。

### 2.2 `AgentLoopManager` 是 `AgentFramework` 的一种特化实现

- `AgentLoopManager` 在长期方向上应当成为 `AgentFramework` 的一个 concrete implementation。
- 但 maintainer 已明确：`AgentLoopManager` 不作为首版迁移目标。
- 首版不再引入 `AgentFrameworkManager -> FrameworkWorker -> AgentFramework` 这套公共三层抽象。

含义：

- `FrameworkWorker` 不再是新架构的一层。
- `AgentLoopWorker` 若继续存在，也只能是 `AgentLoopManager` 的内部实现细节。

### 2.3 Gateway ownership 归 `LLMServerManager`

- Gateway 不由 `AgentFramework` 或 `AgentLoopManager` 拥有。
- Gateway 由 `LLMServerManager` 拥有并托管。
- `GatewayManager` 可以保留，但其定位是 `LLMServerManager` 内部的 session-routing / control-plane 子组件。
- `GatewayActor` 由 `LLMServerManager` 或其 runtime factory 创建，而不是由 `GatewayManager` 自己创建。

### 2.4 Framework 侧依赖的是窄 session capability

- Framework 不应长期直接绑定 `GatewayManager` 具体类型。
- Framework 应依赖一个由 `LLMServerManager` 暴露的窄 session capability。
- 首版建议能力集合：
  - `create_session`
  - `finalize_session`
  - `abort_session`
  - `wait_for_completion`

建议 session handle：

```python
@dataclass
class SessionHandle:
    session_id: str
    base_url: str | None = None
```

### 2.5 Gateway 首版只优先支持 `/v1/chat/completions`

- 首版正式主路径是 session-based OpenAI-compatible chat completion。
- token request ingress 需要预留扩展点，但不是首版优先实现项。
- `AgentLoopManager` 的迁移留到后续阶段。

这意味着：

- 首版优先服务 remote/hosted/OpenAI-compatible agent integration。
- 不为兼容现有 `AgentLoop` 立即把 Gateway 做成双协议入口。

### 2.6 token request ingress 是明确的后续扩展点

- 现有 `AgentLoop` 深度依赖 `AsyncLLMServerManager.generate(...)`。
- 如果未来要把 `AgentLoopManager` 迁移到 Gateway 主路径，需要一个可接收 token request 的 ingress。
- 该能力在本版设计中只做边界预留，不纳入首版交付。

### 2.7 Gateway 与 rollout server 是 `N:M` 关系

- `gateway_count` 可以独立配置。
- `gateway_count = 0` 表示不启动 Gateway 子系统。
- 这意味着 Gateway 与 rollout server 不是强 1:1 绑定关系。
- 更合理的拓扑是：
  - 数量独立
  - lifecycle 从属于 `LLMServerManager`
  - 运行时按路由关联

### 2.8 `wait_for_completion` 与 `/complete` 是一等能力

- `wait_for_completion` 不应删除。
- `POST /sessions/{id}/complete` 保留为可选完成信号与可选 `reward_info` 上传通道。
- 这对 remote/hosted agent 很重要。
- 对 subprocess / coroutine agent，它不是必经路径。

### 2.9 reward 是 shared concern，但不进入 framework 抽象接口

- trajectory reward assignment 是共性问题。
- 但 `compute_reward(...)` 不要求成为 `AgentFramework` 的抽象方法。
- 更合理的落位是：
  - concrete framework implementation 内部逻辑
  - 或 helper utility

首版建议沉淀的共享逻辑：

- reward normalization
- `trajectory -> DataProto` assembler
- 基本 validation

### 2.10 现有 `AgentLoop` 不应与 Gateway 主线双重记账

- 如果某条路径启用了 Gateway 并以其为 trajectory 真相源，则不能继续让旧 `AgentLoop` trajectory bookkeeping 作为生产真相源。
- 双重 bookkeeping 只适合迁移验证，不进入正式运行时设计。
- 因此现有 `AgentLoopManager` 迁移应当在 token ingress 方案明确后单独推进。

## 3. 三类 agent 场景分析

### 3.1 现有 `AgentLoop`

本地代码证据表明：

- trainer-facing contract 已经是 `generate_sequences`。
- 现有路径是 `AgentLoopManager -> AgentLoopWorker -> AgentLoopBase`。
- agent 多数直接调用 `AsyncLLMServerManager.generate(...)`。
- `AgentLoopManager` 当前还自己拥有 rollout server init、load balancer、worker placement、batch orchestration。

对新框架的要求：

- `AgentFramework` 薄接口是可接受的。
- 最大问题不是 framework 顶层接口，而是后续如何让 legacy native loop 过渡到 Gateway truth source。
- 现有 `AgentLoop` 迁移不应阻塞首版。

### 3.2 Aliyun Remote Agent

基于 RFC 和 issue 语义推断：

- 更像 remote service / hosted framework，而不是 VERL-native loop。
- 更关心：
  - `create_session`
  - `wait_for_completion`
  - `/complete`
  - `finalize_session`
- 对 `AgentFramework` 顶层抽象方法没有额外要求。

设计含义：

- 当前“薄 framework + 强 session runtime”方向更适合它。

### 3.3 AWS Bedrock / AgentCore

基于 RFC 和社区引用推断：

- 更像 hosted remote framework。
- 与 Aliyun 类似，更依赖稳定的 session/gateway capability。
- 会强化如下方向：
  - `AgentFramework` 只保留 `generate_sequences`
  - `wait_for_completion` 为正式能力
  - `/complete` 与 `reward_info` 上传通道保留

## 4. 模块落位

建议新增模块：

- `verl/experimental/agent_framework/`
  - `framework.py`
    - `AgentFramework`
  - `types.py`
    - `SessionHandle`
    - `Trajectory`
    - 相关轻量类型
  - `assembler.py`
    - `TrajectoryAssembler`
  - `helpers.py`
    - reward normalization / validation helpers

- `verl/experimental/agent_gateway/`
  - `types.py`
  - `manager.py`
    - `GatewayManager`，作为 `LLMServerManager` 的内部子组件
  - `gateway.py`
    - `GatewayActor`

现有模块处理：

- `verl/experimental/agent_loop/agent_loop.py`
  - 首版不做主路径迁移
  - 后续迁移时再处理 token ingress 与 legacy bookkeeping

- `verl/experimental/fully_async_policy/`
  - 首版不适配
  - 保持现状

## 5. 关键组件职责

### 5.1 `AgentFramework`

负责：

- 暴露统一 trainer-facing `generate_sequences`

不负责：

- 规定内部是 session-based 还是 token-based
- 规定 reward hook 形式
- 规定 worker 拓扑
- 规定 Gateway lifecycle

### 5.2 `LLMServerManager`

负责：

- rollout servers / backend handles ownership
- load balancer ownership
- Gateway 子系统 ownership
- 向 framework 暴露窄 session capability
- 维持 serving/runtime control-plane 能力

### 5.3 `GatewayManager`

定位：

- `LLMServerManager` 内部 session-routing 子组件

负责：

- `session_id -> gateway actor` sticky routing
- `create/finalize/abort/wait` forwarding

不负责：

- 创建 `GatewayActor`
- ownership 决策
- framework-facing 顶级抽象

### 5.4 `GatewayActor`

首版负责：

- session state
- `/v1/chat/completions`
- message-level prefix consistency
- trajectory assembly
- `/complete`

首版不负责：

- token ingress
- `AgentLoop` compatibility bridge

### 5.5 `TrajectoryAssembler` / helpers

负责：

- `trajectory -> DataProto`
- reward normalization
- validation

目标是对齐现有训练可观察输出契约：

- `prompts`
- `responses`
- `response_mask`
- `input_ids`
- `attention_mask`
- `position_ids`
- 可选 `rollout_log_probs`
- `rm_scores`
- `__num_turns__`
- 条件性 `routed_experts`

## 6. 首版接口建议

### 6.1 `AgentFramework`

```python
class AgentFramework(ABC):
    @abstractmethod
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        ...
```

### 6.2 session capability

```python
class SessionRuntime(Protocol):
    async def create_session(self, session_id: str, **kwargs) -> SessionHandle: ...
    async def finalize_session(self, session_id: str) -> list[Trajectory]: ...
    async def abort_session(self, session_id: str) -> None: ...
    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None: ...
```

### 6.3 token ingress 预留

不在首版正式实现，但设计上保留后续能力位：

- 接收 token request
- 服务于 `AgentLoopManager` 迁移
- 与 chat-completions 共享 session / trajectory model

## 7. 首版交付范围

### 本系列 PR1：Gateway + 薄 Framework 主闭环

交付：

- `AgentFramework` 薄抽象
- `LLMServerManager` 拥有 Gateway 子系统的边界设计
- `GatewayManager` / `GatewayActor`
- `/v1/chat/completions`
- `create/finalize/abort/wait`
- `/complete`
- `TrajectoryAssembler`
- 一个最小的 OpenAI-compatible / remote-style reference path
- 组件测试
- 一次轻量 E2E / inspection run

不交付：

- `AgentLoopManager` 迁移
- token ingress
- `ReactAgentLoop` bridge
- `Retool` / `SWE-Agent` validation

### 后续 PR

- token ingress
- `AgentLoopManager` 迁移
- `ReactAgentLoop` / `Retool` / `SWE-Agent` validation
- teacher logprobs / legacy compatibility / fully async 兼容

## 8. 风险与约束

主要风险：

- 如果首版仍试图同时兼容现有 `AgentLoop` 直连 `generate` 语义，容易出现 Gateway 与 legacy bookkeeping 的双重真相源冲突。
- 如果 framework 直接长期依赖 `GatewayManager` 具体类型，后续 `LLMServerManager` 内部调整空间会变小。
- 如果把 token ingress 与 chat-completions 同时作为首版主路径，复杂度会明显上升。

主要约束：

- maintainer 已明确首版优先 `/v1/chat/completions`。
- `AgentLoopManager` 后续再迁移。
- rollout runtime 抽取仍以上游后续工作为准。

## 9. 当前实施原则

- 先把 chat-completions 主路径做通。
- 抽象保持薄，不把 `AgentLoop` 的内部执行模型推成通用框架。
- Gateway ownership 明确归 `LLMServerManager`。
- Aliyun / Bedrock / AgentCore 这类 remote/hosted agent 视为首版设计的重要目标场景。
- `AgentLoop` 是后续迁移对象，不是首版阻塞项。
