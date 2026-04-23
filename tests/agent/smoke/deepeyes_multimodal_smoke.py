"""Thin DeepEyes multimodal smoke harness.

Validates that OpenAICompatibleAgentFramework + GatewayServingRuntime can
drive a multimodal session end-to-end using stub components (no GPU, no
model weights, no trainer).

Run:
    python tests/agent/smoke/deepeyes_multimodal_smoke.py --mode stub
"""
from __future__ import annotations

import argparse
import asyncio
import json

import httpx
import ray

from tests.agent.support import (
    FakeProcessor,
    FakeTokenizer,
    RecordingLoadBalancer,
    RecordingRolloutServer,
    fake_vision_info_extractor,
)
from verl.agent.framework.framework import OpenAICompatibleAgentFramework
from verl.agent.gateway.runtime import GatewayServingRuntime
from verl.utils import tensordict_utils as tu


async def main(mode: str) -> None:
    if mode != "stub":
        raise ValueError(f"Unsupported mode: {mode}")

    ray.init(ignore_reinit_error=True)
    rollout_server = RecordingRolloutServer.remote("DEEPEYES-SMOKE")
    load_balancer = RecordingLoadBalancer.remote("server-0")
    runtime = GatewayServingRuntime(
        servers=[("server-0", rollout_server)],
        load_balancer_handle=load_balancer,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "processor": FakeProcessor(),
            "vision_info_extractor": fake_vision_info_extractor,
            "host": "127.0.0.1",
        },
    )

    prompts = tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "image://deepeyes-sample.png"}},
                            {"type": "text", "text": "Inspect the image and use tools if needed."},
                        ],
                    }
                ]
            ],
            "uid": ["deepeyes-smoke-0"],
            "data_source": ["deepeyes/smoke"],
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index):
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{session.base_url}/chat/completions",
                json={
                    "model": "dummy-model",
                    "tools": [{"type": "function", "function": {"name": "zoom_in", "parameters": {"type": "object"}}}],
                    "messages": raw_prompt,
                },
            )
        response.raise_for_status()

    def reward_fn(ctx):
        return [1.0 for _ in ctx.trajectories]

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=reward_fn,
    )
    framework._build_session_id = lambda prompts, sample_index: "deepeyes-smoke-session"

    try:
        output = await framework.generate_sequences(prompts)
        calls = ray.get(rollout_server.get_calls.remote())
        stats = ray.get(load_balancer.stats.remote())
        summary = {
            "num_trajectories": len(output),
            "uid": tu.get(output, "uid"),
            "data_source": tu.get(output, "data_source"),
            "image_data": calls[0]["image_data"],
            "video_data": calls[0]["video_data"],
            "response_shape": list(output["responses"].shape),
            "load_balancer": stats,
        }
        print(json.dumps(summary, indent=2))
    finally:
        await runtime.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thin DeepEyes multimodal smoke harness.")
    parser.add_argument(
        "--mode",
        default="stub",
        choices=["stub"],
        help="Use CPU-friendly stub components instead of a real rollout backend.",
    )
    args = parser.parse_args()
    asyncio.run(main(mode=args.mode))
