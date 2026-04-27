# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Minimal HuggingFace Transformers async rollout backend.

This backend is intended for smoke validation when a model is supported by
Transformers but not by the high-throughput vLLM/SGLang rollout servers. It
implements the same token-in/token-out server contract used by the agent
gateway, but deliberately keeps trainer-side weight sync as a no-op.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Generator

import ray
import torch
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoModelForImageTextToText, AutoProcessor

from verl.utils.device import get_resource_name
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import RolloutReplica, TokenOutput

logger = logging.getLogger(__name__)


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class ServerAdapter(BaseRollout):
    """No-op trainer-side adapter for the Transformers smoke backend."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        super().__init__(config, model_config, device_mesh)
        self.replica_rank = replica_rank
        self.is_leader_rank = True

    async def resume(self, tags: list[str]):
        del tags

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        del weights, kwargs

    async def release(self):
        pass


class HFTransformersTokenServer:
    """Ray actor target that serves token generation with Transformers."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: DictConfig | HFModelConfig,
        replica_rank: int,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.replica_rank = replica_rank
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        model_path = _get_config_value(model_config, "local_path") or _get_config_value(model_config, "path")
        trust_remote_code = bool(_get_config_value(model_config, "trust_remote_code", False))
        dtype_name = _get_config_value(config, "dtype", "bfloat16")
        dtype = getattr(torch, dtype_name, torch.bfloat16)

        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        custom_chat_template = _get_config_value(model_config, "custom_chat_template")
        if custom_chat_template is not None:
            self._processor.chat_template = custom_chat_template
            if hasattr(self._processor, "tokenizer"):
                self._processor.tokenizer.chat_template = custom_chat_template

        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
        ).to(self._device)
        self._model.eval()

        hf_config = getattr(self._model, "config", None)
        self._image_token_id = getattr(hf_config, "image_token_id", None)
        self._video_token_id = getattr(hf_config, "video_token_id", None)
        tokenizer = getattr(self._processor, "tokenizer", None)
        self._eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) or self._eos_token_id

    def _move_to_device(self, values: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(self._device) if hasattr(value, "to") else value for key, value in values.items()}

    def _build_model_inputs(
        self,
        *,
        prompt_ids: list[int],
        image_data: list[Any] | None,
        video_data: list[Any] | None,
    ) -> dict[str, Any]:
        if video_data:
            raise NotImplementedError("hf_transformers rollout currently supports image inputs only")

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)
        inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, device=self._device),
        }

        multimodal_token_ids = {token_id for token_id in (self._image_token_id, self._video_token_id) if token_id}
        if multimodal_token_ids:
            mask = torch.zeros_like(input_ids, device=self._device)
            for token_id in multimodal_token_ids:
                mask = torch.where(input_ids == token_id, torch.ones_like(mask), mask)
            if mask.any():
                inputs["mm_token_type_ids"] = mask

        if image_data:
            image_inputs = self._processor.image_processor(images=image_data, return_tensors="pt")
            inputs.update(self._move_to_device(dict(image_inputs)))

        return inputs

    def _build_generation_kwargs(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        temperature = float(sampling_params.get("temperature", _get_config_value(self.config, "temperature", 1.0)))
        top_p = float(sampling_params.get("top_p", _get_config_value(self.config, "top_p", 1.0)))
        top_k = int(sampling_params.get("top_k", _get_config_value(self.config, "top_k", -1)))
        max_new_tokens = int(sampling_params.get("max_tokens") or _get_config_value(self.config, "response_length", 512))
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self._eos_token_id,
            "pad_token_id": self._pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(temperature, 1e-5)
            generation_kwargs["top_p"] = top_p
            if top_k > 0:
                generation_kwargs["top_k"] = top_k
        return generation_kwargs

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        **kwargs: Any,
    ) -> TokenOutput:
        del request_id, kwargs
        inputs = self._build_model_inputs(prompt_ids=prompt_ids, image_data=image_data, video_data=video_data)
        generation_kwargs = self._build_generation_kwargs(sampling_params)
        logger.info(
            "hf_transformers generate start replica=%s prompt_len=%s images=%s max_new_tokens=%s",
            self.replica_rank,
            len(prompt_ids),
            len(image_data or []),
            generation_kwargs["max_new_tokens"],
        )
        start = torch.cuda.Event(enable_timing=True) if self._device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if self._device.type == "cuda" else None
        if start is not None:
            start.record()

        with torch.no_grad(), torch.autocast(device_type=self._device.type, dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            output = self._model.generate(
                **inputs,
                **generation_kwargs,
            )
        if end is not None:
            end.record()
            torch.cuda.synchronize(self._device)
            elapsed_ms = start.elapsed_time(end)
        else:
            elapsed_ms = 0.0

        prompt_len = inputs["input_ids"].shape[1]
        token_ids = output.sequences[0, prompt_len:].detach().cpu().tolist()
        logger.info(
            "hf_transformers generate done replica=%s output_len=%s elapsed_ms=%.1f",
            self.replica_rank,
            len(token_ids),
            elapsed_ms,
        )
        stop_reason = "stop" if self._eos_token_id in token_ids else "length"
        return TokenOutput(token_ids=token_ids, stop_reason=stop_reason)

    async def sleep(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def wake_up(self):
        pass

    async def clear_kv_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def abort_all_requests(self):
        return {"aborted_count": 0, "request_ids": []}

    async def resume_generation(self):
        pass

    async def start_profile(self, **kwargs):
        del kwargs

    async def stop_profile(self):
        pass


class HFTransformersReplica(RolloutReplica):
    """RolloutReplica that launches one Transformers token server per replica."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ) -> None:
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix)
        self.server_class = ray.remote(HFTransformersTokenServer)

    async def launch_servers(self):
        if self.world_size != 1:
            raise ValueError("hf_transformers rollout only supports tensor/data/pipeline parallel size 1")
        if len(self.workers) != 1:
            raise ValueError(f"hf_transformers expected one colocated worker, got {len(self.workers)}")

        node_id, cuda_visible_device = await self.workers[0].__ray_call__.remote(
            lambda self: (
                ray.get_runtime_context().get_node_id(),
                ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
            )
        )

        name = f"hf_transformers_server_{self.replica_rank}{self.name_suffix}"
        server = self.server_class.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            runtime_env={
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": str(cuda_visible_device),
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                }
            },
            name=name,
            max_concurrency=self.max_concurrency,
        ).remote(config=self.config, model_config=self.model_config, replica_rank=self.replica_rank)
        self.servers.append(server)
        self._server_handle = server
        self._server_address = name

    async def wake_up(self):
        await asyncio.gather(*[server.wake_up.remote() for server in self.servers])
