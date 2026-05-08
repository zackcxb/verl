"""CPU tests for DeepEyesGatewayDataset shape assumptions."""

from __future__ import annotations

import importlib
import io
import pathlib
import sys

import pytest
import torch
from PIL import Image

TESTS_AGENT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _import_dataset_class():
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("recipe", None)
        sys.path[:] = [path for path in sys.path if path != TESTS_AGENT_ROOT]
        module = importlib.import_module("recipe.deepeyes_with_gateway.dataset")
        return module.DeepEyesGatewayDataset
    finally:
        sys.path[:] = original_sys_path


def _make_dataset(row, *, negative_prompt_key="negative_prompt"):
    DatasetCls = _import_dataset_class()
    dataset = DatasetCls.__new__(DatasetCls)
    dataset.dataframe = [row]
    dataset.prompt_key = "prompt"
    if negative_prompt_key is not None:
        dataset.negative_prompt_key = negative_prompt_key
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.need_tools_kwargs = False
    return dataset


def _png_bytes(color=(200, 40, 40)):
    image = Image.new("RGB", (8, 8), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_dataset_accepts_user_only_message_shape():
    dataset = _make_dataset(
        {
            "prompt": [{"role": "user", "content": "Describe the image."}],
            "extra_info": {},
        }
    )

    item = dataset[0]

    assert item["raw_prompt"][0]["role"] == "system"
    assert item["raw_prompt"][1] == {"role": "user", "content": "Describe the image."}
    assert torch.equal(item["dummy_tensor"], torch.tensor([0], dtype=torch.uint8))


def test_dataset_accepts_missing_negative_prompt_key_attribute():
    dataset = _make_dataset(
        {
            "prompt": [{"role": "user", "content": "Describe the image."}],
            "extra_info": {},
        },
        negative_prompt_key=None,
    )

    item = dataset[0]

    assert item["raw_prompt"][1] == {"role": "user", "content": "Describe the image."}
    assert "raw_negative_prompt" not in item


def test_dataset_accepts_system_then_user_shape():
    dataset = _make_dataset(
        {
            "prompt": [
                {"role": "system", "content": "Ignored legacy system."},
                {"role": "user", "content": "Find the cat."},
            ],
            "extra_info": {},
        }
    )

    item = dataset[0]

    assert item["raw_prompt"][1] == {"role": "user", "content": "Find the cat."}


def test_dataset_rejects_message_shape_without_user():
    dataset = _make_dataset(
        {
            "prompt": [{"role": "assistant", "content": "I am not a valid input."}],
            "extra_info": {},
        }
    )

    with pytest.raises(ValueError, match="Cannot find user message"):
        dataset[0]


def test_dataset_replaces_image_placeholder_and_builds_tool_kwargs():
    image_payload = {"bytes": _png_bytes()}
    dataset = _make_dataset(
        {
            "prompt": [{"role": "user", "content": "Inspect <image> and answer."}],
            "images": [image_payload],
            "extra_info": {},
        }
    )

    item = dataset[0]

    user_content = item["raw_prompt"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "Inspect "}
    assert user_content[1] == {"type": "image", "image": image_payload}
    assert user_content[2] == {"type": "text", "text": " and answer."}
    tool_image = item["tools_kwargs"]["image_zoom_in_tool"]["create_kwargs"]["image"]
    assert tool_image == image_payload
    assert "images" not in item


def test_dataset_preserves_existing_image_content_without_data_uri_conversion():
    image = {"bytes": _png_bytes(color=(20, 40, 200))}
    dataset = _make_dataset(
        {
            "prompt": [{"role": "user", "content": [{"type": "image", "image": image}]}],
            "extra_info": {},
        }
    )

    item = dataset[0]

    assert item["raw_prompt"][1]["content"] == [{"type": "image", "image": image}]
    assert item["tools_kwargs"]["image_zoom_in_tool"]["create_kwargs"]["image"] == image
