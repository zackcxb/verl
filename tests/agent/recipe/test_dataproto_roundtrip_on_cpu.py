"""Verify TensorDict -> DataProto conversion preserves trainer-required fields."""

import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils import tensordict_utils as tu

TRAINER_TENSOR_FIELDS = (
    "prompts",
    "responses",
    "response_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "rollout_log_probs",
    "rm_scores",
)

TRAINER_NON_TENSOR_FIELDS = (
    "multi_modal_inputs",
    "multi_modal_data",
    "__num_turns__",
)


def _make_framework_output(batch_size: int = 2, prompt_len: int = 4, response_len: int = 6) -> TensorDict:
    """Build a TensorDict that mimics assembler + multimodal postprocess output."""
    seq_len = prompt_len + response_len
    tensor_dict = {
        "prompts": torch.randint(0, 100, (batch_size, prompt_len)),
        "responses": torch.randint(0, 100, (batch_size, response_len)),
        "response_mask": torch.ones(batch_size, response_len, dtype=torch.long),
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
        "rollout_log_probs": torch.randn(batch_size, response_len),
        "rm_scores": torch.zeros(batch_size, response_len),
    }
    non_tensor_dict = {
        "multi_modal_inputs": [{"image_grid_thw": torch.tensor([[1, 2, 2]])} for _ in range(batch_size)],
        "multi_modal_data": [{"images": ["fake_img"]} for _ in range(batch_size)],
        "__num_turns__": [3 for _ in range(batch_size)],
    }
    return tu.get_tensordict(tensor_dict=tensor_dict | non_tensor_dict)


def _assert_nested_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return

    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
        return

    assert actual == expected


def _assert_tensor_fields_preserved(actual: TensorDict, expected: TensorDict) -> None:
    for field in TRAINER_TENSOR_FIELDS:
        assert field in actual
        assert torch.equal(actual[field], expected[field]), field


def _assert_non_tensor_fields_preserved(actual: dict[str, object], expected: TensorDict) -> None:
    for field in TRAINER_NON_TENSOR_FIELDS:
        assert field in actual
        _assert_nested_equal(actual[field].tolist(), tu.get(expected, field))


def test_framework_output_to_dataproto_preserves_tensors():
    td = _make_framework_output(batch_size=2)
    dp = DataProto.from_tensordict(td)

    _assert_tensor_fields_preserved(dp.batch, td)
    assert dp.batch["prompts"].shape == (2, 4)
    assert dp.batch["responses"].shape == (2, 6)


def test_framework_output_to_dataproto_preserves_non_tensors():
    td = _make_framework_output(batch_size=2)
    dp = DataProto.from_tensordict(td)

    _assert_non_tensor_fields_preserved(dp.non_tensor_batch, td)


def test_dataproto_roundtrip_through_to_tensordict():
    td_original = _make_framework_output(batch_size=2)
    dp = DataProto.from_tensordict(td_original)
    td_back = dp.to_tensordict()

    _assert_tensor_fields_preserved(td_back, td_original)
    for field in TRAINER_NON_TENSOR_FIELDS:
        _assert_nested_equal(tu.get(td_back, field), tu.get(td_original, field))
