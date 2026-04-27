import torch


def test_hf_transformers_rollout_backend_is_registered():
    from verl.workers.rollout.base import get_rollout_class
    from verl.workers.rollout.replica import get_rollout_replica_class

    assert get_rollout_class("hf_transformers", "async").__name__ == "ServerAdapter"
    assert get_rollout_replica_class("hf_transformers").__name__ == "HFTransformersReplica"


def test_hf_transformers_server_builds_multimodal_model_inputs():
    from verl.workers.rollout.hf_transformers_rollout import HFTransformersTokenServer

    class FakeImageProcessor:
        def __call__(self, *, images, return_tensors):
            assert images == ["image-0"]
            assert return_tensors == "pt"
            return {
                "pixel_values": torch.tensor([[1.0, 2.0]]),
                "image_grid_thw": torch.tensor([[1, 2, 3]]),
            }

    class FakeProcessor:
        image_processor = FakeImageProcessor()

    server = object.__new__(HFTransformersTokenServer)
    server._processor = FakeProcessor()
    server._device = torch.device("cpu")
    server._image_token_id = 248056
    server._video_token_id = 248057

    inputs = server._build_model_inputs(
        prompt_ids=[11, 248056, 248056, 12],
        image_data=["image-0"],
        video_data=None,
    )

    assert inputs["input_ids"].tolist() == [[11, 248056, 248056, 12]]
    assert inputs["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert inputs["mm_token_type_ids"].tolist() == [[0, 1, 1, 0]]
    assert inputs["pixel_values"].tolist() == [[1.0, 2.0]]
    assert inputs["image_grid_thw"].tolist() == [[1, 2, 3]]


def test_hf_transformers_server_builds_explicit_generation_kwargs():
    from verl.workers.rollout.hf_transformers_rollout import HFTransformersTokenServer

    server = object.__new__(HFTransformersTokenServer)
    server.config = {"temperature": 0.0, "top_p": 0.9, "top_k": -1}
    server._eos_token_id = 10
    server._pad_token_id = 10

    kwargs = server._build_generation_kwargs({"max_tokens": 8})

    assert kwargs["max_new_tokens"] == 8
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert kwargs["eos_token_id"] == 10
    assert kwargs["pad_token_id"] == 10
