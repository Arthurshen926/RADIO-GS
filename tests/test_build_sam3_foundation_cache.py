import sys
import types
from pathlib import Path

import pytest
import torch

from radio_gs.models.foundation_cache import load_foundation_cache
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    cast_sam3_model_for_inference,
    filter_images_by_frame_ids,
    frame_id_from_path,
    make_sam3_cache_payload,
    make_sam3_processor,
    normalize_sam3_device,
    output_path_for_image,
    parse_queries,
    resolve_sam3_amp_dtype,
    resolve_sam3_dtype,
    sam3_autocast_context,
    set_requested_cuda_device,
    sha256_file,
    validate_sam3_resolution,
)


def test_parse_queries_accepts_inline_values():
    assert parse_queries("mug, red apple\nchair") == ["mug", "red apple", "chair"]


def test_parse_queries_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        parse_queries("")


def test_normalize_sam3_device_matches_official_builder_contract():
    assert normalize_sam3_device("cuda") == "cuda"
    assert normalize_sam3_device("cuda:0") == "cuda"
    assert normalize_sam3_device("cpu") == "cpu"


def test_set_requested_cuda_device_preserves_explicit_cuda_index(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(device))

    set_requested_cuda_device("cuda:2")

    assert calls == [torch.device("cuda:2")]


def test_filter_images_by_frame_ids_accepts_numeric_and_stem_ids(tmp_path):
    images = [
        tmp_path / "frame_00001.jpg",
        tmp_path / "frame_00041.jpg",
        tmp_path / "frame_00105.jpg",
    ]

    selected = filter_images_by_frame_ids(images, "41,frame_00105")

    assert [path.name for path in selected] == ["frame_00041.jpg", "frame_00105.jpg"]


def test_filter_images_by_frame_ids_rejects_empty_result(tmp_path):
    images = [tmp_path / "frame_00001.jpg"]

    with pytest.raises(ValueError, match="No images matched"):
        filter_images_by_frame_ids(images, "99")


def test_frame_id_from_path_preserves_lerf_frame_stem():
    assert frame_id_from_path(Path("/tmp/frame_00041.jpg")) == "frame_00041"


def test_output_path_for_image_uses_frame_stem(tmp_path):
    assert output_path_for_image(tmp_path, Path("/tmp/frame_00041.jpg")) == (
        tmp_path / "frame_00041.pt"
    )


def test_make_sam3_cache_payload_is_strict_official_cache():
    payload = make_sam3_cache_payload(
        frame_id="rgb_000001",
        queries=["mug"],
        masks=torch.ones(2, 8, 8),
        scores=torch.tensor([0.9, 0.7]),
        boxes=torch.zeros(2, 4),
        mask_query_indices=torch.tensor([0, 0]),
        mask_query_ranks=torch.tensor([0, 1]),
        image_path="/tmp/rgb_000001.png",
    )

    cache = load_foundation_cache(payload, require_official=True)

    assert cache.frame_id == "rgb_000001"
    assert cache.heads["sam3"].mask_logits.shape == (2, 8, 8)
    assert payload["heads"]["sam3"]["mask_tensor_semantics"] == "probability"
    assert cache.heads["sam3"].mask_tensor_semantics == "probability"
    assert torch.equal(cache.heads["sam3"].mask_query_indices, torch.tensor([0, 0]))
    assert torch.equal(cache.heads["sam3"].mask_query_ranks, torch.tensor([0, 1]))
    assert cache.heads["sam3"].producer is not None
    assert cache.heads["sam3"].producer.official is True
    assert cache.heads["sam3"].producer.backend == "facebookresearch/sam3"


def test_make_sam3_cache_payload_records_checkpoint_provenance():
    payload = make_sam3_cache_payload(
        frame_id="rgb_000001",
        queries=["mug"],
        masks=torch.ones(1, 4, 4),
        scores=torch.tensor([0.9]),
        boxes=torch.zeros(1, 4),
        image_path="/tmp/rgb_000001.png",
        checkpoint_path="checkpoints/sam3_modelscope/sam3.pt",
        checkpoint_source="modelscope/facebook/sam3",
        checkpoint_sha256="abc123",
    )

    producer = payload["heads"]["sam3"]["producer"]
    assert producer["checkpoint_path"] == "checkpoints/sam3_modelscope/sam3.pt"
    assert producer["checkpoint_source"] == "modelscope/facebook/sam3"
    assert producer["checkpoint_sha256"] == "abc123"


def test_sha256_file(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_resolve_sam3_dtype_keeps_official_weights_float32_by_default():
    assert resolve_sam3_dtype("cuda", "auto") == torch.float32
    assert resolve_sam3_dtype("cuda:0", "auto") == torch.float32
    assert resolve_sam3_dtype("cpu", "auto") == torch.float32
    assert resolve_sam3_dtype("cuda", "bfloat16") == torch.bfloat16


def test_resolve_sam3_amp_dtype_defaults_to_bfloat16_on_cuda():
    assert resolve_sam3_amp_dtype("cuda", "auto") == torch.bfloat16
    assert resolve_sam3_amp_dtype("cuda:0", "auto") == torch.bfloat16
    assert resolve_sam3_amp_dtype("cpu", "auto") is None
    assert resolve_sam3_amp_dtype("cuda", "off") is None


def test_cast_sam3_model_for_inference_converts_parameters():
    model = torch.nn.Linear(2, 2)

    cast_sam3_model_for_inference(model, torch.bfloat16)

    assert model.weight.dtype == torch.bfloat16


def test_make_sam3_processor_forwards_resolution():
    class DummyProcessor:
        def __init__(self, model, *, device, confidence_threshold, resolution):
            self.model = model
            self.device = device
            self.confidence_threshold = confidence_threshold
            self.resolution = resolution

    model = object()
    processor = make_sam3_processor(
        DummyProcessor,
        model,
        device="cuda",
        confidence_threshold=0.4,
        resolution=672,
    )

    assert processor.model is model
    assert processor.device == "cuda"
    assert processor.confidence_threshold == 0.4
    assert processor.resolution == 672


def test_validate_sam3_resolution_rejects_non_default_without_override():
    assert validate_sam3_resolution(1008, allow_unsafe=False) == 1008
    with pytest.raises(ValueError, match="official SAM3 image model expects 1008"):
        validate_sam3_resolution(672, allow_unsafe=False)
    assert validate_sam3_resolution(672, allow_unsafe=True) == 672


def test_sam3_autocast_context_accepts_cpu_bfloat16():
    with sam3_autocast_context("cpu", torch.bfloat16):
        value = torch.ones(1)

    assert value.item() == 1.0


def test_load_sam3_float32_can_build_on_cpu_before_unchanged_cuda_transfer(
    monkeypatch,
):
    calls = []

    class DummyModel:
        def float(self):
            calls.append(("float",))
            return self

        def to(self, *, device):
            calls.append(("to", device))
            return self

    class DummyProcessor:
        def __init__(self, model, *, device, confidence_threshold, resolution):
            self.model = model
            self.device = device
            self.confidence_threshold = confidence_threshold
            self.resolution = resolution

    def dummy_builder(**kwargs):
        calls.append(("build", kwargs["device"]))
        return DummyModel()

    sam3_module = types.ModuleType("sam3")
    sam3_model_module = types.ModuleType("sam3.model")
    processor_module = types.ModuleType("sam3.model.sam3_image_processor")
    processor_module.Sam3Processor = DummyProcessor
    builder_module = types.ModuleType("sam3.model_builder")
    builder_module.build_sam3_image_model = dummy_builder
    monkeypatch.setitem(sys.modules, "sam3", sam3_module)
    monkeypatch.setitem(sys.modules, "sam3.model", sam3_model_module)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", processor_module)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder_module)

    processor = _load_sam3_model(
        checkpoint_path="checkpoint.pt",
        device="cuda",
        confidence_threshold=0.0,
        dtype="float32",
        resolution=1008,
        build_on_cpu=True,
    )

    assert calls == [("build", "cpu"), ("float",), ("to", "cuda")]
    assert processor.device == "cuda"
