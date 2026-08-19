import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from radio_gs.scripts.build_sam3_automatic_mask_cache import pack_masks
from radio_gs.scripts.build_sam_mask_aligned_language_teacher import (
    PREREGISTRATION_SCHEMA,
    build_crop_pairs,
    build_frame_teacher,
    build_geometric_topology,
    preflight_inputs,
)
from radio_gs.utils.immutable_artifacts import sha256_file


GENERATION = {
    "schema_version": 2,
    "source": "official_sam3_interactive_grid_multimask_hierarchy",
    "official_decoder": True,
    "query_free": True,
    "checkpoint_sha256": "sam-sha",
    "resolution": 1008,
    "dtype": "bfloat16",
    "grid_size": 12,
    "minimum_quality": 0.7,
    "minimum_area_fraction": 0.001,
    "maximum_area_fraction": 0.8,
    "nms_iou": 0.85,
    "minimum_stability": 0.0,
    "stability_offset": 1.0,
    "deduplication": "containment_aware_near_duplicate_only",
    "duplicate_minimum_area_ratio": 0.9,
    "maximum_masks": 0,
}


def _masks() -> np.ndarray:
    masks = np.zeros((3, 8, 8), dtype=bool)
    masks[0, 1:7, 1:7] = True
    masks[1, 2:4, 2:4] = True
    masks[2, 4:6, 4:6] = True
    return masks


def _boxes(masks: np.ndarray) -> torch.Tensor:
    result = []
    for mask in masks:
        y, x = np.where(mask)
        result.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
    return torch.tensor(result, dtype=torch.int32)


def _fixture(tmp_path: Path) -> Path:
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    image_root.mkdir(); mask_root.mkdir()
    image_path = image_root / "frame_00001.jpg"
    values = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    Image.fromarray(values).save(image_path)
    masks = _masks()
    areas = masks.reshape(3, -1).mean(axis=1)
    payload = {
        "packed_masks": pack_masks(masks),
        "mask_shape": [8, 8],
        "scores": torch.tensor([0.9, 0.8, 0.7]),
        "stability": torch.tensor([1.0, 0.95, 0.9]),
        "seed_xy": torch.tensor([[3.0, 3.0], [2.0, 2.0], [5.0, 5.0]]),
        "prompt_index": torch.tensor([0, 1, 2], dtype=torch.int32),
        "candidate_index": torch.tensor([0, 1, 2], dtype=torch.int8),
        "boxes_xyxy": _boxes(masks),
        "proposal_area_fraction": torch.tensor(areas, dtype=torch.float32),
        "metadata": {
            **GENERATION,
            "image": str(image_path.resolve()),
            "source_image_sha256": sha256_file(image_path),
        },
    }
    mask_path = mask_root / "frame_00001.pt"
    torch.save(payload, mask_path)
    manifest = {
        "output_root": str(mask_root.resolve()),
        "generation_contract": GENERATION,
        "images": [
            {
                "image": str(image_path.resolve()),
                "output": str(mask_path.resolve()),
                "source_image_sha256": sha256_file(image_path),
                "output_sha256": sha256_file(mask_path),
                "masks": 3,
            }
        ],
    }
    manifest_path = mask_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checkpoint = tmp_path / "radio.pt"
    checkpoint.write_bytes(b"frozen-radio")
    preregistration = {
        "schema": PREREGISTRATION_SCHEMA,
        "candidate_id": "unit_test",
        "scene": "figurines",
        "source_contract": {
            "query_free": True,
            "source_only": True,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_vocabulary_opened": False,
            "evaluation_rgb_opened": False,
            "text_queries_opened": False,
        },
        "inputs": {
            "source_image_root": str(image_root.resolve()),
            "source_image_stems": ["frame_00001"],
            "mask_root": str(mask_root.resolve()),
            "mask_manifest": str(manifest_path.resolve()),
            "required_mask_generation_contract": GENERATION,
            "radio_checkpoint": str(checkpoint.resolve()),
            "radio_checkpoint_sha256": sha256_file(checkpoint),
            "radio_repo": "/not/loaded/in/preflight",
            "radio_version": "c-radio_v4-h",
        },
        "descriptor_contract": {
            "teacher_space": "official_siglip2_crop_summary",
            "masked_background_rgb": [0.5, 0.5, 0.5],
            "context_expansion": 1.5,
            "crop_resolution": 16,
            "batch_size": 4,
            "output_dtype": "float16",
        },
        "topology_contract": {
            "containment_threshold": 0.95,
            "maximum_child_parent_area_ratio": 0.8,
            "sibling_maximum_iou": 0.05,
            "semantic_assertion": "geometric_candidates_only",
        },
        "output_root": str((tmp_path / "output").resolve()),
    }
    preregistration_path = tmp_path / "preregistration.json"
    preregistration_path.write_text(json.dumps(preregistration))
    return preregistration_path


def test_preflight_accepts_fully_bound_source_only_mask_authority(tmp_path: Path) -> None:
    preregistration, records = preflight_inputs(_fixture(tmp_path))
    assert preregistration["scene"] == "figurines"
    assert len(records) == 1
    assert records[0]["masks"].shape == (3, 8, 8)


def test_preflight_rejects_missing_root_generation_contract(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    preregistration = json.loads(path.read_text())
    manifest_path = Path(preregistration["inputs"]["mask_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("generation_contract")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="root generation contract differs"):
        preflight_inputs(path)


def test_preflight_rejects_mutated_source_rgb(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    preregistration = json.loads(path.read_text())
    image_path = Path(preregistration["inputs"]["source_image_root"]) / "frame_00001.jpg"
    image_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source image SHA differs"):
        preflight_inputs(path)


def test_geometric_topology_keeps_part_whole_and_sibling_candidates() -> None:
    topology = build_geometric_topology(
        _masks(),
        containment_threshold=0.95,
        maximum_child_parent_area_ratio=0.8,
        sibling_maximum_iou=0.05,
    )
    assert topology["direct_parent_index"].tolist() == [-1, 0, 0]
    assert sorted(topology["candidate_part_of_edges"].tolist()) == [[1, 0], [2, 0]]
    assert topology["candidate_sibling_edges"].tolist() == [[1, 2]]


def test_masked_and_context_crops_are_distinct_and_bounded() -> None:
    image = torch.linspace(0, 1, 3 * 8 * 8).reshape(3, 8, 8)
    masks = _masks()
    masked, context = build_crop_pairs(
        image,
        masks,
        _boxes(masks),
        context_expansion=1.5,
        crop_resolution=12,
        masked_background_rgb=[0.5, 0.5, 0.5],
    )
    assert masked.shape == context.shape == (3, 3, 12, 12)
    assert bool(((masked >= 0) & (masked <= 1)).all())
    assert not torch.equal(masked[1], context[1])


class _FakeOfficialRuntime:
    def encode(self, crops: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(crops.shape[0], 1536, device=crops.device)
        result[:, 0] = 1.0
        return result


def test_frame_teacher_emits_two_official_space_descriptors_and_topology(
    tmp_path: Path,
) -> None:
    _preregistration, records = preflight_inputs(_fixture(tmp_path))
    teacher = build_frame_teacher(
        records[0],
        _FakeOfficialRuntime(),
        descriptor_contract={
            "teacher_space": "official_siglip2_crop_summary",
            "masked_background_rgb": [0.5, 0.5, 0.5],
            "context_expansion": 1.5,
            "crop_resolution": 16,
            "batch_size": 2,
            "output_dtype": "float16",
        },
        topology_contract={
            "containment_threshold": 0.95,
            "maximum_child_parent_area_ratio": 0.8,
            "sibling_maximum_iou": 0.05,
            "semantic_assertion": "geometric_candidates_only",
        },
        device=torch.device("cpu"),
    )
    assert teacher["masked_crop_descriptor"].shape == (3, 1536)
    assert teacher["context_crop_descriptor"].shape == (3, 1536)
    assert teacher["masked_crop_descriptor"].dtype == torch.float16
    assert teacher["metadata"]["benchmark_masks_opened"] is False
    assert teacher["topology"]["candidate_sibling_edges"].tolist() == [[1, 2]]
