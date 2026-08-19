import argparse
import hashlib

import numpy as np
import pytest
import torch

from radio_gs.models.sam3_multiscale_hierarchy import (
    CropSpec,
    axis_crop_intervals,
    build_crop_pyramid,
    crop_edge_flags,
    dense_point_grid,
    direct_containment_graph,
    pack_masks,
    remap_crop_mask,
    validate_multiscale_cache_payload,
    validate_source_authority_payload,
)
from radio_gs.scripts.build_sam3_multiscale_hierarchy_cache import generation_contract
from radio_gs.scripts.build_sam3_multiscale_hierarchy_cache import automatic_multiscale_hierarchy
from PIL import Image


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_axis_crop_intervals_are_edge_anchored_overlapping_and_gap_free() -> None:
    intervals = axis_crop_intervals(101, count=4, overlap_ratio=0.25)
    assert intervals[0][0] == 0
    assert intervals[-1][1] == 101
    assert all(a[0] < a[1] for a in intervals)
    assert all(right[0] <= left[1] for left, right in zip(intervals, intervals[1:]))
    assert intervals == axis_crop_intervals(101, count=4, overlap_ratio=0.25)


def test_crop_pyramid_and_dense_grid_are_deterministic_in_full_coordinates() -> None:
    crops = build_crop_pyramid(
        image_width=100,
        image_height=80,
        crop_layers=2,
        overlap_ratio=0.25,
        points_per_side=16,
        point_grid_downscale_factor=2,
    )
    assert len(crops) == 1 + 4 + 16
    assert [crop.grid_side for crop in crops[:5]] == [16, 8, 8, 8, 8]
    assert {crop.grid_side for crop in crops[5:]} == {4}
    crop = crops[2]
    local, full = dense_point_grid(crop)
    assert local.shape == full.shape == (crop.grid_side**2, 2)
    assert np.allclose(full, local + np.asarray(crop.box_xyxy[:2], dtype=np.float32))
    assert np.all(local[:, 0] > 0) and np.all(local[:, 0] < crop.width)
    assert np.all(local[:, 1] > 0) and np.all(local[:, 1] < crop.height)


def test_crop_mask_remaps_exactly_without_resizing() -> None:
    local = np.zeros((3, 4), dtype=bool)
    local[1:, 1:3] = True
    full = remap_crop_mask(
        local, crop_box_xyxy=(2, 3, 6, 6), full_height=8, full_width=9
    )
    assert full.shape == (8, 9)
    assert np.array_equal(full[3:6, 2:6], local)
    assert int(full.sum()) == int(local.sum())
    with pytest.raises(ValueError, match="differs from box raster"):
        remap_crop_mask(
            local[:, :-1], crop_box_xyxy=(2, 3, 6, 6), full_height=8, full_width=9
        )


def test_artificial_crop_edge_is_distinguished_from_image_edge() -> None:
    interior = CropSpec(layer=1, index=1, row=0, column=1, box_xyxy=(40, 0, 100, 60), grid_side=8)
    touches, artificial = crop_edge_flags(
        (0, 0, 30, 20),
        crop=interior,
        image_width=120,
        image_height=80,
        tolerance_pixels=1,
    )
    assert touches == (True, True, False, False)
    assert artificial == (True, False, False, False)
    full = CropSpec(layer=0, index=0, row=0, column=0, box_xyxy=(0, 0, 120, 80), grid_side=16)
    _, full_artificial = crop_edge_flags(
        (0, 0, 120, 80),
        crop=full,
        image_width=120,
        image_height=80,
        tolerance_pixels=1,
    )
    assert full_artificial == (False, False, False, False)


def test_direct_containment_graph_selects_nearest_enclosing_parent() -> None:
    outer = np.zeros((12, 12), dtype=bool); outer[1:11, 1:11] = True
    middle = np.zeros((12, 12), dtype=bool); middle[3:9, 3:9] = True
    inner = np.zeros((12, 12), dtype=bool); inner[4:8, 4:8] = True
    separate = np.zeros((12, 12), dtype=bool); separate[:2, :2] = True
    graph = direct_containment_graph(
        [outer, middle, inner, separate],
        [0.7, 0.8, 0.9, 1.0],
        containment_threshold=0.95,
        minimum_parent_area_ratio=1.05,
    )
    assert graph["parent_index"].tolist() == [-1, 0, 1, -1]
    assert graph["parent_edges"].tolist() == [[0, 1], [1, 2]]
    assert graph["parent_containment"][1:].tolist()[:2] == [1.0, 1.0]


def test_source_authority_forbids_queries_gt_and_target_rgb() -> None:
    payload = {
        "schema_version": 1,
        "contract": "sam3-query-free-source-rgb-authority-v1",
        "information_policy": {
            "registered_source_rgb_only": True,
            "query_text_used": False,
            "benchmark_ground_truth_used": False,
            "target_or_evaluation_rgb_used": False,
        },
        "images": [
            {
                "image_id": "rgb_0",
                "path": "images/rgb_0.png",
                "sha256": _sha("rgb"),
                "rgb_role": "registered_source_or_mapping_view",
            }
        ],
    }
    assert validate_source_authority_payload(payload)[0]["image_id"] == "rgb_0"
    polluted = {**payload, "information_policy": {**payload["information_policy"], "query_text_used": True}}
    with pytest.raises(ValueError, match="information policy differs"):
        validate_source_authority_payload(polluted)
    traversing = {**payload, "images": [{**payload["images"][0], "image_id": "../eval"}]}
    with pytest.raises(ValueError, match="ids and paths"):
        validate_source_authority_payload(traversing)


def test_generation_contract_binds_multiscale_geometry_and_forbidden_inputs() -> None:
    args = argparse.Namespace(
        resolution=1008,
        dtype="bfloat16",
        crop_layers=2,
        crop_overlap_ratio=0.25,
        points_per_side=16,
        point_grid_downscale_factor=2,
        minimum_quality=0.7,
        minimum_stability=0.0,
        stability_offset=1.0,
        minimum_crop_area_fraction=0.001,
        minimum_full_image_area_fraction=0.0001,
        maximum_full_image_area_fraction=0.9,
        crop_edge_tolerance_pixels=2,
        dedup_iou=0.85,
        dedup_near_equal_area_ratio=0.9,
        maximum_masks=0,
        containment_threshold=0.9,
        minimum_parent_area_ratio=1.05,
    )
    contract = generation_contract(args, checkpoint_sha256="a" * 64)
    assert contract["query_free"] is True
    assert contract["crop_pyramid"]["layers_after_full_image"] == 2
    assert contract["point_grid"]["placement"].endswith("row_major")
    assert contract["forbidden_inputs"] == [
        "query_text", "benchmark_ground_truth", "target_or_evaluation_rgb"
    ]
    assert len(contract["digest"]) == 64


def test_cache_validation_is_fail_closed_on_metadata_or_shape_change() -> None:
    metadata = {"identity": "sealed"}
    masks = np.zeros((2, 5, 7), dtype=bool)
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "packed_masks": pack_masks(masks),
        "mask_shape": [5, 7],
        "quality": torch.ones(2),
        "stability": torch.ones(2),
        "seed_xy_full": torch.zeros(2, 2),
        "seed_xy_crop": torch.zeros(2, 2),
        "prompt_index": torch.arange(2),
        "candidate_index": torch.arange(2),
        "crop_index": torch.arange(2),
        "crop_layer": torch.zeros(2),
        "crop_grid_side": torch.full((2,), 16),
        "crop_boxes_xyxy": torch.zeros(2, 4),
        "crop_scale_xy": torch.ones(2, 2),
        "crop_window_area_fraction": torch.ones(2),
        "boxes_xyxy": torch.zeros(2, 4),
        "proposal_area_fraction": torch.ones(2),
        "crop_area_fraction": torch.ones(2),
        "touches_crop_edge": torch.zeros(2, 4, dtype=torch.bool),
        "touches_artificial_crop_edge": torch.zeros(2, 4, dtype=torch.bool),
        "parent_index": torch.full((2,), -1),
        "parent_edges": torch.empty(0, 2, dtype=torch.int64),
        "parent_containment": torch.zeros(2),
        "parent_area_ratio": torch.zeros(2),
    }
    assert validate_multiscale_cache_payload(payload, expected_metadata=metadata) == 2
    with pytest.raises(ValueError, match="identity differs"):
        validate_multiscale_cache_payload(payload, expected_metadata={"identity": "other"})
    broken = {**payload, "boxes_xyxy": torch.zeros(1, 4)}
    with pytest.raises(ValueError, match="boxes_xyxy differs"):
        validate_multiscale_cache_payload(broken, expected_metadata=metadata)


def test_mock_official_decoder_path_materializes_per_mask_geometry() -> None:
    class FakeProcessor:
        def __init__(self) -> None:
            self.model = self
            self.shape = (0, 0)

        def set_image(self, image: Image.Image) -> dict:
            self.shape = (image.height, image.width)
            return {}

        def predict_inst(self, state, **kwargs):
            del state, kwargs
            height, width = self.shape
            mask = np.zeros((1, height, width), dtype=bool)
            mask[:, 2 : height - 2, 2 : width - 2] = True
            logits = np.where(mask, 4.0, -4.0).astype(np.float32)
            return mask, np.asarray([0.9], dtype=np.float32), logits

    args = argparse.Namespace(
        crop_layers=0,
        crop_overlap_ratio=0.25,
        points_per_side=1,
        point_grid_downscale_factor=2,
        minimum_quality=0.7,
        minimum_stability=0.0,
        stability_offset=1.0,
        minimum_crop_area_fraction=0.001,
        minimum_full_image_area_fraction=0.0001,
        maximum_full_image_area_fraction=0.9,
        crop_edge_tolerance_pixels=1,
        dedup_iou=0.85,
        dedup_near_equal_area_ratio=0.9,
        maximum_masks=0,
        containment_threshold=0.9,
        minimum_parent_area_ratio=1.05,
    )
    result = automatic_multiscale_hierarchy(
        FakeProcessor(), Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)), args
    )
    assert result["quality"].tolist() == pytest.approx([0.9])
    assert result["crop_layer"].tolist() == [0]
    assert result["crop_boxes_xyxy"].tolist() == [[0, 0, 12, 10]]
    assert result["crop_scale_xy"].tolist() == [[1.0, 1.0]]
    assert result["boxes_xyxy"].tolist() == [[2, 2, 10, 8]]
    assert result["parent_edges"].shape == (0, 2)
