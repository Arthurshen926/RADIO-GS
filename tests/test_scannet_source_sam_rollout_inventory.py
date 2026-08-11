from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from radio_gs.scripts.build_scannet_source_sam_rollout_inventory import (
    build_scene_record,
)
from radio_gs.scripts.combine_scannet_source_sam_rollout_inventories import combine
from radio_gs.scripts.seal_scannet_source_sam_mask_generation import (
    build as build_mask_seal,
)
from radio_gs.utils.immutable_artifacts import file_record, sha256_file


def _fixture(tmp_path: Path, *, benchmark_opened: bool = False) -> tuple[Path, Path, Path]:
    scene = "scene0001_00"
    source = tmp_path / "source"
    source.mkdir()
    field = source / "canonical_mpr_v2_d256_l128_fusion.pth"
    graph = source / "v2_shared_support_graph_k16.pt"
    mpr = source / "raw_radio_heldout4.pt"
    dino = source / "dino_v3_heldout4.pt"
    sam = source / "sam3_heldout4.pt"
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "best.pth"
    for path in (field, graph, mpr, dino, sam, checkpoint):
        torch.save({}, path)
    config.write_text("feature_height: 2\nfeature_width: 2\n", encoding="utf-8")
    torch.save(
        {
            "schema_version": 1,
            "metadata": {
                "assignment_mode": "raster_gaussian_top1",
                "registration_weight_mode": "alpha_depth",
                "config": str(config),
                "checkpoint": str(checkpoint),
                "selected_frame_indices": [0, 20],
                "excluded_frame_ids": [40],
                "benchmark_images_opened": benchmark_opened,
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
            },
            "assignments": [{}, {}],
        },
        source / "responsibility_heldout4.pt",
    )
    color = tmp_path / "dataset" / scene / "color"
    color.mkdir(parents=True)
    for frame in (0, 20, 40):
        (color / f"{frame}.jpg").write_bytes(b"rgb")
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    receipt = {
        "scene_id": scene,
        "status": "complete_immutable_gaussian_semantic_score_cache",
        "method_family": "canonical_mpr_v3",
        "canonical_field_source": file_record(field),
        "support_graph_source": file_record(graph),
        "mpr_source": file_record(mpr),
        "geometry_checkpoint": file_record(checkpoint),
    }
    receipt_path = receipt_root / f"{scene}.pt.receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path, tmp_path / "dataset", tmp_path / "rollout"


def test_scene_inventory_uses_explicit_source_frames_and_excludes_heldout(tmp_path: Path) -> None:
    receipt, dataset, rollout = _fixture(tmp_path)
    record = build_scene_record(
        scene_id="scene0001_00",
        receipt_path=receipt,
        dataset_root=dataset,
        rollout_root=rollout,
    )
    assert record["source_frame_ids"] == [0, 20]
    assert record["excluded_frame_ids"] == [40]
    assert all(not value.endswith("/40.jpg") for value in record["source_rgb_paths"])
    assert record["access_audit"]["benchmark_predictions_or_metrics_opened"] is False


def test_scene_inventory_rejects_benchmark_rgb_provenance(tmp_path: Path) -> None:
    receipt, dataset, rollout = _fixture(tmp_path, benchmark_opened=True)
    with pytest.raises(ValueError, match="not query free"):
        build_scene_record(
            scene_id="scene0001_00",
            receipt_path=receipt,
            dataset_root=dataset,
            rollout_root=rollout,
        )


def test_combiner_rejects_method_drift(tmp_path: Path) -> None:
    shared = {
        "schema": "radio_gs.scannet_source_sam_paper8_rollout_inventory.v1",
        "status": "frozen_before_source_sam_generation",
        "scenes": [{"scene_id": "a", "source_frame_count": 1}],
        "shared_method": {"grid": 12},
        "promotion_gate": {"all": True},
        "access_audit": {"metric": False},
    }
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps(shared), encoding="utf-8")
    changed = {**shared, "scenes": [{"scene_id": "b", "source_frame_count": 1}],
               "shared_method": {"grid": 8}}
    right.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="different frozen methods"):
        combine([left, right])


def test_mask_generation_seal_requires_complete_declared_frame_axis(tmp_path: Path) -> None:
    receipt, dataset, rollout_root = _fixture(tmp_path)
    scene = build_scene_record(
        scene_id="scene0001_00", receipt_path=receipt,
        dataset_root=dataset, rollout_root=rollout_root,
    )
    params = {
        "resolution": 1008, "grid_size": 12, "minimum_quality": 0.70,
        "minimum_area_fraction": 0.001, "maximum_area_fraction": 0.80,
        "nms_iou": 0.85, "duplicate_minimum_area_ratio": 0.90,
        "maximum_masks": 0,
    }
    rollout = {
        "schema": "radio_gs.scannet_source_sam_paper8_rollout_inventory.v1",
        "scenes": [scene], "shared_method": {"official_sam_parameters": params},
    }
    root = Path(scene["outputs"]["official_sam_mask_root"]["path"])
    root.mkdir()
    checkpoint = tmp_path / "sam.pt"; torch.save({}, checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    reports = []
    for frame, image in zip(scene["source_frame_ids"], scene["source_rgb_paths"]):
        cache = root / f"{frame}.pt"
        torch.save({
            "scores": torch.tensor([0.9]),
            "metadata": {
                "schema_version": 2,
                "source": "official_sam3_interactive_grid_multimask_hierarchy",
                "official_decoder": True, "query_free": True, "image": image,
                "checkpoint_sha256": checkpoint_sha, "grid_size": 12,
                "minimum_quality": 0.70, "minimum_area_fraction": 0.001,
                "maximum_area_fraction": 0.80, "nms_iou": 0.85,
                "duplicate_minimum_area_ratio": 0.90,
            },
        }, cache)
        reports.append({"image": image, "output": str(cache), "masks": 1})
    (root / "manifest.json").write_text(json.dumps({"images": reports}), encoding="utf-8")
    result = build_mask_seal(
        rollout, rollout_path=tmp_path / "rollout.json", rollout_sha256="a" * 64,
        scene_id="scene0001_00", checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
    )
    assert result["source_frame_count"] == 2
    assert result["summary"]["total_masks"] == 2
