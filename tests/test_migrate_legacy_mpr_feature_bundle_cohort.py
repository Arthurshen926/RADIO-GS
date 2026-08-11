from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from radio_gs.field.observation_lifting_contract import canonical_observation_contract
from radio_gs.scripts.migrate_legacy_mpr_feature_bundle_cohort import migrate
from radio_gs.utils.immutable_artifacts import sha256_file


def _cache(space: str, *, radio_sha: str, responsibility_sha: str) -> dict:
    metadata = {
        "feature_space": space,
        "feature_output_bundle_sha256": "",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "observation_lifting_contract": {
            key: value for key, value in canonical_observation_contract().items()
            if key != "requires_full_observation_source_contract"
        },
        "shared_registration_responsibility": True,
        "registration_responsibility_cache_sha256": responsibility_sha,
        "capability_projection_before_mpr": space != "radio",
        "custom_adaptor_head": False,
        "official_adaptor_checkpoint_sha256": radio_sha if space != "radio" else "",
        "config": "config", "checkpoint": "checkpoint",
        "selected_frame_indices": [0, 20], "excluded_frame_ids": [40],
        "aggregation_mode": "raster_gaussian_top1",
        "registration_weight_mode": "alpha_depth",
        "raster_view_fusion": "contribution_mean", "raster_topk": 3,
        "depth_tolerance": 0.08, "relative_depth_tolerance": 0.02,
        "alpha_threshold": 0.02, "normalize_each_view": True,
    }
    return {
        "xyz": torch.arange(6).reshape(2, 3).float(),
        "features": torch.ones(2, 4 if space == "radio" else 2),
        "valid": torch.tensor([True, False]),
        "view_counts": torch.tensor([2, 0]),
        "reliability": torch.tensor([[1.0, 0.5, 0.2], [0.0, 0.0, 0.0]]),
        "metadata": metadata,
    }


def _args(tmp_path: Path) -> argparse.Namespace:
    radio = tmp_path / "radio-checkpoint.pt"
    responsibility = tmp_path / "responsibility.pt"
    torch.save({}, radio); torch.save({}, responsibility)
    radio_sha, responsibility_sha = sha256_file(radio), sha256_file(responsibility)
    inputs = {}
    for space, name in (("radio", "raw"), ("dino_v3", "dino"), ("sam3", "sam")):
        path = tmp_path / f"{name}.pt"
        torch.save(_cache(space, radio_sha=radio_sha, responsibility_sha=responsibility_sha), path)
        inputs[name] = (path, sha256_file(path))
    return argparse.Namespace(
        scene_id="scene",
        raw_cache=str(inputs["raw"][0]), expected_raw_cache_sha256=inputs["raw"][1],
        dino_cache=str(inputs["dino"][0]), expected_dino_cache_sha256=inputs["dino"][1],
        sam_cache=str(inputs["sam"][0]), expected_sam_cache_sha256=inputs["sam"][1],
        responsibility_cache=str(responsibility),
        expected_responsibility_cache_sha256=responsibility_sha,
        radio_checkpoint=str(radio), expected_radio_checkpoint_sha256=radio_sha,
        raw_output=str(tmp_path / "raw-formal.pt"),
        dino_output=str(tmp_path / "dino-formal.pt"),
        sam_output=str(tmp_path / "sam-formal.pt"),
        authority_output=str(tmp_path / "authority.json"),
    )


def test_migration_changes_metadata_only_and_binds_one_bundle(tmp_path: Path) -> None:
    args = _args(tmp_path)
    result = migrate(args)
    bundle = result["feature_output_bundle_sha256"]
    for original_name, output_name in (("raw_cache", "raw_output"), ("dino_cache", "dino_output"), ("sam_cache", "sam_output")):
        original = torch.load(getattr(args, original_name), map_location="cpu", weights_only=True)
        migrated = torch.load(getattr(args, output_name), map_location="cpu", weights_only=True)
        for key in ("xyz", "features", "valid", "view_counts", "reliability"):
            assert torch.equal(original[key], migrated[key])
        assert migrated["metadata"]["feature_output_bundle_sha256"] == bundle
    assert result["tensor_mutation"] is False


def test_migration_rejects_different_view_axis(tmp_path: Path) -> None:
    args = _args(tmp_path)
    payload = torch.load(args.sam_cache, map_location="cpu", weights_only=True)
    payload["metadata"]["selected_frame_indices"] = [0]
    torch.save(payload, args.sam_cache)
    args.expected_sam_cache_sha256 = sha256_file(args.sam_cache)
    with pytest.raises(ValueError, match="policy cohort differs"):
        migrate(args)
