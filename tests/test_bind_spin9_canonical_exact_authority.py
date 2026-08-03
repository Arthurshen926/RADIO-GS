import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from radio_gs.scripts.bind_spin9_canonical_exact_authority import (
    AuthorityError,
    EXPECTED_FIELD_STORAGE,
    EXPECTED_SHARD_CHANNELS,
    RADIO_CHECKPOINT,
    RADIO_CHECKPOINT_SHA256,
    validate_new_field_provenance,
    validate_dense_mpr_identity,
    validate_evaluation_recomputation,
    validate_prediction_embedding,
    validate_prompt_binding,
    validate_sharded_mpr_identity,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prompt_binding_requires_manifest_mask_and_rendered_prompt(tmp_path: Path):
    mask = tmp_path / "prompt.png"
    mask.write_bytes(b"prompt-mask")
    feature = tmp_path / "prompt.pt"
    feature.write_bytes(b"rendered-prompt-feature")
    mask_sha = _sha256(mask)
    feature_sha = _sha256(feature)
    scene = {
        "scene_id": "room",
        "prompt_frame_ids": ["image000"],
        "prompt": {
            "type": "reference_binary_mask",
            "frame_id": "image000",
            "mask_path": mask.name,
        },
        "frames": [
            {
                "frame_id": "image000",
                "ground_truth": mask.name,
                "ground_truth_sha256": mask_sha,
            }
        ],
    }
    prediction = {
        "prompt_frame_id": "image000",
        "prompt_feature_path": str(feature),
        "prompt_feature_sha256": feature_sha,
        "prompt": {
            "type": "reference_binary_mask",
            "paths": {"reference_binary_mask": str(mask)},
            "asset_sha256": {"reference_binary_mask": mask_sha},
        },
    }
    render = {"image000": {"role": "prompt", "feature_path": str(feature)}}
    record = validate_prompt_binding(
        scene="room",
        scene_manifest=scene,
        prediction_scene=prediction,
        render_by_frame=render,
        manifest_base=tmp_path,
    )
    assert record["mask"]["sha256"] == mask_sha
    assert record["feature"]["sha256"] == feature_sha

    prediction["prompt_feature_sha256"] = "0" * 64
    with pytest.raises(AuthorityError, match="prompt feature SHA"):
        validate_prompt_binding(
            scene="room",
            scene_manifest=scene,
            prediction_scene=prediction,
            render_by_frame=render,
            manifest_base=tmp_path,
        )


def test_prediction_embedding_is_explicit_checkpoint_bound(tmp_path: Path):
    checkpoint = tmp_path / "radio.pth"
    checkpoint.write_bytes(b"frozen-radio")
    checkpoint_sha = _sha256(checkpoint)
    method = {
        "embedding_space": {
            "type": "radio_sam3_feature_projection_adaptor_embedding",
            "adaptor_name": "sam3",
            "adaptor_kind": "feature_projection",
            "input_dim": 1280,
            "output_dim": 1024,
            "frozen": True,
            "radio_sam3_adaptor_applied": True,
            "official_sam_decoder": False,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
        }
    }
    record = validate_prediction_embedding(
        method,
        checkpoint=checkpoint,
        expected_sha256=checkpoint_sha,
    )
    assert record["provenance"] == "explicit_file_sha256"
    method["embedding_space"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(AuthorityError, match="checkpoint SHA"):
        validate_prediction_embedding(
            method,
            checkpoint=checkpoint,
            expected_sha256=checkpoint_sha,
        )


def _write_sharded_mpr(
    root: Path,
    name: str,
    *,
    feature_space: str,
    feature_dim: int,
    geometry: dict,
    bundle_sha: str,
    responsibility_sha: str,
    shard_channels=None,
) -> tuple[Path, str, dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    support_path = root / f"{name}.support.pt"
    support_path.write_bytes(f"support-{name}".encode())
    shard_channels = int(shard_channels or feature_dim)
    shard_rows = []
    storage_shards = []
    for start in range(0, feature_dim, shard_channels):
        stop = min(feature_dim, start + shard_channels)
        shard_path = root / f"{name}.channels_{start:05d}_{stop:05d}.bin"
        shard_path.write_bytes(b"\0" * (geometry["num_gaussians"] * (stop - start) * 2))
        shard_rows.append(
            {
                "relative_path": shard_path.name,
                "sha256": _sha256(shard_path),
                "channel_start": start,
                "channel_stop": stop,
                "dtype": "float16",
                "shape": [geometry["num_gaussians"], stop - start],
            }
        )
        storage_shards.append(
            {
                "path": str(shard_path.resolve()),
                "sha256": _sha256(shard_path),
                "channel_start": start,
                "channel_stop": stop,
            }
        )
    metadata = {
        "feature_space": feature_space,
        "feature_storage": "channel_sharded_fp16_row_major",
        "feature_output_bundle_sha256": bundle_sha,
        "observation_lifting_contract": {"name": "canonical-mpr-v1"},
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "registration_responsibility_cache_sha256": responsibility_sha,
        "shared_registration_responsibility": True,
    }
    if feature_space != "radio":
        metadata.update(
            {
                "capability_map_source": "project_raw",
                "capability_projection_before_mpr": True,
                "official_adaptor_checkpoint": str(RADIO_CHECKPOINT),
                "official_adaptor_checkpoint_sha256": RADIO_CHECKPOINT_SHA256,
                "official_adaptor_checkpoint_provenance": "runtime_cli_checkpoint_sha256",
            }
        )
    manifest = {
        "schema": "radio_gs.channel_sharded_mpr.v1",
        "schema_version": 1,
        "layout": "row_major_channel_shards",
        "feature_dtype": "float16",
        "feature_shape": [geometry["num_gaussians"], feature_dim],
        "support": {"relative_path": support_path.name, "sha256": _sha256(support_path)},
        "shards": shard_rows,
        "geometry_fingerprint": geometry,
        "metadata": metadata,
    }
    manifest_path = root / name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = _sha256(manifest_path)
    storage = {
        "storage": "radio_gs.channel_sharded_mpr.v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "support": {"path": str(support_path.resolve()), "sha256": _sha256(support_path)},
        "shards": storage_shards,
    }
    return manifest_path, digest, metadata, storage


def _write_dense_mpr(
    root: Path,
    name: str,
    *,
    feature_space: str,
    feature_dim: int,
    xyz: torch.Tensor,
    bundle_sha: str,
    responsibility_sha: str,
) -> tuple[Path, str, dict, dict, dict]:
    xyz = xyz.float().contiguous()
    geometry = {
        "num_gaussians": int(xyz.shape[0]),
        "xyz_sha256": hashlib.sha256(xyz.numpy().astype("<f4", copy=False).tobytes()).hexdigest(),
    }
    valid = torch.tensor([True, False], dtype=torch.bool)
    counts = torch.tensor([1, 0], dtype=torch.int64)
    reliability = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.float16)
    features = torch.zeros((2, feature_dim), dtype=torch.float16)
    features[0, 0] = 1
    metadata = {
        "feature_space": feature_space,
        "feature_output_bundle_sha256": bundle_sha,
        "observation_lifting_contract": {"name": "canonical-mpr-v1"},
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "registration_responsibility_cache_sha256": responsibility_sha,
        "shared_registration_responsibility": True,
        "num_declared_views": 1,
        "raster_reliability_mode": "legacy_valid",
    }
    if feature_space != "radio":
        metadata.update(
            {
                "capability_map_source": "project_raw",
                "capability_projection_before_mpr": True,
                "official_adaptor_checkpoint": str(RADIO_CHECKPOINT),
                "official_adaptor_checkpoint_sha256": RADIO_CHECKPOINT_SHA256,
                "official_adaptor_checkpoint_provenance": "runtime_cli_checkpoint_sha256",
            }
        )
    path = root / name
    torch.save(
        {
            "xyz": xyz,
            "features": features,
            "valid": valid,
            "view_counts": counts,
            "reliability": reliability,
            "geometry_fingerprint": geometry,
            "metadata": metadata,
        },
        path,
    )
    return path, _sha256(path), metadata, {"storage": "dense_torch_tensor"}, geometry


def test_sharded_mpr_checks_every_member_hash(tmp_path: Path):
    geometry = {"num_gaussians": 2, "xyz_sha256": "1" * 64}
    path, digest, _metadata, _storage = _write_sharded_mpr(
        tmp_path,
        "raw_radio.pt",
        feature_space="radio",
        feature_dim=1280,
        geometry=geometry,
        bundle_sha="2" * 64,
        responsibility_sha="3" * 64,
    )
    record = validate_sharded_mpr_identity(
        path,
        expected_sha256=digest,
        expected_feature_space="radio",
        expected_geometry=geometry,
        expected_feature_bundle_sha256="2" * 64,
    )
    assert record["feature_shape"] == [2, 1280]
    with pytest.raises(AuthorityError, match="channel width"):
        validate_sharded_mpr_identity(
            path,
            expected_sha256=digest,
            expected_feature_space="radio",
            expected_geometry=geometry,
            expected_feature_bundle_sha256="2" * 64,
            expected_shard_channels=512,
        )
    Path(record["shards"][0]["path"]).write_bytes(b"\1" * (2 * 1280 * 2))
    with pytest.raises(AuthorityError, match="shard SHA"):
        validate_sharded_mpr_identity(
            path,
            expected_sha256=digest,
            expected_feature_space="radio",
            expected_geometry=geometry,
            expected_feature_bundle_sha256="2" * 64,
        )


def test_dense_mpr_streams_finite_features_and_rejects_nan(tmp_path: Path):
    xyz = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.float32)
    path, digest, metadata, _storage, geometry = _write_dense_mpr(
        tmp_path,
        "raw_radio.pt",
        feature_space="radio",
        feature_dim=1280,
        xyz=xyz,
        bundle_sha="8" * 64,
        responsibility_sha="9" * 64,
    )
    record = validate_dense_mpr_identity(
        path,
        expected_sha256=digest,
        expected_feature_space="radio",
        expected_geometry=geometry,
        expected_feature_bundle_sha256="8" * 64,
    )
    assert record["storage"] == "dense_torch_tensor"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["features"][0, 0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(AuthorityError, match="non-finite"):
        validate_dense_mpr_identity(
            path,
            expected_sha256=_sha256(path),
            expected_feature_space="radio",
            expected_geometry=geometry,
            expected_feature_bundle_sha256=metadata["feature_output_bundle_sha256"],
        )


def test_new_field_binds_raw_and_capability_shards(tmp_path: Path):
    geometry = {"num_gaussians": 2, "xyz_sha256": "4" * 64}
    bundle, responsibility = "5" * 64, "6" * 64
    raw_path, raw_sha, raw_metadata, raw_storage = _write_sharded_mpr(
        tmp_path, "raw_radio.pt", feature_space="radio", feature_dim=1280,
        geometry=geometry, bundle_sha=bundle, responsibility_sha=responsibility,
        shard_channels=128,
    )
    dino_path, dino_sha, _dino_metadata, dino_storage = _write_sharded_mpr(
        tmp_path, "dino_v3.pt", feature_space="dino_v3", feature_dim=4096,
        geometry=geometry, bundle_sha=bundle, responsibility_sha=responsibility,
        shard_channels=512,
    )
    sam_path, sam_sha, _sam_metadata, sam_storage = _write_sharded_mpr(
        tmp_path, "sam3.pt", feature_space="sam3", feature_dim=1024,
        geometry=geometry, bundle_sha=bundle, responsibility_sha=responsibility,
        shard_channels=512,
    )

    def capability(path: Path, digest: str, storage: dict) -> dict:
        return {
            "path": str(path.resolve()),
            "sha256": digest,
            "projection_order": "official_adaptor_then_geometry_matched_mpr",
            "official_adaptor_checkpoint_sha256": RADIO_CHECKPOINT_SHA256,
            "uses_query_or_benchmark_supervision": False,
            **storage,
        }

    field_path = tmp_path / "canonical_d256_l128_capability_first.pth"
    training = {
        "observation_contract": "canonical-mpr-v1",
        "coefficient_dim": 256,
        "local_dim": 128,
        "primitive_fusion": True,
        "official_capability_loss": True,
        "epochs": 20,
        "min_epochs": 5,
        "target_cosine": 0.985,
        "seed": 0,
        "radio_checkpoint": str(RADIO_CHECKPOINT),
        "expected_radio_checkpoint_sha256": RADIO_CHECKPOINT_SHA256,
        "expected_feature_output_bundle_sha256": bundle,
        "expected_mpr_cache_sha256": raw_sha,
        "dino_mpr_cache": str(dino_path.resolve()),
        "expected_dino_v3_mpr_cache_sha256": dino_sha,
        "sam3_mpr_cache": str(sam_path.resolve()),
        "expected_sam3_mpr_cache_sha256": sam_sha,
    }
    training_sha = hashlib.sha256(
        json.dumps(training, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    torch.save(
        {
            "schema_version": 1,
            "architecture": {
                "feature_dim": 1280,
                "coefficient_dim": 256,
                "local_dim": 128,
                "use_fusion": True,
                "fusion_reliability": True,
            },
            "geometry_fingerprint": geometry,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "capability_target_mode": "official_adaptor_then_geometry_matched_mpr",
            "feature_signature": {"radio_checkpoint_sha256": RADIO_CHECKPOINT_SHA256},
            "training_config": training,
            "training_config_sha256": training_sha,
            "feature_output_bundle_sha256": bundle,
            "mpr_cache": str(raw_path.resolve()),
            "mpr_cache_sha256": raw_sha,
            "mpr_cache_metadata": raw_metadata,
            "mpr_cache_storage": raw_storage,
            "capability_mpr_targets": {
                "dino_v3": capability(dino_path, dino_sha, dino_storage),
                "sam3": capability(sam_path, sam_sha, sam_storage),
            },
        },
        field_path,
    )
    record = validate_new_field_provenance(
        field_path,
        scene="truck",
        expected_sha256=_sha256(field_path),
        expected_geometry=geometry,
    )
    assert record["raw_mpr"]["sha256"] == raw_sha
    assert set(record["capability_mpr"]) == {"dino_v3", "sam3"}
    assert record["shard_channel_contract"] == {
        "radio": 128,
        "dino_v3": 512,
        "sam3": 512,
    }
    with pytest.raises(AuthorityError, match="raw MPR storage contract"):
        validate_new_field_provenance(
            field_path,
            scene="room",
            expected_sha256=_sha256(field_path),
            expected_geometry=geometry,
        )


def test_room_field_requires_dense_raw_and_sam_but_sharded_dino(tmp_path: Path):
    xyz = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.float32)
    bundle, responsibility = "a" * 64, "b" * 64
    raw_path, raw_sha, raw_metadata, raw_storage, geometry = _write_dense_mpr(
        tmp_path, "raw_radio.pt", feature_space="radio", feature_dim=1280,
        xyz=xyz, bundle_sha=bundle, responsibility_sha=responsibility,
    )
    dino_path, dino_sha, _dino_metadata, dino_storage = _write_sharded_mpr(
        tmp_path, "dino_v3.pt", feature_space="dino_v3", feature_dim=4096,
        geometry=geometry, bundle_sha=bundle, responsibility_sha=responsibility,
        shard_channels=256,
    )
    sam_path, sam_sha, _sam_metadata, sam_storage, _ = _write_dense_mpr(
        tmp_path, "sam3.pt", feature_space="sam3", feature_dim=1024,
        xyz=xyz, bundle_sha=bundle, responsibility_sha=responsibility,
    )

    def capability(path: Path, digest: str, storage: dict) -> dict:
        return {
            "path": str(path.resolve()),
            "sha256": digest,
            "projection_order": "official_adaptor_then_geometry_matched_mpr",
            "official_adaptor_checkpoint_sha256": RADIO_CHECKPOINT_SHA256,
            "uses_query_or_benchmark_supervision": False,
            **storage,
        }

    training = {
        "observation_contract": "canonical-mpr-v1",
        "coefficient_dim": 256,
        "local_dim": 128,
        "primitive_fusion": True,
        "official_capability_loss": True,
        "epochs": 20,
        "min_epochs": 5,
        "target_cosine": 0.985,
        "seed": 0,
        "radio_checkpoint": str(RADIO_CHECKPOINT),
        "expected_radio_checkpoint_sha256": RADIO_CHECKPOINT_SHA256,
        "expected_feature_output_bundle_sha256": bundle,
        "expected_mpr_cache_sha256": raw_sha,
        "dino_mpr_cache": str(dino_path.resolve()),
        "expected_dino_v3_mpr_cache_sha256": dino_sha,
        "sam3_mpr_cache": str(sam_path.resolve()),
        "expected_sam3_mpr_cache_sha256": sam_sha,
    }
    training_sha = hashlib.sha256(
        json.dumps(training, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    field_path = tmp_path / "canonical_d256_l128_capability_first.pth"
    torch.save(
        {
            "schema_version": 1,
            "architecture": {
                "feature_dim": 1280,
                "coefficient_dim": 256,
                "local_dim": 128,
                "use_fusion": True,
                "fusion_reliability": True,
            },
            "geometry_fingerprint": geometry,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "capability_target_mode": "official_adaptor_then_geometry_matched_mpr",
            "feature_signature": {"radio_checkpoint_sha256": RADIO_CHECKPOINT_SHA256},
            "training_config": training,
            "training_config_sha256": training_sha,
            "feature_output_bundle_sha256": bundle,
            "mpr_cache": str(raw_path.resolve()),
            "mpr_cache_sha256": raw_sha,
            "mpr_cache_metadata": raw_metadata,
            "mpr_cache_storage": raw_storage,
            "capability_mpr_targets": {
                "dino_v3": capability(dino_path, dino_sha, dino_storage),
                "sam3": capability(sam_path, sam_sha, sam_storage),
            },
        },
        field_path,
    )
    record = validate_new_field_provenance(
        field_path,
        scene="room",
        expected_sha256=_sha256(field_path),
        expected_geometry=geometry,
    )
    assert record["storage_contract"] == {
        "radio": "dense_torch_tensor",
        "dino_v3": "radio_gs.channel_sharded_mpr.v1",
        "sam3": "dense_torch_tensor",
    }
    assert record["shard_channel_contract"] == {"dino_v3": 256}


def test_horns_storage_and_shard_contract_is_fully_sharded_512():
    assert EXPECTED_FIELD_STORAGE["horns"] == {
        "radio": "radio_gs.channel_sharded_mpr.v1",
        "dino_v3": "radio_gs.channel_sharded_mpr.v1",
        "sam3": "radio_gs.channel_sharded_mpr.v1",
    }
    assert EXPECTED_SHARD_CHANNELS["horns"] == {
        "radio": 512,
        "dino_v3": 512,
        "sam3": 512,
    }


def _evaluation_report(ground_truth: Path, prediction: Path) -> dict:
    return {
        "schema_version": 1,
        "protocol_hash": "frozen-protocol",
        "dataset": {
            "foreground_iou": 0.75,
            "pixel_accuracy": 0.875,
            "num_scenes": 1,
            "num_frames": 2,
        },
        "scenes": [
            {
                "scene_id": "room",
                "foreground_iou": 0.75,
                "pixel_accuracy": 0.875,
                "num_frames": 2,
                "frames": [
                    {
                        "frame_id": "frame001",
                        "ground_truth": str(ground_truth),
                        "prediction": str(prediction),
                        "foreground_iou": 0.7,
                        "pixel_accuracy": 0.85,
                    },
                    {
                        "frame_id": "frame002",
                        "ground_truth": str(ground_truth),
                        "prediction": str(prediction),
                        "foreground_iou": 0.8,
                        "pixel_accuracy": 0.9,
                    },
                ],
            }
        ],
        "thresholds": {"policy": "fixed", "value": 0.0},
    }


def test_evaluation_recomputation_accepts_equivalent_workspace_symlinks(tmp_path: Path):
    mounted = tmp_path / "mnt" / "pool"
    mounted.mkdir(parents=True)
    ground_truth = mounted / "mask.png"
    prediction = mounted / "score.npy"
    ground_truth.write_bytes(b"mask")
    prediction.write_bytes(b"score")

    root_alias = tmp_path / "root" / "output"
    root_alias.parent.mkdir(parents=True)
    root_alias.symlink_to(mounted, target_is_directory=True)
    stored = _evaluation_report(root_alias / ground_truth.name, root_alias / prediction.name)
    recomputed = _evaluation_report(ground_truth, prediction)

    validate_evaluation_recomputation(stored, recomputed)


@pytest.mark.parametrize("path_key", ["ground_truth", "prediction"])
def test_evaluation_recomputation_rejects_different_frame_path(
    tmp_path: Path,
    path_key: str,
):
    ground_truth = tmp_path / "mask.png"
    prediction = tmp_path / "score.npy"
    other_ground_truth = tmp_path / "other-mask.png"
    other_prediction = tmp_path / "other-score.npy"
    for path in (ground_truth, prediction, other_ground_truth, other_prediction):
        path.write_bytes(path.name.encode())
    stored = _evaluation_report(ground_truth, prediction)
    recomputed = _evaluation_report(
        other_ground_truth if path_key == "ground_truth" else ground_truth,
        other_prediction if path_key == "prediction" else prediction,
    )

    with pytest.raises(AuthorityError, match="fresh frozen-protocol recomputation"):
        validate_evaluation_recomputation(stored, recomputed)


@pytest.mark.parametrize("mutation", ["dataset_metric", "frame_metric", "frame_order", "extra_field"])
def test_evaluation_recomputation_rejects_any_non_path_difference(
    tmp_path: Path,
    mutation: str,
):
    ground_truth = tmp_path / "mask.png"
    prediction = tmp_path / "score.npy"
    ground_truth.write_bytes(b"mask")
    prediction.write_bytes(b"score")
    stored = _evaluation_report(ground_truth, prediction)
    recomputed = copy.deepcopy(stored)
    if mutation == "dataset_metric":
        recomputed["dataset"]["foreground_iou"] = 0.74
    elif mutation == "frame_metric":
        recomputed["scenes"][0]["frames"][0]["foreground_iou"] = 0.69
    elif mutation == "frame_order":
        recomputed["scenes"][0]["frames"].reverse()
    else:
        recomputed["scenes"][0]["frames"][0]["unexpected"] = True

    with pytest.raises(AuthorityError, match="fresh frozen-protocol recomputation"):
        validate_evaluation_recomputation(stored, recomputed)
