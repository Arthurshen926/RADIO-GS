#!/usr/bin/env python3
"""Bind the canonical-mpr-v3 SPIn local9 exact diagnostic result.

This authority deliberately calls the cohort ``local9``: the official Fork
RGB source is absent, so the result cannot become an official ten-scene row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np
import torch
import yaml

from radio_gs.evaluation.promptable_segmentation import evaluate_manifest
from radio_gs.utils.immutable_artifacts import load_json_object, load_torch_mapping


SCENES = (
    "orchids", "leaves", "fern", "room", "horns",
    "fortress", "pinecone", "truck", "lego",
)
MANIFEST_SHA256 = "e3171f135ae9fa50f5803672cbab4d6ae61a532d44d21721fb6a680294e2aede"
PROTOCOL_HASH = "d8a87284ddc2fde946a5d9de83aec190487e61c72259dc62656be603c2af6752"
GENERAL_FREEZE_SHA256 = "af91f0861d3a15354063579e78f64898801c41f2543d1cf9b352a0a123820916"
PROMPTABLE_REGISTRY_SHA256 = "5d1a044513ce2c5d3850dbd95f4a3505c566ae2649e5d89b7e926daf4a568c54"
RADIO_CHECKPOINT = Path("/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
RADIO_CHECKPOINT_SHA256 = "bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9"
NEW_FIELD_SCENES = frozenset({"room", "horns", "truck"})
_SHARDED_MPR_SCHEMA = "radio_gs.channel_sharded_mpr.v1"
EXPECTED_FIELD_STORAGE = {
    "room": {"radio": "dense_torch_tensor", "dino_v3": _SHARDED_MPR_SCHEMA, "sam3": "dense_torch_tensor"},
    "horns": {"radio": _SHARDED_MPR_SCHEMA, "dino_v3": _SHARDED_MPR_SCHEMA, "sam3": _SHARDED_MPR_SCHEMA},
    "truck": {"radio": _SHARDED_MPR_SCHEMA, "dino_v3": _SHARDED_MPR_SCHEMA, "sam3": _SHARDED_MPR_SCHEMA},
}
EXPECTED_SHARD_CHANNELS = {
    "room": {"dino_v3": 256},
    "horns": {"radio": 512, "dino_v3": 512, "sam3": 512},
    "truck": {"radio": 128, "dino_v3": 512, "sam3": 512},
}


class AuthorityError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuthorityError(f"{label} must be a mapping")
    return value


def _json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _same_path(left: str | Path, right: Path) -> bool:
    return Path(left).expanduser().resolve() == right.resolve()


def _evaluation_with_resolved_frame_paths(
    report: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    """Resolve only evaluator-emitted per-frame artifact paths.

    A result may be produced through a workspace symlink (for example
    ``/root/RADIO-GS``) and recomputed through its mounted target.  Those two
    spellings name the same artifacts, but plain mapping equality rejects
    them.  Keep every other value, field, and list position untouched so the
    final comparison remains the original whole-report equality check.
    """

    normalized = copy.deepcopy(report)
    scenes = normalized.get("scenes")
    _require(isinstance(scenes, list), f"{label} scenes must be a list")
    for scene_index, raw_scene in enumerate(scenes):
        scene = _mapping(raw_scene, f"{label} scene {scene_index}")
        frames = scene.get("frames")
        _require(isinstance(frames, list), f"{label} scene {scene_index} frames must be a list")
        for frame_index, raw_frame in enumerate(frames):
            frame = _mapping(raw_frame, f"{label} scene {scene_index} frame {frame_index}")
            for key in ("ground_truth", "prediction"):
                value = frame.get(key)
                _require(
                    isinstance(value, str) and bool(value),
                    f"{label} scene {scene_index} frame {frame_index} {key} must be a non-empty path",
                )
                frame[key] = str(Path(value).expanduser().resolve())
    return normalized


def validate_evaluation_recomputation(
    evaluation: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    """Require exact reports modulo equivalent per-frame path aliases."""

    stored = _evaluation_with_resolved_frame_paths(evaluation, label="stored evaluation")
    fresh = _evaluation_with_resolved_frame_paths(recomputed, label="recomputed evaluation")
    _require(
        stored == fresh,
        "evaluation does not equal a fresh frozen-protocol recomputation",
    )


def _file_record(path: Path) -> dict[str, str]:
    _require(path.is_file() and not path.is_symlink(), f"artifact is not a regular file: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _frame_map(scene: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = scene.get("frames")
    _require(isinstance(rows, list), f"{scene.get('scene_id')}: manifest frames are absent")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, f"{scene.get('scene_id')}: frame")
        frame_id = str(row.get("frame_id", ""))
        _require(frame_id and frame_id not in result, f"{scene.get('scene_id')}: duplicate/empty frame id")
        result[frame_id] = row
    return result


def validate_prediction_embedding(
    method: Mapping[str, Any],
    *,
    checkpoint: Path = RADIO_CHECKPOINT,
    expected_sha256: str = RADIO_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    """Bind prediction-time projection to the one frozen C-RADIO release."""

    embedding = _mapping(method.get("embedding_space"), "prediction embedding space")
    _require(embedding.get("type") == "radio_sam3_feature_projection_adaptor_embedding", "prediction embedding type differs")
    _require(embedding.get("adaptor_name") == "sam3", "prediction adaptor name differs")
    _require(embedding.get("adaptor_kind") == "feature_projection", "prediction adaptor kind differs")
    _require(embedding.get("input_dim") == 1280 and embedding.get("output_dim") == 1024, "prediction adaptor dimensions differ")
    _require(embedding.get("frozen") is True, "prediction adaptor is not frozen")
    _require(embedding.get("radio_sam3_adaptor_applied") is True, "prediction adaptor was not applied")
    _require(embedding.get("official_sam_decoder") is False, "prediction incorrectly claims an official SAM decoder")
    checkpoint = checkpoint.expanduser().resolve()
    _require(_same_path(str(embedding.get("checkpoint", "")), checkpoint), "prediction adaptor checkpoint path differs")
    actual_sha256 = _sha256(checkpoint)
    _require(actual_sha256 == expected_sha256, "frozen RADIO checkpoint drifted")
    _require(embedding.get("checkpoint_sha256") == expected_sha256, "prediction adaptor checkpoint SHA differs")
    return {
        "path": str(checkpoint),
        "sha256": expected_sha256,
        "provenance": "explicit_file_sha256",
        "adaptor": "sam3_feature_projection_1280_to_1024",
    }


def validate_prompt_binding(
    *,
    scene: str,
    scene_manifest: Mapping[str, Any],
    prediction_scene: Mapping[str, Any],
    render_by_frame: Mapping[str, Mapping[str, Any]],
    manifest_base: Path,
) -> dict[str, Any]:
    """Bind the sole allowed prompt mask and prompt feature to their render row."""

    prompt_ids = scene_manifest.get("prompt_frame_ids")
    _require(isinstance(prompt_ids, list) and len(prompt_ids) == 1, f"{scene}: prompt cohort differs")
    prompt_id = str(prompt_ids[0])
    _require(prediction_scene.get("prompt_frame_id") == prompt_id, f"{scene}: prediction prompt frame differs")
    prompt_spec = _mapping(scene_manifest.get("prompt"), f"{scene}: manifest prompt")
    _require(prompt_spec.get("type") == "reference_binary_mask", f"{scene}: prompt type differs")
    _require(prompt_spec.get("frame_id") == prompt_id, f"{scene}: prompt frame declaration differs")
    frames = _frame_map(scene_manifest)
    _require(prompt_id in frames, f"{scene}: prompt frame record is absent")
    prompt_frame = frames[prompt_id]
    mask = _resolve(manifest_base, str(prompt_spec.get("mask_path", "")))
    frame_mask = _resolve(manifest_base, str(prompt_frame.get("ground_truth", "")))
    _require(mask == frame_mask, f"{scene}: prompt mask does not equal the reference annotation")
    mask_sha256 = _sha256(mask)
    _require(prompt_frame.get("ground_truth_sha256") == mask_sha256, f"{scene}: prompt annotation SHA differs")

    producer_prompt = _mapping(prediction_scene.get("prompt"), f"{scene}: prediction prompt")
    _require(producer_prompt.get("type") == "reference_binary_mask", f"{scene}: producer prompt type differs")
    prompt_paths = _mapping(producer_prompt.get("paths"), f"{scene}: producer prompt paths")
    prompt_hashes = _mapping(producer_prompt.get("asset_sha256"), f"{scene}: producer prompt hashes")
    _require(_same_path(str(prompt_paths.get("reference_binary_mask", "")), mask), f"{scene}: producer used another prompt mask")
    _require(prompt_hashes.get("reference_binary_mask") == mask_sha256, f"{scene}: producer prompt-mask SHA differs")

    _require(prompt_id in render_by_frame, f"{scene}: prompt render row is absent")
    render_row = render_by_frame[prompt_id]
    _require(render_row.get("role") == "prompt", f"{scene}: prompt render role differs")
    prompt_feature = Path(str(prediction_scene.get("prompt_feature_path", ""))).expanduser().resolve()
    _require(prompt_feature == Path(str(render_row.get("feature_path", ""))).expanduser().resolve(), f"{scene}: producer prompt feature is not the bound render")
    prompt_feature_sha256 = _sha256(prompt_feature)
    _require(prediction_scene.get("prompt_feature_sha256") == prompt_feature_sha256, f"{scene}: prompt feature SHA differs")
    return {
        "frame_id": prompt_id,
        "mask": {"path": str(mask), "sha256": mask_sha256},
        "feature": {"path": str(prompt_feature), "sha256": prompt_feature_sha256},
    }


def validate_sharded_mpr_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_feature_space: str,
    expected_geometry: Mapping[str, Any],
    expected_feature_bundle_sha256: str,
    expected_responsibility_sha256: str | None = None,
    expected_shard_channels: int | None = None,
) -> dict[str, Any]:
    """Validate one channel-sharded MPR without materializing feature tensors."""

    manifest, manifest_sha256, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label=f"{expected_feature_space} sharded MPR manifest",
    )
    _require(manifest.get("schema") == _SHARDED_MPR_SCHEMA and manifest.get("schema_version") == 1, "sharded MPR schema differs")
    _require(manifest.get("layout") == "row_major_channel_shards" and manifest.get("feature_dtype") == "float16", "sharded MPR layout/dtype differs")
    shape = manifest.get("feature_shape")
    expected_dim = {"radio": 1280, "dino_v3": 4096, "sam3": 1024}[expected_feature_space]
    _require(isinstance(shape, list) and len(shape) == 2 and int(shape[0]) > 0 and int(shape[1]) == expected_dim, "sharded MPR feature shape differs")
    geometry = _mapping(manifest.get("geometry_fingerprint"), "sharded MPR geometry")
    _require(dict(geometry) == dict(expected_geometry), "sharded MPR geometry identity differs")
    metadata = _mapping(manifest.get("metadata"), "sharded MPR metadata")
    _require(metadata.get("feature_space") == expected_feature_space, "sharded MPR feature space differs")
    _require(metadata.get("feature_storage") == "channel_sharded_fp16_row_major", "sharded MPR storage declaration differs")
    _require(metadata.get("feature_output_bundle_sha256") == expected_feature_bundle_sha256, "sharded MPR feature bundle differs")
    lifting = _mapping(metadata.get("observation_lifting_contract"), "sharded MPR observation contract")
    _require(lifting.get("name") == "canonical-mpr-v1", "sharded MPR observation contract differs")
    for key in ("benchmark_masks_opened", "benchmark_images_opened", "text_queries_opened"):
        _require(metadata.get(key) is False, f"sharded MPR safety declaration differs: {key}")
    responsibility_sha256 = str(metadata.get("registration_responsibility_cache_sha256", ""))
    _require(len(responsibility_sha256) == 64 and metadata.get("shared_registration_responsibility") is True, "sharded MPR responsibility provenance differs")
    if expected_responsibility_sha256 is not None:
        _require(responsibility_sha256 == expected_responsibility_sha256, "sharded MPR responsibility identity differs")
    if expected_feature_space != "radio":
        _require(metadata.get("capability_map_source") == "project_raw", "capability MPR did not project raw RADIO")
        _require(metadata.get("capability_projection_before_mpr") is True, "capability projection order differs")
        _require(_same_path(str(metadata.get("official_adaptor_checkpoint", "")), RADIO_CHECKPOINT), "capability MPR RADIO checkpoint path differs")
        _require(metadata.get("official_adaptor_checkpoint_sha256") == RADIO_CHECKPOINT_SHA256, "capability MPR RADIO checkpoint differs")
        _require(metadata.get("official_adaptor_checkpoint_provenance") == "runtime_cli_checkpoint_sha256", "capability MPR checkpoint provenance differs")

    root = source.parent
    support = _mapping(manifest.get("support"), "sharded MPR support")
    support_relative = Path(str(support.get("relative_path", "")))
    _require(not support_relative.is_absolute() and ".." not in support_relative.parts and len(support_relative.parts) == 1, "sharded MPR support path is unsafe")
    support_path = root / support_relative
    _require(support_path.is_file() and not support_path.is_symlink(), "sharded MPR support is not a regular file")
    support_sha256 = _sha256(support_path)
    _require(support_sha256 == support.get("sha256"), "sharded MPR support SHA differs")

    shard_rows = manifest.get("shards")
    _require(isinstance(shard_rows, list) and shard_rows, "sharded MPR has no shards")
    expected_start = 0
    shard_records: list[dict[str, Any]] = []
    for index, raw in enumerate(shard_rows):
        row = _mapping(raw, f"sharded MPR shard {index}")
        start, stop = int(row.get("channel_start", -1)), int(row.get("channel_stop", -1))
        _require(start == expected_start and start < stop <= expected_dim, "sharded MPR channel coverage differs")
        if expected_shard_channels is not None:
            _require(expected_shard_channels > 0, "expected shard width must be positive")
            _require(stop - start == min(expected_shard_channels, expected_dim - start), "sharded MPR channel width differs from frozen contract")
        _require(row.get("dtype") == "float16" and row.get("shape") == [int(shape[0]), stop - start], "sharded MPR shard shape/dtype differs")
        relative = Path(str(row.get("relative_path", "")))
        _require(not relative.is_absolute() and ".." not in relative.parts and len(relative.parts) == 1, "sharded MPR shard path is unsafe")
        shard = root / relative
        _require(shard.is_file() and not shard.is_symlink(), "sharded MPR shard is not a regular file")
        _require(shard.stat().st_size == int(shape[0]) * (stop - start) * 2, "sharded MPR shard byte count differs")
        shard_sha256 = _sha256(shard)
        _require(shard_sha256 == row.get("sha256"), "sharded MPR shard SHA differs")
        shard_records.append({"path": str(shard.resolve()), "sha256": shard_sha256, "channel_start": start, "channel_stop": stop})
        expected_start = stop
    _require(expected_start == expected_dim, "sharded MPR channel coverage is incomplete")
    return {
        "storage": _SHARDED_MPR_SCHEMA,
        "path": str(source),
        "sha256": manifest_sha256,
        "feature_space": expected_feature_space,
        "feature_shape": [int(shape[0]), expected_dim],
        "shard_channels": expected_shard_channels,
        "geometry_fingerprint": dict(geometry),
        "feature_output_bundle_sha256": expected_feature_bundle_sha256,
        "responsibility_sha256": responsibility_sha256,
        "metadata": dict(metadata),
        "support": {"path": str(support_path.resolve()), "sha256": support_sha256},
        "shards": shard_records,
    }


def _path_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    info = os.stat(path, follow_symlinks=False)
    _require(path.is_file() and not path.is_symlink(), f"artifact is not a regular file: {path}")
    return (int(info.st_dev), int(info.st_ino), int(info.st_size), int(info.st_mtime_ns), int(info.st_ctime_ns))


def _unique_zip_storage(
    archive: zipfile.ZipFile,
    *,
    expected_bytes: int,
    label: str,
) -> zipfile.ZipInfo:
    matches = [
        row
        for row in archive.infolist()
        if "/data/" in row.filename and int(row.file_size) == int(expected_bytes)
    ]
    _require(len(matches) == 1, f"dense MPR {label} storage is absent or ambiguous")
    _require(matches[0].compress_type == zipfile.ZIP_STORED, f"dense MPR {label} storage is unexpectedly compressed")
    return matches[0]


def validate_dense_mpr_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_feature_space: str,
    expected_geometry: Mapping[str, Any],
    expected_feature_bundle_sha256: str,
    expected_responsibility_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a dense MPR in bounded memory by streaming its ZIP storages."""

    source = path.expanduser().resolve()
    before = _path_fingerprint(source)
    actual_sha256 = _sha256(source)
    _require(actual_sha256 == expected_sha256, "dense MPR file SHA differs")
    payload = torch.load(source, map_location="meta", weights_only=True)
    _require(isinstance(payload, Mapping), "dense MPR payload is not a mapping")
    tensors: dict[str, torch.Tensor] = {}
    for key in ("xyz", "features", "valid", "view_counts", "reliability"):
        value = payload.get(key)
        _require(torch.is_tensor(value), f"dense MPR {key} is not a tensor")
        tensors[key] = value
    xyz, features = tensors["xyz"], tensors["features"]
    valid, counts, reliability = tensors["valid"], tensors["view_counts"], tensors["reliability"]
    expected_dim = {"radio": 1280, "dino_v3": 4096, "sam3": 1024}[expected_feature_space]
    _require(xyz.dtype == torch.float32 and xyz.ndim == 2 and xyz.shape[1] == 3, "dense MPR xyz spec differs")
    num_rows = int(xyz.shape[0])
    _require(features.dtype == torch.float16 and tuple(features.shape) == (num_rows, expected_dim), "dense MPR feature spec differs")
    _require(valid.dtype == torch.bool and tuple(valid.shape) == (num_rows,), "dense MPR valid spec differs")
    _require(counts.dtype in {torch.int16, torch.int32, torch.int64, torch.uint8} and tuple(counts.shape) == (num_rows,), "dense MPR view-count spec differs")
    _require(reliability.dtype in {torch.float16, torch.float32} and tuple(reliability.shape) == (num_rows, 3), "dense MPR reliability spec differs")
    for key, value in tensors.items():
        _require(value.is_contiguous() and value.storage_offset() == 0, f"dense MPR {key} is not contiguous")

    geometry = _mapping(payload.get("geometry_fingerprint"), "dense MPR geometry")
    _require(dict(geometry) == dict(expected_geometry), "dense MPR geometry identity differs")
    metadata = _mapping(payload.get("metadata"), "dense MPR metadata")
    _require(metadata.get("feature_space") == expected_feature_space, "dense MPR feature space differs")
    _require(metadata.get("feature_output_bundle_sha256") == expected_feature_bundle_sha256, "dense MPR feature bundle differs")
    lifting = _mapping(metadata.get("observation_lifting_contract"), "dense MPR observation contract")
    _require(lifting.get("name") == "canonical-mpr-v1", "dense MPR observation contract differs")
    for key in ("benchmark_masks_opened", "benchmark_images_opened", "text_queries_opened"):
        _require(metadata.get(key) is False, f"dense MPR safety declaration differs: {key}")
    responsibility_sha256 = str(metadata.get("registration_responsibility_cache_sha256", ""))
    _require(len(responsibility_sha256) == 64 and metadata.get("shared_registration_responsibility") is True, "dense MPR responsibility provenance differs")
    if expected_responsibility_sha256 is not None:
        _require(responsibility_sha256 == expected_responsibility_sha256, "dense MPR responsibility identity differs")
    if expected_feature_space != "radio":
        _require(metadata.get("capability_map_source") == "project_raw", "dense capability MPR did not project raw RADIO")
        _require(metadata.get("capability_projection_before_mpr") is True, "dense capability projection order differs")
        _require(_same_path(str(metadata.get("official_adaptor_checkpoint", "")), RADIO_CHECKPOINT), "dense capability RADIO checkpoint path differs")
        _require(metadata.get("official_adaptor_checkpoint_sha256") == RADIO_CHECKPOINT_SHA256, "dense capability RADIO checkpoint differs")
        _require(metadata.get("official_adaptor_checkpoint_provenance") == "runtime_cli_checkpoint_sha256", "dense capability checkpoint provenance differs")
    _require(str(metadata.get("raster_reliability_mode", "legacy_valid")) == "legacy_valid", "dense MPR reliability policy differs")

    numpy_dtype = {
        torch.float16: np.dtype("<f2"),
        torch.float32: np.dtype("<f4"),
        torch.bool: np.dtype("?"),
        torch.uint8: np.dtype("u1"),
        torch.int16: np.dtype("<i2"),
        torch.int32: np.dtype("<i4"),
        torch.int64: np.dtype("<i8"),
    }
    with zipfile.ZipFile(source) as archive:
        members = {
            key: _unique_zip_storage(
                archive,
                expected_bytes=int(value.numel()) * int(value.element_size()),
                label=key,
            )
            for key, value in tensors.items()
        }
        xyz_bytes = archive.read(members["xyz"])
        _require(hashlib.sha256(xyz_bytes).hexdigest() == geometry.get("xyz_sha256"), "dense MPR xyz bytes differ from geometry fingerprint")
        valid_values = np.frombuffer(archive.read(members["valid"]), dtype=numpy_dtype[valid.dtype]).astype(bool, copy=False)
        count_values = np.frombuffer(archive.read(members["view_counts"]), dtype=numpy_dtype[counts.dtype]).astype(np.int64, copy=False)
        reliability_values = np.frombuffer(archive.read(members["reliability"]), dtype=numpy_dtype[reliability.dtype]).reshape(num_rows, 3).astype(np.float32)
        _require(np.array_equal(valid_values, count_values > 0), "dense MPR valid/count identity differs")
        num_views = int(metadata.get("num_declared_views", 0))
        _require(num_views > 0 and bool((count_values >= 0).all()) and bool((count_values <= num_views).all()), "dense MPR view counts differ")
        _require(bool(np.isfinite(reliability_values).all()), "dense MPR reliability is non-finite")
        _require(bool((reliability_values >= 0).all()) and bool((reliability_values <= 1.001).all()), "dense MPR reliability range differs")
        tolerance = 2e-3 if reliability.dtype == torch.float16 else 1e-6
        _require(np.allclose(reliability_values[:, 0], count_values / float(num_views), atol=tolerance, rtol=0), "dense MPR coverage reliability differs")
        _require(np.allclose(reliability_values[:, 1], valid_values, atol=tolerance, rtol=0), "dense MPR agreement reliability differs")
        _require(np.allclose(reliability_values[:, 2], valid_values, atol=tolerance, rtol=0), "dense MPR support reliability differs")

        rows_per_chunk = max(1, (32 * 1024 * 1024) // (expected_dim * 2))
        with archive.open(members["features"], "r") as handle:
            for row_start in range(0, num_rows, rows_per_chunk):
                row_stop = min(num_rows, row_start + rows_per_chunk)
                expected_bytes = (row_stop - row_start) * expected_dim * 2
                block = handle.read(expected_bytes)
                _require(len(block) == expected_bytes, "dense MPR feature storage ended early")
                values = np.frombuffer(block, dtype="<f2").reshape(row_stop - row_start, expected_dim)
                _require(bool(np.isfinite(values).all()), "dense MPR features are non-finite")
                unsupported = ~valid_values[row_start:row_stop]
                _require(not bool(unsupported.any()) or not bool(np.any(values[unsupported] != 0)), "dense MPR unsupported features are nonzero")
            _require(handle.read(1) == b"", "dense MPR feature storage has trailing bytes")
    _require(_path_fingerprint(source) == before, "dense MPR changed during validation")
    return {
        "storage": "dense_torch_tensor",
        "path": str(source),
        "sha256": actual_sha256,
        "feature_space": expected_feature_space,
        "feature_shape": [num_rows, expected_dim],
        "geometry_fingerprint": dict(geometry),
        "feature_output_bundle_sha256": expected_feature_bundle_sha256,
        "responsibility_sha256": responsibility_sha256,
        "metadata": dict(metadata),
        "finite_validation": "streamed_zip_storage_float16_chunks",
    }


def _validate_field_storage_provenance(
    value: Any,
    record: Mapping[str, Any],
    *,
    label: str,
) -> None:
    provenance = _mapping(value, f"{label} field storage provenance")
    expected_storage = record.get("storage")
    _require(provenance.get("storage") == expected_storage, f"{label} field storage kind differs")
    if expected_storage == "dense_torch_tensor":
        for forbidden in ("manifest_path", "manifest_sha256", "support", "shards"):
            _require(forbidden not in provenance, f"{label} dense field provenance contains sharded member {forbidden}")
        return
    _require(expected_storage == _SHARDED_MPR_SCHEMA, f"{label} expected storage kind is unsupported")
    _require(_same_path(str(provenance.get("manifest_path", "")), Path(str(record["path"]))), f"{label} field manifest path differs")
    _require(provenance.get("manifest_sha256") == record["sha256"], f"{label} field manifest SHA differs")
    support = _mapping(provenance.get("support"), f"{label} field support provenance")
    _require(_same_path(str(support.get("path", "")), Path(str(record["support"]["path"]))), f"{label} field support path differs")
    _require(support.get("sha256") == record["support"]["sha256"], f"{label} field support SHA differs")
    shards = provenance.get("shards")
    _require(isinstance(shards, list) and len(shards) == len(record["shards"]), f"{label} field shard count differs")
    for index, (actual, expected) in enumerate(zip(shards, record["shards"])):
        row = _mapping(actual, f"{label} field shard {index}")
        _require(_same_path(str(row.get("path", "")), Path(str(expected["path"]))), f"{label} field shard path differs")
        for key in ("sha256", "channel_start", "channel_stop"):
            _require(row.get(key) == expected[key], f"{label} field shard {key} differs")


def validate_new_field_provenance(
    field: Path,
    *,
    scene: str,
    expected_sha256: str,
    expected_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Deep-bind a newly built missing-scene field using meta tensors only."""

    payload, field_sha256, source = load_torch_mapping(
        field,
        expected_sha256=expected_sha256,
        map_location="meta",
        label="SPIn missing-scene canonical field",
    )
    _require(scene in EXPECTED_FIELD_STORAGE, "canonical field scene has no frozen storage contract")
    storage_contract = EXPECTED_FIELD_STORAGE[scene]
    _require(payload.get("schema_version") == 1, "canonical field schema differs")
    architecture = _mapping(payload.get("architecture"), "canonical field architecture")
    _require(architecture.get("feature_dim") == 1280, "canonical field RADIO dimension differs")
    _require(architecture.get("coefficient_dim") == 256 and architecture.get("local_dim") == 128, "canonical field compact dimensions differ")
    _require(architecture.get("use_fusion") is True and architecture.get("fusion_reliability") is True, "canonical primitive fusion differs")
    geometry = _mapping(payload.get("geometry_fingerprint"), "canonical field geometry")
    _require(dict(geometry) == dict(expected_geometry), "canonical field geometry provenance differs")
    _require(payload.get("benchmark_masks_opened") is False and payload.get("text_queries_opened") is False, "canonical field query/mask safety differs")
    _require(payload.get("capability_target_mode") == "official_adaptor_then_geometry_matched_mpr", "canonical field capability target mode differs")
    signature = _mapping(payload.get("feature_signature"), "canonical field feature signature")
    _require(signature.get("radio_checkpoint_sha256") == RADIO_CHECKPOINT_SHA256, "canonical field RADIO checkpoint differs")
    training = _mapping(payload.get("training_config"), "canonical field training config")
    expected_training = {
        "observation_contract": "canonical-mpr-v1",
        "coefficient_dim": 256,
        "local_dim": 128,
        "primitive_fusion": True,
        "official_capability_loss": True,
        "epochs": 20,
        "min_epochs": 5,
        "target_cosine": 0.985,
        "seed": 0,
    }
    for key, expected in expected_training.items():
        _require(training.get(key) == expected, f"canonical field training setting differs: {key}")
    training_sha256 = hashlib.sha256(
        json.dumps(dict(training), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _require(payload.get("training_config_sha256") == training_sha256, "canonical field training config SHA differs")
    _require(_same_path(str(training.get("radio_checkpoint", "")), RADIO_CHECKPOINT), "canonical field training checkpoint path differs")
    _require(training.get("expected_radio_checkpoint_sha256") == RADIO_CHECKPOINT_SHA256, "canonical field training checkpoint authority differs")

    bundle_sha256 = str(payload.get("feature_output_bundle_sha256", ""))
    _require(len(bundle_sha256) == 64, "canonical field lacks feature bundle identity")
    _require(training.get("expected_feature_output_bundle_sha256") == bundle_sha256, "canonical field training bundle authority differs")
    raw_sha256 = str(payload.get("mpr_cache_sha256", ""))
    raw_path = source.parent / "raw_radio.pt"
    _require(_same_path(str(payload.get("mpr_cache", "")), raw_path), "canonical field raw MPR path differs")
    _require(training.get("expected_mpr_cache_sha256") == raw_sha256, "canonical field raw MPR caller SHA differs")
    raw_storage_provenance = _mapping(payload.get("mpr_cache_storage"), "canonical field raw storage provenance")
    _require(raw_storage_provenance.get("storage") == storage_contract["radio"], f"{scene}: raw MPR storage contract differs")
    raw_validator = (
        validate_sharded_mpr_identity
        if storage_contract["radio"] == _SHARDED_MPR_SCHEMA
        else validate_dense_mpr_identity
    )
    raw_arguments: dict[str, Any] = {
        "expected_sha256": raw_sha256,
        "expected_feature_space": "radio",
        "expected_geometry": geometry,
        "expected_feature_bundle_sha256": bundle_sha256,
    }
    if storage_contract["radio"] == _SHARDED_MPR_SCHEMA:
        raw_arguments["expected_shard_channels"] = EXPECTED_SHARD_CHANNELS[scene]["radio"]
    raw = raw_validator(raw_path, **raw_arguments)
    _require(raw.get("storage") == storage_contract["radio"], f"{scene}: raw MPR storage contract differs")
    _require(dict(_mapping(payload.get("mpr_cache_metadata"), "canonical field raw MPR metadata")) == dict(raw["metadata"]), "canonical field raw MPR metadata differs")
    _validate_field_storage_provenance(raw_storage_provenance, raw, label="raw MPR")

    capability = _mapping(payload.get("capability_mpr_targets"), "canonical field capability targets")
    _require(set(capability) == {"dino_v3", "sam3"}, "canonical field capability target cohort differs")
    capability_records: dict[str, Any] = {}
    for feature_space, filename in (("dino_v3", "dino_v3.pt"), ("sam3", "sam3.pt")):
        declared = _mapping(capability[feature_space], f"{feature_space} field target")
        target_path = source.parent / filename
        target_sha256 = str(declared.get("sha256", ""))
        _require(_same_path(str(declared.get("path", "")), target_path), f"{feature_space} field target path differs")
        training_path_key = f"{feature_space.replace('dino_v3', 'dino')}_mpr_cache"
        training_sha_key = f"expected_{feature_space}_mpr_cache_sha256"
        _require(_same_path(str(training.get(training_path_key, "")), target_path), f"{feature_space} training target path differs")
        _require(training.get(training_sha_key) == target_sha256, f"{feature_space} training target SHA differs")
        _require(declared.get("projection_order") == "official_adaptor_then_geometry_matched_mpr", f"{feature_space} projection order differs")
        _require(declared.get("official_adaptor_checkpoint_sha256") == RADIO_CHECKPOINT_SHA256, f"{feature_space} field RADIO checkpoint differs")
        _require(declared.get("uses_query_or_benchmark_supervision") is False, f"{feature_space} field target used benchmark supervision")
        _require(declared.get("storage") == storage_contract[feature_space], f"{scene}: {feature_space} MPR storage contract differs")
        target_validator = (
            validate_sharded_mpr_identity
            if storage_contract[feature_space] == _SHARDED_MPR_SCHEMA
            else validate_dense_mpr_identity
        )
        target_arguments: dict[str, Any] = {
            "expected_sha256": target_sha256,
            "expected_feature_space": feature_space,
            "expected_geometry": geometry,
            "expected_feature_bundle_sha256": bundle_sha256,
            "expected_responsibility_sha256": raw["responsibility_sha256"],
        }
        if storage_contract[feature_space] == _SHARDED_MPR_SCHEMA:
            target_arguments["expected_shard_channels"] = EXPECTED_SHARD_CHANNELS[scene][feature_space]
        target = target_validator(target_path, **target_arguments)
        _require(target.get("storage") == storage_contract[feature_space], f"{scene}: {feature_space} MPR storage contract differs")
        _validate_field_storage_provenance(declared, target, label=feature_space)
        capability_records[feature_space] = target
    return {
        "path": str(source),
        "sha256": field_sha256,
        "geometry_fingerprint": dict(geometry),
        "feature_output_bundle_sha256": bundle_sha256,
        "storage_contract": dict(storage_contract),
        "shard_channel_contract": dict(EXPECTED_SHARD_CHANNELS[scene]),
        "raw_mpr": raw,
        "capability_mpr": capability_records,
        "training_config_sha256": payload.get("training_config_sha256"),
    }


def bind(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve()
    manifest_path = _resolve(root, args.manifest)
    asset_map_path = _resolve(root, args.asset_map)
    render_root = _resolve(root, args.render_root)
    prediction_path = _resolve(root, args.prediction_manifest)
    evaluation_path = _resolve(root, args.evaluation)
    output = _resolve(root, args.output)
    for label, path in (
        ("dataset manifest", manifest_path),
        ("asset map", asset_map_path),
        ("prediction manifest", prediction_path),
        ("evaluation", evaluation_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    _require(_sha256(manifest_path) == MANIFEST_SHA256, "SPIn local9 manifest drifted")
    general_freeze = root / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml"
    registry = root / "paper/artifacts/promptable_nvs_protocol_registry.yaml"
    _require(_sha256(general_freeze) == GENERAL_FREEZE_SHA256, "general freeze drifted")
    _require(_sha256(registry) == PROMPTABLE_REGISTRY_SHA256, "promptable registry drifted")

    manifest = _json(manifest_path)
    protocol = _mapping(manifest.get("protocol"), "manifest protocol")
    manifest_scenes = tuple(str(row["scene_id"]) for row in manifest["scenes"])
    _require(manifest_scenes == SCENES, "SPIn local9 cohort/order differs")
    _require(tuple(protocol.get("missing_scenes", [])) == ("fork",), "missing scene must be Fork")
    _require(protocol.get("formal_10scene_eligible") is False, "local9 became ten-scene eligible")
    _require(str(manifest.get("protocol_hash")) == PROTOCOL_HASH, "protocol hash differs")
    _require(protocol.get("score_semantics") == "cosine_similarity_foreground_minus_background", "score semantics differ")
    _require(protocol.get("prediction_representation") == "continuous_margin", "prediction representation differs")
    _require(protocol.get("resize") == "nearest", "score resize differs")
    _require(protocol.get("threshold_comparison") == "greater_or_equal", "threshold comparison differs")
    _require(_mapping(protocol.get("threshold"), "threshold") == {"mode": "fixed", "value": 0.0}, "threshold is not fixed zero")
    _require(protocol.get("within_scene_aggregation") == "unweighted_frame_mean", "frame aggregation differs")
    _require(protocol.get("dataset_aggregation") == "unweighted_macro_over_9_available_scenes_diagnostic", "scene aggregation differs")

    asset_map = _mapping(yaml.safe_load(asset_map_path.read_text(encoding="utf-8")), "asset map")
    _require(asset_map.get("schema_version") == 1 and asset_map.get("benchmark") == "SPIn-NeRF", "asset map schema/benchmark differs")
    _require(asset_map.get("cohort_role") == "local9_full_reference_mask_diagnostic", "asset map cohort differs")
    _require(asset_map.get("official_10scene_eligible") is False, "asset map claim differs")
    _require(asset_map.get("missing_official_scene") == "fork", "asset map missing scene differs")
    _require(_resolve(root, str(asset_map.get("protocol_manifest", ""))) == manifest_path, "asset map protocol path differs")
    _require(asset_map.get("protocol_manifest_sha256") == MANIFEST_SHA256, "asset map protocol SHA differs")
    assets = _mapping(asset_map.get("scenes"), "asset-map scenes")
    _require(tuple(assets) == SCENES, "asset-map cohort/order differs")

    prediction = _json(prediction_path)
    _require(prediction.get("kind") == "promptable_nvs_continuous_score_predictions", "producer kind differs")
    _require(prediction.get("protocol_hash") == PROTOCOL_HASH, "prediction protocol differs")
    method = _mapping(prediction.get("method"), "prediction method")
    _require(method.get("readout") == "reference_prototype_cosine_margin", "readout is not raw prototype cosine margin")
    _require(method.get("score_semantics") == "cosine_similarity_foreground_minus_background", "producer score semantics differ")
    _require(_mapping(method.get("threshold"), "producer threshold").get("value") == 0.0, "producer threshold is not zero")
    _require(_mapping(method.get("threshold"), "producer threshold").get("mode") == "fixed", "producer threshold is not fixed")
    prediction_embedding = validate_prediction_embedding(method)
    prediction_input = _mapping(prediction.get("input"), "prediction input")
    _require(_same_path(str(prediction_input.get("dataset_manifest", "")), manifest_path), "prediction dataset manifest path differs")
    _require(prediction_input.get("dataset_manifest_sha256") == MANIFEST_SHA256, "prediction dataset manifest SHA differs")
    _require(_same_path(str(prediction_input.get("feature_root", "")), render_root), "prediction feature root differs")
    _require(prediction_input.get("feature_pattern") == "{scene_id}/{camera_name}.pt", "prediction feature pattern differs")
    _require(prediction_input.get("feature_layout") == "chw", "prediction feature layout differs")
    safety = _mapping(prediction.get("safety"), "prediction safety")
    _require(safety.get("evaluation_ground_truth_opened") is False, "prediction producer opened evaluation GT")
    _require(safety.get("evaluation_performed") is False, "prediction producer evaluated targets")
    prediction_scenes = prediction.get("scenes")
    _require(isinstance(prediction_scenes, list), "prediction scenes must be a list")
    _require(tuple(str(row["scene_id"]) for row in prediction_scenes) == SCENES, "prediction cohort/order differs")

    manifest_by_scene = {str(row["scene_id"]): row for row in manifest["scenes"]}
    prediction_by_scene = {str(row["scene_id"]): row for row in prediction_scenes}
    predictions = _mapping(prediction.get("predictions"), "predictions")
    prediction_hashes = _mapping(prediction.get("prediction_sha256"), "prediction hashes")
    _require(set(predictions) == set(SCENES) and set(prediction_hashes) == set(SCENES), "prediction score indexes differ")
    prediction_root = prediction_path.parent
    scene_authorities: list[dict[str, Any]] = []
    all_scores = 0
    for scene in SCENES:
        scene_manifest = manifest_by_scene[scene]
        expected_targets = tuple(str(value) for value in scene_manifest["evaluation_frame_ids"])
        expected_rendered = (str(scene_manifest["prompt_frame_ids"][0]), *expected_targets)
        render_path = render_root / scene / "render_manifest.json"
        render = _json(render_path)
        _require(render.get("kind") == "promptable_nvs_gaussfm_render", f"{scene}: render kind differs")
        _require(render.get("scene_id") == scene, f"{scene}: render scene differs")
        _require(render.get("protocol_hash") == PROTOCOL_HASH, f"{scene}: render protocol differs")
        _require(_same_path(str(render.get("manifest", "")), manifest_path), f"{scene}: render dataset manifest path differs")
        _require(render.get("manifest_file_sha256") == MANIFEST_SHA256, f"{scene}: render dataset manifest SHA differs")
        _require(render.get("render_mode") == "canonical_mpr_v3_affine_normalized_splat", f"{scene}: render mode differs")
        render_contract = _mapping(render.get("canonical_render_contract"), f"{scene}: canonical render contract")
        _require(render_contract == {"normalized_splat": True, "affine_decode_after_splat": True, "reliability_splat": False, "screen_refiner": False}, f"{scene}: canonical rendering changed")
        render_safety = _mapping(render.get("safety"), f"{scene}: render safety")
        for key in ("rgb_files_opened", "segmentation_masks_opened", "evaluation_ground_truth_opened", "rgb_refiner_used"):
            _require(render_safety.get(key) is False, f"{scene}: unsafe render flag {key}")
        scene_assets = _mapping(assets.get(scene), f"{scene}: asset map")
        config = _resolve(root, scene_assets["config"])
        geometry = _resolve(root, scene_assets["geometry_checkpoint"])
        field = _resolve(root, scene_assets["canonical_field"])
        camera_map = _resolve(root, scene_assets["camera_map"])
        asset_records = {
            "camera_map": _file_record(camera_map),
            "config": _file_record(config),
            "geometry_checkpoint": _file_record(geometry),
            "canonical_field": _file_record(field),
        }
        for key, path, sha_key, record_key in (
            ("camera_map", camera_map, "camera_map_sha256", "camera_map"),
            ("config", config, "config_sha256", "config"),
            ("checkpoint", geometry, "checkpoint_sha256", "geometry_checkpoint"),
            ("canonical_field_checkpoint", field, "canonical_field_checkpoint_sha256", "canonical_field"),
        ):
            _require(_same_path(render.get(key, ""), path), f"{scene}: {key} path differs")
            _require(asset_records[record_key]["sha256"] == render.get(sha_key), f"{scene}: {key} SHA differs")
        rendered_rows = render.get("outputs")
        _require(isinstance(rendered_rows, list), f"{scene}: render outputs are absent")
        _require(tuple(str(row["frame_id"]) for row in rendered_rows) == expected_rendered, f"{scene}: rendered frame cohort/order differs")
        render_by_frame = {str(row["frame_id"]): row for row in rendered_rows}
        frames = _frame_map(scene_manifest)
        render_feature_hashes: dict[str, str] = {}
        for index, frame in enumerate(expected_rendered):
            render_row = _mapping(rendered_rows[index], f"{scene}/{frame}: render row")
            expected_role = "prompt" if index == 0 else "evaluation"
            _require(render_row.get("role") == expected_role, f"{scene}/{frame}: render role differs")
            _require(render_row.get("camera_name") == frames[frame].get("camera_name"), f"{scene}/{frame}: render camera differs")
            feature_path = Path(str(render_row.get("feature_path", ""))).expanduser().resolve()
            _require(feature_path.is_file() and not feature_path.is_symlink(), f"{scene}/{frame}: rendered feature is not a regular file")

        pred_scene = _mapping(prediction_by_scene[scene], f"{scene}: prediction scene")
        prompt_record = validate_prompt_binding(
            scene=scene,
            scene_manifest=scene_manifest,
            prediction_scene=pred_scene,
            render_by_frame=render_by_frame,
            manifest_base=manifest_path.parent,
        )
        render_feature_hashes[prompt_record["frame_id"]] = prompt_record["feature"]["sha256"]
        output_rows = pred_scene.get("outputs")
        _require(isinstance(output_rows, list), f"{scene}: prediction outputs are absent")
        _require(tuple(str(row["frame_id"]) for row in output_rows) == expected_targets, f"{scene}: scored frame cohort/order differs")
        scene_predictions = _mapping(predictions.get(scene), f"{scene}: prediction paths")
        scene_hashes = _mapping(prediction_hashes.get(scene), f"{scene}: prediction hashes")
        _require(set(scene_predictions) == set(expected_targets) and set(scene_hashes) == set(expected_targets), f"{scene}: score indexes differ")
        score_records: list[dict[str, Any]] = []
        for row in output_rows:
            frame = str(row["frame_id"])
            _require(row.get("score_dtype") == "float32", f"{scene}/{frame}: producer dtype differs")
            relative = str(row["score_path"])
            _require(Path(relative) == Path("scores") / scene / f"{frame}.npy", f"{scene}/{frame}: score path layout differs")
            _require(scene_predictions.get(frame) == relative, f"{scene}/{frame}: score path map differs")
            score_path = (prediction_root / relative).resolve()
            score_sha = _sha256(score_path)
            _require(score_sha == row.get("score_sha256") == scene_hashes.get(frame), f"{scene}/{frame}: score SHA differs")
            score = np.load(score_path, allow_pickle=False)
            _require(score.dtype == np.float32 and score.ndim == 2, f"{scene}/{frame}: score must be float32 HW")
            _require(bool(np.isfinite(score).all()), f"{scene}/{frame}: score is non-finite")
            _require(float(score.min()) >= -2.0001 and float(score.max()) <= 2.0001, f"{scene}/{frame}: score is not a raw cosine margin")
            feature_path = Path(str(row["feature_path"])).resolve()
            feature_sha256 = _sha256(feature_path)
            _require(feature_sha256 == row.get("feature_sha256"), f"{scene}/{frame}: feature SHA differs")
            _require(feature_path == Path(str(render_by_frame[frame]["feature_path"])).resolve(), f"{scene}/{frame}: producer did not consume the bound render")
            render_feature_hashes[frame] = feature_sha256
            score_records.append({"frame_id": frame, "score": str(score_path), "score_sha256": score_sha, "shape": list(score.shape), "min": float(score.min()), "max": float(score.max())})
            all_scores += 1
        field_sha256 = asset_records["canonical_field"]["sha256"]
        field_provenance = (
            validate_new_field_provenance(
                field,
                scene=scene,
                expected_sha256=field_sha256,
                expected_geometry=_mapping(render.get("canonical_field_geometry_fingerprint"), f"{scene}: render field geometry"),
            )
            if scene in NEW_FIELD_SCENES
            else None
        )
        scene_authorities.append({
            "scene_id": scene,
            "render_manifest": str(render_path),
            "render_manifest_sha256": _sha256(render_path),
            "assets": asset_records,
            "prompt": prompt_record,
            "rendered_feature_sha256": render_feature_hashes,
            "new_field_provenance": field_provenance,
            "scores": score_records,
        })

    evaluation = _json(evaluation_path)
    recomputed = evaluate_manifest(manifest_path, prediction_manifest=prediction_path)
    validate_evaluation_recomputation(evaluation, recomputed)
    dataset = _mapping(evaluation.get("dataset"), "evaluation dataset")
    _require(int(dataset.get("num_scenes", -1)) == 9, "evaluation is not nine-scene")
    _require(all_scores == 414 and int(dataset.get("num_frames", -1)) == 414, "evaluation frame count is not the frozen 414")
    _require(math.isfinite(float(dataset["foreground_iou"])), "macro IoU is non-finite")
    _require(math.isfinite(float(dataset["pixel_accuracy"])), "macro accuracy is non-finite")

    sources = (
        "radio_gs/scripts/render_promptable_nvs_features.py",
        "radio_gs/scripts/predict_promptable_nvs_feature_readout.py",
        "radio_gs/scripts/eval_promptable_nvs_segmentation.py",
        "radio_gs/scripts/build_promptable_geometry_render_contract.py",
        "radio_gs/scripts/bind_spin9_canonical_exact_authority.py",
    )
    authority = {
        "schema_version": 1,
        "kind": "spin9_canonical_mpr_v3_exact_local9_result_authority",
        "claim_scope": {
            "benchmark": "SPIn-NeRF",
            "cohort": "local9_full_reference_mask_diagnostic",
            "official_10scene_eligible": False,
            "missing_scene": "fork",
            "published_sparse_point_protocol_exact": False,
            "local9_frozen_score_protocol_exact": True,
        },
        "scoring_contract": {
            "producer_kind": prediction["kind"],
            "readout": method["readout"],
            "score_semantics": "continuous_cosine_margin",
            "stored_values": method["score_semantics"],
            "prediction_representation": "continuous_cosine_margin",
            "threshold": {"comparison": "greater_or_equal", "value": 0.0},
            "resize": "nearest",
            "within_scene_aggregation": "unweighted_frame_mean",
            "dataset_aggregation": "unweighted_macro_over_9_available_scenes",
        },
        "protocol": {"manifest": str(manifest_path), "manifest_sha256": MANIFEST_SHA256, "protocol_hash": PROTOCOL_HASH},
        "freeze_bindings": {
            "evaluation_protocol_freeze_20260801_sha256": GENERAL_FREEZE_SHA256,
            "promptable_nvs_protocol_registry_sha256": PROMPTABLE_REGISTRY_SHA256,
        },
        "asset_map": str(asset_map_path),
        "asset_map_sha256": _sha256(asset_map_path),
        "prediction_embedding": prediction_embedding,
        "prediction_manifest": str(prediction_path),
        "prediction_manifest_sha256": _sha256(prediction_path),
        "evaluation": str(evaluation_path),
        "evaluation_sha256": _sha256(evaluation_path),
        "metrics": dict(dataset),
        "scenes": scene_authorities,
        "source_sha256": {path: _sha256(root / path) for path in sources},
        "safety": {
            "target_ground_truth_opened_by_render_or_prediction": False,
            "target_masks_used_only_by_separate_evaluator": True,
            "target_masks_used_for_calibration": False,
            "target_rgb_opened_at_query": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    receipt = {"authority": str(output), "authority_sha256": _sha256(output), "status": "validated"}
    output.with_suffix(output.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asset-map", required=True)
    parser.add_argument("--render-root", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(bind(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
