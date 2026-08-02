#!/usr/bin/env python3
"""Evaluate Ours scalar query maps with the frozen Occam LERF-2D readout.

This is a CPU-only adapter for already rendered query-score maps.  It does not
encode text, choose scales, tune thresholds, or resize maps.  A bundle must
contain exactly three scale maps for every query at every camera in the frozen
four-scene cohort.  Camera names, deterministic query IDs, annotation files,
map files, scale identities, and source artifacts are all bound explicitly.

The historical pre-rendered feature/OpenCLIP entrypoint remains separate in
``eval_prerendered_lerf_features.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radio_gs.scripts.eval_opengaussian_lerf_baseline import (
    SCENE_GT_FRAMES,
    _coerce_polygons,
    _rasterize_polygons,
)
from radio_gs.scripts.eval_prerendered_lerf_features import (
    LerfObject,
    _bbox_tuple,
    aggregate_scene_results,
    evaluate_relevance_maps,
    resolve_protocol_config,
)
from radio_gs.scripts.validate_evaluation_protocol_freeze import (
    FreezeError,
    load_and_validate,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "radio_gs_lerf2d_three_scale_scalar_query_map_bundle"
RESULT_ARTIFACT_TYPE = "radio_gs_lerf2d_occam_exact_scalar_map_evaluation"
CANONICAL_TASK_ID = "concept_lerf2d_occamlgs"
EXPECTED_REGISTRY_ROW = "lerf2d_occamlgs_official_checkpoint_context_20260731"
SCORE_SEMANTICS = "raw_query_relevance_pre_occam_activation"
EXPECTED_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
EXPECTED_LABELLED_FRAMES = 22
EXPECTED_QUERIES = 208


class ScalarMapProtocolError(ValueError):
    """Raised before scoring when a bundle does not match the frozen protocol."""


@dataclass(frozen=True)
class FrozenLerf2DContract:
    """Resolved, immutable subset of the general protocol freeze."""

    freeze_path: str
    freeze_sha256: str
    freeze_id: str
    canonical_task_id: str
    registry_row: str
    scenes: tuple[str, ...]
    frames_by_scene: Mapping[str, tuple[str, ...]]
    labelled_frames: int
    queries: int
    protocol_config: Mapping[str, Any]


def _sha256_bytes(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _stable_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_stable_regular_file(path: Path, *, label: str) -> bytes:
    """Read one non-symlink regular file and reject concurrent mutation."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe scalar-map evaluation requires O_NOFOLLOW")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
    except FileNotFoundError as error:
        raise ScalarMapProtocolError(f"{label} does not exist: {path}") from error
    except OSError as error:
        raise ScalarMapProtocolError(f"{label} cannot be opened safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ScalarMapProtocolError(f"{label} must be a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read()
        after = os.fstat(descriptor)
        if _stable_identity(before) != _stable_identity(after):
            raise ScalarMapProtocolError(f"{label} changed while being read: {path}")
        return encoded
    finally:
        os.close(descriptor)


def _sha256_stable_regular_file(path: Path, *, label: str) -> str:
    """Hash a potentially large source artifact without loading it into RAM."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe scalar-map evaluation requires O_NOFOLLOW")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
    except FileNotFoundError as error:
        raise ScalarMapProtocolError(f"{label} does not exist: {path}") from error
    except OSError as error:
        raise ScalarMapProtocolError(f"{label} cannot be opened safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ScalarMapProtocolError(f"{label} must be a regular file: {path}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        if _stable_identity(before) != _stable_identity(after):
            raise ScalarMapProtocolError(f"{label} changed while being hashed: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _load_json_bytes(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScalarMapProtocolError(f"{label} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ScalarMapProtocolError(f"{label} must contain a JSON mapping")
    return payload


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScalarMapProtocolError(f"{label} must be a mapping")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ScalarMapProtocolError(f"{label} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ScalarMapProtocolError(f"{label} must be a lowercase SHA256")
    return value


def _bundle_member(root: Path, raw_path: Any, *, label: str) -> Path:
    """Resolve a cache member without allowing absolute paths or symlinks."""

    if not isinstance(raw_path, str) or not raw_path:
        raise ScalarMapProtocolError(f"{label} must be a non-empty relative path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ScalarMapProtocolError(f"{label} must stay inside the bundle")
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ScalarMapProtocolError(f"{label} does not exist: {candidate}") from error
    # This rejects both a symlink at the final component and a symlink in any
    # intermediate directory.  Bundle provenance must not depend on aliases.
    if resolved != candidate.absolute():
        raise ScalarMapProtocolError(f"{label} may not contain symlinks: {candidate}")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ScalarMapProtocolError(f"{label} escaped the bundle: {candidate}") from error
    return resolved


def canonical_query_id(scene: str, frame: str, index: int, query: str) -> str:
    """Return the score-cache ID for one sorted, merged LERF query."""

    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"lerf2d:{scene}:{frame}:{index:03d}:{digest}"


def _occam_protocol_config() -> dict[str, Any]:
    namespace = argparse.Namespace(
        protocol_profile="occam_langsplat_paper",
        mask_thresh=None,
        activation_kernel=None,
        smooth_kernel=None,
        feature_mode=None,
    )
    config = resolve_protocol_config(namespace)
    expected = {
        "mask_thresh": 0.5,
        "activation_kernel": 30,
        "smooth_kernel": 7,
        "feature_mode": "raw",
        "filter_implementation": "opencv_filter2d",
        "mask_smoothing_implementation": "langsplat_legacy",
        "resize_policy": "error_on_mismatch",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ScalarMapProtocolError(
                f"local Occam evaluator drifted at {key}: expected {value!r}, "
                f"got {config.get(key)!r}"
            )
    return config


def contract_from_validated_freeze(
    payload: Mapping[str, Any],
    *,
    freeze_path: Path,
    freeze_sha256: str,
) -> FrozenLerf2DContract:
    """Resolve the LERF-2D task after the general freeze has been validated."""

    tasks = _require_mapping(payload.get("canonical_tasks"), label="canonical_tasks")
    task = _require_mapping(tasks.get(CANONICAL_TASK_ID), label=CANONICAL_TASK_ID)
    if task.get("registry_row") != EXPECTED_REGISTRY_ROW:
        raise ScalarMapProtocolError("frozen LERF-2D registry row drifted")
    cohort = _require_mapping(task.get("cohort"), label="LERF-2D cohort")
    scenes = tuple(cohort.get("scenes", ()))
    if scenes != EXPECTED_SCENES:
        raise ScalarMapProtocolError(
            f"frozen LERF-2D scene cohort must be {EXPECTED_SCENES!r}, got {scenes!r}"
        )
    if cohort.get("labelled_frames") != EXPECTED_LABELLED_FRAMES:
        raise ScalarMapProtocolError("frozen LERF-2D labelled-frame count drifted")
    if cohort.get("queries") != EXPECTED_QUERIES:
        raise ScalarMapProtocolError("frozen LERF-2D query count drifted")

    frozen = _require_mapping(task.get("frozen_protocol"), label="LERF-2D protocol")
    invariants = {
        "camera_lookup": "exact annotation name over all registered cameras",
        "level_selection": "highest raw OpenCLIP relevance peak over three levels",
        "segmentation_threshold": 0.5,
        "activation_filter": "30x30 OpenCV filter2D",
        "smoothing": "legacy 7x7",
        "aggregation": "unweighted equal macro over four scenes",
    }
    for key, value in invariants.items():
        if frozen.get(key) != value:
            raise ScalarMapProtocolError(
                f"frozen LERF-2D invariant {key} must equal {value!r}"
            )
    metrics = frozen.get("metrics")
    if metrics != ["mIoU", "localization_accuracy"]:
        raise ScalarMapProtocolError("frozen LERF-2D metrics drifted")
    openclip = frozen.get("openclip")
    if openclip != "ViT-B-16 / laion2b_s34b_b88k":
        raise ScalarMapProtocolError("frozen LERF-2D OpenCLIP identity drifted")

    freeze_id = payload.get("freeze_id")
    if not isinstance(freeze_id, str) or not freeze_id:
        raise ScalarMapProtocolError("validated freeze has no freeze_id")
    frames = {scene: tuple(SCENE_GT_FRAMES[scene]) for scene in scenes}
    if sum(len(values) for values in frames.values()) != EXPECTED_LABELLED_FRAMES:
        raise ScalarMapProtocolError("local exact-camera frame cohort drifted")
    return FrozenLerf2DContract(
        freeze_path=str(freeze_path),
        freeze_sha256=freeze_sha256,
        freeze_id=freeze_id,
        canonical_task_id=CANONICAL_TASK_ID,
        registry_row=EXPECTED_REGISTRY_ROW,
        scenes=scenes,
        frames_by_scene=frames,
        labelled_frames=EXPECTED_LABELLED_FRAMES,
        queries=EXPECTED_QUERIES,
        protocol_config=_occam_protocol_config(),
    )


def load_frozen_contract(
    freeze_path: Path,
    *,
    repo_root: Path,
    verify_hashes: bool = True,
) -> FrozenLerf2DContract:
    """Validate the general freeze, then bind the exact LERF-2D subset."""

    try:
        canonical = freeze_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ScalarMapProtocolError(f"protocol freeze does not exist: {freeze_path}") from error
    encoded = _read_stable_regular_file(canonical, label="protocol freeze")
    try:
        payload = load_and_validate(
            canonical,
            root=repo_root.resolve(),
            verify_hashes=verify_hashes,
        )
    except FreezeError as error:
        raise ScalarMapProtocolError(f"protocol freeze validation failed: {error}") from error
    # Re-read identity and contents after the validator traversed its authority
    # graph.  A changing freeze cannot authorize this evaluation.
    if _read_stable_regular_file(canonical, label="protocol freeze") != encoded:
        raise ScalarMapProtocolError("protocol freeze changed during validation")
    return contract_from_validated_freeze(
        payload,
        freeze_path=canonical,
        freeze_sha256=_sha256_bytes(encoded),
    )


def _load_annotation(
    path: Path,
    *,
    scene: str,
    frame: str,
) -> tuple[str, list[LerfObject], tuple[int, int], str]:
    encoded = _read_stable_regular_file(path, label=f"{scene}/{frame} annotation")
    payload = _load_json_bytes(encoded, label=f"{scene}/{frame} annotation")
    info = _require_mapping(payload.get("info"), label=f"{scene}/{frame}.info")
    camera_name = info.get("name")
    if not isinstance(camera_name, str) or not camera_name:
        raise ScalarMapProtocolError(f"{scene}/{frame}: annotation has no exact camera name")
    if Path(camera_name).stem != frame:
        raise ScalarMapProtocolError(
            f"{scene}/{frame}: annotation camera stem mismatch: {camera_name!r}"
        )
    try:
        height = int(info["height"])
        width = int(info["width"])
    except (KeyError, TypeError, ValueError) as error:
        raise ScalarMapProtocolError(f"{scene}/{frame}: invalid annotation resolution") from error
    if height <= 0 or width <= 0:
        raise ScalarMapProtocolError(f"{scene}/{frame}: annotation resolution must be positive")

    raw_objects = payload.get("objects", [])
    if not isinstance(raw_objects, list):
        raise ScalarMapProtocolError(f"{scene}/{frame}: objects must be a list")
    masks_by_query: dict[str, np.ndarray] = {}
    bboxes_by_query: dict[str, list[tuple[float, float, float, float]]] = {}
    for raw_object in raw_objects:
        if not isinstance(raw_object, Mapping):
            continue
        query = str(raw_object.get("category", "")).strip()
        polygons = _coerce_polygons(raw_object.get("segmentation"))
        if not query or not polygons:
            continue
        mask = _rasterize_polygons(polygons, height, width)
        if query in masks_by_query:
            masks_by_query[query] = np.logical_or(masks_by_query[query], mask)
        else:
            masks_by_query[query] = mask
        bbox = _bbox_tuple(raw_object.get("bbox", []))
        if bbox is not None:
            bboxes_by_query.setdefault(query, []).append(bbox)
    objects = [
        LerfObject(
            frame=frame,
            query=query,
            mask=masks_by_query[query].astype(bool),
            bboxes=bboxes_by_query.get(query, []),
        )
        for query in sorted(masks_by_query)
    ]
    if not objects:
        raise ScalarMapProtocolError(f"{scene}/{frame}: annotation contains no valid queries")
    return camera_name, objects, (height, width), _sha256_bytes(encoded)


def _validate_scales(raw_scales: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_scales, list) or len(raw_scales) != 3:
        raise ScalarMapProtocolError("score bundle must declare exactly three scales")
    scales: list[dict[str, Any]] = []
    ids: set[str] = set()
    values: set[float] = set()
    for index, raw_scale in enumerate(raw_scales):
        scale = _require_mapping(raw_scale, label=f"scales[{index}]")
        scale_id = scale.get("id")
        raw_value = scale.get("value")
        unit = scale.get("unit")
        if not isinstance(scale_id, str) or not scale_id or scale_id in ids:
            raise ScalarMapProtocolError("scale IDs must be non-empty and unique")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ScalarMapProtocolError("scale values must be finite positive numbers")
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0 or value in values:
            raise ScalarMapProtocolError("scale values must be finite, positive, and unique")
        if not isinstance(unit, str) or not unit:
            raise ScalarMapProtocolError("every scale must declare a non-empty unit")
        ids.add(scale_id)
        values.add(value)
        scales.append({"id": scale_id, "value": value, "unit": unit})
    return scales


def _validate_source_artifacts(
    raw_artifacts: Any,
    *,
    bundle_root: Path,
) -> list[dict[str, str]]:
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ScalarMapProtocolError("score bundle must bind at least one source artifact")
    output: list[dict[str, str]] = []
    roles: set[str] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        artifact = _require_mapping(raw_artifact, label=f"source_artifacts[{index}]")
        role = artifact.get("role")
        if not isinstance(role, str) or not role or role in roles:
            raise ScalarMapProtocolError("source artifact roles must be non-empty and unique")
        path = _bundle_member(
            bundle_root,
            artifact.get("path"),
            label=f"source_artifacts[{index}].path",
        )
        expected_sha = _require_sha256(
            artifact.get("sha256"), label=f"source_artifacts[{index}].sha256"
        )
        actual_sha = _sha256_stable_regular_file(path, label=f"source artifact {role}")
        if actual_sha != expected_sha:
            raise ScalarMapProtocolError(
                f"source artifact SHA256 mismatch for role {role}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        roles.add(role)
        output.append({"role": role, "path": str(path), "sha256": actual_sha})
    return output


def _load_score_map(
    path: Path,
    *,
    expected_sha: str,
    expected_shape: tuple[int, int, int, int],
    label: str,
) -> tuple[np.ndarray, str]:
    encoded = _read_stable_regular_file(path, label=label)
    actual_sha = _sha256_bytes(encoded)
    if actual_sha != expected_sha:
        raise ScalarMapProtocolError(
            f"{label} SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    try:
        array = np.load(io.BytesIO(encoded), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ScalarMapProtocolError(f"{label} is not a valid .npy array") from error
    if not isinstance(array, np.ndarray):
        raise ScalarMapProtocolError(f"{label} must contain one .npy array, not an archive")
    if tuple(array.shape) != expected_shape:
        raise ScalarMapProtocolError(
            f"{label} shape mismatch: expected {expected_shape}, got {tuple(array.shape)}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ScalarMapProtocolError(f"{label} dtype must be floating, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ScalarMapProtocolError(f"{label} contains non-finite scores")
    return np.asarray(array, dtype=np.float32), actual_sha


def evaluate_scalar_map_bundle(
    score_manifest: Path,
    *,
    label_root: Path,
    contract: FrozenLerf2DContract,
) -> dict[str, Any]:
    """Validate and score one complete four-scene scalar-map bundle."""

    manifest_path = score_manifest.resolve(strict=True)
    manifest_encoded = _read_stable_regular_file(manifest_path, label="score manifest")
    manifest = _load_json_bytes(manifest_encoded, label="score manifest")
    bundle_root = manifest_path.parent.resolve()

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ScalarMapProtocolError(f"score manifest schema_version must be {SCHEMA_VERSION}")
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ScalarMapProtocolError(f"score manifest artifact_type must be {ARTIFACT_TYPE!r}")
    method = manifest.get("method")
    if not isinstance(method, str) or not method or method == "OccamLGS":
        raise ScalarMapProtocolError("score manifest must name a non-Occam evaluated method")
    if manifest.get("score_semantics") != SCORE_SEMANTICS:
        raise ScalarMapProtocolError(f"score_semantics must be {SCORE_SEMANTICS!r}")
    if manifest.get("canonical_task_id") != contract.canonical_task_id:
        raise ScalarMapProtocolError("score manifest canonical task mismatch")
    if manifest.get("registry_row") != contract.registry_row:
        raise ScalarMapProtocolError("score manifest registry row mismatch")
    protocol_binding = _require_mapping(
        manifest.get("protocol_freeze"), label="protocol_freeze"
    )
    if protocol_binding.get("freeze_id") != contract.freeze_id:
        raise ScalarMapProtocolError("score manifest freeze_id mismatch")
    if protocol_binding.get("sha256") != contract.freeze_sha256:
        raise ScalarMapProtocolError("score manifest protocol freeze SHA256 mismatch")

    scales = _validate_scales(manifest.get("scales"))
    sources = _validate_source_artifacts(
        manifest.get("source_artifacts"), bundle_root=bundle_root
    )
    raw_scenes = _require_mapping(manifest.get("scenes"), label="scenes")
    if tuple(raw_scenes) != contract.scenes or set(raw_scenes) != set(contract.scenes):
        raise ScalarMapProtocolError(
            "score manifest scene cohort/order mismatch: "
            f"expected {contract.scenes!r}, got {tuple(raw_scenes)!r}"
        )

    scene_results: dict[str, Mapping[str, object]] = {}
    input_receipts: dict[str, Any] = {}
    observed_query_count = 0
    observed_frame_count = 0
    protocol = contract.protocol_config
    for scene in contract.scenes:
        scene_entry = _require_mapping(raw_scenes[scene], label=f"scenes.{scene}")
        raw_frames = _require_mapping(scene_entry.get("frames"), label=f"scenes.{scene}.frames")
        expected_frames = contract.frames_by_scene[scene]
        if tuple(raw_frames) != expected_frames or set(raw_frames) != set(expected_frames):
            raise ScalarMapProtocolError(
                f"{scene}: exact camera cohort/order mismatch; "
                f"expected {expected_frames!r}, got {tuple(raw_frames)!r}"
            )

        objects_by_frame: dict[str, list[LerfObject]] = {}
        maps_by_frame: dict[str, np.ndarray] = {}
        scene_receipts: dict[str, Any] = {}
        for frame in expected_frames:
            observed_frame_count += 1
            entry = _require_mapping(raw_frames[frame], label=f"{scene}/{frame}")
            annotation_path = label_root / scene / f"{frame}.json"
            camera_name, objects, resolution, annotation_sha = _load_annotation(
                annotation_path,
                scene=scene,
                frame=frame,
            )
            if entry.get("annotation_sha256") != annotation_sha:
                raise ScalarMapProtocolError(f"{scene}/{frame}: annotation SHA256 mismatch")
            if entry.get("camera_name") != camera_name:
                raise ScalarMapProtocolError(
                    f"{scene}/{frame}: exact camera name mismatch; "
                    f"expected {camera_name!r}, got {entry.get('camera_name')!r}"
                )
            query_texts = [obj.query for obj in objects]
            query_ids = [
                canonical_query_id(scene, frame, index, query)
                for index, query in enumerate(query_texts)
            ]
            if entry.get("query_texts") != query_texts:
                raise ScalarMapProtocolError(f"{scene}/{frame}: query text/order mismatch")
            if entry.get("query_ids") != query_ids:
                raise ScalarMapProtocolError(f"{scene}/{frame}: query ID/order mismatch")
            expected_resolution = [resolution[0], resolution[1]]
            if entry.get("map_resolution_hw") != expected_resolution:
                raise ScalarMapProtocolError(
                    f"{scene}/{frame}: map resolution binding mismatch; "
                    f"expected {expected_resolution!r}"
                )
            expected_shape = (3, len(objects), resolution[0], resolution[1])
            if entry.get("map_shape_lqhw") != list(expected_shape):
                raise ScalarMapProtocolError(
                    f"{scene}/{frame}: declared map shape mismatch; expected {list(expected_shape)!r}"
                )
            if entry.get("scale_ids") != [scale["id"] for scale in scales]:
                raise ScalarMapProtocolError(f"{scene}/{frame}: scale ID/order mismatch")
            map_path = _bundle_member(
                bundle_root,
                entry.get("map_file"),
                label=f"{scene}/{frame}.map_file",
            )
            expected_map_sha = _require_sha256(
                entry.get("map_sha256"), label=f"{scene}/{frame}.map_sha256"
            )
            score_map, map_sha = _load_score_map(
                map_path,
                expected_sha=expected_map_sha,
                expected_shape=expected_shape,
                label=f"{scene}/{frame} score map",
            )
            objects_by_frame[frame] = objects
            maps_by_frame[frame] = score_map
            observed_query_count += len(objects)
            scene_receipts[frame] = {
                "annotation_path": str(annotation_path.resolve()),
                "annotation_sha256": annotation_sha,
                "camera_name": camera_name,
                "query_ids": query_ids,
                "query_texts": query_texts,
                "map_path": str(map_path),
                "map_sha256": map_sha,
                "map_shape_lqhw": list(expected_shape),
                "map_resolution_hw": expected_resolution,
                "scale_ids": [scale["id"] for scale in scales],
            }
        scene_results[scene] = evaluate_relevance_maps(
            objects_by_frame,
            maps_by_frame,
            mask_thresh=float(protocol["mask_thresh"]),
            activation_kernel=int(protocol["activation_kernel"]),
            smooth_kernel=int(protocol["smooth_kernel"]),
            filter_implementation=str(protocol["filter_implementation"]),
            mask_smoothing_implementation=str(
                protocol["mask_smoothing_implementation"]
            ),
            resize_policy=str(protocol["resize_policy"]),
        )
        input_receipts[scene] = scene_receipts

    if observed_frame_count != contract.labelled_frames:
        raise ScalarMapProtocolError(
            f"observed {observed_frame_count} frames, expected {contract.labelled_frames}"
        )
    if observed_query_count != contract.queries:
        raise ScalarMapProtocolError(
            f"observed {observed_query_count} queries, expected {contract.queries}"
        )
    aggregate = aggregate_scene_results(scene_results)
    if int(aggregate["scene_macro"]["scenes"]) != len(contract.scenes):
        raise ScalarMapProtocolError("four-scene macro aggregation did not cover all scenes")

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RESULT_ARTIFACT_TYPE,
        "status": "complete_exact_frozen_protocol_evaluation",
        "method": method,
        "benchmark": "LERF-2D",
        "adapter": "ours_prerendered_three_scale_scalar_query_maps",
        "score_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_bytes(manifest_encoded),
            "artifact_type": ARTIFACT_TYPE,
            "score_semantics": SCORE_SEMANTICS,
        },
        "protocol_authority": {
            "path": contract.freeze_path,
            "sha256": contract.freeze_sha256,
            "freeze_id": contract.freeze_id,
            "canonical_task_id": contract.canonical_task_id,
            "registry_row": contract.registry_row,
        },
        "protocol_config": dict(protocol),
        "protocol_constraints": {
            "text_encoder_invoked_by_adapter": False,
            "scale_selection": "frozen evaluator argmax activated raw-relevance peak over all three declared scales",
            "threshold_selected_or_tuned": False,
            "resize_or_resample": False,
            "camera_lookup": "exact annotation camera name",
            "query_order": "sorted merged annotation queries with deterministic IDs",
            "aggregation": "unweighted equal macro over four scenes",
        },
        "scales": scales,
        "source_artifacts": sources,
        "cohort": {
            "scenes": list(contract.scenes),
            "labelled_frames": observed_frame_count,
            "queries": observed_query_count,
        },
        "input_receipts": input_receipts,
        "scene_results": scene_results,
        **aggregate,
    }


def write_result_no_clobber(output: Path, payload: Mapping[str, Any]) -> Path:
    """Publish one immutable evaluation result without replacing an old row."""

    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ScalarMapProtocolError(f"output already exists: {output}") from error
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-freeze",
        type=Path,
        default=repo_root / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    contract = load_frozen_contract(
        args.protocol_freeze,
        repo_root=args.repo_root,
        verify_hashes=True,
    )
    result = evaluate_scalar_map_bundle(
        args.score_manifest,
        label_root=args.label_root,
        contract=contract,
    )
    write_result_no_clobber(args.output_json, result)
    macro = result["scene_macro"]
    print(
        f"scene-macro LocAcc={macro['loc_acc']:.4f} "
        f"mIoU={macro['miou']:.4f} scenes={macro['scenes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
