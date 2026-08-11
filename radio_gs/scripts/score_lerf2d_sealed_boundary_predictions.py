#!/usr/bin/env python3
"""Score a sealed LERF-2D RGB-boundary prediction bundle exactly once.

All prediction files and their scalar/RGB lineage are validated before the
first annotation JSON byte is opened.  The scorer never changes a mask,
chooses a candidate, tunes a threshold, or reads RGB pixels.  It only compares
the already sealed masks with the frozen exact-camera annotations.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.scripts.eval_ours_lerf2d_scalar_maps import (
    CANONICAL_TASK_ID,
    EXPECTED_REGISTRY_ROW,
    FrozenLerf2DContract,
    _bundle_member,
    _load_annotation,
    _load_json_bytes,
    _read_stable_regular_file,
    _require_mapping,
    _require_sha256,
    canonical_query_id,
    load_frozen_contract,
    write_result_no_clobber,
)
from radio_gs.scripts.eval_prerendered_lerf_features import _localization_hit
from radio_gs.scripts.materialize_lerf2d_rgb_boundary_predictions import (
    PREDICTION_ARTIFACT_TYPE,
    RgbBoundaryProtocolError,
    _policy_from_mapping,
)
from radio_gs.utils.immutable_artifacts import file_record, sha256_file


SCHEMA_VERSION = 1
RESULT_ARTIFACT_TYPE = "radio_gs_lerf2d_rgb_boundary_exact_evaluation"


def _validate_live_file_record(value: Any, *, label: str) -> Path:
    """Fail closed when a producer/authority file changed after sealing."""

    record = _require_mapping(value, label=label)
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RgbBoundaryProtocolError(f"{label}.path is malformed")
    unresolved = Path(raw_path).expanduser()
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise RgbBoundaryProtocolError(f"{label}.path is unavailable") from error
    if resolved != unresolved.absolute() or not resolved.is_file():
        raise RgbBoundaryProtocolError(f"{label}.path must be a regular non-symlink file")
    expected = _require_sha256(record.get("sha256"), label=f"{label}.sha256")
    if sha256_file(resolved) != expected:
        raise RgbBoundaryProtocolError(f"{label} changed after prediction")
    return resolved


def _load_prediction_array(
    path: Path,
    *,
    expected_sha256: str,
    expected_shape: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    encoded = _read_stable_regular_file(path, label=label)
    if sha256_file(path) != _require_sha256(expected_sha256, label=f"{label} SHA256"):
        raise RgbBoundaryProtocolError(f"{label} SHA256 differs")
    try:
        value = np.load(io.BytesIO(encoded), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RgbBoundaryProtocolError(f"{label} is not a valid npy array") from error
    if not isinstance(value, np.ndarray) or tuple(value.shape) != expected_shape:
        raise RgbBoundaryProtocolError(
            f"{label} shape differs: expected {expected_shape}, got {getattr(value, 'shape', None)}"
        )
    if value.dtype != np.uint8 or not bool(np.isin(value, [0, 1]).all()):
        raise RgbBoundaryProtocolError(f"{label} must be binary uint8")
    return value.astype(bool, copy=False)


def _validate_prediction_before_gt(
    prediction_manifest: Path,
    *,
    prediction_manifest_sha256: str,
    contract: FrozenLerf2DContract,
) -> tuple[dict[str, Any], str, dict[str, dict[str, np.ndarray]]]:
    """Validate the full sealed bundle before any annotation is opened."""

    manifest_path = prediction_manifest.resolve(strict=True)
    encoded = _read_stable_regular_file(manifest_path, label="prediction manifest")
    digest = sha256_file(manifest_path)
    if digest != _require_sha256(
        prediction_manifest_sha256, label="prediction manifest SHA256"
    ):
        raise RgbBoundaryProtocolError("prediction manifest SHA256 differs")
    payload = dict(_load_json_bytes(encoded, label="prediction manifest"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RgbBoundaryProtocolError("prediction schema_version differs")
    if payload.get("artifact_type") != PREDICTION_ARTIFACT_TYPE:
        raise RgbBoundaryProtocolError("prediction artifact_type differs")
    if payload.get("status") != "sealed_before_benchmark_mask_or_metric_access":
        raise RgbBoundaryProtocolError("prediction was not sealed before GT access")
    source_access = _require_mapping(payload.get("source_access"), label="source_access")
    expected_access = {
        "target_rgb_opened": True,
        "benchmark_annotation_json_opened": False,
        "benchmark_segmentation_opened": False,
        "benchmark_bboxes_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_metrics_opened": False,
        "candidate_selected_with_gt": False,
    }
    for key, expected in expected_access.items():
        if source_access.get(key) is not expected:
            raise RgbBoundaryProtocolError(f"prediction source_access.{key} differs")
    _policy_from_mapping(_require_mapping(payload.get("policy"), label="policy"))
    _validate_live_file_record(payload.get("implementation"), label="producer implementation")
    _validate_live_file_record(payload.get("rgb_root_authority"), label="RGB root authority")
    score_manifest = _require_mapping(payload.get("score_manifest"), label="score_manifest")
    if sha256_file(str(score_manifest.get("path"))) != _require_sha256(
        score_manifest.get("sha256"), label="source score manifest SHA256"
    ):
        raise RgbBoundaryProtocolError("source score manifest changed after prediction")
    freeze = _require_mapping(score_manifest.get("protocol_freeze"), label="protocol_freeze")
    if freeze.get("freeze_id") != contract.freeze_id or freeze.get("sha256") != contract.freeze_sha256:
        raise RgbBoundaryProtocolError("prediction protocol freeze differs")
    raw_scenes = _require_mapping(payload.get("scenes"), label="prediction scenes")
    if tuple(raw_scenes) != contract.scenes or set(raw_scenes) != set(contract.scenes):
        raise RgbBoundaryProtocolError("prediction scene cohort/order differs")

    predictions: dict[str, dict[str, np.ndarray]] = {}
    observed_queries = 0
    observed_frames = 0
    for scene in contract.scenes:
        frames = _require_mapping(
            _require_mapping(raw_scenes[scene], label=f"{scene}").get("frames"),
            label=f"{scene}.frames",
        )
        expected_frames = contract.frames_by_scene[scene]
        if tuple(frames) != expected_frames or set(frames) != set(expected_frames):
            raise RgbBoundaryProtocolError(f"{scene}: prediction camera cohort/order differs")
        predictions[scene] = {}
        for frame in expected_frames:
            observed_frames += 1
            entry = _require_mapping(frames[frame], label=f"{scene}/{frame}")
            query_ids = entry.get("query_ids")
            query_texts = entry.get("query_texts")
            resolution = entry.get("resolution_hw")
            queries = entry.get("queries")
            if (
                not isinstance(query_ids, list)
                or not isinstance(query_texts, list)
                or not isinstance(resolution, list)
                or len(resolution) != 2
                or not isinstance(queries, list)
                or len(queries) != len(query_ids)
                or len(query_texts) != len(query_ids)
            ):
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: prediction axes are malformed")
            for index, query in enumerate(queries):
                record = _require_mapping(query, label=f"{scene}/{frame}.queries[{index}]")
                if record.get("query_id") != query_ids[index] or record.get("query_text") != query_texts[index]:
                    raise RgbBoundaryProtocolError(f"{scene}/{frame}: query report axis differs")
                coords = record.get("localization_coords_yx")
                if not isinstance(coords, list) or any(
                    not isinstance(coord, list) or len(coord) != 2 for coord in coords
                ):
                    raise RgbBoundaryProtocolError(f"{scene}/{frame}: localization coords malformed")
            expected_shape = (len(query_ids), int(resolution[0]), int(resolution[1]))
            if entry.get("prediction_shape_qhw") != list(expected_shape):
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: prediction shape binding differs")
            mask_path = _bundle_member(
                manifest_path.parent,
                entry.get("prediction_file"),
                label=f"{scene}/{frame}.prediction_file",
            )
            predictions[scene][frame] = _load_prediction_array(
                mask_path,
                expected_sha256=_require_sha256(
                    entry.get("prediction_sha256"), label=f"{scene}/{frame}.prediction_sha256"
                ),
                expected_shape=expected_shape,
                label=f"{scene}/{frame} prediction",
            )
            # RGB is not reopened by this scorer; identity is nevertheless
            # checked before GT to keep the recorded target-RGB lineage live.
            rgb = _require_mapping(entry.get("target_rgb"), label=f"{scene}/{frame}.target_rgb")
            if sha256_file(str(rgb.get("path"))) != _require_sha256(
                rgb.get("sha256"), label=f"{scene}/{frame}.target_rgb.sha256"
            ):
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: target RGB changed after prediction")
            observed_queries += len(query_ids)
    if observed_frames != contract.labelled_frames or observed_queries != contract.queries:
        raise RgbBoundaryProtocolError("prediction cohort totals differ from frozen contract")
    return payload, digest, predictions


def score(
    prediction_manifest: Path,
    *,
    prediction_manifest_sha256: str,
    label_root: Path,
    contract: FrozenLerf2DContract,
) -> dict[str, Any]:
    """Score an already sealed bundle; GT access starts after validation."""

    payload, digest, predictions = _validate_prediction_before_gt(
        prediction_manifest,
        prediction_manifest_sha256=prediction_manifest_sha256,
        contract=contract,
    )
    raw_scenes = _require_mapping(payload["scenes"], label="prediction scenes")
    scene_rows: dict[str, Any] = {}
    all_ious: list[float] = []
    all_hits: list[bool] = []
    for scene in contract.scenes:
        raw_frames = _require_mapping(
            _require_mapping(raw_scenes[scene], label=scene).get("frames"),
            label=f"{scene}.frames",
        )
        scene_ious: list[float] = []
        scene_hits: list[bool] = []
        frame_rows: dict[str, Any] = {}
        for frame in contract.frames_by_scene[scene]:
            entry = _require_mapping(raw_frames[frame], label=f"{scene}/{frame}")
            # First benchmark mask/bbox byte access occurs here, after every
            # prediction, RGB identity, and source manifest was validated.
            camera_name, objects, resolution, annotation_sha = _load_annotation(
                label_root / scene / f"{frame}.json",
                scene=scene,
                frame=frame,
            )
            query_texts = [obj.query for obj in objects]
            query_ids = [
                canonical_query_id(scene, frame, index, query)
                for index, query in enumerate(query_texts)
            ]
            if (
                entry.get("annotation_sha256") != annotation_sha
                or entry.get("camera_name") != camera_name
                or entry.get("query_texts") != query_texts
                or entry.get("query_ids") != query_ids
                or entry.get("resolution_hw") != list(resolution)
            ):
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: frozen annotation binding differs")
            mask_qhw = predictions[scene][frame]
            query_reports = entry.get("queries")
            assert isinstance(query_reports, list)
            object_rows: list[dict[str, Any]] = []
            for index, obj in enumerate(objects):
                pred = mask_qhw[index]
                gt = np.asarray(obj.mask, dtype=bool)
                intersection = int(np.logical_and(pred, gt).sum())
                union = int(np.logical_or(pred, gt).sum())
                iou = float(intersection / union) if union else 0.0
                coords = torch.as_tensor(query_reports[index]["localization_coords_yx"]).long()
                hit = _localization_hit(coords, obj.bboxes)
                scene_ious.append(iou)
                scene_hits.append(hit)
                all_ious.append(iou)
                all_hits.append(hit)
                object_rows.append(
                    {
                        "query": obj.query,
                        "iou": iou,
                        "loc_hit": bool(hit),
                        "chosen_level": int(query_reports[index]["chosen_level"]),
                        "rgb_candidate_accepted": bool(
                            query_reports[index]["boundary"]["accepted"]
                        ),
                    }
                )
            frame_rows[frame] = {
                "objects": object_rows,
                "miou": float(np.mean([row["iou"] for row in object_rows])),
                "loc_acc": float(np.mean([row["loc_hit"] for row in object_rows])),
            }
        scene_rows[scene] = {
            "miou": float(np.mean(scene_ious)),
            "loc_acc": float(np.mean(scene_hits)),
            "objects": len(scene_ious),
            "frames": frame_rows,
        }
    scene_macro = {
        "miou": float(np.mean([row["miou"] for row in scene_rows.values()])),
        "loc_acc": float(np.mean([row["loc_acc"] for row in scene_rows.values()])),
        "scenes": len(scene_rows),
        "aggregation": "scene_equal_macro",
    }
    query_micro = {
        "miou": float(np.mean(all_ious)),
        "loc_acc": float(np.mean(all_hits)),
        "objects": len(all_ious),
        "aggregation": "query_weighted_micro",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RESULT_ARTIFACT_TYPE,
        "status": "complete_exact_frozen_protocol_evaluation",
        "method": payload.get("method"),
        "benchmark": "LERF-2D",
        "prediction_manifest": {
            "path": str(prediction_manifest.resolve(strict=True)),
            "sha256": digest,
        },
        "scorer_implementation": file_record(Path(__file__).resolve()),
        "protocol_authority": {
            "path": contract.freeze_path,
            "sha256": contract.freeze_sha256,
            "freeze_id": contract.freeze_id,
            "canonical_task_id": CANONICAL_TASK_ID,
            "registry_row": EXPECTED_REGISTRY_ROW,
        },
        "protocol_constraints": {
            "prediction_sealed_before_any_gt_open": True,
            "candidate_selected_with_gt": False,
            "threshold_selected_or_tuned": False,
            "target_rgb_used_by_prediction": True,
            "target_rgb_opened_by_scorer": False,
        },
        "scenes": scene_rows,
        "scene_macro": scene_macro,
        "query_micro": query_micro,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest-sha256", required=True)
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
    result = score(
        args.prediction_manifest,
        prediction_manifest_sha256=args.prediction_manifest_sha256,
        label_root=args.label_root,
        contract=contract,
    )
    write_result_no_clobber(args.output_json, result)
    print(json.dumps(result["scene_macro"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
