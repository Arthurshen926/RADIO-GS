#!/usr/bin/env python3
"""Score a sealed LERF-2D official-SAM3 batch only after full hash validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.scripts.eval_ours_lerf2d_scalar_maps import (
    CANONICAL_TASK_ID,
    EXPECTED_REGISTRY_ROW,
    FrozenLerf2DContract,
    _load_annotation,
    canonical_query_id,
    load_frozen_contract,
    write_result_no_clobber,
)
from radio_gs.scripts.eval_prerendered_lerf_features import _localization_hit
from radio_gs.scripts.materialize_lerf2d_coarse_prediction_receipt import (
    COARSE_ARTIFACT_TYPE,
)
from radio_gs.scripts.refine_lerf2d_coarse_receipt_official_sam3 import (
    FINAL_ARTIFACT_TYPE,
    Sam3Lerf2DProtocolError,
    _bundle_member,
    _load_binary_qhw,
    _require_mapping,
    _require_sha,
    _validate_live_file_record,
)
from radio_gs.utils.immutable_artifacts import file_record, sha256_file


SCHEMA_VERSION = 1
RESULT_ARTIFACT_TYPE = "radio_gs_lerf2d_official_sam3_box_exact_evaluation"
FIXED_POLICY = {
    "coarse_activation_kernel": 30,
    "coarse_mask_threshold": 0.5,
    "coarse_smooth_kernel": 7,
    "sam3_box_padding_pixels": 16,
    "sam3_resolution": 1008,
    "sam3_confidence_threshold": 0.0,
    "sam3_min_initial_iou": 0.05,
    "candidate_selector": "coarse_mask_iou_then_official_score_tie_break",
}


def _parse_scenes(raw: str, contract: FrozenLerf2DContract) -> tuple[str, ...]:
    if not raw.strip():
        return contract.scenes
    requested = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not requested or len(requested) != len(set(requested)):
        raise Sam3Lerf2DProtocolError("--scenes must be a unique comma list")
    if set(requested) - set(contract.scenes):
        raise Sam3Lerf2DProtocolError("--scenes contains a non-frozen scene")
    return tuple(scene for scene in contract.scenes if scene in set(requested))


def _validate_before_gt(
    prediction_receipt: Path,
    *,
    prediction_receipt_sha256: str,
    contract: FrozenLerf2DContract,
    scenes: tuple[str, ...],
) -> tuple[dict[str, Any], str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    receipt_path = prediction_receipt.expanduser().resolve(strict=True)
    digest = sha256_file(receipt_path)
    if digest != _require_sha(prediction_receipt_sha256, label="prediction receipt SHA256"):
        raise Sam3Lerf2DProtocolError("prediction receipt SHA256 differs")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    source = _require_mapping(payload.get("source_access"), label="source_access")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_type") != FINAL_ARTIFACT_TYPE
        or payload.get("status") != "sealed_before_benchmark_mask_or_metric_access"
        or payload.get("policy") != FIXED_POLICY
        or source.get("target_rgb_opened") is not True
        or any(
            source.get(key) is not False
            for key in (
                "benchmark_annotation_json_opened",
                "benchmark_segmentation_opened",
                "benchmark_bboxes_opened",
                "benchmark_masks_opened",
                "benchmark_metrics_opened",
                "candidate_selected_with_gt",
            )
        )
    ):
        raise Sam3Lerf2DProtocolError("prediction receipt violates the pre-GT contract")
    _validate_live_file_record(payload.get("implementation"), label="SAM3 producer")
    _validate_live_file_record(payload.get("rgb_root_authority"), label="RGB authority")
    coarse_record = _require_mapping(
        payload.get("coarse_prediction_receipt"), label="coarse_prediction_receipt"
    )
    coarse_path = Path(str(coarse_record.get("path", ""))).resolve(strict=True)
    if sha256_file(coarse_path) != _require_sha(
        coarse_record.get("sha256"), label="coarse receipt SHA256"
    ):
        raise Sam3Lerf2DProtocolError("coarse receipt changed after SAM3 sealing")
    coarse_payload = json.loads(coarse_path.read_text(encoding="utf-8"))
    coarse_source = _require_mapping(coarse_payload.get("source_access"), label="coarse access")
    if (
        coarse_payload.get("artifact_type") != COARSE_ARTIFACT_TYPE
        or coarse_payload.get("status")
        != "coarse_predictions_sealed_before_target_rgb_or_gt_access"
        or coarse_source.get("target_rgb_opened") is not False
        or any(
            coarse_source.get(key) is not False
            for key in (
                "benchmark_annotation_json_opened",
                "benchmark_segmentation_opened",
                "benchmark_bboxes_opened",
                "benchmark_masks_opened",
                "benchmark_metrics_opened",
                "candidate_selected_with_gt",
            )
        )
    ):
        raise Sam3Lerf2DProtocolError("coarse receipt violates the pre-RGB/GT contract")
    _validate_live_file_record(coarse_payload.get("implementation"), label="coarse producer")
    score_manifest = _require_mapping(
        coarse_payload.get("score_manifest"), label="score_manifest"
    )
    score_path = Path(str(score_manifest.get("path", ""))).resolve(strict=True)
    if sha256_file(score_path) != _require_sha(
        score_manifest.get("sha256"), label="score manifest SHA256"
    ):
        raise Sam3Lerf2DProtocolError("scalar score manifest changed")
    freeze = _require_mapping(score_manifest.get("protocol_freeze"), label="protocol_freeze")
    if freeze.get("freeze_id") != contract.freeze_id or freeze.get("sha256") != contract.freeze_sha256:
        raise Sam3Lerf2DProtocolError("prediction protocol freeze differs")
    raw_scenes = _require_mapping(payload.get("scenes"), label="prediction scenes")
    coarse_scenes = _require_mapping(coarse_payload.get("scenes"), label="coarse scenes")
    if tuple(raw_scenes) != scenes or tuple(coarse_scenes) != scenes:
        raise Sam3Lerf2DProtocolError("prediction scene cohort/order differs")
    arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    observed_queries = 0
    observed_accepted = 0
    for scene in scenes:
        final_frames = _require_mapping(
            _require_mapping(raw_scenes[scene], label=scene).get("frames"),
            label=f"{scene}.frames",
        )
        coarse_frames = _require_mapping(
            _require_mapping(coarse_scenes[scene], label=f"coarse {scene}").get("frames"),
            label=f"coarse {scene}.frames",
        )
        expected_frames = contract.frames_by_scene[scene]
        if tuple(final_frames) != expected_frames or tuple(coarse_frames) != expected_frames:
            raise Sam3Lerf2DProtocolError(f"{scene}: camera cohort/order differs")
        arrays[scene] = {}
        for frame in expected_frames:
            entry = _require_mapping(final_frames[frame], label=f"{scene}/{frame}")
            coarse_entry = _require_mapping(
                coarse_frames[frame], label=f"coarse {scene}/{frame}"
            )
            shape = entry.get("prediction_shape_qhw")
            query_ids, query_texts, query_rows = (
                entry.get("query_ids"),
                entry.get("query_texts"),
                entry.get("queries"),
            )
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or not isinstance(query_ids, list)
                or not isinstance(query_texts, list)
                or not isinstance(query_rows, list)
                or len(query_ids) != int(shape[0])
                or len(query_texts) != int(shape[0])
                or len(query_rows) != int(shape[0])
                or coarse_entry.get("query_ids") != query_ids
                or coarse_entry.get("query_texts") != query_texts
                or coarse_entry.get("prediction_shape_qhw") != shape
            ):
                raise Sam3Lerf2DProtocolError(f"{scene}/{frame}: query axes differ")
            expected_shape = tuple(int(value) for value in shape)
            final_path = _bundle_member(
                receipt_path.parent,
                entry.get("prediction_file"),
                label=f"{scene}/{frame}.prediction_file",
            )
            final = _load_binary_qhw(
                final_path,
                expected_sha256=str(entry.get("prediction_sha256")),
                expected_shape=expected_shape,
                label=f"{scene}/{frame} prediction",
            )
            coarse_root = Path(str(entry.get("coarse_prediction_receipt_root", ""))).resolve(
                strict=True
            )
            if coarse_root != coarse_path.parent:
                raise Sam3Lerf2DProtocolError(f"{scene}/{frame}: coarse root differs")
            coarse_mask_path = _bundle_member(
                coarse_root,
                coarse_entry.get("coarse_prediction_file"),
                label=f"{scene}/{frame}.coarse_prediction_file",
            )
            initial = _load_binary_qhw(
                coarse_mask_path,
                expected_sha256=str(coarse_entry.get("coarse_prediction_sha256")),
                expected_shape=expected_shape,
                label=f"{scene}/{frame} coarse prediction",
            )
            rgb_record = _require_mapping(entry.get("target_rgb"), label="target_rgb")
            rgb_path = Path(str(rgb_record.get("path", ""))).resolve(strict=True)
            if sha256_file(rgb_path) != _require_sha(
                rgb_record.get("sha256"), label=f"{scene}/{frame} RGB SHA256"
            ):
                raise Sam3Lerf2DProtocolError(f"{scene}/{frame}: RGB changed")
            for index, query in enumerate(query_rows):
                row = _require_mapping(query, label=f"{scene}/{frame}.queries[{index}]")
                if row.get("query_id") != query_ids[index] or row.get("query_text") != query_texts[index]:
                    raise Sam3Lerf2DProtocolError(f"{scene}/{frame}: query report differs")
                boundary = _require_mapping(row.get("boundary"), label="boundary")
                observed_accepted += int(bool(boundary.get("accepted")))
            arrays[scene][frame] = (final, initial)
            observed_queries += len(query_ids)
    cohort = _require_mapping(payload.get("cohort"), label="cohort")
    if (
        cohort.get("scenes") != list(scenes)
        or int(cohort.get("queries", -1)) != observed_queries
        or int(cohort.get("sam3_candidate_accepted", -1)) != observed_accepted
    ):
        raise Sam3Lerf2DProtocolError("prediction cohort totals differ")
    payload["_coarse_receipt_path"] = str(coarse_path)
    payload["_coarse_receipt_sha256"] = sha256_file(coarse_path)
    return payload, digest, arrays


def score(
    prediction_receipt: Path,
    *,
    prediction_receipt_sha256: str,
    label_root: Path,
    contract: FrozenLerf2DContract,
    scenes: tuple[str, ...],
) -> dict[str, Any]:
    payload, digest, arrays = _validate_before_gt(
        prediction_receipt,
        prediction_receipt_sha256=prediction_receipt_sha256,
        contract=contract,
        scenes=scenes,
    )
    raw_scenes = _require_mapping(payload.get("scenes"), label="prediction scenes")
    scene_rows: dict[str, Any] = {}
    all_final, all_initial, all_hits = [], [], []
    for scene in scenes:
        frames = _require_mapping(
            _require_mapping(raw_scenes[scene], label=scene).get("frames"),
            label=f"{scene}.frames",
        )
        scene_final, scene_initial, scene_hits = [], [], []
        frame_rows = {}
        for frame in contract.frames_by_scene[scene]:
            entry = _require_mapping(frames[frame], label=f"{scene}/{frame}")
            camera_name, objects, resolution, annotation_sha = _load_annotation(
                label_root / scene / f"{frame}.json", scene=scene, frame=frame
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
                raise Sam3Lerf2DProtocolError(f"{scene}/{frame}: annotation binding differs")
            final_qhw, initial_qhw = arrays[scene][frame]
            query_rows = entry.get("queries")
            assert isinstance(query_rows, list)
            object_rows = []
            for index, obj in enumerate(objects):
                gt = np.asarray(obj.mask, dtype=bool)
                final, initial = final_qhw[index], initial_qhw[index]
                final_union = int(np.logical_or(final, gt).sum())
                initial_union = int(np.logical_or(initial, gt).sum())
                final_iou = (
                    float(np.logical_and(final, gt).sum()) / final_union if final_union else 0.0
                )
                initial_iou = (
                    float(np.logical_and(initial, gt).sum()) / initial_union
                    if initial_union
                    else 0.0
                )
                coords = torch.as_tensor(query_rows[index]["localization_coords_yx"]).long()
                hit = bool(_localization_hit(coords, obj.bboxes))
                boundary = _require_mapping(query_rows[index].get("boundary"), label="boundary")
                object_rows.append(
                    {
                        "query": obj.query,
                        "initial_iou": initial_iou,
                        "iou": final_iou,
                        "delta_iou": final_iou - initial_iou,
                        "loc_hit": hit,
                        "chosen_level": int(query_rows[index]["chosen_level"]),
                        "sam3_attempted": bool(boundary.get("attempted", False)),
                        "sam3_accepted": bool(boundary.get("accepted", False)),
                        "sam3_report": dict(boundary),
                    }
                )
                scene_final.append(final_iou)
                scene_initial.append(initial_iou)
                scene_hits.append(hit)
                all_final.append(final_iou)
                all_initial.append(initial_iou)
                all_hits.append(hit)
            frame_rows[frame] = {
                "initial_miou": float(np.mean([row["initial_iou"] for row in object_rows])),
                "miou": float(np.mean([row["iou"] for row in object_rows])),
                "delta_miou": float(np.mean([row["delta_iou"] for row in object_rows])),
                "loc_acc": float(np.mean([row["loc_hit"] for row in object_rows])),
                "objects": object_rows,
            }
        scene_rows[scene] = {
            "initial_miou": float(np.mean(scene_initial)),
            "miou": float(np.mean(scene_final)),
            "delta_miou": float(np.mean(scene_final) - np.mean(scene_initial)),
            "loc_acc": float(np.mean(scene_hits)),
            "objects": len(scene_final),
            "frames": frame_rows,
        }
    scene_macro = {
        "initial_miou": float(np.mean([row["initial_miou"] for row in scene_rows.values()])),
        "miou": float(np.mean([row["miou"] for row in scene_rows.values()])),
        "delta_miou": float(
            np.mean([row["miou"] for row in scene_rows.values()])
            - np.mean([row["initial_miou"] for row in scene_rows.values()])
        ),
        "loc_acc": float(np.mean([row["loc_acc"] for row in scene_rows.values()])),
        "scenes": len(scene_rows),
        "aggregation": "scene_equal_macro",
    }
    query_micro = {
        "initial_miou": float(np.mean(all_initial)),
        "miou": float(np.mean(all_final)),
        "delta_miou": float(np.mean(all_final) - np.mean(all_initial)),
        "loc_acc": float(np.mean(all_hits)),
        "objects": len(all_final),
        "aggregation": "query_weighted_micro",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RESULT_ARTIFACT_TYPE,
        "status": "complete_after_validated_pre_gt_prediction_receipt",
        "method": payload.get("method"),
        "benchmark": "LERF-2D",
        "prediction_receipt": {
            "path": str(prediction_receipt.resolve(strict=True)),
            "sha256": digest,
        },
        "coarse_prediction_receipt": {
            "path": payload["_coarse_receipt_path"],
            "sha256": payload["_coarse_receipt_sha256"],
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
            "fixed_global_sam3_policy": True,
        },
        "scenes": scene_rows,
        "scene_macro": scene_macro,
        "query_micro": query_micro,
    }


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-receipt", type=Path, required=True)
    parser.add_argument("--prediction-receipt-sha256", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--scenes", default="")
    parser.add_argument(
        "--protocol-freeze",
        type=Path,
        default=repo_root / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = load_frozen_contract(
        args.protocol_freeze, repo_root=args.repo_root, verify_hashes=True
    )
    scenes = _parse_scenes(args.scenes, contract)
    result = score(
        args.prediction_receipt,
        prediction_receipt_sha256=args.prediction_receipt_sha256,
        label_root=args.label_root,
        contract=contract,
        scenes=scenes,
    )
    write_result_no_clobber(args.output_json, result)
    print(json.dumps(result["scene_macro"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
