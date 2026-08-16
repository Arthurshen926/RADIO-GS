#!/usr/bin/env python3
"""Refine sealed NVOS Method-v1 field masks with a target-blind SAM3 box."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from radio_gs.evaluation.promptable_segmentation import resize_mask_nearest
from radio_gs.five_benchmark_method_v1 import METHOD_ID
from radio_gs.querying.transient_rgb_sam import (
    FROZEN_POLICY,
    deterministic_signed_point_trials,
)
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    sam3_autocast_context,
    set_requested_cuda_device,
)
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import (
    DEFAULT_METHOD_AUTHORITY,
    DEFAULT_SAM3_CHECKPOINT,
    FROZEN_SAM3_CHECKPOINT_SHA256,
    SAM_HEIGHT,
    SAM_WIDTH,
    _sha256,
    load_signed_field_prompt,
)
from radio_gs.scripts.refine_lerf_coarse_receipt_official_sam3 import (
    choose_candidate,
    mask_to_box,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    REPO_ROOT
    / "paper/artifacts/nvos_method_v1_field_box_sam3_candidate_preregistration_20260816.json"
)
CANDIDATE_ID = "nvos-method-v1-field-box-sam3-pad16-overlap-v1"
SIGNED_POINT_PREREGISTRATION = (
    REPO_ROOT
    / "paper/artifacts/nvos_method_v1_field_box_signed_points_sam3_candidate_preregistration_20260816.json"
)
SIGNED_POINT_CANDIDATE_ID = "nvos-method-v1-field-box-sam3-signed-point-selector-v1"


def choose_candidate_by_signed_points(
    signed_margin: np.ndarray,
    coarse_mask: np.ndarray,
    candidate_masks: np.ndarray,
    *,
    scores: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a SAM candidate by sealed signed evidence, without target GT."""

    coarse = np.asarray(coarse_mask, dtype=bool)
    masks = np.asarray(candidate_masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    report: dict[str, Any] = {
        "accepted": False,
        "fallback_reason": "",
        "candidate_count": int(masks.shape[0]) if masks.ndim == 3 else 0,
        "selected_index": -1,
        "signed_evidence_score": 0.0,
        "positive_point_inclusion": 0.0,
        "negative_point_exclusion": 0.0,
        "best_initial_overlap": 0.0,
        "selected_score": 0.0,
        "minimum_signed_evidence_score": 0.5,
    }
    if masks.ndim != 3 or masks.shape[-2:] != coarse.shape or not masks.shape[0]:
        report["fallback_reason"] = "candidate_shape_mismatch_or_empty"
        report["candidate_shape"] = list(masks.shape)
        return coarse.copy(), report
    margin = np.asarray(signed_margin, dtype=np.float32)
    points, labels = deterministic_signed_point_trials(
        np.maximum(margin, 0.0),
        np.maximum(-margin, 0.0),
        image_shape=coarse.shape,
        policy=FROZEN_POLICY,
    )
    xy = np.rint(points).astype(np.int64)
    xy[..., 0] = np.clip(xy[..., 0], 0, coarse.shape[1] - 1)
    xy[..., 1] = np.clip(xy[..., 1], 0, coarse.shape[0] - 1)
    positive = labels == 1
    negative = labels == 0
    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if score_values.shape != (masks.shape[0],):
        score_values = np.zeros(masks.shape[0], dtype=np.float32)
    best: tuple[float, float, float, int, float, float] | None = None
    for index, raw_candidate in enumerate(masks):
        candidate = np.asarray(raw_candidate) > 0
        sampled = candidate[xy[..., 1], xy[..., 0]]
        positive_inclusion = float(sampled[positive].mean())
        negative_exclusion = float((~sampled[negative]).mean())
        evidence = 0.5 * (positive_inclusion + negative_exclusion)
        union = int(np.logical_or(candidate, coarse).sum())
        overlap = (
            float(np.logical_and(candidate, coarse).sum() / union) if union else 0.0
        )
        proposal = (
            evidence,
            overlap,
            float(score_values[index]),
            -index,
            positive_inclusion,
            negative_exclusion,
        )
        if best is None or proposal[:4] > best[:4]:
            best = proposal
    assert best is not None
    selected_index = -int(best[3])
    report.update(
        {
            "selected_index": selected_index,
            "signed_evidence_score": float(best[0]),
            "best_initial_overlap": float(best[1]),
            "selected_score": float(best[2]),
            "positive_point_inclusion": float(best[4]),
            "negative_point_exclusion": float(best[5]),
        }
    )
    if best[0] < 0.5:
        report["fallback_reason"] = "low_signed_evidence_score"
        return coarse.copy(), report
    report["accepted"] = True
    report["fallback_reason"] = "accepted"
    return (np.asarray(masks[selected_index]) > 0).astype(bool), report


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_numpy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def predict(args: argparse.Namespace) -> dict[str, Any]:
    scene_ids = [str(value) for value in args.scene_ids]
    if len(scene_ids) != 8 or len(set(scene_ids)) != 8:
        raise ValueError("field-box candidate requires the ordered frozen full8")
    signed_point_selection = args.selection_mode == "signed_points"
    candidate_id = SIGNED_POINT_CANDIDATE_ID if signed_point_selection else CANDIDATE_ID
    preregistration_path = (
        SIGNED_POINT_PREREGISTRATION if signed_point_selection else PREREGISTRATION
    )
    expected_status = (
        "frozen_before_first_signed_point_selected_box_candidate_prediction"
        if signed_point_selection
        else "frozen_before_first_field_box_sam3_candidate_prediction"
    )
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if (
        preregistration.get("status") != expected_status
        or preregistration.get("candidate_id") != candidate_id
        or preregistration.get("access_at_freeze", {}).get(
            (
                "new_candidate_prediction_generated"
                if signed_point_selection
                else "candidate_prediction_generated"
            )
        )
        is not False
    ):
        raise ValueError("field-box candidate preregistration differs")
    sources = [
        load_signed_field_prompt(
            dataset_manifest_path=args.manifest,
            prompt_manifest_path=args.signed_field_prompt_manifest,
            method_authority_path=args.method_authority,
            scene_id=scene_id,
        )
        for scene_id in scene_ids
    ]
    protocol_hashes = {str(source["protocol_hash"]) for source in sources}
    if len(protocol_hashes) != 1:
        raise ValueError("full8 signed prompts disagree on protocol hash")
    output_root = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_root / "prediction_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256):
        raise ValueError("official SAM3 checkpoint SHA-256 differs")
    set_requested_cuda_device(args.device)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint),
        device=args.device,
        confidence_threshold=0.0,
        dtype="float32",
        resolution=SAM_WIDTH,
        point_only=False,
        build_on_cpu=True,
    )
    amp_dtype = torch.bfloat16 if str(args.device).startswith("cuda") else None
    predictions: dict[str, dict[str, str]] = {}
    prediction_hashes: dict[str, dict[str, str]] = {}
    receipts: list[dict[str, str]] = []
    started = time.time()
    for source in sources:
        scene_id = str(source["scene_id"])
        frame_id = str(source["frame_id"])
        target = Image.open(source["target_rgb_path"]).convert("RGB")
        original_size = list(target.size)
        target = target.resize((SAM_WIDTH, SAM_HEIGHT), Image.Resampling.LANCZOS)
        coarse = np.asarray(source["signed_margin"], dtype=np.float32) >= 0.0
        coarse = resize_mask_nearest(coarse, (SAM_HEIGHT, SAM_WIDTH)).astype(bool)
        box = mask_to_box(coarse, padding_pixels=16)
        report: dict[str, Any] = {
            "attempted": box is not None,
            "accepted": False,
            "fallback_reason": "empty_initial_mask" if box is None else "",
            "candidate_count": 0,
            "selected_index": -1,
            "best_initial_overlap": 0.0,
            "selected_score": 0.0,
            "box_prompt_cxcywh_norm": box,
            "box_padding_pixels": 16,
            "minimum_coarse_overlap": 0.05,
        }
        if box is None:
            refined = coarse.copy()
        else:
            with sam3_autocast_context(args.device, amp_dtype):
                state = processor.set_image(target)
                output = processor.add_geometric_prompt(box, True, dict(state))
            masks = output.get("masks")
            if masks is None:
                logits = output.get("masks_logits")
                masks = None if logits is None else logits.float() > 0.0
            if masks is None:
                refined = coarse.copy()
                report["fallback_reason"] = "missing_masks_and_logits"
            else:
                masks_np = (
                    masks.detach().cpu().numpy()
                    if torch.is_tensor(masks)
                    else np.asarray(masks)
                )
                scores = output.get("scores")
                scores_np = (
                    scores.detach().float().cpu().numpy()
                    if torch.is_tensor(scores)
                    else np.asarray(
                        scores if scores is not None else [], dtype=np.float32
                    )
                )
                if signed_point_selection:
                    refined, selected = choose_candidate_by_signed_points(
                        source["signed_margin"],
                        coarse,
                        masks_np,
                        scores=scores_np,
                    )
                else:
                    refined, selected = choose_candidate(
                        coarse, masks_np, scores=scores_np, min_initial_iou=0.05
                    )
                report.update(selected)
        continuous_margin = np.where(refined, 0.5, -0.5).astype(np.float32)
        relative = Path("scores") / scene_id / f"{frame_id}.npy"
        score_path = output_root / relative
        score_sha = _write_numpy(score_path, continuous_margin)
        receipt = {
            "schema_version": 1,
            "artifact_type": "radio_gs_nvos_method_v1_field_box_sam3_receipt",
            "candidate_id": candidate_id,
            "method_id": METHOD_ID,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "signed_field_prompt": {
                "path": str(source["signed_margin_path"]),
                "sha256": source["signed_margin_sha256"],
                "sealed_before_target_rgb_open": True,
            },
            "field": source["feature_render_authority"],
            "target_rgb": {
                "path": str(source["target_rgb_path"]),
                "sha256": source["target_rgb_sha256"],
                "original_size_wh": original_size,
                "sam_size_wh": [SAM_WIDTH, SAM_HEIGHT],
            },
            "coarse": {
                "threshold": 0.0,
                "foreground_pixels": int(coarse.sum()),
                "foreground_fraction": float(coarse.mean()),
            },
            "sam3": report,
            "output": {
                "path": str(score_path),
                "sha256": score_sha,
                "foreground_pixels": int(refined.sum()),
                "foreground_fraction": float(refined.mean()),
            },
            "authorities": {
                "dataset_manifest": str(source["dataset_manifest"]),
                "dataset_manifest_sha256": source["dataset_manifest_sha256"],
                "protocol_hash": source["protocol_hash"],
                "method_authority": str(source["method_authority"]),
                "method_authority_sha256": source["method_authority_sha256"],
                "preregistration": str(preregistration_path),
                "preregistration_sha256": _sha256(preregistration_path),
                "official_sam3_checkpoint": str(checkpoint),
                "official_sam3_checkpoint_sha256": checkpoint_sha,
            },
            "safety": {
                "target_rgb_opened": True,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "candidate_selected_by_coarse_field_overlap": (
                    not signed_point_selection
                ),
                "candidate_selected_by_signed_field_points": (signed_point_selection),
                "coarse_field_overlap_used_as_tie_break": signed_point_selection,
                "target_metric_used_for_selection": False,
                "graph_used": False,
                "connected_component_used": False,
            },
        }
        receipt_path = output_root / "receipts" / f"{scene_id}.json"
        _write_json(receipt_path, receipt)
        predictions[scene_id] = {frame_id: relative.as_posix()}
        prediction_hashes[scene_id] = {frame_id: score_sha}
        receipts.append(
            {
                "scene_id": scene_id,
                "path": str(receipt_path),
                "sha256": _sha256(receipt_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "promptable_nvs_method_v1_field_box_sam3_predictions",
        "candidate_id": candidate_id,
        "method_id": METHOD_ID,
        "protocol_hash": next(iter(protocol_hashes)),
        "scene_order": scene_ids,
        "prediction_root": ".",
        "predictions": predictions,
        "prediction_sha256": prediction_hashes,
        "receipts": receipts,
        "elapsed_seconds": float(time.time() - started),
        "evaluation_performed": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
        "all_eight_predictions_sealed": True,
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "prediction_manifest": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--signed-field-prompt-manifest", required=True)
    parser.add_argument("--scene-id", action="append", dest="scene_ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-authority", default=str(DEFAULT_METHOD_AUTHORITY))
    parser.add_argument("--checkpoint", default=str(DEFAULT_SAM3_CHECKPOINT))
    parser.add_argument(
        "--expected-checkpoint-sha256", default=FROZEN_SAM3_CHECKPOINT_SHA256
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--selection-mode",
        choices=("coarse_iou", "signed_points"),
        default="coarse_iou",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = predict(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "prediction_manifest": report["prediction_manifest"],
                "scene_count": len(report["predictions"]),
                "evaluation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
