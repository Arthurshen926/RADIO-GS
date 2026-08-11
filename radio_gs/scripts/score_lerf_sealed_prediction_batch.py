#!/usr/bin/env python3
"""Score a LERF prediction batch only after validating its immutable receipt."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

sys.path.insert(0, ".")

from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    bootstrap_mean_ci,
    boundary_f_score,
    mask_overlap_stats,
    sha256_file,
    summarize_initial_iou_buckets,
    summarize_ious,
    trimap_iou,
)
from radio_gs.scripts.eval_lerf_grounding import (
    build_gt_masks,
    load_lerf_ovs_labels,
    resolve_lerf_label_dir,
)


def _load_mask(path: Path, *, height: int, width: int) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(f"prediction mask is unreadable: {path}")
    if value.shape != (height, width):
        raise ValueError(
            f"prediction mask shape changed: {value.shape} vs {(height, width)}: {path}"
        )
    return value > 127


def validate_prediction_receipt(
    path: str | Path,
    *,
    expected_sha256: str = "",
) -> Dict[str, Any]:
    receipt_path = Path(path).expanduser().resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    observed_sha256 = sha256_file(receipt_path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise ValueError(
            f"prediction receipt SHA256 mismatch: {observed_sha256} vs {expected_sha256}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact_type = receipt.get("artifact_type")
    allowed_artifacts = {
        "lerf_target_rgb_assisted_pre_metric_prediction_receipt_v1",
        "lerf_strict_feature_field_pre_metric_prediction_receipt_v1",
    }
    strict_feature_field = (
        artifact_type == "lerf_strict_feature_field_pre_metric_prediction_receipt_v1"
    )
    if (
        artifact_type not in allowed_artifacts
        or receipt.get("status")
        != "sealed_before_target_mask_rasterization_and_metric"
        or receipt.get("target_mask_rasterized_before_seal") is not False
        or receipt.get("target_metric_computed_before_seal") is not False
        or receipt.get("target_rgb_opened") is not (not strict_feature_field)
        or receipt.get("target_annotation_inventory_opened") is not True
        or receipt.get("target_annotation_coordinates_loaded") is not False
    ):
        raise ValueError("prediction receipt does not satisfy the pre-metric contract")
    predictions = receipt.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != int(
        receipt.get("prediction_count", -1)
    ):
        raise ValueError("prediction receipt inventory is malformed")
    identities = set()
    for row in predictions:
        identity = (int(row["frame_id"]), str(row["category"]))
        if identity in identities:
            raise ValueError(f"duplicate prediction identity: {identity}")
        identities.add(identity)
        for prefix in ("prediction", "coarse_prediction"):
            mask_path = Path(str(row[f"{prefix}_path"])).expanduser().resolve()
            if not mask_path.is_file() or sha256_file(mask_path) != row[f"{prefix}_sha256"]:
                raise ValueError(f"sealed {prefix} changed after receipt: {identity}")
            mask = _load_mask(
                mask_path,
                height=int(row["height"]),
                width=int(row["width"]),
            )
            expected_pixels = int(
                row["prediction_pixels"]
                if prefix == "prediction"
                else row["coarse_prediction_pixels"]
            )
            if int(mask.sum()) != expected_pixels:
                raise ValueError(f"sealed {prefix} pixel count changed: {identity}")
    receipt["_receipt_path"] = str(receipt_path)
    receipt["_receipt_sha256"] = observed_sha256
    receipt["_receipt_mtime_ns"] = int(receipt_path.stat().st_mtime_ns)
    return receipt


def score_prediction_receipt(
    receipt: Dict[str, Any],
    *,
    label_dir: str,
) -> Dict[str, Any]:
    scene = str(receipt["scene"])
    frame_annotations, categories, img_h, img_w = load_lerf_ovs_labels(label_dir, scene)
    ious: List[float] = []
    initial_ious: List[float] = []
    boundary_scores: List[float] = []
    initial_boundary_scores: List[float] = []
    trimap_scores: List[float] = []
    initial_trimap_scores: List[float] = []
    per_category: Dict[str, List[float]] = {category: [] for category in categories}
    query_details: List[Dict[str, Any]] = []
    for row in receipt["predictions"]:
        frame_id = int(row["frame_id"])
        category = str(row["category"])
        if frame_id not in frame_annotations or category not in per_category:
            raise ValueError(f"sealed prediction is outside the benchmark inventory: {(frame_id, category)}")
        gt_masks = build_gt_masks(frame_annotations[frame_id], categories, img_h, img_w)
        gt = gt_masks[category]
        pred = _load_mask(
            Path(row["prediction_path"]), height=int(row["height"]), width=int(row["width"])
        )
        initial = _load_mask(
            Path(row["coarse_prediction_path"]),
            height=int(row["height"]),
            width=int(row["width"]),
        )
        overlap = mask_overlap_stats(pred, gt)
        initial_overlap = mask_overlap_stats(initial, gt)
        iou = float(overlap["iou"])
        initial_iou = float(initial_overlap["iou"])
        boundary = float(boundary_f_score(pred, gt))
        initial_boundary = float(boundary_f_score(initial, gt))
        trimap = float(trimap_iou(pred, gt))
        initial_trimap = float(trimap_iou(initial, gt))
        sam3_report = dict(row.get("sam3_report", {}))
        ious.append(iou)
        initial_ious.append(initial_iou)
        boundary_scores.append(boundary)
        initial_boundary_scores.append(initial_boundary)
        trimap_scores.append(trimap)
        initial_trimap_scores.append(initial_trimap)
        per_category[category].append(iou)
        query_details.append(
            {
                "frame": f"frame_{frame_id:05d}",
                "frame_id": frame_id,
                "category": category,
                "iou": iou,
                "initial_iou": initial_iou,
                "delta_iou": iou - initial_iou,
                "boundary_f": boundary,
                "initial_boundary_f": initial_boundary,
                "delta_boundary_f": boundary - initial_boundary,
                "trimap_iou": trimap,
                "initial_trimap_iou": initial_trimap,
                "delta_trimap_iou": trimap - initial_trimap,
                "pred_pixels": int(overlap["pred_pixels"]),
                "initial_pred_pixels": int(initial_overlap["pred_pixels"]),
                "gt_pixels": int(overlap["gt_pixels"]),
                "intersection_pixels": int(overlap["intersection_pixels"]),
                "union_pixels": int(overlap["union_pixels"]),
                "sam3_attempted": bool(sam3_report.get("attempted", False)),
                "sam3_accepted": bool(sam3_report.get("accepted", False)),
                "sam3_report": sam3_report,
            }
        )
    if not ious:
        raise RuntimeError("sealed prediction batch contains no scorable predictions")
    summary = summarize_ious(ious)
    initial_summary = summarize_ious(initial_ious)
    summary.update(
        {
            "boundary_f": float(np.asarray(boundary_scores, dtype=np.float32).mean()),
            "trimap_iou": float(np.asarray(trimap_scores, dtype=np.float32).mean()),
            "initial_miou": float(initial_summary["miou"]),
            "initial_acc025": float(initial_summary["acc025"]),
            "initial_acc050": float(initial_summary["acc050"]),
            "initial_boundary_f": float(
                np.asarray(initial_boundary_scores, dtype=np.float32).mean()
            ),
            "initial_trimap_iou": float(
                np.asarray(initial_trimap_scores, dtype=np.float32).mean()
            ),
            "delta_miou": float(np.asarray(ious).mean() - np.asarray(initial_ious).mean()),
            "delta_boundary_f": float(
                np.asarray(boundary_scores).mean()
                - np.asarray(initial_boundary_scores).mean()
            ),
            "delta_trimap_iou": float(
                np.asarray(trimap_scores).mean() - np.asarray(initial_trimap_scores).mean()
            ),
            "per_category": {
                category: summarize_ious(values)
                for category, values in per_category.items()
                if values
            },
            "query_details": query_details,
            "initial_iou_buckets": summarize_initial_iou_buckets(query_details),
            "bootstrap_miou": bootstrap_mean_ci(ious),
            "bootstrap_initial_miou": bootstrap_mean_ci(initial_ious),
        }
    )
    return summary


def _write_no_clobber(path: Path, payload: Dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite a different score report: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale score temporary exists: {temporary}")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-receipt", required=True)
    parser.add_argument("--expected-receipt-sha256", default="")
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    label_dir = resolve_lerf_label_dir(args.label_dir)
    receipt = validate_prediction_receipt(
        args.prediction_receipt,
        expected_sha256=args.expected_receipt_sha256,
    )
    metrics = score_prediction_receipt(receipt, label_dir=label_dir)
    output = Path(args.output).expanduser().resolve()
    payload = {
        "schema_version": 1,
        "artifact_type": (
            "lerf_strict_feature_field_sealed_prediction_metric_v1"
            if receipt.get("artifact_type")
            == "lerf_strict_feature_field_pre_metric_prediction_receipt_v1"
            else "lerf_target_rgb_assisted_sealed_prediction_metric_v1"
        ),
        "status": "complete_after_validated_prediction_receipt",
        "scene": receipt["scene"],
        "protocol": receipt["protocol"],
        "selection": receipt["selection"],
        "prediction_receipt": receipt["_receipt_path"],
        "prediction_receipt_sha256": receipt["_receipt_sha256"],
        "prediction_receipt_mtime_ns": receipt["_receipt_mtime_ns"],
        "metric_written_unix_ns": time.time_ns(),
        "target_metric_computed_after_receipt_validation": True,
        "metrics": metrics,
    }
    _write_no_clobber(output, payload)
    print(json.dumps({"output": str(output), "miou": metrics["miou"], "n": metrics["n"]}))


if __name__ == "__main__":
    main()
