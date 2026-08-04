#!/usr/bin/env python3
"""Label-aware score-quality audit around the frozen LERF3D evaluator.

The primitive score tensor, its complete authority receipt, the renderer
checkpoint, and this audit's method configuration are validated and hashed
before the frozen evaluator is allowed to open benchmark annotations.  AP and
oracle-threshold statistics are explicitly diagnostic: target labels are used
only after the continuous score field has been frozen.

This module does not change the frozen evaluator or the formal protocol.  Its
runtime hook only adds continuous-score diagnostics to the evaluator's normal
result payload.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen_evaluator
from radio_gs.scripts.eval_lerf_adaptive_support_diagnostic import (
    _load_cache_inputs,
    build_frozen_evaluator_argv,
    precompute_adaptive_membership,
    sha256_file,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


AUDIT_CONFIG = {
    "schema_version": 1,
    "continuous_score_source": "frozen_post_knn10_peak_scale_vala_remap",
    "continuous_projection": "visibility_aware_premultiplied_feature_composite",
    "score_resize": "bilinear_to_frozen_gt_resolution",
    "average_precision": "noninterpolated_grouped_by_equal_score",
    "oracle_threshold": "per_object_all_distinct_rendered_score_levels",
    "label_use": "diagnostic_only_after_score_and_method_freeze",
}
REGISTRATION_PATH = Path(
    "paper/artifacts/evidence_to_support_v1_experiment_registration_20260803.json"
)
EXPECTED_REGISTRATION_SHA256 = (
    "7c539fb523c7152446bdc5f28325986a9162baa6c85a5608a66552023aa869c4"
)


def grouped_average_precision(scores: np.ndarray, target: np.ndarray) -> float:
    """Non-interpolated AP with equal scores entering as one threshold group."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=bool).reshape(-1)
    if values.shape != labels.shape or values.size == 0:
        raise ValueError("scores and target must be aligned non-empty vectors")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    positives = int(labels.sum())
    if positives <= 0:
        return 0.0
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    sorted_labels = labels[order].astype(np.int64)
    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), values.size - 1]
    cumulative_tp = np.cumsum(sorted_labels)[group_ends].astype(np.float64)
    selected = (group_ends + 1).astype(np.float64)
    precision = cumulative_tp / selected
    recall = cumulative_tp / float(positives)
    recall_gain = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_gain * precision))


def oracle_threshold_iou(scores: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    """Best labelled IoU over all distinct rendered-score thresholds."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=bool).reshape(-1)
    if values.shape != labels.shape or values.size == 0:
        raise ValueError("scores and target must be aligned non-empty vectors")
    positives = int(labels.sum())
    if positives <= 0:
        return {"iou": 0.0, "threshold": math.inf, "selected_pixels": 0}
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    sorted_labels = labels[order].astype(np.int64)
    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), values.size - 1]
    intersections = np.cumsum(sorted_labels)[group_ends].astype(np.float64)
    selected = (group_ends + 1).astype(np.float64)
    unions = float(positives) + selected - intersections
    ious = intersections / np.maximum(unions, 1.0)
    best = int(np.argmax(ious))
    return {
        "iou": float(ious[best]),
        "threshold": float(sorted_scores[group_ends[best]]),
        "selected_pixels": int(selected[best]),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _selection_quality(row: Mapping[str, Any]) -> dict[str, float]:
    intersection = float(row.get("intersection_pixels", 0))
    predicted = float(row.get("pred_pixels", 0))
    positive = float(row.get("gt_pixels", 0))
    return {
        "miou": float(row.get("iou", 0.0)),
        "positive_coverage": intersection / max(positive, 1.0),
        "selected_purity": intersection / max(predicted, 1.0),
    }


def _index_query_rows(path: str | Path) -> dict[tuple[int, str], Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scene = payload.get("scene", {})
    summaries = scene.get("results", {}) if isinstance(scene, Mapping) else {}
    if not isinstance(summaries, Mapping) or len(summaries) != 1:
        raise ValueError(f"Expected one selection summary in {path}")
    summary = next(iter(summaries.values()))
    rows = summary.get("query_details", []) if isinstance(summary, Mapping) else []
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["frame_id"]), str(row["category"]))
        if key in result:
            raise ValueError(f"Duplicate query row {key} in {path}")
        result[key] = row
    return result


def _continuous_diagnostic(
    *,
    scene: str,
    scene_categories: list[str],
    frame_annotations: dict[int, list[dict]],
    img_h: int,
    img_w: int,
    model: torch.nn.Module,
    renderer: Any,
    dataset: Any,
    scores: torch.Tensor,
    device: torch.device,
    expected_scores_sha256: str,
    formal_rows: Mapping[tuple[int, str], Mapping[str, Any]],
    adaptive_rows: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    actual_hash = frozen_evaluator.tensor_sha256_float32(scores)
    if actual_hash != expected_scores_sha256:
        raise ValueError("evaluator scores differ from the pre-label frozen score tensor")
    features = scores.detach().float().to(device)
    proxy = frozen_evaluator.GaussianSelectionProxy(model, features)
    rows: list[dict[str, Any]] = []
    for frame_id, frame_objects in sorted(frame_annotations.items()):
        pose_w2c = dataset.pose_by_frame_idx.get(frame_id)
        if pose_w2c is None:
            continue
        viewmat = torch.from_numpy(pose_w2c.copy()).float().to(device)
        with torch.no_grad():
            score_maps = (
                renderer.render_features(proxy, viewmat)["feature_map"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        gt_masks = frozen_evaluator.build_gt_masks(
            frame_objects, scene_categories, img_h, img_w
        )
        for category in sorted({str(obj["category"]) for obj in frame_objects}):
            if category not in scene_categories:
                continue
            key = (int(frame_id), category)
            if key not in formal_rows or key not in adaptive_rows:
                raise ValueError(f"Missing frozen comparison row for {key}")
            query_index = scene_categories.index(category)
            gt = np.asarray(gt_masks[category], dtype=bool)
            rendered = frozen_evaluator._resize_float_map(score_maps[query_index], gt.shape)
            flat_scores = np.asarray(rendered, dtype=np.float64).reshape(-1)
            flat_target = gt.reshape(-1)
            oracle = oracle_threshold_iou(flat_scores, flat_target)
            positive_scores = flat_scores[flat_target]
            negative_scores = flat_scores[~flat_target]
            rows.append(
                {
                    "frame_id": int(frame_id),
                    "category": category,
                    "average_precision": grouped_average_precision(flat_scores, flat_target),
                    "auprc": grouped_average_precision(flat_scores, flat_target),
                    "oracle_threshold_iou": float(oracle["iou"]),
                    "oracle_threshold": float(oracle["threshold"]),
                    "oracle_selected_pixels": int(oracle["selected_pixels"]),
                    "positive_prevalence": float(flat_target.mean()),
                    "positive_score_mean": float(positive_scores.mean()),
                    "negative_score_mean": float(negative_scores.mean()),
                    "positive_negative_score_margin": float(
                        positive_scores.mean() - negative_scores.mean()
                    ),
                    "frozen_formal": _selection_quality(formal_rows[key]),
                    "target_blind_recursive_upper_otsu3": _selection_quality(adaptive_rows[key]),
                }
            )
    if not rows:
        raise RuntimeError(f"No score-quality rows produced for {scene}")
    return {
        "status": "label_aware_diagnostic_only_not_formal",
        "scene": scene,
        "objects": len(rows),
        "aggregate_object_mean": {
            "average_precision": _mean(rows, "average_precision"),
            "auprc": _mean(rows, "auprc"),
            "oracle_threshold_iou": _mean(rows, "oracle_threshold_iou"),
            "positive_negative_score_margin": _mean(rows, "positive_negative_score_margin"),
            "frozen_formal_miou": float(np.mean([row["frozen_formal"]["miou"] for row in rows])),
            "frozen_formal_positive_coverage": float(np.mean([row["frozen_formal"]["positive_coverage"] for row in rows])),
            "frozen_formal_selected_purity": float(np.mean([row["frozen_formal"]["selected_purity"] for row in rows])),
            "target_blind_otsu3_miou": float(np.mean([row["target_blind_recursive_upper_otsu3"]["miou"] for row in rows])),
            "target_blind_otsu3_positive_coverage": float(np.mean([row["target_blind_recursive_upper_otsu3"]["positive_coverage"] for row in rows])),
            "target_blind_otsu3_selected_purity": float(np.mean([row["target_blind_recursive_upper_otsu3"]["selected_purity"] for row in rows])),
        },
        "queries": rows,
    }


def _result_path(output_dir: str | Path, scene: str) -> Path:
    return Path(output_dir) / scene / "lerf_direct_3d_selection_results.json"


def run(args: argparse.Namespace) -> Path:
    # Freeze and hash every method-side input before either label-derived
    # comparison artifacts or the benchmark annotations are opened.
    cache = _load_cache_inputs(args.ours_multiscale_query_score_cache)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if checkpoint_sha256 != cache["renderer_geometry_checkpoint_sha256"]:
        raise ValueError("renderer checkpoint differs from score-cache authority")
    processed = precompute_adaptive_membership(cache, otsu_stages=3)
    registration_path = Path(
        str(getattr(args, "experiment_registration", "")).strip()
        or REGISTRATION_PATH
    ).resolve()
    if not registration_path.is_file():
        raise FileNotFoundError(f"Missing experiment registration: {registration_path}")
    registration_sha256 = sha256_file(registration_path)
    expected_registration_sha256 = (
        str(getattr(args, "experiment_registration_sha256", "")).strip()
        or EXPECTED_REGISTRATION_SHA256
    )
    if registration_sha256 != expected_registration_sha256:
        raise ValueError("experiment registration changed after preregistration")
    method_receipt = {
        "audit_config": AUDIT_CONFIG,
        "audit_config_sha256": canonical_json_sha256(AUDIT_CONFIG),
        "query_score_cache": str(Path(args.ours_multiscale_query_score_cache).resolve()),
        "query_score_cache_sha256": sha256_file(args.ours_multiscale_query_score_cache),
        "authority_query_scores_sha256": cache["query_scores_sha256"],
        "processed_scores_sha256": processed["processed_scores_sha256"],
        "renderer_checkpoint_sha256": checkpoint_sha256,
        "frozen_evaluator_source_sha256": sha256_file(frozen_evaluator.__file__),
        "experiment_registration": str(registration_path),
        "experiment_registration_sha256": registration_sha256,
    }
    # Everything below this line is allowed to read labels, but only to score
    # the already frozen method.  No value read here can change its scores,
    # configuration, graph, selection, or threshold.
    formal_rows = _index_query_rows(args.frozen_formal_result)
    adaptive_rows = _index_query_rows(args.target_blind_adaptive_result)

    original_evaluate = frozen_evaluator.evaluate_selection_spec
    diagnostic_holder: list[dict[str, Any]] = []

    def hooked_evaluate_selection_spec(**kwargs):
        if diagnostic_holder:
            raise RuntimeError("score-quality hook must be invoked exactly once")
        diagnostic_holder.append(
            _continuous_diagnostic(
                scene=kwargs["scene"],
                scene_categories=kwargs["scene_categories"],
                frame_annotations=kwargs["frame_annotations"],
                img_h=kwargs["img_h"],
                img_w=kwargs["img_w"],
                model=kwargs["model"],
                renderer=kwargs["renderer"],
                dataset=kwargs["dataset"],
                scores=kwargs["scores"],
                device=kwargs["device"],
                expected_scores_sha256=processed["processed_scores_sha256"],
                formal_rows=formal_rows,
                adaptive_rows=adaptive_rows,
            )
        )
        return original_evaluate(**kwargs)

    previous_argv = sys.argv
    frozen_evaluator.evaluate_selection_spec = hooked_evaluate_selection_spec
    try:
        sys.argv = build_frozen_evaluator_argv(args)
        frozen_evaluator.main()
    finally:
        frozen_evaluator.evaluate_selection_spec = original_evaluate
        sys.argv = previous_argv
    if len(diagnostic_holder) != 1:
        raise RuntimeError("score-quality hook call count differs")
    path = _result_path(args.output_dir, args.scene)
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnostic = diagnostic_holder[0]
    diagnostic["method_receipt_frozen_before_labels"] = method_receipt
    diagnostic["claim_boundary"] = (
        "AP/AUPRC and oracle thresholds use benchmark labels only after the score tensor "
        "and audit method are frozen. They diagnose ranking/separability and are forbidden "
        "for method, model, scene, or threshold selection. Formal and Otsu3 values are "
        "unchanged comparisons imported from their authority-bound result artifacts."
    )
    diagnostic["comparison_artifacts"] = {
        "frozen_formal_result": str(Path(args.frozen_formal_result).resolve()),
        "frozen_formal_result_sha256": sha256_file(args.frozen_formal_result),
        "target_blind_adaptive_result": str(Path(args.target_blind_adaptive_result).resolve()),
        "target_blind_adaptive_result_sha256": sha256_file(args.target_blind_adaptive_result),
    }
    payload["score_quality_diagnostic"] = diagnostic
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", required=True, choices=list(frozen_evaluator.LERF_OVS_SCENES))
    parser.add_argument("--ours_multiscale_query_score_cache", required=True)
    parser.add_argument("--frozen_formal_result", required=True)
    parser.add_argument("--target_blind_adaptive_result", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_mode", default="recursive_upper_otsu3")
    parser.add_argument("--label_dir", default="/mnt/pool/sqy/3d_understanding/lerf_ovs/label")
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument("--text_embedding_cache", default="checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt")
    parser.add_argument("--canonical_embedding_cache", default="checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--experiment-registration",
        default="",
        help=(
            "Optional immutable diagnostic registration. The v1 authority is "
            "retained when this is omitted."
        ),
    )
    parser.add_argument("--experiment-registration-sha256", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(run(args))


if __name__ == "__main__":
    main()
