#!/usr/bin/env python3
"""Run the frozen LERF3D evaluator and append preregistered oracle diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, sha256_file


DIAGNOSTIC_CONTRACT = {
    "schema_version": 1,
    "score_source": "frozen_post_knn10_peak_scale_vala_remap",
    "projection": "visibility_aware_premultiplied_feature_composite",
    "average_precision": "noninterpolated_grouped_by_equal_score",
    "oracle_threshold": "per_object_all_distinct_rendered_score_levels",
    "within_scene_top1": "fraction_of_target_pixels_where_target_query_is_highest_over_all_scene_queries",
    "rank_correlation": "pearson_correlation_between_midranked_rendered_score_and_binary_target",
    "label_use": "diagnostic_only_after_complete_method_score_tensor_freeze",
}


def _grouped_average_precision(scores: np.ndarray, target: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=bool).reshape(-1)
    positives = int(labels.sum())
    if values.shape != labels.shape or values.size == 0 or positives <= 0:
        return 0.0
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    sorted_labels = labels[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), values.size - 1]
    true_positive = np.cumsum(sorted_labels)[ends].astype(np.float64)
    precision = true_positive / (ends + 1)
    recall = true_positive / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _oracle_iou(scores: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=bool).reshape(-1)
    positives = int(labels.sum())
    if values.shape != labels.shape or values.size == 0 or positives <= 0:
        return 0.0, math.inf
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    sorted_labels = labels[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), values.size - 1]
    intersection = np.cumsum(sorted_labels)[ends].astype(np.float64)
    selected = ends.astype(np.float64) + 1
    iou = intersection / np.maximum(positives + selected - intersection, 1.0)
    best = int(np.argmax(iou))
    return float(iou[best]), float(sorted_scores[ends[best]])


def _binary_midrank_correlation(scores: np.ndarray, target: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sorted_values[:-1] != sorted_values[1:]) + 1]
    stops = np.r_[starts[1:], values.size]
    sorted_ranks = np.repeat(0.5 * (starts + stops - 1), stops - starts)
    ranks[order] = sorted_ranks
    ranks -= ranks.mean()
    labels -= labels.mean()
    denominator = np.sqrt(np.sum(ranks * ranks) * np.sum(labels * labels))
    return float(np.sum(ranks * labels) / denominator) if denominator > 0 else 0.0


def _validated_cache(path: str | Path) -> tuple[Mapping[str, Any], Any]:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("oracle score cache must be a mapping")
    xyz = torch.as_tensor(payload.get("xyz")).float().cpu()
    query_ids = tuple(str(value) for value in payload.get("query_ids", []))
    cache = frozen.validate_ours_multiscale_query_score_cache(
        payload,
        expected_xyz=xyz,
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=str(
            payload.get("renderer_geometry_checkpoint_sha256", "")
        ),
    )
    return payload, cache


def _freeze_processed_scores(
    positive_path: str | Path, negative_path: str | Path | None
) -> dict[str, Any]:
    positive_payload, positive = _validated_cache(positive_path)
    if positive.score_semantics == frozen.OURS_CANONICAL_NEGATIVE_PROBABILITY_SEMANTICS:
        if negative_path:
            raise ValueError("probability oracle cannot take a second negative cache")
        values = positive.query_scores
        negative_record = None
    else:
        if not negative_path:
            raise ValueError("raw-cosine oracle requires its canonical-negative cache")
        negative_payload, negative = _validated_cache(negative_path)
        if negative.query_ids != tuple(frozen.NEGATIVE_PROMPTS):
            raise ValueError("oracle negative query axis differs")
        for field in (
            "valid", "scale_ids", "scale_radii_m", "xyz_sha256",
            "field_checkpoint_sha256", "readout_checkpoint_sha256",
            "renderer_geometry_checkpoint_sha256",
        ):
            left = getattr(positive, field)
            right = getattr(negative, field)
            equal = torch.equal(left, right) if torch.is_tensor(left) else left == right
            if not bool(equal):
                raise ValueError(f"oracle positive/negative {field} differs")
        values = frozen.canonical_negative_relevancy_query_scores(
            positive.query_scores, negative.query_scores, logit_scale=10.0
        )
        negative_record = {
            "path": str(Path(negative_path).expanduser().resolve()),
            "sha256": sha256_file(negative_path),
            "authority_query_scores_sha256": negative_payload["authority"][
                "query_scores_sha256"
            ],
        }
    readout = frozen.vala_multiscale_knn_peak_select_scores(
        values,
        torch.as_tensor(positive_payload["xyz"]).float().cpu(),
        k=10,
        chunk_size=65536,
        valid_mask=positive.valid,
        query_contrast="none",
        scale_fusion="peak_select",
    )
    return {
        "scores": readout.scores,
        "scores_sha256": frozen.tensor_sha256_float32(readout.scores),
        "valid": positive.valid,
        "renderer_geometry_checkpoint_sha256": (
            positive.renderer_geometry_checkpoint_sha256
        ),
        "positive": {
            "path": str(Path(positive_path).expanduser().resolve()),
            "sha256": sha256_file(positive_path),
            "authority_query_scores_sha256": positive_payload["authority"][
                "query_scores_sha256"
            ],
        },
        "negative": negative_record,
        "selected_scale_indices": readout.selected_scale_indices.tolist(),
        "raw_smoothed_peaks": readout.raw_smoothed_peaks.tolist(),
    }


def _continuous_diagnostic(
    *,
    scene_categories: list[str],
    frame_annotations: dict[int, list[dict]],
    img_h: int,
    img_w: int,
    model: torch.nn.Module,
    renderer: Any,
    dataset: Any,
    scores: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    proxy = frozen.GaussianSelectionProxy(model, scores.detach().float().to(device))
    rows: list[dict[str, Any]] = []
    for frame_id, objects in sorted(frame_annotations.items()):
        pose = dataset.pose_by_frame_idx.get(frame_id)
        if pose is None:
            continue
        viewmat = torch.from_numpy(pose.copy()).float().to(device)
        with torch.no_grad():
            maps = (
                renderer.render_features(proxy, viewmat)["feature_map"]
                .detach().float().cpu().numpy()
            )
        gt_masks = frozen.build_gt_masks(objects, scene_categories, img_h, img_w)
        for category in sorted({str(item["category"]) for item in objects}):
            if category not in scene_categories:
                continue
            query_index = scene_categories.index(category)
            target = np.asarray(gt_masks[category], dtype=bool)
            resized = np.stack(
                [frozen._resize_float_map(value, target.shape) for value in maps], axis=0
            )
            values = resized[query_index].reshape(-1).astype(np.float64)
            labels = target.reshape(-1)
            oracle_iou, oracle_threshold = _oracle_iou(values, labels)
            positive = values[labels]
            negative = values[~labels]
            top1 = resized.argmax(axis=0) == query_index
            rows.append(
                {
                    "frame_id": int(frame_id),
                    "category": category,
                    "average_precision": _grouped_average_precision(values, labels),
                    "oracle_threshold_iou": oracle_iou,
                    "oracle_threshold": oracle_threshold,
                    "positive_negative_score_margin": float(
                        positive.mean() - negative.mean()
                    ),
                    "within_scene_top1": float(top1[target].mean()),
                    "rank_correlation": _binary_midrank_correlation(values, labels),
                }
            )
    if not rows:
        raise RuntimeError("oracle diagnostic produced no labeled objects")
    aggregate_keys = (
        "average_precision", "oracle_threshold_iou",
        "positive_negative_score_margin", "within_scene_top1", "rank_correlation",
    )
    return {
        "contract": DIAGNOSTIC_CONTRACT,
        "contract_sha256": canonical_json_sha256(DIAGNOSTIC_CONTRACT),
        "objects": len(rows),
        "aggregate_object_mean": {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in aggregate_keys
        },
        "queries": rows,
    }


def _result_path(output_dir: str | Path, scene: str) -> Path:
    return Path(output_dir) / scene / "lerf_direct_3d_selection_results.json"


def run(args: argparse.Namespace) -> Path:
    prereg = Path(args.preregistration).expanduser().resolve()
    if not prereg.is_file() or sha256_file(prereg) != args.expected_preregistration_sha256:
        raise ValueError("oracle diagnostic preregistration differs")
    frozen_scores = _freeze_processed_scores(
        args.ours_multiscale_query_score_cache,
        args.ours_multiscale_negative_score_cache or None,
    )
    if sha256_file(args.checkpoint) != frozen_scores[
        "renderer_geometry_checkpoint_sha256"
    ]:
        raise ValueError("renderer checkpoint differs from frozen oracle scores")

    holder: list[dict[str, Any]] = []
    original = frozen.evaluate_selection_spec

    def hooked(**kwargs):
        if holder:
            raise RuntimeError("oracle diagnostic hook invoked more than once")
        observed = frozen.tensor_sha256_float32(kwargs["scores"])
        if observed != frozen_scores["scores_sha256"]:
            raise ValueError("frozen evaluator processed scores differ")
        holder.append(
            _continuous_diagnostic(
                scene_categories=kwargs["scene_categories"],
                frame_annotations=kwargs["frame_annotations"],
                img_h=kwargs["img_h"], img_w=kwargs["img_w"],
                model=kwargs["model"], renderer=kwargs["renderer"],
                dataset=kwargs["dataset"], scores=kwargs["scores"],
                device=kwargs["device"],
            )
        )
        return original(**kwargs)

    argv = [
        "eval_lerf_direct_3d_selection.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--scene", args.scene,
        "--protocol_preset", "vala_repo_3d",
        "--score_threshold", "0.6",
        "--label_dir", args.label_dir,
        "--output_dir", args.output_dir,
        "--summary_head_weights", args.summary_head_weights,
        "--text_embedding_cache", args.text_embedding_cache,
        "--canonical_embedding_cache", args.canonical_embedding_cache,
        "--ours_multiscale_query_score_cache", args.ours_multiscale_query_score_cache,
        "--gpu", str(args.gpu),
    ]
    if args.ours_multiscale_negative_score_cache:
        argv.extend(
            [
                "--ours_multiscale_negative_score_cache",
                args.ours_multiscale_negative_score_cache,
            ]
        )
    previous_argv = sys.argv
    frozen.evaluate_selection_spec = hooked
    try:
        sys.argv = argv
        frozen.main()
    finally:
        frozen.evaluate_selection_spec = original
        sys.argv = previous_argv
    if len(holder) != 1:
        raise RuntimeError("oracle diagnostic hook call count differs")
    result_path = _result_path(args.output_dir, args.scene)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    summaries = payload["scene"]["results"]
    if len(summaries) != 1:
        raise ValueError("frozen result must contain exactly one selection summary")
    formal = next(iter(summaries.values()))
    query_rows = formal["query_details"]
    holder[0]["aggregate_object_mean"].update(
        {
            "frozen_fixed_threshold_miou": float(formal["miou"]),
            "selected_purity": float(
                np.mean(
                    [
                        float(row["intersection_pixels"])
                        / max(float(row["pred_pixels"]), 1.0)
                        for row in query_rows
                    ]
                )
            ),
            "positive_coverage": float(
                np.mean(
                    [
                        float(row["intersection_pixels"])
                        / max(float(row["gt_pixels"]), 1.0)
                        for row in query_rows
                    ]
                )
            ),
        }
    )
    holder[0]["method_receipt_frozen_before_labels"] = {
        "positive_score_cache": frozen_scores["positive"],
        "negative_score_cache": frozen_scores["negative"],
        "processed_scores_sha256": frozen_scores["scores_sha256"],
        "selected_scale_indices": frozen_scores["selected_scale_indices"],
        "renderer_checkpoint_sha256": sha256_file(args.checkpoint),
        "frozen_evaluator_sha256": sha256_file(frozen.__file__),
        "preregistration": {
            "path": str(prereg),
            "sha256": args.expected_preregistration_sha256,
        },
    }
    payload["teacher_view_oracle_diagnostic"] = holder[0]
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", default="figurines")
    parser.add_argument("--ours_multiscale_query_score_cache", required=True)
    parser.add_argument("--ours_multiscale_negative_score_cache", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_dir", default="/mnt/pool/sqy/3d_understanding/lerf_ovs/label")
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument("--text_embedding_cache", default="checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt")
    parser.add_argument("--canonical_embedding_cache", default="checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt")
    parser.add_argument("--gpu", type=int, default=0)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
