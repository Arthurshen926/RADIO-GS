"""Diagnose a frozen LERF run without selecting a new benchmark readout.

This analysis is intentionally downstream of the complete frozen evaluation.
It distinguishes missing support from broad, query-insensitive support using
only already-sealed score caches and evaluator receipts.  It never writes a
new score cache or proposes a threshold.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _summary(values: Sequence[float]) -> dict[str, float]:
    finite = [float(value) for value in values]
    if not finite:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(finite)
    return {
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "p90": float(ordered[int(0.9 * (len(ordered) - 1))]),
        "max": float(ordered[-1]),
    }


def _pairwise_jaccard(mask: torch.Tensor) -> dict[str, float]:
    if mask.ndim != 2 or mask.shape[1] < 2 or mask.dtype != torch.bool:
        raise ValueError("selection mask must be boolean [N,Q] with Q >= 2")
    # Q is small while N can exceed half a million.  One matrix product avoids
    # launching a full-column boolean reduction for every query pair.
    numeric = mask.float()
    intersections = numeric.T @ numeric
    counts = numeric.sum(dim=0)
    unions = counts[:, None] + counts[None, :] - intersections
    jaccard = intersections / unions.clamp_min(1.0)
    values: list[float] = []
    for left in range(mask.shape[1]):
        for right in range(left + 1, mask.shape[1]):
            values.append(float(jaccard[left, right].item()))
    return _summary(values)


def _posterior_correlation(scores: torch.Tensor) -> dict[str, float]:
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("posterior scores must be [N,Q] with Q >= 2")
    centered = scores.float() - scores.float().mean(dim=0, keepdim=True)
    normalized = centered / centered.norm(dim=0, keepdim=True).clamp_min(1e-12)
    correlation = normalized.T @ normalized
    off_diagonal = correlation[
        ~torch.eye(correlation.shape[0], dtype=torch.bool)
    ].tolist()
    return _summary(off_diagonal)


def _result_payload(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scene = payload["scene"]
    return scene["results"][scene["best_by_miou"]]


def _projection_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    details = list(result["query_details"])
    precision = [
        float(row["intersection_pixels"]) / max(int(row["pred_pixels"]), 1)
        for row in details
    ]
    recall = [
        float(row["intersection_pixels"]) / max(int(row["gt_pixels"]), 1)
        for row in details
    ]
    overselect = [float(row["overselect_ratio"]) for row in details]
    return {
        "miou": float(result["miou"]),
        "sample_count": int(result["n"]),
        "pixel_precision": _summary(precision),
        "pixel_recall": _summary(recall),
        "predicted_to_gt_area_ratio": _summary(overselect),
    }


def _top_fraction_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("top-fraction selection must be in (0,1]")
    top_count = max(1, round(float(fraction) * scores.shape[0]))
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for query_index in range(scores.shape[1]):
        indices = torch.topk(scores[:, query_index], top_count).indices
        mask[indices, query_index] = True
    return mask


def analyze_scene(
    *,
    scene: str,
    score_cache: Path,
    strict_report: Path,
    top2_report: Path,
) -> dict[str, Any]:
    payload = torch.load(score_cache, map_location="cpu", weights_only=False)
    scores = payload.get("query_scores")
    identity_scores = payload.get("identity_query_scores")
    metadata = dict(payload.get("metadata", {}))
    query_names = [str(value) for value in metadata.get("query_names", [])]
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise ValueError(f"{score_cache} does not contain [N,Q] query_scores")
    if (
        not isinstance(identity_scores, torch.Tensor)
        or identity_scores.shape != scores.shape
    ):
        raise ValueError("score cache lacks row-aligned identity_query_scores")
    if len(query_names) != scores.shape[1]:
        raise ValueError("score-cache query names do not match score columns")
    if not torch.isfinite(scores).all() or not torch.isfinite(identity_scores).all():
        raise ValueError("posterior cache contains non-finite values")

    strict_mask = scores > 0.5
    top2_mask = _top_fraction_mask(scores, 0.02)
    identity_top2_mask = _top_fraction_mask(identity_scores, 0.02)

    strict = _projection_diagnostics(_result_payload(strict_report))
    top2 = _projection_diagnostics(_result_payload(top2_report))
    selected_fractions = strict_mask.float().mean(dim=0).tolist()
    cross_query = {
        "identity_pearson": _posterior_correlation(identity_scores),
        "posterior_pearson": _posterior_correlation(scores),
        "strict_selection_jaccard": _pairwise_jaccard(strict_mask),
        "identity_top2_jaccard": _pairwise_jaccard(identity_top2_mask),
        "posterior_top2_jaccard": _pairwise_jaccard(top2_mask),
    }
    broad_support = bool(
        strict["pixel_recall"]["mean"] > 0.7
        and strict["pixel_precision"]["mean"] < 0.2
        and strict["predicted_to_gt_area_ratio"]["median"] > 5.0
    )
    query_insensitive = bool(
        cross_query["posterior_pearson"]["median"] > 0.5
        or cross_query["strict_selection_jaccard"]["median"] > 0.15
    )
    child_collapse = bool(
        cross_query["posterior_pearson"]["median"]
        - cross_query["identity_pearson"]["median"]
        > 0.25
        and cross_query["posterior_top2_jaccard"]["median"]
        > cross_query["identity_top2_jaccard"]["median"]
    )
    return {
        "scene": scene,
        "gaussian_count": int(scores.shape[0]),
        "query_count": int(scores.shape[1]),
        "strict_selected_fraction": _summary(selected_fractions),
        "strict": strict,
        "top2": top2,
        "cross_query": cross_query,
        "failure_signals": {
            "broad_support_not_missing_support": broad_support,
            "query_insensitive_extent": query_insensitive,
            "child_posterior_collapses_identity_diversity": child_collapse,
            "top2_remains_below_0.15_miou": bool(top2["miou"] < 0.15),
        },
        "inputs": [
            {"role": "score_cache", "path": str(score_cache), "sha256": sha256_file(score_cache)},
            {"role": "strict_report", "path": str(strict_report), "sha256": sha256_file(strict_report)},
            {"role": "top2_report", "path": str(top2_report), "sha256": sha256_file(top2_report)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--score-cache-root", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result_root = Path(args.result_root).resolve(strict=True)
    cache_root = Path(args.score_cache_root).resolve(strict=True)
    rows = []
    for scene in args.scene:
        rows.append(
            analyze_scene(
                scene=scene,
                score_cache=(cache_root / f"{scene}.pt").resolve(strict=True),
                strict_report=(
                    result_root
                    / "lerf3d_strict"
                    / scene
                    / scene
                    / "lerf_direct_3d_selection_results.json"
                ).resolve(strict=True),
                top2_report=(
                    result_root
                    / "lerf3d_top2"
                    / scene
                    / scene
                    / "lerf_direct_3d_selection_results.json"
                ).resolve(strict=True),
            )
        )

    broad_count = sum(
        int(row["failure_signals"]["broad_support_not_missing_support"])
        for row in rows
    )
    insensitive_count = sum(
        int(row["failure_signals"]["query_insensitive_extent"])
        for row in rows
    )
    collapse_count = sum(
        int(
            row["failure_signals"][
                "child_posterior_collapses_identity_diversity"
            ]
        )
        for row in rows
    )
    top2_failure_count = sum(
        int(row["failure_signals"]["top2_remains_below_0.15_miou"])
        for row in rows
    )
    required = max(3, len(rows) - 1)
    design_defect = bool(
        broad_count >= required
        and insensitive_count >= required
        and collapse_count >= required
        and top2_failure_count >= required
    )
    report = {
        "schema": "radio_gs.sugm_v3.lerf_posterior_failure_diagnostic.v2",
        "diagnostic_only_after_frozen_benchmark": True,
        "used_for_method_selection": False,
        "scenes": rows,
        "scene_counts": {
            "total": len(rows),
            "broad_support_not_missing_support": broad_count,
            "query_insensitive_extent": insensitive_count,
            "child_posterior_collapses_identity_diversity": collapse_count,
            "top2_below_0.15_miou": top2_failure_count,
        },
        "current_design_defect_supported": design_defect,
        "conclusion": (
            "the instance/calibration child collapses distinct clean identities "
            "into broad, spatially scattered extent; coverage is not the "
            "remaining bottleneck"
            if design_defect
            else "failure mode is not consistent across the evaluated cohort"
        ),
    }
    write_frozen_json(Path(args.output).resolve(), report)
    print(json.dumps(report["scene_counts"], sort_keys=True))
    print(report["conclusion"])


if __name__ == "__main__":
    main()
