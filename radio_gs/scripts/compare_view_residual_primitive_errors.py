#!/usr/bin/env python3
"""Check whether a frozen view residual reduces conflict-linked held-out error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.evaluation.view_consistency import pearson_spearman


def _mean_error(cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.as_tensor(cache["error_weight"]).float()
    error = torch.as_tensor(cache["error_sum"]).float() / weight.clamp_min(1e-8)
    return error, weight


def _decile(error: torch.Tensor, disagreement: torch.Tensor, valid: torch.Tensor) -> dict:
    values = disagreement[valid]
    errors = error[valid]
    low_threshold = torch.quantile(values, 0.1)
    high_threshold = torch.quantile(values, 0.9)
    low = values <= low_threshold
    high = values >= high_threshold
    return {
        "bottom10_mean_error": float(errors[low].mean()),
        "top10_mean_error": float(errors[high].mean()),
        "top_minus_bottom_error": float(errors[high].mean() - errors[low].mean()),
    }


def compare(args: argparse.Namespace) -> dict:
    consistency = torch.load(args.consistency_cache, map_location="cpu")
    base = torch.load(args.base_error_cache, map_location="cpu")
    residual = torch.load(args.residual_error_cache, map_location="cpu")
    if str(base.get("geometry_xyz_sha256", "")) != str(
        residual.get("geometry_xyz_sha256", "")
    ) or str(base.get("geometry_xyz_sha256", "")) != str(
        consistency.get("geometry_xyz_sha256", "")
    ):
        raise ValueError("comparison caches use different geometry rows")
    if list(base.get("selected_frame_ids", [])) != list(
        residual.get("selected_frame_ids", [])
    ):
        raise ValueError("base and residual held-out frame sets differ")
    if "zero_mean_view_residual" not in str(residual.get("representation", "")):
        raise ValueError("residual cache does not identify the zero-mean residual path")
    disagreement = torch.as_tensor(consistency["view_disagreement"]).float()
    train_count = torch.as_tensor(consistency["observation_count"]).long()
    base_error, base_weight = _mean_error(base)
    residual_error, residual_weight = _mean_error(residual)
    valid = (train_count >= 2) & (base_weight > 0) & (residual_weight > 0)
    if int(valid.sum()) <= 1:
        raise RuntimeError("insufficient common primitive support")
    improvement = base_error - residual_error
    base_correlation = pearson_spearman(disagreement[valid], base_error[valid])
    residual_correlation = pearson_spearman(
        disagreement[valid], residual_error[valid]
    )
    improvement_correlation = pearson_spearman(
        disagreement[valid], improvement[valid]
    )
    base_decile = _decile(base_error, disagreement, valid)
    residual_decile = _decile(residual_error, disagreement, valid)
    improvement_decile = _decile(-improvement, disagreement, valid)
    # Negated improvement is passed to the generic error helper; invert the
    # reported means back to positive improvement for readability.
    high_improvement = -float(improvement_decile["top10_mean_error"])
    low_improvement = -float(improvement_decile["bottom10_mean_error"])
    base_spearman = base_correlation["spearman"]
    residual_spearman = residual_correlation["spearman"]
    report = {
        "schema_version": 1,
        "audit": "zero_mean_view_residual_causal_check_v1",
        "protocol": {
            "checkpoint_frozen_before_heldout_access": True,
            "heldout_frames": list(base.get("selected_frame_ids", [])),
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
        "artifacts": {
            "consistency_cache": str(Path(args.consistency_cache).resolve()),
            "base_error_cache": str(Path(args.base_error_cache).resolve()),
            "residual_error_cache": str(Path(args.residual_error_cache).resolve()),
        },
        "common_rows": int(valid.sum()),
        "base": {
            "mean_primitive_error": float(base_error[valid].mean()),
            "disagreement_error_correlation": base_correlation,
            **base_decile,
        },
        "residual": {
            "mean_primitive_error": float(residual_error[valid].mean()),
            "disagreement_error_correlation": residual_correlation,
            **residual_decile,
        },
        "improvement": {
            "mean": float(improvement[valid].mean()),
            "disagreement_improvement_correlation": improvement_correlation,
            "bottom10_disagreement_mean_improvement": low_improvement,
            "top10_disagreement_mean_improvement": high_improvement,
            "top_minus_bottom_improvement": high_improvement - low_improvement,
        },
        "decision": {
            "view_context_error_reduced": (
                base_spearman is not None
                and residual_spearman is not None
                and float(residual_spearman) < float(base_spearman)
                and high_improvement > low_improvement
                and float(improvement[valid].mean()) > 0
            ),
            "remaining_error_requires_next_audit": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consistency-cache", required=True)
    parser.add_argument("--base-error-cache", required=True)
    parser.add_argument("--residual-error-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args), indent=2))


if __name__ == "__main__":
    main()

