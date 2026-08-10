#!/usr/bin/env python3
"""Materialize the source-selected Gaussian valid-domain LERF readout.

This independent producer does not modify the frozen evaluator or the
promoted uniform valid-domain implementation.  It fails closed unless the
immutable two-scene source gate selected exactly the global ``gaussian``
policy, then emits an ordinary external query-score cache for a separately
authorized frozen metric runner.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.querying import (
    reliability_weighted_valid_domain_knn_readout as weighted,
)
from radio_gs.scripts import materialize_lerf_valid_domain_knn_candidate as v1
from radio_gs.scripts import (
    select_lerf_source_only_reliability_weighted_knn as selector,
)
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_gaussian_valid_domain_knn_external_scores.v1"
SCHEMA_VERSION = 1
SELECTED_POLICY_ID = "gaussian"


def access_audit() -> dict[str, bool]:
    return {
        "source_only_policy_gate_opened": True,
        "raw_positive_query_score_cache_opened": True,
        "raw_canonical_negative_score_cache_opened": True,
        "reliability_sidecar_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_or_labels_opened": False,
        "target_metrics_opened": False,
        "target_metrics_computed": False,
        "gpu_used": False,
        "result_dependent_parameters": False,
    }


def validate_source_gate(path: str | Path, digest: str) -> dict[str, Any]:
    payload, _, _ = load_json_object(
        path, expected_sha256=digest, label="Gaussian valid-domain source gate"
    )
    selection = payload.get("selection") if isinstance(payload, Mapping) else None
    if (
        payload.get("schema") != selector.RESULT_SCHEMA
        or payload.get("schema_version") != selector.SCHEMA_VERSION
        or payload.get("status") != "complete_source_only_weighted_knn_gate"
        or payload.get("implementation")
        != file_record(Path(selector.__file__).resolve())
        or payload.get("method_contract") != selector.method_contract()
        or payload.get("access_audit") != selector.access_audit()
        or payload.get("metric_executed") is not False
        or payload.get("metric_execution_authorized") is not True
        or not isinstance(selection, Mapping)
        or selection.get("selected_policy_id") != SELECTED_POLICY_ID
        or selection.get("fallback_uniform_used") is not False
        or selection.get("target_metric_execution_authorized") is not True
        or payload.get("next_gate")
        != "materialize_one_global_selected_policy_then_one_shot_frozen_metric"
    ):
        raise ValueError("Gaussian valid-domain source gate differs")
    candidates = selection.get("candidate_grid")
    selected_rows = (
        [row for row in candidates if row.get("policy_id") == SELECTED_POLICY_ID]
        if isinstance(candidates, list)
        else []
    )
    if (
        len(selected_rows) != 1
        or selected_rows[0].get("eligible") is not True
        or selected_rows[0].get("strict_pooled_mean_improvement") is not True
        or selected_rows[0].get("every_source_mean_nonregression") is not True
        or selected_rows[0].get("every_source_p05_nonregression") is not True
    ):
        raise ValueError("Gaussian valid-domain selected gate row differs")
    prereg = payload.get("preregistration")
    if not isinstance(prereg, Mapping) or set(prereg) != {"path", "sha256"}:
        raise ValueError("Gaussian valid-domain gate preregistration differs")
    selector.validate_preregistration(prereg["path"], prereg["sha256"])
    return dict(payload)


def build_candidate(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    xyz: torch.Tensor,
    valid: torch.Tensor,
    *,
    chunk_size: int,
) -> weighted.ValidDomainMultiscaleReadout:
    reliability = torch.zeros(valid.numel(), dtype=torch.float32)
    return weighted.reliability_weighted_valid_domain_multiscale_readout(
        positive_scores,
        negative_scores,
        xyz,
        valid,
        reliability,
        policy_id=SELECTED_POLICY_ID,
        k=weighted.KNN_K,
        chunk_size=chunk_size,
        logit_scale=v1.LOGIT_SCALE,
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_cache).expanduser().resolve()
    report_path = Path(args.output_report).expanduser().resolve()
    if str(output) != args.output_cache or str(report_path) != args.output_report:
        raise ValueError("Gaussian valid-domain output paths must be canonical")
    if output == report_path:
        raise ValueError("Gaussian valid-domain output paths must differ")
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("Gaussian valid-domain output already exists")
    gate = validate_source_gate(args.source_gate, args.source_gate_sha256)
    (
        positive_raw,
        _negative_raw,
        positive,
        negative,
        positive_record,
        negative_record,
    ) = v1._load_pair(
        args.positive_cache,
        args.positive_cache_sha256,
        args.negative_cache,
        args.negative_cache_sha256,
    )
    result = build_candidate(
        positive.query_scores,
        negative.query_scores,
        torch.as_tensor(positive_raw["xyz"]),
        positive.valid,
        chunk_size=args.chunk_size,
    )
    if (
        not bool(torch.isfinite(result.scores).all())
        or bool((result.scores < 0).any())
        or bool((result.scores > 1).any())
        or bool((result.scores[~positive.valid] != 0).any())
    ):
        raise ValueError("Gaussian valid-domain final scores differ")

    authority = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": args.scene_id,
        "method": {
            "contract": weighted.CONTRACT,
            "implementation": file_record(Path(weighted.__file__).resolve()),
            "producer": file_record(Path(__file__).resolve()),
            "selected_policy_id": SELECTED_POLICY_ID,
            "weight_formula": (
                "normalize(exp(-0.5*(distance/farthest_knn_distance)^2))"
            ),
            "local_bandwidth": "per_center_farthest_valid_knn_distance",
            "knn_k": weighted.KNN_K,
            "knn_domain": "covered_valid_primitive_xyz_only",
            "self_semantics": "retained_in_knn_then_0.5_raw_outer_blend",
            "effective_self_weight_at_k10": 0.55,
            "knn_blend": "0.5*raw+0.5*normalized_gaussian_neighbor_estimate",
            "per_scale_minmax": True,
            "released_two_x_minus_one_clip": True,
            "scale_selection": (
                "highest_raw_gaussian_valid_domain_knn_smoothed_peak_per_query"
            ),
            "per_scene_or_per_query_hyperparameters": False,
        },
        "source_gate": file_record(args.source_gate),
        "source_gate_selected_policy": gate["selection"]["selected_policy_id"],
        "source_artifacts": {
            "positive_raw_score_cache": positive_record,
            "canonical_negative_raw_score_cache": negative_record,
        },
        "geometry_axis": {
            "num_gaussians": int(positive.valid.numel()),
            "valid_gaussians": int(positive.valid.sum()),
            "xyz_sha256": positive.xyz_sha256,
            "renderer_geometry_checkpoint_sha256": (
                positive.renderer_geometry_checkpoint_sha256
            ),
        },
        "query_axis": list(positive.query_ids),
        "scale_axis": [
            {"id": scale_id, "radius_m": radius}
            for scale_id, radius in zip(positive.scale_ids, positive.scale_radii_m)
        ],
        "selected_scale_indices": result.selected_scale_indices.tolist(),
        "raw_smoothed_peaks": result.raw_smoothed_peaks.tolist(),
        "final_query_scores_sha256": frozen.tensor_sha256_typed(result.scores),
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
    }
    payload = {
        "schema": SCHEMA,
        "query_scores": result.scores.float().contiguous(),
        "valid": positive.valid.bool().contiguous(),
        "xyz": torch.as_tensor(positive_raw["xyz"]).detach().float().cpu().contiguous(),
        "metadata": {
            "query_names": list(positive.query_ids),
            "score_semantics": (
                "canonical_negative_probability_gaussian_valid_domain_knn10_"
                "peak_scale_minmax"
            ),
            "score_postprocess": (
                "none_already_gaussian_valid_domain_knn10_per_scale_minmax"
            ),
        },
        "authority": authority,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    report = {
        **authority,
        "status": "complete_source_selected_gaussian_premetric_readout",
        "output_cache": file_record(output),
        "score_min_valid": float(result.scores[positive.valid].min()),
        "score_max_valid": float(result.scores[positive.valid].max()),
        "finite": True,
    }
    write_frozen_json(report_path, report)
    return {**report, "output_report": file_record(report_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-gate", required=True)
    parser.add_argument("--source-gate-sha256", required=True)
    parser.add_argument("--positive-cache", required=True)
    parser.add_argument("--positive-cache-sha256", required=True)
    parser.add_argument("--negative-cache", required=True)
    parser.add_argument("--negative-cache-sha256", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--chunk-size", type=int, default=65536)
    return parser


def main() -> None:
    result = materialize(build_parser().parse_args())
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA",
    "SELECTED_POLICY_ID",
    "access_audit",
    "build_candidate",
    "materialize",
    "validate_source_gate",
]
