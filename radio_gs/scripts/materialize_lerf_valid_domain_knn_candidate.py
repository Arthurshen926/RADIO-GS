#!/usr/bin/env python3
"""Materialize a target-blind LERF valid-domain-kNN readout candidate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radio_gs.querying import valid_domain_knn_readout as readout
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_valid_domain_knn_external_scores.v1"
KNN_K = 10
LOGIT_SCALE = 10.0


def access_audit() -> dict[str, bool]:
    return {
        "raw_positive_query_score_cache_opened": True,
        "raw_canonical_negative_score_cache_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_opened": False,
        "target_metrics_computed": False,
        "gpu_used": False,
        "result_dependent_parameters": False,
    }


def _load_pair(
    positive_path: str | Path,
    positive_sha256: str,
    negative_path: str | Path,
    negative_sha256: str,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    frozen.OursMultiscaleQueryScoreCache,
    frozen.OursMultiscaleQueryScoreCache,
    dict[str, str],
    dict[str, str],
]:
    positive_raw, _, _ = load_torch_mapping(
        positive_path,
        expected_sha256=positive_sha256,
        map_location="cpu",
        label="valid-domain positive raw scores",
    )
    negative_raw, _, _ = load_torch_mapping(
        negative_path,
        expected_sha256=negative_sha256,
        map_location="cpu",
        label="valid-domain canonical-negative raw scores",
    )
    query_ids = tuple(str(value) for value in positive_raw.get("query_ids", []))
    negative_ids = tuple(str(value) for value in negative_raw.get("query_ids", []))
    xyz = torch.as_tensor(positive_raw.get("xyz")).detach().float().cpu()
    renderer_sha = str(
        positive_raw.get("renderer_geometry_checkpoint_sha256", "")
    )
    positive = frozen.validate_ours_multiscale_query_score_cache(
        positive_raw,
        expected_xyz=xyz,
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=renderer_sha,
    )
    negative = frozen.validate_ours_multiscale_query_score_cache(
        negative_raw,
        expected_xyz=xyz,
        expected_query_ids=negative_ids,
        expected_renderer_geometry_checkpoint_sha256=renderer_sha,
    )
    if negative_ids != tuple(frozen.NEGATIVE_PROMPTS):
        raise ValueError("canonical-negative query axis differs")
    if (
        positive.score_semantics != "raw_independent_normalized_cosine"
        or negative.score_semantics != "raw_independent_normalized_cosine"
    ):
        raise ValueError("valid-domain candidate requires raw normalized cosine caches")
    for field in (
        "valid",
        "scale_ids",
        "scale_radii_m",
        "xyz_sha256",
        "field_checkpoint_sha256",
        "readout_checkpoint_sha256",
        "renderer_geometry_checkpoint_sha256",
    ):
        left, right = getattr(positive, field), getattr(negative, field)
        same = torch.equal(left, right) if torch.is_tensor(left) else left == right
        if not bool(same):
            raise ValueError(f"positive/canonical-negative {field} differs")
    return (
        positive_raw,
        negative_raw,
        positive,
        negative,
        file_record(positive_path),
        file_record(negative_path),
    )


def build_candidate(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    xyz: torch.Tensor,
    valid: torch.Tensor,
    *,
    chunk_size: int,
) -> readout.ValidDomainMultiscaleReadout:
    return readout.valid_domain_multiscale_readout(
        positive_scores,
        negative_scores,
        xyz,
        valid,
        k=KNN_K,
        chunk_size=chunk_size,
        logit_scale=LOGIT_SCALE,
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_cache).expanduser().resolve()
    report_path = Path(args.output_report).expanduser().resolve()
    if str(args.output_cache) != str(output) or str(args.output_report) != str(
        report_path
    ):
        raise ValueError("output paths must be canonical absolute")
    if output == report_path:
        raise ValueError("output cache and report paths must differ")
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("valid-domain candidate output already exists")

    (
        positive_raw,
        _negative_raw,
        positive,
        negative,
        positive_record,
        negative_record,
    ) = _load_pair(
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
    if not bool(torch.isfinite(result.scores).all()):
        raise ValueError("valid-domain final scores are nonfinite")
    if bool((result.scores < 0).any()) or bool((result.scores > 1).any()):
        raise ValueError("valid-domain final scores must remain in [0,1]")
    if bool((result.scores[~positive.valid] != 0).any()):
        raise ValueError("invalid primitive scores must be exact zero")

    authority = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene_id": args.scene_id,
        "method": {
            "contract": readout.CONTRACT,
            "producer": file_record(Path(__file__).resolve()),
            "implementation": file_record(Path(readout.__file__).resolve()),
            "canonical_negative_logit_scale": LOGIT_SCALE,
            "knn_k": KNN_K,
            "knn_domain": "covered_valid_primitive_xyz_only",
            "knn_blend": "0.5*raw+0.5*valid_neighbor_mean",
            "per_scale_minmax": True,
            "released_two_x_minus_one_clip": True,
            "scale_selection": "highest_raw_valid_domain_knn_smoothed_peak_per_query",
            "per_scene_or_per_query_hyperparameters": False,
        },
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
            "score_semantics": "canonical_negative_probability_valid_domain_knn10_peak_scale_minmax",
            "score_postprocess": "none_already_valid_domain_knn10_per_scale_minmax",
        },
        "authority": authority,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    report = {
        **authority,
        "status": "complete_source_only_premetric_valid_domain_readout",
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
    "KNN_K",
    "LOGIT_SCALE",
    "SCHEMA",
    "access_audit",
    "build_candidate",
    "materialize",
]
