#!/usr/bin/env python3
"""Materialize a target-blind adaptive operating-point LERF candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.querying import adaptive_otsu_score_calibration as calibration
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_adaptive_otsu_external_scores.v1"


def access_audit() -> dict[str, bool]:
    return {
        "parent_primitive_score_cache_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_opened": False,
        "target_metrics_computed": False,
        "gpu_used": False,
        "scene_or_query_specific_parameters": False,
    }


def _load_parent(path: str, digest: str) -> tuple[Mapping[str, Any], dict[str, str]]:
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=digest,
        map_location="cpu",
        label="adaptive Otsu parent score cache",
    )
    scores = payload.get("query_scores")
    valid = payload.get("valid")
    xyz = payload.get("xyz")
    metadata = payload.get("metadata")
    if (
        not isinstance(scores, torch.Tensor)
        or scores.dtype != torch.float32
        or scores.ndim != 2
        or not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
        or not isinstance(xyz, torch.Tensor)
        or xyz.shape != (scores.shape[0], 3)
        or not isinstance(metadata, Mapping)
        or not isinstance(metadata.get("query_names"), list)
        or len(metadata["query_names"]) != scores.shape[1]
    ):
        raise ValueError("parent external score cache axes differ")
    return payload, file_record(path)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_cache).expanduser().resolve()
    report_path = Path(args.output_report).expanduser().resolve()
    if str(output) != args.output_cache or str(report_path) != args.output_report:
        raise ValueError("output paths must be canonical absolute")
    if output == report_path or output.exists() or report_path.exists():
        raise FileExistsError("adaptive Otsu outputs must be new and distinct")
    parent, parent_record = _load_parent(args.parent_cache, args.parent_cache_sha256)
    result = calibration.calibrate_to_frozen_threshold(
        parent["query_scores"],
        parent["valid"],
        stages=args.stages,
    )
    names = [str(value) for value in parent["metadata"]["query_names"]]
    payload = {
        "schema": SCHEMA,
        "query_scores": result.scores,
        "valid": parent["valid"].detach().bool().cpu().contiguous(),
        "xyz": parent["xyz"].detach().float().cpu().contiguous(),
        "metadata": {
            "scene_id": args.scene_id,
            "query_names": names,
            "score_semantics": "monotone_recursive_upper_otsu_calibrated_relevance",
            "score_postprocess": "none_already_calibrated_to_frozen_threshold_0p6",
            "source_thresholds": result.source_thresholds.tolist(),
            "otsu_stages": result.stages,
            "parent_score_cache": parent_record,
        },
        "authority": {
            "contract": calibration.CONTRACT,
            "implementation": file_record(Path(calibration.__file__).resolve()),
            "producer": file_record(Path(__file__).resolve()),
            "parent_score_cache": parent_record,
            "frozen_threshold": calibration.FROZEN_THRESHOLD,
            "within_query_order_preserved": True,
            "selection_equivalence": "calibrated>0.6 iff parent>recursive_upper_otsu_threshold",
            "access_audit": access_audit(),
            "metric_execution_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    report = {
        "schema": SCHEMA,
        "status": "complete_target_blind_premetric_adaptive_calibration",
        "scene_id": args.scene_id,
        "method": payload["authority"],
        "query_names": names,
        "source_thresholds": result.source_thresholds.tolist(),
        "selected_counts": result.selected_counts.tolist(),
        "valid_primitives": int(parent["valid"].sum()),
        "output_cache": file_record(output),
    }
    write_frozen_json(report_path, report)
    return {**report, "output_report": file_record(report_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--parent-cache-sha256", required=True)
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-report", required=True)
    return parser


def main() -> None:
    print(json.dumps(materialize(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "access_audit", "materialize"]
