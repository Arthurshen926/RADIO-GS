"""Aggregate preregistered geometry evidence without opening raw benchmark data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.v4.contracts.geometry_receipt import sha256_file


def _candidate_record(
    oracle: dict[str, float],
    source: dict[str, float],
    minimum_delta: float,
) -> dict[str, Any]:
    oracle_directions = {
        "same_view_roundtrip": oracle["roundtrip_delta"],
        "tracked_cross_view_transfer": oracle["transfer_delta"],
        "boundary_leakage_reduction": oracle["boundary_leakage_reduction"],
        "mutually_exclusive_purity": oracle["purity_delta"],
    }
    source_directions = {
        "same_view_roundtrip": source["roundtrip_delta"],
        "boundary_leakage_reduction": source["same_view_leakage_reduction"],
        "mutually_exclusive_purity": source["mutually_exclusive_purity_delta"],
    }
    oracle_improved = sum(value > minimum_delta for value in oracle_directions.values())
    source_improved = sum(value > minimum_delta for value in source_directions.values())
    purity_non_regression = (
        oracle["purity_delta"] >= 0
        and source["mutually_exclusive_purity_delta"] >= 0
    )
    passes = oracle_improved >= 3 and source_improved >= 3 and purity_non_regression
    return {
        "passes_scene_gate": passes,
        "oracle_primary_directions": oracle_directions,
        "oracle_primary_improved_count": oracle_improved,
        "source_primary_directions": source_directions,
        "source_primary_improved_count": source_improved,
        "purity_non_regression": purity_non_regression,
        "coverage_delta": {
            "oracle": oracle["coverage_delta"],
            "source": source["coverage_delta"],
        },
        "registration_ambiguity": {
            "oracle_effective_contributors_reduction": oracle["effective_contributors_reduction"],
            "source_effective_contributors_reduction": source["effective_contributors_reduction"],
        },
        "untracked_source_cross_view_best_match_delta_not_gating": source["transfer_delta"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    oracle_path = Path(args.oracle_report).resolve(strict=True)
    source_path = Path(args.source_report).resolve(strict=True)
    oracle_report = json.loads(oracle_path.read_text())
    source_report = json.loads(source_path.read_text())
    oracle_comparison = oracle_report["comparison_to_gaussian"]
    source_comparison = source_report["comparison_to_gaussian"]
    carriers = {
        "mesh_oracle": _candidate_record(
            oracle_comparison["mesh_oracle"],
            source_comparison["mesh_surface"],
            args.minimum_delta,
        ),
        "sparse_surface": _candidate_record(
            oracle_comparison["mesh_derived_sparse_surface"],
            source_comparison["mesh_derived_sparse_surface"],
            args.minimum_delta,
        ),
    }
    lerf_paths = [Path(value).resolve(strict=True) for value in getattr(args, "lerf_report", [])]
    lerf_scenes = {}
    for path in lerf_paths:
        lerf = json.loads(path.read_text())
        if lerf.get("schema") != "radio_gs.surface_object_memory_v4.lerf_source_mask_geometry_gate.v1":
            raise ValueError(f"unexpected LERF geometry report schema: {path}")
        label = str(lerf["scene_label"])
        if label in lerf_scenes:
            raise ValueError(f"duplicate LERF scene label: {label}")
        lerf_scenes[label] = {
            "passes_scene_gate": bool(lerf["passes_scene_gate"]),
            "primary_directions": lerf["primary_directions"],
            "comparison_to_gaussian": lerf["comparison_to_gaussian"],
            "coverage_is_reported_not_compensatory": bool(
                lerf["coverage_is_reported_not_compensatory"]
            ),
            "projection_configuration": lerf["projection_configuration"],
        }
    evaluated = 1 + len(lerf_scenes)
    all_geometry_scenes_pass = (
        carriers["mesh_oracle"]["passes_scene_gate"]
        and carriers["sparse_surface"]["passes_scene_gate"]
        and all(record["passes_scene_gate"] for record in lerf_scenes.values())
    )
    milestone_complete = evaluated == args.expected_scene_count and all_geometry_scenes_pass
    report = {
        "schema": "radio_gs.surface_object_memory_v4.geometry_gate.v1",
        "cohort_key": args.cohort_key,
        "minimum_directional_delta": args.minimum_delta,
        "minimum_improved_primary_metrics": 3,
        "evaluated_scene_count": evaluated,
        "expected_scene_count": args.expected_scene_count,
        "scene_gate": carriers,
        "lerf_scene_gates": lerf_scenes,
        "scan_oracle_stop_rule_triggered": not carriers["mesh_oracle"]["passes_scene_gate"],
        "sparse_surface_scene_gate_passed": carriers["sparse_surface"]["passes_scene_gate"],
        "milestone_1_complete": milestone_complete,
        "object_codebook_authorized": milestone_complete,
        "object_codebook_authorized_scope": "oracle_only" if milestone_complete else None,
        "query_encoder_authorized": False,
        "compression_authorized": False,
        "object_codebook_block_reason": None if milestone_complete else (
            "geometry ladder cohort incomplete or a preregistered scene failed its directional gate"
        ),
        "inputs": {
            "oracle_report": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
            "source_report": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "lerf_reports": [
                {"path": str(path), "sha256": sha256_file(path)} for path in lerf_paths
            ],
        },
        "metric_policy": {
            "coverage_cannot_compensate_for_purity_regression": True,
            "source_best_target_proposal_transfer_is_non_gating_without_tracks": True,
            "magnitude_is_reported_even_when_directional_gate_is_used": True,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-report", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--cohort-key", required=True)
    parser.add_argument("--expected-scene-count", type=int, required=True)
    parser.add_argument("--minimum-delta", type=float, default=0.0)
    parser.add_argument("--lerf-report", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.expected_scene_count <= 0 or args.minimum_delta < 0:
        parser.error("expected scene count must be positive and minimum delta non-negative")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
