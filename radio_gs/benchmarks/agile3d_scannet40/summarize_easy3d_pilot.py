#!/usr/bin/env python3
"""Validate paired Easy3D reports and freeze a source-grounded protocol policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .evaluate_easy3d import PAPER_IOU


SUMMARY_SCHEMA = "easy3d-agile3d-dual-contract-pilot-summary-v2"
CONTRACTS = ("agile3d_release", "easy3d_released_code")
COMMON_PROVENANCE_KEYS = (
    "easy3d_commit",
    "checkpoint_sha256",
    "dataset_root",
    "object_ids_sha256",
    "object_classes_sha256",
    "scene_count",
    "object_count",
    "voxel_size_m",
    "max_scene_size_m",
    "preprocessing",
    "preprocessing_cache_schema",
    "preprocessing_manifest_sha256",
    "max_clicks",
    "amp_bfloat16",
    "object_batch_size",
    "formal_selection",
    "runtime",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_summary(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    cohort = report["cohorts"]["complete_release_selection"]
    metrics = cohort["metrics_query_micro"]
    gaps = {
        key: float(metrics[key] - paper)
        for key, paper in PAPER_IOU.items()
    }
    absolute_gaps = [abs(value) for value in gaps.values()]
    return {
        "results_path": str(report_path.resolve()),
        "results_sha256": _sha256(report_path),
        "object_count": int(cohort["object_count"]),
        "evaluated_object_count": int(cohort["evaluated_object_count"]),
        "failed_object_count": int(cohort["failed_object_count"]),
        "metrics_query_micro": {
            key: float(metrics[key]) for key in PAPER_IOU
        },
        "paper_gap": gaps,
        "paper_gap_percentage_points": {
            key: float(value * 100.0) for key, value in gaps.items()
        },
        "mean_absolute_paper_gap_percentage_points": float(
            100.0 * sum(absolute_gaps) / len(absolute_gaps)
        ),
        "maximum_absolute_paper_gap_percentage_points": float(
            100.0 * max(absolute_gaps)
        ),
        "sum_scene_elapsed_seconds": float(elapsed_seconds),
    }


def summarize_reports(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    report_paths: Mapping[str, Path],
    elapsed_seconds: Mapping[str, float],
) -> dict[str, Any]:
    if set(reports) != set(CONTRACTS):
        raise ValueError(f"paired pilot requires contracts {CONTRACTS}")
    for contract, report in reports.items():
        provenance = report.get("provenance", {})
        if provenance.get("interaction_contract") != contract:
            raise ValueError(f"{contract}: report contract mismatch")
        if bool(provenance.get("formal_selection")):
            raise ValueError(f"{contract}: formal output is not a pilot")
        if report.get("status") != "declared_pilot_or_partial":
            raise ValueError(f"{contract}: unexpected pilot status")
        if report.get("object_failures"):
            raise ValueError(f"{contract}: pilot contains object failures")
    first = reports[CONTRACTS[0]]
    second = reports[CONTRACTS[1]]
    common_mismatches = {
        key: {
            CONTRACTS[0]: first["provenance"].get(key),
            CONTRACTS[1]: second["provenance"].get(key),
        }
        for key in COMMON_PROVENANCE_KEYS
        if first["provenance"].get(key) != second["provenance"].get(key)
    }
    if common_mismatches:
        raise ValueError(
            f"paired pilot provenance differs: {common_mismatches}"
        )
    first_keys = [str(row["key"]) for row in first["rows"]]
    second_keys = [str(row["key"]) for row in second["rows"]]
    if first_keys != second_keys or len(set(first_keys)) != len(first_keys):
        raise ValueError("paired pilot object rows do not align exactly")

    summaries = {
        contract: _contract_summary(
            reports[contract],
            report_path=report_paths[contract],
            elapsed_seconds=elapsed_seconds[contract],
        )
        for contract in CONTRACTS
    }
    # The paper explicitly says it follows the AGILE3D benchmark.  Freeze that
    # source-grounded contract before inspecting any pilot-to-paper gap; using
    # a published test row as a protocol selector would be post-hoc
    # calibration.  The released Easy3D forward remains a coextensive
    # sensitivity contract.
    primary = "agile3d_release"
    metric_advantage = {
        key: float(
            100.0
            * (
                summaries["agile3d_release"]["metrics_query_micro"][key]
                - summaries["easy3d_released_code"][
                    "metrics_query_micro"
                ][key]
            )
        )
        for key in PAPER_IOU
    }
    provenance = first["provenance"]
    return {
        "summary_schema": SUMMARY_SCHEMA,
        "status": "complete_protocol_sensitivity_audit",
        "pilot_scene_count": int(provenance["scene_count"]),
        "pilot_object_count": int(provenance["object_count"]),
        "pilot_scene_ids": sorted(
            {str(row["scene_id"]) for row in first["rows"]}
        ),
        "paper_metrics": dict(PAPER_IOU),
        "common_provenance": {
            key: provenance[key] for key in COMMON_PROVENANCE_KEYS
        },
        "contracts": summaries,
        "agile3d_release_advantage_percentage_points": metric_advantage,
        "primary_contract": primary,
        "paper_gap_used_for_contract_selection": False,
        "primary_contract_basis": (
            "source-grounded: the Easy3D paper states that quantitative "
            "evaluation follows the AGILE3D benchmark"
        ),
        "formal_run_policy": (
            "run agile3d_release as the paper-facing primary and "
            "easy3d_released_code as a full-cohort implementation "
            "sensitivity on the same 312 scenes and 10,357 objects"
        ),
    }


def _scene_elapsed(report_path: Path) -> float:
    shard_root = report_path.parent / "scene_shards"
    shards = sorted(shard_root.glob("*.json"))
    if not shards:
        raise ValueError(f"pilot has no scene shards: {shard_root}")
    return float(
        sum(
            float(json.loads(path.read_text(encoding="utf-8"))[
                "elapsed_seconds"
            ])
            for path in shards
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agile3d-release-results", required=True)
    parser.add_argument("--easy3d-released-code-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "agile3d_release": Path(args.agile3d_release_results),
        "easy3d_released_code": Path(args.easy3d_released_code_results),
    }
    reports = {
        contract: json.loads(path.read_text(encoding="utf-8"))
        for contract, path in paths.items()
    }
    summary = summarize_reports(
        reports,
        report_paths=paths,
        elapsed_seconds={
            contract: _scene_elapsed(path)
            for contract, path in paths.items()
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != summary:
            raise ValueError("existing pilot summary differs; use a new path")
    else:
        output.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "primary_contract": summary["primary_contract"],
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
