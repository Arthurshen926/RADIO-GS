#!/usr/bin/env python3
"""Summarize completed Easy3D shards without promoting a partial run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluate_easy3d import (
    EVALUATOR_SCHEMA_VERSION,
    PAPER_IOU,
    aggregate_trajectory_rows,
    reference_cohort_audit,
)
from .protocol import load_official_object_list


SUMMARY_SCHEMA = "easy3d-hardware-interrupted-prefix-diagnostic-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_shard_set_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def summarize_interrupted_prefix(
    *,
    run_root: Path,
    data_root: Path,
    agile_reference_csv: Path,
    failure_record: Mapping[str, Any],
) -> dict[str, Any]:
    shards = sorted((run_root / "scene_shards").glob("*.json"))
    completed = failure_record["completed_prefix"]
    if len(shards) != int(completed["scene_shard_count"]):
        raise ValueError("completed shard count changed after failure record")
    objects = load_official_object_list(data_root)
    all_scene_ids = sorted({item.scene_id for item in objects})
    completed_scene_ids = [path.stem for path in shards]
    if completed_scene_ids != all_scene_ids[: len(shards)]:
        raise ValueError("completed shards are not the formal scene prefix")

    rows: list[dict[str, Any]] = []
    elapsed_seconds = 0.0
    expected_common = {
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "checkpoint_sha256": failure_record["assets"][
            "checkpoint_sha256"
        ],
        "easy3d_commit": failure_record["assets"]["easy3d_commit"],
        "object_ids_sha256": failure_record["assets"][
            "object_ids_sha256"
        ],
        "preprocessing_manifest_sha256": failure_record["assets"][
            "preprocessing_manifest_sha256"
        ],
        "interaction_contract": "agile3d_release",
        "max_clicks": 10,
        "amp_bfloat16": True,
        "object_batch_size": 4,
    }
    for path in shards:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected_common.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{path.name}: interrupted shard provenance mismatch: "
                f"{mismatches}"
            )
        if payload.get("scene_id") != path.stem:
            raise ValueError(f"{path.name}: scene ID mismatch")
        shard_rows = list(payload.get("rows", []))
        if len(shard_rows) != len(payload.get("object_keys", [])):
            raise ValueError(f"{path.name}: incomplete object rows")
        rows.extend(shard_rows)
        elapsed_seconds += float(payload["elapsed_seconds"])
    keys = [str(row["key"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("interrupted prefix repeats object keys")
    failures = [
        row
        for row in rows
        if str(row.get("status", "evaluated")) != "evaluated"
    ]
    evaluated = [
        row
        for row in rows
        if str(row.get("status", "evaluated")) == "evaluated"
    ]
    if len(rows) != int(completed["evaluated_object_count"]):
        raise ValueError("interrupted object count changed after failure record")

    cohort_audit = reference_cohort_audit(objects, agile_reference_csv)
    legacy_keys = set(cohort_audit["legacy_matched_keys"])
    legacy_rows = [row for row in evaluated if str(row["key"]) in legacy_keys]
    complete_metrics = aggregate_trajectory_rows(
        evaluated, max_clicks=10
    )
    legacy_metrics = aggregate_trajectory_rows(
        legacy_rows, max_clicks=10
    )
    paper_gaps = {
        key: float(complete_metrics[key] - value)
        for key, value in PAPER_IOU.items()
    }
    return {
        "summary_schema": SUMMARY_SCHEMA,
        "status": "hardware_interrupted_diagnostic",
        "not_formal_result": True,
        "paper_metric_comparable": False,
        "warning": (
            "This is a lexicographic 68-scene prefix stopped by GPU loss, "
            "not the 312-scene formal cohort and not a representative pilot."
        ),
        "run_root": str(run_root.resolve()),
        "failure_record_path": str(
            (run_root / "formal_attempt_001_failure.json").resolve()
        ),
        "failure_record_sha256": _sha256(
            run_root / "formal_attempt_001_failure.json"
        ),
        "scene_shard_count": len(shards),
        "formal_scene_count": 312,
        "scene_completion_fraction": float(len(shards) / 312),
        "object_row_count": len(rows),
        "formal_object_count": 10357,
        "object_completion_fraction": float(len(rows) / 10357),
        "evaluated_object_count": len(evaluated),
        "failed_object_count": len(failures),
        "sum_scene_elapsed_seconds": elapsed_seconds,
        "first_scene": completed_scene_ids[0],
        "last_scene": completed_scene_ids[-1],
        "next_scene_to_recompute": failure_record["completed_prefix"][
            "next_scene_to_recompute"
        ],
        "canonical_scene_shard_set_sha256": (
            _canonical_shard_set_sha256(shards)
        ),
        "provenance": expected_common,
        "cohorts": {
            "interrupted_complete_release_prefix": {
                "object_count": len(rows),
                "evaluated_object_count": len(evaluated),
                "failed_object_count": len(failures),
                "metrics_query_micro": complete_metrics,
                "paper_gap_diagnostic": paper_gaps,
            },
            "interrupted_legacy_key_intersection_prefix": {
                "object_count": len(
                    [row for row in rows if str(row["key"]) in legacy_keys]
                ),
                "evaluated_object_count": len(legacy_rows),
                "failed_object_count": len(
                    [
                        row
                        for row in failures
                        if str(row["key"]) in legacy_keys
                    ]
                ),
                "metrics_query_micro": legacy_metrics,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--agile-reference-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    failure_path = run_root / "formal_attempt_001_failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    report = summarize_interrupted_prefix(
        run_root=run_root,
        data_root=Path(args.data_root),
        agile_reference_csv=Path(args.agile_reference_csv),
        failure_record=failure,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != report:
            raise ValueError(
                "existing interrupted-prefix diagnostic differs"
            )
    else:
        output.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scene_shard_count": report["scene_shard_count"],
                "object_row_count": report["object_row_count"],
                "failed_object_count": report["failed_object_count"],
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
