#!/usr/bin/env python3
"""Merge disjoint exact AGILE3D evaluator shards into one formal report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .protocol import aggregate_official_metrics, load_official_object_list


def merge(
    benchmark_root: str | Path,
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    max_clicks: int = 20,
) -> dict:
    """Merge scene-disjoint evaluator reports without changing a trajectory.

    A shard is produced by the ordinary evaluator with ``--scene-names``.
    This function never recomputes predictions or clicks: it only validates
    exact official object coverage, restores official object order, and
    re-aggregates the released metrics over all trajectories.
    """

    root = Path(benchmark_root)
    expected = load_official_object_list(root)
    expected_by_key = {(item.scene_id, int(item.object_id)): item for item in expected}
    row_by_key: dict[tuple[str, int], dict] = {}
    coverage_by_scene: dict[str, dict] = {}
    protocol_values: dict[str, object] | None = None

    if not inputs:
        raise ValueError("at least one shard report is required")
    for raw_path in inputs:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        if payload.get("benchmark") != "AGILE3D ScanNet40 single-object":
            raise ValueError(f"{path} is not an AGILE3D single-object report")
        protocol = dict(payload.get("protocol", {}))
        required = {
            "voxel_size_m",
            "max_clicks",
            "click_policy",
            "clicked_labels_forced",
            "test_set_calibration",
            "selection_mode",
            "unary_mode",
            "appearance_unary_weight",
            "boundary_unary_weight",
            "observation_lift_mode",
            "observation_lift_neighbors",
            "observation_lift_maximum_distance_m",
        }
        missing = required - set(protocol)
        if missing:
            raise ValueError(f"{path} lacks formal protocol keys: {sorted(missing)}")
        if protocol_values is None:
            protocol_values = {key: protocol[key] for key in required}
        elif protocol_values != {key: protocol[key] for key in required}:
            raise ValueError(f"{path} uses a different evaluator/query protocol")
        for scene in payload.get("scene_coverage", []):
            scene_id = str(scene.get("scene_id", ""))
            if scene_id not in {item.scene_id for item in expected}:
                raise ValueError(f"{path} reports unknown scene coverage: {scene_id}")
            previous = coverage_by_scene.get(scene_id)
            if previous is not None and previous != scene:
                raise ValueError(f"conflicting coverage records for {scene_id}")
            coverage_by_scene[scene_id] = scene
        for row in payload.get("rows", []):
            key = (str(row.get("scene_id", "")), int(row.get("object_id", -1)))
            expected_item = expected_by_key.get(key)
            if expected_item is None:
                raise ValueError(f"{path} reports unknown AGILE3D object {key}")
            if str(row.get("semantic_class", "")) != expected_item.semantic_class:
                raise ValueError(f"{path} semantic class disagrees for {key}")
            if key in row_by_key:
                raise ValueError(f"duplicate AGILE3D object across shards: {key}")
            trajectory = {int(step): float(value) for step, value in row["trajectory"].items()}
            if set(trajectory) != set(range(1, int(max_clicks) + 1)):
                raise ValueError(f"{path} has incomplete trajectory for {key}")
            row_by_key[key] = row

    missing_objects = [
        item.key for item in expected if (item.scene_id, int(item.object_id)) not in row_by_key
    ]
    if missing_objects:
        raise ValueError(
            f"merged AGILE3D report misses {len(missing_objects)} official objects, "
            f"including {missing_objects[:5]}"
        )
    expected_scenes = {item.scene_id for item in expected}
    missing_scenes = sorted(expected_scenes - set(coverage_by_scene))
    if missing_scenes:
        raise ValueError(
            f"merged AGILE3D report lacks coverage for {len(missing_scenes)} scenes, "
            f"including {missing_scenes[:5]}"
        )

    ordered_rows = [row_by_key[(item.scene_id, int(item.object_id))] for item in expected]
    ordered_coverage = [coverage_by_scene[scene] for scene in sorted(expected_scenes)]
    assert protocol_values is not None
    report = {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "protocol": {
            "official_preprocessed_data": str(root.resolve()),
            "objects": len(ordered_rows),
            "scenes": len(ordered_coverage),
            **protocol_values,
        },
        "scene_coverage": ordered_coverage,
        "coverage_summary": {
            "mean_feature_coverage": float(
                sum(float(row["feature_coverage"]) for row in ordered_coverage)
                / len(ordered_coverage)
            ),
            "minimum_feature_coverage": float(
                min(float(row["feature_coverage"]) for row in ordered_coverage)
            ),
            "mean_projectable_fraction": float(
                sum(
                    float(row["observation_lift"]["projectable_fraction"])
                    for row in ordered_coverage
                )
                / len(ordered_coverage)
            ),
            "minimum_projectable_fraction": float(
                min(
                    float(row["observation_lift"]["projectable_fraction"])
                    for row in ordered_coverage
                )
            ),
        },
        "metrics": aggregate_official_metrics(
            [row["trajectory"] for row in ordered_rows], max_clicks=int(max_clicks)
        ),
        "rows": ordered_rows,
        "merge": {
            "source_shards": [str(Path(path).resolve()) for path in inputs],
            "prediction_recomputed": False,
            "clicks_recomputed": False,
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-clicks", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            merge(
                args.benchmark_root,
                args.inputs,
                args.output,
                max_clicks=args.max_clicks,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
