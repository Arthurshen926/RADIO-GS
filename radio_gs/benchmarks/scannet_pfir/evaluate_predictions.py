#!/usr/bin/env python3
"""Evaluate mesh-domain PFIR predictions using evaluator-only GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radio_gs.benchmarks.scannet_pfir.build_benchmark import find_scene_annotations
from radio_gs.benchmarks.scannet_pfir.evaluation import (
    evaluate_instance_ranking,
    evaluate_instance_selection,
)
from radio_gs.benchmarks.scannet_pfir.protocol import load_mesh_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--annotations-root", action="append", required=True)
    parser.add_argument("--track", choices=("ranking", "selection"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    benchmark_root = Path(args.benchmark_dir)
    internal = json.loads(
        (benchmark_root / "manifest.internal.json").read_text(encoding="utf-8")
    )
    records = internal["queries"]
    scenes = sorted({str(record["scene_id"]) for record in records})
    xyz_by_scene, instances_by_scene = {}, {}
    for scene in scenes:
        mesh, aggregation, segmentation = find_scene_annotations(
            scene, args.annotations_root
        )
        xyz, instances, _ = load_mesh_instances(mesh, aggregation, segmentation)
        xyz_by_scene[scene], instances_by_scene[scene] = xyz, instances
    prediction_root = Path(args.prediction_dir)
    predictions = {
        str(record["query_id"]): np.load(
            prediction_root / f"{record['query_id']}.npy", allow_pickle=False
        )
        for record in records
    }
    if args.track == "ranking":
        report = evaluate_instance_ranking(records, predictions, instances_by_scene)
    else:
        report = evaluate_instance_selection(
            records, predictions, instances_by_scene, xyz_by_scene
        )
    report["benchmark_version"] = internal["benchmark_version"]
    report["prediction_domain"] = "official_scannet_annotation_mesh_vertices"
    report["test_calibration"] = False
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
