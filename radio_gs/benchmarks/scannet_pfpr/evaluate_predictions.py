#!/usr/bin/env python3
"""Evaluate score vectors on the frozen ScanNet-PFPR candidate point domain."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import (
    ProtocolConfig,
    SUPPORTED_BENCHMARK_VERSIONS,
    aggregate_query_metrics,
    evaluate_ranked_locations,
    fixed_radius_nms,
    protocol_config_from_record,
)


def _config_from_manifest(payload: dict[str, Any]) -> ProtocolConfig:
    return protocol_config_from_record(
        str(payload.get("benchmark_version", "")),
        payload.get("protocol_config", {}),
    )


def evaluate(
    benchmark_dir: str | Path,
    prediction_dir: str | Path,
    output: str | Path,
    *,
    scene_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Evaluate one finite score vector per query without opening instances."""

    root = Path(benchmark_dir)
    predictions = Path(prediction_dir)
    private = json.loads((root / "manifest.evaluator.json").read_text(encoding="utf-8"))
    benchmark_version = str(private.get("benchmark_version", ""))
    if benchmark_version not in SUPPORTED_BENCHMARK_VERSIONS:
        raise ValueError("not a supported ScanNet-PFPR evaluator manifest")
    config = _config_from_manifest(private)
    requested = {str(value) for value in scene_names if str(value)}
    candidate_by_scene: dict[str, np.ndarray] = {}
    for record in private.get("scene_domains", []):
        scene_id = str(record.get("scene_id", ""))
        path = Path(str(record.get("candidate_xyz_path", "")))
        if not scene_id or not path.is_file() or scene_id in candidate_by_scene:
            raise ValueError("PFPR evaluator has an invalid or duplicate scene domain")
        xyz = np.load(path, allow_pickle=False)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError(f"PFPR candidate domain is invalid: {path}")
        candidate_by_scene[scene_id] = np.asarray(xyz, dtype=np.float32)
    if requested:
        unknown = sorted(requested - set(candidate_by_scene))
        if unknown:
            raise ValueError(f"PFPR pilot requests unknown scenes: {unknown}")
        candidate_by_scene = {
            scene: values
            for scene, values in candidate_by_scene.items()
            if scene in requested
        }

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    maximum = max(config.retrieval_ks)
    for query in private.get("queries", []):
        query_id = str(query.get("query_id", ""))
        scene_id = str(query.get("scene_id", ""))
        if requested and scene_id not in requested:
            continue
        anchor = np.asarray(query.get("anchor_world_xyz"), dtype=np.float32)
        if not query_id or query_id in seen or scene_id not in candidate_by_scene:
            raise ValueError("PFPR evaluator query/domain alignment is invalid")
        seen.add(query_id)
        score_path = predictions / f"{query_id}.npy"
        if not score_path.is_file():
            raise FileNotFoundError(f"missing PFPR score vector: {score_path}")
        scores = np.load(score_path, allow_pickle=False)
        candidates = candidate_by_scene[scene_id]
        if np.asarray(scores).shape != (len(candidates),):
            raise ValueError(f"{query_id}: score vector does not align with public candidate domain")
        selected = fixed_radius_nms(
            candidates,
            scores,
            radius_m=config.nms_radius_m,
            maximum=maximum,
        )
        if not len(selected):
            raise AssertionError("finite score vector must yield at least one PFPR hypothesis")
        metrics = evaluate_ranked_locations(candidates[selected], anchor, config=config)
        rows.append(
            {
                "query_id": query_id,
                "scene_id": scene_id,
                "selected_candidate_indices": [int(value) for value in selected],
                **metrics,
            }
        )
    if not rows:
        raise ValueError("PFPR evaluator received no queries")
    query_micro = aggregate_query_metrics(rows, config=config)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(row)
    per_scene = {
        scene: aggregate_query_metrics(scene_rows, config=config)
        for scene, scene_rows in sorted(by_scene.items())
    }
    scene_macro = {
        key: float(np.mean([metrics[key] for metrics in per_scene.values()]))
        for key in query_micro
    }
    report: dict[str, Any] = {
        "benchmark": benchmark_version,
        "protocol": {
            "candidate_domain": "public_annotation_mesh_geometry_quantized_5cm",
            "score_input": "one_query_score_vector_aligned_to_public_candidate_domain",
            "spatial_nms_radius_m": float(config.nms_radius_m),
            "distance": "euclidean_3d_anchor_distance",
            "instance_identity_used": False,
            "query_pose_or_depth_used_by_method": False,
            "query_count": len(rows),
            "scene_count": len(per_scene),
            "scene_selection": (
                "all_frozen_scenes"
                if not requested
                else "evaluator_only_declared_pilot_subset"
            ),
        },
        "metrics_query_micro": query_micro,
        "metrics_scene_macro": scene_macro,
        "per_scene": per_scene,
        "rows": rows,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scene-names",
        default="",
        help="evaluator-only pilot subset; omit for the frozen 20-scene result",
    )
    args = parser.parse_args()
    scenes = tuple(
        value for value in str(args.scene_names).replace(",", " ").split() if value
    )
    print(
        json.dumps(
            evaluate(
                args.benchmark_dir,
                args.prediction_dir,
                args.output,
                scene_names=scenes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
