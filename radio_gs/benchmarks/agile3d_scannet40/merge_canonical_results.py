#!/usr/bin/env python3
"""Merge a fixed-scene direct-canonical AGILE3D promotion without recomputing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from radio_gs.field.observation_lifting_contract import (
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES,
)

from .protocol import (
    aggregate_official_metrics,
    interaction_health_metrics,
    load_official_object_list,
)


_PROTOCOL_KEYS = (
    "result_status",
    "formal_comparable",
    "diagnostic_no_support_gate",
    "observation_contract",
    "support_gate_required",
    "minimum_support_fraction",
    "field_checkpoint_name",
    "capability_cache_name",
    "support_graph_name",
    "reliability_cache_name",
    "canonical_mpr_contract",
    "canonical_mpr_coverage_ranked",
    "voxel_size_m",
    "max_clicks",
    "click_policy",
    "clicked_labels_forced",
    "test_set_calibration",
    "world_query",
    "observation_lift",
    "official_point_readout",
    "readout_candidate_k",
    "readout_support_threshold",
    "evaluation_voxel_size_m",
    "click_seed_kernel",
    "seed_candidate_k",
    "hard_seed_topk",
    "seed_temperature",
    "prototype_count",
    "prototype_strategy",
    "solver_type",
    "laplacian_weight",
    "cg_iterations",
    "support_threshold",
    "hard_seed_threshold",
    "hard_seed_conflict_policy",
    "hard_seed_conflict_margin",
    "unary_edge_contrast",
    "world_point_prototype_mode",
    "world_point_max_prototypes",
    "world_point_prototype_weighting",
    "appearance_unary_weight",
    "boundary_unary_weight",
    "feature_calibration",
    "background_centroids",
    "background_negative_policy",
    "calibration_sample_size",
    "centroid_iterations",
    "score_calibration",
    "channel_confidence_mode",
    "negative_spatial_mode",
    "negative_spatial_steps",
    "negative_spatial_decay",
    "spatial_log_weight",
    "spatial_floor",
)

# These two fields were introduced after the first direct-canonical pilot.  A
# missing value has one unambiguous historical behavior, so normalize only
# these metadata omissions while preserving the immutable trajectories.
_HISTORICAL_PROTOCOL_DEFAULTS = {
    "result_status": "historical",
    "formal_comparable": False,
    "diagnostic_no_support_gate": False,
    "field_checkpoint_name": "canonical_mpr_v2.pt",
    "support_gate_required": False,
    "minimum_support_fraction": 0.95,
    "capability_cache_name": "official_dino_sam3_views.pt",
    "support_graph_name": "shared_support_graph_k16.pt",
    "reliability_cache_name": "",
    "canonical_mpr_contract": "canonical-mpr-v1",
    "canonical_mpr_coverage_ranked": False,
    "click_seed_kernel": "evaluator_voxel_convolved",
    "readout_candidate_k": 64,
    "readout_support_threshold": 1e-6,
    "seed_candidate_k": 64,
    "hard_seed_topk": 0,
    "seed_temperature": 1.0,
    "prototype_count": 4,
    "prototype_strategy": "weighted_fps",
    "solver_type": "confidence_random_walker",
    "laplacian_weight": 1.0,
    "cg_iterations": 64,
    "support_threshold": 0.5,
    "hard_seed_threshold": 0.20,
    "hard_seed_conflict_policy": "positive_priority",
    "hard_seed_conflict_margin": 0.0,
    "unary_edge_contrast": 0.0,
    "world_point_max_prototypes": 0,
    "world_point_prototype_weighting": "support_mass",
    "appearance_unary_weight": 1.0,
    "boundary_unary_weight": 0.35,
    "feature_calibration": "none",
    "background_centroids": 0,
    "background_negative_policy": "pooled_mean",
    "calibration_sample_size": 8192,
    "centroid_iterations": 4,
    "score_calibration": "none",
    "channel_confidence_mode": "none",
    "negative_spatial_mode": "none",
    "negative_spatial_steps": 4,
    "negative_spatial_decay": 0.8,
    "spatial_log_weight": 0.25,
    "spatial_floor": 0.01,
}


def _stable_support_record(record: dict[str, Any]) -> dict[str, Any]:
    """Ignore cache reuse bookkeeping when comparing duplicated shard audits."""

    return {
        str(key): value
        for key, value in dict(record).items()
        if str(key) != "geometry_cache_reused"
    }


def merge(
    benchmark_root: str | Path,
    expected_scene_report: str | Path | None,
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    expected_scenes: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate scene/object completeness then aggregate immutable trajectories."""

    if not inputs:
        raise ValueError("at least one direct-canonical AGILE shard is required")
    root = Path(benchmark_root)
    declared_scenes = [str(scene) for scene in expected_scenes]
    if expected_scene_report is not None:
        expected_payload = json.loads(
            Path(expected_scene_report).read_text(encoding="utf-8")
        )
        report_scenes = [
            str(item["scene_id"])
            for item in expected_payload.get("scene_coverage", [])
        ]
        if declared_scenes and report_scenes != declared_scenes:
            raise ValueError(
                "expected-scene-report and explicit expected-scenes disagree"
            )
        declared_scenes = report_scenes
    if not declared_scenes or len(set(declared_scenes)) != len(declared_scenes):
        raise ValueError(
            "provide one ordered duplicate-free scene list through an expected "
            "scene report or --expected-scenes"
        )
    objects = [
        item for item in load_official_object_list(root) if item.scene_id in set(declared_scenes)
    ]
    expected_by_key = {(item.scene_id, int(item.object_id)): item for item in objects}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    support: dict[str, dict[str, Any]] = {}
    common_protocol: dict[str, Any] | None = None
    for raw_path in inputs:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("benchmark") != "AGILE3D ScanNet40 single-object":
            raise ValueError(f"{path} is not an AGILE3D report")
        protocol = dict(payload.get("protocol", {}))
        missing = set(_PROTOCOL_KEYS) - set(protocol) - set(_HISTORICAL_PROTOCOL_DEFAULTS)
        if missing:
            raise ValueError(f"{path} lacks direct-canonical protocol keys: {sorted(missing)}")
        selected_protocol = {
            key: protocol.get(key, _HISTORICAL_PROTOCOL_DEFAULTS.get(key))
            for key in _PROTOCOL_KEYS
        }
        if common_protocol is None:
            common_protocol = selected_protocol
        elif common_protocol != selected_protocol:
            raise ValueError(f"{path} has a different direct-canonical protocol")
        for record in payload.get("scene_support", []):
            scene = str(record.get("scene_id", ""))
            if scene not in declared_scenes:
                raise ValueError(f"{path} has an unknown support record: {scene}")
            if "continuous_support_fraction" not in record:
                raise ValueError(f"{path} support record lacks continuous_support_fraction")
            existing = support.get(scene)
            if existing is not None:
                # Object shards repeat the same label-free support audit for a
                # scene.  It is valid only if the field/source record is
                # identical; a differing record would mean the two shards did
                # not evaluate the same canonical field and cannot be merged.
                if _stable_support_record(existing) != _stable_support_record(dict(record)):
                    raise ValueError(
                        f"{path} has a conflicting duplicate support record: {scene}"
                    )
                continue
            support[scene] = dict(record)
        for row in payload.get("rows", []):
            key = (str(row.get("scene_id", "")), int(row.get("object_id", -1)))
            expected = expected_by_key.get(key)
            if expected is None or str(row.get("semantic_class", "")) != expected.semantic_class:
                raise ValueError(f"{path} reports an invalid AGILE object: {key}")
            if key in rows:
                raise ValueError(f"duplicate direct-canonical AGILE object: {key}")
            trajectory = {int(step): float(value) for step, value in row.get("trajectory", {}).items()}
            if set(trajectory) != set(range(1, int(common_protocol["max_clicks"]) + 1)):
                raise ValueError(f"{path} has incomplete trajectory for {key}")
            rows[key] = dict(row)
    missing_scenes = [scene for scene in declared_scenes if scene not in support]
    missing_objects = [item.key for item in objects if (item.scene_id, int(item.object_id)) not in rows]
    if missing_scenes or missing_objects:
        raise ValueError(
            f"direct-canonical merge incomplete: scenes={missing_scenes[:5]}, objects={missing_objects[:5]}"
        )
    assert common_protocol is not None
    ordered_rows = [rows[(item.scene_id, int(item.object_id))] for item in objects]
    ordered_support = [support[scene] for scene in declared_scenes]
    fractions = np.asarray(
        [float(item["continuous_support_fraction"]) for item in ordered_support], dtype=np.float64
    )
    if str(common_protocol["observation_contract"]) in {
        "scannet_full_observation_pilot",
        "scannet_full_observation_v1",
    }:
        if not bool(common_protocol.get("support_gate_required", False)):
            raise ValueError(
                "full-observation results must have passed the support gate"
            )
        threshold = float(common_protocol.get("minimum_support_fraction", 0.0))
        if threshold <= 0 or bool((fractions < threshold).any()):
            raise ValueError(
                "full-observation merge contains a scene below its "
                "declared continuous-support gate"
            )
        if (
            str(common_protocol.get("canonical_mpr_contract", ""))
            not in CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES
            or not bool(common_protocol.get("canonical_mpr_coverage_ranked", False))
        ):
            raise ValueError(
                "full-observation merge requires coverage-ranked canonical "
                "full-observation MPR evidence"
            )
    report: dict[str, Any] = {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "protocol": {
            "official_preprocessed_data": str(root.resolve()),
            "expected_scene_report": (
                str(Path(expected_scene_report).resolve())
                if expected_scene_report is not None
                else ""
            ),
            "expected_scenes": declared_scenes,
            "objects": len(ordered_rows),
            "scenes": len(ordered_support),
            **common_protocol,
        },
        "scene_support": ordered_support,
        "support_summary": {
            "mean_continuous_support_fraction": float(fractions.mean()),
            "minimum_continuous_support_fraction": float(fractions.min()),
            "scenes_meeting_095": int((fractions >= 0.95).sum()),
            "scenes_total": int(len(fractions)),
        },
        "metrics": aggregate_official_metrics(
            [row["trajectory"] for row in ordered_rows],
            max_clicks=int(common_protocol["max_clicks"]),
        ),
        "interaction_health": interaction_health_metrics(
            [row["trajectory"] for row in ordered_rows],
            seed_satisfaction=(
                [row["seed_satisfaction"] for row in ordered_rows]
                if all("seed_satisfaction" in row for row in ordered_rows)
                else None
            ),
            max_clicks=int(common_protocol["max_clicks"]),
        ),
        "rows": ordered_rows,
        "merge": {
            "source_shards": [str(Path(path).resolve()) for path in inputs],
            "prediction_recomputed": False,
            "clicks_recomputed": False,
            "object_shards_merged": [
                dict(json.loads(Path(path).read_text(encoding="utf-8")).get("shard", {}))
                for path in inputs
            ],
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--expected-scene-report", default="")
    parser.add_argument(
        "--expected-scenes",
        default="",
        help="comma/space-delimited fixed pilot scenes; an alternative to an expected scene report",
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected_scenes = [
        value
        for value in str(args.expected_scenes).replace(",", " ").split()
        if value
    ]
    if not str(args.expected_scene_report).strip() and not expected_scenes:
        parser.error("provide --expected-scene-report or --expected-scenes")
    print(
        json.dumps(
            merge(
                args.benchmark_root,
                args.expected_scene_report or None,
                args.inputs,
                args.output,
                expected_scenes=expected_scenes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
