#!/usr/bin/env python3
"""Aggregate LUDVIG protocol results without mixing scenes, runs, or cohorts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import yaml


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "paper" / "artifacts" / "promptable_nvs_protocol_registry.yaml"


class AggregationError(RuntimeError):
    pass


def _paper_context(registry: dict) -> dict:
    return next(
        row
        for row in registry["reported_context"]
        if row["method_id"] == "marrie_et_al_iccv_2025_ludvig_sam"
    )


def aggregate(input_root: Path, benchmark: str) -> dict:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    context = _paper_context(registry)
    records: dict[str, dict[int, float]] = defaultdict(dict)
    checkpoint_hashes: dict[str, set[str]] = defaultdict(set)
    manifests: list[dict] = []
    for manifest_path in sorted(input_root.rglob("run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        expected = "NVOS" if benchmark == "nvos" else "SPIn-NeRF"
        if manifest.get("benchmark") != expected:
            continue
        if manifest.get("protocol_id") != "ludvig_official_online_multiview_v1":
            raise AggregationError(f"Unexpected protocol in {manifest_path}")
        if manifest.get("strict_unseen_exact_match") is not False:
            raise AggregationError(f"Unsafe strict-unseen label in {manifest_path}")
        geometry_protocol = manifest.get("geometry_protocol")
        expected_training_visibility = geometry_protocol == "released_all_view"
        if (
            manifest.get(
                "target_rgb_visible_during_gaussian_splatting_training"
            )
            is not expected_training_visibility
        ):
            raise AggregationError(
                "Geometry label/target-training visibility mismatch in "
                f"{manifest_path}"
            )
        checkpoint_hash = manifest.get("gs_source_sha256")
        if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
            raise AggregationError(
                f"Missing gs_source_sha256 in {manifest_path}"
            )
        result_paths = list(manifest_path.parent.rglob("protocol_result.json"))
        if len(result_paths) != 1:
            raise AggregationError(
                f"Expected exactly one result below {manifest_path.parent}; "
                f"found {len(result_paths)}"
            )
        result = json.loads(result_paths[0].read_text(encoding="utf-8"))
        score = (
            result["selected_iou"]
            if benchmark == "nvos"
            else result["scene_mean_iou"]
        )
        scene = manifest["scene"]
        seed = manifest.get("seed")
        if not isinstance(seed, int):
            raise AggregationError(f"Missing integer seed in {manifest_path}")
        if seed in records[scene]:
            raise AggregationError(
                f"Duplicate completed scene/seed {scene}/{seed}; paper run "
                f"aggregation requires unique seeds"
            )
        records[scene][seed] = 100.0 * float(score)
        checkpoint_hashes[scene].add(checkpoint_hash)
        manifests.append(manifest)
    if not records:
        raise AggregationError(f"No completed {benchmark} results under {input_root}")
    geometry_protocols = {
        manifest.get("geometry_protocol") for manifest in manifests
    }
    if len(geometry_protocols) != 1 or None in geometry_protocols:
        raise AggregationError(
            "All aggregated runs must declare one identical geometry_protocol; "
            f"found {sorted(str(item) for item in geometry_protocols)}"
        )
    geometry_protocol = next(iter(geometry_protocols))
    inconsistent_checkpoints = {
        scene: sorted(hashes)
        for scene, hashes in checkpoint_hashes.items()
        if len(hashes) != 1
    }
    if inconsistent_checkpoints:
        raise AggregationError(
            "All seeds of a scene must evaluate the same 3DGS checkpoint; "
            f"found {inconsistent_checkpoints}"
        )

    per_scene = {
        scene: {
            "seeds": sorted(seed_values),
            "run_values_iou_percent": [
                seed_values[seed] for seed in sorted(seed_values)
            ],
            "local_mean_iou_percent": mean(seed_values.values()),
            "paper_iou_percent": context["published_per_scene_iou"][
                "nvos" if benchmark == "nvos" else "spin_nerf"
            ][scene],
            "delta_local_minus_paper": mean(seed_values.values())
            - context["published_per_scene_iou"][
                "nvos" if benchmark == "nvos" else "spin_nerf"
            ][scene],
            "num_runs": len(seed_values),
            "gs_source_sha256": next(iter(checkpoint_hashes[scene])),
        }
        for scene, seed_values in sorted(records.items())
    }
    local_macro = mean(row["local_mean_iou_percent"] for row in per_scene.values())
    paper_same_scene_macro = mean(row["paper_iou_percent"] for row in per_scene.values())
    expected_scenes = (
        registry["protocols"]["ludvig_official_online_multiview_v1"]["nvos"]["tasks"]
        if benchmark == "nvos"
        else registry["protocols"][
            "ludvig_spin_nerf_9scene_without_fork_diagnostic_v1"
        ]["scenes"]
    )
    missing = sorted(set(expected_scenes) - set(per_scene))
    extra = sorted(set(per_scene) - set(expected_scenes))
    complete_cohort = not missing and not extra
    all_scenes_have_three_runs = all(
        row["num_runs"] == 3 for row in per_scene.values()
    )
    required_paper_seeds = [0, 1, 2]
    all_scenes_have_required_seeds = all(
        row["seeds"] == required_paper_seeds for row in per_scene.values()
    )
    released_all_view_geometry = geometry_protocol == "released_all_view"
    per_scene_three_seed_check = (
        released_all_view_geometry and all_scenes_have_required_seeds
    )
    for row in per_scene.values():
        row["eligible_for_three_seed_paper_protocol_check"] = (
            released_all_view_geometry
            and row["seeds"] == required_paper_seeds
        )
    summary = {
        "schema_version": 1,
        "method": "LUDVIG-SAM",
        "benchmark": "NVOS" if benchmark == "nvos" else "SPIn-NeRF",
        "protocol_id": "ludvig_official_online_multiview_v1",
        "geometry_protocol": geometry_protocol,
        "metric": "foreground_iou",
        "metric_source": (
            "selected_iou_fixed_threshold"
            if benchmark == "nvos"
            else "scene_mean_iou_after_reference_calibration"
        ),
        "oracle_values_aggregated": False,
        "aggregation": "seed_mean_per_scene_then_equal_weight_scene_macro",
        "strict_unseen_exact_match": False,
        "cohort": list(per_scene),
        "expected_cohort": expected_scenes,
        "missing_scenes": missing,
        "extra_scenes": extra,
        "complete_requested_cohort": complete_cohort,
        "per_scene": per_scene,
        "local_scene_macro_iou_percent": local_macro,
        "paper_same_scene_macro_iou_percent": paper_same_scene_macro,
        "delta_local_minus_paper": local_macro - paper_same_scene_macro,
        "paper_full_benchmark_iou_percent": context["values"][
            "nvos" if benchmark == "nvos" else "spin_nerf"
        ]["miou"],
        "eligible_for_full_cohort_single_run_report": complete_cohort,
        "eligible_for_full_cohort_three_seed_hybrid_report": (
            complete_cohort and all_scenes_have_required_seeds
        ),
        "eligible_for_paper_protocol_comparison": (
            complete_cohort
            and all_scenes_have_required_seeds
            and released_all_view_geometry
        ),
        "eligible_for_per_scene_three_seed_paper_protocol_check": (
            per_scene_three_seed_check
        ),
        "eligible_for_strict_unseen_claim": False,
        "paper_requires_three_runs": True,
        "required_paper_seeds": required_paper_seeds,
        "all_scenes_have_three_runs": all_scenes_have_three_runs,
        "all_scenes_have_required_seeds": all_scenes_have_required_seeds,
        "released_all_view_geometry": released_all_view_geometry,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("nvos", "spin"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.input_root, args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
