#!/usr/bin/env python3
"""Inventory local inputs for the preregistered ScanNet-UQIS-9 cohort.

This is a diagnostic asset audit, not a release constructor.  In particular,
finding a mesh and annotations does not establish query-frame visibility,
projection purity, field coverage, or exclusion compliance.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from radio_gs.benchmarks.scannet_pfir.build_benchmark import (
    find_scene_annotations,
)
from radio_gs.benchmarks.scannet_pfir.protocol import load_mesh_instances
from radio_gs.benchmarks.scannet_uqis.construction import (
    REFERIT3D_VIEW_DEPENDENCE_RULE,
    load_reference_rows,
    select_view_independent_expression,
)
from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION,
    FROZEN_PROTOCOL_CONFIG,
    PREREGISTERED_TEST_SCENES,
    PREREGISTERED_REPLACEMENTS,
    sha256_file,
)


STRUCTURAL_RAW_LABELS = frozenset({"wall", "floor", "ceiling"})


def _find_sens(scene_id: str, roots: Iterable[str | Path]) -> Path:
    searched: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root)
        candidates = (
            root / "scans" / scene_id / f"{scene_id}.sens",
            root / scene_id / f"{scene_id}.sens",
            root / f"{scene_id}.sens",
        )
        searched.extend(candidates)
        match = next((path for path in candidates if path.is_file()), None)
        if match is not None:
            return match
    raise FileNotFoundError(
        f"{scene_id}: .sens not found; searched " + ", ".join(map(str, searched))
    )


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _scene_record(
    scene_id: str,
    *,
    sens_roots: tuple[Path, ...],
    annotation_roots: tuple[Path, ...],
    reference_rows_by_target: Mapping[tuple[str, int], list[Mapping[str, Any]]] | None,
) -> dict[str, Any]:
    try:
        sens_path = _find_sens(scene_id, sens_roots)
        mesh_path, aggregation_path, segmentation_path = find_scene_annotations(
            scene_id, annotation_roots
        )
        xyz, _instance_ids, metadata = load_mesh_instances(
            mesh_path, aggregation_path, segmentation_path
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        return {
            "scene_id": scene_id,
            "status": "missing_or_invalid_official_asset",
            "error": str(error),
        }

    eligible_geometry = {
        int(instance_id): row
        for instance_id, row in metadata.items()
        if str(row["label"]).strip().lower() not in STRUCTURAL_RAW_LABELS
        and int(row["num_vertices"]) >= FROZEN_PROTOCOL_CONFIG.min_mesh_vertices
    }
    label_counts = Counter(
        str(row["label"]).strip().lower() for row in eligible_geometry.values()
    )
    same_class_geometry = sorted(
        instance_id
        for instance_id, row in eligible_geometry.items()
        if label_counts[str(row["label"]).strip().lower()] >= 2
    )

    expression_instances: list[int] = []
    if reference_rows_by_target is not None:
        for instance_id in eligible_geometry:
            try:
                select_view_independent_expression(
                    reference_rows_by_target.get((scene_id, instance_id - 1), ()),
                    scene_id=scene_id,
                    official_instance_id=instance_id,
                )
            except ValueError:
                continue
            expression_instances.append(instance_id)
    expression_labels = Counter(
        str(eligible_geometry[instance_id]["label"]).strip().lower()
        for instance_id in expression_instances
    )
    expression_same_class = sum(
        expression_labels[
            str(eligible_geometry[instance_id]["label"]).strip().lower()
        ]
        >= 2
        for instance_id in expression_instances
    )
    coarse_reference_ready = bool(
        reference_rows_by_target is not None
        and len(expression_instances) >= FROZEN_PROTOCOL_CONFIG.min_targets_per_scene
        and len(expression_labels)
        >= FROZEN_PROTOCOL_CONFIG.min_semantic_categories_per_scene
        and expression_same_class
        >= FROZEN_PROTOCOL_CONFIG.min_same_class_targets_per_scene
    )

    return {
        "scene_id": scene_id,
        "status": "official_geometry_present",
        "sens": _file_record(sens_path),
        "mesh": _file_record(mesh_path),
        "aggregation": _file_record(aggregation_path),
        "segmentation": _file_record(segmentation_path),
        "mesh_vertex_count": int(xyz.shape[0]),
        "official_instance_count": int(len(metadata)),
        "non_structural_instances_at_least_500_vertices": int(
            len(eligible_geometry)
        ),
        "raw_semantic_categories_among_those_instances": int(len(label_counts)),
        "instances_with_same_raw_class_distractor": int(len(same_class_geometry)),
        "view_independent_reference_expression_instances": (
            None if reference_rows_by_target is None else int(len(expression_instances))
        ),
        "reference_expression_raw_semantic_categories": (
            None if reference_rows_by_target is None else int(len(expression_labels))
        ),
        "reference_expression_same_raw_class_instances": (
            None if reference_rows_by_target is None else int(expression_same_class)
        ),
        "coarse_reference_geometry_ready": coarse_reference_ready,
        "formal_target_eligibility_computed": False,
    }


def audit_candidate_assets(
    *,
    sens_roots: Iterable[str | Path],
    annotation_roots: Iterable[str | Path],
    reference_annotations: str | Path | None = None,
    referit3d_rule_source: str | Path | None = None,
) -> dict[str, Any]:
    """Return a content-bound local inventory for all frozen test candidates."""

    sens = tuple(Path(value) for value in sens_roots)
    annotations = tuple(Path(value) for value in annotation_roots)
    if not sens or not annotations:
        raise ValueError("at least one sensor root and annotation root are required")
    reference_path = (
        None if reference_annotations is None else Path(reference_annotations).resolve()
    )
    rule_source_path = (
        None if referit3d_rule_source is None else Path(referit3d_rule_source).resolve()
    )
    if rule_source_path is not None and not rule_source_path.is_file():
        raise FileNotFoundError(rule_source_path)
    reference_rows = (
        None if reference_path is None else load_reference_rows(reference_path)
    )
    reference_index: dict[tuple[str, int], list[Mapping[str, Any]]] | None = None
    if reference_rows is not None:
        grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for row in reference_rows:
            try:
                grouped[(str(row.get("scan_id", "")), int(row.get("target_id")))].append(row)
            except (TypeError, ValueError):
                continue
        reference_index = dict(grouped)
    scene_order = (*PREREGISTERED_TEST_SCENES, *PREREGISTERED_REPLACEMENTS)
    scenes = [
        _scene_record(
            scene_id,
            sens_roots=sens,
            annotation_roots=annotations,
            reference_rows_by_target=reference_index,
        )
        for scene_id in scene_order
    ]
    complete_official_assets = all(
        scene["status"] == "official_geometry_present" for scene in scenes
    )
    blockers = []
    if not complete_official_assets:
        blockers.append("one_or_more_preregistered_scenes_lack_official_local_assets")
    if reference_path is None:
        blockers.append("nr3d_or_equivalent_reference_annotations_not_provided")
    elif not all(
        scene["coarse_reference_geometry_ready"]
        for scene in scenes[: len(PREREGISTERED_TEST_SCENES)]
    ):
        blockers.append("one_or_more_primary_scenes_fail_coarse_nr3d_geometry_constraints")
    blockers.extend(
        (
            "visibility_projection_purity_and_surface_coverage_not_derived",
            "query_frame_union_exclusion_and_mapping_observation_receipts_not_built",
            "formal_release_constructor_intentionally_not_implemented",
        )
    )
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "audit_kind": "candidate_asset_inventory",
        "candidate_scene_order": list(PREREGISTERED_TEST_SCENES),
        "replacement_scene_order": list(PREREGISTERED_REPLACEMENTS),
        "reference_annotations": (
            None if reference_path is None else _file_record(reference_path)
        ),
        "reference_derivation": (
            None
            if reference_path is None
            else {
                "view_dependence_rule": REFERIT3D_VIEW_DEPENDENCE_RULE,
                "correct_guess_required": True,
                "mentions_target_class_required": True,
                "minimum_tokens": 2,
                "maximum_tokens": 64,
                "rule_source": (
                    None if rule_source_path is None else _file_record(rule_source_path)
                ),
            }
        ),
        "scene_count": len(scenes),
        "complete_official_scene_assets": complete_official_assets,
        "formal_benchmark_eligible": False,
        "blockers": blockers,
        "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens-root", action="append", required=True)
    parser.add_argument("--annotation-root", action="append", required=True)
    parser.add_argument("--reference-annotations")
    parser.add_argument("--referit3d-rule-source")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_candidate_assets(
        sens_roots=args.sens_root,
        annotation_roots=args.annotation_root,
        reference_annotations=args.reference_annotations,
        referit3d_rule_source=args.referit3d_rule_source,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
