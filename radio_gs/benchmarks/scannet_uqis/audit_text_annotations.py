#!/usr/bin/env python3
"""Content-bind and audit Nr3D target coverage before visual construction."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .scannet_assets import find_scene_annotations, load_mesh_instances

from .construction import (
    REFERIT3D_VIEW_DEPENDENCE_RULE,
    load_reference_rows,
    select_view_independent_expression,
)
from .protocol import (
    BENCHMARK_VERSION,
    FROZEN_PROTOCOL_CONFIG,
    PREREGISTERED_REPLACEMENTS,
    PREREGISTERED_TEST_SCENES,
    canonical_json_sha256,
    sha256_file,
)


STRUCTURAL_LABELS = frozenset({"wall", "floor", "ceiling"})


def _binding(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def audit_text_annotations(
    *,
    annotation_roots: Iterable[str | Path],
    nr3d_path: str | Path,
    referit3d_rule_source: str | Path,
    scene_ids: Iterable[str] = (*PREREGISTERED_TEST_SCENES, *PREREGISTERED_REPLACEMENTS),
) -> dict[str, Any]:
    """Audit only model-independent language/geometry eligibility facts."""

    ordered_scenes = tuple(map(str, scene_ids))
    if not ordered_scenes or len(set(ordered_scenes)) != len(ordered_scenes):
        raise ValueError("scene_ids must be non-empty and unique")
    nr3d_binding = _binding(nr3d_path)
    rule_binding = _binding(referit3d_rule_source)
    rows = load_reference_rows(nr3d_path)
    index: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            index[(str(row.get("scan_id", "")), int(row.get("target_id")))].append(row)
        except (TypeError, ValueError):
            continue
    scenes: list[dict[str, Any]] = []
    for scene_id in ordered_scenes:
        try:
            mesh, aggregation, segmentation = find_scene_annotations(
                scene_id, annotation_roots
            )
            _xyz, _ids, metadata = load_mesh_instances(mesh, aggregation, segmentation)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            scenes.append(
                {"scene_id": scene_id, "status": "missing_geometry", "error": str(error)}
            )
            continue
        targets = []
        for instance_id, metadata_row in sorted(metadata.items()):
            label = str(metadata_row["label"]).strip().lower()
            if (
                label in STRUCTURAL_LABELS
                or int(metadata_row["num_vertices"])
                < FROZEN_PROTOCOL_CONFIG.min_mesh_vertices
            ):
                continue
            try:
                expression = select_view_independent_expression(
                    index.get((scene_id, instance_id - 1), ()),
                    scene_id=scene_id,
                    official_instance_id=instance_id,
                )
            except ValueError:
                continue
            targets.append(
                {
                    "official_instance_id": int(instance_id),
                    "raw_semantic_label": label,
                    "mesh_vertices": int(metadata_row["num_vertices"]),
                    "expression_annotation_id": expression["annotation_id"],
                    "expression_source": expression["source"],
                    "view_dependence_rule": expression["view_dependence_rule"],
                }
            )
        counts = Counter(row["raw_semantic_label"] for row in targets)
        same_class = sum(counts[row["raw_semantic_label"]] >= 2 for row in targets)
        reasons = []
        if len(targets) < FROZEN_PROTOCOL_CONFIG.min_targets_per_scene:
            reasons.append("fewer_than_six_nr3d_geometry_targets")
        if len(counts) < FROZEN_PROTOCOL_CONFIG.min_semantic_categories_per_scene:
            reasons.append("fewer_than_four_raw_semantic_categories")
        if same_class < FROZEN_PROTOCOL_CONFIG.min_same_class_targets_per_scene:
            reasons.append("fewer_than_three_same_class_targets")
        scenes.append(
            {
                "scene_id": scene_id,
                "status": "coarse_text_geometry_pass" if not reasons else "coarse_text_geometry_fail",
                "failure_reasons": reasons,
                "eligible_target_count": len(targets),
                "raw_semantic_category_count": len(counts),
                "same_raw_class_target_count": same_class,
                "targets": targets,
                "geometry_bindings": {
                    "mesh": _binding(mesh),
                    "aggregation": _binding(aggregation),
                    "segmentation": _binding(segmentation),
                },
            }
        )
    body = {
        "schema_version": "scannet_uqis_text_annotation_audit_v1",
        "benchmark_version": BENCHMARK_VERSION,
        "status": "diagnostic_complete",
        "formal_benchmark_eligible": False,
        "nr3d": nr3d_binding,
        "nr3d_row_count": len(rows),
        "referit3d_view_dependence_rule": {
            "identity": REFERIT3D_VIEW_DEPENDENCE_RULE,
            "source": rule_binding,
        },
        "selection_flags": {
            "correct_guess": True,
            "mentions_target_class": True,
            "minimum_tokens": 2,
            "maximum_tokens": 64,
        },
        "scene_order": list(ordered_scenes),
        "scenes": scenes,
        "remaining_formal_gates": [
            "scanrefer_supplement_if_nr3d_scene_coverage_fails",
            "query_projection_and_purity",
            "three_frame_target_cover",
            "field_surface_coverage_after_union_exclusion",
        ],
    }
    return {**body, "audit_sha256": canonical_json_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", action="append", required=True)
    parser.add_argument("--nr3d", required=True)
    parser.add_argument("--referit3d-rule-source", required=True)
    parser.add_argument("--scene-id", action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_text_annotations(
        annotation_roots=args.annotation_root,
        nr3d_path=args.nr3d,
        referit3d_rule_source=args.referit3d_rule_source,
        scene_ids=(
            args.scene_id
            if args.scene_id
            else (*PREREGISTERED_TEST_SCENES, *PREREGISTERED_REPLACEMENTS)
        ),
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
