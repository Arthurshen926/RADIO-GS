#!/usr/bin/env python3
"""Select a class-complete, benchmark-blind ScanNet source cohort.

The cohort is selected exclusively from official ScanNet annotations and
already materialized, query-blind canonical fields.  No benchmark prediction,
mask, or metric is read.  This closes the provenance gap between the official
semantic authority and a categorical posterior fitted on independent scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from plyfile import PlyData

from radio_gs.data.scannet_source_region_semantics import (
    load_scannet_raw_to_nyu40,
    official_vertex_nyu40_labels,
    sha256_file,
)
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)


SCHEMA = "radio_gs.scannet_class_complete_source_cohort.v1"


def select_class_complete_cohort(
    scene_classes: Mapping[str, Iterable[int]], required_classes: Iterable[int]
) -> list[str]:
    """Deterministic greedy set cover, with scene id as the stable tie-break."""

    required = frozenset(int(value) for value in required_classes)
    normalized = {
        str(scene_id): frozenset(int(value) for value in values) & required
        for scene_id, values in scene_classes.items()
    }
    uncovered = set(required)
    selected: list[str] = []
    while uncovered:
        ranked = sorted(
            (
                (-len(classes & uncovered), scene_id)
                for scene_id, classes in normalized.items()
                if scene_id not in selected
            )
        )
        if not ranked or -ranked[0][0] == 0:
            missing = sorted(uncovered)
            raise ValueError(f"source cohort cannot cover NYU40 classes {missing}")
        winner = ranked[0][1]
        selected.append(winner)
        uncovered.difference_update(normalized[winner])
    return selected


def class_coverage_counts(
    scene_classes: Mapping[str, Iterable[int]], required_classes: Iterable[int]
) -> dict[int, int]:
    """Count independent scene support for each required semantic class."""

    required = sorted(set(int(value) for value in required_classes))
    normalized = [set(int(value) for value in values) for values in scene_classes.values()]
    return {value: sum(value in classes for classes in normalized) for value in required}


def _asset_record(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=True)
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _scene_assets(
    scene_id: str, annotation_root: Path, reconstruction_root: Path
) -> dict[str, Path] | None:
    annotation = annotation_root / scene_id
    canonical = reconstruction_root / "canonical_fields" / scene_id
    render = reconstruction_root / "render_contracts"
    assets = {
        "mesh": annotation / f"{scene_id}_vh_clean_2.ply",
        "segmentation": annotation / f"{scene_id}_vh_clean_2.0.010000.segs.json",
        "aggregation": annotation / f"{scene_id}.aggregation.json",
        "render_config": render / f"{scene_id}.yaml",
        "geometry_checkpoint": render / f"{scene_id}.geometry_renderer.pth",
        "canonical_field": canonical / "canonical_mpr_v2.pt",
        "canonical_field_receipt": canonical / "canonical_mpr_v2.pt.json",
    }
    return assets if all(path.is_file() for path in assets.values()) else None


def audit_scene(
    scene_id: str,
    assets: Mapping[str, Path],
    raw_to_nyu40: Mapping[str, int],
    required_classes: Sequence[int],
    *,
    hash_assets: bool,
) -> dict[str, Any]:
    mesh = PlyData.read(str(assets["mesh"]))
    vertex_count = int(mesh["vertex"].count)
    segmentation = json.loads(assets["segmentation"].read_text())
    aggregation = json.loads(assets["aggregation"].read_text())
    labels, label_audit = official_vertex_nyu40_labels(
        scene_id=scene_id,
        vertex_count=vertex_count,
        segmentation=segmentation,
        aggregation=aggregation,
        raw_to_nyu40=raw_to_nyu40,
    )
    coverage = sorted(set(int(value) for value in labels) & set(required_classes))
    record = {
        "scene_id": scene_id,
        "covered_nyu40_ids": coverage,
        "covered_class_names": [NYU40_ID_TO_NAME[value] for value in coverage],
        "official_label_audit": label_audit,
        "assets": {},
    }
    for name, path in sorted(assets.items()):
        resolved = path.expanduser().resolve(strict=True)
        record["assets"][name] = (
            _asset_record(resolved)
            if hash_assets
            else {"path": str(resolved), "size_bytes": resolved.stat().st_size}
        )
    return record


def build_audit(
    *,
    annotation_root: Path,
    reconstruction_root: Path,
    label_tsv: Path,
    hash_assets: bool = True,
) -> dict[str, Any]:
    required = list(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"])
    raw_to_nyu40 = load_scannet_raw_to_nyu40(label_tsv)
    records = []
    for annotation in sorted(annotation_root.glob("scene????_??")):
        assets = _scene_assets(annotation.name, annotation_root, reconstruction_root)
        if assets is not None:
            records.append(
                audit_scene(
                    annotation.name,
                    assets,
                    raw_to_nyu40,
                    required,
                    hash_assets=hash_assets,
                )
            )
    if not records:
        raise ValueError("no annotation scene has a compatible canonical field")
    selected_ids = select_class_complete_cohort(
        {record["scene_id"]: record["covered_nyu40_ids"] for record in records},
        required,
    )
    by_id = {record["scene_id"]: record for record in records}
    selected = [by_id[scene_id] for scene_id in selected_ids]
    union = sorted(
        set().union(*(set(record["covered_nyu40_ids"]) for record in selected))
    )
    coverage_counts = class_coverage_counts(
        {record["scene_id"]: record["covered_nyu40_ids"] for record in records},
        required,
    )
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "selection": "deterministic_greedy_set_cover_max_new_classes_then_scene_id",
        "semantic_authority": (
            "official_mesh_to_official_segments_to_official_aggregation_to_nyu40"
        ),
        "field_authority": "query_blind_canonical_mpr_v2",
        "required_nyu40_ids": required,
        "required_class_names": [NYU40_ID_TO_NAME[value] for value in required],
        "compatible_scene_count": len(records),
        "compatible_scene_coverage": [
            {
                "scene_id": record["scene_id"],
                "covered_nyu40_ids": record["covered_nyu40_ids"],
                "covered_class_names": record["covered_class_names"],
            }
            for record in records
        ],
        "selected_scene_ids": selected_ids,
        "selected_coverage_nyu40_ids": union,
        "compatible_scene_coverage_count_by_nyu40_id": {
            str(key): value for key, value in coverage_counts.items()
        },
        "scene_loso_identifiable": all(value >= 2 for value in coverage_counts.values()),
        "scene_loso_missing_redundancy_nyu40_ids": [
            key for key, value in coverage_counts.items() if value < 2
        ],
        "selected_records": selected,
        "label_tsv": _asset_record(label_tsv) if hash_assets else {"path": str(label_tsv.resolve())},
        "access_contract": {
            "benchmark_masks_opened": False,
            "benchmark_predictions_opened": False,
            "benchmark_metrics_opened": False,
            "paper8_scene_labels_opened": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--label-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-asset-hashes", action="store_true")
    args = parser.parse_args()
    payload = build_audit(
        annotation_root=args.annotation_root.expanduser().resolve(strict=True),
        reconstruction_root=args.reconstruction_root.expanduser().resolve(strict=True),
        label_tsv=args.label_tsv.expanduser().resolve(strict=True),
        hash_assets=not args.skip_asset_hashes,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "compatible_scene_count": payload["compatible_scene_count"],
        "selected_scene_ids": payload["selected_scene_ids"],
        "selected_coverage_nyu40_ids": payload["selected_coverage_nyu40_ids"],
    }, indent=2))


if __name__ == "__main__":
    main()
