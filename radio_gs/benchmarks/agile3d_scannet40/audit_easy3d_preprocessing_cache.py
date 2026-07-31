#!/usr/bin/env python3
"""Audit the immutable official-worker Easy3D preprocessing cache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_easy3d import (
    EASY3D_AUDITED_COMMIT,
    OFFICIAL_OBJECT_CLASSES_SHA256,
    OFFICIAL_OBJECT_IDS_SHA256,
    _git_commit,
    _sha256,
    aggregate_quantization_diagnostics,
    load_official_worker_cached_scene,
    validate_official_worker_cache_manifest,
)
from .protocol import Agile3DObject, load_official_object_list


AUDIT_SCHEMA = "official-easy3d-worker-cache-audit-v1"


def audit_cache(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    easy3d_repo = Path(args.easy3d_repo).resolve()
    cache_root = Path(args.cache_root).resolve()
    output_path = Path(args.output).resolve()
    easy3d_commit = _git_commit(easy3d_repo)
    object_ids_sha = _sha256(data_root / "single" / "object_ids.npy")
    object_classes_sha = _sha256(
        data_root / "single" / "object_classes.txt"
    )
    if easy3d_commit != EASY3D_AUDITED_COMMIT:
        raise ValueError("cache audit requires the audited Easy3D commit")
    if object_ids_sha != OFFICIAL_OBJECT_IDS_SHA256:
        raise ValueError("cache audit requires the canonical object IDs")
    if object_classes_sha != OFFICIAL_OBJECT_CLASSES_SHA256:
        raise ValueError("cache audit requires the canonical object classes")

    objects = load_official_object_list(data_root)
    by_scene: dict[str, list[Agile3DObject]] = defaultdict(list)
    for item in objects:
        by_scene[item.scene_id].append(item)
    manifest, manifest_sha, manifest_rows = (
        validate_official_worker_cache_manifest(
            cache_root,
            data_root=data_root,
            easy3d_commit=easy3d_commit,
            object_ids_sha256=object_ids_sha,
            required_scene_ids=by_scene,
            formal=True,
        )
    )

    scene_rows: list[dict[str, Any]] = []
    quantization_rows = []
    raw_missing: list[str] = []
    voxel_missing: list[str] = []
    class_counts: Counter[str] = Counter()
    for index, scene_id in enumerate(sorted(by_scene), start=1):
        scene, metadata = load_official_worker_cached_scene(
            cache_root,
            scene_id,
            easy3d_commit=easy3d_commit,
            object_ids_sha256=object_ids_sha,
        )
        manifest_row = manifest_rows[scene_id]
        for hash_key in ("npz_sha256", "array_content_sha256"):
            if metadata.get(hash_key) != manifest_row.get(hash_key):
                raise ValueError(
                    f"{scene_id}: metadata/manifest {hash_key} mismatch"
                )
        point_ids = set(np.unique(scene.point_labels).tolist())
        voxel_ids = set(np.unique(scene.voxel_labels).tolist())
        scene_raw_missing = [
            item.key
            for item in by_scene[scene_id]
            if item.object_id not in point_ids
        ]
        scene_voxel_missing = [
            item.key
            for item in by_scene[scene_id]
            if item.object_id not in voxel_ids
        ]
        raw_missing.extend(scene_raw_missing)
        voxel_missing.extend(scene_voxel_missing)
        classes = Counter(item.semantic_class for item in by_scene[scene_id])
        class_counts.update(classes)
        diagnostics = dict(scene.quantization_diagnostics)
        quantization_rows.append(diagnostics)
        scene_rows.append(
            {
                "scene_id": scene_id,
                "point_count": int(len(scene.point_labels)),
                "voxel_count": int(len(scene.coordinates)),
                "object_count": int(len(by_scene[scene_id])),
                "unique_class_count": int(len(classes)),
                "class_counts": dict(sorted(classes.items())),
                "raw_missing_object_count": len(scene_raw_missing),
                "voxel_missing_object_count": len(scene_voxel_missing),
                "npz_sha256": metadata["npz_sha256"],
                "array_content_sha256": metadata[
                    "array_content_sha256"
                ],
            }
        )
        if index % 25 == 0 or index == len(by_scene):
            print(
                f"[{index}/{len(by_scene)}] verified {scene_id}",
                flush=True,
            )

    report = {
        "audit_schema": AUDIT_SCHEMA,
        "status": (
            "formal_complete"
            if not raw_missing and not voxel_missing
            else "formal_with_missing_objects"
        ),
        "provenance": {
            "data_root": str(data_root),
            "easy3d_repository_root": str(easy3d_repo),
            "easy3d_commit": easy3d_commit,
            "cache_root": str(cache_root),
            "cache_schema": manifest["cache_schema"],
            "cache_manifest_sha256": manifest_sha,
            "cache_selection": manifest["selection"],
            "object_ids_sha256": object_ids_sha,
            "object_classes_sha256": object_classes_sha,
        },
        "scene_count": len(scene_rows),
        "object_count": len(objects),
        "class_count": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "raw_missing_objects": raw_missing,
        "voxel_missing_objects": voxel_missing,
        "quantization_diagnostics": aggregate_quantization_diagnostics(
            quantization_rows
        ),
        "scenes": scene_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != report:
            raise ValueError(
                "existing cache audit differs; use a new output path"
            )
    else:
        output_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--easy3d-repo", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_cache(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "scene_count": report["scene_count"],
                "object_count": report["object_count"],
                "raw_missing_object_count": len(
                    report["raw_missing_objects"]
                ),
                "voxel_missing_object_count": len(
                    report["voxel_missing_objects"]
                ),
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
