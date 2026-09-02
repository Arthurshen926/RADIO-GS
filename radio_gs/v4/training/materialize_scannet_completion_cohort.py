"""Materialize a query-free ScanNet-PFIR cohort for the v4 completion oracle.

The PFIR field release keeps query-independent RGB/pose observations separate
from ScanNet instance annotations.  This utility joins those immutable inputs
without reading a held-out RGB image: eight coverage-ordered cameras are bound,
four source RGBs are deterministically resized for the frozen C-RADIO raster,
and the remaining four cameras are geometry-only held-out authorities.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from radio_gs.v4.completion.scannet import _uniform_indices
from radio_gs.v4.contracts.geometry_receipt import sha256_file


SCHEMA = "radio_gs.surface_object_memory_v4.scannet_pfir_completion_stage.v3"
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
COHORT_SELECTION_SALT = "radio_gs_v4_completion_cohort16_v1"
VALIDATION_SELECTION_SALT = "radio_gs_v4_completion_val4_v1"
FROZEN_RADIO_GRID_HW = (60, 81)


def _json_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scene_family(scene_id: str) -> str:
    parts = str(scene_id).split("_")
    if len(parts) != 2 or not parts[0].startswith("scene"):
        raise ValueError(f"invalid ScanNet scene identity {scene_id!r}")
    return parts[0]


def _radio_grid_hw(source_size_wh: tuple[int, int]) -> tuple[int, int]:
    width, height = map(int, source_size_wh)
    # This mirrors extract_radio_features._compute_scaled_radio_resolution at
    # resolution_scale=1 without importing the heavyweight extractor module.
    radio_height = max(16, round(height / 16) * 16)
    radio_width = max(16, round(width / 16) * 16)
    return radio_height // 16, radio_width // 16


def _validate_pose(pose: np.ndarray, *, label: str) -> None:
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{label} must be one finite 4x4 matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{label} has an invalid homogeneous bottom row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{label} does not contain an orthonormal rotation")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError(f"{label} does not contain a proper rotation")


def _field_scene(field_roots: list[Path], scene_id: str) -> Path:
    matches = [
        (root / scene_id).resolve(strict=True)
        for root in field_roots
        if (root / scene_id).is_dir()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"scene {scene_id!r} must occur in exactly one PFIR field shard"
        )
    return matches[0]


def _rgb_path(directory: Path, frame_id: int) -> Path:
    stem = f"{int(frame_id):06d}"
    matches = [directory / f"{stem}{suffix}" for suffix in (".jpg", ".png", ".jpeg")]
    available = [path.resolve(strict=True) for path in matches if path.is_file()]
    if len(available) != 1:
        raise FileNotFoundError(f"frame {stem} must have exactly one PFIR RGB file")
    return available[0]


def _pose_path(directory: Path, frame_id: int) -> Path:
    return (directory / f"{int(frame_id):06d}.txt").resolve(strict=True)


def _annotation_paths(annotation_root: Path, scene_id: str) -> dict[str, Path]:
    directory = (annotation_root / scene_id).resolve(strict=True)
    meshes = sorted(directory.glob("*_vh_clean_2.ply"))
    segments = sorted(directory.glob("*.segs.json"))
    aggregations = sorted(directory.glob("*.aggregation.json"))
    if len(meshes) != 1 or len(segments) != 1 or len(aggregations) != 1:
        raise FileNotFoundError(
            f"scene {scene_id!r} lacks one exact mesh/segmentation/aggregation triple"
        )
    return {
        "directory": directory,
        "mesh": meshes[0].resolve(strict=True),
        "segmentation": segments[0].resolve(strict=True),
        "aggregation": aggregations[0].resolve(strict=True),
    }


def _safe_link(source: Path, destination: Path, *, directory: bool = False) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace staged path {destination}")
    destination.symlink_to(source.resolve(strict=True), target_is_directory=directory)


def _hardlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace staged path {destination}")
    os.link(source.resolve(strict=True), destination)


def _resize_source_rgb(source: Path, destination: Path, size: tuple[int, int]) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace staged path {destination}")
    with Image.open(source) as image:
        resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        resized.save(destination, format="JPEG", quality=95, subsampling=0)


def materialize_scene(
    *,
    scene_id: str,
    field_roots: list[Path],
    annotation_root: Path,
    output_root: Path,
    total_view_count: int = 8,
    observation_view_count: int = 4,
) -> dict[str, Any]:
    if total_view_count < 2 or not 0 < observation_view_count < total_view_count:
        raise ValueError("completion staging needs disjoint source and held-out views")
    field = _field_scene(field_roots, scene_id)
    annotations = _annotation_paths(annotation_root, scene_id)
    contract_path = (field / "field_source_contract.json").resolve(strict=True)
    contract = json.loads(contract_path.read_text())
    if (
        contract.get("scene_id") != scene_id
        or contract.get("frame_selection_policy") != "depth_voxel_coverage"
        or contract.get("uses_instances_or_semantic_labels") is not False
        or contract.get("contains_instance_or_label_directories") is not False
    ):
        raise ValueError("PFIR field source is not the frozen query-free coverage contract")
    source_size = tuple(map(int, contract.get("source_color_size", ())))
    if len(source_size) != 2 or min(source_size) <= 0:
        raise ValueError("PFIR contract lacks the original C-RADIO source raster size")
    radio_grid_hw = _radio_grid_hw(source_size)
    if radio_grid_hw != FROZEN_RADIO_GRID_HW:
        raise ValueError(
            f"PFIR source raster {source_size} yields RADIO grid {radio_grid_hw}, "
            f"not the frozen {FROZEN_RADIO_GRID_HW} completion grid"
        )
    intrinsic_path = (field / "intrinsic" / "intrinsic_color.txt").resolve(strict=True)
    intrinsic = np.loadtxt(intrinsic_path, dtype=np.float64)
    if (
        intrinsic.shape != (4, 4)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
    ):
        raise ValueError("PFIR color intrinsic must be one finite 4x4 matrix")

    selected: list[dict[str, Any]] = []
    seen = set()
    for raw_id in contract.get("selection_order_frame_indices", []):
        frame_id = int(raw_id)
        if frame_id in seen:
            continue
        seen.add(frame_id)
        try:
            pose_path = _pose_path(field / "pose", frame_id)
        except FileNotFoundError:
            continue
        pose = np.loadtxt(pose_path, dtype=np.float64)
        _validate_pose(pose, label=f"pose {pose_path}")
        selected.append({"frame_id": frame_id, "pose_path": pose_path, "pose": pose})
        if len(selected) == total_view_count:
            break
    if len(selected) != total_view_count:
        raise RuntimeError(f"scene {scene_id!r} has too few valid coverage-ordered frames")

    observation_positions = _uniform_indices(total_view_count, observation_view_count)
    observation_position_set = set(observation_positions)
    heldout_positions = [
        index for index in range(total_view_count) if index not in observation_position_set
    ]
    scene_root = output_root.resolve() / scene_id
    if scene_root.exists() or scene_root.is_symlink():
        raise FileExistsError(f"refusing to replace existing completion stage {scene_root}")
    color_root = scene_root / "color"
    source_root = scene_root / "source_rgb"
    color_root.mkdir(parents=True)
    source_root.mkdir()

    frame_records = []
    transform_frames = []
    for position, row in enumerate(selected):
        stem = f"{row['frame_id']:06d}"
        staged_rgb = color_root / f"{stem}.jpg"
        is_source = position in observation_position_set
        if is_source:
            source_rgb = _rgb_path(field / "color", row["frame_id"])
            _resize_source_rgb(source_rgb, staged_rgb, source_size)
            _hardlink(staged_rgb, source_root / staged_rgb.name)
        transform = row["pose"] @ GL_TO_CV
        transform_frames.append(
            {"file_path": f"color/{stem}", "transform_matrix": transform.tolist()}
        )
        frame_record = {
            "position": position,
            "frame_id": int(row["frame_id"]),
            "role": "source_observation" if is_source else "heldout_geometry_only",
            "pose_path": str(row["pose_path"]),
            "pose_sha256": sha256_file(row["pose_path"]),
            "staged_rgb_present": is_source,
            "rgb_content_opened_or_decoded": is_source,
            "rgb_content_hashed": is_source,
        }
        if is_source:
            frame_record.update({
                "source_rgb_path": str(source_rgb),
                "source_rgb_sha256": sha256_file(source_rgb),
                "staged_rgb_path": str(staged_rgb.absolute()),
                "staged_rgb_resized": True,
                "staged_rgb_sha256": sha256_file(staged_rgb),
            })
        frame_records.append(frame_record)

    transforms = {
        "fl_x": float(intrinsic[0, 0]),
        "fl_y": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "w": int(source_size[0]),
        "h": int(source_size[1]),
        "frames": transform_frames,
    }
    transforms_path = scene_root / "transforms.json"
    transforms_path.write_text(
        json.dumps(transforms, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _safe_link(annotations["mesh"], scene_root / "points3d.ply")
    _safe_link(annotations["directory"], scene_root / "instance_annotations", directory=True)

    receipt = {
        "schema": SCHEMA,
        "scene_id": scene_id,
        "scene_family": _scene_family(scene_id),
        "field_scene_root": str(field),
        "field_shard_root": str(field.parent),
        "selection_policy": "first_valid_pfir_depth_voxel_coverage_order_v1",
        "coordinate_conversion": "raw_scannet_opencv_c2w_times_diag_1_minus1_minus1_1_to_nerf_json",
        "source_rgb_resize": {
            "method": "PIL_bilinear_RGB_JPEG_quality95_subsampling0",
            "size_wh": list(source_size),
            "frozen_radio_grid_hw": list(radio_grid_hw),
            "applies_only_to_source_observations": True,
            "heldout_rgb_path_resolved": False,
            "heldout_rgb_materialized_in_scene_root": False,
            "heldout_rgb_decoded_or_opened": False,
            "heldout_rgb_content_hashed": False,
        },
        "total_view_count": total_view_count,
        "observation_view_count": observation_view_count,
        "observation_positions": observation_positions,
        "heldout_positions": heldout_positions,
        "field_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "intrinsic": {"path": str(intrinsic_path), "sha256": sha256_file(intrinsic_path)},
        "annotation_inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in annotations.items()
            if key != "directory"
        },
        "frames": frame_records,
        "transforms": {"path": str(transforms_path.resolve()), "sha256": sha256_file(transforms_path)},
        "source_rgb_directory": str(source_root.resolve()),
    }
    receipt["receipt_sha256"] = _json_sha256(receipt)
    receipt_path = scene_root / "stage_manifest.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.scene_limit < 0:
        raise ValueError("scene limit cannot be negative")
    if args.validation_per_field_shard < 0:
        raise ValueError("validation scenes per field shard cannot be negative")
    field_roots = sorted(
        (Path(value).resolve(strict=True) for value in args.field_root), key=str
    )
    annotation_root = Path(args.annotation_root).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    scene_ids = list(dict.fromkeys(args.scene_id))
    if len({_scene_family(scene_id) for scene_id in scene_ids}) != len(scene_ids):
        raise ValueError("explicit completion cohort must be physical-family disjoint")
    candidate_scene_ids = list(scene_ids)
    selection_policy = "explicit_scene_ids"
    if not scene_ids:
        annotated = {path.name for path in annotation_root.glob("scene*") if path.is_dir()}
        import hashlib

        candidates_by_shard: list[list[str]] = []
        for root in field_roots:
            eligible = []
            for path in root.glob("scene*"):
                if not path.is_dir() or path.name not in annotated:
                    continue
                contract_path = path / "field_source_contract.json"
                try:
                    contract = json.loads(contract_path.read_text())
                    source_size = tuple(map(int, contract.get("source_color_size", ())))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if len(source_size) != 2 or _radio_grid_hw(source_size) != FROZEN_RADIO_GRID_HW:
                    continue
                eligible.append(path.name)
            eligible.sort(
                key=lambda value: hashlib.sha256(
                    f"{COHORT_SELECTION_SALT}\0{value}".encode("utf-8")
                ).hexdigest()
            )
            candidates_by_shard.append(eligible)
        candidate_scene_ids = sorted(
            scene_id for rows in candidates_by_shard for scene_id in rows
        )
        if args.scene_limit > 0 and args.scene_limit % len(field_roots):
            raise ValueError("balanced scene limit must divide evenly across field shards")
        per_shard_limit = (
            args.scene_limit // len(field_roots) if args.scene_limit > 0 else None
        )
        used_families: set[str] = set()
        scene_ids = []
        for rows in candidates_by_shard:
            selected_from_shard = []
            for scene_id in rows:
                family = _scene_family(scene_id)
                if family in used_families:
                    continue
                used_families.add(family)
                selected_from_shard.append(scene_id)
                if per_shard_limit is not None and len(selected_from_shard) == per_shard_limit:
                    break
            if per_shard_limit is not None and len(selected_from_shard) != per_shard_limit:
                raise RuntimeError("a PFIR field shard has too few eligible physical families")
            scene_ids.extend(selected_from_shard)
        selection_policy = (
            "eligible_frozen_radio_grid_sha256_rank_physical_family_disjoint_"
            + (
                "balanced_per_field_shard_v1"
                if per_shard_limit is not None
                else "all_eligible_per_field_shard_v1"
            )
        )
    elif args.scene_limit:
        raise ValueError("--scene-limit applies only to automatic cohort selection")
    if not scene_ids:
        raise FileNotFoundError("no PFIR field/annotation scene intersection was found")
    records = [
        materialize_scene(
            scene_id=scene_id,
            field_roots=field_roots,
            annotation_root=annotation_root,
            output_root=output_root,
            total_view_count=args.total_view_count,
            observation_view_count=args.observation_view_count,
        )
        for scene_id in scene_ids
    ]
    records_by_scene = {row["scene_id"]: row for row in records}
    validation_scene_ids: list[str] = []
    if args.validation_per_field_shard:
        import hashlib

        by_shard: dict[str, list[str]] = {}
        for row in records:
            by_shard.setdefault(row["field_shard_root"], []).append(row["scene_id"])
        for shard in sorted(by_shard):
            ranked = sorted(
                by_shard[shard],
                key=lambda value: hashlib.sha256(
                    f"{VALIDATION_SELECTION_SALT}\0{value}".encode("utf-8")
                ).hexdigest(),
            )
            if len(ranked) <= args.validation_per_field_shard:
                raise ValueError("validation selection would leave a field shard without training scenes")
            validation_scene_ids.extend(ranked[: args.validation_per_field_shard])
    validation_set = set(validation_scene_ids)
    training_scene_ids = [scene_id for scene_id in scene_ids if scene_id not in validation_set]
    shard_counts: dict[str, dict[str, int]] = {}
    for scene_id in scene_ids:
        shard = records_by_scene[scene_id]["field_shard_root"]
        counts = shard_counts.setdefault(shard, {"total": 0, "train": 0, "validation": 0})
        counts["total"] += 1
        counts["validation" if scene_id in validation_set else "train"] += 1
    manifest = {
        "schema": "radio_gs.surface_object_memory_v4.scannet_pfir_completion_cohort.v3",
        "scene_ids": scene_ids,
        "scene_count": len(records),
        "selection": {
            "policy": selection_policy,
            "salt": COHORT_SELECTION_SALT if not args.scene_id else None,
            "candidate_scene_count": len(candidate_scene_ids),
            "candidate_scene_ids_sha256": _json_sha256(sorted(candidate_scene_ids)),
            "scene_limit": int(args.scene_limit),
        },
        "split": {
            "training_scene_ids": training_scene_ids,
            "validation_scene_ids": validation_scene_ids,
            "validation_selection_salt": (
                VALIDATION_SELECTION_SALT if args.validation_per_field_shard else None
            ),
            "validation_per_field_shard": int(args.validation_per_field_shard),
            "physical_family_disjoint": True,
            "field_shard_counts": shard_counts,
        },
        "records": [
            {
                "scene_id": row["scene_id"],
                "stage_manifest": str(output_root / row["scene_id"] / "stage_manifest.json"),
                "receipt_sha256": row["receipt_sha256"],
            }
            for row in records
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "cohort_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-root", action="append", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--validation-per-field-shard", type=int, default=0)
    parser.add_argument("--total-view-count", type=int, default=8)
    parser.add_argument("--observation-view-count", type=int, default=4)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"scene_count": report["scene_count"], "scene_ids": report["scene_ids"]}, indent=2))


if __name__ == "__main__":
    main()
