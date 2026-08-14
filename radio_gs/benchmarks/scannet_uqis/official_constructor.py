#!/usr/bin/env python3
"""Derive UQIS scene/target records from content-bound official assets."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence
import zlib

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .scannet_assets import (
    STRUCTURAL_LABELS,
    STRUCTURAL_NYU40_IDS,
    discover_frames,
    find_scene_annotations,
    load_matrix,
    load_mesh_instances,
    resolve_frame_observations,
)

from .construction import (
    align_depth_to_color_raster,
    build_image_query_crop,
    derive_paired_point_prompt,
    load_reference_rows,
    recompute_union_frame_exclusion,
    select_query_frame_cover,
    select_profiled_expression,
    select_view_independent_expression,
)
from .protocol import (
    BENCHMARK_VERSION,
    BENCHMARK_VERSION_V2_CANDIDATE,
    COHORT_DERIVATION_LEDGER,
    FROZEN_PROTOCOL_CONFIG,
    PREREGISTERED_TEST_SCENES,
    UQISProtocolConfig,
    canonical_json_sha256,
    sha256_file,
)


TEXT_PROFILE_LEGACY_V1 = "legacy_v1"
TEXT_PROFILE_UQIS_V2 = "uqis_v2_core_relational"
TEXT_PROFILES = (TEXT_PROFILE_LEGACY_V1, TEXT_PROFILE_UQIS_V2)
UQIS_V2_CONSTRUCTION_CANDIDATE = BENCHMARK_VERSION_V2_CANDIDATE


def read_sens_pose_inventory(path: str | Path) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Read only frame poses/order from a ScanNet `.sens`, skipping image payloads."""

    source = Path(path)
    poses: dict[str, np.ndarray] = {}
    with source.open("rb") as handle:
        version = struct.unpack("I", handle.read(4))[0]
        if version != 4:
            raise ValueError(f"{source}: unsupported .sens version {version}")
        name_length = struct.unpack("Q", handle.read(8))[0]
        handle.seek(name_length + 16 * 4 * 4, 1)
        handle.seek(4 + 4 + 4 * 4 + 4, 1)
        frame_count = struct.unpack("Q", handle.read(8))[0]
        for index in range(frame_count):
            pose_bytes = handle.read(16 * 4)
            if len(pose_bytes) != 16 * 4:
                raise ValueError(f"{source}: truncated frame pose inventory")
            pose = np.asarray(struct.unpack("f" * 16, pose_bytes), dtype=np.float64).reshape(4, 4)
            handle.seek(8 + 8, 1)
            color_bytes = struct.unpack("Q", handle.read(8))[0]
            depth_bytes = struct.unpack("Q", handle.read(8))[0]
            handle.seek(color_bytes + depth_bytes, 1)
            poses[f"{index:06d}"] = pose
    return tuple(poses), poses


def sample_sens_mapping_surfaces(
    path: str | Path, *, frame_stride: int, depth_stride: int
) -> dict[str, np.ndarray]:
    """Stream a frozen stride of RGB-independent depth surfaces from `.sens`."""

    source = Path(path)
    result: dict[str, np.ndarray] = {}
    with source.open("rb") as handle:
        version = struct.unpack("I", handle.read(4))[0]
        if version != 4:
            raise ValueError(f"{source}: unsupported .sens version {version}")
        name_length = struct.unpack("Q", handle.read(8))[0]
        handle.seek(name_length, 1)
        intrinsic_color = np.asarray(struct.unpack("f" * 16, handle.read(64))).reshape(4, 4)
        handle.seek(64, 1)  # extrinsic_color
        intrinsic_depth = np.asarray(struct.unpack("f" * 16, handle.read(64))).reshape(4, 4)
        handle.seek(64, 1)  # extrinsic_depth
        _color_compression = struct.unpack("i", handle.read(4))[0]
        depth_compression = struct.unpack("i", handle.read(4))[0]
        _color_width, _color_height, depth_width, depth_height = struct.unpack(
            "IIII", handle.read(16)
        )
        depth_shift = float(struct.unpack("f", handle.read(4))[0])
        frame_count = struct.unpack("Q", handle.read(8))[0]
        if depth_compression not in {0, 1} or depth_shift <= 0:
            raise ValueError(f"{source}: unsupported depth stream")
        kd = intrinsic_depth[:3, :3].astype(np.float64)
        for index in range(frame_count):
            pose = np.asarray(struct.unpack("f" * 16, handle.read(64)), dtype=np.float64).reshape(4, 4)
            handle.seek(16, 1)
            color_bytes = struct.unpack("Q", handle.read(8))[0]
            depth_bytes = struct.unpack("Q", handle.read(8))[0]
            handle.seek(color_bytes, 1)
            payload = handle.read(depth_bytes)
            if index % int(frame_stride) or not np.isfinite(pose).all():
                continue
            raw = zlib.decompress(payload) if depth_compression == 1 else payload
            depth = np.frombuffer(raw, dtype=np.uint16).reshape(depth_height, depth_width)
            yy, xx = np.mgrid[0:depth_height:int(depth_stride), 0:depth_width:int(depth_stride)]
            z = depth[yy, xx].astype(np.float64) / depth_shift
            valid = np.isfinite(z) & (z > 0.20) & (z < 8.0)
            x = (xx - kd[0, 2]) * z / kd[0, 0]
            y = (yy - kd[1, 2]) * z / kd[1, 1]
            camera = np.stack(
                [x[valid], y[valid], z[valid], np.ones(int(valid.sum()))], axis=1
            )
            result[f"{index:06d}"] = (camera @ pose.T)[:, :3].astype(np.float32)
    if not result:
        raise ValueError(f"{source}: no finite mapping surfaces sampled")
    return result


def _reference_index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
    result: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        try:
            key = (str(row.get("scan_id", "")), int(row.get("target_id")))
        except (TypeError, ValueError):
            continue
        result.setdefault(key, []).append(row)
    return result


def _file_binding(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    return {"path": str(source), "bytes": int(source.stat().st_size), "sha256": sha256_file(source)}


def _structurally_valid_subsets(
    candidates: Mapping[int, Mapping[str, Any]], config: UQISProtocolConfig
) -> Iterable[tuple[int, ...]]:
    ids = sorted(candidates)
    for count in range(min(config.max_targets_per_scene, len(ids)), config.min_targets_per_scene - 1, -1):
        for subset in itertools.combinations(ids, count):
            classes = [str(candidates[value]["raw_semantic_label"]) for value in subset]
            if len(set(classes)) < config.min_semantic_categories_per_scene:
                continue
            if max((classes.count(value) for value in set(classes)), default=0) > 2:
                continue
            same_class_targets = sum(
                any(
                    other != instance_id
                    and str(candidates[other]["raw_semantic_label"])
                    == str(candidates[instance_id]["raw_semantic_label"])
                    for other in ids
                )
                for instance_id in subset
            )
            if same_class_targets >= config.min_same_class_targets_per_scene:
                yield subset


def construct_scene(
    *,
    scene_id: str,
    frames_root: str | Path,
    sens_path: str | Path,
    annotation_roots: Sequence[str | Path],
    reference_index: Mapping[tuple[str, int], list[Mapping[str, Any]]],
    output_root: str | Path,
    config: UQISProtocolConfig = FROZEN_PROTOCOL_CONFIG,
    text_profile: str = TEXT_PROFILE_LEGACY_V1,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Construct one scene using no method predictions or evaluator metrics."""

    if text_profile not in TEXT_PROFILES:
        raise ValueError(f"unsupported text profile: {text_profile}")

    frames = discover_frames(Path(frames_root) / scene_id)
    frame_by_id = {frame.frame_id: frame for frame in frames}
    mesh, aggregation, segmentation = find_scene_annotations(scene_id, annotation_roots)
    mesh_xyz, mesh_ids, metadata = load_mesh_instances(mesh, aggregation, segmentation)
    depth_k = load_matrix(Path(frames_root) / scene_id / "intrinsics_depth.txt")[:3, :3]
    color_k = load_matrix(Path(frames_root) / scene_id / "intrinsics_color.txt")[:3, :3]
    tree = cKDTree(mesh_xyz)
    per_frame = {
        frame.frame_id: resolve_frame_observations(
            frame,
            mesh_xyz,
            mesh_ids,
            depth_k,
            color_k,
            depth_stride=2,
            maximum_mesh_distance_m=config.mesh_correspondence_distance_m,
            mesh_tree=tree,
        )
        for frame in frames
    }
    base_candidates: dict[int, dict[str, Any]] = {}
    for instance_id, meta in sorted(metadata.items()):
        raw_label = str(meta["label"]).strip().lower()
        if raw_label in STRUCTURAL_LABELS or int(meta["num_vertices"]) < config.min_mesh_vertices:
            continue
        try:
            selector = (
                select_profiled_expression
                if text_profile == TEXT_PROFILE_UQIS_V2
                else select_view_independent_expression
            )
            expression = selector(
                reference_index.get((scene_id, instance_id - 1), ()),
                scene_id=scene_id,
                official_instance_id=instance_id,
            )
        except ValueError:
            continue
        observations = {}
        for frame in frames:
            matches = [
                value
                for value in per_frame[frame.frame_id].values()
                if value.instance_id_3d == instance_id
                and value.resolution_purity >= config.min_projection_purity
                and value.nyu40_class_id not in STRUCTURAL_NYU40_IDS
                and (
                    value.pixel_count >= config.min_query_pixels
                    or value.image_fraction >= config.min_query_fraction
                )
            ]
            if matches:
                observations[frame.frame_id] = max(matches, key=lambda value: (value.pixel_count, value.resolution_purity, -value.encoded_2d_id))
        if observations:
            class_votes: dict[int, int] = {}
            for value in observations.values():
                class_votes[value.nyu40_class_id] = class_votes.get(value.nyu40_class_id, 0) + value.pixel_count
            class_id = max(class_votes, key=lambda value: (class_votes[value], -value))
            base_candidates[instance_id] = {
                "metadata": meta,
                "expression": expression,
                "nyu40_class_id": int(class_id),
                "raw_semantic_label": raw_label,
                "query_observations": observations,
            }
    full_frame_ids, full_poses = read_sens_pose_inventory(sens_path)
    mapping_surfaces = sample_sens_mapping_surfaces(
        sens_path,
        frame_stride=config.coverage_frame_stride,
        depth_stride=config.coverage_depth_stride,
    )
    mapping_visible_instances: dict[str, set[int]] = {}
    for frame_id, points in mapping_surfaces.items():
        distance, nearest = tree.query(points, k=1)
        valid = np.isfinite(distance) & (
            distance <= config.mesh_correspondence_distance_m
        )
        mapping_visible_instances[frame_id] = {
            int(value) for value in mesh_ids[nearest[valid]] if int(value) > 0
        }
    sparse_ids = tuple(frame.frame_id for frame in frames)
    if not set(sparse_ids).issubset(full_frame_ids):
        raise ValueError(f"{scene_id}: sparse projection frames escape the .sens order")

    chosen = chosen_query_frames = chosen_excluded = None
    final_candidates: dict[int, dict[str, Any]] = {}
    proposals_by_count: dict[
        int, dict[tuple[str, ...], tuple[tuple[Any, ...], tuple[int, ...], Any]]
    ] = {}
    structural_subset_count = 0
    for subset in _structurally_valid_subsets(base_candidates, config):
        structural_subset_count += 1
        try:
            cover = select_query_frame_cover(
                {
                    instance_id: {
                        frame_id: float(observation.pixel_count)
                        for frame_id, observation in base_candidates[instance_id]["query_observations"].items()
                    }
                    for instance_id in subset
                },
                maximum_frames=config.max_query_frames_per_scene,
            )
        except ValueError:
            continue
        query_frames = tuple(cover.frame_ids)
        utility = sum(cover.target_scores[value] for value in subset)
        key = (len(query_frames), -utility, query_frames, subset)
        proposals = proposals_by_count.setdefault(len(subset), {})
        previous = proposals.get(query_frames)
        if previous is None or key < previous[0]:
            proposals[query_frames] = (key, subset, cover)

    audited_camera_sets = 0
    failed_coverage_subsets = 0
    best_coverage_by_instance = {instance_id: 0.0 for instance_id in base_candidates}
    best_visibility_by_instance = {instance_id: 0 for instance_id in base_candidates}
    audit_cache: dict[tuple[str, ...], tuple[tuple[str, ...], dict[int, dict[str, Any]]]] = {}
    for count in sorted(proposals_by_count, reverse=True):
        selected_for_count = None
        for key, subset, cover in sorted(proposals_by_count[count].values()):
            query_frames = tuple(cover.frame_ids)
            if query_frames not in audit_cache:
                audited_camera_sets += 1
                excluded = recompute_union_frame_exclusion(
                    full_frame_ids,
                    full_poses,
                    query_frames,
                    temporal_radius=config.temporal_exclusion_radius,
                    translation_m=config.near_pose_translation_m,
                    rotation_deg=config.near_pose_rotation_deg,
                )
                field_ids = set(sparse_ids) - set(excluded)
                mapping_surface_points = [
                    points
                    for frame_id, points in mapping_surfaces.items()
                    if frame_id not in set(excluded)
                ]
                mapping_tree = cKDTree(np.concatenate(mapping_surface_points, axis=0))
                audited: dict[int, dict[str, Any]] = {}
                for instance_id, candidate in base_candidates.items():
                    distance, _nearest = mapping_tree.query(
                        mesh_xyz[mesh_ids == instance_id], k=1
                    )
                    coverage = float(
                        np.mean(distance <= config.coverage_distance_m)
                    )
                    audited[instance_id] = {
                        **candidate,
                        "field_visibility_count": sum(
                            instance_id in instance_ids
                            for frame_id, instance_ids in mapping_visible_instances.items()
                            if frame_id not in set(excluded)
                        ),
                        "field_surface_coverage": coverage,
                    }
                    best_coverage_by_instance[instance_id] = max(
                        best_coverage_by_instance[instance_id], coverage
                    )
                    best_visibility_by_instance[instance_id] = max(
                        best_visibility_by_instance[instance_id],
                        audited[instance_id]["field_visibility_count"],
                    )
                audit_cache[query_frames] = (excluded, audited)
            excluded, audited = audit_cache[query_frames]
            if any(
                audited[value]["field_visibility_count"] < config.min_field_visibility_count
                or audited[value]["field_surface_coverage"] < config.min_field_surface_coverage
                for value in subset
            ):
                failed_coverage_subsets += 1
                continue
            coverage = sum(audited[value]["field_surface_coverage"] for value in subset)
            selected_for_count = (
                (key[0], key[1], -coverage, query_frames, subset),
                subset,
                query_frames,
                excluded,
                audited,
            )
            break
        if selected_for_count is not None:
            _, chosen, chosen_query_frames, chosen_excluded, final_candidates = selected_for_count
            break
    if chosen is None or chosen_query_frames is None or chosen_excluded is None:
        class_counts: dict[int, int] = {}
        for value in base_candidates.values():
            class_id = int(value["nyu40_class_id"])
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        raise RuntimeError(
            f"{scene_id}: no formal target/query-frame construction exists; "
            f"base_candidates={len(base_candidates)}, nyu40_class_counts={class_counts}, "
            f"structural_subsets={structural_subset_count}, "
            f"camera_cover_proposals={sum(map(len, proposals_by_count.values()))}, "
            f"audited_camera_sets={audited_camera_sets}, "
            f"failed_coverage_subsets={failed_coverage_subsets}, "
            f"best_coverage_by_instance={best_coverage_by_instance}, "
            f"best_visibility_by_instance={best_visibility_by_instance}"
        )

    output = Path(output_root).resolve()
    scene_assets = output / "scene_assets" / scene_id
    crop_root = output / "private_crops" / scene_id
    scene_assets.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)
    mesh_path = scene_assets / "mesh_xyz.npy"
    ids_path = scene_assets / "mesh_instance_ids.npy"
    np.save(mesh_path, mesh_xyz.astype(np.float32), allow_pickle=False)
    np.save(ids_path, mesh_ids.astype(np.int32), allow_pickle=False)

    volumes = {
        instance_id: float(np.prod(np.maximum(np.ptp(mesh_xyz[mesh_ids == instance_id], axis=0), 1e-6)))
        for instance_id in chosen
    }
    cuts = np.quantile(list(volumes.values()), [1 / 3, 2 / 3])
    target_records = []
    for instance_id in chosen:
        candidate = final_candidates[instance_id]
        observations = candidate["query_observations"]
        frame_id = min(
            (value for value in chosen_query_frames if value in observations),
            key=lambda value: (-observations[value].pixel_count, value),
        )
        observation = observations[frame_id]
        frame = frame_by_id[frame_id]
        instance_image = np.asarray(Image.open(frame.instance), dtype=np.int64)
        mask = instance_image == observation.encoded_2d_id
        rgb = Image.open(frame.rgb).convert("RGB")
        crop, _box = build_image_query_crop(
            rgb, mask,
            padding_fraction=config.crop_padding_fraction,
            output_size=config.crop_size_px,
            fill_rgb=config.crop_out_of_bounds_fill_rgb,
        )
        crop_path = crop_root / f"instance_{instance_id:03d}.png"
        crop.save(crop_path, format="PNG", optimize=False, compress_level=9)
        depth = np.asarray(Image.open(frame.depth), dtype=np.uint16)
        aligned_depth_m = align_depth_to_color_raster(depth, depth_k, color_k, rgb.size)
        prompt = derive_paired_point_prompt(
            mask,
            aligned_depth_m,
            color_k,
            load_matrix(frame.pose),
            depth_scale=1.0,
        )
        class_id = int(candidate["nyu40_class_id"])
        raw_semantic_label = str(candidate["raw_semantic_label"])
        distractors = sorted(
            other for other, value in final_candidates.items()
            if other != instance_id
            and str(value["raw_semantic_label"]) == raw_semantic_label
            and value["field_visibility_count"] >= config.min_field_visibility_count
            and value["field_surface_coverage"] >= config.min_field_surface_coverage
        )
        volume = volumes[instance_id]
        size_bucket = "small" if volume <= cuts[0] else "medium" if volume <= cuts[1] else "large"
        expression = candidate["expression"]
        record = {
            "scene_id": scene_id,
                "instance_id": int(instance_id),
                "nyu40_class_id": class_id,
                "raw_semantic_label": raw_semantic_label,
                "mesh_vertex_count": int(candidate["metadata"]["num_vertices"]),
                "size_bucket": size_bucket,
                "same_class_distractor_instance_ids": distractors,
                "query_frame_id": frame_id,
                "expression": expression["expression"],
                "expression_annotation_id": expression["annotation_id"],
                "expression_source": expression["source"],
                "expression_view_independent": True,
                "expression_view_dependence_rule": expression["view_dependence_rule"],
                "crop_rgb_path": str(crop_path),
                "crop_rgb_sha256": sha256_file(crop_path),
                "camera_to_world": load_matrix(frame.pose).tolist(),
                "camera_intrinsics": color_k.tolist(),
                "raster_size": list(rgb.size),
                "positive_pixel_uv": prompt["positive_pixel_uv"],
                "click_depth_m": prompt["click_depth_m"],
                "point_world_xyz": prompt["point_world_xyz"],
                "projection_pixels": int(observation.pixel_count),
                "projection_fraction": float(observation.image_fraction),
                "projection_purity": float(observation.resolution_purity),
                "field_surface_coverage": float(candidate["field_surface_coverage"]),
            "field_visibility_count": int(candidate["field_visibility_count"]),
        }
        if text_profile == TEXT_PROFILE_UQIS_V2:
            record.update(
                {
                    "evaluation_tier": expression["evaluation_tier"],
                    "relational_language_required": expression[
                        "relational_language_required"
                    ],
                    "spatial_language_evidence": expression[
                        "spatial_language_evidence"
                    ],
                }
            )
        target_records.append(record)
    field_ids = [
        frame_id
        for frame_id in full_frame_ids
        if frame_id not in set(chosen_excluded)
        and np.isfinite(full_poses[frame_id]).all()
    ]
    scene_record = {
        "scene_id": scene_id,
        "mesh_xyz_path": str(mesh_path),
        "mesh_instance_ids_path": str(ids_path),
        "query_frame_ids": list(chosen_query_frames),
        "withheld_frame_ids": list(chosen_excluded),
        "field_frame_ids": field_ids,
        "max_query_frames": config.max_query_frames_per_scene,
    }
    receipt_body = {
        "schema_version": (
            "scannet_uqis_official_scene_derivation_v2_candidate"
            if text_profile == TEXT_PROFILE_UQIS_V2
            else "scannet_uqis_official_scene_derivation_v1"
        ),
        "benchmark_version": (
            UQIS_V2_CONSTRUCTION_CANDIDATE
            if text_profile == TEXT_PROFILE_UQIS_V2
            else BENCHMARK_VERSION
        ),
        "scene_id": scene_id,
        "status": "construction_complete",
        "method_predictions_opened": False,
        "sources": {
            "sens": _file_binding(sens_path),
            "mesh": _file_binding(mesh),
            "aggregation": _file_binding(aggregation),
            "segmentation": _file_binding(segmentation),
            "query_frame_derivation_receipt": (
                _file_binding(
                    Path(frames_root)
                    / scene_id
                    / "query_frame_derivation_receipt.json"
                )
                if (
                    Path(frames_root)
                    / scene_id
                    / "query_frame_derivation_receipt.json"
                ).is_file()
                else None
            ),
        },
        "full_sensor_frame_count": len(full_frame_ids),
        "nonfinite_pose_frame_count": sum(
            not np.isfinite(full_poses[frame_id]).all()
            for frame_id in full_frame_ids
        ),
        "sparse_projection_frame_count": len(sparse_ids),
        "coverage_surface_frame_count": len(mapping_surfaces),
        "query_frame_ids": list(chosen_query_frames),
        "withheld_frame_count": len(chosen_excluded),
        "field_frame_count": len(field_ids),
        "target_instance_ids": list(chosen),
        "protocol_config": asdict(config),
    }
    if text_profile == TEXT_PROFILE_UQIS_V2:
        receipt_body["text_profile"] = text_profile
    receipt = {**receipt_body, "receipt_sha256": canonical_json_sha256(receipt_body)}
    return scene_record, target_records, receipt


def _construct_scene_job(job: tuple[Any, ...]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    (
        scene_id,
        frames_root,
        sens_root,
        annotation_roots,
        nr3d_path,
        output_root,
        geometry_query_frames_root,
        text_profile,
    ) = job
    reference_index = _reference_index(load_reference_rows(nr3d_path))
    return construct_scene(
        scene_id=scene_id,
        frames_root=(
            geometry_query_frames_root
            if geometry_query_frames_root
            and (Path(geometry_query_frames_root) / scene_id).is_dir()
            else frames_root
        ),
        sens_path=Path(sens_root) / "scans" / scene_id / f"{scene_id}.sens",
        annotation_roots=annotation_roots,
        reference_index=reference_index,
        output_root=output_root,
        text_profile=text_profile,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--sens-root", required=True)
    parser.add_argument("--annotation-root", action="append", required=True)
    parser.add_argument("--nr3d", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--geometry-query-frames-root")
    parser.add_argument(
        "--text-profile",
        choices=TEXT_PROFILES,
        default=TEXT_PROFILE_LEGACY_V1,
        help=(
            "legacy_v1 preserves the frozen v0.1 selector; "
            "uqis_v2_core_relational prefers non-spatial Nr3D text and emits "
            "a non-formal v0.2 construction candidate"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path(args.nr3d).resolve()
    scenes, targets, receipts = [], [], []
    jobs = [
        (
            scene_id,
            args.frames_root,
            args.sens_root,
            tuple(args.annotation_root),
            str(reference_path),
            str(output),
            args.geometry_query_frames_root,
            args.text_profile,
        )
        for scene_id in PREREGISTERED_TEST_SCENES
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_construct_scene_job, jobs))
    else:
        results = [_construct_scene_job(job) for job in jobs]
    for scene, scene_targets, receipt in results:
        scenes.append(scene)
        targets.extend(scene_targets)
        receipts.append(receipt)
    output_benchmark_version = (
        UQIS_V2_CONSTRUCTION_CANDIDATE
        if args.text_profile == TEXT_PROFILE_UQIS_V2
        else BENCHMARK_VERSION
    )
    construction_receipts = {
        "benchmark_version": output_benchmark_version,
        "nr3d": _file_binding(reference_path),
        "cohort_derivation_ledger": list(COHORT_DERIVATION_LEDGER),
        "scenes": receipts,
    }
    if args.text_profile == TEXT_PROFILE_UQIS_V2:
        construction_receipts.update(
            {
                "formal_benchmark_eligible": False,
                "text_profile": args.text_profile,
            }
        )
    payloads = {
        "scene_records.json": {"benchmark_version": output_benchmark_version, "scenes": scenes},
        "target_records.json": {"benchmark_version": output_benchmark_version, "targets": targets},
        "construction_receipts.json": construction_receipts,
    }
    for name, payload in payloads.items():
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"scene_count": len(scenes), "target_count": len(targets), "output_root": str(output)}, indent=2))


if __name__ == "__main__":
    main()
