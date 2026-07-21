"""Strict construction contract for ScanNet-PFIR-Small.

The method-facing query contains only a real RGB crop.  Camera calibration,
depth, 2-D instance projections and ScanNet's 3-D annotations are private
construction/evaluation data and must never be consumed by a queried method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree


BENCHMARK_VERSION = "scannet-pfir-small-v1"
STRUCTURAL_NYU40_IDS = frozenset({1, 2, 22})
STRUCTURAL_LABELS = frozenset({"wall", "floor", "ceiling"})


@dataclass(frozen=True)
class ProtocolConfig:
    """Frozen v1 construction constants.

    Values are intentionally benchmark-level constants.  They may be changed
    only by minting another benchmark version, never after inspecting method
    results on test scenes.
    """

    bbox_padding: float = 0.10
    min_mesh_vertices: int = 500
    min_query_pixels: int = 1000
    min_query_fraction: float = 0.01
    max_border_fraction: float = 0.05
    min_resolution_purity: float = 0.90
    min_field_visibility_count: int = 5
    min_instance_surface_coverage: float = 0.70
    temporal_exclusion_radius: int = 5
    near_pose_translation_m: float = 0.10
    near_pose_rotation_deg: float = 8.0
    max_instances_per_scene: int = 8
    queries_per_instance: int = 2
    max_query_frames_per_scene: int = 6
    depth_stride: int = 2
    maximum_mesh_distance_m: float = 0.08
    coverage_distance_m: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 <= self.bbox_padding <= 1.0):
            raise ValueError("bbox_padding must be in [0,1]")
        for name in (
            "min_query_fraction",
            "max_border_fraction",
            "min_resolution_purity",
            "min_instance_surface_coverage",
        ):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0,1]")
        if self.queries_per_instance != 2:
            raise ValueError("v1 fixes one easy/medium and one hard query per instance")
        if self.depth_stride <= 0 or self.temporal_exclusion_radius < 0:
            raise ValueError("stride/radius must be non-negative")


@dataclass(frozen=True)
class FramePaths:
    frame_id: str
    rgb: Path
    depth: Path
    instance: Path
    label: Path
    pose: Path


@dataclass
class FrameInstanceObservation:
    encoded_2d_id: int
    nyu40_class_id: int
    pixel_count: int
    image_fraction: float
    border_fraction: float
    bbox_xyxy: tuple[int, int, int, int]
    instance_id_3d: int
    resolution_purity: float
    valid_depth_votes: int
    observed_world_xyz: np.ndarray


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    values = np.asarray(mask, dtype=bool)
    shape = np.asarray(values.shape, dtype=np.int64).tobytes()
    return hashlib.sha256(shape + np.packbits(values.reshape(-1)).tobytes()).hexdigest()


def load_matrix(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid finite 4x4 matrix: {path}")
    return matrix


def discover_frames(scene_dir: str | Path) -> list[FramePaths]:
    root = Path(scene_dir)
    frames: list[FramePaths] = []
    for rgb in sorted((root / "color").glob("*.jpg")):
        paths = {
            "depth": root / "depth" / f"{rgb.stem}.png",
            "instance": root / "instance" / f"{rgb.stem}.png",
            "label": root / "label" / f"{rgb.stem}.png",
            "pose": root / "pose" / f"{rgb.stem}.txt",
        }
        if all(path.is_file() for path in paths.values()):
            try:
                load_matrix(paths["pose"])
            except ValueError:
                continue
            frames.append(FramePaths(rgb.stem, rgb, **paths))
    if not frames:
        raise FileNotFoundError(f"no complete finite ScanNet frames in {root}")
    return frames


def padded_bbox(mask: np.ndarray, padding: float = 0.10) -> tuple[int, int, int, int]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not bool(values.any()):
        raise ValueError("mask must be a non-empty 2-D array")
    y, x = np.nonzero(values)
    x0, x1 = int(x.min()), int(x.max()) + 1
    y0, y1 = int(y.min()), int(y.max()) + 1
    pad_x = int(math.ceil((x1 - x0) * float(padding)))
    pad_y = int(math.ceil((y1 - y0) * float(padding)))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(values.shape[1], x1 + pad_x),
        min(values.shape[0], y1 + pad_y),
    )


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    return padded_bbox(mask, 0.0)


def _border_fraction(mask: np.ndarray, width: int = 2) -> float:
    values = np.asarray(mask, dtype=bool)
    border = np.zeros_like(values)
    border[:width] = True
    border[-width:] = True
    border[:, :width] = True
    border[:, -width:] = True
    return float(np.logical_and(values, border).sum() / max(int(values.sum()), 1))


def load_mesh_instances(
    mesh_path: str | Path,
    aggregation_path: str | Path,
    segmentation_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    """Load official annotation mesh and map vertices to objectId+1."""

    ply = PlyData.read(str(mesh_path))
    vertex = ply["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    segmentation = json.loads(Path(segmentation_path).read_text(encoding="utf-8"))
    aggregation = json.loads(Path(aggregation_path).read_text(encoding="utf-8"))
    segment_ids = np.asarray(segmentation.get("segIndices", []), dtype=np.int64)
    if segment_ids.shape != (xyz.shape[0],):
        raise ValueError("segIndices and annotation-mesh vertex rows do not align")
    instance_ids = np.zeros(xyz.shape[0], dtype=np.int32)
    metadata: dict[int, dict[str, Any]] = {}
    for group in aggregation.get("segGroups", []):
        instance_id = int(group["objectId"]) + 1
        selected = np.isin(
            segment_ids, np.asarray(group.get("segments", []), dtype=np.int64)
        )
        if bool((instance_ids[selected] != 0).any()):
            raise ValueError(f"overlapping ScanNet segments for instance {instance_id}")
        instance_ids[selected] = instance_id
        metadata[instance_id] = {
            "object_id": int(group["objectId"]),
            "label": str(group.get("label", "")),
            "num_vertices": int(selected.sum()),
        }
    if not metadata:
        raise ValueError("aggregation contains no non-empty 3-D instances")
    return xyz, instance_ids, metadata


def _depth_to_labeled_world(
    depth: np.ndarray,
    instance_image: np.ndarray,
    depth_intrinsic: np.ndarray,
    color_intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift depth samples and read their corresponding color-space instance ID."""

    height, width = depth.shape
    yy, xx = np.mgrid[0:height:int(stride), 0:width:int(stride)]
    z = depth[yy, xx].astype(np.float64) / 1000.0
    valid = np.isfinite(z) & (z > 0.20) & (z < 8.0)
    x = (xx - depth_intrinsic[0, 2]) * z / depth_intrinsic[0, 0]
    y = (yy - depth_intrinsic[1, 2]) * z / depth_intrinsic[1, 1]
    uc = color_intrinsic[0, 0] * x / np.maximum(z, 1e-8) + color_intrinsic[0, 2]
    vc = color_intrinsic[1, 1] * y / np.maximum(z, 1e-8) + color_intrinsic[1, 2]
    ui = np.rint(uc).astype(np.int64)
    vi = np.rint(vc).astype(np.int64)
    valid &= (
        (ui >= 0)
        & (vi >= 0)
        & (ui < instance_image.shape[1])
        & (vi < instance_image.shape[0])
    )
    encoded = instance_image[vi[valid], ui[valid]].astype(np.int64)
    camera = np.stack(
        [x[valid], y[valid], z[valid], np.ones(int(valid.sum()))], axis=1
    )
    world = (camera @ camera_to_world.T)[:, :3].astype(np.float32)
    return world, encoded


def resolve_frame_observations(
    frame: FramePaths,
    mesh_xyz: np.ndarray,
    mesh_instance_ids: np.ndarray,
    depth_intrinsic: np.ndarray,
    color_intrinsic: np.ndarray,
    *,
    depth_stride: int = 2,
    maximum_mesh_distance_m: float = 0.08,
    mesh_tree: cKDTree | None = None,
) -> dict[int, FrameInstanceObservation]:
    """Independently confirm official 2-D projections against the 3-D mesh."""

    instance_image = np.asarray(Image.open(frame.instance), dtype=np.int64)
    label_image = np.asarray(Image.open(frame.label), dtype=np.int64)
    depth = np.asarray(Image.open(frame.depth), dtype=np.uint16)
    if instance_image.shape != label_image.shape:
        raise ValueError(f"instance/label projection mismatch in {frame.frame_id}")
    if label_image.size and (
        int(label_image.min()) < 0 or int(label_image.max()) > 40
    ):
        raise ValueError(
            f"{frame.frame_id}: semantic projections are not in NYU40 id space"
        )
    world, encoded = _depth_to_labeled_world(
        depth,
        instance_image,
        depth_intrinsic,
        color_intrinsic,
        load_matrix(frame.pose),
        stride=depth_stride,
    )
    nonzero = encoded > 0
    world, encoded = world[nonzero], encoded[nonzero]
    if world.size:
        tree = mesh_tree if mesh_tree is not None else cKDTree(np.asarray(mesh_xyz))
        distance, nearest = tree.query(world, k=1)
        matched_3d = np.asarray(mesh_instance_ids)[nearest]
        valid_match = (
            np.isfinite(distance)
            & (distance <= float(maximum_mesh_distance_m))
            & (matched_3d > 0)
        )
    else:
        matched_3d = np.empty(0, dtype=np.int32)
        valid_match = np.empty(0, dtype=bool)

    observations: dict[int, FrameInstanceObservation] = {}
    image_pixels = int(instance_image.size)
    for encoded_id in np.unique(instance_image):
        encoded_id = int(encoded_id)
        if encoded_id <= 0:
            continue
        mask = instance_image == encoded_id
        pixel_count = int(mask.sum())
        labels, label_counts = np.unique(label_image[mask], return_counts=True)
        nonzero_labels = labels > 0
        if bool(nonzero_labels.any()):
            labels, label_counts = labels[nonzero_labels], label_counts[nonzero_labels]
            nyu40_id = int(labels[int(np.argmax(label_counts))])
        else:
            nyu40_id = 0
        selected = (encoded == encoded_id) & valid_match
        votes = matched_3d[selected]
        if votes.size:
            vote_ids, vote_counts = np.unique(votes, return_counts=True)
            best = int(np.argmax(vote_counts))
            instance_id = int(vote_ids[best])
            purity = float(vote_counts[best] / votes.size)
            points = world[selected][votes == instance_id]
        else:
            instance_id, purity = 0, 0.0
            points = np.empty((0, 3), dtype=np.float32)
        observations[encoded_id] = FrameInstanceObservation(
            encoded_2d_id=encoded_id,
            nyu40_class_id=nyu40_id,
            pixel_count=pixel_count,
            image_fraction=float(pixel_count / image_pixels),
            border_fraction=_border_fraction(mask),
            bbox_xyxy=_tight_bbox(mask),
            instance_id_3d=instance_id,
            resolution_purity=purity,
            valid_depth_votes=int(votes.size),
            observed_world_xyz=points.astype(np.float32, copy=False),
        )
    return observations


def pose_distance(
    first_camera_to_world: np.ndarray, second_camera_to_world: np.ndarray
) -> tuple[float, float]:
    first = np.asarray(first_camera_to_world, dtype=np.float64)
    second = np.asarray(second_camera_to_world, dtype=np.float64)
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return translation, float(np.degrees(np.arccos(cosine)))


def exclusion_frame_ids(
    frames: Sequence[FramePaths],
    query_frame_ids: Iterable[str],
    *,
    temporal_radius: int = 5,
    translation_m: float = 0.10,
    rotation_deg: float = 8.0,
) -> list[str]:
    """Union temporal and pose-near exclusions for all scene queries."""

    index = {frame.frame_id: i for i, frame in enumerate(frames)}
    poses = {frame.frame_id: load_matrix(frame.pose) for frame in frames}
    excluded: set[str] = set()
    for query_id in query_frame_ids:
        if query_id not in index:
            raise ValueError(f"unknown query frame: {query_id}")
        centre = index[query_id]
        lo, hi = max(0, centre - temporal_radius), min(len(frames), centre + temporal_radius + 1)
        excluded.update(frame.frame_id for frame in frames[lo:hi])
        for frame in frames:
            translation, rotation = pose_distance(poses[query_id], poses[frame.frame_id])
            if translation < float(translation_m) and rotation < float(rotation_deg):
                excluded.add(frame.frame_id)
    return sorted(excluded, key=lambda value: index[value])


def instance_surface_coverage(
    target_xyz: np.ndarray,
    observed_world_xyz: Sequence[np.ndarray],
    *,
    distance_m: float = 0.05,
) -> float:
    points = [np.asarray(value, dtype=np.float32) for value in observed_world_xyz if len(value)]
    if not points or len(target_xyz) == 0:
        return 0.0
    observed = np.concatenate(points, axis=0)
    distance, _ = cKDTree(observed).query(np.asarray(target_xyz), k=1)
    return float(np.mean(distance <= float(distance_m)))


def _nearest_field_pose(
    query: FramePaths, field_frames: Sequence[FramePaths]
) -> tuple[float, float]:
    distances = [
        pose_distance(load_matrix(query.pose), load_matrix(frame.pose))
        for frame in field_frames
    ]
    return min(distances, key=lambda value: (value[0], value[1]))


def _eligible_observation(
    observation: FrameInstanceObservation,
    metadata: Mapping[int, Mapping[str, Any]],
    config: ProtocolConfig,
) -> bool:
    meta = metadata.get(observation.instance_id_3d, {})
    label = str(meta.get("label", "")).strip().lower()
    return bool(
        observation.instance_id_3d > 0
        and 0 < observation.nyu40_class_id <= 40
        and observation.resolution_purity > config.min_resolution_purity
        and int(meta.get("num_vertices", 0)) >= config.min_mesh_vertices
        and (
            observation.pixel_count >= config.min_query_pixels
            or observation.image_fraction >= config.min_query_fraction
        )
        and observation.border_fraction <= config.max_border_fraction
        and observation.nyu40_class_id not in STRUCTURAL_NYU40_IDS
        and label not in STRUCTURAL_LABELS
    )


def _choose_two_views(
    candidates: Sequence[tuple[FramePaths, FrameInstanceObservation]],
) -> list[tuple[str, FramePaths, FrameInstanceObservation]]:
    """Deterministically choose complete and hard views without model scores."""

    maximum_area = max(row[1].pixel_count for row in candidates)
    easy = max(
        candidates,
        key=lambda row: (
            row[1].pixel_count / maximum_area - row[1].border_fraction,
            row[1].resolution_purity,
            -int(row[0].frame_id),
        ),
    )
    remaining = [row for row in candidates if row[0].frame_id != easy[0].frame_id]
    if not remaining:
        return []
    easy_pose = load_matrix(easy[0].pose)
    # A hard crop has less visible area and/or a larger viewpoint change.  The
    # fixed score is construction-only and never examines any method output.
    def hard_key(row: tuple[FramePaths, FrameInstanceObservation]) -> tuple[float, float, int]:
        _, angle = pose_distance(easy_pose, load_matrix(row[0].pose))
        relative_area = row[1].pixel_count / maximum_area
        difficulty = (1.0 - relative_area) + min(angle / 90.0, 1.0)
        return difficulty, -row[1].resolution_purity, -int(row[0].frame_id)

    hard = max(remaining, key=hard_key)
    return [("easy_medium", easy[0], easy[1]), ("hard", hard[0], hard[1])]


def _choose_global_query_pairs(
    by_instance: Mapping[int, Sequence[tuple[FramePaths, FrameInstanceObservation]]],
    *,
    maximum_instances: int,
    maximum_query_frames: int = 6,
) -> dict[int, list[tuple[str, FramePaths, FrameInstanceObservation]]]:
    """Share held-out frames across objects instead of deleting the whole scan.

    Selecting each instance independently can produce sixteen different query
    frames; their ±5 neighborhoods may cover a short ScanNet sequence.  This
    deterministic set-cover variant first finds the frame pair supporting the
    most objects, then adds only frames that increase the number of objects
    with two eligible views.  It uses projection/pose metadata only.
    """

    usable = {
        int(instance_id): list(candidates)
        for instance_id, candidates in by_instance.items()
        if len({frame.frame_id for frame, _ in candidates}) >= 2
    }
    frame_ids = sorted(
        {frame.frame_id for candidates in usable.values() for frame, _ in candidates},
        key=int,
    )
    availability = {
        instance_id: {frame.frame_id for frame, _ in candidates}
        for instance_id, candidates in usable.items()
    }
    if len(frame_ids) < 2:
        return {}

    def supported(pool: set[str]) -> set[int]:
        return {
            instance_id
            for instance_id, visible in availability.items()
            if len(visible.intersection(pool)) >= 2
        }

    best_pair: tuple[str, str] | None = None
    best_key: tuple[int, int, int] | None = None
    for first_index, first in enumerate(frame_ids[:-1]):
        for second in frame_ids[first_index + 1 :]:
            pool = {first, second}
            count = min(len(supported(pool)), int(maximum_instances))
            # Prefer temporal separation after object support; it gives a
            # deterministic viewpoint-diversity proxy before loading poses.
            key = (count, abs(int(second) - int(first)), -int(first))
            if best_key is None or key > best_key:
                best_pair, best_key = (first, second), key
    assert best_pair is not None
    pool = set(best_pair)
    while len(pool) < int(maximum_query_frames):
        current = supported(pool)
        if len(current) >= int(maximum_instances):
            break
        candidate: str | None = None
        candidate_key: tuple[int, int, int] | None = None
        for frame_id in frame_ids:
            if frame_id in pool:
                continue
            expanded = supported(pool | {frame_id})
            newly_supported = len(expanded - current)
            partial_progress = sum(
                min(2, len(visible.intersection(pool | {frame_id})))
                - min(2, len(visible.intersection(pool)))
                for visible in availability.values()
            )
            separation = min(abs(int(frame_id) - int(value)) for value in pool)
            key = (newly_supported, partial_progress, separation)
            if candidate_key is None or key > candidate_key:
                candidate, candidate_key = frame_id, key
        if candidate is None or candidate_key is None or candidate_key[:2] == (0, 0):
            break
        pool.add(candidate)

    chosen: dict[int, list[tuple[str, FramePaths, FrameInstanceObservation]]] = {}
    for instance_id in sorted(supported(pool)):
        candidates = [
            row for row in usable[instance_id] if row[0].frame_id in pool
        ]
        pair = _choose_two_views(candidates)
        if len(pair) == 2:
            chosen[instance_id] = pair
    return chosen


def build_scene_records(
    *,
    scene_id: str,
    frames_root: str | Path,
    mesh_path: str | Path,
    aggregation_path: str | Path,
    segmentation_path: str | Path,
    config: ProtocolConfig = ProtocolConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construct all valid query records for one scene.

    The returned records still contain private GT fields.  Call
    :func:`freeze_manifest` to split method-facing and evaluator-only files.
    """

    scene_dir = Path(frames_root) / scene_id
    frames = discover_frames(scene_dir)
    depth_intrinsic = load_matrix(scene_dir / "intrinsics_depth.txt")
    color_intrinsic = load_matrix(scene_dir / "intrinsics_color.txt")
    mesh_xyz, mesh_instances, metadata = load_mesh_instances(
        mesh_path, aggregation_path, segmentation_path
    )
    mesh_tree = cKDTree(mesh_xyz)
    per_frame: dict[str, dict[int, FrameInstanceObservation]] = {}
    for frame in frames:
        per_frame[frame.frame_id] = resolve_frame_observations(
            frame,
            mesh_xyz,
            mesh_instances,
            depth_intrinsic,
            color_intrinsic,
            depth_stride=config.depth_stride,
            maximum_mesh_distance_m=config.maximum_mesh_distance_m,
            mesh_tree=mesh_tree,
        )

    by_instance: dict[int, list[tuple[FramePaths, FrameInstanceObservation]]] = {}
    for frame in frames:
        for observation in per_frame[frame.frame_id].values():
            if _eligible_observation(observation, metadata, config):
                by_instance.setdefault(observation.instance_id_3d, []).append(
                    (frame, observation)
                )
    pair_candidates = _choose_global_query_pairs(
        by_instance,
        maximum_instances=config.max_instances_per_scene,
        maximum_query_frames=config.max_query_frames_per_scene,
    )

    def audit_selection(
        selection: Mapping[
            int, Sequence[tuple[str, FramePaths, FrameInstanceObservation]]
        ],
    ) -> tuple[list[str], list[FramePaths], dict[int, tuple[int, float]], bool]:
        query_ids = [
            frame.frame_id for rows in selection.values() for _, frame, _ in rows
        ]
        excluded_ids = exclusion_frame_ids(
            frames,
            query_ids,
            temporal_radius=config.temporal_exclusion_radius,
            translation_m=config.near_pose_translation_m,
            rotation_deg=config.near_pose_rotation_deg,
        )
        field = [
            frame for frame in frames if frame.frame_id not in set(excluded_ids)
        ]
        selection_audits: dict[int, tuple[int, float]] = {}
        valid = True
        for instance_id in selection:
            observations = [
                observation
                for frame in field
                for observation in per_frame[frame.frame_id].values()
                if (
                    observation.instance_id_3d == instance_id
                    and observation.resolution_purity > config.min_resolution_purity
                )
            ]
            coverage = instance_surface_coverage(
                mesh_xyz[mesh_instances == instance_id],
                [observation.observed_world_xyz for observation in observations],
                distance_m=config.coverage_distance_m,
            )
            selection_audits[instance_id] = (len(observations), coverage)
            valid &= bool(
                len(observations) >= config.min_field_visibility_count
                and coverage >= config.min_instance_surface_coverage
            )
        return excluded_ids, field, selection_audits, valid

    # First reject pairs that cannot support their own fully-held-out field.
    # Then greedily add pairs only when *all* already accepted instances retain
    # the frozen visibility/coverage contract.  Removing all failures in one
    # pass is incorrect: their combined exclusions can make individually valid
    # objects appear invalid.
    individual: dict[int, tuple[int, float, int]] = {}
    for instance_id, rows in pair_candidates.items():
        excluded_one, _, audits_one, valid_one = audit_selection(
            {instance_id: rows}
        )
        if valid_one:
            visibility, coverage = audits_one[instance_id]
            individual[instance_id] = (
                visibility,
                coverage,
                len(excluded_one),
            )

    # Prefer objects that create same-category distractors, then pairs leaving
    # more field evidence, then stable IDs.  All criteria are construction-only.
    class_frequency: dict[int, int] = {}
    for instance_id in individual:
        class_id = pair_candidates[instance_id][0][2].nyu40_class_id
        class_frequency[class_id] = class_frequency.get(class_id, 0) + 1
    ordered_instances = sorted(
        individual,
        key=lambda instance_id: (
            -int(
                class_frequency[
                    pair_candidates[instance_id][0][2].nyu40_class_id
                ]
                > 1
            ),
            individual[instance_id][2],
            -individual[instance_id][1],
            -individual[instance_id][0],
            instance_id,
        ),
    )
    chosen: dict[
        int, list[tuple[str, FramePaths, FrameInstanceObservation]]
    ] = {}
    for instance_id in ordered_instances:
        if len(chosen) >= config.max_instances_per_scene:
            break
        trial = {**chosen, instance_id: pair_candidates[instance_id]}
        _, _, _, valid_trial = audit_selection(trial)
        if valid_trial:
            chosen[instance_id] = pair_candidates[instance_id]

    if not chosen:
        return [], {
            "scene_id": scene_id,
            "status": "no_valid_queries",
            "frame_count": len(frames),
            "resolved_candidate_instances": len(by_instance),
        }

    excluded, field_frames, audits, valid_final = audit_selection(chosen)
    if not valid_final:
        raise RuntimeError("internal PFIR selection/audit disagreement")
    field_ids = [frame.frame_id for frame in field_frames]
    field_hash = canonical_json_sha256(field_ids)
    # Ranking is against every reconstructable non-structural instance in the
    # scene, not merely the at-most-eight instances selected as query targets.
    field_observations: dict[int, list[FrameInstanceObservation]] = {}
    for frame in field_frames:
        for observation in per_frame[frame.frame_id].values():
            if (
                observation.instance_id_3d > 0
                and observation.resolution_purity > config.min_resolution_purity
            ):
                field_observations.setdefault(
                    observation.instance_id_3d, []
                ).append(observation)
    candidate_class_ids: dict[int, int] = {}
    for instance_id, observations in field_observations.items():
        meta = metadata.get(instance_id, {})
        if (
            int(meta.get("num_vertices", 0)) < config.min_mesh_vertices
            or str(meta.get("label", "")).strip().lower() in STRUCTURAL_LABELS
            or len(observations) < config.min_field_visibility_count
        ):
            continue
        coverage = instance_surface_coverage(
            mesh_xyz[mesh_instances == instance_id],
            [observation.observed_world_xyz for observation in observations],
            distance_m=config.coverage_distance_m,
        )
        if coverage < config.min_instance_surface_coverage:
            continue
        class_votes: dict[int, int] = {}
        for observation in observations:
            class_votes[observation.nyu40_class_id] = (
                class_votes.get(observation.nyu40_class_id, 0)
                + observation.pixel_count
            )
        class_id = max(class_votes, key=lambda value: (class_votes[value], -value))
        if class_id <= 0 or class_id > 40 or class_id in STRUCTURAL_NYU40_IDS:
            continue
        candidate_class_ids[instance_id] = int(class_id)
    candidate_ids = sorted(candidate_class_ids)
    missing_targets = sorted(set(chosen) - set(candidate_ids))
    if missing_targets:
        raise RuntimeError(
            f"query targets failed final ranking-candidate audit: {missing_targets}"
        )
    records: list[dict[str, Any]] = []
    for instance_id in sorted(chosen):
        class_id = chosen[instance_id][0][2].nyu40_class_id
        visibility_count, coverage = audits[instance_id]
        same_category = sum(
            candidate_class_ids[other] == class_id
            for other in candidate_ids
            if other != instance_id
        )
        for difficulty, frame, observation in chosen[instance_id]:
            instance_image = np.asarray(Image.open(frame.instance), dtype=np.int64)
            target_2d_mask = instance_image == observation.encoded_2d_id
            bbox = padded_bbox(target_2d_mask, config.bbox_padding)
            translation, rotation = _nearest_field_pose(frame, field_frames)
            query_id = f"{scene_id}_{frame.frame_id}_i{instance_id:03d}_{difficulty}"
            records.append(
                {
                    "benchmark_version": BENCHMARK_VERSION,
                    "query_id": query_id,
                    "scene_id": scene_id,
                    "space_id": scene_id.split("_")[0],
                    "query_frame_id": frame.frame_id,
                    "query_rgb_path": str(frame.rgb.resolve()),
                    "query_rgb_sha256": sha256_file(frame.rgb),
                    "instance_id_3d": instance_id,
                    "encoded_instance_id_2d": observation.encoded_2d_id,
                    "nyu40_class_id": class_id,
                    "instance_label": metadata[instance_id]["label"],
                    "bbox_xyxy": list(bbox),
                    "bbox_padding": config.bbox_padding,
                    "mask_sha256": mask_sha256(target_2d_mask),
                    "query_type": "bbox",
                    "difficulty": difficulty,
                    "field_frame_manifest_sha256": field_hash,
                    "field_frame_ids": field_ids,
                    "query_exclusion_frames": excluded,
                    "nearest_field_pose_translation": translation,
                    "nearest_field_pose_rotation": rotation,
                    "instance_field_visibility_count": visibility_count,
                    "instance_surface_coverage": coverage,
                    "instance_mesh_vertex_count": metadata[instance_id]["num_vertices"],
                    "resolution_purity": observation.resolution_purity,
                    "same_category_distractor_count": same_category,
                    "candidate_instance_ids_3d": candidate_ids,
                    "candidate_instance_class_ids": {
                        str(other): candidate_class_ids[other]
                        for other in candidate_ids
                    },
                    "method_visible_query_fields": ["crop_rgb", "scene_id"],
                    "query_pose_used_by_method": False,
                    "query_depth_used_by_method": False,
                    "query_mask_used_by_method": False,
                }
            )
    scene_report = {
        "scene_id": scene_id,
        "status": "ok",
        "frame_count": len(frames),
        "field_frame_count": len(field_frames),
        "excluded_frame_count": len(excluded),
        "instance_count": len(chosen),
        "ranking_candidate_instance_count": len(candidate_ids),
        "query_count": len(records),
        "field_frame_manifest_sha256": field_hash,
        "protocol_config": asdict(config),
    }
    return records, scene_report


def audit_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = list(payload.get("queries", []))
    errors: list[str] = []
    query_ids: set[str] = set()
    scene_fields: dict[str, tuple[str, ...]] = {}
    for record in records:
        query_id = str(record.get("query_id", ""))
        if not query_id or query_id in query_ids:
            errors.append(f"missing/duplicate query_id: {query_id!r}")
        query_ids.add(query_id)
        query_frame = str(record.get("query_frame_id", ""))
        field_ids = tuple(map(str, record.get("field_frame_ids", [])))
        exclusions = set(map(str, record.get("query_exclusion_frames", [])))
        if query_frame in field_ids or query_frame not in exclusions:
            errors.append(f"{query_id}: query frame leakage")
        if exclusions.intersection(field_ids):
            errors.append(f"{query_id}: excluded frame appears in field")
        expected_hash = canonical_json_sha256(list(field_ids))
        if record.get("field_frame_manifest_sha256") != expected_hash:
            errors.append(f"{query_id}: field frame hash mismatch")
        previous = scene_fields.setdefault(str(record.get("scene_id")), field_ids)
        if previous != field_ids:
            errors.append(f"{query_id}: scene was not reconstructed once")
        if record.get("query_type") != "bbox" or record.get("bbox_padding") != 0.10:
            errors.append(f"{query_id}: non-v1 query crop contract")
        if payload.get("visibility") == "public":
            leaked = sorted(
                {
                    "instance_id_3d",
                    "encoded_instance_id_2d",
                    "nyu40_class_id",
                    "instance_label",
                    "candidate_instance_ids_3d",
                    "candidate_instance_class_ids",
                }.intersection(record)
            )
            if leaked:
                errors.append(f"{query_id}: private GT fields leaked: {leaked}")
    return {
        "benchmark_version": payload.get("benchmark_version"),
        "query_count": len(records),
        "scene_count": len(scene_fields),
        "valid": not errors,
        "errors": errors,
    }


def freeze_manifest(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    split_role: str,
    scene_reports: Sequence[Mapping[str, Any]] = (),
    config: ProtocolConfig = ProtocolConfig(),
    selection_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write immutable public/private manifests and deterministic bbox crops."""

    if split_role not in {"dev", "test"}:
        raise ValueError("split_role must be dev or test")
    root = Path(output_dir)
    crop_root = root / "queries" / "bbox"
    masked_crop_root = root / "queries" / "masked_oracle"
    crop_root.mkdir(parents=True, exist_ok=True)
    masked_crop_root.mkdir(parents=True, exist_ok=True)
    full_records: list[dict[str, Any]] = []
    for source in sorted(records, key=lambda row: str(row["query_id"])):
        record = dict(source)
        class_ids = [
            int(record["nyu40_class_id"]),
            *[
                int(value)
                for value in record["candidate_instance_class_ids"].values()
            ],
        ]
        if any(value <= 0 or value > 40 for value in class_ids):
            raise ValueError(
                f"{record['query_id']}: class IDs are not valid NYU40 ids"
            )
        image = Image.open(record["query_rgb_path"]).convert("RGB")
        crop_path = crop_root / f"{record['query_id']}.png"
        bbox = tuple(record["bbox_xyxy"])
        image.crop(bbox).save(crop_path)
        record["crop_rgb_path"] = str(crop_path.resolve())
        record["crop_rgb_sha256"] = sha256_file(crop_path)
        instance_image = np.asarray(
            Image.open(
                Path(record["query_rgb_path"]).parent.parent
                / "instance"
                / f"{record['query_frame_id']}.png"
            ),
            dtype=np.int64,
        )
        binary = instance_image == int(record["encoded_instance_id_2d"])
        rgb = np.asarray(image, dtype=np.uint8)
        masked = np.where(binary[..., None], rgb, 0)
        x0, y0, x1, y1 = bbox
        masked_path = masked_crop_root / f"{record['query_id']}.png"
        Image.fromarray(masked[y0:y1, x0:x1]).save(masked_path)
        record["masked_crop_rgb_path"] = str(masked_path.resolve())
        record["masked_crop_rgb_sha256"] = sha256_file(masked_path)
        full_records.append(record)

    private_keys = {
        "instance_id_3d",
        "encoded_instance_id_2d",
        "nyu40_class_id",
        "instance_label",
        "same_category_distractor_count",
        "candidate_instance_ids_3d",
        "candidate_instance_class_ids",
        "query_rgb_path",
        "masked_crop_rgb_path",
        "masked_crop_rgb_sha256",
    }
    public_records = [
        {key: value for key, value in row.items() if key not in private_keys}
        for row in full_records
    ]
    gt_records = [
        {
            "query_id": row["query_id"],
            "scene_id": row["scene_id"],
            "instance_id_3d": row["instance_id_3d"],
            "nyu40_class_id": row["nyu40_class_id"],
            "candidate_instance_ids_3d": row["candidate_instance_ids_3d"],
            "candidate_instance_class_ids": row["candidate_instance_class_ids"],
        }
        for row in full_records
    ]
    common = {
        "benchmark_version": BENCHMARK_VERSION,
        "split_role": split_role,
        "protocol_config": asdict(config),
        "scene_reports": list(scene_reports),
        "scene_selection": dict(selection_metadata or {}),
    }
    public = {**common, "visibility": "public", "queries": public_records}
    private = {**common, "visibility": "evaluator_only", "queries": gt_records}
    full = {**common, "visibility": "internal", "queries": full_records}
    method_records = [
        {
            "benchmark_version": BENCHMARK_VERSION,
            "query_id": row["query_id"],
            "scene_id": row["scene_id"],
            "query_type": "bbox",
            "difficulty": row["difficulty"],
            "crop_rgb_path": row["crop_rgb_path"],
            "crop_rgb_sha256": row["crop_rgb_sha256"],
            "available_method_inputs": ["scene_id", "crop_rgb"],
        }
        for row in full_records
    ]
    masked_method_records = [
        {
            **{key: value for key, value in row.items() if key != "query_type"},
            "query_type": "masked_crop_oracle",
            "crop_rgb_path": full_records[index]["masked_crop_rgb_path"],
            "crop_rgb_sha256": full_records[index]["masked_crop_rgb_sha256"],
        }
        for index, row in enumerate(method_records)
    ]
    method = {**common, "visibility": "method_input", "queries": method_records}
    masked_method = {
        **common,
        "visibility": "method_input",
        "track_variant": "masked_crop_oracle",
        "queries": masked_method_records,
    }
    for name, payload in (
        ("manifest.method.json", method),
        ("manifest.masked_oracle.method.json", masked_method),
        ("manifest.public.json", public),
        ("manifest.evaluator.json", private),
        ("manifest.internal.json", full),
    ):
        (root / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    audit = audit_manifest(public)
    if not audit["valid"]:
        raise RuntimeError(f"frozen public manifest failed audit: {audit['errors']}")
    hashes = {
        name: sha256_file(root / name)
        for name in (
            "manifest.method.json",
            "manifest.masked_oracle.method.json",
            "manifest.public.json",
            "manifest.evaluator.json",
            "manifest.internal.json",
        )
    }
    release = {
        **common,
        "query_count": len(full_records),
        "scene_count": len({row["scene_id"] for row in full_records}),
        "manifest_sha256": hashes,
        "audit": audit,
    }
    (root / "release.json").write_text(
        json.dumps(release, indent=2, sort_keys=True), encoding="utf-8"
    )
    return release
