"""Frozen information contract for ScanNet-UQIS-9.

The external seam is deliberately small: construction code supplies normalized
scene and target records to :func:`freeze_release`; methods consume exactly one
of the four query manifests; the evaluator alone consumes target pairing and
instance identities.  Dataset parsing and model adapters live outside this
module so neither can weaken the information firewall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


BENCHMARK_VERSION = "scannet-uqis-9-v0.1"
BENCHMARK_VERSION_V2_CANDIDATE = "scannet-uqis-9-v0.2-construction-candidate"
PREDICTION_DOMAIN = "official_scannet_mesh_vertex_probability"
FORMAL_RELEASE_IMPLEMENTED = False
PREREGISTERED_TEST_SCENES = (
    "scene0030_00",
    "scene0249_00",
    "scene0353_00",
    "scene0435_00",
    "scene0700_00",
    "scene0704_00",
    "scene0246_00",
    "scene0203_00",
    "scene0535_00",
)
PREREGISTERED_REPLACEMENTS: tuple[str, ...] = ()
COHORT_DERIVATION_LEDGER = (
    {
        "scene_id": "scene0046_00",
        "decision": "excluded_before_formal_release",
        "reason": "nr3d_eligible_targets_span_only_three_raw_semantic_categories",
    },
    {
        "scene_id": "scene0412_00",
        "decision": "excluded_before_formal_release",
        "reason": "nr3d_eligible_targets_span_only_three_raw_semantic_categories",
    },
    {
        "scene_id": "scene0550_00",
        "decision": "excluded_before_formal_release",
        "reason": "only_two_nr3d_eligible_geometry_targets",
    },
    {
        "scene_id": "scene0678_00",
        "decision": "excluded_before_formal_release",
        "reason": "nr3d_eligible_targets_span_only_three_raw_semantic_categories",
    },
    {
        "scene_id": "scene0246_00",
        "decision": "included_as_preregistered_replacement",
        "reason": "seven_nr3d_eligible_targets_across_five_raw_semantic_categories",
    },
    {
        "scene_id": "scene0203_00",
        "decision": "included_before_any_formal_method_evaluation",
        "reason": "sixteen_nr3d_eligible_targets_across_seven_raw_semantic_categories",
    },
    {
        "scene_id": "scene0535_00",
        "decision": "included_before_any_formal_method_evaluation",
        "reason": "eight_nr3d_eligible_targets_across_four_raw_semantic_categories",
    },
    {
        "scene_id": None,
        "decision": "freeze_nine_scene_cohort",
        "reason": "user_accepted_nine_scenes_without_mining_full_validation_for_a_tenth",
    },
)
QUERY_MANIFEST_NAMES = {
    "text": "query_manifest.text.json",
    "image": "query_manifest.image.json",
    "point_2d": "query_manifest.2d_point.json",
    "point_3d": "query_manifest.3d_point.json",
}
RELEASE_MANIFEST_NAMES = (
    "scene_manifest.json",
    "target_manifest.public.json",
    "target_manifest.evaluator.json",
    "field_exclusion_manifest.json",
    *QUERY_MANIFEST_NAMES.values(),
)


class QueryModality(str, Enum):
    """The four independently executed authorized query inputs."""

    TEXT = "text"
    IMAGE = "image"
    POINT_2D = "point_2d"
    POINT_3D = "point_3d"


@dataclass(frozen=True)
class UQISProtocolConfig:
    """Versioned construction and evaluation constants.

    Changing any value mints a different benchmark release.  Test labels or
    predictions must never be used to change these values in-place.
    """

    min_targets_per_scene: int = 6
    max_targets_per_scene: int = 8
    max_query_frames_per_scene: int = 3
    min_same_class_targets_per_scene: int = 3
    min_semantic_categories_per_scene: int = 4
    min_mesh_vertices: int = 500
    min_query_pixels: int = 1000
    min_query_fraction: float = 0.01
    min_projection_purity: float = 0.90
    min_field_surface_coverage: float = 0.70
    min_field_visibility_count: int = 5
    temporal_exclusion_radius: int = 5
    near_pose_translation_m: float = 0.10
    near_pose_rotation_deg: float = 8.0
    mesh_correspondence_distance_m: float = 0.08
    coverage_distance_m: float = 0.05
    coverage_frame_stride: int = 20
    coverage_depth_stride: int = 4
    crop_padding_fraction: float = 0.15
    crop_size_px: int = 224
    crop_out_of_bounds_fill_rgb: tuple[int, int, int] = (0, 0, 0)
    fixed_probability_threshold: float = 0.5
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20260813

    def __post_init__(self) -> None:
        if not 1 <= self.min_targets_per_scene <= self.max_targets_per_scene:
            raise ValueError("target count bounds are invalid")
        if self.max_query_frames_per_scene <= 0:
            raise ValueError("max_query_frames_per_scene must be positive")
        for name in (
            "min_query_fraction",
            "min_projection_purity",
            "min_field_surface_coverage",
            "crop_padding_fraction",
            "fixed_probability_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if (
            self.crop_size_px <= 0
            or self.bootstrap_samples <= 0
            or self.coverage_frame_stride <= 0
            or self.coverage_depth_stride <= 0
        ):
            raise ValueError("crop/bootstrap sizes must be positive")
        if len(self.crop_out_of_bounds_fill_rgb) != 3 or any(
            not 0 <= int(value) <= 255
            for value in self.crop_out_of_bounds_fill_rgb
        ):
            raise ValueError("crop fill must be an RGB byte triplet")
        if self.fixed_probability_threshold != 0.5:
            raise ValueError("v0.1 fixes the calibrated probability threshold at 0.5")


FROZEN_PROTOCOL_CONFIG = UQISProtocolConfig()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} must be a finite length-{length} vector")
    return [float(item) for item in array]


def _finite_matrix(value: Any, shape: tuple[int, int], label: str) -> list[list[float]]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} must be a finite {shape} matrix")
    return [[float(item) for item in row] for row in array]


def _opaque_query_id(
    *, salt: bytes, scene_id: str, instance_id: int, modality: QueryModality
) -> str:
    if len(salt) < 16:
        raise ValueError("query ID salt must contain at least 16 bytes")
    material = (
        salt
        + b"\0"
        + BENCHMARK_VERSION.encode("utf-8")
        + b"\0"
        + scene_id.encode("utf-8")
        + b"\0"
        + str(int(instance_id)).encode("ascii")
        + b"\0"
        + modality.value.encode("ascii")
    )
    return "uq_" + hashlib.sha256(material).hexdigest()[:32]


def _target_sort_key(record: Mapping[str, Any]) -> tuple[str, int]:
    return str(record["scene_id"]), int(record["instance_id"])


def _normalize_scene_record(source: Mapping[str, Any]) -> dict[str, Any]:
    scene_id = str(source.get("scene_id", ""))
    if not scene_id:
        raise ValueError("scene record has no scene_id")
    mesh_path = Path(str(source.get("mesh_xyz_path", ""))).resolve()
    instance_path = Path(str(source.get("mesh_instance_ids_path", ""))).resolve()
    if not mesh_path.is_file() or not instance_path.is_file():
        raise ValueError(f"{scene_id}: frozen mesh domain files are missing")
    mesh_xyz = np.load(mesh_path, allow_pickle=False)
    instance_ids = np.load(instance_path, allow_pickle=False)
    if mesh_xyz.ndim != 2 or mesh_xyz.shape[1] != 3 or not np.isfinite(mesh_xyz).all():
        raise ValueError(f"{scene_id}: mesh_xyz must be finite [V,3]")
    if instance_ids.shape != (mesh_xyz.shape[0],) or instance_ids.dtype.kind not in "iu":
        raise ValueError(f"{scene_id}: mesh instance IDs must be integer [V]")
    field_ids = [str(value) for value in source.get("field_frame_ids", [])]
    excluded = [str(value) for value in source.get("withheld_frame_ids", [])]
    query_frames = [str(value) for value in source.get("query_frame_ids", [])]
    if not field_ids or not excluded or not query_frames:
        raise ValueError(f"{scene_id}: field/query frame domains must be non-empty")
    if len(query_frames) > int(source.get("max_query_frames", len(query_frames))):
        raise ValueError(f"{scene_id}: query-frame limit was exceeded")
    if set(field_ids).intersection(excluded) or not set(query_frames).issubset(excluded):
        raise ValueError(f"{scene_id}: field/query exclusion leakage")
    return {
        "scene_id": scene_id,
        "mesh_xyz_path": str(mesh_path),
        "mesh_xyz_sha256": sha256_file(mesh_path),
        "mesh_instance_ids_path": str(instance_path),
        "mesh_instance_ids_sha256": sha256_file(instance_path),
        "mesh_vertices": int(mesh_xyz.shape[0]),
        "query_frame_ids": query_frames,
        "query_frame_set_sha256": canonical_json_sha256(sorted(query_frames)),
        "withheld_frame_ids": excluded,
        "withheld_frame_set_sha256": canonical_json_sha256(sorted(excluded)),
        "field_frame_ids": field_ids,
        "field_frame_manifest_sha256": canonical_json_sha256(field_ids),
    }


def _validate_target(
    source: Mapping[str, Any],
    scene: Mapping[str, Any],
    config: UQISProtocolConfig,
) -> dict[str, Any]:
    scene_id = str(source.get("scene_id", ""))
    instance_id = int(source.get("instance_id", 0))
    if scene_id != scene["scene_id"] or instance_id <= 0:
        raise ValueError("target scene/instance identity is invalid")
    query_frame_id = str(source.get("query_frame_id", ""))
    if query_frame_id not in set(scene["query_frame_ids"]):
        raise ValueError(f"{scene_id}/i{instance_id}: unknown query frame")
    expression = str(source.get("expression", "")).strip()
    if not expression or not bool(source.get("expression_view_independent", False)):
        raise ValueError(f"{scene_id}/i{instance_id}: main text must be view-independent")
    crop_path = Path(str(source.get("crop_rgb_path", ""))).resolve()
    if not crop_path.is_file():
        raise ValueError(f"{scene_id}/i{instance_id}: image crop is missing")
    try:
        with Image.open(crop_path) as crop_image:
            crop_size = crop_image.size
            crop_mode = crop_image.mode
    except OSError as error:
        raise ValueError(f"{scene_id}/i{instance_id}: image crop is unreadable") from error
    if crop_size != (config.crop_size_px, config.crop_size_px) or crop_mode != "RGB":
        raise ValueError(
            f"{scene_id}/i{instance_id}: image crop must be "
            f"{config.crop_size_px}x{config.crop_size_px} RGB"
        )
    crop_hash = sha256_file(crop_path)
    declared_crop_hash = source.get("crop_rgb_sha256")
    if declared_crop_hash is not None and str(declared_crop_hash) != crop_hash:
        raise ValueError(f"{scene_id}/i{instance_id}: image crop hash mismatch")
    camera_to_world = _finite_matrix(
        source.get("camera_to_world"), (4, 4), "camera_to_world"
    )
    camera_intrinsics = _finite_matrix(
        source.get("camera_intrinsics"), (3, 3), "camera_intrinsics"
    )
    raster_size = [int(value) for value in source.get("raster_size", [])]
    if len(raster_size) != 2 or min(raster_size) <= 0:
        raise ValueError("raster_size must be [width,height]")
    pixel = [int(value) for value in source.get("positive_pixel_uv", [])]
    if len(pixel) != 2 or not (0 <= pixel[0] < raster_size[0] and 0 <= pixel[1] < raster_size[1]):
        raise ValueError("positive 2-D pixel lies outside the prompt raster")
    point = _finite_vector(source.get("point_world_xyz"), 3, "point_world_xyz")
    depth_m = float(source.get("click_depth_m", 0.0))
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("click_depth_m must be finite and positive")
    expected = np.asarray(camera_to_world, dtype=np.float64) @ np.array(
        [
            (pixel[0] - camera_intrinsics[0][2]) * depth_m / camera_intrinsics[0][0],
            (pixel[1] - camera_intrinsics[1][2]) * depth_m / camera_intrinsics[1][1],
            depth_m,
            1.0,
        ],
        dtype=np.float64,
    )
    if not np.allclose(expected[:3], point, atol=1e-5, rtol=0.0):
        raise ValueError(
            f"{scene_id}/i{instance_id}: 2-D and 3-D prompts are not the same surface point"
        )
    distractors = sorted({int(value) for value in source.get("same_class_distractor_instance_ids", [])})
    if instance_id in distractors or any(value <= 0 for value in distractors):
        raise ValueError("same-class distractor IDs are invalid")
    size_bucket = str(source.get("size_bucket", ""))
    if size_bucket not in {"small", "medium", "large"}:
        raise ValueError("size_bucket must be small, medium, or large")
    class_id = int(source.get("nyu40_class_id", 0))
    raw_semantic_label = str(
        source.get("raw_semantic_label", source.get("instance_label", f"nyu40:{class_id}"))
    ).strip().lower()
    mesh_vertex_count = int(source.get("mesh_vertex_count", 0))
    projection_pixels = int(source.get("projection_pixels", 0))
    projection_fraction = float(source.get("projection_fraction", 0.0))
    projection_purity = float(source.get("projection_purity", 0.0))
    field_surface_coverage = float(source.get("field_surface_coverage", 0.0))
    field_visibility_count = int(source.get("field_visibility_count", 0))
    if not 1 <= class_id <= 40:
        raise ValueError(f"{scene_id}/i{instance_id}: NYU40 class is invalid")
    if not raw_semantic_label:
        raise ValueError(f"{scene_id}/i{instance_id}: raw semantic label is missing")
    if mesh_vertex_count < config.min_mesh_vertices:
        raise ValueError(f"{scene_id}/i{instance_id}: target mesh is too small")
    if not (
        projection_pixels >= config.min_query_pixels
        or projection_fraction >= config.min_query_fraction
    ):
        raise ValueError(f"{scene_id}/i{instance_id}: query projection is too small")
    if not np.isfinite(projection_purity) or projection_purity < config.min_projection_purity:
        raise ValueError(f"{scene_id}/i{instance_id}: projection purity is too low")
    if (
        not np.isfinite(field_surface_coverage)
        or field_surface_coverage < config.min_field_surface_coverage
    ):
        raise ValueError(f"{scene_id}/i{instance_id}: field surface coverage is too low")
    if field_visibility_count < config.min_field_visibility_count:
        raise ValueError(f"{scene_id}/i{instance_id}: field visibility is too low")
    return {
        "scene_id": scene_id,
        "instance_id": instance_id,
        "nyu40_class_id": class_id,
        "raw_semantic_label": raw_semantic_label,
        "mesh_vertex_count": mesh_vertex_count,
        "size_bucket": size_bucket,
        "same_class_distractor_instance_ids": distractors,
        "query_frame_id": query_frame_id,
        "expression": expression,
        "expression_annotation_id": str(source.get("expression_annotation_id", "")),
        "expression_source": str(source.get("expression_source", "nr3d")),
        "expression_view_independent": True,
        "crop_rgb_path": str(crop_path),
        "crop_rgb_sha256": crop_hash,
        "camera_to_world": camera_to_world,
        "camera_intrinsics": camera_intrinsics,
        "raster_size": raster_size,
        "positive_pixel_uv": pixel,
        "click_depth_m": depth_m,
        "point_world_xyz": point,
        "projection_pixels": projection_pixels,
        "projection_fraction": projection_fraction,
        "projection_purity": projection_purity,
        "field_surface_coverage": field_surface_coverage,
        "field_visibility_count": field_visibility_count,
    }


def freeze_release(
    scene_records: Sequence[Mapping[str, Any]],
    target_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    split_role: str,
    query_id_salt: bytes,
    config: UQISProtocolConfig = UQISProtocolConfig(),
    allow_incomplete_pilot: bool = False,
) -> dict[str, Any]:
    """Freeze a result-ineligible pilot harness and bind it by hash.

    ``query_id_salt`` remains evaluator-private.  The public release stores
    only its commitment, so query IDs are reproducible by the release owner
    without exposing a target-pairing oracle to methods.

    Formal dev/test freezing is intentionally unavailable in this initial
    implementation.  It requires an end-to-end constructor that recomputes
    target eligibility and frame exclusion from content-bound ScanNet and
    ReferIt3D assets, plus mapping/runtime receipts.  Refusing those split
    roles prevents hand-authored pilot records from being laundered into a
    benchmark row with the same version string.
    """

    if split_role not in {"dev", "test", "pilot"}:
        raise ValueError("split_role must be dev, test, or pilot")
    if split_role != "pilot":
        if config != FROZEN_PROTOCOL_CONFIG:
            raise ValueError("formal UQIS releases require the frozen v0.1 config")
        requested_scenes = tuple(sorted(str(row.get("scene_id", "")) for row in scene_records))
        if split_role == "test" and requested_scenes != tuple(
            sorted(PREREGISTERED_TEST_SCENES)
        ):
            raise ValueError(
                "formal UQIS test scenes must match the preregistered candidate set"
            )
        raise RuntimeError(
            "formal UQIS freezing is disabled until the official-asset constructor, "
            "replacement ledger, exclusion receipt, and mapping receipt are implemented"
        )
    if split_role == "pilot" and not allow_incomplete_pilot:
        raise ValueError("pilot releases require allow_incomplete_pilot=True")
    normalized_scenes = [_normalize_scene_record(record) for record in scene_records]
    scene_by_id = {record["scene_id"]: record for record in normalized_scenes}
    if len(scene_by_id) != len(normalized_scenes):
        raise ValueError("scene IDs must be unique")
    normalized_targets = [
        _validate_target(
            record,
            scene_by_id[str(record.get("scene_id", ""))],
            config,
        )
        for record in sorted(target_records, key=_target_sort_key)
    ]
    identities = [(record["scene_id"], record["instance_id"]) for record in normalized_targets]
    if len(set(identities)) != len(identities):
        raise ValueError("a ScanNet instance may appear only once in a release")
    targets_by_scene: dict[str, list[dict[str, Any]]] = {
        scene_id: [] for scene_id in scene_by_id
    }
    for record in normalized_targets:
        targets_by_scene[record["scene_id"]].append(record)
    if set(targets_by_scene) != {record["scene_id"] for record in normalized_targets}:
        raise ValueError("every scene must contain at least one target")
    readiness_errors: list[str] = []
    for scene_id, rows in targets_by_scene.items():
        if not config.min_targets_per_scene <= len(rows) <= config.max_targets_per_scene:
            readiness_errors.append(f"{scene_id}: target count {len(rows)}")
        if sum(bool(row["same_class_distractor_instance_ids"]) for row in rows) < config.min_same_class_targets_per_scene:
            readiness_errors.append(f"{scene_id}: too few same-class targets")
        if len({row["raw_semantic_label"] for row in rows}) < config.min_semantic_categories_per_scene:
            readiness_errors.append(f"{scene_id}: too few semantic categories")
        class_counts = {
            class_id: sum(row["raw_semantic_label"] == class_id for row in rows)
            for class_id in {row["raw_semantic_label"] for row in rows}
        }
        if max(class_counts.values(), default=0) > 2:
            readiness_errors.append(f"{scene_id}: more than two targets share a class")
        if len(scene_by_id[scene_id]["query_frame_ids"]) > config.max_query_frames_per_scene:
            readiness_errors.append(f"{scene_id}: too many query frames")
    if split_role != "pilot" and readiness_errors:
        raise ValueError("formal release is incomplete: " + "; ".join(readiness_errors))
    if split_role == "test" and len(normalized_scenes) != 9:
        raise ValueError("formal ScanNet-UQIS-9 test releases require exactly 9 scenes")
    if split_role == "dev" and len(normalized_scenes) != 4:
        raise ValueError("formal ScanNet-UQIS dev releases require exactly 4 scenes")

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty release: {root}")
    root.mkdir(parents=True, exist_ok=True)
    common = {
        "benchmark_version": BENCHMARK_VERSION,
        "split_role": split_role,
        "release_tier": "pilot_harness",
        "formal_benchmark_eligible": False,
        "protocol_config": asdict(config),
        "protocol_config_sha256": canonical_json_sha256(asdict(config)),
        "query_id_salt_sha256": hashlib.sha256(query_id_salt).hexdigest(),
    }
    public_domains = [
        {
            "scene_id": scene["scene_id"],
            "mesh_xyz_path": scene["mesh_xyz_path"],
            "mesh_xyz_sha256": scene["mesh_xyz_sha256"],
            "mesh_vertices": scene["mesh_vertices"],
        }
        for scene in normalized_scenes
    ]
    private_domains = [
        {
            **public,
            "mesh_instance_ids_path": scene["mesh_instance_ids_path"],
            "mesh_instance_ids_sha256": scene["mesh_instance_ids_sha256"],
        }
        for public, scene in zip(public_domains, normalized_scenes)
    ]
    query_rows: dict[QueryModality, list[dict[str, Any]]] = {
        modality: [] for modality in QueryModality
    }
    evaluator_targets: list[dict[str, Any]] = []
    for target_index, target in enumerate(normalized_targets):
        ids = {
            modality: _opaque_query_id(
                salt=query_id_salt,
                scene_id=target["scene_id"],
                instance_id=target["instance_id"],
                modality=modality,
            )
            for modality in QueryModality
        }
        base = {"scene_id": target["scene_id"]}
        sanitized_crop_path = root / "method_assets" / "image" / f"{ids[QueryModality.IMAGE]}.png"
        sanitized_crop_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(target["crop_rgb_path"]) as source_crop:
            # Re-encoding strips source metadata and the opaque basename hides
            # frame/instance/bbox conventions from the method workspace.
            source_crop.convert("RGB").save(
                sanitized_crop_path,
                format="PNG",
                optimize=False,
                compress_level=9,
            )
        sanitized_crop_hash = sha256_file(sanitized_crop_path)
        query_rows[QueryModality.TEXT].append(
            {
                **base,
                "query_id": ids[QueryModality.TEXT],
                "modality": QueryModality.TEXT.value,
                "expression": target["expression"],
                "available_method_inputs": ["scene_id", "expression"],
            }
        )
        query_rows[QueryModality.IMAGE].append(
            {
                **base,
                "query_id": ids[QueryModality.IMAGE],
                "modality": QueryModality.IMAGE.value,
                "crop_rgb_path": str(sanitized_crop_path),
                "crop_rgb_sha256": sanitized_crop_hash,
                "available_method_inputs": ["scene_id", "crop_rgb"],
            }
        )
        query_rows[QueryModality.POINT_2D].append(
            {
                **base,
                "query_id": ids[QueryModality.POINT_2D],
                "modality": QueryModality.POINT_2D.value,
                "camera_to_world": target["camera_to_world"],
                "camera_intrinsics": target["camera_intrinsics"],
                "raster_size": target["raster_size"],
                "positive_pixel_uv": target["positive_pixel_uv"],
                "available_method_inputs": [
                    "scene_id",
                    "camera_to_world",
                    "camera_intrinsics",
                    "raster_size",
                    "positive_pixel_uv",
                ],
            }
        )
        query_rows[QueryModality.POINT_3D].append(
            {
                **base,
                "query_id": ids[QueryModality.POINT_3D],
                "modality": QueryModality.POINT_3D.value,
                "point_world_xyz": target["point_world_xyz"],
                "available_method_inputs": ["scene_id", "point_world_xyz"],
            }
        )
        evaluator_targets.append(
            {
                "target_id": f"target_{target_index:04d}",
                "scene_id": target["scene_id"],
                "instance_id": target["instance_id"],
                "nyu40_class_id": target["nyu40_class_id"],
                "raw_semantic_label": target["raw_semantic_label"],
                "mesh_vertex_count": target["mesh_vertex_count"],
                "size_bucket": target["size_bucket"],
                "same_class_distractor_instance_ids": target[
                    "same_class_distractor_instance_ids"
                ],
                "query_frame_id": target["query_frame_id"],
                "click_depth_m": target["click_depth_m"],
                "point_world_xyz": target["point_world_xyz"],
                "expression_annotation_id": target["expression_annotation_id"],
                "expression_source": target["expression_source"],
                "queries": {modality.value: ids[modality] for modality in QueryModality},
            }
        )

    scene_manifest = {
        **common,
        "visibility": "public_method_domain",
        "scene_domains": public_domains,
    }
    public_scene_rows = []
    for scene_id in sorted(targets_by_scene):
        rows = targets_by_scene[scene_id]
        public_scene_rows.append(
            {
                "scene_id": scene_id,
                "target_count": len(rows),
                "same_class_target_count": sum(
                    bool(row["same_class_distractor_instance_ids"]) for row in rows
                ),
                "semantic_category_count": len({row["raw_semantic_label"] for row in rows}),
                "size_bucket_counts": {
                    bucket: sum(row["size_bucket"] == bucket for row in rows)
                    for bucket in ("small", "medium", "large")
                },
            }
        )
    target_public = {
        **common,
        "visibility": "public_aggregate_only",
        "target_count": len(normalized_targets),
        "scenes": public_scene_rows,
        "readiness_errors": readiness_errors,
    }
    target_evaluator = {
        **common,
        "visibility": "evaluator_only",
        "scene_domains": private_domains,
        "targets": evaluator_targets,
    }
    field_exclusion = {
        **common,
        "visibility": "mapping_input",
        "scenes": [
            {
                "scene_id": scene["scene_id"],
                "query_frame_count": len(scene["query_frame_ids"]),
                "query_frame_set_sha256": scene["query_frame_set_sha256"],
                "withheld_frame_ids": scene["withheld_frame_ids"],
                "withheld_frame_set_sha256": scene["withheld_frame_set_sha256"],
                "field_frame_ids": scene["field_frame_ids"],
                "field_frame_manifest_sha256": scene["field_frame_manifest_sha256"],
            }
            for scene in normalized_scenes
        ],
    }
    payloads: dict[str, Mapping[str, Any]] = {
        "scene_manifest.json": scene_manifest,
        "target_manifest.public.json": target_public,
        "target_manifest.evaluator.json": target_evaluator,
        "field_exclusion_manifest.json": field_exclusion,
    }
    for modality in QueryModality:
        payloads[QUERY_MANIFEST_NAMES[modality.value]] = {
            **common,
            "visibility": "method_input",
            "modality": modality.value,
            "prediction_domain": PREDICTION_DOMAIN,
            "scene_domains": public_domains,
            "queries": sorted(query_rows[modality], key=lambda row: row["query_id"]),
        }
    for name, payload in payloads.items():
        _write_json(root / name, payload)
    audit = audit_payloads(payloads, check_files=True)
    if not audit["valid"]:
        raise RuntimeError("frozen release failed audit: " + "; ".join(audit["errors"]))
    hashes = {name: sha256_file(root / name) for name in RELEASE_MANIFEST_NAMES}
    release = {
        **common,
        "status": "pilot_harness",
        "formal_benchmark_eligible": False,
        "formal_release_implemented": FORMAL_RELEASE_IMPLEMENTED,
        "scene_count": len(normalized_scenes),
        "target_count": len(normalized_targets),
        "query_count": len(normalized_targets) * len(QueryModality),
        "manifest_sha256": hashes,
        "audit": audit,
    }
    _write_json(root / "release.json", release)
    return release


_QUERY_EXACT_KEYS = {
    QueryModality.TEXT: {
        "query_id",
        "scene_id",
        "modality",
        "expression",
        "available_method_inputs",
    },
    QueryModality.IMAGE: {
        "query_id",
        "scene_id",
        "modality",
        "crop_rgb_path",
        "crop_rgb_sha256",
        "available_method_inputs",
    },
    QueryModality.POINT_2D: {
        "query_id",
        "scene_id",
        "modality",
        "camera_to_world",
        "camera_intrinsics",
        "raster_size",
        "positive_pixel_uv",
        "available_method_inputs",
    },
    QueryModality.POINT_3D: {
        "query_id",
        "scene_id",
        "modality",
        "point_world_xyz",
        "available_method_inputs",
    },
}

_COMMON_MANIFEST_KEYS = {
    "benchmark_version",
    "split_role",
    "release_tier",
    "formal_benchmark_eligible",
    "protocol_config",
    "protocol_config_sha256",
    "query_id_salt_sha256",
}
_QUERY_MANIFEST_EXACT_KEYS = _COMMON_MANIFEST_KEYS | {
    "visibility",
    "modality",
    "prediction_domain",
    "scene_domains",
    "queries",
}
_SCENE_MANIFEST_EXACT_KEYS = _COMMON_MANIFEST_KEYS | {
    "visibility",
    "scene_domains",
}
_TARGET_PUBLIC_EXACT_KEYS = _COMMON_MANIFEST_KEYS | {
    "visibility",
    "target_count",
    "scenes",
    "readiness_errors",
}
_TARGET_PUBLIC_SCENE_EXACT_KEYS = {
    "scene_id",
    "target_count",
    "same_class_target_count",
    "semantic_category_count",
    "size_bucket_counts",
}
_TARGET_EVALUATOR_EXACT_KEYS = _COMMON_MANIFEST_KEYS | {
    "visibility",
    "scene_domains",
    "targets",
}
_EVALUATOR_TARGET_EXACT_KEYS = {
    "target_id",
    "scene_id",
    "instance_id",
    "nyu40_class_id",
    "raw_semantic_label",
    "mesh_vertex_count",
    "size_bucket",
    "same_class_distractor_instance_ids",
    "query_frame_id",
    "click_depth_m",
    "point_world_xyz",
    "expression_annotation_id",
    "expression_source",
    "queries",
}
_PRIVATE_SCENE_DOMAIN_EXACT_KEYS = {
    "scene_id",
    "mesh_xyz_path",
    "mesh_xyz_sha256",
    "mesh_vertices",
    "mesh_instance_ids_path",
    "mesh_instance_ids_sha256",
}
_FIELD_EXCLUSION_EXACT_KEYS = _COMMON_MANIFEST_KEYS | {
    "visibility",
    "scenes",
}
_FIELD_EXCLUSION_SCENE_EXACT_KEYS = {
    "scene_id",
    "query_frame_count",
    "query_frame_set_sha256",
    "withheld_frame_ids",
    "withheld_frame_set_sha256",
    "field_frame_ids",
    "field_frame_manifest_sha256",
}
_RELEASE_EXACT_KEYS = _COMMON_MANIFEST_KEYS | {
    "status",
    "formal_release_implemented",
    "scene_count",
    "target_count",
    "query_count",
    "manifest_sha256",
    "audit",
}
_AVAILABLE_METHOD_INPUTS = {
    QueryModality.TEXT: ["scene_id", "expression"],
    QueryModality.IMAGE: ["scene_id", "crop_rgb"],
    QueryModality.POINT_2D: [
        "scene_id",
        "camera_to_world",
        "camera_intrinsics",
        "raster_size",
        "positive_pixel_uv",
    ],
    QueryModality.POINT_3D: ["scene_id", "point_world_xyz"],
}


def audit_payloads(
    payloads: Mapping[str, Mapping[str, Any]], *, check_files: bool
) -> dict[str, Any]:
    """Fail closed on leaked fields, broken pairing, or changed spatial domains."""

    errors: list[str] = []
    required = set(RELEASE_MANIFEST_NAMES)
    missing = sorted(required - set(payloads))
    if missing:
        return {"valid": False, "errors": [f"missing manifests: {missing}"]}
    expected_common: dict[str, Any] | None = None
    for name in required:
        payload = payloads[name]
        if payload.get("benchmark_version") != BENCHMARK_VERSION:
            errors.append(f"{name}: benchmark version mismatch")
        common = {key: payload.get(key) for key in _COMMON_MANIFEST_KEYS}
        if expected_common is None:
            expected_common = common
        elif common != expected_common:
            errors.append(f"{name}: common release identity differs")
        config = payload.get("protocol_config")
        if not isinstance(config, Mapping) or payload.get(
            "protocol_config_sha256"
        ) != canonical_json_sha256(config):
            errors.append(f"{name}: protocol-config digest mismatch")
        if payload.get("release_tier") != "pilot_harness":
            errors.append(f"{name}: unsupported release tier")
        if payload.get("formal_benchmark_eligible") is not False:
            errors.append(f"{name}: pilot manifest claims formal eligibility")
        try:
            _require_sha256(payload.get("query_id_salt_sha256"), "query ID salt hash")
        except ValueError as error:
            errors.append(f"{name}: {error}")
    scene_payload = payloads["scene_manifest.json"]
    if set(scene_payload) != _SCENE_MANIFEST_EXACT_KEYS:
        errors.append("scene_manifest: top-level fields changed")
    if scene_payload.get("visibility") != "public_method_domain":
        errors.append("scene_manifest: visibility changed")
    public_domains = scene_payload.get("scene_domains", [])
    scene_ids = {str(row.get("scene_id", "")) for row in public_domains}
    if not scene_ids or "" in scene_ids or len(scene_ids) != len(public_domains):
        errors.append("scene_manifest: invalid/duplicate scene IDs")
    for domain in public_domains:
        if set(domain) != {
            "scene_id",
            "mesh_xyz_path",
            "mesh_xyz_sha256",
            "mesh_vertices",
        }:
            errors.append(f"{domain.get('scene_id')}: public scene-domain fields changed")
            continue
        if check_files:
            path = Path(str(domain["mesh_xyz_path"]))
            if not path.is_file() or sha256_file(path) != domain["mesh_xyz_sha256"]:
                errors.append(f"{domain['scene_id']}: public mesh-domain hash mismatch")
    query_ids: set[str] = set()
    query_by_modality: dict[QueryModality, set[str]] = {}
    for modality in QueryModality:
        name = QUERY_MANIFEST_NAMES[modality.value]
        payload = payloads[name]
        if set(payload) != _QUERY_MANIFEST_EXACT_KEYS:
            errors.append(f"{name}: top-level fields changed")
        if payload.get("visibility") != "method_input" or payload.get("modality") != modality.value:
            errors.append(f"{name}: modality/visibility mismatch")
        if payload.get("prediction_domain") != PREDICTION_DOMAIN:
            errors.append(f"{name}: prediction domain changed")
        if payload.get("scene_domains") != public_domains:
            errors.append(f"{name}: scene domains diverge")
        current: set[str] = set()
        for row in payload.get("queries", []):
            query_id = str(row.get("query_id", ""))
            if not query_id.startswith("uq_") or len(query_id) != 35:
                errors.append(f"{name}: non-opaque query ID {query_id!r}")
            if query_id in query_ids:
                errors.append(f"{name}: query ID reused across modalities")
            query_ids.add(query_id)
            current.add(query_id)
            if set(row) != _QUERY_EXACT_KEYS[modality]:
                errors.append(f"{query_id}: method-visible fields changed")
            if str(row.get("scene_id", "")) not in scene_ids:
                errors.append(f"{query_id}: unknown scene")
            if row.get("modality") != modality.value:
                errors.append(f"{query_id}: modality mismatch")
            if row.get("available_method_inputs") != _AVAILABLE_METHOD_INPUTS[modality]:
                errors.append(f"{query_id}: available method inputs changed")
            if modality is QueryModality.IMAGE and check_files:
                path = Path(str(row.get("crop_rgb_path", "")))
                if not path.is_file() or sha256_file(path) != row.get("crop_rgb_sha256"):
                    errors.append(f"{query_id}: image crop hash mismatch")
            if modality is QueryModality.POINT_2D:
                try:
                    _finite_matrix(row.get("camera_to_world"), (4, 4), "camera_to_world")
                    _finite_matrix(row.get("camera_intrinsics"), (3, 3), "camera_intrinsics")
                except ValueError as error:
                    errors.append(f"{query_id}: {error}")
                serialized = json.dumps(row, sort_keys=True).lower()
                if any(token in serialized for token in ("rendered_rgb", "rgb_path", "image_embedding", "target_mask")):
                    errors.append(f"{query_id}: strict 2-D prompt leaked image/mask evidence")
        if len(current) != len(payload.get("queries", [])):
            errors.append(f"{name}: duplicate query IDs")
        query_by_modality[modality] = current
    counts = {modality: len(values) for modality, values in query_by_modality.items()}
    if len(set(counts.values())) != 1:
        errors.append(f"modality query counts differ: {counts}")

    evaluator = payloads["target_manifest.evaluator.json"]
    if set(evaluator) != _TARGET_EVALUATOR_EXACT_KEYS:
        errors.append("target evaluator manifest: top-level fields changed")
    if evaluator.get("visibility") != "evaluator_only":
        errors.append("target evaluator manifest is not evaluator-only")
    private_domains = evaluator.get("scene_domains", [])
    for domain in private_domains:
        if set(domain) != _PRIVATE_SCENE_DOMAIN_EXACT_KEYS:
            errors.append(
                f"{domain.get('scene_id')}: evaluator scene-domain fields changed"
            )
            continue
        public = {key: value for key, value in domain.items() if not key.startswith("mesh_instance_ids")}
        if public not in public_domains:
            errors.append(f"{domain.get('scene_id')}: evaluator/public mesh domains differ")
        if check_files:
            path = Path(str(domain.get("mesh_instance_ids_path", "")))
            if not path.is_file() or sha256_file(path) != domain.get("mesh_instance_ids_sha256"):
                errors.append(f"{domain.get('scene_id')}: private instance-domain hash mismatch")
    paired: set[str] = set()
    target_ids: set[str] = set()
    for target in evaluator.get("targets", []):
        if set(target) != _EVALUATOR_TARGET_EXACT_KEYS:
            errors.append(
                f"{target.get('target_id')}: evaluator target fields changed"
            )
        target_id = str(target.get("target_id", ""))
        if not target_id or target_id in target_ids:
            errors.append(f"duplicate/missing evaluator target ID {target_id!r}")
        target_ids.add(target_id)
        queries = target.get("queries", {})
        if set(queries) != {modality.value for modality in QueryModality}:
            errors.append(f"{target_id}: incomplete modality pairing")
            continue
        for modality in QueryModality:
            query_id = str(queries[modality.value])
            if query_id not in query_by_modality[modality]:
                errors.append(f"{target_id}: unknown {modality.value} query")
            if query_id in paired:
                errors.append(f"{target_id}: query paired more than once")
            paired.add(query_id)
    if paired != query_ids:
        errors.append("evaluator pairing does not cover the exact method query set")

    target_public = payloads["target_manifest.public.json"]
    if set(target_public) != _TARGET_PUBLIC_EXACT_KEYS:
        errors.append("target public manifest: top-level fields changed")
    if target_public.get("visibility") != "public_aggregate_only":
        errors.append("target public manifest: visibility changed")
    for row in target_public.get("scenes", []):
        if set(row) != _TARGET_PUBLIC_SCENE_EXACT_KEYS:
            errors.append(f"{row.get('scene_id')}: public target summary fields changed")
        if set(row.get("size_bucket_counts", {})) != {"small", "medium", "large"}:
            errors.append(f"{row.get('scene_id')}: size-bucket summary changed")

    exclusions = payloads["field_exclusion_manifest.json"]
    if set(exclusions) != _FIELD_EXCLUSION_EXACT_KEYS:
        errors.append("field exclusion manifest: top-level fields changed")
    if exclusions.get("visibility") != "mapping_input":
        errors.append("field exclusion manifest: visibility changed")
    exclusion_scene_ids: set[str] = set()
    for row in exclusions.get("scenes", []):
        if set(row) != _FIELD_EXCLUSION_SCENE_EXACT_KEYS:
            errors.append(f"{row.get('scene_id')}: exclusion fields changed")
        scene_id = str(row.get("scene_id", ""))
        exclusion_scene_ids.add(scene_id)
        field_ids = list(map(str, row.get("field_frame_ids", [])))
        withheld = list(map(str, row.get("withheld_frame_ids", [])))
        if set(field_ids).intersection(withheld):
            errors.append(f"{scene_id}: withheld frame leaked into field")
        if row.get("field_frame_manifest_sha256") != canonical_json_sha256(field_ids):
            errors.append(f"{scene_id}: field-frame digest mismatch")
        if row.get("withheld_frame_set_sha256") != canonical_json_sha256(sorted(withheld)):
            errors.append(f"{scene_id}: withheld-frame digest mismatch")
    if exclusion_scene_ids != scene_ids:
        errors.append("field-exclusion scenes differ from scene domain")
    query_frames_by_scene: dict[str, set[str]] = {scene_id: set() for scene_id in scene_ids}
    for target in evaluator.get("targets", []):
        query_frames_by_scene[str(target.get("scene_id", ""))].add(
            str(target.get("query_frame_id", ""))
        )
    exclusion_by_scene = {
        str(row.get("scene_id")): row for row in exclusions.get("scenes", [])
    }
    for scene_id, query_frames in query_frames_by_scene.items():
        row = exclusion_by_scene.get(scene_id, {})
        withheld = set(map(str, row.get("withheld_frame_ids", [])))
        if not query_frames.issubset(withheld):
            errors.append(f"{scene_id}: a query frame is not withheld")
        if row.get("query_frame_set_sha256") != canonical_json_sha256(sorted(query_frames)):
            errors.append(f"{scene_id}: private/public query-frame commitment mismatch")
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "scene_count": len(scene_ids),
        "target_count": len(target_ids),
        "query_count": len(query_ids),
        "valid": not errors,
        "errors": errors,
    }


def audit_release(root: str | Path, *, check_files: bool = True) -> dict[str, Any]:
    release_root = Path(root).resolve()
    release_path = release_root / "release.json"
    if not release_path.is_file():
        return {"valid": False, "errors": [f"missing release.json: {release_path}"]}
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
        payloads = {
            name: json.loads((release_root / name).read_text(encoding="utf-8"))
            for name in RELEASE_MANIFEST_NAMES
        }
    except (OSError, json.JSONDecodeError) as error:
        return {"valid": False, "errors": [f"unreadable release: {error}"]}
    report = audit_payloads(payloads, check_files=check_files)
    payload_audit = {**report, "errors": list(report.get("errors", []))}
    if set(release) != _RELEASE_EXACT_KEYS:
        report["errors"].append("release.json top-level fields changed")
        report["valid"] = False
    manifest_common = {
        key: payloads["scene_manifest.json"].get(key)
        for key in _COMMON_MANIFEST_KEYS
    }
    release_common = {key: release.get(key) for key in _COMMON_MANIFEST_KEYS}
    if release_common != manifest_common:
        report["errors"].append("release/common manifest identity differs")
        report["valid"] = False
    release_config = release.get("protocol_config")
    if not isinstance(release_config, Mapping) or release.get(
        "protocol_config_sha256"
    ) != canonical_json_sha256(release_config):
        report["errors"].append("release protocol-config digest mismatch")
        report["valid"] = False
    for count_key in ("scene_count", "target_count", "query_count"):
        if release.get(count_key) != payload_audit.get(count_key):
            report["errors"].append(f"release {count_key} differs from manifests")
            report["valid"] = False
    if release.get("audit") != payload_audit:
        report["errors"].append("release embedded audit differs from fresh audit")
        report["valid"] = False
    if (
        release.get("status") != "pilot_harness"
        or release.get("release_tier") != "pilot_harness"
        or release.get("formal_benchmark_eligible") is not False
        or release.get("formal_release_implemented") is not False
    ):
        report["errors"].append("release eligibility/status changed")
        report["valid"] = False
    if not check_files:
        report["errors"].append(
            "asset hashes were skipped; this diagnostic audit is not valid"
        )
        report["valid"] = False
    expected = release.get("manifest_sha256", {})
    actual = {
        name: sha256_file(release_root / name)
        for name in RELEASE_MANIFEST_NAMES
        if (release_root / name).is_file()
    }
    if expected != actual:
        report["errors"].append("release manifest hashes do not match")
        report["valid"] = False
    report["release_hashes_match"] = expected == actual
    report["formal_benchmark_eligible"] = False
    return report
