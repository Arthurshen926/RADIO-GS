"""ScanNet preparation for the scene-disjoint completion oracle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from radio_gs.v4.carrier import Camera, MeshCarrier, SurfaceVoxelCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.evaluation.object_oracle_gate import _load_instance_labels
from radio_gs.v4.registration.evidence_fusion import fuse_evidence_tables


CACHE_SCHEMA = "radio_gs.surface_object_memory_v4.scannet_completion_scene.v4"
RADIO_BACKBONE_DIMENSION = 1280
RADIO_PROJECTION_DIMENSION = 64
RADIO_CHECKPOINT_SHA256 = "bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9"
RADIO_PROJECTION_SALT = b"radio_gs_v4_fixed_rademacher_jl64_v1"
RADIO_PROJECTION_SHA256 = "9c38ec18c82219e404fdecc4ee65ebefa7f7e803ac92b4e0ce5ae81c78fef663"
MASK_DROPOUT_KEEP_PROBABILITY = 0.5
MASK_DROPOUT_SALT = "radio_gs_v4_scannet_source_object_view_mask_dropout_v1"


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _radio_feature_layout() -> list[str]:
    return [
        "source_rgb_r",
        "source_rgb_g",
        "source_rgb_b",
        "source_rgb_available",
        *[
            f"source_radio_fixed_jl64_{index:02d}"
            for index in range(RADIO_PROJECTION_DIMENSION)
        ],
        "normal_x",
        "normal_y",
        "normal_z",
    ]


def _radio_projection_matrix() -> torch.Tensor:
    """Return the preregistered, data-independent RADIO projection."""

    values = np.empty(
        (RADIO_BACKBONE_DIMENSION, RADIO_PROJECTION_DIMENSION), dtype=np.float32
    )
    scale = 1.0 / math.sqrt(RADIO_PROJECTION_DIMENSION)
    for row in range(RADIO_BACKBONE_DIMENSION):
        row_bytes = int(row).to_bytes(4, "little", signed=False)
        for column in range(RADIO_PROJECTION_DIMENSION):
            digest = hashlib.sha256(
                RADIO_PROJECTION_SALT
                + row_bytes
                + int(column).to_bytes(4, "little", signed=False)
            ).digest()
            values[row, column] = scale if digest[0] & 1 else -scale
    observed = hashlib.sha256(values.astype("<f4", copy=False).tobytes(order="C")).hexdigest()
    if observed != RADIO_PROJECTION_SHA256:
        raise RuntimeError("fixed RADIO projection differs from its preregistered digest")
    return torch.from_numpy(values)


def _mask_dropout_record(scene_id: str, frame_key: str, object_id: int) -> dict[str, Any]:
    message = "\0".join(
        (str(scene_id), str(frame_key), str(int(object_id)), MASK_DROPOUT_SALT)
    ).encode("utf-8")
    digest = hashlib.sha256(message).hexdigest()
    # p=0.5 is an exact high-bit decision and therefore needs no floating-point
    # comparison or platform-specific random-number generator.
    kept = int(digest[:16], 16) < (1 << 63)
    return {
        "scene_id": str(scene_id),
        "frame_id": str(frame_key),
        "object_id": int(object_id),
        "sha256": digest,
        "kept": kept,
    }


def _positive_mask_support(
    carrier: SurfaceVoxelCarrier,
    cameras: list[Camera],
    *,
    scene_id: str,
    element_object_id: torch.Tensor,
    object_ids: list[int],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Simulate sparse, perfect-precision positive object masks from source views."""

    element_object_id = torch.as_tensor(element_object_id, dtype=torch.long).cpu()
    if element_object_id.shape != (carrier.num_elements,):
        raise ValueError("element object identities must align with the carrier")
    identifiers = list(map(int, object_ids))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("mask-dropout object identities must be unique")
    supported = torch.zeros(carrier.num_elements, dtype=torch.bool)
    records: list[dict[str, Any]] = []
    for camera in cameras:
        visible_ids = torch.unique(carrier.project(camera).element_ids)
        visible_object = element_object_id[visible_ids]
        for object_id in identifiers:
            record = _mask_dropout_record(scene_id, camera.key, object_id)
            records.append(record)
            if record["kept"]:
                supported[visible_ids[visible_object == object_id]] = True
    return supported, records


def _load_ply(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from plyfile import PlyData
    except ImportError as error:
        raise RuntimeError("ScanNet completion preparation requires plyfile") from error
    ply = PlyData.read(str(path), known_list_len={"face": {"vertex_indices": 3}})
    vertex = ply["vertex"].data
    required = ("x", "y", "z")
    if any(name not in vertex.dtype.names for name in required):
        raise ValueError("ScanNet mesh PLY lacks xyz vertex fields")
    xyz = np.stack([vertex[name] for name in ("x", "y", "z")], -1).astype(np.float32)
    faces = np.vstack(ply["face"].data["vertex_indices"]).astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("ScanNet completion expects a triangular mesh")
    triangle = xyz[faces]
    face_normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    normals = np.zeros_like(xyz)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normal)
    normal_length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(normal_length, 1e-12)
    return torch.from_numpy(xyz), torch.from_numpy(faces), torch.from_numpy(normals)


def _source_rgb_path(scene_directory: Path, camera: Camera) -> Path:
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = scene_directory / "color" / f"{camera.key}{suffix}"
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise FileNotFoundError(f"source observation RGB is missing for frame {camera.key!r}")


def _load_source_rgb(path: Path, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).copy()
    return torch.from_numpy(array / 255.0)


def _source_radio_paths(
    radio_feature_root: Path,
    scene_id: str,
    cameras: list[Camera],
    rgb_paths: list[Path],
    *,
    height: int,
    width: int,
) -> tuple[Path, list[Path], dict[str, Any]]:
    """Resolve only explicitly selected source-frame RADIO tensors."""

    if len(cameras) != len(rgb_paths) or not cameras:
        raise ValueError("source cameras and RGB paths must align before RADIO resolution")
    scene_feature_root = (radio_feature_root / scene_id).resolve(strict=True)
    manifest_path = (scene_feature_root / "frame_manifest.json").resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("scene") != scene_id:
        raise ValueError("RADIO frame manifest has the wrong scene identity")
    radio = manifest.get("radio", {})
    backbone = manifest.get("features", {}).get("backbone", {})
    if (
        not isinstance(radio, dict)
        or radio.get("version") != "c-radio_v4-h"
        or radio.get("checkpoint_sha256") != RADIO_CHECKPOINT_SHA256
        or not str(radio.get("repo_hubconf_sha256", ""))
        or list(radio.get("requested_adaptors", []))
    ):
        raise ValueError("RADIO frame manifest is not the exact frozen adaptor-free C-RADIO v4-H source")
    if backbone != {
        "subdir": "backbone",
        "dim": RADIO_BACKBONE_DIMENSION,
        "grid": [height, width],
        "dtype": "float16",
    }:
        raise ValueError("RADIO frame manifest has the wrong backbone contract")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames or manifest.get("num_frames") != len(frames):
        raise ValueError("RADIO frame manifest has an incomplete frame inventory")
    selected_keys = [camera.key for camera in cameras]
    manifest_keys = [
        Path(str(record.get("source_file", ""))).stem
        if isinstance(record, dict)
        else ""
        for record in frames
    ]
    if (
        len(set(manifest_keys)) != len(manifest_keys)
        or len(set(selected_keys)) != len(selected_keys)
        or set(manifest_keys) != set(selected_keys)
    ):
        raise ValueError("RADIO frame manifest must contain exactly the selected source cameras")
    if (
        manifest.get("radio_input_resolution_hw") != [height * 16, width * 16]
        or manifest.get("resolution_scale") != 1.0
        or manifest.get("sliding_window") is not False
    ):
        raise ValueError("RADIO source extraction raster contract differs")
    execution = manifest.get("execution")
    if (
        not isinstance(execution, dict)
        or execution.get("resume_partial") is not True
        or execution.get("resume_contract") != ".extract_resume_contract.json"
        or not str(execution.get("resume_contract_sha256", ""))
        or execution.get("committed_frame_validation")
        != "same_fd_sha256_weights_only_dtype_shape_finite_v2"
    ):
        raise ValueError("RADIO source extraction lacks strict committed-output provenance")
    bundle = manifest.get("output_bundle")
    resume_digest = str(execution["resume_contract_sha256"])
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema_version") != 1
        or bundle.get("contract") != "radio-feature-output-bundle-v1"
        or bundle.get("resume_contract_sha256") != resume_digest
        or manifest.get("output_bundle_sha256") != _canonical_json_sha256(bundle)
    ):
        raise ValueError("RADIO source output bundle contract differs")
    bundle_frames = bundle.get("frames")
    if not isinstance(bundle_frames, list) or len(bundle_frames) != len(frames):
        raise ValueError("RADIO source output bundle frame inventory differs")
    by_key: dict[str, dict[str, Any]] = {}
    saved_stems: set[str] = set()
    for record in frames:
        if not isinstance(record, dict):
            raise ValueError("RADIO frame manifest contains a malformed frame record")
        source_file = Path(str(record.get("source_file", "")))
        key = source_file.stem
        saved_stem = str(record.get("saved_stem", ""))
        if (
            not key
            or source_file.name != str(record.get("source_file", ""))
            or key in by_key
            or not saved_stem.startswith("rgb_")
            or Path(saved_stem).name != saved_stem
            or saved_stem in saved_stems
        ):
            raise ValueError("RADIO frame manifest frame identities are unsafe or duplicated")
        saved_suffix = saved_stem.removeprefix("rgb_")
        if key.isdigit():
            if not saved_suffix.isdigit() or int(saved_suffix) != int(key):
                raise ValueError("RADIO numeric saved stem differs from its source camera")
        elif saved_suffix != key:
            raise ValueError("RADIO saved stem differs from its source camera")
        by_key[key] = record
        saved_stems.add(saved_stem)
    bundle_by_key: dict[str, dict[str, Any]] = {}
    for bundle_record in bundle_frames:
        if not isinstance(bundle_record, dict) or not isinstance(
            bundle_record.get("frame"), dict
        ):
            raise ValueError("RADIO source output bundle contains a malformed frame")
        bundle_key = Path(str(bundle_record["frame"].get("source_file", ""))).stem
        if bundle_key in bundle_by_key or bundle_record["frame"] != by_key.get(bundle_key):
            raise ValueError("RADIO source output bundle frame inventory differs")
        bundle_by_key[bundle_key] = bundle_record
    if set(bundle_by_key) != set(by_key):
        raise ValueError("RADIO source output bundle frame inventory differs")
    feature_directory = (scene_feature_root / "backbone").resolve(strict=True)
    paths: list[Path] = []
    for camera, rgb_path in zip(cameras, rgb_paths):
        record = by_key.get(camera.key)
        if record is None:
            raise ValueError(f"RADIO manifest omits exact source camera {camera.key!r}")
        saved_stem = str(record["saved_stem"])
        declared_source_sha = str(record.get("source_sha256", ""))
        if not declared_source_sha or declared_source_sha != sha256_file(rgb_path):
            raise ValueError("RADIO source image digest differs from the sealed source RGB")
        path = (feature_directory / f"{saved_stem}.pt").resolve(strict=True)
        if path.parent != feature_directory:
            raise ValueError("RADIO source tensor escaped the manifest backbone directory")
        bundle_record = bundle_by_key[camera.key]
        if (
            set(bundle_record) != {
                "frame", "marker_relative_path", "marker_sha256", "feature_signature", "tensors"
            }
            or bundle_record.get("marker_relative_path")
            != f".extract_frame_commits/{saved_stem}.json"
            or len(str(bundle_record.get("marker_sha256", ""))) != 64
            or bundle_record.get("feature_signature") != manifest.get("features")
        ):
            raise ValueError("RADIO source output bundle frame record differs")
        tensors = bundle_record.get("tensors")
        if not isinstance(tensors, list):
            raise ValueError("RADIO source output bundle tensor inventory is invalid")
        relative_path = f"backbone/{saved_stem}.pt"
        if [
            str(tensor.get("relative_path", ""))
            for tensor in tensors
            if isinstance(tensor, dict)
        ] != [relative_path, f"summary/{saved_stem}.pt"]:
            raise ValueError("RADIO source output bundle contains a noncanonical tensor set")
        declared = [
            tensor
            for tensor in tensors
            if isinstance(tensor, dict) and tensor.get("relative_path") == relative_path
        ]
        if len(declared) != 1 or declared[0] != {
            "relative_path": relative_path,
            "sha256": sha256_file(path),
            "dtype": "float16",
            "shape": [RADIO_BACKBONE_DIMENSION, height, width],
            "num_bytes": RADIO_BACKBONE_DIMENSION * height * width * 2,
        }:
            raise ValueError("RADIO source backbone tensor differs from its output bundle record")
        paths.append(path)
    return manifest_path, paths, {
        "radio_version": str(radio["version"]),
        "radio_checkpoint_sha256": str(radio["checkpoint_sha256"]),
        "radio_manifest_source_rgb_sha_bound": True,
        "radio_manifest_frame_count": len(frames),
        "radio_output_bundle_sha256": str(manifest["output_bundle_sha256"]),
        "radio_resume_contract_sha256": resume_digest,
    }


def _load_source_radio(path: Path, height: int, width: int) -> torch.Tensor:
    value = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=True)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float16
        or value.shape != (RADIO_BACKBONE_DIMENSION, height, width)
        or not torch.isfinite(value).all()
    ):
        raise ValueError("source RADIO feature must be a finite plain fp16 [1280,H,W] tensor")
    tokens = F.normalize(value.float().permute(1, 2, 0), dim=-1, eps=1e-12)
    projected = tokens @ _radio_projection_matrix()
    if bool((projected.norm(dim=-1) <= 1e-12).any()):
        raise ValueError("source RADIO projection produced a degenerate token")
    return F.normalize(projected, dim=-1, eps=1e-12)


def _local_features_from_source_rgb(
    carrier: SurfaceVoxelCarrier,
    cameras: list[Camera],
    rgb_paths: list[Path],
    normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(cameras) != len(rgb_paths) or not cameras:
        raise ValueError("source RGB paths and observation cameras must align")
    tables = []
    for camera, path in zip(cameras, rgb_paths):
        rgb = _load_source_rgb(path, camera.height, camera.width)
        tables.append(carrier.lift(rgb, camera))
    fused = fuse_evidence_tables(tables)
    available = fused.weight_sum > 0
    appearance = (fused.mean * 2.0 - 1.0) * available[:, None]
    features = torch.cat((appearance, available.float()[:, None], normals), -1)
    return features, available


def _local_features_from_source_rgb_and_radio(
    carrier: SurfaceVoxelCarrier,
    cameras: list[Camera],
    rgb_paths: list[Path],
    radio_paths: list[Path],
    normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift source-only RGB and RADIO through the identical carrier relation."""

    if not cameras or len(cameras) != len(rgb_paths) or len(cameras) != len(radio_paths):
        raise ValueError("source RGB/RADIO paths and observation cameras must align")
    tables = []
    for camera, rgb_path, radio_path in zip(cameras, rgb_paths, radio_paths):
        rgb = _load_source_rgb(rgb_path, camera.height, camera.width)
        radio = _load_source_radio(radio_path, camera.height, camera.width)
        tables.append(carrier.lift(torch.cat((rgb, radio), dim=-1), camera))
    fused = fuse_evidence_tables(tables)
    available = fused.weight_sum > 0
    rgb = (fused.mean[:, :3] * 2.0 - 1.0) * available[:, None]
    radio = F.normalize(fused.mean[:, 3:], dim=-1, eps=1e-12) * available[:, None]
    features = torch.cat(
        (rgb, available.float()[:, None], radio, torch.as_tensor(normals).float()), dim=-1
    )
    return features, available


def _render_valid_surface(
    carrier: SurfaceVoxelCarrier,
    posterior: torch.Tensor,
    valid_elements: torch.Tensor,
    camera: Camera,
) -> torch.Tensor:
    projection = carrier.project(camera)
    retained = torch.as_tensor(valid_elements, dtype=torch.bool)[projection.element_ids]
    element_ids = projection.element_ids[retained]
    pixel_ids = projection.pixel_ids[retained]
    weights = projection.weights[retained]
    posterior = torch.as_tensor(posterior, dtype=torch.float32)
    output = torch.zeros(projection.num_pixels, posterior.shape[1])
    output.index_add_(0, pixel_ids, posterior[element_ids] * weights[:, None])
    denominator = torch.zeros(projection.num_pixels)
    # Invalid/boundary carrier elements contribute zero posterior but still
    # occupy projection mass.  Filtering them out of the denominator inflates
    # valid values wherever the two strata overlap.
    denominator.index_add_(0, projection.pixel_ids, projection.weights)
    return (output / denominator.clamp_min(1e-12)[:, None]).reshape(
        camera.height, camera.width, posterior.shape[1]
    )


def _soft_iou_values(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    prediction = prediction.reshape(-1, prediction.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    intersection = (prediction * target).sum(0)
    union = prediction.sum(0) + target.sum(0) - intersection
    valid = (target.sum(0) > 0) & (union > 0)
    return list(map(float, intersection[valid] / union[valid].clamp_min(1e-12)))


def _load_all_cameras(transforms_path: Path, height: int, width: int) -> list[Camera]:
    payload = json.loads(transforms_path.read_text())
    scale_x, scale_y = width / int(payload["w"]), height / int(payload["h"])
    intrinsic = torch.tensor([
        [float(payload["fl_x"]) * scale_x, 0, float(payload["cx"]) * scale_x],
        [0, float(payload["fl_y"]) * scale_y, float(payload["cy"]) * scale_y],
        [0, 0, 1],
    ], dtype=torch.float64)
    gl_to_cv = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))
    cameras = []
    for frame in payload["frames"]:
        key = Path(frame["file_path"]).stem
        pose = torch.tensor(frame["transform_matrix"], dtype=torch.float64) @ gl_to_cv
        cameras.append(Camera(key, intrinsic, pose, height, width))
    if len(cameras) < 2:
        raise ValueError("completion oracle requires at least two cameras")
    return cameras


def _uniform_indices(total: int, count: int) -> list[int]:
    count = min(int(count), total)
    if count <= 0:
        return []
    return sorted(set(np.linspace(0, total - 1, count).round().astype(np.int64).tolist()))


def _camera_record(camera: Camera) -> dict[str, Any]:
    return {
        "key": camera.key,
        "intrinsic": camera.intrinsic,
        "camera_to_world": camera.camera_to_world,
        "height": camera.height,
        "width": camera.width,
    }


def camera_from_record(record: dict[str, Any]) -> Camera:
    return Camera(
        str(record["key"]), record["intrinsic"], record["camera_to_world"],
        int(record["height"]), int(record["width"]),
    )


@torch.no_grad()
def prepare_scene(
    scene_root: Path,
    radio_feature_root: Path,
    scene_id: str,
    *,
    output: Path,
    voxel_size: float,
    maximum_splat_radius: int,
    surface_band_voxels: float,
    maximum_contributors_per_pixel: int,
    observation_view_count: int,
    heldout_view_count: int,
    feature_height: int,
    feature_width: int,
    minimum_observed_elements: int,
    minimum_total_elements: int,
    minimum_voxel_instance_purity: float,
) -> dict[str, Any]:
    if minimum_observed_elements <= 0 or minimum_total_elements <= 0:
        raise ValueError("completion object support minima must be positive")
    directory = (scene_root / scene_id).resolve(strict=True)
    mesh_path = (directory / "points3d.ply").resolve(strict=True)
    transforms_path = (directory / "transforms.json").resolve(strict=True)
    annotation = (directory / "instance_annotations").resolve(strict=True)
    segmentation_path = next(annotation.glob("*.segs.json")).resolve(strict=True)
    aggregation_path = next(annotation.glob("*.aggregation.json")).resolve(strict=True)
    vertices, triangles, normals = _load_ply(mesh_path)
    vertex_instances, object_ids = _load_instance_labels(
        segmentation_path, aggregation_path, vertices.shape[0]
    )
    keys = torch.floor(vertices / voxel_size).long()
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    element_count = int(inverse.max()) + 1
    count = torch.bincount(inverse, minlength=element_count).float()
    centres = torch.zeros(element_count, 3).index_add_(0, inverse, vertices) / count[:, None]
    reduced_normals = torch.zeros(element_count, 3).index_add_(0, inverse, normals)
    reduced_normals = torch.nn.functional.normalize(reduced_normals, dim=-1, eps=1e-12)

    if not 0.5 <= minimum_voxel_instance_purity <= 1:
        raise ValueError("minimum voxel instance purity must be in [0.5, 1]")
    instance_stride = int(vertex_instances.max()) + 1
    pair = inverse * instance_stride + vertex_instances
    unique_pair, pair_count = torch.unique(pair, return_counts=True)
    pair_voxel = torch.div(unique_pair, instance_stride, rounding_mode="floor")
    pair_instance = unique_pair.remainder(instance_stride)
    maximum_count = torch.zeros(element_count, dtype=torch.long)
    maximum_count.scatter_reduce_(0, pair_voxel, pair_count, reduce="amax", include_self=True)
    winner = pair_count == maximum_count[pair_voxel]
    element_instance = torch.full((element_count,), instance_stride, dtype=torch.long)
    element_instance.scatter_reduce_(
        0, pair_voxel[winner], pair_instance[winner], reduce="amin", include_self=True
    )
    instance_purity = maximum_count.float() / count.clamp_min(1)
    completion_valid = instance_purity >= minimum_voxel_instance_purity
    element_instance[~completion_valid] = 0
    carrier = SurfaceVoxelCarrier(
        centres, voxel_size, normals=reduced_normals,
        maximum_splat_radius=maximum_splat_radius,
        surface_band_voxels=surface_band_voxels,
        maximum_contributors_per_pixel=maximum_contributors_per_pixel,
    )
    cameras = _load_all_cameras(transforms_path, feature_height, feature_width)
    observation_indices = _uniform_indices(len(cameras), observation_view_count)
    remaining = [index for index in range(len(cameras)) if index not in set(observation_indices)]
    heldout_positions = _uniform_indices(len(remaining), heldout_view_count)
    heldout_indices = [remaining[index] for index in heldout_positions]
    if not observation_indices or not heldout_indices:
        raise ValueError("observation and held-out camera sets must both be non-empty")
    observation_cameras = [cameras[index] for index in observation_indices]
    observation_rgb_paths = [
        _source_rgb_path(directory, camera) for camera in observation_cameras
    ]
    radio_manifest_path, observation_radio_paths, radio_provenance = _source_radio_paths(
        radio_feature_root,
        scene_id,
        observation_cameras,
        observation_rgb_paths,
        height=feature_height,
        width=feature_width,
    )
    local_features, feature_available = _local_features_from_source_rgb_and_radio(
        carrier,
        observation_cameras,
        observation_rgb_paths,
        observation_radio_paths,
        reduced_normals,
    )
    source_visible = feature_available

    candidate_object_ids = []
    for object_id in object_ids:
        member = element_instance == object_id
        if (
            int(member.sum()) >= minimum_total_elements
            and int((member & source_visible).sum()) >= minimum_observed_elements
        ):
            candidate_object_ids.append(int(object_id))
    candidate_mask_support, candidate_dropout_records = _positive_mask_support(
        carrier,
        observation_cameras,
        scene_id=scene_id,
        element_object_id=element_instance,
        object_ids=candidate_object_ids,
    )
    retained_object_ids = [
        object_id
        for object_id in candidate_object_ids
        if int(((element_instance == object_id) & candidate_mask_support).sum())
        >= minimum_observed_elements
    ]
    if len(retained_object_ids) < 2:
        raise RuntimeError("too few observed object tokens survive completion preparation")
    lookup = torch.full((int(vertex_instances.max()) + 1,), -1, dtype=torch.long)
    lookup[torch.tensor(retained_object_ids)] = torch.arange(len(retained_object_ids))
    token_index = lookup[element_instance]
    annotated = (token_index >= 0) & completion_valid
    mask_supported = candidate_mask_support & annotated
    membership_observed = mask_supported.clone()
    if any(
        int((mask_supported & (token_index == token)).sum()) < minimum_observed_elements
        for token in range(len(retained_object_ids))
    ):
        raise RuntimeError("a retained completion token lacks preregistered mask support")
    observed_fraction = float(membership_observed[annotated].float().mean())
    source_visible_fraction = float(source_visible[annotated].float().mean())
    visible_unmasked = annotated & source_visible & ~membership_observed
    never_visible = annotated & ~source_visible
    retained_object_set = set(retained_object_ids)
    retained_dropout_records = [
        record
        for record in candidate_dropout_records
        if int(record["object_id"]) in retained_object_set
    ]

    # Held-out 2-D authority is the original annotated mesh, never a raster
    # generated by the sparse carrier under evaluation.
    mesh_oracle = MeshCarrier(vertices, triangles)
    vertex_token_index = lookup[vertex_instances]
    vertex_membership = torch.zeros(vertices.shape[0], len(retained_object_ids))
    annotated_vertex = vertex_token_index >= 0
    vertex_membership[annotated_vertex, vertex_token_index[annotated_vertex]] = 1.0
    surface_membership = torch.zeros(element_count, len(retained_object_ids))
    surface_membership[annotated, token_index[annotated]] = 1.0
    heldout_mesh_target_rasters = []
    ceiling_values = []
    ceiling_per_view = []
    for index in heldout_indices:
        camera = cameras[index]
        mesh_target = mesh_oracle.render_posterior(vertex_membership, camera)
        surface_ceiling = _render_valid_surface(
            carrier, surface_membership, completion_valid, camera
        )
        values = _soft_iou_values(surface_ceiling, mesh_target)
        ceiling_values.extend(values)
        ceiling_per_view.append({
            "camera_key": camera.key,
            "soft_miou": float(np.mean(values)) if values else None,
            "visible_token_count": len(values),
        })
        heldout_mesh_target_rasters.append(mesh_target.to(torch.float16))
    sealed_inputs = (
        HashedInput.seal("surface_mesh", mesh_path),
        HashedInput.seal("camera_transforms", transforms_path),
        HashedInput.seal("instance_segmentation", segmentation_path),
        HashedInput.seal("instance_aggregation", aggregation_path),
        *tuple(
            HashedInput.seal(f"source_observation_rgb_{index}", path)
            for index, path in enumerate(observation_rgb_paths)
        ),
        HashedInput.seal("source_radio_frame_manifest", radio_manifest_path),
        *tuple(
            HashedInput.seal(f"source_observation_radio_backbone_{index}", path)
            for index, path in enumerate(observation_radio_paths)
        ),
    )
    geometry_receipt = GeometryReceipt(
        carrier="sparse_surface_voxel",
        coordinate_convention="scannet_mesh_nerf_opengl_to_opencv_feature_raster",
        inputs=sealed_inputs,
        source_rgb_opened=True,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=True,
        metadata={
            "scene_disjoint_supervised_completion_oracle": True,
            "observed_instance_membership_is_oracle_input": True,
            "observed_instance_membership_is_mask_supported_only": True,
            "unobserved_instance_membership_is_training_target_only": True,
            "full_instance_membership_is_training_target_only": False,
            "source_visibility_is_not_membership_observation": True,
            "projection_configuration_has_no_implicit_defaults": True,
            "voxel_size": voxel_size,
            "maximum_splat_radius": maximum_splat_radius,
            "surface_band_voxels": surface_band_voxels,
            "maximum_contributors_per_pixel": maximum_contributors_per_pixel,
            "minimum_voxel_instance_purity": minimum_voxel_instance_purity,
            "mesh_rgb_consumed": False,
            "observation_rgb_and_radio_lifted_through_same_carrier": True,
            "unobserved_appearance_is_zero": True,
            "appearance_availability_bit_appended": True,
            "source_radio_opened": True,
            "heldout_radio_opened": False,
            "radio_backbone_dimension": RADIO_BACKBONE_DIMENSION,
            "radio_projection_dimension": RADIO_PROJECTION_DIMENSION,
            "radio_projection_method": "sha256_entry_rademacher_jl_v1",
            "radio_projection_salt": RADIO_PROJECTION_SALT.decode("ascii"),
            "radio_projection_sha256": RADIO_PROJECTION_SHA256,
            "radio_pixel_normalized_before_projection": True,
            "radio_pixel_normalized_after_projection": True,
            "radio_element_normalized_after_lift": True,
            **radio_provenance,
            "observation_frame_ids": [camera.key for camera in observation_cameras],
            "source_radio_frame_ids": [camera.key for camera in observation_cameras],
            "heldout_frame_ids": [cameras[index].key for index in heldout_indices],
            "heldout_rgb_opened": False,
            "heldout_target_authority": "original_mesh_vertex_instance_raycast",
            "heldout_target_uses_sparse_carrier": False,
            "mask_dropout_method": "sha256_scene_frame_original_object_salt_v1",
            "mask_dropout_salt": MASK_DROPOUT_SALT,
            "mask_dropout_keep_probability": MASK_DROPOUT_KEEP_PROBABILITY,
            "mask_dropout_depends_on_validation_outcomes": False,
        },
    )
    mask_dropout_receipt = {
        "schema": "radio_gs.surface_object_memory_v4.source_mask_dropout.v1",
        "method": "sha256_scene_frame_original_object_salt_v1",
        "hash_input_order": ["scene_id", "frame_id", "original_object_id", "salt"],
        "salt": MASK_DROPOUT_SALT,
        "keep_probability": MASK_DROPOUT_KEEP_PROBABILITY,
        "candidate_object_ids": candidate_object_ids,
        "retained_object_ids": retained_object_ids,
        "records": retained_dropout_records,
    }
    payload = {
        "schema": CACHE_SCHEMA,
        "scene_id": scene_id,
        "centres": centres,
        "normals": reduced_normals,
        "local_features": local_features,
        "appearance_available": feature_available,
        "feature_available": feature_available,
        "source_visible": source_visible,
        "mask_supported": mask_supported,
        "membership_observed": membership_observed,
        "completion_valid": completion_valid,
        "token_index": token_index,
        "object_ids": retained_object_ids,
        # Compatibility alias consumed by the current completion runtime.  It is
        # membership evidence, not generic source/feature visibility.
        "observed_visible": membership_observed,
        "mask_dropout_receipt": mask_dropout_receipt,
        "observation_cameras": [_camera_record(camera) for camera in observation_cameras],
        "heldout_cameras": [_camera_record(cameras[index]) for index in heldout_indices],
        "heldout_mesh_target_rasters": heldout_mesh_target_rasters,
        "surface_perfect_membership_ceiling": {
            "heldout_2d_soft_miou": float(np.mean(ceiling_values)) if ceiling_values else 0.0,
            "token_view_count": len(ceiling_values),
            "per_view": ceiling_per_view,
        },
        "configuration": {
            "voxel_size": voxel_size,
            "maximum_splat_radius": maximum_splat_radius,
            "surface_band_voxels": surface_band_voxels,
            "maximum_contributors_per_pixel": maximum_contributors_per_pixel,
            "feature_height": feature_height,
            "feature_width": feature_width,
            "observation_view_count": len(observation_indices),
            "heldout_view_count": len(heldout_indices),
            "minimum_observed_elements": minimum_observed_elements,
            "minimum_total_elements": minimum_total_elements,
            "minimum_voxel_instance_purity": minimum_voxel_instance_purity,
            "radio_backbone_dimension": RADIO_BACKBONE_DIMENSION,
            "radio_projection_dimension": RADIO_PROJECTION_DIMENSION,
            "radio_projection_sha256": RADIO_PROJECTION_SHA256,
            "mask_dropout_keep_probability": MASK_DROPOUT_KEEP_PROBABILITY,
            "mask_dropout_salt": MASK_DROPOUT_SALT,
            "local_feature_layout": _radio_feature_layout(),
        },
        "geometry_receipt": geometry_receipt.to_dict(),
        "input_receipt": [value.__dict__ for value in sealed_inputs],
        "statistics": {
            "element_count": element_count,
            "annotated_element_count": int(annotated.sum()),
            "retained_token_count": len(retained_object_ids),
            "observed_annotated_fraction": observed_fraction,
            "source_visible_annotated_fraction": source_visible_fraction,
            "mask_supported_annotated_count": int((mask_supported & annotated).sum()),
            "visible_unmasked_annotated_count": int(visible_unmasked.sum()),
            "visible_unmasked_annotated_fraction": float(
                visible_unmasked.sum() / annotated.sum().clamp_min(1)
            ),
            "never_visible_annotated_count": int(never_visible.sum()),
            "never_visible_annotated_fraction": float(
                never_visible.sum() / annotated.sum().clamp_min(1)
            ),
            "candidate_token_count_before_mask_support": len(candidate_object_ids),
            "dropped_insufficient_mask_support_object_count": (
                len(candidate_object_ids) - len(retained_object_ids)
            ),
            "mean_voxel_instance_purity": float(instance_purity.mean()),
            "low_purity_voxel_count": int((instance_purity < minimum_voxel_instance_purity).sum()),
            "completion_valid_element_count": int(completion_valid.sum()),
            "valid_null_surface_count": int(((token_index < 0) & completion_valid).sum()),
            "appearance_available_fraction": float(feature_available.float().mean()),
            "mask_supported_element_fraction": float(mask_supported.float().mean()),
            "surface_perfect_membership_heldout_2d_soft_miou": (
                float(np.mean(ceiling_values)) if ceiling_values else 0.0
            ),
        },
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "scene_id": scene_id,
        "output": str(output),
        "sha256": sha256_file(output),
        **payload["statistics"],
    }


def load_scene_cache(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("schema") != CACHE_SCHEMA:
        raise ValueError("not a v4 ScanNet completion scene cache")
    required = (
        "scene_id", "centres", "normals", "local_features", "token_index", "object_ids",
        "observed_visible", "appearance_available", "feature_available", "source_visible",
        "mask_supported", "membership_observed", "mask_dropout_receipt",
        "completion_valid", "geometry_receipt",
        "input_receipt", "configuration", "observation_cameras", "heldout_cameras",
        "heldout_mesh_target_rasters", "surface_perfect_membership_ceiling",
    )
    if any(key not in payload for key in required):
        raise ValueError("completion scene cache is incomplete")
    configuration = payload["configuration"]
    centres = torch.as_tensor(payload["centres"], dtype=torch.float32)
    normals = torch.as_tensor(payload["normals"], dtype=torch.float32)
    if centres.ndim != 2 or centres.shape[1] != 3 or not torch.isfinite(centres).all():
        raise ValueError("completion centres must be a finite [E, 3] tensor")
    element_count = int(centres.shape[0])
    if normals.shape != centres.shape or not torch.isfinite(normals).all():
        raise ValueError("completion normals must be finite and align with centres")
    if torch.as_tensor(payload["local_features"]).shape[0] != element_count:
        raise ValueError("completion local features do not align")
    if torch.as_tensor(payload["token_index"]).shape != (element_count,):
        raise ValueError("completion token targets do not align")
    for key in (
        "observed_visible", "appearance_available", "feature_available", "source_visible",
        "mask_supported", "membership_observed", "completion_valid",
    ):
        if torch.as_tensor(payload[key]).shape != (element_count,):
            raise ValueError(f"completion {key} does not align")
    appearance_available = torch.as_tensor(payload["appearance_available"], dtype=torch.bool)
    feature_available = torch.as_tensor(payload["feature_available"], dtype=torch.bool)
    source_visible = torch.as_tensor(payload["source_visible"], dtype=torch.bool)
    mask_supported = torch.as_tensor(payload["mask_supported"], dtype=torch.bool)
    membership_observed = torch.as_tensor(payload["membership_observed"], dtype=torch.bool)
    observed_visible = torch.as_tensor(payload["observed_visible"], dtype=torch.bool)
    completion_valid = torch.as_tensor(payload["completion_valid"], dtype=torch.bool)
    if not (
        torch.equal(appearance_available, feature_available)
        and torch.equal(feature_available, source_visible)
    ):
        raise ValueError("completion source visibility and feature availability disagree")
    if not (
        torch.equal(mask_supported, membership_observed)
        and torch.equal(membership_observed, observed_visible)
    ):
        raise ValueError("completion mask support and membership observation disagree")
    if bool((membership_observed & ~source_visible).any()):
        raise ValueError("completion membership support must be a subset of source visibility")
    local_features = torch.as_tensor(payload["local_features"], dtype=torch.float32)
    if local_features.shape[1] != 4 + RADIO_PROJECTION_DIMENSION + 3:
        raise ValueError("completion local feature schema must be RGB+availability+RADIO64+normal")
    expected_feature_layout = _radio_feature_layout()
    if configuration.get("local_feature_layout") != expected_feature_layout:
        raise ValueError("completion local feature layout is not the frozen source RGB/RADIO contract")
    if not torch.isfinite(local_features).all():
        raise ValueError("completion local features must be finite")
    if bool(((local_features[:, :3] < -1) | (local_features[:, :3] > 1)).any()):
        raise ValueError("completion source-RGB features must remain in [-1, 1]")
    radio_start = 4
    radio_stop = radio_start + RADIO_PROJECTION_DIMENSION
    if not torch.equal(local_features[:, -3:], normals):
        raise ValueError("completion local normal features disagree with carrier normals")
    if not torch.equal(local_features[:, 3] > 0.5, appearance_available):
        raise ValueError("completion appearance bit disagrees with availability")
    if bool((local_features[~appearance_available, :3] != 0).any()) or bool(
        (local_features[~appearance_available, radio_start:radio_stop] != 0).any()
    ):
        raise ValueError("unobserved completion appearance must be exactly zero")
    observed_radio_norm = local_features[appearance_available, radio_start:radio_stop].norm(dim=-1)
    if observed_radio_norm.numel() and not torch.allclose(
        observed_radio_norm, torch.ones_like(observed_radio_norm), atol=5e-5, rtol=5e-5
    ):
        raise ValueError("available completion RADIO features must have unit norm")
    receipt = payload.get("geometry_receipt", {})
    metadata = receipt.get("metadata", {})
    expected_flags = {
        "source_rgb_opened": True,
        "target_rgb_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": True,
    }
    for key, expected in expected_flags.items():
        if receipt.get(key) is not expected:
            raise ValueError(f"completion cache geometry receipt has invalid {key}")
    for key in (
        "voxel_size", "maximum_splat_radius", "surface_band_voxels",
        "maximum_contributors_per_pixel", "minimum_voxel_instance_purity",
    ):
        if metadata.get(key) != configuration.get(key):
            raise ValueError(f"completion cache geometry receipt disagrees on {key}")
    if metadata.get("mesh_rgb_consumed") is not False:
        raise ValueError("completion cache must forbid mesh RGB features")
    if metadata.get("observed_instance_membership_is_oracle_input") is not True:
        raise ValueError("completion cache must disclose observed oracle membership input")
    if metadata.get("unobserved_instance_membership_is_training_target_only") is not True:
        raise ValueError("completion cache must restrict target-only membership to unobserved rows")
    if metadata.get("full_instance_membership_is_training_target_only") is not False:
        raise ValueError("completion cache cannot claim full membership is target-only")
    if metadata.get("observed_instance_membership_is_mask_supported_only") is not True:
        raise ValueError("completion cache must restrict observations to mask-supported membership")
    if metadata.get("source_visibility_is_not_membership_observation") is not True:
        raise ValueError("completion cache conflates source visibility with membership observation")
    if metadata.get("heldout_rgb_opened") is not False:
        raise ValueError("completion cache opened held-out RGB")
    if metadata.get("source_radio_opened") is not True or metadata.get("heldout_radio_opened") is not False:
        raise ValueError("completion cache has an invalid source-only RADIO disclosure")
    expected_radio_contract = {
        "radio_version": "c-radio_v4-h",
        "radio_checkpoint_sha256": RADIO_CHECKPOINT_SHA256,
        "radio_manifest_source_rgb_sha_bound": True,
        "radio_backbone_dimension": RADIO_BACKBONE_DIMENSION,
        "radio_projection_dimension": RADIO_PROJECTION_DIMENSION,
        "radio_projection_method": "sha256_entry_rademacher_jl_v1",
        "radio_projection_salt": RADIO_PROJECTION_SALT.decode("ascii"),
        "radio_projection_sha256": RADIO_PROJECTION_SHA256,
        "radio_pixel_normalized_before_projection": True,
        "radio_pixel_normalized_after_projection": True,
        "radio_element_normalized_after_lift": True,
    }
    if any(metadata.get(key) != value for key, value in expected_radio_contract.items()):
        raise ValueError("completion cache RADIO projection receipt differs")
    if (
        int(metadata.get("radio_manifest_frame_count", -1))
        != int(configuration.get("observation_view_count", -2))
        or not str(metadata.get("radio_output_bundle_sha256", ""))
        or not str(metadata.get("radio_resume_contract_sha256", ""))
    ):
        raise ValueError("completion cache RADIO source-only bundle receipt differs")
    if (
        configuration.get("radio_backbone_dimension") != RADIO_BACKBONE_DIMENSION
        or configuration.get("radio_projection_dimension") != RADIO_PROJECTION_DIMENSION
        or configuration.get("radio_projection_sha256") != RADIO_PROJECTION_SHA256
    ):
        raise ValueError("completion cache RADIO configuration differs")
    expected_dropout_contract = {
        "mask_dropout_method": "sha256_scene_frame_original_object_salt_v1",
        "mask_dropout_salt": MASK_DROPOUT_SALT,
        "mask_dropout_keep_probability": MASK_DROPOUT_KEEP_PROBABILITY,
        "mask_dropout_depends_on_validation_outcomes": False,
    }
    if any(metadata.get(key) != value for key, value in expected_dropout_contract.items()):
        raise ValueError("completion cache mask-dropout receipt differs")
    if (
        configuration.get("mask_dropout_salt") != MASK_DROPOUT_SALT
        or configuration.get("mask_dropout_keep_probability")
        != MASK_DROPOUT_KEEP_PROBABILITY
    ):
        raise ValueError("completion cache mask-dropout configuration differs")
    if metadata.get("heldout_target_authority") != "original_mesh_vertex_instance_raycast":
        raise ValueError("completion held-out target is not the independent mesh authority")
    if metadata.get("heldout_target_uses_sparse_carrier") is not False:
        raise ValueError("completion held-out target cannot use the sparse carrier")
    input_receipt = payload.get("input_receipt", [])
    if list(receipt.get("inputs", ())) != input_receipt:
        raise ValueError("completion input receipt disagrees with the geometry receipt")
    rgb_inputs = [
        row for row in input_receipt
        if str(row.get("role", "")).startswith("source_observation_rgb_")
    ]
    observation_cameras = payload["observation_cameras"]
    expected_observation = int(configuration.get("observation_view_count", -1))
    if len(rgb_inputs) != expected_observation or len(observation_cameras) != expected_observation:
        raise ValueError("completion cache must seal exactly the opened observation RGB files")
    roles = [str(row["role"]) for row in rgb_inputs]
    if roles != [f"source_observation_rgb_{index}" for index in range(len(rgb_inputs))]:
        raise ValueError("completion source RGB receipt roles must be complete and ordered")
    radio_manifest_inputs = [
        row for row in input_receipt if row.get("role") == "source_radio_frame_manifest"
    ]
    radio_inputs = [
        row for row in input_receipt
        if str(row.get("role", "")).startswith("source_observation_radio_backbone_")
    ]
    radio_roles = [str(row["role"]) for row in radio_inputs]
    if len(radio_manifest_inputs) != 1 or radio_roles != [
        f"source_observation_radio_backbone_{index}"
        for index in range(expected_observation)
    ]:
        raise ValueError("completion cache must seal exactly the ordered source RADIO tensors")
    all_roles = [str(row.get("role", "")) for row in input_receipt]
    if all_roles != [
        "surface_mesh", "camera_transforms", "instance_segmentation",
        "instance_aggregation", *roles, "source_radio_frame_manifest", *radio_roles,
    ]:
        raise ValueError("completion sealed inputs contain an unexpected source or target")
    if any("heldout" in role and ("rgb" in role or "radio" in role) for role in all_roles):
        raise ValueError("completion cache sealed a held-out appearance input")
    token_index = torch.as_tensor(payload["token_index"], dtype=torch.long)
    token_count = len(payload["object_ids"])
    if token_count < 1 or len(set(map(int, payload["object_ids"]))) != token_count:
        raise ValueError("completion object identities must be non-empty and unique")
    if bool(((token_index < -1) | (token_index >= token_count)).any()):
        raise ValueError("completion token targets exceed the retained token set")
    if bool(((token_index >= 0) & ~completion_valid).any()):
        raise ValueError("invalid boundary voxels cannot retain object targets")
    if bool((membership_observed & ((token_index < 0) | ~completion_valid)).any()):
        raise ValueError("mask-supported membership must be a valid retained-object positive")
    minimum_support = int(configuration.get("minimum_observed_elements", -1))
    if minimum_support <= 0 or any(
        int((membership_observed & (token_index == token)).sum()) < minimum_support
        for token in range(token_count)
    ):
        raise ValueError("a retained completion token lacks minimum mask-supported positives")
    target_rasters = payload["heldout_mesh_target_rasters"]
    heldout_cameras = payload["heldout_cameras"]
    expected_heldout = int(configuration.get("heldout_view_count", -1))
    if len(target_rasters) != expected_heldout or len(heldout_cameras) != expected_heldout:
        raise ValueError("completion cache held-out cameras and mesh targets are incomplete")
    height = int(configuration.get("feature_height", -1))
    width = int(configuration.get("feature_width", -1))
    observation_keys = [str(record.get("key", "")) for record in observation_cameras]
    heldout_keys = [str(record.get("key", "")) for record in heldout_cameras]
    if observation_keys != list(metadata.get("observation_frame_ids", [])):
        raise ValueError("completion observation cameras disagree with the geometry receipt")
    if observation_keys != list(metadata.get("source_radio_frame_ids", [])):
        raise ValueError("completion source RADIO frames disagree with observation cameras")
    if heldout_keys != list(metadata.get("heldout_frame_ids", [])):
        raise ValueError("completion held-out cameras disagree with the geometry receipt")
    if len(set(observation_keys + heldout_keys)) != len(observation_keys) + len(heldout_keys):
        raise ValueError("completion observation and held-out camera identities must be disjoint")
    for record in (*observation_cameras, *heldout_cameras):
        camera = camera_from_record(record)
        if camera.height != height or camera.width != width:
            raise ValueError("completion camera raster shape disagrees with configuration")
    dropout_receipt = payload["mask_dropout_receipt"]
    expected_dropout_header = {
        "schema": "radio_gs.surface_object_memory_v4.source_mask_dropout.v1",
        "method": "sha256_scene_frame_original_object_salt_v1",
        "hash_input_order": ["scene_id", "frame_id", "original_object_id", "salt"],
        "salt": MASK_DROPOUT_SALT,
        "keep_probability": MASK_DROPOUT_KEEP_PROBABILITY,
    }
    if not isinstance(dropout_receipt, dict) or any(
        dropout_receipt.get(key) != value for key, value in expected_dropout_header.items()
    ):
        raise ValueError("completion mask-dropout receipt header differs")
    if list(map(int, dropout_receipt.get("retained_object_ids", []))) != list(
        map(int, payload["object_ids"])
    ):
        raise ValueError("completion mask-dropout retained identities differ")
    candidate_ids = list(map(int, dropout_receipt.get("candidate_object_ids", [])))
    if len(candidate_ids) != len(set(candidate_ids)) or not set(map(int, payload["object_ids"])).issubset(
        candidate_ids
    ):
        raise ValueError("completion mask-dropout candidate identities are invalid")
    element_object_id = torch.full((element_count,), -1, dtype=torch.long)
    for token, object_id in enumerate(map(int, payload["object_ids"])):
        element_object_id[token_index == token] = object_id
    support_carrier = SurfaceVoxelCarrier(
        centres,
        float(configuration["voxel_size"]),
        normals=normals,
        maximum_splat_radius=int(configuration["maximum_splat_radius"]),
        surface_band_voxels=float(configuration["surface_band_voxels"]),
        maximum_contributors_per_pixel=int(configuration["maximum_contributors_per_pixel"]),
    )
    source_cameras = [camera_from_record(record) for record in observation_cameras]
    expected_source_visible = torch.zeros(element_count, dtype=torch.bool)
    for camera in source_cameras:
        expected_source_visible[support_carrier.project(camera).element_ids] = True
    if not torch.equal(expected_source_visible, source_visible):
        raise ValueError("completion source visibility differs from source camera projections")
    expected_support, expected_records = _positive_mask_support(
        support_carrier,
        source_cameras,
        scene_id=str(payload["scene_id"]),
        element_object_id=element_object_id,
        object_ids=list(map(int, payload["object_ids"])),
    )
    expected_support &= token_index >= 0
    if not torch.equal(expected_support, membership_observed):
        raise ValueError("completion mask-supported membership differs from deterministic dropout")
    if dropout_receipt.get("records") != expected_records:
        raise ValueError("completion mask-dropout decisions differ from their hash inputs")
    for target in target_rasters:
        values = torch.as_tensor(target)
        if values.shape != (height, width, token_count):
            raise ValueError("completion held-out mesh target has the wrong shape")
        if not torch.isfinite(values).all() or bool(((values < 0) | (values > 1)).any()):
            raise ValueError("completion held-out mesh target must be finite probabilities")
    ceiling = payload["surface_perfect_membership_ceiling"]
    if not isinstance(ceiling, dict) or not 0 <= float(
        ceiling.get("heldout_2d_soft_miou", -1)
    ) <= 1:
        raise ValueError("completion sparse-carrier ceiling is missing or invalid")
    payload["cache_path"] = str(resolved)
    payload["cache_sha256"] = sha256_file(resolved)
    return payload
