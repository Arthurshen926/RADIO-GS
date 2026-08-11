#!/usr/bin/env python3
"""Render frozen primitive/query scores for the exact LERF-2D cameras.

This is a rendering-only bridge between the immutable per-scene
``[primitive,scale,query]`` cache and
``eval_ours_lerf2d_scalar_maps.py``.  It never encodes text, chooses a scale,
normalizes scores, applies a threshold, opens a mask, or resizes an output.
Every map is the repository's canonical gsplat alpha-composited scalar color
with a zero background at the annotation camera's native resolution.

The scene-authority JSON is intentionally external to the protocol freeze: it
binds the method artifacts being evaluated, while the freeze binds the public
benchmark readout.  All file records are SHA-256 checked before rendering and
rechecked before the completed bundle is published.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.data.lerf_dataset import LERFDataset
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    GaussianSelectionProxy,
    build_mask_renderer,
    validate_ours_multiscale_query_score_cache,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.eval_ours_lerf2d_scalar_maps import (
    ARTIFACT_TYPE as EVALUATOR_ARTIFACT_TYPE,
    CANONICAL_TASK_ID,
    EXPECTED_REGISTRY_ROW,
    SCORE_SEMANTICS,
    FrozenLerf2DContract,
    _load_json_bytes,
    _read_stable_regular_file,
    canonical_query_id,
    load_frozen_contract,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    SHARED_AUTHORITY_CONTRACT,
    _tensor_sha256,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache_fp32 import (
    SHARED_AUTHORITY_CONTRACT as FP32_SHARED_AUTHORITY_CONTRACT,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    fsync_directory,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    stable_descriptor_load,
    write_bytes_noclobber,
    write_frozen_json,
)


SCHEMA_VERSION = 1
AUTHORITY_ARTIFACT_TYPE = "radio_gs_lerf2d_scalar_map_render_authority"
RECEIPT_ARTIFACT_TYPE = "radio_gs_lerf2d_scalar_map_render_receipt"
RENDER_OPERATOR = "canonical_gsplat_alpha_composited_scalar_zero_background"


class ScalarMapRenderError(ValueError):
    """Raised before publication when renderer authority does not match."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str


@dataclass(frozen=True)
class SceneBinding:
    scene: str
    scene_root: Path
    config: FileBinding
    geometry_checkpoint: FileBinding
    geometry_ply: FileBinding
    query_score_cache: FileBinding
    camera_sources: Mapping[str, FileBinding]


@dataclass(frozen=True)
class FrameBinding:
    frame: str
    annotation_path: Path
    annotation_sha256: str
    camera_name: str
    resolution_hw: tuple[int, int]
    query_texts: tuple[str, ...]
    query_ids: tuple[str, ...]


@dataclass
class SceneRuntime:
    model: torch.nn.Module
    config: object
    dataset: LERFDataset


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScalarMapRenderError(f"{label} must be a mapping")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise ScalarMapRenderError(
            f"{label} fields differ: expected {sorted(keys)}, got {sorted(value)}"
        )


def _require_sha(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ScalarMapRenderError(f"{label} must be a lowercase SHA-256")
    return text


def _canonical_directory(path: object, *, label: str) -> Path:
    if not isinstance(path, str) or not path:
        raise ScalarMapRenderError(f"{label} must be a non-empty path")
    raw = Path(os.path.abspath(os.path.expanduser(path)))
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ScalarMapRenderError(f"{label} does not exist: {raw}") from exc
    if not resolved.is_dir():
        raise ScalarMapRenderError(f"{label} must be a directory: {resolved}")
    return resolved


def _file_binding(value: object, *, label: str) -> FileBinding:
    record = _require_mapping(value, label=label)
    _require_exact_keys(record, {"path", "sha256"}, label=label)
    expected = _require_sha(record["sha256"], label=f"{label}.sha256")
    try:
        _, observed, source = stable_descriptor_load(
            str(record["path"]),
            lambda handle: None,
            expected_sha256=expected,
            label=label,
        )
    except (OSError, ValueError) as exc:
        raise ScalarMapRenderError(str(exc)) from exc
    return FileBinding(source, observed)


def _camera_source_paths(scene_root: Path) -> dict[str, Path]:
    transforms = scene_root / "transforms.json"
    if transforms.is_file():
        return {"transforms_json": transforms}
    sparse = scene_root / "sparse" / "0"
    paths = {
        "colmap_cameras_bin": sparse / "cameras.bin",
        "colmap_images_bin": sparse / "images.bin",
    }
    sidecar = scene_root / "hwf_cxcy.npy"
    if sidecar.is_file():
        paths["nex_intrinsics"] = sidecar
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ScalarMapRenderError(
            "camera source is incomplete: " + ", ".join(missing)
        )
    return paths


def _parse_scene_binding(scene: str, value: object) -> SceneBinding:
    raw = _require_mapping(value, label=f"scenes.{scene}")
    _require_exact_keys(
        raw,
        {
            "scene_root",
            "config",
            "geometry_checkpoint",
            "geometry_ply",
            "query_score_cache",
            "camera_sources",
        },
        label=f"scenes.{scene}",
    )
    scene_root = _canonical_directory(raw["scene_root"], label=f"{scene}.scene_root")
    camera_records = _require_mapping(
        raw["camera_sources"], label=f"{scene}.camera_sources"
    )
    discovered = _camera_source_paths(scene_root)
    if tuple(camera_records) != tuple(discovered) or set(camera_records) != set(discovered):
        raise ScalarMapRenderError(
            f"{scene}: camera source roles/order differ: "
            f"{tuple(camera_records)} vs {tuple(discovered)}"
        )
    camera_sources: dict[str, FileBinding] = {}
    for role, path in discovered.items():
        binding = _file_binding(
            camera_records[role], label=f"{scene}.camera_sources.{role}"
        )
        if binding.path != path.resolve(strict=True):
            raise ScalarMapRenderError(f"{scene}: {role} path differs from scene root")
        camera_sources[role] = binding
    return SceneBinding(
        scene=scene,
        scene_root=scene_root,
        config=_file_binding(raw["config"], label=f"{scene}.config"),
        geometry_checkpoint=_file_binding(
            raw["geometry_checkpoint"], label=f"{scene}.geometry_checkpoint"
        ),
        geometry_ply=_file_binding(raw["geometry_ply"], label=f"{scene}.geometry_ply"),
        query_score_cache=_file_binding(
            raw["query_score_cache"], label=f"{scene}.query_score_cache"
        ),
        camera_sources=camera_sources,
    )


def load_render_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    contract: FrozenLerf2DContract,
) -> tuple[str, Path, str, Path, Mapping[str, SceneBinding]]:
    """Load and validate all non-benchmark method inputs."""

    payload, digest, source = load_json_object(
        path,
        expected_sha256=_require_sha(expected_sha256, label="authority SHA-256"),
        label="LERF-2D render authority",
    )
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "artifact_type",
            "method",
            "canonical_task_id",
            "registry_row",
            "protocol_freeze",
            "label_root",
            "scenes",
        },
        label="render authority",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ScalarMapRenderError("render authority schema_version differs")
    if payload["artifact_type"] != AUTHORITY_ARTIFACT_TYPE:
        raise ScalarMapRenderError("render authority artifact_type differs")
    method = payload["method"]
    if not isinstance(method, str) or not method or method == "OccamLGS":
        raise ScalarMapRenderError("render authority method is invalid")
    if payload["canonical_task_id"] != contract.canonical_task_id:
        raise ScalarMapRenderError("render authority canonical task differs")
    if payload["registry_row"] != contract.registry_row:
        raise ScalarMapRenderError("render authority registry row differs")
    freeze = _require_mapping(payload["protocol_freeze"], label="protocol_freeze")
    _require_exact_keys(freeze, {"freeze_id", "sha256"}, label="protocol_freeze")
    if freeze["freeze_id"] != contract.freeze_id or freeze["sha256"] != contract.freeze_sha256:
        raise ScalarMapRenderError("render authority protocol freeze differs")
    label_root = _canonical_directory(payload["label_root"], label="label_root")
    scenes = _require_mapping(payload["scenes"], label="scenes")
    if tuple(scenes) != contract.scenes or set(scenes) != set(contract.scenes):
        raise ScalarMapRenderError("render authority scene cohort/order differs")
    bindings = {
        scene: _parse_scene_binding(scene, scenes[scene]) for scene in contract.scenes
    }
    return str(method), source, digest, label_root, bindings


def _frame_bindings(
    scene: str,
    *,
    label_root: Path,
    frames: Sequence[str],
) -> tuple[list[FrameBinding], tuple[str, ...]]:
    result: list[FrameBinding] = []
    all_queries: set[str] = set()
    for frame in frames:
        annotation = label_root / scene / f"{frame}.json"
        encoded = _read_stable_regular_file(
            annotation, label=f"{scene}/{frame} annotation metadata"
        )
        payload = _load_json_bytes(
            encoded, label=f"{scene}/{frame} annotation metadata"
        )
        info = _require_mapping(payload.get("info"), label=f"{scene}/{frame}.info")
        camera_name = info.get("name")
        if not isinstance(camera_name, str) or not camera_name:
            raise ScalarMapRenderError(f"{scene}/{frame}: annotation has no camera name")
        if Path(camera_name).stem != frame:
            raise ScalarMapRenderError(f"{scene}/{frame}: annotation camera stem differs")
        try:
            resolution = (int(info["height"]), int(info["width"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ScalarMapRenderError(
                f"{scene}/{frame}: annotation resolution is malformed"
            ) from exc
        if min(resolution) <= 0:
            raise ScalarMapRenderError(f"{scene}/{frame}: annotation resolution is invalid")
        raw_objects = payload.get("objects")
        if not isinstance(raw_objects, list):
            raise ScalarMapRenderError(f"{scene}/{frame}: annotation objects differ")
        # The renderer is allowed to bind camera/query metadata only.  It never
        # accesses segmentation or bbox fields; those remain private to the
        # downstream frozen evaluator.
        queries = tuple(
            sorted(
                {
                    str(raw.get("category", "")).strip()
                    for raw in raw_objects
                    if isinstance(raw, Mapping)
                    and str(raw.get("category", "")).strip()
                }
            )
        )
        if not queries:
            raise ScalarMapRenderError(f"{scene}/{frame}: annotation has no queries")
        digest = hashlib.sha256(encoded).hexdigest()
        ids = tuple(
            canonical_query_id(scene, frame, index, query)
            for index, query in enumerate(queries)
        )
        all_queries.update(queries)
        result.append(
            FrameBinding(
                frame=frame,
                annotation_path=annotation.resolve(strict=True),
                annotation_sha256=digest,
                camera_name=camera_name,
                resolution_hw=(int(resolution[0]), int(resolution[1])),
                query_texts=queries,
                query_ids=ids,
            )
        )
    return result, tuple(sorted(all_queries))


def _exact_camera_map(dataset: LERFDataset) -> dict[str, tuple[str, np.ndarray]]:
    result: dict[str, tuple[str, np.ndarray]] = {}
    for raw_path, pose in zip(dataset.file_paths, dataset.poses_w2c):
        name = Path(str(raw_path)).name
        if not name:
            raise ScalarMapRenderError("registered camera has an empty basename")
        if name in result:
            raise ScalarMapRenderError(f"duplicate registered camera name: {name!r}")
        matrix = np.asarray(pose, dtype=np.float32)
        if matrix.shape != (4, 4) or not bool(np.isfinite(matrix).all()):
            raise ScalarMapRenderError(f"registered camera {name!r} pose is malformed")
        result[name] = (str(raw_path), matrix)
    if not result:
        raise ScalarMapRenderError("scene has no registered cameras")
    return result


def _load_runtime(binding: SceneBinding, *, label_root: Path, device: torch.device) -> SceneRuntime:
    config = load_config(str(binding.config.path))
    configured_root = Path(str(getattr(config, "scene_root", ""))).resolve(strict=True)
    if configured_root != binding.scene_root:
        raise ScalarMapRenderError(f"{binding.scene}: config scene_root differs")
    if str(getattr(config, "scene", binding.scene)) != binding.scene:
        raise ScalarMapRenderError(f"{binding.scene}: config scene name differs")
    ply = Path(str(getattr(config, "ply_path", ""))).resolve(strict=True)
    if ply != binding.geometry_ply.path:
        raise ScalarMapRenderError(f"{binding.scene}: config PLY path differs")
    model, _codec, _renderer, _sharpener, _refiner, loaded, _hybrid = load_render_pipeline(
        str(binding.config.path),
        str(binding.geometry_checkpoint.path),
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
        expected_checkpoint_sha256=binding.geometry_checkpoint.sha256,
    )
    dataset = LERFDataset(
        scene_root=str(binding.scene_root),
        feature_dir=str(binding.scene_root / ".renderer_requires_no_features"),
        annotation_dir=str(label_root / binding.scene),
        feature_height=int(getattr(loaded, "image_height")),
        feature_width=int(getattr(loaded, "image_width")),
        allow_empty_features=True,
    )
    return SceneRuntime(model=model, config=loaded, dataset=dataset)


def _validate_cache_authority(
    payload: Mapping[str, Any],
    *,
    expected_query_ids: Sequence[str],
    expected_renderer_geometry_sha256: str,
) -> list[dict[str, Any]]:
    authority = _require_mapping(payload.get("authority"), label="cache.authority")
    if authority.get("contract") not in {
        SHARED_AUTHORITY_CONTRACT,
        FP32_SHARED_AUTHORITY_CONTRACT,
    }:
        raise ScalarMapRenderError("query-score cache shared authority differs")
    if authority.get("query_scores_sha256") != _tensor_sha256(payload["query_scores"]):
        raise ScalarMapRenderError("query-score tensor SHA-256 differs from authority")
    query_axis = _require_mapping(authority.get("query_axis"), label="query_axis")
    if query_axis.get("ids") != list(expected_query_ids):
        raise ScalarMapRenderError("query-score authority query axis differs")
    if query_axis.get("order_sha256") != canonical_json_sha256(list(expected_query_ids)):
        raise ScalarMapRenderError("query-score authority query-order SHA-256 differs")
    geometry_axis = _require_mapping(authority.get("geometry_axis"), label="geometry_axis")
    if (
        geometry_axis.get("renderer_geometry_checkpoint_sha256")
        != expected_renderer_geometry_sha256
    ):
        raise ScalarMapRenderError("query-score authority geometry checkpoint differs")
    consumer = _require_mapping(
        _require_mapping(authority.get("consumer_contracts"), label="consumer_contracts").get(
            "lerf2d_scalar_map_renderer"
        ),
        label="lerf2d_scalar_map_renderer",
    )
    if consumer.get("score_semantics") != SCORE_SEMANTICS:
        raise ScalarMapRenderError("query-score renderer semantics differ")
    if consumer.get("tensor_layout_before_render") != "[primitive_row,scale,query]":
        raise ScalarMapRenderError("query-score renderer tensor layout differs")
    if consumer.get("query_text_axis") != list(expected_query_ids):
        raise ScalarMapRenderError("query-score renderer query axis differs")
    constraints = _require_mapping(
        authority.get("calibration_constraints"), label="calibration_constraints"
    )
    if not constraints or set(constraints.values()) != {False}:
        raise ScalarMapRenderError("query-score cache is not calibration-free")
    scales = authority.get("scale_axis")
    if not isinstance(scales, list) or len(scales) != 3:
        raise ScalarMapRenderError("query-score cache must declare three scales")
    scale_records = [dict(_require_mapping(record, label="scale_axis record")) for record in scales]
    if consumer.get("scale_ids") != [record.get("id") for record in scale_records]:
        raise ScalarMapRenderError("query-score renderer scale axis differs")
    for record in scale_records:
        if set(record) != {"id", "value", "unit"} or record.get("unit") != "meter":
            raise ScalarMapRenderError("query-score cache scale record differs")
        value = record.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScalarMapRenderError("query-score cache scale value is malformed")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ScalarMapRenderError("query-score cache scale value is malformed")
        if record.get("id") != str(float(value)):
            raise ScalarMapRenderError("query-score cache native scale ID differs")
    if any(
        float(left["value"]) >= float(right["value"])
        for left, right in zip(scale_records, scale_records[1:])
    ):
        raise ScalarMapRenderError("query-score cache native scale order differs")
    return scale_records


def load_scene_query_scores(
    binding: SceneBinding,
    *,
    model: torch.nn.Module,
    expected_query_ids: Sequence[str],
) -> tuple[torch.Tensor, tuple[str, ...], list[dict[str, Any]]]:
    payload, digest, source = load_torch_mapping(
        binding.query_score_cache.path,
        expected_sha256=binding.query_score_cache.sha256,
        map_location="cpu",
        label=f"{binding.scene} query-score cache",
    )
    if digest != binding.query_score_cache.sha256 or source != binding.query_score_cache.path:
        raise ScalarMapRenderError(f"{binding.scene}: query-score cache identity changed")
    try:
        cache = validate_ours_multiscale_query_score_cache(
            payload,
            expected_xyz=model.get_xyz().detach().cpu(),
            expected_query_ids=expected_query_ids,
            expected_renderer_geometry_checkpoint_sha256=(
                binding.geometry_checkpoint.sha256
            ),
        )
    except ValueError as exc:
        raise ScalarMapRenderError(str(exc)) from exc
    scales = _validate_cache_authority(
        payload,
        expected_query_ids=expected_query_ids,
        expected_renderer_geometry_sha256=binding.geometry_checkpoint.sha256,
    )
    if bool(cache.query_scores[~cache.valid].ne(0).any()):
        raise ScalarMapRenderError("invalid primitive rows must contain exact zero scores")
    return cache.query_scores.contiguous(), cache.query_ids, scales


def render_frame_score_maps(
    renderer: object,
    model: torch.nn.Module,
    pose_w2c: np.ndarray,
    scene_scores_n3q: torch.Tensor,
    query_indices: Sequence[int],
    *,
    height: int,
    width: int,
    device: torch.device,
) -> np.ndarray:
    """Apply the canonical unnormalized scalar-color observation operator."""

    if scene_scores_n3q.ndim != 3 or int(scene_scores_n3q.shape[1]) != 3:
        raise ScalarMapRenderError("scene scores must be [N,3,Q]")
    indices = torch.as_tensor(tuple(query_indices), dtype=torch.long)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ScalarMapRenderError("frame query indices must be non-empty")
    if int(indices.min()) < 0 or int(indices.max()) >= int(scene_scores_n3q.shape[2]):
        raise ScalarMapRenderError("frame query index is outside the cache axis")
    viewmat = torch.from_numpy(np.asarray(pose_w2c, dtype=np.float32)).to(device)
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for scale_index in range(3):
            rows = scene_scores_n3q[:, scale_index, indices].to(
                device=device, dtype=torch.float32
            )
            proxy = GaussianSelectionProxy(model, rows)
            result = renderer.render_feature_rows(
                proxy,
                viewmat,
                rows,
                feature_height=int(height),
                feature_width=int(width),
                alpha_normalize=False,
            )
            feature_map = torch.as_tensor(result["feature_map"]).detach().float().cpu()
            expected = (int(indices.numel()), int(height), int(width))
            if tuple(feature_map.shape) != expected:
                raise ScalarMapRenderError(
                    f"renderer output shape differs: {tuple(feature_map.shape)} vs {expected}"
                )
            if not bool(torch.isfinite(feature_map).all()):
                raise ScalarMapRenderError("renderer produced non-finite scalar scores")
            outputs.append(feature_map)
    return torch.stack(outputs).numpy().astype(np.float32, copy=False)


def _npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return handle.getvalue()


def _source_records(binding: SceneBinding) -> dict[str, Any]:
    return {
        "config": {"path": str(binding.config.path), "sha256": binding.config.sha256},
        "geometry_checkpoint": {
            "path": str(binding.geometry_checkpoint.path),
            "sha256": binding.geometry_checkpoint.sha256,
        },
        "geometry_ply": {
            "path": str(binding.geometry_ply.path),
            "sha256": binding.geometry_ply.sha256,
        },
        "query_score_cache": {
            "path": str(binding.query_score_cache.path),
            "sha256": binding.query_score_cache.sha256,
        },
        "camera_sources": {
            role: {"path": str(value.path), "sha256": value.sha256}
            for role, value in binding.camera_sources.items()
        },
    }


def _recheck_scene_sources(binding: SceneBinding) -> None:
    for label, value in (
        ("config", binding.config),
        ("geometry checkpoint", binding.geometry_checkpoint),
        ("geometry PLY", binding.geometry_ply),
        ("query-score cache", binding.query_score_cache),
        *[(f"camera source {role}", record) for role, record in binding.camera_sources.items()],
    ):
        if sha256_file(value.path) != value.sha256:
            raise ScalarMapRenderError(f"{binding.scene}: {label} changed during rendering")


def render_bundle(
    *,
    authority_path: str | Path,
    authority_sha256: str,
    protocol_freeze: str | Path,
    repo_root: str | Path,
    output_bundle: str | Path,
    device: str | torch.device,
    runtime_loader: Callable[..., SceneRuntime] = _load_runtime,
) -> dict[str, Any]:
    """Render and atomically publish one complete four-scene bundle."""

    repo = Path(repo_root).resolve(strict=True)
    contract = load_frozen_contract(Path(protocol_freeze), repo_root=repo, verify_hashes=True)
    method, authority_source, authority_digest, label_root, bindings = load_render_authority(
        authority_path, expected_sha256=authority_sha256, contract=contract
    )
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ScalarMapRenderError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(torch_device)

    output = Path(os.path.abspath(os.path.expanduser(os.fspath(output_bundle))))
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve(strict=True) / output.name
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output bundle already exists: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        manifest_scenes: dict[str, Any] = {}
        receipt_scenes: dict[str, Any] = {}
        shared_scales: list[dict[str, Any]] | None = None
        total_frames = 0
        total_queries = 0
        for scene in contract.scenes:
            binding = bindings[scene]
            frames, scene_query_axis = _frame_bindings(
                scene,
                label_root=label_root,
                frames=contract.frames_by_scene[scene],
            )
            runtime = runtime_loader(binding, label_root=label_root, device=torch_device)
            config_height = int(getattr(runtime.config, "image_height"))
            config_width = int(getattr(runtime.config, "image_width"))
            if any(frame.resolution_hw != (config_height, config_width) for frame in frames):
                raise ScalarMapRenderError(
                    f"{scene}: annotation resolution differs from canonical camera raster"
                )
            camera_map = _exact_camera_map(runtime.dataset)
            scores, cache_queries, scales = load_scene_query_scores(
                binding, model=runtime.model, expected_query_ids=scene_query_axis
            )
            if shared_scales is None:
                shared_scales = scales
            elif scales != shared_scales:
                raise ScalarMapRenderError("scale identities differ across scenes")
            query_lookup = {query: index for index, query in enumerate(cache_queries)}
            scene_scale_ids = [str(record["id"]) for record in scales]
            renderer = build_mask_renderer(
                runtime.config,
                height=config_height,
                width=config_width,
                device=torch_device,
            )
            scene_frames: dict[str, Any] = {}
            receipt_frames: dict[str, Any] = {}
            for frame in frames:
                camera = camera_map.get(frame.camera_name)
                if camera is None:
                    raise ScalarMapRenderError(
                        f"{scene}/{frame.frame}: exact annotation camera "
                        f"{frame.camera_name!r} is not registered"
                    )
                indices = [query_lookup[query] for query in frame.query_texts]
                score_map = render_frame_score_maps(
                    renderer,
                    runtime.model,
                    camera[1],
                    scores,
                    indices,
                    height=config_height,
                    width=config_width,
                    device=torch_device,
                )
                relative = Path("maps") / scene / f"{frame.frame}.npy"
                target = staging / relative
                encoded = _npy_bytes(score_map)
                write_bytes_noclobber(target, encoded)
                map_sha = hashlib.sha256(encoded).hexdigest()
                shape = [3, len(frame.query_texts), config_height, config_width]
                scene_frames[frame.frame] = {
                    "annotation_sha256": frame.annotation_sha256,
                    "camera_name": frame.camera_name,
                    "query_texts": list(frame.query_texts),
                    "query_ids": list(frame.query_ids),
                    "map_file": relative.as_posix(),
                    "map_sha256": map_sha,
                    "map_shape_lqhw": shape,
                    "map_resolution_hw": [config_height, config_width],
                    "scale_ids": scene_scale_ids,
                }
                receipt_frames[frame.frame] = {
                    "annotation": {
                        "path": str(frame.annotation_path),
                        "sha256": frame.annotation_sha256,
                    },
                    "camera_name": frame.camera_name,
                    "registered_camera_path": camera[0],
                    "pose_w2c_sha256": hashlib.sha256(
                        np.asarray(camera[1], dtype="<f4").tobytes(order="C")
                    ).hexdigest(),
                    "query_cache_indices": indices,
                    "query_ids": list(frame.query_ids),
                    "map_sha256": map_sha,
                    "map_shape_lqhw": shape,
                }
                total_frames += 1
                total_queries += len(frame.query_texts)
            _recheck_scene_sources(binding)
            manifest_scenes[scene] = {"frames": scene_frames}
            receipt_scenes[scene] = {
                "sources": _source_records(binding),
                "query_text_axis": list(cache_queries),
                "query_text_axis_sha256": canonical_json_sha256(list(cache_queries)),
                "geometry_xyz_sha256": _tensor_sha256(runtime.model.get_xyz().detach().cpu()),
                "frames": receipt_frames,
            }
        if total_frames != contract.labelled_frames or total_queries != contract.queries:
            raise ScalarMapRenderError(
                f"rendered cohort differs: frames={total_frames}, queries={total_queries}"
            )
        assert shared_scales is not None
        implementation = Path(__file__).resolve(strict=True)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": RECEIPT_ARTIFACT_TYPE,
            "status": "complete_frozen_query_score_render",
            "method": method,
            "render_operator": RENDER_OPERATOR,
            "render_semantics": {
                "alpha_composited": True,
                "alpha_normalized": False,
                "background_score": 0.0,
                "text_encoded": False,
                "scale_reduced": False,
                "threshold_applied": False,
                "map_resized_or_resampled": False,
                "benchmark_masks_opened": False,
                "benchmark_annotation_metadata_opened": True,
                "benchmark_segmentation_or_bbox_fields_accessed": False,
            },
            "protocol_freeze": {
                "path": contract.freeze_path,
                "sha256": contract.freeze_sha256,
                "freeze_id": contract.freeze_id,
            },
            "render_authority": {
                "path": str(authority_source),
                "sha256": authority_digest,
            },
            "renderer_implementation": file_record(implementation),
            "device": str(torch_device),
            "cohort": {"frames": total_frames, "queries": total_queries},
            "scenes": receipt_scenes,
        }
        receipt_path = staging / "renderer_receipt.json"
        write_frozen_json(receipt_path, receipt)
        receipt_digest = sha256_file(receipt_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": EVALUATOR_ARTIFACT_TYPE,
            "method": method,
            "score_semantics": SCORE_SEMANTICS,
            "canonical_task_id": CANONICAL_TASK_ID,
            "registry_row": EXPECTED_REGISTRY_ROW,
            "protocol_freeze": {
                "freeze_id": contract.freeze_id,
                "sha256": contract.freeze_sha256,
            },
            "scales": shared_scales,
            "source_artifacts": [
                {
                    "role": "renderer_receipt",
                    "path": "renderer_receipt.json",
                    "sha256": receipt_digest,
                }
            ],
            "scenes": manifest_scenes,
        }
        manifest_path = staging / "manifest.json"
        write_frozen_json(manifest_path, manifest)
        manifest_digest = sha256_file(manifest_path)
        fsync_directory(staging)
        os.rename(staging, output)
        fsync_directory(output.parent)
        return {
            "status": "complete_frozen_query_score_render",
            "bundle": str(output),
            "manifest": str(output / "manifest.json"),
            "manifest_sha256": manifest_digest,
            "renderer_receipt_sha256": receipt_digest,
            "frames": total_frames,
            "queries": total_queries,
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-authority", required=True)
    parser.add_argument("--scene-authority-sha256", required=True)
    parser.add_argument(
        "--protocol-freeze",
        default=str(repo / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml"),
    )
    parser.add_argument("--repo-root", default=str(repo))
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = render_bundle(
        authority_path=args.scene_authority,
        authority_sha256=args.scene_authority_sha256,
        protocol_freeze=args.protocol_freeze,
        repo_root=args.repo_root,
        output_bundle=args.output_bundle,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
