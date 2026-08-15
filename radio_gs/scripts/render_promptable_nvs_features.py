#!/usr/bin/env python3
"""Render GaussFM features for protocol-locked NVOS/SPIn-NeRF views.

This renderer is intentionally narrower than the general downstream renderer:
it obtains cameras from the COLMAP sparse model, renders only the declared
prompt/evaluation cameras, and never opens an RGB image or a segmentation mask.
RGB-guided refiners are rejected so an NVOS target cannot re-enter at query
time through a screen-space postprocessor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.data.lerf_dataset import _parse_colmap_sparse
from radio_gs.data.promptable_nvs_manifest import (
    ManifestError,
    validate_manifest as validate_dataset_manifest,
)


class PromptableRenderError(ValueError):
    """Raised when rendering would violate or ambiguously map the protocol."""


CAMERA_MAP_SCHEMA_VERSION = 1
_SPLIT_PREFIX_RE = re.compile(r"^[01]_(?P<camera>.+)$")
_CANONICAL_IMAGE_INDEX_RE = re.compile(r"^image(?P<index>[0-9]{3})$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_float32_rows(values: torch.Tensor) -> str:
    array = (
        values.detach()
        .float()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_canonical_feature_source(
    payload: Mapping[str, Any],
    *,
    num_gaussians: int,
    geometry_xyz_sha256: str,
) -> None:
    """Reject a canonical field that is supervised or belongs to other rows."""

    if int(payload.get("schema_version", -1)) != 1:
        raise PromptableRenderError("Canonical feature source is not schema-v1")
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise PromptableRenderError("Canonical feature source lacks architecture")
    if int(architecture.get("num_gaussians", -1)) != int(num_gaussians):
        raise PromptableRenderError("Canonical field and geometry row counts differ")
    if int(architecture.get("feature_dim", -1)) != 1280:
        raise PromptableRenderError("Canonical NVOS field must decode 1280d RADIO")
    fingerprint = payload.get("geometry_fingerprint")
    if (
        not isinstance(fingerprint, Mapping)
        or str(fingerprint.get("xyz_sha256") or "") != geometry_xyz_sha256
    ):
        raise PromptableRenderError("Canonical field and geometry xyz rows differ")
    if payload.get("benchmark_masks_opened") is not False:
        raise PromptableRenderError("Canonical field lacks benchmark-mask exclusion authority")
    if payload.get("text_queries_opened") is not False:
        raise PromptableRenderError("Canonical field lacks text-query exclusion authority")


def validate_factorized_feature_source(
    payload: Mapping[str, Any],
    *,
    num_gaussians: int,
    geometry_xyz_sha256: str,
) -> None:
    """Validate a complete schema-v2 Method-v1 field for feature-only rendering."""

    from radio_gs.five_benchmark_method_v1 import validate_complete_field_payload

    try:
        validate_complete_field_payload(payload)
    except ValueError as error:
        raise PromptableRenderError(str(error)) from error
    architecture = payload["architecture"]
    fingerprint = payload["geometry_fingerprint"]
    if (
        int(architecture.get("num_gaussians", -1)) != int(num_gaussians)
        or int(fingerprint.get("num_gaussians", -1)) != int(num_gaussians)
        or str(fingerprint.get("xyz_sha256", "")) != geometry_xyz_sha256
    ):
        raise PromptableRenderError(
            "Factorized Method-v1 field and geometry rows differ"
        )


def _safe_component(value: str, *, role: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "\\" in value:
        raise PromptableRenderError(f"Unsafe {role}: {value!r}")
    return value


def build_rgb_to_colmap_mapping(
    rgb_paths: Sequence[str | Path],
    colmap_file_paths: Sequence[str | Path],
    *,
    scene_id: str,
) -> list[dict[str, Any]]:
    """Build the only permitted RGB-to-camera association.

    Matching is deliberately narrow and ordered: exact basename stem first,
    then removal of one official ``0_``/``1_`` split prefix, and finally a
    strict ``imageNNN`` canonical index into the lexicographically sorted
    COLMAP camera list.  There is no fuzzy/nearest-name fallback.
    """

    scene_id = _safe_component(str(scene_id), role="scene_id")
    rgb_items = sorted(
        (Path(path).expanduser().resolve() for path in rgb_paths),
        key=lambda path: path.name,
    )
    colmap_items = [Path(str(path)) for path in colmap_file_paths]
    if not rgb_items:
        raise PromptableRenderError(f"Scene {scene_id} has no RGB files to map")
    if not colmap_items:
        raise PromptableRenderError(f"Scene {scene_id} has no COLMAP cameras to map")

    rgb_by_stem: dict[str, Path] = {}
    for path in rgb_items:
        if path.stem in rgb_by_stem:
            raise PromptableRenderError(
                f"Scene {scene_id} has duplicate RGB basename stem {path.stem!r}: "
                f"{rgb_by_stem[path.stem]} and {path}"
            )
        rgb_by_stem[path.stem] = path

    colmap_by_stem: dict[str, Path] = {}
    for path in colmap_items:
        if path.stem in colmap_by_stem:
            raise PromptableRenderError(
                f"Scene {scene_id} has duplicate COLMAP basename stem {path.stem!r}: "
                f"{colmap_by_stem[path.stem]} and {path}"
            )
        colmap_by_stem[path.stem] = path
    lexicographic_colmap = sorted(colmap_items, key=lambda path: path.name)

    records: list[dict[str, Any]] = []
    used_colmap: dict[str, str] = {}
    for rgb_rank, rgb_path in enumerate(rgb_items):
        rgb_stem = rgb_path.stem
        canonical_index: int | None = None
        if rgb_stem in colmap_by_stem:
            colmap_path = colmap_by_stem[rgb_stem]
            rule = "exact_case_sensitive_basename_stem"
        else:
            prefix_match = _SPLIT_PREFIX_RE.fullmatch(rgb_stem)
            stripped = prefix_match.group("camera") if prefix_match is not None else None
            if stripped is not None and stripped in colmap_by_stem:
                colmap_path = colmap_by_stem[stripped]
                rule = "strip_official_0_or_1_split_prefix_then_exact_stem"
            else:
                index_match = _CANONICAL_IMAGE_INDEX_RE.fullmatch(rgb_stem)
                if index_match is None:
                    raise PromptableRenderError(
                        f"Scene {scene_id} cannot map RGB {rgb_path.name!r}; permitted "
                        "rules are exact stem, one 0_/1_ prefix strip, or strict imageNNN "
                        "canonical index. No nearest-name guessing is allowed."
                    )
                canonical_index = int(index_match.group("index"))
                if canonical_index >= len(lexicographic_colmap):
                    raise PromptableRenderError(
                        f"Scene {scene_id} RGB {rgb_path.name!r} requests canonical "
                        f"COLMAP index {canonical_index}, but only "
                        f"{len(lexicographic_colmap)} cameras exist"
                    )
                colmap_path = lexicographic_colmap[canonical_index]
                rule = "imageNNN_canonical_index_to_lexicographic_colmap_camera"

        colmap_stem = colmap_path.stem
        if colmap_stem in used_colmap:
            raise PromptableRenderError(
                f"Scene {scene_id} maps both RGB {used_colmap[colmap_stem]!r} and "
                f"{rgb_stem!r} to COLMAP camera {colmap_stem!r}"
            )
        used_colmap[colmap_stem] = rgb_stem
        records.append(
            {
                "rgb_rank": rgb_rank,
                "rgb_camera_name": rgb_stem,
                "rgb_path": str(rgb_path),
                "colmap_camera_name": colmap_stem,
                "colmap_file_path": str(colmap_path),
                "match_rule": rule,
                "canonical_index": canonical_index,
            }
        )
    return records


def validate_locked_camera_mapping(
    payload: Mapping[str, Any],
    *,
    scene_id: str,
) -> list[dict[str, Any]]:
    """Validate a queue-produced mapping without opening any RGB image."""

    if int(payload.get("schema_version", -1)) != CAMERA_MAP_SCHEMA_VERSION:
        raise PromptableRenderError("Unsupported or missing camera-map schema_version")
    if str(payload.get("scene_id") or "") != str(scene_id):
        raise PromptableRenderError(
            f"Camera map scene {payload.get('scene_id')!r} does not match {scene_id!r}"
        )
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise PromptableRenderError("Camera map must contain a non-empty records list")

    allowed_rules = {
        "exact_case_sensitive_basename_stem",
        "strip_official_0_or_1_split_prefix_then_exact_stem",
        "imageNNN_canonical_index_to_lexicographic_colmap_camera",
    }
    records: list[dict[str, Any]] = []
    rgb_names: set[str] = set()
    colmap_names: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise PromptableRenderError(f"Camera-map record {index} is not an object")
        rgb_name = _safe_component(
            str(raw.get("rgb_camera_name") or ""), role="rgb_camera_name"
        )
        colmap_name = _safe_component(
            str(raw.get("colmap_camera_name") or ""), role="colmap_camera_name"
        )
        rgb_path = Path(str(raw.get("rgb_path") or "")).expanduser()
        colmap_path = Path(str(raw.get("colmap_file_path") or ""))
        rule = str(raw.get("match_rule") or "")
        if not rgb_path.is_absolute():
            raise PromptableRenderError(
                f"Camera-map RGB path must be absolute for {rgb_name!r}: {rgb_path}"
            )
        if rgb_path.stem != rgb_name:
            raise PromptableRenderError(
                f"Camera-map RGB stem mismatch: {rgb_path.stem!r} != {rgb_name!r}"
            )
        if colmap_path.stem != colmap_name:
            raise PromptableRenderError(
                f"Camera-map COLMAP stem mismatch: {colmap_path.stem!r} != {colmap_name!r}"
            )
        if rule not in allowed_rules:
            raise PromptableRenderError(
                f"Camera-map record {rgb_name!r} has unsupported match_rule {rule!r}"
            )
        if rgb_name in rgb_names or colmap_name in colmap_names:
            raise PromptableRenderError(
                f"Camera map is not one-to-one at RGB {rgb_name!r} / "
                f"COLMAP {colmap_name!r}"
            )
        rgb_names.add(rgb_name)
        colmap_names.add(colmap_name)
        records.append(dict(raw))

    declared_rgb_to_colmap = payload.get("rgb_camera_to_colmap_camera")
    if declared_rgb_to_colmap is not None:
        if not isinstance(declared_rgb_to_colmap, Mapping):
            raise PromptableRenderError("rgb_camera_to_colmap_camera must be an object")
        expected = {
            str(record["rgb_camera_name"]): str(record["colmap_camera_name"])
            for record in records
        }
        if {str(key): str(value) for key, value in declared_rgb_to_colmap.items()} != expected:
            raise PromptableRenderError(
                "rgb_camera_to_colmap_camera disagrees with camera-map records"
            )
    return records


def validate_feature_only_config(config: Any) -> None:
    """Reject every known RGB-dependent path in the reusable-field track."""

    boolean_forbidden = (
        "use_refiner",
        "refiner_rgb_guide",
        "self_guided",
        "train_sh",
    )
    enabled = [name for name in boolean_forbidden if bool(getattr(config, name, False))]
    if enabled:
        raise PromptableRenderError(
            "Reusable feature-field rendering forbids RGB/refiner options: "
            + ", ".join(enabled)
        )
    if float(getattr(config, "rgb_loss_weight", 0.0)) != 0.0:
        raise PromptableRenderError("Reusable feature-field track requires rgb_loss_weight=0")
    populated_rgb_dirs = [
        name
        for name in ("rgb_dir", "val_rgb_dir")
        if str(getattr(config, name, "") or "").strip()
    ]
    if populated_rgb_dirs:
        raise PromptableRenderError(
            "Reusable feature-field config must not expose query RGB directories: "
            + ", ".join(populated_rgb_dirs)
        )


def _raw_scene(manifest: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    matches = [
        scene
        for scene in manifest.get("scenes", [])
        if isinstance(scene, Mapping) and str(scene.get("scene_id")) == scene_id
    ]
    if len(matches) != 1:
        raise PromptableRenderError(
            f"Expected exactly one manifest scene {scene_id!r}; found {len(matches)}"
        )
    return matches[0]


def resolve_protocol_views(
    manifest: Mapping[str, Any],
    *,
    scene_id: str,
    scene_root: str | Path,
    camera_mapping: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve declared RGB cameras through the queue-locked camera map."""

    try:
        normalized = validate_dataset_manifest(manifest, check_files=False)
    except ManifestError as error:
        raise PromptableRenderError(str(error)) from error
    normalized_scene = next(
        (scene for scene in normalized["scenes"] if scene["scene_id"] == scene_id),
        None,
    )
    if normalized_scene is None:
        raise PromptableRenderError(f"Scene {scene_id!r} is not in the locked manifest")
    if normalized_scene["calibration_frame_ids"]:
        raise PromptableRenderError("Calibration views are forbidden in this fixed-threshold track")

    raw_scene = _raw_scene(manifest, scene_id)
    raw_frames = raw_scene.get("frames", [])
    if isinstance(raw_frames, Mapping):
        frames = {str(key): value for key, value in raw_frames.items()}
    else:
        frames = {
            str(frame.get("frame_id")): frame
            for frame in raw_frames
            if isinstance(frame, Mapping)
        }

    colmap = _parse_colmap_sparse(Path(scene_root).expanduser().resolve())
    by_stem: dict[str, tuple[str, np.ndarray]] = {}
    for file_path, c2w in zip(colmap["file_paths"], colmap["c2w_list"]):
        stem = Path(str(file_path)).stem
        if stem in by_stem:
            raise PromptableRenderError(
                f"Duplicate COLMAP camera basename stem {stem!r} in {scene_root}"
            )
        by_stem[stem] = (str(file_path), np.asarray(c2w, dtype=np.float32))

    mapping_records = validate_locked_camera_mapping(
        camera_mapping,
        scene_id=scene_id,
    )
    mapping_by_rgb = {
        str(record["rgb_camera_name"]): record for record in mapping_records
    }
    missing_locked_colmap = sorted(
        {
            str(record["colmap_camera_name"])
            for record in mapping_records
        }
        - set(by_stem)
    )
    if missing_locked_colmap:
        raise PromptableRenderError(
            f"Scene {scene_id} locked cameras are absent from COLMAP: "
            f"{missing_locked_colmap}"
        )

    frame_ids = list(
        dict.fromkeys(
            list(normalized_scene["prompt_frame_ids"])
            + list(normalized_scene["evaluation_frame_ids"])
        )
    )
    resolved: list[dict[str, Any]] = []
    used_cameras: set[str] = set()
    for frame_id in frame_ids:
        raw_frame = frames.get(str(frame_id))
        if not isinstance(raw_frame, Mapping):
            raise PromptableRenderError(f"Scene {scene_id}/{frame_id} lacks raw frame metadata")
        camera_name = _safe_component(
            str(raw_frame.get("camera_name") or ""), role="camera_name"
        )
        if camera_name in used_cameras:
            raise PromptableRenderError(
                f"Scene {scene_id} maps multiple protocol frames to camera {camera_name}"
            )
        mapping_record = mapping_by_rgb.get(camera_name)
        if mapping_record is None:
            raise PromptableRenderError(
                f"Scene {scene_id}/{frame_id} RGB camera {camera_name!r} is absent "
                "from the locked RGB-to-COLMAP map"
            )
        used_cameras.add(camera_name)
        colmap_camera_name = str(mapping_record["colmap_camera_name"])
        colmap_path, c2w = by_stem[colmap_camera_name]
        locked_colmap_path = Path(str(mapping_record["colmap_file_path"]))
        if Path(colmap_path) != locked_colmap_path:
            raise PromptableRenderError(
                f"Scene {scene_id}/{frame_id} locked COLMAP path changed for "
                f"{colmap_camera_name!r}: {locked_colmap_path} != {colmap_path}"
            )
        resolved.append(
            {
                "frame_id": str(frame_id),
                "camera_name": camera_name,
                "colmap_camera_name": colmap_camera_name,
                "camera_match_rule": str(mapping_record["match_rule"]),
                "role": (
                    "prompt"
                    if frame_id in normalized_scene["prompt_frame_ids"]
                    else "evaluation"
                ),
                "colmap_file_path": colmap_path,
                "w2c": np.linalg.inv(c2w).astype(np.float32),
            }
        )
    return resolved


def _atomic_torch_save(path: Path, value: torch.Tensor, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Rendered feature already exists (use --overwrite): {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@torch.inference_mode()
def render_protocol_scene(
    manifest_path: str | Path,
    *,
    scene_id: str,
    camera_map_path: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    canonical_field_checkpoint_path: str | Path | None = None,
    canonical_field_checkpoint_schema: str = "canonical-v1",
    expected_canonical_field_checkpoint_sha256: str = "",
    output_dir: str | Path,
    device: str = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render one scene while keeping evaluation masks and target RGB unopened."""

    manifest_source = Path(manifest_path).expanduser().resolve()
    camera_map_source = Path(camera_map_path).expanduser().resolve()
    config_source = Path(config_path).expanduser().resolve()
    checkpoint_source = Path(checkpoint_path).expanduser().resolve()
    canonical_field_source = (
        Path(canonical_field_checkpoint_path).expanduser().resolve()
        if canonical_field_checkpoint_path is not None
        else None
    )
    if canonical_field_checkpoint_schema not in {"canonical-v1", "factorized-v2"}:
        raise PromptableRenderError(
            "canonical_field_checkpoint_schema must be canonical-v1 or factorized-v2"
        )
    for path, label in (
        (manifest_source, "manifest"),
        (camera_map_source, "camera map"),
        (config_source, "config"),
        (checkpoint_source, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if canonical_field_source is not None and not canonical_field_source.is_file():
        raise FileNotFoundError(
            f"canonical field checkpoint not found: {canonical_field_source}"
        )
    if (
        canonical_field_source is not None
        and expected_canonical_field_checkpoint_sha256
        and _sha256(canonical_field_source)
        != str(expected_canonical_field_checkpoint_sha256)
    ):
        raise PromptableRenderError("Canonical field checkpoint SHA256 differs")
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    camera_mapping = json.loads(camera_map_source.read_text(encoding="utf-8"))
    normalized = validate_dataset_manifest(manifest, check_files=False)
    protocol_hash = str(normalized["protocol_hash"])
    config = load_config(str(config_source))
    validate_feature_only_config(config)

    scene_id = _safe_component(scene_id, role="scene_id")
    configured_scene = str(getattr(config, "scene", ""))
    if configured_scene != scene_id:
        raise PromptableRenderError(
            f"Config scene {configured_scene!r} does not match requested scene {scene_id!r}"
        )
    scene_root = Path(str(getattr(config, "scene_root", ""))).expanduser().resolve()
    views = resolve_protocol_views(
        manifest,
        scene_id=scene_id,
        scene_root=scene_root,
        camera_mapping=camera_mapping,
    )

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise PromptableRenderError(f"CUDA device requested but unavailable: {device}")

    canonical_field = None
    canonical_payload: Mapping[str, Any] | None = None
    if canonical_field_source is None:
        # eval_rendered historically keeps its device as a module global.  Set it
        # before loading so every model/checkpoint tensor lands on the requested
        # device without adding an RGB-dependent evaluation path.
        from radio_gs.scripts import eval_rendered

        eval_rendered.device = torch_device
        model, codec, renderer, sharpener, refiner, _, is_hybrid = (
            eval_rendered.load_model_and_render(
                str(config_source), str(checkpoint_source)
            )
        )
        if refiner is not None:
            raise PromptableRenderError(
                "Feature-only protocol unexpectedly constructed a refiner"
            )
    else:
        from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline

        model, _codec, renderer, _sharpener, refiner, _, _is_hybrid = (
            load_render_pipeline(
                str(config_source),
                str(checkpoint_source),
                torch_device,
                strict_checkpoint_contract=True,
                load_ply_rgb_features=False,
            )
        )
        if refiner is not None:
            raise PromptableRenderError(
                "Canonical protocol unexpectedly constructed a screen refiner"
            )
        geometry_xyz_sha256 = _sha256_float32_rows(model.get_xyz())
        if canonical_field_checkpoint_schema == "factorized-v2":
            from radio_gs.field import load_factorized_canonical_field_checkpoint

            canonical_field, canonical_payload, _factorized_signature = (
                load_factorized_canonical_field_checkpoint(
                    canonical_field_source,
                    map_location="cpu",
                    expected_sha256=(
                        expected_canonical_field_checkpoint_sha256 or None
                    ),
                )
            )
            validate_factorized_feature_source(
                canonical_payload,
                num_gaussians=int(model.get_xyz().shape[0]),
                geometry_xyz_sha256=geometry_xyz_sha256,
            )
        else:
            from radio_gs.field import load_canonical_field_checkpoint

            canonical_field, canonical_payload = load_canonical_field_checkpoint(
                canonical_field_source, map_location="cpu"
            )
            validate_canonical_feature_source(
                canonical_payload,
                num_gaussians=int(model.get_xyz().shape[0]),
                geometry_xyz_sha256=geometry_xyz_sha256,
            )
        canonical_field = canonical_field.to(torch_device).eval()

    output_root = Path(output_dir).expanduser().resolve()
    outputs: list[dict[str, Any]] = []
    for view in views:
        pose = torch.from_numpy(view["w2c"][None]).to(torch_device)
        if canonical_field is None:
            result = renderer.render_features_batch(model, pose)
            rendered = sharpener(result["feature_map"])
            if is_hybrid:
                rendered = eval_rendered._hybrid_decode(
                    model, rendered, result, pose, renderer.K
                )
            decoded = codec.decoder(rendered).squeeze(0).float().cpu()
        else:
            from radio_gs.rendering.coefficient_renderer import render_canonical_radio

            result = render_canonical_radio(
                renderer,
                model,
                canonical_field,
                pose.squeeze(0),
                feature_height=int(getattr(config, "feature_height")),
                feature_width=int(getattr(config, "feature_width")),
                use_reliability=False,
            )
            decoded = result["feature_map"].float().cpu()
        destination = output_root / scene_id / f"{view['camera_name']}.pt"
        _atomic_torch_save(destination, decoded, overwrite=overwrite)
        outputs.append(
            {
                "frame_id": view["frame_id"],
                "camera_name": view["camera_name"],
                "colmap_camera_name": view["colmap_camera_name"],
                "camera_match_rule": view["camera_match_rule"],
                "role": view["role"],
                "feature_path": str(destination),
                "feature_sha256": _sha256(destination),
                "shape": list(decoded.shape),
                "dtype": "float32",
            }
        )

    report = {
        "schema_version": 1,
        "kind": "promptable_nvs_gaussfm_render",
        "protocol_hash": protocol_hash,
        "scene_id": scene_id,
        "manifest": str(manifest_source),
        "manifest_file_sha256": _sha256(manifest_source),
        "camera_map": str(camera_map_source),
        "camera_map_sha256": _sha256(camera_map_source),
        "config": str(config_source),
        "config_sha256": _sha256(config_source),
        "checkpoint": str(checkpoint_source),
        "checkpoint_sha256": _sha256(checkpoint_source),
        "render_mode": (
            "factorized_v2_affine_normalized_splat"
            if canonical_field_source is not None
            and canonical_field_checkpoint_schema == "factorized-v2"
            else "canonical_mpr_v3_affine_normalized_splat"
            if canonical_field_source is not None
            else "legacy_reusable_hcd_screen_field"
        ),
        "canonical_field_checkpoint_schema": (
            canonical_field_checkpoint_schema
            if canonical_field_source is not None
            else None
        ),
        "canonical_field_checkpoint": (
            str(canonical_field_source) if canonical_field_source is not None else None
        ),
        "canonical_field_checkpoint_sha256": (
            _sha256(canonical_field_source)
            if canonical_field_source is not None
            else None
        ),
        "canonical_field_geometry_fingerprint": (
            dict(canonical_payload.get("geometry_fingerprint", {}))
            if canonical_payload is not None
            else None
        ),
        "canonical_render_contract": (
            {
                "normalized_splat": True,
                "affine_decode_after_splat": True,
                "reliability_splat": False,
                "screen_refiner": False,
            }
            if canonical_field_source is not None
            else None
        ),
        "feature_layout": "chw",
        "outputs": outputs,
        "safety": {
            "rgb_files_opened": False,
            "segmentation_masks_opened": False,
            "evaluation_ground_truth_opened": False,
            "rgb_refiner_used": False,
            "camera_mapping": "queue_locked_rgb_to_colmap_map",
        },
    }
    report_path = output_root / scene_id / "render_manifest.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["render_manifest"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--camera-map", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--canonical-field-checkpoint",
        type=Path,
        default=None,
        help=(
            "Use a canonical-mpr-v3 field as the sole feature source; --checkpoint "
            "then supplies only its row-aligned frozen geometry carrier."
        ),
    )
    parser.add_argument(
        "--canonical-field-checkpoint-schema",
        choices=("canonical-v1", "factorized-v2"),
        default="canonical-v1",
    )
    parser.add_argument("--expected-canonical-field-checkpoint-sha256", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = render_protocol_scene(
        args.manifest,
        scene_id=args.scene_id,
        camera_map_path=args.camera_map,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        canonical_field_checkpoint_path=args.canonical_field_checkpoint,
        canonical_field_checkpoint_schema=args.canonical_field_checkpoint_schema,
        expected_canonical_field_checkpoint_sha256=(
            args.expected_canonical_field_checkpoint_sha256
        ),
        output_dir=args.output_dir,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "scene_id": report["scene_id"],
                "protocol_hash": report["protocol_hash"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "render_manifest": report["render_manifest"],
                "num_views": len(report["outputs"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
