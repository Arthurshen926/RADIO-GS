#!/usr/bin/env python3
"""Prepare (but never launch) protocol-locked GaussFM benchmark jobs.

The ``prepare`` command validates an official NVOS or SPIn-NeRF manifest,
materializes exact COLMAP pose/index files and generated config overlays, then
writes a JSON queue plus an opt-in shell runner.  Preparing the queue is CPU
only.  The generated runner refuses to execute until ``ALLOW_GPU=1`` is set.

The ``evaluate`` command is the separate, explicit scoring boundary.  It is
the only command in this file that opens evaluation ground-truth masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from radio_gs.config import load_config
from radio_gs.data.lerf_dataset import _parse_colmap_sparse
from radio_gs.data.promptable_nvs_manifest import (
    ManifestError,
    validate_manifest as validate_dataset_manifest,
)
from radio_gs.data.view_split import select_image_indices
from radio_gs.evaluation.promptable_segmentation import evaluate_manifest
from radio_gs.scripts.extract_radio_features import (
    _collect_image_paths,
    _compute_scaled_radio_resolution,
)
from radio_gs.scripts.render_promptable_nvs_features import (
    CAMERA_MAP_SCHEMA_VERSION,
    PromptableRenderError,
    build_rgb_to_colmap_mapping,
    validate_feature_only_config,
)


PENDING_SHA256 = "PENDING_SHA256_AFTER_STAGE_COMPLETION"
QUEUE_SCHEMA_VERSION = 1
NVOS_TRACK = "nvos_official_scribble_reusable_feature_field"
SPIN_FULL_MASK_TRACK = "spin_full_reference_mask_propagation"


class QueuePreparationError(ValueError):
    """Raised when a queue cannot be generated without guessing or leakage."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_component(value: str, *, role: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "\\" in value:
        raise QueuePreparationError(f"Unsafe {role}: {value!r}")
    return value


def _scene_map(source: str | Path | Mapping[str, str] | None) -> dict[str, Path]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        payload = source
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scene-root map not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise QueuePreparationError("scene_root_map must be a JSON object")
    return {str(key): Path(str(value)).expanduser().resolve() for key, value in payload.items()}


def _has_colmap(root: Path) -> bool:
    sparse = root / "sparse" / "0"
    return (sparse / "cameras.bin").is_file() and (sparse / "images.bin").is_file()


def _discover_scene_root(
    raw_scene: Mapping[str, Any],
    *,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not _has_colmap(root):
            raise QueuePreparationError(
                f"Explicit scene root lacks sparse/0 cameras.bin+images.bin: {root}"
            )
        return root

    rgb_directory = Path(str(raw_scene.get("rgb_directory") or "")).expanduser().resolve()
    if not rgb_directory.is_dir():
        raise QueuePreparationError(
            f"Scene {raw_scene.get('scene_id')} has no exact rgb_directory: {rgb_directory}"
        )
    candidates: list[Path] = []
    current = rgb_directory
    for _ in range(5):
        if _has_colmap(current) and current not in candidates:
            candidates.append(current)
        if current.parent == current:
            break
        current = current.parent
    if len(candidates) != 1:
        raise QueuePreparationError(
            f"Could not uniquely infer COLMAP scene root for {raw_scene.get('scene_id')} "
            f"from {rgb_directory}; candidates={[str(path) for path in candidates]}. "
            "Supply an explicit --scene-root-map."
        )
    return candidates[0]


def _raw_scene(manifest: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in manifest.get("scenes", [])
        if isinstance(item, Mapping) and str(item.get("scene_id")) == scene_id
    ]
    if len(matches) != 1:
        raise QueuePreparationError(
            f"Expected exactly one raw scene {scene_id!r}; found {len(matches)}"
        )
    return matches[0]


def _raw_frames(raw_scene: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = raw_scene.get("frames", [])
    if isinstance(raw, Mapping):
        frames = {str(key): value for key, value in raw.items() if isinstance(value, Mapping)}
    else:
        frames = {
            str(value.get("frame_id")): value
            for value in raw
            if isinstance(value, Mapping)
        }
    return frames


def _protocol_exclusions(
    *,
    benchmark: str,
    normalized_scene: Mapping[str, Any],
    raw_scene: Mapping[str, Any],
) -> tuple[str, ...]:
    if benchmark == "spin_nerf":
        return ()
    if benchmark != "nvos":
        raise QueuePreparationError(f"Unsupported benchmark: {benchmark!r}")
    frames = _raw_frames(raw_scene)
    stems: list[str] = []
    for frame_id in normalized_scene["evaluation_frame_ids"]:
        raw_frame = frames.get(str(frame_id))
        if raw_frame is None or not raw_frame.get("camera_name"):
            raise QueuePreparationError(
                f"NVOS {normalized_scene['scene_id']}/{frame_id} lacks camera_name"
            )
        stems.append(str(raw_frame["camera_name"]))
    if not stems:
        raise QueuePreparationError(f"NVOS {normalized_scene['scene_id']} has no target")
    return tuple(dict.fromkeys(stems))


def _track_metadata(benchmark: str, prompt_type: str) -> dict[str, Any]:
    if benchmark == "nvos" and prompt_type == "fixed_positive_negative_scribbles":
        return {
            "id": NVOS_TRACK,
            "prompt": "official_fixed_positive_negative_scribbles",
            "saga_same_prompt_main_table_eligible": False,
            "comparison_note": (
                "NVOS follows the official scribble task. Published baselines still require "
                "their own target-view/training-policy provenance before same-protocol labeling."
            ),
        }
    if benchmark == "spin_nerf" and prompt_type == "single_reference_binary_mask":
        return {
            "id": SPIN_FULL_MASK_TRACK,
            "prompt": "full_binary_mask_on_first_reference_frame",
            "saga_same_prompt_main_table_eligible": False,
            "comparison_note": (
                "This is a full-reference-mask propagation track. SAGA final inference uses "
                "2D point prompts, and SPIn-NeRF starts from sparse positive/negative clicks; "
                "do not place these scores in a same-prompt SAGA main comparison unless the "
                "exact Appendix-A.2 point set is separately frozen."
            ),
            "required_for_point_prompt_track": "frozen_exact_positive_negative_point_coordinates",
        }
    raise QueuePreparationError(
        f"No queue track is registered for benchmark={benchmark!r}, prompt_type={prompt_type!r}"
    )


def _colmap_by_stem(scene_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    parsed = _parse_colmap_sparse(scene_root)
    lookup: dict[str, dict[str, Any]] = {}
    for file_path, c2w in zip(parsed["file_paths"], parsed["c2w_list"]):
        path = Path(str(file_path))
        stem = path.stem
        if stem in lookup:
            raise QueuePreparationError(
                f"Duplicate COLMAP basename stem {stem!r} in {scene_root}"
            )
        absolute = path if path.is_absolute() else scene_root / path
        lookup[stem] = {
            "file_path": str(path),
            "absolute_path": absolute.resolve(),
            "c2w": np.asarray(c2w, dtype=np.float32),
        }
    return lookup, parsed


def _training_feature_index(
    image_path: Path,
    *,
    source_rank: int,
) -> int:
    # Queue commands force extract_radio_features --frame-id-mode source_rank.
    # Keeping this as an explicit helper binds pose filenames to that contract.
    del image_path
    return int(source_rank)


def _common_rgb_size(paths: Sequence[Path], *, scene_id: str) -> tuple[int, int]:
    """Read only permitted training RGB headers and require one resolution."""

    size: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            current = (int(image.width), int(image.height))
        if size is None:
            size = current
        elif current != size:
            raise QueuePreparationError(
                f"Scene {scene_id} training RGB resolutions differ: "
                f"{size} versus {current} for {path}"
            )
    if size is None:
        raise QueuePreparationError(f"Scene {scene_id} has no permitted training RGB")
    return size


def _materialize_scene_inputs(
    *,
    raw_scene: Mapping[str, Any],
    normalized_scene: Mapping[str, Any],
    benchmark: str,
    scene_root: Path,
    scene_dir: Path,
) -> dict[str, Any]:
    scene_id = str(normalized_scene["scene_id"])
    rgb_directory = Path(str(raw_scene.get("rgb_directory") or "")).expanduser().resolve()
    image_paths, sort_mode = _collect_image_paths(str(rgb_directory))
    exclusions = _protocol_exclusions(
        benchmark=benchmark,
        normalized_scene=normalized_scene,
        raw_scene=raw_scene,
    )
    retained_indices, excluded_names = select_image_indices(
        image_paths, exclusions, min_remaining=2
    )
    retained_paths = [image_paths[index].resolve() for index in retained_indices]

    declared_training = [
        frame
        for frame in raw_scene.get("training_frames", [])
        if isinstance(frame, Mapping)
    ]
    declared_by_stem: dict[str, Path] = {}
    for frame in declared_training:
        stem = str(frame.get("camera_name") or frame.get("frame_id") or "")
        path = Path(str(frame.get("rgb_path") or "")).expanduser().resolve()
        if not stem or stem in declared_by_stem:
            raise QueuePreparationError(
                f"Scene {scene_id} has missing/duplicate training camera stem {stem!r}"
            )
        if not path.is_file():
            raise QueuePreparationError(f"Scene {scene_id} training RGB is missing: {path}")
        declared_by_stem[stem] = path

    retained_by_stem = {path.stem: path for path in retained_paths}
    if set(retained_by_stem) != set(declared_by_stem):
        raise QueuePreparationError(
            f"Scene {scene_id} locked training_frames do not match exact extraction set; "
            f"missing_from_manifest={sorted(set(retained_by_stem) - set(declared_by_stem))}, "
            f"extra_in_manifest={sorted(set(declared_by_stem) - set(retained_by_stem))}"
        )
    mismatched_paths = [
        stem
        for stem in retained_by_stem
        if retained_by_stem[stem] != declared_by_stem[stem]
    ]
    if mismatched_paths:
        raise QueuePreparationError(
            f"Scene {scene_id} manifest/extractor RGB paths differ for {mismatched_paths}"
        )

    colmap, camera = _colmap_by_stem(scene_root)
    mapping_records = build_rgb_to_colmap_mapping(
        image_paths,
        camera["file_paths"],
        scene_id=scene_id,
    )
    mapping_by_rgb = {
        str(record["rgb_camera_name"]): record for record in mapping_records
    }
    required_protocol_stems = {
        str(frame.get("camera_name") or "")
        for frame in _raw_frames(raw_scene).values()
    }
    missing_mapping = sorted(
        (set(retained_by_stem) | required_protocol_stems) - set(mapping_by_rgb)
    )
    if missing_mapping:
        raise QueuePreparationError(
            f"Scene {scene_id} RGB cameras are absent from the locked "
            f"RGB-to-COLMAP mapping: {missing_mapping}"
        )

    mapped_colmap_names = {str(record["colmap_camera_name"]) for record in mapping_records}
    if not mapped_colmap_names.issubset(colmap):
        raise QueuePreparationError(
            f"Scene {scene_id} mapping references absent COLMAP cameras: "
            f"{sorted(mapped_colmap_names - set(colmap))}"
        )

    if benchmark == "nvos":
        leaked = sorted(set(exclusions) & set(retained_by_stem))
        if leaked:
            raise QueuePreparationError(f"NVOS {scene_id} target leaked after exclusion: {leaked}")
        target_training_leak = sorted(set(exclusions) & set(declared_by_stem))
        if target_training_leak:
            raise QueuePreparationError(
                f"NVOS {scene_id} target exists in locked training_frames: {target_training_leak}"
            )

    rgb_width, rgb_height = _common_rgb_size(retained_paths, scene_id=scene_id)
    calibration_width, calibration_height = int(camera["w"]), int(camera["h"])
    scale_x = float(rgb_width) / float(calibration_width)
    scale_y = float(rgb_height) / float(calibration_height)
    scaled_camera = dict(camera)
    scaled_camera.update(
        {
            "w": rgb_width,
            "h": rgb_height,
            "fl_x": float(camera["fl_x"]) * scale_x,
            "fl_y": float(camera["fl_y"]) * scale_y,
            "cx": float(camera["cx"]) * scale_x,
            "cy": float(camera["cy"]) * scale_y,
        }
    )

    camera_map_path = scene_dir / "rgb_to_colmap_camera_mapping.json"
    camera_map_payload = {
        "schema_version": CAMERA_MAP_SCHEMA_VERSION,
        "kind": "promptable_nvs_rgb_to_colmap_camera_mapping",
        "scene_id": scene_id,
        "scene_root": str(scene_root),
        "rgb_directory": str(rgb_directory),
        "policy_order": [
            "exact_case_sensitive_basename_stem",
            "strip_official_0_or_1_split_prefix_then_exact_stem",
            "imageNNN_canonical_index_to_lexicographic_colmap_camera",
        ],
        "nearest_or_fuzzy_matching": "forbidden",
        "colmap_camera_count": len(colmap),
        "rgb_count": len(image_paths),
        "complete_colmap_coverage": mapped_colmap_names == set(colmap),
        "unmapped_colmap_camera_names": sorted(set(colmap) - mapped_colmap_names),
        "calibration_resolution_wh": [calibration_width, calibration_height],
        "rgb_resolution_wh": [rgb_width, rgb_height],
        "intrinsics_scale_xy": [scale_x, scale_y],
        "rgb_camera_to_colmap_camera": {
            str(record["rgb_camera_name"]): str(record["colmap_camera_name"])
            for record in mapping_records
        },
        "colmap_camera_to_rgb_path": {
            str(record["colmap_camera_name"]): str(record["rgb_path"])
            for record in mapping_records
        },
        "records": mapping_records,
    }
    _write_json(camera_map_path, camera_map_payload)

    pose_dir = scene_dir / "poses_c2w_by_feature_id"
    pose_dir.mkdir(parents=True, exist_ok=True)
    feature_records: list[dict[str, Any]] = []
    seen_feature_ids: dict[int, str] = {}
    for source_rank, path in enumerate(retained_paths):
        feature_id = _training_feature_index(path, source_rank=source_rank)
        if feature_id in seen_feature_ids:
            raise QueuePreparationError(
                f"Scene {scene_id} extractor would collide at rgb_{feature_id}.pt: "
                f"{seen_feature_ids[feature_id]} and {path.name}"
            )
        seen_feature_ids[feature_id] = path.name
        mapping_record = mapping_by_rgb[path.stem]
        colmap_camera_name = str(mapping_record["colmap_camera_name"])
        np.savetxt(
            pose_dir / f"{feature_id}.txt",
            colmap[colmap_camera_name]["c2w"],
            fmt="%.9g",
        )
        feature_records.append(
            {
                "feature_frame_id": feature_id,
                "source_rank": source_rank,
                "camera_name": path.stem,
                "colmap_camera_name": colmap_camera_name,
                "camera_match_rule": str(mapping_record["match_rule"]),
                "rgb_path": str(path),
                "pose_path": str(pose_dir / f"{feature_id}.txt"),
            }
        )

    feature_ids = [record["feature_frame_id"] for record in feature_records]
    # Train on every protocol-permitted RGB.  Validation is a deterministic
    # overlapping audit subset, so no legal SPIn view is withheld and NVOS's
    # target is absent from both lists.
    val_stride = max(1, len(feature_ids) // max(1, min(8, len(feature_ids))))
    val_ids = feature_ids[::val_stride]
    if not val_ids:
        val_ids = [feature_ids[0]]
    train_ids_path = scene_dir / "train_frame_ids.json"
    val_ids_path = scene_dir / "val_frame_ids.json"
    exclusion_path = scene_dir / "excluded_image_stems.json"
    _write_json(train_ids_path, {"frame_ids": feature_ids})
    _write_json(val_ids_path, {"frame_ids": val_ids})
    _write_json(exclusion_path, {"excluded_image_stems": list(exclusions)})
    _write_json(
        scene_dir / "feature_pose_mapping.json",
        {
            "scene_id": scene_id,
            "image_sort_mode": sort_mode,
            "excluded_image_names": excluded_names,
            "records": feature_records,
        },
    )
    return {
        "scene_root": scene_root,
        "rgb_directory": rgb_directory,
        "excluded_stems": exclusions,
        "exclusion_path": exclusion_path,
        "pose_dir": pose_dir,
        "camera_map_path": camera_map_path,
        "train_ids_path": train_ids_path,
        "val_ids_path": val_ids_path,
        "feature_records": feature_records,
        "camera": scaled_camera,
        "colmap_camera": camera,
    }


def _safe_config_overrides() -> dict[str, Any]:
    """Fields that remove mask/RGB/task-specific training from the main track."""

    return {
        "use_refiner": False,
        "refiner_rgb_guide": False,
        "refiner_depth_guide": False,
        "refiner_alpha_guide": False,
        "refiner_boundary_guide": False,
        "self_guided": False,
        "train_sh": False,
        "rgb_loss_weight": 0.0,
        "rgb_dir": "",
        "val_rgb_dir": "",
        "depth_dir": "",
        "val_depth_dir": "",
        "semantics_dir": "",
        "val_semantics_dir": "",
        "instance_dir": "",
        "val_instance_dir": "",
        "depth_loss_weight": 0.0,
        "geom_depth_loss_weight": 0.0,
        "frozen_depth_head_weight": 0.0,
        "frozen_depth_gradient_weight": 0.0,
        "seg_loss_weight": 0.0,
        "frozen_seg_head_weight": 0.0,
        "hybrid_semantic_aux_weight": 0.0,
        "grounding_query_loss_weight": 0.0,
        "samclip_mask_loss_weight": 0.0,
        "samclip_contrastive_loss_weight": 0.0,
        "samclip_background_loss_weight": 0.0,
        "foundation_cache_weight": 0.0,
        "foundation_cache_mask_logit_weight": 0.0,
        "foundation_cache_mask_boundary_weight": 0.0,
        "foundation_cache_token_weight": 0.0,
        "foundation_cache_region_consistency_weight": 0.0,
        "foundation_cache_region_separation_weight": 0.0,
        "foundation_cache_feature_boundary_weight": 0.0,
        "radio_adaptor_mask_logit_weight": 0.0,
        "radio_adaptor_cross_view_mask_propagation_weight": 0.0,
        "direct_point_loss_weight": 0.0,
        "direct_point_text_loss_weight": 0.0,
        "direct_point_query_logit_distill_weight": 0.0,
        "direct_point_query_support_distill_weight": 0.0,
        "text_heatmap_distill_weight": 0.0,
        "resume_from": "",
        "warmstart_from": "",
    }


def _assert_effective_config_safe(config_path: Path) -> None:
    config = load_config(str(config_path))
    try:
        validate_feature_only_config(config)
    except PromptableRenderError as error:
        raise QueuePreparationError(str(error)) from error
    nonzero_forbidden = {
        name: float(getattr(config, name, 0.0))
        for name in (
            "seg_loss_weight",
            "frozen_seg_head_weight",
            "samclip_mask_loss_weight",
            "samclip_contrastive_loss_weight",
            "samclip_background_loss_weight",
            "foundation_cache_weight",
            "radio_adaptor_mask_logit_weight",
            "radio_adaptor_cross_view_mask_propagation_weight",
            "grounding_query_loss_weight",
        )
        if float(getattr(config, name, 0.0)) != 0.0
    }
    if nonzero_forbidden:
        raise QueuePreparationError(
            f"Generated config retains task/mask supervision: {nonzero_forbidden}"
        )


def _command(*parts: str | Path) -> list[str]:
    return [str(part) for part in parts]


def _write_runner(path: Path, plan: Mapping[str, Any]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "if [[ \"${ALLOW_GPU:-0}\" != \"1\" ]]; then",
        "  echo 'Refusing to launch. Set ALLOW_GPU=1 after auditing queue_plan.json.' >&2",
        "  exit 2",
        "fi",
        "export CUDA_VISIBLE_DEVICES=\"${GPU:-0}\"",
        "verify_sha256() {",
        "  local expected=\"$1\"",
        "  local artifact=\"$2\"",
        "  local actual",
        "  actual=\"$(sha256sum \"$artifact\" | awk '{print $1}')\"",
        "  if [[ \"$actual\" != \"$expected\" ]]; then",
        "    echo \"SHA-256 mismatch: $artifact\" >&2",
        "    exit 3",
        "  fi",
        "}",
        f"verify_sha256 {shlex.quote(str(plan['manifest_file_sha256']))} "
        f"{shlex.quote(str(plan['manifest']))}",
        f"verify_sha256 {shlex.quote(str(plan['base_config_sha256']))} "
        f"{shlex.quote(str(plan['base_config']))}",
        "",
    ]
    for artifact in plan.get("code_artifacts", []):
        lines.append(
            f"verify_sha256 {shlex.quote(str(artifact['sha256']))} "
            f"{shlex.quote(str(artifact['path']))}"
        )
    if plan.get("code_artifacts"):
        lines.append("")
    sam_checkpoint = plan.get("sam_embedding", {}).get("checkpoint")
    sam_sha256 = plan.get("sam_embedding", {}).get("checkpoint_sha256")
    if sam_checkpoint and sam_sha256:
        lines.append(
            f"verify_sha256 {shlex.quote(str(sam_sha256))} "
            f"{shlex.quote(str(sam_checkpoint))}"
        )
        lines.append("")
    for scene in plan["scenes"]:
        lines.append(f"# {scene['scene_id']}")
        lines.append(
            f"verify_sha256 {shlex.quote(str(scene['camera_mapping']['sha256']))} "
            f"{shlex.quote(str(scene['camera_mapping']['path']))}"
        )
        lines.append(
            f"verify_sha256 {shlex.quote(str(scene['config_sha256']))} "
            f"{shlex.quote(str(scene['config']))}"
        )
        for stage in ("geometry", "feature_extraction", "feature_field", "render"):
            lines.append(shlex.join(scene["commands"][stage]))
        lines.append("")
    lines.append("# Dataset-level prediction and scoring")
    lines.append(shlex.join(plan["dataset_commands"]["predict"]))
    lines.append(shlex.join(plan["dataset_commands"]["evaluate"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def prepare_queue(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    base_config: str | Path,
    scene_root_map: str | Path | Mapping[str, str] | None = None,
    python_executable: str | Path = sys.executable,
    geometry_iters: int = 30000,
    resolution_scale: float = 1.0,
    radio_repo: str | Path | None = None,
    sam_adaptor_checkpoint: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize a CPU-only queue plan; no training/render command is run."""

    manifest_source = Path(manifest_path).expanduser().resolve()
    base_config_source = Path(base_config).expanduser().resolve()
    if not manifest_source.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_source}")
    if not base_config_source.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config_source}")
    if int(geometry_iters) <= 0:
        raise QueuePreparationError("geometry_iters must be positive")
    if float(resolution_scale) <= 0:
        raise QueuePreparationError("resolution_scale must be positive")

    destination = Path(output_root).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Queue output is non-empty (use --overwrite for generated files): {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    try:
        normalized = validate_dataset_manifest(manifest, check_files=True)
    except ManifestError as error:
        raise QueuePreparationError(str(error)) from error
    benchmark = str(manifest.get("benchmark") or "")
    track = _track_metadata(benchmark, str(normalized["protocol"].get("prompt_type", "")))
    threshold = normalized["protocol"]["threshold"]
    if threshold != {"mode": "fixed", "value": 0.0}:
        raise QueuePreparationError(
            "Main GaussFM queue requires frozen threshold {mode: fixed, value: 0.0}; "
            "target/test calibration is never generated"
        )
    if any(scene["calibration_frame_ids"] for scene in normalized["scenes"]):
        raise QueuePreparationError("Calibration frames are forbidden in the main queue")

    roots = _scene_map(scene_root_map)
    unknown_roots = sorted(set(roots) - {scene["scene_id"] for scene in normalized["scenes"]})
    if unknown_roots:
        raise QueuePreparationError(f"scene_root_map contains unknown scenes: {unknown_roots}")

    project_root = Path(__file__).resolve().parents[2]
    python = str(Path(python_executable).expanduser())
    base_effective = load_config(str(base_config_source))
    resolved_radio_repo = Path(
        str(radio_repo or getattr(base_effective, "radio_repo", "") or "/root/RADIO")
    ).expanduser().resolve()
    if not resolved_radio_repo.is_dir():
        raise QueuePreparationError(f"RADIO repository not found: {resolved_radio_repo}")
    adaptor_path = (
        Path(sam_adaptor_checkpoint).expanduser().resolve()
        if sam_adaptor_checkpoint is not None
        else None
    )
    if adaptor_path is not None and not adaptor_path.is_file():
        raise FileNotFoundError(f"SAM adaptor checkpoint not found: {adaptor_path}")

    rendered_root = destination / "rendered_features"
    prediction_root = destination / "predictions"
    evaluation_path = destination / "evaluation.json"
    scene_plans: list[dict[str, Any]] = []
    for normalized_scene in normalized["scenes"]:
        scene_id = _safe_component(str(normalized_scene["scene_id"]), role="scene_id")
        raw_scene = _raw_scene(manifest, scene_id)
        scene_root = _discover_scene_root(raw_scene, explicit=roots.get(scene_id))
        scene_dir = destination / "scenes" / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        materialized = _materialize_scene_inputs(
            raw_scene=raw_scene,
            normalized_scene=normalized_scene,
            benchmark=benchmark,
            scene_root=scene_root,
            scene_dir=scene_dir,
        )
        camera = materialized["camera"]
        target_h, target_w = _compute_scaled_radio_resolution(
            int(camera["h"]), int(camera["w"]), float(resolution_scale), patch_size=16
        )
        geometry_dir = scene_dir / "geometry"
        feature_dir = scene_dir / "radio_features"
        field_dir = scene_dir / "feature_field"
        geometry_ply = (
            geometry_dir
            / "point_cloud"
            / f"iteration_{int(geometry_iters)}"
            / "point_cloud.ply"
        )
        field_checkpoint = field_dir / "checkpoints" / "best.pth"
        config_path = scene_dir / "gaussfm_main_track.yaml"
        overlay: dict[str, Any] = {
            "base_config": str(base_config_source),
            "exp_name": f"gaussfm_{benchmark}_{scene_id}_main_track",
            "output_dir": str(field_dir),
            "scene": scene_id,
            "scene_root": str(scene_root),
            "dataset_type": "lerf",
            "ply_path": str(geometry_ply),
            "radio_repo": str(resolved_radio_repo),
            "feature_dir": str(feature_dir),
            "val_feature_dir": str(feature_dir),
            "pose_file": "",
            "val_pose_file": "",
            "pose_dir": str(materialized["pose_dir"]),
            "val_pose_dir": str(materialized["pose_dir"]),
            "train_frame_ids_path": str(materialized["train_ids_path"]),
            "val_frame_ids_path": str(materialized["val_ids_path"]),
            "image_height": int(camera["h"]),
            "image_width": int(camera["w"]),
            "fx": float(camera["fl_x"]),
            "fy": float(camera["fl_y"]),
            "cx": float(camera["cx"]),
            "cy": float(camera["cy"]),
            "feature_height": int(target_h // 16),
            "feature_width": int(target_w // 16),
            "mixed_split": False,
            "camera_map_path": str(materialized["camera_map_path"]),
            # Unknown metadata fields are ignored by RadioGSConfig but remain
            # in the snapshot for protocol provenance.
            "queue_protocol_hash": normalized["protocol_hash"],
            "queue_protocol_track": track["id"],
            "queue_checkpoint_sha256": PENDING_SHA256,
        }
        overlay.update(_safe_config_overrides())
        config_path.write_text(
            yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        _assert_effective_config_safe(config_path)

        geometry_command = _command(
            python,
            project_root / "radio_gs/scripts/train_colmap_gs.py",
            "--scene_root",
            scene_root,
            "--output_dir",
            geometry_dir,
            "--iters",
            str(int(geometry_iters)),
            "--device",
            "cuda",
            "--image-map-json",
            materialized["camera_map_path"],
            "--image-scale",
            str(float(resolution_scale)),
        )
        feature_command = _command(
            python,
            project_root / "radio_gs/scripts/extract_radio_features.py",
            "--scene",
            scene_id,
            "--image_dir",
            materialized["rgb_directory"],
            "--output_dir",
            feature_dir,
            "--radio_repo",
            resolved_radio_repo,
            "--radio_version",
            str(getattr(base_effective, "radio_version", "c-radio_v4-h")),
            "--resolution_scale",
            str(float(resolution_scale)),
            "--frame-id-mode",
            "source_rank",
            "--device",
            "cuda",
            "--amp",
        )
        if benchmark == "nvos":
            geometry_command.extend(
                ["--exclude-image-stems-file", str(materialized["exclusion_path"])]
            )
            feature_command.extend(
                ["--exclude-image-stems-file", str(materialized["exclusion_path"])]
            )
        field_command = _command(
            python,
            project_root / "radio_gs/scripts/train_feature_field.py",
            "--config",
            config_path,
        )
        render_command = _command(
            python,
            project_root / "radio_gs/scripts/render_promptable_nvs_features.py",
            "--manifest",
            manifest_source,
            "--scene-id",
            scene_id,
            "--camera-map",
            materialized["camera_map_path"],
            "--config",
            config_path,
            "--checkpoint",
            field_checkpoint,
            "--output-dir",
            rendered_root,
            "--device",
            "cuda",
        )
        scene_plans.append(
            {
                "scene_id": scene_id,
                "scene_root": str(scene_root),
                "rgb_directory": str(materialized["rgb_directory"]),
                "camera_mapping": {
                    "path": str(materialized["camera_map_path"]),
                    "sha256": _sha256(materialized["camera_map_path"]),
                    "policy": "locked_explicit_rgb_to_colmap_no_nearest",
                },
                "excluded_image_stems": list(materialized["excluded_stems"]),
                "training_feature_frame_ids": [
                    record["feature_frame_id"] for record in materialized["feature_records"]
                ],
                "resolution": {
                    "geometry_rgb_scale_xy": [
                        float(camera["w"]) / float(materialized["colmap_camera"]["w"]),
                        float(camera["h"]) / float(materialized["colmap_camera"]["h"]),
                    ],
                    "geometry_training_image_scale": float(resolution_scale),
                    "geometry_training_hw": [
                        max(1, int(round(float(camera["h"]) * float(resolution_scale)))),
                        max(1, int(round(float(camera["w"]) * float(resolution_scale)))),
                    ],
                    "radio_input_scale": float(resolution_scale),
                    "source_image_hw": [int(camera["h"]), int(camera["w"])],
                    "radio_input_hw": [int(target_h), int(target_w)],
                    "feature_grid_hw": [int(target_h // 16), int(target_w // 16)],
                },
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "commands": {
                    "geometry": geometry_command,
                    "feature_extraction": feature_command,
                    "feature_field": field_command,
                    "render": render_command,
                },
                "artifacts": {
                    "geometry_ply": {
                        "path": str(geometry_ply),
                        "sha256": PENDING_SHA256,
                    },
                    "feature_field_checkpoint": {
                        "path": str(field_checkpoint),
                        "sha256": PENDING_SHA256,
                    },
                    "render_manifest": str(rendered_root / scene_id / "render_manifest.json"),
                },
                "safety": {
                    "target_rgb_excluded_from_geometry_rgb_loss": benchmark == "nvos",
                    "sparse_points_with_target_observations_removed": benchmark == "nvos",
                    "upstream_camera_calibration_shared_exception": benchmark == "nvos",
                    "target_rgb_excluded_from_feature_extraction": benchmark == "nvos",
                    "target_absent_from_train_frame_ids": benchmark == "nvos",
                    "target_absent_from_val_frame_ids": benchmark == "nvos",
                    "mask_paths_present_in_training_config": False,
                    "rgb_refiner": False,
                    "target_calibration": False,
                },
            }
        )

    predict_command = _command(
        python,
        project_root / "radio_gs/scripts/predict_promptable_nvs_feature_readout.py",
        "--manifest",
        manifest_source,
        "--output-dir",
        prediction_root,
        "--feature-root",
        rendered_root,
        "--feature-pattern",
        "{scene_id}/{camera_name}.pt",
        "--feature-layout",
        "chw",
        "--method-name",
        "GaussFM reusable field + frozen RADIO SAM3-adaptor prototype readout",
    )
    if adaptor_path is not None:
        predict_command.extend(
            [
                "--radio-sam3-adaptor-checkpoint",
                str(adaptor_path),
                "--adaptor-device",
                "cuda",
            ]
        )
    prediction_manifest = prediction_root / "prediction_manifest.json"
    evaluate_command = _command(
        python,
        Path(__file__).resolve(),
        "evaluate",
        "--manifest",
        manifest_source,
        "--prediction-manifest",
        prediction_manifest,
        "--output",
        evaluation_path,
    )
    code_relative_paths = (
        "radio_gs/data/benchmark_paths.py",
        "radio_gs/data/lerf_dataset.py",
        "radio_gs/data/promptable_nvs_manifest.py",
        "radio_gs/data/view_split.py",
        "radio_gs/evaluation/promptable_feature_readout.py",
        "radio_gs/evaluation/promptable_segmentation.py",
        "radio_gs/rendering/feature_renderer.py",
        "radio_gs/scripts/extract_radio_features.py",
        "radio_gs/scripts/predict_promptable_nvs_feature_readout.py",
        "radio_gs/scripts/prepare_promptable_nvs_gaussfm_queue.py",
        "radio_gs/scripts/render_promptable_nvs_features.py",
        "radio_gs/scripts/train_colmap_gs.py",
        "radio_gs/scripts/train_feature_field.py",
    )
    code_artifacts = [
        {
            "path": str((project_root / relative).resolve()),
            "sha256": _sha256(project_root / relative),
        }
        for relative in code_relative_paths
    ]
    plan: dict[str, Any] = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "kind": "promptable_nvs_gaussfm_queue",
        "status": "prepared_not_run",
        "benchmark": benchmark,
        "track": track,
        "protocol_hash": normalized["protocol_hash"],
        "manifest": str(manifest_source),
        "manifest_file_sha256": _sha256(manifest_source),
        "base_config": str(base_config_source),
        "base_config_sha256": _sha256(base_config_source),
        "code_artifacts": code_artifacts,
        "sam_embedding": {
            "type": (
                "frozen_radio_sam3_feature_projection"
                if adaptor_path is not None
                else "raw_gaussfm_radio_embedding"
            ),
            "checkpoint": str(adaptor_path) if adaptor_path is not None else None,
            "checkpoint_sha256": (
                _sha256(adaptor_path)
                if adaptor_path is not None
                else None
            ),
            "official_sam_decoder": False,
        },
        "scenes": scene_plans,
        "dataset_commands": {
            "predict": predict_command,
            "evaluate": evaluate_command,
        },
        "outputs": {
            "rendered_features": str(rendered_root),
            "prediction_manifest": str(prediction_manifest),
            "evaluation": str(evaluation_path),
        },
        "protocol_guards": {
            "aggregation": normalized["protocol"]["aggregation"],
            "metrics": normalized["protocol"]["metrics"],
            "prediction_representation": normalized["protocol"][
                "prediction_representation"
            ],
            "threshold_comparison": normalized["protocol"]["threshold_comparison"],
            "threshold": threshold,
            "target_test_calibration": "forbidden",
            "reference_scoring": False,
            "training_mask_use": "none",
            "query_target_rgb_use": "none",
            "nvos_geometry_provenance": (
                "held-out target RGB is excluded from photometric geometry loss and "
                "sparse points observed by the target camera are removed; poses and "
                "intrinsics still come from the upstream joint reconstruction"
                if benchmark == "nvos"
                else "all protocol-permitted SPIn RGB views may train the field"
            ),
            "fully_target_pixel_independent_camera_calibration": False,
            "same_prompt_saga_main_table_eligible": track[
                "saga_same_prompt_main_table_eligible"
            ],
        },
        "execution": {
            "commands_executed_during_prepare": 0,
            "gpu_used_during_prepare": False,
            "runner_requires_environment": {"ALLOW_GPU": "1", "GPU": "CUDA index"},
        },
    }
    plan_path = destination / "queue_plan.json"
    runner_path = destination / "run_plan.sh"
    _write_json(plan_path, plan)
    _write_runner(runner_path, plan)
    plan["plan_path"] = str(plan_path)
    plan["runner_path"] = str(runner_path)
    return plan


def evaluate_predictions(
    manifest_path: str | Path,
    prediction_manifest: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Score protocol-bound predictions and persist the exact evaluator report."""

    manifest_source = Path(manifest_path).expanduser().resolve()
    raw_manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    validate_dataset_manifest(raw_manifest, check_files=True)
    result = evaluate_manifest(
        manifest_source,
        prediction_manifest=Path(prediction_manifest).expanduser().resolve(),
    )
    output = Path(output_path).expanduser().resolve()
    _write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Generate configs and a non-running queue")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--base-config", type=Path, required=True)
    prepare.add_argument("--scene-root-map", type=Path)
    prepare.add_argument("--python-executable", default=sys.executable)
    prepare.add_argument("--geometry-iters", type=int, default=30000)
    prepare.add_argument(
        "--resolution-scale",
        type=float,
        default=1.0,
        help="Shared geometry-RGB and RADIO-input scale (use 0.25 for NVOS factor-4)",
    )
    prepare.add_argument("--radio-repo", type=Path)
    prepare.add_argument("--sam-adaptor-checkpoint", type=Path)
    prepare.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Explicitly score saved predictions")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--prediction-manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        plan = prepare_queue(
            args.manifest,
            args.output_root,
            base_config=args.base_config,
            scene_root_map=args.scene_root_map,
            python_executable=args.python_executable,
            geometry_iters=args.geometry_iters,
            resolution_scale=args.resolution_scale,
            radio_repo=args.radio_repo,
            sam_adaptor_checkpoint=args.sam_adaptor_checkpoint,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "status": plan["status"],
                    "benchmark": plan["benchmark"],
                    "protocol_hash": plan["protocol_hash"],
                    "scenes": len(plan["scenes"]),
                    "plan": plan["plan_path"],
                    "runner": plan["runner_path"],
                    "gpu_used": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    result = evaluate_predictions(args.manifest, args.prediction_manifest, args.output)
    print(json.dumps(result["dataset"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, QueuePreparationError, PromptableRenderError) as error:
        raise SystemExit(f"queue operation failed: {error}") from error
