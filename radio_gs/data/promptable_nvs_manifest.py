"""Fail-closed dataset manifests for NVOS and SPIn-NeRF segmentation.

The dictionaries produced here are consumed directly by
``radio_gs.evaluation.promptable_segmentation``.  Dataset-specific metadata is
kept as extra fields, but there is deliberately no second ``tasks`` schema.

RGB/annotation association is explicit:

* NVOS uses exact, case-sensitive camera basenames.
* Four SPIn-NeRF LLFF exports use the documented ``imageNNN`` annotation index
  into the lexicographically sorted camera list.  Sparse indices (notably
  orchids ``image015`` after ``image013``) are never compacted.
* The remaining SPIn-NeRF scenes use exact basenames.  Lego and Truck may use
  the official ``0_``/``1_`` split prefix, which is stripped only after an
  exact source-camera match is found.

Any missing or ambiguous source fails with :class:`ManifestError`; the builder
never guesses a nearest filename or silently drops a benchmark scene.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from radio_gs.evaluation.promptable_segmentation import (
    AGGREGATION,
    METRIC_NAMES,
    RESIZE_POLICY,
    TASK_NAME,
    compute_protocol_hash,
    validate_manifest as validate_evaluation_manifest,
)


SCHEMA_VERSION = 1

NVOS_TASKS = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)

# The order follows the frozen ten-scene reporting cohort.  Aggregation is a
# scene macro, but retaining one canonical order makes audits deterministic.
SPIN_SCENE_FOLDERS = (
    ("orchids", "orchids (llff)"),
    ("leaves", "leaves (llff)"),
    ("fern", "fern (llff)"),
    ("room", "room (llff)"),
    ("horns", "horns (llff)"),
    ("fortress", "fortress (llff)"),
    ("fork", "fork (nerf_supervision)"),
    ("pinecone", "pinecone (nerf_real_360)"),
    ("truck", "Truck (Tanks & Temples)"),
    ("lego", "lego_real_night_radial"),
)
SPIN_SCENES = tuple(scene_id for scene_id, _ in SPIN_SCENE_FOLDERS)
SPIN_DIAGNOSTIC_SCENES = tuple(scene_id for scene_id in SPIN_SCENES if scene_id != "fork")
SPIN_EXPECTED_MASK_COUNTS = {
    "orchids": 24,
    "leaves": 26,
    "fern": 20,
    "room": 41,
    "horns": 62,
    "fortress": 42,
    "fork": 38,
    "pinecone": 99,
    "truck": 65,
    "lego": 44,
}

_SPIN_INDEXED_LLFF_SCENES = frozenset({"fern", "fortress", "leaves", "orchids"})
_SPIN_PREFIXED_SCENES = frozenset({"lego", "truck"})
_IMAGE_INDEX_RE = re.compile(r"^image(?P<index>[0-9]+)$", re.IGNORECASE)
_SPLIT_PREFIX_RE = re.compile(r"^[01]_(?P<camera>.+)$")
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})


class ManifestError(ValueError):
    """Raised when a benchmark layout cannot be mapped without guessing."""


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ManifestError(f"Image directory does not exist: {root}")
    files = sorted(
        (
            path.resolve()
            for path in root.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )
    if not files:
        raise ManifestError(f"Image directory contains no supported images: {root}")
    return files


def _require_one(paths: Iterable[Path], description: str) -> Path:
    matches = sorted((path.resolve() for path in paths), key=lambda path: path.name)
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise ManifestError(
            f"Expected exactly one {description}; found {len(matches)} ({names})"
        )
    return matches[0]


def _visible_directories(root: Path) -> set[str]:
    if not root.is_dir():
        raise ManifestError(f"Annotation directory does not exist: {root}")
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
    }


def _require_cohort(root: Path, expected: Sequence[str], label: str) -> None:
    actual = _visible_directories(root)
    wanted = set(expected)
    if actual != wanted:
        raise ManifestError(
            f"Incomplete {label} cohort at {root}; "
            f"missing={sorted(wanted - actual)}, unexpected={sorted(actual - wanted)}"
        )


def _stem_lookup(images: Sequence[Path], *, label: str) -> dict[str, tuple[int, Path]]:
    lookup: dict[str, tuple[int, Path]] = {}
    for index, path in enumerate(images):
        if path.stem in lookup:
            raise ManifestError(
                f"Duplicate RGB basename stem {path.stem!r} in {label}: "
                f"{lookup[path.stem][1]} and {path}"
            )
        lookup[path.stem] = (index, path)
    return lookup


def _resolve_exact_image_dir(path: str | Path, *, label: str) -> Path:
    """Resolve an explicit scene path, rejecting ambiguous image pyramids."""

    root = _absolute(path)
    if not root.is_dir():
        raise ManifestError(f"{label} source directory does not exist: {root}")
    direct = [
        item
        for item in root.iterdir()
        if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if direct:
        return root
    candidates = [
        child.resolve()
        for child in (root / "images_4", root / "images")
        if child.is_dir()
        and any(
            item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
            for item in child.iterdir()
        )
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ManifestError(
            f"{label} has both images_4 and images. Pass the exact image directory "
            f"instead of the scene root: {root}"
        )
    raise ManifestError(
        f"{label} contains neither direct images nor a unique images/images_4 directory: {root}"
    )


def _resolve_nvos_scene_dir(rgb_root: Path, task_id: str) -> Path:
    base_scene = "horns" if task_id.startswith("horns_") else task_id
    aliases = tuple(
        dict.fromkeys(
            (task_id, base_scene, f"{task_id}_undistort", f"{base_scene}_undistort")
        )
    )
    matches: list[Path] = []
    for alias in aliases:
        candidate = rgb_root / alias
        if not candidate.is_dir():
            continue
        try:
            resolved = _resolve_exact_image_dir(candidate, label=f"NVOS {task_id}")
        except ManifestError as error:
            # Existing but malformed/ambiguous input must be reported, not
            # hidden by falling through to another alias.
            raise ManifestError(str(error)) from error
        if resolved not in matches:
            matches.append(resolved)
    if len(matches) != 1:
        raise ManifestError(
            f"Expected one NVOS RGB source for {task_id} below {rgb_root}; "
            f"aliases={list(aliases)}, matches={[str(path) for path in matches]}"
        )
    return matches[0]


def _fixed_threshold(value: float) -> dict[str, Any]:
    value = float(value)
    if not math.isfinite(value):
        raise ManifestError("Fixed threshold must be finite")
    return {"mode": "fixed", "value": value}


def _base_protocol(
    *,
    benchmark: str,
    dataset_version: str,
    prompt_type: str,
    threshold: float,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "dataset_version": dataset_version,
        "task": TASK_NAME,
        "prompt_type": prompt_type,
        "metrics": list(METRIC_NAMES),
        "aggregation": AGGREGATION,
        "resize": RESIZE_POLICY,
        "prediction_representation": "continuous_margin",
        "threshold_comparison": "greater_or_equal",
        "empty_union_value": 1.0,
        "allow_reference_scoring": False,
        "threshold": _fixed_threshold(threshold),
        "score_semantics": "cosine_similarity_foreground_minus_background",
        "score_temperature": "none",
    }


def _frame(
    frame_id: str,
    *,
    ground_truth: Path | None,
    rgb_path: Path,
    annotation_rgb_path: Path | None,
    canonical_index: int,
    annotation_sorted_index: int | None,
    rgb_sorted_index: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame_id": frame_id,
        "ground_truth": str(ground_truth) if ground_truth is not None else None,
        "gt_mask_path": str(ground_truth) if ground_truth is not None else None,
        "canonical_index": canonical_index,
        "annotation_sorted_index": annotation_sorted_index,
        "rgb_sorted_index": rgb_sorted_index,
        "camera_name": rgb_path.stem,
        "rgb_path": str(rgb_path),
        "annotation_rgb_path": (
            str(annotation_rgb_path) if annotation_rgb_path is not None else None
        ),
        "feature_path": None,
    }
    if ground_truth is not None:
        payload["ground_truth_sha256"] = _sha256(ground_truth)
    return payload


def _finalize(manifest: dict[str, Any], *, check_files: bool) -> dict[str, Any]:
    manifest["protocol_hash"] = compute_protocol_hash(manifest)
    validate_manifest(manifest, check_files=check_files)
    return manifest


def build_nvos_manifest(
    annotation_root: str | Path,
    rgb_root: str | Path,
    *,
    threshold: float = 0.0,
    validate: bool = True,
) -> dict[str, Any]:
    """Build the strict eight-task NVOS unseen-target manifest.

    The official target camera is removed from ``training_frames`` for every
    task.  The scribble reference has ``ground_truth: null`` because the NVOS
    release provides scribbles, not a full reference-view object mask.
    """

    annotations = _absolute(annotation_root)
    rgb_source = _absolute(rgb_root)
    masks_root = annotations / "masks"
    reference_root = annotations / "reference_image"
    scribbles_root = annotations / "scribbles"
    for root, label in (
        (masks_root, "NVOS mask"),
        (reference_root, "NVOS reference-image"),
        (scribbles_root, "NVOS scribble"),
    ):
        _require_cohort(root, NVOS_TASKS, label)

    scenes: list[dict[str, Any]] = []
    prompt_hashes: dict[str, dict[str, str]] = {}
    for task_id in NVOS_TASKS:
        mask_dir = masks_root / task_id
        reference_dir = reference_root / task_id
        scribble_dir = scribbles_root / task_id

        gt_mask = _require_one(
            (
                path
                for path in mask_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".png"
                and path.stem.endswith("_mask")
            ),
            f"NVOS target mask for {task_id}",
        )
        target_id = gt_mask.stem[: -len("_mask")]
        target_annotation_rgb = _require_one(
            (path for path in _image_files(mask_dir) if path.stem == target_id),
            f"NVOS target RGB annotation copy for {task_id}/{target_id}",
        )
        reference_annotation_rgb = _require_one(
            _image_files(reference_dir), f"NVOS reference RGB for {task_id}"
        )
        reference_id = reference_annotation_rgb.stem
        if reference_id == target_id:
            raise ManifestError(f"NVOS {task_id} reference and target both use {target_id}")

        scribble_images = _image_files(scribble_dir)
        positive = _require_one(
            (path for path in scribble_images if path.stem.startswith("pos_")),
            f"positive scribble for {task_id}",
        )
        negative = _require_one(
            (path for path in scribble_images if path.stem.startswith("neg_")),
            f"negative scribble for {task_id}",
        )
        visualization = _require_one(
            (path for path in scribble_images if path.stem.startswith("vis_")),
            f"scribble visualization for {task_id}",
        )

        rgb_dir = _resolve_nvos_scene_dir(rgb_source, task_id)
        rgb_images = _image_files(rgb_dir)
        lookup = _stem_lookup(rgb_images, label=f"NVOS {task_id}")
        missing_roles = [
            f"{role}={frame_id}"
            for role, frame_id in (("reference", reference_id), ("target", target_id))
            if frame_id not in lookup
        ]
        if missing_roles:
            raise ManifestError(
                f"NVOS {task_id} annotation/RGB mapping failed in {rgb_dir}: "
                + ", ".join(missing_roles)
            )

        ref_index, ref_rgb = lookup[reference_id]
        target_index, target_rgb = lookup[target_id]
        reference_frame = _frame(
            reference_id,
            ground_truth=None,
            rgb_path=ref_rgb,
            annotation_rgb_path=reference_annotation_rgb,
            canonical_index=ref_index,
            annotation_sorted_index=None,
            rgb_sorted_index=ref_index,
        )
        target_frame = _frame(
            target_id,
            ground_truth=gt_mask,
            rgb_path=target_rgb,
            annotation_rgb_path=target_annotation_rgb,
            canonical_index=target_index,
            annotation_sorted_index=0,
            rgb_sorted_index=target_index,
        )
        training_frames = [
            {
                "frame_id": path.stem,
                "camera_name": path.stem,
                "rgb_sorted_index": index,
                "rgb_path": str(path),
            }
            for index, path in enumerate(rgb_images)
            if path.stem != target_id
        ]
        if target_id in {frame["frame_id"] for frame in training_frames}:
            raise AssertionError("NVOS target exclusion invariant failed")

        scenes.append(
            {
                "scene_id": task_id,
                "base_scene_id": "horns" if task_id.startswith("horns_") else task_id,
                "prompt_frame_ids": [reference_id],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": [target_id],
                "frames": [reference_frame, target_frame],
                "prompt": {
                    "type": "positive_negative_scribbles",
                    "frame_id": reference_id,
                    "positive_path": str(positive),
                    "negative_path": str(negative),
                    "visualization_path": str(visualization),
                },
                "annotation_scene_dir": str(mask_dir.resolve()),
                "rgb_directory": str(rgb_dir),
                "frame_mapping": {
                    "strategy": "exact_case_sensitive_basename_stem",
                    "rgb_order": "lexicographic_filename",
                    "verified": True,
                },
                "target_rgb_policy": "excluded_from_field_training_and_query",
                "excluded_training_frame_ids": [target_id],
                "training_frames": training_frames,
            }
        )
        prompt_hashes[task_id] = {
            "positive": _sha256(positive),
            "negative": _sha256(negative),
        }

    protocol = _base_protocol(
        benchmark="NVOS",
        dataset_version="official-nvos-2022-v1+nex-llff-undistorted",
        prompt_type="fixed_positive_negative_scribbles",
        threshold=threshold,
    )
    protocol.update(
        {
            "cohort": list(NVOS_TASKS),
            "prompt_asset_sha256": prompt_hashes,
            "prompt_support": "complete_official_positive_negative_scribble_masks",
            "exactly_comparable_to_published_saga": False,
            "published_saga_prompt_note": (
                "random positive/negative subset from scribbles; point count and "
                "seed not disclosed"
            ),
            "target_rgb_during_field_training": "forbidden",
            "target_rgb_at_query": "forbidden",
            "target_mask_use": "scoring_only",
            "within_scene_aggregation": "single_official_target",
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "nvos",
        "annotation_root": str(annotations),
        "rgb_root": str(rgb_source),
        "protocol": protocol,
        "scenes": scenes,
    }
    if validate:
        return _finalize(manifest, check_files=True)
    manifest["protocol_hash"] = compute_protocol_hash(manifest)
    return manifest


def _base_spin_masks(scene_dir: Path) -> list[Path]:
    return sorted(
        (
            path.resolve()
            for path in scene_dir.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() == ".png"
            and not path.stem.lower().endswith(("_cutout", "_pseudo"))
        ),
        key=lambda path: path.name,
    )


def _spin_canonical_index(scene_id: str, mask: Path, sorted_index: int) -> int:
    if scene_id not in _SPIN_INDEXED_LLFF_SCENES:
        return sorted_index
    match = _IMAGE_INDEX_RE.fullmatch(mask.stem)
    if match is None:
        raise ManifestError(
            f"SPIn-NeRF {scene_id} requires imageNNN annotation ids; found {mask.name}"
        )
    return int(match.group("index"))


def _map_spin_rgb(
    *,
    scene_id: str,
    mask: Path,
    canonical_index: int,
    rgb_images: Sequence[Path],
    lookup: Mapping[str, tuple[int, Path]],
) -> tuple[int, Path, str]:
    if scene_id in _SPIN_INDEXED_LLFF_SCENES:
        if canonical_index >= len(rgb_images):
            raise ManifestError(
                f"SPIn-NeRF {scene_id}/{mask.name} maps to canonical RGB index "
                f"{canonical_index}, but {len(rgb_images)} source cameras exist"
            )
        return canonical_index, rgb_images[canonical_index], "imageNNN_to_rgb_index"

    if mask.stem in lookup:
        index, path = lookup[mask.stem]
        return index, path, "exact_case_sensitive_basename_stem"

    if scene_id in _SPIN_PREFIXED_SCENES:
        match = _SPLIT_PREFIX_RE.fullmatch(mask.stem)
        if match is not None and match.group("camera") in lookup:
            index, path = lookup[match.group("camera")]
            return index, path, "strip_official_0_or_1_split_prefix_then_exact_stem"

    raise ManifestError(
        f"No explicit RGB mapping for SPIn-NeRF {scene_id}/{mask.name}; "
        "only the scene's declared protocol mapping is permitted"
    )


def _normalize_spin_rgb_dirs(
    scene_rgb_dirs: Mapping[str, str | Path],
    *,
    expected_scenes: Sequence[str] = SPIN_SCENES,
) -> dict[str, Path]:
    if not isinstance(scene_rgb_dirs, Mapping):
        raise ManifestError("scene_rgb_dirs must be a mapping of scene id to image directory")
    expected = tuple(str(scene_id) for scene_id in expected_scenes)
    unknown = sorted(set(scene_rgb_dirs) - set(expected))
    missing = sorted(set(expected) - set(scene_rgb_dirs))
    if unknown or missing:
        raise ManifestError(
            "SPIn-NeRF RGB mapping must explicitly cover the requested cohort; "
            f"missing={missing}, unknown={unknown}"
        )
    return {
        scene_id: _resolve_exact_image_dir(
            scene_rgb_dirs[scene_id], label=f"SPIn-NeRF {scene_id}"
        )
        for scene_id in expected
    }


def build_spin_manifest(
    annotation_root: str | Path,
    scene_rgb_dirs: Mapping[str, str | Path],
    *,
    threshold: float = 0.0,
    enforce_official_counts: bool = True,
    diagnostic_missing_fork: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """Build the formal ten-scene or explicitly labelled nine-scene diagnostic."""

    annotations = _absolute(annotation_root)
    expected_folders = tuple(folder for _, folder in SPIN_SCENE_FOLDERS)
    _require_cohort(annotations, expected_folders, "SPIn-NeRF multiview-segmentation")
    cohort = SPIN_DIAGNOSTIC_SCENES if diagnostic_missing_fork else SPIN_SCENES
    rgb_dirs = _normalize_spin_rgb_dirs(scene_rgb_dirs, expected_scenes=cohort)

    scenes: list[dict[str, Any]] = []
    for scene_id, annotation_folder in SPIN_SCENE_FOLDERS:
        if scene_id not in cohort:
            continue
        scene_dir = annotations / annotation_folder
        masks = _base_spin_masks(scene_dir)
        if len(masks) < 2:
            raise ManifestError(
                f"SPIn-NeRF {scene_id} needs one prompt mask and at least one target; "
                f"found {len(masks)} base masks"
            )
        if enforce_official_counts and len(masks) != SPIN_EXPECTED_MASK_COUNTS[scene_id]:
            raise ManifestError(
                f"SPIn-NeRF {scene_id} official cohort has "
                f"{SPIN_EXPECTED_MASK_COUNTS[scene_id]} masks; found {len(masks)}"
            )

        canonical_indices = [
            _spin_canonical_index(scene_id, mask, index)
            for index, mask in enumerate(masks)
        ]
        if len(canonical_indices) != len(set(canonical_indices)):
            raise ManifestError(f"Duplicate canonical annotation index in {scene_id}")
        if scene_id == "orchids":
            if 14 in canonical_indices:
                raise ManifestError(
                    "Official SPIn-NeRF orchids annotations skip image014; found it unexpectedly"
                )
            if enforce_official_counts and canonical_indices != list(range(14)) + list(
                range(15, 25)
            ):
                raise ManifestError(
                    "SPIn-NeRF orchids indices differ from the official sparse 0..13,15..24 cohort"
                )

        rgb_images = _image_files(rgb_dirs[scene_id])
        lookup = _stem_lookup(rgb_images, label=f"SPIn-NeRF {scene_id}")
        frames: list[dict[str, Any]] = []
        strategies: set[str] = set()
        mapped_cameras: set[str] = set()
        for annotation_index, (mask, canonical_index) in enumerate(
            zip(masks, canonical_indices)
        ):
            rgb_index, rgb_path, strategy = _map_spin_rgb(
                scene_id=scene_id,
                mask=mask,
                canonical_index=canonical_index,
                rgb_images=rgb_images,
                lookup=lookup,
            )
            if rgb_path.stem in mapped_cameras:
                raise ManifestError(
                    f"SPIn-NeRF {scene_id} maps multiple annotations to camera {rgb_path.stem}"
                )
            mapped_cameras.add(rgb_path.stem)
            strategies.add(strategy)
            frames.append(
                _frame(
                    mask.stem,
                    ground_truth=mask,
                    rgb_path=rgb_path,
                    annotation_rgb_path=None,
                    canonical_index=canonical_index,
                    annotation_sorted_index=annotation_index,
                    rgb_sorted_index=rgb_index,
                )
            )

        reference = frames[0]
        evaluation = frames[1:]
        scenes.append(
            {
                "scene_id": scene_id,
                "prompt_frame_ids": [reference["frame_id"]],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": [frame["frame_id"] for frame in evaluation],
                "frames": frames,
                "prompt": {
                    "type": "reference_binary_mask",
                    "frame_id": reference["frame_id"],
                    "mask_path": reference["ground_truth"],
                },
                "annotation_scene_dir": str(scene_dir.resolve()),
                "rgb_directory": str(rgb_dirs[scene_id]),
                "frame_mapping": {
                    "strategies": sorted(strategies),
                    "annotation_order": "lexicographic_base_mask_filename",
                    "rgb_order": "lexicographic_filename",
                    "known_skipped_canonical_indices": (
                        [14] if scene_id == "orchids" else []
                    ),
                    "verified": True,
                },
                "target_rgb_policy": "allowed_for_field_training_but_forbidden_at_query",
                "training_frames": [
                    {
                        "frame_id": path.stem,
                        "camera_name": path.stem,
                        "rgb_sorted_index": index,
                        "rgb_path": str(path),
                    }
                    for index, path in enumerate(rgb_images)
                ],
            }
        )

    protocol = _base_protocol(
        benchmark="SPIn-NeRF",
        dataset_version="official-spin-nerf-multiview-segmentation-2023-v1",
        prompt_type="single_reference_binary_mask",
        threshold=threshold,
    )
    protocol.update(
        {
            "cohort": list(cohort),
            "protocol_role": "full_reference_mask_support_diagnostic",
            "canonical_paper_interaction": "sparse_positive_negative_point_prompts",
            "exactly_comparable_to_published_saga": False,
            "published_saga_prompt_note": (
                "random subset inside/outside reference mask; point count and seed "
                "not disclosed"
            ),
            "reference_selection": "first_canonical_official_annotation",
            "reference_scored": False,
            "target_rgb_during_field_training": "allowed",
            "target_rgb_at_query": "forbidden_for_reusable_feature_field_track",
            "target_mask_use": "scoring_only",
            "within_scene_aggregation": "unweighted_frame_mean",
            "dataset_aggregation": (
                "unweighted_macro_over_9_available_scenes_diagnostic"
                if diagnostic_missing_fork
                else "unweighted_macro_over_10_scenes"
            ),
            "formal_10scene_eligible": not diagnostic_missing_fork,
            "missing_scenes": ["fork"] if diagnostic_missing_fork else [],
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": (
            "spin_nerf_diagnostic_9scene" if diagnostic_missing_fork else "spin_nerf"
        ),
        "annotation_root": str(annotations),
        "scene_rgb_dirs": {key: str(value) for key, value in rgb_dirs.items()},
        "protocol": protocol,
        "scenes": scenes,
    }
    if validate:
        return _finalize(manifest, check_files=True)
    manifest["protocol_hash"] = compute_protocol_hash(manifest)
    return manifest


# Both spellings are used in papers/repositories; keep discoverable aliases.
build_spinnerf_manifest = build_spin_manifest
build_spinerf_manifest = build_spin_manifest


def validate_manifest(
    manifest: Mapping[str, Any], *, check_files: bool = True
) -> dict[str, Any]:
    """Validate evaluator schema plus benchmark-specific leakage/mapping rules."""

    normalized = validate_evaluation_manifest(manifest)
    benchmark = str(manifest.get("benchmark", ""))
    expected = (
        NVOS_TASKS
        if benchmark == "nvos"
        else SPIN_SCENES
        if benchmark == "spin_nerf"
        else SPIN_DIAGNOSTIC_SCENES
        if benchmark == "spin_nerf_diagnostic_9scene"
        else None
    )
    if expected is None:
        raise ManifestError(f"Unsupported benchmark metadata: {benchmark!r}")
    scene_ids = [scene["scene_id"] for scene in normalized["scenes"]]
    if scene_ids != list(expected):
        raise ManifestError(
            f"{benchmark} scene cohort/order mismatch: got={scene_ids}, expected={list(expected)}"
        )

    raw_scenes = {
        str(scene.get("scene_id")): scene
        for scene in manifest.get("scenes", [])
        if isinstance(scene, Mapping)
    }
    protocol = normalized["protocol"]
    expected_protocol_prompt = {
        "nvos": "fixed_positive_negative_scribbles",
        "spin_nerf": "single_reference_binary_mask",
        "spin_nerf_diagnostic_9scene": "single_reference_binary_mask",
    }[benchmark]
    if protocol.get("prompt_type") != expected_protocol_prompt:
        raise ManifestError(
            f"{benchmark} protocol.prompt_type must be "
            f"{expected_protocol_prompt!r}; got {protocol.get('prompt_type')!r}. "
            "Point-prompt tracks require a separately implemented frozen sampler."
        )

    annotation_root_value = manifest.get("annotation_root")
    if not annotation_root_value:
        raise ManifestError(f"{benchmark} manifest lacks annotation_root")
    annotation_root = _absolute(str(annotation_root_value))
    evaluation_gt_paths: set[Path] = set()
    for scene in normalized["scenes"]:
        for frame_id in scene["evaluation_frame_ids"]:
            value = scene["frames"][frame_id].get("ground_truth")
            if value:
                evaluation_gt_paths.add(_absolute(str(value)))

    for scene in normalized["scenes"]:
        scene_id = scene["scene_id"]
        raw_scene = raw_scenes[scene_id]
        prompt_ids = set(scene["prompt_frame_ids"])
        evaluation_ids = set(scene["evaluation_frame_ids"])
        if prompt_ids & evaluation_ids:
            raise ManifestError(f"{scene_id} reference frame is incorrectly scored")
        if not raw_scene.get("frame_mapping", {}).get("verified"):
            raise ManifestError(f"{scene_id} RGB/annotation mapping is not verified")

        raw_frames = raw_scene.get("frames", [])
        if isinstance(raw_frames, Mapping):
            raw_frames = list(raw_frames.values())
        for frame in raw_frames:
            if not isinstance(frame, Mapping):
                raise ManifestError(f"{scene_id} contains a non-object frame")
            for key in ("rgb_path", "camera_name", "rgb_sorted_index"):
                if frame.get(key) in (None, ""):
                    raise ManifestError(f"{scene_id}/{frame.get('frame_id')} lacks {key}")
            if check_files:
                for key in ("ground_truth", "rgb_path", "annotation_rgb_path"):
                    value = frame.get(key)
                    if value is not None and not Path(str(value)).is_file():
                        raise ManifestError(
                            f"{scene_id}/{frame.get('frame_id')} {key} does not exist: {value}"
                        )
                ground_truth = frame.get("ground_truth")
                if ground_truth is not None:
                    expected_digest = frame.get("ground_truth_sha256")
                    if not expected_digest:
                        raise ManifestError(
                            f"{scene_id}/{frame.get('frame_id')} lacks ground_truth_sha256"
                        )
                    actual_digest = _sha256(_absolute(str(ground_truth)))
                    if actual_digest != str(expected_digest):
                        raise ManifestError(
                            f"{scene_id}/{frame.get('frame_id')} ground-truth SHA-256 "
                            f"mismatch: {actual_digest} != {expected_digest}"
                        )

        if benchmark == "nvos":
            training_ids = {
                str(frame.get("frame_id"))
                for frame in raw_scene.get("training_frames", [])
                if isinstance(frame, Mapping)
            }
            leaked = sorted(evaluation_ids & training_ids)
            if leaked:
                raise ManifestError(
                    f"NVOS {scene_id} target RGB leaked into training: {leaked}"
                )
            prompt = raw_scene.get("prompt", {})
            if prompt.get("type") != "positive_negative_scribbles":
                raise ManifestError(f"NVOS {scene_id} prompt metadata is invalid")
            expected_prompt_dir = (annotation_root / "scribbles" / scene_id).resolve()
            declared_hashes = protocol.get("prompt_asset_sha256", {}).get(scene_id)
            if not isinstance(declared_hashes, Mapping):
                raise ManifestError(f"NVOS {scene_id} lacks frozen prompt hashes")
            if check_files:
                for key in ("positive_path", "negative_path", "visualization_path"):
                    value = prompt.get(key)
                    if not value or not Path(str(value)).is_file():
                        raise ManifestError(f"NVOS {scene_id} missing prompt asset {key}: {value}")
                    resolved = _absolute(str(value))
                    if resolved.parent != expected_prompt_dir:
                        raise ManifestError(
                            f"NVOS {scene_id} {key} is outside its official scribble "
                            f"directory: {resolved}"
                        )
                    if resolved in evaluation_gt_paths:
                        raise ManifestError(
                            f"NVOS {scene_id} prompt asset aliases evaluation GT: {resolved}"
                        )
                for key, digest_key in (
                    ("positive_path", "positive"),
                    ("negative_path", "negative"),
                ):
                    actual_digest = _sha256(_absolute(str(prompt[key])))
                    expected_digest = str(declared_hashes.get(digest_key, ""))
                    if actual_digest != expected_digest:
                        raise ManifestError(
                            f"NVOS {scene_id} {digest_key} scribble SHA-256 mismatch: "
                            f"{actual_digest} != {expected_digest}"
                        )
        else:
            prompt = raw_scene.get("prompt", {})
            if prompt.get("type") != "reference_binary_mask":
                raise ManifestError(f"SPIn-NeRF {scene_id} prompt metadata is invalid")
            reference_id = scene["prompt_frame_ids"][0]
            reference_gt = scene["frames"][reference_id].get("ground_truth")
            if not reference_gt:
                raise ManifestError(f"SPIn-NeRF {scene_id} reference mask is missing")
            prompt_mask = prompt.get("mask_path")
            if not prompt_mask or _absolute(str(prompt_mask)) != _absolute(str(reference_gt)):
                raise ManifestError(
                    f"SPIn-NeRF {scene_id} prompt.mask_path must be the declared "
                    "reference-frame GT, never an evaluation mask"
                )
            if _absolute(str(prompt_mask)) in evaluation_gt_paths:
                raise ManifestError(
                    f"SPIn-NeRF {scene_id} reference prompt aliases evaluation GT"
                )
            expected_folder = dict(SPIN_SCENE_FOLDERS)[scene_id]
            expected_scene_dir = (annotation_root / expected_folder).resolve()
            if _absolute(str(prompt_mask)).parent != expected_scene_dir:
                raise ManifestError(
                    f"SPIn-NeRF {scene_id} reference prompt is outside the official "
                    f"annotation directory: {prompt_mask}"
                )

    return normalized


def write_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    """Validate and write a stable JSON manifest."""

    validate_manifest(manifest, check_files=True)
    destination = _absolute(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_scene_rgb_map(path: str | Path) -> dict[str, str]:
    """Load an explicit ``{scene_id: image_directory}`` JSON mapping."""

    source = _absolute(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"Could not read SPIn-NeRF RGB map {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ManifestError(f"SPIn-NeRF RGB map must be a JSON object: {source}")
    return {str(key): str(value) for key, value in payload.items()}


__all__ = [
    "ManifestError",
    "NVOS_TASKS",
    "SCHEMA_VERSION",
    "SPIN_EXPECTED_MASK_COUNTS",
    "SPIN_DIAGNOSTIC_SCENES",
    "SPIN_SCENE_FOLDERS",
    "SPIN_SCENES",
    "build_nvos_manifest",
    "build_spin_manifest",
    "build_spinerf_manifest",
    "build_spinnerf_manifest",
    "load_scene_rgb_map",
    "validate_manifest",
    "write_manifest",
]
