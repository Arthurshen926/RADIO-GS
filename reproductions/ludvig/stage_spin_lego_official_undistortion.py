"""Fail-closed staging for SPIn-NeRF's released Lego undistortion.

The Lego archive contains both the 1020x768 SIMPLE_RADIAL COLMAP input and
the official 1015x764 PINHOLE output.  RGB and annotation filenames retain a
SPIn-NeRF ``0_``/``1_`` split prefix, while COLMAP uses canonical five-digit
names.  This module validates those two representations before creating only
per-attempt symlinks; it never rewrites the downloaded asset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image

from reproductions.ludvig.run_ludvig_sam import (
    ProtocolError,
    _ensure_link,
    _image_hashes,
    _max_pose_delta,
    _read_colmap_cameras_binary,
    _read_colmap_image_poses,
    _scaled_pinhole_audit,
    _sha256,
)


LEGO_REGISTERED_VIEWS = 102
LEGO_RAW_RESOLUTION = (1020, 768)
LEGO_UNDISTORTED_RESOLUTION = (1015, 764)
LEGO_RAW_CAMERA_PARAMS = (
    745.3564686190166,
    510.0,
    384.0,
    0.004476572911261259,
)
LEGO_UNDISTORTED_CAMERA_PARAMS = (
    745.3564686190166,
    745.3564686190166,
    507.5,
    382.0,
)
LEGO_POSE_EQUIVALENCE_TOLERANCE = 1e-12
LEGO_REFERENCE_MASK = "0_00001.png"
LEGO_ANNOTATED_VIEWS = 44
LEGO_MASK_VALUES = (0, 1)
LEGO_ANNOTATION_RELATIVE = (
    Path("SPIn-NeRF")
    / "multiview_annotations"
    / "lego_real_night_radial"
)


def _expected_names() -> tuple[set[str], set[str]]:
    canonical = {
        f"{index:05d}.png" for index in range(LEGO_REGISTERED_VIEWS)
    }
    prefixed = {
        f"{'1' if index % 16 == 0 else '0'}_{index:05d}.png"
        for index in range(LEGO_REGISTERED_VIEWS)
    }
    return canonical, prefixed


def _prefixed_inventory(
    directory: Path,
    *,
    expected_names: set[str],
    expected_canonical_names: set[str],
    expected_size: tuple[int, int],
    label: str,
) -> tuple[dict[str, Path], dict[str, str]]:
    if not directory.is_dir():
        raise ProtocolError(f"Missing Lego {label} directory: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.suffix == ".png")
    names = {path.name for path in paths}
    if len(paths) != LEGO_REGISTERED_VIEWS or names != expected_names:
        raise ProtocolError(
            f"Lego {label} names are not the frozen 0_/1_ split cohort"
        )

    canonical_to_source: dict[str, Path] = {}
    source_to_canonical: dict[str, str] = {}
    sizes: set[tuple[int, int]] = set()
    for path in paths:
        match = re.fullmatch(r"[01]_(\d{5})\.png", path.name)
        if match is None:
            raise ProtocolError(f"Invalid Lego split-prefixed RGB name: {path.name}")
        canonical = f"{match.group(1)}.png"
        if canonical in canonical_to_source:
            raise ProtocolError(f"Duplicate Lego canonical RGB mapping: {canonical}")
        with Image.open(path) as image:
            sizes.add(image.size)
        canonical_to_source[canonical] = path
        source_to_canonical[path.name] = canonical
    if set(canonical_to_source) != expected_canonical_names:
        raise ProtocolError(f"Lego {label} prefix removal is not a complete bijection")
    if sizes != {expected_size}:
        raise ProtocolError(
            f"Lego {label} dimensions changed: expected {expected_size}, "
            f"found {sorted(sizes)}"
        )
    return canonical_to_source, source_to_canonical


def _validate_annotations(
    annotation_scene: Path,
    *,
    image_mapping: dict[str, str],
    converted_poses: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    if not annotation_scene.is_dir():
        raise ProtocolError(f"Missing Lego annotations: {annotation_scene}")
    annotation_paths = sorted(annotation_scene.glob("*.png"))
    base_masks = sorted(
        path
        for path in annotation_paths
        if not path.stem.endswith("_cutout")
        and not path.stem.endswith("_pseudo")
    )
    if (
        len(base_masks) != LEGO_ANNOTATED_VIEWS
        or len(annotation_paths) != 3 * LEGO_ANNOTATED_VIEWS
    ):
        raise ProtocolError(
            "Lego annotations must contain 44 masks, 44 cutouts, and 44 pseudos"
        )
    if base_masks[0].name != LEGO_REFERENCE_MASK:
        raise ProtocolError(
            f"Lego reference role changed: expected {LEGO_REFERENCE_MASK} first"
        )

    annotation_cameras: list[str] = []
    for mask in base_masks:
        if (
            re.fullmatch(r"[01]_\d{5}\.png", mask.name) is None
            or mask.name not in image_mapping
        ):
            raise ProtocolError(
                f"Lego annotation has no exact split-prefix RGB mapping: {mask.name}"
            )
        canonical = image_mapping[mask.name]
        if canonical not in converted_poses:
            raise ProtocolError(
                f"Lego annotation maps to an unregistered camera: {mask.name}"
            )
        variants = {
            "mask": mask,
            "cutout": annotation_scene / f"{mask.stem}_cutout.png",
            "pseudo": annotation_scene / f"{mask.stem}_pseudo.png",
        }
        expected_modes = {"mask": "L", "cutout": "RGBA", "pseudo": "RGB"}
        for role, path in variants.items():
            if not path.is_file():
                raise ProtocolError(f"Missing Lego annotation {role} for {mask.name}")
            with Image.open(path) as image:
                if (
                    image.size != LEGO_UNDISTORTED_RESOLUTION
                    or image.mode != expected_modes[role]
                ):
                    raise ProtocolError(
                        f"Lego annotation {path.name} changed dimensions or mode"
                    )
                if role == "mask" and set(image.tobytes()) != set(
                    LEGO_MASK_VALUES
                ):
                    raise ProtocolError(
                        "Lego ground-truth mask encoding changed from exact "
                        f"values {LEGO_MASK_VALUES}: {path.name}"
                    )
        annotation_cameras.append(Path(canonical).stem)
    if len(set(annotation_cameras)) != LEGO_ANNOTATED_VIEWS:
        raise ProtocolError("Lego annotation-to-camera mapping is not one-to-one")

    roles = {
        "annotation_scene": str(annotation_scene.resolve()),
        "reference_mask": base_masks[0].name,
        "reference_camera": annotation_cameras[0],
        "reference_scored": False,
        "target_masks": LEGO_ANNOTATED_VIEWS - 1,
        "target_masks_scoring_only": True,
        "annotation_rgb_used_for_training": False,
        "all_masks": LEGO_ANNOTATED_VIEWS,
        "mask_dimensions": list(LEGO_UNDISTORTED_RESOLUTION),
        "mask_values": list(LEGO_MASK_VALUES),
        "mapping_rule": "remove_exact_0_or_1_split_prefix",
        "all_mapped_to_registered_cameras": True,
    }
    hashes = {path.name: _sha256(path) for path in annotation_paths}
    return roles, hashes


def _stage_spin_lego_pinhole_colmap(
    source_scene: Path,
    annotation_scene: Path,
    staging_scene: Path,
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    """Audit and stage the exact official 102-view Lego PINHOLE output."""

    if (target_width, target_height) != LEGO_UNDISTORTED_RESOLUTION:
        raise ProtocolError(
            "Lego staging requires the frozen 1015x764 evaluation size; "
            f"found {target_width}x{target_height}"
        )
    expected_canonical_names, expected_prefixed_names = _expected_names()
    raw_sparse = source_scene / "sparse" / "0"
    converted_sparse = source_scene / "sparse"

    raw_cameras = _read_colmap_cameras_binary(raw_sparse / "cameras.bin")
    converted_cameras = _read_colmap_cameras_binary(
        converted_sparse / "cameras.bin"
    )
    if len(raw_cameras) != 1 or len(converted_cameras) != 1:
        raise ProtocolError(
            "Audited Lego staging expects one raw and one converted camera"
        )
    raw_camera = raw_cameras[0]
    camera = converted_cameras[0]
    if (
        raw_camera["camera_id"] != 1
        or raw_camera["model"] != "SIMPLE_RADIAL"
        or (raw_camera["width"], raw_camera["height"]) != LEGO_RAW_RESOLUTION
        or tuple(raw_camera["params"]) != LEGO_RAW_CAMERA_PARAMS
    ):
        raise ProtocolError(
            "Lego raw camera contract changed from the frozen SIMPLE_RADIAL "
            "1020x768 calibration"
        )
    if (
        camera["camera_id"] != 1
        or camera["model"] != "PINHOLE"
        or (camera["width"], camera["height"])
        != LEGO_UNDISTORTED_RESOLUTION
        or tuple(camera["params"]) != LEGO_UNDISTORTED_CAMERA_PARAMS
    ):
        raise ProtocolError(
            "Lego official undistortion changed from the frozen PINHOLE "
            "1015x764 calibration"
        )
    intrinsics_audit = _scaled_pinhole_audit(camera, target_width, target_height)

    raw_poses = _read_colmap_image_poses(raw_sparse / "images.bin")
    converted_poses = _read_colmap_image_poses(converted_sparse / "images.bin")
    if set(raw_poses) != expected_canonical_names:
        raise ProtocolError(
            "Lego raw sparse names are not the frozen 00000..00101 cohort"
        )
    if set(converted_poses) != expected_canonical_names:
        raise ProtocolError(
            "Lego converted sparse names are not the frozen 00000..00101 cohort"
        )
    max_qvec_delta, max_tvec_delta = _max_pose_delta(raw_poses, converted_poses)
    if (
        max_qvec_delta > LEGO_POSE_EQUIVALENCE_TOLERANCE
        or max_tvec_delta > LEGO_POSE_EQUIVALENCE_TOLERANCE
    ):
        raise ProtocolError(
            "Lego official undistortion changed camera poses beyond 1e-12: "
            f"q={max_qvec_delta}, t={max_tvec_delta}"
        )
    if {pose["camera_id"] for pose in raw_poses.values()} != {1} or {
        pose["camera_id"] for pose in converted_poses.values()
    } != {1}:
        raise ProtocolError(
            "Lego registered images do not all use the audited shared camera"
        )

    converted_images, image_mapping = _prefixed_inventory(
        source_scene / "images",
        expected_names=expected_prefixed_names,
        expected_canonical_names=expected_canonical_names,
        expected_size=LEGO_UNDISTORTED_RESOLUTION,
        label="official undistorted RGB",
    )
    raw_resized_images, raw_image_mapping = _prefixed_inventory(
        source_scene / "images_resized",
        expected_names=expected_prefixed_names,
        expected_canonical_names=expected_canonical_names,
        expected_size=LEGO_RAW_RESOLUTION,
        label="raw resized RGB",
    )
    if image_mapping != raw_image_mapping:
        raise ProtocolError(
            "Lego raw and undistorted RGB split-prefix mappings differ"
        )
    annotation_roles, annotation_hashes = _validate_annotations(
        annotation_scene,
        image_mapping=image_mapping,
        converted_poses=converted_poses,
    )

    sparse_names = ("cameras.bin", "images.bin", "points3D.bin")
    raw_sparse_hashes = {name: _sha256(raw_sparse / name) for name in sparse_names}
    converted_sparse_hashes = {
        name: _sha256(converted_sparse / name) for name in sparse_names
    }
    rgb_hashes = _image_hashes(source_scene / "images")
    raw_resized_rgb_hashes = _image_hashes(source_scene / "images_resized")
    mapping_sha256 = hashlib.sha256(
        json.dumps(
            image_mapping,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    # Mutate only the new per-attempt staging directory, and only after every
    # source-side validation and hash has succeeded.
    for canonical, source in sorted(converted_images.items()):
        _ensure_link(staging_scene / "images" / canonical, source)
    for name in sparse_names:
        _ensure_link(
            staging_scene / "sparse" / "0" / name,
            converted_sparse / name,
        )

    return {
        "strategy": "stage_native_spin_lego_official_undistortion",
        "source_scene": str(source_scene.resolve()),
        "raw_sparse_source": str(raw_sparse.resolve()),
        "converted_sparse_source": str(converted_sparse.resolve()),
        "staged_scene": str(staging_scene.resolve()),
        "raw_camera_model": raw_camera["model"],
        "raw_camera_dimensions": list(LEGO_RAW_RESOLUTION),
        "raw_camera_params": raw_camera["params"],
        "camera_model": camera["model"],
        "camera_metadata_dimensions": list(LEGO_UNDISTORTED_RESOLUTION),
        "camera_params": camera["params"],
        "registered_images": len(converted_poses),
        "rgb_images": len(converted_images),
        "rgb_dimensions": list(LEGO_UNDISTORTED_RESOLUTION),
        "raw_resized_rgb_images": len(raw_resized_images),
        "raw_resized_rgb_dimensions": list(LEGO_RAW_RESOLUTION),
        "max_qvec_delta_vs_raw_sparse": max_qvec_delta,
        "max_tvec_delta_vs_raw_sparse": max_tvec_delta,
        "pose_equivalence_tolerance": LEGO_POSE_EQUIVALENCE_TOLERANCE,
        "intrinsics": intrinsics_audit,
        "image_name_mapping": {
            "rule": "remove_exact_0_or_1_split_prefix",
            "entries": image_mapping,
            "mapping_sha256": mapping_sha256,
            "bijective": True,
        },
        "annotation_roles": annotation_roles,
        "raw_sparse_sha256": raw_sparse_hashes,
        "source_sha256": converted_sparse_hashes,
        "rgb_sha256": rgb_hashes,
        "raw_resized_rgb_sha256": raw_resized_rgb_hashes,
        "annotation_sha256": annotation_hashes,
        "raw_dataset_modified": False,
    }
