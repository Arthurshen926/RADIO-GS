#!/usr/bin/env python3
"""Train released NVOS all-view geometry with pinned original 3DGS.

The LUDVIG repository vendors gaussian-splatting as a normal directory, so it
does not provide a gitlink that can be checked out directly.  The companion
lock file records the official revision whose training entrypoint is byte
identical to LUDVIG's vendored entrypoint.  This launcher verifies that
revision and all of its relevant gitlinks, stages the already-undistorted
NVOS PINHOLE model, and serializes the only GPU section with the shared GPU 0
lock.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any
from urllib.parse import unquote, urlparse

from reproductions.ludvig.run_ludvig_sam import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_DRIVER_LIBRARY_DIR,
    LOCK_PATH,
    NVOS_GEOMETRY_REGISTERED_IMAGES,
    NVOS_TASK_TO_GEOMETRY_SCENE,
    _driver_library,
    _stage_nvos_pinhole_colmap,
    _stage_spin_llff_pinhole_colmap,
    _stage_spin_truck_pinhole_colmap,
)
from reproductions.ludvig.stage_spin_lego_official_undistortion import (
    LEGO_ANNOTATION_RELATIVE,
    LEGO_ANNOTATED_VIEWS,
    LEGO_MASK_VALUES,
    LEGO_REFERENCE_MASK,
    _stage_spin_lego_pinhole_colmap,
)


ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = Path(__file__).with_name("official_3dgs.lock.json")
DEFAULT_RELEASED_ALL_VIEW_ROOT = (
    ROOT
    / "output"
    / "protocol_audit_20260731"
    / "ludvig"
    / "nvos"
    / "released_all_view"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_RELEASED_ALL_VIEW_ROOT / "fern" / "training"
OFFICIAL_3DGS_COMMIT = "f7a116fb1397d9842239127d39dc212f93171f70"
LUDVIG_COMMIT = "4461fc515439bb498a75d71738a1e73cf7a452ed"
RASTERIZER_COMMIT = "8064f52ca233942bdec2d1a1451c026deedd320b"
SIMPLE_KNN_COMMIT = "44f764299fa305faf6ec5ebd99939e0508331503"
GLM_COMMIT = "5c46b9c07008ae65cb81ab79cd677ecc1934b903"
TRAIN_ENTRYPOINT_SHA256 = (
    "c5a61947e2abcf56bf83451ae9633799d96894910ea2982a01f209c47cec462d"
)
NVOS_GEOMETRY_SCENES = tuple(dict.fromkeys(NVOS_TASK_TO_GEOMETRY_SCENE.values()))
TARGET_WIDTH = 1600
TARGET_HEIGHT = 1199

NATIVE_NVOS_PINHOLE_CONTRACT = "native_nvos_llff_pinhole_undistortion"
VERIFIED_SPIN_NVOS_PINHOLE_REUSE_CONTRACT = (
    "verified_byte_identical_spin_raw_reusing_nvos_pinhole_undistortion"
)
NATIVE_SPIN_TRUCK_PINHOLE_CONTRACT = (
    "native_spin_graphdeco_truck_half_resolution_pinhole"
)
NATIVE_SPIN_LEGO_PINHOLE_CONTRACT = (
    "native_spin_lego_official_undistortion_with_split_prefix_mapping"
)
NATIVE_SPIN_PINECONE_PINHOLE_CONTRACT = (
    "native_spin_pinecone_pinned_colmap_undistortion"
)


@dataclass(frozen=True)
class AllViewTrainingSpec:
    """Frozen dataset identity for one exact original-3DGS training job."""

    benchmark: str
    scene: str
    geometry_scene: str
    converted_source_relative: Path
    expected_registered_images: int
    evaluation_render_resolution: tuple[int, int]
    default_output_root: Path
    source_asset_contract: str
    raw_identity_source_relative: Path | None = None


class TrainingProtocolError(RuntimeError):
    """Raised before GPU work when exact training provenance is not satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _automatic_training_resolution(
    source_width: int,
    source_height: int,
) -> tuple[int, int]:
    """Reproduce pinned original-3DGS ``resolution=-1`` integer behavior."""

    if (
        type(source_width) is not int
        or type(source_height) is not int
        or source_width <= 0
        or source_height <= 0
    ):
        raise TrainingProtocolError(
            f"Invalid source resolution: {source_width}x{source_height}"
        )
    global_down = source_width / TARGET_WIDTH if source_width > TARGET_WIDTH else 1
    return (
        int(source_width / global_down),
        int(source_height / global_down),
    )


def _training_output_root(scene: str, output_root: Path | None) -> Path:
    if scene not in NVOS_GEOMETRY_REGISTERED_IMAGES:
        raise TrainingProtocolError(f"Unknown NVOS geometry scene: {scene}")
    if output_root is not None:
        return output_root.resolve()
    return (DEFAULT_RELEASED_ALL_VIEW_ROOT / scene / "training").resolve()


def _validate_relative_asset_path(path: Path, label: str) -> None:
    if not isinstance(path, Path) or path.is_absolute() or ".." in path.parts:
        raise TrainingProtocolError(
            f"{label} must be a benchmark-root-relative path without '..': {path}"
        )


def _validate_training_spec(spec: AllViewTrainingSpec) -> None:
    """Reject internally inconsistent dataset identities before creating a run."""

    for label, value in (
        ("benchmark", spec.benchmark),
        ("scene", spec.scene),
        ("geometry_scene", spec.geometry_scene),
    ):
        if not isinstance(value, str) or not value or value.strip() != value:
            raise TrainingProtocolError(f"Invalid training spec {label}: {value!r}")
    if spec.scene != spec.geometry_scene:
        raise TrainingProtocolError(
            "Exact all-view geometry training requires scene == geometry_scene"
        )
    _validate_relative_asset_path(
        spec.converted_source_relative,
        "converted_source_relative",
    )
    if (
        type(spec.expected_registered_images) is not int
        or spec.expected_registered_images <= 0
    ):
        raise TrainingProtocolError(
            "expected_registered_images must be a positive integer"
        )
    render_resolution = spec.evaluation_render_resolution
    if (
        not isinstance(render_resolution, tuple)
        or len(render_resolution) != 2
        or any(type(value) is not int or value <= 0 for value in render_resolution)
    ):
        raise TrainingProtocolError(
            "evaluation_render_resolution must be two positive integers"
        )
    if not isinstance(spec.default_output_root, Path):
        raise TrainingProtocolError("default_output_root must be a Path")

    if spec.source_asset_contract == NATIVE_NVOS_PINHOLE_CONTRACT:
        if spec.benchmark != "NVOS" or spec.raw_identity_source_relative is not None:
            raise TrainingProtocolError(
                "Native NVOS source contract requires benchmark=NVOS and no "
                "cross-dataset identity source"
            )
    elif spec.source_asset_contract == NATIVE_SPIN_TRUCK_PINHOLE_CONTRACT:
        if (
            spec.benchmark != "SPIn-NeRF"
            or spec.scene != "truck"
            or spec.raw_identity_source_relative is not None
            or spec.expected_registered_images != 251
            or spec.evaluation_render_resolution != (979, 546)
        ):
            raise TrainingProtocolError(
                "Native Truck source contract requires the frozen SPIn-NeRF "
                "truck/251-view/979x546 asset and no cross-dataset identity source"
            )
    elif spec.source_asset_contract == NATIVE_SPIN_LEGO_PINHOLE_CONTRACT:
        if (
            spec.benchmark != "SPIn-NeRF"
            or spec.scene != "lego"
            or spec.raw_identity_source_relative is not None
            or spec.expected_registered_images != 102
            or spec.evaluation_render_resolution != (1015, 764)
        ):
            raise TrainingProtocolError(
                "Native Lego source contract requires the frozen SPIn-NeRF "
                "lego/102-view/1015x764 official undistortion and no "
                "cross-dataset identity source"
            )
    elif spec.source_asset_contract == NATIVE_SPIN_PINECONE_PINHOLE_CONTRACT:
        if (
            spec.benchmark != "SPIn-NeRF"
            or spec.scene != "pinecone"
            or spec.raw_identity_source_relative is None
            or spec.expected_registered_images != 99
            or spec.evaluation_render_resolution != (1600, 1199)
        ):
            raise TrainingProtocolError(
                "Native Pinecone source contract requires the frozen "
                "SPIn-NeRF pinecone/99-view/1600x1199 audited COLMAP "
                "undistortion and an explicit raw identity source"
            )
        _validate_relative_asset_path(
            spec.raw_identity_source_relative,
            "raw_identity_source_relative",
        )
    elif spec.source_asset_contract == VERIFIED_SPIN_NVOS_PINHOLE_REUSE_CONTRACT:
        if spec.benchmark != "SPIn-NeRF" or spec.raw_identity_source_relative is None:
            raise TrainingProtocolError(
                "Verified SPIn reuse requires benchmark=SPIn-NeRF and an explicit "
                "raw identity source"
            )
        _validate_relative_asset_path(
            spec.raw_identity_source_relative,
            "raw_identity_source_relative",
        )
    else:
        raise TrainingProtocolError(
            f"Unknown source asset contract: {spec.source_asset_contract}"
        )


def _nvos_training_spec(scene: str) -> AllViewTrainingSpec:
    expected_registered_images = NVOS_GEOMETRY_REGISTERED_IMAGES.get(scene)
    if expected_registered_images is None:
        raise TrainingProtocolError(f"Unknown NVOS geometry scene: {scene}")
    return AllViewTrainingSpec(
        benchmark="NVOS",
        scene=scene,
        geometry_scene=scene,
        converted_source_relative=(
            Path("NVOS") / "llff_undistorted" / f"{scene}_undistort"
        ),
        expected_registered_images=expected_registered_images,
        evaluation_render_resolution=(TARGET_WIDTH, TARGET_HEIGHT),
        default_output_root=_training_output_root(scene, None),
        source_asset_contract=NATIVE_NVOS_PINHOLE_CONTRACT,
    )


def _stage_training_input(
    spec: AllViewTrainingSpec,
    benchmark_root: Path,
    staging_scene: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stage a PINHOLE scene and return camera plus source-identity audits."""

    converted_source = benchmark_root / spec.converted_source_relative
    evaluation_width, evaluation_height = spec.evaluation_render_resolution
    if spec.source_asset_contract == NATIVE_NVOS_PINHOLE_CONTRACT:
        pinhole_audit = _stage_nvos_pinhole_colmap(
            converted_source,
            staging_scene,
            evaluation_width,
            evaluation_height,
        )
        source_audit = {
            "source_asset_contract": spec.source_asset_contract,
            "converted_source_scene": str(converted_source.resolve()),
            "raw_identity_source_scene": None,
            "raw_scene_identity_proven": None,
            "raw_dataset_modified": pinhole_audit.get("raw_dataset_modified"),
        }
        return pinhole_audit, source_audit

    if spec.source_asset_contract == NATIVE_SPIN_TRUCK_PINHOLE_CONTRACT:
        pinhole_audit = _stage_spin_truck_pinhole_colmap(
            converted_source,
            staging_scene,
            evaluation_width,
            evaluation_height,
        )
        if pinhole_audit.get("strategy") != (
            "stage_native_graphdeco_truck_pinhole"
        ):
            raise TrainingProtocolError("Truck staging strategy changed")
        if pinhole_audit.get("raw_dataset_modified") is not False:
            raise TrainingProtocolError("Truck staging must not modify source assets")
        if pinhole_audit.get("camera_metadata_dimensions") != [1957, 1091]:
            raise TrainingProtocolError(
                "Truck staging omitted the frozen 1957x1091 COLMAP metadata "
                "resolution"
            )
        sparse_hashes = pinhole_audit.get("source_sha256")
        if not isinstance(sparse_hashes, dict) or set(sparse_hashes) != {
            "cameras.bin",
            "images.bin",
            "points3D.bin",
        }:
            raise TrainingProtocolError("Incomplete Truck sparse hash audit")
        rgb_hashes = pinhole_audit.get("rgb_sha256")
        if not isinstance(rgb_hashes, dict) or len(rgb_hashes) != (
            spec.expected_registered_images
        ):
            raise TrainingProtocolError("Incomplete Truck RGB hash audit")
        all_hashes = [*sparse_hashes.values(), *rgb_hashes.values()]
        if any(
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in all_hashes
        ):
            raise TrainingProtocolError("Invalid digest in Truck source audit")
        source_audit = {
            "source_asset_contract": spec.source_asset_contract,
            "strategy": pinhole_audit["strategy"],
            "native_source_scene": str(converted_source.resolve()),
            "raw_identity_source_scene": None,
            "raw_scene_identity_proven": None,
            "source_sparse_sha256": sparse_hashes,
            "source_rgb_sha256": rgb_hashes,
            "source_rgb_images": len(rgb_hashes),
            "raw_dataset_modified": False,
        }
        return pinhole_audit, source_audit

    if spec.source_asset_contract == NATIVE_SPIN_LEGO_PINHOLE_CONTRACT:
        annotation_source = benchmark_root / LEGO_ANNOTATION_RELATIVE
        pinhole_audit = _stage_spin_lego_pinhole_colmap(
            converted_source,
            annotation_source,
            staging_scene,
            evaluation_width,
            evaluation_height,
        )
        if pinhole_audit.get("strategy") != (
            "stage_native_spin_lego_official_undistortion"
        ):
            raise TrainingProtocolError("Lego staging strategy changed")
        if pinhole_audit.get("raw_dataset_modified") is not False:
            raise TrainingProtocolError("Lego staging must not modify source assets")

        expected_sparse_names = {"cameras.bin", "images.bin", "points3D.bin"}
        raw_sparse_hashes = pinhole_audit.get("raw_sparse_sha256")
        sparse_hashes = pinhole_audit.get("source_sha256")
        if (
            not isinstance(raw_sparse_hashes, dict)
            or set(raw_sparse_hashes) != expected_sparse_names
            or not isinstance(sparse_hashes, dict)
            or set(sparse_hashes) != expected_sparse_names
        ):
            raise TrainingProtocolError("Incomplete Lego sparse hash audit")

        rgb_hashes = pinhole_audit.get("rgb_sha256")
        raw_resized_rgb_hashes = pinhole_audit.get("raw_resized_rgb_sha256")
        annotation_hashes = pinhole_audit.get("annotation_sha256")
        if (
            not isinstance(rgb_hashes, dict)
            or len(rgb_hashes) != spec.expected_registered_images
            or not isinstance(raw_resized_rgb_hashes, dict)
            or len(raw_resized_rgb_hashes) != spec.expected_registered_images
        ):
            raise TrainingProtocolError("Incomplete Lego RGB hash audit")
        if (
            not isinstance(annotation_hashes, dict)
            or len(annotation_hashes) != 3 * LEGO_ANNOTATED_VIEWS
        ):
            raise TrainingProtocolError("Incomplete Lego annotation hash audit")

        mapping = pinhole_audit.get("image_name_mapping")
        mapping_entries = mapping.get("entries") if isinstance(mapping, dict) else None
        if (
            not isinstance(mapping, dict)
            or mapping.get("rule") != "remove_exact_0_or_1_split_prefix"
            or mapping.get("bijective") is not True
            or not isinstance(mapping_entries, dict)
            or len(mapping_entries) != spec.expected_registered_images
        ):
            raise TrainingProtocolError("Incomplete Lego split-prefix mapping audit")

        annotation_roles = pinhole_audit.get("annotation_roles")
        required_roles = {
            "reference_mask": LEGO_REFERENCE_MASK,
            "reference_camera": "00001",
            "reference_scored": False,
            "target_masks": LEGO_ANNOTATED_VIEWS - 1,
            "target_masks_scoring_only": True,
            "annotation_rgb_used_for_training": False,
            "all_masks": LEGO_ANNOTATED_VIEWS,
            "mask_values": list(LEGO_MASK_VALUES),
            "all_mapped_to_registered_cameras": True,
        }
        if not isinstance(annotation_roles, dict) or any(
            type(annotation_roles.get(key)) is not type(expected)
            or annotation_roles.get(key) != expected
            for key, expected in required_roles.items()
        ):
            raise TrainingProtocolError("Lego annotation-role audit changed")

        all_hashes = [
            *raw_sparse_hashes.values(),
            *sparse_hashes.values(),
            *rgb_hashes.values(),
            *raw_resized_rgb_hashes.values(),
            *annotation_hashes.values(),
            mapping.get("mapping_sha256"),
        ]
        if any(
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in all_hashes
        ):
            raise TrainingProtocolError("Invalid digest in Lego source audit")

        source_audit = {
            "source_asset_contract": spec.source_asset_contract,
            "strategy": pinhole_audit["strategy"],
            "native_source_scene": str(converted_source.resolve()),
            "annotation_source_scene": str(annotation_source.resolve()),
            "raw_identity_source_scene": None,
            "raw_scene_identity_proven": None,
            "raw_sparse_sha256": raw_sparse_hashes,
            "source_sparse_sha256": sparse_hashes,
            "source_rgb_sha256": rgb_hashes,
            "raw_resized_rgb_sha256": raw_resized_rgb_hashes,
            "annotation_sha256": annotation_hashes,
            "image_name_mapping": mapping,
            "annotation_roles": annotation_roles,
            "source_rgb_images": len(rgb_hashes),
            "raw_dataset_modified": False,
        }
        return pinhole_audit, source_audit

    raw_source = benchmark_root / spec.raw_identity_source_relative
    identity_audit = _stage_spin_llff_pinhole_colmap(
        raw_source,
        converted_source,
        staging_scene,
        evaluation_width,
        evaluation_height,
    )
    if identity_audit.get("strategy") != (
        "reuse_verified_identical_llff_colmap_undistortion"
    ):
        raise TrainingProtocolError("SPIn source identity audit strategy changed")
    if identity_audit.get("raw_scene_identity_proven") is not True:
        raise TrainingProtocolError("SPIn raw scene identity was not proven")
    if identity_audit.get("raw_dataset_modified") is not False:
        raise TrainingProtocolError("SPIn raw dataset must remain unmodified")
    if identity_audit.get("raw_rgb_images") != spec.expected_registered_images:
        raise TrainingProtocolError(
            f"Expected {spec.expected_registered_images} raw SPIn RGBs, found "
            f"{identity_audit.get('raw_rgb_images')}"
        )
    sparse_hashes = identity_audit.get("raw_sparse_sha256")
    if not isinstance(sparse_hashes, dict) or set(sparse_hashes) != {
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    }:
        raise TrainingProtocolError("Incomplete raw SPIn sparse hash audit")
    rgb_hashes = identity_audit.get("raw_rgb_sha256")
    if not isinstance(rgb_hashes, dict) or len(rgb_hashes) != (
        spec.expected_registered_images
    ):
        raise TrainingProtocolError("Incomplete raw SPIn RGB hash audit")
    all_hashes = [*sparse_hashes.values(), *rgb_hashes.values()]
    if any(
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in all_hashes
    ):
        raise TrainingProtocolError("Invalid digest in raw SPIn identity audit")
    pinhole_audit = identity_audit.get("pinhole")
    if not isinstance(pinhole_audit, dict):
        raise TrainingProtocolError("SPIn source identity audit omitted PINHOLE audit")
    source_audit = {
        "source_asset_contract": spec.source_asset_contract,
        "strategy": identity_audit["strategy"],
        "raw_identity_source_scene": str(raw_source.resolve()),
        "converted_source_scene": str(converted_source.resolve()),
        "raw_scene_identity_proven": True,
        "raw_sparse_sha256": sparse_hashes,
        "raw_rgb_sha256": rgb_hashes,
        "raw_rgb_images": identity_audit["raw_rgb_images"],
        "raw_dataset_modified": False,
    }
    return pinhole_audit, source_audit


def _git(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_head(checkout: Path) -> str:
    return _git(checkout, "rev-parse", "HEAD")


def _require_clean(checkout: Path, label: str) -> None:
    status = _git(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise TrainingProtocolError(
            f"{label} checkout has tracked modifications:\n{status}"
        )


def _literal_class_defaults(source: str, class_name: str) -> dict[str, Any]:
    """Extract literal ``self.<name> = value`` defaults from ``__init__``."""

    tree = ast.parse(source)
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if class_node is None:
        raise TrainingProtocolError(f"Missing class {class_name}")
    init_node = next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ),
        None,
    )
    if init_node is None:
        raise TrainingProtocolError(f"Missing {class_name}.__init__")
    defaults: dict[str, Any] = {}
    for node in init_node.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue
        try:
            defaults[target.attr] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return defaults


def _validate_source(
    checkout: Path,
    ludvig_checkout: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    if lock.get("commit") != OFFICIAL_3DGS_COMMIT:
        raise TrainingProtocolError("official_3dgs.lock.json has an unknown commit")
    if _git_head(checkout) != OFFICIAL_3DGS_COMMIT:
        raise TrainingProtocolError(
            f"3DGS checkout must be {OFFICIAL_3DGS_COMMIT}; "
            f"found {_git_head(checkout)}"
        )
    _require_clean(checkout, "3DGS")

    submodules = {
        "submodules/diff-gaussian-rasterization": RASTERIZER_COMMIT,
        "submodules/simple-knn": SIMPLE_KNN_COMMIT,
    }
    resolved_submodules: dict[str, str] = {}
    for relative, expected in submodules.items():
        submodule = checkout / relative
        if not submodule.exists():
            raise TrainingProtocolError(f"Missing initialized submodule: {submodule}")
        found = _git_head(submodule)
        if found != expected:
            raise TrainingProtocolError(
                f"{relative} must be {expected}; found {found}"
            )
        _require_clean(submodule, relative)
        resolved_submodules[relative] = found
    glm = checkout / "submodules" / "diff-gaussian-rasterization" / "third_party" / "glm"
    if not glm.exists() or _git_head(glm) != GLM_COMMIT:
        raise TrainingProtocolError(
            f"rasterizer GLM must be initialized at {GLM_COMMIT}"
        )
    _require_clean(glm, "rasterizer third_party/glm")
    resolved_submodules["submodules/diff-gaussian-rasterization/third_party/glm"] = (
        _git_head(glm)
    )

    source_hashes = {}
    for relative, expected in lock["source_sha256"].items():
        source_path = checkout / relative
        found = _sha256(source_path)
        if found != expected:
            raise TrainingProtocolError(
                f"Source hash mismatch for {relative}: expected {expected}, found {found}"
            )
        source_hashes[relative] = found

    train_hash = _sha256(checkout / "train.py")
    if train_hash != TRAIN_ENTRYPOINT_SHA256:
        raise TrainingProtocolError("Pinned official train.py hash changed")
    if _git_head(ludvig_checkout) != LUDVIG_COMMIT:
        raise TrainingProtocolError(
            f"LUDVIG checkout must be {LUDVIG_COMMIT}; "
            f"found {_git_head(ludvig_checkout)}"
        )
    ludvig_train = ludvig_checkout / "gaussiansplatting" / "train.py"
    if _sha256(ludvig_train) != train_hash:
        raise TrainingProtocolError(
            "Official pinned train.py is no longer byte-identical to LUDVIG's "
            "vendored training entrypoint"
        )
    tree_entry = _git(
        ludvig_checkout,
        "ls-tree",
        "HEAD",
        "gaussiansplatting",
    )
    if not tree_entry.startswith("040000 tree "):
        raise TrainingProtocolError(
            "Expected LUDVIG gaussian-splatting to be a vendored tree, not a gitlink"
        )

    argument_source = (checkout / "arguments" / "__init__.py").read_text(
        encoding="utf-8"
    )
    optimization_defaults = _literal_class_defaults(
        argument_source, "OptimizationParams"
    )
    model_defaults = _literal_class_defaults(argument_source, "ModelParams")
    for key, expected in lock["training_defaults"].items():
        if optimization_defaults.get(key) != expected:
            raise TrainingProtocolError(
                f"Official 3DGS default {key} changed: "
                f"expected {expected}, found {optimization_defaults.get(key)}"
            )
    for key, expected in lock["model_defaults"].items():
        source_key = f"_{key}" if f"_{key}" in model_defaults else key
        if model_defaults.get(source_key) != expected:
            raise TrainingProtocolError(
                f"Official 3DGS model default {key} changed: expected "
                f"{expected}, found {model_defaults.get(source_key)}"
            )

    return {
        "repository": lock["repository"],
        "commit": OFFICIAL_3DGS_COMMIT,
        "checkout": str(checkout),
        "tracked_source_clean": True,
        "source_sha256": source_hashes,
        "submodules": resolved_submodules,
        "training_defaults": lock["training_defaults"],
        "model_defaults": lock["model_defaults"],
        "ludvig_commit": LUDVIG_COMMIT,
        "ludvig_vendored_tree_entry": tree_entry,
        "ludvig_train_entrypoint_sha256": train_hash,
        "reconstruction_evidence": lock["selection"],
    }


def _extension_path(dependency_root: Path, package: str) -> Path:
    candidates = sorted((dependency_root / package).glob("_C*.so"))
    if len(candidates) != 1:
        raise TrainingProtocolError(
            f"Expected exactly one compiled {package} extension under "
            f"{dependency_root}; found {len(candidates)}"
        )
    return candidates[0]


def _runtime_environment(
    dependency_root: Path,
    driver_library_dir: Path,
    *,
    expose_gpu: bool,
) -> tuple[dict[str, str], Path]:
    driver_library = _driver_library(driver_library_dir)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0" if expose_gpu else ""
    environment["PYTHONPATH"] = str(dependency_root) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    environment["LD_LIBRARY_PATH"] = (
        str(driver_library_dir)
        + os.pathsep
        + "/usr/local/cuda/lib64"
        + os.pathsep
        + environment.get("LD_LIBRARY_PATH", "")
    )
    return environment, driver_library


def _validate_dependencies(
    checkout: Path,
    python: Path,
    dependency_root: Path,
    driver_library_dir: Path,
    preflight_log: Path,
) -> dict[str, Any]:
    if not python.exists():
        raise TrainingProtocolError(f"Missing Python interpreter: {python}")
    if not dependency_root.is_dir():
        raise TrainingProtocolError(
            f"Missing isolated dependency directory: {dependency_root}"
        )
    extensions = {
        package: _extension_path(dependency_root, package)
        for package in ("diff_gaussian_rasterization", "simple_knn")
    }
    expected_install_sources = {
        "diff_gaussian_rasterization": (
            checkout / "submodules" / "diff-gaussian-rasterization"
        ),
        "simple_knn": checkout / "submodules" / "simple-knn",
    }
    install_sources = {}
    for package, expected_source in expected_install_sources.items():
        dist_info = sorted(
            dependency_root.glob(f"{package}-*.dist-info/direct_url.json")
        )
        if len(dist_info) != 1:
            raise TrainingProtocolError(
                f"Expected one direct_url.json for {package}; found {len(dist_info)}"
            )
        direct_url = json.loads(dist_info[0].read_text(encoding="utf-8"))["url"]
        parsed = urlparse(direct_url)
        if parsed.scheme != "file":
            raise TrainingProtocolError(
                f"{package} was not built from the locked local checkout"
            )
        installed_from = Path(unquote(parsed.path)).resolve()
        if installed_from != expected_source.resolve():
            raise TrainingProtocolError(
                f"{package} was built from {installed_from}, expected "
                f"{expected_source.resolve()}"
            )
        install_sources[package] = {
            "direct_url": direct_url,
            "resolved_source": str(installed_from),
        }
    environment, driver_library = _runtime_environment(
        dependency_root,
        driver_library_dir,
        expose_gpu=False,
    )
    command = [str(python), str(checkout / "train.py"), "--help"]
    completed = subprocess.run(
        command,
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
    )
    preflight_log.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise TrainingProtocolError(
            f"Pinned train.py dependency preflight failed; see {preflight_log}"
        )
    required_help = (
        "--position_lr_init",
        "--densify_until_iter",
        "--save_iterations",
        "--resolution RESOLUTION",
    )
    if any(token not in completed.stdout for token in required_help):
        raise TrainingProtocolError(
            "Pinned train.py help is missing expected original-3DGS arguments"
        )
    runtime_command = [
        str(python),
        "-c",
        (
            "import json, torch; "
            "print(json.dumps({'torch': torch.__version__, "
            "'torch_cuda': torch.version.cuda}))"
        ),
    ]
    runtime_completed = subprocess.run(
        runtime_command,
        cwd=checkout,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime_versions = json.loads(runtime_completed.stdout)
    return {
        "python": str(python),
        "dependency_root": str(dependency_root),
        "runtime_versions": runtime_versions,
        "install_sources": install_sources,
        "compiled_extensions": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in extensions.items()
        },
        "driver_library": str(driver_library.resolve()),
        "driver_library_sha256": _sha256(driver_library),
        "cpu_only_train_help_preflight": {
            "command": command,
            "cuda_visible_devices": "",
            "returncode": completed.returncode,
            "log": str(preflight_log),
            "log_sha256": _sha256(preflight_log),
        },
    }


def _training_command(
    python: Path,
    checkout: Path,
    staged_scene: Path,
    model_path: Path,
) -> list[str]:
    return [
        str(python),
        str(checkout / "train.py"),
        "--source_path",
        str(staged_scene),
        "--model_path",
        str(model_path),
        "--iterations",
        "30000",
        "--test_iterations",
        "-1",
        "--save_iterations",
        "30000",
        "--quiet",
    ]


def _parse_namespace(path: Path) -> dict[str, Any]:
    expression = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "Namespace"
    ):
        raise TrainingProtocolError(f"Unexpected cfg_args syntax: {path}")
    values = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise TrainingProtocolError(f"Unexpected cfg_args expansion: {path}")
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return values


def _parse_ply_vertex_count(path: Path) -> int:
    vertex_count = None
    found_end = False
    with path.open("rb") as handle:
        for _ in range(512):
            line = handle.readline()
            if not line:
                break
            if len(line) > 4096:
                raise TrainingProtocolError(f"Invalid PLY header line in {path}")
            decoded = line.decode("ascii", errors="strict").strip()
            match = re.fullmatch(r"element vertex ([0-9]+)", decoded)
            if match:
                vertex_count = int(match.group(1))
            if decoded == "end_header":
                found_end = True
                break
    if not found_end or vertex_count is None or vertex_count <= 0:
        raise TrainingProtocolError(f"Invalid or empty 3DGS PLY: {path}")
    return vertex_count


def _validate_training_output(
    run_dir: Path,
    model_path: Path,
    *,
    expected_registered_images: int,
    expected_source_resolution: tuple[int, int],
    expected_camera_metadata_resolution: tuple[int, int] | None = None,
) -> dict[str, Any]:
    point_cloud = (
        model_path
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply"
    )
    if not point_cloud.is_file():
        raise TrainingProtocolError(f"Missing final 30k point cloud: {point_cloud}")
    cfg_args = _parse_namespace(model_path / "cfg_args")
    if cfg_args.get("eval") is not False:
        raise TrainingProtocolError("All-view training unexpectedly enabled --eval")
    if cfg_args.get("resolution") != -1:
        raise TrainingProtocolError(
            f"Expected original automatic resolution -1, found {cfg_args.get('resolution')}"
        )
    cameras = json.loads((model_path / "cameras.json").read_text(encoding="utf-8"))
    if not isinstance(cameras, list):
        raise TrainingProtocolError("cameras.json must contain a camera list")
    if len(cameras) != expected_registered_images:
        raise TrainingProtocolError(
            f"Expected {expected_registered_images} all-view cameras, "
            f"found {len(cameras)}"
        )
    camera_metadata_resolution = (
        expected_source_resolution
        if expected_camera_metadata_resolution is None
        else expected_camera_metadata_resolution
    )
    if (
        not isinstance(camera_metadata_resolution, tuple)
        or len(camera_metadata_resolution) != 2
        or any(
            type(value) is not int or value <= 0
            for value in camera_metadata_resolution
        )
    ):
        raise TrainingProtocolError(
            "expected_camera_metadata_resolution must be two positive integers"
        )
    serialized_camera_resolutions = {
        (camera.get("width"), camera.get("height"))
        for camera in cameras
        if isinstance(camera, dict)
    }
    if serialized_camera_resolutions != {camera_metadata_resolution}:
        raise TrainingProtocolError(
            "Training cameras do not match the staged COLMAP metadata "
            f"resolution: expected {camera_metadata_resolution}, found "
            f"{sorted(serialized_camera_resolutions, key=repr)}"
        )
    effective_resolution = _automatic_training_resolution(
        *expected_source_resolution
    )
    return {
        "point_cloud": str(point_cloud),
        "point_cloud_sha256": _sha256(point_cloud),
        "point_cloud_size_bytes": point_cloud.stat().st_size,
        "point_cloud_vertices": _parse_ply_vertex_count(point_cloud),
        "cfg_args": cfg_args,
        "registered_all_view_cameras": len(cameras),
        "source_rgb_resolution": list(expected_source_resolution),
        "camera_metadata_resolution": list(camera_metadata_resolution),
        # Kept as a schema-v1 compatibility alias for source_rgb_resolution.
        "source_camera_resolution": list(expected_source_resolution),
        "effective_training_resolution": list(effective_resolution),
        "target_rgb_visible_during_training": True,
        "model_path": str(model_path),
        "run_dir": str(run_dir),
    }


def launch_all_view_training(
    args: argparse.Namespace,
    spec: AllViewTrainingSpec,
) -> Path:
    _validate_training_spec(spec)
    scene = spec.scene
    expected_registered_images = spec.expected_registered_images
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.attempt_id):
        raise TrainingProtocolError(
            "--attempt-id must contain only letters, digits, '.', '_' or '-'"
        )
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else spec.default_output_root.resolve()
    )
    run_dir = output_root / "attempts" / args.attempt_id
    if run_dir.exists():
        raise TrainingProtocolError(
            f"Refusing to reuse immutable attempt directory: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "training_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "preflighting",
        "method": "original-3DGS",
        "benchmark": spec.benchmark,
        "scene": scene,
        "geometry_scene": spec.geometry_scene,
        "geometry_protocol": "released_all_view",
        "expected_registered_images": expected_registered_images,
        "evaluation_render_resolution": list(
            spec.evaluation_render_resolution
        ),
        "source_asset_contract": spec.source_asset_contract,
        "attempt_id": args.attempt_id,
        "created_at": _utc_now(),
        "lock_file": str(LOCK_FILE),
        "lock_file_sha256": _sha256(LOCK_FILE),
        "gpu_lock": str(LOCK_PATH),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        source_provenance = _validate_source(
            args.upstream.resolve(),
            args.ludvig_upstream.resolve(),
            lock,
        )
        preflight_log = run_dir / "dependency_preflight.log"
        dependency_provenance = _validate_dependencies(
            args.upstream.resolve(),
            args.python.resolve(),
            args.dependency_root.resolve(),
            args.driver_library_dir.resolve(),
            preflight_log,
        )
        staged_scene = run_dir / "staging" / "colmap_pinhole_undistorted"
        camera_audit, geometry_input_provenance = _stage_training_input(
            spec,
            args.benchmark_root.resolve(),
            staged_scene,
        )
        if camera_audit.get("registered_images") != expected_registered_images:
            raise TrainingProtocolError(
                f"Expected {expected_registered_images} {scene} views, found "
                f"{camera_audit.get('registered_images')}"
            )
        if camera_audit.get("rgb_images") != expected_registered_images:
            raise TrainingProtocolError(
                f"Expected {expected_registered_images} staged {scene} RGBs, found "
                f"{camera_audit.get('rgb_images')}"
            )
        if camera_audit.get("raw_dataset_modified") is not False:
            raise TrainingProtocolError("Staging must not modify the source dataset")
        rgb_dimensions = camera_audit.get("rgb_dimensions")
        if (
            not isinstance(rgb_dimensions, list)
            or len(rgb_dimensions) != 2
            or any(type(value) is not int or value <= 0 for value in rgb_dimensions)
        ):
            raise TrainingProtocolError(
                f"Invalid staged RGB resolution for {scene}: {rgb_dimensions}"
            )
        source_resolution = tuple(rgb_dimensions)
        camera_metadata_dimensions = camera_audit.get(
            "camera_metadata_dimensions",
            rgb_dimensions,
        )
        if (
            not isinstance(camera_metadata_dimensions, list)
            or len(camera_metadata_dimensions) != 2
            or any(
                type(value) is not int or value <= 0
                for value in camera_metadata_dimensions
            )
        ):
            raise TrainingProtocolError(
                f"Invalid staged COLMAP metadata resolution for {scene}: "
                f"{camera_metadata_dimensions}"
            )
        camera_metadata_resolution = tuple(camera_metadata_dimensions)
        effective_training_resolution = _automatic_training_resolution(
            *source_resolution
        )
        model_path = run_dir / "model"
        command = _training_command(
            args.python.resolve(),
            args.upstream.resolve(),
            staged_scene,
            model_path,
        )
        manifest.update(
            {
                "source_provenance": source_provenance,
                "dependency_provenance": dependency_provenance,
                "geometry_input_provenance": geometry_input_provenance,
                "camera_audit": camera_audit,
                "training_command": command,
                "effective_training_protocol": {
                    "registered_training_views": expected_registered_images,
                    "held_out_training_views": 0,
                    "eval_split_enabled": False,
                    "iterations": 30000,
                    "resolution_argument": -1,
                    "source_rgb_resolution": list(source_resolution),
                    "camera_metadata_resolution": list(
                        camera_metadata_resolution
                    ),
                    "source_camera_resolution": list(source_resolution),
                    "effective_resolution": list(
                        effective_training_resolution
                    ),
                    "evaluation_render_resolution": list(
                        spec.evaluation_render_resolution
                    ),
                    "test_iterations_override": [-1],
                    "save_iterations": [30000],
                    "rng_seed": 0,
                    "hyperparameters": lock["training_defaults"],
                    "algorithm_source_modified": False,
                    "environment_compatibility_source_patch": None,
                },
            }
        )
        if args.dry_run:
            manifest["status"] = "dry_run"
            manifest["completed_at"] = _utc_now()
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            return manifest_path

        environment, _driver = _runtime_environment(
            args.dependency_root.resolve(),
            args.driver_library_dir.resolve(),
            expose_gpu=True,
        )
        manifest["status"] = "queued"
        manifest["queued_at"] = _utc_now()
        manifest["cuda_visible_devices"] = environment["CUDA_VISIBLE_DEVICES"]
        queue_started_epoch = time.time()
        wall_started = time.monotonic()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        log_path = run_dir / "stdout_stderr.log"
        gpu_started_marker = run_dir / "gpu_started_at.txt"
        locked_script = (
            "date -u +%Y-%m-%dT%H:%M:%S.%NZ"
            f" > {shlex.quote(str(gpu_started_marker))}; "
            f"exec {shlex.join(command)}"
        )
        locked_command = ["flock", str(LOCK_PATH), "-c", locked_script]
        try:
            with log_path.open("w") as log_handle:
                completed = subprocess.run(
                    locked_command,
                    cwd=args.upstream.resolve(),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            manifest["completed_at"] = _utc_now()
            manifest["wall_time_seconds"] = time.monotonic() - wall_started
            manifest["log"] = str(log_path)
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            raise

        completed_epoch = time.time()
        manifest["returncode"] = completed.returncode
        manifest["completed_at"] = _utc_now()
        manifest["wall_time_seconds"] = time.monotonic() - wall_started
        manifest["log"] = str(log_path)
        manifest["log_sha256"] = _sha256(log_path)
        if gpu_started_marker.exists():
            gpu_started_epoch = gpu_started_marker.stat().st_mtime
            manifest["gpu_started_at"] = gpu_started_marker.read_text(
                encoding="utf-8"
            ).strip()
            manifest["queue_wait_seconds"] = max(
                0.0, gpu_started_epoch - queue_started_epoch
            )
            manifest["gpu_wall_time_seconds"] = max(
                0.0, completed_epoch - gpu_started_epoch
            )
        if completed.returncode:
            manifest["status"] = "failed"
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            raise subprocess.CalledProcessError(completed.returncode, command)

        manifest["status"] = "validating"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["training_output"] = _validate_training_output(
            run_dir,
            model_path,
            expected_registered_images=expected_registered_images,
            expected_source_resolution=source_resolution,
            expected_camera_metadata_resolution=(
                camera_metadata_resolution
            ),
        )
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path
    except BaseException as error:
        if manifest.get("status") not in {"failed", "interrupted", "complete"}:
            manifest["status"] = (
                "failed_validation"
                if manifest.get("status") == "validating"
                else "failed_preflight"
            )
            manifest["completed_at"] = _utc_now()
            manifest["error_type"] = type(error).__name__
            manifest["error"] = str(error)
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
        raise


def launch(args: argparse.Namespace) -> Path:
    """Backward-compatible NVOS entrypoint."""

    return launch_all_view_training(args, _nvos_training_spec(args.scene))


def _add_common_training_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("/root/baselines/gaussian-splatting-ludvig-audit"),
    )
    parser.add_argument(
        "--ludvig-upstream",
        type=Path,
        default=Path("/root/baselines/LUDVIG"),
    )
    parser.add_argument(
        "--dependency-root",
        type=Path,
        default=Path(
            "/root/baselines/"
            "gaussian-splatting-ludvig-audit-deps/f7a116f-sm86"
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/miniconda3/envs/cybersim_agent/bin/python"),
    )
    parser.add_argument(
        "--driver-library-dir",
        type=Path,
        default=DEFAULT_DRIVER_LIBRARY_DIR,
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Training directory containing attempts/. By default this is "
            "the immutable training spec's dataset/scene output directory."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    parser = _add_common_training_arguments(argparse.ArgumentParser())
    parser.add_argument(
        "--scene",
        choices=NVOS_GEOMETRY_SCENES,
        default="fern",
        help=(
            "Unique NVOS geometry scene. horns_center and horns_left share "
            "the horns geometry checkpoint."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(launch(parse_args()))
    except TrainingProtocolError as error:
        raise SystemExit(f"protocol error: {error}") from error
