"""CPU-only, fail-closed staging for a LUDVIG-to-PFPR smoke run.

Phase A deliberately stops before importing PyTorch or any upstream LUDVIG
runtime.  It binds the public/method benchmark manifests, the query-held-out
ScanNet source contract, every selected source file through a repo-frozen
adapter ledger, the shared Gaussian geometry, and the official DINOv2
checkpoint to immutable hashes.  It then stages a PINHOLE COLMAP text scene
from a deterministic coverage-order prefix and proves the camera conversion
against both frozen world/pixel fixtures and a serialization round trip.

The DINO extraction, scene-only PCA, inverse-render uplift, crop scoring, and
PFPR evaluation phases do not exist in this module.  The companion wrapper
must reject attempts to request those phases until audited implementations are
added.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import AbstractSet, Any, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from radio_gs.benchmarks.scannet_pfpr.protocol import (
    PFPR_V2_BENCHMARK_VERSION,
    canonical_json_sha256,
    protocol_config_from_record,
    validate_field_query_exclusion_commitment,
)


LUDVIG_AUDITED_COMMIT = "4461fc515439bb498a75d71738a1e73cf7a452ed"
FIELD_CONTRACT_VERSION = "scannet_full_observation_pfpr_queryheldout_v1"
FIELD_SOURCE_POLICY = (
    "full_sens_greedy_depth_voxel_coverage_excluding_pfpr_query_source_frames"
)
OFFICIAL_DINO_CHECKPOINT_NAME = "dinov2_vitg14_reg4_pretrain.pth"
OFFICIAL_DINO_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/"
    "dinov2_vitg14_reg4_pretrain.pth"
)
OFFICIAL_DINO_CHECKPOINT_SIZE = 4_546_140_349
# Meta's CDN does not publish a SHA-256 header for this object.  This digest is
# corroborated by an exact-name, exact-size Git-LFS mirror and must be checked
# after downloading from the official URL.
OFFICIAL_DINO_CHECKPOINT_SHA256 = (
    "746ecb8c6301c645c5c855be91687d274587d6e48fdaec4a729753160b34a283"
)
OFFICIAL_DINO_CHECKPOINT_SHA256_PROVENANCE = (
    "exact_name_and_size_git_lfs_mirror_corroboration_not_meta_signed"
)
FROZEN_METHOD_MANIFEST_SHA256 = (
    "25ec3ca9d576d1862c229261ed9141a1657ab68be6890c9506d831053562417f"
)
FROZEN_PUBLIC_MANIFEST_SHA256 = (
    "e0ceae7a829363a036e0a34ba425be9cc5bc11bd0ebe1c49b93f28434caa58b9"
)
FROZEN_SOURCE_ADAPTER_LEDGER_SHA256_BY_SCENE = {
    "scene0050_02": "f56768c1808e84f9685b08ba7a7fffd7519b33c37c40285c49a475166bfcfcf1"
}
FROZEN_SOURCE_ADAPTER_LEDGER_FILENAME_BY_SCENE = {
    "scene0050_02": "ludvig_pfpr_scene0050_02_source_adapter_ledger_v1.json"
}
SMOKE_VIEW_COUNT = 120
EXPECTED_FIELD_FRAME_COUNT = 960
EXPECTED_GAUSSIAN_COUNT = 300_000
SOURCE_ADAPTER_LEDGER_VERSION = "ludvig_pfpr_source_adapter_ledger_v1"
SOURCE_ADAPTER_TRUST_MODEL = "trust_on_first_audit"
SOURCE_ADAPTER_AUDIT_SCOPE = "independent_cpu_source_adapter_tofu"
SOURCE_ADAPTER_TRUST_STATEMENT = (
    "hashes_frozen_after_manual_protocol_audit_without_evaluator_private_inputs"
)
POSE_CONVENTION = "camera_to_world"
CAMERA_AXES = "opencv_x_right_y_down_z_forward"
INTRINSICS_MODEL = "PINHOLE"
POSE_FIXTURE_CAMERA_XYZ = (0.125, -0.075, 2.0)
POSE_FIXTURE_COUNT = 3
SOURCE_INVENTORY_CANONICALIZATION = (
    "sha256_of_RFC8259_canonical_JSON_ordered_records_sort_keys_compact_utf8"
)
SOURCE_INVENTORY_RECORD_SCHEMA = {
    "ordering": "coverage_prefix_order",
    "modalities": ["color", "depth", "pose"],
    "record_fields": ["frame_id", "modalities"],
    "modality_binding_fields": ["relative_path", "bytes", "sha256"],
}
ROUNDTRIP_PIXEL_TOLERANCE = 1e-4
ROUNDTRIP_CAMERA_TOLERANCE = 1e-5
ROTATION_INPUT_TOLERANCE = 1e-4
POSE_FIXTURE_TOLERANCE = 1e-8

UPSTREAM_AUDIT_FILES = (
    "README.md",
    "configs/dif_NVOS.yaml",
    "dinov2/configs/vitg14_pretrain.yaml",
    "dinov2/setup.py",
    "dinov2/dino_utils.py",
    "predictors/dino.py",
    "ludvig_uplift.py",
    "utils/image.py",
    "utils/sliding_windows.py",
    "utils/solver.py",
)

METHOD_QUERY_KEYS = frozenset(
    {
        "available_method_inputs",
        "benchmark_version",
        "crop_rgb_path",
        "crop_rgb_sha256",
        "query_id",
        "scene_id",
    }
)
PROHIBITED_METHOD_KEYS = frozenset(
    {
        "anchor_world_xyz",
        "source_depth_pixel",
        "source_frame_id",
        "instance_id",
        "class_id",
        "mask",
        "pose",
        "depth",
    }
)


class LudvigPFPRPhaseAError(RuntimeError):
    """Raised before any GPU work when a phase-A contract is invalid."""


@dataclass(frozen=True)
class PhaseAConfig:
    """All externally bound inputs for one immutable CPU staging attempt."""

    scene_id: str
    benchmark_dir: Path
    source_scene: Path
    field_contract_sha256: str
    source_adapter_ledger: Path
    expected_source_adapter_ledger_sha256: str
    geometry_ply: Path
    geometry_sha256: str
    dino_checkpoint: Path
    ludvig_upstream: Path
    output_dir: Path
    view_count: int = SMOKE_VIEW_COUNT
    expected_field_frame_count: int = EXPECTED_FIELD_FRAME_COUNT
    expected_gaussian_count: int = EXPECTED_GAUSSIAN_COUNT
    expected_method_manifest_sha256: str = FROZEN_METHOD_MANIFEST_SHA256
    expected_public_manifest_sha256: str = FROZEN_PUBLIC_MANIFEST_SHA256
    expected_checkpoint_size: int = OFFICIAL_DINO_CHECKPOINT_SIZE
    expected_checkpoint_sha256: str = OFFICIAL_DINO_CHECKPOINT_SHA256
    expected_ludvig_commit: str = LUDVIG_AUDITED_COMMIT
    upstream_audit_files: tuple[str, ...] = UPSTREAM_AUDIT_FILES


def sha256_file(path: Path) -> str:
    """Hash one regular file without following any unrelated directory tree."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise LudvigPFPRPhaseAError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LudvigPFPRPhaseAError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LudvigPFPRPhaseAError(f"Invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise LudvigPFPRPhaseAError(f"{label} must be a JSON object: {path}")
    return value


def _require_exact_keys(
    value: Any, expected: AbstractSet[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        found = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise LudvigPFPRPhaseAError(
            f"{label} fields changed: expected {sorted(expected)}, found {found}"
        )
    return value


def _validate_file_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    expected_digest = _require_sha256(expected, f"expected {label} hash")
    if not path.is_file():
        raise LudvigPFPRPhaseAError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise LudvigPFPRPhaseAError(
            f"{label} SHA-256 mismatch: expected {expected_digest}, found {actual}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def _resolve_source_file(
    source_scene: Path, relative_path: Path, label: str
) -> tuple[Path, Path]:
    """Resolve one field input and reject symlink/path escape from the scene."""

    root = source_scene.resolve()
    lexical = root / relative_path
    if not lexical.is_file():
        raise LudvigPFPRPhaseAError(f"Missing {label}: {lexical}")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise LudvigPFPRPhaseAError(
            f"{label} resolves outside source_scene: {lexical} -> {resolved}"
        ) from error
    return lexical, resolved


def audit_checkpoint(
    path: Path,
    *,
    expected_size: int = OFFICIAL_DINO_CHECKPOINT_SIZE,
    expected_sha256: str = OFFICIAL_DINO_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    """Require the exact official LUDVIG DINOv2 filename, size, and digest."""

    checkpoint = path.resolve()
    if checkpoint.name != OFFICIAL_DINO_CHECKPOINT_NAME:
        raise LudvigPFPRPhaseAError(
            "DINO checkpoint basename must be exactly "
            f"{OFFICIAL_DINO_CHECKPOINT_NAME!r}; found {checkpoint.name!r}"
        )
    if not checkpoint.is_file():
        raise LudvigPFPRPhaseAError(f"Missing DINO checkpoint: {checkpoint}")
    size = checkpoint.stat().st_size
    if size != int(expected_size):
        raise LudvigPFPRPhaseAError(
            f"DINO checkpoint size mismatch: expected {expected_size}, found {size}"
        )
    expected_digest = _require_sha256(expected_sha256, "expected checkpoint hash")
    digest = sha256_file(checkpoint)
    if digest != expected_digest:
        raise LudvigPFPRPhaseAError(
            "DINO checkpoint SHA-256 mismatch: "
            f"expected {expected_digest}, found {digest}"
        )
    return {
        "path": str(checkpoint),
        "filename": checkpoint.name,
        "bytes": size,
        "sha256": digest,
        "official_url": OFFICIAL_DINO_CHECKPOINT_URL,
        "sha256_provenance": OFFICIAL_DINO_CHECKPOINT_SHA256_PROVENANCE,
        "meta_published_sha256": False,
    }


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LudvigPFPRPhaseAError(
            f"Unable to audit LUDVIG git checkout: {checkout}"
        ) from error
    return completed.stdout.strip()


def audit_ludvig_upstream(
    checkout: Path,
    *,
    expected_commit: str = LUDVIG_AUDITED_COMMIT,
    audited_files: Sequence[str] = UPSTREAM_AUDIT_FILES,
) -> dict[str, Any]:
    """Bind phase A to a LUDVIG commit while recording local audited patches."""

    root = checkout.resolve()
    if not root.is_dir():
        raise LudvigPFPRPhaseAError(f"Missing LUDVIG checkout: {root}")
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit != str(expected_commit):
        raise LudvigPFPRPhaseAError(
            f"LUDVIG commit mismatch: expected {expected_commit}, found {commit}"
        )
    file_hashes: dict[str, str] = {}
    for relative in audited_files:
        path = root / relative
        if not path.is_file():
            raise LudvigPFPRPhaseAError(
                f"Missing audited LUDVIG source file: {relative}"
            )
        file_hashes[str(relative)] = sha256_file(path)
    status = _git_output(root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "path": str(root),
        "commit": commit,
        "source_sha256": file_hashes,
        "working_tree_porcelain": status.splitlines() if status else [],
        "working_tree_clean": not bool(status),
        "phase_a_executes_upstream_code": False,
        "later_phase_source_lock_complete": False,
    }


def _assert_no_prohibited_method_keys(value: Any, path: str = "method_manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in PROHIBITED_METHOD_KEYS:
                raise LudvigPFPRPhaseAError(
                    f"Method manifest exposes prohibited key {key!r} at {path}"
                )
            _assert_no_prohibited_method_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_prohibited_method_keys(child, f"{path}[{index}]")


def _scene_domain(manifest: Mapping[str, Any], scene_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in manifest.get("scene_domains", [])
        if isinstance(item, dict) and str(item.get("scene_id")) == scene_id
    ]
    if len(matches) != 1:
        raise LudvigPFPRPhaseAError(
            f"PFPR manifest must contain exactly one domain for {scene_id}"
        )
    return matches[0]


def audit_benchmark(
    benchmark_dir: Path,
    scene_id: str,
    *,
    expected_method_sha256: str = FROZEN_METHOD_MANIFEST_SHA256,
    expected_public_sha256: str = FROZEN_PUBLIC_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Audit only method-visible/public PFPR artifacts, never evaluator GT."""

    root = benchmark_dir.resolve()
    method_path = root / "manifest.method.json"
    public_path = root / "manifest.public.json"
    method_binding = _validate_file_hash(
        method_path, expected_method_sha256, "PFPR method manifest"
    )
    public_binding = _validate_file_hash(
        public_path, expected_public_sha256, "PFPR public manifest"
    )
    method = _load_json(method_path, "PFPR method manifest")
    public = _load_json(public_path, "PFPR public manifest")
    _assert_no_prohibited_method_keys(method)
    versions = {str(method.get("benchmark_version")), str(public.get("benchmark_version"))}
    if versions != {PFPR_V2_BENCHMARK_VERSION}:
        raise LudvigPFPRPhaseAError(
            f"LUDVIG PFPR staging requires {PFPR_V2_BENCHMARK_VERSION}: {versions}"
        )
    try:
        method_config = protocol_config_from_record(
            PFPR_V2_BENCHMARK_VERSION, method.get("protocol_config")
        )
        public_config = protocol_config_from_record(
            PFPR_V2_BENCHMARK_VERSION, public.get("protocol_config")
        )
    except (TypeError, ValueError) as error:
        raise LudvigPFPRPhaseAError("Invalid frozen PFPR protocol config") from error
    if method_config != public_config:
        raise LudvigPFPRPhaseAError("PFPR method/public protocol configs disagree")

    queries = [
        item
        for item in method.get("queries", [])
        if isinstance(item, dict) and str(item.get("scene_id")) == scene_id
    ]
    if len(queries) != int(method_config.anchors_per_scene):
        raise LudvigPFPRPhaseAError(
            f"{scene_id} must expose {method_config.anchors_per_scene} method queries"
        )
    query_bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        if set(query) != METHOD_QUERY_KEYS:
            raise LudvigPFPRPhaseAError(
                f"Method query fields changed for {query.get('query_id')}: {sorted(query)}"
            )
        if query.get("available_method_inputs") != ["scene_id", "crop_rgb"]:
            raise LudvigPFPRPhaseAError("PFPR method input visibility changed")
        query_id = str(query.get("query_id", ""))
        if not query_id or query_id in seen:
            raise LudvigPFPRPhaseAError("PFPR method query IDs are empty or duplicated")
        seen.add(query_id)
        crop = Path(str(query.get("crop_rgb_path", ""))).resolve()
        binding = _validate_file_hash(
            crop, str(query.get("crop_rgb_sha256", "")), f"PFPR crop {query_id}"
        )
        with Image.open(crop) as image:
            if image.size != (method_config.patch_size_px, method_config.patch_size_px):
                raise LudvigPFPRPhaseAError(
                    f"PFPR crop dimensions changed for {query_id}: {image.size}"
                )
            binding["dimensions"] = list(image.size)
            binding["mode"] = image.mode
        binding["query_id"] = query_id
        query_bindings.append(binding)

    method_domain = _scene_domain(method, scene_id)
    public_domain = _scene_domain(public, scene_id)
    if method_domain != public_domain:
        raise LudvigPFPRPhaseAError("PFPR method/public scene domains disagree")
    if method_domain.get("geometry_only") is not True:
        raise LudvigPFPRPhaseAError("PFPR candidate domain is no longer geometry-only")
    candidate = Path(str(method_domain.get("candidate_xyz_path", ""))).resolve()
    candidate_binding = _validate_file_hash(
        candidate,
        str(method_domain.get("candidate_xyz_sha256", "")),
        f"PFPR candidate domain {scene_id}",
    )
    try:
        xyz = np.load(candidate, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise LudvigPFPRPhaseAError("Unable to load PFPR public candidate domain") from error
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or len(xyz) != int(method_domain.get("candidate_points", -1))
        or not np.isfinite(xyz).all()
    ):
        raise LudvigPFPRPhaseAError("PFPR public candidate domain shape/content changed")
    candidate_binding["points"] = int(len(xyz))
    return {
        "benchmark_dir": str(root),
        "benchmark_version": PFPR_V2_BENCHMARK_VERSION,
        "method_manifest": method_binding,
        "public_manifest": public_binding,
        "evaluator_manifest_opened": False,
        "method_visible_inputs": ["scene_id", "crop_rgb"],
        "query_raster_contract": method_config.query_raster_contract,
        "query_count": len(query_bindings),
        "queries": query_bindings,
        "candidate_domain": candidate_binding,
        "excluded_query_source_frame_ids_sha256": str(
            method_domain.get("excluded_query_source_frame_ids_sha256", "")
        ),
        "protocol_config": method.get("protocol_config"),
    }


def audit_field_contract(
    source_scene: Path,
    scene_id: str,
    public_exclusion_digest: str,
    *,
    expected_sha256: str,
    expected_frame_count: int = EXPECTED_FIELD_FRAME_COUNT,
    view_count: int = SMOKE_VIEW_COUNT,
) -> tuple[dict[str, Any], list[int]]:
    """Validate the query-held-out contract and select its coverage prefix."""

    root = source_scene.resolve()
    contract_path = root / "field_source_contract.json"
    binding = _validate_file_hash(
        contract_path, expected_sha256, "PFPR field source contract"
    )
    contract = _load_json(contract_path, "PFPR field source contract")
    if contract.get("field_contract_version") != FIELD_CONTRACT_VERSION:
        raise LudvigPFPRPhaseAError("PFPR field contract version changed")
    if str(contract.get("scene_id")) != scene_id:
        raise LudvigPFPRPhaseAError("PFPR field contract scene does not match")
    if int(contract.get("field_frame_count", -1)) != int(expected_frame_count):
        raise LudvigPFPRPhaseAError(
            "PFPR field frame count changed: "
            f"expected {expected_frame_count}, found {contract.get('field_frame_count')}"
        )
    if contract.get("frame_selection_policy") != "depth_voxel_coverage":
        raise LudvigPFPRPhaseAError("PFPR field is not coverage-order selected")
    policy = str(contract.get("source_policy", ""))
    if policy != FIELD_SOURCE_POLICY:
        raise LudvigPFPRPhaseAError(
            f"PFPR field source policy changed: expected {FIELD_SOURCE_POLICY!r}"
        )
    if contract.get("materialization_mode") != "decoded_sens":
        raise LudvigPFPRPhaseAError("PFPR field is not a decoded .sens materialization")
    source_sens_sha256 = _require_sha256(
        str(contract.get("source_sens_sha256", "")), "field source .sens hash"
    )
    for key in (
        "uses_private_anchor",
        "uses_private_depth_pixel",
        "uses_instances_or_semantic_labels",
        "contains_instance_or_label_directories",
    ):
        if contract.get(key) is not False:
            raise LudvigPFPRPhaseAError(f"PFPR field privacy flag is not false: {key}")
    try:
        validate_field_query_exclusion_commitment(
            PFPR_V2_BENCHMARK_VERSION,
            public_exclusion_digest,
            str(contract.get("excluded_query_source_frame_ids_sha256", "")),
        )
    except ValueError as error:
        raise LudvigPFPRPhaseAError(
            "PFPR field/query exclusion commitment mismatch"
        ) from error
    try:
        order = [int(value) for value in contract["selection_order_frame_indices"]]
        selected = [int(value) for value in contract["selected_frame_indices"]]
    except (KeyError, TypeError, ValueError) as error:
        raise LudvigPFPRPhaseAError("PFPR field frame inventories are invalid") from error
    if (
        len(order) != expected_frame_count
        or len(set(order)) != expected_frame_count
        or len(selected) != expected_frame_count
        or set(order) != set(selected)
    ):
        raise LudvigPFPRPhaseAError(
            "PFPR coverage order and materialized frame inventory disagree"
        )
    if int(contract.get("max_field_frames", -1)) != expected_frame_count:
        raise LudvigPFPRPhaseAError("PFPR max_field_frames changed")
    if str(contract.get("field_frame_manifest_sha256", "")) != canonical_json_sha256(
        selected
    ):
        raise LudvigPFPRPhaseAError("PFPR selected frame manifest digest changed")
    if int(view_count) != SMOKE_VIEW_COUNT:
        raise LudvigPFPRPhaseAError(
            f"Phase-A smoke is frozen to {SMOKE_VIEW_COUNT} views; found {view_count}"
        )
    if view_count > len(order):
        raise LudvigPFPRPhaseAError("PFPR coverage prefix is shorter than requested")
    prefix = order[:view_count]
    binding.update(
        {
            "field_contract_version": FIELD_CONTRACT_VERSION,
            "scene_id": scene_id,
            "field_frame_count": expected_frame_count,
            "frame_selection_policy": "depth_voxel_coverage",
            "source_policy": policy,
            "materialization_mode": "decoded_sens",
            "source_sens_sha256": source_sens_sha256,
            "excluded_query_source_frame_ids_sha256": str(
                contract.get("excluded_query_source_frame_ids_sha256")
            ),
            "privacy_flags": {
                key: contract[key]
                for key in (
                    "uses_private_anchor",
                    "uses_private_depth_pixel",
                    "uses_instances_or_semantic_labels",
                    "contains_instance_or_label_directories",
                )
            },
            "declared_color_size": contract.get("color_size"),
            "coverage_prefix_count": view_count,
            "coverage_prefix_frame_ids": prefix,
            "coverage_prefix_sha256": canonical_json_sha256(prefix),
        }
    )
    return binding, prefix


def _read_ply_header(path: Path) -> tuple[int, list[str]]:
    vertex_count: Optional[int] = None
    properties: list[str] = []
    consumed = 0
    with path.open("rb") as handle:
        for raw in handle:
            consumed += len(raw)
            if consumed > 1024 * 1024:
                raise LudvigPFPRPhaseAError("Gaussian PLY header exceeds 1 MiB")
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property "):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break
        else:
            raise LudvigPFPRPhaseAError("Gaussian PLY has no end_header")
    if vertex_count is None:
        raise LudvigPFPRPhaseAError("Gaussian PLY has no vertex count")
    return vertex_count, properties


def audit_geometry(
    path: Path,
    *,
    expected_sha256: str,
    expected_gaussians: int = EXPECTED_GAUSSIAN_COUNT,
) -> dict[str, Any]:
    binding = _validate_file_hash(path.resolve(), expected_sha256, "shared Gaussian PLY")
    try:
        count, properties = _read_ply_header(path.resolve())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        if isinstance(error, LudvigPFPRPhaseAError):
            raise
        raise LudvigPFPRPhaseAError("Invalid shared Gaussian PLY header") from error
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required - set(properties))
    if count != int(expected_gaussians) or missing:
        raise LudvigPFPRPhaseAError(
            "Shared Gaussian geometry changed: "
            f"vertices={count}, expected={expected_gaussians}, missing={missing}"
        )
    binding.update(
        {
            "gaussians": count,
            "property_count": len(properties),
            "required_graphdeco_properties_present": True,
            "training_protocol": "shared_RADIO_GS_15k_not_official_LUDVIG_30k",
        }
    )
    return binding


def _load_matrix(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise LudvigPFPRPhaseAError(f"Missing {label}: {path}")
    try:
        matrix = np.loadtxt(str(path), dtype=np.float64).reshape(4, 4)
    except (OSError, ValueError) as error:
        raise LudvigPFPRPhaseAError(f"Invalid {label}: {path}") from error
    if not np.isfinite(matrix).all():
        raise LudvigPFPRPhaseAError(f"Non-finite {label}: {path}")
    return matrix


def _is_pinhole_intrinsics(matrix: np.ndarray) -> bool:
    expected = np.array(
        [
            [matrix[0, 0], 0.0, matrix[0, 2], 0.0],
            [0.0, matrix[1, 1], matrix[1, 2], 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return bool(
        matrix[0, 0] > 0
        and matrix[1, 1] > 0
        and np.allclose(matrix, expected, atol=1e-10, rtol=0.0)
    )


def audit_depth_intrinsics(source_scene: Path, image_size: tuple[int, int]) -> dict[str, Any]:
    """Bind the resized depth-aligned RGB to the depth, not color, intrinsics."""

    _depth_path, depth_resolved = _resolve_source_file(
        source_scene, Path("intrinsics_depth.txt"), "ScanNet depth intrinsics"
    )
    _color_path, color_resolved = _resolve_source_file(
        source_scene, Path("intrinsics_color.txt"), "ScanNet color intrinsics"
    )
    depth = _load_matrix(depth_resolved, "ScanNet depth intrinsics")
    color = _load_matrix(color_resolved, "ScanNet color intrinsics")
    width, height = image_size
    if not _is_pinhole_intrinsics(depth):
        raise LudvigPFPRPhaseAError("ScanNet depth intrinsics are not PINHOLE-compatible")
    if not _is_pinhole_intrinsics(color):
        raise LudvigPFPRPhaseAError("ScanNet color intrinsics are not PINHOLE-compatible")
    expected_center = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    principal = np.array([depth[0, 2], depth[1, 2]])
    if not np.allclose(principal, expected_center, atol=1e-6):
        raise LudvigPFPRPhaseAError(
            "LUDVIG CamScene requires centered intrinsics for this adapter; "
            f"expected {expected_center.tolist()}, found {principal.tolist()}"
        )
    if np.allclose(depth, color, atol=1e-8):
        raise LudvigPFPRPhaseAError(
            "Depth and original-color intrinsics unexpectedly coincide; "
            "the depth-aligned raster contract may have changed"
        )
    depth_binding = {
        "relative_path": "intrinsics_depth.txt",
        "bytes": depth_resolved.stat().st_size,
        "sha256": sha256_file(depth_resolved),
    }
    color_binding = {
        "relative_path": "intrinsics_color.txt",
        "bytes": color_resolved.stat().st_size,
        "sha256": sha256_file(color_resolved),
    }
    return {
        "selected_role": f"depth_intrinsics_for_{width}x{height}_depth_aligned_RGB",
        "selected_path": str(depth_resolved),
        "selected_sha256": depth_binding["sha256"],
        "rejected_original_color_path": str(color_resolved),
        "rejected_original_color_sha256": color_binding["sha256"],
        "source_bindings": {"depth": depth_binding, "color": color_binding},
        "image_dimensions": [width, height],
        "fx": float(depth[0, 0]),
        "fy": float(depth[1, 1]),
        "cx": float(depth[0, 2]),
        "cy": float(depth[1, 2]),
        "matrix": depth.tolist(),
    }


def _orthonormalized_c2w(raw: np.ndarray, frame_id: int) -> tuple[np.ndarray, float]:
    if not np.allclose(raw[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise LudvigPFPRPhaseAError(f"Frame {frame_id} pose has invalid homogeneous row")
    rotation = raw[:3, :3]
    orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    if max(orthogonality, determinant_error) > ROTATION_INPUT_TOLERANCE:
        raise LudvigPFPRPhaseAError(
            f"Frame {frame_id} pose is not a rigid c2w transform"
        )
    u, _singular, vt = np.linalg.svd(rotation)
    normalized = u @ vt
    if np.linalg.det(normalized) < 0:
        u[:, -1] *= -1
        normalized = u @ vt
    correction = float(np.max(np.abs(normalized - rotation)))
    if correction > ROTATION_INPUT_TOLERANCE:
        raise LudvigPFPRPhaseAError(
            f"Frame {frame_id} pose needs excessive rotation correction: {correction}"
        )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = normalized
    result[:3, 3] = raw[:3, 3]
    return result, correction


def rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """Return COLMAP's canonical ``qw,qx,qy,qz`` quaternion."""

    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * scale,
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = int(np.argmax(np.diag(r)))
        if diagonal == 0:
            scale = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array(
                [
                    (r[2, 1] - r[1, 2]) / scale,
                    0.25 * scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                ]
            )
        elif diagonal == 1:
            scale = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array(
                [
                    (r[0, 2] - r[2, 0]) / scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    0.25 * scale,
                    (r[1, 2] + r[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array(
                [
                    (r[1, 0] - r[0, 1]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                    (r[1, 2] + r[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1
    return q


def qvec_to_rotation_matrix(qvec: Sequence[float]) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _camera_roundtrip(
    c2w: np.ndarray,
    qvec: np.ndarray,
    tvec: np.ndarray,
    intrinsics: Mapping[str, Any],
) -> tuple[float, float, float]:
    width, height = [int(value) for value in intrinsics["image_dimensions"]]
    fx, fy, cx, cy = [float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")]
    reconstructed_w2c = np.eye(4, dtype=np.float64)
    reconstructed_w2c[:3, :3] = qvec_to_rotation_matrix(qvec)
    reconstructed_w2c[:3, 3] = tvec
    expected_w2c = np.linalg.inv(c2w)
    matrix_error = float(np.max(np.abs(reconstructed_w2c - expected_w2c)))
    pixels = (
        (cx, cy, 0.5),
        (0.25 * (width - 1), 0.25 * (height - 1), 1.5),
        (0.75 * (width - 1), 0.75 * (height - 1), 4.0),
    )
    max_pixel = 0.0
    max_camera = 0.0
    for u, v, depth in pixels:
        camera = np.array(
            [(u - cx) * depth / fx, (v - cy) * depth / fy, depth, 1.0],
            dtype=np.float64,
        )
        world = c2w @ camera
        recovered = reconstructed_w2c @ world
        if recovered[2] <= 0:
            raise LudvigPFPRPhaseAError("Camera round trip produced non-positive depth")
        recovered_u = fx * recovered[0] / recovered[2] + cx
        recovered_v = fy * recovered[1] / recovered[2] + cy
        max_pixel = max(max_pixel, abs(recovered_u - u), abs(recovered_v - v))
        max_camera = max(max_camera, float(np.max(np.abs(recovered[:3] - camera[:3]))))
    return matrix_error, max_pixel, max_camera


def audit_selected_source(
    source_scene: Path,
    frame_ids: Sequence[int],
    declared_color_size: Sequence[int],
) -> tuple[dict[str, Any], dict[int, np.ndarray], dict[int, Path]]:
    if len(declared_color_size) != 2:
        raise LudvigPFPRPhaseAError("PFPR field contract has no color dimensions")
    image_size = (int(declared_color_size[0]), int(declared_color_size[1]))
    entries: list[dict[str, Any]] = []
    canonical_records: list[dict[str, Any]] = []
    modality_records: dict[str, list[dict[str, Any]]] = {
        "color": [],
        "depth": [],
        "pose": [],
    }
    poses: dict[int, np.ndarray] = {}
    colors: dict[int, Path] = {}
    max_rotation_correction = 0.0
    for frame_id in frame_ids:
        stem = f"{int(frame_id):06d}"
        relative_paths = {
            "color": Path("color") / f"{stem}.jpg",
            "depth": Path("depth") / f"{stem}.png",
            "pose": Path("pose") / f"{stem}.txt",
        }
        paths = {
            role: _resolve_source_file(
                source_scene,
                relative,
                f"selected PFPR {role} for frame {frame_id}",
            )[1]
            for role, relative in relative_paths.items()
        }
        with Image.open(paths["color"]) as image:
            if image.size != image_size or image.mode != "RGB":
                raise LudvigPFPRPhaseAError(
                    f"Selected RGB frame {frame_id} changed dimensions or mode"
                )
        with Image.open(paths["depth"]) as depth:
            if depth.size != image_size:
                raise LudvigPFPRPhaseAError(
                    f"Selected depth frame {frame_id} is not RGB-aligned"
                )
        raw_pose = _load_matrix(paths["pose"], f"ScanNet c2w pose {frame_id}")
        pose, correction = _orthonormalized_c2w(raw_pose, int(frame_id))
        max_rotation_correction = max(max_rotation_correction, correction)
        poses[int(frame_id)] = pose
        colors[int(frame_id)] = paths["color"]
        canonical_modalities: dict[str, dict[str, Any]] = {}
        output_modalities: dict[str, dict[str, Any]] = {}
        for role in ("color", "depth", "pose"):
            canonical_binding = {
                "relative_path": relative_paths[role].as_posix(),
                "bytes": paths[role].stat().st_size,
                "sha256": sha256_file(paths[role]),
            }
            canonical_modalities[role] = canonical_binding
            output_modalities[role] = {
                "path": str(paths[role]),
                **canonical_binding,
            }
            modality_records[role].append(
                {"frame_id": int(frame_id), **canonical_binding}
            )
        canonical_records.append(
            {"frame_id": int(frame_id), "modalities": canonical_modalities}
        )
        entries.append(
            {
                "frame_id": int(frame_id),
                **output_modalities,
                "rotation_orthonormalization_max_abs_delta": correction,
            }
        )
    return (
        {
            "frame_count": len(entries),
            "ordered_inventory": entries,
            "ordered_inventory_sha256": canonical_json_sha256(entries),
            "canonicalization": SOURCE_INVENTORY_CANONICALIZATION,
            "record_schema": SOURCE_INVENTORY_RECORD_SCHEMA,
            "canonical_records_sha256": canonical_json_sha256(canonical_records),
            "modality_records_sha256": {
                role: canonical_json_sha256(records)
                for role, records in modality_records.items()
            },
            "all_rgb_depth_pose_present": True,
            "all_rgb_depth_dimensions": list(image_size),
            "max_rotation_orthonormalization_abs_delta": max_rotation_correction,
        },
        poses,
        colors,
    )


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise LudvigPFPRPhaseAError(f"{label} is not numeric") from error
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise LudvigPFPRPhaseAError(f"{label} must be a finite length-{length} vector")
    return vector


def _audit_pose_fixture(
    fixture: Any,
    *,
    poses_c2w: Mapping[int, np.ndarray],
    intrinsics: Mapping[str, Any],
) -> dict[str, Any]:
    record = _require_exact_keys(
        fixture,
        {
            "camera_center_world",
            "camera_xyz",
            "frame_id",
            "pixel_uv",
            "world_xyz",
        },
        "source adapter pose fixture",
    )
    try:
        frame_id = int(record["frame_id"])
    except (TypeError, ValueError) as error:
        raise LudvigPFPRPhaseAError("Pose fixture frame_id is invalid") from error
    if frame_id not in poses_c2w:
        raise LudvigPFPRPhaseAError(
            f"Pose fixture frame {frame_id} is outside the coverage prefix"
        )
    camera_xyz = _finite_vector(record["camera_xyz"], 3, "pose fixture camera_xyz")
    if not np.allclose(
        camera_xyz,
        np.asarray(POSE_FIXTURE_CAMERA_XYZ),
        atol=0.0,
        rtol=0.0,
    ):
        raise LudvigPFPRPhaseAError("Pose fixture camera point changed")
    expected_center = _finite_vector(
        record["camera_center_world"], 3, "pose fixture camera center"
    )
    expected_world = _finite_vector(record["world_xyz"], 3, "pose fixture world_xyz")
    expected_pixel = _finite_vector(record["pixel_uv"], 2, "pose fixture pixel_uv")

    c2w = poses_c2w[frame_id]
    w2c = np.linalg.inv(c2w)
    actual_center = c2w[:3, 3]
    actual_world = (c2w @ np.append(camera_xyz, 1.0))[:3]
    recovered_camera = (w2c @ np.append(expected_world, 1.0))[:3]
    recovered_center = (w2c @ np.append(expected_center, 1.0))[:3]
    fx, fy, cx, cy = [float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")]
    if recovered_camera[2] <= 0:
        raise LudvigPFPRPhaseAError(
            f"Pose fixture frame {frame_id} projects behind the OpenCV camera"
        )
    actual_pixel = np.array(
        [
            fx * recovered_camera[0] / recovered_camera[2] + cx,
            fy * recovered_camera[1] / recovered_camera[2] + cy,
        ],
        dtype=np.float64,
    )
    if not (expected_pixel[0] > cx and expected_pixel[1] < cy):
        raise LudvigPFPRPhaseAError(
            "Pose fixture does not encode OpenCV x-right/y-down pixel axes"
        )
    errors = {
        "camera_center_world_max_abs_error": float(
            np.max(np.abs(actual_center - expected_center))
        ),
        "camera_to_world_point_max_abs_error": float(
            np.max(np.abs(actual_world - expected_world))
        ),
        "world_to_camera_point_max_abs_error": float(
            np.max(np.abs(recovered_camera - camera_xyz))
        ),
        "camera_center_to_origin_max_abs_error": float(
            np.max(np.abs(recovered_center))
        ),
        "world_to_pixel_max_abs_error": float(
            np.max(np.abs(actual_pixel - expected_pixel))
        ),
    }
    if max(errors.values()) > POSE_FIXTURE_TOLERANCE:
        raise LudvigPFPRPhaseAError(
            f"Trusted c2w/OpenCV pose fixture failed for frame {frame_id}: {errors}"
        )
    return {"frame_id": frame_id, **errors}


def audit_source_adapter_ledger(
    path: Path,
    *,
    expected_sha256: str,
    scene_id: str,
    field_contract: Mapping[str, Any],
    frame_ids: Sequence[int],
    source_inventory: Mapping[str, Any],
    intrinsics: Mapping[str, Any],
    poses_c2w: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    """Bind materialized field bytes and independent camera-semantics fixtures."""

    binding = _validate_file_hash(
        path.resolve(), expected_sha256, "trusted source adapter ledger"
    )
    ledger = _load_json(path.resolve(), "trusted source adapter ledger")
    _require_exact_keys(
        ledger,
        {
            "camera_contract",
            "coverage_prefix",
            "provenance",
            "scene_id",
            "schema_version",
            "selected_source_inventory",
            "trust_model",
        },
        "source adapter ledger",
    )
    if ledger.get("schema_version") != SOURCE_ADAPTER_LEDGER_VERSION:
        raise LudvigPFPRPhaseAError("Source adapter ledger schema changed")
    if str(ledger.get("scene_id")) != scene_id:
        raise LudvigPFPRPhaseAError("Source adapter ledger scene changed")
    if ledger.get("trust_model") != SOURCE_ADAPTER_TRUST_MODEL:
        raise LudvigPFPRPhaseAError("Source adapter ledger trust model changed")

    provenance = _require_exact_keys(
        ledger.get("provenance"),
        {
            "audit_date_utc",
            "audit_scope",
            "evaluator_private_manifest_opened",
            "field_contract_sha256",
            "materialization_mode",
            "query_private_anchor_pose_depth_used",
            "source_sens_sha256",
            "statement",
        },
        "source adapter ledger provenance",
    )
    if not str(provenance.get("audit_date_utc", "")):
        raise LudvigPFPRPhaseAError("Source adapter ledger audit date is empty")
    if provenance.get("audit_scope") != SOURCE_ADAPTER_AUDIT_SCOPE:
        raise LudvigPFPRPhaseAError("Source adapter ledger audit scope changed")
    if provenance.get("statement") != SOURCE_ADAPTER_TRUST_STATEMENT:
        raise LudvigPFPRPhaseAError("Source adapter ledger trust statement changed")
    if provenance.get("evaluator_private_manifest_opened") is not False:
        raise LudvigPFPRPhaseAError("Source adapter audit opened evaluator-private data")
    if provenance.get("query_private_anchor_pose_depth_used") is not False:
        raise LudvigPFPRPhaseAError("Source adapter audit used query-private data")
    if provenance.get("field_contract_sha256") != field_contract.get("sha256"):
        raise LudvigPFPRPhaseAError("Source adapter ledger field contract hash changed")
    if provenance.get("source_sens_sha256") != field_contract.get("source_sens_sha256"):
        raise LudvigPFPRPhaseAError("Source adapter ledger source .sens hash changed")
    if provenance.get("materialization_mode") != field_contract.get(
        "materialization_mode"
    ):
        raise LudvigPFPRPhaseAError("Source adapter materialization mode changed")

    coverage = _require_exact_keys(
        ledger.get("coverage_prefix"),
        {"count", "ordered_frame_ids_sha256"},
        "source adapter coverage prefix",
    )
    if int(coverage.get("count", -1)) != len(frame_ids) or coverage.get(
        "ordered_frame_ids_sha256"
    ) != canonical_json_sha256(list(frame_ids)):
        raise LudvigPFPRPhaseAError("Source adapter coverage prefix changed")

    inventory = _require_exact_keys(
        ledger.get("selected_source_inventory"),
        {
            "canonical_records_sha256",
            "canonicalization",
            "modality_records_sha256",
            "record_count",
            "record_schema",
        },
        "source adapter selected inventory",
    )
    if inventory.get("canonicalization") != SOURCE_INVENTORY_CANONICALIZATION:
        raise LudvigPFPRPhaseAError("Source adapter inventory canonicalization changed")
    if inventory.get("record_schema") != SOURCE_INVENTORY_RECORD_SCHEMA:
        raise LudvigPFPRPhaseAError("Source adapter inventory record schema changed")
    if int(inventory.get("record_count", -1)) != len(frame_ids):
        raise LudvigPFPRPhaseAError("Source adapter inventory count changed")
    if inventory.get("canonical_records_sha256") != source_inventory.get(
        "canonical_records_sha256"
    ):
        raise LudvigPFPRPhaseAError(
            "Selected RGB/depth/pose inventory does not match trusted ledger"
        )
    expected_modality_hashes = inventory.get("modality_records_sha256")
    if (
        not isinstance(expected_modality_hashes, Mapping)
        or set(expected_modality_hashes) != {"color", "depth", "pose"}
        or dict(expected_modality_hashes)
        != dict(source_inventory.get("modality_records_sha256", {}))
    ):
        raise LudvigPFPRPhaseAError(
            "Selected source modality hashes do not match trusted ledger"
        )

    camera = _require_exact_keys(
        ledger.get("camera_contract"),
        {
            "camera_axes",
            "image_dimensions",
            "intrinsics_bindings",
            "intrinsics_model",
            "pose_convention",
            "pose_fixtures",
        },
        "source adapter camera contract",
    )
    if camera.get("pose_convention") != POSE_CONVENTION:
        raise LudvigPFPRPhaseAError("Source adapter pose convention is not c2w")
    if camera.get("camera_axes") != CAMERA_AXES:
        raise LudvigPFPRPhaseAError("Source adapter camera axes are not ScanNet/OpenCV")
    if camera.get("intrinsics_model") != INTRINSICS_MODEL:
        raise LudvigPFPRPhaseAError("Source adapter intrinsics model is not PINHOLE")
    if camera.get("image_dimensions") != intrinsics.get("image_dimensions"):
        raise LudvigPFPRPhaseAError("Source adapter camera image dimensions changed")
    expected_intrinsics = camera.get("intrinsics_bindings")
    if (
        not isinstance(expected_intrinsics, Mapping)
        or set(expected_intrinsics) != {"color", "depth"}
        or dict(expected_intrinsics) != dict(intrinsics.get("source_bindings", {}))
    ):
        raise LudvigPFPRPhaseAError(
            "Source intrinsics files do not match trusted adapter ledger"
        )

    fixtures = camera.get("pose_fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != POSE_FIXTURE_COUNT:
        raise LudvigPFPRPhaseAError(
            f"Source adapter requires exactly {POSE_FIXTURE_COUNT} pose fixtures"
        )
    fixture_audits = [
        _audit_pose_fixture(item, poses_c2w=poses_c2w, intrinsics=intrinsics)
        for item in fixtures
    ]
    fixture_ids = [item["frame_id"] for item in fixture_audits]
    if len(set(fixture_ids)) != POSE_FIXTURE_COUNT:
        raise LudvigPFPRPhaseAError("Source adapter pose fixture IDs are duplicated")
    binding.update(
        {
            "schema_version": SOURCE_ADAPTER_LEDGER_VERSION,
            "scene_id": scene_id,
            "trust_model": SOURCE_ADAPTER_TRUST_MODEL,
            "provenance": dict(provenance),
            "coverage_prefix_sha256": coverage["ordered_frame_ids_sha256"],
            "selected_source_inventory_sha256": inventory[
                "canonical_records_sha256"
            ],
            "pose_convention": POSE_CONVENTION,
            "camera_axes": CAMERA_AXES,
            "pose_fixture_audits": fixture_audits,
            "pose_fixtures_passed": True,
        }
    )
    return binding


def _write_placeholder_points_ply(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "comment camera-loader placeholder; not Gaussian geometry",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property float nx",
                "property float ny",
                "property float nz",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 0 0 0 0 0 0 0",
                "",
            )
        ),
        encoding="ascii",
    )


def stage_colmap_text(
    staging_root: Path,
    frame_ids: Sequence[int],
    poses_c2w: Mapping[int, np.ndarray],
    color_paths: Mapping[int, Path],
    intrinsics: Mapping[str, Any],
) -> dict[str, Any]:
    """Stage symlinked RGBs and a one-camera PINHOLE COLMAP text model."""

    images_dir = staging_root / "images"
    sparse_dir = staging_root / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=False)
    sparse_dir.mkdir(parents=True, exist_ok=False)
    width, height = [int(value) for value in intrinsics["image_dimensions"]]
    fx, fy, cx, cy = [float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")]
    cameras_path = sparse_dir / "cameras.txt"
    cameras_path.write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n",
        encoding="ascii",
    )
    image_lines = [
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] is intentionally empty for feature uplift staging",
    ]
    staged_images: list[dict[str, Any]] = []
    maximum_matrix_error = 0.0
    maximum_pixel_error = 0.0
    maximum_camera_error = 0.0
    for rank, frame_id in enumerate(frame_ids):
        c2w = poses_c2w[int(frame_id)]
        w2c = np.linalg.inv(c2w)
        qvec = rotation_matrix_to_qvec(w2c[:3, :3])
        tvec = w2c[:3, 3]
        matrix_error, pixel_error, camera_error = _camera_roundtrip(
            c2w, qvec, tvec, intrinsics
        )
        maximum_matrix_error = max(maximum_matrix_error, matrix_error)
        maximum_pixel_error = max(maximum_pixel_error, pixel_error)
        maximum_camera_error = max(maximum_camera_error, camera_error)
        staged_name = f"{rank:06d}_{int(frame_id):06d}.jpg"
        destination = images_dir / staged_name
        os.symlink(str(color_paths[int(frame_id)].resolve()), str(destination))
        values = [*qvec.tolist(), *tvec.tolist()]
        image_lines.append(
            f"{rank + 1} "
            + " ".join(f"{value:.17g}" for value in values)
            + f" 1 {staged_name}"
        )
        image_lines.append("")
        staged_images.append(
            {
                "rank": rank,
                "frame_id": int(frame_id),
                "staged_name": staged_name,
                "source": str(color_paths[int(frame_id)].resolve()),
                "qvec_wxyz": qvec.tolist(),
                "tvec": tvec.tolist(),
            }
        )
    if maximum_matrix_error > ROUNDTRIP_CAMERA_TOLERANCE:
        raise LudvigPFPRPhaseAError(
            f"COLMAP matrix round-trip error is too large: {maximum_matrix_error}"
        )
    if maximum_pixel_error > ROUNDTRIP_PIXEL_TOLERANCE:
        raise LudvigPFPRPhaseAError(
            f"COLMAP pixel round-trip error is too large: {maximum_pixel_error}"
        )
    if maximum_camera_error > ROUNDTRIP_CAMERA_TOLERANCE:
        raise LudvigPFPRPhaseAError(
            f"COLMAP camera round-trip error is too large: {maximum_camera_error}"
        )
    images_path = sparse_dir / "images.txt"
    images_path.write_text("\n".join(image_lines) + "\n", encoding="ascii")
    points_path = sparse_dir / "points3D.ply"
    _write_placeholder_points_ply(points_path)
    return {
        "layout": "PINHOLE_COLMAP_text_with_symlinked_RGB",
        "root_relative_to_attempt": "staging/colmap",
        "camera_model": "PINHOLE",
        "registered_images": len(staged_images),
        "images_are_symlinks": True,
        "staged_images": staged_images,
        "staged_images_sha256": canonical_json_sha256(staged_images),
        "cameras_txt_sha256": sha256_file(cameras_path),
        "images_txt_sha256": sha256_file(images_path),
        "points3D_ply_sha256": sha256_file(points_path),
        "points3D_role": "camera_loader_placeholder_not_Gaussian_geometry",
        "pose_convention": "ScanNet_c2w_inverted_to_COLMAP_world_to_camera_qvec_tvec",
        "roundtrip": {
            "maximum_w2c_matrix_abs_error": maximum_matrix_error,
            "maximum_pixel_error_px": maximum_pixel_error,
            "maximum_camera_coordinate_abs_error": maximum_camera_error,
            "pixel_tolerance_px": ROUNDTRIP_PIXEL_TOLERANCE,
            "camera_tolerance": ROUNDTRIP_CAMERA_TOLERANCE,
            "passed": True,
        },
    }


def _repository_commit(root: Path) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_phase_a(config: PhaseAConfig, *, argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Validate all phase-A inputs and atomically materialize one attempt."""

    output = config.output_dir.resolve()
    if output.exists():
        raise LudvigPFPRPhaseAError(
            f"Refusing to overwrite an existing PFPR/LUDVIG attempt: {output}"
        )
    if (
        not config.scene_id
        or "/" in config.scene_id
        or config.scene_id != Path(config.scene_id).name
    ):
        raise LudvigPFPRPhaseAError(f"Invalid ScanNet scene ID: {config.scene_id!r}")
    if int(config.view_count) != SMOKE_VIEW_COUNT:
        raise LudvigPFPRPhaseAError(
            f"Phase-A smoke is frozen to {SMOKE_VIEW_COUNT} views"
        )

    # Complete every expensive/fail-closed validation before creating the
    # attempt directory.  None of these helpers imports torch or touches CUDA.
    checkpoint = audit_checkpoint(
        config.dino_checkpoint,
        expected_size=config.expected_checkpoint_size,
        expected_sha256=config.expected_checkpoint_sha256,
    )
    upstream = audit_ludvig_upstream(
        config.ludvig_upstream,
        expected_commit=config.expected_ludvig_commit,
        audited_files=config.upstream_audit_files,
    )
    benchmark = audit_benchmark(
        config.benchmark_dir,
        config.scene_id,
        expected_method_sha256=config.expected_method_manifest_sha256,
        expected_public_sha256=config.expected_public_manifest_sha256,
    )
    contract, frame_ids = audit_field_contract(
        config.source_scene,
        config.scene_id,
        benchmark["excluded_query_source_frame_ids_sha256"],
        expected_sha256=config.field_contract_sha256,
        expected_frame_count=config.expected_field_frame_count,
        view_count=config.view_count,
    )
    geometry = audit_geometry(
        config.geometry_ply,
        expected_sha256=config.geometry_sha256,
        expected_gaussians=config.expected_gaussian_count,
    )
    declared_size = contract.get("declared_color_size")
    if not isinstance(declared_size, list) or len(declared_size) != 2:
        raise LudvigPFPRPhaseAError("PFPR field contract color_size is invalid")
    intrinsics = audit_depth_intrinsics(
        config.source_scene.resolve(),
        (int(declared_size[0]), int(declared_size[1])),
    )
    source_inventory, poses, colors = audit_selected_source(
        config.source_scene.resolve(), frame_ids, declared_size
    )
    source_adapter = audit_source_adapter_ledger(
        config.source_adapter_ledger,
        expected_sha256=config.expected_source_adapter_ledger_sha256,
        scene_id=config.scene_id,
        field_contract=contract,
        frame_ids=frame_ids,
        source_inventory=source_inventory,
        intrinsics=intrinsics,
        poses_c2w=poses,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.phase_a_tmp_", dir=str(output.parent))
    )
    try:
        colmap = stage_colmap_text(
            temporary / "staging" / "colmap",
            frame_ids,
            poses,
            colors,
            intrinsics,
        )
        manifest: dict[str, Any] = {
            "schema_version": "ludvig_pfpr_phase_a_v1",
            "status": "phase_a_complete_phase_b_available_not_run",
            "result_eligible": False,
            "official_ludvig_reproduction": False,
            "paper_metric_comparable": False,
            "scene_id": config.scene_id,
            "attempt_dir": str(output),
            "argv": list(argv or []),
            "repository_commit_at_stage": _repository_commit(Path(__file__).resolve().parents[3]),
            "gpu_work_started": False,
            "torch_imported_by_phase_a": False,
            "evaluator_private_manifest_opened": False,
            "phase_status": {
                "phase_a_cpu_staging": "complete",
                "phase_b_dino_scene_features_and_pca": "available_separate_not_run",
                "phase_c_inverse_render_uplift": "not_implemented_fail_closed",
                "phase_d_pfpr_crop_scoring": "not_implemented_fail_closed",
                "phase_e_pfpr_evaluation": "not_run_until_b_to_d_are_audited",
            },
            "checkpoint": checkpoint,
            "ludvig_upstream": upstream,
            "benchmark": benchmark,
            "field_contract": contract,
            "source_adapter_ledger": source_adapter,
            "geometry": geometry,
            "view_selection": {
                "policy": "coverage_prefix",
                "count": len(frame_ids),
                "ordered_frame_ids": frame_ids,
                "ordered_frame_ids_sha256": canonical_json_sha256(frame_ids),
                "query_source_frames_excluded_by_public_digest_commitment": True,
            },
            "source_inventory": source_inventory,
            "camera_intrinsics": intrinsics,
            "colmap_staging": colmap,
            "phase_b_method_contract": {
                "encoder": "released_LUDVIG_vendored_DINOv2_ViT_g14_without_register_tokens",
                "checkpoint_file": "official_DINOv2_ViT_g14_reg4",
                "checkpoint_load_contract": (
                    "released_strict_false_with_only_register_tokens_unexpected; "
                    "audited_reproduction_filters_only_register_tokens_then_strict_true"
                ),
                "scene_pca_components": 40,
                "scene_pca_fit_scope": "scene_views_only",
                "query_pca_rule": "reuse_frozen_scene_mean_std_components_only",
                "uplift": "official_LUDVIG_inverse_render_weights",
                "query_pooling": "center_3x3_L2_mean",
                "candidate_readout": "continuous_opacity_weighted_Gaussian_5cm_cell",
                "implemented_in_repository": True,
                "executed_in_phase_a": False,
            },
            "fail_closed_reason": (
                "This artifact proves only CPU input and camera staging. Phase B "
                "must run through the separately audited dino-pca entry point; "
                "uplift/scorer/evaluation remain unavailable."
            ),
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def reject_unimplemented_phase(phase: str) -> None:
    """Reject every phase that could otherwise be mistaken for a GPU result."""

    if str(phase) not in {"phase-a", "dino-pca"}:
        raise LudvigPFPRPhaseAError(
            f"Requested {phase!r}, but later phases are not implemented; "
            "only phase-a and dino-pca are implemented. "
            "Uplift/scorer/evaluation remain fail-closed."
        )
