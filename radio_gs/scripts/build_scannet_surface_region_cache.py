#!/usr/bin/env python3
"""Build query-free 3-D surface-region/official-summary pairs from ScanNet.

Only ``color``, ``depth``, ``pose`` and the two intrinsic files are opened.
Semantic/instance labels are deliberately forbidden from the input contract.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import stat
import tempfile
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_region_summary import surface_region_geometry_v2
from radio_gs.interfaces.surface_scene_intermediate import (
    SourceFileBinding,
    SurfaceSceneFrameBinding,
    SurfaceSceneIntermediate,
    SurfaceSceneIntermediateContract,
    assert_exact_surface_scene_replay,
    default_graph_config_dict,
    load_surface_scene_intermediate,
    save_surface_scene_intermediate,
    scientific_tensor_bundle_sha256,
    scientific_tensor_sha256,
)
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.scripts.build_canonical_support_graph import deterministic_feature_hash
from radio_gs.scripts.surface_region_scene_resume import (
    RESUME_CONTRACT_ARTIFACT_TYPE,
    RESUME_SCHEMA_VERSION,
    SCENE_PARTIAL_SUFFIX,
    SCENE_TERMINAL_SUFFIX,
    append_scene_rows,
    commit_scene_partial,
    decode_rng_state,
    encode_rng_state,
    load_scene_partial,
    open_or_create_resume_contract,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


FORBIDDEN_EVAL_SCENES = {
    "scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00",
    "scene0140_00", "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00",
}
FIXED_CORE_TEACHER_SEMANTICS = (
    "fixed_core_geodesic_support_without_input_context_v1"
)
TEACHER_CROP_PROTOCOL = (
    "core_support_defined_unmasked_bbox_min24_context_pad0_v1"
)
SCENE_INTERMEDIATE_MANIFEST_ARTIFACT_TYPE = (
    "surface-scene-intermediate-manifest-v1"
)
SCENE_INTERMEDIATE_AUTHORITY_ARTIFACT_TYPE = (
    "surface-scene-intermediate-authority-v1"
)
SCENE_INTERMEDIATE_SCHEMA_VERSION = 1
SCENE_INTERMEDIATE_DATA_NAME = "intermediate.pt"
SCENE_INTERMEDIATE_AUTHORITY_NAME = "authority.json"
SCENE_INTERMEDIATE_MANIFEST_NAME = "manifest.json"
TEACHER_REPLAY_AUTHORITY_ARTIFACT_TYPE = (
    "surface-region-teacher-replay-authority-v1"
)
TEACHER_REPLAY_AUTHORITY_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_mapping(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )
    return value


def _absolute_without_resolving(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _resolved_intermediate_root(
    path: str | Path,
    *,
    create: bool,
) -> Path:
    """Resolve one controlled root while rejecting a symlink final component."""

    lexical = _absolute_without_resolving(path)
    if os.path.lexists(lexical) and lexical.is_symlink():
        raise ValueError(
            f"refuse symlinked scene-intermediate root: {lexical}"
        )
    if create:
        lexical.mkdir(parents=True, exist_ok=True)
    resolved = lexical.resolve(strict=True)
    info = os.lstat(resolved)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(
            f"scene-intermediate root is not a real directory: {resolved}"
        )
    return resolved


def _require_nosymlink_relative_path(
    *,
    resolved_root: Path,
    relative_path: Path,
    label: str,
    final_kind: str,
) -> Path:
    """Walk one relative path with lstat so no descendant can redirect it."""

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label} escapes its controlled root")
    current = resolved_root
    parts = relative_path.parts
    if not parts:
        raise ValueError(f"{label} cannot name the controlled root")
    for index, component in enumerate(parts):
        if component in {"", ".", ".."}:
            raise ValueError(f"{label} has an invalid path component")
        current = current / component
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} has a symlink component: {current}")
        is_final = index == len(parts) - 1
        if not is_final and not stat.S_ISDIR(info.st_mode):
            raise ValueError(
                f"{label} ancestor is not a directory: {current}"
            )
        if is_final:
            if final_kind == "directory" and not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"{label} is not a directory: {current}")
            if final_kind == "regular" and not stat.S_ISREG(info.st_mode):
                raise ValueError(f"{label} is not a regular file: {current}")
    return current


def _require_intermediate_tree_path(
    root: Path,
    path: Path,
    *,
    label: str,
    final_kind: str,
) -> Path:
    source = _absolute_without_resolving(path)
    try:
        relative = source.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes scene-intermediate root") from error
    return _require_nosymlink_relative_path(
        resolved_root=root,
        relative_path=relative,
        label=label,
        final_kind=final_kind,
    )


def _reject_scene_intermediate_tree_symlinks(root: Path) -> None:
    for current_root, directories, files in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        for name in [*directories, *files]:
            entry = Path(current_root) / name
            if stat.S_ISLNK(os.lstat(entry).st_mode):
                raise ValueError(
                    f"scene-intermediate tree has a symlink component: {entry}"
                )


def _require_scene_intermediate_root_entries(
    root: Path,
    *,
    scenes: list[str],
    manifest_expected: bool,
) -> None:
    expected = set(scenes)
    if manifest_expected:
        expected.add(SCENE_INTERMEDIATE_MANIFEST_NAME)
    actual = {entry.name for entry in os.scandir(root)}
    if actual != expected:
        raise ValueError(
            "scene-intermediate root entries differ: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _require_scene_source_path(
    dataset_root: Path,
    path: Path,
    *,
    label: str,
    final_kind: str,
) -> Path:
    """Allow the dataset-root alias once, then reject every descendant link."""

    lexical_root = _absolute_without_resolving(dataset_root)
    lexical_path = _absolute_without_resolving(path)
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the dataset root") from error
    return _require_nosymlink_relative_path(
        resolved_root=lexical_root.resolve(strict=True),
        relative_path=relative,
        label=label,
        final_kind=final_kind,
    )


def _verify_scene_intermediate_sources_nosymlink(
    dataset_root: Path,
    scene_name: str,
    frames: list[tuple[Path, Path, Path]],
) -> None:
    scene_dir = dataset_root / scene_name
    _require_scene_source_path(
        dataset_root,
        scene_dir,
        label=f"scene-intermediate scene {scene_name}",
        final_kind="directory",
    )
    for name in ("intrinsics_depth.txt", "intrinsics_color.txt"):
        _require_scene_source_path(
            dataset_root,
            scene_dir / name,
            label=f"scene-intermediate source {scene_name}/{name}",
            final_kind="regular",
        )
    for frame_index, (color, depth, pose) in enumerate(frames):
        for role, path in (
            ("color", color),
            ("depth", depth),
            ("pose", pose),
        ):
            _require_scene_source_path(
                dataset_root,
                path,
                label=(
                    f"scene-intermediate source {scene_name} frame "
                    f"{frame_index} {role}"
                ),
                final_kind="regular",
            )


def _source_binding_without_final_symlink(
    path: str | Path,
    *,
    label: str,
) -> SourceFileBinding:
    lexical = _absolute_without_resolving(path)
    info = os.lstat(lexical)
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{label} cannot be a symlink: {lexical}")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file: {lexical}")
    return SourceFileBinding.from_path(lexical)


def _scene_intermediate_common_bindings(
    args: argparse.Namespace,
) -> tuple[SourceFileBinding, dict[str, SourceFileBinding]]:
    package_root = Path(__file__).resolve().parents[1]
    implementations = {
        "scene_builder": Path(__file__).resolve(),
        "scene_intermediate_contract": (
            package_root / "interfaces" / "surface_scene_intermediate.py"
        ),
        "radio_runtime": package_root / "interfaces" / "frozen_radio_views.py",
        "radio_adaptors": package_root / "models" / "radio_adaptors.py",
        "feature_hash": package_root / "scripts" / "build_canonical_support_graph.py",
        "support_graph": package_root / "querying" / "support_solver.py",
    }
    return (
        _source_binding_without_final_symlink(
            args.radio_checkpoint,
            label="scene-intermediate RADIO checkpoint",
        ),
        {
            role: _source_binding_without_final_symlink(
                path,
                label=f"scene-intermediate implementation {role}",
            )
            for role, path in implementations.items()
        },
    )


def _binding_from_file_record(record: object, *, label: str) -> SourceFileBinding:
    value = _require_exact_mapping(
        record,
        {"path", "sha256"},
        label=label,
    )
    return SourceFileBinding(
        path=str(value["path"]),
        sha256=str(value["sha256"]),
    )


def _surface_scene_intermediate_contract(
    *,
    args: argparse.Namespace,
    scene_name: str,
    scene_input_record: dict,
    region_contract: SurfaceRegionContractV2,
    radio_version: str,
    radio_checkpoint: SourceFileBinding,
    implementation_sources: dict[str, SourceFileBinding],
) -> SurfaceSceneIntermediateContract:
    if scene_input_record.get("input_contract_status") != "bound":
        raise RuntimeError(
            "cannot construct scene-intermediate contract from unavailable "
            f"inputs: {scene_name}"
        )
    frames = []
    for frame in scene_input_record["frames"]:
        color = _binding_from_file_record(
            frame["color"],
            label=f"scene-intermediate {scene_name} color binding",
        )
        frames.append(
            SurfaceSceneFrameBinding(
                frame_id=Path(color.path).stem,
                color=color,
                depth=_binding_from_file_record(
                    frame["depth"],
                    label=f"scene-intermediate {scene_name} depth binding",
                ),
                pose=_binding_from_file_record(
                    frame["pose"],
                    label=f"scene-intermediate {scene_name} pose binding",
                ),
            )
        )
    return SurfaceSceneIntermediateContract(
        scene=scene_name,
        source_frames=tuple(frames),
        depth_intrinsic=_binding_from_file_record(
            scene_input_record["intrinsics"]["depth"],
            label=f"scene-intermediate {scene_name} depth intrinsic",
        ),
        color_intrinsic=_binding_from_file_record(
            scene_input_record["intrinsics"]["color"],
            label=f"scene-intermediate {scene_name} color intrinsic",
        ),
        radio_checkpoint=radio_checkpoint,
        radio_version=radio_version,
        radio_resolution=int(args.radio_resolution),
        depth_stride=int(args.depth_stride),
        voxel_size=float(args.voxel_size),
        adaptor_names={
            "appearance": "dino_v3_7b",
            "boundary": "sam3",
        },
        adaptor_batch_size=int(args.adaptor_batch_size),
        affinity_dimension=int(args.affinity_dim),
        graph_config=default_graph_config_dict(region_contract.graph_config()),
        implementation_sources=implementation_sources,
    )


def _scene_intermediate_paths(
    root: Path,
    scene_name: str,
) -> tuple[Path, Path, Path]:
    if Path(scene_name).name != scene_name or scene_name in {".", ".."}:
        raise ValueError(f"invalid scene-intermediate scene name: {scene_name!r}")
    directory = root / scene_name
    return (
        directory,
        directory / SCENE_INTERMEDIATE_DATA_NAME,
        directory / SCENE_INTERMEDIATE_AUTHORITY_NAME,
    )


def _scene_intermediate_authority_payload(
    value: SurfaceSceneIntermediate,
    *,
    artifact_sha256: str,
) -> dict:
    tensor_digests = scientific_tensor_sha256(value)
    return {
        "artifact_type": SCENE_INTERMEDIATE_AUTHORITY_ARTIFACT_TYPE,
        "schema_version": SCENE_INTERMEDIATE_SCHEMA_VERSION,
        "scene": value.contract.scene,
        "artifact": {
            "file_name": SCENE_INTERMEDIATE_DATA_NAME,
            "sha256": artifact_sha256,
        },
        "contract": value.contract.to_dict(),
        "contract_sha256": value.contract.digest,
        "tensor_bundle_sha256": scientific_tensor_bundle_sha256(value),
        "tensor_sha256": tensor_digests,
    }


def _load_published_scene_intermediate(
    directory: Path,
    *,
    root: Path,
    expected_contract: SurfaceSceneIntermediateContract,
    expected_authority_sha256: str | None = None,
    expected_manifest_record: dict | None = None,
) -> tuple[SurfaceSceneIntermediate, dict]:
    _require_intermediate_tree_path(
        root,
        directory,
        label=f"scene-intermediate directory {expected_contract.scene}",
        final_kind="directory",
    )
    entries = {entry.name for entry in os.scandir(directory)}
    expected_entries = {
        SCENE_INTERMEDIATE_DATA_NAME,
        SCENE_INTERMEDIATE_AUTHORITY_NAME,
    }
    if entries != expected_entries:
        raise ValueError(
            "scene-intermediate directory entries differ: "
            f"missing={sorted(expected_entries - entries)}, "
            f"unexpected={sorted(entries - expected_entries)}"
        )
    data_path = directory / SCENE_INTERMEDIATE_DATA_NAME
    authority_path = directory / SCENE_INTERMEDIATE_AUTHORITY_NAME
    _require_intermediate_tree_path(
        root,
        data_path,
        label=f"scene-intermediate data {expected_contract.scene}",
        final_kind="regular",
    )
    _require_intermediate_tree_path(
        root,
        authority_path,
        label=f"scene-intermediate authority {expected_contract.scene}",
        final_kind="regular",
    )
    authority, authority_sha256, _ = load_json_object(
        authority_path,
        expected_sha256=expected_authority_sha256,
        label=f"scene-intermediate authority {expected_contract.scene}",
    )
    authority = _require_exact_mapping(
        authority,
        {
            "artifact_type",
            "schema_version",
            "scene",
            "artifact",
            "contract",
            "contract_sha256",
            "tensor_bundle_sha256",
            "tensor_sha256",
        },
        label=f"scene-intermediate authority {expected_contract.scene}",
    )
    if (
        authority["artifact_type"]
        != SCENE_INTERMEDIATE_AUTHORITY_ARTIFACT_TYPE
        or authority["schema_version"] != SCENE_INTERMEDIATE_SCHEMA_VERSION
        or authority["scene"] != expected_contract.scene
    ):
        raise ValueError("scene-intermediate authority header differs")
    artifact = _require_exact_mapping(
        authority["artifact"],
        {"file_name", "sha256"},
        label="scene-intermediate artifact authority",
    )
    if artifact["file_name"] != SCENE_INTERMEDIATE_DATA_NAME:
        raise ValueError("scene-intermediate artifact file name differs")
    authority_contract = SurfaceSceneIntermediateContract.from_dict(
        authority["contract"]
    )
    if (
        authority_contract.digest != expected_contract.digest
        or authority_contract.to_dict() != expected_contract.to_dict()
        or authority["contract_sha256"] != expected_contract.digest
    ):
        raise ValueError("scene-intermediate external contract differs")
    value = load_surface_scene_intermediate(
        data_path,
        expected_contract=expected_contract,
        expected_file_sha256=str(artifact["sha256"]),
    )
    tensor_digests = scientific_tensor_sha256(value)
    tensor_bundle_sha256 = scientific_tensor_bundle_sha256(value)
    if (
        authority["tensor_sha256"] != tensor_digests
        or authority["tensor_bundle_sha256"] != tensor_bundle_sha256
    ):
        raise ValueError("scene-intermediate external tensor authority differs")
    provenance = {
        "scene": expected_contract.scene,
        "authority_sha256": authority_sha256,
        "artifact_sha256": str(artifact["sha256"]),
        "contract_sha256": expected_contract.digest,
        "tensor_bundle_sha256": tensor_bundle_sha256,
    }
    if expected_manifest_record is not None:
        expected_record = _require_exact_mapping(
            expected_manifest_record,
            set(provenance),
            label="scene-intermediate manifest scene record",
        )
        if expected_record != provenance:
            raise ValueError(
                "scene-intermediate manifest and scene authority differ"
            )
    return value, provenance


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_incomplete_scene_intermediate_pending(
    pending: Path,
    *,
    root: Path,
    scene_name: str,
) -> None:
    expected = root / f".{scene_name}.pending"
    if pending != expected or pending.parent != root:
        raise RuntimeError("refuse to remove an unexpected pending directory")
    _require_intermediate_tree_path(
        root,
        pending,
        label=f"scene-intermediate pending directory {scene_name}",
        final_kind="directory",
    )
    for current_root, directories, files in os.walk(
        pending,
        topdown=True,
        followlinks=False,
    ):
        for name in [*directories, *files]:
            entry = Path(current_root) / name
            if stat.S_ISLNK(os.lstat(entry).st_mode):
                raise ValueError(
                    f"scene-intermediate pending tree has a symlink: {entry}"
                )
    shutil.rmtree(pending)
    _fsync_directory(root)


def _recover_scene_intermediate_if_available(
    root: Path,
    *,
    scene_name: str,
    expected_contract: SurfaceSceneIntermediateContract,
) -> tuple[SurfaceSceneIntermediate, dict] | None:
    directory, _data_path, _authority_path = _scene_intermediate_paths(
        root,
        scene_name,
    )
    pending = root / f".{scene_name}.pending"
    if os.path.lexists(directory):
        value = _load_published_scene_intermediate(
            directory,
            root=root,
            expected_contract=expected_contract,
        )
        if os.path.lexists(pending):
            _remove_incomplete_scene_intermediate_pending(
                pending,
                root=root,
                scene_name=scene_name,
            )
        return value
    if not os.path.lexists(pending):
        return None
    pending_data = pending / SCENE_INTERMEDIATE_DATA_NAME
    pending_authority = pending / SCENE_INTERMEDIATE_AUTHORITY_NAME
    if not (
        os.path.lexists(pending_data)
        and os.path.lexists(pending_authority)
    ):
        _remove_incomplete_scene_intermediate_pending(
            pending,
            root=root,
            scene_name=scene_name,
        )
        return None
    # Once both files exist, treat the pending directory as a completed
    # publication candidate.  Any contract, digest, graph, or symlink failure
    # is authoritative and must not be hidden by deleting and recomputing it.
    recovered = _load_published_scene_intermediate(
        pending,
        root=root,
        expected_contract=expected_contract,
    )
    os.rename(pending, directory)
    _fsync_directory(root)
    return _load_published_scene_intermediate(
        directory,
        root=root,
        expected_contract=expected_contract,
        expected_authority_sha256=recovered[1]["authority_sha256"],
    )


def _publish_scene_intermediate(
    root: Path,
    value: SurfaceSceneIntermediate,
) -> tuple[SurfaceSceneIntermediate, dict]:
    scene_name = value.contract.scene
    existing = _recover_scene_intermediate_if_available(
        root,
        scene_name=scene_name,
        expected_contract=value.contract,
    )
    if existing is not None:
        assert_exact_surface_scene_replay(value, existing[0])
        return existing
    directory, _data_path, _authority_path = _scene_intermediate_paths(
        root,
        scene_name,
    )
    pending = root / f".{scene_name}.pending"
    os.mkdir(pending)
    artifact = save_surface_scene_intermediate(
        value,
        pending / SCENE_INTERMEDIATE_DATA_NAME,
    )
    write_frozen_json(
        pending / SCENE_INTERMEDIATE_AUTHORITY_NAME,
        _scene_intermediate_authority_payload(
            value,
            artifact_sha256=artifact.file_sha256,
        ),
    )
    checked = _load_published_scene_intermediate(
        pending,
        root=root,
        expected_contract=value.contract,
    )
    assert_exact_surface_scene_replay(value, checked[0])
    os.rename(pending, directory)
    _fsync_directory(root)
    return _load_published_scene_intermediate(
        directory,
        root=root,
        expected_contract=value.contract,
        expected_authority_sha256=checked[1]["authority_sha256"],
    )


def _scene_intermediate_manifest_payload(
    root: Path,
    *,
    scenes: list[str],
    scene_records: list[dict],
    run_contract: dict,
) -> dict:
    return {
        "artifact_type": SCENE_INTERMEDIATE_MANIFEST_ARTIFACT_TYPE,
        "schema_version": SCENE_INTERMEDIATE_SCHEMA_VERSION,
        "root": str(root),
        "selected_scenes": list(scenes),
        "scene_records": scene_records,
        "run_contract": run_contract,
        "publication": (
            "atomic_scene_directory_then_external_sha_manifest_v1"
        ),
    }


def _publish_scene_intermediate_manifest(
    root: Path,
    *,
    scenes: list[str],
    contracts: dict[str, SurfaceSceneIntermediateContract],
    run_contract: dict,
) -> dict:
    scene_records = []
    for scene_name in scenes:
        directory, _data_path, _authority_path = _scene_intermediate_paths(
            root,
            scene_name,
        )
        _value, provenance = _load_published_scene_intermediate(
            directory,
            root=root,
            expected_contract=contracts[scene_name],
        )
        scene_records.append(provenance)
    payload = _scene_intermediate_manifest_payload(
        root,
        scenes=scenes,
        scene_records=scene_records,
        run_contract=run_contract,
    )
    manifest_path = root / SCENE_INTERMEDIATE_MANIFEST_NAME
    _require_scene_intermediate_root_entries(
        root,
        scenes=scenes,
        manifest_expected=os.path.lexists(manifest_path),
    )
    if os.path.lexists(manifest_path):
        existing, _digest, _path = load_json_object(
            manifest_path,
            label="scene-intermediate output manifest",
        )
        if existing != payload:
            raise ValueError(
                "existing scene-intermediate manifest differs from this build"
            )
    else:
        write_frozen_json(manifest_path, payload)
    manifest_record = file_record(manifest_path)
    return {
        "mode": "fresh_publish",
        "root": str(root),
        "manifest": manifest_record,
        "run_contract_sha256": _json_sha256(run_contract),
        "scene_records": scene_records,
    }


def _load_scene_intermediate_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_scenes: list[str],
    expected_run_contract: dict,
) -> tuple[Path, dict[str, dict], dict]:
    lexical = _absolute_without_resolving(path)
    if lexical.name != SCENE_INTERMEDIATE_MANIFEST_NAME:
        raise ValueError(
            "scene-intermediate manifest must use the canonical manifest.json name"
        )
    manifest, digest, manifest_path = load_json_object(
        lexical,
        expected_sha256=expected_sha256,
        label="scene-intermediate replay manifest",
    )
    manifest = _require_exact_mapping(
        manifest,
        {
            "artifact_type",
            "schema_version",
            "root",
            "selected_scenes",
            "scene_records",
            "run_contract",
            "publication",
        },
        label="scene-intermediate replay manifest",
    )
    root = _resolved_intermediate_root(manifest_path.parent, create=False)
    _reject_scene_intermediate_tree_symlinks(root)
    if (
        manifest["artifact_type"]
        != SCENE_INTERMEDIATE_MANIFEST_ARTIFACT_TYPE
        or manifest["schema_version"] != SCENE_INTERMEDIATE_SCHEMA_VERSION
        or manifest["root"] != str(root)
        or manifest["selected_scenes"] != list(expected_scenes)
        or manifest["run_contract"] != expected_run_contract
        or manifest["publication"]
        != "atomic_scene_directory_then_external_sha_manifest_v1"
    ):
        raise ValueError("scene-intermediate replay manifest contract differs")
    _require_scene_intermediate_root_entries(
        root,
        scenes=expected_scenes,
        manifest_expected=True,
    )
    records = manifest["scene_records"]
    if not isinstance(records, list) or len(records) != len(expected_scenes):
        raise ValueError("scene-intermediate replay scene records differ")
    by_scene = {}
    for expected_scene, raw_record in zip(expected_scenes, records):
        record = _require_exact_mapping(
            raw_record,
            {
                "scene",
                "authority_sha256",
                "artifact_sha256",
                "contract_sha256",
                "tensor_bundle_sha256",
            },
            label="scene-intermediate replay scene record",
        )
        if record["scene"] != expected_scene or expected_scene in by_scene:
            raise ValueError(
                "scene-intermediate replay scene order or uniqueness differs"
            )
        directory, data_path, authority_path = _scene_intermediate_paths(
            root,
            expected_scene,
        )
        _require_intermediate_tree_path(
            root,
            directory,
            label=f"scene-intermediate replay directory {expected_scene}",
            final_kind="directory",
        )
        _require_intermediate_tree_path(
            root,
            data_path,
            label=f"scene-intermediate replay data {expected_scene}",
            final_kind="regular",
        )
        _require_intermediate_tree_path(
            root,
            authority_path,
            label=f"scene-intermediate replay authority {expected_scene}",
            final_kind="regular",
        )
        by_scene[expected_scene] = record
    return root, by_scene, {
        "mode": "exact_replay",
        "root": str(root),
        "manifest": {
            "path": str(manifest_path),
            "sha256": digest,
        },
        "run_contract_sha256": _json_sha256(expected_run_contract),
        "scene_records": records,
    }


def _thermal_pause(
    device: torch.device,
    seconds_per_image: float,
    *,
    image_count: int = 1,
) -> None:
    """Synchronize a CUDA burst before an execution-only cooling pause."""

    seconds = float(seconds_per_image)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("thermal pacing seconds must be finite and non-negative")
    if not isinstance(image_count, int) or image_count <= 0:
        raise ValueError("thermal pacing image_count must be a positive integer")
    delay = seconds * image_count
    if delay == 0:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    time.sleep(delay)


def _surface_region_id(
    scene: str,
    seed: int,
    radius_m: float,
    teacher_contract_sha256: str,
    teacher_support_sha256: str,
) -> str:
    payload = {
        "scene": str(scene),
        "seed": int(seed),
        "physical_radius_m": float(radius_m),
        "teacher_region_contract_sha256": str(
            teacher_contract_sha256
        ),
        "teacher_support_sha256": str(teacher_support_sha256),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _teacher_support_sha256(
    xyz: torch.Tensor,
    rows: torch.Tensor,
) -> str:
    """Bind a region identity to ordered voxel IDs and physical coordinates."""

    indices = torch.as_tensor(rows).detach().cpu().to(torch.int64).contiguous()
    coordinates = (
        torch.as_tensor(xyz)[indices]
        .detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
    )
    digest = hashlib.sha256()
    digest.update(np.asarray(indices.shape, dtype=np.int64).tobytes())
    digest.update(indices.numpy().tobytes())
    digest.update(np.asarray(coordinates.shape, dtype=np.int64).tobytes())
    digest.update(coordinates.numpy().tobytes())
    return digest.hexdigest()


def _teacher_target_sha256(
    summary_tokens: torch.Tensor,
    crop_summaries: torch.Tensor,
    mask: torch.Tensor,
) -> str:
    """Digest the exact padded official target tensors for one region."""

    digest = hashlib.sha256()
    for value in (summary_tokens, crop_summaries, mask):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _teacher_target_protocol(
    args: argparse.Namespace,
    teacher_contract: SurfaceRegionContractV2,
    runtime: OfficialCropSummaryRuntime | None,
    *,
    radio_version: str | None = None,
    radio_checkpoint_sha256: str | None = None,
) -> dict:
    if runtime is not None:
        effective_radio_version = str(runtime.version)
        effective_checkpoint_sha256 = str(
            runtime.radio_checkpoint_sha256
        )
    else:
        effective_radio_version = str(radio_version or "").strip()
        effective_checkpoint_sha256 = str(
            radio_checkpoint_sha256 or ""
        ).strip()
        if not effective_radio_version or len(effective_checkpoint_sha256) != 64:
            raise ValueError(
                "lightweight teacher protocol requires RADIO version and SHA-256"
            )
    return {
        "schema_version": 1,
        "support_semantics": FIXED_CORE_TEACHER_SEMANTICS,
        "teacher_region_contract_sha256": teacher_contract.digest,
        "crop_protocol": TEACHER_CROP_PROTOCOL,
        "frame_selection": "sorted_valid_frames_even_spacing_v1",
        "frames_per_scene": int(args.frames_per_scene),
        "minimum_visible_support_tokens": int(
            args.min_visible_tokens
        ),
        "maximum_teacher_views": int(args.teacher_views),
        "crop_resize_resolution": int(args.radio_resolution),
        "radio_version": effective_radio_version,
        "radio_checkpoint_sha256": effective_checkpoint_sha256,
        "target_padding": "left_aligned_zero_padding_v1",
        "teacher_medoid": "official_descriptor_pairwise_consensus_v1",
    }


def _json_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_teacher_replay_cache(
    path: str,
    *,
    authority_path: str = "",
    authority_sha256: str = "",
    expected_split_role: str,
    expected_split_file_sha256: str,
    expected_dataset_root: str,
    expected_teacher_contract_sha256: str,
    expected_teacher_target_protocol_sha256: str,
    expected_radio_checkpoint_sha256: str,
    expected_excluded_physical_spaces: set[str],
    expected_regions_per_scene: int,
    expected_teacher_views: int,
) -> tuple[
    dict,
    dict[str, list[tuple[int, dict]]],
    dict,
    dict,
] | None:
    raw = str(path or "").strip()
    if not raw:
        return None
    cache_path = Path(raw)
    try:
        payload, cache_sha256, cache_path = load_torch_mapping(
            cache_path,
            map_location="cpu",
            label="SurfaceRegion teacher replay cache",
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"teacher replay cache is missing: {cache_path}"
        ) from error
    metadata = payload.get("metadata", {})
    current_builder_sha256 = _sha256(Path(__file__).resolve())
    if (
        metadata.get("schema_version") != 3
        or metadata.get("split_role") != expected_split_role
        or metadata.get("split_file_sha256")
        != expected_split_file_sha256
        or metadata.get("dataset_root") != expected_dataset_root
        or metadata.get("teacher_region_contract_sha256")
        != expected_teacher_contract_sha256
        or metadata.get("teacher_target_protocol_sha256")
        != expected_teacher_target_protocol_sha256
        or metadata.get("radio_checkpoint_sha256")
        != expected_radio_checkpoint_sha256
        or set(metadata.get("excluded_physical_spaces", []))
        != expected_excluded_physical_spaces
        or metadata.get("teacher_region_semantics")
        != FIXED_CORE_TEACHER_SEMANTICS
        or metadata.get("teacher_target_schema_version") != 1
        or metadata.get("teacher_crop_protocol")
        != TEACHER_CROP_PROTOCOL
        or metadata.get("teacher_target_source")
        != "fresh_official_runtime"
        or metadata.get("teacher_regions_saturated") != 0
        or metadata.get("physical_space_disjoint") is not True
        or metadata.get("regions_per_scene_requested")
        != int(expected_regions_per_scene)
        or metadata.get("teacher_views_requested")
        != int(expected_teacher_views)
    ):
        raise ValueError(
            "teacher replay cache differs from the frozen teacher protocol"
        )
    source_builder_sha256 = metadata.get("builder_script_sha256")
    replay_authority: dict[str, str] = {}
    raw_authority = str(authority_path or "").strip()
    raw_authority_sha256 = str(authority_sha256 or "").strip()
    if source_builder_sha256 == current_builder_sha256:
        if raw_authority or raw_authority_sha256:
            raise ValueError(
                "teacher replay authority is only valid for a historical "
                "builder provenance mismatch"
            )
    else:
        if not raw_authority or not raw_authority_sha256:
            raise ValueError(
                "teacher replay cache builder provenance differs; an exact "
                "external replay authority and SHA-256 are required"
            )
        authority, authority_digest, authority_source = load_json_object(
            raw_authority,
            expected_sha256=raw_authority_sha256,
            label="historical teacher replay authority",
        )
        authority = _require_exact_mapping(
            authority,
            {
                "artifact_type",
                "schema_version",
                "authorization_scope",
                "run_manifest",
                "cache",
                "split_role",
                "split_file_sha256",
                "scene_names",
                "teacher_region_contract_sha256",
                "teacher_target_protocol_sha256",
                "radio_checkpoint_sha256",
                "source_builder_script_sha256",
            },
            label="historical teacher replay authority",
        )
        if (
            authority["artifact_type"]
            != TEACHER_REPLAY_AUTHORITY_ARTIFACT_TYPE
            or authority["schema_version"]
            != TEACHER_REPLAY_AUTHORITY_SCHEMA_VERSION
            or authority["authorization_scope"]
            != "exact_historical_cache_fixed_teacher_replay_only"
            or authority["cache"]
            != {"path": str(cache_path), "sha256": cache_sha256}
            or authority["split_role"] != expected_split_role
            or authority["split_file_sha256"]
            != expected_split_file_sha256
            or authority["scene_names"] != metadata.get("scene_names")
            or authority["teacher_region_contract_sha256"]
            != expected_teacher_contract_sha256
            or authority["teacher_target_protocol_sha256"]
            != expected_teacher_target_protocol_sha256
            or authority["radio_checkpoint_sha256"]
            != expected_radio_checkpoint_sha256
            or authority["source_builder_script_sha256"]
            != source_builder_sha256
        ):
            raise ValueError(
                "historical teacher replay authority differs from the exact "
                "cache and frozen teacher protocol"
            )
        validate_file_record(
            authority["run_manifest"],
            label="historical teacher replay authority run manifest",
        )
        replay_authority = {
            "path": str(authority_source),
            "sha256": authority_digest,
        }
    if (
        _json_sha256(metadata.get("teacher_target_protocol", {}))
        != expected_teacher_target_protocol_sha256
    ):
        raise ValueError("teacher replay protocol payload has a wrong digest")
    try:
        raw_contract = metadata["teacher_region_contract"]
        expanded_contract = {
            **raw_contract,
            "radii_m": tuple(raw_contract["radii_m"]),
        }
        expanded_contract.setdefault(
            "token_candidate_limit",
            int(raw_contract["maximum_tokens"]),
        )
        source_contract = SurfaceRegionContractV2(
            **expanded_contract
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "teacher replay cache has an invalid teacher contract"
        ) from error
    if source_contract.digest != expected_teacher_contract_sha256:
        raise ValueError("teacher replay contract payload has a wrong digest")
    if metadata.get("failed_scenes"):
        raise ValueError("teacher replay cache contains failed scenes")
    if any(
        metadata.get(key, True)
        for key in (
            "uses_benchmark_scenes",
            "uses_benchmark_test_vocabulary",
            "annotations_opened",
            "labels_opened",
            "instances_opened",
            "masks_opened",
            "text_opened",
        )
    ):
        raise ValueError("teacher replay cache violates the query-free contract")
    records = metadata.get("region_records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("teacher replay cache has no region records")
    required = (
        "official_summary_tokens",
        "official_crop_summaries",
        "teacher_mask",
    )
    if any(key not in payload for key in required) or any(
        len(payload[key]) != len(records) for key in required
    ):
        raise ValueError("teacher replay cache tensors do not align with records")
    by_scene: dict[str, list[tuple[int, dict]]] = {}
    seen: set[str] = set()
    for row, record in enumerate(records):
        if (
            record.get("teacher_target_source")
            != "fresh_official_runtime"
            or record.get("teacher_region_saturated") is not False
            or int(record.get("teacher_region_tokens", 0)) <= 0
        ):
            raise ValueError(
                "teacher replay cache contains non-fresh or saturated targets"
            )
        region_id = str(record.get("region_id", ""))
        expected_id = _surface_region_id(
            record["scene"],
            int(record["seed"]),
            float(record["physical_radius_m"]),
            expected_teacher_contract_sha256,
            str(record.get("teacher_support_sha256", "")),
        )
        if (
            not str(record.get("teacher_support_sha256", ""))
            or not region_id
            or region_id != expected_id
            or region_id in seen
        ):
            raise ValueError("teacher replay cache has invalid region IDs")
        summary_tokens = torch.as_tensor(
            payload["official_summary_tokens"][row]
        )
        crop_summaries = torch.as_tensor(
            payload["official_crop_summaries"][row]
        )
        teacher_mask = torch.as_tensor(
            payload["teacher_mask"][row]
        ).bool()
        view_count = int(teacher_mask.sum())
        expected_mask = (
            torch.arange(int(expected_teacher_views)) < view_count
        )
        if (
            summary_tokens.dtype != torch.float16
            or crop_summaries.dtype != torch.float16
            or summary_tokens.shape
            != (int(expected_teacher_views), 1280)
            or crop_summaries.ndim != 2
            or crop_summaries.shape[0] != int(expected_teacher_views)
            or teacher_mask.shape != (int(expected_teacher_views),)
            or view_count < 2
            or len(record.get("teacher_views", [])) != view_count
            or not torch.equal(teacher_mask.cpu(), expected_mask)
            or not torch.isfinite(summary_tokens).all()
            or not torch.isfinite(crop_summaries).all()
            or bool(summary_tokens[~teacher_mask].count_nonzero())
            or bool(crop_summaries[~teacher_mask].count_nonzero())
            or not 0 <= int(record.get("teacher_medoid", -1)) < view_count
        ):
            raise ValueError(
                "teacher replay cache has malformed target tensors"
            )
        target_sha256 = _teacher_target_sha256(
            summary_tokens,
            crop_summaries,
            teacher_mask,
        )
        if record.get("teacher_target_sha256") != target_sha256:
            raise ValueError("teacher replay target digest is inconsistent")
        seen.add(region_id)
        by_scene.setdefault(str(record["scene"]), []).append((row, record))
    if set(by_scene) != set(metadata.get("scene_names", [])):
        raise ValueError("teacher replay cache scene metadata is inconsistent")
    if metadata.get("scene_region_counts") != {
        scene: len(values)
        for scene, values in sorted(by_scene.items())
    }:
        raise ValueError("teacher replay cache region counts are inconsistent")
    provenance = {
        "path": str(cache_path),
        "sha256": cache_sha256,
    }
    return payload, by_scene, provenance, replay_authority


def _physical_space(scene_name: str) -> str:
    """Return ScanNet's physical-space identifier (``sceneXXXX``)."""

    return str(scene_name).split("_", 1)[0]


def _read_scene_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _excluded_spaces(
    scene_files: str,
    scene_names: str,
) -> tuple[set[str], list[dict[str, str]]]:
    """Compile an auditable physical-space exclusion set.

    Excluding the whole ``sceneXXXX`` space prevents a different rescan from
    leaking the development/test environment into the global readout.
    """

    names = [
        value
        for value in str(scene_names).replace(",", " ").split()
        if value
    ]
    records: list[dict[str, str]] = []
    for value in str(scene_files).replace(",", " ").split():
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"exclude scene file is missing: {path}")
        names.extend(_read_scene_file(path))
        records.append(file_record(path))
    spaces = {_physical_space(name) for name in names}
    return spaces, records


def _scene_names(
    split_file: Path,
    root: Path,
    *,
    excluded_physical_spaces: set[str] | None = None,
) -> list[str]:
    names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    names = [name for name in names if name not in FORBIDDEN_EVAL_SCENES]
    excluded = excluded_physical_spaces or set()
    names = [name for name in names if _physical_space(name) not in excluded]
    return [name for name in names if (root / name).is_dir()]


def _frame_paths(scene_dir: Path) -> list[tuple[Path, Path, Path]]:
    triples = []
    for color in sorted((scene_dir / "color").glob("*.jpg")):
        depth = scene_dir / "depth" / f"{color.stem}.png"
        pose = scene_dir / "pose" / f"{color.stem}.txt"
        if depth.is_file() and pose.is_file():
            matrix = np.loadtxt(pose)
            if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                triples.append((color, depth, pose))
    return triples


def _select_evenly(values: list, count: int) -> list:
    if len(values) <= int(count):
        return values
    indices = np.linspace(0, len(values) - 1, int(count)).round().astype(int)
    return [values[int(index)] for index in indices]


def _load_color(path: Path) -> torch.Tensor:
    return pil_to_tensor(Image.open(path).convert("RGB")).float().div_(255.0)


def _load_depth(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path), dtype=np.float32).copy()).div_(1000.0)


def _lift_observation(
    depth: torch.Tensor,
    depth_intrinsic: torch.Tensor,
    color_intrinsic: torch.Tensor,
    camera_to_world: torch.Tensor,
    spatial: torch.Tensor,
    *,
    stride: int,
    color_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift one depth map and bilinearly sample the matching RADIO map."""

    height, width = depth.shape
    ys = torch.arange(0, height, int(stride), dtype=torch.long)
    xs = torch.arange(0, width, int(stride), dtype=torch.long)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    z = depth[yy, xx]
    valid = torch.isfinite(z) & (z > 0.20) & (z < 8.0)
    x = (xx.float() - depth_intrinsic[0, 2]) * z / depth_intrinsic[0, 0]
    y = (yy.float() - depth_intrinsic[1, 2]) * z / depth_intrinsic[1, 1]
    camera = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)[valid]
    world = (camera @ camera_to_world.T)[:, :3]

    # Color and depth share the exported pose but have different intrinsics.
    u = color_intrinsic[0, 0] * x / z.clamp_min(1e-6) + color_intrinsic[0, 2]
    v = color_intrinsic[1, 1] * y / z.clamp_min(1e-6) + color_intrinsic[1, 2]
    color_width, color_height = (float(value) for value in color_size)
    # ``align_corners=False`` maps pixel centres, not pixel corners.  Omitting
    # the half-pixel term creates a fixed lifting bias in both image axes.
    grid = torch.stack(
        [2.0 * (u + 0.5) / color_width - 1.0,
         2.0 * (v + 0.5) / color_height - 1.0], dim=-1
    )[valid]
    features = F.grid_sample(
        spatial[None].float(), grid[None, None], mode="bilinear",
        padding_mode="border", align_corners=False,
    )[0, :, 0].T
    footprint = (z[valid] / depth_intrinsic[0, 0] * float(stride)).clamp_min(1e-3)
    return world, features, footprint


def _voxel_fuse(
    xyz: torch.Tensor, features: torch.Tensor, footprint: torch.Tensor, voxel_size: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = torch.floor(xyz / float(voxel_size)).to(torch.int64)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    count = torch.bincount(inverse, minlength=unique.shape[0]).float()
    fused_xyz = torch.zeros(unique.shape[0], 3).index_add_(0, inverse, xyz) / count[:, None]
    fused_features = torch.zeros(unique.shape[0], features.shape[1]).index_add_(
        0, inverse, features
    ) / count[:, None]
    fused_footprint = torch.zeros(unique.shape[0]).index_add_(0, inverse, footprint) / count
    return fused_xyz, fused_features, fused_footprint, count


def _project_region_box(
    xyz: torch.Tensor, depth: torch.Tensor, depth_intrinsic: torch.Tensor,
    color_intrinsic: torch.Tensor, camera_to_world: torch.Tensor,
    color_size: tuple[int, int], *, min_visible: int, context_pad: float = 0.12,
) -> list[int] | None:
    world_to_camera = torch.linalg.inv(camera_to_world)
    camera = torch.cat([xyz, torch.ones(len(xyz), 1)], dim=1) @ world_to_camera.T
    z = camera[:, 2]
    valid = z > 0.15
    if not bool(valid.any()):
        return None
    camera = camera[valid]
    z = z[valid]
    ud = depth_intrinsic[0, 0] * camera[:, 0] / z + depth_intrinsic[0, 2]
    vd = depth_intrinsic[1, 1] * camera[:, 1] / z + depth_intrinsic[1, 2]
    ix, iy = ud.round().long(), vd.round().long()
    inside = (ix >= 0) & (iy >= 0) & (ix < depth.shape[1]) & (iy < depth.shape[0])
    visible = torch.zeros_like(inside)
    if bool(inside.any()):
        observed = depth[iy[inside], ix[inside]]
        visible[inside] = (observed > 0) & ((observed - z[inside]).abs() < 0.10)
    if int(visible.sum()) < int(min_visible):
        return None
    camera, z = camera[visible], z[visible]
    u = color_intrinsic[0, 0] * camera[:, 0] / z + color_intrinsic[0, 2]
    v = color_intrinsic[1, 1] * camera[:, 1] / z + color_intrinsic[1, 2]
    width, height = color_size
    x0, x1 = float(u.min()), float(u.max())
    y0, y1 = float(v.min()), float(v.max())
    pad = float(context_pad) * max(x1 - x0, y1 - y0, 16.0)
    left, right = max(0, int(math.floor(x0 - pad))), min(width, int(math.ceil(x1 + pad)))
    top, bottom = max(0, int(math.floor(y0 - pad))), min(height, int(math.ceil(y1 + pad)))
    # Isolated/small surface components are valid inference regions too.  Give
    # their teacher observation a deterministic minimum pixel footprint rather
    # than silently dropping them from the bridge's training support.
    minimum_crop = 24
    if right - left < minimum_crop:
        centre = 0.5 * (left + right)
        left = max(0, min(width - minimum_crop, int(round(centre - minimum_crop / 2))))
        right = min(width, left + minimum_crop)
    if bottom - top < minimum_crop:
        centre = 0.5 * (top + bottom)
        top = max(0, min(height - minimum_crop, int(round(centre - minimum_crop / 2))))
        bottom = min(height, top + minimum_crop)
    if right <= left or bottom <= top:
        return None
    return [top, left, bottom, right]


def _teacher_medoid(tokens: torch.Tensor, descriptors: torch.Tensor | None = None) -> int:
    """Select the view nearest the multiview centre in official semantic space."""
    values = tokens if descriptors is None else descriptors
    normalized = F.normalize(values.float(), dim=-1, eps=1e-8)
    return int((normalized @ normalized.T).sum(dim=1).argmax())


def _voxel_reliability_v2(
    xyz: torch.Tensor,
    features: torch.Tensor,
    voxel_size: float,
    fused_features: torch.Tensor,
) -> torch.Tensor:
    """Coverage/agreement geometric mean matching canonical reliability semantics."""
    keys = torch.floor(xyz / float(voxel_size)).to(torch.int64)
    _unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    count = torch.bincount(inverse, minlength=fused_features.shape[0]).float()
    direction = F.normalize(features.float(), dim=-1, eps=1e-8)
    centre = F.normalize(fused_features.float(), dim=-1, eps=1e-8)
    cosine = (direction * centre[inverse]).sum(-1).clamp(-1, 1)
    agreement = torch.zeros_like(count).index_add_(0, inverse, (cosine + 1.0) * 0.5)
    agreement = agreement / count.clamp_min(1.0)
    coverage = 1.0 - torch.exp(-count / 2.0)
    return (coverage.clamp_min(1e-6) * agreement.clamp_min(1e-6)).sqrt()


def _candidate_reliability_from_geometric(
    geometric_reliability: torch.Tensor | None,
    *,
    mode: str,
    num_rows: int,
) -> torch.Tensor:
    """Apply candidate semantics without changing the shared scene domain."""

    rows = int(num_rows)
    if rows <= 0:
        raise ValueError("candidate reliability requires a positive scene size")
    if geometric_reliability is not None:
        if (
            not torch.is_tensor(geometric_reliability)
            or geometric_reliability.dtype != torch.float32
            or geometric_reliability.device.type != "cpu"
            or geometric_reliability.shape != (rows,)
            or not bool(torch.isfinite(geometric_reliability).all())
            or bool((geometric_reliability < 0).any())
            or bool((geometric_reliability > 1).any())
        ):
            raise ValueError(
                "geometric reliability must be finite CPU float32 [N] in [0,1]"
            )
    if mode == "uniform_valid":
        return (
            torch.ones_like(geometric_reliability)
            if geometric_reliability is not None
            else torch.ones(rows, dtype=torch.float32)
        )
    if mode == "geometric_mean_observation_agreement":
        if geometric_reliability is None:
            raise RuntimeError("geometric reliability was not materialized")
        return geometric_reliability
    raise ValueError(f"unsupported candidate reliability mode: {mode}")


def _teacher_region_contract(
    input_contract: SurfaceRegionContractV2,
    candidate_limit: int,
) -> SurfaceRegionContractV2:
    """Build a fixed core-only teacher support independent of input sampling."""

    limit = int(candidate_limit)
    if limit < input_contract.minimum_tokens:
        raise ValueError(
            "teacher-region candidate limit is below the minimum token count"
        )
    return replace(
        input_contract,
        context_ratio=1.0,
        maximum_tokens=limit,
        minimum_tokens=1,
        token_candidate_limit=limit,
        token_subsampling="nearest_geodesic_then_node_index",
        core_token_fraction=1.0,
        # Reliability weights condition the student input only; they neither
        # define the physical teacher ball nor the official crop target.
        reliability_semantics="uniform_valid",
    )


def _normalize_resume_cli(args: argparse.Namespace, resume_dir: Path) -> dict:
    """Capture every parsed CLI field without dropping execution-only knobs."""

    values = dict(vars(args))
    values["resume_dir"] = str(resume_dir)
    for name in (
        "dataset_root",
        "split_file",
        "output",
        "radio_repo",
        "radio_checkpoint",
        "scene_graph_output_root",
        "scene_intermediate_output_root",
        "scene_intermediate_manifest",
        "teacher_replay_cache",
    ):
        raw = str(values.get(name, "") or "").strip()
        values[name] = str(Path(raw).resolve()) if raw else ""
    for name in (
        "scene_intermediate_output_root",
        "scene_intermediate_manifest",
        "scene_intermediate_manifest_sha256",
    ):
        if not str(values.get(name, "") or "").strip():
            values.pop(name, None)
    return dict(sorted(values.items()))


def _scene_intermediate_run_contract(
    args: argparse.Namespace,
    *,
    dataset_root: str,
    split_file: Path,
    split_file_sha256: str,
    split_role: str,
    scenes: list[str],
    excluded_physical_spaces: set[str],
    exclusion_files: list[dict[str, str]],
) -> dict:
    split_binding = _source_binding_without_final_symlink(
        split_file,
        label="scene-intermediate split file",
    )
    if split_binding.sha256 != split_file_sha256:
        raise ValueError("scene-intermediate split binding differs")
    explicit_scenes = [
        value
        for value in str(args.scene_names).replace(",", " ").split()
        if value
    ]
    return {
        "builder": file_record(Path(__file__).resolve()),
        "dataset_root": dataset_root,
        "split_role": split_role,
        "split_file": split_binding.to_dict(),
        "selection": {
            "mode": (
                "explicit_scene_names"
                if explicit_scenes
                else "deterministic_shard"
            ),
            "scene_names_argument": explicit_scenes,
            "shard_count": int(args.shard_count),
            "shard_index": int(args.shard_index),
            "max_scenes": int(args.max_scenes),
            "selected_scenes": list(scenes),
        },
        "excluded_physical_spaces": sorted(excluded_physical_spaces),
        "exclusion_files": exclusion_files,
    }


def _selected_scene_input_contract(
    root: Path,
    scenes: list[str],
    *,
    frames_per_scene: int,
) -> tuple[dict[str, list[tuple[Path, Path, Path]]], list[dict]]:
    """Bind exactly the RGB-D-pose and intrinsic files selected per scene."""

    selected: dict[str, list[tuple[Path, Path, Path]]] = {}
    records: list[dict] = []
    for scene_name in scenes:
        scene_dir = root / scene_name
        try:
            frames = _select_evenly(
                _frame_paths(scene_dir),
                int(frames_per_scene),
            )
            selected[scene_name] = frames
            records.append(
                {
                    "scene": scene_name,
                    "input_contract_status": "bound",
                    "intrinsics": {
                        "depth": file_record(
                            scene_dir / "intrinsics_depth.txt"
                        ),
                        "color": file_record(
                            scene_dir / "intrinsics_color.txt"
                        ),
                    },
                    "frames": [
                        {
                            "color": file_record(color),
                            "depth": file_record(depth),
                            "pose": file_record(pose),
                        }
                        for color, depth, pose in frames
                    ],
                }
            )
        except Exception as error:
            # Preserve the historical per-scene failure semantics.  A broken
            # scene remains in the immutable run contract but does not prevent
            # other scenes from producing complete rows.
            selected.setdefault(scene_name, [])
            records.append(
                {
                    "scene": scene_name,
                    "input_contract_status": "unavailable",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return selected, records


def _scene_rows_from_accumulators(
    *,
    scene_start: int,
    records: list[dict],
    tensor_rows: dict[str, list[torch.Tensor]],
) -> dict:
    return {
        **{
            key: torch.stack(values[scene_start:])
            for key, values in tensor_rows.items()
        },
        "records": records[scene_start:],
    }


def _verify_scene_input_contract(record: dict) -> None:
    if record.get("input_contract_status") != "bound":
        raise RuntimeError(
            "scene inputs could not be bound into the durable resume contract: "
            f"{record.get('error', 'unknown input error')}"
        )
    intrinsics = record["intrinsics"]
    validate_file_record(
        intrinsics["depth"],
        label="SurfaceRegion depth intrinsic",
    )
    validate_file_record(
        intrinsics["color"],
        label="SurfaceRegion color intrinsic",
    )
    for frame in record["frames"]:
        for role in ("color", "depth", "pose"):
            validate_file_record(
                frame[role],
                label=f"SurfaceRegion scene frame {role}",
            )


@torch.no_grad()
def build(args: argparse.Namespace) -> dict:
    root, split_file = Path(args.dataset_root), Path(args.split_file)
    output = Path(args.output)
    if os.path.lexists(output) and not bool(
        getattr(args, "overwrite_output", False)
    ):
        raise FileExistsError(
            f"output already exists; refuse to mix cache runs: {output}"
        )
    if output.is_symlink():
        raise ValueError(f"refuse symlinked SurfaceRegion output: {output}")
    dataset_root = str(root.resolve())
    split_file_sha256 = _sha256(split_file)
    builder_script_sha256 = _sha256(Path(__file__).resolve())
    split_role = str(args.split_role)
    if split_role not in {"train", "validation"}:
        raise ValueError("split-role must be train or validation")
    excluded_spaces, exclusion_files = _excluded_spaces(
        args.exclude_scene_files,
        args.exclude_scene_names,
    )
    scenes = _scene_names(
        split_file,
        root,
        excluded_physical_spaces=excluded_spaces,
    )
    if str(args.scene_names).strip():
        requested = [
            value for value in str(args.scene_names).replace(",", " ").split()
            if value
        ]
        missing = sorted(set(requested) - set(scenes))
        if missing:
            raise ValueError(f"requested scenes are absent/forbidden: {missing}")
        scenes = requested
    else:
        scenes = scenes[int(args.shard_index)::int(args.shard_count)][:int(args.max_scenes)]
    if not scenes:
        raise RuntimeError("no ScanNet scenes selected")
    if FORBIDDEN_EVAL_SCENES.intersection(scenes):
        raise RuntimeError("paper evaluation scenes leaked into bridge data")
    leaked_spaces = {_physical_space(name) for name in scenes} & excluded_spaces
    if leaked_spaces:
        raise RuntimeError(
            f"excluded benchmark physical spaces leaked into bridge data: "
            f"{sorted(leaked_spaces)}"
        )
    intermediate_output_raw = str(
        getattr(args, "scene_intermediate_output_root", "") or ""
    ).strip()
    intermediate_manifest_raw = str(
        getattr(args, "scene_intermediate_manifest", "") or ""
    ).strip()
    intermediate_manifest_sha256 = str(
        getattr(args, "scene_intermediate_manifest_sha256", "") or ""
    ).strip()
    if intermediate_output_raw and intermediate_manifest_raw:
        raise ValueError(
            "scene-intermediate output and replay modes are mutually exclusive"
        )
    if bool(intermediate_manifest_raw) != bool(
        intermediate_manifest_sha256
    ):
        raise ValueError(
            "scene-intermediate replay requires both manifest path and SHA-256"
        )
    intermediate_run_contract = (
        _scene_intermediate_run_contract(
            args,
            dataset_root=dataset_root,
            split_file=split_file,
            split_file_sha256=split_file_sha256,
            split_role=split_role,
            scenes=scenes,
            excluded_physical_spaces=excluded_spaces,
            exclusion_files=exclusion_files,
        )
        if intermediate_output_raw or intermediate_manifest_raw
        else {}
    )
    intermediate_output_root = (
        _resolved_intermediate_root(
            intermediate_output_raw,
            create=True,
        )
        if intermediate_output_raw
        else None
    )
    intermediate_replay_root = None
    intermediate_manifest_by_scene: dict[str, dict] = {}
    intermediate_run_provenance: dict = {}
    if intermediate_manifest_raw:
        (
            intermediate_replay_root,
            intermediate_manifest_by_scene,
            intermediate_run_provenance,
        ) = _load_scene_intermediate_manifest(
            intermediate_manifest_raw,
            expected_sha256=intermediate_manifest_sha256,
            expected_scenes=scenes,
            expected_run_contract=intermediate_run_contract,
        )
    intermediate_enabled = bool(
        intermediate_output_root is not None
        or intermediate_replay_root is not None
    )
    radio_checkpoint_binding = None
    intermediate_implementation_sources: dict[
        str, SourceFileBinding
    ] = {}
    if intermediate_enabled:
        (
            radio_checkpoint_binding,
            intermediate_implementation_sources,
        ) = _scene_intermediate_common_bindings(args)
    lightweight_exact_replay = bool(
        intermediate_replay_root is not None
        and str(getattr(args, "teacher_replay_cache", "") or "").strip()
    )
    runtime = None
    if lightweight_exact_replay:
        assert radio_checkpoint_binding is not None
        runtime_version = str(args.radio_version)
        runtime_checkpoint_sha256 = radio_checkpoint_binding.sha256
    else:
        runtime = OfficialCropSummaryRuntime.load(
            checkpoint_path=args.radio_checkpoint,
            radio_repo=args.radio_repo,
            version=args.radio_version,
            device=args.device,
        )
        runtime_version = str(runtime.version)
        runtime_checkpoint_sha256 = str(
            runtime.radio_checkpoint_sha256
        )
    device = torch.device(args.device)
    thermal_pacing_seconds = float(
        args.radio_thermal_pacing_seconds_per_image
    )
    if (
        not math.isfinite(thermal_pacing_seconds)
        or thermal_pacing_seconds < 0
    ):
        raise ValueError(
            "radio-thermal-pacing-seconds-per-image must be finite and non-negative"
        )
    contract = SurfaceRegionContractV2(
        radii_m=tuple(float(v) for v in str(args.region_radii).replace(",", " ").split()),
        context_ratio=float(args.context_ratio),
        neighbors=int(args.graph_neighbors),
        maximum_tokens=int(args.max_tokens),
        minimum_tokens=int(args.min_tokens),
        path_cost_mode=str(args.path_cost_mode),
        path_affinity_floor=float(args.path_affinity_floor),
        token_subsampling=str(args.token_subsampling),
        token_candidate_limit=int(args.token_candidate_limit),
        core_token_fraction=float(args.core_token_fraction),
        reliability_semantics=str(
            getattr(
                args,
                "region_reliability_mode",
                "geometric_mean_observation_agreement",
            )
        ),
    )
    teacher_contract = _teacher_region_contract(
        contract,
        int(args.teacher_region_candidate_limit),
    )
    teacher_target_protocol = _teacher_target_protocol(
        args,
        teacher_contract,
        runtime,
        radio_version=runtime_version,
        radio_checkpoint_sha256=runtime_checkpoint_sha256,
    )
    teacher_target_protocol_sha256 = _json_sha256(
        teacher_target_protocol
    )
    teacher_replay = _load_teacher_replay_cache(
        str(getattr(args, "teacher_replay_cache", "")),
        authority_path=str(
            getattr(args, "teacher_replay_authority", "")
        ),
        authority_sha256=str(
            getattr(args, "teacher_replay_authority_sha256", "")
        ),
        expected_split_role=split_role,
        expected_split_file_sha256=split_file_sha256,
        expected_dataset_root=dataset_root,
        expected_teacher_contract_sha256=teacher_contract.digest,
        expected_teacher_target_protocol_sha256=(
            teacher_target_protocol_sha256
        ),
        expected_radio_checkpoint_sha256=(
            runtime_checkpoint_sha256
        ),
        expected_excluded_physical_spaces=excluded_spaces,
        expected_regions_per_scene=int(args.regions_per_scene),
        expected_teacher_views=int(args.teacher_views),
    )
    replay_payload = None
    replay_by_scene: dict[str, list[tuple[int, dict]]] = {}
    replay_provenance: dict[str, str] = {}
    replay_authority_provenance: dict[str, str] = {}
    if teacher_replay is not None:
        (
            replay_payload,
            replay_by_scene,
            replay_provenance,
            replay_authority_provenance,
        ) = teacher_replay
        if set(replay_by_scene) != set(scenes):
            raise ValueError(
                "teacher replay cache scene set differs from this shard"
            )
        wrong_counts = {
            scene: len(replay_by_scene[scene])
            for scene in scenes
            if len(replay_by_scene[scene])
            != int(args.regions_per_scene)
        }
        if wrong_counts:
            raise ValueError(
                "teacher replay cache has incomplete scene region counts: "
                f"{wrong_counts}"
            )
    teacher_target_source = (
        "exact_cache_replay"
        if replay_payload is not None
        else "fresh_official_runtime"
    )
    selected_frames, scene_input_records = _selected_scene_input_contract(
        root,
        scenes,
        frames_per_scene=int(args.frames_per_scene),
    )
    scene_input_by_name = {
        str(record["scene"]): record
        for record in scene_input_records
    }
    scene_intermediate_contracts: dict[
        str, SurfaceSceneIntermediateContract
    ] = {}
    if intermediate_enabled:
        assert radio_checkpoint_binding is not None
        if radio_checkpoint_binding.sha256 != runtime_checkpoint_sha256:
            raise ValueError(
                "scene-intermediate checkpoint binding differs from runtime"
            )
        for scene_name in scenes:
            _verify_scene_intermediate_sources_nosymlink(
                root,
                scene_name,
                selected_frames[scene_name],
            )
            _verify_scene_input_contract(scene_input_by_name[scene_name])
            expected_scene_contract = _surface_scene_intermediate_contract(
                args=args,
                scene_name=scene_name,
                scene_input_record=scene_input_by_name[scene_name],
                region_contract=contract,
                radio_version=runtime_version,
                radio_checkpoint=radio_checkpoint_binding,
                implementation_sources=intermediate_implementation_sources,
            )
            manifest_record = intermediate_manifest_by_scene.get(scene_name)
            if (
                manifest_record is not None
                and manifest_record["contract_sha256"]
                != expected_scene_contract.digest
            ):
                raise ValueError(
                    "scene-intermediate manifest graph/source contract differs "
                    f"for {scene_name}"
                )
            scene_intermediate_contracts[scene_name] = (
                expected_scene_contract
            )
    if intermediate_output_root is not None:
        intermediate_resume_input = {
            "mode": "fresh_publish",
            "root": str(intermediate_output_root),
            "manifest_path": str(
                intermediate_output_root
                / SCENE_INTERMEDIATE_MANIFEST_NAME
            ),
            "scene_contract_sha256": {
                scene: scene_intermediate_contracts[scene].digest
                for scene in scenes
            },
        }
    elif intermediate_replay_root is not None:
        intermediate_resume_input = intermediate_run_provenance
    else:
        intermediate_resume_input = {}
    resume_dir_raw = str(getattr(args, "resume_dir", "") or "").strip()
    resume_dir = (
        Path(resume_dir_raw).resolve()
        if resume_dir_raw
        else output.with_name(f"{output.name}.scene-resume").resolve()
    )
    resume_contract = {
        "artifact_type": RESUME_CONTRACT_ARTIFACT_TYPE,
        "schema_version": RESUME_SCHEMA_VERSION,
        "builder": {
            "entrypoint": file_record(Path(__file__).resolve()),
            "scene_resume_implementation": file_record(
                Path(__file__).with_name(
                    "surface_region_scene_resume.py"
                ).resolve()
            ),
        },
        "cli": _normalize_resume_cli(args, resume_dir),
        "inputs": {
            "dataset_root": dataset_root,
            "split_file": file_record(split_file),
            "exclusion_files": exclusion_files,
            "excluded_physical_spaces": sorted(excluded_spaces),
            "radio": {
                "repository": str(Path(args.radio_repo).resolve()),
                "version": runtime_version,
                "checkpoint": {
                    "path": str(Path(args.radio_checkpoint).resolve()),
                    "sha256": runtime_checkpoint_sha256,
                },
            },
            "teacher_replay_cache": replay_provenance,
            **(
                {"scene_intermediate": intermediate_resume_input}
                if intermediate_enabled
                else {}
            ),
            "scenes": scene_input_records,
        },
        "selected_scenes": list(scenes),
        "row_contract": {
            "regions_per_scene": int(args.regions_per_scene),
            "maximum_tokens": int(args.max_tokens),
            "feature_dimension": 1280,
            "geometry_dimension": 14,
            "teacher_views": int(args.teacher_views),
            "region_contract_sha256": contract.digest,
            "teacher_region_contract_sha256": teacher_contract.digest,
            "teacher_target_protocol_sha256": (
                teacher_target_protocol_sha256
            ),
            **(
                {
                    "scene_intermediate_contract_sha256": {
                        scene: scene_intermediate_contracts[scene].digest
                        for scene in scenes
                    }
                }
                if intermediate_enabled
                else {}
            ),
        },
        "resume_protocol": {
            "scene_partial_suffix": SCENE_PARTIAL_SUFFIX,
            "scene_terminal_suffix": SCENE_TERMINAL_SUFFIX,
            "partial_deserialization": (
                "torch_weights_only_same_fd_external_sha256_v1"
            ),
            "publication": "data_noclobber_then_sha_terminal_noclobber_v1",
            "random_state": "python_random_getstate_before_after_basic_types_v1",
            "merge_order": "selected_scene_order_then_region_row_order_v1",
            "promotable_glob": "final_cache_only_no_scene_partial_pt_suffix_v1",
        },
    }
    (
        resume_dir,
        resume_contract_record,
        resume_contract_payload_sha256,
    ) = open_or_create_resume_contract(resume_dir, resume_contract)
    adaptors = {}
    if intermediate_replay_root is None:
        adaptors = {
            "appearance": load_radio_adaptor_from_checkpoint(
                args.radio_checkpoint,
                "dino_v3_7b",
                kind="feature_projection",
            ).to(device).eval(),
            "boundary": load_radio_adaptor_from_checkpoint(
                args.radio_checkpoint,
                "sam3",
                kind="feature_projection",
            ).to(device).eval(),
        }
    for adaptor in adaptors.values():
        adaptor.requires_grad_(False)
    rng = random.Random(int(args.seed) + int(args.shard_index) * 100003)
    records, feature_rows, geometry_rows, masks, reliability_rows = [], [], [], [], []
    teacher_tokens, teacher_descriptors, teacher_masks, anchor_rows = [], [], [], []
    tensor_rows = {
        "radio_features": feature_rows,
        "geometry": geometry_rows,
        "token_mask": masks,
        "reliability": reliability_rows,
        "official_summary_tokens": teacher_tokens,
        "official_crop_summaries": teacher_descriptors,
        "teacher_mask": teacher_masks,
        "anchor_index": anchor_rows,
    }
    aligned_rows = (
        records,
        feature_rows,
        geometry_rows,
        masks,
        reliability_rows,
        teacher_tokens,
        teacher_descriptors,
        teacher_masks,
        anchor_rows,
    )
    failures = {}
    radii = contract.radii_m
    scene_intermediate_provenance_by_scene: dict[str, dict] = {}
    for scene_index, scene_name in enumerate(scenes):
        rng_state_before = rng.getstate()
        resumed = load_scene_partial(
            resume_dir,
            scene_index=scene_index,
            scene_name=scene_name,
            expected_rows=int(args.regions_per_scene),
            maximum_tokens=int(args.max_tokens),
            teacher_views=int(args.teacher_views),
            contract_record=resume_contract_record,
            contract_payload_sha256=resume_contract_payload_sha256,
        )
        if resumed is not None:
            if resumed["rng_state_before"] != encode_rng_state(
                rng_state_before
            ):
                raise RuntimeError(
                    "scene resume RNG predecessor differs from uninterrupted "
                    f"execution at {scene_index}:{scene_name}"
                )
            append_scene_rows(
                resumed["rows"],
                records=records,
                tensor_rows=tensor_rows,
            )
            rng.setstate(decode_rng_state(resumed["rng_state_after"]))
            continue
        scene_dir = root / scene_name
        scene_start = len(records)
        try:
            _verify_scene_input_contract(
                scene_input_by_name[scene_name]
            )
            frames = selected_frames[scene_name]
            if len(frames) < 2:
                raise RuntimeError("fewer than two valid RGB-D-pose frames")
            kd = torch.from_numpy(np.loadtxt(scene_dir / "intrinsics_depth.txt")).float()
            kc = torch.from_numpy(np.loadtxt(scene_dir / "intrinsics_color.txt")).float()
            frame_data = []
            scene_intermediate = None
            scene_intermediate_provenance = None
            expected_scene_intermediate_contract = (
                scene_intermediate_contracts.get(scene_name)
            )
            if intermediate_replay_root is not None:
                assert expected_scene_intermediate_contract is not None
                manifest_record = intermediate_manifest_by_scene[scene_name]
                (
                    scene_intermediate,
                    scene_intermediate_provenance,
                ) = _load_published_scene_intermediate(
                    intermediate_replay_root / scene_name,
                    root=intermediate_replay_root,
                    expected_contract=expected_scene_intermediate_contract,
                    expected_authority_sha256=str(
                        manifest_record["authority_sha256"]
                    ),
                    expected_manifest_record=manifest_record,
                )
            elif intermediate_output_root is not None:
                assert expected_scene_intermediate_contract is not None
                recovered = _recover_scene_intermediate_if_available(
                    intermediate_output_root,
                    scene_name=scene_name,
                    expected_contract=expected_scene_intermediate_contract,
                )
                if recovered is not None:
                    (
                        scene_intermediate,
                        scene_intermediate_provenance,
                    ) = recovered

            lifted_xyz, lifted_features, lifted_footprint = [], [], []
            for color_path, depth_path, pose_path in frames:
                color, depth = _load_color(color_path), _load_depth(depth_path)
                pose = torch.from_numpy(np.loadtxt(pose_path)).float()
                if scene_intermediate is None:
                    input_image = F.interpolate(
                        color[None],
                        (
                            int(args.radio_resolution),
                            int(args.radio_resolution),
                        ),
                        mode="bilinear",
                        align_corners=False,
                    ).to(device)
                    if runtime is None:
                        raise RuntimeError(
                            "fresh scene intermediate requires the RADIO runtime"
                        )
                    spatial = runtime.encode_training_pair(input_image)[0][0].cpu()
                    _thermal_pause(device, thermal_pacing_seconds)
                    xyz_observation, feat, footprint = _lift_observation(
                        depth,
                        kd,
                        kc,
                        pose,
                        spatial,
                        stride=int(args.depth_stride),
                        color_size=(
                            int(color.shape[2]),
                            int(color.shape[1]),
                        ),
                    )
                    lifted_xyz.append(xyz_observation)
                    lifted_features.append(feat)
                    lifted_footprint.append(footprint)
                frame_data.append((color_path, color, depth, pose))
            reliability_mode = str(
                getattr(
                    args,
                    "region_reliability_mode",
                    "geometric_mean_observation_agreement",
                )
            )
            if scene_intermediate is None:
                xyz, features, _footprint, _counts = _voxel_fuse(
                    torch.cat(lifted_xyz),
                    torch.cat(lifted_features),
                    torch.cat(lifted_footprint),
                    float(args.voxel_size),
                )
                geometric_reliability_all = None
                if intermediate_enabled or reliability_mode != "uniform_valid":
                    geometric_reliability_all = _voxel_reliability_v2(
                        torch.cat(lifted_xyz),
                        torch.cat(lifted_features),
                        float(args.voxel_size),
                        features,
                    )
                features = F.normalize(
                    features.float(),
                    dim=-1,
                    eps=1e-8,
                )
                projected = {}
                if set(adaptors) != {"appearance", "boundary"}:
                    raise RuntimeError(
                        "fresh scene graph requires both frozen RADIO adaptors"
                    )
                for name, adaptor in adaptors.items():
                    chunks = []
                    for start in range(
                        0,
                        len(features),
                        int(args.adaptor_batch_size),
                    ):
                        value = adaptor(
                            features[
                                start:start + int(args.adaptor_batch_size)
                            ].to(device)
                        )
                        chunks.append(
                            F.normalize(
                                value.float(),
                                dim=-1,
                                eps=1e-8,
                            ).cpu()
                        )
                    projected[name] = deterministic_feature_hash(
                        torch.cat(chunks),
                        int(args.affinity_dim),
                    )
                graph = contract.build_graph(
                    xyz,
                    appearance_features=projected["appearance"],
                    boundary_features=projected["boundary"],
                )
                if intermediate_output_root is not None:
                    assert geometric_reliability_all is not None
                    assert expected_scene_intermediate_contract is not None
                    fresh_scene_intermediate = SurfaceSceneIntermediate(
                        contract=expected_scene_intermediate_contract,
                        xyz=xyz,
                        radio_features=features,
                        geometric_reliability=geometric_reliability_all,
                        graph=graph,
                    )
                    (
                        scene_intermediate,
                        scene_intermediate_provenance,
                    ) = _publish_scene_intermediate(
                        intermediate_output_root,
                        fresh_scene_intermediate,
                    )
                    xyz = scene_intermediate.xyz
                    features = scene_intermediate.radio_features
                    geometric_reliability_all = (
                        scene_intermediate.geometric_reliability
                    )
                    graph = scene_intermediate.graph
            else:
                xyz = scene_intermediate.xyz
                features = scene_intermediate.radio_features
                geometric_reliability_all = (
                    scene_intermediate.geometric_reliability
                )
                graph = scene_intermediate.graph
            reliability_all = _candidate_reliability_from_geometric(
                geometric_reliability_all,
                mode=reliability_mode,
                num_rows=len(features),
            )
            if scene_intermediate_provenance is not None:
                scene_intermediate_provenance_by_scene[scene_name] = (
                    scene_intermediate_provenance
                )
            if str(args.scene_graph_output_root).strip():
                graph_root = Path(args.scene_graph_output_root)
                graph_root.mkdir(parents=True, exist_ok=True)
                graph_payload = {
                    "schema_version": 1,
                    "scene": scene_name,
                    "xyz": xyz,
                    "edge_index": graph.edge_index,
                    "edge_weight": graph.edge_weight,
                    "raw_affinity": graph.raw_affinity,
                    "local_sigma": graph.local_sigma,
                    "edge_channels": graph.edge_channels,
                    "depth_intrinsic": kd,
                    "color_intrinsic": kc,
                    "frames": [
                        {
                            "color": str(color_path.resolve()),
                            "depth": str(depth_path.resolve()),
                            "pose": pose,
                        }
                        for (color_path, _color, _depth, pose),
                            (_color_path, depth_path, _pose_path)
                        in zip(frame_data, frames)
                    ],
                    "metadata": {
                        "source": "ScanNet_frames_25k_query_free",
                        "labels_opened": False, "instances_opened": False,
                        "masks_opened": False, "text_opened": False,
                        "region_contract": contract.to_dict(),
                        "region_contract_sha256": contract.digest,
                        "teacher_region_semantics": FIXED_CORE_TEACHER_SEMANTICS,
                        "teacher_region_contract": (
                            teacher_contract.to_dict()
                        ),
                        "teacher_region_contract_sha256": (
                            teacher_contract.digest
                        ),
                        "teacher_target_protocol": (
                            teacher_target_protocol
                        ),
                        "teacher_target_protocol_sha256": (
                            teacher_target_protocol_sha256
                        ),
                    },
                }
                torch.save(graph_payload, graph_root / f"{scene_name}.pt")
            prepared_graph = contract.prepare_graph(graph, xyz)
            if replay_payload is None:
                candidates = list(range(len(xyz)))
                rng.shuffle(candidates)
                candidate_specs = [
                    (int(seed), None, None)
                    for seed in candidates
                ]
            else:
                candidate_specs = [
                    (
                        int(record["seed"]),
                        int(source_row),
                        record,
                    )
                    for source_row, record in replay_by_scene[scene_name]
                ]
            scene_regions = 0
            for seed, replay_row, replay_record in candidate_specs:
                if (
                    replay_payload is None
                    and scene_regions >= int(args.regions_per_scene)
                ):
                    break
                radius = (
                    float(replay_record["physical_radius_m"])
                    if replay_record is not None
                    else radii[scene_regions % len(radii)]
                )
                rows, core, _geodesic = contract.expand(
                    graph, xyz, seed, radius, prepared_graph=prepared_graph
                )
                idx = rows
                teacher_idx, teacher_core, _teacher_geodesic = (
                    teacher_contract.expand(
                        graph,
                        xyz,
                        seed,
                        radius,
                        include_context=False,
                        prepared_graph=prepared_graph,
                    )
                )
                teacher_region_saturated = (
                    len(teacher_idx)
                    >= int(args.teacher_region_candidate_limit)
                )
                if teacher_region_saturated:
                    raise RuntimeError(
                        "teacher region hit its Dijkstra candidate limit; "
                        "increase --teacher-region-candidate-limit"
                    )
                if not bool(teacher_core.all()):
                    raise RuntimeError(
                        "core-only teacher expansion admitted context tokens"
                    )
                teacher_support_sha256 = _teacher_support_sha256(
                    xyz,
                    teacher_idx,
                )
                crops, views = [], []
                for color_path, color, depth, pose in frame_data:
                    box = _project_region_box(
                        xyz[teacher_idx], depth, kd, kc, pose,
                        (int(color.shape[2]), int(color.shape[1])),
                        min_visible=min(
                            int(args.min_visible_tokens),
                            len(teacher_idx),
                        ),
                        context_pad=0.0,
                    )
                    if box is None:
                        continue
                    top, left, bottom, right = box
                    if replay_payload is None:
                        crops.append(F.interpolate(
                            color[:, top:bottom, left:right][None],
                            (
                                int(args.radio_resolution),
                                int(args.radio_resolution),
                            ),
                            mode="bilinear",
                            align_corners=False,
                        )[0])
                    views.append({"frame": color_path.name, "crop_box_tlbr": box})
                    if len(views) >= int(args.teacher_views):
                        break
                if len(views) < 2:
                    continue
                region_id = _surface_region_id(
                    scene_name,
                    seed,
                    radius,
                    teacher_contract.digest,
                    teacher_support_sha256,
                )
                if replay_payload is None:
                    if runtime is None:
                        raise RuntimeError(
                            "fresh teacher targets require the RADIO runtime"
                        )
                    _, tokens, descriptors = runtime.encode_training_pair(
                        torch.stack(crops).to(device)
                    )
                    _thermal_pause(
                        device,
                        thermal_pacing_seconds,
                        image_count=len(crops),
                    )
                    view_count = len(crops)
                    max_views = int(args.teacher_views)
                    padded_tokens = torch.zeros(
                        max_views, 1280, dtype=torch.float16
                    )
                    padded_descriptors = torch.zeros(
                        max_views,
                        descriptors.shape[1],
                        dtype=torch.float16,
                    )
                    padded_teacher_mask = torch.zeros(
                        max_views, dtype=torch.bool
                    )
                    padded_tokens[:view_count] = tokens.half().cpu()
                    padded_descriptors[:view_count] = (
                        descriptors.half().cpu()
                    )
                    padded_teacher_mask[:view_count] = True
                    teacher_medoid = _teacher_medoid(tokens, descriptors)
                else:
                    assert replay_record is not None and replay_row is not None
                    source_views = replay_record.get("teacher_views", [])
                    if (
                        json.dumps(views, sort_keys=True)
                        != json.dumps(source_views, sort_keys=True)
                        or int(replay_record["teacher_region_tokens"])
                        != len(teacher_idx)
                        or replay_record.get("teacher_region_saturated")
                        is not False
                        or str(
                            replay_record.get(
                                "teacher_support_sha256",
                                "",
                            )
                        )
                        != teacher_support_sha256
                        or str(replay_record["region_id"]) != region_id
                    ):
                        raise RuntimeError(
                            "recomputed fixed-core teacher geometry differs "
                            "from the replay cache"
                        )
                    padded_tokens = torch.as_tensor(
                        replay_payload["official_summary_tokens"][replay_row]
                    ).clone()
                    padded_descriptors = torch.as_tensor(
                        replay_payload["official_crop_summaries"][replay_row]
                    ).clone()
                    padded_teacher_mask = torch.as_tensor(
                        replay_payload["teacher_mask"][replay_row]
                    ).bool().clone()
                    max_views = int(args.teacher_views)
                    if (
                        padded_tokens.dtype != torch.float16
                        or padded_descriptors.dtype != torch.float16
                        or padded_tokens.shape != (max_views, 1280)
                        or padded_descriptors.ndim != 2
                        or padded_descriptors.shape[0] != max_views
                        or padded_teacher_mask.shape != (max_views,)
                    ):
                        raise RuntimeError(
                            "teacher replay tensors have incompatible shapes "
                            "or dtypes"
                        )
                    view_count = int(padded_teacher_mask.sum())
                    expected_mask = (
                        torch.arange(max_views) < view_count
                    )
                    if (
                        view_count != len(views)
                        or view_count < 2
                        or not torch.equal(
                            padded_teacher_mask.cpu(),
                            expected_mask,
                        )
                    ):
                        raise RuntimeError(
                            "teacher replay mask differs from recorded views"
                        )
                    teacher_medoid = int(replay_record["teacher_medoid"])
                    if not 0 <= teacher_medoid < view_count:
                        raise RuntimeError(
                            "teacher replay medoid is outside valid views"
                        )
                padded_features = torch.zeros(int(args.max_tokens), 1280, dtype=torch.float16)
                padded_geometry = torch.zeros(int(args.max_tokens), 14, dtype=torch.float16)
                padded_mask = torch.zeros(int(args.max_tokens), dtype=torch.bool)
                padded_reliability = torch.zeros(int(args.max_tokens), 1, dtype=torch.float16)
                n = len(idx)
                rel = reliability_all[idx, None]
                scale = graph.local_sigma[idx, None].expand(-1, 3).clamp_min(1e-4)
                anchor_local = int(torch.where(idx == int(seed))[0][0])
                geom = surface_region_geometry_v2(
                    xyz[idx], scale, rel, float(radius), anchor_index=anchor_local,
                    core_mask=core,
                )
                padded_features[:n] = features[idx].half()
                padded_geometry[:n] = geom.half()
                padded_mask[:n] = True; padded_reliability[:n] = rel.half()
                feature_rows.append(padded_features); geometry_rows.append(padded_geometry)
                masks.append(padded_mask); reliability_rows.append(padded_reliability)
                teacher_tokens.append(padded_tokens); teacher_descriptors.append(padded_descriptors)
                teacher_masks.append(padded_teacher_mask)
                teacher_target_sha256 = _teacher_target_sha256(
                    padded_tokens,
                    padded_descriptors,
                    padded_teacher_mask,
                )
                record = {
                    "region_id": region_id,
                    "scene": scene_name, "seed": int(seed), "physical_radius_m": radius,
                    "tokens": n, "teacher_views": views,
                    "teacher_medoid": teacher_medoid,
                    "teacher_region_tokens": int(len(teacher_idx)),
                    "teacher_support_sha256": (
                        teacher_support_sha256
                    ),
                    "teacher_region_saturated": False,
                    "teacher_target_source": teacher_target_source,
                    "teacher_target_sha256": teacher_target_sha256,
                    "anchor_local_index": anchor_local,
                    "core_tokens": int(core.sum()),
                    "below_nominal_minimum": bool(len(idx) < contract.minimum_tokens),
                }
                if replay_row is not None:
                    record["teacher_replay_source_row"] = int(replay_row)
                    record["teacher_replay_source_region_id"] = str(
                        replay_record["region_id"]
                    )
                if scene_intermediate_provenance is not None:
                    record["scene_intermediate_contract_sha256"] = str(
                        scene_intermediate_provenance["contract_sha256"]
                    )
                    record[
                        "scene_intermediate_tensor_bundle_sha256"
                    ] = str(
                        scene_intermediate_provenance[
                            "tensor_bundle_sha256"
                        ]
                    )
                records.append(record)
                anchor_rows.append(
                    torch.tensor(anchor_local, dtype=torch.long)
                )
                scene_regions += 1
            if scene_regions != int(args.regions_per_scene):
                raise RuntimeError(
                    "incomplete multi-view surface-region set: "
                    f"{scene_regions}/{int(args.regions_per_scene)}"
                )
            _verify_scene_input_contract(
                scene_input_by_name[scene_name]
            )
        except Exception as error:
            for collection in aligned_rows:
                del collection[scene_start:]
            failures[scene_name] = f"{type(error).__name__}: {error}"
            continue
        scene_rows = _scene_rows_from_accumulators(
            scene_start=scene_start,
            records=records,
            tensor_rows=tensor_rows,
        )
        commit_scene_partial(
            resume_dir,
            scene_index=scene_index,
            scene_name=scene_name,
            scene_rows=scene_rows,
            rng_state_before=rng_state_before,
            rng_state_after=rng.getstate(),
            expected_rows=int(args.regions_per_scene),
            maximum_tokens=int(args.max_tokens),
            teacher_views=int(args.teacher_views),
            contract_record=resume_contract_record,
            contract_payload_sha256=resume_contract_payload_sha256,
        )
    if replay_payload is not None and failures:
        raise RuntimeError(
            "teacher replay must be exact for every scene; failures: "
            f"{failures}"
        )
    if not records:
        raise RuntimeError(f"all scenes failed: {failures}")
    if intermediate_output_root is not None:
        intermediate_run_provenance = (
            _publish_scene_intermediate_manifest(
                intermediate_output_root,
                scenes=scenes,
                contracts=scene_intermediate_contracts,
                run_contract=intermediate_run_contract,
            )
        )
    elif intermediate_replay_root is not None:
        for scene_name in scenes:
            manifest_record = intermediate_manifest_by_scene[scene_name]
            _value, provenance = _load_published_scene_intermediate(
                intermediate_replay_root / scene_name,
                root=intermediate_replay_root,
                expected_contract=scene_intermediate_contracts[scene_name],
                expected_authority_sha256=str(
                    manifest_record["authority_sha256"]
                ),
                expected_manifest_record=manifest_record,
            )
            scene_intermediate_provenance_by_scene[scene_name] = provenance
    metadata = {
        "schema_version": 3, "training_scope": "global_cross_scene_3d_surface_v2",
        "dataset_id": "ScanNet_frames_25k_query_free",
        "dataset_root": dataset_root,
        "split_role": split_role,
        "split_file": str(split_file.resolve()),
        "split_file_sha256": split_file_sha256,
        "builder_script_sha256": builder_script_sha256,
        "uses_benchmark_test_vocabulary": False, "uses_benchmark_scenes": False,
        "annotations_opened": False, "labels_opened": False, "instances_opened": False,
        "masks_opened": False, "text_opened": False,
        "region_construction": "shared_surface_region_contract_v2",
        "region_contract": contract.to_dict(),
        "region_contract_version": contract.version,
        "region_contract_sha256": contract.digest,
        "teacher_region_semantics": FIXED_CORE_TEACHER_SEMANTICS,
        "teacher_region_contract": teacher_contract.to_dict(),
        "teacher_region_contract_sha256": teacher_contract.digest,
        "teacher_target_schema_version": 1,
        "teacher_crop_protocol": TEACHER_CROP_PROTOCOL,
        "teacher_target_protocol": teacher_target_protocol,
        "teacher_target_protocol_sha256": (
            teacher_target_protocol_sha256
        ),
        "teacher_target_source": teacher_target_source,
        "teacher_replay_cache": replay_provenance,
        "teacher_replay_authority": replay_authority_provenance,
        "teacher_regions_saturated": sum(
            bool(record["teacher_region_saturated"])
            for record in records
        ),
        "regions_per_scene_requested": int(args.regions_per_scene),
        "teacher_views_requested": int(args.teacher_views),
        "complete_scene_regions": True,
        "radio_version": runtime_version,
        "radio_checkpoint_sha256": runtime_checkpoint_sha256,
        "execution_radio_thermal_pacing_seconds_per_image": (
            thermal_pacing_seconds
        ),
        "durable_scene_resume": {
            "schema_version": RESUME_SCHEMA_VERSION,
            "contract": resume_contract_record,
            "contract_payload_sha256": (
                resume_contract_payload_sha256
            ),
            "partial_suffix": SCENE_PARTIAL_SUFFIX,
            "terminal_suffix": SCENE_TERMINAL_SUFFIX,
            "merge_order": (
                "selected_scene_order_then_region_row_order_v1"
            ),
        },
        "scene_names": sorted({record["scene"] for record in records}),
        "scene_region_counts": {
            scene: sum(record["scene"] == scene for record in records)
            for scene in sorted({record["scene"] for record in records})
        },
        "regions_below_nominal_minimum": sum(
            bool(record["below_nominal_minimum"]) for record in records
        ),
        "region_records": records, "failed_scenes": failures,
        "forbidden_eval_scenes": sorted(FORBIDDEN_EVAL_SCENES),
        "excluded_physical_spaces": sorted(excluded_spaces),
        "exclusion_files": exclusion_files,
        "physical_space_disjoint": True,
    }
    if intermediate_enabled:
        metadata["scene_intermediate"] = intermediate_run_provenance
    payload = {
        "radio_features": torch.stack(feature_rows),
        "geometry": torch.stack(geometry_rows), "token_mask": torch.stack(masks),
        "reliability": torch.stack(reliability_rows),
        "official_summary_tokens": torch.stack(teacher_tokens),
        "official_crop_summaries": torch.stack(teacher_descriptors),
        "teacher_mask": torch.stack(teacher_masks), "metadata": metadata,
        "anchor_index": torch.stack(anchor_rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if bool(getattr(args, "overwrite_output", False)):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        write_torch_noclobber(output, payload)
    output_record = file_record(output)
    report = {
        "output": output_record["path"], "regions": len(records),
        "scenes": len(metadata["scene_names"]), "failed_scenes": failures,
        "split_role": split_role, "split_file_sha256": metadata["split_file_sha256"],
        "teacher_target_source": teacher_target_source,
        "teacher_replay_cache": replay_provenance,
        "teacher_replay_authority": replay_authority_provenance,
    }
    if intermediate_enabled:
        report["scene_intermediate"] = intermediate_run_provenance
    sidecar = output.with_suffix(output.suffix + ".json")
    if bool(getattr(args, "overwrite_output", False)) and os.path.lexists(
        sidecar
    ):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            dir=sidecar.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, sidecar)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        write_frozen_json(sidecar, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split-role", choices=("train", "validation"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume-dir",
        default="",
        help=(
            "Dedicated durable per-scene resume directory. Scene partials "
            "are immutable, SHA-terminal-bound, and never use a .pt suffix."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Atomically replace an explicitly targeted existing cache.",
    )
    parser.add_argument(
        "--scene-intermediate-output-root",
        default="",
        help=(
            "Dedicated control-only root for atomically published, "
            "candidate-independent per-scene RADIO/geometry/graph artifacts."
        ),
    )
    parser.add_argument(
        "--scene-intermediate-manifest",
        default="",
        help=(
            "Exact control manifest.json used to replay per-scene "
            "intermediates without RADIO/adaptor scene inference."
        ),
    )
    parser.add_argument(
        "--scene-intermediate-manifest-sha256",
        default="",
        help=(
            "Required external SHA-256 authority for "
            "--scene-intermediate-manifest."
        ),
    )
    parser.add_argument("--max-scenes", type=int, default=16)
    parser.add_argument("--scene-names", default="")
    parser.add_argument(
        "--exclude-scene-files",
        default="",
        help=(
            "Comma/space-separated benchmark scene-list files. Every rescan "
            "of each listed sceneXXXX physical space is excluded."
        ),
    )
    parser.add_argument(
        "--exclude-scene-names",
        default="",
        help="Additional comma/space-separated scenes or sceneXXXX spaces to exclude.",
    )
    parser.add_argument("--frames-per-scene", type=int, default=8)
    parser.add_argument("--regions-per-scene", type=int, default=12)
    parser.add_argument("--region-radii", default="0.25,0.45,0.70")
    parser.add_argument("--context-ratio", type=float, default=1.20)
    parser.add_argument("--graph-neighbors", type=int, default=16)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--depth-stride", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--token-subsampling",
        choices=("nearest_geodesic_then_node_index", "core_context_radial_stratified_v1"),
        default="core_context_radial_stratified_v1",
        help=(
            "Declared fixed-budget region-token selection policy. New caches "
            "preserve a core/context quota; frozen legacy contracts remain loadable."
        ),
    )
    parser.add_argument(
        "--token-candidate-limit", type=int, default=1024,
        help=(
            "Maximum settled Dijkstra candidates before deterministic token "
            "selection; must exceed max-tokens to expose the context shell."
        ),
    )
    parser.add_argument(
        "--core-token-fraction", type=float, default=0.60,
        help="Core quota for core/context stratified sampling.",
    )
    parser.add_argument(
        "--region-reliability-mode",
        choices=(
            "geometric_mean_observation_agreement",
            "uniform_valid",
        ),
        default="uniform_valid",
        help=(
            "New caches default to a matched train/inference abstention contract "
            "until canonical MPR stores real multiview agreement."
        ),
    )
    parser.add_argument(
        "--path-cost-mode",
        choices=("euclidean", "appearance_boundary_geometric"),
        default="euclidean",
        help="Query-free shortest-path cost; relation weighting is manifest-locked.",
    )
    parser.add_argument("--path-affinity-floor", type=float, default=1e-4)
    parser.add_argument("--min-visible-tokens", type=int, default=12)
    parser.add_argument("--teacher-views", type=int, default=3)
    parser.add_argument(
        "--teacher-region-candidate-limit",
        type=int,
        default=4096,
        help=(
            "Fixed core-only Dijkstra budget used to define teacher crops "
            "independently of the input token-sampling ablation."
        ),
    )
    parser.add_argument(
        "--teacher-replay-cache",
        default="",
        help=(
            "Fresh fixed-core cache for the identical shard. When set, seed, "
            "radius, teacher support/views and official target tensors are "
            "validated and replayed exactly; any mismatch aborts the build."
        ),
    )
    parser.add_argument(
        "--teacher-replay-authority",
        default="",
        help=(
            "Narrow external authority permitting one byte-exact historical "
            "cache whose recorded builder SHA differs from this builder. "
            "All teacher/scenes/split/checkpoint fields remain fail-closed."
        ),
    )
    parser.add_argument(
        "--teacher-replay-authority-sha256",
        default="",
        help=(
            "Required external SHA-256 for --teacher-replay-authority. "
            "Supplying an authority for a current-builder cache is rejected."
        ),
    )
    parser.add_argument("--adaptor-batch-size", type=int, default=4096)
    parser.add_argument("--affinity-dim", type=int, default=256)
    parser.add_argument("--radio-resolution", type=int, default=384)
    parser.add_argument(
        "--radio-thermal-pacing-seconds-per-image",
        type=float,
        default=0.0,
        help=(
            "Execution-only cooling pause per image after each RADIO encode; "
            "does not change samples or feature arithmetic."
        ),
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--scene-graph-output-root", default="",
        help="Optional query-free per-scene graph export for relation calibration",
    )
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--radio-checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
