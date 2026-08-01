#!/usr/bin/env python3
"""Freeze an audited three-seed SurfaceRegion query-free selection on CPU.

This companion deliberately does not run a benchmark or choose a fortunate
single seed.  It revalidates the completed fixed-teacher screen, recomputes the
frozen query-free rule, and binds the selected candidate's complete seed set.
The resulting bundle keeps the text-response/benchmark gate explicitly closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2
from radio_gs.scripts.build_scannet_surface_region_cache import (
    FIXED_CORE_TEACHER_SEMANTICS,
    FORBIDDEN_EVAL_SCENES,
    TEACHER_CROP_PROTOCOL,
    _physical_space,
    _read_scene_file,
    _surface_region_id,
    _teacher_region_contract,
    _teacher_target_sha256,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_surface_region_summary_readout_v2,
    load_torch_mapping,
    sha256_file,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_region_query_free_three_seed_bundle"
COMPLETION_ARTIFACT_TYPE = "surface_region_query_free_promotion_completion"
SCREEN_NAME = "surface-region-fixed-teacher-replay-v2"
CONTROL = "control_c256_geometric"
REQUIRED_SEEDS = (0, 1, 2)
EXPECTED_CANDIDATES = (
    CONTROL,
    "context_c1024_geometric",
    "context_c1024_uniform",
    "core_c1024_geometric",
)
METRIC_KEYS = (
    "summary_token_cosine",
    "mean_descriptor_cosine",
    "all_view_descriptor_cosine",
)
QUERY_FREE_FLAGS = (
    "uses_benchmark_scenes",
    "uses_benchmark_test_vocabulary",
    "annotations_opened",
    "labels_opened",
    "instances_opened",
    "masks_opened",
    "text_opened",
)
IMPLEMENTATION_SOURCES = (
    "radio_gs/scripts/build_scannet_surface_region_cache.py",
    "radio_gs/scripts/train_surface_region_summary_readout.py",
    "radio_gs/interfaces/surface_region_contract.py",
    "radio_gs/interfaces/surface_region_summary.py",
    "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
)
SCANNET_SCENE_PATTERN = re.compile(r"^scene\d{4}_\d{2}$")
SHARD_SCENE_LIMIT = 100
SUMMARY_TOKEN_DIM = 1280
SIGLIP2_DESCRIPTOR_DIM = 1536
VALIDATION_RECOMPUTE_TOLERANCE = 5e-5
EVALUATION_KEYS = (
    "radio_features",
    "geometry",
    "token_mask",
    "reliability",
    "official_summary_tokens",
    "official_crop_summaries",
    "teacher_mask",
    "anchor_index",
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _json_object(path: Path) -> dict:
    value, _, _ = load_json_object(path, label="Surface promotion JSON artifact")
    return value


def _torch_mapping(path: Path) -> Mapping:
    value, _, _ = load_torch_mapping(
        path,
        map_location="cpu",
        label="Surface promotion torch artifact",
    )
    return value


def _resolved_file(raw: object, *, label: str) -> Path:
    path = Path(str(raw)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"bound {label} is missing: {path}")
    return path


def _check_file_binding(raw: object, expected: object, *, label: str) -> Path:
    path = _resolved_file(raw, label=label)
    _require(_is_sha256(expected), f"{label} lacks a valid SHA256")
    _require(_sha256(path) == str(expected), f"{label} SHA256 mismatch")
    return path


def _finite_number(value: object, *, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _close(left: object, right: object, *, tolerance: float = 1e-7) -> bool:
    return math.isclose(
        _finite_number(left, label="left comparison value"),
        _finite_number(right, label="right comparison value"),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def _scene_name(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a scene name")
    _require(
        SCANNET_SCENE_PATTERN.fullmatch(value) is not None,
        f"{label} has invalid ScanNet scene syntax: {value!r}",
    )
    return value


def _normalized_exclusion_records(value: object, *, label: str) -> list[dict[str, str]]:
    _require(isinstance(value, list), f"{label} must be a list")
    records: dict[str, str] = {}
    for row in value:
        _require(isinstance(row, Mapping), f"{label} contains a non-object row")
        path = str(Path(str(row.get("path", ""))).resolve())
        digest = row.get("sha256")
        _require(_is_sha256(digest), f"{label} contains an invalid SHA256")
        _require(path not in records, f"{label} contains a duplicate path")
        records[path] = str(digest)
    return [
        {"path": path, "sha256": digest}
        for path, digest in sorted(records.items())
    ]


def _derive_manifest_semantics(manifest: Mapping) -> dict:
    """Recompute scene shards and exclusions from the manifest-bound inputs."""

    explicit = manifest.get("excluded_scene_names")
    _require(
        isinstance(explicit, list) and explicit,
        "manifest lacks explicit excluded scene names",
    )
    explicit_names = [
        _scene_name(value, label="explicit excluded scene") for value in explicit
    ]
    _require(
        explicit_names == sorted(set(explicit_names)),
        "manifest explicit excluded scene names are not sorted and unique",
    )
    _require(
        FORBIDDEN_EVAL_SCENES.issubset(set(explicit_names)),
        "manifest does not explicitly exclude every forbidden evaluation scene",
    )

    exclusions = manifest["exclusion_files"]
    excluded_names = list(explicit_names)
    exclusion_records = []
    for raw_path, digest in exclusions.items():
        path = Path(str(raw_path)).resolve()
        # The binding itself was checked before this semantic pass.  Parse the
        # exact builder format rather than trusting an asserted physical-space list.
        excluded_names.extend(
            _scene_name(value, label=f"excluded scene from {path}")
            for value in _read_scene_file(path)
        )
        exclusion_records.append({"path": str(path), "sha256": str(digest)})
    excluded_spaces = sorted({_physical_space(name) for name in excluded_names})
    exclusion_records = _normalized_exclusion_records(
        exclusion_records,
        label="manifest exclusion records",
    )

    dataset_root = Path(str(manifest["dataset_root"])).resolve()
    cache_contract = manifest["cache_contract"]
    role_scenes: dict[str, list[str]] = {}
    shards: dict[str, dict[int, list[str]]] = {}
    for role in ("train", "validation"):
        split_path = Path(str(manifest[f"{role}_split"])).resolve()
        raw_names = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        names = [_scene_name(value, label=f"{role} split scene") for value in raw_names]
        _require(
            len(names) == len(set(names)),
            f"{role} split contains duplicate scenes",
        )
        names = [name for name in names if name not in FORBIDDEN_EVAL_SCENES]
        names = [name for name in names if _physical_space(name) not in excluded_spaces]
        missing = [name for name in names if not (dataset_root / name).is_dir()]
        _require(not missing, f"{role} split scenes are missing from the dataset: {missing}")
        _require(bool(names), f"{role} split has no eligible scenes")
        role_scenes[role] = names
        shard_count = int(cache_contract[f"{role}_shards"])
        shards[role] = {
            shard: names[shard::shard_count][:SHARD_SCENE_LIMIT]
            for shard in range(shard_count)
        }
        _require(
            all(shards[role][shard] for shard in range(shard_count)),
            f"{role} split leaves an empty cache shard",
        )
        flattened = [
            scene
            for shard in range(shard_count)
            for scene in shards[role][shard]
        ]
        _require(
            set(flattened) == set(names) and len(flattened) == len(names),
            f"{role} shard derivation does not cover the split exactly",
        )

    train_spaces = {_physical_space(scene) for scene in role_scenes["train"]}
    validation_spaces = {
        _physical_space(scene) for scene in role_scenes["validation"]
    }
    _require(
        not train_spaces & validation_spaces,
        "train/validation splits share a physical ScanNet space",
    )
    _require(
        not (train_spaces | validation_spaces) & set(excluded_spaces),
        "derived split scenes overlap excluded physical spaces",
    )
    return {
        "excluded_physical_spaces": excluded_spaces,
        "exclusion_files": exclusion_records,
        "role_scenes": role_scenes,
        "shards": shards,
    }


def _contract_from_metadata(metadata: Mapping, key: str) -> SurfaceRegionContractV2:
    raw = metadata.get(key)
    _require(isinstance(raw, Mapping), f"cache lacks {key}")
    specification = dict(raw)
    specification["radii_m"] = tuple(specification["radii_m"])
    specification.setdefault(
        "token_candidate_limit",
        int(specification["maximum_tokens"]),
    )
    try:
        return SurfaceRegionContractV2(**specification)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cache has an invalid {key}") from error


def _validate_manifest(output_root: Path) -> tuple[Path, dict, Path, dict]:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = output_root / "run_manifest.json"
    manifest = _json_object(manifest_path)
    _require(manifest.get("schema_version") == 1, "unexpected run-manifest schema")
    _require(manifest.get("screen") == SCREEN_NAME, "unexpected SurfaceRegion screen")
    expected_candidates = {
        CONTROL: {
            "context_ratio": 1.20,
            "token_candidate_limit": 256,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "fresh_official_runtime",
        },
        "context_c1024_geometric": {
            "context_ratio": 1.20,
            "token_candidate_limit": 1024,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "exact_cache_replay",
        },
        "context_c1024_uniform": {
            "context_ratio": 1.20,
            "token_candidate_limit": 1024,
            "reliability": "uniform_valid",
            "teacher_source": "exact_cache_replay",
        },
        "core_c1024_geometric": {
            "context_ratio": 1.00,
            "token_candidate_limit": 1024,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "exact_cache_replay",
        },
    }
    _require(
        manifest.get("candidates") == expected_candidates,
        "run manifest has an unexpected ordered candidate contract",
    )
    expected_readout = {
        "hidden_dim": 256,
        "epochs": 60,
        "patience": 10,
        "batch_size": 16,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "token_weight": 0.25,
        "relation_weight": 0.1,
        "reliability_attention_mode": "log_prior",
        "seeds": list(REQUIRED_SEEDS),
    }
    _require(
        manifest.get("readout_contract") == expected_readout,
        "run manifest has an unexpected readout contract",
    )
    _require(
        manifest.get("selection_contract")
        == {
            "minimum_mean_score_gain": 0.001,
            "minimum_seed_wins": 2,
            "maximum_component_drop": 0.002,
            "uses_benchmark_queries": False,
        },
        "run manifest has an unexpected query-free selection contract",
    )
    _require(
        manifest.get("cache_contract", {}).get("train_shards") == 4
        and manifest.get("cache_contract", {}).get("validation_shards") == 2,
        "run manifest has unexpected shard counts",
    )
    cache_contract = manifest["cache_contract"]
    expected_cache_values = {
        "frames_per_scene": 8,
        "regions_per_scene": 12,
        "region_radii_m": [0.25, 0.45, 0.70],
        "maximum_tokens": 256,
        "teacher_region_candidate_limit": 4096,
        "path_cost_mode": "appearance_boundary_geometric",
        "path_affinity_floor": 1e-4,
        "token_subsampling": "core_context_radial_stratified_v1",
        "core_token_fraction": 0.60,
        "teacher_views": 3,
        "seed": 0,
    }
    _require(
        all(cache_contract.get(key) == value for key, value in expected_cache_values.items()),
        "run manifest has an unexpected semantic cache contract",
    )
    _require(
        isinstance(cache_contract.get("adaptor_batch_size"), int)
        and cache_contract["adaptor_batch_size"] > 0
        and _finite_number(
            cache_contract.get("radio_thermal_pacing_seconds_per_image"),
            label="cache thermal pacing",
        )
        >= 0,
        "run manifest has invalid execution-only cache settings",
    )

    runner = repo_root / "radio_gs/scripts/run_surface_region_context_recovery_screen.sh"
    _require(runner.is_file(), "frozen SurfaceRegion runner is missing")
    _require(
        _sha256(runner) == manifest.get("runner_sha256"),
        "SurfaceRegion runner differs from the run manifest",
    )
    implementations = manifest.get("implementation_sources")
    _require(
        isinstance(implementations, Mapping)
        and set(implementations) == set(IMPLEMENTATION_SOURCES),
        "run manifest implementation source set differs",
    )
    for relative in IMPLEMENTATION_SOURCES:
        path = repo_root / relative
        _require(path.is_file(), f"bound implementation source is missing: {relative}")
        _require(
            _sha256(path) == implementations[relative],
            f"bound implementation source changed: {relative}",
        )

    for name in ("train", "validation"):
        _check_file_binding(
            manifest.get(f"{name}_split"),
            manifest.get(f"{name}_split_sha256"),
            label=f"{name} split",
        )
    _check_file_binding(
        manifest.get("radio_checkpoint"),
        manifest.get("radio_checkpoint_sha256"),
        label="RADIO checkpoint",
    )
    exclusions = manifest.get("exclusion_files")
    _require(isinstance(exclusions, Mapping) and exclusions, "manifest lacks exclusions")
    for raw_path, digest in exclusions.items():
        _check_file_binding(raw_path, digest, label="benchmark exclusion file")
    _require(
        Path(str(manifest.get("dataset_root", ""))).resolve().is_dir(),
        "dataset root is missing",
    )
    _require(
        Path(str(manifest.get("radio_repo", ""))).resolve().is_dir(),
        "RADIO repository is missing",
    )
    thermal = manifest.get("thermal_safety_contract", {})
    guard = _check_file_binding(
        thermal.get("guard"), thermal.get("guard_sha256"), label="thermal guard"
    )
    _require(
        guard == repo_root / "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
        "manifest binds an unexpected thermal guard",
    )
    semantics = _derive_manifest_semantics(manifest)
    return manifest_path, manifest, repo_root, semantics


def _record_teacher_identity(record: Mapping) -> dict:
    keys = (
        "region_id",
        "scene",
        "seed",
        "physical_radius_m",
        "teacher_views",
        "teacher_medoid",
        "teacher_region_tokens",
        "teacher_support_sha256",
        "teacher_region_saturated",
        "teacher_target_sha256",
    )
    _require(all(key in record for key in keys), "cache region record is incomplete")
    return {key: record[key] for key in keys}


def _validate_teacher_view(value: object, *, label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(
        isinstance(value.get("frame"), str) and bool(value["frame"]),
        f"{label} lacks a frame identity",
    )
    box = value.get("crop_box_tlbr")
    _require(
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(item, int) and not isinstance(item, bool) for item in box),
        f"{label} has an invalid crop box",
    )
    top, left, bottom, right = (int(item) for item in box)
    _require(
        top >= 0 and left >= 0 and bottom > top and right > left,
        f"{label} has a degenerate crop box",
    )


def _validate_cache_payload(
    payload: Mapping,
    *,
    path: Path,
    role: str,
    shard: int,
    candidate: str,
    specification: Mapping,
    manifest: Mapping,
    manifest_semantics: Mapping,
    builder_sha256: str,
) -> dict:
    metadata = payload.get("metadata")
    _require(isinstance(metadata, Mapping), f"{path} lacks metadata")
    _require(
        metadata.get("schema_version") == 3 and metadata.get("split_role") == role,
        f"{path} has a wrong cache schema/split",
    )
    _require(
        metadata.get("training_scope") == "global_cross_scene_3d_surface_v2"
        and metadata.get("dataset_id") == "ScanNet_frames_25k_query_free"
        and metadata.get("region_construction") == "shared_surface_region_contract_v2",
        f"{path} has unexpected cache semantics",
    )
    for flag in QUERY_FREE_FLAGS:
        _require(metadata.get(flag) is False, f"{path} does not certify {flag}=false")
    _require(metadata.get("physical_space_disjoint") is True, f"{path} is not disjoint")
    _require(not metadata.get("failed_scenes"), f"{path} contains failed scenes")
    _require(metadata.get("complete_scene_regions") is True, f"{path} is incomplete")
    _require(metadata.get("teacher_regions_saturated") == 0, f"{path} saturated teacher")
    _require(
        metadata.get("builder_script_sha256") == builder_sha256,
        f"{path} builder provenance differs",
    )
    _require(
        metadata.get("dataset_root") == str(Path(str(manifest["dataset_root"])).resolve()),
        f"{path} dataset root differs",
    )
    _require(
        metadata.get("split_file")
        == str(Path(str(manifest[f"{role}_split"])).resolve())
        and metadata.get("split_file_sha256") == manifest[f"{role}_split_sha256"],
        f"{path} split hash differs",
    )
    expected_scenes = sorted(manifest_semantics["shards"][role][int(shard)])
    _require(
        metadata.get("scene_names") == expected_scenes,
        f"{path} scene set differs from the manifest-bound split shard",
    )
    _require(
        metadata.get("forbidden_eval_scenes") == sorted(FORBIDDEN_EVAL_SCENES)
        and metadata.get("excluded_physical_spaces")
        == manifest_semantics["excluded_physical_spaces"]
        and _normalized_exclusion_records(
            metadata.get("exclusion_files"),
            label=f"{path} exclusion records",
        )
        == manifest_semantics["exclusion_files"],
        f"{path} exclusion semantics differ from manifest-bound inputs",
    )
    _require(
        not {
            _physical_space(scene) for scene in expected_scenes
        }
        & set(manifest_semantics["excluded_physical_spaces"]),
        f"{path} contains an excluded physical space",
    )
    _require(
        metadata.get("radio_checkpoint_sha256") == manifest["radio_checkpoint_sha256"]
        and metadata.get("radio_version") == manifest["radio_version"],
        f"{path} RADIO provenance differs",
    )

    contract = _contract_from_metadata(metadata, "region_contract")
    _require(
        contract.digest == metadata.get("region_contract_sha256"),
        f"{path} region-contract digest mismatch",
    )
    _require(
        metadata.get("region_contract_version") == contract.version,
        f"{path} region-contract version mismatch",
    )
    cache_contract = manifest["cache_contract"]
    _require(
        tuple(contract.radii_m) == tuple(cache_contract["region_radii_m"])
        and contract.maximum_tokens == int(cache_contract["maximum_tokens"])
        and contract.path_cost_mode == cache_contract["path_cost_mode"]
        and contract.path_affinity_floor == float(cache_contract["path_affinity_floor"])
        and contract.token_subsampling == cache_contract["token_subsampling"]
        and contract.core_token_fraction == float(cache_contract["core_token_fraction"])
        and contract.context_ratio == float(specification["context_ratio"])
        and contract.token_candidate_limit == int(specification["token_candidate_limit"])
        and contract.reliability_semantics == specification["reliability"],
        f"{path} region contract differs from the manifest",
    )
    teacher_contract = _contract_from_metadata(metadata, "teacher_region_contract")
    expected_teacher_contract = _teacher_region_contract(
        contract,
        int(cache_contract["teacher_region_candidate_limit"]),
    )
    _require(
        teacher_contract.to_dict() == expected_teacher_contract.to_dict()
        and teacher_contract.digest == expected_teacher_contract.digest
        and teacher_contract.digest == metadata.get("teacher_region_contract_sha256"),
        f"{path} teacher contract differs from the fixed-core builder contract",
    )
    protocol = metadata.get("teacher_target_protocol")
    _require(isinstance(protocol, Mapping), f"{path} lacks teacher protocol")
    expected_protocol = {
        "schema_version": 1,
        "support_semantics": FIXED_CORE_TEACHER_SEMANTICS,
        "teacher_region_contract_sha256": teacher_contract.digest,
        "crop_protocol": TEACHER_CROP_PROTOCOL,
        "frame_selection": "sorted_valid_frames_even_spacing_v1",
        "frames_per_scene": int(cache_contract["frames_per_scene"]),
        "minimum_visible_support_tokens": 12,
        "maximum_teacher_views": int(cache_contract["teacher_views"]),
        "crop_resize_resolution": 384,
        "radio_version": manifest["radio_version"],
        "radio_checkpoint_sha256": manifest["radio_checkpoint_sha256"],
        "target_padding": "left_aligned_zero_padding_v1",
        "teacher_medoid": "official_descriptor_pairwise_consensus_v1",
    }
    _require(
        dict(protocol) == expected_protocol
        and _canonical_json_sha256(protocol)
        == metadata.get("teacher_target_protocol_sha256"),
        f"{path} teacher protocol digest mismatch",
    )
    _require(
        metadata.get("teacher_target_source") == specification["teacher_source"],
        f"{path} teacher source differs",
    )
    _require(
        metadata.get("teacher_region_semantics")
        == FIXED_CORE_TEACHER_SEMANTICS
        and metadata.get("teacher_crop_protocol") == TEACHER_CROP_PROTOCOL
        and metadata.get("teacher_target_schema_version") == 1,
        f"{path} teacher semantics differ",
    )
    _require(
        metadata.get("regions_per_scene_requested")
        == int(cache_contract["regions_per_scene"])
        and metadata.get("teacher_views_requested") == int(cache_contract["teacher_views"]),
        f"{path} teacher sampling contract differs",
    )
    _require(
        _close(
            metadata.get("execution_radio_thermal_pacing_seconds_per_image"),
            cache_contract["radio_thermal_pacing_seconds_per_image"],
        ),
        f"{path} thermal pacing provenance differs",
    )

    records = metadata.get("region_records")
    _require(isinstance(records, list) and records, f"{path} lacks region records")
    row_count = len(records)
    for key in EVALUATION_KEYS:
        _require(key in payload, f"{path} lacks {key}")
        tensor = torch.as_tensor(payload[key])
        _require(
            tensor.device.type == "cpu" and tensor.shape[0] == row_count,
            f"{path} has a misaligned tensor {key}",
        )

    radio_features = torch.as_tensor(payload["radio_features"])
    geometry = torch.as_tensor(payload["geometry"])
    token_mask = torch.as_tensor(payload["token_mask"])
    reliability = torch.as_tensor(payload["reliability"])
    anchor_index = torch.as_tensor(payload["anchor_index"])
    maximum_tokens = int(cache_contract["maximum_tokens"])
    _require(
        radio_features.dtype == torch.float16
        and radio_features.shape == (row_count, maximum_tokens, SUMMARY_TOKEN_DIM)
        and geometry.dtype == torch.float16
        and geometry.shape == (row_count, maximum_tokens, 14)
        and token_mask.dtype == torch.bool
        and token_mask.shape == (row_count, maximum_tokens)
        and reliability.dtype == torch.float16
        and reliability.shape == (row_count, maximum_tokens, 1)
        and anchor_index.dtype == torch.int64
        and anchor_index.shape == (row_count,),
        f"{path} has malformed student input tensors",
    )
    _require(
        bool(torch.isfinite(radio_features).all())
        and bool(torch.isfinite(geometry).all())
        and bool(torch.isfinite(reliability).all())
        and not bool(((reliability < 0) | (reliability > 1)).any()),
        f"{path} has non-finite/out-of-range student inputs",
    )
    token_counts = token_mask.sum(dim=1).to(torch.int64)
    expected_token_mask = (
        torch.arange(maximum_tokens, dtype=torch.int64)[None]
        < token_counts[:, None]
    )
    rows_index = torch.arange(row_count, dtype=torch.int64)
    _require(
        bool((token_counts > 0).all())
        and torch.equal(token_mask.cpu(), expected_token_mask)
        and bool((anchor_index >= 0).all())
        and bool((anchor_index < maximum_tokens).all())
        and bool(token_mask[rows_index, anchor_index].all())
        and not bool(radio_features[~token_mask].count_nonzero())
        and not bool(geometry[~token_mask].count_nonzero())
        and not bool(reliability[~token_mask].count_nonzero()),
        f"{path} has invalid token padding/anchors",
    )

    summary_tokens = torch.as_tensor(payload["official_summary_tokens"])
    crop_summaries = torch.as_tensor(payload["official_crop_summaries"])
    teacher_mask = torch.as_tensor(payload["teacher_mask"])
    teacher_views = int(cache_contract["teacher_views"])
    _require(
        summary_tokens.dtype == torch.float16
        and summary_tokens.shape
        == (row_count, teacher_views, SUMMARY_TOKEN_DIM)
        and crop_summaries.dtype == torch.float16
        and crop_summaries.shape
        == (row_count, teacher_views, SIGLIP2_DESCRIPTOR_DIM)
        and teacher_mask.dtype == torch.bool
        and teacher_mask.shape == (row_count, teacher_views),
        f"{path} has malformed teacher target tensors",
    )
    _require(
        bool(torch.isfinite(summary_tokens).all())
        and bool(torch.isfinite(crop_summaries).all()),
        f"{path} has non-finite teacher targets",
    )
    view_counts = teacher_mask.sum(dim=1).to(torch.int64)
    expected_teacher_mask = (
        torch.arange(teacher_views, dtype=torch.int64)[None]
        < view_counts[:, None]
    )
    _require(
        bool((view_counts >= 2).all())
        and torch.equal(teacher_mask.cpu(), expected_teacher_mask)
        and not bool(summary_tokens[~teacher_mask].count_nonzero())
        and not bool(crop_summaries[~teacher_mask].count_nonzero()),
        f"{path} has invalid teacher mask/padding",
    )
    _require(
        len({str(record.get("region_id", "")) for record in records}) == row_count
        and all(str(record.get("region_id", "")) for record in records),
        f"{path} has duplicate/empty region IDs",
    )
    scenes = sorted({str(record.get("scene", "")) for record in records})
    _require(all(scenes), f"{path} has an empty scene identity")
    counts = {
        scene: sum(str(record.get("scene")) == scene for record in records)
        for scene in scenes
    }
    regions_per_scene = int(cache_contract["regions_per_scene"])
    _require(
        scenes == expected_scenes
        and metadata.get("scene_names") == scenes
        and metadata.get("scene_region_counts") == counts
        and all(count == regions_per_scene for count in counts.values())
        and row_count == len(expected_scenes) * regions_per_scene,
        f"{path} scene metadata is inconsistent",
    )
    teacher_identities = []
    for row_index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"{path} has a non-object region record")
        scene = _scene_name(record.get("scene"), label=f"{path} record scene")
        _require(scene in expected_scenes, f"{path} record belongs to another scene")
        seed = record.get("seed")
        radius = record.get("physical_radius_m")
        _require(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
            f"{path} record has an invalid seed",
        )
        radius_value = _finite_number(radius, label=f"{path} record radius")
        _require(
            any(_close(radius_value, value) for value in contract.radii_m),
            f"{path} record has a radius outside the contract",
        )
        support_sha = record.get("teacher_support_sha256")
        _require(_is_sha256(support_sha), f"{path} record has an invalid support SHA256")
        expected_region_id = _surface_region_id(
            scene,
            int(seed),
            radius_value,
            teacher_contract.digest,
            str(support_sha),
        )
        _require(
            record.get("region_id") == expected_region_id,
            f"{path} record has an invalid region ID",
        )
        _require(
            record.get("tokens") == int(token_counts[row_index])
            and record.get("anchor_local_index") == int(anchor_index[row_index]),
            f"{path} record disagrees with student token tensors",
        )
        source_views = record.get("teacher_views")
        view_count = int(view_counts[row_index])
        _require(
            isinstance(source_views, list) and len(source_views) == view_count,
            f"{path} record has an invalid teacher-view count",
        )
        for view_index, view in enumerate(source_views):
            _validate_teacher_view(
                view,
                label=f"{path} record {row_index} teacher view {view_index}",
            )
        teacher_region_tokens = record.get("teacher_region_tokens")
        _require(
            isinstance(teacher_region_tokens, int)
            and not isinstance(teacher_region_tokens, bool)
            and 0 < teacher_region_tokens
            < int(cache_contract["teacher_region_candidate_limit"])
            and record.get("teacher_region_saturated") is False
            and record.get("teacher_target_source")
            == metadata.get("teacher_target_source"),
            f"{path} record has invalid teacher-support provenance",
        )
        normalized_descriptors = F.normalize(
            crop_summaries[row_index, :view_count].float(),
            dim=-1,
            eps=1e-8,
        )
        _require(
            bool((normalized_descriptors.norm(dim=-1) > 0).all()),
            f"{path} record has a zero valid teacher descriptor",
        )
        expected_medoid = int(
            (normalized_descriptors @ normalized_descriptors.T)
            .sum(dim=1)
            .argmax()
        )
        _require(
            record.get("teacher_medoid") == expected_medoid,
            f"{path} record teacher medoid differs from target tensors",
        )
        target_sha = _teacher_target_sha256(
            summary_tokens[row_index],
            crop_summaries[row_index],
            teacher_mask[row_index],
        )
        _require(
            record.get("teacher_target_sha256") == target_sha,
            f"{path} record teacher target digest is inconsistent",
        )
        teacher_identities.append(_record_teacher_identity(record))
    return {
        "metadata": metadata,
        "records": teacher_identities,
        "regions": row_count,
        "feature_dim": int(radio_features.shape[-1]),
        "teacher_tensors": {
            key: torch.as_tensor(payload[key])
            for key in (
                "official_summary_tokens",
                "official_crop_summaries",
                "teacher_mask",
            )
        },
    }


def _validate_cache_sidecar(
    sidecar_path: Path,
    *,
    cache_path: Path,
    role: str,
    row: Mapping,
    metadata: Mapping,
) -> str:
    sidecar = _json_object(sidecar_path)
    _require(
        Path(str(sidecar.get("output", ""))).resolve() == cache_path,
        f"{sidecar_path} points to another cache",
    )
    _require(sidecar.get("regions") == row["regions"], f"{sidecar_path} region drift")
    _require(
        sidecar.get("scenes") == len(metadata.get("scene_names", [])),
        f"{sidecar_path} scene-count drift",
    )
    _require(sidecar.get("split_role") == role, f"{sidecar_path} split drift")
    _require(not sidecar.get("failed_scenes"), f"{sidecar_path} records failures")
    _require(
        sidecar.get("split_file_sha256") == metadata.get("split_file_sha256")
        and sidecar.get("teacher_target_source") == metadata.get("teacher_target_source")
        and sidecar.get("teacher_replay_cache") == metadata.get("teacher_replay_cache"),
        f"{sidecar_path} provenance differs from its cache",
    )
    return _sha256(sidecar_path)


def _validate_pairing_and_caches(
    output_root: Path,
    manifest_path: Path,
    manifest: Mapping,
    manifest_semantics: Mapping,
) -> tuple[Path, dict, dict[str, dict], list[dict]]:
    pairing_path = output_root / "cache_pairing.json"
    pairing = _json_object(pairing_path)
    _require(
        pairing.get("schema_version") == 1
        and pairing.get("status") == "exact_teacher_replay_verified",
        "cache pairing is not an exact-replay report",
    )
    _require(
        Path(str(pairing.get("run_manifest", ""))).resolve() == manifest_path
        and pairing.get("run_manifest_sha256") == _sha256(manifest_path),
        "cache pairing is bound to another run manifest",
    )
    _require(
        pairing.get("benchmark_queries_opened") is False
        and pairing.get("benchmark_masks_opened") is False,
        "cache pairing is not query-free",
    )
    raw_rows = pairing.get("caches")
    _require(isinstance(raw_rows, list), "cache pairing lacks cache rows")
    expected_keys = {
        (candidate, role, shard)
        for candidate in EXPECTED_CANDIDATES
        for role, count in (("train", 4), ("validation", 2))
        for shard in range(count)
    }
    rows = {}
    for row in raw_rows:
        _require(isinstance(row, Mapping), "cache pairing contains a non-object row")
        key = (row.get("candidate"), row.get("role"), row.get("shard"))
        _require(key in expected_keys and key not in rows, "cache pairing key set differs")
        rows[key] = row
    _require(set(rows) == expected_keys, "cache pairing is incomplete")

    builder_sha = manifest["implementation_sources"][
        "radio_gs/scripts/build_scannet_surface_region_cache.py"
    ]
    cache_state: dict[str, dict] = {candidate: {} for candidate in EXPECTED_CANDIDATES}
    seen_region_ids: dict[str, dict[str, set[str]]] = {
        candidate: {"train": set(), "validation": set()}
        for candidate in EXPECTED_CANDIDATES
    }
    bindings: list[dict] = []
    for role, count in (("train", 4), ("validation", 2)):
        for shard in range(count):
            control_payload = None
            control_state = None
            control_path = output_root / "caches" / CONTROL / f"{role}_shard{shard}.pt"
            for candidate in EXPECTED_CANDIDATES:
                row = rows[(candidate, role, shard)]
                path = output_root / "caches" / candidate / f"{role}_shard{shard}.pt"
                _require(
                    Path(str(row.get("path", ""))).resolve() == path,
                    "cache pairing points outside the frozen candidate layout",
                )
                _require(_is_sha256(row.get("sha256")), f"{path} lacks a cache SHA256")
                _require(_sha256(path) == row["sha256"], f"{path} cache SHA256 mismatch")
                payload = _torch_mapping(path)
                state = _validate_cache_payload(
                    payload,
                    path=path,
                    role=role,
                    shard=shard,
                    candidate=candidate,
                    specification=manifest["candidates"][candidate],
                    manifest=manifest,
                    manifest_semantics=manifest_semantics,
                    builder_sha256=builder_sha,
                )
                _require(state["regions"] == row.get("regions"), f"{path} region count drift")
                _require(
                    state["metadata"].get("teacher_target_protocol_sha256")
                    == row.get("teacher_target_protocol_sha256"),
                    f"{path} teacher protocol differs from pairing",
                )
                sidecar_path = path.with_suffix(path.suffix + ".json")
                sidecar_sha = _validate_cache_sidecar(
                    sidecar_path,
                    cache_path=path,
                    role=role,
                    row=row,
                    metadata=state["metadata"],
                )
                if candidate == CONTROL:
                    _require(
                        path == control_path
                        and state["metadata"].get("teacher_target_source")
                        == "fresh_official_runtime"
                        and state["metadata"].get("teacher_replay_cache") == {},
                        f"{path} is not a fresh control cache",
                    )
                    control_payload, control_state = payload, state
                else:
                    _require(control_payload is not None and control_state is not None, "control ordering failed")
                    expected_replay = {"path": str(control_path), "sha256": rows[(CONTROL, role, shard)]["sha256"]}
                    _require(
                        state["metadata"].get("teacher_target_source") == "exact_cache_replay"
                        and state["metadata"].get("teacher_replay_cache") == expected_replay,
                        f"{path} does not bind the fresh control cache",
                    )
                    _require(state["records"] == control_state["records"], f"{path} teacher identities drift")
                    for tensor_key, tensor in state["teacher_tensors"].items():
                        _require(
                            torch.equal(tensor, control_state["teacher_tensors"][tensor_key]),
                            f"{path} does not exactly replay {tensor_key}",
                        )
                current_ids = {
                    str(record["region_id"]) for record in state["records"]
                }
                _require(
                    not current_ids & seen_region_ids[candidate][role],
                    f"{path} duplicates a region ID from another shard",
                )
                seen_region_ids[candidate][role].update(current_ids)
                state["path"] = str(path)
                state["sha256"] = str(row["sha256"])
                cache_state[candidate][(role, shard)] = state
                bindings.append(
                    {
                        "candidate": candidate,
                        "role": role,
                        "shard": shard,
                        "path": str(path),
                        "sha256": row["sha256"],
                        "sidecar": str(sidecar_path),
                        "sidecar_sha256": sidecar_sha,
                        "regions": state["regions"],
                        "region_contract_sha256": state["metadata"]["region_contract_sha256"],
                        "teacher_target_protocol_sha256": state["metadata"]["teacher_target_protocol_sha256"],
                    }
                )
    return pairing_path, pairing, cache_state, bindings


def _expected_training_config(
    output_root: Path,
    manifest: Mapping,
    candidate: str,
    seed: int,
) -> dict:
    contract = manifest["readout_contract"]
    checkpoint = output_root / "readouts" / f"{candidate}_seed{seed}.pt"
    return {
        "train_caches": str(output_root / "caches" / candidate / "train_shard*.pt"),
        "validation_caches": str(output_root / "caches" / candidate / "validation_shard*.pt"),
        "output": str(checkpoint),
        "hidden_dim": int(contract["hidden_dim"]),
        "epochs": int(contract["epochs"]),
        "patience": int(contract["patience"]),
        "batch_size": int(contract["batch_size"]),
        "learning_rate": float(contract["learning_rate"]),
        "weight_decay": float(contract["weight_decay"]),
        "token_weight": float(contract["token_weight"]),
        "relation_weight": float(contract["relation_weight"]),
        "reliability_attention_mode": contract["reliability_attention_mode"],
        "canonical_noise_degrees": 0.0,
        "canonical_noise_calibration": "",
        "seed": seed,
        "device": "cuda:0",
        "radio_checkpoint": str(Path(str(manifest["radio_checkpoint"])).resolve()),
    }


def _validate_merged_cache_provenance(
    value: object,
    *,
    output_root: Path,
    candidate: str,
    role: str,
    cache_state: Mapping,
    manifest: Mapping,
) -> None:
    _require(isinstance(value, Mapping), f"{candidate} checkpoint lacks {role} provenance")
    count = 4 if role == "train" else 2
    expected_paths = [
        str(output_root / "caches" / candidate / f"{role}_shard{shard}.pt")
        for shard in range(count)
    ]
    _require(value.get("cache_paths") == expected_paths, f"{candidate} {role} cache paths differ")
    _require(
        value.get("split_hashes") == [manifest[f"{role}_split_sha256"]],
        f"{candidate} {role} split provenance differs",
    )
    contract_hashes = {
        cache_state[(role, shard)]["metadata"]["region_contract_sha256"]
        for shard in range(count)
    }
    _require(len(contract_hashes) == 1, f"{candidate} {role} contracts differ")
    contract_hash = next(iter(contract_hashes))
    _require(
        value.get("region_contract_sha256") == contract_hash
        and value.get("region_contract")
        == cache_state[(role, 0)]["metadata"]["region_contract"],
        f"{candidate} {role} contract provenance differs",
    )
    expected_teacher = {
        "semantics": cache_state[(role, 0)]["metadata"]["teacher_region_semantics"],
        "contract": cache_state[(role, 0)]["metadata"]["teacher_region_contract"],
        "contract_sha256": cache_state[(role, 0)]["metadata"]["teacher_region_contract_sha256"],
        "target_source": cache_state[(role, 0)]["metadata"]["teacher_target_source"],
        "target_protocol_sha256": cache_state[(role, 0)]["metadata"]["teacher_target_protocol_sha256"],
    }
    for shard in range(count):
        current_metadata = cache_state[(role, shard)]["metadata"]
        current_teacher = {
            "semantics": current_metadata["teacher_region_semantics"],
            "contract": current_metadata["teacher_region_contract"],
            "contract_sha256": current_metadata["teacher_region_contract_sha256"],
            "target_source": current_metadata["teacher_target_source"],
            "target_protocol_sha256": current_metadata[
                "teacher_target_protocol_sha256"
            ],
        }
        _require(
            current_teacher == expected_teacher,
            f"{candidate} {role} cache teacher contracts differ",
        )
    _require(
        value.get("teacher_region") == expected_teacher,
        f"{candidate} {role} teacher provenance differs",
    )
    expected_scenes = sorted(
        {
            str(scene)
            for shard in range(count)
            for scene in cache_state[(role, shard)]["metadata"]["scene_names"]
        }
    )
    _require(
        value.get("scenes") == expected_scenes,
        f"{candidate} {role} scene provenance differs",
    )
    expected_spaces = cache_state[(role, 0)]["metadata"].get(
        "excluded_physical_spaces"
    )
    expected_exclusions = cache_state[(role, 0)]["metadata"].get(
        "exclusion_files"
    )
    for shard in range(count):
        metadata = cache_state[(role, shard)]["metadata"]
        _require(
            metadata.get("excluded_physical_spaces") == expected_spaces
            and metadata.get("exclusion_files") == expected_exclusions,
            f"{candidate} {role} cache exclusion contracts differ",
        )
    _require(
        value.get("radio_checkpoint_sha256") == manifest["radio_checkpoint_sha256"]
        and value.get("physical_space_disjoint") is True
        and value.get("excluded_physical_spaces") == expected_spaces
        and value.get("exclusion_files") == expected_exclusions,
        f"{candidate} {role} checkpoint provenance is not disjoint",
    )


def _load_summary_head(radio_checkpoint: Path) -> torch.nn.Module:
    """Load the frozen official projection once for CPU metric recomputation."""

    from radio_gs.models.siglip_projection import SigLIP2SummaryHead

    try:
        head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_checkpoint))
    except Exception as error:
        raise ValueError(
            "could not load the manifest-bound official SigLIP2 summary head"
        ) from error
    return head.cpu().eval().requires_grad_(False)


def _load_validation_evaluation_data(
    cache_state: Mapping,
) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {key: [] for key in EVALUATION_KEYS}
    for shard in range(2):
        state = cache_state[("validation", shard)]
        path = Path(state["path"])
        _require(
            _sha256(path) == state["sha256"],
            f"{path} changed before validation metric recomputation",
        )
        payload = _torch_mapping(path)
        for key in EVALUATION_KEYS:
            parts[key].append(torch.as_tensor(payload[key]).cpu())
    return {key: torch.cat(values, dim=0) for key, values in parts.items()}


@torch.inference_mode()
def _evaluate_readout_cpu(
    model: SurfaceRegionSummaryReadoutV2,
    head: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, float]:
    """Recompute the trainer's validation metrics from tensors and weights."""

    _require(batch_size > 0, "validation evaluator batch size must be positive")
    token_cosines: list[float] = []
    descriptor_cosines: list[float] = []
    multiview_cosines: list[float] = []
    rows_total = len(data["radio_features"])
    _require(rows_total > 0, "validation evaluator received no rows")
    model = model.cpu().eval().requires_grad_(False)
    head = head.cpu().eval().requires_grad_(False)
    for start in range(0, rows_total, int(batch_size)):
        stop = min(start + int(batch_size), rows_total)
        rows = torch.arange(start, stop, dtype=torch.int64)
        tokens = data["official_summary_tokens"][rows].float()
        descriptors = F.normalize(
            data["official_crop_summaries"][rows].float(),
            dim=-1,
        )
        teacher_mask = data["teacher_mask"][rows].bool()
        similarity = torch.einsum("bvd,bwd->bvw", descriptors, descriptors)
        similarity = similarity.masked_fill(~teacher_mask[:, None, :], 0.0)
        medoid = similarity.sum(-1).masked_fill(~teacher_mask, -1e9).argmax(-1)
        batch = torch.arange(len(rows), dtype=torch.int64)
        target_token = tokens[batch, medoid]
        weights = teacher_mask.float() / teacher_mask.sum(1, keepdim=True)
        target_descriptor = F.normalize(
            (descriptors * weights[..., None]).sum(1),
            dim=-1,
            eps=1e-8,
        )
        predicted = model(
            data["radio_features"][rows],
            data["geometry"][rows],
            anchor_index=data["anchor_index"][rows],
            token_mask=data["token_mask"][rows],
            reliability=data["reliability"][rows],
        )
        projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1)
        _require(
            predicted.shape == target_token.shape
            and projected.shape == target_descriptor.shape
            and bool(torch.isfinite(predicted).all())
            and bool(torch.isfinite(projected).all()),
            "validation evaluator produced malformed/non-finite predictions",
        )
        token_cosines.extend(
            F.cosine_similarity(predicted, target_token, dim=-1).tolist()
        )
        descriptor_cosines.extend(
            F.cosine_similarity(projected, target_descriptor, dim=-1).tolist()
        )
        pairwise = torch.einsum("bd,bvd->bv", projected, descriptors)
        multiview_cosines.extend(pairwise[teacher_mask].tolist())
    return {
        "summary_token_cosine": sum(token_cosines) / len(token_cosines),
        "mean_descriptor_cosine": sum(descriptor_cosines)
        / len(descriptor_cosines),
        "all_view_descriptor_cosine": sum(multiview_cosines)
        / len(multiview_cosines),
    }


def _require_metrics_match(
    reported: Mapping,
    recomputed: Mapping,
    *,
    label: str,
    tolerance: float = VALIDATION_RECOMPUTE_TOLERANCE,
) -> None:
    for key in METRIC_KEYS:
        _require(
            _close(reported.get(key), recomputed.get(key), tolerance=tolerance),
            f"{label} {key} differs from CPU recomputation",
        )


def _untrained_baseline_model(
    *,
    feature_dim: int,
    hidden_dim: int,
    reliability_attention_mode: str,
    seed: int,
) -> SurfaceRegionSummaryReadoutV2:
    # Training seeds CPU initialization before moving the module to CUDA.  A
    # forked RNG reproduces that state without perturbing finalizer callers.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return SurfaceRegionSummaryReadoutV2(
            feature_dim=int(feature_dim),
            hidden_dim=int(hidden_dim),
            reliability_attention_mode=str(reliability_attention_mode),
        ).cpu().eval().requires_grad_(False)


def _validate_checkpoint_and_sidecar(
    output_root: Path,
    manifest: Mapping,
    cache_state: Mapping,
    validation_data: Mapping[str, torch.Tensor],
    summary_head: torch.nn.Module,
    *,
    candidate: str,
    seed: int,
) -> dict:
    checkpoint_path = output_root / "readouts" / f"{candidate}_seed{seed}.pt"
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    checkpoint_sha = _sha256(checkpoint_path)
    model, checkpoint, _, _ = load_surface_region_summary_readout_v2(
        checkpoint_path,
        map_location="cpu",
    )
    model.cpu().eval().requires_grad_(False)
    architecture = checkpoint.get("architecture")
    _require(isinstance(architecture, Mapping), f"{checkpoint_path} lacks architecture")
    _require(
        architecture.get("name") == "surface_region_summary_readout_v2"
        and architecture.get("hidden_dim") == manifest["readout_contract"]["hidden_dim"]
        and architecture.get("reliability_attention_mode", "log_prior")
        == manifest["readout_contract"]["reliability_attention_mode"],
        f"{checkpoint_path} architecture differs",
    )
    expected_feature_dim = cache_state[("train", 0)]["feature_dim"]
    _require(
        architecture.get("feature_dim") == expected_feature_dim,
        f"{checkpoint_path} feature dimension differs from caches",
    )
    contract_hash = cache_state[("train", 0)]["metadata"]["region_contract_sha256"]
    _require(
        architecture.get("contract_sha256") == contract_hash,
        f"{checkpoint_path} contract digest differs",
    )
    for key, tensor in checkpoint.get("state_dict", {}).items():
        value = torch.as_tensor(tensor)
        _require(value.device.type == "cpu", f"{checkpoint_path}:{key} is not CPU resident")
        if value.is_floating_point():
            _require(bool(torch.isfinite(value).all()), f"{checkpoint_path}:{key} is non-finite")

    expected_config = _expected_training_config(output_root, manifest, candidate, seed)
    _require(
        checkpoint.get("training_config") == expected_config,
        f"{checkpoint_path} training configuration differs",
    )
    provenance = checkpoint.get("provenance")
    _require(isinstance(provenance, Mapping), f"{checkpoint_path} lacks provenance")
    _require(
        provenance.get("training_scope") == "global_cross_scene_3d_surface_v2"
        and provenance.get("frozen") is True
        and provenance.get("uses_benchmark_scenes") is False
        and provenance.get("uses_benchmark_test_vocabulary") is False
        and provenance.get("scene_disjoint") is True
        and provenance.get("official_summary_head") == "c-radio_v4 siglip2-g"
        and provenance.get("custom_text_projection") is False
        and _close(provenance.get("canonical_direction_noise_degrees"), 0.0)
        and provenance.get("canonical_noise_calibration") == "",
        f"{checkpoint_path} violates the query-free readout contract",
    )
    _require(
        provenance.get("random_seed_contract")
        == {
            "seed": seed,
            "model_initialization": True,
            "data_order": True,
            "canonical_noise": True,
        },
        f"{checkpoint_path} seed provenance differs",
    )
    _validate_merged_cache_provenance(
        provenance.get("train"),
        output_root=output_root,
        candidate=candidate,
        role="train",
        cache_state=cache_state,
        manifest=manifest,
    )
    _require(
        provenance["train"].get("region_contract_sha256")
        == provenance["validation"].get("region_contract_sha256")
        and provenance["train"].get("teacher_region")
        == provenance["validation"].get("teacher_region"),
        f"{checkpoint_path} train/validation contracts differ",
    )
    _validate_merged_cache_provenance(
        provenance.get("validation"),
        output_root=output_root,
        candidate=candidate,
        role="validation",
        cache_state=cache_state,
        manifest=manifest,
    )
    _require(
        provenance.get("region_contract_sha256") == contract_hash
        and provenance.get("region_contract")
        == cache_state[("train", 0)]["metadata"]["region_contract"],
        f"{checkpoint_path} top-level contract provenance differs",
    )

    history = checkpoint.get("history")
    _require(isinstance(history, list) and history, f"{checkpoint_path} lacks history")
    _require(
        [row.get("epoch") for row in history] == list(range(1, len(history) + 1))
        and len(history) <= expected_config["epochs"],
        f"{checkpoint_path} history epochs differ",
    )
    for row in history:
        for key in ("loss", "selection_score", *METRIC_KEYS):
            _finite_number(row.get(key), label=f"{checkpoint_path} history {key}")
    best_score = max(float(row["selection_score"]) for row in history)
    best_epoch = next(
        int(row["epoch"])
        for row in history
        if float(row["selection_score"]) == best_score
    )
    _require(
        checkpoint.get("best_epoch") == best_epoch
        and _close(checkpoint.get("best_selection_score"), best_score),
        f"{checkpoint_path} best checkpoint selection is inconsistent",
    )

    sidecar = _json_object(sidecar_path)
    _require(
        Path(str(sidecar.get("output", ""))).resolve() == checkpoint_path,
        f"{sidecar_path} points to another checkpoint",
    )
    _require(sidecar.get("checkpoint_sha256") == checkpoint_sha, f"{sidecar_path} checkpoint SHA mismatch")
    _require(sidecar.get("architecture") == architecture, f"{sidecar_path} architecture drift")
    _require(
        sidecar.get("best_epoch") == best_epoch
        and _close(sidecar.get("best_selection_score"), best_score),
        f"{sidecar_path} best score drift",
    )
    _require(
        sidecar.get("untrained_baseline") == checkpoint.get("untrained_baseline"),
        f"{sidecar_path} baseline drift",
    )
    baseline = checkpoint.get("untrained_baseline")
    _require(isinstance(baseline, Mapping), f"{checkpoint_path} lacks baseline metrics")
    for key in METRIC_KEYS:
        _finite_number(baseline.get(key), label=f"{checkpoint_path} baseline {key}")
    baseline_score = 0.5 * (
        float(baseline["mean_descriptor_cosine"])
        + float(baseline["all_view_descriptor_cosine"])
    )
    _require(
        _close(checkpoint.get("untrained_baseline_score"), baseline_score)
        and _close(sidecar.get("selection_score_delta"), best_score - baseline_score),
        f"{sidecar_path} baseline delta is inconsistent",
    )
    validation = sidecar.get("validation")
    _require(isinstance(validation, Mapping), f"{sidecar_path} lacks validation metrics")
    for key in METRIC_KEYS:
        _finite_number(validation.get(key), label=f"{sidecar_path} validation {key}")
    validation_score = 0.5 * (
        float(validation["mean_descriptor_cosine"])
        + float(validation["all_view_descriptor_cosine"])
    )
    best_history_row = history[best_epoch - 1]
    _require(
        _close(validation_score, best_score, tolerance=1e-6)
        and all(
            _close(validation[key], best_history_row[key], tolerance=1e-6)
            for key in METRIC_KEYS
        ),
        f"{sidecar_path} validation score differs from the selected checkpoint",
    )
    train_scenes = set(provenance["train"].get("scenes", []))
    validation_scenes = set(provenance["validation"].get("scenes", []))
    _require(not train_scenes & validation_scenes, f"{checkpoint_path} has scene leakage")
    _require(
        sidecar.get("train_scenes") == len(train_scenes)
        and sidecar.get("validation_scenes") == len(validation_scenes)
        and sidecar.get("scene_overlap") == [],
        f"{sidecar_path} scene counts differ",
    )

    recomputed_validation = _evaluate_readout_cpu(
        model,
        summary_head,
        validation_data,
        batch_size=int(expected_config["batch_size"]),
    )
    _require_metrics_match(
        validation,
        recomputed_validation,
        label=f"{sidecar_path} validation",
    )
    recomputed_score = 0.5 * (
        recomputed_validation["mean_descriptor_cosine"]
        + recomputed_validation["all_view_descriptor_cosine"]
    )
    _require(
        _close(
            recomputed_score,
            best_score,
            tolerance=VALIDATION_RECOMPUTE_TOLERANCE,
        ),
        f"{sidecar_path} selected score differs from CPU recomputation",
    )
    baseline_model = _untrained_baseline_model(
        feature_dim=int(architecture["feature_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        reliability_attention_mode=str(
            architecture.get("reliability_attention_mode", "log_prior")
        ),
        seed=seed,
    )
    recomputed_baseline = _evaluate_readout_cpu(
        baseline_model,
        summary_head,
        validation_data,
        batch_size=int(expected_config["batch_size"]),
    )
    _require_metrics_match(
        baseline,
        recomputed_baseline,
        label=f"{checkpoint_path} untrained baseline",
    )
    recomputed_baseline_score = 0.5 * (
        recomputed_baseline["mean_descriptor_cosine"]
        + recomputed_baseline["all_view_descriptor_cosine"]
    )
    return {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "architecture_sha256": architecture["digest"],
        "best_epoch": best_epoch,
        "best_selection_score": float(recomputed_score),
        "selection_score_delta": float(
            recomputed_score - recomputed_baseline_score
        ),
        "validation": {
            key: float(recomputed_validation[key]) for key in METRIC_KEYS
        },
        "metric_source": "cpu_recomputed_from_bound_validation_caches",
        "reported_best_selection_score": float(sidecar["best_selection_score"]),
        "reported_selection_score_delta": float(sidecar["selection_score_delta"]),
        "reported_validation": {
            key: float(validation[key]) for key in METRIC_KEYS
        },
        "recomputed_untrained_baseline": {
            key: float(recomputed_baseline[key]) for key in METRIC_KEYS
        },
    }


def _recompute_candidate_rows(readouts: Mapping[str, Sequence[Mapping]], selection: Mapping) -> tuple[dict, str]:
    rows = {}
    for candidate in EXPECTED_CANDIDATES:
        reports = [
            {
                "seed": int(value["seed"]),
                "checkpoint": value["checkpoint"],
                "checkpoint_sha256": value["checkpoint_sha256"],
                "best_epoch": int(value["best_epoch"]),
                "best_selection_score": float(value["best_selection_score"]),
                "selection_score_delta": float(value["selection_score_delta"]),
                "validation": dict(value["validation"]),
            }
            for value in readouts[candidate]
        ]
        rows[candidate] = {
            "seeds": reports,
            "mean_selection_score": statistics.fmean(
                value["best_selection_score"] for value in reports
            ),
            "mean_validation": {
                key: statistics.fmean(value["validation"][key] for value in reports)
                for key in METRIC_KEYS
            },
        }
    control = rows[CONTROL]
    eligible = []
    for candidate, values in rows.items():
        values["mean_score_gain_over_control"] = (
            values["mean_selection_score"] - control["mean_selection_score"]
        )
        values["seed_wins_over_control"] = sum(
            current["best_selection_score"] > reference["best_selection_score"]
            for current, reference in zip(values["seeds"], control["seeds"])
        )
        values["component_drops_from_control"] = {
            key: control["mean_validation"][key] - values["mean_validation"][key]
            for key in METRIC_KEYS
        }
        values["eligible_for_query_free_promotion"] = (
            candidate != CONTROL
            and values["mean_score_gain_over_control"]
            >= float(selection["minimum_mean_score_gain"])
            and values["seed_wins_over_control"] >= int(selection["minimum_seed_wins"])
            and max(values["component_drops_from_control"].values())
            <= float(selection["maximum_component_drop"])
        )
        if values["eligible_for_query_free_promotion"]:
            eligible.append(candidate)
    selected = (
        max(
            eligible,
            key=lambda name: (
                rows[name]["mean_selection_score"],
                rows[name]["seed_wins_over_control"],
                name,
            ),
        )
        if eligible
        else CONTROL
    )
    return rows, selected


def _reported_readout_view(
    readouts: Mapping[str, Sequence[Mapping]],
) -> dict[str, list[dict]]:
    return {
        candidate: [
            {
                **value,
                "best_selection_score": value[
                    "reported_best_selection_score"
                ],
                "selection_score_delta": value[
                    "reported_selection_score_delta"
                ],
                "validation": value["reported_validation"],
            }
            for value in values
        ]
        for candidate, values in readouts.items()
    }


def _validate_screen(
    output_root: Path,
    manifest_path: Path,
    manifest: Mapping,
    pairing_path: Path,
    readouts: Mapping[str, Sequence[Mapping]],
) -> tuple[Path, dict, dict, str, Path]:
    screen_path = output_root / "query_free_screen.json"
    screen = _json_object(screen_path)
    reported_rows, reported_selected = _recompute_candidate_rows(
        _reported_readout_view(readouts),
        manifest["selection_contract"],
    )
    rows, selected = _recompute_candidate_rows(
        readouts,
        manifest["selection_contract"],
    )
    _require(
        selected == reported_selected,
        "CPU-recomputed validation metrics change the reported candidate selection",
    )
    expected_status = (
        "query_free_candidate_selected_benchmark_gate_still_closed"
        if reported_selected != CONTROL
        else "query_free_control_retained"
    )
    expected = {
        "schema_version": 1,
        "selection_status": expected_status,
        "selected_candidate": reported_selected,
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "cache_pairing_report": str(pairing_path),
        "cache_pairing_report_sha256": _sha256(pairing_path),
        "candidates": reported_rows,
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
        "next_gate": (
            "freeze the selected query-free readout, then evaluate benchmarks "
            "without changing graph, unary, score, or connected-selection rules"
        ),
    }
    _require(screen == expected, "query-free screen differs from strict recomputation")
    complete_path = output_root / "screen.complete"
    _require(complete_path.is_file() and complete_path.stat().st_size > 0, "screen completion marker is missing")
    try:
        dt.datetime.fromisoformat(complete_path.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise ValueError("screen completion marker is not an ISO-8601 timestamp") from error
    _require(
        complete_path.stat().st_mtime_ns >= screen_path.stat().st_mtime_ns,
        "screen completion marker predates the selection report",
    )
    return screen_path, screen, rows, selected, complete_path


def _write_atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize(
    output_root: Path,
    *,
    promotion_manifest: Path | None = None,
    completion: Path | None = None,
) -> dict:
    """Validate and freeze the completed screen without evaluating a benchmark."""

    output_root = Path(output_root).resolve()
    _require(output_root.is_dir(), f"SurfaceRegion output root is missing: {output_root}")
    manifest_path, manifest, _, manifest_semantics = _validate_manifest(output_root)
    pairing_path, _, cache_state, cache_bindings = _validate_pairing_and_caches(
        output_root,
        manifest_path,
        manifest,
        manifest_semantics,
    )
    summary_head = _load_summary_head(Path(str(manifest["radio_checkpoint"])).resolve())

    readouts: dict[str, list[dict]] = {}
    all_readout_bindings = []
    for candidate in EXPECTED_CANDIDATES:
        validation_data = _load_validation_evaluation_data(cache_state[candidate])
        values = []
        for seed in REQUIRED_SEEDS:
            value = _validate_checkpoint_and_sidecar(
                output_root,
                manifest,
                cache_state[candidate],
                validation_data,
                summary_head,
                candidate=candidate,
                seed=seed,
            )
            values.append(value)
            all_readout_bindings.append({"candidate": candidate, **value})
        readouts[candidate] = values
        del validation_data

    screen_path, validated_screen, rows, selected, screen_complete = _validate_screen(
        output_root,
        manifest_path,
        manifest,
        pairing_path,
        readouts,
    )
    finalizer_path = Path(__file__).resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "query_free_three_seed_bundle_frozen_benchmark_gate_closed",
        "selected_candidate": selected,
        "seed_selection_policy": "all_required_seeds_no_single_seed_selection",
        "required_seeds": list(REQUIRED_SEEDS),
        "selected_readouts": [
            {"candidate": selected, **value} for value in readouts[selected]
        ],
        "query_free_selection": {
            "control": CONTROL,
            "contract": dict(manifest["selection_contract"]),
            "metric_source": "cpu_recomputed_from_bound_validation_caches",
            "cpu_gpu_metric_tolerance": VALIDATION_RECOMPUTE_TOLERANCE,
            "selected_candidate_metrics": rows[selected],
            "reported_metric_source": "bound_query_free_screen",
            "reported_selected_candidate_metrics": validated_screen["candidates"][
                selected
            ],
        },
        "benchmark_gate": {
            "status": "closed_not_evaluated",
            "text_response_gate": "required_before_benchmark_evaluation",
            "benchmark_queries_opened": False,
            "benchmark_masks_opened": False,
            "main_result_eligible": False,
        },
        "bindings": {
            "finalizer": {"path": str(finalizer_path), "sha256": _sha256(finalizer_path)},
            "run_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "cache_pairing": {"path": str(pairing_path), "sha256": _sha256(pairing_path)},
            "query_free_screen": {"path": str(screen_path), "sha256": _sha256(screen_path)},
            "screen_completion": {"path": str(screen_complete), "sha256": _sha256(screen_complete)},
            "caches": cache_bindings,
            "all_compared_readouts": all_readout_bindings,
        },
    }

    promotion_path = (
        Path(promotion_manifest).resolve()
        if promotion_manifest is not None
        else output_root / "query_free_promotion_bundle.json"
    )
    completion_path = (
        Path(completion).resolve()
        if completion is not None
        else output_root / "query_free_promotion.complete.json"
    )
    _require(promotion_path != completion_path, "promotion and completion paths must differ")
    if promotion_path.exists():
        _require(
            _json_object(promotion_path) == payload,
            "existing promotion manifest differs from strict recomputation",
        )
    else:
        _require(
            not completion_path.exists(),
            "completion exists without its promotion manifest",
        )
        _write_atomic_json(promotion_path, payload)

    completion_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETION_ARTIFACT_TYPE,
        "status": "complete_benchmark_gate_closed",
        "promotion_manifest": str(promotion_path),
        "promotion_manifest_sha256": _sha256(promotion_path),
        "selected_candidate": selected,
        "required_seeds": list(REQUIRED_SEEDS),
        "benchmark_gate_status": "closed_not_evaluated",
        "main_result_eligible": False,
        "finalizer_sha256": _sha256(finalizer_path),
    }
    if completion_path.exists():
        _require(
            _json_object(completion_path) == completion_payload,
            "existing promotion completion differs from strict recomputation",
        )
    else:
        _write_atomic_json(completion_path, completion_payload)
    return {
        "promotion_manifest": str(promotion_path),
        "promotion_manifest_sha256": _sha256(promotion_path),
        "completion": str(completion_path),
        "completion_sha256": _sha256(completion_path),
        "selected_candidate": selected,
        "required_seeds": list(REQUIRED_SEEDS),
        "benchmark_gate_status": "closed_not_evaluated",
        "main_result_eligible": False,
        "device": "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--promotion-manifest", type=Path)
    parser.add_argument("--completion", type=Path)
    args = parser.parse_args()
    result = finalize(
        args.output_root,
        promotion_manifest=args.promotion_manifest,
        completion=args.completion,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
