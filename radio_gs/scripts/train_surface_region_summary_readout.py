#!/usr/bin/env python3
"""Train the global query-free 3-D surface-region RADIO summary readout."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_contract import (
    SurfaceRegionContractV2,
    SurfaceRegionContractV3,
    SurfaceRegionContractV4,
)
from radio_gs.interfaces.surface_region_selection import (
    surface_region_contract_from_metadata,
)
from radio_gs.interfaces.surface_region_summary import (
    JOINT_CONTEXT_POOLING,
    SEPARATE_CONTEXT_POOLING,
    SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION,
    SurfaceRegionSummaryReadoutV2,
    SurfaceRegionSummaryReadoutV3,
    SURFACE_REGION_V3_GATED_RAW_PRIOR,
    SURFACE_REGION_V3_LEGACY_RAW_BASE,
    SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.losses.surface_region_codebook_loss import (
    latent_query_responses,
    scene_listwise_and_hard_negative_loss,
)
from radio_gs.training.surface_region_sparse_support import (
    SparseTokenSupport,
    deterministic_sparse_token_support,
)
from radio_gs.training.surface_region_eligibility_completion import (
    STRUCTURED_ELIGIBILITY_POLICY,
)
from radio_gs.utils.immutable_artifacts import (
    load_sha_bound_project_checkpoint_mapping,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


FIXED_SPARSE_VALIDATION_SEED = 0
FIXED_SPARSE_VALIDATION_EPOCH = 0
SPARSE_SUPPORT_AUGMENTATION_DEFAULT = False
_DERIVED_REGION_ID_VERSION = "surface-region-training-identity-v1"
TARGET_BLIND_RESPONSE_TEMPERATURE = 0.05
TARGET_BLIND_BASE_SELECTION_MIN = 0.9255443984
TARGET_BLIND_PROFILE_P05_MIN = 0.7909541130065918
TARGET_BLIND_TOP1_MIN = 0.28217822313308716
TARGET_BLIND_RANK_P05_MIN = -0.09290908616781235
TARGET_BLIND_SMOOTH_L1_MAX = 0.00011793594161234796
TARGET_BLIND_ADDENDUM = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/surface_region_v3_target_blind_text_response_addendum_20260805.json"
)


def _validate_v3_cache_payload(
    payload: dict,
    metadata: dict,
    contract: SurfaceRegionContractV3 | SurfaceRegionContractV4,
    *,
    label: str,
) -> list[str]:
    """Validate the non-negotiable V3 tensor and support-totality contract."""

    required = (
        "radio_features",
        "geometry",
        "token_mask",
        "support_fill_mask",
        "reliability",
    )
    if any(key not in payload for key in required):
        raise ValueError(f"{label} lacks required V3 cache tensors")
    features = torch.as_tensor(payload["radio_features"])
    geometry = torch.as_tensor(payload["geometry"])
    token_mask = torch.as_tensor(payload["token_mask"])
    support_fill = torch.as_tensor(payload["support_fill_mask"])
    reliability = torch.as_tensor(payload["reliability"])
    if token_mask.dtype != torch.bool or support_fill.dtype != torch.bool:
        raise ValueError(f"{label} V3 token/support-fill masks must be boolean")
    if (
        geometry.ndim != 3
        or geometry.shape[-1] != 16
        or features.ndim != 3
        or features.shape[-1] != 1280
        or features.shape[:2] != geometry.shape[:2]
        or token_mask.shape != geometry.shape[:2]
        or support_fill.shape != token_mask.shape
        or reliability.shape != (*token_mask.shape, 1)
    ):
        raise ValueError(f"{label} V3 cache tensors have invalid geometry-16 alignment")
    if (
        contract.feature_normalization
        != "l2_direction_plus_log_raw_norm_v1"
    ):
        raise ValueError(f"{label} V3 cache has the wrong RADIO feature gauge")
    fill_channel = geometry[..., 14]
    if (
        not torch.equal(fill_channel, support_fill.to(fill_channel.dtype))
        or bool((support_fill & ~token_mask).any())
    ):
        raise ValueError(
            f"{label} V3 support_fill_mask must exactly match geometry index 14 "
            "and be a token_mask subset"
        )
    padding = ~token_mask
    if any(
        bool(torch.count_nonzero(tensor[padding]))
        for tensor in (features, geometry, reliability)
    ):
        raise ValueError(
            f"{label} V3 features/geometry/reliability padding must be exactly zero"
        )
    if not all(
        bool(torch.isfinite(tensor).all())
        for tensor in (features, geometry, reliability)
    ):
        raise ValueError(f"{label} V3 cache tensors must be finite")
    active_norm = torch.linalg.vector_norm(features.float(), dim=-1)[token_mask]
    if not torch.allclose(
        active_norm,
        torch.ones_like(active_norm),
        rtol=2e-3,
        atol=2e-3,
    ):
        raise ValueError(
            f"{label} V3 radio_features must use the unit L2 direction gauge"
        )
    counts = token_mask.sum(dim=1)
    records = metadata.get("region_records")
    if not isinstance(records, list) or len(records) != len(token_mask):
        raise ValueError(f"{label} V3 cache lacks aligned region records")
    for row, record in enumerate(records):
        fill_count = int(support_fill[row].sum())
        semantic_count = int(counts[row]) - fill_count
        if (
            not isinstance(record, dict)
            or record.get("minimum_satisfied") is not True
            or int(record.get("tokens", -1)) != int(counts[row])
            or int(record.get("support_fill_tokens", -1)) != fill_count
            or int(record.get("semantic_tokens", -1)) != semantic_count
            or int(counts[row]) < int(contract.minimum_tokens)
        ):
            raise ValueError(
                f"{label} V3 region records do not certify minimum_satisfied"
            )
    completion = metadata.get("eligibility_completion")
    if completion is None:
        return ["full_support"] * len(records)
    if not isinstance(completion, dict):
        raise ValueError(f"{label} V3 eligibility completion metadata is invalid")
    variants = int(completion.get("variants_per_teacher_region", 0))
    if (
        completion.get("schema_version") != 1
        or completion.get("validation_checkpoint_selection")
        != "full_support_rows_only"
        or variants <= 0
        or completion.get("policy") != STRUCTURED_ELIGIBILITY_POLICY
    ):
        raise ValueError(f"{label} V3 eligibility completion contract differs")
    roles: list[str] = []
    full_by_id: dict[str, int] = {}
    completion_by_full: dict[str, list[int]] = {}
    for row, record in enumerate(records):
        region_id = str(record.get("region_id", ""))
        paired = str(record.get("paired_full_region_id", ""))
        role = str(record.get("row_role", ""))
        if (
            not region_id
            or not paired
            or int(record.get("eligibility_variants_per_teacher_region", -1))
            != variants
        ):
            raise ValueError(f"{label} V3 paired row identity is invalid")
        if role == "full_support":
            if (
                paired != region_id
                or int(record.get("eligibility_variant_index", -2)) != -1
                or region_id in full_by_id
            ):
                raise ValueError(f"{label} V3 full-support row identity differs")
            full_by_id[region_id] = row
        elif role == "eligibility_completion":
            variant = int(record.get("eligibility_variant_index", -1))
            digest = str(record.get("eligibility_sha256", ""))
            if (
                not 0 <= variant < variants
                or len(digest) != 64
                or any(value not in "0123456789abcdef" for value in digest)
                or int(record.get("support_fill_tokens", 0)) <= 0
                or record.get("eligibility_policy")
                != STRUCTURED_ELIGIBILITY_POLICY
            ):
                raise ValueError(f"{label} V3 completion identity/fill differs")
            semantic = int(record.get("semantic_tokens", -1))
            semantic_eligible = int(
                record.get("eligibility_semantic_eligible_tokens", -2)
            )
            nominal = int(
                record.get("eligibility_nominal_semantic_keep_tokens", -3)
            )
            expected_fill = int(
                record.get("eligibility_expected_fill_tokens", -4)
            )
            fallback = record.get("eligibility_extreme_graph_fallback")
            if (
                semantic != semantic_eligible
                or int(record.get("support_fill_tokens", -1))
                != expected_fill
                or int(record.get("tokens", -1))
                != semantic + expected_fill
                or not isinstance(fallback, bool)
                or (not fallback and semantic != nominal)
                or (
                    fallback
                    and not str(
                        record.get(
                            "eligibility_extreme_graph_fallback_reason", ""
                        )
                    )
                )
            ):
                raise ValueError(f"{label} V3 completion budget differs")
            completion_by_full.setdefault(paired, []).append(row)
        else:
            raise ValueError(f"{label} V3 row has an unknown completion role")
        roles.append(role)
    if set(completion_by_full) != set(full_by_id):
        raise ValueError(f"{label} V3 completion pairs are incomplete")
    for full_id, full_row in full_by_id.items():
        variant_rows = completion_by_full[full_id]
        if sorted(
            int(records[row]["eligibility_variant_index"])
            for row in variant_rows
        ) != list(range(variants)):
            raise ValueError(f"{label} V3 completion variant indices differ")
        for row in variant_rows:
            for tensor in (
                torch.as_tensor(payload["official_summary_tokens"]),
                torch.as_tensor(payload["official_crop_summaries"]),
                torch.as_tensor(payload["teacher_mask"]),
            ):
                if not torch.equal(tensor[row], tensor[full_row]):
                    raise ValueError(
                        f"{label} V3 pairs do not share exact teacher tensors"
                    )
            for key in (
                "scene",
                "seed",
                "physical_radius_m",
                "teacher_support_sha256",
                "teacher_target_sha256",
            ):
                if records[row].get(key) != records[full_row].get(key):
                    raise ValueError(
                        f"{label} V3 pairs do not share frozen teacher identity"
                    )
    completion_rows = len(records) - len(full_by_id)
    fill_tokens = sum(
        int(record["support_fill_tokens"])
        for record in records
        if record["row_role"] == "eligibility_completion"
    )
    selected_tokens = sum(
        int(record["tokens"])
        for record in records
        if record["row_role"] == "eligibility_completion"
    )
    if (
        completion_rows != len(full_by_id) * variants
        or int(completion.get("full_support_rows", -1)) != len(full_by_id)
        or int(completion.get("completion_variant_rows", -1))
        != completion_rows
        or int(completion.get("completion_rows_with_fill", -1))
        != completion_rows
        or int(completion.get("completion_support_fill_tokens", -1))
        != fill_tokens
        or int(completion.get("completion_selected_tokens", -1))
        != selected_tokens
        or int(completion.get("nominal_semantic_keep_tokens", -1))
        != int(contract.minimum_tokens)
        - max(1, int(contract.minimum_tokens) // 6)
        or int(completion.get("nominal_support_fill_tokens", -1))
        != max(1, int(contract.minimum_tokens) // 6)
        or int(completion.get("extreme_graph_fallback_rows", -1))
        != sum(
            bool(record["eligibility_extreme_graph_fallback"])
            for record in records
            if record["row_role"] == "eligibility_completion"
        )
    ):
        raise ValueError(f"{label} V3 completion coverage metadata differs")
    return roles


def _paths(raw: str) -> list[Path]:
    result = []
    for value in str(raw).replace(",", " ").split():
        matches = (
            [Path(path) for path in sorted(glob.glob(value))]
            if any(c in value for c in "*?[")
            else [Path(value)]
        )
        result.extend(matches)
    if not result or any(not path.is_file() for path in result):
        raise FileNotFoundError("surface-region cache list is empty or missing")
    return result


def _load_target_blind_text_bank(
    path: Path,
    *,
    expected_sha256: str,
    expected_split: str,
) -> tuple[torch.Tensor, dict[str, object], tuple[str, ...]]:
    """Load one immutable ImageNet-derived bank without benchmark vocabulary."""

    payload, digest, source = load_sha_bound_project_checkpoint_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label=f"target-blind {expected_split} text bank",
    )
    embeddings = payload.get("embeddings")
    queries = payload.get("queries")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "target_blind_text_embedding_cache"
        or payload.get("algorithm_version") != "siglip2-target-blind-split-v1"
        or payload.get("split") != expected_split
        or payload.get("benchmark_vocabulary_opened") is not False
        or payload.get("uses_benchmark_vocabulary_for_construction") is not False
        or payload.get("prompt_templates") != ["{query}"]
        or payload.get("text_canonicalization")
        != "official_c_radio_siglip2_g"
        or not isinstance(queries, list)
        or len(queries) == 0
        or len(set(str(value) for value in queries)) != len(queries)
        or not isinstance(embeddings, torch.Tensor)
        or embeddings.shape != (len(queries), 1536)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("target-blind text-bank contract differs")
    normalized = F.normalize(embeddings.float(), dim=-1, eps=1e-8)
    return normalized, {
        "path": str(source),
        "sha256": digest,
        "split": expected_split,
        "queries": len(queries),
        "embedding_tensor_sha256": str(
            payload.get("embedding_tensor_sha256", "")
        ),
        "ordered_records_sha256": str(
            payload.get("ordered_records_sha256", "")
        ),
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
    }, tuple(str(value) for value in queries)


def _center_target_blind_text_bank(
    embeddings: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Remove the frozen bank's common direction and restore unit directions."""

    source = F.normalize(torch.as_tensor(embeddings).float(), dim=-1, eps=1e-8)
    if source.ndim != 2 or source.shape[0] < 2:
        raise ValueError("target-blind centering needs at least two text directions")
    centered = source - source.mean(dim=0, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=-1)
    if not bool(torch.isfinite(centered).all()) or float(norms.min()) <= 1e-8:
        raise ValueError("target-blind centered text bank is degenerate")
    centered = F.normalize(centered, dim=-1, eps=1e-8)
    return centered, {
        "gauge": "normalize(l2_text_direction_minus_bank_mean)_v1",
        "mean_direction_norm_before_centering": float(
            torch.linalg.vector_norm(source.mean(dim=0))
        ),
        "centered_mean_max_abs": float(centered.mean(dim=0).abs().max()),
    }


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _seed_training(
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    """Seed model initialization, augmentation, and data order coherently."""

    value = int(seed)
    if value < 0:
        raise ValueError("training seed must be non-negative")
    torch.manual_seed(value)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(value)
    return torch.Generator().manual_seed(value)


def _tensor_digest(value: torch.Tensor) -> str:
    """Hash tensor content without serialization or row-order dependence."""

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _stable_training_region_id(
    payload: dict,
    record: dict,
    row: int,
    *,
    scene: str,
) -> str:
    """Return an existing region ID or derive one from canonical row state.

    Legacy caches may lack ``region_id``.  Their fallback identity binds the
    available scene/seed/radius fields and compact content fingerprints at the
    anchor and teacher target.  It deliberately excludes shard path, shard
    number, and row number, so repartitioning or reordering a cache is inert.
    """

    existing = str(record.get("region_id", ""))
    if existing:
        return existing
    anchor = int(torch.as_tensor(payload["anchor_index"])[int(row)])
    mask = torch.as_tensor(payload["token_mask"])[int(row)].bool()
    if anchor < 0 or anchor >= len(mask) or not bool(mask[anchor]):
        raise ValueError("cannot derive a region ID from an invalid anchor")
    canonical: dict[str, object] = {
        "version": _DERIVED_REGION_ID_VERSION,
        "scene": str(record.get("scene", scene)),
        "anchor_local_index": int(record.get("anchor_local_index", anchor)),
        "valid_tokens": int(mask.sum()),
        "anchor_radio_sha256": _tensor_digest(
            torch.as_tensor(payload["radio_features"])[int(row), anchor]
        ),
        "anchor_geometry_sha256": _tensor_digest(
            torch.as_tensor(payload["geometry"])[int(row), anchor]
        ),
        "valid_geometry_sha256": _tensor_digest(
            torch.as_tensor(payload["geometry"])[int(row), mask]
        ),
        "teacher_summary_sha256": _tensor_digest(
            torch.as_tensor(payload["official_summary_tokens"])[int(row)]
        ),
    }
    for key in (
        "seed",
        "physical_radius_m",
        "teacher_region_contract_sha256",
        "teacher_support_sha256",
        "core_tokens",
    ):
        if key in record:
            canonical[key] = record[key]
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load(
    paths: list[Path],
    expected_role: str,
    *,
    derive_region_ids: bool = False,
    expected_sha256: Sequence[str] | None = None,
) -> tuple[dict, dict]:
    keys = (
        "radio_features", "geometry", "token_mask", "reliability",
        "official_summary_tokens", "official_crop_summaries", "teacher_mask",
        "anchor_index",
    )
    parts = {key: [] for key in (*keys, "support_fill_mask")}
    scenes = set(); hashes = []; contracts = []; contract_specs = []
    contract_versions: list[str] = []
    teacher_region_specs = []
    radio_checkpoint_hashes = []
    region_ids: set[str] = set()
    row_region_ids: list[str] = []
    row_scenes: list[str] = []
    row_roles: list[str] = []
    eligibility_contracts: list[dict[str, object]] = []
    row_scenes_complete = True
    excluded_spaces: set[str] | None = None
    exclusion_files: dict[str, str] = {}
    cache_artifacts: list[dict[str, str]] = []
    if expected_sha256 is not None and len(expected_sha256) != len(paths):
        raise ValueError("surface-region cache paths and SHA-256 authorities differ")
    for path_index, path in enumerate(paths):
        expected_digest = (
            str(expected_sha256[path_index])
            if expected_sha256 is not None
            else None
        )
        payload, actual_digest, resolved_path = load_torch_mapping(
            path,
            expected_sha256=expected_digest,
            map_location="cpu",
            label="SurfaceRegion training cache",
        )
        cache_artifacts.append(
            {"path": str(resolved_path), "sha256": actual_digest}
        )
        metadata = payload.get("metadata", {})
        if metadata.get("split_role") != expected_role:
            raise ValueError(f"{path} has wrong 3-D cache schema/split")
        contract = surface_region_contract_from_metadata(metadata)
        if type(contract) not in {
            SurfaceRegionContractV2,
            SurfaceRegionContractV3,
            SurfaceRegionContractV4,
        }:
            raise ValueError(f"{path} uses an unsupported surface-region contract")
        expected_cache_schema = (
            4
            if type(contract) in {SurfaceRegionContractV3, SurfaceRegionContractV4}
            else 3
        )
        if metadata.get("schema_version") != expected_cache_schema:
            raise ValueError(f"{path} has wrong 3-D cache schema/split")
        if type(contract) in {SurfaceRegionContractV3, SurfaceRegionContractV4}:
            cache_roles = _validate_v3_cache_payload(
                payload,
                metadata,
                contract,
                label=str(path),
            )
            completion = metadata.get("eligibility_completion")
            eligibility_contracts.append(
                {
                    "enabled": completion is not None,
                    "schema_version": (
                        completion.get("schema_version")
                        if isinstance(completion, dict)
                        else None
                    ),
                    "policy": (
                        completion.get("policy")
                        if isinstance(completion, dict)
                        else None
                    ),
                    "variants_per_teacher_region": (
                        completion.get("variants_per_teacher_region")
                        if isinstance(completion, dict)
                        else 0
                    ),
                    "validation_checkpoint_selection": (
                        completion.get("validation_checkpoint_selection")
                        if isinstance(completion, dict)
                        else "full_support_rows_only"
                    ),
                }
            )
        else:
            cache_roles = ["full_support"] * len(payload["radio_features"])
            eligibility_contracts.append(
                {
                    "enabled": False,
                    "schema_version": None,
                    "policy": None,
                    "variants_per_teacher_region": 0,
                    "validation_checkpoint_selection": (
                        "full_support_rows_only"
                    ),
                }
            )
        row_roles.extend(cache_roles)
        if contract_versions and contract.version != contract_versions[0]:
            raise ValueError("surface-region cache contract versions differ")
        contracts.append(contract.digest)
        contract_specs.append(contract.to_dict())
        contract_versions.append(contract.version)
        teacher_semantics = metadata.get(
            "teacher_region_semantics",
            "selected_core_and_context_extent_legacy",
        )
        cache_records = metadata.get("region_records", [])
        row_scene_start = len(row_scenes)
        if teacher_semantics == (
            "fixed_core_geodesic_support_without_input_context_v1"
        ):
            target_protocol = metadata.get(
                "teacher_target_protocol",
                {},
            )
            protocol_digest = hashlib.sha256(
                json.dumps(
                    target_protocol,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if (
                metadata.get("teacher_target_source")
                not in {"fresh_official_runtime", "exact_cache_replay"}
                or metadata.get("teacher_regions_saturated") != 0
                or metadata.get("complete_scene_regions") is not True
                or metadata.get("teacher_target_schema_version") != 1
                or metadata.get("teacher_crop_protocol")
                != (
                    "core_support_defined_unmasked_bbox_min24_"
                    "context_pad0_v1"
                )
                or metadata.get("teacher_target_protocol_sha256")
                != protocol_digest
            ):
                raise ValueError(
                    f"{path} has an incomplete fixed-core teacher protocol"
                )
            if len(cache_records) != len(payload["radio_features"]):
                raise ValueError(
                    f"{path} has misaligned fixed-core region records"
                )
            for record in cache_records:
                region_id = str(record.get("region_id", ""))
                scene_id = str(record.get("scene", ""))
                if (
                    not region_id
                    or region_id in region_ids
                    or not scene_id
                    or scene_id not in metadata.get("scene_names", [])
                ):
                    raise ValueError(
                        "surface-region caches contain duplicate/invalid "
                        "region IDs or scene bindings"
                    )
                region_ids.add(region_id)
                row_scenes.append(scene_id)
        else:
            cache_scenes = [str(value) for value in metadata.get("scene_names", [])]
            if cache_records:
                if len(cache_records) != len(payload["radio_features"]):
                    raise ValueError(
                        f"{path} has misaligned legacy region records"
                    )
                for record in cache_records:
                    scene_id = str(record.get("scene", ""))
                    if not scene_id or scene_id not in cache_scenes:
                        raise ValueError(
                            "surface-region legacy caches contain invalid "
                            "row-to-scene bindings"
                        )
                    row_scenes.append(scene_id)
            elif len(cache_scenes) == 1:
                row_scenes.extend(
                    [cache_scenes[0]] * len(payload["radio_features"])
                )
            else:
                row_scenes_complete = False
        if derive_region_ids:
            cache_rows = len(payload["radio_features"])
            cache_row_scenes = row_scenes[row_scene_start:]
            if len(cache_row_scenes) != cache_rows:
                cache_row_scenes = [""] * cache_rows
            identity_records = cache_records or [{} for _ in range(cache_rows)]
            if len(identity_records) != cache_rows:
                raise ValueError(f"{path} has misaligned region identity records")
            row_region_ids.extend(
                _stable_training_region_id(
                    payload,
                    dict(identity_records[row]),
                    row,
                    scene=cache_row_scenes[row],
                )
                for row in range(cache_rows)
            )
        teacher_region_specs.append(
            {
                "semantics": teacher_semantics,
                "contract": metadata.get("teacher_region_contract"),
                "contract_sha256": metadata.get(
                    "teacher_region_contract_sha256",
                    "",
                ),
                "target_source": metadata.get(
                    "teacher_target_source",
                    "legacy_in_cache",
                ),
                "target_protocol_sha256": metadata.get(
                    "teacher_target_protocol_sha256",
                    "",
                ),
            }
        )
        radio_checkpoint_sha256 = str(
            metadata.get("radio_checkpoint_sha256", "")
        )
        if not radio_checkpoint_sha256:
            raise ValueError(f"{path} lacks RADIO checkpoint provenance")
        radio_checkpoint_hashes.append(radio_checkpoint_sha256)
        if any(metadata.get(key, True) for key in (
            "uses_benchmark_scenes", "uses_benchmark_test_vocabulary",
            "annotations_opened", "labels_opened", "instances_opened",
            "masks_opened", "text_opened",
        )):
            raise ValueError(f"{path} violates the query-free scene-disjoint contract")
        scenes.update(str(value) for value in metadata["scene_names"])
        hashes.append(str(metadata["split_file_sha256"]))
        cache_exclusions = {
            str(value) for value in metadata.get("excluded_physical_spaces", [])
        }
        if excluded_spaces is None:
            excluded_spaces = cache_exclusions
        elif cache_exclusions != excluded_spaces:
            raise ValueError("surface-region cache exclusion contracts differ")
        if not bool(metadata.get("physical_space_disjoint", True)):
            raise ValueError(f"{path} does not certify physical-space disjointness")
        for record in metadata.get("exclusion_files", []):
            resolved = str(record["path"])
            digest = str(record["sha256"])
            previous = exclusion_files.setdefault(resolved, digest)
            if previous != digest:
                raise ValueError("surface-region exclusion file hashes differ")
        for key in keys:
            parts[key].append(torch.as_tensor(payload[key]))
        if type(contract) in {SurfaceRegionContractV3, SurfaceRegionContractV4}:
            parts["support_fill_mask"].append(
                torch.as_tensor(payload["support_fill_mask"])
            )
    merged = {
        key: torch.cat(value, dim=0)
        for key, value in parts.items()
        if value
    }
    if derive_region_ids:
        if len(row_region_ids) != len(merged["radio_features"]):
            raise ValueError("surface-region row/identity bindings are misaligned")
        merged["region_ids"] = row_region_ids
    if row_scenes_complete:
        if len(row_scenes) != len(merged["radio_features"]):
            raise ValueError("surface-region row/scene bindings are misaligned")
        merged["scene_ids"] = row_scenes
    if len(row_roles) != len(merged["radio_features"]):
        raise ValueError("surface-region row/completion-role bindings are misaligned")
    merged["row_roles"] = row_roles
    if len(set(contracts)) != 1:
        raise ValueError("surface-region cache contracts differ")
    if len(set(contract_versions)) != 1:
        raise ValueError("surface-region cache contract versions differ")
    if any(spec != contract_specs[0] for spec in contract_specs[1:]):
        raise ValueError("surface-region cache contract specifications differ")
    if any(
        spec != teacher_region_specs[0]
        for spec in teacher_region_specs[1:]
    ):
        raise ValueError("surface-region teacher contracts differ")
    if len(set(radio_checkpoint_hashes)) != 1:
        raise ValueError("surface-region RADIO checkpoints differ")
    if any(
        contract != eligibility_contracts[0]
        for contract in eligibility_contracts[1:]
    ):
        raise ValueError("surface-region eligibility completion contracts differ")
    merged_meta = {"scenes": sorted(scenes), "split_hashes": sorted(set(hashes)),
                   "cache_paths": [str(path.resolve()) for path in paths],
                   "region_contract_sha256": contracts[0]}
    merged_meta["region_contract"] = contract_specs[0]
    merged_meta["region_contract_version"] = contract_versions[0]
    merged_meta["teacher_region"] = teacher_region_specs[0]
    merged_meta["radio_checkpoint_sha256"] = (
        radio_checkpoint_hashes[0]
    )
    merged_meta["excluded_physical_spaces"] = sorted(excluded_spaces or set())
    merged_meta["exclusion_files"] = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(exclusion_files.items())
    ]
    merged_meta["physical_space_disjoint"] = True
    merged_meta["eligibility_completion"] = eligibility_contracts[0]
    if expected_sha256 is not None:
        merged_meta["cache_artifacts"] = cache_artifacts
    return merged, merged_meta


def _select_cache_rows(data: dict, rows: torch.Tensor) -> dict:
    """Select aligned cache rows without dropping tensor/list provenance."""

    index = torch.as_tensor(rows).detach().cpu().long()
    total = len(data["radio_features"])
    selected = index.tolist()
    result: dict = {}
    for key, value in data.items():
        if torch.is_tensor(value) and value.ndim >= 1 and len(value) == total:
            result[key] = value[index]
        elif isinstance(value, list) and len(value) == total:
            result[key] = [value[row] for row in selected]
        else:
            result[key] = value
    return result


def _completion_validation_views(data: dict) -> tuple[dict, dict | None]:
    """Separate checkpoint-authoritative full rows from robustness variants."""

    roles = [str(value) for value in data.get("row_roles", [])]
    if len(roles) != len(data["radio_features"]):
        raise ValueError("validation cache lacks aligned completion row roles")
    full_rows = torch.tensor(
        [row for row, role in enumerate(roles) if role == "full_support"],
        dtype=torch.long,
    )
    completion_rows = torch.tensor(
        [
            row
            for row, role in enumerate(roles)
            if role == "eligibility_completion"
        ],
        dtype=torch.long,
    )
    if not len(full_rows):
        raise ValueError("validation cache has no full-support authority rows")
    unknown = set(roles) - {"full_support", "eligibility_completion"}
    if unknown:
        raise ValueError(f"validation cache has unknown row roles: {sorted(unknown)}")
    return (
        _select_cache_rows(data, full_rows),
        _select_cache_rows(data, completion_rows)
        if len(completion_rows)
        else None,
    )


def _eligibility_completion_training_rows(
    data: dict,
    *,
    completion_weight: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return the row-role training support and its explicit sampling contract.

    The control is intentionally binary.  Weight ``1`` is the historical
    uniform-without-replacement schedule over every cache row; weight ``0``
    removes completion variants from the epoch order rather than silently
    retaining them with a zero loss.  This keeps optimizer steps, scheduler
    semantics, and provenance aligned.
    """

    weight = float(completion_weight)
    if weight not in (0.0, 1.0):
        raise ValueError(
            "eligibility completion training weight must be 0.0 or 1.0"
        )
    roles = [str(value) for value in data.get("row_roles", [])]
    total_rows = len(data["radio_features"])
    if len(roles) != total_rows:
        raise ValueError("training cache lacks aligned completion row roles")
    unknown = set(roles) - {"full_support", "eligibility_completion"}
    if unknown:
        raise ValueError(f"training cache has unknown row roles: {sorted(unknown)}")
    full_rows = torch.tensor(
        [row for row, role in enumerate(roles) if role == "full_support"],
        dtype=torch.long,
    )
    completion_rows = torch.tensor(
        [
            row
            for row, role in enumerate(roles)
            if role == "eligibility_completion"
        ],
        dtype=torch.long,
    )
    if not len(full_rows):
        raise ValueError("training cache has no full-support authority rows")

    # Preserve the exact historical identity ordering for the default.  In
    # particular, do not concatenate role partitions before randperm: that
    # would change V2 and V3 weight-1 training even with the same RNG seed.
    selected = (
        torch.arange(total_rows, dtype=torch.long)
        if weight == 1.0
        else full_rows
    )
    sampled_completion_rows = len(completion_rows) if weight == 1.0 else 0
    contract: dict[str, object] = {
        "schema_version": 1,
        "purpose": "query_free_generic_diagnostic",
        "sampling": "uniform_without_replacement_over_positive_weight_rows_v1",
        "requested_completion_training_weight": weight,
        "full_support_sampling_weight": 1.0,
        "completion_sampling_weight": weight,
        "full_support_rows_available": len(full_rows),
        "completion_rows_available": len(completion_rows),
        "full_support_rows_sampled_per_epoch": len(full_rows),
        "completion_rows_sampled_per_epoch": sampled_completion_rows,
        "total_rows_sampled_per_epoch": len(selected),
        "paired_rows_have_equal_sampling_weight": (
            not len(completion_rows) or weight == 1.0
        ),
        "validation_checkpoint_selection": "full_support_rows_only",
        "completion_validation_authority": "diagnostic_robustness_gate_only",
    }
    return selected, contract


def _training_epoch_order(
    candidate_rows: torch.Tensor,
    *,
    total_rows: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Shuffle selected rows while keeping the default schedule bit-exact."""

    rows = torch.as_tensor(candidate_rows).detach().cpu().long()
    if len(rows) == int(total_rows) and torch.equal(
        rows, torch.arange(int(total_rows), dtype=torch.long)
    ):
        return torch.randperm(int(total_rows), generator=generator)
    return rows[torch.randperm(len(rows), generator=generator)]


def _scene_complete_epoch_batches(
    candidate_rows: torch.Tensor,
    scene_ids: Sequence[str],
    *,
    target_rows: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Pack complete scenes so listwise response supervision is well-defined."""

    selected = torch.as_tensor(candidate_rows).detach().cpu().long()
    if target_rows <= 0 or len(scene_ids) == 0:
        raise ValueError("scene-complete batches need positive size and scene IDs")
    grouped: dict[str, list[int]] = {}
    for raw_row in selected.tolist():
        if raw_row < 0 or raw_row >= len(scene_ids):
            raise ValueError("scene-complete candidate row is outside scene IDs")
        scene = str(scene_ids[raw_row])
        if not scene:
            raise ValueError("scene-complete batches require non-empty scene IDs")
        grouped.setdefault(scene, []).append(raw_row)
    if not grouped or any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("every selected training scene needs at least two rows")
    names = sorted(grouped)
    order = torch.randperm(len(names), generator=generator).tolist()
    batches: list[torch.Tensor] = []
    pending: list[int] = []
    for index in order:
        pending.extend(grouped[names[index]])
        if len(pending) >= int(target_rows):
            batches.append(torch.tensor(pending, dtype=torch.long))
            pending = []
    if pending:
        batches.append(torch.tensor(pending, dtype=torch.long))
    flattened = torch.cat(batches)
    if len(flattened) != len(selected) or set(flattened.tolist()) != set(
        selected.tolist()
    ):
        raise RuntimeError("scene-complete batches changed the candidate row set")
    return batches


def _gradient_norm(
    loss: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=False,
    )
    value = torch.stack(
        [gradient.detach().float().square().sum() for gradient in gradients]
    ).sum().sqrt()
    result = float(value.cpu())
    if not math.isfinite(result) or result <= 1e-12:
        raise ValueError("target-blind response gradient calibration is degenerate")
    return result


def _text_response_metrics(
    student: torch.Tensor,
    teacher: torch.Tensor,
    scene_ids: Sequence[str],
) -> dict[str, float]:
    """Measure response fidelity and within-scene retrieval geometry."""

    predicted = torch.as_tensor(student).detach().float().cpu()
    target = torch.as_tensor(teacher).detach().float().cpu()
    if (
        predicted.ndim != 2
        or target.shape != predicted.shape
        or len(scene_ids) != len(predicted)
    ):
        raise ValueError("text responses and scene IDs must align as [regions,queries]")
    profile = F.cosine_similarity(predicted, target, dim=-1)
    spearman: list[torch.Tensor] = []
    overlap: list[torch.Tensor] = []
    top1: list[torch.Tensor] = []
    for scene in sorted(set(str(value) for value in scene_ids)):
        rows = torch.tensor(
            [
                index
                for index, value in enumerate(scene_ids)
                if str(value) == scene
            ],
            dtype=torch.long,
        )
        if rows.numel() < 2:
            continue
        scene_prediction = predicted[rows]
        scene_target = target[rows]
        predicted_rank = scene_prediction.argsort(dim=0).argsort(dim=0).float()
        target_rank = scene_target.argsort(dim=0).argsort(dim=0).float()
        predicted_rank -= predicted_rank.mean(dim=0)
        target_rank -= target_rank.mean(dim=0)
        correlation = (predicted_rank * target_rank).sum(dim=0) / (
            predicted_rank.square().sum(dim=0).sqrt()
            * target_rank.square().sum(dim=0).sqrt()
        ).clamp_min(1e-8)
        spearman.append(correlation)
        keep = max(1, int(math.ceil(0.1 * len(rows))))
        predicted_top = scene_prediction.topk(keep, dim=0).indices
        target_top = scene_target.topk(keep, dim=0).indices
        overlap.append(
            (predicted_top[:, None, :] == target_top[None, :, :])
            .any(dim=1)
            .float()
            .mean(dim=0)
        )
        top1.append(
            (scene_prediction.argmax(dim=0) == scene_target.argmax(dim=0)).float()
        )
    if not spearman:
        raise ValueError("text-response validation needs complete multi-row scenes")
    rank = torch.cat(spearman)
    top_overlap = torch.cat(overlap)
    support = torch.cat(top1)
    return {
        "text_response_smooth_l1": float(F.smooth_l1_loss(predicted, target)),
        "response_profile_cosine_mean": float(profile.mean()),
        "response_profile_cosine_p05": float(torch.quantile(profile, 0.05)),
        "ranking_spearman_mean": float(rank.mean()),
        "ranking_spearman_p05": float(torch.quantile(rank, 0.05)),
        "top_decile_overlap_mean": float(top_overlap.mean()),
        "top_decile_overlap_p05": float(torch.quantile(top_overlap, 0.05)),
        "support_top1_agreement": float(support.mean()),
    }


def _target_blind_generic_gate(
    metrics: Mapping[str, float],
    *,
    completion_nonregression: bool,
) -> dict[str, object]:
    """Apply the immutable centered-dev noninferiority and response gates."""

    conditions = {
        "base_selection_score": (
            _selection_score(dict(metrics)) >= TARGET_BLIND_BASE_SELECTION_MIN
        ),
        "response_profile_cosine_p05": (
            float(metrics["response_profile_cosine_p05"])
            >= TARGET_BLIND_PROFILE_P05_MIN
        ),
        "support_top1_agreement": (
            float(metrics["support_top1_agreement"])
            >= TARGET_BLIND_TOP1_MIN
        ),
        "ranking_spearman_p05": (
            float(metrics["ranking_spearman_p05"])
            >= TARGET_BLIND_RANK_P05_MIN
        ),
        "text_response_smooth_l1": (
            float(metrics["text_response_smooth_l1"])
            <= TARGET_BLIND_SMOOTH_L1_MAX
        ),
        "completion_nonregression": bool(completion_nonregression),
    }
    return {
        "schema_version": 1,
        "authority": "centered_target_blind_imagenet1k_dev_v1",
        "thresholds": {
            "base_selection_score_min": TARGET_BLIND_BASE_SELECTION_MIN,
            "response_profile_cosine_p05_min": TARGET_BLIND_PROFILE_P05_MIN,
            "support_top1_agreement_min": TARGET_BLIND_TOP1_MIN,
            "ranking_spearman_p05_min": TARGET_BLIND_RANK_P05_MIN,
            "text_response_smooth_l1_max": TARGET_BLIND_SMOOTH_L1_MAX,
            "completion_nonregression_required": True,
        },
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def _view_set_token_loss(
    predicted: torch.Tensor,
    teacher_tokens: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> torch.Tensor:
    """Average valid-view cosine error per region, then equally over regions."""

    student = torch.as_tensor(predicted).float()
    teacher = torch.as_tensor(teacher_tokens).float()
    mask = torch.as_tensor(teacher_mask).bool()
    if (
        student.ndim != 2
        or teacher.ndim != 3
        or teacher.shape[0] != student.shape[0]
        or teacher.shape[2] != student.shape[1]
        or mask.shape != teacher.shape[:2]
        or bool((mask.sum(dim=1) == 0).any())
    ):
        raise ValueError("view-set token tensors must align with valid teacher views")
    cosine = torch.einsum(
        "bd,bvd->bv",
        F.normalize(student, dim=-1, eps=1e-8),
        F.normalize(teacher, dim=-1, eps=1e-8),
    )
    per_region = ((1.0 - cosine) * mask.float()).sum(dim=1) / mask.sum(
        dim=1
    ).float()
    return per_region.mean()


def _targets(data: dict, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = data["official_summary_tokens"][rows].float()
    descriptors = F.normalize(data["official_crop_summaries"][rows].float(), dim=-1)
    mask = data["teacher_mask"][rows].bool()
    # The medoid is selected in the official SigLIP2 descriptor space, not
    # by averaging or comparing backbone summary tokens.
    normalized = descriptors
    similarity = torch.einsum("bvd,bwd->bvw", normalized, normalized)
    similarity = similarity.masked_fill(~mask[:, None, :], 0.0)
    medoid = similarity.sum(-1).masked_fill(~mask, -1e9).argmax(-1)
    batch = torch.arange(len(rows))
    target_token = tokens[batch, medoid]
    weights = mask.float() / mask.sum(1, keepdim=True)
    target_descriptor = F.normalize(
        (descriptors * weights[..., None]).sum(1), dim=-1, eps=1e-8
    )
    return target_token, target_descriptor, descriptors, mask


def inject_tangent_direction_noise(
    features: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    angle_degrees: float,
) -> torch.Tensor:
    """Apply isotropic canonical-reconstruction noise to unit RADIO directions."""

    values = F.normalize(torch.as_tensor(features).float(), dim=-1, eps=1e-8)
    mask = torch.as_tensor(token_mask, device=values.device).bool()
    if angle_degrees <= 0:
        return values * mask[..., None]
    tangent = torch.randn_like(values)
    tangent = tangent - (tangent * values).sum(-1, keepdim=True) * values
    tangent = F.normalize(tangent, dim=-1, eps=1e-8)
    # Half-normal angular noise matches a non-negative reconstruction error;
    # clipping avoids rare, unphysical augmentation outliers.
    angle = torch.randn(values.shape[:-1], device=values.device).abs().clamp_max(2.0)
    angle = angle * (float(angle_degrees) * torch.pi / 180.0)
    result = values * angle.cos()[..., None] + tangent * angle.sin()[..., None]
    return F.normalize(result, dim=-1, eps=1e-8) * mask[..., None]


def _selection_score(metrics: dict[str, float]) -> float:
    return 0.5 * (
        float(metrics["mean_descriptor_cosine"])
        + float(metrics["all_view_descriptor_cosine"])
    )


def _validate_sparse_support_inputs(
    data: dict,
    *,
    minimum_tokens: int,
    label: str,
) -> None:
    """Fail before training when the declared sparse interval is impossible."""

    region_ids = [str(value) for value in data.get("region_ids", [])]
    rows = len(data["radio_features"])
    if len(region_ids) != rows or any(not value for value in region_ids):
        raise ValueError(f"{label} lacks stable per-region identities")
    if len(set(region_ids)) != len(region_ids):
        raise ValueError(f"{label} contains duplicate region identities")
    mask = torch.as_tensor(data["token_mask"]).bool()
    if mask.shape[0] != rows or bool((mask.sum(1) < int(minimum_tokens)).any()):
        raise ValueError(
            f"{label} contains support below the sparse minimum_tokens"
        )


def _sparsify_inputs(
    data: dict,
    rows: torch.Tensor,
    device: torch.device,
    *,
    minimum_tokens: int,
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SparseTokenSupport]:
    """Select and synchronously zero one sparse surface-region minibatch."""

    row_list = torch.as_tensor(rows).detach().cpu().long().tolist()
    # Keep SHA256/count selection on CPU.  Constructing it on CUDA would add
    # one synchronization per row without moving any useful model work there.
    original_mask = torch.as_tensor(data["token_mask"])[rows].bool()
    anchors = torch.as_tensor(data["anchor_index"])[rows].long()
    host_selection = deterministic_sparse_token_support(
        original_mask,
        anchor_index=anchors,
        region_ids=[str(data["region_ids"][row]) for row in row_list],
        minimum_tokens=int(minimum_tokens),
        seed=int(seed),
        epoch=int(epoch),
    )
    selection = SparseTokenSupport(
        token_mask=host_selection.token_mask.to(device),
        kept_counts=host_selection.kept_counts,
        available_counts=host_selection.available_counts,
    )
    zeroed = selection.zero_tensors(
        {
            "radio_features": torch.as_tensor(data["radio_features"])[rows].to(device),
            "geometry": torch.as_tensor(data["geometry"])[rows].to(device),
            "reliability": torch.as_tensor(data["reliability"])[rows].to(device),
        }
    )
    return (
        zeroed["radio_features"],
        zeroed["geometry"],
        zeroed["reliability"],
        selection,
    )


@torch.no_grad()
def _evaluate(
    model,
    head,
    data,
    device,
    batch_size: int,
    *,
    sparse_minimum_tokens: int | None = None,
    sparse_seed: int = FIXED_SPARSE_VALIDATION_SEED,
    sparse_epoch: int = FIXED_SPARSE_VALIDATION_EPOCH,
    text_bank: torch.Tensor | None = None,
    response_temperature: float = TARGET_BLIND_RESPONSE_TEMPERATURE,
) -> dict:
    token_cos, descriptor_cos, multiview_cos = [], [], []
    predicted_descriptors: list[torch.Tensor] = []
    for start in range(0, len(data["radio_features"]), int(batch_size)):
        rows = torch.arange(start, min(start + int(batch_size), len(data["radio_features"])))
        token, descriptor, all_descriptors, teacher_mask = _targets(data, rows)
        if sparse_minimum_tokens is None:
            radio_features = data["radio_features"][rows].to(device)
            geometry = data["geometry"][rows].to(device)
            reliability = data["reliability"][rows].to(device)
            token_mask = data["token_mask"][rows].to(device)
        else:
            radio_features, geometry, reliability, selection = _sparsify_inputs(
                data,
                rows,
                device,
                minimum_tokens=int(sparse_minimum_tokens),
                seed=int(sparse_seed),
                epoch=int(sparse_epoch),
            )
            token_mask = selection.token_mask
        predicted = model(
            radio_features, geometry,
            anchor_index=data["anchor_index"][rows].to(device),
            token_mask=token_mask,
            reliability=reliability,
        )
        projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1)
        if text_bank is not None:
            predicted_descriptors.append(projected.cpu())
        token_cos.extend(F.cosine_similarity(predicted.cpu(), token, dim=-1).tolist())
        descriptor_cos.extend(F.cosine_similarity(projected.cpu(), descriptor, dim=-1).tolist())
        pair = torch.einsum("bd,bvd->bv", projected.cpu(), all_descriptors)
        multiview_cos.extend(pair[teacher_mask].tolist())
    metrics = {
        "summary_token_cosine": sum(token_cos) / len(token_cos),
        "mean_descriptor_cosine": sum(descriptor_cos) / len(descriptor_cos),
        "all_view_descriptor_cosine": sum(multiview_cos) / len(multiview_cos),
    }
    if text_bank is not None:
        if "scene_ids" not in data:
            raise ValueError("target-blind response validation requires scene IDs")
        bank = F.normalize(torch.as_tensor(text_bank).float().cpu(), dim=-1)
        predicted_descriptor = torch.cat(predicted_descriptors)
        rows = torch.arange(len(predicted_descriptor))
        _, _, all_descriptors, teacher_mask = _targets(data, rows)
        student_response = predicted_descriptor @ bank.T
        teacher_response = latent_query_responses(
            all_descriptors,
            bank,
            mask=teacher_mask,
            temperature=float(response_temperature),
        )
        metrics.update(
            _text_response_metrics(
                student_response,
                teacher_response,
                list(data["scene_ids"]),
            )
        )
    return metrics


def _fixed_sparse_validation_report(
    *,
    minimum_tokens: int,
    baseline_full: dict[str, float],
    baseline_sparse: dict[str, float],
    candidate_full: dict[str, float],
    candidate_sparse: dict[str, float],
) -> dict:
    """Build the parameter-free held-out sparse-support robustness gate."""

    baseline_full_score = _selection_score(baseline_full)
    baseline_sparse_score = _selection_score(baseline_sparse)
    candidate_full_score = _selection_score(candidate_full)
    candidate_sparse_score = _selection_score(candidate_sparse)
    values = (
        baseline_full_score,
        baseline_sparse_score,
        candidate_full_score,
        candidate_sparse_score,
    )
    checks = {
        "all_scores_finite": all(math.isfinite(value) for value in values),
        "sparse_nonregression_from_untrained_baseline": (
            candidate_sparse_score >= baseline_sparse_score
        ),
        "sparse_support_penalty_not_increased": (
            max(0.0, candidate_full_score - candidate_sparse_score)
            <= max(0.0, baseline_full_score - baseline_sparse_score) + 1e-8
        ),
    }
    return {
        "schema": "surface_region_fixed_sparse_validation_v1",
        "sampling": {
            "minimum_tokens": int(minimum_tokens),
            "seed": FIXED_SPARSE_VALIDATION_SEED,
            "epoch": FIXED_SPARSE_VALIDATION_EPOCH,
            "region_key": "stable_region_id",
            "count_distribution": "sha256_keyed_log_uniform_integer",
            "validation_support": "fixed_sparse_diagnostic_only",
            "checkpoint_selection_support": "full",
        },
        "untrained_full": baseline_full,
        "untrained_sparse": baseline_sparse,
        "candidate_full": candidate_full,
        "candidate_sparse": candidate_sparse,
        "untrained_sparse_selection_score": baseline_sparse_score,
        "candidate_sparse_selection_score": candidate_sparse_score,
        "candidate_full_minus_sparse_score": (
            candidate_full_score - candidate_sparse_score
        ),
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def _build_versioned_readout(
    args: argparse.Namespace,
    *,
    contract_version: str,
    device: torch.device,
) -> tuple[SurfaceRegionSummaryReadoutV2 | SurfaceRegionSummaryReadoutV3, dict]:
    """Construct the readout and disclose its effective versioned settings."""

    if contract_version == "surface-region-contract-v2":
        reliability_mode = str(
            getattr(args, "reliability_attention_mode", "log_prior")
        )
        context_mode = str(
            getattr(args, "context_pooling_mode", JOINT_CONTEXT_POOLING)
        )
        model: SurfaceRegionSummaryReadoutV2 | SurfaceRegionSummaryReadoutV3 = (
            SurfaceRegionSummaryReadoutV2(
                hidden_dim=int(args.hidden_dim),
                reliability_attention_mode=reliability_mode,
                context_pooling_mode=context_mode,
            )
        )
        contract = {
            "contract_version": contract_version,
            "checkpoint_schema_version": 3,
            "training_scope": "global_cross_scene_3d_surface_v2",
            "effective_reliability_attention_mode": reliability_mode,
            "effective_context_pooling_mode": context_mode,
        }
    elif contract_version == "surface-region-contract-v3":
        # V2 CLI defaults are deliberately not inherited.  These two choices
        # are immutable parts of the V3 gauge/reliability architecture.
        base_output_mode = str(
            getattr(
                args,
                "v3_base_output_mode",
                SURFACE_REGION_V3_LEGACY_RAW_BASE,
            )
        )
        model = SurfaceRegionSummaryReadoutV3(
            hidden_dim=int(args.hidden_dim),
            reliability_attention_mode="input_only",
            context_pooling_mode=JOINT_CONTEXT_POOLING,
            base_output_mode=base_output_mode,
        )
        contract = {
            "contract_version": contract_version,
            "checkpoint_schema_version": (
                SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION
                if base_output_mode == SURFACE_REGION_V3_GATED_RAW_PRIOR
                else SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION
            ),
            "training_scope": "global_cross_scene_3d_surface_v3",
            "effective_reliability_attention_mode": "input_only",
            "effective_context_pooling_mode": JOINT_CONTEXT_POOLING,
        }
        if base_output_mode == SURFACE_REGION_V3_GATED_RAW_PRIOR:
            contract["effective_base_output_mode"] = base_output_mode
    else:
        raise ValueError(f"unsupported readout contract version: {contract_version}")
    return model.to(device), contract


def train(args: argparse.Namespace) -> dict:
    response_gradient_ratio = float(
        getattr(args, "text_response_gradient_ratio", 0.0)
    )
    if not 0.0 <= response_gradient_ratio <= 1.0:
        raise ValueError("text-response gradient ratio must lie in [0,1]")
    response_enabled = response_gradient_ratio > 0.0
    response_temperature = float(
        getattr(
            args,
            "text_response_temperature",
            TARGET_BLIND_RESPONSE_TEMPERATURE,
        )
    )
    if response_enabled and response_temperature != TARGET_BLIND_RESPONSE_TEMPERATURE:
        raise ValueError("target-blind response temperature is frozen at 0.05")
    sparse_enabled = bool(
        getattr(
            args,
            "sparse_support_augmentation",
            SPARSE_SUPPORT_AUGMENTATION_DEFAULT,
        )
    )
    train_data, train_meta = _load(
        _paths(args.train_caches),
        "train",
        derive_region_ids=sparse_enabled,
    )
    val_data, val_meta = _load(
        _paths(args.validation_caches),
        "validation",
        derive_region_ids=sparse_enabled,
    )
    val_full_data, val_completion_data = _completion_validation_views(
        val_data
    )
    completion_training_rows, completion_training_contract = (
        _eligibility_completion_training_rows(
            train_data,
            completion_weight=float(
                getattr(args, "eligibility_completion_training_weight", 1.0)
            ),
        )
    )
    sparse_minimum_tokens = int(
        train_meta["region_contract"]["minimum_tokens"]
    )
    overlap = set(train_meta["scenes"]) & set(val_meta["scenes"])
    if overlap:
        raise ValueError(f"train/validation scene leakage: {sorted(overlap)}")
    if (
        train_meta["region_contract_version"]
        != val_meta["region_contract_version"]
    ):
        raise ValueError("train/validation region contract versions differ")
    if train_meta["region_contract_sha256"] != val_meta["region_contract_sha256"]:
        raise ValueError("train/validation region contracts differ")
    if (
        train_meta["excluded_physical_spaces"]
        != val_meta["excluded_physical_spaces"]
    ):
        raise ValueError("train/validation benchmark exclusion contracts differ")
    if train_meta["teacher_region"] != val_meta["teacher_region"]:
        raise ValueError("train/validation teacher protocols differ")
    if (
        train_meta["radio_checkpoint_sha256"]
        != val_meta["radio_checkpoint_sha256"]
    ):
        raise ValueError("train/validation RADIO checkpoints differ")
    if (
        _sha256_file(Path(args.radio_checkpoint))
        != train_meta["radio_checkpoint_sha256"]
    ):
        raise ValueError(
            "training RADIO checkpoint differs from cache provenance"
        )
    fit_text: torch.Tensor | None = None
    validation_text: torch.Tensor | None = None
    fit_text_record: dict[str, object] | None = None
    validation_text_record: dict[str, object] | None = None
    if response_enabled:
        if "scene_ids" not in train_data or "scene_ids" not in val_full_data:
            raise ValueError("target-blind response training requires exact scene IDs")
        fit_text, fit_text_record, fit_queries = _load_target_blind_text_bank(
            Path(str(getattr(args, "fit_text_bank", ""))),
            expected_sha256=str(getattr(args, "fit_text_bank_sha256", "")),
            expected_split="fit",
        )
        validation_text, validation_text_record, validation_queries = (
            _load_target_blind_text_bank(
                Path(str(getattr(args, "validation_text_bank", ""))),
                expected_sha256=str(
                    getattr(args, "validation_text_bank_sha256", "")
                ),
                expected_split="dev",
            )
        )
        if set(fit_queries) & set(validation_queries):
            raise ValueError("target-blind fit/dev text queries overlap")
        fit_text, fit_centering = _center_target_blind_text_bank(fit_text)
        validation_text, validation_centering = _center_target_blind_text_bank(
            validation_text
        )
        fit_text_record.update(fit_centering)
        validation_text_record.update(validation_centering)
    if sparse_enabled:
        _validate_sparse_support_inputs(
            train_data,
            minimum_tokens=sparse_minimum_tokens,
            label="training cache",
        )
        _validate_sparse_support_inputs(
            val_full_data,
            minimum_tokens=sparse_minimum_tokens,
            label="full-support validation cache",
        )
        if val_completion_data is not None:
            _validate_sparse_support_inputs(
                val_completion_data,
                minimum_tokens=sparse_minimum_tokens,
                label="completion robustness validation cache",
            )
    device = torch.device(args.device)
    generator = _seed_training(int(args.seed), device=device)
    model, readout_training_contract = _build_versioned_readout(
        args,
        contract_version=str(train_meta["region_contract_version"]),
        device=device,
    )
    head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint).to(device).eval()
    for parameter in head.parameters(): parameter.requires_grad_(False)
    if fit_text is not None:
        fit_text = fit_text.to(device)
    model.eval()
    baseline = _evaluate(
        model,
        head,
        val_full_data,
        device,
        int(args.batch_size),
        text_bank=validation_text,
        response_temperature=response_temperature,
    )
    completion_baseline = (
        _evaluate(
            model,
            head,
            val_completion_data,
            device,
            int(args.batch_size),
            text_bank=validation_text,
            response_temperature=response_temperature,
        )
        if val_completion_data is not None
        else None
    )
    baseline_score = _selection_score(baseline)
    baseline_sparse = (
        _evaluate(
            model,
            head,
            val_full_data,
            device,
            int(args.batch_size),
            sparse_minimum_tokens=sparse_minimum_tokens,
        )
        if sparse_enabled
        else None
    )
    print(json.dumps({"untrained_baseline": baseline, "selection_score": baseline_score}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate),
                                  weight_decay=float(args.weight_decay))
    best_score, best_epoch, best_state, history, stale = -1.0, 0, None, [], 0
    best_key: tuple[float, ...] | None = None
    response_calibration: dict[str, float] | None = None
    response_warmup_epochs = int(
        getattr(args, "text_response_warmup_epochs", 3)
    )
    response_selection_floor = float(
        getattr(args, "text_response_selection_floor", 0.0)
    )
    if response_enabled and (
        response_warmup_epochs < 0
        or not 0.0 <= response_selection_floor <= 1.0
    ):
        raise ValueError("target-blind response warmup/floor is invalid")
    trainable_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    for epoch in range(int(args.epochs)):
        if response_enabled:
            epoch_batches = _scene_complete_epoch_batches(
                completion_training_rows,
                list(train_data["scene_ids"]),
                target_rows=int(args.batch_size),
                generator=generator,
            )
        else:
            order = _training_epoch_order(
                completion_training_rows,
                total_rows=len(train_data["radio_features"]),
                generator=generator,
            )
            epoch_batches = [
                order[start : start + int(args.batch_size)]
                for start in range(0, len(order), int(args.batch_size))
            ]
        losses = []
        response_losses: list[float] = []
        independent_losses: list[float] = []
        listwise_losses: list[float] = []
        hard_negative_losses: list[float] = []
        kept_tokens = 0
        available_tokens = 0
        augmented_regions = 0
        model.train()
        for rows in epoch_batches:
            target_token, target_descriptor, all_descriptors, teacher_mask = _targets(train_data, rows)
            if sparse_enabled:
                (
                    batch_features,
                    batch_geometry,
                    batch_reliability,
                    selection,
                ) = _sparsify_inputs(
                    train_data,
                    rows,
                    device,
                    minimum_tokens=sparse_minimum_tokens,
                    seed=int(args.seed),
                    epoch=epoch,
                )
                token_mask = selection.token_mask
                kept_tokens += int(selection.kept_counts.sum())
                available_tokens += int(selection.available_counts.sum())
                augmented_regions += len(rows)
            else:
                token_mask = train_data["token_mask"][rows].to(device)
                batch_features = train_data["radio_features"][rows].to(device)
                batch_geometry = train_data["geometry"][rows].to(device)
                batch_reliability = train_data["reliability"][rows].to(device)
            radio_features = inject_tangent_direction_noise(
                batch_features, token_mask,
                angle_degrees=float(args.canonical_noise_degrees),
            )
            predicted = model(
                radio_features,
                batch_geometry,
                anchor_index=train_data["anchor_index"][rows].to(device),
                token_mask=token_mask,
                reliability=batch_reliability,
            )
            projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1)
            target_token, target_descriptor = target_token.to(device), target_descriptor.to(device)
            all_descriptors = all_descriptors.to(device)
            teacher_mask = teacher_mask.to(device)
            if response_enabled:
                teacher_tokens = torch.as_tensor(
                    train_data["official_summary_tokens"]
                )[rows].to(device).float()
                token_loss = _view_set_token_loss(
                    predicted,
                    teacher_tokens,
                    teacher_mask,
                )
            else:
                token_loss = (
                    1 - F.cosine_similarity(predicted, target_token, dim=-1)
                ).mean()
            all_view_cosine = torch.einsum("bd,bvd->bv", projected, all_descriptors)
            descriptor_loss = (1 - all_view_cosine)[teacher_mask].mean()
            teacher_rel = target_descriptor @ target_descriptor.T
            predicted_rel = projected @ projected.T
            relation_loss = F.smooth_l1_loss(predicted_rel, teacher_rel)
            main_loss = (float(args.token_weight) * token_loss + descriptor_loss
                         + float(args.relation_weight) * relation_loss)
            if response_enabled:
                assert fit_text is not None
                student_response = projected @ fit_text.T
                teacher_response = latent_query_responses(
                    all_descriptors,
                    fit_text,
                    mask=teacher_mask,
                    temperature=response_temperature,
                )
                independent_loss = F.smooth_l1_loss(
                    student_response, teacher_response
                )
                listwise_loss, hard_negative_loss = (
                    scene_listwise_and_hard_negative_loss(
                        student_response,
                        teacher_response,
                        [
                            str(train_data["scene_ids"][int(row)])
                            for row in rows
                        ],
                    )
                )
                response_loss = (
                    independent_loss + listwise_loss + hard_negative_loss
                ) / 3.0
                if epoch < response_warmup_epochs:
                    loss = main_loss
                else:
                    if response_calibration is None:
                        main_norm = _gradient_norm(
                            main_loss,
                            trainable_parameters,
                            retain_graph=True,
                        )
                        response_norm = _gradient_norm(
                            response_loss,
                            trainable_parameters,
                            retain_graph=True,
                        )
                        response_calibration = {
                            "main_gradient_l2": main_norm,
                            "response_gradient_l2": response_norm,
                            "response_gradient_ratio": response_gradient_ratio,
                            "response_lambda": (
                                response_gradient_ratio
                                * main_norm
                                / response_norm
                            ),
                        }
                        print(
                            json.dumps(
                                {"text_response_gradient_calibration": response_calibration}
                            ),
                            flush=True,
                        )
                    loss = (
                        main_loss
                        + response_calibration["response_lambda"] * response_loss
                    )
                response_losses.append(float(response_loss.detach()))
                independent_losses.append(float(independent_loss.detach()))
                listwise_losses.append(float(listwise_loss.detach()))
                hard_negative_losses.append(float(hard_negative_loss.detach()))
            else:
                loss = main_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval(); metrics = _evaluate(
            model,
            head,
            val_full_data,
            device,
            int(args.batch_size),
            text_bank=validation_text,
            response_temperature=response_temperature,
        )
        score = _selection_score(metrics)
        record = {"epoch": epoch + 1, "loss": sum(losses) / len(losses),
                  "selection_score": score, **metrics}
        if response_enabled:
            record["train_text_response"] = {
                "combined": sum(response_losses) / len(response_losses),
                "independent": sum(independent_losses) / len(independent_losses),
                "listwise": sum(listwise_losses) / len(listwise_losses),
                "hard_negative": (
                    sum(hard_negative_losses) / len(hard_negative_losses)
                ),
            }
        if sparse_enabled:
            record.update(
                {
                    "sparse_support_mean_kept_tokens": (
                        kept_tokens / augmented_regions
                    ),
                    "sparse_support_kept_fraction": (
                        kept_tokens / available_tokens
                    ),
                }
            )
        history.append(record); print(json.dumps(record), flush=True)
        if response_enabled:
            feasible = score >= response_selection_floor
            key = (
                float(metrics["support_top1_agreement"]),
                float(metrics["ranking_spearman_p05"]),
                float(metrics["response_profile_cosine_p05"]),
                score,
                -float(metrics["text_response_smooth_l1"]),
            )
            improved = feasible and (best_key is None or key > best_key)
        else:
            key = (score,)
            improved = score > best_score
        if improved:
            best_score, best_epoch, best_key, stale = score, epoch + 1, key, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif best_state is not None:
            # A feasibility floor must not consume early-stopping patience
            # before the first eligible checkpoint exists.  Otherwise a
            # healthy warm-up can terminate solely because it starts below
            # the frozen noninferiority threshold.
            stale += 1
        if int(args.patience) and stale >= int(args.patience): break
    if best_state is None:
        raise RuntimeError(
            "no checkpoint met the target-blind response base-selection floor"
        )
    if response_enabled and response_calibration is None:
        raise RuntimeError("target-blind response training never reached calibration")
    model.load_state_dict(best_state)
    final_validation = _evaluate(
        model,
        head,
        val_full_data,
        device,
        int(args.batch_size),
        text_bank=validation_text,
        response_temperature=response_temperature,
    )
    completion_validation = None
    if val_completion_data is not None:
        assert completion_baseline is not None
        candidate_completion = _evaluate(
            model,
            head,
            val_completion_data,
            device,
            int(args.batch_size),
            text_bank=validation_text,
            response_temperature=response_temperature,
        )
        completion_validation = {
            "authority": "diagnostic_only_never_checkpoint_selection",
            "rows": len(val_completion_data["radio_features"]),
            "untrained": completion_baseline,
            "candidate": candidate_completion,
            "untrained_selection_score": _selection_score(
                completion_baseline
            ),
            "candidate_selection_score": _selection_score(
                candidate_completion
            ),
            "nonregression_gate_passed": (
                _selection_score(candidate_completion)
                >= _selection_score(completion_baseline)
            ),
        }
    target_blind_generic_gate = None
    if response_enabled:
        completion_nonregression = (
            completion_validation is None
            or bool(completion_validation["nonregression_gate_passed"])
        )
        target_blind_generic_gate = _target_blind_generic_gate(
            final_validation,
            completion_nonregression=completion_nonregression,
        )
    sparse_validation = None
    if sparse_enabled:
        assert baseline_sparse is not None
        candidate_sparse = _evaluate(
            model,
            head,
            val_full_data,
            device,
            int(args.batch_size),
            sparse_minimum_tokens=sparse_minimum_tokens,
        )
        sparse_validation = _fixed_sparse_validation_report(
            minimum_tokens=sparse_minimum_tokens,
            baseline_full=baseline,
            baseline_sparse=baseline_sparse,
            candidate_full=final_validation,
            candidate_sparse=candidate_sparse,
        )
    architecture = model.architecture(train_meta["region_contract_sha256"])
    sparse_support_contract = {
        "enabled": sparse_enabled,
        "sampling": "sha256_seed_epoch_region_id_log_uniform_v1",
        "minimum_tokens_source": "region_contract.minimum_tokens",
        "minimum_tokens": sparse_minimum_tokens,
        "anchor_preserved": True,
        "token_order": "existing_contract_geodesic_order",
        "synchronously_zeroed": [
            "radio_features",
            "geometry",
            "reliability",
        ],
        "validation_model_selection_support": "full",
        "fixed_sparse_validation": (
            {
                "seed": FIXED_SPARSE_VALIDATION_SEED,
                "epoch": FIXED_SPARSE_VALIDATION_EPOCH,
            }
            if sparse_enabled
            else None
        ),
    }
    completion_role_mask = torch.tensor(
        [
            role == "eligibility_completion"
            for role in train_data["row_roles"]
        ],
        dtype=torch.bool,
    )
    support_fill_mask = train_data.get("support_fill_mask")
    completion_fill_tokens = (
        int(
            torch.as_tensor(support_fill_mask)[completion_role_mask].sum()
        )
        if support_fill_mask is not None
        else 0
    )
    completion_selected_tokens = int(
        torch.as_tensor(train_data["token_mask"])[completion_role_mask].sum()
    )
    completion_training_contract.update(
        {
            "enabled": bool(
                train_meta["eligibility_completion"]["enabled"]
            ),
            # These legacy names now intentionally mean rows that actually
            # enter each epoch, not merely rows present in the cache.
            "full_support_training_rows": completion_training_contract[
                "full_support_rows_sampled_per_epoch"
            ],
            "completion_training_rows": completion_training_contract[
                "completion_rows_sampled_per_epoch"
            ],
            "completion_training_row_fraction": (
                int(
                    completion_training_contract[
                        "completion_rows_sampled_per_epoch"
                    ]
                )
                / int(
                    completion_training_contract[
                        "total_rows_sampled_per_epoch"
                    ]
                )
            ),
            "completion_training_fill_token_fraction": (
                completion_fill_tokens / completion_selected_tokens
                if completion_selected_tokens
                and float(
                    completion_training_contract[
                        "completion_sampling_weight"
                    ]
                )
                > 0.0
                else 0.0
            ),
            "completion_cache_fill_token_fraction": (
                completion_fill_tokens / completion_selected_tokens
                if completion_selected_tokens
                else 0.0
            ),
            "epochs_trained": len(history),
            "actual_full_support_row_samples": (
                int(
                    completion_training_contract[
                        "full_support_rows_sampled_per_epoch"
                    ]
                )
                * len(history)
            ),
            "actual_completion_row_samples": (
                int(
                    completion_training_contract[
                        "completion_rows_sampled_per_epoch"
                    ]
                )
                * len(history)
            ),
        }
    )
    provenance = {
        "training_scope": readout_training_contract["training_scope"],
        "frozen": True,
        "uses_benchmark_scenes": False, "uses_benchmark_test_vocabulary": False,
        "train": train_meta, "validation": val_meta,
        "scene_disjoint": True, "official_summary_head": "c-radio_v4 siglip2-g",
        "custom_text_projection": False,
        "region_contract_sha256": train_meta["region_contract_sha256"],
        "region_contract": train_meta["region_contract"],
        "canonical_direction_noise_degrees": float(args.canonical_noise_degrees),
        "canonical_noise_calibration": str(args.canonical_noise_calibration),
        "sparse_support_augmentation": sparse_support_contract,
        "eligibility_completion_training": completion_training_contract,
        "random_seed_contract": {
            "seed": int(args.seed),
            "model_initialization": True,
            "data_order": True,
            "canonical_noise": True,
            "sparse_support": sparse_enabled,
        },
    }
    if response_enabled:
        assert fit_text_record is not None
        assert validation_text_record is not None
        assert response_calibration is not None
        provenance["target_blind_text_response"] = {
            "schema_version": 1,
            "benchmark_vocabulary_opened": False,
            "teacher": (
                "temperature_0.05_logmeanexp_over_frozen_official_view_descriptors"
            ),
            "student": "single_query_invariant_official_descriptor_response",
            "text_direction_gauge": (
                "normalize(l2_text_direction_minus_bank_mean)_v1"
            ),
            "token_target": "equal_region_equal_valid_view_cosine_set_v1",
            "scene_batching": "complete_scene_deterministic_packing_v1",
            "response_losses": [
                "row_local_smooth_l1",
                "within_scene_listwise_kl",
                "teacher_top_runner_up_gap_smooth_l1",
            ],
            "fit_text_bank": fit_text_record,
            "validation_text_bank": validation_text_record,
            "fit_dev_query_disjoint": True,
            "temperature": response_temperature,
            "warmup_epochs": response_warmup_epochs,
            "gradient_calibration": response_calibration,
            "checkpoint_selection_key": list(best_key or ()),
            "generic_gate": target_blind_generic_gate,
            "preregistration_addendum": {
                "path": str(TARGET_BLIND_ADDENDUM),
                "sha256": _sha256_file(TARGET_BLIND_ADDENDUM),
            },
        }
    if train_meta["region_contract_version"] == "surface-region-contract-v3":
        provenance["surface_region_v3"] = {
            "cache_geometry_dim": 16,
            "support_fill_mask_validated": True,
            "padding_exact_zero_validated": [
                "radio_features",
                "geometry",
                "reliability",
            ],
            "all_regions_minimum_satisfied": True,
            "feature_normalization": train_meta["region_contract"][
                "feature_normalization"
            ],
            "effective_reliability_attention_mode": (
                readout_training_contract[
                    "effective_reliability_attention_mode"
                ]
            ),
            "effective_context_pooling_mode": readout_training_contract[
                "effective_context_pooling_mode"
            ],
        }
        if (
            readout_training_contract.get("effective_base_output_mode")
            == SURFACE_REGION_V3_GATED_RAW_PRIOR
        ):
            provenance["surface_region_v3"]["effective_base_output_mode"] = (
                SURFACE_REGION_V3_GATED_RAW_PRIOR
            )
    training_config = dict(vars(args))
    if not response_enabled:
        for key in (
            "fit_text_bank",
            "fit_text_bank_sha256",
            "validation_text_bank",
            "validation_text_bank_sha256",
            "text_response_gradient_ratio",
            "text_response_warmup_epochs",
            "text_response_selection_floor",
            "text_response_temperature",
        ):
            training_config.pop(key, None)
    if (
        str(
            training_config.get(
                "v3_base_output_mode",
                SURFACE_REGION_V3_LEGACY_RAW_BASE,
            )
        )
        == SURFACE_REGION_V3_LEGACY_RAW_BASE
    ):
        # Keep the pre-schema8/default schema7 payload byte-contract free of
        # the opt-in experiment knob.  The legacy architecture serialization,
        # state dict, training contract, and training config remain unchanged.
        training_config.pop("v3_base_output_mode", None)
    payload = {"schema_version": readout_training_contract["checkpoint_schema_version"],
               "architecture": architecture,
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
               "provenance": provenance, "history": history, "best_epoch": best_epoch,
               "best_selection_score": best_score, "untrained_baseline": baseline,
               "untrained_baseline_score": baseline_score,
               "sparse_validation": sparse_validation,
               "eligibility_completion_validation": completion_validation,
               "target_blind_generic_gate": target_blind_generic_gate,
               "training_config": training_config}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {"output": str(output.resolve()), "checkpoint_sha256": digest,
              "architecture": architecture, "best_epoch": best_epoch,
              "checkpoint_schema_version": readout_training_contract[
                  "checkpoint_schema_version"
              ],
              "training_scope": readout_training_contract["training_scope"],
              "best_selection_score": best_score,
              "untrained_baseline": baseline,
              "selection_score_delta": best_score - baseline_score,
              "validation": final_validation,
              "eligibility_completion_validation": completion_validation,
              "target_blind_generic_gate": target_blind_generic_gate,
              "eligibility_completion_training": completion_training_contract,
              "sparse_support_augmentation": sparse_support_contract,
              "sparse_validation": sparse_validation,
              "train_scenes": len(train_meta["scenes"]),
              "validation_scenes": len(val_meta["scenes"]), "scene_overlap": []}
    report_path = output.with_suffix(output.suffix + ".json")
    write_frozen_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument(
        "--reliability-attention-mode",
        choices=("log_prior", "input_only"),
        default="log_prior",
        help=(
            "Keep the frozen multiplicative confidence prior or use reliability "
            "only through the geometry input to avoid train/inference prior shift."
        ),
    )
    parser.add_argument(
        "--context-pooling-mode",
        choices=(JOINT_CONTEXT_POOLING, SEPARATE_CONTEXT_POOLING),
        default=JOINT_CONTEXT_POOLING,
        help=(
            "Keep legacy joint attention or preserve the core base while "
            "pooling context as a separately normalized conditioning stream."
        ),
    )
    parser.add_argument("--canonical-noise-degrees", type=float, default=0.0)
    parser.add_argument(
        "--canonical-noise-calibration", default="",
        help="Frozen-field angular-residual audit used to set the augmentation",
    )
    parser.add_argument(
        "--sparse-support-augmentation",
        action="store_true",
        default=SPARSE_SUPPORT_AUGMENTATION_DEFAULT,
        help=(
            "Enable deterministic region-ID-keyed log-uniform token support "
            "during training. Disabled by default; full-support validation "
            "remains the checkpoint-selection authority."
        ),
    )
    parser.add_argument(
        "--eligibility-completion-training-weight",
        type=float,
        choices=(0.0, 1.0),
        default=1.0,
        help=(
            "Query-free generic diagnostic control: 1.0 preserves uniform "
            "training over full-support and eligibility-completion rows; 0.0 "
            "trains on full-support rows only. Validation always selects on "
            "full support and reports completion variants diagnostically."
        ),
    )
    parser.add_argument(
        "--v3-base-output-mode",
        choices=(
            SURFACE_REGION_V3_LEGACY_RAW_BASE,
            SURFACE_REGION_V3_GATED_RAW_PRIOR,
        ),
        default=SURFACE_REGION_V3_LEGACY_RAW_BASE,
        help=(
            "V3-only query-free output composition. The default preserves "
            "schema-7 fixed raw-base checkpoints exactly; the gated mode "
            "uses schema 8 and learns one global sigmoid weight for the "
            "pooled raw-amplitude prior."
        ),
    )
    parser.add_argument(
        "--fit-text-bank",
        default="",
        help="Immutable target-blind ImageNet fit embedding artifact.",
    )
    parser.add_argument("--fit-text-bank-sha256", default="")
    parser.add_argument(
        "--validation-text-bank",
        default="",
        help="Disjoint immutable target-blind ImageNet dev embedding artifact.",
    )
    parser.add_argument("--validation-text-bank-sha256", default="")
    parser.add_argument(
        "--text-response-gradient-ratio",
        type=float,
        default=0.0,
        help=(
            "Opt-in target-blind response objective gradient ratio. Zero keeps "
            "the historical trainer path unchanged."
        ),
    )
    parser.add_argument("--text-response-warmup-epochs", type=int, default=3)
    parser.add_argument(
        "--text-response-selection-floor",
        type=float,
        default=TARGET_BLIND_BASE_SELECTION_MIN,
    )
    parser.add_argument(
        "--text-response-temperature",
        type=float,
        default=TARGET_BLIND_RESPONSE_TEMPERATURE,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radio-checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    args = parser.parse_args(); print(json.dumps(train(args), indent=2))


if __name__ == "__main__": main()
