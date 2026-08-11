#!/usr/bin/env python3
"""Train one compact, query-independent canonical RADIO field from MPR targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from radio_gs.field import (
    CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    CANONICAL_OBSERVATION_CONTRACT_NAME,
    CanonicalGaussianField,
    FeatureSpaceSignature,
    canonical_observation_contract,
    fit_affine_basis,
    load_canonical_field_checkpoint,
    validate_basis_conditioning,
    validate_observation_contract_metadata,
    observation_contract_sha256,
)
from radio_gs.field.checkpoint import load_factorized_canonical_field_checkpoint
from radio_gs.field.factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION,
    CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
    CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
    FactorizedRadioFieldSignature,
    canonical_factorized_radio_contract,
    factorized_radio_checkpoint_metadata,
)
from radio_gs.interfaces.frozen_radio_views import FrozenRadioViews
from radio_gs.rendering.contribution_compositor import (
    EXACT_CENTER_UNCERTAINTY_CONTRACT,
    MARGINAL_RESPONSIBILITY_CONTRACT,
)
from radio_gs.training.canonical_field_losses import (
    CanonicalFieldLossConfig,
    canonical_primitive_loss,
    hard_boundary_relation_ranking_loss,
)
from radio_gs.training.primitive_consensus import (
    PrimitiveConsensus,
    consensus_target_rows,
)
from radio_gs.training.tensor_cache_io import ShardedMPRCache, load_mpr_cache
from radio_gs.training.factorized_radio_cache import (
    CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2,
    FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
    FactorizedRadioTrainingCache,
    canonical_factorized_radio_builder_contract_v2,
    factorized_radio_builder_contract_v2_sha256,
    load_factorized_radio_training_cache,
)
from radio_gs.training.factorized_radio_loss import (
    FACTORIZED_RADIO_EXACT_MARGINAL_VISIBILITY_SAFE_LOSS_CONTRACT,
    FACTORIZED_RADIO_RECONSTRUCTION_LOSS_CONTRACT,
    FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY,
    FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE,
)
from radio_gs.training.source_only_multiteacher import (
    SOURCE_DESCRIPTOR_WEIGHT,
    SOURCE_RELATION_BATCH_SIZE,
    SOURCE_RELATION_WEIGHT,
    FrozenPrimitiveSiglipProxy,
    evaluate_source_only_gates,
    load_source_only_multiteacher_bundle,
    primitive_source_descriptor_loss,
    source_descriptor_metrics,
    source_relation_batch_loss,
    source_relation_metrics,
)
from radio_gs.training.source_only_sam_structure import (
    SAM_STRUCTURE_BATCH_SIZE,
    SAM_STRUCTURE_WEIGHT,
    evaluate_source_sam_structure_gates,
    load_source_only_sam_structure_bundle,
    source_sam_structure_batch_loss,
    source_sam_structure_metrics,
    validate_single_radio_checkpoint_payload,
)
from radio_gs.training.source_only_sam_relative_structure import (
    SAM_RELATIVE_GUARD_BATCH_SIZE,
    SAM_RELATIVE_TRIPLET_BATCH_SIZE,
    evaluate_source_sam_relative_full_gates,
    evaluate_source_sam_relative_gates,
    load_source_only_sam_relative_bundle,
    source_sam_relative_batch_loss,
    source_sam_relative_metrics,
)
from radio_gs.training.source_only_sam_capability_relative_structure import (
    evaluate_source_sam_capability_relative_full_gates,
    evaluate_source_sam_capability_relative_gates,
    load_source_only_sam_capability_relative_bundle,
    source_sam_capability_pair_metrics,
    source_sam_capability_relative_batch_loss,
    source_sam_capability_relative_metrics,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


CAPABILITY_TARGET_CONTRACT_MATCHED_TOP1 = "matched_top1"
CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL = "matched_exact_marginal"
CAPABILITY_TARGET_CONTRACT_FIELD_A = "field_a_exact_adjoint"
CAPABILITY_TARGET_CONTRACT_FIELD_C = "field_c_exact_center_uncertainty"
LEGACY_FACTORIZED_CAPABILITY_RECEIPT_EXPERIMENT = (
    "canonical-factorized-radio-v1-label-free-source-gate"
)
FORMAL_FACTORIZED_CAPABILITY_RECEIPT_EXPERIMENT = (
    "canonical-factorized-radio-v1-formal-capability-cohort"
)
RELATION_OBJECTIVE_DISABLED = "disabled"
RELATION_OBJECTIVE_FIELD_B = "field_b_boundary_ranking_v1"
FIELD_B_RELATION_WEIGHT = CanonicalFieldLossConfig().relation_weight
STRICT_OBSERVATION_CONTRACT_MODES = frozenset(
    {
        CANONICAL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    }
)


def _factorized_loss_reliability_policy(
    factorized: FactorizedRadioTrainingCache | None,
    *,
    capability_target_contract: str,
) -> str:
    """Bind visibility weighting to one source-only exact-marginal authority."""

    if factorized is None or (
        str(capability_target_contract) != CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL
    ):
        return FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY
    metadata = factorized.metadata
    registration_sha256 = str(
        metadata.get("registration_responsibility_cache_sha256", "")
    )
    expected_authority = {
        **FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
        "registration_responsibility_cache_sha256": registration_sha256,
    }
    if (
        factorized.provenance().get("storage")
        != CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2
        or metadata.get("builder_contract")
        != canonical_factorized_radio_builder_contract_v2()
        or metadata.get("builder_contract_sha256")
        != factorized_radio_builder_contract_v2_sha256()
        or re.fullmatch(r"[0-9a-f]{64}", registration_sha256) is None
        or metadata.get("visibility_purity_authority") != expected_authority
        or metadata.get("query_independent") is not True
        or any(
            metadata.get(name) is not False
            for name in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError(
            "matched exact-marginal visibility-safe loss requires its frozen "
            "source-only measured-purity authority"
        )
    return FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE


def _validate_training_observation_contract_metadata(
    metadata: dict,
    mode: str,
) -> None:
    """Apply a canonical contract only when the caller requested one.

    ``compatible-legacy`` still receives the deep tensor, hash, geometry, and
    contamination checks in ``load_mpr_cache``.  It must not silently be
    interpreted as the default canonical lifting contract.
    """

    requested = str(mode)
    if requested in STRICT_OBSERVATION_CONTRACT_MODES:
        validate_observation_contract_metadata(
            metadata,
            require_declaration=True,
            contract_name=requested,
        )
    elif requested not in {"compatible-legacy", "unchecked"}:
        raise ValueError(f"unsupported training observation contract: {requested}")


def _assert_initial_field_signature_compatible(
    expected: FeatureSpaceSignature,
    actual: FeatureSpaceSignature,
    *,
    allow_legacy_normalization: bool,
) -> dict:
    expected_values = expected.to_dict()
    actual_values = actual.to_dict()
    expected_values.pop("field_checkpoint_sha256", None)
    actual_values.pop("field_checkpoint_sha256", None)
    mismatches = {
        key: [expected_values.get(key), actual_values.get(key)]
        for key in sorted(set(expected_values) | set(actual_values))
        if expected_values.get(key) != actual_values.get(key)
    }
    if not mismatches:
        return {"legacy_normalization_mismatch_accepted": False}
    if (
        allow_legacy_normalization
        and set(mismatches) == {"normalization"}
        and mismatches["normalization"] == ["radio_direction_unit", "none"]
    ):
        return {
            "legacy_normalization_mismatch_accepted": True,
            "mpr_declared_normalization": "radio_direction_unit",
            "initial_checkpoint_recorded_normalization": "none",
            "checkpoint_signature_preserved": True,
            "feature_values_modified": False,
        }
    expected.assert_compatible(
        actual,
        allow_field_checkpoint_difference=True,
    )
    raise AssertionError("unreachable signature compatibility path")


def _load_field_b_relation_triplets(
    path: str | Path,
    *,
    expected_sha256: str,
    num_rows: int,
    capability_valid: torch.Tensor,
    geometry_fingerprint: dict,
    expected_dino_sha256: str,
    expected_sam3_sha256: str,
) -> tuple[dict[str, torch.Tensor], dict]:
    if not expected_sha256:
        raise ValueError("Field-B relation cache requires a trusted SHA-256")
    payload, digest, source = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="Field-B relation triplet cache",
    )
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Field-B relation cache schema differs")
    pairs = torch.as_tensor(payload.get("pair_index")).long().cpu()
    margin = torch.as_tensor(payload.get("teacher_margin")).float().cpu().reshape(-1)
    channel = (
        torch.as_tensor(payload.get("boundary_channel"))
        .to(torch.uint8)
        .cpu()
        .reshape(-1)
    )
    if (
        pairs.ndim != 2
        or pairs.shape != (2, 2 * margin.numel())
        or channel.shape != margin.shape
    ):
        raise ValueError("Field-B pair/margin/channel tensors do not align")
    if (
        margin.numel() == 0
        or not bool(torch.isfinite(margin).all())
        or bool(((margin <= 0) | (margin > 1)).any())
    ):
        raise ValueError("Field-B teacher margins must be non-empty in (0,1]")
    if bool((pairs < 0).any()) or int(pairs.max()) >= int(num_rows):
        raise ValueError("Field-B pair rows are outside the canonical field")
    count = margin.numel()
    positive = pairs[:, :count]
    negative = pairs[:, count:]
    if (
        not torch.equal(positive[0], negative[0])
        or bool((positive[0] == positive[1]).any())
        or bool((negative[0] == negative[1]).any())
        or bool((positive[1] == negative[1]).any())
    ):
        raise ValueError("Field-B triplet topology is malformed")
    valid = torch.as_tensor(capability_valid).bool().cpu().reshape(-1)
    if valid.shape != (int(num_rows),) or not bool(valid[pairs].all()):
        raise ValueError("Field-B triplets leave exact capability-valid rows")
    metadata = dict(payload.get("metadata", {}))
    expected_policy = {
        "schema_version": "canonical_field_b_boundary_relation_triplets_v1",
        "construction": "exact_capability_local_hard_boundary_ranking_v1",
        "neighbors": 16,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "fixed_scalar_margin": None,
    }
    mismatched = [
        key for key, value in expected_policy.items() if metadata.get(key) != value
    ]
    if mismatched:
        raise ValueError(f"Field-B relation policy differs: {mismatched}")
    if metadata.get("geometry_fingerprint") != geometry_fingerprint:
        raise ValueError("Field-B relation geometry differs")
    if (
        dict(metadata.get("dino_mpr", {})).get("sha256") != expected_dino_sha256
        or dict(metadata.get("sam3_mpr", {})).get("sha256") != expected_sam3_sha256
    ):
        raise ValueError("Field-B relation exact capability targets differ")
    return {
        "pair_index": pairs,
        "teacher_margin": margin,
        "boundary_channel": channel,
    }, {
        "path": str(source),
        "sha256": digest,
        "triplets": int(count),
        "metadata": metadata,
    }


def _field_b_relation_batch_loss(
    field: CanonicalGaussianField,
    official_views: FrozenRadioViews,
    relation_cache: dict[str, torch.Tensor],
    triplet_indices: torch.Tensor,
) -> torch.Tensor:
    triplets = int(relation_cache["teacher_margin"].numel())
    selected = torch.as_tensor(triplet_indices).long().cpu()
    if selected.numel() == 0:
        return field.local_codes.sum() * 0.0
    columns = torch.cat([selected, selected + triplets])
    global_pairs = relation_cache["pair_index"][:, columns]
    unique_rows, inverse = torch.unique(
        global_pairs.reshape(-1), sorted=True, return_inverse=True
    )
    local_pairs = inverse.reshape_as(global_pairs).to(field.local_codes.device)
    predicted_radio = field.radio_features(unique_rows.to(field.local_codes.device))
    predicted_dino = official_views.project_dino_primitives(predicted_radio)
    predicted_sam3 = official_views.project_sam3_primitives(predicted_radio)
    return hard_boundary_relation_ranking_loss(
        predicted_dino,
        predicted_sam3,
        local_pairs,
        relation_cache["teacher_margin"][selected],
    )


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _target_shape(
    consensus: PrimitiveConsensus | ShardedMPRCache,
) -> tuple[int, int]:
    if isinstance(consensus, ShardedMPRCache):
        return consensus.shape
    return int(consensus.targets.shape[0]), int(consensus.targets.shape[1])


def _basis_fit_values(
    consensus: PrimitiveConsensus | ShardedMPRCache,
    valid_rows: torch.Tensor,
    *,
    max_samples: int,
    seed: int,
) -> torch.Tensor:
    """Materialize only the exact frozen PCA sample for a sharded target."""

    if not isinstance(consensus, ShardedMPRCache):
        return consensus.targets[valid_rows]
    selected = valid_rows
    if selected.numel() > int(max_samples):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        order = torch.randperm(selected.numel(), generator=generator)[
            : int(max_samples)
        ]
        selected = selected[order]
    return consensus.fetch_rows(selected).float()


@torch.no_grad()
def _initialize_codes_streaming(
    field: CanonicalGaussianField,
    decoder,
    consensus: PrimitiveConsensus | ShardedMPRCache,
    *,
    batch_size: int = 4096,
) -> None:
    num_rows, _feature_dim = _target_shape(consensus)
    device = field.local_codes.device
    for start in range(0, num_rows, int(batch_size)):
        stop = min(start + int(batch_size), num_rows)
        rows = torch.arange(start, stop, dtype=torch.long)
        target = consensus_target_rows(consensus, rows).to(device).float()
        field.local_codes[start:stop].copy_(decoder.encode(target))


def _consensus_from_cache(
    cache: dict | ShardedMPRCache,
    *,
    preserve_target_dtype: bool = False,
) -> PrimitiveConsensus | ShardedMPRCache:
    if isinstance(cache, ShardedMPRCache):
        return cache
    targets = torch.as_tensor(cache["features"]).cpu()
    if not preserve_target_dtype:
        targets = targets.float()
    valid = torch.as_tensor(cache["valid"]).bool().cpu()
    counts = torch.as_tensor(cache["view_counts"]).long().cpu()
    reliability = cache.get("reliability")
    if reliability is None:
        maximum = max(1, int(counts.max()) if counts.numel() else 1)
        reliability = torch.stack(
            [counts.float() / maximum, valid.float(), valid.float()], dim=-1
        )
    else:
        reliability = torch.as_tensor(reliability).float().cpu()
    metadata = dict(cache.get("metadata", {}))
    if metadata.get("aggregation_mode") in {
        "raster_marginal_responsibility",
        "raster_exact_center_uncertainty",
    }:
        expected_contract = (
            MARGINAL_RESPONSIBILITY_CONTRACT
            if metadata.get("aggregation_mode") == "raster_marginal_responsibility"
            else EXACT_CENTER_UNCERTAINTY_CONTRACT
        )
        if metadata.get("marginal_responsibility_contract") != expected_contract:
            raise ValueError("marginal MPR responsibility contract differs")
        purity = torch.as_tensor(cache.get("visibility_purity")).float().cpu()
        if purity.shape != valid.shape:
            raise ValueError("marginal MPR visibility purity does not align")
        reliability = reliability.clone()
        reliability[:, 2] = purity
    return PrimitiveConsensus(
        targets=targets,
        valid=valid,
        observation_count=counts,
        reliability=reliability,
        per_view_agreement=torch.empty(0, targets.shape[0]),
    )


def _load_capability_mpr_target(
    path: str | Path,
    *,
    expected_space: str,
    raw_cache: dict,
    raw_metadata: dict,
    radio_checkpoint_sha256: str,
    expected_cache_sha256: str = "",
    expected_feature_output_bundle_sha256: str = "",
    target_contract: str = CAPABILITY_TARGET_CONTRACT_MATCHED_TOP1,
) -> tuple[PrimitiveConsensus, dict]:
    """Load an official-adaptor-before-MPR target with strict provenance."""

    cache, cache_sha256, cache_path = load_mpr_cache(
        path,
        expected_sha256=str(expected_cache_sha256) or None,
        expected_feature_space=expected_space,
        require_reliability=True,
        require_formal_safety=True,
    )
    metadata = dict(cache.get("metadata", {}))
    target_contract = str(target_contract)
    if target_contract == CAPABILITY_TARGET_CONTRACT_MATCHED_TOP1:
        raw_contract = raw_metadata.get("observation_lifting_contract", {})
        raw_contract_name = (
            str(raw_contract.get("name", CANONICAL_OBSERVATION_CONTRACT_NAME))
            if isinstance(raw_contract, dict)
            else CANONICAL_OBSERVATION_CONTRACT_NAME
        )
        validate_observation_contract_metadata(
            metadata,
            require_declaration="observation_lifting_contract" in raw_metadata,
            contract_name=raw_contract_name,
        )
    elif target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_A:
        exact_policy = {
            "aggregation_mode": "raster_adjoint",
            "raster_view_fusion": "contribution_mean",
            "raster_reliability_mode": "mean_resultant",
            "normalize_each_view": True,
            "per_view_normalization_applied": True,
            "per_view_normalization_stage": ("pixel_feature_before_raster_lifting"),
        }
        mismatched_exact = [
            key
            for key, expected in exact_policy.items()
            if metadata.get(key) != expected
        ]
        if mismatched_exact:
            raise ValueError(
                f"{expected_space} Field-A exact capability policy differs: "
                f"{sorted(mismatched_exact)}"
            )
        if raw_metadata.get("aggregation_mode") != "raster_adjoint":
            raise ValueError(
                "Field-A capability observation reference must use raster_adjoint"
            )
    elif target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_C:
        exact_policy = {
            "aggregation_mode": "raster_exact_center_uncertainty",
            "registration_weight_mode": ("exact_front_to_back_adjoint_center"),
            "raster_view_fusion": "contribution_mean",
            "raster_reliability_mode": "mean_resultant",
            "normalize_each_view": True,
            "per_view_normalization_applied": True,
            "per_view_normalization_stage": ("pixel_feature_before_raster_lifting"),
            "marginal_responsibility_contract": (EXACT_CENTER_UNCERTAINTY_CONTRACT),
            "visibility_uncertainty_semantics": (
                "per_primitive_sum_weight_times_responsibility_over_sum_weight"
            ),
            "alpha_threshold": 0.0,
        }
        mismatched_exact = [
            key
            for key, expected in exact_policy.items()
            if metadata.get(key) != expected
        ]
        if mismatched_exact:
            raise ValueError(
                f"{expected_space} Field-C marginal capability policy differs: "
                f"{sorted(mismatched_exact)}"
            )
        if raw_metadata.get("aggregation_mode") != "raster_exact_center_uncertainty":
            raise ValueError(
                "Field-C raw observation target must use marginal responsibility"
            )
    else:
        raise ValueError(f"unsupported capability target contract: {target_contract}")
    if str(metadata.get("feature_space", "")) != str(expected_space):
        raise ValueError(f"expected a {expected_space} MPR cache")
    safety = {
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "capability_projection_before_mpr": True,
        "custom_adaptor_head": False,
    }
    for key, expected in safety.items():
        if metadata.get(key) is not expected:
            raise ValueError(f"{expected_space} MPR violates safety contract: {key}")
    if str(metadata.get("official_adaptor_checkpoint_sha256", "")) != str(
        radio_checkpoint_sha256
    ):
        raise ValueError(f"{expected_space} MPR uses another RADIO checkpoint")
    capability_map_source = str(metadata.get("capability_map_source", "project_raw"))
    if capability_map_source not in {"project_raw", "official_extracted"}:
        raise ValueError(
            f"{expected_space} MPR has an unsupported capability map source "
            f"{capability_map_source!r}"
        )
    if target_contract in {
        CAPABILITY_TARGET_CONTRACT_FIELD_A,
        CAPABILITY_TARGET_CONTRACT_FIELD_C,
    }:
        expected_bundle = str(expected_feature_output_bundle_sha256 or "")
        if re.fullmatch(r"[0-9a-f]{64}", expected_bundle) is None:
            raise ValueError(
                f"{target_contract} requires a trusted feature output bundle SHA-256"
            )
        if metadata.get("feature_output_bundle_sha256") != expected_bundle:
            raise ValueError(
                f"{expected_space} {target_contract} target belongs to another feature "
                "output bundle"
            )
        if capability_map_source != "project_raw":
            raise ValueError(
                f"{expected_space} {target_contract} preregistration requires project_raw"
            )
    if capability_map_source == "official_extracted":
        if (
            str(
                metadata.get(
                    "official_adaptor_checkpoint_provenance",
                    "",
                )
            )
            != "explicit_file_sha256"
        ):
            raise ValueError(
                f"{expected_space} direct official MPR is not bound to the "
                "extraction checkpoint SHA256"
            )
        required_native_provenance = {
            "capability_native_map_manifest",
            "capability_native_map_manifest_sha256",
            "capability_native_map_radio_checkpoint_load_contract",
            "capability_adaptor_execution",
        }
        missing_native_provenance = sorted(
            key for key in required_native_provenance if not str(metadata.get(key, ""))
        )
        if missing_native_provenance:
            raise ValueError(
                f"{expected_space} direct official MPR lacks native-map provenance: "
                f"{missing_native_provenance}"
            )
        if (
            str(metadata.get("capability_adaptor_execution", ""))
            != "official_c_radio_runtime_adaptor_output"
        ):
            raise ValueError(
                f"{expected_space} direct official MPR did not use the official "
                "C-RADIO runtime adaptor output"
            )
        if (
            metadata.get("capability_native_map_radio_checkpoint_load_contract")
            != "external_sha256_same_fd_restricted_pickle_hub_injection_v1"
        ):
            raise ValueError(
                f"{expected_space} direct official MPR used an unrestricted "
                "RADIO checkpoint loader"
            )
        expected_bundle = str(expected_feature_output_bundle_sha256 or "")
        if (
            not expected_bundle
            or metadata.get("feature_output_bundle_sha256") != expected_bundle
            or metadata.get("capability_native_map_output_bundle_sha256")
            != expected_bundle
        ):
            raise ValueError(
                f"{expected_space} MPR belongs to another feature output bundle"
            )

    raw_xyz = torch.as_tensor(raw_cache["xyz"]).float().cpu()
    target_xyz = torch.as_tensor(cache.get("xyz")).float().cpu()
    if target_xyz.shape != raw_xyz.shape or _sha256_tensor_rows(
        target_xyz
    ) != _sha256_tensor_rows(raw_xyz):
        raise ValueError(f"{expected_space} MPR geometry does not align with raw MPR")
    raw_valid = torch.as_tensor(raw_cache["valid"]).bool().cpu()
    target_valid = torch.as_tensor(cache.get("valid")).bool().cpu()
    raw_counts = torch.as_tensor(raw_cache["view_counts"]).long().cpu()
    target_counts = torch.as_tensor(cache.get("view_counts")).long().cpu()
    if not torch.equal(target_valid, raw_valid) or not torch.equal(
        target_counts, raw_counts
    ):
        raise ValueError(
            f"{expected_space} MPR must use the exact raw-MPR observation support"
        )
    responsibility_sha256 = str(
        metadata.get("registration_responsibility_cache_sha256", "")
    )
    if (
        target_contract == CAPABILITY_TARGET_CONTRACT_MATCHED_TOP1
        and str(raw_metadata.get("aggregation_mode", "")) == "raster_gaussian_top1"
    ):
        raw_responsibility_sha256 = str(
            raw_metadata.get("registration_responsibility_cache_sha256", "")
        )
        if (
            not raw_responsibility_sha256
            or not bool(raw_metadata.get("shared_registration_responsibility", False))
            or not bool(metadata.get("shared_registration_responsibility", False))
            or responsibility_sha256 != raw_responsibility_sha256
        ):
            raise ValueError(
                f"{expected_space} MPR must reuse the exact raw-MPR "
                "registration responsibility sidecar"
            )
    policy_keys = (
        "config",
        "checkpoint",
        "selected_frame_indices",
        "excluded_frame_ids",
        "aggregation_mode",
        "registration_weight_mode",
        "raster_view_fusion",
        "raster_topk",
        "depth_tolerance",
        "relative_depth_tolerance",
        "alpha_threshold",
        "normalize_each_view",
    )
    mismatched = [
        key for key in policy_keys if metadata.get(key) != raw_metadata.get(key)
    ]
    if mismatched:
        raise ValueError(
            f"{expected_space} MPR policy differs from raw MPR: {mismatched}"
        )
    consensus = _consensus_from_cache(cache, preserve_target_dtype=True)
    feature_dim = (
        consensus.feature_dim
        if isinstance(consensus, ShardedMPRCache)
        else int(consensus.targets.shape[1])
    )
    return consensus, {
        "path": str(cache_path.resolve()),
        "sha256": cache_sha256,
        "feature_space": expected_space,
        "feature_dim": int(feature_dim),
        "target_contract": target_contract,
        "projection_order": (
            "official_adaptor_then_exact_raster_adjoint_contribution_mpr"
            if target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_A
            else (
                "official_adaptor_then_exact_center_plus_uncertainty_mpr"
                if target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_C
                else "official_adaptor_then_geometry_matched_mpr"
            )
        ),
        "official_adaptor_name": metadata.get("official_adaptor_name"),
        "official_adaptor_checkpoint_sha256": metadata.get(
            "official_adaptor_checkpoint_sha256"
        ),
        "official_adaptor_checkpoint_provenance": metadata.get(
            "official_adaptor_checkpoint_provenance", ""
        ),
        "capability_map_source": capability_map_source,
        "capability_native_map_manifest": metadata.get(
            "capability_native_map_manifest", ""
        ),
        "capability_native_map_manifest_sha256": metadata.get(
            "capability_native_map_manifest_sha256", ""
        ),
        "feature_output_bundle_sha256": metadata.get(
            "feature_output_bundle_sha256", ""
        ),
        "capability_native_map_output_bundle_sha256": metadata.get(
            "capability_native_map_output_bundle_sha256", ""
        ),
        "capability_native_map_radio_checkpoint_load_contract": metadata.get(
            "capability_native_map_radio_checkpoint_load_contract", ""
        ),
        "capability_native_map_grid": metadata.get("capability_native_map_grid", []),
        "capability_adaptor_execution": metadata.get(
            "capability_adaptor_execution", ""
        ),
        "selected_frame_indices": metadata.get("selected_frame_indices", []),
        "aggregation_mode": metadata.get("aggregation_mode", ""),
        "raster_view_fusion": metadata.get("raster_view_fusion", ""),
        "raster_reliability_mode": metadata.get(
            "raster_reliability_mode", "legacy_valid"
        ),
        "registration_responsibility_cache_sha256": responsibility_sha256,
        "uses_query_or_benchmark_supervision": False,
        **(
            consensus.provenance()
            if isinstance(consensus, ShardedMPRCache)
            else {"storage": "dense_torch_tensor"}
        ),
    }


def _load_factorized_matched_capability_targets(
    *,
    factorized: FactorizedRadioTrainingCache,
    reference_path: str | Path,
    reference_expected_sha256: str,
    dino_path: str | Path,
    dino_expected_sha256: str,
    sam3_path: str | Path,
    sam3_expected_sha256: str,
    correction_path: str | Path,
    correction_expected_sha256: str,
    radio_checkpoint_sha256: str,
) -> tuple[dict[str, PrimitiveConsensus | ShardedMPRCache], dict[str, dict], dict]:
    """Load one hash-bound matched-top1 cohort through a narrow receipt.

    Historical cohorts may use the frozen July compatibility receipt.  New
    cohorts must use the formal receipt and carry the same sealed feature
    bundle as the factorized RADIO cache.  Keeping the two modes explicit
    prevents fresh scene assets from masquerading as legacy exceptions.
    """

    repo_root = Path(__file__).resolve().parents[2]

    def resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else repo_root / path).resolve()

    correction, correction_sha256, correction_source = load_json_object(
        resolve(correction_path),
        expected_sha256=correction_expected_sha256,
        label="factorized capability legacy receipt correction",
    )
    experiment = str(correction.get("experiment", ""))
    legacy_cohort = experiment == LEGACY_FACTORIZED_CAPABILITY_RECEIPT_EXPERIMENT
    formal_cohort = experiment == FORMAL_FACTORIZED_CAPABILITY_RECEIPT_EXPERIMENT
    if legacy_cohort:
        target_access_is_false = (
            correction.get("scientific_contract_unchanged", {}).get("target_access")
            is False
        )
        cohort_bundle_sha256 = ""
    elif formal_cohort:
        target_access = correction.get("target_access")
        target_access_is_false = isinstance(target_access, dict) and all(
            target_access.get(key) is False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
                "target_metrics_used_for_selection",
            )
        )
        cohort_bundle_sha256 = str(correction.get("feature_output_bundle_sha256", ""))
        if (
            correction.get("artifact_type") != "factorized_capability_cohort_authority"
            or int(correction.get("schema_version", -1)) != 1
            or re.fullmatch(r"[0-9a-f]{64}", cohort_bundle_sha256) is None
            or factorized.metadata.get("feature_output_bundle_sha256")
            != cohort_bundle_sha256
        ):
            raise ValueError("formal factorized capability receipt differs")
    else:
        target_access_is_false = False
        cohort_bundle_sha256 = ""
    if not target_access_is_false:
        raise ValueError("factorized capability correction receipt differs")
    authorities = correction.get("frozen_cache_authorities")
    if not isinstance(authorities, dict) or set(authorities) != {
        "radio",
        "dino_v3",
        "sam3",
    }:
        raise ValueError("factorized capability correction authorities differ")
    requested = {
        "radio": (resolve(reference_path), reference_expected_sha256),
        "dino_v3": (resolve(dino_path), dino_expected_sha256),
        "sam3": (resolve(sam3_path), sam3_expected_sha256),
    }
    for name, (path, digest) in requested.items():
        authority = authorities.get(name)
        if not isinstance(authority, dict) or (
            resolve(str(authority.get("path", ""))) != path
            or authority.get("sha256") != str(digest)
        ):
            raise ValueError(f"factorized capability {name} receipt authority differs")

    reference, reference_sha256, reference_source = load_mpr_cache(
        resolve(reference_path),
        expected_sha256=reference_expected_sha256,
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=formal_cohort,
    )
    reference_metadata = dict(reference.get("metadata", {}))
    factorized_metadata = factorized.metadata
    if formal_cohort and (
        reference_metadata.get("feature_output_bundle_sha256") != cohort_bundle_sha256
    ):
        raise ValueError("formal factorized capability reference bundle differs")
    factorized_geometry = dict(factorized.geometry_fingerprint)
    reference_geometry = dict(reference.get("geometry_fingerprint", {}))
    if reference_geometry != factorized_geometry:
        raise ValueError("factorized capability reference geometry differs")
    reference_valid = torch.as_tensor(reference.get("valid")).bool().cpu()
    reference_counts = torch.as_tensor(reference.get("view_counts")).long().cpu()
    if not torch.equal(reference_valid, factorized.valid) or not torch.equal(
        reference_counts, factorized.view_counts
    ):
        raise ValueError("factorized capability reference support differs")
    source_only = (
        "benchmark_images_opened",
        "benchmark_masks_opened",
        "text_queries_opened",
    )
    if any(reference_metadata.get(key) is not False for key in source_only):
        raise ValueError("factorized capability reference is not source-only")
    responsibility_sha256 = str(
        factorized_metadata.get("registration_responsibility_cache_sha256", "")
    )
    if (
        not responsibility_sha256
        or reference_metadata.get("registration_responsibility_cache_sha256")
        != responsibility_sha256
        or reference_metadata.get("shared_registration_responsibility") is not True
    ):
        raise ValueError("factorized capability reference responsibility differs")
    common_policy = (
        "config",
        "checkpoint",
        "selected_dataset_indices",
        "selected_frame_indices",
        "aggregation_mode",
        "registration_weight_mode",
        "raster_view_fusion",
    )
    mismatched_reference = [
        key
        for key in common_policy
        if reference_metadata.get(key) != factorized_metadata.get(key)
    ]
    if mismatched_reference:
        raise ValueError(
            "factorized capability reference policy differs: " f"{mismatched_reference}"
        )
    lifting = reference_metadata.get("observation_lifting_contract")
    if not isinstance(lifting, dict) or (
        lifting.get("name") != "canonical-mpr-v1"
        or lifting.get("feature_projection_order") != "per_view_before_mpr"
        or lifting.get("responsibility_sharing")
        != "exact_sidecar_across_feature_spaces"
        or lifting.get("query_independent") is not True
    ):
        raise ValueError("factorized capability reference lifting differs")

    targets: dict[str, PrimitiveConsensus | ShardedMPRCache] = {}
    provenance: dict[str, dict] = {}
    legacy_policy = (
        "config",
        "checkpoint",
        "selected_dataset_indices",
        "selected_frame_indices",
        "excluded_frame_ids",
        "aggregation_mode",
        "registration_weight_mode",
        "raster_view_fusion",
        "raster_topk",
        "depth_tolerance",
        "relative_depth_tolerance",
        "alpha_threshold",
        "normalize_each_view",
        "per_view_normalization_applied",
        "per_view_normalization_stage",
    )
    for name, path, expected_sha256 in (
        ("dino_v3", dino_path, dino_expected_sha256),
        ("sam3", sam3_path, sam3_expected_sha256),
    ):
        cache, digest, source = load_mpr_cache(
            resolve(path),
            expected_sha256=expected_sha256,
            expected_feature_space=name,
            require_reliability=True,
            require_formal_safety=formal_cohort,
        )
        metadata = dict(cache.get("metadata", {}))
        if formal_cohort and (
            metadata.get("feature_output_bundle_sha256") != cohort_bundle_sha256
        ):
            raise ValueError(f"formal factorized {name} capability bundle differs")
        if any(metadata.get(key) is not False for key in source_only) or (
            metadata.get("capability_projection_before_mpr") is not True
            or metadata.get("custom_adaptor_head") is not False
            or metadata.get("capability_map_source") != "project_raw"
            or metadata.get("shared_registration_responsibility") is not True
            or metadata.get("official_adaptor_checkpoint_sha256")
            != radio_checkpoint_sha256
            or metadata.get("registration_responsibility_cache_sha256")
            != responsibility_sha256
        ):
            raise ValueError(f"factorized {name} capability lineage differs")
        if dict(cache.get("geometry_fingerprint", {})) != reference_geometry:
            raise ValueError(f"factorized {name} capability geometry differs")
        if not torch.equal(
            torch.as_tensor(cache.get("valid")).bool().cpu(), reference_valid
        ) or not torch.equal(
            torch.as_tensor(cache.get("view_counts")).long().cpu(), reference_counts
        ):
            raise ValueError(f"factorized {name} capability support differs")
        mismatched = [
            key
            for key in legacy_policy
            if metadata.get(key) != reference_metadata.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"factorized {name} capability policy differs: {mismatched}"
            )
        target_lifting = metadata.get("observation_lifting_contract")
        if not isinstance(target_lifting, dict) or (
            target_lifting.get("name") != "canonical-mpr-v1"
            or target_lifting.get("feature_projection_order") != "per_view_before_mpr"
            or target_lifting.get("responsibility_sharing")
            != "exact_sidecar_across_feature_spaces"
            or target_lifting.get("query_independent") is not True
        ):
            raise ValueError(f"factorized {name} capability lifting differs")
        targets[name] = _consensus_from_cache(cache, preserve_target_dtype=True)
        provenance[name] = {
            "path": str(source.resolve()),
            "sha256": digest,
            "feature_space": name,
            "target_contract": CAPABILITY_TARGET_CONTRACT_MATCHED_TOP1,
            "projection_order": "official_adaptor_then_geometry_matched_mpr",
            "official_adaptor_checkpoint_sha256": radio_checkpoint_sha256,
            "registration_responsibility_cache_sha256": responsibility_sha256,
            "historical_feature_bundle_receipt_compatibility": legacy_cohort,
            "formal_feature_bundle_authority": formal_cohort,
            "feature_output_bundle_sha256": (
                cohort_bundle_sha256 if formal_cohort else ""
            ),
            "uses_query_or_benchmark_supervision": False,
        }
    reference_provenance = {
        "path": str(reference_source.resolve()),
        "sha256": reference_sha256,
        "geometry_fingerprint": reference_geometry,
        "registration_responsibility_cache_sha256": responsibility_sha256,
        "legacy_receipt_correction": {
            "path": str(correction_source.resolve()),
            "sha256": correction_sha256,
        },
        "capability_cohort_authority_mode": (
            "formal_feature_bundle_v1" if formal_cohort else "legacy_compatibility_v1"
        ),
        "historical_feature_bundle_receipt_compatibility": legacy_cohort,
        "formal_feature_bundle_authority": formal_cohort,
        "feature_output_bundle_sha256": (cohort_bundle_sha256 if formal_cohort else ""),
        "uses_query_or_benchmark_supervision": False,
    }
    return targets, provenance, reference_provenance


def _load_factorized_exact_marginal_capability_targets(
    *,
    factorized: FactorizedRadioTrainingCache,
    reference_path: str | Path,
    reference_expected_sha256: str,
    dino_path: str | Path,
    dino_expected_sha256: str,
    sam3_path: str | Path,
    sam3_expected_sha256: str,
    correction_path: str | Path,
    correction_expected_sha256: str,
    radio_checkpoint_sha256: str,
) -> tuple[dict[str, PrimitiveConsensus | ShardedMPRCache], dict[str, dict], dict]:
    """Load a builder-v2 cohort sharing one exact sparse marginal authority.

    This intentionally does not reuse the historical top-1 compatibility
    bridge.  Every one of the factorized/raw/DINO/SAM caches must be a formal,
    source-only artifact with identical geometry, frames, semantic support,
    and responsibility authority.  The factorized cache retains its raw
    amplitude-aware core contract; only conventional caches declare the
    normalized exact-marginal observation-lifting contract.
    """

    repo_root = Path(__file__).resolve().parents[2]

    def resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else repo_root / path).resolve()

    expected_lifting = canonical_observation_contract(
        CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME
    )
    expected_lifting_sha256 = observation_contract_sha256(expected_lifting)
    factorized_metadata = factorized.metadata
    if (
        factorized_metadata.get("builder_contract")
        != canonical_factorized_radio_builder_contract_v2()
        or factorized_metadata.get("builder_contract_sha256")
        != factorized_radio_builder_contract_v2_sha256()
        or factorized_metadata.get("observation_lifting_contract")
        != canonical_factorized_radio_contract()
        or factorized_metadata.get("observation_lifting_contract_sha256")
        != CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
        or factorized.provenance().get("storage")
        != CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2
    ):
        raise ValueError(
            "factorized exact-marginal target requires builder-v2 exact support "
            "and the unchanged raw-amplitude factorized core contract"
        )

    source_only_flags = (
        "benchmark_images_opened",
        "benchmark_masks_opened",
        "text_queries_opened",
    )
    if (
        any(factorized_metadata.get(key) is not False for key in source_only_flags)
        or factorized_metadata.get("query_independent") is not True
    ):
        raise ValueError("factorized exact-marginal cache is not source-only")
    shared_semantic_policy = {
        "semantic_assignment_gate": ("pre_adaptor_raw_radio_l2_norm_strictly_positive"),
        "view_count_semantics": (
            "views_with_pre_adaptor_raw_radio_l2_norm_strictly_positive"
        ),
        "geometric_visibility_semantics": (
            "independent_exact_base_weight_authority_includes_zero_amplitude_hits"
        ),
    }
    factorized_semantic_policy = {
        **shared_semantic_policy,
        "valid_semantics": (
            "positive_raw_radio_amplitude_responsibility_mass_and_"
            "nonzero_direction_resultant"
        ),
        "invalid_row_purity_policy": (
            "core_v1_requires_zero_for_semantically_invalid_rows"
        ),
    }
    if any(
        factorized_metadata.get(key) != value
        for key, value in factorized_semantic_policy.items()
    ):
        raise ValueError("factorized exact-marginal semantic evidence policy differs")
    geometric_authority_fields = (
        "geometric_view_counts_sha256",
        "geometric_visible_gaussian_count",
        "semantic_valid_gaussian_count",
        "geometric_visible_semantic_invalid_gaussian_count",
    )
    if (
        re.fullmatch(
            r"[0-9a-f]{64}",
            str(factorized_metadata.get("geometric_view_counts_sha256", "")),
        )
        is None
    ):
        raise ValueError("factorized exact-marginal geometric counts authority differs")

    correction, correction_sha256, correction_source = load_json_object(
        resolve(correction_path),
        expected_sha256=correction_expected_sha256,
        label="factorized exact-marginal capability cohort authority",
    )
    cohort_bundle_sha256 = str(correction.get("feature_output_bundle_sha256", ""))
    target_access = correction.get("target_access")
    if (
        correction.get("experiment") != FORMAL_FACTORIZED_CAPABILITY_RECEIPT_EXPERIMENT
        or correction.get("artifact_type") != "factorized_capability_cohort_authority"
        or int(correction.get("schema_version", -1)) != 1
        or re.fullmatch(r"[0-9a-f]{64}", cohort_bundle_sha256) is None
        or cohort_bundle_sha256
        != factorized_metadata.get("feature_output_bundle_sha256")
        or not isinstance(target_access, dict)
        or any(
            target_access.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
                "target_metrics_used_for_selection",
            )
        )
    ):
        raise ValueError("formal exact-marginal capability cohort authority differs")

    authorities = correction.get("frozen_cache_authorities")
    if not isinstance(authorities, dict) or set(authorities) != {
        "radio",
        "dino_v3",
        "sam3",
    }:
        raise ValueError("formal exact-marginal cache authorities differ")
    requested = {
        "radio": (resolve(reference_path), str(reference_expected_sha256)),
        "dino_v3": (resolve(dino_path), str(dino_expected_sha256)),
        "sam3": (resolve(sam3_path), str(sam3_expected_sha256)),
    }
    for name, (path, digest) in requested.items():
        authority = authorities.get(name)
        if not isinstance(authority, dict) or (
            resolve(str(authority.get("path", ""))) != path
            or authority.get("sha256") != digest
        ):
            raise ValueError(f"formal exact-marginal {name} authority differs")

    reference, reference_sha256, reference_source = load_mpr_cache(
        requested["radio"][0],
        expected_sha256=requested["radio"][1],
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=True,
    )
    reference_metadata = dict(reference.get("metadata", {}))
    validate_observation_contract_metadata(
        reference_metadata,
        require_declaration=True,
        contract_name=CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
    )
    if (
        reference_metadata.get("feature_output_bundle_sha256") != cohort_bundle_sha256
        or reference_metadata.get("capability_projection_before_mpr") is not False
        or reference_metadata.get("custom_adaptor_head") is not False
    ):
        raise ValueError("formal exact-marginal raw feature bundle differs")
    conventional_semantic_policy = {
        **shared_semantic_policy,
        "valid_semantics": (
            "positive_pre_adaptor_raw_radio_amplitude_responsibility_mass"
        ),
        "invalid_row_purity_policy": (
            "mpr_schema_v1_requires_zero_for_semantically_invalid_rows"
        ),
    }
    if any(
        reference_metadata.get(key) != value
        for key, value in conventional_semantic_policy.items()
    ) or any(
        reference_metadata.get(key) != factorized_metadata.get(key)
        for key in geometric_authority_fields
    ):
        raise ValueError("factorized exact-marginal raw semantic authority differs")

    reference_geometry = dict(reference.get("geometry_fingerprint", {}))
    reference_valid = torch.as_tensor(reference.get("valid")).bool().cpu()
    reference_counts = torch.as_tensor(reference.get("view_counts")).long().cpu()
    if (
        reference_geometry != dict(factorized.geometry_fingerprint)
        or not torch.equal(reference_valid, factorized.valid)
        or not torch.equal(reference_counts, factorized.view_counts)
    ):
        raise ValueError("factorized exact-marginal raw geometry/support differs")

    common_frames = (
        "selected_dataset_indices",
        "selected_frame_indices",
        "num_declared_views",
    )
    if any(
        reference_metadata.get(key) != factorized_metadata.get(key)
        for key in common_frames
    ):
        raise ValueError("factorized exact-marginal raw frame authority differs")

    responsibility_sha256 = str(
        factorized_metadata.get("registration_responsibility_cache_sha256", "")
    )
    responsibility_contract = factorized_metadata.get(
        "registration_responsibility_contract"
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", responsibility_sha256) is None
        or not isinstance(responsibility_contract, dict)
        or factorized_metadata.get("shared_registration_responsibility") is not True
        or reference_metadata.get("registration_responsibility_cache_sha256")
        != responsibility_sha256
        or reference_metadata.get("registration_responsibility_contract")
        != responsibility_contract
        or reference_metadata.get("shared_registration_responsibility") is not True
    ):
        raise ValueError("factorized exact-marginal raw responsibility differs")

    targets: dict[str, PrimitiveConsensus | ShardedMPRCache] = {}
    provenance: dict[str, dict] = {}
    for name in ("dino_v3", "sam3"):
        cache, digest, source = load_mpr_cache(
            requested[name][0],
            expected_sha256=requested[name][1],
            expected_feature_space=name,
            require_reliability=True,
            require_formal_safety=True,
        )
        metadata = dict(cache.get("metadata", {}))
        validate_observation_contract_metadata(
            metadata,
            require_declaration=True,
            contract_name=CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
        )
        valid = torch.as_tensor(cache.get("valid")).bool().cpu()
        counts = torch.as_tensor(cache.get("view_counts")).long().cpu()
        if (
            dict(cache.get("geometry_fingerprint", {})) != reference_geometry
            or not torch.equal(valid, reference_valid)
            or not torch.equal(counts, reference_counts)
        ):
            raise ValueError(
                f"factorized exact-marginal {name} geometry/support differs"
            )
        if any(
            metadata.get(key) != reference_metadata.get(key) for key in common_frames
        ):
            raise ValueError(
                f"factorized exact-marginal {name} frame authority differs"
            )
        if any(
            metadata.get(key) != value
            for key, value in conventional_semantic_policy.items()
        ) or any(
            metadata.get(key) != reference_metadata.get(key)
            for key in geometric_authority_fields
        ):
            raise ValueError(
                f"factorized exact-marginal {name} semantic authority differs"
            )
        if (
            metadata.get("observation_lifting_contract") != expected_lifting
            or metadata.get("observation_lifting_contract_sha256")
            != expected_lifting_sha256
            or metadata.get("registration_responsibility_cache_sha256")
            != responsibility_sha256
            or metadata.get("registration_responsibility_contract")
            != responsibility_contract
            or metadata.get("shared_registration_responsibility") is not True
        ):
            raise ValueError(f"factorized exact-marginal {name} lifting differs")
        if (
            metadata.get("feature_output_bundle_sha256") != cohort_bundle_sha256
            or any(metadata.get(key) is not False for key in source_only_flags)
            or metadata.get("capability_projection_before_mpr") is not True
            or metadata.get("custom_adaptor_head") is not False
            or metadata.get("capability_map_source") != "project_raw"
            or metadata.get("official_adaptor_checkpoint_sha256")
            != radio_checkpoint_sha256
        ):
            raise ValueError(f"factorized exact-marginal {name} feature target differs")
        targets[name] = _consensus_from_cache(cache, preserve_target_dtype=True)
        provenance[name] = {
            "path": str(source.resolve()),
            "sha256": digest,
            "feature_space": name,
            "target_contract": CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL,
            "projection_order": ("official_adaptor_then_shared_exact_marginal_mpr"),
            "official_adaptor_checkpoint_sha256": radio_checkpoint_sha256,
            "registration_responsibility_cache_sha256": responsibility_sha256,
            "observation_lifting_contract_sha256": expected_lifting_sha256,
            "feature_output_bundle_sha256": cohort_bundle_sha256,
            "uses_query_or_benchmark_supervision": False,
        }

    reference_provenance = {
        "path": str(reference_source.resolve()),
        "sha256": reference_sha256,
        "geometry_fingerprint": reference_geometry,
        "registration_responsibility_cache_sha256": responsibility_sha256,
        "observation_lifting_contract_sha256": expected_lifting_sha256,
        "capability_cohort_authority": {
            "path": str(correction_source.resolve()),
            "sha256": correction_sha256,
        },
        "capability_cohort_authority_mode": "formal_exact_marginal_v1",
        "feature_output_bundle_sha256": cohort_bundle_sha256,
        "uses_query_or_benchmark_supervision": False,
    }
    return targets, provenance, reference_provenance


def _load_field_a_observation_reference(
    path: str | Path,
    *,
    primary_raw_cache: dict | ShardedMPRCache,
    primary_raw_metadata: dict,
    expected_cache_sha256: str,
) -> tuple[dict | ShardedMPRCache, dict, dict]:
    """Load the raw-RADIO adjoint cache defining Field-A observations.

    Field-A keeps the current canonical raw MPR objective unchanged.  This
    second raw cache binds only the view/geometry/operator support on which
    the exact DINO and SAM auxiliary teachers must have been constructed.
    """

    expected = str(expected_cache_sha256)
    if not expected:
        raise ValueError("Field-A observation reference requires a trusted SHA-256")
    cache, digest, source = load_mpr_cache(
        path,
        expected_sha256=expected,
        expected_feature_space="radio",
        require_reliability=True,
        # The frozen exact-adjoint support reference predates feature-bundle
        # receipts.  Field-A never consumes its feature values: only its
        # hash-bound geometry, validity, observation counts, views, and
        # operator policy define the support shared by the newly formal DINO
        # and SAM targets.  Do not invent provenance by rewriting this cache;
        # validate its legacy payload deeply and enforce contamination/policy
        # declarations explicitly below instead.
        require_formal_safety=False,
    )
    metadata = dict(cache.get("metadata", {}))
    contaminated = [
        key
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
        if metadata.get(key) is not False
    ]
    if contaminated:
        raise ValueError(
            f"Field-A observation reference is not query-independent: {contaminated}"
        )
    expected_policy = {
        "aggregation_mode": "raster_adjoint",
        "raster_view_fusion": "contribution_mean",
        "normalize_each_view": True,
    }
    mismatched = [
        key for key, value in expected_policy.items() if metadata.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            f"Field-A observation reference policy differs: {sorted(mismatched)}"
        )
    for key in ("config", "checkpoint", "excluded_frame_ids", "alpha_threshold"):
        if metadata.get(key) != primary_raw_metadata.get(key):
            raise ValueError(
                f"Field-A observation reference differs from primary raw MPR: {key}"
            )
    primary_xyz = torch.as_tensor(primary_raw_cache["xyz"]).float().cpu()
    reference_xyz = torch.as_tensor(cache["xyz"]).float().cpu()
    if reference_xyz.shape != primary_xyz.shape or _sha256_tensor_rows(
        reference_xyz
    ) != _sha256_tensor_rows(primary_xyz):
        raise ValueError("Field-A observation reference geometry differs")
    return (
        cache,
        metadata,
        {
            "path": str(source.resolve()),
            "sha256": digest,
            "aggregation_mode": metadata["aggregation_mode"],
            "raster_view_fusion": metadata["raster_view_fusion"],
            "selected_frame_indices": list(metadata.get("selected_frame_indices", [])),
            "uses_query_or_benchmark_supervision": False,
            **(
                cache.provenance()
                if isinstance(cache, ShardedMPRCache)
                else {"storage": "dense_torch_tensor"}
            ),
        },
    )


@torch.no_grad()
def _reconstruction_metrics(
    field: CanonicalGaussianField,
    consensus: PrimitiveConsensus,
    rows: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    cosines: list[torch.Tensor] = []
    rmses: list[torch.Tensor] = []
    device = field.local_codes.device
    for start in range(0, rows.numel(), batch_size):
        batch = rows[start : start + batch_size]
        predicted = field.radio_features(batch.to(device)).float().cpu()
        target = consensus_target_rows(consensus, batch).float()
        cosines.append(F.cosine_similarity(predicted, target, dim=-1, eps=1e-8))
        rmses.append((predicted - target).square().mean(dim=-1).sqrt())
    cosine = torch.cat(cosines)
    rmse = torch.cat(rmses)
    return {
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(cosine.quantile(0.05)),
        "mean_rmse": float(rmse.mean()),
    }


@torch.no_grad()
def _factorized_amplitude_metrics(
    field: CanonicalGaussianField,
    target: FactorizedRadioTrainingCache,
    rows: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    """Measure the independently supervised raw-RADIO gauge."""

    absolute_log_errors: list[torch.Tensor] = []
    predicted_norms: list[torch.Tensor] = []
    target_norms: list[torch.Tensor] = []
    device = field.local_codes.device
    for start in range(0, rows.numel(), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        predicted = field.radio_features(batch.to(device)).float().cpu()
        predicted_norm = torch.linalg.vector_norm(predicted, dim=-1)
        predicted_log = torch.log(
            torch.sqrt(predicted_norm.square() + torch.finfo(torch.float32).eps ** 2)
        )
        target_log = target.log_amplitude[batch].float()
        absolute_log_errors.append((predicted_log - target_log).abs())
        predicted_norms.append(predicted_norm)
        target_norms.append(torch.exp(target_log))
    error = torch.cat(absolute_log_errors)
    predicted_norm = torch.cat(predicted_norms)
    target_norm = torch.cat(target_norms)
    return {
        "mean_abs_log_amplitude_error": float(error.mean()),
        "p95_abs_log_amplitude_error": float(error.quantile(0.95)),
        "predicted_norm_median": float(predicted_norm.median()),
        "target_norm_median": float(target_norm.median()),
    }


@torch.no_grad()
def _capability_reconstruction_metrics(
    field: CanonicalGaussianField,
    official_views: FrozenRadioViews,
    targets: dict[str, PrimitiveConsensus | ShardedMPRCache],
    rows: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    values: dict[str, list[torch.Tensor]] = {name: [] for name in targets}
    device = field.local_codes.device
    for start in range(0, rows.numel(), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        radio = field.radio_features(batch.to(device)).float()
        for name, consensus in targets.items():
            valid = consensus.valid[batch]
            if not bool(valid.any()):
                continue
            projected = (
                official_views.project_dino_primitives(radio)
                if name == "dino_v3"
                else official_views.project_sam3_primitives(radio)
            )
            target = consensus_target_rows(consensus, batch).to(device).float()
            values[name].append(
                F.cosine_similarity(
                    projected[valid.to(device)],
                    target[valid.to(device)],
                    dim=-1,
                    eps=1e-8,
                ).cpu()
            )
    report: dict[str, float] = {}
    for name, parts in values.items():
        if not parts:
            report[f"{name}_target_mean_cosine"] = 0.0
            report[f"{name}_target_p05_cosine"] = 0.0
            continue
        cosine = torch.cat(parts)
        report[f"{name}_target_mean_cosine"] = float(cosine.mean())
        report[f"{name}_target_p05_cosine"] = float(torch.quantile(cosine, 0.05))
    return report


@torch.no_grad()
def _cross_basis_projection(
    local_decoder, output_decoder
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local PCA coordinates into the higher-rank output PCA coordinates."""

    scale_ratio = local_decoder.scale / output_decoder.scale
    output_inverse = output_decoder.encoding_projection()
    matrix = (local_decoder.basis.transpose(0, 1) * scale_ratio[None]) @ output_inverse
    bias = (
        (local_decoder.mean - output_decoder.mean) / output_decoder.scale
    ) @ output_inverse
    return matrix.transpose(0, 1).contiguous(), bias.contiguous()


def train(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    for candidate in (output, report_path):
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(
                f"refuse to overwrite canonical field artifact: {candidate}"
            )
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    basis_fit_device = torch.device(
        str(getattr(args, "basis_fit_device", "")).strip() or "cpu"
    )
    if basis_fit_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA basis fitting requested but CUDA is unavailable")
    observation_contract_mode = str(getattr(args, "observation_contract", "unchecked"))
    capability_target_contract = str(
        getattr(
            args,
            "capability_target_contract",
            CAPABILITY_TARGET_CONTRACT_MATCHED_TOP1,
        )
    )
    factorized_mode = (
        observation_contract_mode == CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME
    )
    expected_feature_bundle_sha256 = str(
        getattr(args, "expected_feature_output_bundle_sha256", "")
    )
    radio_hash = sha256_file(args.radio_checkpoint)
    expected_radio_hash = str(getattr(args, "expected_radio_checkpoint_sha256", ""))
    if factorized_mode and re.fullmatch(r"[0-9a-f]{64}", expected_radio_hash) is None:
        raise ValueError(
            "canonical-factorized-radio-v1 requires the official RADIO SHA-256"
        )
    if expected_radio_hash and radio_hash != expected_radio_hash:
        raise ValueError("RADIO checkpoint differs from caller authority")
    factorized_target: FactorizedRadioTrainingCache | None = None
    if factorized_mode:
        if bool(getattr(args, "fusion_reliability", True)):
            raise ValueError(
                "canonical-factorized-radio-v1 requires --no-fusion-reliability"
            )
        if capability_target_contract not in {
            CAPABILITY_TARGET_CONTRACT_MATCHED_TOP1,
            CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL,
        }:
            raise ValueError(
                "factorized RADIO accepts only matched top1 or exact-marginal "
                "capability targets"
            )
        if str(getattr(args, "relation_objective", RELATION_OBJECTIVE_DISABLED)) != (
            RELATION_OBJECTIVE_DISABLED
        ):
            raise ValueError(
                "factorized RADIO does not accept legacy relation objectives"
            )
        factorized_target = load_factorized_radio_training_cache(
            args.mpr_cache,
            expected_sha256=str(getattr(args, "expected_mpr_cache_sha256", "")),
            expected_feature_output_bundle_sha256=(expected_feature_bundle_sha256),
        )
        cache: dict | ShardedMPRCache = factorized_target.support_mapping()
        mpr_cache_sha256 = factorized_target.sha256
        mpr_cache_path = factorized_target.source
        metadata = dict(factorized_target.metadata)
        consensus = factorized_target.as_consensus()
        raw_mpr_storage_provenance = factorized_target.provenance()
    else:
        cache, mpr_cache_sha256, mpr_cache_path = load_mpr_cache(
            args.mpr_cache,
            expected_sha256=(
                str(getattr(args, "expected_mpr_cache_sha256", "")) or None
            ),
            expected_feature_space="radio",
            require_reliability=True,
            require_formal_safety=(
                observation_contract_mode in STRICT_OBSERVATION_CONTRACT_MODES
                or capability_target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_C
            ),
        )
        metadata = dict(cache.get("metadata", {}))
        _validate_training_observation_contract_metadata(
            metadata,
            observation_contract_mode,
        )
        consensus = _consensus_from_cache(cache)
        raw_mpr_storage_provenance = (
            cache.provenance()
            if isinstance(cache, ShardedMPRCache)
            else {"storage": "dense_torch_tensor"}
        )
    if metadata.get("benchmark_masks_opened", False) or metadata.get(
        "text_queries_opened", False
    ):
        raise ValueError(
            "MPR training cache is contaminated by benchmark queries or masks"
        )
    if str(metadata.get("feature_space", "radio")) != "radio":
        raise ValueError(
            "canonical main field must reconstruct raw RADIO, not a query head"
        )
    if (
        observation_contract_mode in STRICT_OBSERVATION_CONTRACT_MODES
        or factorized_mode
    ) and (
        not expected_feature_bundle_sha256
        or metadata.get("feature_output_bundle_sha256")
        != expected_feature_bundle_sha256
    ):
        raise ValueError("raw MPR belongs to another feature output bundle")
    capability_observation_cache = cache
    capability_observation_metadata = metadata
    capability_observation_provenance: dict = {}
    if capability_target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_A:
        if not bool(args.official_capability_loss):
            raise ValueError("Field-A requires --official-capability-loss")
        if not str(args.dino_mpr_cache).strip() or not str(args.sam3_mpr_cache).strip():
            raise ValueError("Field-A requires both exact DINO and SAM3 MPR targets")
        for name in ("dino_v3", "sam3"):
            if not str(getattr(args, f"expected_{name}_mpr_cache_sha256", "")):
                raise ValueError(
                    f"Field-A requires a trusted SHA-256 for the {name} MPR target"
                )
        (
            capability_observation_cache,
            capability_observation_metadata,
            (capability_observation_provenance),
        ) = _load_field_a_observation_reference(
            getattr(args, "capability_observation_reference_mpr_cache", ""),
            primary_raw_cache=cache,
            primary_raw_metadata=metadata,
            expected_cache_sha256=str(
                getattr(
                    args,
                    "expected_capability_observation_reference_mpr_cache_sha256",
                    "",
                )
            ),
        )
    elif capability_target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_C:
        if not bool(args.official_capability_loss):
            raise ValueError("Field-C requires --official-capability-loss")
        if metadata.get("aggregation_mode") != "raster_exact_center_uncertainty":
            raise ValueError("Field-C requires an exact-center uncertainty raw MPR")
        if not str(args.dino_mpr_cache).strip() or not str(args.sam3_mpr_cache).strip():
            raise ValueError("Field-C requires both DINO and SAM3 MPR targets")
        for name in ("dino_v3", "sam3"):
            if not str(getattr(args, f"expected_{name}_mpr_cache_sha256", "")):
                raise ValueError(
                    f"Field-C requires a trusted SHA-256 for the {name} MPR target"
                )
        if (
            not expected_feature_bundle_sha256
            or metadata.get("feature_output_bundle_sha256")
            != expected_feature_bundle_sha256
        ):
            raise ValueError("Field-C raw MPR belongs to another feature bundle")
    elif (
        capability_target_contract == CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL
    ):
        if not factorized_mode:
            raise ValueError(
                "matched_exact_marginal capability targets require factorized RADIO"
            )
    elif capability_target_contract != CAPABILITY_TARGET_CONTRACT_MATCHED_TOP1:
        raise ValueError(
            f"unsupported capability target contract: {capability_target_contract}"
        )
    capability_targets: dict[str, PrimitiveConsensus | ShardedMPRCache] = {}
    capability_target_provenance: dict[str, dict] = {}
    if factorized_mode:
        required_factorized_capability = {
            "official_capability_loss": bool(args.official_capability_loss),
            "dino_mpr_cache": bool(str(args.dino_mpr_cache).strip()),
            "sam3_mpr_cache": bool(str(args.sam3_mpr_cache).strip()),
            "expected_dino_v3_mpr_cache_sha256": bool(
                str(getattr(args, "expected_dino_v3_mpr_cache_sha256", ""))
            ),
            "expected_sam3_mpr_cache_sha256": bool(
                str(getattr(args, "expected_sam3_mpr_cache_sha256", ""))
            ),
            "factorized_capability_reference_mpr_cache": bool(
                str(
                    getattr(args, "factorized_capability_reference_mpr_cache", "")
                ).strip()
            ),
            "expected_factorized_capability_reference_mpr_cache_sha256": bool(
                str(
                    getattr(
                        args,
                        "expected_factorized_capability_reference_mpr_cache_sha256",
                        "",
                    )
                )
            ),
            "factorized_capability_legacy_receipt_correction": bool(
                str(
                    getattr(
                        args,
                        "factorized_capability_legacy_receipt_correction",
                        "",
                    )
                ).strip()
            ),
            "expected_factorized_capability_legacy_receipt_correction_sha256": bool(
                str(
                    getattr(
                        args,
                        "expected_factorized_capability_legacy_receipt_correction_sha256",
                        "",
                    )
                )
            ),
        }
        missing = sorted(
            name
            for name, present in required_factorized_capability.items()
            if not present
        )
        if missing:
            raise ValueError(
                "factorized RADIO requires a frozen matched capability cohort: "
                f"{missing}"
            )
        assert factorized_target is not None
        factorized_loader = (
            _load_factorized_exact_marginal_capability_targets
            if capability_target_contract
            == CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL
            else _load_factorized_matched_capability_targets
        )
        (
            capability_targets,
            capability_target_provenance,
            (capability_observation_provenance),
        ) = factorized_loader(
            factorized=factorized_target,
            reference_path=getattr(args, "factorized_capability_reference_mpr_cache"),
            reference_expected_sha256=str(
                getattr(
                    args,
                    "expected_factorized_capability_reference_mpr_cache_sha256",
                )
            ),
            dino_path=args.dino_mpr_cache,
            dino_expected_sha256=str(args.expected_dino_v3_mpr_cache_sha256),
            sam3_path=args.sam3_mpr_cache,
            sam3_expected_sha256=str(args.expected_sam3_mpr_cache_sha256),
            correction_path=getattr(
                args, "factorized_capability_legacy_receipt_correction"
            ),
            correction_expected_sha256=str(
                getattr(
                    args,
                    "expected_factorized_capability_legacy_receipt_correction_sha256",
                )
            ),
            radio_checkpoint_sha256=radio_hash,
        )
    else:
        for name, path in (
            ("dino_v3", args.dino_mpr_cache),
            ("sam3", args.sam3_mpr_cache),
        ):
            if not str(path).strip():
                continue
            target, provenance = _load_capability_mpr_target(
                path,
                expected_space=name,
                raw_cache=capability_observation_cache,
                raw_metadata=capability_observation_metadata,
                radio_checkpoint_sha256=radio_hash,
                expected_cache_sha256=str(
                    getattr(args, f"expected_{name}_mpr_cache_sha256", "")
                ),
                expected_feature_output_bundle_sha256=(expected_feature_bundle_sha256),
                target_contract=capability_target_contract,
            )
            capability_targets[name] = target
            capability_target_provenance[name] = provenance
    if capability_targets and not args.official_capability_loss:
        raise ValueError(
            "auxiliary capability MPR targets require --official-capability-loss"
        )
    relation_objective = str(
        getattr(args, "relation_objective", RELATION_OBJECTIVE_DISABLED)
    )
    relation_weight = float(getattr(args, "relation_weight", 0.0))
    relation_cache: dict[str, torch.Tensor] | None = None
    relation_provenance: dict = {}
    field_b_registration_provenance: dict = {}
    relation_path = str(getattr(args, "relation_triplet_cache", "")).strip()
    relation_expected_sha = str(
        getattr(args, "expected_relation_triplet_cache_sha256", "")
    )
    if relation_objective == RELATION_OBJECTIVE_DISABLED:
        if (
            relation_weight != 0.0
            or relation_path
            or relation_expected_sha
            or str(getattr(args, "field_b_experiment_registration", "")).strip()
            or str(
                getattr(
                    args,
                    "expected_field_b_experiment_registration_sha256",
                    "",
                )
            ).strip()
        ):
            raise ValueError(
                "disabled relation objective requires zero weight, no cache, and no Field-B registration"
            )
    elif relation_objective == RELATION_OBJECTIVE_FIELD_B:
        if capability_target_contract != CAPABILITY_TARGET_CONTRACT_FIELD_A or set(
            capability_targets
        ) != {"dino_v3", "sam3"}:
            raise ValueError("Field-B requires both Field-A exact capability targets")
        if not bool(args.official_capability_loss):
            raise ValueError("Field-B requires frozen official capability views")
        if relation_weight != FIELD_B_RELATION_WEIGHT:
            raise ValueError(
                f"Field-B v1 freezes the pre-existing relation weight at "
                f"{FIELD_B_RELATION_WEIGHT}"
            )
        registration, registration_sha, registration_path = load_json_object(
            args.field_b_experiment_registration,
            expected_sha256=(args.expected_field_b_experiment_registration_sha256),
            label="Field-B experiment registration",
        )
        if registration.get("schema_version") != (
            "canonical_field_b_boundary_relation_registration_v1"
        ):
            raise ValueError("Field-B experiment registration schema differs")
        source_hashes = dict(registration.get("source_hashes", {}))
        current_source_hashes = {
            "trainer": sha256_file(Path(__file__).resolve()),
            "losses": sha256_file(
                Path(__file__).parents[1] / "training" / "canonical_field_losses.py"
            ),
        }
        if any(
            source_hashes.get(key) != value
            for key, value in current_source_hashes.items()
        ):
            raise ValueError("Field-B registered training source SHA-256 differs")
        immutable_inputs = dict(registration.get("immutable_inputs", {}))
        expected_registered_inputs = {
            "field_a_checkpoint": str(args.expected_initial_field_checkpoint_sha256),
            "primary_raw_mpr": str(mpr_cache_sha256),
            "exact_observation_reference": str(
                args.expected_capability_observation_reference_mpr_cache_sha256
            ),
            "exact_dino_v3": str(args.expected_dino_v3_mpr_cache_sha256),
            "exact_sam3": str(args.expected_sam3_mpr_cache_sha256),
            "official_c_radio_checkpoint": str(radio_hash),
        }
        mismatched_inputs = {
            key: [dict(immutable_inputs.get(key, {})).get("sha256"), value]
            for key, value in expected_registered_inputs.items()
            if dict(immutable_inputs.get(key, {})).get("sha256") != value
        }
        if mismatched_inputs:
            raise ValueError(
                f"Field-B registered immutable inputs differ: {mismatched_inputs}"
            )
        registered_loss = dict(registration.get("loss_contract", {}))
        actual_loss = {
            "raw_mpr_weight": float(args.mpr_weight),
            "dino_v3_weight": float(args.dino_weight),
            "sam3_weight": float(args.sam3_weight),
            "relation_weight": relation_weight,
            "coefficient_weight": float(args.coefficient_weight),
            "basis_orthogonality_weight": float(args.basis_orthogonality_weight),
        }
        if any(registered_loss.get(key) != value for key, value in actual_loss.items()):
            raise ValueError("Field-B registered loss contract differs")
        registered_training = dict(registration.get("training_contract", {}))
        actual_training = {
            "epochs": int(args.epochs),
            "min_epochs": int(args.min_epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "validation_fraction": float(args.validation_fraction),
            "seed": int(args.seed),
        }
        if any(
            registered_training.get(key) != value
            for key, value in actual_training.items()
        ):
            raise ValueError("Field-B registered training contract differs")
        field_b_registration_provenance = {
            "path": str(registration_path),
            "sha256": registration_sha,
            "source_hashes": current_source_hashes,
        }
        dino_valid = torch.as_tensor(capability_targets["dino_v3"].valid).bool()
        sam_valid = torch.as_tensor(capability_targets["sam3"].valid).bool()
        if not torch.equal(dino_valid, sam_valid):
            raise ValueError("Field-B exact DINO/SAM valid rows differ")
        relation_cache, relation_provenance = _load_field_b_relation_triplets(
            relation_path,
            expected_sha256=relation_expected_sha,
            num_rows=_target_shape(consensus)[0],
            capability_valid=dino_valid,
            geometry_fingerprint=dict(cache.get("geometry_fingerprint", {})),
            expected_dino_sha256=str(args.expected_dino_v3_mpr_cache_sha256),
            expected_sam3_sha256=str(args.expected_sam3_mpr_cache_sha256),
        )
    else:
        raise ValueError(f"unsupported relation objective: {relation_objective}")
    primitive_positions = torch.as_tensor(cache["xyz"]).float().cpu()
    valid_rows = torch.where(consensus.valid)[0]
    consensus_num_rows, consensus_feature_dim = _target_shape(consensus)
    signature = FeatureSpaceSignature(
        radio_version=args.radio_version,
        radio_checkpoint_sha256=radio_hash,
        raw_feature_dim=consensus_feature_dim,
        adaptor_name="backbone",
        token_type="primitive",
        normalization=(
            "radio_raw_full"
            if factorized_mode
            else (
                "radio_direction_unit"
                if bool(metadata.get("normalize_each_view", False))
                else "radio_raw_full"
            )
        ),
        crop_policy=(
            "training_views_canonical_factorized_radio_v1"
            if factorized_mode
            else "training_views_depth_alpha_checked_mpr"
        ),
        # The field stores exactly the declared MPR RADIO semantics.  Semantic alignment is a
        # separately selected, frozen capability view and is never part of the
        # field checkpoint contract.
        semantic_alignment="none",
    )
    factorized_field_signature = (
        FactorizedRadioFieldSignature.create(signature) if factorized_mode else None
    )
    initial_field_provenance: dict = {}
    if str(args.initial_field_checkpoint).strip():
        initial_path = Path(args.initial_field_checkpoint)
        initial_expected_sha256 = (
            str(
                getattr(
                    args,
                    "expected_initial_field_checkpoint_sha256",
                    "",
                )
            )
            or None
        )
        if factorized_mode:
            assert factorized_field_signature is not None
            field, initial_payload, _initial_factorized_signature = (
                load_factorized_canonical_field_checkpoint(
                    initial_path,
                    map_location="cpu",
                    expected_sha256=initial_expected_sha256,
                    expected_signature=factorized_field_signature,
                )
            )
        else:
            field, initial_payload = load_canonical_field_checkpoint(
                initial_path,
                map_location="cpu",
                expected_sha256=initial_expected_sha256,
            )
        if initial_payload.get("benchmark_masks_opened", False) or initial_payload.get(
            "text_queries_opened", False
        ):
            raise ValueError("initial field used benchmark masks or text queries")
        if factorized_mode and (
            initial_payload.get("factorized_cache_sha256") != mpr_cache_sha256
            or initial_payload.get("feature_output_bundle_sha256")
            != expected_feature_bundle_sha256
        ):
            raise ValueError("initial factorized field source lineage differs")
        if field.num_gaussians != consensus_num_rows:
            raise ValueError("initial field Gaussian count differs from the MPR cache")
        if field.decoder.feature_dim != consensus_feature_dim:
            raise ValueError("initial field RADIO dimension differs from the MPR cache")
        signature_compatibility = (
            {"factorized_schema_v2_exact": True}
            if factorized_mode
            else _assert_initial_field_signature_compatible(
                signature,
                field.signature,
                allow_legacy_normalization=(
                    observation_contract_mode == "compatible-legacy"
                ),
            )
        )
        expected_geometry = str(
            cache.get("geometry_fingerprint", {}).get("xyz_sha256", "")
        )
        actual_geometry = str(
            initial_payload.get("geometry_fingerprint", {}).get("xyz_sha256", "")
        )
        if not expected_geometry or actual_geometry != expected_geometry:
            raise ValueError("initial field geometry differs from the MPR cache")
        if factorized_mode:
            if field.reliability.shape != (consensus_num_rows, 0):
                raise ValueError("factorized initial field reconstructed reliability")
        else:
            if field.reliability.shape != consensus.reliability.shape:
                raise ValueError(
                    "initial field reliability shape differs from the MPR cache"
                )
            # Registration support is part of the current raw target contract.
            # Updating this fixed buffer affects control and treatment identically;
            # no learned state or query signal is introduced.
            with torch.no_grad():
                field.reliability.copy_(consensus.reliability)
        field = field.to(device)
        basis_fit_report = dict(initial_payload.get("basis_fit_report", {}))
        initial_field_provenance = {
            "path": str(initial_path.resolve()),
            "sha256": sha256_file(initial_path),
            "source_final_metrics": initial_payload.get("final_metrics", {}),
            "source_final_capability_metrics": initial_payload.get(
                "final_capability_metrics", {}
            ),
            "source_capability_target_mode": initial_payload.get(
                "capability_target_mode", "legacy_or_unspecified"
            ),
            "source_training_epochs": len(initial_payload.get("history", [])),
            "architecture_reused_exactly": True,
            "learned_state_reinitialized": False,
            "fixed_reliability_refreshed_from_current_raw_mpr": (not factorized_mode),
            "factorized_target_reliability_entered_field": False,
            "signature_compatibility": signature_compatibility,
        }
    else:
        if valid_rows.numel() < int(args.coefficient_dim):
            raise ValueError("too few valid primitive targets for the requested basis")
        basis_values = _basis_fit_values(
            consensus,
            valid_rows,
            max_samples=int(args.pca_samples),
            seed=int(args.seed),
        ).to(basis_fit_device)
        decoder, fit_report = fit_affine_basis(
            basis_values,
            int(args.coefficient_dim),
            standardize=not args.no_standardize,
            max_samples=(
                int(args.pca_samples)
                if not isinstance(consensus, ShardedMPRCache)
                else int(basis_values.shape[0])
            ),
            seed=int(args.seed),
            trainable_basis=not args.freeze_basis,
        )
        del basis_values
        basis_fit_report = asdict(fit_report)
        local_dim = (
            int(args.local_dim)
            if int(args.local_dim) > 0
            else int(args.coefficient_dim)
        )
        use_fusion = bool(args.primitive_fusion)
        if (
            local_dim != int(args.coefficient_dim) or int(args.spatial_coarse_dim) > 0
        ) and not use_fusion:
            raise ValueError("compact local/spatial codes require --primitive-fusion")
        spatial_hash = None
        if int(args.spatial_coarse_dim) > 0:
            spatial_hash = {
                "output_dim": int(args.spatial_coarse_dim),
                "num_levels": int(args.hash_levels),
                "features_per_level": int(args.hash_features_per_level),
                "log2_hashmap_size": int(args.hash_log2_size),
                "base_resolution": int(args.hash_base_resolution),
                "max_resolution": int(args.hash_max_resolution),
                "hidden_dim": int(args.hash_hidden_dim),
            }
        field = CanonicalGaussianField(
            num_gaussians=consensus_num_rows,
            decoder=decoder,
            signature=signature,
            local_dim=local_dim,
            coarse_dim=int(args.spatial_coarse_dim),
            primitive_positions=(
                primitive_positions if spatial_hash is not None else None
            ),
            spatial_hash=spatial_hash,
            reliability=None if factorized_mode else consensus.reliability,
            fusion_reliability=(
                False if factorized_mode else bool(args.fusion_reliability)
            ),
            hidden_dim=int(args.hidden_dim),
            fusion_residual_blocks=int(getattr(args, "fusion_residual_blocks", 0)),
            use_fusion=use_fusion,
        ).to(device)
        with torch.no_grad():
            if local_dim == int(args.coefficient_dim):
                if isinstance(consensus, ShardedMPRCache):
                    _initialize_codes_streaming(field, decoder, consensus)
                    encoded = None
                else:
                    encoded = decoder.encode(consensus.targets.to(device))
            else:
                local_basis_values = _basis_fit_values(
                    consensus,
                    valid_rows,
                    max_samples=int(args.pca_samples),
                    seed=int(args.seed),
                ).to(basis_fit_device)
                local_decoder, _local_fit_report = fit_affine_basis(
                    local_basis_values,
                    local_dim,
                    standardize=not args.no_standardize,
                    max_samples=(
                        int(args.pca_samples)
                        if not isinstance(consensus, ShardedMPRCache)
                        else int(local_basis_values.shape[0])
                    ),
                    seed=int(args.seed),
                    trainable_basis=False,
                )
                if isinstance(consensus, ShardedMPRCache):
                    local_decoder.to(device)
                    _initialize_codes_streaming(field, local_decoder, consensus)
                    local_decoder.cpu()
                    encoded = None
                else:
                    encoded = local_decoder.encode(
                        consensus.targets.to(basis_fit_device)
                    ).to(device)
                del local_basis_values
                if field.fusion is None:
                    raise RuntimeError("local compression requires primitive fusion")
                weight, bias = _cross_basis_projection(
                    local_decoder.cpu(), decoder.cpu()
                )
                decoder.to(device)
                field.fusion.initialize_base_projection(weight, bias)
            if encoded is not None:
                field.local_codes.copy_(encoded)

    official_views = None
    if args.official_capability_loss:
        official_views = FrozenRadioViews.from_radio_checkpoint(
            args.radio_checkpoint,
            expected_sha256=radio_hash,
        ).to(device)
    source_multiteacher = None
    source_descriptor_projector = None
    control_source_multiteacher_metrics: dict[str, float] = {}
    source_manifest_path = str(
        getattr(args, "source_only_multiteacher_manifest", "")
    ).strip()
    source_manifest_expected_sha256 = str(
        getattr(
            args,
            "expected_source_only_multiteacher_manifest_sha256",
            "",
        )
    ).strip()
    if source_manifest_path:
        if not factorized_mode or factorized_target is None:
            raise ValueError(
                "source-only multiteacher distillation requires "
                "canonical-factorized-radio-v1"
            )
        if not str(args.initial_field_checkpoint).strip() or re.fullmatch(
            r"[0-9a-f]{64}",
            str(args.expected_initial_field_checkpoint_sha256),
        ) is None:
            raise ValueError(
                "source-only multiteacher distillation requires one hash-bound "
                "control field initialization"
            )
        if re.fullmatch(r"[0-9a-f]{64}", source_manifest_expected_sha256) is None:
            raise ValueError(
                "source-only multiteacher manifest requires a caller SHA-256"
            )
        source_multiteacher = load_source_only_multiteacher_bundle(
            source_manifest_path,
            expected_sha256=source_manifest_expected_sha256,
            expected_xyz=factorized_target.xyz,
            expected_valid=factorized_target.valid,
            expected_factorized_radio_cache_sha256=mpr_cache_sha256,
            expected_field_checkpoint_sha256=str(
                args.expected_initial_field_checkpoint_sha256
            ),
            expected_radio_checkpoint_sha256=radio_hash,
        )
        source_descriptor_projector = FrozenPrimitiveSiglipProxy.from_radio_checkpoint(
            args.radio_checkpoint, expected_sha256=radio_hash
        ).to(device)
        field.eval()
        control_source_multiteacher_metrics.update(
            source_descriptor_metrics(
                field,
                projector=source_descriptor_projector,
                teacher=source_multiteacher.primitive,
                rows=valid_rows,
                batch_size=int(args.eval_batch_size),
            )
        )
        control_source_multiteacher_metrics.update(
            source_relation_metrics(
                field,
                projector=source_descriptor_projector,
                teacher=source_multiteacher.primitive,
                relation=source_multiteacher.relation,
            )
        )
    source_sam_structure = None
    source_sam_training_teacher = None
    control_source_sam_metrics: dict[str, float] = {}
    source_sam_manifest_path = str(
        getattr(args, "source_only_sam_structure_manifest", "")
    ).strip()
    source_sam_manifest_expected_sha256 = str(
        getattr(
            args,
            "expected_source_only_sam_structure_manifest_sha256",
            "",
        )
    ).strip()
    if source_sam_manifest_path:
        if not str(args.initial_field_checkpoint).strip() or re.fullmatch(
            r"[0-9a-f]{64}",
            str(args.expected_initial_field_checkpoint_sha256),
        ) is None:
            raise ValueError(
                "source-only official-SAM structure supervision requires one "
                "hash-bound control field initialization"
            )
        if re.fullmatch(r"[0-9a-f]{64}", source_sam_manifest_expected_sha256) is None:
            raise ValueError(
                "source-only official-SAM structure manifest requires a caller SHA-256"
            )
        source_sam_structure = load_source_only_sam_structure_bundle(
            source_sam_manifest_path,
            expected_sha256=source_sam_manifest_expected_sha256,
            expected_xyz=(
                factorized_target.xyz
                if factorized_target is not None
                else torch.as_tensor(cache["xyz"]).float().cpu()
            ),
            expected_valid=(
                factorized_target.valid
                if factorized_target is not None
                else torch.as_tensor(cache["valid"]).bool().cpu()
            ),
            expected_canonical_radio_cache_sha256=mpr_cache_sha256,
            expected_field_checkpoint_sha256=str(
                args.expected_initial_field_checkpoint_sha256
            ),
        )
        field.eval()
        control_source_sam_metrics = source_sam_structure_metrics(
            field, teacher=source_sam_structure.teacher
        )
    source_sam_relative = None
    source_sam_relative_training_teacher = None
    control_source_sam_relative_pair_metrics: dict[str, float] = {}
    control_source_sam_relative_metrics: dict[str, float] = {}
    source_sam_relative_manifest_path = str(
        getattr(args, "source_only_sam_relative_structure_manifest", "")
    ).strip()
    source_sam_relative_expected_sha256 = str(
        getattr(
            args,
            "expected_source_only_sam_relative_structure_manifest_sha256",
            "",
        )
    ).strip()
    if source_sam_relative_manifest_path:
        if not str(args.initial_field_checkpoint).strip() or re.fullmatch(
            r"[0-9a-f]{64}",
            str(args.expected_initial_field_checkpoint_sha256),
        ) is None:
            raise ValueError(
                "source-only official-SAM relative supervision requires one "
                "hash-bound control field initialization"
            )
        if re.fullmatch(r"[0-9a-f]{64}", source_sam_relative_expected_sha256) is None:
            raise ValueError(
                "source-only official-SAM relative manifest requires a caller SHA-256"
            )
        source_sam_relative = load_source_only_sam_relative_bundle(
            source_sam_relative_manifest_path,
            expected_sha256=source_sam_relative_expected_sha256,
            expected_xyz=(
                factorized_target.xyz
                if factorized_target is not None
                else torch.as_tensor(cache["xyz"]).float().cpu()
            ),
            expected_valid=(
                factorized_target.valid
                if factorized_target is not None
                else torch.as_tensor(cache["valid"]).bool().cpu()
            ),
            expected_canonical_radio_cache_sha256=mpr_cache_sha256,
            expected_field_checkpoint_sha256=str(
                args.expected_initial_field_checkpoint_sha256
            ),
            control_field=field,
        )
        field.eval()
        control_source_sam_relative_pair_metrics = source_sam_structure_metrics(
            field, teacher=source_sam_relative.teacher.pair_teacher
        )
        control_source_sam_relative_metrics = source_sam_relative_metrics(
            field, teacher=source_sam_relative.teacher
        )
    source_sam_capability_relative = None
    source_sam_capability_relative_training_teacher = None
    control_source_sam_capability_pair_metrics: dict[str, float] = {}
    control_source_sam_capability_relative_metrics: dict[str, float] = {}
    source_sam_capability_relative_manifest_path = str(
        getattr(
            args,
            "source_only_sam_capability_relative_structure_manifest",
            "",
        )
    ).strip()
    source_sam_capability_relative_expected_sha256 = str(
        getattr(
            args,
            "expected_source_only_sam_capability_relative_structure_manifest_sha256",
            "",
        )
    ).strip()
    if source_sam_capability_relative_manifest_path:
        if official_views is None:
            raise ValueError(
                "source-only SAM capability-relative supervision requires frozen "
                "official DINOv3 and SAM3 adaptors"
            )
        if not str(args.initial_field_checkpoint).strip() or re.fullmatch(
            r"[0-9a-f]{64}",
            str(args.expected_initial_field_checkpoint_sha256),
        ) is None:
            raise ValueError(
                "source-only SAM capability-relative supervision requires one "
                "hash-bound control field initialization"
            )
        if re.fullmatch(
            r"[0-9a-f]{64}",
            source_sam_capability_relative_expected_sha256,
        ) is None:
            raise ValueError(
                "source-only SAM capability-relative manifest requires a caller SHA-256"
            )
        source_sam_capability_relative = (
            load_source_only_sam_capability_relative_bundle(
                source_sam_capability_relative_manifest_path,
                expected_sha256=(
                    source_sam_capability_relative_expected_sha256
                ),
                expected_xyz=(
                    factorized_target.xyz
                    if factorized_target is not None
                    else torch.as_tensor(cache["xyz"]).float().cpu()
                ),
                expected_valid=(
                    factorized_target.valid
                    if factorized_target is not None
                    else torch.as_tensor(cache["valid"]).bool().cpu()
                ),
                expected_canonical_radio_cache_sha256=mpr_cache_sha256,
                expected_field_checkpoint_sha256=str(
                    args.expected_initial_field_checkpoint_sha256
                ),
                control_field=field,
                official_views=official_views,
            )
        )
        field.eval()
        control_source_sam_capability_pair_metrics = (
            source_sam_capability_pair_metrics(
                field,
                official_views=official_views,
                teacher=source_sam_capability_relative.teacher,
            )
        )
        control_source_sam_capability_relative_metrics = (
            source_sam_capability_relative_metrics(
                field,
                official_views=official_views,
                teacher=source_sam_capability_relative.teacher,
            )
        )
        # Reuse the report's stable pair/triplet slots while keeping the v3
        # capability-relation metric names explicit inside those mappings.
        control_source_sam_relative_pair_metrics = dict(
            control_source_sam_capability_pair_metrics
        )
        control_source_sam_relative_metrics = dict(
            control_source_sam_capability_relative_metrics
        )
    factorized_reliability_policy = _factorized_loss_reliability_policy(
        factorized_target,
        capability_target_contract=capability_target_contract,
    )
    loss_config = CanonicalFieldLossConfig(
        mpr_weight=float(args.mpr_weight),
        dino_weight=float(args.dino_weight if official_views is not None else 0.0),
        sam3_weight=float(args.sam3_weight if official_views is not None else 0.0),
        relation_weight=relation_weight,
        coefficient_weight=float(args.coefficient_weight),
        basis_orthogonality_weight=float(args.basis_orthogonality_weight),
    )
    capability_reliability_policy = (
        "field_a_boundary_safe"
        if capability_target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_A
        else (
            "field_c_visibility_safe"
            if capability_target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_C
            else (
                FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE
                if capability_target_contract
                == CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL
                else "legacy_mean"
            )
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    order = valid_rows[torch.randperm(valid_rows.numel(), generator=generator)]
    validation_count = max(
        1, int(round(order.numel() * float(args.validation_fraction)))
    )
    validation_rows = order[:validation_count]
    training_rows = order[validation_count:]
    if training_rows.numel() == 0:
        training_rows = validation_rows
    if relation_cache is not None:
        triplets = int(relation_cache["teacher_margin"].numel())
        training_mask = torch.zeros(consensus_num_rows, dtype=torch.bool)
        training_mask[training_rows] = True
        positive = relation_cache["pair_index"][:, :triplets]
        negative = relation_cache["pair_index"][:, triplets:]
        keep = training_mask[positive].all(dim=0) & training_mask[negative].all(dim=0)
        if not bool(keep.any()):
            raise ValueError("Field-B has no triplets wholly inside training rows")
        relation_cache = {
            "pair_index": torch.cat([positive[:, keep], negative[:, keep]], dim=1),
            "teacher_margin": relation_cache["teacher_margin"][keep],
            "boundary_channel": relation_cache["boundary_channel"][keep],
        }
        relation_provenance["training_triplets_after_validation_exclusion"] = int(
            keep.sum()
        )
        relation_provenance["validation_rows_used_by_relation"] = False
    if source_sam_structure is not None:
        training_mask = torch.zeros(consensus_num_rows, dtype=torch.bool)
        training_mask[training_rows] = True
        source_sam_keep = training_mask[
            source_sam_structure.teacher.global_edge_index
        ].all(dim=0)
        if not bool(source_sam_keep.any()):
            raise ValueError(
                "source-only official-SAM structure has no training-only edges"
            )
        source_sam_training_teacher = source_sam_structure.teacher.subset(
            source_sam_keep
        )
        if (
            not bool(source_sam_training_teacher.same_relation.any())
            or not bool((~source_sam_training_teacher.same_relation).any())
        ):
            raise ValueError(
                "source-only official-SAM training split requires both relation classes"
            )
    if source_sam_relative is not None:
        training_mask = torch.zeros(consensus_num_rows, dtype=torch.bool)
        training_mask[training_rows] = True
        relative_pair_keep = training_mask[
            source_sam_relative.teacher.pair_teacher.global_edge_index
        ].all(dim=0)
        relative_triplet_keep = training_mask[
            source_sam_relative.teacher.triplet_index
        ].all(dim=0)
        if not bool(relative_pair_keep.any()) or not bool(relative_triplet_keep.any()):
            raise ValueError(
                "source-only official-SAM relative teacher has no training-only support"
            )
        source_sam_relative_training_teacher = (
            source_sam_relative.teacher.training_subset(
                pair_keep=relative_pair_keep,
                triplet_keep=relative_triplet_keep,
            )
        )
        if (
            not bool(
                source_sam_relative_training_teacher.pair_teacher.same_relation.any()
            )
            or not bool(
                (~source_sam_relative_training_teacher.pair_teacher.same_relation).any()
            )
        ):
            raise ValueError(
                "source-only official-SAM relative guard requires both relation classes"
            )
    if source_sam_capability_relative is not None:
        training_mask = torch.zeros(consensus_num_rows, dtype=torch.bool)
        training_mask[training_rows] = True
        capability_pair_keep = training_mask[
            source_sam_capability_relative.teacher.pair_teacher.global_edge_index
        ].all(dim=0)
        capability_triplet_keep = training_mask[
            source_sam_capability_relative.teacher.triplet_index
        ].all(dim=0)
        if not bool(capability_pair_keep.any()) or not bool(
            capability_triplet_keep.any()
        ):
            raise ValueError(
                "source-only official-SAM capability-relative teacher has no "
                "training-only support"
            )
        source_sam_capability_relative_training_teacher = (
            source_sam_capability_relative.teacher.training_subset(
                pair_keep=capability_pair_keep,
                triplet_keep=capability_triplet_keep,
            )
        )
        if (
            not bool(
                source_sam_capability_relative_training_teacher.pair_teacher.same_relation.any()
            )
            or not bool(
                (~source_sam_capability_relative_training_teacher.pair_teacher.same_relation).any()
            )
        ):
            raise ValueError(
                "source-only official-SAM capability-relative guard requires both "
                "relation classes"
            )
    active_source_sam_relative_training_teacher = (
        source_sam_capability_relative_training_teacher
        if source_sam_capability_relative_training_teacher is not None
        else source_sam_relative_training_teacher
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in field.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        epoch_order = training_rows[
            torch.randperm(training_rows.numel(), generator=generator)
        ]
        totals: list[float] = []
        relation_totals: list[float] = []
        source_descriptor_totals: list[float] = []
        source_relation_totals: list[float] = []
        source_sam_structure_totals: list[float] = []
        source_sam_relative_totals: list[float] = []
        relation_order = (
            torch.randperm(
                relation_cache["teacher_margin"].numel(), generator=generator
            )
            if relation_cache is not None
            else torch.empty(0, dtype=torch.long)
        )
        source_relation_order = (
            torch.randperm(
                source_multiteacher.relation.num_edges, generator=generator
            )
            if source_multiteacher is not None
            else torch.empty(0, dtype=torch.long)
        )
        source_sam_order = (
            torch.randperm(
                source_sam_training_teacher.num_edges, generator=generator
            )
            if source_sam_training_teacher is not None
            else torch.empty(0, dtype=torch.long)
        )
        source_sam_relative_triplet_order = (
            torch.randperm(
                active_source_sam_relative_training_teacher.num_triplets,
                generator=generator,
            )
            if active_source_sam_relative_training_teacher is not None
            else torch.empty(0, dtype=torch.long)
        )
        source_sam_relative_guard_order = (
            torch.randperm(
                active_source_sam_relative_training_teacher.pair_teacher.num_edges,
                generator=generator,
            )
            if active_source_sam_relative_training_teacher is not None
            else torch.empty(0, dtype=torch.long)
        )
        epoch_batches = max(
            1,
            (epoch_order.numel() + int(args.batch_size) - 1) // int(args.batch_size),
        )
        if source_sam_order.numel():
            source_sam_order = source_sam_order[
                : min(
                    int(source_sam_order.numel()),
                    epoch_batches * SAM_STRUCTURE_BATCH_SIZE,
                )
            ]
        if source_sam_relative_triplet_order.numel():
            source_sam_relative_triplet_order = source_sam_relative_triplet_order[
                : min(
                    int(source_sam_relative_triplet_order.numel()),
                    epoch_batches * SAM_RELATIVE_TRIPLET_BATCH_SIZE,
                )
            ]
            source_sam_relative_guard_order = source_sam_relative_guard_order[
                : min(
                    int(source_sam_relative_guard_order.numel()),
                    epoch_batches * SAM_RELATIVE_GUARD_BATCH_SIZE,
                )
            ]
        field.train()
        for batch_index, start in enumerate(
            range(0, epoch_order.numel(), int(args.batch_size))
        ):
            rows = epoch_order[start : start + int(args.batch_size)].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _stats = canonical_primitive_loss(
                field,
                consensus,
                rows,
                official_views=official_views,
                capability_targets=capability_targets,
                capability_reliability_policy=capability_reliability_policy,
                factorized_target=factorized_target,
                factorized_reliability_policy=factorized_reliability_policy,
                config=loss_config,
            )
            if source_multiteacher is not None:
                assert source_descriptor_projector is not None
                source_descriptor_loss, _source_stats = (
                    primitive_source_descriptor_loss(
                        field.radio_features(rows),
                        rows.detach().cpu(),
                        projector=source_descriptor_projector,
                        teacher=source_multiteacher.primitive,
                    )
                )
                loss = loss + SOURCE_DESCRIPTOR_WEIGHT * source_descriptor_loss
                source_descriptor_totals.append(
                    float(source_descriptor_loss.detach())
                )
                relation_start = batch_index * SOURCE_RELATION_BATCH_SIZE
                relation_stop = min(
                    relation_start + SOURCE_RELATION_BATCH_SIZE,
                    int(source_relation_order.numel()),
                )
                selected_source_edges = source_relation_order[
                    relation_start:relation_stop
                ]
                if selected_source_edges.numel():
                    source_relation_loss = source_relation_batch_loss(
                        field,
                        projector=source_descriptor_projector,
                        teacher=source_multiteacher.primitive,
                        relation=source_multiteacher.relation,
                        edge_indices=selected_source_edges,
                    )
                    loss = loss + SOURCE_RELATION_WEIGHT * source_relation_loss
                    source_relation_totals.append(
                        float(source_relation_loss.detach())
                    )
            if source_sam_training_teacher is not None:
                sam_start = batch_index * source_sam_order.numel() // epoch_batches
                sam_stop = (
                    (batch_index + 1) * source_sam_order.numel() // epoch_batches
                )
                selected_sam_edges = source_sam_order[sam_start:sam_stop]
                if selected_sam_edges.numel():
                    source_sam_loss, _source_sam_stats = (
                        source_sam_structure_batch_loss(
                            field,
                            teacher=source_sam_training_teacher,
                            edge_indices=selected_sam_edges,
                        )
                    )
                    loss = loss + SAM_STRUCTURE_WEIGHT * source_sam_loss
                    source_sam_structure_totals.append(
                        float(source_sam_loss.detach())
                    )
            if active_source_sam_relative_training_teacher is not None:
                relative_triplet_start = (
                    batch_index
                    * source_sam_relative_triplet_order.numel()
                    // epoch_batches
                )
                relative_triplet_stop = (
                    (batch_index + 1)
                    * source_sam_relative_triplet_order.numel()
                    // epoch_batches
                )
                relative_guard_start = (
                    batch_index
                    * source_sam_relative_guard_order.numel()
                    // epoch_batches
                )
                relative_guard_stop = (
                    (batch_index + 1)
                    * source_sam_relative_guard_order.numel()
                    // epoch_batches
                )
                selected_relative_triplets = source_sam_relative_triplet_order[
                    relative_triplet_start:relative_triplet_stop
                ]
                selected_relative_guards = source_sam_relative_guard_order[
                    relative_guard_start:relative_guard_stop
                ]
                if (
                    selected_relative_triplets.numel()
                    and selected_relative_guards.numel()
                ):
                    if source_sam_capability_relative_training_teacher is not None:
                        assert official_views is not None
                        source_sam_relative_loss, _source_sam_relative_stats = (
                            source_sam_capability_relative_batch_loss(
                                field,
                                official_views=official_views,
                                teacher=(
                                    source_sam_capability_relative_training_teacher
                                ),
                                triplet_indices=selected_relative_triplets,
                                guard_edge_indices=selected_relative_guards,
                            )
                        )
                    else:
                        source_sam_relative_loss, _source_sam_relative_stats = (
                            source_sam_relative_batch_loss(
                                field,
                                teacher=source_sam_relative_training_teacher,
                                triplet_indices=selected_relative_triplets,
                                guard_edge_indices=selected_relative_guards,
                            )
                        )
                    loss = loss + SAM_STRUCTURE_WEIGHT * source_sam_relative_loss
                    source_sam_relative_totals.append(
                        float(source_sam_relative_loss.detach())
                    )
            if relation_cache is not None:
                relation_start = batch_index * relation_order.numel() // epoch_batches
                relation_stop = (
                    (batch_index + 1) * relation_order.numel() // epoch_batches
                )
                selected_triplets = relation_order[relation_start:relation_stop]
                relation_loss = _field_b_relation_batch_loss(
                    field,
                    official_views,
                    relation_cache,
                    selected_triplets,
                )
                loss = loss + loss_config.relation_weight * relation_loss
                relation_totals.append(float(relation_loss.detach()))
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach()))
        field.eval()
        validation = _reconstruction_metrics(
            field, consensus, validation_rows, int(args.eval_batch_size)
        )
        if factorized_target is not None:
            validation.update(
                _factorized_amplitude_metrics(
                    field,
                    factorized_target,
                    validation_rows,
                    int(args.eval_batch_size),
                )
            )
        if official_views is not None and capability_targets:
            validation.update(
                _capability_reconstruction_metrics(
                    field,
                    official_views,
                    capability_targets,
                    validation_rows,
                    int(args.eval_batch_size),
                )
            )
        if source_multiteacher is not None:
            assert source_descriptor_projector is not None
            validation.update(
                source_descriptor_metrics(
                    field,
                    projector=source_descriptor_projector,
                    teacher=source_multiteacher.primitive,
                    rows=validation_rows,
                    batch_size=int(args.eval_batch_size),
                )
            )
        record = {
            "epoch": epoch + 1,
            "loss": sum(totals) / max(1, len(totals)),
            "relation_ranking_loss": (
                sum(relation_totals) / len(relation_totals) if relation_totals else 0.0
            ),
            "source_descriptor_loss": (
                sum(source_descriptor_totals) / len(source_descriptor_totals)
                if source_descriptor_totals
                else 0.0
            ),
            "source_relation_loss": (
                sum(source_relation_totals) / len(source_relation_totals)
                if source_relation_totals
                else 0.0
            ),
            "source_sam_structure_loss": (
                sum(source_sam_structure_totals)
                / len(source_sam_structure_totals)
                if source_sam_structure_totals
                else 0.0
            ),
            "source_sam_relative_loss": (
                sum(source_sam_relative_totals)
                / len(source_sam_relative_totals)
                if source_sam_relative_totals
                else 0.0
            ),
            **validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if (
            not factorized_mode
            and epoch + 1 >= int(args.min_epochs)
            and validation["mean_cosine"] >= float(args.target_cosine)
        ):
            break

    field.eval()
    final_metrics = _reconstruction_metrics(
        field, consensus, valid_rows, int(args.eval_batch_size)
    )
    if factorized_target is not None:
        final_metrics.update(
            _factorized_amplitude_metrics(
                field,
                factorized_target,
                valid_rows,
                int(args.eval_batch_size),
            )
        )
    final_capability_metrics = (
        _capability_reconstruction_metrics(
            field,
            official_views,
            capability_targets,
            valid_rows,
            int(args.eval_batch_size),
        )
        if official_views is not None and capability_targets
        else {}
    )
    final_source_multiteacher_metrics: dict[str, float] = {}
    if source_multiteacher is not None:
        assert source_descriptor_projector is not None
        final_source_multiteacher_metrics.update(
            source_descriptor_metrics(
                field,
                projector=source_descriptor_projector,
                teacher=source_multiteacher.primitive,
                rows=valid_rows,
                batch_size=int(args.eval_batch_size),
            )
        )
    source_gate_decision: dict = {}
    if source_multiteacher is not None:
        final_source_multiteacher_metrics.update(
            source_relation_metrics(
                field,
                projector=source_descriptor_projector,
                teacher=source_multiteacher.primitive,
                relation=source_multiteacher.relation,
            )
        )
        source_gate_decision = evaluate_source_only_gates(
            manifest=source_multiteacher.manifest,
            control_source_metrics=control_source_multiteacher_metrics,
            candidate_primary_metrics=final_metrics,
            candidate_capability_metrics=final_capability_metrics,
            candidate_source_metrics=final_source_multiteacher_metrics,
            primitive_state_summary=source_multiteacher.primitive_state_summary,
        )
    final_source_sam_metrics: dict[str, float] = {}
    source_sam_gate_decision: dict[str, object] = {}
    if source_sam_structure is not None:
        final_source_sam_metrics = source_sam_structure_metrics(
            field, teacher=source_sam_structure.teacher
        )
        source_sam_gate_decision = evaluate_source_sam_structure_gates(
            control=control_source_sam_metrics,
            candidate=final_source_sam_metrics,
        )
    final_source_sam_relative_pair_metrics: dict[str, float] = {}
    final_source_sam_relative_metrics: dict[str, float] = {}
    source_sam_relative_gate_decision: dict[str, object] = {}
    if source_sam_relative is not None:
        final_source_sam_relative_pair_metrics = source_sam_structure_metrics(
            field, teacher=source_sam_relative.teacher.pair_teacher
        )
        final_source_sam_relative_metrics = source_sam_relative_metrics(
            field, teacher=source_sam_relative.teacher
        )
        source_sam_relative_structure_decision = evaluate_source_sam_relative_gates(
            control_pair=control_source_sam_relative_pair_metrics,
            candidate_pair=final_source_sam_relative_pair_metrics,
            control_relative=control_source_sam_relative_metrics,
            candidate_relative=final_source_sam_relative_metrics,
        )
        source_sam_relative_gate_decision = (
            evaluate_source_sam_relative_full_gates(
                manifest=source_sam_relative.manifest,
                control_primary=initial_field_provenance["source_final_metrics"],
                candidate_primary=final_metrics,
                control_capability=initial_field_provenance[
                    "source_final_capability_metrics"
                ],
                candidate_capability=final_capability_metrics,
                structure_decision=source_sam_relative_structure_decision,
            )
        )
    elif source_sam_capability_relative is not None:
        assert official_views is not None
        final_source_sam_relative_pair_metrics = (
            source_sam_capability_pair_metrics(
                field,
                official_views=official_views,
                teacher=source_sam_capability_relative.teacher,
            )
        )
        final_source_sam_relative_metrics = (
            source_sam_capability_relative_metrics(
                field,
                official_views=official_views,
                teacher=source_sam_capability_relative.teacher,
            )
        )
        source_sam_relative_structure_decision = (
            evaluate_source_sam_capability_relative_gates(
                control_pair=control_source_sam_relative_pair_metrics,
                candidate_pair=final_source_sam_relative_pair_metrics,
                control_relative=control_source_sam_relative_metrics,
                candidate_relative=final_source_sam_relative_metrics,
            )
        )
        source_sam_relative_gate_decision = (
            evaluate_source_sam_capability_relative_full_gates(
                manifest=source_sam_capability_relative.manifest,
                control_primary=initial_field_provenance["source_final_metrics"],
                candidate_primary=final_metrics,
                control_capability=initial_field_provenance[
                    "source_final_capability_metrics"
                ],
                candidate_capability=final_capability_metrics,
                structure_decision=source_sam_relative_structure_decision,
            )
        )
    field.cpu()
    output.parent.mkdir(parents=True, exist_ok=True)
    architecture = {
        "num_gaussians": field.num_gaussians,
        "feature_dim": field.decoder.feature_dim,
        "coefficient_dim": field.decoder.coefficient_dim,
        "local_dim": field.local_codes.shape[1],
        "coarse_dim": field.coarse_dim,
        "spatial_hash": (
            field.spatial_encoder.architecture()
            if field.spatial_encoder is not None
            else None
        ),
        "position_storage": "normalized_fp16" if field.coarse_dim else "none",
        "fusion_reliability": field.fusion_reliability,
        "hidden_dim": (
            int(field.fusion.network[0].out_features)
            if field.fusion is not None
            else int(args.hidden_dim)
        ),
        "fusion_residual_blocks": int(field.fusion_residual_blocks),
        "use_fusion": field.fusion is not None,
        "trainable_basis": bool(field.decoder.basis.requires_grad),
        "trainable_statistics": bool(
            field.decoder.mean.requires_grad or field.decoder.scale.requires_grad
        ),
    }
    training_config = {key: value for key, value in vars(args).items()}
    training_config_sha256 = hashlib.sha256(
        json.dumps(
            training_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    basis_conditioning = validate_basis_conditioning(field.decoder.basis).to_dict()
    active_source_sam_relative_bundle = (
        source_sam_capability_relative
        if source_sam_capability_relative is not None
        else source_sam_relative
    )
    payload = {
        "architecture": architecture,
        "state_dict": field.state_dict(),
        "geometry_fingerprint": cache.get("geometry_fingerprint", {}),
        "mpr_cache": str(mpr_cache_path),
        "mpr_cache_sha256": mpr_cache_sha256,
        "mpr_cache_storage": raw_mpr_storage_provenance,
        "mpr_cache_metadata": metadata,
        "feature_output_bundle_sha256": expected_feature_bundle_sha256,
        "basis_fit_report": basis_fit_report,
        "basis_conditioning": basis_conditioning,
        "initial_field_checkpoint": initial_field_provenance,
        "loss_config": asdict(loss_config),
        "capability_target_mode": (
            "official_adaptor_then_exact_raster_adjoint_contribution_mpr"
            if capability_target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_A
            else (
                "official_adaptor_then_exact_center_plus_uncertainty_mpr"
                if capability_target_contract == CAPABILITY_TARGET_CONTRACT_FIELD_C
                else (
                    "official_adaptor_then_shared_exact_marginal_mpr"
                    if capability_target_contract
                    == CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL
                    and capability_targets
                    else (
                        "official_adaptor_then_geometry_matched_mpr"
                        if capability_targets
                        else (
                            "adaptor_of_raw_mpr_target"
                            if official_views is not None
                            else "none"
                        )
                    )
                )
            )
        ),
        "capability_target_contract": capability_target_contract,
        "capability_reliability_policy": capability_reliability_policy,
        "factorized_reliability_policy": factorized_reliability_policy,
        "capability_observation_reference": capability_observation_provenance,
        "capability_mpr_targets": capability_target_provenance,
        "relation_objective": relation_objective,
        "relation_triplet_cache": relation_provenance,
        "field_b_experiment_registration": field_b_registration_provenance,
        "training_config": training_config,
        "training_config_sha256": training_config_sha256,
        "history": history,
        "final_metrics": final_metrics,
        "final_capability_metrics": final_capability_metrics,
        "source_only_multiteacher": (
            {
                "manifest": {
                    "path": str(source_multiteacher.manifest_source),
                    "sha256": source_multiteacher.manifest_sha256,
                },
                "primitive_descriptor_teacher": {
                    "path": str(source_multiteacher.primitive.source),
                    "sha256": source_multiteacher.primitive.sha256,
                },
                "primitive_state": source_multiteacher.primitive_state_summary,
                "relation_edges": source_multiteacher.relation.num_edges,
                "descriptor_weight": SOURCE_DESCRIPTOR_WEIGHT,
                "relation_weight": SOURCE_RELATION_WEIGHT,
                "query_free": True,
                "source_only": True,
            }
            if source_multiteacher is not None
            else {}
        ),
        "source_only_sam_structure": (
            {
                "manifest": {
                    "path": str(source_sam_structure.manifest_source),
                    "sha256": source_sam_structure.manifest_sha256,
                },
                "relation_cache": {
                    "path": str(
                        source_sam_structure.teacher.relation_cache_source
                    ),
                    "sha256": (
                        source_sam_structure.teacher.relation_cache_sha256
                    ),
                },
                "relation_edges": source_sam_structure.teacher.num_edges,
                "loss_weight": SAM_STRUCTURE_WEIGHT,
                "persistent_semantic_feature": "canonical_radio_only",
                "teacher_payload_saved": False,
                "query_time_source_rgb": False,
                "query_time_target_rgb": False,
            }
            if source_sam_structure is not None
            else {}
        ),
        "source_only_sam_relative_structure": (
            {
                "manifest": {
                    "path": str(active_source_sam_relative_bundle.manifest_source),
                    "sha256": active_source_sam_relative_bundle.manifest_sha256,
                },
                **(
                    {
                        "base_relative_manifest": dict(
                            source_sam_capability_relative.manifest[
                                "base_relative_manifest"
                            ]
                        ),
                        "official_adaptor_checkpoint": dict(
                            source_sam_capability_relative.manifest[
                                "official_adaptor_checkpoint"
                            ]
                        ),
                        "relation_space": "official_dino_sam_geometric_mean_v3",
                    }
                    if source_sam_capability_relative is not None
                    else {
                        "base_structure_manifest": dict(
                            source_sam_relative.manifest[
                                "base_structure_manifest"
                            ]
                        ),
                        "relation_space": "raw_radio_cosine_v2",
                    }
                ),
                "relation_cache": {
                    "path": str(
                        active_source_sam_relative_bundle.teacher.pair_teacher.relation_cache_source
                    ),
                    "sha256": (
                        active_source_sam_relative_bundle.teacher.pair_teacher.relation_cache_sha256
                    ),
                },
                "relation_edges": active_source_sam_relative_bundle.teacher.pair_teacher.num_edges,
                "scale_matched_triplets": active_source_sam_relative_bundle.teacher.num_triplets,
                "loss_weight": SAM_STRUCTURE_WEIGHT,
                "persistent_semantic_feature": "canonical_radio_only",
                "teacher_payload_saved": False,
                "query_time_source_rgb": False,
                "query_time_target_rgb": False,
            }
            if active_source_sam_relative_bundle is not None
            else {}
        ),
        "final_source_multiteacher_metrics": final_source_multiteacher_metrics,
        "control_source_multiteacher_metrics": control_source_multiteacher_metrics,
        "source_gate_decision": source_gate_decision,
        "final_source_sam_structure_metrics": final_source_sam_metrics,
        "control_source_sam_structure_metrics": control_source_sam_metrics,
        "source_sam_structure_gate_decision": source_sam_gate_decision,
        "final_source_sam_relative_pair_metrics": (
            final_source_sam_relative_pair_metrics
        ),
        "control_source_sam_relative_pair_metrics": (
            control_source_sam_relative_pair_metrics
        ),
        "final_source_sam_relative_metrics": final_source_sam_relative_metrics,
        "control_source_sam_relative_metrics": control_source_sam_relative_metrics,
        "source_sam_relative_gate_decision": source_sam_relative_gate_decision,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    if factorized_mode:
        assert factorized_field_signature is not None
        exact_visibility_safe = (
            factorized_reliability_policy
            == FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE
        )
        payload.update(
            {
                "schema_version": (
                    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION
                ),
                "checkpoint_contract": (CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT),
                "factorized_radio_metadata": factorized_radio_checkpoint_metadata(
                    factorized_field_signature
                ),
                "factorized_cache_sha256": mpr_cache_sha256,
                "reliability": torch.empty(field.num_gaussians, 0),
                "factorized_loss_contract": {
                    "name": (
                        FACTORIZED_RADIO_EXACT_MARGINAL_VISIBILITY_SAFE_LOSS_CONTRACT
                        if exact_visibility_safe
                        else FACTORIZED_RADIO_RECONSTRUCTION_LOSS_CONTRACT
                    ),
                    "amplitude_weight": 0.25,
                    "target_reliability_columns_entered_field": False,
                    **(
                        {
                            "reliability_policy": factorized_reliability_policy,
                            "target_reliability_columns_entered_loss": [
                                "directional_resultant",
                                "log_amplitude_std",
                                "observation_evidence",
                                "visibility_purity",
                            ],
                            "visibility_purity_authority": dict(
                                factorized_target.metadata[
                                    "visibility_purity_authority"
                                ]
                            ),
                            "visibility_purity_weighting": (
                                "multiply_legacy_row_weight_by_uniform_half_purity"
                            ),
                        }
                        if exact_visibility_safe
                        else {}
                    ),
                    "official_capability_projection": (
                        "predicted_raw_radio_then_official_adaptor"
                    ),
                    "early_stopping": "disabled_run_all_registered_epochs",
                },
            }
        )
    else:
        payload.update(
            {
                "schema_version": 1,
                "feature_signature": field.signature.to_dict(),
                "reliability": consensus.reliability.half(),
            }
        )
    validate_single_radio_checkpoint_payload(payload)
    write_torch_noclobber(output, payload)
    report = {
        "output": str(output),
        "num_gaussians": field.num_gaussians,
        "valid_gaussians": int(valid_rows.numel()),
        "coefficient_dim": field.decoder.coefficient_dim,
        "local_dim": field.local_codes.shape[1],
        "coarse_dim": field.coarse_dim,
        "basis_fit": basis_fit_report,
        "basis_conditioning": basis_conditioning,
        "initial_field_checkpoint": initial_field_provenance,
        "final_metrics": final_metrics,
        "final_capability_metrics": final_capability_metrics,
        "final_source_multiteacher_metrics": final_source_multiteacher_metrics,
        "control_source_multiteacher_metrics": control_source_multiteacher_metrics,
        "source_gate_decision": source_gate_decision,
        "source_only_multiteacher": payload["source_only_multiteacher"],
        "source_only_sam_structure": payload["source_only_sam_structure"],
        "source_only_sam_relative_structure": payload[
            "source_only_sam_relative_structure"
        ],
        "final_source_sam_structure_metrics": final_source_sam_metrics,
        "control_source_sam_structure_metrics": control_source_sam_metrics,
        "source_sam_structure_gate_decision": source_sam_gate_decision,
        "final_source_sam_relative_pair_metrics": (
            final_source_sam_relative_pair_metrics
        ),
        "control_source_sam_relative_pair_metrics": (
            control_source_sam_relative_pair_metrics
        ),
        "final_source_sam_relative_metrics": final_source_sam_relative_metrics,
        "control_source_sam_relative_metrics": control_source_sam_relative_metrics,
        "source_sam_relative_gate_decision": source_sam_relative_gate_decision,
        "capability_target_mode": payload["capability_target_mode"],
        "capability_target_contract": capability_target_contract,
        "capability_reliability_policy": capability_reliability_policy,
        "factorized_reliability_policy": factorized_reliability_policy,
        "capability_observation_reference": capability_observation_provenance,
        "capability_mpr_targets": capability_target_provenance,
        "relation_objective": relation_objective,
        "relation_triplet_cache": relation_provenance,
        "field_b_experiment_registration": field_b_registration_provenance,
        "mpr_cache_sha256": mpr_cache_sha256,
        "mpr_cache_storage": raw_mpr_storage_provenance,
        "feature_output_bundle_sha256": expected_feature_bundle_sha256,
        "training_config_sha256": training_config_sha256,
        "feature_signature": field.signature.to_dict(),
        "factorized_field_signature": (
            factorized_field_signature.to_dict()
            if factorized_field_signature is not None
            else None
        ),
        "xyz_sha256": _sha256_tensor_rows(torch.as_tensor(cache["xyz"])),
    }
    write_frozen_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument(
        "--expected-mpr-cache-sha256",
        default="",
        help="Caller-trusted SHA-256 for the raw RADIO MPR cache.",
    )
    parser.add_argument(
        "--observation-contract",
        choices=[
            CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
            CANONICAL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
            "compatible-legacy",
            "unchecked",
        ],
        default=CANONICAL_OBSERVATION_CONTRACT_NAME,
        help="Require the shared dataset-independent MPR contract for new fields.",
    )
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument(
        "--expected-radio-checkpoint-sha256",
        default="",
        help="Caller-trusted SHA-256 for the official RADIO checkpoint.",
    )
    parser.add_argument(
        "--expected-feature-output-bundle-sha256",
        default="",
        help="Caller-trusted SHA-256 of the extracted feature output bundle.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--initial-field-checkpoint",
        default="",
        help=(
            "Continue from an exactly geometry/signature-compatible canonical "
            "field. Its architecture and learned state are reused unchanged; "
            "architecture initialization flags are ignored."
        ),
    )
    parser.add_argument(
        "--expected-initial-field-checkpoint-sha256",
        default="",
        help="Caller-trusted SHA-256 for --initial-field-checkpoint.",
    )
    parser.add_argument(
        "--source-only-multiteacher-manifest",
        default="",
        help=(
            "Optional immutable Stage-B manifest adding query-free primitive "
            "descriptor and local semantic-relation teachers."
        ),
    )
    parser.add_argument(
        "--expected-source-only-multiteacher-manifest-sha256",
        default="",
        help="Caller-trusted SHA-256 for the Stage-B manifest.",
    )
    parser.add_argument(
        "--source-only-sam-structure-manifest",
        default="",
        help=(
            "Optional immutable mapping-time official-SAM relation manifest. "
            "It supervises only the existing RADIO direction and is absent "
            "from the query-time field checkpoint."
        ),
    )
    parser.add_argument(
        "--expected-source-only-sam-structure-manifest-sha256",
        default="",
        help="Caller-trusted SHA-256 for the source-only SAM structure manifest.",
    )
    parser.add_argument(
        "--source-only-sam-relative-structure-manifest",
        default="",
        help=(
            "Optional immutable v2 manifest compiling legal source official-SAM "
            "relations into scale-matched no-harm RADIO ordering supervision."
        ),
    )
    parser.add_argument(
        "--expected-source-only-sam-relative-structure-manifest-sha256",
        default="",
        help="Caller-trusted SHA-256 for the v2 source-only SAM manifest.",
    )
    parser.add_argument(
        "--source-only-sam-capability-relative-structure-manifest",
        default="",
        help=(
            "Optional immutable v3 manifest applying the same source-only SAM "
            "triplets after frozen official DINOv3/SAM3 adaptors."
        ),
    )
    parser.add_argument(
        "--expected-source-only-sam-capability-relative-structure-manifest-sha256",
        default="",
        help="Caller-trusted SHA-256 for the v3 capability-relative manifest.",
    )
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coefficient-dim", type=int, default=256)
    parser.add_argument(
        "--local-dim",
        type=int,
        default=0,
        help="Per-Gaussian code dimension; 0 uses coefficient-dim.",
    )
    parser.add_argument("--spatial-coarse-dim", type=int, default=0)
    parser.add_argument("--hash-levels", type=int, default=8)
    parser.add_argument("--hash-features-per-level", type=int, default=2)
    parser.add_argument("--hash-log2-size", type=int, default=15)
    parser.add_argument("--hash-base-resolution", type=int, default=8)
    parser.add_argument("--hash-max-resolution", type=int, default=512)
    parser.add_argument("--hash-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--fusion-reliability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose fixed MPR observation reliability to primitive fusion.",
    )
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument(
        "--fusion-residual-blocks",
        type=int,
        default=0,
        help=(
            "Optional token-wise coefficient residual depth after primitive "
            "local/coarse/reliability fusion; zero preserves schema-v1 behavior."
        ),
    )
    parser.add_argument(
        "--primitive-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional local/coarse/reliability residual fusion; stage-1 main is direct local coefficients.",
    )
    parser.add_argument("--pca-samples", type=int, default=50000)
    parser.add_argument(
        "--basis-fit-device",
        default="",
        help=(
            "Optional device for the source-only PCA fit; empty preserves the "
            "historical CPU path. Use cuda:0 inside the selected "
            "CUDA_VISIBLE_DEVICES scope for faster high-rank fits."
        ),
    )
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--freeze-basis", action="store_true")
    parser.add_argument("--official-capability-loss", action="store_true")
    parser.add_argument(
        "--capability-target-contract",
        choices=[
            CAPABILITY_TARGET_CONTRACT_MATCHED_TOP1,
            CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL,
            CAPABILITY_TARGET_CONTRACT_FIELD_A,
            CAPABILITY_TARGET_CONTRACT_FIELD_C,
        ],
        default=CAPABILITY_TARGET_CONTRACT_MATCHED_TOP1,
        help=(
            "Keep the historical geometry-matched capability target, or opt "
            "into Field-A exact raster-adjoint targets or Field-C exact "
            "marginal-responsibility targets with visibility uncertainty."
        ),
    )
    parser.add_argument(
        "--capability-observation-reference-mpr-cache",
        default="",
        help=(
            "For Field-A only, raw-RADIO raster-adjoint cache that freezes the "
            "exact view/geometry/operator support shared by DINO and SAM."
        ),
    )
    parser.add_argument(
        "--expected-capability-observation-reference-mpr-cache-sha256",
        default="",
        help="Caller-trusted SHA-256 for the Field-A observation reference.",
    )
    parser.add_argument(
        "--dino-mpr-cache",
        default="",
        help=(
            "Optional query-free target built by applying the official DINOv3 "
            "spatial adaptor to each 2-D teacher view before matched MPR."
        ),
    )
    parser.add_argument(
        "--expected-dino-v3-mpr-cache-sha256",
        default="",
        help="Caller-trusted SHA-256 for --dino-mpr-cache.",
    )
    parser.add_argument(
        "--sam3-mpr-cache",
        default="",
        help=(
            "Optional query-free target built by applying the official SAM3 "
            "spatial adaptor to each 2-D teacher view before matched MPR."
        ),
    )
    parser.add_argument(
        "--expected-sam3-mpr-cache-sha256",
        default="",
        help="Caller-trusted SHA-256 for --sam3-mpr-cache.",
    )
    parser.add_argument(
        "--factorized-capability-reference-mpr-cache",
        default="",
        help=(
            "For canonical-factorized-radio-v1 only, the frozen raw "
            "matched-top1 MPR defining DINO/SAM support and operator policy."
        ),
    )
    parser.add_argument(
        "--expected-factorized-capability-reference-mpr-cache-sha256",
        default="",
        help="Caller-trusted SHA-256 for the factorized capability reference.",
    )
    parser.add_argument(
        "--factorized-capability-legacy-receipt-correction",
        default="",
        help=(
            "Immutable authority freezing the raw/DINO/SAM matched-top1 cohort. "
            "Accepts the historical compatibility receipt or the strict formal "
            "feature-bundle cohort schema."
        ),
    )
    parser.add_argument(
        "--expected-factorized-capability-legacy-receipt-correction-sha256",
        default="",
        help="Caller-trusted SHA-256 for the capability cohort authority.",
    )
    parser.add_argument(
        "--factorized-capability-cohort-authority",
        default="",
        help=(
            "Preferred name for a strict formal raw/DINO/SAM cohort authority. "
            "Mutually exclusive with the legacy compatibility option."
        ),
    )
    parser.add_argument(
        "--expected-factorized-capability-cohort-authority-sha256",
        default="",
        help="Caller-trusted SHA-256 for the formal capability cohort authority.",
    )
    parser.add_argument("--mpr-weight", type=float, default=1.0)
    parser.add_argument("--dino-weight", type=float, default=0.20)
    parser.add_argument("--sam3-weight", type=float, default=0.20)
    parser.add_argument(
        "--relation-objective",
        choices=(RELATION_OBJECTIVE_DISABLED, RELATION_OBJECTIVE_FIELD_B),
        default=RELATION_OBJECTIVE_DISABLED,
        help=(
            "Default-off relation supervision. Field-B v1 consumes a "
            "query-independent exact-capability boundary-triplet cache."
        ),
    )
    parser.add_argument(
        "--field-b-experiment-registration",
        default="",
        help="Independent preregistration required by Field-B v1.",
    )
    parser.add_argument(
        "--expected-field-b-experiment-registration-sha256",
        default="",
        help="Caller-trusted SHA-256 for the Field-B preregistration.",
    )
    parser.add_argument(
        "--relation-triplet-cache",
        default="",
        help="Field-B query-independent pair_index/teacher-margin cache.",
    )
    parser.add_argument(
        "--expected-relation-triplet-cache-sha256",
        default="",
        help="Caller-trusted SHA-256 for the Field-B triplet cache.",
    )
    parser.add_argument(
        "--relation-weight",
        type=float,
        default=0.0,
        help=(
            "Default zero. Field-B v1 accepts only the pre-existing global "
            "CanonicalFieldLossConfig default of 0.05."
        ),
    )
    parser.add_argument("--coefficient-weight", type=float, default=1e-5)
    parser.add_argument("--basis-orthogonality-weight", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="Minimum optimization epochs before the raw-MPR early-stop rule applies.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--target-cosine", type=float, default=0.985)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.fusion_residual_blocks < 0:
        parser.error("--fusion-residual-blocks cannot be negative")
    if args.fusion_residual_blocks and not args.primitive_fusion:
        parser.error("--fusion-residual-blocks requires --primitive-fusion")
    if args.min_epochs <= 0 or args.min_epochs > args.epochs:
        parser.error("--min-epochs must lie in [1, --epochs]")
    source_multiteacher_pair = (
        args.source_only_multiteacher_manifest,
        args.expected_source_only_multiteacher_manifest_sha256,
    )
    if any(source_multiteacher_pair) and not all(source_multiteacher_pair):
        parser.error(
            "Stage-B source-only multiteacher manifest path and SHA-256 "
            "are both required"
        )
    if any(source_multiteacher_pair) and args.observation_contract != (
        CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME
    ):
        parser.error(
            "Stage-B source-only multiteacher distillation requires "
            "--observation-contract canonical-factorized-radio-v1"
        )
    source_sam_structure_pair = (
        args.source_only_sam_structure_manifest,
        args.expected_source_only_sam_structure_manifest_sha256,
    )
    if any(source_sam_structure_pair) and not all(source_sam_structure_pair):
        parser.error(
            "source-only SAM structure manifest path and SHA-256 are both required"
        )
    if any(source_sam_structure_pair) and args.observation_contract in {
        "compatible-legacy",
        "unchecked",
    }:
        parser.error(
            "source-only SAM structure supervision requires a strict canonical "
            "observation contract"
        )
    source_sam_relative_pair = (
        args.source_only_sam_relative_structure_manifest,
        args.expected_source_only_sam_relative_structure_manifest_sha256,
    )
    if any(source_sam_relative_pair) and not all(source_sam_relative_pair):
        parser.error(
            "source-only SAM relative manifest path and SHA-256 are both required"
        )
    if any(source_sam_structure_pair) and any(source_sam_relative_pair):
        parser.error(
            "v1 absolute and v2 relative source-only SAM objectives are mutually exclusive"
        )
    if any(source_sam_relative_pair) and args.observation_contract in {
        "compatible-legacy",
        "unchecked",
    }:
        parser.error(
            "source-only SAM relative supervision requires a strict canonical "
            "observation contract"
        )
    source_sam_capability_relative_pair = (
        args.source_only_sam_capability_relative_structure_manifest,
        args.expected_source_only_sam_capability_relative_structure_manifest_sha256,
    )
    if any(source_sam_capability_relative_pair) and not all(
        source_sam_capability_relative_pair
    ):
        parser.error(
            "source-only SAM capability-relative manifest path and SHA-256 are "
            "both required"
        )
    if any(source_sam_capability_relative_pair) and (
        any(source_sam_structure_pair) or any(source_sam_relative_pair)
    ):
        parser.error(
            "v3 capability-relative and earlier source-only SAM objectives are "
            "mutually exclusive"
        )
    if any(source_sam_capability_relative_pair) and args.observation_contract in {
        "compatible-legacy",
        "unchecked",
    }:
        parser.error(
            "source-only SAM capability-relative supervision requires a strict "
            "canonical observation contract"
        )
    if any(source_sam_capability_relative_pair) and not args.official_capability_loss:
        parser.error(
            "source-only SAM capability-relative supervision requires "
            "--official-capability-loss"
        )
    if args.capability_target_contract != CAPABILITY_TARGET_CONTRACT_FIELD_A and (
        args.capability_observation_reference_mpr_cache
        or args.expected_capability_observation_reference_mpr_cache_sha256
    ):
        parser.error(
            "Observation-reference arguments require "
            "--capability-target-contract field_a_exact_adjoint"
        )
    formal_cohort_pair = (
        args.factorized_capability_cohort_authority,
        args.expected_factorized_capability_cohort_authority_sha256,
    )
    if any(formal_cohort_pair) and not all(formal_cohort_pair):
        parser.error(
            "formal factorized capability cohort path and SHA-256 are both required"
        )
    legacy_cohort_pair = (
        args.factorized_capability_legacy_receipt_correction,
        args.expected_factorized_capability_legacy_receipt_correction_sha256,
    )
    if any(formal_cohort_pair) and any(legacy_cohort_pair):
        parser.error(
            "formal and legacy factorized capability cohort authorities are "
            "mutually exclusive"
        )
    if all(formal_cohort_pair):
        args.factorized_capability_legacy_receipt_correction = formal_cohort_pair[0]
        args.expected_factorized_capability_legacy_receipt_correction_sha256 = (
            formal_cohort_pair[1]
        )
    factorized_capability_arguments = (
        args.factorized_capability_reference_mpr_cache,
        args.expected_factorized_capability_reference_mpr_cache_sha256,
        args.factorized_capability_legacy_receipt_correction,
        args.expected_factorized_capability_legacy_receipt_correction_sha256,
    )
    if args.observation_contract != CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME and any(
        factorized_capability_arguments
    ):
        parser.error(
            "factorized capability authority arguments require "
            "--observation-contract canonical-factorized-radio-v1"
        )
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
