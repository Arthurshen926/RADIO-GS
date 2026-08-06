#!/usr/bin/env python3
"""Derive a disposable text-space cache from the canonical field and v3 readout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2,
    load_factorized_field_support,
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_contract import (
    SurfaceRegionContractV2,
    SurfaceRegionContractV3,
)
from radio_gs.interfaces.surface_region_full_scalar_contract import (
    FullScalarRegionSummary,
    SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256,
    SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
    aggregate_surface_region_full_scalars,
    apply_full_scalar_normalization,
    build_full_scalar_support_routing,
    validate_full_scalar_normalization_authority,
)
from radio_gs.interfaces.surface_region_full_scalar_residual_checkpoint import (
    load_surface_region_full_scalar_residual_checkpoint,
)
from radio_gs.interfaces.surface_region_full_scalar_training_certificate import (
    load_training_certificate,
)
from radio_gs.interfaces.surface_region_selection import (
    RegionSelection,
    as_region_selection,
    surface_region_contract_from_metadata,
)
from radio_gs.interfaces.surface_region_query_router import (
    SurfaceRegionQueryRouterV1,
)
from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryReadoutV3,
    SURFACE_REGION_V3_GATED_RAW_PRIOR,
    SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION,
    SurfaceRegionSummaryResidualCodebookV1,
    surface_region_state_dict_sha256,
    surface_region_effective_reliability_v3,
    surface_region_geometry_v2,
    surface_region_geometry_v3,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.models.surface_region_dual_descriptor import (
    SurfaceRegionAcceptedV2FullScalarResidualV1,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.build_canonical_primitive_semantic_cache import (
    canonical_reconstruction_confidence,
)
from radio_gs.scripts.build_primitive_text_score_cache import (
    apply_completion_evidence,
)
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_surface_region_summary_readout_v2,
    load_torch_mapping,
    load_torch_payload,
    sha256_file,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1 = "canonical-v1"

_FULL_SCALAR_OVERLAY_ARGUMENTS = (
    "accepted_v2_full_scalar_state",
    "accepted_v2_full_scalar_state_sha256",
    "full_scalar_normalization_authority",
    "full_scalar_normalization_authority_sha256",
    "full_scalar_residual_checkpoint",
    "full_scalar_residual_checkpoint_sha256",
    "full_scalar_training_certificate",
    "full_scalar_training_certificate_sha256",
)


@dataclass(frozen=True)
class AcceptedV2FullScalarOverlayResult:
    """One post-e0 overlay result with auditable, row-aligned routing."""

    semantic_descriptor: torch.Tensor
    overlap_candidate_mask: torch.Tensor
    base_only_fallback_mask: torch.Tensor
    source_ood_fallback_mask: torch.Tensor
    effective_update_mask: torch.Tensor


@dataclass(frozen=True)
class AcceptedV2FullScalarRuntimeCarrier:
    """Prevalidated CPU carrier that avoids revalidating all state per batch."""

    scalar_source: torch.Tensor
    reliability_source: torch.Tensor
    global_to_compact: torch.Tensor
    overlap_mask: torch.Tensor
    base_only_mask: torch.Tensor
    exact_only_mask: torch.Tensor
    neither_mask: torch.Tensor


def _build_full_scalar_runtime_carrier(
    exact_state: object,
    routing: object,
) -> AcceptedV2FullScalarRuntimeCarrier:
    """Materialize immutable gather tables after the strict state validation."""

    valid = torch.as_tensor(exact_state.valid).bool().cpu()
    global_rows = torch.as_tensor(exact_state.global_rows).long().cpu()
    scalar_source = exact_state.scalar_encoding_input().float().cpu().contiguous()
    reliability = (
        exact_state.legacy_geometric_reliability().float().cpu().contiguous()
    )
    if (
        global_rows.shape != (int(valid.sum()),)
        or scalar_source.shape != (global_rows.numel(), 6)
        or reliability.shape != (global_rows.numel(),)
        or not bool(torch.isfinite(scalar_source).all())
        or not bool(torch.isfinite(reliability).all())
        or bool((reliability < 0).any())
    ):
        raise ValueError("prevalidated full-scalar runtime state differs")
    global_to_compact = torch.full((valid.numel(),), -1, dtype=torch.long)
    global_to_compact[global_rows] = torch.arange(global_rows.numel())
    return AcceptedV2FullScalarRuntimeCarrier(
        scalar_source=scalar_source,
        reliability_source=reliability,
        global_to_compact=global_to_compact,
        overlap_mask=torch.as_tensor(routing.overlap_mask).bool().cpu(),
        base_only_mask=torch.as_tensor(routing.base_only_fallback_mask).bool().cpu(),
        exact_only_mask=torch.as_tensor(routing.exact_only_abstain_mask).bool().cpu(),
        neither_mask=torch.as_tensor(routing.neither_abstain_mask).bool().cpu(),
    )


def _aggregate_prevalidated_full_scalars(
    carrier: AcceptedV2FullScalarRuntimeCarrier,
    region_global_rows: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor,
) -> FullScalarRegionSummary:
    """Execute the contract-identical 6D→18D aggregation without revalidation."""

    rows = torch.as_tensor(region_global_rows).long().cpu()
    mask = torch.as_tensor(token_mask).bool().cpu()
    anchor = torch.as_tensor(anchor_index).long().cpu()
    if (
        rows.ndim != 2
        or mask.shape != rows.shape
        or anchor.shape != (rows.shape[0],)
        or bool((mask.sum(dim=1) <= 0).any())
        or bool((anchor < 0).any())
        or bool((anchor >= rows.shape[1]).any())
        or not bool(mask[torch.arange(rows.shape[0]), anchor].all())
    ):
        raise ValueError("prevalidated full-scalar region layout differs")
    active_rows = rows[mask]
    domain_size = carrier.global_to_compact.numel()
    if active_rows.numel() and (
        int(active_rows.min()) < 0 or int(active_rows.max()) >= domain_size
    ):
        raise ValueError("full-scalar region global row is outside state geometry")
    sorted_rows = rows.masked_fill(~mask, domain_size).sort(dim=1).values
    duplicate_active = (sorted_rows[:, 1:] == sorted_rows[:, :-1]) & (
        sorted_rows[:, 1:] < domain_size
    )
    if bool(duplicate_active.any()):
        raise ValueError("active full-scalar region rows must be unique")
    safe_rows = torch.where(mask, rows, torch.zeros_like(rows))
    token_overlap = mask & carrier.overlap_mask[safe_rows]
    token_base_only = mask & carrier.base_only_mask[safe_rows]
    token_exact_only = mask & carrier.exact_only_mask[safe_rows]
    anchor_rows = safe_rows[torch.arange(rows.shape[0]), anchor]
    use_full = carrier.overlap_mask[anchor_rows]
    base_fallback = carrier.base_only_mask[anchor_rows]
    abstain = carrier.exact_only_mask[anchor_rows] | carrier.neither_mask[
        anchor_rows
    ]

    compact_rows = carrier.global_to_compact[safe_rows]
    safe_compact = compact_rows.clamp_min(0)
    token_scalars = carrier.scalar_source[safe_compact].masked_fill(
        ~token_overlap[..., None], 0.0
    )
    token_weights = carrier.reliability_source[safe_compact].masked_fill(
        ~token_overlap, 0.0
    )
    # Preserve the contract's operation order exactly.  Besides being faster
    # than a Python per-region loop, this avoids making an OOD decision from a
    # numerically different reduction at the source-envelope boundary.
    total = token_weights.sum(dim=1, keepdim=True)
    if bool((use_full & (total[:, 0] <= 0)).any()):
        raise ValueError("overlap route lacks positive legacy reliability")
    safe_total = total.clamp_min(torch.finfo(torch.float32).tiny)
    mean = (token_scalars * token_weights[..., None]).sum(dim=1) / safe_total
    variance = (
        (token_scalars - mean[:, None]).square()
        * token_weights[..., None]
    ).sum(dim=1) / safe_total
    anchor_values = token_scalars[torch.arange(rows.shape[0]), anchor]
    summary = torch.cat(
        (anchor_values, mean, variance.clamp_min(0).sqrt()), dim=1
    ).masked_fill(~use_full[:, None], 0.0)
    if bool(summary[~use_full].ne(0).any()) or bool(
        token_scalars[~mask].ne(0).any()
    ):
        raise RuntimeError("prevalidated full-scalar aggregation leaked padding/state")
    return FullScalarRegionSummary(
        summary=summary,
        token_scalars=token_scalars,
        token_overlap_mask=token_overlap,
        token_base_only_mask=token_base_only,
        token_exact_only_mask=token_exact_only,
        use_full_scalar_mask=use_full,
        base_fallback_mask=base_fallback,
        abstain_mask=abstain,
    )


def _full_scalar_overlay_arguments(
    args: argparse.Namespace,
) -> dict[str, str] | None:
    """Return the all-or-none full-scalar CLI bundle.

    Keeping this separate from the legacy factorized-field switches is
    intentional: the exact state is an overlay sidecar and can never become
    the authority for the accepted base graph, field, or validity mask.
    """

    values = {
        name: str(getattr(args, name, "")).strip()
        for name in _FULL_SCALAR_OVERLAY_ARGUMENTS
    }
    if not any(values.values()):
        return None
    missing = sorted(name for name, value in values.items() if not value)
    if missing:
        raise ValueError(
            "accepted-V2 full-scalar overlay requires every state/normalization/"
            f"certificate/residual path and SHA-256 argument; missing {missing}"
        )
    return values


def _validate_full_scalar_overlay_mode(
    *,
    contract: object,
    field_schema: str,
    canonical_radio_source: str,
    context_field: torch.nn.Module | None,
    query_router_mode: bool,
) -> None:
    """Fail closed unless the overlay is downstream of the accepted V2 base."""

    if type(contract) is not SurfaceRegionContractV2:
        raise ValueError("full-scalar overlay requires the exact accepted V2 contract")
    if field_schema != CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1:
        raise ValueError(
            "full-scalar overlay cannot replace the accepted V2 field with a "
            "factorized field"
        )
    if canonical_radio_source != "field_decode":
        raise ValueError("full-scalar overlay requires accepted V2 field decoding")
    if context_field is not None:
        raise ValueError("full-scalar overlay forbids a changed directional context field")
    if query_router_mode:
        raise ValueError("full-scalar overlay and query-router modes are mutually exclusive")


def _validate_full_scalar_runtime_base_authority(
    checkpoint_payload: Mapping[str, Any],
    *,
    readout_payload: Mapping[str, Any],
    readout_checkpoint_sha256: str,
    contract_sha256: str,
) -> None:
    """Bind the residual's five-part accepted-V2 authority to this runtime."""

    accepted = checkpoint_payload.get("accepted_v2_authority")
    architecture = readout_payload.get("architecture")
    state = readout_payload.get("state_dict")
    provenance = readout_payload.get("provenance")
    if (
        not isinstance(accepted, Mapping)
        or not isinstance(architecture, Mapping)
        or not isinstance(state, Mapping)
        or not isinstance(provenance, Mapping)
    ):
        raise ValueError("full-scalar runtime lacks accepted V2 authority")
    observed = {
        "checkpoint_sha256": str(readout_checkpoint_sha256),
        "architecture_sha256": str(architecture.get("digest", "")),
        "state_dict_sha256": surface_region_state_dict_sha256(state),
        "provenance_sha256": canonical_json_sha256(dict(provenance)),
        "contract_sha256": str(contract_sha256),
    }
    if dict(accepted) != observed:
        raise ValueError("full-scalar residual/runtime accepted V2 authority differs")


def apply_accepted_v2_full_scalar_overlay(
    base_descriptor: torch.Tensor,
    *,
    residual: SurfaceRegionAcceptedV2FullScalarResidualV1,
    exact_state: object,
    runtime_carrier: AcceptedV2FullScalarRuntimeCarrier | None = None,
    normalization_authority: Mapping[str, Any],
    accepted_base_valid: torch.Tensor,
    accepted_global_rows: torch.Tensor,
    local_region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor,
) -> AcceptedV2FullScalarOverlayResult:
    """Apply the scalar branch strictly after an immutable accepted-V2 ``e0``.

    ``local_region_rows`` are indices into the accepted support graph.  The
    explicit gather through ``accepted_global_rows`` is the only permitted
    bridge to the exact-state global geometry domain.
    """

    base = torch.as_tensor(base_descriptor)
    graph_rows = torch.as_tensor(local_region_rows).detach().long().cpu()
    graph_to_global = torch.as_tensor(accepted_global_rows).detach().long().cpu()
    if graph_rows.ndim != 2 or graph_to_global.ndim != 1:
        raise ValueError("accepted region/global row tensors have invalid ranks")
    active = torch.as_tensor(token_mask).detach().bool().cpu()
    if active.shape != graph_rows.shape:
        raise ValueError("accepted region rows and token mask differ")
    active_rows = graph_rows[active]
    if active_rows.numel() and (
        int(active_rows.min()) < 0 or int(active_rows.max()) >= graph_to_global.numel()
    ):
        raise ValueError("accepted local region row is outside the frozen graph")
    # Padding is never semantically gathered.  Zero is only a safe placeholder
    # that the aggregation contract subsequently masks to exact zero.
    safe_local = torch.where(active, graph_rows, torch.zeros_like(graph_rows))
    region_global_rows = graph_to_global[safe_local]
    if runtime_carrier is None:
        aggregate = aggregate_surface_region_full_scalars(
            exact_state,
            torch.as_tensor(accepted_base_valid).detach().bool().cpu(),
            region_global_rows,
            active,
            torch.as_tensor(anchor_index).detach().long().cpu(),
        )
    else:
        aggregate = _aggregate_prevalidated_full_scalars(
            runtime_carrier,
            region_global_rows,
            active,
            torch.as_tensor(anchor_index).detach().long().cpu(),
        )
    if bool(torch.as_tensor(aggregate.abstain_mask).any()):
        raise RuntimeError(
            "accepted V2 descriptor batch unexpectedly contains an exact-only/"
            "neither anchor"
        )
    normalized = apply_full_scalar_normalization(
        aggregate.summary,
        aggregate.use_full_scalar_mask,
        normalization_authority,
    )
    overlap = torch.as_tensor(aggregate.use_full_scalar_mask).bool().cpu()
    base_only = torch.as_tensor(aggregate.base_fallback_mask).bool().cpu()
    source_ood = torch.as_tensor(normalized.ood_mask).bool().cpu()
    if not torch.equal(overlap | base_only, torch.ones_like(overlap)) or bool(
        (overlap & base_only).any()
    ):
        raise RuntimeError("accepted V2 overlay anchors do not form overlap/base-only")
    if bool((source_ood & ~overlap).any()):
        raise RuntimeError("source-envelope OOD escaped the overlap support")
    if not torch.equal(
        torch.as_tensor(normalized.use_full_scalar_mask).bool().cpu(),
        overlap & ~source_ood,
    ) or not torch.equal(
        torch.as_tensor(normalized.base_fallback_mask).bool().cpu(),
        source_ood,
    ):
        raise RuntimeError("full-scalar normalization routing differs")
    fallback = base_only | source_ood

    # The model owns normalization from raw 18-D values.  The authority's
    # normalized tensor is deliberately ignored here; using it would apply the
    # frozen median/robust scale twice.
    diagnostics = residual.forward_with_diagnostics(
        base,
        aggregate.summary.to(device=base.device, dtype=base.dtype),
        ood_mask=fallback.to(base.device),
    )
    if not torch.equal(diagnostics.base_descriptor, base):
        raise RuntimeError("full-scalar residual changed its accepted V2 base input")
    if not torch.equal(diagnostics.ood_fallback.cpu(), fallback):
        raise RuntimeError("full-scalar residual fallback routing differs")
    semantic = diagnostics.semantic_descriptor
    if semantic.shape != base.shape or not bool(torch.isfinite(semantic).all()):
        raise RuntimeError("full-scalar semantic descriptor is malformed")
    if not torch.equal(semantic[fallback.to(base.device)], base[fallback.to(base.device)]):
        raise RuntimeError("full-scalar fallback is not bitwise accepted V2")
    norms = torch.linalg.vector_norm(semantic, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise RuntimeError("full-scalar semantic descriptor left the unit gauge")
    update = diagnostics.tangent_update.detach().cpu()
    effective = overlap & ~source_ood & update.ne(0).any(dim=-1)
    return AcceptedV2FullScalarOverlayResult(
        semantic_descriptor=semantic,
        overlap_candidate_mask=overlap,
        base_only_fallback_mask=base_only,
        source_ood_fallback_mask=source_ood,
        effective_update_mask=effective,
    )


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _atomic_torch_save(payload: object, output: Path) -> None:
    """Durably publish a first-writer-wins cache without replacement."""

    write_torch_noclobber(output, payload)


def load_surface_factorized_state_bundle(
    field_checkpoint: str | Path,
    *,
    expected_field_checkpoint_sha256: str,
    expected_factorized_radio_cache_sha256: str,
    state_path: str | Path,
    expected_state_sha256: str,
):
    """Load the shared schema-v2 support and its exact state sidecar."""

    support = load_factorized_field_support(
        field_checkpoint,
        expected_field_checkpoint_sha256=expected_field_checkpoint_sha256,
        expected_mpr_cache_sha256=expected_factorized_radio_cache_sha256,
    )
    state = load_factorized_primitive_state(
        state_path,
        expected_sha256=expected_state_sha256,
        expected_field_checkpoint_sha256=expected_field_checkpoint_sha256,
        expected_factorized_radio_cache_sha256=support.cache.sha256,
        expected_xyz=support.cache.xyz,
        expected_valid=support.cache.valid,
    )
    return support, state


def directional_anchor_tokens(
    context_tokens: torch.Tensor,
    anchor_values: torch.Tensor,
    anchor_index: torch.Tensor,
) -> torch.Tensor:
    """Replace only the anchor feature in each canonical context region."""

    tokens = torch.as_tensor(context_tokens)
    values = torch.as_tensor(
        anchor_values, device=tokens.device, dtype=tokens.dtype
    )
    anchors = torch.as_tensor(
        anchor_index, device=tokens.device, dtype=torch.long
    )
    if tokens.ndim != 3:
        raise ValueError("directional context tokens must be [batch,width,channels]")
    if values.shape != (tokens.shape[0], tokens.shape[2]):
        raise ValueError("directional anchor values must be [batch,channels]")
    if anchors.shape != (tokens.shape[0],):
        raise ValueError("directional anchor indices must be [batch]")
    if anchors.numel() and (
        int(anchors.min()) < 0 or int(anchors.max()) >= tokens.shape[1]
    ):
        raise IndexError("directional anchor index is outside its region")
    result = tokens.clone()
    result[torch.arange(tokens.shape[0], device=tokens.device), anchors] = values
    return result


def surface_region_radio_tokens(
    features: torch.Tensor,
    *,
    normalization: str,
) -> torch.Tensor:
    """Apply the frozen SurfaceRegion feature-gauge contract.

    SurfaceRegion V2 training caches contain L2-normalized RADIO directions,
    but the accepted legacy checkpoint was empirically calibrated against the
    raw compact-field gauge at inference.  Keep both modes explicit and bind
    the selected mode into cache authority.  ``l2_direction`` is valid only
    with a readout trained and validated for that gauge.
    """

    values = torch.as_tensor(features)
    mode = str(normalization)
    if mode == "legacy_raw":
        return values
    if mode not in {
        "l2_direction",
        "l2_direction_plus_log_raw_norm_v1",
    }:
        raise ValueError(
            "surface-region RADIO normalization must be l2_direction, "
            "l2_direction_plus_log_raw_norm_v1, or legacy_raw"
        )
    normalized = F.normalize(values.float(), dim=-1, eps=1e-8)
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("surface-region RADIO direction is non-finite")
    return normalized.to(dtype=values.dtype)


def validate_query_router_deployment_gauge(
    *,
    normalization: str,
    experiment_registration: Mapping[str, Any] | None,
    gauge_authority: Mapping[str, Any] | None,
) -> str:
    """Authorize the trained L2 route or the narrow accepted-V2 raw sentinel."""

    mode = str(normalization)
    if mode == "l2_direction":
        if gauge_authority is not None:
            raise ValueError("L2 query-router deployment cannot claim raw-gauge authority")
        return "trained_l2_direction"
    if mode != "legacy_raw":
        raise ValueError("query-router deployment gauge is unsupported")
    if not isinstance(experiment_registration, Mapping) or (
        experiment_registration.get("registration")
        != "surface_region_accepted_physical_v2_residual_router_v1"
    ):
        raise ValueError("raw-gauge query router requires the exact preregistration")
    if not isinstance(gauge_authority, Mapping) or (
        gauge_authority.get("registration")
        != "surface_region_accepted_physical_v2_deployment_gauge_parity_addendum_v1"
    ):
        raise ValueError("raw-gauge query router requires the exact gauge authority")
    evidence = gauge_authority.get("evidence_without_benchmark_labels")
    rule = gauge_authority.get("gauge_resolution_rule")
    attribution = gauge_authority.get("attribution_constraint")
    if (
        not isinstance(evidence, Mapping)
        or str(evidence.get("accepted_positive_cache_sha256", ""))
        != "3366c96839fe392d6cc1f6d55939691787e095491581c63006f74e44862d4cac"
        or str(evidence.get("accepted_negative_cache_sha256", ""))
        != "d778361b2ea860cfb07f586ace82cb1558df5129a55ad462b097e4e6c366ba90"
        or evidence.get("benchmark_masks_opened") is not False
        or evidence.get("benchmark_metrics_opened") is not False
        or not isinstance(rule, Mapping)
        or "legacy_raw" not in list(rule.get("candidates", []))
        or not isinstance(attribution, Mapping)
        or attribution.get("generic_training_gauge") != "l2_direction"
    ):
        raise ValueError("raw-gauge query-router authority evidence differs")
    return "accepted_v2_legacy_raw_mixed_gauge_sentinel"


def project_surface_codebook_slots(
    head: SigLIP2SummaryHead,
    slot_tokens: torch.Tensor,
) -> torch.Tensor:
    """Project every hypothesis through the training-identical head shape.

    The official head is deliberately invoked once per slot with an exact
    ``[B,1,1280]`` input.  A single ``[B,4,1280]`` GEMM is mathematically
    equivalent but not bitwise equivalent, and the query router was trained
    under the former numerical contract.
    """

    tokens = torch.as_tensor(slot_tokens)
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (4, 1280):
        raise ValueError("surface codebook slots must be [B,4,1280]")
    projected = [
        head(tokens[:, slot].contiguous()[:, None])[:, 0].float()
        for slot in range(4)
    ]
    return F.normalize(torch.stack(projected, dim=1), dim=-1, eps=1e-8)


def _state_dicts_bitwise_equal(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key].detach().cpu(), right[key].detach().cpu())
        for key in left
    )


def _load_query_router_negative_text(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    payload, digest, source = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="query-router generic-negative text bank",
    )
    queries = ["object", "things", "stuff", "texture"]
    embeddings = payload.get("embeddings")
    if (
        payload.get("text_encoder") != "siglip2"
        or payload.get("queries") != queries
        or payload.get("prompt_templates") != ["{query}"]
        or not isinstance(embeddings, torch.Tensor)
        or tuple(embeddings.shape) != (4, 1536)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("query-router generic-negative text contract differs")
    return F.normalize(embeddings.float(), dim=-1, eps=1e-8), {
        "path": str(source),
        "sha256": digest,
        "queries": queries,
    }


class _ResumeStateError(ValueError):
    pass


def _resume_failure(resume_dir: Path, reason: str) -> _ResumeStateError:
    quarantine = resume_dir.with_name(f"{resume_dir.name}.quarantine-required")
    return _ResumeStateError(
        f"stale/corrupt semantic resume state: {reason}; "
        f"quarantine path: {quarantine} (not deleted automatically)"
    )


def _load_or_create_resume_contract(
    resume_dir: Path,
    payload: Mapping[str, Any],
) -> str:
    resume_dir.mkdir(parents=True, exist_ok=True)
    contract_path = resume_dir / "contract.json"
    expected = dict(payload)
    expected_digest = canonical_json_sha256(expected)
    if contract_path.exists() or contract_path.is_symlink():
        try:
            observed, _, _ = load_json_object(
                contract_path,
                label="semantic resume contract",
            )
        except Exception as exc:
            raise _resume_failure(resume_dir, "contract cannot be reopened") from exc
        if observed != expected:
            raise _resume_failure(resume_dir, "contract differs from this stage")
    else:
        write_frozen_json(contract_path, expected)
    return expected_digest


def _resume_paths(
    resume_dir: Path,
    *,
    phase: str,
    start: int,
    stop: int,
) -> tuple[Path, Path]:
    stem = f"{phase}_{int(start):09d}_{int(stop):09d}"
    return resume_dir / f"{stem}.pt", resume_dir / f"{stem}.complete.json"


def _load_resume_tensor(
    resume_dir: Path,
    *,
    phase: str,
    start: int,
    stop: int,
    contract_sha256: str,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
) -> torch.Tensor | None:
    shard, terminal = _resume_paths(
        resume_dir,
        phase=phase,
        start=start,
        stop=stop,
    )
    shard_present = shard.exists() or shard.is_symlink()
    terminal_present = terminal.exists() or terminal.is_symlink()
    if not shard_present and not terminal_present:
        return None
    if shard_present != terminal_present:
        raise _resume_failure(resume_dir, f"partial {phase} batch {start}:{stop}")
    try:
        marker, _, _ = load_json_object(terminal, label=f"{phase} resume terminal")
        if (
            set(marker)
            != {
                "schema_version",
                "artifact_type",
                "phase",
                "start",
                "stop",
                "contract_sha256",
                "tensor",
                "shape",
                "dtype",
            }
            or marker.get("schema_version") != 1
            or marker.get("artifact_type") != "surface_semantic_resume_batch"
            or marker.get("phase") != phase
            or marker.get("start") != start
            or marker.get("stop") != stop
            or marker.get("contract_sha256") != contract_sha256
            or marker.get("shape") != list(expected_shape)
            or marker.get("dtype") != str(expected_dtype)
        ):
            raise _resume_failure(resume_dir, f"{phase} batch marker differs")
        tensor_path = validate_file_record(
            marker["tensor"],
            label=f"{phase} resume tensor",
        )
        if tensor_path != shard.resolve():
            raise _resume_failure(resume_dir, f"{phase} batch path differs")
        value, _, _ = load_torch_payload(
            shard,
            expected_sha256=marker["tensor"]["sha256"],
            map_location="cpu",
            label=f"{phase} resume tensor",
        )
    except _ResumeStateError:
        raise
    except Exception as exc:
        raise _resume_failure(resume_dir, f"{phase} batch cannot be reopened") from exc
    if (
        not torch.is_tensor(value)
        or tuple(value.shape) != expected_shape
        or value.dtype != expected_dtype
        or not bool(torch.isfinite(value).all())
    ):
        raise _resume_failure(resume_dir, f"{phase} batch tensor differs")
    return value


def _commit_resume_tensor(
    resume_dir: Path,
    *,
    phase: str,
    start: int,
    stop: int,
    contract_sha256: str,
    value: torch.Tensor,
) -> None:
    shard, terminal = _resume_paths(
        resume_dir,
        phase=phase,
        start=start,
        stop=stop,
    )
    tensor = value.detach().cpu().contiguous()
    write_torch_noclobber(shard, tensor)
    marker = {
        "schema_version": 1,
        "artifact_type": "surface_semantic_resume_batch",
        "phase": phase,
        "start": int(start),
        "stop": int(stop),
        "contract_sha256": contract_sha256,
        "tensor": file_record(shard),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    write_frozen_json(terminal, marker)


def _pace_after_commit(device: torch.device, seconds: float) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    time.sleep(float(seconds))


def _validate_resume_inventory(
    resume_dir: Path,
    *,
    row_count: int,
    radio_batch_size: int,
    semantic_batch_size: int,
    semantic_phase: str,
) -> None:
    allowed = {"contract.json"}
    for phase, batch_size in (
        ("radio", int(radio_batch_size)),
        (str(semantic_phase), int(semantic_batch_size)),
    ):
        for start in range(0, int(row_count), batch_size):
            stop = min(int(row_count), start + batch_size)
            shard, terminal = _resume_paths(
                resume_dir,
                phase=phase,
                start=start,
                stop=stop,
            )
            allowed.update((shard.name, terminal.name))
    unexpected = sorted(path.name for path in resume_dir.iterdir() if path.name not in allowed)
    if unexpected:
        raise _resume_failure(
            resume_dir,
            f"unexpected files {unexpected[:5]}",
        )


def _adjacency(graph: dict, neighbors: int) -> torch.Tensor:
    """Keep strongest surface-conditioned outgoing edges plus a self slot."""
    count = int(graph["xyz"].shape[0]); k = int(neighbors)
    edge = torch.as_tensor(graph["edge_index"]).long()
    affinity = torch.as_tensor(graph["raw_affinity"]).float()
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(count)]
    for src, dst, weight in zip(edge[0].tolist(), edge[1].tolist(), affinity.tolist()):
        buckets[src].append((float(weight), int(dst)))
    result = torch.arange(count)[:, None].expand(-1, k).clone()
    for row, entries in enumerate(buckets):
        selected = [dst for _weight, dst in sorted(entries, reverse=True)[:k]]
        if selected:
            result[row, :len(selected)] = torch.tensor(selected)
    return result


def two_hop_physical_regions(
    centers: torch.Tensor, adjacency: torch.Tensor, xyz: torch.Tensor, radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique two-hop surface candidates clipped by physical radius."""
    center = torch.as_tensor(centers, device=adjacency.device).long()
    first = adjacency[center]
    second = adjacency[first].flatten(1)
    rows = torch.cat([center[:, None], first, second], dim=1)
    rows, _ = rows.sort(dim=1)
    unique = torch.ones_like(rows, dtype=torch.bool)
    unique[:, 1:] = rows[:, 1:] != rows[:, :-1]
    distance = torch.linalg.vector_norm(xyz[rows] - xyz[center, None], dim=-1)
    mask = unique & (distance <= float(radius))
    # The center is guaranteed to survive sorting and the radius test.
    if not bool(mask.any(dim=1).all()):
        raise RuntimeError("physical surface region lost its center")
    return rows, mask


def completion_primary_valid(
    mpr: dict,
    output_valid: torch.Tensor,
) -> torch.Tensor | None:
    """Recover the explicit primary/fallback partition from a fused MPR."""

    metadata = dict(mpr.get("metadata", {}))
    if (
        metadata.get("construction")
        != "dominant_primary_with_query_free_support_completion"
    ):
        return None
    observed = torch.as_tensor(output_valid).bool().reshape(-1)
    reliability = torch.as_tensor(mpr.get("reliability")).float()
    if reliability.ndim != 2 or reliability.shape[0] != observed.numel():
        raise ValueError("completed MPR reliability does not align with rows")
    if reliability.shape[1] < 3:
        raise ValueError("completed MPR lacks its primary indicator channel")
    primary = observed & (reliability[:, 2] > 0.5)
    expected = metadata.get("primary_valid_count")
    if expected is None or int(primary.sum()) != int(expected):
        raise ValueError("completed MPR primary partition count mismatch")
    if not bool(primary.any()) or torch.equal(primary, observed):
        raise ValueError("completed MPR does not contain a distinct fallback partition")
    return primary


def preserve_primary_region_tokens(
    rows: torch.Tensor,
    mask: torch.Tensor,
    centers: torch.Tensor,
    primary_valid: torch.Tensor | None,
) -> torch.Tensor:
    """Prevent fallback rows from changing an existing primary descriptor."""

    active = torch.as_tensor(mask).bool()
    if primary_valid is None:
        return active
    neighbors = torch.as_tensor(rows).long()
    center_rows = torch.as_tensor(centers).long().reshape(-1)
    primary = torch.as_tensor(primary_valid).bool()
    if neighbors.shape != active.shape or center_rows.shape != (
        neighbors.shape[0],
    ):
        raise ValueError("surface region rows, mask, and centers must align")
    if neighbors.numel() and int(neighbors.max()) >= primary.numel():
        raise ValueError("surface region rows exceed the primary partition")
    return active & (
        primary[neighbors] | (neighbors == center_rows[:, None])
    )


def _load_versioned_surface_region_readout(
    path: Path,
    *,
    expected_sha256: str | None,
) -> tuple[torch.nn.Module, dict, str, Path]:
    """Load the immutable V2 or gauge-explicit V3 readout fail-closed."""

    payload, digest, source = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="surface-region summary checkpoint",
    )
    architecture = payload.get("architecture", {})
    name = architecture.get("name") if isinstance(architecture, Mapping) else None
    if name == "surface_region_summary_readout_v2":
        return load_surface_region_summary_readout_v2(
            path,
            expected_sha256=expected_sha256,
            map_location="cpu",
        )
    if name != "surface_region_summary_readout_v3":
        raise ValueError("unsupported surface-region readout architecture")
    model, _reopened = SurfaceRegionSummaryReadoutV3.from_checkpoint(
        path,
        map_location="cpu",
    )
    return model, payload, digest, source


def validate_surface_region_readout_deployment_authority(
    payload: Mapping,
    *,
    contract: SurfaceRegionContractV2 | SurfaceRegionContractV3,
    radio_checkpoint_sha256: str,
    readout_checkpoint_sha256: str = "",
    legacy_radio_authority: Mapping[str, Any] | None = None,
) -> None:
    """Bind architecture, training provenance, contract, and RADIO exactly."""

    architecture = payload.get("architecture")
    provenance = payload.get("provenance")
    if not isinstance(architecture, Mapping) or not isinstance(
        provenance, Mapping
    ):
        raise ValueError("surface-region readout lacks deployment authority")
    contract_hashes = {
        str(architecture.get("contract_sha256", "")),
        str(provenance.get("region_contract_sha256", "")),
        str(contract.digest),
    }
    if "" in contract_hashes or len(contract_hashes) != 1:
        raise ValueError(
            "surface-region readout architecture/provenance contract differs"
        )
    expected_radio = str(radio_checkpoint_sha256)
    if not expected_radio:
        raise ValueError("current RADIO checkpoint lacks a SHA-256 authority")
    training_radio = []
    for split in ("train", "validation"):
        split_provenance = provenance.get(split)
        if not isinstance(split_provenance, Mapping):
            raise ValueError(
                f"surface-region readout lacks {split} RADIO authority"
            )
        training_radio.append(
            str(split_provenance.get("radio_checkpoint_sha256", ""))
        )
    if any(training_radio):
        if any(value != expected_radio for value in training_radio):
            raise ValueError(
                "surface-region readout training/current RADIO authority differs"
            )
        if legacy_radio_authority is not None:
            raise ValueError(
                "inline RADIO provenance cannot be combined with legacy authority"
            )
    else:
        _validate_legacy_surface_region_v2_radio_authority(
            payload,
            contract=contract,
            readout_checkpoint_sha256=str(readout_checkpoint_sha256),
            radio_checkpoint_sha256=expected_radio,
            authority=legacy_radio_authority,
        )
    if payload.get("schema_version") == SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION:
        mode = str(architecture.get("base_output_mode", ""))
        v3_provenance = provenance.get("surface_region_v3")
        if mode != SURFACE_REGION_V3_GATED_RAW_PRIOR or not isinstance(
            v3_provenance, Mapping
        ) or str(v3_provenance.get("effective_base_output_mode", "")) != mode:
            raise ValueError(
                "gated V3 readout architecture/provenance mode differs"
            )


def _validate_legacy_surface_region_v2_radio_authority(
    payload: Mapping[str, Any],
    *,
    contract: SurfaceRegionContractV2 | SurfaceRegionContractV3,
    readout_checkpoint_sha256: str,
    radio_checkpoint_sha256: str,
    authority: Mapping[str, Any] | None,
) -> None:
    """Close only the immutable accepted-V2 checkpoint's missing RADIO field.

    The historical checkpoint retained exact train/validation cache paths but
    omitted their already-present RADIO SHA from its summarized provenance.
    Reopen and hash every declared cache so this compatibility route is at
    least as strong as an inline declaration and cannot authorize another
    readout, contract, split, or RADIO checkpoint.
    """

    architecture = payload.get("architecture")
    provenance = payload.get("provenance")
    if (
        not isinstance(contract, SurfaceRegionContractV2)
        or not isinstance(architecture, Mapping)
        or architecture.get("name") != "surface_region_summary_readout_v2"
        or not isinstance(provenance, Mapping)
        or not isinstance(authority, Mapping)
        or authority.get("schema_version") != 1
        or authority.get("registration")
        != "surface_region_accepted_v2_legacy_radio_training_authority_v1"
        or str(authority.get("readout_checkpoint_sha256", ""))
        != str(readout_checkpoint_sha256)
        or str(authority.get("region_contract_sha256", "")) != contract.digest
        or str(authority.get("radio_checkpoint_sha256", ""))
        != str(radio_checkpoint_sha256)
        or authority.get("cache_validation", {}).get("benchmark_masks_opened")
        is not False
        or authority.get("cache_validation", {}).get("benchmark_metrics_opened")
        is not False
    ):
        raise ValueError("legacy SurfaceRegion V2 RADIO authority differs")

    for split, expected_role in (("train", "train"), ("validation", "validation")):
        split_provenance = provenance.get(split)
        split_authority = authority.get(split)
        if not isinstance(split_provenance, Mapping) or not isinstance(
            split_authority, Mapping
        ):
            raise ValueError(f"legacy RADIO authority lacks {split} split")
        cache_records = split_authority.get("caches")
        if not isinstance(cache_records, list) or not cache_records:
            raise ValueError(f"legacy RADIO authority {split} caches differ")
        authority_paths = [str(record.get("path", "")) for record in cache_records]
        if authority_paths != list(split_provenance.get("cache_paths", [])):
            raise ValueError(f"legacy RADIO authority {split} cache paths differ")
        declared_split_hash = str(split_authority.get("split_file_sha256", ""))
        if list(split_provenance.get("split_hashes", [])) != [declared_split_hash]:
            raise ValueError(f"legacy RADIO authority {split} split hash differs")
        for index, record in enumerate(cache_records):
            cache_path = validate_file_record(
                record,
                label=f"legacy SurfaceRegion V2 {split} cache {index}",
            )
            cache, _, _ = load_torch_mapping(
                cache_path,
                expected_sha256=str(record["sha256"]),
                map_location="cpu",
                label=f"legacy SurfaceRegion V2 {split} cache {index}",
            )
            metadata = cache.get("metadata")
            if (
                not isinstance(metadata, Mapping)
                or str(metadata.get("split_role", "")) != expected_role
                or str(metadata.get("split_file_sha256", ""))
                != declared_split_hash
                or str(metadata.get("region_contract_sha256", ""))
                != contract.digest
                or str(metadata.get("radio_version", ""))
                != str(authority.get("radio_version", ""))
                or str(metadata.get("radio_checkpoint_sha256", ""))
                != str(radio_checkpoint_sha256)
                or metadata.get("uses_benchmark_scenes") is not False
                or metadata.get("uses_benchmark_test_vocabulary") is not False
                or metadata.get("labels_opened") is not False
                or metadata.get("text_opened") is not False
            ):
                raise ValueError(
                    f"legacy SurfaceRegion V2 {split} cache metadata differs"
                )


def expand_surface_region_v3_batch_at_radius(
    contract: SurfaceRegionContractV3,
    support: PrimitiveSupportGraph,
    xyz: torch.Tensor,
    centers: torch.Tensor,
    radius: float,
    *,
    prepared_graph,
    primary_local: torch.Tensor | None,
) -> list[RegionSelection]:
    """Return ordered V3 selections under primary/fallback eligibility.

    A primary anchor traverses only the primary-induced graph.  A fallback
    anchor may use every output-valid row.  Grouping anchors by policy retains
    batched expansion without reintroducing the old post-expansion filter.
    """

    anchor_rows = torch.as_tensor(centers).detach().long().cpu().reshape(-1)
    if anchor_rows.numel() == 0:
        return []
    output_valid = torch.ones(support.num_nodes, dtype=torch.bool)
    primary = (
        None
        if primary_local is None
        else torch.as_tensor(primary_local).detach().bool().cpu().reshape(-1)
    )
    if primary is not None and primary.shape != output_valid.shape:
        raise ValueError("primary partition does not align with the support graph")
    grouped: list[RegionSelection | None] = [None] * len(anchor_rows)
    primary_flags = (
        torch.zeros(len(anchor_rows), dtype=torch.bool)
        if primary is None
        else primary[anchor_rows]
    )
    for wants_primary in (True, False):
        offsets = torch.where(primary_flags == wants_primary)[0]
        if offsets.numel() == 0:
            continue
        eligibility = primary if wants_primary and primary is not None else output_valid
        selected_anchors = anchor_rows[offsets]
        expanded = contract.expand_batch(
            support,
            xyz,
            selected_anchors.tolist(),
            float(radius),
            prepared_graph=prepared_graph,
            selection_eligibility=eligibility,
        )
        for offset, anchor, raw in zip(
            offsets.tolist(), selected_anchors.tolist(), expanded
        ):
            grouped[offset] = as_region_selection(
                raw,
                anchor_row=int(anchor),
            ).pad_to(contract.maximum_tokens)
    if any(value is None for value in grouped):
        raise RuntimeError("V3 primary/fallback expansion left an anchor unresolved")
    return [value for value in grouped if value is not None]


@torch.no_grad()
def build(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    canonical_radio_source = str(
        getattr(args, "canonical_radio_source", "field_decode")
    )
    if canonical_radio_source not in {"field_decode", "mpr_teacher"}:
        raise ValueError("unsupported canonical RADIO source")
    radio_feature_normalization = str(
        getattr(args, "radio_feature_normalization", "legacy_raw")
    )
    if radio_feature_normalization not in {
        "l2_direction",
        "l2_direction_plus_log_raw_norm_v1",
        "legacy_raw",
    }:
        raise ValueError("unsupported surface-region RADIO normalization")
    registration_arg = str(getattr(args, "experiment_registration", "")).strip()
    registration_record = file_record(registration_arg) if registration_arg else None
    registration_payload = (
        load_json_object(
            registration_arg,
            label="surface-region experiment registration",
        )[0]
        if registration_arg
        else None
    )
    if canonical_radio_source == "mpr_teacher" and registration_record is None:
        raise ValueError("mpr_teacher capacity diagnostics require preregistration")
    field_path, graph_path, readout_path = map(Path, (
        args.field_checkpoint, args.support_graph, args.readout_checkpoint,
    ))
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"semantic output already exists: {output}")
    router_values = {
        "codebook": str(
            getattr(args, "residual_codebook_checkpoint", "")
        ).strip(),
        "codebook_sha256": str(
            getattr(args, "residual_codebook_checkpoint_sha256", "")
        ).strip(),
        "router": str(
            getattr(args, "query_router_checkpoint", "")
        ).strip(),
        "router_sha256": str(
            getattr(args, "query_router_checkpoint_sha256", "")
        ).strip(),
        "negative_text": str(
            getattr(args, "generic_negative_text_cache", "")
        ).strip(),
        "negative_text_sha256": str(
            getattr(args, "generic_negative_text_cache_sha256", "")
        ).strip(),
        "control_output": str(
            getattr(args, "router_control_output", "")
        ).strip(),
    }
    query_router_mode = any(router_values.values())
    if query_router_mode and not all(router_values.values()):
        missing = sorted(key for key, value in router_values.items() if not value)
        raise ValueError(
            "query-router mode requires every codebook/router/negative/control "
            f"argument; missing {missing}"
        )
    full_scalar_values = _full_scalar_overlay_arguments(args)
    router_control_output = (
        Path(router_values["control_output"]).resolve()
        if query_router_mode
        else None
    )
    if router_control_output is not None:
        if router_control_output == output:
            raise ValueError("router candidate and exact-control outputs must differ")
        if router_control_output.exists() or router_control_output.is_symlink():
            raise FileExistsError(
                f"router control output already exists: {router_control_output}"
            )
    radio_batch_size = int(args.radio_batch_size)
    semantic_batch_size = int(args.semantic_batch_size)
    pacing_seconds = float(args.thermal_pacing_seconds_per_batch)
    if (
        radio_batch_size <= 0
        or semantic_batch_size <= 0
        or not math.isfinite(pacing_seconds)
        or pacing_seconds <= 0.0
    ):
        raise ValueError("batch sizes and thermal pacing must be positive")

    field_expected = str(getattr(args, "field_checkpoint_sha256", "")).strip() or None
    graph_expected = str(getattr(args, "support_graph_sha256", "")).strip() or None
    readout_expected = str(getattr(args, "readout_checkpoint_sha256", "")).strip() or None
    mpr_expected = str(getattr(args, "mpr_cache_sha256", "")).strip() or None
    radio_expected = str(getattr(args, "radio_checkpoint_sha256", "")).strip() or None
    field_schema = str(
        getattr(
            args,
            "field_checkpoint_schema",
            CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1,
        )
    )
    factorized_state_arg = str(
        getattr(args, "factorized_primitive_state", "")
    ).strip()
    factorized_state_expected = str(
        getattr(args, "factorized_primitive_state_sha256", "")
    ).strip()
    factorized_support = None
    factorized_state = None
    factorized_state_record = None
    if field_schema == CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1:
        if factorized_state_arg or factorized_state_expected:
            raise ValueError(
                "legacy SurfaceRegion field cannot bind factorized primitive state"
            )
        field, field_payload = load_canonical_field_checkpoint(
            field_path,
            map_location="cpu",
            expected_sha256=field_expected,
        )
    elif field_schema == FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2:
        if field_expected is None:
            raise ValueError(
                "factorized SurfaceRegion field requires its trusted SHA-256"
            )
        if canonical_radio_source != "field_decode":
            raise ValueError("factorized SurfaceRegion forbids MPR-teacher decoding")
        if not factorized_state_arg or not factorized_state_expected:
            raise ValueError(
                "factorized SurfaceRegion requires a trusted primitive-state sidecar"
            )
        factorized_support, factorized_state = load_surface_factorized_state_bundle(
            field_path,
            expected_field_checkpoint_sha256=field_expected,
            expected_factorized_radio_cache_sha256=(mpr_expected or ""),
            state_path=factorized_state_arg,
            expected_state_sha256=factorized_state_expected,
        )
        field = factorized_support.field
        field_payload = dict(factorized_support.field_payload)
        factorized_state_record = file_record(factorized_state_arg)
    else:
        raise ValueError("unsupported SurfaceRegion field checkpoint schema")
    context_field_arg = str(
        getattr(args, "directional_context_field_checkpoint", "")
    ).strip()
    context_field_expected = str(
        getattr(args, "directional_context_field_checkpoint_sha256", "")
    ).strip() or None
    context_field = None
    context_field_payload = None
    context_field_record = None
    if context_field_arg:
        if field_schema != CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1:
            raise ValueError(
                "factorized SurfaceRegion forbids an independently gauged context field"
            )
        if canonical_radio_source != "field_decode":
            raise ValueError(
                "directional anchor context requires canonical field decoding"
            )
        if context_field_expected is None:
            raise ValueError(
                "directional anchor context requires its trusted SHA-256"
            )
        context_field, context_field_payload = load_canonical_field_checkpoint(
            context_field_arg,
            map_location="cpu",
            expected_sha256=context_field_expected,
        )
        context_field_record = file_record(context_field_arg)
        if context_field_payload.get("geometry_fingerprint") != field_payload.get(
            "geometry_fingerprint"
        ):
            raise ValueError("directional anchor/context field geometry differs")
        if context_field_payload.get("feature_signature") != field_payload.get(
            "feature_signature"
        ):
            raise ValueError("directional anchor/context feature signatures differ")
        for payload, label in (
            (field_payload, "directional anchor field"),
            (context_field_payload, "directional context field"),
        ):
            if any(
                payload.get(key) is not False
                for key in (
                    "benchmark_images_opened",
                    "benchmark_masks_opened",
                    "text_queries_opened",
                )
            ):
                raise ValueError(f"{label} is task contaminated")
    graph, _, _ = load_torch_mapping(
        graph_path,
        expected_sha256=graph_expected,
        map_location="cpu",
        label="surface support graph",
    )
    readout, readout_payload, _, _ = _load_versioned_surface_region_readout(
        readout_path,
        expected_sha256=readout_expected,
    )
    if readout_payload["provenance"].get("uses_benchmark_scenes", True):
        raise ValueError("readout provenance is benchmark contaminated")
    mpr_path = Path(field_payload["mpr_cache"]).resolve()
    if factorized_support is None:
        mpr, _, _ = load_torch_mapping(
            mpr_path,
            expected_sha256=mpr_expected,
            map_location="cpu",
            label="canonical field MPR cache",
        )
    else:
        mpr = None
        if mpr_path != factorized_support.cache.source.resolve():
            raise ValueError("factorized SurfaceRegion cache path differs from field")
        if mpr_expected and mpr_expected != factorized_support.cache.sha256:
            raise ValueError("factorized SurfaceRegion cache SHA-256 differs")
    field_record = file_record(field_path)
    graph_record = file_record(graph_path)
    readout_record = file_record(readout_path)
    mpr_record = file_record(mpr_path)
    radio_record = file_record(args.radio_checkpoint)
    for expected, record, label in (
        (field_expected, field_record, "field"),
        (graph_expected, graph_record, "support graph"),
        (readout_expected, readout_record, "readout"),
        (mpr_expected, mpr_record, "MPR"),
        (radio_expected, radio_record, "RADIO"),
    ):
        if expected is not None and record["sha256"] != expected:
            raise ValueError(f"{label} checkpoint SHA-256 differs")
    legacy_radio_authority_arg = str(
        getattr(args, "readout_legacy_radio_authority", "")
    ).strip()
    legacy_radio_authority_sha256 = str(
        getattr(args, "readout_legacy_radio_authority_sha256", "")
    ).strip()
    legacy_radio_authority = None
    legacy_radio_authority_record = None
    if legacy_radio_authority_arg or legacy_radio_authority_sha256:
        if not legacy_radio_authority_arg or not legacy_radio_authority_sha256:
            raise ValueError(
                "legacy readout RADIO authority path and SHA-256 are both required"
            )
        (
            legacy_radio_authority,
            _legacy_radio_authority_digest,
            _legacy_radio_authority_source,
        ) = load_json_object(
            legacy_radio_authority_arg,
            expected_sha256=legacy_radio_authority_sha256,
            label="legacy SurfaceRegion V2 RADIO authority",
        )
        legacy_radio_authority_record = {
            "path": str(_legacy_radio_authority_source),
            "sha256": _legacy_radio_authority_digest,
        }
    xyz_global = (
        torch.as_tensor(mpr["xyz"]).float().cpu()
        if mpr is not None
        else factorized_state.xyz
    )
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    xyz = torch.as_tensor(graph["xyz"]).float().cpu()
    if not torch.equal(xyz, xyz_global[global_rows]):
        raise ValueError("support graph and canonical field geometry differ")
    if factorized_state is not None:
        if not torch.equal(global_rows, factorized_state.global_rows):
            raise ValueError(
                "factorized primitive state and support graph rows differ"
            )
        graph_metadata = graph.get("metadata")
        capability_metadata = (
            graph_metadata.get("capability_metadata")
            if isinstance(graph_metadata, Mapping)
            else None
        )
        if (
            not isinstance(capability_metadata, Mapping)
            or capability_metadata.get("field_checkpoint_schema_version") != 2
            or capability_metadata.get("field_checkpoint_sha256") != field_expected
            or capability_metadata.get("factorized_radio_cache_sha256")
            != factorized_support.cache.sha256
            or capability_metadata.get(
                "factorized_radio_field_signature_sha256"
            )
            != factorized_support.field_signature.digest
        ):
            raise ValueError(
                "factorized SurfaceRegion graph lineage differs from field/state"
            )
    output_valid = torch.zeros(len(xyz_global), dtype=torch.bool)
    output_valid[global_rows] = True
    primary_valid = (
        completion_primary_valid(mpr, output_valid) if mpr is not None else None
    )
    primary_local = (
        primary_valid[global_rows] if primary_valid is not None else None
    )
    provenance = readout_payload["provenance"]
    training_scope = str(provenance.get("training_scope", ""))
    if (
        not training_scope.startswith("global_cross_scene")
        or provenance.get("uses_benchmark_scenes", True)
        or provenance.get("uses_benchmark_test_vocabulary", True)
        or provenance.get("scene_disjoint") is not True
    ):
        raise ValueError("readout provenance is not frozen global cross-scene training")
    contract = surface_region_contract_from_metadata(
        {
            **provenance,
            "region_contract_version": provenance.get(
                "region_contract_version",
                provenance["region_contract"].get("version"),
            ),
        }
    )
    if str(args.region_radii).strip():
        requested = tuple(
            float(value) for value in str(args.region_radii).replace(",", " ").split()
        )
        if requested != contract.radii_m:
            raise ValueError("CLI radii differ from the frozen readout contract")
    contract.assert_compatible({
        "region_contract_version": contract.version,
        "region_contract_sha256": provenance["region_contract_sha256"],
    })
    validate_surface_region_readout_deployment_authority(
        readout_payload,
        contract=contract,
        radio_checkpoint_sha256=radio_record["sha256"],
        readout_checkpoint_sha256=readout_record["sha256"],
        legacy_radio_authority=legacy_radio_authority,
    )
    if isinstance(contract, SurfaceRegionContractV3) and context_field is not None:
        raise ValueError(
            "V3 gauge-explicit deployment does not permit an independently "
            "gauged directional context field"
        )
    if radio_feature_normalization != contract.feature_normalization and not (
        type(contract) is SurfaceRegionContractV2
        and radio_feature_normalization == "legacy_raw"
        and contract.feature_normalization == "l2_direction"
    ):
        raise ValueError(
            "requested RADIO gauge differs from the frozen SurfaceRegion contract"
        )
    full_scalar_state = None
    full_scalar_state_record = None
    full_scalar_normalization = None
    full_scalar_normalization_record = None
    full_scalar_residual = None
    full_scalar_residual_payload = None
    full_scalar_residual_record = None
    full_scalar_certificate = None
    full_scalar_certificate_record = None
    full_scalar_routing = None
    full_scalar_runtime_carrier = None
    if full_scalar_values is not None:
        _validate_full_scalar_overlay_mode(
            contract=contract,
            field_schema=field_schema,
            canonical_radio_source=canonical_radio_source,
            context_field=context_field,
            query_router_mode=query_router_mode,
        )
        full_scalar_state = load_factorized_primitive_state(
            full_scalar_values["accepted_v2_full_scalar_state"],
            expected_sha256=full_scalar_values[
                "accepted_v2_full_scalar_state_sha256"
            ],
            # Geometry is shared, while validity deliberately is not: the
            # exact state may have support outside the immutable accepted base.
            expected_xyz=xyz_global,
        )
        full_scalar_state_record = file_record(
            full_scalar_values["accepted_v2_full_scalar_state"]
        )
        if (
            full_scalar_state_record["sha256"]
            != full_scalar_values["accepted_v2_full_scalar_state_sha256"]
        ):
            raise ValueError("full-scalar state SHA-256 differs")
        normalization_payload, normalization_digest, normalization_source = (
            load_torch_mapping(
                full_scalar_values["full_scalar_normalization_authority"],
                expected_sha256=full_scalar_values[
                    "full_scalar_normalization_authority_sha256"
                ],
                map_location="cpu",
                label="full-scalar normalization authority",
            )
        )
        full_scalar_normalization = validate_full_scalar_normalization_authority(
            normalization_payload
        )
        full_scalar_normalization_record = {
            "path": str(normalization_source),
            "sha256": normalization_digest,
        }
        (
            full_scalar_certificate,
            full_scalar_certificate_record,
        ) = load_training_certificate(
            full_scalar_values["full_scalar_training_certificate"],
            expected_sha256=full_scalar_values[
                "full_scalar_training_certificate_sha256"
            ],
        )
        if (
            full_scalar_certificate["normalization_authority"]["sha256"]
            != normalization_digest
        ):
            raise ValueError(
                "full-scalar certificate/normalization authority differs"
            )
        (
            full_scalar_residual,
            full_scalar_residual_payload,
        ) = load_surface_region_full_scalar_residual_checkpoint(
            full_scalar_values["full_scalar_residual_checkpoint"],
            expected_checkpoint_sha256=full_scalar_values[
                "full_scalar_residual_checkpoint_sha256"
            ],
            normalization_authority=full_scalar_normalization,
            expected_normalization_authority_sha256=normalization_digest,
            expected_source_state_cohort_authority_sha256=(
                full_scalar_normalization["source_state_cohort_sha256"]
            ),
            training_certificate=full_scalar_certificate,
            expected_training_certificate_sha256=(
                full_scalar_certificate_record["sha256"]
            ),
            map_location="cpu",
        )
        full_scalar_residual_record = file_record(
            full_scalar_values["full_scalar_residual_checkpoint"]
        )
        if (
            full_scalar_residual_record["sha256"]
            != full_scalar_values["full_scalar_residual_checkpoint_sha256"]
        ):
            raise ValueError("full-scalar residual checkpoint SHA-256 differs")
        _validate_full_scalar_runtime_base_authority(
            full_scalar_residual_payload,
            readout_payload=readout_payload,
            readout_checkpoint_sha256=readout_record["sha256"],
            contract_sha256=contract.digest,
        )
        full_scalar_routing = build_full_scalar_support_routing(
            output_valid,
            full_scalar_state,
        )
        if bool(
            (
                full_scalar_routing.exact_only_abstain_mask
                | full_scalar_routing.neither_abstain_mask
            )[output_valid].any()
        ):
            raise RuntimeError("accepted V2 validity admitted an abstain route")
        full_scalar_runtime_carrier = _build_full_scalar_runtime_carrier(
            full_scalar_state,
            full_scalar_routing,
        )
    stream_text = bool(str(args.stream_text_queries).strip())
    preserve_streamed_text_scales = bool(
        getattr(args, "preserve_streamed_text_scales", False)
    )
    if preserve_streamed_text_scales and not stream_text:
        raise ValueError(
            "--preserve-streamed-text-scales requires --stream-text-queries"
        )
    residual_codebook = None
    query_router = None
    router_negative_text = None
    codebook_record = None
    router_record = None
    negative_text_record = None
    router_gauge_authority_record = None
    router_deployment_gauge = None
    if query_router_mode:
        if not stream_text or not preserve_streamed_text_scales:
            raise ValueError(
                "query-router mode requires streamed text with all native scales"
            )
        if type(contract) is not SurfaceRegionContractV2:
            raise ValueError(
                "query-router benchmark deployment requires the accepted V2 contract"
            )
        gauge_authority_arg = str(
            getattr(args, "query_router_gauge_authority", "")
        ).strip()
        gauge_authority_sha256 = str(
            getattr(args, "query_router_gauge_authority_sha256", "")
        ).strip()
        gauge_authority_payload = None
        if gauge_authority_arg or gauge_authority_sha256:
            if not gauge_authority_arg or not gauge_authority_sha256:
                raise ValueError(
                    "query-router gauge authority path and SHA-256 are both required"
                )
            router_gauge_authority_record = file_record(gauge_authority_arg)
            if router_gauge_authority_record["sha256"] != gauge_authority_sha256:
                raise ValueError("query-router gauge authority SHA-256 differs")
            gauge_authority_payload, _, _ = load_json_object(
                gauge_authority_arg,
                expected_sha256=gauge_authority_sha256,
                label="query-router gauge authority",
            )
        router_deployment_gauge = validate_query_router_deployment_gauge(
            normalization=radio_feature_normalization,
            experiment_registration=registration_payload,
            gauge_authority=gauge_authority_payload,
        )
        codebook_path = Path(router_values["codebook"])
        router_path = Path(router_values["router"])
        negative_text_path = Path(router_values["negative_text"])
        codebook_record = file_record(codebook_path)
        router_record = file_record(router_path)
        if codebook_record["sha256"] != router_values["codebook_sha256"]:
            raise ValueError("residual codebook SHA-256 differs")
        if router_record["sha256"] != router_values["router_sha256"]:
            raise ValueError("query-router SHA-256 differs")
        residual_codebook, codebook_payload = (
            SurfaceRegionSummaryResidualCodebookV1.from_checkpoint(
                codebook_path, map_location="cpu"
            )
        )
        query_router, router_payload = SurfaceRegionQueryRouterV1.from_checkpoint(
            router_path, map_location="cpu"
        )
        router_negative_text, negative_text_record = (
            _load_query_router_negative_text(
                negative_text_path,
                expected_sha256=router_values["negative_text_sha256"],
            )
        )
        codebook_architecture = codebook_payload.get("architecture", {})
        codebook_provenance = codebook_payload.get("provenance", {})
        router_architecture = router_payload.get("architecture", {})
        router_provenance = router_payload.get("provenance", {})
        if (
            codebook_architecture.get("control_sha256")
            != readout_record["sha256"]
            or codebook_architecture.get("contract_sha256") != contract.digest
            or codebook_provenance.get("region_contract_sha256")
            != contract.digest
            or codebook_provenance.get("control_readout", {}).get("sha256")
            != readout_record["sha256"]
            or codebook_provenance.get("uses_benchmark_scenes") is not False
            or codebook_provenance.get("uses_benchmark_test_vocabulary")
            is not False
            or codebook_provenance.get("benchmark_images_opened") is not False
            or codebook_provenance.get("benchmark_masks_opened") is not False
            or codebook_architecture.get("canonical_gauge")
            != "caller_provided_exact_frozen_v2"
            or codebook_architecture.get("residual_gauge")
            != "l2_direction_tangent_plus_log_norm"
        ):
            raise ValueError("residual codebook authority differs or is contaminated")
        if not _state_dicts_bitwise_equal(
            residual_codebook.base.state_dict(), readout.state_dict()
        ):
            raise ValueError("residual codebook base is not the exact frozen readout")
        if (
            router_architecture.get("codebook_sha256")
            != codebook_record["sha256"]
            or router_provenance.get("frozen_codebook", {}).get("sha256")
            != codebook_record["sha256"]
            or router_provenance.get("generic_negative_text_bank", {}).get(
                "sha256"
            )
            != negative_text_record["sha256"]
            or router_provenance.get("generic_negative_text_bank", {}).get(
                "queries"
            )
            != negative_text_record["queries"]
            or router_provenance.get("score_contract")
            != "canonical_negative_bernoulli_query_first"
            or float(router_provenance.get("logit_scale", -1.0)) != 10.0
            or router_payload.get("generic_gate", {}).get("overall_pass")
            is not True
            or router_provenance.get("uses_benchmark_scenes") is not False
            or router_provenance.get("uses_benchmark_test_vocabulary") is not False
            or router_provenance.get("benchmark_images_opened") is not False
            or router_provenance.get("benchmark_masks_opened") is not False
        ):
            raise ValueError("query-router authority differs or is contaminated")
        training_radio_sha = codebook_provenance.get("train", {}).get(
            "radio_checkpoint_sha256"
        )
        if training_radio_sha != radio_record["sha256"]:
            raise ValueError("query-router RADIO authority differs")
    resume_dir = (
        Path(args.resume_dir).resolve()
        if str(args.resume_dir).strip()
        else output.with_name(f"{output.name}.resume")
    )
    text_record = (
        file_record(args.text_embedding_cache)
        if stream_text and str(args.text_embedding_cache).strip()
        else None
    )
    resume_contract = {
        "schema_version": 1,
        "artifact_type": "surface_semantic_resume_contract",
        "output": str(output),
        **(
            {"router_control_output": str(router_control_output)}
            if router_control_output is not None
            else {}
        ),
        "inputs": {
            "field": field_record,
            **(
                {"directional_context_field": context_field_record}
                if context_field_record is not None
                else {}
            ),
            "mpr": mpr_record,
            **(
                {"factorized_primitive_state": factorized_state_record}
                if factorized_state_record is not None
                else {}
            ),
            **(
                {
                    "accepted_v2_full_scalar_state": full_scalar_state_record,
                    "full_scalar_normalization_authority": (
                        full_scalar_normalization_record
                    ),
                    "full_scalar_residual_checkpoint": (
                        full_scalar_residual_record
                    ),
                }
                if full_scalar_residual is not None
                else {}
            ),
            "support_graph": graph_record,
            "readout": readout_record,
            **(
                {"readout_legacy_radio_authority": legacy_radio_authority_record}
                if legacy_radio_authority_record is not None
                else {}
            ),
            "radio": radio_record,
            "text": text_record,
            **(
                {
                    "residual_codebook": codebook_record,
                    "query_router": router_record,
                    "generic_negative_text": negative_text_record,
                    **(
                        {"query_router_gauge_authority": router_gauge_authority_record}
                        if router_gauge_authority_record is not None
                        else {}
                    ),
                }
                if query_router_mode
                else {}
            ),
            **(
                {"experiment_registration": registration_record}
                if registration_record is not None
                else {}
            ),
        },
        "region_contract": contract.to_dict(),
        "region_contract_sha256": contract.digest,
        **(
            {
                "accepted_v2_full_scalar_overlay": {
                    "contract_sha256": (
                        SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256
                    ),
                    "summary_names_sha256": (
                        SURFACE_REGION_FULL_SCALAR_NAMES_SHA256
                    ),
                    "source_state_cohort_sha256": (
                        full_scalar_normalization[
                            "source_state_cohort_sha256"
                        ]
                    ),
                    "accepted_valid_sha256": tensor_sha256(output_valid),
                    "accepted_global_rows_sha256": tensor_sha256(global_rows),
                }
            }
            if full_scalar_residual is not None
            else {}
        ),
        "radio_feature_normalization": radio_feature_normalization,
        "radio_batch_size": radio_batch_size,
        "semantic_batch_size": semantic_batch_size,
        "stream_text_queries": str(args.stream_text_queries),
        **(
            {
                "score_contract": (
                    "canonical_negative_bernoulli_query_router_v1"
                ),
                "logit_scale": 10.0,
                "paired_resume_channel_order": [
                    "router_candidate",
                    "exact_frozen_v2_slot0_control",
                ],
            }
            if query_router_mode
            else {}
        ),
        **(
            {"preserve_streamed_text_scales": True}
            if preserve_streamed_text_scales
            else {}
        ),
        **(
            {"canonical_radio_source": canonical_radio_source}
            if canonical_radio_source == "mpr_teacher"
            else {}
        ),
        "device_type": device.type,
        "thermal_pacing_seconds_per_batch": pacing_seconds,
        "implementation": file_record(Path(__file__).resolve()),
    }
    resume_contract_sha256 = _load_or_create_resume_contract(
        resume_dir,
        resume_contract,
    )
    support = PrimitiveSupportGraph(
        edge_index=graph["edge_index"], edge_weight=graph["edge_weight"],
        raw_affinity=graph["raw_affinity"], local_sigma=graph["local_sigma"],
        num_nodes=len(xyz), edge_channels=graph.get("edge_channels", {}),
    )
    prepared_graph = contract.prepare_graph(support, xyz)
    field, readout = field.to(device).eval(), readout.to(device).eval()
    if context_field is not None:
        context_field = context_field.to(device).eval()
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        args.radio_checkpoint,
        **({"expected_sha256": radio_expected} if radio_expected else {}),
    ).to(device).eval()
    if residual_codebook is not None:
        residual_codebook = residual_codebook.to(device).eval()
    if query_router is not None:
        query_router = query_router.to(device).eval()
    if full_scalar_residual is not None:
        full_scalar_residual = full_scalar_residual.to(device).eval()
    if router_negative_text is not None:
        router_negative_text = router_negative_text.to(device)
    for module in (
        field,
        readout,
        head,
        context_field,
        residual_codebook,
        query_router,
        full_scalar_residual,
    ):
        if module is None:
            continue
        for parameter in module.parameters(): parameter.requires_grad_(False)
    semantic_phase = (
        "query_router_v1_probability_and_v2_control_multiscale"
        if query_router_mode
        else "text_scores_multiscale"
        if preserve_streamed_text_scales
        else ("text_scores" if stream_text else "semantic")
    )
    _validate_resume_inventory(
        resume_dir,
        row_count=len(global_rows),
        radio_batch_size=radio_batch_size,
        semantic_batch_size=semantic_batch_size,
        semantic_phase=semantic_phase,
    )
    radio = torch.empty(len(global_rows), 1280, dtype=torch.float16, device=device)
    anchor_radio = (
        torch.empty_like(radio) if context_field is not None else None
    )
    for start in range(0, len(global_rows), radio_batch_size):
        stop = min(start + radio_batch_size, len(global_rows))
        selected_rows = global_rows[start:stop].to(device)
        if anchor_radio is not None:
            anchor_radio[start:stop] = field.radio_features(selected_rows).half()
        cached = _load_resume_tensor(
            resume_dir,
            phase="radio",
            start=start,
            stop=stop,
            contract_sha256=resume_contract_sha256,
            expected_shape=(stop - start, 1280),
            expected_dtype=torch.float16,
        )
        if cached is not None:
            radio[start:stop] = cached.to(device)
            continue
        if canonical_radio_source == "field_decode":
            source_field = context_field if context_field is not None else field
            computed = source_field.radio_features(selected_rows).half()
        else:
            if mpr is None:
                raise RuntimeError("factorized SurfaceRegion has no MPR-teacher route")
            computed = torch.as_tensor(mpr["features"])[
                global_rows[start:stop]
            ].to(device=device, dtype=torch.float16)
        radio[start:stop] = computed
        _commit_resume_tensor(
            resume_dir,
            phase="radio",
            start=start,
            stop=stop,
            contract_sha256=resume_contract_sha256,
            value=computed,
        )
        _pace_after_commit(device, pacing_seconds)
    semantic_confidence = None
    if primary_valid is not None:
        teacher_radio = torch.as_tensor(mpr["features"])[global_rows]
        observation_counts = torch.as_tensor(mpr["view_counts"])[global_rows]
        local_confidence = torch.zeros(
            len(global_rows), dtype=torch.float16
        )
        for start in range(0, len(global_rows), radio_batch_size):
            stop = min(start + radio_batch_size, len(global_rows))
            local_confidence[start:stop] = canonical_reconstruction_confidence(
                radio[start:stop].float(),
                teacher_radio[start:stop].to(
                    device=device, dtype=torch.float32
                ),
                torch.ones(stop - start, dtype=torch.bool, device=device),
                primary_local[start:stop].to(device),
                observation_counts[start:stop].to(device),
            ).half().cpu()
        semantic_confidence = torch.zeros(
            len(xyz_global), dtype=torch.float16
        )
        semantic_confidence[global_rows] = local_confidence
    if factorized_state is not None:
        # This compatibility value is the only factorized-state quantity
        # presented to an unchanged V2/V3 readout in phase one.  The complete
        # six-column state is merely lineage-bound; no new residual/head is
        # introduced here.
        reliability = factorized_state.legacy_geometric_reliability()
    elif contract.reliability_semantics == "uniform_valid":
        reliability = torch.ones(len(global_rows), dtype=torch.float32)
    else:
        reliability_source = torch.as_tensor(mpr.get("reliability")).float()[
            global_rows
        ]
        if reliability_source.ndim != 2 or reliability_source.shape[1] < 2:
            raise ValueError(
                "canonical MPR reliability needs coverage/agreement channels"
            )
        reliability = (
            reliability_source[:, :2].clamp_min(1e-6).log().mean(-1).exp()
        )
        reliability[(reliability_source[:, :2] <= 0).any(-1)] = 0.0
    reliability = reliability.to(device)
    local_scale = torch.as_tensor(graph["local_sigma"]).float().clamp_min(1e-4).to(device)
    xyz_device = xyz.to(device)
    radii = contract.radii_m
    accepted_valid_authority = output_valid.clone()
    accepted_global_rows_authority = global_rows.clone()
    accepted_scale_authority = tuple(float(value) for value in radii)
    full_scalar_scale_statistics = (
        [
            {
                "overlap_candidate_rows": 0,
                "base_only_fallback_rows": 0,
                "source_ood_fallback_rows": 0,
                "effective_update_rows": 0,
            }
            for _ in radii
        ]
        if full_scalar_residual is not None
        else None
    )
    text_queries: list[str] = []
    text_embeddings = None
    if stream_text:
        if not args.text_embedding_cache:
            raise ValueError("streaming text queries require --text-embedding-cache")
        text_payload, _, _ = load_torch_mapping(
            args.text_embedding_cache,
            expected_sha256=(text_record or {}).get("sha256"),
            map_location="cpu",
            label="streamed text embedding cache",
        )
        available = [str(value) for value in text_payload.get("queries", [])]
        text_queries = [
            value.strip() for value in str(args.stream_text_queries).split(",")
            if value.strip()
        ]
        if preserve_streamed_text_scales and len(set(text_queries)) != len(
            text_queries
        ):
            raise ValueError("multiscale streamed text query IDs must be unique")
        lookup = {name: index for index, name in enumerate(available)}
        missing = [name for name in text_queries if name not in lookup]
        if missing:
            raise ValueError(f"streaming text queries are absent: {missing}")
        text_embeddings = F.normalize(
            torch.as_tensor(text_payload["embeddings"])[
                torch.tensor([lookup[name] for name in text_queries])
            ].float(), dim=-1, eps=1e-8,
        )
        streamed_scores = torch.zeros(
            (
                (len(xyz_global), len(radii), len(text_queries))
                if preserve_streamed_text_scales
                else (len(xyz_global), len(text_queries))
            ),
            dtype=torch.float16,
        )
        router_control_scores = (
            torch.zeros_like(streamed_scores) if query_router_mode else None
        )
        descriptors_by_scale = None
    else:
        descriptors_by_scale = torch.zeros(
            len(global_rows), len(radii), 1536, dtype=torch.float16
        )
    for start in range(0, len(global_rows), semantic_batch_size):
        stop = min(start + semantic_batch_size, len(global_rows))
        cached_shape = (
            (stop - start, len(radii), len(text_queries), 2)
            if query_router_mode
            else (
                (stop - start, len(radii), len(text_queries))
                if preserve_streamed_text_scales
                else (stop - start, len(text_queries))
            )
            if stream_text
            else (stop - start, len(radii), 1536)
        )
        cached = _load_resume_tensor(
            resume_dir,
            phase=semantic_phase,
            start=start,
            stop=stop,
            contract_sha256=resume_contract_sha256,
            expected_shape=cached_shape,
            expected_dtype=torch.float16,
        )
        if cached is not None:
            if stream_text:
                if query_router_mode:
                    assert router_control_scores is not None
                    streamed_scores[global_rows[start:stop]] = cached[..., 0]
                    router_control_scores[global_rows[start:stop]] = cached[..., 1]
                else:
                    streamed_scores[global_rows[start:stop]] = cached
            else:
                assert descriptors_by_scale is not None
                descriptors_by_scale[start:stop] = cached
            continue
        centers_cpu = torch.arange(start, stop)
        batch_streamed_scores = None
        for scale_index, radius in enumerate(radii):
            if isinstance(contract, SurfaceRegionContractV3):
                selections = expand_surface_region_v3_batch_at_radius(
                    contract,
                    support,
                    xyz,
                    centers_cpu,
                    float(radius),
                    prepared_graph=prepared_graph,
                    primary_local=primary_local,
                )
                rows = torch.stack([value.rows for value in selections])
                mask = torch.stack([value.token_mask for value in selections])
                core = torch.stack([value.core_mask for value in selections])
                context = torch.stack([value.context_mask for value in selections])
                support_fill = torch.stack(
                    [value.support_fill_mask for value in selections]
                )
                recovery_distance = torch.stack(
                    [value.recovery_distance for value in selections]
                )
                anchor_local = torch.tensor(
                    [value.anchor_index for value in selections],
                    dtype=torch.long,
                )
            else:
                regions = contract.expand_batch(
                    support, xyz, centers_cpu.tolist(), radius,
                    prepared_graph=prepared_graph,
                )
                batch = len(regions); width = contract.maximum_tokens
                rows = torch.zeros(batch, width, dtype=torch.long)
                mask = torch.zeros(batch, width, dtype=torch.bool)
                core = torch.zeros(batch, width, dtype=torch.bool)
                anchor_local = torch.zeros(batch, dtype=torch.long)
                for offset, (region_rows, region_core, _distance) in enumerate(regions):
                    count = len(region_rows)
                    rows[offset, :count] = region_rows
                    mask[offset, :count] = True
                    core[offset, :count] = region_core
                    anchor_local[offset] = int(
                        torch.where(region_rows == centers_cpu[offset])[0][0]
                    )
                mask = preserve_primary_region_tokens(
                    rows,
                    mask,
                    centers_cpu,
                    primary_local,
                )
                core &= mask
                context = mask & ~core
                support_fill = torch.zeros_like(mask)
                recovery_distance = torch.full(
                    mask.shape,
                    torch.inf,
                    dtype=torch.float32,
                )
            rows, mask, core, context, support_fill, recovery_distance, anchor_local = (
                rows.to(device),
                mask.to(device),
                core.to(device),
                context.to(device),
                support_fill.to(device),
                recovery_distance.to(device),
                anchor_local.to(device),
            )
            token_xyz = xyz_device[rows]
            token_scale = local_scale[rows, None].expand(-1, -1, 3)
            primitive_reliability = reliability[rows, None]
            radio_tokens_raw = radio[rows]
            if anchor_radio is not None:
                radio_tokens_raw = directional_anchor_tokens(
                    radio_tokens_raw, anchor_radio[start:stop], anchor_local
                )
            radio_tokens = surface_region_radio_tokens(
                radio_tokens_raw,
                normalization=radio_feature_normalization,
            )
            if isinstance(contract, SurfaceRegionContractV3):
                token_reliability = surface_region_effective_reliability_v3(
                    primitive_reliability,
                    recovery_distance,
                    float(radius),
                    support_fill_mask=support_fill,
                    token_mask=mask,
                )
                raw_norm = torch.linalg.vector_norm(
                    radio_tokens_raw.float(),
                    dim=-1,
                    keepdim=True,
                )
                geometry = surface_region_geometry_v3(
                    token_xyz,
                    token_scale,
                    token_reliability,
                    float(radius),
                    raw_radio_l2_norm=raw_norm,
                    anchor_index=anchor_local,
                    core_mask=core,
                    context_mask=context,
                    support_fill_mask=support_fill,
                    token_mask=mask,
                )
                radio_tokens = radio_tokens.masked_fill(~mask[..., None], 0.0)
            else:
                token_reliability = primitive_reliability
                geometry = surface_region_geometry_v2(
                    token_xyz, token_scale, token_reliability, float(radius),
                    anchor_index=anchor_local, core_mask=core, token_mask=mask,
                )
            if query_router_mode:
                assert residual_codebook is not None
                assert query_router is not None
                assert router_negative_text is not None
                codebook_output = residual_codebook.forward_codebook(
                    radio_tokens,
                    geometry,
                    token_mask=mask,
                    reliability=token_reliability,
                    anchor_index=anchor_local,
                )
                direct_v2 = readout(
                    radio_tokens,
                    geometry,
                    token_mask=mask,
                    reliability=token_reliability,
                    anchor_index=anchor_local,
                )
                if not torch.equal(codebook_output.canonical_token, direct_v2):
                    raise RuntimeError("residual codebook canonical path changed V2")
                if not torch.equal(codebook_output.slot_tokens[:, 0], direct_v2):
                    raise RuntimeError("residual codebook slot zero changed V2")
                slot_descriptors = project_surface_codebook_slots(
                    head, codebook_output.slot_tokens
                )
                assert text_embeddings is not None
                router_output = query_router(
                    slot_descriptors,
                    codebook_output.slot_tokens,
                    text_embeddings.to(device),
                    router_negative_text,
                    logit_scale=10.0,
                )
                scale_scores = router_output.response.cpu()
                scale_control = router_output.slot_scores[:, 0].cpu()
                if batch_streamed_scores is None:
                    batch_streamed_scores = torch.empty(
                        batch,
                        len(radii),
                        len(text_queries),
                        2,
                        dtype=torch.float32,
                    )
                batch_streamed_scores[:, scale_index, :, 0] = scale_scores
                batch_streamed_scores[:, scale_index, :, 1] = scale_control
                continue
            summary = readout(
                radio_tokens, geometry, token_mask=mask,
                reliability=token_reliability, anchor_index=anchor_local,
            )
            # This is the immutable accepted-V2 e0 authority.  The optional
            # scalar path is deliberately downstream of selection, RADIO,
            # geometry, reliability, readout, and the singleton official head.
            base_descriptor = F.normalize(
                head(summary[:, None])[:, 0].float(), dim=-1
            )
            if full_scalar_residual is not None:
                assert full_scalar_state is not None
                assert full_scalar_normalization is not None
                assert full_scalar_scale_statistics is not None
                assert full_scalar_runtime_carrier is not None
                overlay = apply_accepted_v2_full_scalar_overlay(
                    base_descriptor,
                    residual=full_scalar_residual,
                    exact_state=full_scalar_state,
                    runtime_carrier=full_scalar_runtime_carrier,
                    normalization_authority=full_scalar_normalization,
                    accepted_base_valid=accepted_valid_authority,
                    accepted_global_rows=accepted_global_rows_authority,
                    local_region_rows=rows,
                    token_mask=mask,
                    anchor_index=anchor_local,
                )
                descriptor = overlay.semantic_descriptor.half()
                scale_statistics = full_scalar_scale_statistics[scale_index]
                scale_statistics["overlap_candidate_rows"] += int(
                    overlay.overlap_candidate_mask.sum()
                )
                scale_statistics["base_only_fallback_rows"] += int(
                    overlay.base_only_fallback_mask.sum()
                )
                scale_statistics["source_ood_fallback_rows"] += int(
                    overlay.source_ood_fallback_mask.sum()
                )
                scale_statistics["effective_update_rows"] += int(
                    overlay.effective_update_mask.sum()
                )
            else:
                # Keep the legacy cast at the same point and in the same dtype.
                descriptor = base_descriptor.half()
            if stream_text:
                # Match the warm-cache compiler exactly: descriptors are
                # quantized to fp16 before normalized cosine and scores are
                # finally stored as fp16 primitive unaries.
                assert text_embeddings is not None
                # Deliberately perform this tiny Q-way dot product on CPU.
                # The warm-cache compiler also reloads fp16 descriptors on
                # CPU, so this makes cold/warm unaries bitwise reproducible
                # instead of merely close across CUDA/CPU reduction kernels.
                scale_scores = F.normalize(
                    descriptor.cpu().float(), dim=-1, eps=1e-8
                ) @ text_embeddings.T
                if preserve_streamed_text_scales:
                    if batch_streamed_scores is None:
                        batch_streamed_scores = torch.empty(
                            batch,
                            len(radii),
                            len(text_queries),
                            dtype=torch.float32,
                        )
                    batch_streamed_scores[:, scale_index] = scale_scores
                else:
                    batch_streamed_scores = (
                        scale_scores
                        if batch_streamed_scores is None
                        else torch.maximum(batch_streamed_scores, scale_scores)
                    )
            else:
                assert descriptors_by_scale is not None
                descriptors_by_scale[start:stop, scale_index] = descriptor.cpu()
        if stream_text:
            assert batch_streamed_scores is not None
            committed = batch_streamed_scores.half()
            if query_router_mode:
                assert router_control_scores is not None
                streamed_scores[global_rows[start:stop]] = committed[..., 0]
                router_control_scores[global_rows[start:stop]] = committed[..., 1]
            else:
                streamed_scores[global_rows[start:stop]] = committed
        else:
            assert descriptors_by_scale is not None
            committed = descriptors_by_scale[start:stop].clone()
        _commit_resume_tensor(
            resume_dir,
            phase=semantic_phase,
            start=start,
            stop=stop,
            contract_sha256=resume_contract_sha256,
            value=committed,
        )
        _pace_after_commit(device, pacing_seconds)
    if (
        not torch.equal(output_valid, accepted_valid_authority)
        or not torch.equal(global_rows, accepted_global_rows_authority)
        or tuple(float(value) for value in radii) != accepted_scale_authority
    ):
        raise RuntimeError(
            "full-scalar execution changed accepted validity, row order, or scales"
        )
    full_scalar_metadata = None
    if full_scalar_residual is not None:
        assert full_scalar_state is not None
        assert full_scalar_state_record is not None
        assert full_scalar_normalization is not None
        assert full_scalar_normalization_record is not None
        assert full_scalar_residual_record is not None
        assert full_scalar_residual_payload is not None
        assert full_scalar_certificate is not None
        assert full_scalar_certificate_record is not None
        assert full_scalar_routing is not None
        assert full_scalar_scale_statistics is not None
        overlap_count = int(full_scalar_routing.overlap_mask.sum())
        base_only_count = int(full_scalar_routing.base_only_fallback_mask.sum())
        exact_only_count = int(full_scalar_routing.exact_only_abstain_mask.sum())
        neither_count = int(full_scalar_routing.neither_abstain_mask.sum())
        for scale_statistics in full_scalar_scale_statistics:
            if (
                scale_statistics["overlap_candidate_rows"] != overlap_count
                or scale_statistics["base_only_fallback_rows"] != base_only_count
            ):
                raise RuntimeError(
                    "full-scalar batch routing differs from the global support partition"
                )
        full_scalar_metadata = {
            "schema_version": 1,
            "contract_sha256": SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256,
            "summary_names_sha256": SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
            "placement": "strictly_after_accepted_v2_official_descriptor_e0",
            "base_components_mutated": False,
            "raw_summary_presented_to_residual": True,
            "normalization_authority_used_for_ood_and_model_buffers_only": True,
            "target_factorized_primitive_state": full_scalar_state_record,
            "target_factorized_primitive_state_contract_sha256": (
                full_scalar_state.contract_sha256
            ),
            "normalization_authority": full_scalar_normalization_record,
            "source_state_cohort_sha256": full_scalar_normalization[
                "source_state_cohort_sha256"
            ],
            "residual_checkpoint": full_scalar_residual_record,
            "training_certificate": full_scalar_certificate_record,
            "training_certificate_content_sha256": (
                full_scalar_certificate["content_sha256"]
            ),
            "residual_checkpoint_contract_sha256": (
                full_scalar_residual_payload["contract_sha256"]
            ),
            "residual_model_architecture_sha256": (
                full_scalar_residual_payload["model_architecture_sha256"]
            ),
            "residual_model_state_dict_sha256": (
                full_scalar_residual_payload["model_state_dict_sha256"]
            ),
            "accepted_v2_authority": dict(
                full_scalar_residual_payload["accepted_v2_authority"]
            ),
            "source_authority": dict(
                full_scalar_residual_payload["source_authority"]
            ),
            "support_partition": {
                "overlap_candidate_rows": overlap_count,
                "accepted_base_only_fallback_rows": base_only_count,
                "exact_only_abstain_rows": exact_only_count,
                "neither_abstain_rows": neither_count,
                "accepted_valid_sha256": tensor_sha256(accepted_valid_authority),
                "accepted_global_rows_sha256": tensor_sha256(
                    accepted_global_rows_authority
                ),
                "target_exact_valid_sha256": tensor_sha256(
                    full_scalar_state.valid
                ),
                "candidate_valid_equals_accepted_base_bitwise": True,
                "candidate_global_rows_equal_accepted_base_bitwise": True,
                "exact_only_never_enabled": True,
            },
            "scale_radii_m": list(accepted_scale_authority),
            "per_scale_execution": full_scalar_scale_statistics,
        }
    readout_sha256 = readout_record["sha256"]
    radio_sha256 = radio_record["sha256"]
    metadata = {
        "schema_version": 5, "feature_space": "official_siglip2_summary_descriptor_multiscale",
        "source": (
            "canonical_radio_surface_region_residual_codebook_query_router"
            if query_router_mode
            else "canonical_radio_surface_region_readout"
        ),
        "construction": (
            "surface_residual_codebook_slotwise_official_head_then_query_router"
            if query_router_mode
            else (
                "accepted_v2_official_descriptor_then_full_scalar_residual_v1"
                if full_scalar_metadata is not None
                else "canonical_radio_surface_region_readout_then_official_summary_head"
            )
        ),
        "canonical_radio_source": (
            "field_decode_only"
            if canonical_radio_source == "field_decode"
            else "frozen_mpr_full_1280_teacher"
        ),
        "mpr_radio_features_opened": canonical_radio_source == "mpr_teacher",
        **(
            {
                "capacity_diagnostic_only": True,
                "experiment_registration": registration_record,
            }
            if canonical_radio_source == "mpr_teacher"
            else {}
        ),
        "readout_checkpoint": str(readout_path.resolve()),
        "readout_checkpoint_sha256": readout_sha256,
        "bridge_checkpoint_sha256": readout_sha256,
        "bridge_training_scope": "global_cross_scene",
        "bridge_training_scope_detail": training_scope,
        **(
            {"accepted_v2_full_scalar_overlay": full_scalar_metadata}
            if full_scalar_metadata is not None
            else {}
        ),
        **(
            {
                "residual_codebook_checkpoint": codebook_record["path"],
                "residual_codebook_checkpoint_sha256": codebook_record["sha256"],
                "query_router_checkpoint": router_record["path"],
                "query_router_checkpoint_sha256": router_record["sha256"],
                "generic_negative_text_cache": negative_text_record["path"],
                "generic_negative_text_cache_sha256": negative_text_record[
                    "sha256"
                ],
                "generic_negative_queries": negative_text_record["queries"],
                "query_router_score_contract": (
                    "canonical_negative_bernoulli_query_first"
                ),
                "query_router_logit_scale": 10.0,
                "slot_projection_contract": (
                    "four_independent_official_head_calls_Bx1x1280"
                ),
                "representation_query_set_invariant": True,
                "query_router_query_dependent": True,
                "exact_frozen_v2_slot0_control": True,
                "query_router_deployment_gauge": router_deployment_gauge,
                "mixed_gauge_transfer_diagnostic": (
                    radio_feature_normalization == "legacy_raw"
                ),
                **(
                    {
                        "query_router_gauge_authority": (
                            router_gauge_authority_record
                        )
                    }
                    if router_gauge_authority_record is not None
                    else {}
                ),
                **(
                    {"experiment_registration": registration_record}
                    if registration_record is not None
                    else {}
                ),
            }
            if query_router_mode
            else {}
        ),
        "field_checkpoint": field_record["path"],
        "field_checkpoint_sha256": field_record["sha256"],
        **(
            {
                "field_checkpoint_schema": field_schema,
                "factorized_primitive_state": factorized_state_record["path"],
                "factorized_primitive_state_sha256": factorized_state_record[
                    "sha256"
                ],
                "factorized_primitive_state_contract_sha256": (
                    factorized_state.contract_sha256
                ),
                "factorized_primitive_state_schema": factorized_state.schema,
                "factorized_primitive_state_schema_version": (
                    factorized_state.schema_version
                ),
                "factorized_radio_cache_sha256": factorized_support.cache.sha256,
                "factorized_radio_field_signature_sha256": (
                    factorized_support.field_signature.digest
                ),
                "factorized_state_readout_policy": (
                    "legacy_geometric_reliability_only_no_residual_v1"
                ),
                "factorized_state_full_scalar_encoder_used": False,
                "visibility_purity_measurement_available": bool(
                    factorized_state.visibility_purity_known.all()
                ),
                "visibility_purity_known_count": int(
                    factorized_state.visibility_purity_known.sum()
                ),
                "visibility_purity_unknown_count": int(
                    (~factorized_state.visibility_purity_known).sum()
                ),
                "visibility_purity_authority": dict(
                    factorized_state.metadata["visibility_purity_authority"]
                ),
            }
            if factorized_state_record is not None
            else {}
        ),
        **(
            {
                "directional_context_field_checkpoint": context_field_record[
                    "path"
                ],
                "directional_context_field_checkpoint_sha256": (
                    context_field_record["sha256"]
                ),
                "directional_readout_policy": (
                    "anchor_mode_with_frozen_canonical_neighborhood_context"
                ),
            }
            if context_field_record is not None
            else {}
        ),
        "mpr_cache": mpr_record["path"],
        "mpr_cache_sha256": mpr_record["sha256"],
        "field_geometry_xyz_sha256": field_payload.get(
            "geometry_fingerprint", {}
        ).get("xyz_sha256"),
        "support_graph": str(graph_path.resolve()),
        "support_graph_sha256": graph_record["sha256"],
        "official_radio_checkpoint_sha256": radio_sha256,
        "radio_checkpoint_sha256": radio_sha256,
        "region_radii_m": list(radii), "region_topology": contract.expansion,
        "readout_batch_size": int(args.semantic_batch_size),
        "region_contract": contract.to_dict(),
        "region_contract_version": contract.version,
        "region_contract_sha256": contract.digest,
        "radio_feature_normalization": radio_feature_normalization,
        "radio_feature_normalization_authority": (
            "frozen_surface_region_contract"
            if radio_feature_normalization in {
                "l2_direction",
                "l2_direction_plus_log_raw_norm_v1",
            }
            else "explicit_legacy_reproduction_override"
        ),
        "query_set_invariant": not query_router_mode,
        "benchmark_images_opened": False,
        "official_summary_head": True, "custom_text_projection": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": query_router_mode,
        "cache_role": "disposable_derivative_not_scene_memory",
        "row_storage": "sparse_valid_rows_with_global_row_index",
        "scale_storage": "all_scales_preserved; mean_descriptor_legacy_only",
        "resume_contract_sha256": resume_contract_sha256,
        "thermal_pacing_seconds_per_batch": pacing_seconds,
        "completion_context_policy": (
            "primary_induced_graph_or_all_valid_fallback_v1"
            if isinstance(contract, SurfaceRegionContractV3)
            and primary_valid is not None
            else "all_valid_v3"
            if isinstance(contract, SurfaceRegionContractV3)
            else "primary_plus_center"
            if primary_valid is not None
            else "all_valid"
        ),
        **(
            {
                "minimum_token_policy": contract.minimum_token_policy,
                "support_fill_semantics": contract.support_fill_semantics,
                "support_fill_reliability": (
                    "primitive_reliability_times_exp_negative_recovery_"
                    "distance_over_radius_v1"
                ),
                "selection_eligibility": (
                    "eligible_induced_traversal_primary_or_all_valid_fallback_v1"
                ),
            }
            if isinstance(contract, SurfaceRegionContractV3)
            else {}
        ),
        "primary_valid_count": (
            int(primary_valid.sum()) if primary_valid is not None else None
        ),
        "semantic_confidence": (
            {
                "source": "canonical_radio_reconstruction_fidelity",
                "nonzero_count": int((semantic_confidence > 0).sum()),
                "mean_valid": float(
                    semantic_confidence[output_valid].float().mean()
                ),
            }
            if semantic_confidence is not None
            else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if router_control_output is not None:
        router_control_output.parent.mkdir(parents=True, exist_ok=True)
    if stream_text:
        if preserve_streamed_text_scales:
            # The frozen LERF Direct3D protocol consumes all three raw scales
            # and performs its own fixed KNN10/peak-scale readout.  Completion
            # or scale reduction here would change that protocol.
            streamed_scores[~output_valid] = 0
            if router_control_scores is not None:
                router_control_scores[~output_valid] = 0
            completion = {
                "applied": False,
                "reason": "frozen_direct3d_requires_raw_unreduced_scale_scores",
            }
        else:
            streamed_scores, completion = apply_completion_evidence(
                streamed_scores,
                output_valid,
                semantic_confidence=semantic_confidence,
                primary_valid=primary_valid,
                routing=str(
                    getattr(args, "completion_routing", "primary_first")
                ),
                primary_support_threshold=float(
                    getattr(args, "primary_support_threshold", 0.5)
                ),
                primary_support_mode=str(
                    getattr(args, "primary_support_mode", "relative_peak")
                ),
                primary_support_margin=float(
                    getattr(args, "primary_support_margin", 0.02)
                ),
            )
        score_metadata = {
            "schema_version": (
                4
                if query_router_mode
                else 3
                if preserve_streamed_text_scales
                else 2
            ),
            "feature_space": (
                "primitive_canonical_negative_probability_multiscale_unreduced"
                if query_router_mode
                else "primitive_text_query_scores_multiscale_unreduced"
                if preserve_streamed_text_scales
                else "primitive_text_query_scores"
            ),
            "construction": (
                "surface_residual_codebook_slotwise_head_then_query_router"
                if query_router_mode
                else "cold_streaming_surface_region_readout_then_independent_cosine"
                if preserve_streamed_text_scales
                else "cold_streaming_surface_region_readout_then_cosine_max"
            ),
            "scoring": (
                "canonical_negative_bernoulli_query_router_v1"
                if query_router_mode
                else "raw_independent_normalized_cosine"
                if preserve_streamed_text_scales
                else "cosine"
            ),
            **(
                {
                    "score_semantics": "canonical_negative_bernoulli_probability",
                    "value_range": [0.0, 1.0],
                    "logit_scale": 10.0,
                    "generic_negative_queries": negative_text_record["queries"],
                    "probability_route": "query_router_v1",
                }
                if query_router_mode
                else {}
            ),
            "scale_aggregation": (
                "none_frozen_downstream_only"
                if preserve_streamed_text_scales
                else "max"
            ),
            "scale_count": len(radii),
            **(
                {"scale_radii_m": list(radii)}
                if preserve_streamed_text_scales
                else {}
            ),
            "score_chunk_size": int(args.semantic_batch_size),
            "query_names": text_queries,
            "text_embedding_cache": str(Path(args.text_embedding_cache).resolve()),
            **(
                {
                    "text_embedding_cache_sha256": (text_record or {}).get(
                        "sha256"
                    ),
                    "streaming_implementation": file_record(
                        Path(__file__).resolve()
                    ),
                }
                if preserve_streamed_text_scales
                else {}
            ),
            "semantic_cache_materialized": False,
            "completion": completion,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": True,
            "semantic_provenance": metadata,
        }
        score_payload = {
            "xyz": xyz_global,
            "features": streamed_scores,
            "valid": output_valid,
            "metadata": score_metadata,
        }
        if primary_valid is not None:
            score_payload["primary_valid"] = primary_valid
        if semantic_confidence is not None:
            score_payload["semantic_confidence"] = semantic_confidence
        control_report = None
        if query_router_mode:
            assert router_control_output is not None
            assert router_control_scores is not None
            if (
                bool((streamed_scores.float() < 0.0).any())
                or bool((streamed_scores.float() > 1.0).any())
                or bool((router_control_scores.float() < 0.0).any())
                or bool((router_control_scores.float() > 1.0).any())
            ):
                raise RuntimeError("query-router Bernoulli responses left [0,1]")
        _atomic_torch_save(score_payload, output)
        if query_router_mode:
            assert router_control_output is not None
            assert router_control_scores is not None
            control_metadata = dict(score_metadata)
            control_metadata.update(
                {
                    "construction": (
                        "surface_residual_codebook_exact_frozen_v2_slot0"
                    ),
                    "scoring": (
                        "canonical_negative_bernoulli_frozen_v2_slot0"
                    ),
                    "probability_route": "exact_frozen_v2_slot0_control",
                    "paired_candidate_output": str(output),
                }
            )
            control_payload = {
                "xyz": xyz_global,
                "features": router_control_scores,
                "valid": output_valid,
                "metadata": control_metadata,
            }
            if primary_valid is not None:
                control_payload["primary_valid"] = primary_valid
            if semantic_confidence is not None:
                control_payload["semantic_confidence"] = semantic_confidence
            _atomic_torch_save(control_payload, router_control_output)
            control_report = {
                "output": str(router_control_output),
                "valid_primitives": int(output_valid.sum()),
                "total_primitives": len(output_valid),
                "num_queries": len(text_queries),
                "semantic_cache_materialized": False,
                "metadata": control_metadata,
            }
            write_frozen_json(
                router_control_output.with_suffix(
                    router_control_output.suffix + ".json"
                ),
                control_report,
            )
        report = {
            "output": str(output.resolve()),
            "valid_primitives": int(output_valid.sum()),
            "total_primitives": len(output_valid),
            "num_queries": len(text_queries),
            "semantic_cache_materialized": False,
            "metadata": score_metadata,
            **(
                {"paired_exact_control": control_report}
                if control_report is not None
                else {}
            ),
        }
        write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
        return report
    assert descriptors_by_scale is not None
    descriptors = F.normalize(descriptors_by_scale.float().mean(1), dim=-1).half()
    # Semantic descriptors dominate cache size (1536 fp16 values per row).  Do
    # not materialize zero descriptors for invalid/background primitives.  The
    # global geometry and explicit row index retain an exact, lossless mapping;
    # consumers expand only when their downstream score representation needs it.
    semantic_payload = {
        "xyz": xyz_global,
        "features": descriptors,
        "summary_features": descriptors,
        "global_rows": global_rows,
        "features_by_scale": descriptors_by_scale,
        "valid": output_valid,
        "metadata": metadata,
    }
    if primary_valid is not None:
        semantic_payload["primary_valid"] = primary_valid
    if semantic_confidence is not None:
        semantic_payload["semantic_confidence"] = semantic_confidence
    _atomic_torch_save(semantic_payload, output)
    # Pose-free image querying only consumes the already aggregated descriptor,
    # never the retained per-scale tensor.  Save an exact, provenance-identical
    # derivative alongside the full cache so each query does not repeatedly
    # deserialize several otherwise unused gigabytes.  This is an execution
    # representation change only: ``features`` is byte-for-byte the tensor
    # stored in the full semantic cache.
    query_output = (
        Path(args.query_output)
        if str(args.query_output).strip()
        else output.with_name(f"{output.stem}_query{output.suffix}")
    )
    query_payload = {
        "xyz": xyz_global,
        "features": descriptors,
        "global_rows": global_rows,
        "valid": output_valid,
        "metadata": metadata,
    }
    if primary_valid is not None:
        query_payload["primary_valid"] = primary_valid
    if semantic_confidence is not None:
        query_payload["semantic_confidence"] = semantic_confidence
    _atomic_torch_save(query_payload, query_output)
    report = {"output": str(output.resolve()), "valid_primitives": int(output_valid.sum()),
              "total_primitives": len(output_valid), "metadata": metadata}
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    write_frozen_json(
        output.with_suffix(output.suffix + ".provenance.json"),
        {"cache": str(output.resolve()), "inputs": metadata},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument(
        "--field-checkpoint-schema",
        choices=(
            CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1,
            FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2,
        ),
        default=CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1,
        help=(
            "Explicit field schema; factorized-v2 has no legacy fallback and "
            "requires a SHA-bound primitive-state sidecar."
        ),
    )
    parser.add_argument("--factorized-primitive-state", default="")
    parser.add_argument("--factorized-primitive-state-sha256", default="")
    parser.add_argument(
        "--accepted-v2-full-scalar-state",
        default="",
        help=(
            "Independent exact-marginal primitive-state sidecar used only "
            "after the accepted V2 descriptor e0 has been produced."
        ),
    )
    parser.add_argument("--accepted-v2-full-scalar-state-sha256", default="")
    parser.add_argument("--full-scalar-normalization-authority", default="")
    parser.add_argument(
        "--full-scalar-normalization-authority-sha256", default=""
    )
    parser.add_argument("--full-scalar-residual-checkpoint", default="")
    parser.add_argument("--full-scalar-residual-checkpoint-sha256", default="")
    parser.add_argument("--full-scalar-training-certificate", default="")
    parser.add_argument("--full-scalar-training-certificate-sha256", default="")
    parser.add_argument(
        "--canonical-radio-source",
        choices=("field_decode", "mpr_teacher"),
        default="field_decode",
        help=(
            "Diagnostic source for the 1280-D RADIO rows. The default preserves "
            "the compact canonical-field path; mpr_teacher is a full-capacity "
            "label-free teacher upper-bound and must not be reported as the field."
        ),
    )
    parser.add_argument(
        "--experiment-registration",
        default="",
        help=(
            "Immutable preregistration receipt. Required for the label-free "
            "mpr_teacher full-capacity diagnostic."
        ),
    )
    parser.add_argument("--field-checkpoint-sha256", default="")
    parser.add_argument(
        "--directional-context-field-checkpoint",
        default="",
        help=(
            "Optional frozen canonical context field. When set, the primary "
            "field supplies only each region anchor while neighboring tokens "
            "come from this context field."
        ),
    )
    parser.add_argument(
        "--directional-context-field-checkpoint-sha256", default=""
    )
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--support-graph-sha256", default="")
    parser.add_argument("--readout-checkpoint", required=True)
    parser.add_argument("--readout-checkpoint-sha256", default="")
    parser.add_argument(
        "--readout-legacy-radio-authority",
        default="",
        help=(
            "Narrow external authority for an immutable legacy V2 readout "
            "whose train/validation cache paths are present but whose RADIO "
            "SHA was omitted from the summarized checkpoint provenance."
        ),
    )
    parser.add_argument("--readout-legacy-radio-authority-sha256", default="")
    parser.add_argument("--residual-codebook-checkpoint", default="")
    parser.add_argument("--residual-codebook-checkpoint-sha256", default="")
    parser.add_argument("--query-router-checkpoint", default="")
    parser.add_argument("--query-router-checkpoint-sha256", default="")
    parser.add_argument("--query-router-gauge-authority", default="")
    parser.add_argument("--query-router-gauge-authority-sha256", default="")
    parser.add_argument("--generic-negative-text-cache", default="")
    parser.add_argument("--generic-negative-text-cache-sha256", default="")
    parser.add_argument(
        "--router-control-output",
        default="",
        help=(
            "Paired exact frozen-V2 slot-zero probability output. Required "
            "with the residual codebook/query router so attribution does not "
            "confound the new region contract with router attention."
        ),
    )
    parser.add_argument("--mpr-cache-sha256", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--query-output", default="",
        help=(
            "Optional compact descriptor-only sidecar for pose-free querying; "
            "defaults next to --output."
        ),
    )
    parser.add_argument("--region-radii", default="")
    parser.add_argument("--graph-neighbors", type=int, default=16)
    parser.add_argument("--radio-batch-size", type=int, default=4096)
    parser.add_argument("--semantic-batch-size", type=int, default=256)
    parser.add_argument("--resume-dir", default="")
    parser.add_argument(
        "--radio-feature-normalization",
        choices=(
            "l2_direction",
            "l2_direction_plus_log_raw_norm_v1",
            "legacy_raw",
        ),
        default="legacy_raw",
        help=(
            "Gauge presented to the frozen SurfaceRegion readout. legacy_raw "
            "is the historical V2 override; V3 requires the explicit "
            "direction-plus-log-raw-norm contract."
        ),
    )
    parser.add_argument(
        "--thermal-pacing-seconds-per-batch",
        type=float,
        default=1.0,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-embedding-cache", default="")
    parser.add_argument(
        "--completion-routing",
        choices=("primary_first", "direct", "primary_only"),
        default="primary_first",
    )
    parser.add_argument(
        "--primary-support-mode",
        choices=("absolute", "relative_peak"),
        default="relative_peak",
    )
    parser.add_argument("--primary-support-threshold", type=float, default=0.5)
    parser.add_argument("--primary-support-margin", type=float, default=0.02)
    parser.add_argument(
        "--stream-text-queries", default="",
        help=(
            "Optional ordered comma-separated queries. When set, execute the "
            "readout and cosine scoring as a cold stream and save only scalar "
            "primitive unaries, never a 1536D semantic cache."
        ),
    )
    parser.add_argument(
        "--preserve-streamed-text-scales",
        action="store_true",
        help=(
            "Keep raw [primitive,3,query] cosine scores for the frozen LERF "
            "Direct3D protocol. This disables both scale reduction and "
            "completion in the streamed derivative."
        ),
    )
    parser.add_argument("--radio-checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    parser.add_argument("--radio-checkpoint-sha256", default="")
    args = parser.parse_args(); print(json.dumps(build(args), indent=2))


if __name__ == "__main__": main()
