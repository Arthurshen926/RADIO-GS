"""Sparse, candidate-audited context overlay for immutable AcceptedV2 regions.

This module does not define another surface-region base readout.  The accepted
V2 descriptor remains an external immutable authority.  The only semantic
carrier produced here is a deterministic reliability-weighted pool of V4
``context_mask`` RADIO directions.  V4 core and support-fill tokens are never
part of that carrier.

The frozen V4 implementation stops after ``token_candidate_limit`` settled
nodes and does not expose why the search stopped.  The helper below probes one
additional settled node.  A row is ``search_complete`` exactly when no
additional eligible semantic node exists inside the declared context radius.
Rows which hit the candidate cap may be audited, but their pooled carrier is
required to remain exact zero and can never authorize a residual.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

import numpy as np
import torch

from radio_gs.interfaces.surface_region_contract import (
    PreparedSurfaceRegionGraphV3,
    SurfaceRegionContractV4,
    _bounded_dijkstra_eligible_batch,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


TYPED_CONTEXT_AUTHORITY_SCHEMA = (
    "radio_gs.surface_region_accepted_v2_typed_context_authority.v1"
)
TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION = 1
TYPED_CONTEXT_FEATURE_DIM = 1280
TYPED_CONTEXT_STATISTIC_NAMES = (
    "log_physical_radius_m",
    "log1p_context_token_count",
    "context_fraction_of_selected_semantic_tokens",
    "context_reliability_mean",
    "context_reliability_population_std",
    "context_geodesic_over_radius_mean",
    "context_geodesic_over_radius_population_std",
    "context_weighted_directional_resultant_length",
    "context_log_raw_radio_norm_mean",
    "context_log_raw_radio_norm_population_std",
    "context_to_anchor_radio_cosine_mean",
    "context_to_anchor_radio_cosine_population_std",
)
TYPED_CONTEXT_STATISTIC_DIM = len(TYPED_CONTEXT_STATISTIC_NAMES)
TYPED_CONTEXT_STATISTIC_NAMES_SHA256 = canonical_json_sha256(
    list(TYPED_CONTEXT_STATISTIC_NAMES)
)
TERMINATION_COMPLETE = "complete_within_context_limit"
TERMINATION_CANDIDATE_CAP = "candidate_cap_reached"
TYPED_CONTEXT_TERMINATIONS = (
    TERMINATION_COMPLETE,
    TERMINATION_CANDIDATE_CAP,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCANNET_SCENE = re.compile(r"^(scene\d{4})_\d{2}$")


def typed_context_overlay_contract() -> dict[str, Any]:
    """Return the immutable Stage-A typed-context overlay contract."""

    v4 = SurfaceRegionContractV4()
    return {
        "schema": TYPED_CONTEXT_AUTHORITY_SCHEMA,
        "schema_version": TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION,
        "base": "external_immutable_accepted_v2_descriptor_not_copied",
        "context_selection": {
            "contract": v4.to_dict(),
            "contract_sha256": v4.digest,
            "semantic_type": "v4_context_mask_only",
            "v4_core_allowed": False,
            "support_fill_allowed": False,
            "candidate_completion_probe": (
                "settle_token_candidate_limit_plus_one_then_disable_cap_hit_v1"
            ),
            "cap_hit_action": "audit_rows_but_zero_carrier_and_disable_residual",
        },
        "carrier": {
            "name": "pooled_context_radio_direction",
            "dimension": TYPED_CONTEXT_FEATURE_DIM,
            "storage_dtype": "float16",
            "pool_accumulation_dtype": "float32",
            "pool": "legacy_reliability_weighted_spherical_mean_v1",
            "official_summary_head_applied": False,
            "invalid_value": "exact_all_zero",
        },
        "statistics": {
            "dimension": TYPED_CONTEXT_STATISTIC_DIM,
            "names": list(TYPED_CONTEXT_STATISTIC_NAMES),
            "names_sha256": TYPED_CONTEXT_STATISTIC_NAMES_SHA256,
            "storage_dtype": "float32",
            "dispersion": "population_standard_deviation",
            "invalid_value": "exact_all_zero",
        },
        "sparse_row_audit": {
            "storage": "csr_offsets_plus_local_and_global_context_rows",
            "dense_region_by_token_feature_tensor_allowed": False,
        },
        "validity": (
            "search_complete_and_context_present_and_positive_reliability_"
            "weight_and_positive_directional_resultant"
        ),
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
    }


TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256 = canonical_json_sha256(
    typed_context_overlay_contract()
)


@dataclass(frozen=True)
class CandidateCompleteTypedSelection:
    """One bounded V4 typed selection plus an explicit completion result."""

    rows: torch.Tensor
    core_mask: torch.Tensor
    context_mask: torch.Tensor
    semantic_geodesic_distance: torch.Tensor
    candidate_probe_count: int
    search_complete: bool
    termination: str

    def __post_init__(self) -> None:
        rows = torch.as_tensor(self.rows).detach().long().cpu().reshape(-1).clone()
        core = torch.as_tensor(self.core_mask).detach().bool().cpu().reshape(-1).clone()
        context = (
            torch.as_tensor(self.context_mask).detach().bool().cpu().reshape(-1).clone()
        )
        distances = (
            torch.as_tensor(self.semantic_geodesic_distance)
            .detach()
            .float()
            .cpu()
            .reshape(-1)
            .clone()
        )
        if (
            rows.numel() <= 0
            or core.shape != rows.shape
            or context.shape != rows.shape
            or distances.shape != rows.shape
            or not bool((core ^ context).all())
            or not bool(torch.isfinite(distances).all())
            or bool((distances < 0).any())
            or int(torch.unique(rows).numel()) != rows.numel()
        ):
            raise ValueError("typed-context candidate selection tensors differ")
        probe_count = int(self.candidate_probe_count)
        complete = bool(self.search_complete)
        termination = str(self.termination)
        if probe_count < rows.numel() or termination not in TYPED_CONTEXT_TERMINATIONS:
            raise ValueError("typed-context candidate termination differs")
        if complete != (termination == TERMINATION_COMPLETE):
            raise ValueError("typed-context search-complete flag differs")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "core_mask", core)
        object.__setattr__(self, "context_mask", context)
        object.__setattr__(self, "semantic_geodesic_distance", distances)
        object.__setattr__(self, "candidate_probe_count", probe_count)
        object.__setattr__(self, "search_complete", complete)
        object.__setattr__(self, "termination", termination)


def candidate_complete_typed_context_batch(
    contract: SurfaceRegionContractV4,
    prepared_graph: PreparedSurfaceRegionGraphV3,
    anchors: Sequence[int] | torch.Tensor,
    radius_m: float,
    *,
    selection_eligibility: torch.Tensor | np.ndarray | None = None,
) -> list[CandidateCompleteTypedSelection]:
    """Return V4 typed selections with an exact candidate-cap audit.

    The first ``token_candidate_limit`` settled candidates are passed to the
    frozen V4 selector unchanged.  One additional settlement slot is used only
    to determine whether V4 omitted an eligible node inside ``context_ratio *
    radius``.  It never enters the returned typed selection.
    """

    if not isinstance(contract, SurfaceRegionContractV4):
        raise TypeError("typed-context selection requires SurfaceRegionContractV4")
    if not isinstance(prepared_graph, PreparedSurfaceRegionGraphV3):
        raise TypeError("typed-context selection requires a prepared V4 graph")
    if prepared_graph.contract_sha256 != contract.digest:
        raise ValueError("typed-context prepared graph belongs to another contract")
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("typed-context radius must be positive and finite")
    anchor_values = torch.as_tensor(anchors).detach().long().cpu().reshape(-1)
    if anchor_values.numel() == 0:
        return []
    if bool((anchor_values < 0).any()) or bool(
        (anchor_values >= prepared_graph.num_nodes).any()
    ):
        raise ValueError("typed-context anchor is outside prepared graph")
    base_eligibility = contract._base_eligibility(
        selection_eligibility, prepared_graph.num_nodes
    )
    probe_width = min(
        int(contract.token_candidate_limit) + 1,
        int(prepared_graph.num_nodes),
    )
    semantic = prepared_graph.semantic_csr
    probe_rows, probe_distances, probe_counts = _bounded_dijkstra_eligible_batch(
        semantic.indptr.astype(np.int64, copy=False),
        semantic.indices.astype(np.int64, copy=False),
        semantic.data.astype(np.float64, copy=False),
        anchor_values.numpy().astype(np.int64, copy=False),
        base_eligibility,
        radius * float(contract.context_ratio),
        probe_width,
    )
    results: list[CandidateCompleteTypedSelection] = []
    cap = int(contract.token_candidate_limit)
    for batch_index, anchor in enumerate(anchor_values.tolist()):
        probe_count = int(probe_counts[batch_index])
        if probe_count <= 0 or int(probe_rows[batch_index, 0]) != int(anchor):
            raise RuntimeError("typed-context Dijkstra lost its anchor")
        search_complete = probe_count <= cap
        candidate_count = min(probe_count, cap)
        candidate_rows = probe_rows[batch_index, :candidate_count]
        candidate_distances = probe_distances[batch_index, :candidate_count]
        selected_rows, selected_distances = contract._select_semantic_candidates(
            candidate_rows,
            candidate_distances,
            int(anchor),
            radius,
        )
        selected_distances_f32 = np.asarray(
            selected_distances, dtype=np.float32
        )
        core = selected_distances_f32 <= radius + 1e-7
        results.append(
            CandidateCompleteTypedSelection(
                rows=torch.from_numpy(
                    np.asarray(selected_rows, dtype=np.int64).copy()
                ),
                core_mask=torch.from_numpy(core.copy()),
                context_mask=torch.from_numpy((~core).copy()),
                semantic_geodesic_distance=torch.from_numpy(
                    selected_distances_f32.copy()
                ),
                candidate_probe_count=probe_count,
                search_complete=search_complete,
                termination=(
                    TERMINATION_COMPLETE
                    if search_complete
                    else TERMINATION_CANDIDATE_CAP
                ),
            )
        )
    return results


def candidate_complete_typed_context(
    contract: SurfaceRegionContractV4,
    prepared_graph: PreparedSurfaceRegionGraphV3,
    anchor: int,
    radius_m: float,
    *,
    selection_eligibility: torch.Tensor | np.ndarray | None = None,
) -> CandidateCompleteTypedSelection:
    """Single-row wrapper exactly matching the batch helper."""

    return candidate_complete_typed_context_batch(
        contract,
        prepared_graph,
        [int(anchor)],
        radius_m,
        selection_eligibility=selection_eligibility,
    )[0]


@dataclass(frozen=True)
class PooledTypedContext:
    direction: torch.Tensor
    statistics: torch.Tensor
    context_present: bool
    pool_valid: bool


def _population_std(values: torch.Tensor) -> torch.Tensor:
    return values.float().std(unbiased=False) if values.numel() > 1 else values.new_zeros(())


def pool_typed_context_radio(
    raw_context_radio: torch.Tensor,
    reliability: torch.Tensor,
    semantic_geodesic_distance: torch.Tensor,
    *,
    raw_anchor_radio: torch.Tensor,
    radius_m: float,
    selected_semantic_token_count: int,
    search_complete: bool,
    context_ratio: float = 1.20,
) -> PooledTypedContext:
    """Build the fixed 1280-D carrier and its 12-D typed statistics."""

    values = torch.as_tensor(raw_context_radio).detach().float().cpu()
    weights = torch.as_tensor(reliability).detach().float().cpu().reshape(-1)
    distances = (
        torch.as_tensor(semantic_geodesic_distance)
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    anchor = torch.as_tensor(raw_anchor_radio).detach().float().cpu().reshape(-1)
    count = int(values.shape[0]) if values.ndim == 2 else -1
    radius = float(radius_m)
    semantic_count = int(selected_semantic_token_count)
    if (
        values.ndim != 2
        or values.shape[1] != TYPED_CONTEXT_FEATURE_DIM
        or weights.shape != (count,)
        or distances.shape != (count,)
        or anchor.shape != (TYPED_CONTEXT_FEATURE_DIM,)
        or not math.isfinite(radius)
        or radius <= 0
        or semantic_count < count
        or semantic_count <= 0
    ):
        raise ValueError("typed-context pooling inputs do not align")
    if (
        not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(weights).all())
        or not bool(torch.isfinite(distances).all())
        or not bool(torch.isfinite(anchor).all())
        or bool((weights < 0).any())
        or bool((weights > 1).any())
    ):
        raise ValueError("typed-context pooling inputs must be finite and valid")
    zero_direction = torch.zeros(TYPED_CONTEXT_FEATURE_DIM, dtype=torch.float16)
    zero_statistics = torch.zeros(TYPED_CONTEXT_STATISTIC_DIM, dtype=torch.float32)
    present = count > 0
    if not present:
        return PooledTypedContext(zero_direction, zero_statistics, False, False)
    context_norm = torch.linalg.vector_norm(values, dim=-1)
    anchor_norm = torch.linalg.vector_norm(anchor)
    if bool((context_norm <= 0).any()) or float(anchor_norm) <= 0:
        raise ValueError("typed-context raw RADIO vectors must be nonzero")
    normalized_distance = distances / radius
    if bool((distances <= radius + 1e-7).any()) or bool(
        (distances > radius * float(context_ratio) + 1e-5).any()
    ):
        raise ValueError("typed-context rows are outside the declared shell")
    if not bool(search_complete):
        return PooledTypedContext(zero_direction, zero_statistics, True, False)
    weight_sum = weights.sum()
    if float(weight_sum) <= 0:
        return PooledTypedContext(zero_direction, zero_statistics, True, False)
    directions = values / context_norm[:, None]
    weighted_mean = (directions * weights[:, None]).sum(dim=0) / weight_sum
    resultant = torch.linalg.vector_norm(weighted_mean)
    if not bool(torch.isfinite(resultant)) or float(resultant) <= 1e-12:
        return PooledTypedContext(zero_direction, zero_statistics, True, False)
    pooled = (weighted_mean / resultant).half()
    anchor_direction = anchor / anchor_norm
    anchor_cosine = directions @ anchor_direction
    log_norm = torch.log(context_norm)
    statistics = torch.stack(
        (
            torch.tensor(math.log(radius), dtype=torch.float32),
            torch.tensor(math.log1p(count), dtype=torch.float32),
            torch.tensor(count / semantic_count, dtype=torch.float32),
            weights.mean(),
            _population_std(weights),
            normalized_distance.mean(),
            _population_std(normalized_distance),
            resultant,
            log_norm.mean(),
            _population_std(log_norm),
            anchor_cosine.mean(),
            _population_std(anchor_cosine),
        )
    ).float()
    if statistics.shape != (TYPED_CONTEXT_STATISTIC_DIM,) or not bool(
        torch.isfinite(statistics).all()
    ):
        raise RuntimeError("typed-context statistics are non-finite")
    return PooledTypedContext(pooled, statistics, True, True)


def _tensor_channel_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def typed_context_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    tensor_names = (
        "canonical_region_indices",
        "scale_indices",
        "anchor_local_rows",
        "anchor_global_rows",
        "pooled_context_radio_direction",
        "typed_context_statistics",
        "context_present",
        "candidate_search_complete",
        "typed_context_valid",
        "candidate_probe_count",
        "context_token_count",
        "context_token_row_offsets",
        "context_token_local_rows",
        "context_token_global_rows",
    )
    return {
        **{name: _tensor_channel_sha256(value[name]) for name in tensor_names},
        "region_row_ids": canonical_json_sha256(value["region_row_ids"]),
        "candidate_termination": canonical_json_sha256(
            value["candidate_termination"]
        ),
    }


def _validate_sha(value: object, *, label: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _validate_file_record_shape(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value.get("path", ""))
    if not path:
        raise ValueError(f"{label} file path is empty")
    return {"path": path, "sha256": _validate_sha(value["sha256"], label=label)}


def typed_context_source_access() -> dict[str, bool]:
    return {
        "source_rgb_used": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
    }


def validate_typed_context_authority(value: object) -> dict[str, Any]:
    """Fail closed on any carrier, sparse-row, lineage, or padding drift."""

    if not isinstance(value, Mapping):
        raise ValueError("typed-context authority must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "scene_id",
        "physical_space_id",
        "producer",
        "input_authority",
        "region_row_ids",
        "canonical_region_indices",
        "scale_indices",
        "anchor_local_rows",
        "anchor_global_rows",
        "pooled_context_radio_direction",
        "typed_context_statistics",
        "context_present",
        "candidate_search_complete",
        "typed_context_valid",
        "candidate_termination",
        "candidate_probe_count",
        "context_token_count",
        "context_token_row_offsets",
        "context_token_local_rows",
        "context_token_global_rows",
        "channel_sha256",
        "source_access",
    }
    if set(payload) != required or "accepted_v2_e0" in payload:
        raise ValueError("typed-context authority fields differ")
    contract = typed_context_overlay_contract()
    scene = str(payload.get("scene_id", ""))
    matched = _SCANNET_SCENE.fullmatch(scene)
    if (
        payload.get("schema") != TYPED_CONTEXT_AUTHORITY_SCHEMA
        or payload.get("schema_version") != TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256") != TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256
        or matched is None
        or payload.get("physical_space_id") != matched.group(1)
        or payload.get("source_access") != typed_context_source_access()
        or any(payload.get("source_access", {}).values())
    ):
        raise ValueError("typed-context authority contract or access differs")
    payload["producer"] = _validate_file_record_shape(
        payload.get("producer"), label="typed-context producer"
    )
    inputs = payload.get("input_authority")
    input_keys = {
        "accepted_v2_canonical_region_authority",
        "accepted_region_channel_sha256",
        "accepted_region_fingerprints_sha256",
        "factorized_field_checkpoint",
        "factorized_primitive_state",
        "factorized_radio_cache_sha256",
        "support_graph",
        "primitive_row_authority_sha256",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != input_keys:
        raise ValueError("typed-context input authority differs")
    frozen_inputs = dict(inputs)
    for name in (
        "accepted_v2_canonical_region_authority",
        "factorized_field_checkpoint",
        "factorized_primitive_state",
        "support_graph",
    ):
        frozen_inputs[name] = _validate_file_record_shape(
            frozen_inputs[name], label=f"typed-context {name}"
        )
    for name in (
        "accepted_region_channel_sha256",
        "accepted_region_fingerprints_sha256",
        "factorized_radio_cache_sha256",
        "primitive_row_authority_sha256",
    ):
        frozen_inputs[name] = _validate_sha(
            frozen_inputs[name], label=f"typed-context {name}"
        )
    payload["input_authority"] = frozen_inputs

    canonical = payload.get("canonical_region_indices")
    scales = payload.get("scale_indices")
    anchor_local = payload.get("anchor_local_rows")
    anchor_global = payload.get("anchor_global_rows")
    direction = payload.get("pooled_context_radio_direction")
    statistics = payload.get("typed_context_statistics")
    present = payload.get("context_present")
    complete = payload.get("candidate_search_complete")
    valid = payload.get("typed_context_valid")
    probe_count = payload.get("candidate_probe_count")
    context_count = payload.get("context_token_count")
    offsets = payload.get("context_token_row_offsets")
    local_rows = payload.get("context_token_local_rows")
    global_rows = payload.get("context_token_global_rows")
    region_ids = payload.get("region_row_ids")
    terminations = payload.get("candidate_termination")
    regions = int(canonical.numel()) if torch.is_tensor(canonical) else -1
    aligned = (
        scales,
        anchor_local,
        anchor_global,
        present,
        complete,
        valid,
        probe_count,
        context_count,
    )
    if (
        not torch.is_tensor(canonical)
        or canonical.dtype != torch.long
        or canonical.ndim != 1
        or regions <= 0
        or regions > 4096
        or (regions > 1 and not bool((canonical[1:] > canonical[:-1]).all()))
        or any(not torch.is_tensor(item) or item.shape != (regions,) for item in aligned)
        or scales.dtype != torch.long
        or anchor_local.dtype != torch.long
        or anchor_global.dtype != torch.long
        or present.dtype != torch.bool
        or complete.dtype != torch.bool
        or valid.dtype != torch.bool
        or probe_count.dtype != torch.long
        or context_count.dtype != torch.long
        or bool((scales < 0).any())
        or bool((anchor_local < 0).any())
        or bool((anchor_global < 0).any())
        or not torch.is_tensor(direction)
        or direction.dtype != torch.float16
        or direction.shape != (regions, TYPED_CONTEXT_FEATURE_DIM)
        or not torch.is_tensor(statistics)
        or statistics.dtype != torch.float32
        or statistics.shape != (regions, TYPED_CONTEXT_STATISTIC_DIM)
        or not bool(torch.isfinite(direction).all())
        or not bool(torch.isfinite(statistics).all())
        or not isinstance(region_ids, list)
        or len(region_ids) != regions
        or len(set(region_ids)) != regions
        or any(not isinstance(item, str) or not item for item in region_ids)
        or not isinstance(terminations, list)
        or len(terminations) != regions
        or any(item not in TYPED_CONTEXT_TERMINATIONS for item in terminations)
    ):
        raise ValueError("typed-context aligned tensor layout differs")
    expected_complete = torch.tensor(
        [item == TERMINATION_COMPLETE for item in terminations], dtype=torch.bool
    )
    v4_specification = contract["context_selection"]["contract"]
    candidate_cap = int(v4_specification["token_candidate_limit"])
    maximum_tokens = int(v4_specification["maximum_tokens"])
    scale_count = len(v4_specification["radii_m"])
    if (
        not torch.equal(complete.cpu(), expected_complete)
        or not torch.equal(present.cpu(), context_count.cpu() > 0)
        or bool((valid & ~(present & complete)).any())
        or bool((probe_count <= 0).any())
        or bool((probe_count[complete] > candidate_cap).any())
        or bool((probe_count[~complete] != candidate_cap + 1).any())
        or bool((context_count < 0).any())
        or bool((context_count > maximum_tokens - 1).any())
        or bool((scales >= scale_count).any())
    ):
        raise ValueError("typed-context routing or termination differs")
    inactive = ~valid
    if bool(direction[inactive].count_nonzero()) or bool(
        statistics[inactive].count_nonzero()
    ):
        raise ValueError("inactive typed-context carrier must be exact zero")
    if bool(valid.any()):
        norms = torch.linalg.vector_norm(direction[valid].float(), dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=1e-3):
            raise ValueError("active typed-context carrier is not unit L2")
    if (
        not torch.is_tensor(offsets)
        or offsets.dtype != torch.long
        or offsets.shape != (regions + 1,)
        or int(offsets[0]) != 0
        or bool((offsets[1:] < offsets[:-1]).any())
        or not torch.is_tensor(local_rows)
        or local_rows.dtype != torch.long
        or local_rows.ndim != 1
        or not torch.is_tensor(global_rows)
        or global_rows.dtype != torch.long
        or global_rows.shape != local_rows.shape
        or int(offsets[-1]) != int(local_rows.numel())
        or bool((local_rows < 0).any())
        or bool((global_rows < 0).any())
        or not torch.equal(offsets[1:] - offsets[:-1], context_count)
    ):
        raise ValueError("typed-context sparse row audit differs")
    for row in range(regions):
        start, stop = int(offsets[row]), int(offsets[row + 1])
        if (
            int(torch.unique(local_rows[start:stop]).numel()) != stop - start
            or int(torch.unique(global_rows[start:stop]).numel()) != stop - start
        ):
            raise ValueError("typed-context row audit repeats a context token")
    expected_channels = typed_context_channel_sha256(payload)
    if payload.get("channel_sha256") != expected_channels:
        raise ValueError("typed-context channel SHA-256 differs")
    return {
        **payload,
        "canonical_region_indices": canonical.detach().cpu().contiguous(),
        "scale_indices": scales.detach().cpu().contiguous(),
        "pooled_context_radio_direction": direction.detach().cpu().contiguous(),
        "typed_context_statistics": statistics.detach().cpu().contiguous(),
        "context_token_row_offsets": offsets.detach().cpu().contiguous(),
        "context_token_local_rows": local_rows.detach().cpu().contiguous(),
        "context_token_global_rows": global_rows.detach().cpu().contiguous(),
    }


__all__ = [
    "CandidateCompleteTypedSelection",
    "PooledTypedContext",
    "TERMINATION_CANDIDATE_CAP",
    "TERMINATION_COMPLETE",
    "TYPED_CONTEXT_AUTHORITY_SCHEMA",
    "TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION",
    "TYPED_CONTEXT_FEATURE_DIM",
    "TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256",
    "TYPED_CONTEXT_STATISTIC_DIM",
    "TYPED_CONTEXT_STATISTIC_NAMES",
    "TYPED_CONTEXT_STATISTIC_NAMES_SHA256",
    "candidate_complete_typed_context",
    "candidate_complete_typed_context_batch",
    "pool_typed_context_radio",
    "typed_context_channel_sha256",
    "typed_context_overlay_contract",
    "typed_context_source_access",
    "validate_typed_context_authority",
]
