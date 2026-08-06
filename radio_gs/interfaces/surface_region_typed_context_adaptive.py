"""Typed-budget-complete adaptive context selection for AcceptedV2 overlays.

Stage A conservatively required exhaustion of every V4 candidate inside the
context radius.  That is stronger than necessary.  Dijkstra settles nodes in
``(distance, row)`` order, so the final bounded typed selection is immutable
as soon as the complete core count is known and enough leading context nodes
have been observed to satisfy the reserved context budget (including donation
when the core is short).  This module versions that weaker, exact proof.

No frozen V2, V4, or Stage-A implementation is changed.  The V4 mechanism is
instantiated with a versioned 192-core/64-context budget and remains bounded
to 256 published semantic tokens.
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
from radio_gs.interfaces.surface_region_typed_context import (
    TYPED_CONTEXT_FEATURE_DIM,
    TYPED_CONTEXT_STATISTIC_DIM,
    TYPED_CONTEXT_STATISTIC_NAMES,
    TYPED_CONTEXT_STATISTIC_NAMES_SHA256,
    typed_context_source_access,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA = (
    "radio_gs.surface_region_accepted_v2_typed_context_authority.v2"
)
ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION = 2
ADAPTIVE_CONTEXT_BUDGET = 64
ADAPTIVE_MAXIMUM_TOKENS = 256
ADAPTIVE_CORE_BUDGET = ADAPTIVE_MAXIMUM_TOKENS - ADAPTIVE_CONTEXT_BUDGET
ADAPTIVE_CORE_TOKEN_FRACTION = ADAPTIVE_CORE_BUDGET / ADAPTIVE_MAXIMUM_TOKENS
ADAPTIVE_INITIAL_PROBE_WIDTH = 1025
ADAPTIVE_PROBE_GROWTH_FACTOR = 4
ADAPTIVE_MAX_BATCH_SIZE = 8
ADAPTIVE_WORKING_MEMORY_CEILING_BYTES = 512 * 1024 * 1024
TERMINATION_TYPED_BUDGET = "typed_budget_satisfied"
TERMINATION_NATURAL = "natural_exhaustion"
ADAPTIVE_TERMINATIONS = (TERMINATION_TYPED_BUDGET, TERMINATION_NATURAL)
_FIXED_WORKING_BYTES = 1024 * 1024
_PER_NODE_WORKING_BYTES = 32
_PER_DIRECTED_EDGE_HEAP_BOUND_BYTES = 32
_PER_PROBE_BATCH_ROW_BYTES = 16
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCANNET_SCENE = re.compile(r"^(scene\d{4})_\d{2}$")


def adaptive_typed_context_v4_contract() -> SurfaceRegionContractV4:
    """Return the isolated V4 mechanism with a 192/64 typed budget."""

    return SurfaceRegionContractV4(core_token_fraction=ADAPTIVE_CORE_TOKEN_FRACTION)


def adaptive_typed_context_overlay_contract() -> dict[str, Any]:
    v4 = adaptive_typed_context_v4_contract()
    return {
        "schema": ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA,
        "schema_version": ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION,
        "base": "external_immutable_accepted_v2_descriptor_not_copied",
        "typed_selection": {
            "mechanism": "frozen_surface_region_contract_v4_parameterization",
            "contract": v4.to_dict(),
            "contract_sha256": v4.digest,
            "maximum_tokens": ADAPTIVE_MAXIMUM_TOKENS,
            "core_budget": ADAPTIVE_CORE_BUDGET,
            "context_budget": ADAPTIVE_CONTEXT_BUDGET,
            "selection": (
                "leading_core_and_context_then_core_first_context_second_donation"
            ),
            "carrier_tokens": "final_selected_v4_context_rows_only",
            "full_context_shell_pooling_allowed": False,
            "support_fill_allowed": False,
        },
        "selection_completion": {
            "ordering": "dijkstra_distance_then_primitive_row",
            "proof": (
                "after_first_context_core_count_is_final_and_leading_context_"
                "count_reaches_max_64_or_256_minus_core_count"
            ),
            "natural_exhaustion_is_complete": True,
            "all_candidates_inside_context_radius_must_be_exhausted": False,
        },
        "adaptive_probe": {
            "initial_width": ADAPTIVE_INITIAL_PROBE_WIDTH,
            "growth_factor": ADAPTIVE_PROBE_GROWTH_FACTOR,
            "maximum_width": "prepared_graph_num_nodes",
            "maximum_batch_size": ADAPTIVE_MAX_BATCH_SIZE,
            "working_memory_ceiling_bytes": ADAPTIVE_WORKING_MEMORY_CEILING_BYTES,
            "estimated_bytes": (
                "1MiB_plus_32_times_num_nodes_plus_32_times_semantic_csr_nnz_"
                "heap_bound_plus_16_times_batch_times_probe_width"
            ),
            "ceiling_action": "reduce_batch_then_fail_closed_if_one_row_exceeds",
        },
        "carrier": {
            "name": "pooled_context_radio_direction",
            "dimension": TYPED_CONTEXT_FEATURE_DIM,
            "storage_dtype": "float16",
            "pool": "legacy_reliability_weighted_spherical_mean_v1",
            "official_summary_head_applied": False,
            "invalid_value": "exact_all_zero",
        },
        "statistics": {
            "dimension": TYPED_CONTEXT_STATISTIC_DIM,
            "names": list(TYPED_CONTEXT_STATISTIC_NAMES),
            "names_sha256": TYPED_CONTEXT_STATISTIC_NAMES_SHA256,
            "storage_dtype": "float32",
            "invalid_value": "exact_all_zero",
        },
        "sparse_row_audit": {
            "storage": "csr_offsets_plus_local_and_global_final_context_rows",
            "dense_region_by_token_feature_tensor_allowed": False,
        },
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
    }


ADAPTIVE_TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256 = canonical_json_sha256(
    adaptive_typed_context_overlay_contract()
)


def estimate_adaptive_probe_working_bytes(
    *,
    num_nodes: int,
    num_directed_edges: int,
    probe_width: int,
    batch_size: int,
) -> int:
    nodes = int(num_nodes)
    edges = int(num_directed_edges)
    width = int(probe_width)
    batch = int(batch_size)
    if nodes <= 0 or edges < 0 or width <= 0 or batch <= 0 or width > nodes:
        raise ValueError("adaptive typed-context memory estimate inputs differ")
    return (
        _FIXED_WORKING_BYTES
        + _PER_NODE_WORKING_BYTES * nodes
        + _PER_DIRECTED_EDGE_HEAP_BOUND_BYTES * edges
        + _PER_PROBE_BATCH_ROW_BYTES * batch * width
    )


def _effective_batch_size(
    *,
    num_nodes: int,
    num_directed_edges: int,
    probe_width: int,
    requested_batch_size: int,
    memory_ceiling_bytes: int,
) -> int:
    requested = min(int(requested_batch_size), ADAPTIVE_MAX_BATCH_SIZE)
    ceiling = int(memory_ceiling_bytes)
    if requested <= 0 or ceiling <= 0:
        raise ValueError("adaptive typed-context batch or memory ceiling differs")
    base = (
        _FIXED_WORKING_BYTES
        + _PER_NODE_WORKING_BYTES * int(num_nodes)
        + _PER_DIRECTED_EDGE_HEAP_BOUND_BYTES * int(num_directed_edges)
    )
    per_row = _PER_PROBE_BATCH_ROW_BYTES * int(probe_width)
    available = ceiling - base
    maximum = available // per_row
    if maximum < 1:
        raise MemoryError(
            "adaptive typed-context one-row probe exceeds working-memory ceiling"
        )
    return min(requested, int(maximum))


@dataclass(frozen=True)
class AdaptiveTypedBudgetSelection:
    rows: torch.Tensor
    core_mask: torch.Tensor
    context_mask: torch.Tensor
    semantic_geodesic_distance: torch.Tensor
    termination: str
    final_probe_width: int
    settled_candidate_count: int
    adaptive_round_count: int

    def __post_init__(self) -> None:
        rows = torch.as_tensor(self.rows).detach().long().cpu().reshape(-1).clone()
        core = torch.as_tensor(self.core_mask).detach().bool().cpu().reshape(-1).clone()
        context = (
            torch.as_tensor(self.context_mask).detach().bool().cpu().reshape(-1).clone()
        )
        distance = (
            torch.as_tensor(self.semantic_geodesic_distance)
            .detach()
            .float()
            .cpu()
            .reshape(-1)
            .clone()
        )
        if (
            rows.numel() <= 0
            or rows.numel() > ADAPTIVE_MAXIMUM_TOKENS
            or core.shape != rows.shape
            or context.shape != rows.shape
            or distance.shape != rows.shape
            or not bool((core ^ context).all())
            or not bool(torch.isfinite(distance).all())
            or bool((distance < 0).any())
            or int(torch.unique(rows).numel()) != rows.numel()
            or str(self.termination) not in ADAPTIVE_TERMINATIONS
            or int(self.final_probe_width) <= 0
            or int(self.settled_candidate_count) <= 0
            or int(self.settled_candidate_count) > int(self.final_probe_width)
            or int(self.adaptive_round_count) <= 0
        ):
            raise ValueError("adaptive typed-budget selection differs")
        if int(context.sum()) > ADAPTIVE_MAXIMUM_TOKENS - 1:
            raise ValueError("adaptive typed-budget context count differs")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "core_mask", core)
        object.__setattr__(self, "context_mask", context)
        object.__setattr__(self, "semantic_geodesic_distance", distance)

    @property
    def search_complete(self) -> bool:
        """Compatibility property: the final bounded selection is proved complete."""

        return True


@dataclass(frozen=True)
class AdaptiveTypedBudgetBatch:
    selections: tuple[AdaptiveTypedBudgetSelection, ...]
    memory_ceiling_bytes: int
    maximum_estimated_working_bytes: int
    requested_batch_size: int

    def __post_init__(self) -> None:
        if (
            not self.selections
            or int(self.memory_ceiling_bytes) <= 0
            or int(self.maximum_estimated_working_bytes) <= 0
            or int(self.maximum_estimated_working_bytes)
            > int(self.memory_ceiling_bytes)
            or int(self.requested_batch_size) <= 0
            or int(self.requested_batch_size) > ADAPTIVE_MAX_BATCH_SIZE
        ):
            raise ValueError("adaptive typed-budget batch audit differs")


def _selection_is_complete(
    distances: np.ndarray,
    *,
    radius: float,
    settled_count: int,
    probe_width: int,
    num_nodes: int,
    maximum_tokens: int,
    context_budget: int,
) -> tuple[bool, str | None]:
    natural = int(settled_count) < int(probe_width) or int(probe_width) == int(
        num_nodes
    )
    if natural:
        return True, TERMINATION_NATURAL
    values = np.asarray(distances[:settled_count], dtype=np.float64)
    context = values > float(radius) + 1e-7
    first_context = np.flatnonzero(context)
    if first_context.size == 0:
        return False, None
    core_count = int(first_context[0])
    context_count = int(context.sum())
    required_context = max(int(context_budget), int(maximum_tokens) - core_count)
    if context_count >= required_context:
        return True, TERMINATION_TYPED_BUDGET
    return False, None


def adaptive_typed_budget_context_batch(
    contract: SurfaceRegionContractV4,
    prepared_graph: PreparedSurfaceRegionGraphV3,
    anchors: Sequence[int] | torch.Tensor,
    radius_m: float,
    *,
    selection_eligibility: torch.Tensor | np.ndarray | None = None,
    initial_probe_width: int = ADAPTIVE_INITIAL_PROBE_WIDTH,
    growth_factor: int = ADAPTIVE_PROBE_GROWTH_FACTOR,
    batch_size: int = ADAPTIVE_MAX_BATCH_SIZE,
    memory_ceiling_bytes: int = ADAPTIVE_WORKING_MEMORY_CEILING_BYTES,
) -> AdaptiveTypedBudgetBatch:
    """Adaptively prove the final typed selection without exhausting the shell."""

    if not isinstance(contract, SurfaceRegionContractV4):
        raise TypeError("adaptive typed-context requires SurfaceRegionContractV4")
    if not isinstance(prepared_graph, PreparedSurfaceRegionGraphV3):
        raise TypeError("adaptive typed-context requires a prepared V4 graph")
    if prepared_graph.contract_sha256 != contract.digest:
        raise ValueError("adaptive typed-context prepared graph contract differs")
    maximum_tokens = int(contract.maximum_tokens)
    context_budget = maximum_tokens - int(
        round(maximum_tokens * float(contract.core_token_fraction))
    )
    if (
        maximum_tokens != ADAPTIVE_MAXIMUM_TOKENS
        or context_budget != ADAPTIVE_CONTEXT_BUDGET
    ):
        raise ValueError("adaptive typed-context requires the frozen 192/64 budget")
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("adaptive typed-context radius must be positive and finite")
    anchor_values = torch.as_tensor(anchors).detach().long().cpu().reshape(-1)
    if anchor_values.numel() == 0:
        raise ValueError("adaptive typed-context requires at least one anchor")
    num_nodes = int(prepared_graph.num_nodes)
    if bool((anchor_values < 0).any()) or bool((anchor_values >= num_nodes).any()):
        raise ValueError("adaptive typed-context anchor is outside prepared graph")
    initial = int(initial_probe_width)
    growth = int(growth_factor)
    requested_batch = int(batch_size)
    ceiling = int(memory_ceiling_bytes)
    if initial <= 0 or growth <= 1:
        raise ValueError("adaptive typed-context probe schedule differs")
    if requested_batch <= 0 or requested_batch > ADAPTIVE_MAX_BATCH_SIZE:
        raise ValueError("adaptive typed-context batch exceeds the strict maximum")
    base_eligibility = contract._base_eligibility(
        selection_eligibility, num_nodes
    )
    semantic = prepared_graph.semantic_csr
    num_directed_edges = int(semantic.nnz)
    results: list[AdaptiveTypedBudgetSelection | None] = [None] * anchor_values.numel()
    rounds = torch.zeros(anchor_values.numel(), dtype=torch.long)
    unfinished = torch.arange(anchor_values.numel(), dtype=torch.long)
    probe_width = min(initial, num_nodes)
    maximum_estimated = 0
    while unfinished.numel():
        effective_batch = _effective_batch_size(
            num_nodes=num_nodes,
            num_directed_edges=num_directed_edges,
            probe_width=probe_width,
            requested_batch_size=requested_batch,
            memory_ceiling_bytes=ceiling,
        )
        next_unfinished: list[int] = []
        for start in range(0, unfinished.numel(), effective_batch):
            batch_indices = unfinished[start : start + effective_batch]
            current_batch = int(batch_indices.numel())
            estimated = estimate_adaptive_probe_working_bytes(
                num_nodes=num_nodes,
                num_directed_edges=num_directed_edges,
                probe_width=probe_width,
                batch_size=current_batch,
            )
            if estimated > ceiling:
                raise MemoryError("adaptive typed-context probe exceeds memory ceiling")
            maximum_estimated = max(maximum_estimated, estimated)
            rows, distances, counts = _bounded_dijkstra_eligible_batch(
                semantic.indptr.astype(np.int64, copy=False),
                semantic.indices.astype(np.int64, copy=False),
                semantic.data.astype(np.float64, copy=False),
                anchor_values[batch_indices].numpy().astype(np.int64, copy=False),
                base_eligibility,
                radius * float(contract.context_ratio),
                probe_width,
            )
            for local, global_index in enumerate(batch_indices.tolist()):
                rounds[global_index] += 1
                settled = int(counts[local])
                anchor = int(anchor_values[global_index])
                if settled <= 0 or int(rows[local, 0]) != anchor:
                    raise RuntimeError("adaptive typed-context Dijkstra lost its anchor")
                complete, termination = _selection_is_complete(
                    distances[local],
                    radius=radius,
                    settled_count=settled,
                    probe_width=probe_width,
                    num_nodes=num_nodes,
                    maximum_tokens=maximum_tokens,
                    context_budget=context_budget,
                )
                if not complete:
                    next_unfinished.append(global_index)
                    continue
                selected_rows, selected_distances = contract._select_semantic_candidates(
                    rows[local, :settled],
                    distances[local, :settled],
                    anchor,
                    radius,
                )
                selected_distance_f32 = np.asarray(
                    selected_distances, dtype=np.float32
                )
                core = selected_distance_f32 <= radius + 1e-7
                results[global_index] = AdaptiveTypedBudgetSelection(
                    rows=torch.from_numpy(
                        np.asarray(selected_rows, dtype=np.int64).copy()
                    ),
                    core_mask=torch.from_numpy(core.copy()),
                    context_mask=torch.from_numpy((~core).copy()),
                    semantic_geodesic_distance=torch.from_numpy(
                        selected_distance_f32.copy()
                    ),
                    termination=str(termination),
                    final_probe_width=probe_width,
                    settled_candidate_count=settled,
                    adaptive_round_count=int(rounds[global_index]),
                )
        if not next_unfinished:
            break
        if probe_width >= num_nodes:
            raise RuntimeError("adaptive typed-context failed to complete at num_nodes")
        unfinished = torch.tensor(next_unfinished, dtype=torch.long)
        next_width = min(num_nodes, probe_width * growth)
        if next_width <= probe_width:
            raise RuntimeError("adaptive typed-context probe width did not increase")
        probe_width = next_width
    if any(value is None for value in results):
        raise RuntimeError("adaptive typed-context omitted a selection")
    return AdaptiveTypedBudgetBatch(
        selections=tuple(value for value in results if value is not None),
        memory_ceiling_bytes=ceiling,
        maximum_estimated_working_bytes=maximum_estimated,
        requested_batch_size=requested_batch,
    )


def adaptive_typed_budget_context(
    contract: SurfaceRegionContractV4,
    prepared_graph: PreparedSurfaceRegionGraphV3,
    anchor: int,
    radius_m: float,
    **kwargs: Any,
) -> AdaptiveTypedBudgetSelection:
    return adaptive_typed_budget_context_batch(
        contract, prepared_graph, [int(anchor)], radius_m, **kwargs
    ).selections[0]


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


def adaptive_typed_context_channel_sha256(
    value: Mapping[str, Any]
) -> dict[str, str]:
    tensor_names = (
        "canonical_region_indices",
        "scale_indices",
        "anchor_local_rows",
        "anchor_global_rows",
        "pooled_context_radio_direction",
        "typed_context_statistics",
        "context_present",
        "selection_complete",
        "typed_context_valid",
        "final_probe_width",
        "settled_candidate_count",
        "adaptive_round_count",
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


def _require_sha(value: object, *, label: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _file_record_shape(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value.get("path", ""))
    if not path:
        raise ValueError(f"{label} file path is empty")
    return {"path": path, "sha256": _require_sha(value["sha256"], label=label)}


def validate_adaptive_typed_context_authority(value: object) -> dict[str, Any]:
    """Validate the v2 typed-budget authority without opening its inputs."""

    if not isinstance(value, Mapping):
        raise ValueError("adaptive typed-context authority must be a mapping")
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
        "selection_complete",
        "typed_context_valid",
        "candidate_termination",
        "final_probe_width",
        "settled_candidate_count",
        "adaptive_round_count",
        "context_token_count",
        "context_token_row_offsets",
        "context_token_local_rows",
        "context_token_global_rows",
        "memory_audit",
        "channel_sha256",
        "source_access",
    }
    if set(payload) != required or "accepted_v2_e0" in payload:
        raise ValueError("adaptive typed-context authority fields differ")
    contract = adaptive_typed_context_overlay_contract()
    scene = str(payload.get("scene_id", ""))
    matched = _SCANNET_SCENE.fullmatch(scene)
    if (
        payload.get("schema") != ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA
        or payload.get("schema_version")
        != ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256")
        != ADAPTIVE_TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256
        or matched is None
        or payload.get("physical_space_id") != matched.group(1)
        or payload.get("source_access") != typed_context_source_access()
        or any(payload.get("source_access", {}).values())
    ):
        raise ValueError("adaptive typed-context contract or source access differs")
    payload["producer"] = _file_record_shape(
        payload.get("producer"), label="adaptive typed-context producer"
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
        raise ValueError("adaptive typed-context input authority differs")
    frozen_inputs = dict(inputs)
    for name in (
        "accepted_v2_canonical_region_authority",
        "factorized_field_checkpoint",
        "factorized_primitive_state",
        "support_graph",
    ):
        frozen_inputs[name] = _file_record_shape(
            frozen_inputs[name], label=f"adaptive typed-context {name}"
        )
    for name in (
        "accepted_region_channel_sha256",
        "accepted_region_fingerprints_sha256",
        "factorized_radio_cache_sha256",
        "primitive_row_authority_sha256",
    ):
        frozen_inputs[name] = _require_sha(
            frozen_inputs[name], label=f"adaptive typed-context {name}"
        )
    payload["input_authority"] = frozen_inputs

    canonical = payload.get("canonical_region_indices")
    scales = payload.get("scale_indices")
    anchor_local = payload.get("anchor_local_rows")
    anchor_global = payload.get("anchor_global_rows")
    direction = payload.get("pooled_context_radio_direction")
    statistics = payload.get("typed_context_statistics")
    present = payload.get("context_present")
    selection_complete = payload.get("selection_complete")
    valid = payload.get("typed_context_valid")
    widths = payload.get("final_probe_width")
    settled = payload.get("settled_candidate_count")
    rounds = payload.get("adaptive_round_count")
    counts = payload.get("context_token_count")
    offsets = payload.get("context_token_row_offsets")
    local_rows = payload.get("context_token_local_rows")
    global_rows = payload.get("context_token_global_rows")
    terminations = payload.get("candidate_termination")
    region_ids = payload.get("region_row_ids")
    regions = int(canonical.numel()) if torch.is_tensor(canonical) else -1
    aligned = (
        scales,
        anchor_local,
        anchor_global,
        present,
        selection_complete,
        valid,
        widths,
        settled,
        rounds,
        counts,
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
        or selection_complete.dtype != torch.bool
        or valid.dtype != torch.bool
        or widths.dtype != torch.long
        or settled.dtype != torch.long
        or rounds.dtype != torch.long
        or counts.dtype != torch.long
        or bool((scales < 0).any())
        or bool((scales >= 3).any())
        or bool((anchor_local < 0).any())
        or bool((anchor_global < 0).any())
        or bool((widths <= 0).any())
        or bool((settled <= 0).any())
        or bool((settled > widths).any())
        or bool((rounds <= 0).any())
        or bool((counts < 0).any())
        or bool((counts > ADAPTIVE_MAXIMUM_TOKENS - 1).any())
        or not torch.is_tensor(direction)
        or direction.dtype != torch.float16
        or direction.shape != (regions, TYPED_CONTEXT_FEATURE_DIM)
        or not torch.is_tensor(statistics)
        or statistics.dtype != torch.float32
        or statistics.shape != (regions, TYPED_CONTEXT_STATISTIC_DIM)
        or not bool(torch.isfinite(direction).all())
        or not bool(torch.isfinite(statistics).all())
        or not isinstance(terminations, list)
        or len(terminations) != regions
        or any(item not in ADAPTIVE_TERMINATIONS for item in terminations)
        or not isinstance(region_ids, list)
        or len(region_ids) != regions
        or len(set(region_ids)) != regions
        or any(not isinstance(item, str) or not item for item in region_ids)
        or not torch.equal(present, counts > 0)
        or not bool(selection_complete.all())
        or bool((valid & ~present).any())
    ):
        raise ValueError("adaptive typed-context tensor layout differs")
    inactive = ~valid
    if bool(direction[inactive].count_nonzero()) or bool(
        statistics[inactive].count_nonzero()
    ):
        raise ValueError("inactive adaptive typed-context carrier must be exact zero")
    if bool(valid.any()):
        norms = torch.linalg.vector_norm(direction[valid].float(), dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=1e-3):
            raise ValueError("adaptive typed-context carrier is not unit L2")
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
        or not torch.equal(offsets[1:] - offsets[:-1], counts)
    ):
        raise ValueError("adaptive typed-context sparse row audit differs")
    for row in range(regions):
        start, stop = int(offsets[row]), int(offsets[row + 1])
        if (
            int(torch.unique(local_rows[start:stop]).numel()) != stop - start
            or int(torch.unique(global_rows[start:stop]).numel()) != stop - start
        ):
            raise ValueError("adaptive typed-context repeats a context row")
    memory = payload.get("memory_audit")
    if (
        not isinstance(memory, Mapping)
        or set(memory)
        != {
            "memory_ceiling_bytes",
            "maximum_estimated_working_bytes",
            "requested_batch_size",
        }
        or int(memory["memory_ceiling_bytes"])
        != ADAPTIVE_WORKING_MEMORY_CEILING_BYTES
        or int(memory["maximum_estimated_working_bytes"]) <= 0
        or int(memory["maximum_estimated_working_bytes"])
        > int(memory["memory_ceiling_bytes"])
        or int(memory["requested_batch_size"]) <= 0
        or int(memory["requested_batch_size"]) > ADAPTIVE_MAX_BATCH_SIZE
    ):
        raise ValueError("adaptive typed-context memory audit differs")
    if payload.get("channel_sha256") != adaptive_typed_context_channel_sha256(
        payload
    ):
        raise ValueError("adaptive typed-context channel SHA-256 differs")
    return {
        **payload,
        "canonical_region_indices": canonical.detach().cpu().contiguous(),
        "pooled_context_radio_direction": direction.detach().cpu().contiguous(),
        "typed_context_statistics": statistics.detach().cpu().contiguous(),
        "context_token_row_offsets": offsets.detach().cpu().contiguous(),
        "context_token_local_rows": local_rows.detach().cpu().contiguous(),
        "context_token_global_rows": global_rows.detach().cpu().contiguous(),
    }


__all__ = [
    "ADAPTIVE_CONTEXT_BUDGET",
    "ADAPTIVE_CORE_BUDGET",
    "ADAPTIVE_INITIAL_PROBE_WIDTH",
    "ADAPTIVE_MAX_BATCH_SIZE",
    "ADAPTIVE_MAXIMUM_TOKENS",
    "ADAPTIVE_PROBE_GROWTH_FACTOR",
    "ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA",
    "ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION",
    "ADAPTIVE_TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256",
    "ADAPTIVE_WORKING_MEMORY_CEILING_BYTES",
    "AdaptiveTypedBudgetBatch",
    "AdaptiveTypedBudgetSelection",
    "TERMINATION_NATURAL",
    "TERMINATION_TYPED_BUDGET",
    "adaptive_typed_budget_context",
    "adaptive_typed_budget_context_batch",
    "adaptive_typed_context_channel_sha256",
    "adaptive_typed_context_overlay_contract",
    "adaptive_typed_context_v4_contract",
    "estimate_adaptive_probe_working_bytes",
    "validate_adaptive_typed_context_authority",
]
