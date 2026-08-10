"""Scale-aware primitive-unary fusion with a frozen native-V3 relation.

This module deliberately keeps semantic evidence and object support separate.
The incoming primitive relevance is the complete frozen multiscale query
readout *before* its per-query scale gather.  A registered SurfaceRegion uses
only the relevance of its own scale to seed the source-promoted native-V3
co-membership graph.  The graph may then add a bounded positive residual, but
it may never replace, lower, binarize, or globally rescale the primitive
unary.

The construction addresses the failure mode of the earlier multi-region
unions: selecting a graph region no longer turns its whole semantic core into
unit foreground.  Instead, a primitive keeps its local query evidence and is
raised by a noisy-OR residual whose strength is the conjunction of seed
evidence and source-calibrated relation-path evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.querying.absolute_relevance_relation_readout import (
    absolute_relevance_relation_readout,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


READOUT_SCHEMA = "radio_gs.scale_aware_native_v3_support_fusion.v1"
READOUT_SCHEMA_VERSION = 1

# These are not target-scene choices.  0.60 is the frozen VALA primitive-mask
# boundary; the other constants are the source-promoted native-V3 rule.
SEMANTIC_BOUNDARY = 0.60
RELATION_THRESHOLD = 0.85
MAXIMUM_REGIONS = 8
PATH_METHOD = "widest_path"


def readout_contract() -> dict[str, Any]:
    return {
        "schema": READOUT_SCHEMA,
        "schema_version": READOUT_SCHEMA_VERSION,
        "input_unary": (
            "frozen_knn10_independent_per_scale_minmax_primitive_relevance_N_S_Q"
        ),
        "region_unary": "same_scale_registered_anchor_primitive_relevance",
        "seed": (
            "highest_region_unary_restricted_to_the_frozen_query_selected_scale"
        ),
        "relation": {
            "source_promoted_native_v3_threshold": RELATION_THRESHOLD,
            "maximum_regions": MAXIMUM_REGIONS,
            "path_method": PATH_METHOD,
        },
        "semantic_boundary": SEMANTIC_BOUNDARY,
        "completion": (
            "p_final=max(p_selected_scale,"
            "p_region_scale+(1-p_region_scale)*normalized_seed_excess*"
            "normalized_path_excess)"
        ),
        "region_to_primitive": "registered_semantic_core_only",
        "invariants": {
            "primitive_unary_never_decreases": True,
            "isolated_seed_is_exact_primitive_unary": True,
            "failed_seed_gate_is_exact_primitive_unary": True,
            "seed_region_is_not_uniformly_filled": True,
            "relation_support_bounded_by_eight": True,
            "candidate_uses_its_registered_scale": True,
            "binary_region_union": False,
            "query_axis_opaque": True,
            "scene_specific_parameters": False,
        },
    }


READOUT_CONTRACT_SHA256 = canonical_json_sha256(readout_contract())


@dataclass(frozen=True)
class ScaleAwareNativeV3Fusion:
    primitive_unary: torch.Tensor
    primitive_relevance_by_scale: torch.Tensor
    final_primitive_relevance: torch.Tensor
    region_anchor_relevance: torch.Tensor
    seed_region_indices: torch.Tensor
    query_gate: torch.Tensor
    relation_selected_region_masks: torch.Tensor
    relation_path_support: torch.Tensor
    changed_primitive_query_cells: int


def _validated_inputs(
    *,
    primitive_relevance_by_scale: torch.Tensor,
    selected_scale_indices: torch.Tensor,
    primitive_valid: torch.Tensor,
    region_rows: torch.Tensor,
    semantic_core_mask: torch.Tensor,
    region_anchor_positions: torch.Tensor,
    region_scale_indices: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    relevance = (
        torch.as_tensor(primitive_relevance_by_scale)
        .detach()
        .float()
        .cpu()
        .contiguous()
    )
    selected_scale = (
        torch.as_tensor(selected_scale_indices)
        .detach()
        .long()
        .cpu()
        .reshape(-1)
        .contiguous()
    )
    valid = (
        torch.as_tensor(primitive_valid)
        .detach()
        .bool()
        .cpu()
        .reshape(-1)
        .contiguous()
    )
    rows = torch.as_tensor(region_rows).detach().long().cpu().contiguous()
    core = (
        torch.as_tensor(semantic_core_mask)
        .detach()
        .bool()
        .cpu()
        .contiguous()
    )
    anchor = (
        torch.as_tensor(region_anchor_positions)
        .detach()
        .long()
        .cpu()
        .reshape(-1)
        .contiguous()
    )
    scale = (
        torch.as_tensor(region_scale_indices)
        .detach()
        .long()
        .cpu()
        .reshape(-1)
        .contiguous()
    )
    if relevance.ndim != 3 or min(relevance.shape) <= 0:
        raise ValueError("primitive relevance must be nonempty [N,S,Q]")
    primitive_count, scale_count, query_count = relevance.shape
    region_count = int(rows.shape[0]) if rows.ndim == 2 else -1
    if (
        not bool(torch.isfinite(relevance).all())
        or bool((relevance < 0.0).any())
        or bool((relevance > 1.0).any())
        or selected_scale.shape != (query_count,)
        or bool((selected_scale < 0).any())
        or bool((selected_scale >= scale_count).any())
        or valid.shape != (primitive_count,)
        or not bool(valid.any())
        or region_count <= 0
        or core.shape != rows.shape
        or anchor.shape != (region_count,)
        or scale.shape != (region_count,)
        or bool((anchor < 0).any())
        or bool((anchor >= rows.shape[1]).any())
        or bool((scale < 0).any())
        or bool((scale >= scale_count).any())
        or not bool(core.any(dim=1).all())
        or not bool(core[torch.arange(region_count), anchor].all())
    ):
        raise ValueError("scale-aware native-V3 axes differ")
    active = rows[core]
    if (
        active.numel() <= 0
        or bool((active < 0).any())
        or bool((active >= primitive_count).any())
    ):
        raise ValueError("SurfaceRegion core differs from the primitive axis")
    for region in range(region_count):
        members = rows[region, core[region]]
        if int(torch.unique(members).numel()) != int(members.numel()):
            raise ValueError("SurfaceRegion semantic core contains duplicates")
    return relevance, selected_scale, valid, rows, core, anchor, scale


def scale_aware_native_v3_support_fusion(
    *,
    primitive_relevance_by_scale: torch.Tensor,
    selected_scale_indices: torch.Tensor,
    primitive_valid: torch.Tensor,
    region_rows: torch.Tensor,
    semantic_core_mask: torch.Tensor,
    region_anchor_positions: torch.Tensor,
    region_scale_indices: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
) -> ScaleAwareNativeV3Fusion:
    """Fuse a strong scale-specific unary with the fixed native-V3 relation.

    Query scales must be the frozen raw-smoothed-peak choices.  The query axis
    remains opaque; neither identifiers nor strings enter this function.
    """

    (
        relevance,
        selected_scale,
        valid,
        rows,
        core,
        anchor,
        region_scale,
    ) = _validated_inputs(
        primitive_relevance_by_scale=primitive_relevance_by_scale,
        selected_scale_indices=selected_scale_indices,
        primitive_valid=primitive_valid,
        region_rows=region_rows,
        semantic_core_mask=semantic_core_mask,
        region_anchor_positions=region_anchor_positions,
        region_scale_indices=region_scale_indices,
    )
    primitive_count, _scale_count, query_count = relevance.shape
    query_axis = torch.arange(query_count)
    base = relevance[:, selected_scale, query_axis].contiguous()
    base[~valid] = 0.0

    region_count = int(rows.shape[0])
    anchor_rows = rows[torch.arange(region_count), anchor]
    region_anchor_relevance = relevance[
        anchor_rows, region_scale, :
    ].contiguous()

    # Only a same-scale region may own the semantic seed.  Off-scale regions
    # remain available as relation-path targets, and keep their own registered
    # scale when their primitive evidence is fused below.
    seed_unary = region_anchor_relevance.clone()
    same_scale = region_scale[:, None] == selected_scale[None, :]
    seed_unary[~same_scale] = 0.0
    relation = absolute_relevance_relation_readout(
        region_absolute_relevance=seed_unary,
        pair_indices=pair_indices,
        pair_probabilities=pair_probabilities,
        absolute_boundary=SEMANTIC_BOUNDARY,
        relation_threshold=RELATION_THRESHOLD,
        maximum_regions=MAXIMUM_REGIONS,
        path_method=PATH_METHOD,
    )

    final = base.clone()
    for query in range(query_count):
        if not bool(relation.query_gate[query]):
            continue
        seed = int(relation.seed_region_indices[query])
        seed_score = float(seed_unary[seed, query])
        seed_excess = max(
            0.0,
            min(1.0, (seed_score - SEMANTIC_BOUNDARY) / (1.0 - SEMANTIC_BOUNDARY)),
        )
        selected_regions = torch.where(
            relation.selected_region_masks[:, query]
        )[0].tolist()
        for region in selected_regions:
            # A seed without a relation target must be an exact no-op.  This
            # prevents the old single-region binary-fill failure.
            if region == seed:
                continue
            path_score = float(relation.path_support[region, query])
            path_excess = max(
                0.0,
                min(
                    1.0,
                    (path_score - RELATION_THRESHOLD)
                    / (1.0 - RELATION_THRESHOLD),
                ),
            )
            completion = seed_excess * path_excess
            if completion <= 0.0:
                continue
            members = rows[region, core[region]]
            members = members[valid[members]]
            if members.numel() == 0:
                continue
            local = relevance[members, int(region_scale[region]), query]
            proposed = local + (1.0 - local) * completion
            final[members, query] = torch.maximum(final[members, query], proposed)

    final[~valid] = 0.0
    if (
        not bool(torch.isfinite(final).all())
        or bool((final < 0.0).any())
        or bool((final > 1.0).any())
        or bool((final[valid] < base[valid]).any())
        or bool(final[~valid].count_nonzero())
        or int(relation.selected_region_masks.sum(dim=0).max()) > MAXIMUM_REGIONS
    ):
        raise RuntimeError("scale-aware native-V3 fusion invariant failed")
    return ScaleAwareNativeV3Fusion(
        primitive_unary=base.contiguous(),
        primitive_relevance_by_scale=relevance,
        final_primitive_relevance=final.contiguous(),
        region_anchor_relevance=region_anchor_relevance,
        seed_region_indices=relation.seed_region_indices,
        query_gate=relation.query_gate,
        relation_selected_region_masks=relation.selected_region_masks,
        relation_path_support=relation.path_support,
        changed_primitive_query_cells=int((final != base).sum()),
    )


__all__ = [
    "MAXIMUM_REGIONS",
    "PATH_METHOD",
    "READOUT_CONTRACT_SHA256",
    "READOUT_SCHEMA",
    "RELATION_THRESHOLD",
    "SEMANTIC_BOUNDARY",
    "ScaleAwareNativeV3Fusion",
    "readout_contract",
    "scale_aware_native_v3_support_fusion",
]
