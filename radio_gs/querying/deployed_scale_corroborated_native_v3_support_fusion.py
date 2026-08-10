"""Native-V3 completion corroborated in the deployed query-scale domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.querying import (
    corroborated_scale_aware_native_v3_support_fusion as registered_corroboration,
)
from radio_gs.querying import scale_aware_native_v3_support_fusion as legacy
from radio_gs.querying.absolute_relevance_relation_readout import (
    absolute_relevance_relation_readout,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


READOUT_SCHEMA = "radio_gs.deployed_scale_corroborated_native_v3_support_fusion.v1"
READOUT_SCHEMA_VERSION = 1
SEMANTIC_BOUNDARY = legacy.SEMANTIC_BOUNDARY
RELATION_THRESHOLD = legacy.RELATION_THRESHOLD
MAXIMUM_REGIONS = legacy.MAXIMUM_REGIONS
PATH_METHOD = legacy.PATH_METHOD


def readout_contract() -> dict[str, Any]:
    return {
        "schema": READOUT_SCHEMA,
        "schema_version": READOUT_SCHEMA_VERSION,
        "parent_contract_sha256": (
            registered_corroboration.READOUT_CONTRACT_SHA256
        ),
        "seed": legacy.readout_contract()["seed"],
        "relation": legacy.readout_contract()["relation"],
        "semantic_boundary": SEMANTIC_BOUNDARY,
        "target_corroboration": (
            "at_least_one_valid_target_core_primitive_at_frozen_query_"
            "selected_scale_already_reaches_semantic_boundary"
        ),
        "completion_evidence": (
            "target_registered_scale_local_unary_as_in_parent_contract"
        ),
        "invariants": {
            "primitive_unary_never_decreases": True,
            "failed_or_isolated_seed_is_exact_primitive_unary": True,
            "uncorroborated_target_is_exact_primitive_unary": True,
            "seed_region_is_not_uniformly_filled": True,
            "corroboration_matches_deployed_consumer_scale": True,
            "relation_support_bounded_by_eight": True,
            "query_axis_opaque": True,
            "scene_specific_parameters": False,
            "new_tunable_constants": False,
        },
    }


READOUT_CONTRACT_SHA256 = canonical_json_sha256(readout_contract())


@dataclass(frozen=True)
class DeployedScaleCorroboratedFusion:
    primitive_unary: torch.Tensor
    primitive_relevance_by_scale: torch.Tensor
    final_primitive_relevance: torch.Tensor
    region_anchor_relevance: torch.Tensor
    seed_region_indices: torch.Tensor
    query_gate: torch.Tensor
    relation_selected_region_masks: torch.Tensor
    deployed_corroborated_region_masks: torch.Tensor
    relation_path_support: torch.Tensor
    changed_primitive_query_cells: int


def deployed_scale_corroborated_native_v3_support_fusion(
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
) -> DeployedScaleCorroboratedFusion:
    (
        relevance,
        selected_scale,
        valid,
        rows,
        core,
        anchor,
        region_scale,
    ) = legacy._validated_inputs(
        primitive_relevance_by_scale=primitive_relevance_by_scale,
        selected_scale_indices=selected_scale_indices,
        primitive_valid=primitive_valid,
        region_rows=region_rows,
        semantic_core_mask=semantic_core_mask,
        region_anchor_positions=region_anchor_positions,
        region_scale_indices=region_scale_indices,
    )
    _primitive_count, _scale_count, query_count = relevance.shape
    query_axis = torch.arange(query_count)
    base = relevance[:, selected_scale, query_axis].contiguous()
    base[~valid] = 0.0
    region_count = int(rows.shape[0])
    anchor_rows = rows[torch.arange(region_count), anchor]
    region_anchor_relevance = relevance[
        anchor_rows, region_scale, :
    ].contiguous()
    seed_unary = region_anchor_relevance.clone()
    seed_unary[region_scale[:, None] != selected_scale[None, :]] = 0.0
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
    corroborated = torch.zeros_like(relation.selected_region_masks)
    for query in range(query_count):
        if not bool(relation.query_gate[query]):
            continue
        seed = int(relation.seed_region_indices[query])
        seed_score = float(seed_unary[seed, query])
        seed_excess = max(
            0.0,
            min(1.0, (seed_score - SEMANTIC_BOUNDARY) / (1.0 - SEMANTIC_BOUNDARY)),
        )
        for region in torch.where(
            relation.selected_region_masks[:, query]
        )[0].tolist():
            if region == seed:
                continue
            members = rows[region, core[region]]
            members = members[valid[members]]
            if members.numel() == 0:
                continue
            deployed = relevance[members, int(selected_scale[query]), query]
            if not bool((deployed >= SEMANTIC_BOUNDARY).any()):
                continue
            corroborated[region, query] = True
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
        or bool((corroborated & ~relation.selected_region_masks).any())
        or int(relation.selected_region_masks.sum(dim=0).max()) > MAXIMUM_REGIONS
    ):
        raise RuntimeError("deployed-scale corroboration invariant failed")
    return DeployedScaleCorroboratedFusion(
        primitive_unary=base.contiguous(),
        primitive_relevance_by_scale=relevance,
        final_primitive_relevance=final.contiguous(),
        region_anchor_relevance=region_anchor_relevance,
        seed_region_indices=relation.seed_region_indices,
        query_gate=relation.query_gate,
        relation_selected_region_masks=relation.selected_region_masks,
        deployed_corroborated_region_masks=corroborated.contiguous(),
        relation_path_support=relation.path_support,
        changed_primitive_query_cells=int((final != base).sum()),
    )
