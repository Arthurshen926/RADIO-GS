"""Coverage-deficit native-V3 residual with continuous source reliability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.querying import scale_aware_native_v3_support_fusion as legacy
from radio_gs.querying.absolute_relevance_relation_readout import (
    _bounded_path_support,
    _validated_graph,
    absolute_relevance_relation_readout,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


READOUT_SCHEMA = "radio_gs.coverage_deficit_native_v3_support_fusion.v1"
READOUT_SCHEMA_VERSION = 1
SEMANTIC_BOUNDARY = legacy.SEMANTIC_BOUNDARY
RELATION_THRESHOLD = legacy.RELATION_THRESHOLD
MAXIMUM_REGIONS = legacy.MAXIMUM_REGIONS
PATH_METHOD = legacy.PATH_METHOD


def readout_contract() -> dict[str, Any]:
    return {
        "schema": READOUT_SCHEMA,
        "schema_version": READOUT_SCHEMA_VERSION,
        "parent_contract_sha256": legacy.READOUT_CONTRACT_SHA256,
        "semantic_boundary": SEMANTIC_BOUNDARY,
        "relation": legacy.readout_contract()["relation"],
        "source_observation_strength": (
            "clip(2*minimum_mean_observation_evidence-1,0,1); derived_from_"
            "the_frozen_count_over_count_plus_one_encoding_so_one_view_is_zero"
        ),
        "effective_edge_strength": (
            "normalized_relation_probability_excess_times_source_observation_strength"
        ),
        "target_anchor_strength": (
            "normalized_frozen_selected_scale_canonical_anchor_excess"
        ),
        "covered_mass": (
            "mean_normalized_positive_excess_over_valid_target_semantic_core_"
            "at_frozen_selected_scale"
        ),
        "covered_core_quantile": "median_of_the_same_normalized_positive_excess",
        "coverage_deficit": "(1-covered_mass)*(1-covered_core_quantile)",
        "completion_strength": (
            "seed_excess*effective_widest_path_excess*target_anchor_strength*"
            "coverage_deficit"
        ),
        "completion_evidence": "target_registered_scale_local_unary",
        "invariants": {
            "primitive_unary_never_decreases": True,
            "seed_region_is_not_filled": True,
            "failed_or_isolated_seed_is_exact_O1": True,
            "single_view_effective_path_is_exact_O1": True,
            "low_reliability_edge_limit_is_exact_O1": True,
            "missing_target_anchor_evidence_is_exact_O1": True,
            "saturated_target_core_residual_tends_to_zero": True,
            "query_axis_opaque": True,
            "scene_specific_parameters": False,
            "new_tunable_constants": False,
        },
    }


READOUT_CONTRACT_SHA256 = canonical_json_sha256(readout_contract())


@dataclass(frozen=True)
class CoverageDeficitNativeV3Fusion:
    primitive_unary: torch.Tensor
    primitive_relevance_by_scale: torch.Tensor
    final_primitive_relevance: torch.Tensor
    seed_region_indices: torch.Tensor
    query_gate: torch.Tensor
    relation_selected_region_masks: torch.Tensor
    effective_path_support: torch.Tensor
    target_anchor_strength: torch.Tensor
    covered_mass: torch.Tensor
    covered_core_quantile: torch.Tensor
    coverage_deficit: torch.Tensor
    completion_strength: torch.Tensor
    changed_primitive_query_cells: int


def _effective_path_support(
    *,
    region_count: int,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    pair_observation_evidence: torch.Tensor,
    seeds: torch.Tensor,
    query_gate: torch.Tensor,
) -> torch.Tensor:
    probability = torch.as_tensor(pair_probabilities).detach().float().cpu()
    observation = (
        torch.as_tensor(pair_observation_evidence).detach().float().cpu().reshape(-1)
    )
    if (
        probability.shape != observation.shape
        or not bool(torch.isfinite(observation).all())
        or bool((observation < 0.0).any())
        or bool((observation > 1.0).any())
    ):
        raise ValueError("coverage-deficit observation evidence differs")
    relation_excess = (
        (probability - RELATION_THRESHOLD) / (1.0 - RELATION_THRESHOLD)
    ).clamp(0.0, 1.0)
    # observation_evidence is frozen as positive_view_count/(count+1).
    # Consequently a single view is exactly 0.5 and must contribute zero.
    observation_strength = (2.0 * observation - 1.0).clamp(0.0, 1.0)
    effective_excess = relation_excess * observation_strength
    effective_probability = torch.where(
        probability >= RELATION_THRESHOLD,
        RELATION_THRESHOLD + (1.0 - RELATION_THRESHOLD) * effective_excess,
        torch.zeros_like(probability),
    )
    adjacency = _validated_graph(
        region_count=region_count,
        pair_indices=pair_indices,
        pair_probabilities=effective_probability,
        relation_threshold=RELATION_THRESHOLD,
    )
    support = torch.zeros(region_count, int(seeds.numel()), dtype=torch.float32)
    for query, seed in enumerate(seeds.tolist()):
        if not bool(query_gate[query]):
            continue
        _selected, values = _bounded_path_support(
            seed=int(seed),
            adjacency=adjacency,
            maximum_regions=region_count,
            method=PATH_METHOD,
        )
        support[:, query] = values
    return support.contiguous()


def coverage_deficit_native_v3_support_fusion(
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
    pair_observation_evidence: torch.Tensor,
) -> CoverageDeficitNativeV3Fusion:
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
    region_anchor = relevance[anchor_rows, region_scale, :].contiguous()
    seed_unary = region_anchor.clone()
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
    effective_path = _effective_path_support(
        region_count=region_count,
        pair_indices=pair_indices,
        pair_probabilities=pair_probabilities,
        pair_observation_evidence=pair_observation_evidence,
        seeds=relation.seed_region_indices,
        query_gate=relation.query_gate,
    )

    final = base.clone()
    anchor_strength = torch.zeros_like(seed_unary)
    covered_mass = torch.zeros_like(seed_unary)
    covered_quantile = torch.zeros_like(seed_unary)
    deficit = torch.zeros_like(seed_unary)
    completion_strength = torch.zeros_like(seed_unary)
    for query in range(query_count):
        if not bool(relation.query_gate[query]):
            continue
        seed = int(relation.seed_region_indices[query])
        seed_excess = float(
            ((seed_unary[seed, query] - SEMANTIC_BOUNDARY)
             / (1.0 - SEMANTIC_BOUNDARY)).clamp(0.0, 1.0)
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
            positive_excess = (
                (deployed - SEMANTIC_BOUNDARY) / (1.0 - SEMANTIC_BOUNDARY)
            ).clamp(0.0, 1.0)
            mass = positive_excess.mean()
            quantile = positive_excess.median()
            target_deficit = (1.0 - mass) * (1.0 - quantile)
            target_anchor = relevance[
                int(anchor_rows[region]), int(selected_scale[query]), query
            ]
            target_anchor_strength = (
                (target_anchor - SEMANTIC_BOUNDARY) / (1.0 - SEMANTIC_BOUNDARY)
            ).clamp(0.0, 1.0)
            path_strength = (
                (effective_path[region, query] - RELATION_THRESHOLD)
                / (1.0 - RELATION_THRESHOLD)
            ).clamp(0.0, 1.0)
            completion = (
                seed_excess
                * float(path_strength)
                * float(target_anchor_strength)
                * float(target_deficit)
            )
            anchor_strength[region, query] = target_anchor_strength
            covered_mass[region, query] = mass
            covered_quantile[region, query] = quantile
            deficit[region, query] = target_deficit
            completion_strength[region, query] = completion
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
        or bool((completion_strength[~relation.selected_region_masks] != 0).any())
        or int(relation.selected_region_masks.sum(dim=0).max()) > MAXIMUM_REGIONS
    ):
        raise RuntimeError("coverage-deficit fusion invariant failed")
    return CoverageDeficitNativeV3Fusion(
        primitive_unary=base.contiguous(),
        primitive_relevance_by_scale=relevance,
        final_primitive_relevance=final.contiguous(),
        seed_region_indices=relation.seed_region_indices,
        query_gate=relation.query_gate,
        relation_selected_region_masks=relation.selected_region_masks,
        effective_path_support=effective_path,
        target_anchor_strength=anchor_strength.contiguous(),
        covered_mass=covered_mass.contiguous(),
        covered_core_quantile=covered_quantile.contiguous(),
        coverage_deficit=deficit.contiguous(),
        completion_strength=completion_strength.contiguous(),
        changed_primitive_query_cells=int((final != base).sum()),
    )
