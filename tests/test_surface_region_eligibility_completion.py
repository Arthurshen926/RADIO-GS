import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV3
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.training.surface_region_eligibility_completion import (
    STRUCTURED_ELIGIBILITY_POLICY,
    completion_region_id,
    structured_eligibility_variant,
)


def _line_graph(count: int) -> tuple[torch.Tensor, PrimitiveSupportGraph]:
    xyz = torch.stack(
        [
            torch.arange(count, dtype=torch.float32) * 0.01,
            torch.zeros(count),
            torch.zeros(count),
        ],
        dim=1,
    )
    edge = torch.tensor(
        [(row, row + 1) for row in range(count - 1)],
        dtype=torch.long,
    ).T.contiguous()
    affinity = torch.ones(edge.shape[1])
    graph = PrimitiveSupportGraph(
        edge_index=edge,
        edge_weight=torch.ones(edge.shape[1]),
        raw_affinity=affinity,
        local_sigma=torch.full((count,), 0.01),
        num_nodes=count,
        edge_channels={"appearance": affinity, "boundary": affinity},
    )
    return xyz, graph


def test_structured_eligibility_is_deterministic_and_exercises_real_v3_fill() -> None:
    xyz, graph = _line_graph(80)
    contract = SurfaceRegionContractV3(
        radii_m=(0.50,),
        context_ratio=1.2,
        minimum_tokens=24,
        maximum_tokens=32,
        token_candidate_limit=80,
    )
    prepared = contract.prepare_graph(graph, xyz)
    first = structured_eligibility_variant(
        contract=contract,
        prepared_graph=prepared,
        anchor=0,
        radius_m=0.50,
        teacher_region_id="frozen-teacher-region",
        variant_index=0,
    )
    repeated = structured_eligibility_variant(
        contract=contract,
        prepared_graph=prepared,
        anchor=0,
        radius_m=0.50,
        teacher_region_id="frozen-teacher-region",
        variant_index=0,
    )
    second = structured_eligibility_variant(
        contract=contract,
        prepared_graph=prepared,
        anchor=0,
        radius_m=0.50,
        teacher_region_id="frozen-teacher-region",
        variant_index=1,
    )

    assert first.policy == STRUCTURED_ELIGIBILITY_POLICY
    assert torch.equal(first.mask, repeated.mask)
    assert first.mask_sha256 == repeated.mask_sha256
    assert first.semantic_eligible_tokens < contract.minimum_tokens
    assert first.globally_eligible_tokens >= contract.minimum_tokens
    assert first.nominal_semantic_keep_tokens == 20
    assert first.semantic_eligible_tokens == 20

    expansion = contract.expand(
        graph,
        xyz,
        0,
        0.50,
        prepared_graph=prepared,
        selection_eligibility=first.mask,
    )
    assert bool(expansion.support_fill_mask.any())
    assert bool(first.mask[expansion.rows].all())
    assert int(
        (expansion.core_mask | expansion.context_mask).sum()
    ) == 20
    assert int(expansion.support_fill_mask.sum()) == 4
    assert len(expansion.rows) == contract.minimum_tokens
    first_id = completion_region_id(
        teacher_region_id="frozen-teacher-region", variant=first
    )
    assert first_id == completion_region_id(
        teacher_region_id="frozen-teacher-region", variant=repeated
    )
    assert first_id != completion_region_id(
        teacher_region_id="frozen-teacher-region", variant=second
    )
