import torch

from radio_gs.querying.observation_clamped_harmonic import (
    ObservationClampedHarmonicConfig,
)
from radio_gs.querying.source_loo_component_gate import (
    BoundaryRingPairs,
    component_brier_records,
    recover_graph_off_field_prior,
    source_loo_predictions,
    unknown_component_labels,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _graph(edge_pairs: list[tuple[int, int]], count: int) -> PrimitiveSupportGraph:
    directed = []
    for left, right in edge_pairs:
        directed.extend(((left, right), (right, left)))
    edge_index = torch.tensor(directed, dtype=torch.long).T.contiguous()
    weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return PrimitiveSupportGraph(
        edge_index=edge_index,
        edge_weight=weight,
        raw_affinity=weight,
        local_sigma=torch.ones(count),
        num_nodes=count,
        edge_channels={},
    )


def test_field_prior_recovery_is_exact_probability_mixture_inverse():
    field = torch.tensor([0.2, 0.8, 0.4])
    source = torch.tensor([1.0, 0.0, 0.9])
    confidence = torch.tensor([0.25, 0.5, 1.0])
    fused = (1 - confidence) * field + confidence * source
    recovered, eligible = recover_graph_off_field_prior(
        fused, source, confidence
    )
    assert torch.allclose(recovered[:2], field[:2])
    assert eligible.tolist() == [True, True, False]
    assert recovered[2] == fused[2]


def test_unknown_components_do_not_merge_through_observed_rows():
    graph = _graph([(0, 1), (1, 2), (2, 3)], 4)
    labels, count = unknown_component_labels(
        graph, torch.tensor([False, True, False, False])
    )
    assert count == 2
    assert labels[0] >= 0 and labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert labels[1] == -1


def test_four_fold_component_gate_accepts_only_strict_source_brier_gain():
    # One unknown center and four observed boundary-ring rows.  Holding out any
    # boundary row leaves three identical fused boundaries, so the harmonic
    # prediction is 0.5 and strictly improves over the graph-off value 0 for a
    # source target of 1 in every fold.
    graph = _graph([(0, 1), (0, 2), (0, 3), (0, 4)], 5)
    fused = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5])
    field = torch.zeros(5)
    source_probability = torch.ones(5)
    confidence = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5])
    eligible = confidence > 0
    folds = torch.tensor([0, 0, 1, 2, 3])
    predictions = source_loo_predictions(
        graph,
        fused_probability=fused,
        field_probability=field,
        source_confidence=confidence,
        validation_eligible=eligible,
        fold_assignment=folds,
        config=ObservationClampedHarmonicConfig(
            cg_iterations=64, cg_tolerance=1e-7
        ),
    )
    rings = BoundaryRingPairs(
        component=torch.zeros(4, dtype=torch.long),
        observed_row=torch.tensor([1, 2, 3, 4]),
        affinity=torch.ones(4, dtype=torch.float64),
    )
    records, accepted = component_brier_records(
        component_count=1,
        rings=rings,
        predictions=predictions,
        field_probability=field,
        source_probability=source_probability,
        source_confidence=confidence,
        validation_eligible=eligible,
        fold_assignment=folds,
    )
    assert accepted.tolist() == [True]
    assert records[0]["strictly_improves_every_fold"] is True

    # Exact equality is not a gain: the no-margin gate must fail closed.
    equal_predictions = [field.clone() for _ in range(4)]
    _records, accepted = component_brier_records(
        component_count=1,
        rings=rings,
        predictions=equal_predictions,
        field_probability=field,
        source_probability=source_probability,
        source_confidence=confidence,
        validation_eligible=eligible,
        fold_assignment=folds,
    )
    assert accepted.tolist() == [False]
