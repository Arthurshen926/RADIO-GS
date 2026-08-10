import torch

from radio_gs.querying.absolute_relevance_relation_readout import (
    absolute_relevance_relation_readout,
    readout_contract,
)


def test_absolute_readout_is_monotone_bounded_and_keeps_failed_query_exact():
    unary = torch.tensor(
        [
            [0.80, 0.45],
            [0.30, 0.50],
            [0.20, 0.40],
            [0.10, 0.30],
        ]
    )
    pairs = torch.tensor([[0, 1, 2], [1, 2, 3]])
    probability = torch.tensor([0.95, 0.90, 0.70])
    output = absolute_relevance_relation_readout(
        region_absolute_relevance=unary,
        pair_indices=pairs,
        pair_probabilities=probability,
        absolute_boundary=0.5,
        relation_threshold=0.8,
        maximum_regions=3,
        path_method="widest_path",
    )
    assert output.query_gate.tolist() == [True, False]
    assert output.seed_region_indices.tolist() == [0, 1]
    assert torch.equal(output.final_relevance[:, 1], unary[:, 1])
    assert output.final_relevance[0, 0].item() == unary[0, 0].item()
    assert output.final_relevance[1, 0] > unary[1, 0]
    assert output.final_relevance[2, 0] > unary[2, 0]
    assert output.final_relevance[3, 0].item() == unary[3, 0].item()
    assert bool((output.final_relevance >= unary).all())
    assert bool((output.final_relevance <= 1).all())


def test_absolute_readout_relation_threshold_and_contract_do_not_replace_unary():
    unary = torch.tensor([[0.9], [0.4]])
    output = absolute_relevance_relation_readout(
        region_absolute_relevance=unary,
        pair_indices=torch.tensor([[0], [1]]),
        pair_probabilities=torch.tensor([0.79]),
        absolute_boundary=0.5,
        relation_threshold=0.8,
        maximum_regions=2,
    )
    assert torch.equal(output.final_relevance, unary)
    contract = readout_contract()
    assert contract["invariants"]["final_not_below_absolute_unary"] is True
    assert contract["invariants"]["rank_or_minmax_normalization"] is False
    assert contract["legacy_readout_default_changed"] is False


def test_absolute_readout_rejects_unsorted_or_out_of_range_graph():
    kwargs = {
        "region_absolute_relevance": torch.tensor([[0.8], [0.4], [0.3]]),
        "pair_probabilities": torch.tensor([0.9, 0.9]),
        "absolute_boundary": 0.5,
        "relation_threshold": 0.8,
        "maximum_regions": 3,
    }
    try:
        absolute_relevance_relation_readout(
            pair_indices=torch.tensor([[1, 0], [2, 1]]), **kwargs
        )
    except ValueError as error:
        assert "sorted unique" in str(error)
    else:
        raise AssertionError("unsorted pairs were accepted")
