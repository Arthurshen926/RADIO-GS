import dataclasses

import pytest
import torch

from radio_gs.querying.source_conditioned_hierarchy_completion import (
    ACTION_FIELD_BASE,
    ACTION_HIERARCHY,
    HierarchyOOFResult,
    apply_source_conditioned_completion,
    build_maximum_spanning_forest,
    deterministic_group_folds,
    hierarchy_probability_from_source,
    make_source_observation,
    run_source_only_hierarchy_oof,
)


GRAPH_SHA = "a" * 64
SOURCE_SHA = "b" * 64
FOLD_SHA = "c" * 64


def _line_hierarchy(count: int = 12):
    undirected = [(index, index + 1) for index in range(count - 1)]
    # Make the middle edge the final inter-part merge.  Reciprocal storage
    # mirrors the canonical support-graph cache schema.
    affinity = [0.1 if first == count // 2 - 1 else 1.0 for first, _ in undirected]
    directed = undirected + [(second, first) for first, second in undirected]
    weights = affinity + affinity
    return build_maximum_spanning_forest(
        torch.tensor(directed).T,
        torch.tensor(weights),
        torch.arange(100, 100 + count),
        support_graph_sha256=GRAPH_SHA,
        expected_support_graph_sha256=GRAPH_SHA,
    )


def _source(positive: torch.Tensor, negative: torch.Tensor):
    mass = positive + negative
    q = torch.full_like(mass, 0.5, dtype=torch.float64)
    observed = mass > 0
    q[observed] = positive[observed] / mass[observed]
    confidence = torch.zeros_like(q)
    confidence[observed] = 0.8
    return make_source_observation(
        torch.arange(100, 100 + len(positive)),
        positive,
        negative,
        q,
        confidence,
        authority_sha256=SOURCE_SHA,
    )


def _balanced_groups() -> torch.Tensor:
    """Return six group ids for every deterministic fold."""
    by_fold = {fold: [] for fold in range(3)}
    candidate = 0
    while any(len(values) < 6 for values in by_fold.values()):
        fold = int(deterministic_group_folds(torch.tensor([candidate]))[0])
        if len(by_fold[fold]) < 6:
            by_fold[fold].append(candidate)
        candidate += 1
    # The first six rows are positive and the last six negative.  Give every
    # fold two groups from each class.
    positive = [value for fold in range(3) for value in by_fold[fold][:2]]
    negative = [value for fold in range(3) for value in by_fold[fold][2:4]]
    return torch.tensor(positive + negative)


def test_kruskal_hierarchy_is_identical_under_edge_permutation_and_ties():
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
    )
    weights = torch.ones(edges.shape[1])
    rows = torch.tensor([10, 20, 30, 40])
    first = build_maximum_spanning_forest(
        edges,
        weights,
        rows,
        support_graph_sha256=GRAPH_SHA,
        expected_support_graph_sha256=GRAPH_SHA,
    )
    permutation = torch.tensor([7, 3, 5, 1, 6, 0, 4, 2])
    second = build_maximum_spanning_forest(
        edges[:, permutation],
        weights[permutation],
        rows,
        support_graph_sha256=GRAPH_SHA,
        expected_support_graph_sha256=GRAPH_SHA,
    )
    assert first.content_sha256 == second.content_sha256
    torch.testing.assert_close(first.parent, second.parent)
    torch.testing.assert_close(first.left, second.left)
    torch.testing.assert_close(first.right, second.right)


def test_vectorized_edge_canonicalization_scales_to_dense_synthetic_graph():
    """Exercise the million-edge class without target data or a GPU.

    Fifty thousand leaves with eight reciprocal ring offsets produce 800k
    directed edges.  This guards against reintroducing a Python tuple/dict for
    every graph edge while remaining small enough for a focused CPU test.
    """

    leaf_count = 50_000
    offsets = torch.arange(1, 9, dtype=torch.long)
    first = torch.arange(leaf_count, dtype=torch.long).repeat(len(offsets))
    repeated_offsets = offsets.repeat_interleave(leaf_count)
    second = torch.remainder(first + repeated_offsets, leaf_count)
    forward = torch.stack([first, second])
    edges = torch.cat([forward, forward.flip(0)], dim=1)
    forward_affinity = (1.0 / repeated_offsets.double()).contiguous()
    affinity = torch.cat([forward_affinity, forward_affinity])

    hierarchy = build_maximum_spanning_forest(
        edges,
        affinity,
        torch.arange(leaf_count),
        support_graph_sha256=GRAPH_SHA,
        expected_support_graph_sha256=GRAPH_SHA,
    )
    assert edges.shape[1] == 800_000
    assert hierarchy.leaf_count == leaf_count
    assert hierarchy.node_count == 2 * leaf_count - 1
    assert hierarchy.roots.numel() == 1


def test_graph_and_hierarchy_authorities_fail_closed():
    with pytest.raises(ValueError, match="unknown support-graph"):
        build_maximum_spanning_forest(
            torch.tensor([[0], [1]]),
            torch.ones(1),
            torch.tensor([0, 1]),
            support_graph_sha256=GRAPH_SHA,
            expected_support_graph_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="reciprocal"):
        build_maximum_spanning_forest(
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([0.8, 0.7]),
            torch.tensor([0, 1]),
            support_graph_sha256=GRAPH_SHA,
            expected_support_graph_sha256=GRAPH_SHA,
        )


def test_p_h_is_bounded_keeps_multiple_branches_and_does_not_make_missing_negative():
    hierarchy = build_maximum_spanning_forest(
        torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        torch.ones(4),
        torch.arange(4),
        support_graph_sha256=GRAPH_SHA,
        expected_support_graph_sha256=GRAPH_SHA,
    )
    positive = torch.tensor([1.0, 0.0, 2.0, 0.0])
    no_negative = torch.zeros(4)
    result = hierarchy_probability_from_source(
        hierarchy, torch.full((4,), 0.1), positive, no_negative
    )
    assert result.positive_branch_components == 2
    assert result.positive_seed_leaves.tolist() == [0, 2]
    torch.testing.assert_close(result.hierarchy_probability, torch.ones(4, dtype=torch.float64))
    assert bool(((result.hierarchy_probability >= 0) & (result.hierarchy_probability <= 1)).all())

    # A row with no responsibility is neutral.  Explicit negative mass, and
    # only explicit negative mass, lowers its ancestor proposal probability.
    explicit_negative = hierarchy_probability_from_source(
        hierarchy,
        torch.full((4,), 0.1),
        positive,
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )
    assert result.proposal_probability[1] == 1.0
    assert explicit_negative.proposal_probability[1] == 0.5


def test_source_probability_shape_and_content_provenance_fail_closed():
    positive = torch.tensor([1.0, 0.0])
    negative = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="q_obs differs"):
        make_source_observation(
            torch.tensor([0, 1]),
            positive,
            negative,
            torch.tensor([0.8, 0.2]),
            torch.tensor([0.5, 0.5]),
            authority_sha256=SOURCE_SHA,
        )
    valid = make_source_observation(
        torch.tensor([0, 1]),
        positive,
        negative,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.5, 0.5]),
        authority_sha256=SOURCE_SHA,
    )
    with pytest.raises(ValueError, match="content digest"):
        dataclasses.replace(valid, content_sha256="e" * 64)


def test_group_fold_assignment_never_splits_a_raster_or_subtree_group():
    groups = torch.tensor([11, 11, 11, 29, 29, 42, 42, 42])
    folds = deterministic_group_folds(groups)
    for group in groups.unique():
        assert folds[groups == group].unique().numel() == 1


def test_source_only_oof_selects_hierarchy_when_ancestor_completion_is_better():
    hierarchy = _line_hierarchy()
    positive = torch.cat([torch.ones(6), torch.zeros(6)]).double()
    negative = torch.cat([torch.zeros(6), torch.ones(6)]).double()
    observation = _source(positive, negative)
    groups = _balanced_groups()
    seen = []

    def predictor(training_positive, training_negative, fold):
        heldout = deterministic_group_folds(groups) == fold
        assert bool((training_positive[heldout] == 0).all())
        assert bool((training_negative[heldout] == 0).all())
        seen.append(fold)
        return torch.full((12,), 0.5)

    result = run_source_only_hierarchy_oof(
        hierarchy,
        observation,
        groups,
        predictor,
        fold_unit_authority_sha256=FOLD_SHA,
        expected_fold_unit_authority_sha256=FOLD_SHA,
        expected_support_graph_sha256=GRAPH_SHA,
        expected_source_authority_sha256=SOURCE_SHA,
        minimum_class_rows=1,
    )
    assert seen == [0, 1, 2]
    assert result.selected_action == ACTION_HIERARCHY
    assert (
        result.metrics[ACTION_HIERARCHY]["responsibility_balanced_log_loss"]
        < result.metrics[ACTION_FIELD_BASE]["responsibility_balanced_log_loss"]
    )
    assert set(result.metrics[ACTION_HIERARCHY]) == {
        "responsibility_balanced_log_loss",
        "responsibility_weighted_auc",
        "responsibility_weighted_brier",
        "responsibility_weighted_ece10",
        "proposal_support_size",
    }


def test_exact_metric_and_support_tie_returns_base_unchanged_then_fuses_source():
    hierarchy = _line_hierarchy()
    positive = torch.cat([torch.ones(6), torch.zeros(6)]).double()
    negative = torch.cat([torch.zeros(6), torch.ones(6)]).double()
    observation = _source(positive, negative)
    groups = _balanced_groups()

    result = run_source_only_hierarchy_oof(
        hierarchy,
        observation,
        groups,
        lambda positive, negative, fold: torch.ones(12),
        fold_unit_authority_sha256=FOLD_SHA,
        expected_fold_unit_authority_sha256=FOLD_SHA,
        expected_support_graph_sha256=GRAPH_SHA,
        expected_source_authority_sha256=SOURCE_SHA,
        minimum_class_rows=1,
    )
    assert result.selected_action == ACTION_FIELD_BASE
    assert result.metrics[ACTION_FIELD_BASE] == result.metrics[ACTION_HIERARCHY]

    field = torch.linspace(0.1, 0.9, 12)
    completed = apply_source_conditioned_completion(
        hierarchy,
        observation,
        field,
        result,
        expected_support_graph_sha256=GRAPH_SHA,
        expected_source_authority_sha256=SOURCE_SHA,
        expected_fold_unit_authority_sha256=FOLD_SHA,
    )
    torch.testing.assert_close(completed.p_hierarchy, field.double())
    expected = 0.2 * field.double() + 0.8 * observation.q_obs
    torch.testing.assert_close(completed.p_final, expected)
    assert completed.proposal is None

    forged = dataclasses.replace(result, source_authority_sha256="f" * 64)
    with pytest.raises(ValueError, match="OOF decision provenance"):
        apply_source_conditioned_completion(
            hierarchy,
            observation,
            field,
            forged,
            expected_support_graph_sha256=GRAPH_SHA,
            expected_source_authority_sha256=SOURCE_SHA,
            expected_fold_unit_authority_sha256=FOLD_SHA,
        )
