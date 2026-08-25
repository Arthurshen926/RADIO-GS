import torch

from radio_gs.querying.identity_seeded_object_topology import (
    compile_view_exclusive_physical_tracks,
    identity_seeded_object_topology_posterior,
    proposal_query_indices_from_names,
)


def test_proposal_quality_does_not_dilute_large_object_identity() -> None:
    scores = torch.tensor([[1.0], [0.1], [0.1], [0.9]])
    # Rows 0/1/2 form the same large proposal in two views.  Its proposal mean
    # text score is low, but the immutable identity seed selects it correctly.
    rows = torch.tensor([0, 1, 2, 0, 1, 2, 3])
    props = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    weights = torch.tensor([0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.7])
    posterior, stats = identity_seeded_object_topology_posterior(
        scores,
        rows,
        props,
        weights,
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 0, -1]),
        unknown_policy="negative_outside_topology",
    )
    assert torch.allclose(
        posterior[:, 0], torch.tensor([1.0, 35.0 / 36.0, 8.0 / 9.0, 0.0])
    )
    assert stats["identity_peaks_preserved_exactly"] is True
    assert stats["num_queries_with_object_consensus"] == 1


def test_proposal_confidence_is_not_a_membership_temperature() -> None:
    scores = torch.tensor([[1.0], [0.1]])
    posterior, _ = identity_seeded_object_topology_posterior(
        scores,
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 1, 1]),
        # A proposal-global confidence of roughly 0.6 must not prevent the
        # accepted mask interior from crossing the fixed 0.6 decision rule.
        torch.tensor([0.60, 0.54, 0.60, 0.54]),
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
    )
    assert posterior[1, 0] > 0.6
    assert posterior[0, 0] == scores[0, 0]


def test_sparse_unobserved_rows_preserve_text_prior_by_default() -> None:
    scores = torch.tensor([[1.0], [0.1], [0.7]])
    posterior, stats = identity_seeded_object_topology_posterior(
        scores,
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 1, 1]),
        torch.ones(4),
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
    )
    assert posterior[2, 0] == scores[2, 0]
    assert stats["unknown_policy"] == "preserve_text_prior"


def test_quality_is_multiview_evidence_not_membership_threshold() -> None:
    scores = torch.tensor([[1.0], [0.1]])
    posterior, stats = identity_seeded_object_topology_posterior(
        scores,
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 1, 1]),
        # Pure conditional membership survives cache construction unchanged.
        torch.tensor([1.0, 0.8, 1.0, 0.8]),
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
        proposal_scores=torch.tensor([0.8, 0.8]),
        membership_calibration="pure_probability",
        use_proposal_quality=True,
    )
    # Noisy-or of two quality-weighted observations: 1-(1-.64)^2=.8704.
    assert torch.allclose(posterior[1, 0], torch.tensor(0.8704), atol=1e-6)
    assert stats["proposal_quality_role"] == (
        "proposal_ranking_and_multiview_observation_confidence"
    )


def test_missing_object_consensus_is_exact_identity_fallback() -> None:
    scores = torch.tensor([[1.0], [0.2]])
    posterior, stats = identity_seeded_object_topology_posterior(
        scores,
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
        torch.ones(2),
        torch.tensor([0]),
        torch.tensor([0]),
        minimum_object_views=2,
    )
    assert torch.equal(posterior, scores)
    assert stats["num_queries_with_object_consensus"] == 0


def test_competing_query_proposal_cannot_create_identity() -> None:
    scores = torch.tensor([[1.0, 0.0], [0.1, 1.0], [0.0, 0.1]])
    rows = torch.tensor([0, 2, 0, 2, 1, 2, 1, 2])
    props = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    posterior, _ = identity_seeded_object_topology_posterior(
        scores,
        rows,
        props,
        torch.ones(8),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([1, 1, 1, 1]),
    )
    assert torch.equal(posterior[:, 0], scores[:, 0])
    assert posterior[0, 0] == 1.0


def test_query_name_mapping_is_case_and_space_stable() -> None:
    assert proposal_query_indices_from_names(
        [" Green Apple ", "unknown"], ["green apple", "bag"]
    ).tolist() == [0, -1]


def test_query_free_hierarchy_is_reused_for_each_query_identity() -> None:
    scores = torch.tensor([[1.0, 0.0], [0.1, 1.0], [0.1, 0.1]])
    rows = torch.tensor([0, 2, 0, 2, 1, 2, 1, 2])
    props = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    posterior, stats = identity_seeded_object_topology_posterior(
        scores, rows, props, torch.ones(8),
        torch.tensor([0, 1, 0, 1]), torch.full((4,), -1),
        unknown_policy="negative_outside_topology",
    )
    assert posterior[2, 0] > 0.5
    assert posterior[2, 1] > 0.5
    assert posterior[0, 0] == 1.0 and posterior[1, 1] == 1.0
    assert stats["query_independent_mask_hierarchy"] is True
    assert stats["capability_track"] == "query_free_source_sam_exact_mpr_object_topology"


def test_physical_track_forest_prevents_same_view_chain_percolation() -> None:
    tracks = compile_view_exclusive_physical_tracks(
        torch.tensor([0, 1, 0]), torch.tensor([1, 2, 2]),
        torch.tensor([0.9, 0.8, 0.7]),
        # Nodes 0 and 2 conflict because they are alternative proposals in view 0.
        torch.tensor([0, 1, 0]),
    )
    assert tracks[0] == tracks[1]
    assert tracks[2] == -1


def test_identity_selects_one_cross_view_physical_track() -> None:
    scores = torch.tensor([[1.0], [0.1], [0.2], [0.1], [0.1]])
    rows = torch.tensor([0, 1, 0, 1, 2, 4, 2, 4])
    props = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    posterior, stats = identity_seeded_object_topology_posterior(
        scores, rows, props, torch.ones(8),
        torch.tensor([0, 1, 0, 1]), torch.full((4,), -1),
        proposal_track_indices=torch.tensor([0, 0, 1, 1]),
        unknown_policy="negative_outside_topology",
    )
    assert posterior[1, 0] > 0.5
    assert posterior[4, 0] == 0
    assert stats["selected_physical_track_per_query"] == [0]
