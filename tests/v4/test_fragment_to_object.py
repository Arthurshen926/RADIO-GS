from __future__ import annotations


def test_nonseed_fragment_obeys_explicit_cannot_link():
    import torch
    from radio_gs.v4.object_memory.fragment_to_object import _soft_assignments

    affinity = torch.tensor([[1., .2, .99], [.2, 1., .4], [.99, .4, 1.]])
    forbidden = torch.zeros(3, 3, dtype=torch.bool)
    forbidden[0, 2] = forbidden[2, 0] = True
    for normalization in ("full_then_top2", "top2_then_normalize"):
        assignment, null = _soft_assignments(
            affinity, [[0], [1]], torch.tensor([0, 1, 0]),
            torch.tensor([-1, -1, -1]), temperature=.1, null_affinity=.45,
            top_k=2, inherit_hierarchy_root=False,
            normalization=normalization, cannot_link=forbidden,
        )
        assert assignment[2, 0] == 0
        assert torch.allclose(assignment.sum(-1) + null, torch.ones(3))

import pytest
import torch
from torch import nn

from radio_gs.v4.object_memory.fragment_to_object import (
    FragmentAssociationConfiguration,
    _constrained_groups,
    _disjoint_same_view_cannot_link,
    _pairwise_geometry,
    _cross_view_support_count,
    _group_medoid_indices,
    _pairwise_visibility_conflict,
    _soft_assignments,
    build_fragment_object_hypotheses,
)


def test_disjoint_policy_preserves_partial_overlap_and_unknown_support():
    positive = torch.tensor([
        [1., 1., 0., 0.], [0., 1., 1., 0.],
        [0., 0., 0., 1.], [0., 0., 0., 0.],
        [0., 0., 0., 1.],
    ])
    geometry, _, _ = _pairwise_geometry(
        positive, torch.zeros(4, 3), evidence_threshold=0.05,
    )
    forbidden = _disjoint_same_view_cannot_link(geometry, torch.tensor([0, 0, 0, 0, 1]))
    assert 0 < geometry[0, 1] < 0.5  # Previous predicate falsely forbade this pair.
    assert not forbidden[0, 1]
    assert forbidden[0, 2]
    assert not forbidden[3].any()  # Empty support is unknown, not negative.
    assert not forbidden[:, 3].any()
    assert not forbidden[4].any()  # Different views are not this constraint.
    assert not forbidden.diagonal().any()


def test_global_grouping_merges_cross_view_fragments_but_not_same_view_roots():
    affinity = torch.tensor([
        [1.0, 0.9, 0.8],
        [0.9, 1.0, 0.85],
        [0.8, 0.85, 1.0],
    ])
    groups = _constrained_groups(
        affinity,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0, 1]),
        minimum_affinity=0.5,
    )
    assert len(groups) == 2
    assert not any(0 in group and 1 in group for group in groups)
    assert any(2 in group and len(group) == 2 for group in groups)


def test_explicit_cannot_links_allow_nested_same_view_fragments_only_when_safe():
    affinity = torch.tensor([
        [1.0, 0.9, 0.8],
        [0.9, 1.0, 0.8],
        [0.8, 0.8, 1.0],
    ])
    cannot_link = torch.tensor([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=torch.bool)
    groups = _constrained_groups(
        affinity,
        torch.arange(3),
        torch.tensor([0, 0, 1]),
        minimum_affinity=0.5,
        cannot_link=cannot_link,
    )
    assert any(set(group) == {0, 1, 2} for group in groups)

    cannot_link[0, 1] = cannot_link[1, 0] = True
    groups = _constrained_groups(
        affinity,
        torch.arange(3),
        torch.tensor([0, 0, 1]),
        minimum_affinity=0.5,
        cannot_link=cannot_link,
    )
    assert not any(0 in group and 1 in group for group in groups)


def test_fragment_object_hypotheses_keep_top2_and_explicit_null():
    result = build_fragment_object_hypotheses(
        fragment_positive=torch.tensor([
            [1.0, 0.8, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.8],
            [0.0, 0.0, 0.9, 1.0],
        ]),
        element_centres=torch.tensor([
            [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
            [2.0, 0.0, 0.0], [2.1, 0.0, 0.0],
        ]),
        fragment_view_index=torch.tensor([0, 1, 0, 1]),
        fragment_local_index=torch.tensor([0, 0, 1, 1]),
        parent_index=torch.tensor([-1, -1, -1, -1]),
        quality=torch.ones(4),
        appearance_prototypes=None,
        configuration=FragmentAssociationConfiguration(
            merge_affinity=0.5,
            geometry_weight=0.8,
            appearance_weight=0.0,
            spatial_weight=0.2,
        ),
    )
    assignment = result["fragment_assignment"]
    null = result["fragment_null_probability"]
    assert assignment.shape == (4, 2)
    assert (assignment > 0).sum(-1).max() <= 2
    assert torch.allclose(assignment.sum(-1) + null, torch.ones(4))
    assert result["observed_membership"].shape == (4, 2)
    assert result["pairwise_audit"]["object_hypothesis_count"] == 2


def test_association_configuration_rejects_non_normalized_weights():
    with pytest.raises(ValueError, match="sum to one"):
        FragmentAssociationConfiguration(geometry_weight=1.0).validate()


def test_cross_view_support_counts_distinct_views_not_fragments():
    affinity = torch.tensor([
        [1.0, 0.9, 0.8, 0.1],
        [0.9, 1.0, 0.7, 0.2],
        [0.8, 0.7, 1.0, 0.6],
        [0.1, 0.2, 0.6, 1.0],
    ])
    support = _cross_view_support_count(
        affinity,
        torch.tensor([0, 1, 1, 2]),
        minimum_affinity=0.5,
    )
    assert support.tolist() == [1, 1, 2, 1]


def test_visibility_conflict_uses_mutual_explicit_negative_evidence():
    conflict = _pairwise_visibility_conflict(
        torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        torch.tensor([0, 1, 1]),
        torch.ones(2, 2, dtype=torch.bool),
        evidence_threshold=0.05,
    )
    assert conflict[0, 1] == 0
    assert conflict[0, 2] == 1
    assert conflict[1, 2] == 1


def test_visibility_conflict_requires_both_directions():
    conflict = _pairwise_visibility_conflict(
        torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        torch.tensor([0, 1]),
        torch.tensor([[1, 1], [0, 1]], dtype=torch.bool),
        evidence_threshold=0.05,
    )
    assert conflict[0, 1] == 0


def test_group_medoid_balances_centrality_and_source_quality():
    medoids = _group_medoid_indices(
        [[0, 1, 2], [3]],
        torch.tensor([
            [1.0, 0.9, 0.2, 0.0],
            [0.9, 1.0, 0.8, 0.0],
            [0.2, 0.8, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        torch.tensor([1.0, 1.0, 1.0, 0.7]),
    )
    assert medoids.tolist() == [1, 3]


def test_top2_fragment_assignment_is_invariant_to_unselected_hypotheses():
    base = torch.tensor([
        [1.0, 0.8, 0.1],
        [0.8, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])
    expanded = torch.tensor([
        [1.0, 0.8, 0.1, 0.2, 0.3],
        [0.8, 1.0, 0.2, 0.1, 0.2],
        [0.1, 0.2, 1.0, 0.2, 0.1],
        [0.2, 0.1, 0.2, 1.0, 0.1],
        [0.3, 0.2, 0.1, 0.1, 1.0],
    ])
    kwargs = {
        "groups": [[0], [1], [2]],
        "view_index": torch.tensor([0, 1, 2]),
        "global_parent": torch.tensor([-1, -1, -1]),
        "temperature": 0.1,
        "null_affinity": 0.45,
        "top_k": 2,
        "inherit_hierarchy_root": False,
        "normalization": "top2_then_normalize",
    }
    known_base, null_base = _soft_assignments(base, **kwargs)
    kwargs["groups"] = [[0], [1], [2], [3], [4]]
    kwargs["view_index"] = torch.tensor([0, 1, 2, 3, 4])
    kwargs["global_parent"] = torch.tensor([-1, -1, -1, -1, -1])
    known_expanded, null_expanded = _soft_assignments(expanded, **kwargs)
    assert torch.allclose(known_base[0, :2], known_expanded[0, :2])
    assert torch.allclose(null_base[0], null_expanded[0])


def test_top2_assignment_never_drops_own_seed_on_exact_ties():
    assignment, null = _soft_assignments(
        torch.ones(3, 3),
        [[0], [1], [2]],
        torch.zeros(3, dtype=torch.long),
        torch.full((3,), -1, dtype=torch.long),
        temperature=0.1,
        null_affinity=0.45,
        top_k=2,
        inherit_hierarchy_root=False,
        normalization="full_then_top2",
        cannot_link=torch.zeros(3, 3, dtype=torch.bool),
    )
    assert bool((assignment.sum(0) > 0).all())
    assert torch.allclose(assignment.sum(-1) + null, torch.ones(3))


def test_all_fragment_seeds_do_not_force_child_into_broad_parent_identity():
    result = build_fragment_object_hypotheses(
        fragment_positive=torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]),
        element_centres=torch.tensor([
            [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0], [1.1, 0.0, 0.0],
        ]),
        fragment_view_index=torch.tensor([0, 0, 0]),
        fragment_local_index=torch.tensor([0, 1, 2]),
        parent_index=torch.tensor([-1, 0, 0]),
        quality=torch.ones(3),
        appearance_prototypes=None,
        configuration=FragmentAssociationConfiguration(
            merge_affinity=0.5,
            geometry_weight=0.8,
            appearance_weight=0.0,
            spatial_weight=0.2,
            seed_policy="all_fragments",
        ),
    )
    assert result["pairwise_audit"]["seed_fragment_count"] == 3
    assert result["pairwise_audit"]["hierarchy_root_fragment_count"] == 1
    assert result["pairwise_audit"]["object_hypothesis_count"] == 3
    assert result["pairwise_audit"]["seed_policy"] == "all_fragments"


def test_cross_view_supported_fragments_seed_and_unsupported_fragment_assigns():
    result = build_fragment_object_hypotheses(
        fragment_positive=torch.tensor([
            [1.0, 1.0, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]),
        element_centres=torch.tensor([
            [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
            [2.0, 0.0, 0.0], [2.1, 0.0, 0.0],
        ]),
        fragment_view_index=torch.tensor([0, 1, 2]),
        fragment_local_index=torch.tensor([0, 0, 0]),
        parent_index=torch.tensor([-1, -1, -1]),
        quality=torch.ones(3),
        appearance_prototypes=None,
        configuration=FragmentAssociationConfiguration(
            merge_affinity=0.5,
            geometry_weight=0.8,
            appearance_weight=0.0,
            spatial_weight=0.2,
            seed_policy="cross_view_supported",
            minimum_seed_support_views=1,
        ),
    )
    assert result["pairwise_audit"]["seed_fragment_count"] == 2
    assert result["pairwise_audit"]["object_hypothesis_count"] == 1
    assert result["fragment_assignment"].shape == (3, 1)


def test_learned_affinity_replaces_hand_weighted_pair_score():
    class ConstantAffinity(nn.Module):
        def forward(self, features):
            return torch.full(
                features.shape[:-1], 8.0, dtype=features.dtype,
                device=features.device,
            )

    result = build_fragment_object_hypotheses(
        fragment_positive=torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        element_centres=torch.tensor([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        ]),
        fragment_view_index=torch.tensor([0, 1]),
        fragment_local_index=torch.tensor([0, 0]),
        parent_index=torch.tensor([-1, -1]),
        quality=torch.ones(2),
        appearance_prototypes=None,
        view_visibility=torch.ones(2, 2, dtype=torch.bool),
        learned_affinity_model=ConstantAffinity(),
        configuration=FragmentAssociationConfiguration(
            merge_affinity=0.99,
            geometry_weight=0.8,
            appearance_weight=0.0,
            spatial_weight=0.2,
        ),
    )
    assert result["pairwise_audit"]["object_hypothesis_count"] == 1
    assert result["pairwise_audit"]["pair_affinity_source"].endswith("mlp")
    assert result["pairwise_audit"]["fragment_assignment_affinity_source"].startswith(
        "fixed_weighted"
    )
