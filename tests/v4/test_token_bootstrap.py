import torch

from radio_gs.v4.object_memory import ObservedObjectEvidence, SurfaceTokenBootstrap


def evidence(positive, view_id):
    positive = torch.tensor(positive, dtype=torch.float32)
    visible = torch.ones_like(positive)
    return ObservedObjectEvidence.from_positive_visibility(
        positive,
        visible,
        view_ids=torch.full((positive.shape[0],), view_id),
        quality=torch.ones(positive.shape[0]),
    )


def test_bootstrap_creates_roots_and_reuses_token_for_parts_and_later_views():
    centres = torch.tensor(
        [[0.0, 0, 0], [0.1, 0, 0], [1.0, 0, 0], [1.1, 0, 0]], dtype=torch.float32
    )
    model = SurfaceTokenBootstrap(centres, minimum_overlap=0.1)
    first = model.process_view(
        evidence([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]], 0),
        element_visibility=torch.ones(3, 4),
        parent_index=torch.tensor([-1, -1, 0]),
    )
    assert model.num_tokens == 2
    assert first.token_ids[2] == first.token_ids[0]
    assert not first.created[2]

    second = model.process_view(
        evidence([[0.8, 1, 0, 0]], 1),
        element_visibility=torch.ones(1, 4),
        parent_index=torch.tensor([-1]),
    )
    assert model.num_tokens == 2
    assert second.token_ids[0] == first.token_ids[0]
    assert not second.created[0]


def test_unmatched_part_selects_null_and_cannot_create_token():
    model = SurfaceTokenBootstrap(torch.eye(3), minimum_overlap=0.2)
    result = model.process_view(
        evidence([[1, 0, 0]], 0),
        element_visibility=torch.ones(1, 3),
        parent_index=torch.tensor([2]),
    )
    assert model.num_tokens == 0
    assert result.token_ids.tolist() == [-1]
    assert result.null_probability.tolist() == [1.0]


def test_bootstrap_accumulates_query_free_proposal_descriptors():
    model = SurfaceTokenBootstrap(torch.eye(3), appearance_weight=0.25)
    model.process_view(
        evidence([[1, 0, 0]], 0),
        element_visibility=torch.ones(1, 3),
        parent_index=torch.tensor([-1]),
        proposal_descriptors=torch.tensor([[1.0, 0.0]]),
    )
    model.process_view(
        evidence([[1, 0, 0]], 1),
        element_visibility=torch.ones(1, 3),
        parent_index=torch.tensor([-1]),
        proposal_descriptors=torch.tensor([[0.8, 0.2]]),
    )
    assert model.num_tokens == 1
    assert model.descriptor_sum is not None
    assert model.descriptor_mass.shape == (1,)


def test_frozen_batch_groups_cross_view_roots_before_single_commit():
    centres = torch.tensor(
        [[0.0, 0, 0], [0.1, 0, 0], [1.0, 0, 0], [1.1, 0, 0]], dtype=torch.float32
    )
    model = SurfaceTokenBootstrap(centres, batch_birth_overlap=0.2)
    results = model.process_batch(
        [
            evidence([[1, 1, 0, 0], [0, 0, 1, 1]], 10),
            evidence([[0.8, 1, 0, 0], [0, 0, 1, 0.8]], 20),
        ],
        element_visibilities=[torch.ones(2, 4), torch.ones(2, 4)],
        parent_indices=[torch.tensor([-1, -1]), torch.tensor([-1, -1])],
    )
    assert model.num_tokens == 2
    assert results[0].token_ids.tolist() == results[1].token_ids.tolist()
    assert sum(int(result.created.sum()) for result in results) == 2


def test_frozen_batch_part_inherits_explicit_parent_group():
    model = SurfaceTokenBootstrap(torch.eye(3), batch_birth_overlap=0.2)
    results = model.process_batch(
        [evidence([[1, 1, 0], [1, 0, 0]], 3)],
        element_visibilities=[torch.ones(2, 3)],
        parent_indices=[torch.tensor([-1, 0])],
    )
    assert model.num_tokens == 1
    assert results[0].token_ids.tolist() == [0, 0]
    assert results[0].granularity.tolist() == [0, 1]


def test_frozen_batch_existing_token_accepts_only_one_root_per_view():
    model = SurfaceTokenBootstrap(torch.eye(3), minimum_overlap=0.05)
    model.process_batch(
        [evidence([[1, 1, 0]], 1)],
        element_visibilities=[torch.ones(1, 3)],
        parent_indices=[torch.tensor([-1])],
    )
    result = model.process_batch(
        [evidence([[1, 0.8, 0], [0.9, 1, 0]], 2)],
        element_visibilities=[torch.ones(2, 3)],
        parent_indices=[torch.tensor([-1, -1])],
    )[0]
    assert model.num_tokens == 2
    assert len(set(result.token_ids.tolist())) == 2


def test_frozen_batch_sealed_identity_overrides_missing_surface_overlap():
    model = SurfaceTokenBootstrap(torch.eye(3), batch_birth_overlap=0.9)
    results = model.process_batch(
        [evidence([[1, 0, 0]], 10), evidence([[0, 0, 1]], 20)],
        element_visibilities=[torch.ones(1, 3), torch.ones(1, 3)],
        parent_indices=[torch.tensor([-1]), torch.tensor([-1])],
        proposal_identity_ids=[torch.tensor([4]), torch.tensor([4])],
    )
    assert model.num_tokens == 1
    assert results[0].token_ids.tolist() == results[1].token_ids.tolist() == [0]


def test_frozen_batch_unlabelled_root_can_attach_to_identity_core():
    model = SurfaceTokenBootstrap(torch.eye(3), batch_birth_overlap=0.5)
    results = model.process_batch(
        [
            evidence([[1, 0, 0]], 10),
            evidence([[0, 1, 0]], 20),
            evidence([[0, 1, 0]], 30),
        ],
        element_visibilities=[torch.ones(1, 3)] * 3,
        parent_indices=[torch.tensor([-1])] * 3,
        proposal_identity_ids=[torch.tensor([4]), torch.tensor([4]), torch.tensor([-1])],
    )
    assert model.num_tokens == 1
    assert [int(result.token_ids[0]) for result in results] == [0, 0, 0]
