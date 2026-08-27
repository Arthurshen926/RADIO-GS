import torch

from radio_gs.v3.training.instance_upper_bound import (
    ExtraCodeArm,
    FrozenProjectionArm,
    MaskEpisode,
    build_known_pixel_authority,
    episode_objective,
    relation_contrastive_loss,
    same_view_different_peers,
)
from radio_gs.v3.training.run_instance_upper_bound import (
    materialize_canonical_memory,
    radio_preservation_summary,
    sampled_source_visual_loss,
    training_support_for_heldout,
)


def test_known_authority_uses_positive_and_explicit_different_only():
    masks = torch.tensor([
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
    ], dtype=torch.bool)
    target, known = build_known_pixel_authority(
        masks, 0, 10, torch.tensor([10, 11, 12]),
        torch.tensor([10]), torch.tensor([11]),
    )
    assert target.tolist() == [[True, False], [False, False]]
    assert known.tolist() == [[True, True], [True, False]]
    assert same_view_different_peers(masks, 0, torch.tensor([10, 11, 12])) == (11, 12)


def test_oracle_receives_gradient_through_exact_render_closure():
    arm = ExtraCodeArm(3, output_dim=2)
    episode = MaskEpisode(
        proposal_index=0,
        view_index=0,
        gaussian_ids=torch.tensor([0, 1, 2]),
        pixel_ids=torch.tensor([0, 0, 1]),
        contribution_weights=torch.tensor([0.5, 0.5, 1.0]),
        target=torch.tensor([[1, 0]], dtype=torch.bool),
        known=torch.tensor([[1, 1]], dtype=torch.bool),
        boundary=torch.tensor([[1, 1]], dtype=torch.bool),
        unknown=torch.tensor([[0, 0]], dtype=torch.bool),
        scale=0.5,
    )
    loss = episode_objective(
        arm(), (torch.tensor([0, 1]), torch.ones(2)), episode, temperature=0.2
    )
    loss.backward()
    assert arm.code.grad is not None
    assert torch.isfinite(arm.code.grad).all()


def test_oracle_is_scale_conditioned_without_a_second_code_table():
    arm = ExtraCodeArm(4, output_dim=3)
    assert not torch.equal(arm(0.05), arm(0.8))
    assert arm.code.shape == (4, 3)


def test_frozen_projection_checkpoint_does_not_duplicate_d512_field():
    arm = FrozenProjectionArm(torch.zeros(4, 512))
    assert "latent" not in arm.state_dict()
    assert set(arm.state_dict()) == {
        "projection.weight",
        "scale_adapter.weight",
        "scale_adapter.bias",
    }


def test_heldout_support_uses_only_known_same_training_proposals():
    supports = (
        (torch.tensor([0]), torch.tensor([1.0])),
        (torch.tensor([1]), torch.tensor([0.8])),
        (torch.tensor([], dtype=torch.long), torch.tensor([])),
    )
    relation = {
        "edge_left": torch.tensor([0, 0]),
        "edge_right": torch.tensor([1, 2]),
        "edge_relation": torch.tensor([1, -1], dtype=torch.int8),
    }
    result = training_support_for_heldout(0, {1, 2}, supports, relation)
    assert result is not None
    assert result[0].tolist() == [1]


def test_materialize_canonical_memory_uses_public_post_fusion_query():
    class Field:
        num_gaussians = 5

        class Decoder:
            coefficient_dim = 3

        decoder = Decoder()

        def __init__(self):
            self.calls = []

        def query_memory(self, rows, *, representation):
            assert representation == "coefficients"
            self.calls.append(rows.clone())
            return rows[:, None].repeat(1, 3).float() + 10.0

    field = Field()
    value = materialize_canonical_memory(field, chunk_size=2)
    assert value.shape == (5, 3)
    assert value[4].tolist() == [14.0, 14.0, 14.0]
    assert [rows.tolist() for rows in field.calls] == [[0, 1], [2, 3], [4]]


def test_radio_preservation_summary_is_exact_for_unmodified_writeback():
    from radio_gs.v3.training.low_rank_writeback import LowRankWritebackArm

    arm = LowRankWritebackArm(
        torch.randn(5, 512),
        radio_basis=torch.randn(7, 512),
        radio_mean=torch.randn(7),
        radio_scale=torch.rand(7) + 0.5,
        rank=2,
    )
    summary = radio_preservation_summary(arm, chunk_size=2)
    assert summary["mean_cosine"] > 0.99999
    assert summary["p05_cosine"] > 0.99999


def test_sampled_source_visual_loss_uses_exact_hits_and_teacher():
    from radio_gs.v3.training.low_rank_writeback import LowRankWritebackArm

    latent = torch.zeros(2, 512)
    latent[0, 0] = 1
    latent[1, 1] = 1
    arm = LowRankWritebackArm(
        latent,
        radio_basis=torch.eye(2, 512),
        radio_mean=torch.zeros(2),
        radio_scale=torch.ones(2),
        rank=2,
    )
    episode = MaskEpisode(
        proposal_index=0,
        view_index=1,
        gaussian_ids=torch.tensor([0, 1]),
        pixel_ids=torch.tensor([0, 1]),
        contribution_weights=torch.ones(2),
        target=torch.ones(1, 2, dtype=torch.bool),
        known=torch.ones(1, 2, dtype=torch.bool),
        boundary=torch.zeros(1, 2, dtype=torch.bool),
        unknown=torch.zeros(1, 2, dtype=torch.bool),
        scale=0.5,
    )
    loss = sampled_source_visual_loss(
        arm,
        episode,
        torch.eye(2),
        [0, 1],
        __import__("random").Random(1),
        pixel_budget=2,
    )
    assert abs(float(loss)) < 1e-6


def test_relation_contrastive_excludes_unknown_and_trains_both_known_classes():
    embedding = torch.nn.functional.normalize(torch.randn(6, 4, requires_grad=True), dim=-1)
    supports = tuple((torch.tensor([index]), torch.tensor([1.0])) for index in range(6))
    loss = relation_contrastive_loss(
        embedding, supports,
        torch.tensor([0, 2, 4]), torch.tensor([1, 3, 5]),
        torch.tensor([1, 0, -1], dtype=torch.int8),
    )
    loss.backward()
    assert torch.isfinite(loss)
