import torch

from radio_gs.v3.training.learned_source_codec import apply_codec, _principal_codec
from radio_gs.v3.evaluation.structured_source_capability import _same_pixel_retrieval
from radio_gs.v3.training.fit_render_metric_codec import _loss
from radio_gs.v3.training.native_visual_codec import GatedResidualVisualCodec, _matched_pairs
from radio_gs.v3.training.refine_native_visual_memory import _render_loss


def test_apply_codec_centers_before_projection():
    values = torch.tensor([[2.0, 5.0], [4.0, 9.0]])
    mean = torch.tensor([1.0, 3.0])
    basis = torch.tensor([[1.0], [2.0]])
    assert torch.equal(apply_codec(values, mean, basis), torch.tensor([[5.0], [15.0]]))


def test_principal_codec_selects_dominant_axis():
    samples = torch.tensor([
        [-3.0, 0.1], [-2.0, -0.1], [2.0, 0.1], [3.0, -0.1]
    ])
    mean, basis, retained = _principal_codec(samples, 1, torch.device("cpu"))
    assert torch.allclose(mean, torch.tensor([0.0, 0.0]), atol=1e-6)
    assert abs(float(basis[0, 0])) > 0.99
    assert retained > 0.99


def test_same_pixel_retrieval_reports_identity_ceiling():
    target = torch.eye(4)
    top1, top5, margin = _same_pixel_retrieval(target, target, 4)
    assert top1 == 1.0
    assert top5 == 1.0
    assert margin == 1.0


def test_render_metric_loss_is_finite_for_exact_identity_episode():
    episode = {
        "features": torch.eye(3).half(),
        "inverse": torch.arange(3),
        "pixel_ids": torch.arange(3),
        "weights": torch.ones(3).half(),
        "target": torch.eye(3).half(),
        "num_pixels": 3,
    }
    loss = _loss(episode, torch.zeros(3), temperature=0.1)
    assert torch.isfinite(loss)
    assert float(loss) < 0.001


def test_gated_residual_visual_codec_outputs_unit_d320():
    model = GatedResidualVisualCodec(
        radio_dim=8, dino_dim=6, output_dim=4,
        radio_rank=3, dino_rank=2, hidden_dim=7,
    )
    embedding, radio, dino = model(torch.randn(5, 8), torch.randn(5, 6))
    assert embedding.shape == (5, 4)
    assert radio.shape == (5, 8)
    assert dino.shape == (5, 6)
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_matched_pairs_are_cross_residue_same_gaussian():
    observations = [
        (torch.tensor([2, 5]), torch.tensor([10, 11])),
        (torch.tensor([2, 7]), torch.tensor([20, 21])),
    ]
    pairs = _matched_pairs(
        observations, [1, 2], pairs_per_view_pair=8, seed=0
    )
    assert torch.equal(pairs, torch.tensor([[0, 10, 1, 20, 2]]))


def test_renderer_refinement_loss_updates_only_visual_parameter():
    visual = torch.randn(7, 4, requires_grad=True)
    initial = visual.detach().clone()
    episode = {
        "rows": torch.tensor([1, 3, 5]),
        "inverse": torch.tensor([0, 1, 2]),
        "pixels": torch.tensor([0, 1, 1]),
        "weights": torch.tensor([1.0, 0.6, 0.4]),
        "target": torch.randn(2, 4),
        "num_pixels": 2,
    }
    loss, cosine, correspondence, anchor = _render_loss(
        visual, initial, episode, temperature=0.1, anchor_weight=0.1
    )
    loss.backward()
    assert visual.grad is not None
    assert torch.isfinite(torch.stack((loss, cosine, correspondence, anchor))).all()
    assert visual.grad[[0, 2, 4, 6]].abs().max() == 0
