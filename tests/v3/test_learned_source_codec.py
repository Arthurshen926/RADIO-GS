import torch

from radio_gs.v3.training.learned_source_codec import apply_codec, _principal_codec
from radio_gs.v3.evaluation.structured_source_capability import _same_pixel_retrieval
from radio_gs.v3.training.fit_render_metric_codec import _loss


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
