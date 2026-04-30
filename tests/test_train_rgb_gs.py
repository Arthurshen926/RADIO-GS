import torch

from radio_gs.scripts import train_rgb_gs
from radio_gs.scripts.train_rgb_gs import SimpleGaussianModel


def _init_data(num_points: int = 2):
    return {
        "means": torch.zeros(num_points, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_points),
        "log_scales": torch.zeros(num_points, 3),
        "logit_opacity": torch.zeros(num_points, 1),
        "sh_dc": torch.zeros(num_points, 3),
    }


def test_simple_gaussian_model_uses_rgb_colors_for_degree_zero(monkeypatch):
    seen = {}

    def fake_rasterization(**kwargs):
        colors = kwargs["colors"]
        seen["colors_shape"] = tuple(colors.shape)
        height = kwargs["height"]
        width = kwargs["width"]
        return (
            torch.zeros(1, height, width, 3),
            torch.zeros(1, height, width, 1),
            {},
        )

    monkeypatch.setattr(train_rgb_gs, "rasterization", fake_rasterization)
    model = SimpleGaussianModel(_init_data(), sh_degree=0)
    viewmat = torch.eye(4)
    k = torch.eye(3)

    model.render(viewmat, k, W=4, H=3)

    assert seen["colors_shape"] == (2, 3)
