from pathlib import Path

import torch

from radio_gs.models.radio_adaptors import (
    RadioMLPAdaptor,
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)


def _state(prefix: str, input_dim: int = 4, hidden_dim: int = 6, output_dim: int = 3):
    return {
        f"{prefix}.fc1.weight": torch.randn(hidden_dim, input_dim),
        f"{prefix}.fc1.bias": torch.randn(hidden_dim),
        f"{prefix}.blocks.0.0.weight": torch.ones(hidden_dim),
        f"{prefix}.blocks.0.0.bias": torch.zeros(hidden_dim),
        f"{prefix}.blocks.0.2.weight": torch.randn(hidden_dim, hidden_dim),
        f"{prefix}.blocks.0.2.bias": torch.randn(hidden_dim),
        f"{prefix}.blocks.1.0.weight": torch.ones(hidden_dim),
        f"{prefix}.blocks.1.0.bias": torch.zeros(hidden_dim),
        f"{prefix}.blocks.1.2.weight": torch.randn(hidden_dim, hidden_dim),
        f"{prefix}.blocks.1.2.bias": torch.randn(hidden_dim),
        f"{prefix}.final.0.weight": torch.ones(hidden_dim),
        f"{prefix}.final.0.bias": torch.zeros(hidden_dim),
        f"{prefix}.final.2.weight": torch.randn(output_dim, hidden_dim),
        f"{prefix}.final.2.bias": torch.randn(output_dim),
    }


def test_load_radio_adaptor_from_checkpoint_supports_dino_v3_alias(tmp_path: Path):
    ckpt = {"state_dict": _state("_feature_projections.dino_v3_7b")}
    path = tmp_path / "radio.pth"
    torch.save(ckpt, path)

    adaptor = load_radio_adaptor_from_checkpoint(path, "dino_v3", kind="feature_projection")

    assert isinstance(adaptor, RadioMLPAdaptor)
    assert adaptor.input_dim == 4
    assert adaptor.output_dim == 3


def test_project_feature_map_with_adaptor_preserves_spatial_shape():
    adaptor = RadioMLPAdaptor(input_dim=4, hidden_dim=6, output_dim=3, num_blocks=2)
    features = torch.randn(2, 4, 5, 7)

    projected = project_feature_map_with_adaptor(features, adaptor)

    assert projected.shape == (2, 3, 5, 7)
    norms = projected.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
