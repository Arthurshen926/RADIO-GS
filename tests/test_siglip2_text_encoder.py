import torch
from torch import nn

from radio_gs.scripts.eval_lerf_grounding import (
    _canonicalize_siglip2_text,
    _resolve_siglip2_text_max_length,
    _restore_siglip2_text_head_from_state,
)


class _TextConfig:
    hidden_size = 1152
    projection_size = 1536


class _Config:
    text_config = _TextConfig()


class _FakeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1152, 1152)


class _FakeSiglipModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.text_model = _FakeTextModel()


def test_restore_siglip2_text_head_uses_checkpoint_projection_size() -> None:
    model = _FakeSiglipModel()
    weight = torch.randn(1536, 1152)
    bias = torch.randn(1536)

    restored = _restore_siglip2_text_head_from_state(
        model,
        {
            "text_model.head.weight": weight,
            "text_model.head.bias": bias,
        },
    )

    assert restored is True
    assert tuple(model.text_model.head.weight.shape) == (1536, 1152)
    assert torch.equal(model.text_model.head.weight, weight)
    assert torch.equal(model.text_model.head.bias, bias)


def test_resolve_siglip2_text_max_length_prefers_text_config() -> None:
    class TextConfig:
        max_position_embeddings = 64

    class Config:
        text_config = TextConfig()

    assert _resolve_siglip2_text_max_length(Config()) == 64


def test_canonicalize_siglip2_text_matches_official_radio_adaptor() -> None:
    assert _canonicalize_siglip2_text(" Stainless_steel pots! ") == "stainless steel pots"
    assert _canonicalize_siglip2_text("pour-over   vessel") == "pourover vessel"
