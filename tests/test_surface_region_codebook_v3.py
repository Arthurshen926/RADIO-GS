from __future__ import annotations

import torch

from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryCodebookV3,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    features = torch.randn(2, 6, 16)
    geometry = torch.randn(2, 6, 14)
    geometry[..., 8:10] = 0
    geometry[:, :3, 8] = 1
    geometry[:, 3:, 9] = 1
    mask = torch.ones(2, 6, dtype=torch.bool)
    anchor = torch.zeros(2, dtype=torch.long)
    return features, geometry, mask, anchor


def test_v3_codebook_is_direction_identified_and_normalized() -> None:
    model = SurfaceRegionSummaryCodebookV3(
        feature_dim=16,
        hidden_dim=8,
        slots=3,
    ).eval()
    features, geometry, mask, anchor = _inputs()
    first = model.forward_codebook(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
    )
    scales = torch.linspace(0.25, 4.0, 6).view(1, 6, 1)
    second = model.forward_codebook(
        features * scales,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
    )
    assert first.slot_tokens.shape == (2, 3, 16)
    assert first.slot_priors.shape == (2, 3)
    assert torch.allclose(first.slot_priors.sum(-1), torch.ones(2))
    assert torch.allclose(first.canonical_token, second.canonical_token, atol=1e-6)
    assert torch.allclose(first.slot_tokens, second.slot_tokens, atol=1e-6)


def test_v3_forward_is_the_prior_marginal_token() -> None:
    model = SurfaceRegionSummaryCodebookV3(
        feature_dim=16,
        hidden_dim=8,
        slots=3,
    ).eval()
    features, geometry, mask, anchor = _inputs()
    codebook = model.forward_codebook(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
    )
    direct = model(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=torch.rand(2, 6, 1),
    )
    expected = torch.einsum(
        "bk,bkd->bd", codebook.slot_priors, codebook.slot_tokens
    )
    assert torch.equal(direct, codebook.canonical_token)
    assert torch.allclose(direct, expected)


def test_v3_rejects_overlapping_core_context_flags() -> None:
    model = SurfaceRegionSummaryCodebookV3(
        feature_dim=16,
        hidden_dim=8,
        slots=3,
    )
    features, geometry, mask, anchor = _inputs()
    geometry[:, 0, 9] = 1
    try:
        model.forward_codebook(
            features,
            geometry,
            anchor_index=anchor,
            token_mask=mask,
        )
    except ValueError as error:
        assert "disjoint core/context" in str(error)
    else:  # pragma: no cover
        raise AssertionError("overlapping core/context flags were accepted")


def test_v3_checkpoint_round_trip(tmp_path) -> None:
    model = SurfaceRegionSummaryCodebookV3(
        feature_dim=16,
        hidden_dim=8,
        slots=3,
    ).eval()
    path = tmp_path / "readout.pt"
    torch.save(
        {
            "schema_version": 4,
            "architecture": model.architecture("contract"),
            "state_dict": model.state_dict(),
        },
        path,
    )
    loaded, payload = SurfaceRegionSummaryCodebookV3.from_checkpoint(path)
    assert payload["schema_version"] == 4
    features, geometry, mask, anchor = _inputs()
    expected = model(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
    )
    actual = loaded(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
    )
    assert torch.equal(actual, expected)
