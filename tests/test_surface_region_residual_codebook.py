from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryResidualCodebookV1,
)
from radio_gs.losses.surface_region_codebook_loss import (
    gauge_aware_permutation_set_matching_loss,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.train_surface_region_residual_codebook import (
    _assert_checkpoint_training_contract,
    _evaluate,
    _evaluate_v2_control_max,
    _head_descriptors,
)


def _model_and_inputs() -> tuple[
    SurfaceRegionSummaryResidualCodebookV1,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(11)
    model = SurfaceRegionSummaryResidualCodebookV1(
        feature_dim=16,
        hidden_dim=8,
        control_sha256="control",
    )
    features = torch.randn(2, 6, 16)
    geometry = torch.randn(2, 6, 14)
    geometry[..., 8:10] = 0
    geometry[:, :3, 8] = 1
    geometry[:, 3:, 9] = 1
    mask = torch.ones(2, 6, dtype=torch.bool)
    anchor = torch.zeros(2, dtype=torch.long)
    reliability = torch.tensor(
        [
            [[1.0], [0.9], [0.4], [0.2], [0.1], [0.05]],
            [[0.05], [0.1], [0.2], [0.4], [0.9], [1.0]],
        ]
    )
    return model, features, geometry, mask, anchor, reliability


def test_residual_codebook_exactly_preserves_v2_canonical_and_slot0() -> None:
    model, features, geometry, mask, anchor, reliability = _model_and_inputs()
    expected = model.base(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    output = model.forward_codebook(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    assert torch.equal(output.canonical_token, expected)
    assert torch.equal(output.slot_tokens[:, 0], expected)
    assert torch.equal(output.slot_tokens[:, 1:], expected[:, None].expand(-1, 3, -1))
    assert output.slot_priors.tolist() == [[1.0, 0.0, 0.0, 0.0]] * 2


def test_residual_codebook_passes_reliability_to_frozen_v2() -> None:
    model, features, geometry, mask, anchor, reliability = _model_and_inputs()
    with_reliability = model(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    direct = model.base(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    without_reliability = model.base(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
    )
    assert torch.equal(with_reliability, direct)
    assert not torch.equal(with_reliability, without_reliability)


def test_residual_codebook_train_keeps_base_frozen_eval() -> None:
    model, *_ = _model_and_inputs()
    model.train()
    assert model.training
    assert not model.base.training
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert all(parameter.grad is None for parameter in model.base.parameters())


def test_residual_codebook_best_slot_cannot_regress() -> None:
    model, features, geometry, mask, anchor, reliability = _model_and_inputs()
    with torch.no_grad():
        model.residual_head[-1].bias[:16] = torch.linspace(-0.3, 0.3, 16)
    output = model.forward_codebook(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    teacher = F.normalize(torch.randn(2, 3, 16), dim=-1)
    slots = F.normalize(output.slot_tokens, dim=-1)
    fallback = F.normalize(output.canonical_token, dim=-1)
    candidate_best = torch.einsum("bvd,bkd->bvk", teacher, slots).amax(-1)
    control = torch.einsum("bvd,bd->bv", teacher, fallback)
    assert bool((candidate_best >= control).all())


def test_zero_initialized_residual_has_nonzero_set_gradient() -> None:
    model, features, geometry, mask, anchor, reliability = _model_and_inputs()
    output = model.forward_codebook(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    residual = output.slot_tokens[:, 1:]
    teacher = residual[:, :1].detach() * 1.8
    descriptor = F.normalize(residual, dim=-1)
    teacher_descriptor = F.normalize(teacher, dim=-1)
    loss, _ = gauge_aware_permutation_set_matching_loss(
        residual,
        descriptor,
        teacher,
        teacher_descriptor,
        torch.ones(2, 1, dtype=torch.bool),
        token_direction_weight=0.25,
        token_log_norm_weight=0.25,
    )
    loss.backward()
    gradient = model.residual_head[-1].bias.grad
    assert gradient is not None
    assert abs(float(gradient[-1])) > 0


def test_residual_codebook_checkpoint_round_trip(tmp_path) -> None:
    model, features, geometry, mask, anchor, reliability = _model_and_inputs()
    path = tmp_path / "residual.pt"
    torch.save(
        {
            "schema_version": 5,
            "architecture": model.architecture("contract"),
            "state_dict": model.state_dict(),
        },
        path,
    )
    loaded, payload = SurfaceRegionSummaryResidualCodebookV1.from_checkpoint(path)
    assert payload["schema_version"] == 5
    expected = model.forward_codebook(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    actual = loaded.forward_codebook(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    assert torch.equal(actual.canonical_token, expected.canonical_token)
    assert torch.equal(actual.slot_tokens, expected.slot_tokens)


def test_identical_slots_use_bitwise_identical_official_head_path() -> None:
    torch.manual_seed(17)
    head = SigLIP2SummaryHead().eval()
    canonical = torch.randn(2, 1280)
    residual = canonical[:, None, :].expand(-1, 3, -1).clone()
    fallback, descriptors = _head_descriptors(head, canonical, residual)
    assert torch.equal(descriptors[:, 0], fallback)
    assert torch.equal(descriptors[:, 1], fallback)
    assert torch.equal(descriptors[:, 2], fallback)
    assert torch.equal(descriptors[:, 3], fallback)


def test_untrained_codebook_metrics_equal_hard_max_v2_control() -> None:
    torch.manual_seed(23)
    model = SurfaceRegionSummaryResidualCodebookV1(
        feature_dim=1280,
        hidden_dim=8,
        control_sha256="control",
    ).eval()
    head = SigLIP2SummaryHead().eval()
    rows, tokens, views = 4, 6, 3
    features = torch.randn(rows, tokens, 1280)
    geometry = torch.randn(rows, tokens, 14)
    geometry[..., 8:10] = 0
    geometry[:, :3, 8] = 1
    geometry[:, 3:, 9] = 1
    teacher_tokens = torch.randn(rows, views, 1280)
    teacher_descriptors = F.normalize(torch.randn(rows, views, 1536), dim=-1)
    data = {
        "radio_features": features,
        "geometry": geometry,
        "token_mask": torch.ones(rows, tokens, dtype=torch.bool),
        "anchor_index": torch.zeros(rows, dtype=torch.long),
        "reliability": torch.rand(rows, tokens, 1).clamp_min(0.05),
        "official_summary_tokens": teacher_tokens,
        "official_crop_summaries": teacher_descriptors,
        "teacher_mask": torch.ones(rows, views, dtype=torch.bool),
        "scene_ids": ["a", "a", "b", "b"],
    }
    text = F.normalize(torch.randn(5, 1536), dim=-1)
    control = _evaluate_v2_control_max(
        model.base,
        head,
        data,
        text,
        torch.device("cpu"),
        batch_size=2,
    )
    candidate = _evaluate(
        model,
        head,
        data,
        text,
        torch.device("cpu"),
        batch_size=2,
    )
    for key, value in control.items():
        if key == "query_slot_usage":
            continue
        assert candidate[key] == value
    assert candidate["query_slot_usage"] == [1.0, 0.0, 0.0, 0.0]


def test_checkpoint_training_contract_is_bound_fail_closed() -> None:
    payload = {
        "architecture": {"contract_sha256": "contract"},
        "provenance": {
            "region_contract_sha256": "contract",
            "train": {
                "region_contract_sha256": "contract",
                "radio_checkpoint_sha256": "radio",
            },
            "validation": {
                "region_contract_sha256": "contract",
                "radio_checkpoint_sha256": "radio",
            },
        },
    }
    _assert_checkpoint_training_contract(
        payload,
        expected_contract_sha256="contract",
        expected_radio_sha256="radio",
        label="test checkpoint",
    )
    payload["provenance"]["validation"]["region_contract_sha256"] = "other"
    with pytest.raises(ValueError, match="region contract differs"):
        _assert_checkpoint_training_contract(
            payload,
            expected_contract_sha256="contract",
            expected_radio_sha256="radio",
            label="test checkpoint",
        )
