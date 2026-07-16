import pytest
import torch

from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_alignment_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_cross_view_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_cross_view_mask_propagation_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_cross_view_propagation_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_local_affinity_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_mask_logit_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_masked_render_losses
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_peak_background_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_region_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_relation_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_token_contrast_loss


class ScaleAdaptor(torch.nn.Module):
    def forward(self, x):
        return x * 2.0


class IdentityAdaptor(torch.nn.Module):
    def forward(self, x):
        return x


def test_compute_radio_adaptor_alignment_loss_returns_zero_without_adaptors():
    decoded = torch.randn(1, 4, 2, 2)
    target = decoded.clone()

    loss, stats = compute_radio_adaptor_alignment_loss(decoded, target, {})

    assert loss.item() == 0.0
    assert stats == {}


def test_compute_radio_adaptor_alignment_loss_matches_identical_features():
    decoded = torch.randn(1, 4, 2, 2)
    target = decoded.clone()

    loss, stats = compute_radio_adaptor_alignment_loss(
        decoded,
        target,
        {"sam3": ScaleAdaptor()},
    )

    assert loss.item() < 1e-6
    assert "sam3" in stats


def test_masked_render_losses_match_identical_visible_capability() -> None:
    decoded = torch.randn(1, 4, 3, 3)
    valid = torch.ones(1, 3, 3, dtype=torch.bool)

    alignment, local, stats = compute_radio_adaptor_masked_render_losses(
        decoded,
        decoded.clone(),
        {"sam3": IdentityAdaptor()},
        valid,
    )

    assert alignment.item() < 1e-6
    assert local.item() < 1e-6
    assert stats["sam3"]["visible_pixels"].item() == 9
    assert stats["sam3"]["visible_pairs"].item() == 12


def test_masked_render_losses_ignore_unsupported_pixels_and_pairs() -> None:
    target = torch.randn(1, 4, 3, 3)
    decoded = target.clone()
    decoded[:, :, 1, 1] = -target[:, :, 1, 1]
    valid = torch.ones(1, 3, 3, dtype=torch.bool)
    valid[:, 1, 1] = False

    alignment, local, _ = compute_radio_adaptor_masked_render_losses(
        decoded,
        target,
        {"sam3": IdentityAdaptor()},
        valid,
    )

    assert alignment.item() < 1e-6
    assert local.item() < 1e-6


def test_masked_render_losses_support_teacher_boundary_balancing() -> None:
    torch.manual_seed(7)
    target = torch.randn(1, 4, 4, 4)
    decoded = target.clone()
    decoded[:, :, :, 2:] = decoded[:, :, :, 2:].flip(-1)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    _alignment, local, stats = compute_radio_adaptor_masked_render_losses(
        decoded,
        target,
        {"sam3": IdentityAdaptor()},
        valid,
        local_balance_quantile=0.2,
    )
    assert torch.isfinite(local)
    assert local.item() > 0
    assert stats["sam3"]["local_balance_quantile"].item() == pytest.approx(0.2)


def test_compute_radio_adaptor_relation_loss_matches_identical_pairwise_structure():
    decoded = torch.randn(1, 4, 3, 3)

    loss, stats = compute_radio_adaptor_relation_loss(
        decoded,
        decoded.clone(),
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
    )

    assert loss.item() < 1e-6
    assert "dino_v3" in stats


def test_compute_radio_adaptor_relation_loss_detects_spatial_structure_changes():
    target = torch.randn(1, 4, 3, 3)
    decoded = target.flip(-1)

    loss, _ = compute_radio_adaptor_relation_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
    )

    assert loss.item() > 1e-4


def test_compute_radio_adaptor_local_affinity_loss_matches_teacher_neighborhoods():
    decoded = torch.randn(1, 4, 4, 4)

    loss, stats = compute_radio_adaptor_local_affinity_loss(
        decoded,
        decoded.clone(),
        {"dino_v3": IdentityAdaptor()},
        downsample=1,
        radius=1,
    )

    assert loss.item() < 1e-6
    assert "dino_v3" in stats


def test_compute_radio_adaptor_local_affinity_loss_detects_local_topology_changes():
    torch.manual_seed(0)
    target = torch.randn(1, 4, 4, 4)
    decoded = target.clone()
    decoded[..., 1:3, 1:3] = decoded[..., 1:3, 1:3].flip(-1)

    loss, _ = compute_radio_adaptor_local_affinity_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        downsample=1,
        radius=1,
    )

    assert loss.item() > 1e-4


def test_compute_radio_adaptor_region_loss_matches_teacher_soft_regions():
    decoded = torch.randn(1, 4, 4, 4)

    loss, stats = compute_radio_adaptor_region_loss(
        decoded,
        decoded.clone(),
        {"sam3": IdentityAdaptor()},
        num_anchors=4,
    )

    assert loss.item() < 1e-6
    assert "sam3" in stats


def test_compute_radio_adaptor_mask_logit_loss_matches_teacher_logits():
    decoded = torch.randn(1, 4, 4, 4)

    loss, stats = compute_radio_adaptor_mask_logit_loss(
        decoded,
        decoded.clone(),
        {"sam3": IdentityAdaptor()},
        num_anchors=4,
    )

    assert loss.item() < 1e-6
    assert "sam3" in stats


def test_compute_radio_adaptor_mask_logit_loss_detects_assignment_changes():
    target = torch.randn(1, 4, 4, 4)
    decoded = target.flip(-1)

    loss, _ = compute_radio_adaptor_mask_logit_loss(
        decoded,
        target,
        {"sam3": IdentityAdaptor()},
        num_anchors=4,
        temperature=0.1,
    )

    assert loss.item() > 1e-4


def test_compute_radio_adaptor_cross_view_loss_matches_identical_cross_view_structure():
    decoded = torch.randn(2, 4, 3, 3)

    loss, stats = compute_radio_adaptor_cross_view_loss(
        decoded,
        decoded.clone(),
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
    )

    assert loss.item() < 1e-6
    assert "dino_v3" in stats


def test_compute_radio_adaptor_cross_view_loss_detects_pair_structure_changes():
    target = torch.randn(2, 4, 3, 3)
    decoded = target.clone()
    decoded[1] = decoded[1].flip(-1)

    loss, _ = compute_radio_adaptor_cross_view_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
    )

    assert loss.item() > 1e-4


def test_compute_radio_adaptor_cross_view_transport_cycle_matches_teacher_topology():
    decoded = torch.randn(2, 4, 3, 3)

    loss, stats = compute_radio_adaptor_cross_view_loss(
        decoded,
        decoded.clone(),
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
        objective="transport_cycle",
    )

    assert loss.item() < 1e-6
    assert "dino_v3" in stats


def test_compute_radio_adaptor_cross_view_transport_cycle_detects_topology_changes():
    target = torch.randn(2, 4, 3, 3)
    decoded = target.clone()
    decoded[1] = decoded[1].flip(-1)

    loss, _ = compute_radio_adaptor_cross_view_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
        objective="transport_cycle",
    )

    assert loss.item() > 1e-4


def test_compute_radio_adaptor_cross_view_propagation_loss_matches_teacher_maps():
    decoded = torch.randn(2, 4, 3, 3)

    loss, stats = compute_radio_adaptor_cross_view_propagation_loss(
        decoded,
        decoded.clone(),
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
        num_anchors=4,
        temperature=0.2,
    )

    assert loss.item() < 1e-6
    assert "dino_v3" in stats


def test_compute_radio_adaptor_cross_view_propagation_loss_detects_target_swaps():
    target = torch.randn(2, 4, 3, 3)
    decoded = target.clone()
    decoded[1] = decoded[1].flip(-1)

    loss, _ = compute_radio_adaptor_cross_view_propagation_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
        num_anchors=4,
        temperature=0.2,
    )

    assert loss.item() > 1e-4


def test_compute_radio_adaptor_cross_view_propagation_loss_distinctive_anchors_focus_confident_tokens():
    target = torch.zeros(2, 4, 1, 5)
    target[:, 0, 0, [0, 1, 4]] = 1.0
    target[:, 1, 0, 2] = 1.0
    target[:, 2, 0, 3] = 1.0
    decoded = target.clone()
    decoded[0, :, 0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    decoded[0, :, 0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    linspace_loss, _ = compute_radio_adaptor_cross_view_propagation_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=5,
        num_anchors=2,
        temperature=0.2,
        anchor_strategy="linspace",
    )
    distinctive_loss, _ = compute_radio_adaptor_cross_view_propagation_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=5,
        num_anchors=2,
        temperature=0.2,
        anchor_strategy="distinctive",
    )

    assert linspace_loss.item() < 1e-6
    assert distinctive_loss.item() > 1e-3


def test_compute_radio_adaptor_cross_view_mask_propagation_loss_matches_teacher_maps():
    decoded = torch.randn(2, 4, 3, 3)

    loss, stats = compute_radio_adaptor_cross_view_mask_propagation_loss(
        decoded,
        decoded.clone(),
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
        num_anchors=4,
        temperature=0.2,
    )

    assert loss.item() < 1e-6
    assert "dino_v3" in stats


def test_compute_radio_adaptor_cross_view_mask_propagation_loss_detects_transport_changes():
    target = torch.randn(2, 4, 3, 3)
    decoded = target.clone()
    decoded[1] = decoded[1].flip(-1)

    loss, _ = compute_radio_adaptor_cross_view_mask_propagation_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
        num_anchors=4,
        temperature=0.2,
    )

    assert loss.item() > 1e-4


def test_compute_radio_adaptor_token_contrast_loss_matches_same_token_teacher():
    decoded = torch.randn(1, 4, 3, 3)

    loss, stats = compute_radio_adaptor_token_contrast_loss(
        decoded,
        decoded.clone(),
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
        temperature=0.07,
    )

    assert loss.item() < 1e-4
    assert "dino_v3" in stats


def test_compute_radio_adaptor_token_contrast_loss_penalizes_hard_negative_swaps():
    target = torch.randn(1, 4, 3, 3)
    decoded = target.clone()
    decoded[..., 0], decoded[..., -1] = target[..., -1], target[..., 0]

    loss, _ = compute_radio_adaptor_token_contrast_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
        temperature=0.07,
    )

    assert loss.item() > 0.1


def test_compute_radio_adaptor_peak_background_loss_matches_teacher_margins():
    decoded = torch.randn(1, 4, 3, 3)

    loss, stats = compute_radio_adaptor_peak_background_loss(
        decoded,
        decoded.clone(),
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
        num_anchors=4,
        temperature=0.2,
    )

    assert loss.item() < 1e-6
    assert "dino_v3" in stats


def test_compute_radio_adaptor_peak_background_loss_penalizes_peak_collapse():
    target = torch.randn(1, 4, 3, 3)
    decoded = torch.zeros_like(target)

    loss, _ = compute_radio_adaptor_peak_background_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=9,
        num_anchors=4,
        temperature=0.2,
    )

    assert loss.item() > 1e-3


def test_compute_radio_adaptor_peak_background_loss_uses_hard_negative():
    target = torch.tensor(
        [[[[1.0, 0.98, -1.0]], [[0.0, 0.20, 0.0]]]],
        dtype=torch.float32,
    )
    decoded = target.clone()
    decoded[:, :, :, 1] = decoded[:, :, :, 0]

    loss, _ = compute_radio_adaptor_peak_background_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=3,
        num_anchors=1,
        temperature=0.05,
    )

    assert loss.item() > 0.02


def test_compute_radio_adaptor_peak_background_loss_distinctive_anchors_focus_confident_tokens():
    target = torch.zeros(1, 4, 1, 5)
    target[:, 0, 0, [0, 1, 4]] = 1.0
    target[:, 1, 0, 2] = 1.0
    target[:, 2, 0, 3] = 1.0
    decoded = target.clone()
    decoded[0, :, 0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    decoded[0, :, 0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    linspace_loss, _ = compute_radio_adaptor_peak_background_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=5,
        num_anchors=2,
        temperature=0.2,
        anchor_strategy="linspace",
    )
    distinctive_loss, _ = compute_radio_adaptor_peak_background_loss(
        decoded,
        target,
        {"dino_v3": IdentityAdaptor()},
        max_tokens=5,
        num_anchors=2,
        temperature=0.2,
        anchor_strategy="distinctive",
    )

    assert linspace_loss.item() < 1e-6
    assert distinctive_loss.item() > 1e-3
