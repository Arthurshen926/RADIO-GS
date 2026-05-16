import torch

from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_alignment_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_cross_view_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_mask_logit_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_region_loss
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_relation_loss


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
