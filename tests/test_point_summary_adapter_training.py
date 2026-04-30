from __future__ import annotations

import torch
import torch.nn.functional as F

from radio_gs.scripts.train_scannet_point_summary_adapter import (
    compute_adapter_training_loss,
    merge_adapter_checkpoint,
)


def test_adapter_loss_can_detach_compact_features_from_field_gradients():
    compact = torch.eye(2, requires_grad=True)
    teacher_summary = F.normalize(torch.flip(torch.eye(2), dims=[1]), dim=-1)
    adapter = torch.nn.Linear(2, 2, bias=False)
    adapter.weight.data.copy_(torch.eye(2))

    loss, stats = compute_adapter_training_loss(
        compact,
        teacher_summary,
        adapter,
        summary_weight=1.0,
        text_distill_weight=0.0,
        detach_compact=True,
    )
    loss.backward()

    assert compact.grad is None
    assert adapter.weight.grad is not None
    assert adapter.weight.grad.abs().sum() > 0
    assert torch.allclose(stats["summary_loss"], torch.tensor(1.0))


def test_adapter_text_distill_matches_teacher_distribution():
    compact = torch.eye(2)
    teacher_summary = F.normalize(torch.flip(torch.eye(2), dims=[1]), dim=-1)
    adapter = torch.nn.Linear(2, 2, bias=False)
    adapter.weight.data.copy_(torch.eye(2))
    text_embeddings = F.normalize(torch.eye(2), dim=-1)

    loss, stats = compute_adapter_training_loss(
        compact,
        teacher_summary,
        adapter,
        text_embeddings=text_embeddings,
        summary_weight=0.0,
        text_distill_weight=1.0,
        text_distill_temperature=1.0,
        detach_compact=True,
    )

    expected = F.kl_div(
        F.log_softmax(torch.eye(2), dim=-1),
        F.softmax(torch.flip(torch.eye(2), dims=[1]), dim=-1),
        reduction="batchmean",
    )
    assert torch.allclose(loss, expected)
    assert torch.allclose(stats["text_distill_loss"], expected)
    assert torch.allclose(stats["text_distill_agreement"], torch.tensor(0.0))


def test_adapter_text_pseudo_ce_matches_teacher_argmax_targets():
    compact = torch.eye(2)
    teacher_summary = F.normalize(torch.flip(torch.eye(2), dims=[1]), dim=-1)
    adapter = torch.nn.Linear(2, 2, bias=False)
    adapter.weight.data.copy_(torch.eye(2))
    text_embeddings = F.normalize(torch.eye(2), dim=-1)

    loss, stats = compute_adapter_training_loss(
        compact,
        teacher_summary,
        adapter,
        text_embeddings=text_embeddings,
        summary_weight=0.0,
        text_distill_weight=0.0,
        text_pseudo_ce_weight=1.0,
        detach_compact=True,
    )

    expected = F.cross_entropy(torch.eye(2), torch.tensor([1, 0]))
    assert torch.allclose(loss, expected)
    assert torch.allclose(stats["text_pseudo_ce_loss"], expected)
    assert torch.allclose(stats["text_pseudo_ce_valid_ratio"], torch.tensor(1.0))
    assert torch.allclose(stats["text_pseudo_ce_agreement"], torch.tensor(0.0))


def test_adapter_text_pseudo_ce_confidence_threshold_filters_points():
    compact = torch.eye(2)
    teacher_summary = F.normalize(torch.flip(torch.eye(2), dims=[1]), dim=-1)
    adapter = torch.nn.Linear(2, 2, bias=False)
    adapter.weight.data.copy_(torch.eye(2))
    text_embeddings = F.normalize(torch.eye(2), dim=-1)

    loss, stats = compute_adapter_training_loss(
        compact,
        teacher_summary,
        adapter,
        text_embeddings=text_embeddings,
        summary_weight=0.0,
        text_distill_weight=0.0,
        text_pseudo_ce_weight=1.0,
        text_pseudo_ce_confidence_threshold=0.99,
        detach_compact=True,
    )

    assert torch.allclose(loss, torch.tensor(0.0))
    assert torch.allclose(stats["text_pseudo_ce_loss"], torch.tensor(0.0))
    assert torch.allclose(stats["text_pseudo_ce_valid_ratio"], torch.tensor(0.0))


def test_adapter_decoder_anchor_loss_keeps_adapter_close_to_decoded_summary():
    compact = torch.eye(2)
    teacher_summary = F.normalize(torch.eye(2), dim=-1)
    decoder_anchor_summary = F.normalize(torch.flip(torch.eye(2), dims=[1]), dim=-1)
    adapter = torch.nn.Linear(2, 2, bias=False)
    adapter.weight.data.copy_(torch.eye(2))

    loss, stats = compute_adapter_training_loss(
        compact,
        teacher_summary,
        adapter,
        summary_weight=0.0,
        text_distill_weight=0.0,
        decoder_anchor_summary=decoder_anchor_summary,
        decoder_anchor_weight=2.0,
        detach_compact=True,
    )

    expected_anchor = torch.tensor(1.0)
    assert torch.allclose(stats["decoder_anchor_loss"], expected_anchor)
    assert torch.allclose(loss, expected_anchor * 2.0)


def test_merge_adapter_checkpoint_preserves_base_model_state():
    base = {
        "epoch": 30,
        "model_state_dict": {"x": torch.tensor([1.0])},
        "codec_state_dict": {"y": torch.tensor([2.0])},
    }
    adapter_state = {"net.0.weight": torch.tensor([3.0])}

    merged = merge_adapter_checkpoint(
        base,
        adapter_state,
        metadata={"scene": "scene0000_00"},
        epoch=5,
        best_metric=0.25,
    )

    assert merged["model_state_dict"] is base["model_state_dict"]
    assert merged["codec_state_dict"] is base["codec_state_dict"]
    assert torch.equal(merged["point_summary_adapter_state_dict"]["net.0.weight"], torch.tensor([3.0]))
    assert merged["point_summary_adapter_metadata"]["scene"] == "scene0000_00"
    assert merged["point_summary_adapter_best_metric"] == 0.25
    assert merged["point_summary_adapter_epoch"] == 5
