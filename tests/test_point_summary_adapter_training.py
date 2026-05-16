from __future__ import annotations

import torch
import torch.nn.functional as F

from radio_gs.scripts.train_scannet_point_summary_adapter import (
    _build_teacher_sample_weights,
    compute_adapter_training_loss,
    compute_text_rank_distillation_loss,
    merge_adapter_checkpoint,
    _load_teacher_cache_class_names,
    _load_teacher_cache,
    _prepare_teacher_summary,
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


def test_adapter_summary_loss_uses_sample_weights():
    compact = torch.eye(2)
    teacher_summary = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), dim=-1)
    adapter = torch.nn.Linear(2, 2, bias=False)
    adapter.weight.data.copy_(torch.eye(2))

    loss, stats = compute_adapter_training_loss(
        compact,
        teacher_summary,
        adapter,
        summary_weight=1.0,
        text_distill_weight=0.0,
        sample_weights=torch.tensor([1.0, 0.0]),
        detach_compact=True,
    )

    assert torch.allclose(loss, torch.tensor(0.0))
    assert torch.allclose(stats["summary_loss"], torch.tensor(0.0))
    assert torch.allclose(stats["sample_weight_mean"], torch.tensor(0.5))


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


def test_teacher_cache_accepts_registered_summary_features(tmp_path):
    path = tmp_path / "teacher.pt"
    summary = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), dim=-1)
    torch.save(
        {
            "xyz": torch.arange(9, dtype=torch.float32).view(3, 3),
            "summary_features": summary,
            "valid": torch.tensor([True, False, True]),
        },
        path,
    )

    teacher = _load_teacher_cache(str(path), torch.device("cpu"), valid_only=True)

    assert torch.equal(teacher["indices"], torch.tensor([0, 2]))
    assert "features" not in teacher
    assert torch.allclose(teacher["summary_features"], summary[[0, 2]])


def test_teacher_cache_loads_registration_view_counts(tmp_path):
    path = tmp_path / "teacher.pt"
    torch.save(
        {
            "xyz": torch.arange(12, dtype=torch.float32).view(4, 3),
            "summary_features": torch.zeros(4, 2),
            "valid": torch.tensor([True, False, True, True]),
            "view_counts": torch.tensor([4, 0, 16, 64]),
        },
        path,
    )

    teacher = _load_teacher_cache(str(path), torch.device("cpu"), valid_only=True)
    weights = _build_teacher_sample_weights(
        teacher,
        mode="log",
        min_weight=0.0,
    )

    assert torch.equal(teacher["view_counts"], torch.tensor([4.0, 16.0, 64.0]))
    assert weights.shape == torch.Size([3])
    assert torch.all(weights > 0)
    assert torch.allclose(weights.mean(), torch.tensor(1.0))
    assert weights[0] < weights[1] < weights[2]


def test_teacher_sample_weights_support_clipped_log_mode():
    cache = {
        "view_counts": torch.tensor([0.0, 1.0, 10.0, 100.0]),
        "valid": torch.tensor([False, True, True, True]),
    }
    weights = _build_teacher_sample_weights(
        cache,
        mode="clipped_log",
        min_weight=0.25,
        percentile_low=0.0,
        percentile_high=75.0,
    )
    assert weights[0].item() == 0.0
    assert weights[1:].min().item() >= 0.25
    assert weights.max().item() <= 1.0


def test_adapter_text_rank_loss_matches_teacher_ordering():
    pred = torch.tensor([[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]])
    teacher = torch.tensor([[2.0, 0.0, -2.0], [0.0, 1.0, -1.0]])
    loss, stats = compute_text_rank_distillation_loss(
        pred,
        teacher,
        sample_weights=torch.tensor([1.0, 0.5]),
        margin=0.1,
        topk=2,
    )
    assert loss.item() >= 0.0
    assert stats["rank_pairs"].item() > 0


def test_prepare_teacher_summary_uses_cached_summary_without_projection():
    summary = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 1.0]]), dim=-1)
    teacher = {"summary_features": summary.clone()}
    projection = torch.nn.Linear(2, 2)

    prepared = _prepare_teacher_summary(
        teacher,
        projection,
        chunk_size=1,
        device=torch.device("cpu"),
    )

    assert torch.allclose(prepared, summary)
    assert "summary_features" not in teacher


def test_load_teacher_cache_class_names_reads_lerf_metadata(tmp_path):
    path = tmp_path / "teacher.pt"
    torch.save(
        {
            "xyz": torch.zeros(1, 3),
            "summary_features": torch.zeros(1, 2),
            "valid": torch.tensor([True]),
            "metadata": {"categories": ["ramen bowl", "chopsticks"]},
        },
        path,
    )

    assert _load_teacher_cache_class_names(str(path)) == ["ramen bowl", "chopsticks"]
