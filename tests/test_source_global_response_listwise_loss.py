from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from radio_gs.evaluation.source_query_response_hard_negatives import (
    build_negative_authority,
)
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.losses.source_global_response_listwise_loss import (
    SourceGlobalResponseLossConfig,
    load_frozen_source_response_authority,
    recommended_v2_config,
    source_global_response_listwise_loss,
    zero_weight_config,
)
from radio_gs.utils.immutable_artifacts import sha256_file


ACCEPTED_SHA = "a" * 64
TEACHER_SHA = "b" * 64
TEACHER_CHANNEL_SHA = "c" * 64
FIT_SHA = "d" * 64


def _fixture(tmp_path: Path):
    teacher = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.4358899, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.4358899],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    generator = torch.Generator().manual_seed(7)
    text = F.normalize(torch.randn(806, 3, generator=generator), dim=-1)
    channels = {
        "anchor_region_indices": torch.tensor([0, 1, 2, 3]),
        "negative_region_indices": torch.tensor([1, 0, 3, 2]),
        "source_codes": torch.tensor([3, 3, 3, 3]),
        "teacher_similarity_ranks": torch.tensor([0, 0, 0, 0]),
        "response_nearest_ranks": torch.tensor([0, 0, 0, 0]),
        "teacher_cosines": torch.tensor([0.9, 0.9, 0.9, 0.9]),
        "response_profile_cosines": torch.tensor([0.95, 0.95, 0.95, 0.95]),
        "anchor_scale_indices": torch.zeros(4, dtype=torch.int64),
        "negative_scale_indices": torch.zeros(4, dtype=torch.int64),
        "row_offsets": torch.arange(5, dtype=torch.int64),
    }
    authority = build_negative_authority(
        scene_id="scene0001_00",
        canonical_region_indices=torch.arange(4),
        region_fingerprints=[f"region-{index}" for index in range(4)],
        channels=channels,
        input_authority={
            "accepted_v2": {"sha256": ACCEPTED_SHA},
            "official_multiview_siglip2_teacher": {
                "sha256": TEACHER_SHA,
                "channel_sha256": {
                    "pair_descriptors": TEACHER_CHANNEL_SHA,
                },
            },
            "fit_text_bank": {
                "split": "fit",
                "queries": 806,
                "sha256": FIT_SHA,
                "embedding_tensor_sha256": tensor_sha256(text),
                "benchmark_vocabulary_opened": False,
                "uses_benchmark_vocabulary_for_construction": False,
            },
        },
    )
    path = tmp_path / "negative.pt"
    torch.save(authority, path)
    digest = sha256_file(path)
    loaded = load_frozen_source_response_authority(
        path,
        expected_file_sha256=digest,
        expected_content_authority_sha256=authority[
            "content_authority_sha256"
        ],
        expected_scene_id="scene0001_00",
        expected_accepted_v2_file_sha256=ACCEPTED_SHA,
        expected_teacher_file_sha256=TEACHER_SHA,
        expected_teacher_pair_descriptors_sha256=TEACHER_CHANNEL_SHA,
        expected_fit_text_bank_file_sha256=FIT_SHA,
    )
    return teacher, text, loaded, path, digest


def _loss(teacher, text, authority, student, base, *, config=None, canonical=None):
    return source_global_response_listwise_loss(
        base,
        student,
        teacher,
        torch.arange(4, dtype=torch.int64),
        text,
        torch.arange(4, dtype=torch.int64) if canonical is None else canonical,
        authority,
        accepted_v2_file_sha256=ACCEPTED_SHA,
        teacher_file_sha256=TEACHER_SHA,
        teacher_pair_descriptors_sha256=TEACHER_CHANNEL_SHA,
        fit_text_bank_file_sha256=FIT_SHA,
        config=config,
    )


def test_zero_weight_returns_base_loss_object_and_exact_gradient_without_auxiliary_reads() -> None:
    value = torch.tensor(2.0, requires_grad=True)
    base = value.square()
    total, metrics = source_global_response_listwise_loss(
        base,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        None,  # type: ignore[arg-type]
        accepted_v2_file_sha256="invalid",
        teacher_file_sha256="invalid",
        teacher_pair_descriptors_sha256="invalid",
        fit_text_bank_file_sha256="invalid",
        config=zero_weight_config(),
    )
    assert total is base
    total.backward()
    assert value.grad.item() == 4.0
    assert all(
        item == 0 or (isinstance(item, torch.Tensor) and item.item() == 0)
        for item in metrics.values()
    )


def test_recommended_config_and_perfect_teacher_are_zero_auxiliary(tmp_path: Path) -> None:
    teacher, text, authority, _, _ = _fixture(tmp_path)
    student = teacher.clone().requires_grad_(True)
    base = student.sum() * 0.0
    total, metrics = _loss(
        teacher, text, authority, student, base, config=recommended_v2_config()
    )
    assert recommended_v2_config() == SourceGlobalResponseLossConfig(
        auxiliary_weight=0.25
    )
    assert metrics["centered_response_profile_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["pairwise_ordering_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["hard_negative_triplet_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert total.item() == pytest.approx(0.0, abs=1e-6)


def test_candidate_receives_gradients_but_teacher_and_text_remain_frozen(tmp_path: Path) -> None:
    teacher, text, authority, _, _ = _fixture(tmp_path)
    teacher.requires_grad_(True)
    text.requires_grad_(True)
    student = torch.roll(teacher.detach(), shifts=1, dims=0).requires_grad_(True)
    base = student.sum() * 0.0
    total, metrics = _loss(teacher, text, authority, student, base)
    assert total.item() > 0
    assert metrics["valid_profile_queries"] == 806
    assert metrics["valid_pair_query_units"] > 0
    assert metrics["hard_negative_pairs"] == 4
    total.backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0
    assert teacher.grad is None
    assert text.grad is None


def test_partial_scene_and_runtime_hash_drift_fail_closed(tmp_path: Path) -> None:
    teacher, text, authority, _, _ = _fixture(tmp_path)
    student = teacher.clone().requires_grad_(True)
    base = student.sum() * 0.0
    with pytest.raises(ValueError, match="complete canonical scene"):
        _loss(
            teacher,
            text,
            authority,
            student[:3],
            base,
            canonical=torch.arange(3),
        )
    with pytest.raises(ValueError, match="runtime accepted_v2 binding"):
        source_global_response_listwise_loss(
            base,
            student,
            teacher,
            torch.arange(4),
            text,
            torch.arange(4),
            authority,
            accepted_v2_file_sha256="e" * 64,
            teacher_file_sha256=TEACHER_SHA,
            teacher_pair_descriptors_sha256=TEACHER_CHANNEL_SHA,
            fit_text_bank_file_sha256=FIT_SHA,
        )


def test_file_and_content_hashes_are_both_required(tmp_path: Path) -> None:
    _, _, authority, path, digest = _fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_frozen_source_response_authority(
            path,
            expected_file_sha256="f" * 64,
            expected_content_authority_sha256=authority.content_authority_sha256,
            expected_scene_id=authority.scene_id,
            expected_accepted_v2_file_sha256=ACCEPTED_SHA,
            expected_teacher_file_sha256=TEACHER_SHA,
            expected_teacher_pair_descriptors_sha256=TEACHER_CHANNEL_SHA,
            expected_fit_text_bank_file_sha256=FIT_SHA,
        )
    with pytest.raises(ValueError, match="content authority differs"):
        load_frozen_source_response_authority(
            path,
            expected_file_sha256=digest,
            expected_content_authority_sha256="f" * 64,
            expected_scene_id=authority.scene_id,
            expected_accepted_v2_file_sha256=ACCEPTED_SHA,
            expected_teacher_file_sha256=TEACHER_SHA,
            expected_teacher_pair_descriptors_sha256=TEACHER_CHANNEL_SHA,
            expected_fit_text_bank_file_sha256=FIT_SHA,
        )
