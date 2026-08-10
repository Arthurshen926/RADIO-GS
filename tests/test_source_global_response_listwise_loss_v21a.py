from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.evaluation.source_query_response_hard_negatives import (
    build_negative_authority,
)
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.losses.source_global_response_listwise_loss import (
    load_frozen_source_response_authority,
)
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    CANONICAL_NEGATIVE_MODEL,
    load_frozen_canonical_negative_bank,
)
from radio_gs.losses.source_global_response_listwise_loss_v21a import (
    source_global_response_listwise_loss_v21a,
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
            ]
        ),
        dim=-1,
    )
    generator = torch.Generator().manual_seed(31)
    text = F.normalize(torch.randn(806, 3, generator=generator), dim=-1)
    channels = {
        "anchor_region_indices": torch.tensor([0, 1, 2, 3]),
        "negative_region_indices": torch.tensor([1, 0, 3, 2]),
        "source_codes": torch.tensor([3, 3, 3, 3]),
        "teacher_similarity_ranks": torch.zeros(4, dtype=torch.int64),
        "response_nearest_ranks": torch.zeros(4, dtype=torch.int64),
        "teacher_cosines": torch.tensor([0.9, 0.9, 0.9, 0.9]),
        "response_profile_cosines": torch.tensor([0.95, 0.95, 0.95, 0.95]),
        "anchor_scale_indices": torch.zeros(4, dtype=torch.int64),
        "negative_scale_indices": torch.zeros(4, dtype=torch.int64),
        "row_offsets": torch.arange(5, dtype=torch.int64),
    }
    raw = build_negative_authority(
        scene_id="scene0001_00",
        canonical_region_indices=torch.arange(4),
        region_fingerprints=[f"region-{index}" for index in range(4)],
        channels=channels,
        input_authority={
            "accepted_v2": {"sha256": ACCEPTED_SHA},
            "official_multiview_siglip2_teacher": {
                "sha256": TEACHER_SHA,
                "channel_sha256": {"pair_descriptors": TEACHER_CHANNEL_SHA},
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
    authority_path = tmp_path / "authority.pt"
    torch.save(raw, authority_path)
    authority = load_frozen_source_response_authority(
        authority_path,
        expected_file_sha256=sha256_file(authority_path),
        expected_content_authority_sha256=raw["content_authority_sha256"],
        expected_scene_id="scene0001_00",
        expected_accepted_v2_file_sha256=ACCEPTED_SHA,
        expected_teacher_file_sha256=TEACHER_SHA,
        expected_teacher_pair_descriptors_sha256=TEACHER_CHANNEL_SHA,
        expected_fit_text_bank_file_sha256=FIT_SHA,
    )
    negatives = torch.zeros(4, 3)
    negatives[:, -1] = 1.0
    negative_path = tmp_path / "negative.pt"
    torch.save(
        {
            "queries": [f"negative-{index}" for index in range(4)],
            "prompt_templates": ["{query}"],
            "text_encoder": "siglip2",
            "model_name": CANONICAL_NEGATIVE_MODEL,
            "embeddings": negatives,
        },
        negative_path,
    )
    negative_bank = load_frozen_canonical_negative_bank(
        negative_path, expected_file_sha256=sha256_file(negative_path)
    )
    return teacher, text, authority, negative_bank


def _loss(tmp_path: Path, trainable: torch.Tensor, *, training: bool):
    teacher, text, authority, negative = _fixture(tmp_path)
    student = torch.flip(teacher, dims=(0,)).clone().requires_grad_(True)
    return source_global_response_listwise_loss_v21a(
        student.sum() * 0.0,
        student,
        teacher,
        torch.arange(4),
        text,
        torch.arange(4),
        authority,
        negative,
        accepted_v2_file_sha256=ACCEPTED_SHA,
        teacher_file_sha256=TEACHER_SHA,
        teacher_pair_descriptors_sha256=TEACHER_CHANNEL_SHA,
        fit_text_bank_file_sha256=FIT_SHA,
        trainable_region_mask=trainable,
        training=training,
    )


def test_training_triplet_requires_trainable_anchor_but_pairwise_keeps_either_endpoint(
    tmp_path: Path,
) -> None:
    _, metrics = _loss(
        tmp_path, torch.tensor([True, False, False, False]), training=True
    )
    assert metrics["pairwise_objective_hard_negative_pairs"] == 2
    assert metrics["triplet_objective_hard_negative_pairs"] == 1
    assert metrics["triplet_nonanchor_only_pairs_excluded"] == 1
    assert float(metrics["triplet_anchor_trainable_coverage"]) == 0.25


def test_validation_retains_all_pairs_for_both_terms(tmp_path: Path) -> None:
    _, metrics = _loss(
        tmp_path, torch.tensor([True, False, False, False]), training=False
    )
    assert metrics["pairwise_objective_hard_negative_pairs"] == 4
    assert metrics["triplet_objective_hard_negative_pairs"] == 4
    assert float(metrics["triplet_anchor_trainable_coverage"]) == 1.0


def test_anchor_only_triplet_still_backpropagates(tmp_path: Path) -> None:
    total, metrics = _loss(
        tmp_path, torch.tensor([True, False, False, False]), training=True
    )
    assert torch.isfinite(total)
    assert torch.isfinite(metrics["hard_negative_triplet_loss"])
    total.backward()
