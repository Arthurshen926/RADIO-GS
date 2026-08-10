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
    load_frozen_source_response_authority,
)
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    CANONICAL_NEGATIVE_MODEL,
    O4_TEACHER_RESPONSE_TEMPERATURE,
    load_frozen_canonical_negative_bank,
    load_frozen_compositional_generic_bank,
    source_global_response_listwise_loss_v21,
    zero_weight_v21_config,
)
from radio_gs.utils.immutable_artifacts import sha256_file


ACCEPTED_SHA = "a" * 64
TEACHER_SHA = "b" * 64
TEACHER_CHANNEL_SHA = "c" * 64
FIT_SHA = "d" * 64
FROZEN_V2_SHA = "552e7bf0e4d83e9346af731e6ce9eaf891968b14f32b49f728b5188c5e012ae7"


def _canonical_negative_bank(tmp_path: Path, dimension: int):
    embeddings = torch.zeros(4, dimension, dtype=torch.float32)
    embeddings[:, -1] = 1.0
    payload = {
        "queries": [f"generic-negative-{index}" for index in range(4)],
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": CANONICAL_NEGATIVE_MODEL,
        "embeddings": embeddings,
    }
    path = tmp_path / "canonical_negative.pt"
    torch.save(payload, path)
    return load_frozen_canonical_negative_bank(
        path, expected_file_sha256=sha256_file(path)
    )


def _fixture(
    tmp_path: Path,
    *,
    teacher: torch.Tensor | None = None,
    text: torch.Tensor | None = None,
):
    if teacher is None:
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
    if text is None:
        generator = torch.Generator().manual_seed(17)
        text = F.normalize(
            torch.randn(806, teacher.shape[1], generator=generator), dim=-1
        )
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
    raw_authority = build_negative_authority(
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
    path = tmp_path / "negative_authority.pt"
    torch.save(raw_authority, path)
    authority = load_frozen_source_response_authority(
        path,
        expected_file_sha256=sha256_file(path),
        expected_content_authority_sha256=raw_authority["content_authority_sha256"],
        expected_scene_id="scene0001_00",
        expected_accepted_v2_file_sha256=ACCEPTED_SHA,
        expected_teacher_file_sha256=TEACHER_SHA,
        expected_teacher_pair_descriptors_sha256=TEACHER_CHANNEL_SHA,
        expected_fit_text_bank_file_sha256=FIT_SHA,
    )
    canonical_negative = _canonical_negative_bank(tmp_path, teacher.shape[1])
    return teacher, text, authority, canonical_negative


def _loss(
    teacher,
    text,
    authority,
    canonical_negative,
    student,
    *,
    compositional_banks=(),
    trainable_region_mask=None,
    exclude_both_immutable_pairs=False,
    config=None,
):
    base = student.sum() * 0.0
    return source_global_response_listwise_loss_v21(
        base,
        student,
        teacher,
        torch.arange(4, dtype=torch.int64),
        text,
        torch.arange(4, dtype=torch.int64),
        authority,
        canonical_negative,
        accepted_v2_file_sha256=ACCEPTED_SHA,
        teacher_file_sha256=TEACHER_SHA,
        teacher_pair_descriptors_sha256=TEACHER_CHANNEL_SHA,
        fit_text_bank_file_sha256=FIT_SHA,
        compositional_banks=compositional_banks,
        trainable_region_mask=trainable_region_mask,
        exclude_both_immutable_pairs=exclude_both_immutable_pairs,
        config=config,
    )


def test_training_pair_denominator_excludes_only_both_immutable_pairs(
    tmp_path: Path,
) -> None:
    teacher, text, authority, canonical_negative = _fixture(tmp_path)
    trainable = torch.tensor([True, False, False, False])
    student = teacher.clone().requires_grad_(True)
    _, training = _loss(
        teacher,
        text,
        authority,
        canonical_negative,
        student,
        trainable_region_mask=trainable,
        exclude_both_immutable_pairs=True,
    )
    _, validation = _loss(
        teacher,
        text,
        authority,
        canonical_negative,
        student,
        trainable_region_mask=trainable,
        exclude_both_immutable_pairs=False,
    )
    assert training["authority_hard_negative_pairs"] == 4
    assert training["objective_hard_negative_pairs"] == 2
    assert training["both_immutable_pairs_excluded"] == 2
    assert float(training["pair_trainable_endpoint_coverage"]) == 0.5
    assert validation["objective_hard_negative_pairs"] == 4
    assert validation["both_immutable_pairs_excluded"] == 0
    assert float(validation["pair_trainable_endpoint_coverage"]) == 0.5


def test_training_pair_denominator_fails_closed_without_trainable_pair(
    tmp_path: Path,
) -> None:
    teacher, text, authority, canonical_negative = _fixture(tmp_path)
    with pytest.raises(ValueError, match="no hard-negative pair"):
        _loss(
            teacher,
            text,
            authority,
            canonical_negative,
            teacher,
            trainable_region_mask=torch.zeros(4, dtype=torch.bool),
            exclude_both_immutable_pairs=True,
        )


def test_stable_small_margin_units_keep_nonzero_pairwise_gradient(
    tmp_path: Path,
) -> None:
    x = torch.tensor([-0.010, 0.000, 0.010, 0.020])
    teacher = torch.stack(
        [x, torch.sqrt(1.0 - x.square()), torch.zeros_like(x)], dim=-1
    )
    text = torch.zeros(806, 3)
    text[:, 0] = 1.0
    teacher, text, authority, canonical_negative = _fixture(
        tmp_path, teacher=teacher, text=text
    )
    student = torch.flip(teacher, dims=(0,)).clone().requires_grad_(True)
    _, metrics = _loss(teacher, text, authority, canonical_negative, student)
    pairwise = metrics["continuous_pairwise_relevance_loss"]
    assert torch.is_tensor(pairwise) and pairwise.item() > 0
    assert metrics["small_margin_positive_weight_units"] > 0
    pairwise.backward()
    assert student.grad is not None
    assert float(student.grad.abs().sum()) > 0


def test_v21_teacher_response_matches_o4_equal_view_normalized_lse() -> None:
    from radio_gs.losses import source_global_response_listwise_loss as v2

    teacher_views = F.normalize(
        torch.tensor(
            [
                [[1.0, 0.0], [0.6, 0.8], [0.0, 0.0]],
                [[0.0, 1.0], [-0.8, 0.6], [0.8, 0.6]],
            ]
        ),
        dim=-1,
    )
    teacher_mask = torch.tensor([[True, True, False], [True, True, True]])
    query = F.normalize(torch.tensor([[0.9, 0.1], [-0.2, 0.8]]), dim=-1)
    observed = v2._teacher_response_chunk(
        teacher_views,
        teacher_mask,
        query,
        temperature=O4_TEACHER_RESPONSE_TEMPERATURE,
    )
    beta = 10.0
    cosine = torch.einsum("rvd,qd->rvq", teacher_views, query)
    weights = teacher_mask.float()
    weights = weights / weights.sum(dim=1, keepdim=True)
    expected = (
        torch.logsumexp(
            beta * cosine
            + torch.where(
                weights > 0,
                weights.clamp_min(torch.finfo(torch.float32).tiny).log(),
                torch.full_like(weights, -torch.inf),
            )[..., None],
            dim=1,
        )
        / beta
    )
    assert torch.allclose(observed, expected, atol=2e-7, rtol=0)


def test_absolute_response_offset_is_supervised_when_centered_profile_matches(
    tmp_path: Path,
) -> None:
    teacher_x = torch.tensor([-0.30, -0.10, 0.10, 0.30])
    student_x = teacher_x + 0.10
    teacher = torch.stack(
        [
            teacher_x,
            torch.sqrt(1.0 - teacher_x.square()),
            torch.zeros_like(teacher_x),
        ],
        dim=-1,
    )
    student = torch.stack(
        [
            student_x,
            torch.sqrt(1.0 - student_x.square()),
            torch.zeros_like(student_x),
        ],
        dim=-1,
    ).requires_grad_(True)
    text = torch.zeros(806, 3)
    text[:, 0] = 1.0
    teacher, text, authority, canonical_negative = _fixture(
        tmp_path, teacher=teacher, text=text
    )
    _, metrics = _loss(teacher, text, authority, canonical_negative, student)
    assert metrics["centered_response_profile_loss"].item() == pytest.approx(
        0.0, abs=2e-6
    )
    assert metrics["absolute_response_loss"].item() > 0.05
    metrics["absolute_response_loss"].backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0


def test_compositional_component_extends_only_query_axis(tmp_path: Path) -> None:
    teacher, text, authority, canonical_negative = _fixture(tmp_path)
    embeddings = F.normalize(torch.tensor([[0.2, 0.8, 0.1], [0.7, -0.2, 0.4]]), dim=-1)
    payload = {
        "artifact_type": "target_blind_compositional_text_embedding_cache",
        "split": "fit",
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "prompt_templates": ["{query}"],
        "queries": ["generic-composition-a", "generic-composition-b"],
        "text_encoder": {"model_id": CANONICAL_NEGATIVE_MODEL},
        "embeddings": embeddings,
        "embedding_tensor_sha256": tensor_sha256(embeddings),
    }
    path = tmp_path / "compositional.pt"
    torch.save(payload, path)
    component = load_frozen_compositional_generic_bank(
        path,
        expected_file_sha256=sha256_file(path),
        component_id="generic_composition",
        loss_weight=0.20,
    )
    student = teacher.clone().requires_grad_(True)
    _, metrics = _loss(
        teacher,
        text,
        authority,
        canonical_negative,
        student,
        compositional_banks=(component,),
    )
    assert metrics["generic_query_rows"] == 808
    assert metrics["compositional_query_rows"] == 2
    assert metrics["generic_bank_components"] == 2


def test_uniform_optional_row_duplication_does_not_change_component_weight(
    tmp_path: Path,
) -> None:
    teacher, text, authority, canonical_negative = _fixture(tmp_path)
    base_embeddings = F.normalize(
        torch.tensor([[0.2, 0.8, 0.1], [0.7, -0.2, 0.4]]), dim=-1
    )

    def component(
        identifier: str, embeddings: torch.Tensor, *, loss_weight: float = 0.20
    ):
        payload = {
            "artifact_type": "target_blind_compositional_text_embedding_cache",
            "split": "fit",
            "benchmark_vocabulary_opened": False,
            "uses_benchmark_vocabulary_for_construction": False,
            "prompt_templates": ["{query}"],
            "queries": [
                f"generic-{identifier}-{index}" for index in range(embeddings.shape[0])
            ],
            "text_encoder": {"model_id": CANONICAL_NEGATIVE_MODEL},
            "embeddings": embeddings,
            "embedding_tensor_sha256": tensor_sha256(embeddings),
        }
        path = tmp_path / f"{identifier}.pt"
        torch.save(payload, path)
        return load_frozen_compositional_generic_bank(
            path,
            expected_file_sha256=sha256_file(path),
            component_id=identifier,
            loss_weight=loss_weight,
        )

    original = component("original", base_embeddings)
    duplicated = component("duplicated", base_embeddings.repeat_interleave(2, dim=0))
    student = torch.roll(teacher, shifts=1, dims=0).requires_grad_(True)
    _, original_metrics = _loss(
        teacher,
        text,
        authority,
        canonical_negative,
        student,
        compositional_banks=(original,),
    )
    _, duplicated_metrics = _loss(
        teacher,
        text,
        authority,
        canonical_negative,
        student,
        compositional_banks=(duplicated,),
    )
    for name in (
        "centered_response_profile_loss",
        "absolute_response_loss",
        "absolute_relevance_loss",
        "continuous_pairwise_relevance_loss",
        "auxiliary_loss",
    ):
        assert duplicated_metrics[name].item() == pytest.approx(
            original_metrics[name].item(), abs=2e-6
        )


def test_explicit_component_weight_changes_cross_component_reduction_only(
    tmp_path: Path,
) -> None:
    teacher, text, authority, canonical_negative = _fixture(tmp_path)
    embeddings = F.normalize(torch.tensor([[0.2, 0.8, 0.1], [0.7, -0.2, 0.4]]), dim=-1)

    def component(identifier: str, loss_weight: float):
        payload = {
            "artifact_type": "target_blind_compositional_text_embedding_cache",
            "split": "fit",
            "benchmark_vocabulary_opened": False,
            "uses_benchmark_vocabulary_for_construction": False,
            "prompt_templates": ["{query}"],
            "queries": [f"generic-{identifier}-a", f"generic-{identifier}-b"],
            "text_encoder": {"model_id": CANONICAL_NEGATIVE_MODEL},
            "embeddings": embeddings,
            "embedding_tensor_sha256": tensor_sha256(embeddings),
        }
        path = tmp_path / f"{identifier}.pt"
        torch.save(payload, path)
        return load_frozen_compositional_generic_bank(
            path,
            expected_file_sha256=sha256_file(path),
            component_id=identifier,
            loss_weight=loss_weight,
        )

    student = torch.roll(teacher, shifts=1, dims=0)
    low = component("low_weight", 0.05)
    high = component("high_weight", 0.30)
    _, low_metrics = _loss(
        teacher,
        text,
        authority,
        canonical_negative,
        student.clone().requires_grad_(True),
        compositional_banks=(low,),
    )
    _, high_metrics = _loss(
        teacher,
        text,
        authority,
        canonical_negative,
        student.clone().requires_grad_(True),
        compositional_banks=(high,),
    )
    assert low_metrics["generic_bank_component_weight_sum"].item() == pytest.approx(
        0.30
    )
    assert high_metrics["generic_bank_component_weight_sum"].item() == pytest.approx(
        0.55
    )
    assert low_metrics["auxiliary_loss"].item() != pytest.approx(
        high_metrics["auxiliary_loss"].item(), abs=1e-7
    )


@pytest.mark.parametrize("loss_weight", [0.0, -0.1, float("inf"), float("nan")])
def test_compositional_component_rejects_invalid_loss_weight(
    tmp_path: Path, loss_weight: float
) -> None:
    embeddings = F.normalize(torch.tensor([[0.2, 0.8, 0.1]]), dim=-1)
    payload = {
        "artifact_type": "target_blind_compositional_text_embedding_cache",
        "split": "fit",
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "prompt_templates": ["{query}"],
        "queries": ["generic-composition"],
        "text_encoder": {"model_id": CANONICAL_NEGATIVE_MODEL},
        "embeddings": embeddings,
        "embedding_tensor_sha256": tensor_sha256(embeddings),
    }
    path = tmp_path / "invalid_weight.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="loss_weight"):
        load_frozen_compositional_generic_bank(
            path,
            expected_file_sha256=sha256_file(path),
            component_id="invalid_weight",
            loss_weight=loss_weight,
        )


def test_zero_weight_returns_base_object_without_reading_v21_authorities() -> None:
    value = torch.tensor(3.0, requires_grad=True)
    base = value.square()
    total, metrics = source_global_response_listwise_loss_v21(
        base,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        accepted_v2_file_sha256="invalid",
        teacher_file_sha256="invalid",
        teacher_pair_descriptors_sha256="invalid",
        fit_text_bank_file_sha256="invalid",
        config=zero_weight_v21_config(),
    )
    assert total is base
    total.backward()
    assert value.grad.item() == 6.0
    assert all(
        value == 0 or (torch.is_tensor(value) and value.item() == 0)
        for value in metrics.values()
    )


def test_v21_uses_shared_inference_relevance_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from radio_gs.losses import source_global_response_listwise_loss_v21 as module

    teacher, text, authority, canonical_negative = _fixture(tmp_path)
    original = module.cosine_relevancy_torch
    calls = []

    def recorded(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "cosine_relevancy_torch", recorded)
    _loss(
        teacher,
        text,
        authority,
        canonical_negative,
        teacher.clone().requires_grad_(True),
    )
    assert calls
    assert all(call["logit_scale"] == 10.0 for call in calls)
    assert all(call["assume_normalized"] is True for call in calls)


def test_frozen_v2_is_byte_identical_and_v21_contains_no_target_vocabulary() -> None:
    repository = Path(__file__).resolve().parents[1]
    assert (
        sha256_file(
            repository / "radio_gs/losses/source_global_response_listwise_loss.py"
        )
        == FROZEN_V2_SHA
    )
    implementation = (
        repository / "radio_gs/losses/source_global_response_listwise_loss_v21.py"
    ).read_text()
    preregistration = (
        repository
        / "paper/artifacts/source_global_response_listwise_loss_v21_preregistration_20260806.json"
    ).read_text()
    forbidden = (
        "tesla door handle",
        "green toy chair",
        "rubber duck with buoy",
        "rubber duck with hat",
    )
    assert all(term not in implementation for term in forbidden)
    assert all(term not in preregistration for term in forbidden)


def test_real_frozen_canonical_negative_bank_is_sha_bound() -> None:
    repository = Path(__file__).resolve().parents[1]
    bank = load_frozen_canonical_negative_bank(
        repository / "checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt",
        expected_file_sha256=(
            "18d2aac56b50a9670ffe04b397d23a4652dd44fe8f18ed7a309a82b6c1102b67"
        ),
    )
    assert bank.embeddings.shape == (4, 1536)
    assert bank.model_id == CANONICAL_NEGATIVE_MODEL
