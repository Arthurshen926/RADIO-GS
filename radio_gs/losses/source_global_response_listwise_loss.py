"""Scene-global generic-query response supervision for bounded V2 residuals.

This module owns no parameters and does not load a text encoder.  It consumes
only caller-verified official teacher observations, frozen target-blind fit
embeddings, and a SHA-bound scene hard-negative authority.  The exact zero
auxiliary-weight path returns the caller's base loss object unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.evaluation.source_query_response_hard_negatives import (
    validate_negative_authority,
)
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.utils.immutable_artifacts import load_torch_payload


RESPONSE_TEMPERATURE = 0.05
NORMALIZED_PAIR_GAP_FLOOR = 0.05
TRIPLET_MARGIN_CEILING = 0.10
TRIPLET_MARGIN_SCALE = 0.50
QUERY_CHUNK_ROWS = 128
PAIR_CHUNK_ROWS = 4096
RECOMMENDED_AUXILIARY_WEIGHT = 0.25


@dataclass(frozen=True)
class SourceGlobalResponseLossConfig:
    auxiliary_weight: float = RECOMMENDED_AUXILIARY_WEIGHT
    response_temperature: float = RESPONSE_TEMPERATURE
    normalized_pair_gap_floor: float = NORMALIZED_PAIR_GAP_FLOOR
    triplet_margin_ceiling: float = TRIPLET_MARGIN_CEILING
    triplet_margin_scale: float = TRIPLET_MARGIN_SCALE
    query_chunk_rows: int = QUERY_CHUNK_ROWS
    pair_chunk_rows: int = PAIR_CHUNK_ROWS

    def __post_init__(self) -> None:
        scalars = {
            "auxiliary_weight": self.auxiliary_weight,
            "response_temperature": self.response_temperature,
            "normalized_pair_gap_floor": self.normalized_pair_gap_floor,
            "triplet_margin_ceiling": self.triplet_margin_ceiling,
            "triplet_margin_scale": self.triplet_margin_scale,
        }
        if any(not math.isfinite(float(value)) for value in scalars.values()):
            raise ValueError("source-global response loss scalars must be finite")
        if self.auxiliary_weight < 0:
            raise ValueError("auxiliary_weight must be nonnegative")
        if self.response_temperature <= 0:
            raise ValueError("response_temperature must be positive")
        if not 0 <= self.normalized_pair_gap_floor < 1:
            raise ValueError("normalized_pair_gap_floor must lie in [0,1)")
        if self.triplet_margin_ceiling < 0 or self.triplet_margin_scale < 0:
            raise ValueError("triplet margin constants must be nonnegative")
        if self.query_chunk_rows <= 0 or self.pair_chunk_rows <= 0:
            raise ValueError("query and pair chunk sizes must be positive")


@dataclass(frozen=True)
class FrozenSourceResponseAuthority:
    payload: dict[str, Any]
    file_sha256: str
    content_authority_sha256: str
    scene_id: str
    accepted_v2_file_sha256: str
    teacher_file_sha256: str
    teacher_pair_descriptors_sha256: str
    fit_text_bank_file_sha256: str
    fit_text_embedding_tensor_sha256: str


def recommended_v2_config() -> SourceGlobalResponseLossConfig:
    """Return the single preregistered configuration; no scene knobs exist."""

    return SourceGlobalResponseLossConfig()


def zero_weight_config() -> SourceGlobalResponseLossConfig:
    """Return the exact compatibility configuration for the existing loss."""

    return SourceGlobalResponseLossConfig(auxiliary_weight=0.0)


def _lower_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def load_frozen_source_response_authority(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_content_authority_sha256: str,
    expected_scene_id: str,
    expected_accepted_v2_file_sha256: str,
    expected_teacher_file_sha256: str,
    expected_teacher_pair_descriptors_sha256: str,
    expected_fit_text_bank_file_sha256: str,
) -> FrozenSourceResponseAuthority:
    """Load and bind a hard-negative authority to all training inputs."""

    expected_file = _lower_sha256(
        expected_file_sha256, label="hard-negative authority file"
    )
    value, digest, _ = load_torch_payload(
        path,
        expected_sha256=expected_file,
        map_location="cpu",
        label="source-global hard-negative authority",
    )
    payload = validate_negative_authority(value)
    content = _lower_sha256(
        expected_content_authority_sha256,
        label="hard-negative content authority",
    )
    if payload["content_authority_sha256"] != content:
        raise ValueError("hard-negative content authority differs")
    if payload["scene_id"] != str(expected_scene_id):
        raise ValueError("hard-negative scene identity differs")
    inputs = payload["input_authority"]
    try:
        accepted = inputs["accepted_v2"]
        teacher = inputs["official_multiview_siglip2_teacher"]
        fit = inputs["fit_text_bank"]
    except (KeyError, TypeError) as exc:
        raise ValueError("hard-negative input authority is incomplete") from exc
    accepted_sha = _lower_sha256(
        expected_accepted_v2_file_sha256, label="AcceptedV2 authority"
    )
    teacher_sha = _lower_sha256(
        expected_teacher_file_sha256, label="official teacher authority"
    )
    teacher_pair_descriptors_sha = _lower_sha256(
        expected_teacher_pair_descriptors_sha256,
        label="official teacher pair descriptor channel",
    )
    fit_sha = _lower_sha256(
        expected_fit_text_bank_file_sha256, label="fit text bank"
    )
    if accepted.get("sha256") != accepted_sha:
        raise ValueError("hard-negative AcceptedV2 binding differs")
    teacher_channels = teacher.get("channel_sha256")
    if (
        teacher.get("sha256") != teacher_sha
        or not isinstance(teacher_channels, dict)
        or teacher_channels.get("pair_descriptors")
        != teacher_pair_descriptors_sha
    ):
        raise ValueError("hard-negative teacher binding differs")
    if (
        fit.get("split") != "fit"
        or int(fit.get("queries", -1)) != 806
        or fit.get("sha256") != fit_sha
        or fit.get("benchmark_vocabulary_opened") is not False
        or fit.get("uses_benchmark_vocabulary_for_construction") is not False
    ):
        raise ValueError("hard-negative generic fit text binding differs")
    fit_tensor_sha = _lower_sha256(
        fit.get("embedding_tensor_sha256"), label="fit text embedding tensor"
    )
    access = payload["source_access"]
    if (
        access.get("generic_target_blind_text_bank_opened") is not True
        or access.get("benchmark_text_queries_opened") is not False
        or access.get("text_queries_opened") is not False
    ):
        raise ValueError("hard-negative text access semantics differ")
    return FrozenSourceResponseAuthority(
        payload=payload,
        file_sha256=digest,
        content_authority_sha256=content,
        scene_id=str(expected_scene_id),
        accepted_v2_file_sha256=accepted_sha,
        teacher_file_sha256=teacher_sha,
        teacher_pair_descriptors_sha256=teacher_pair_descriptors_sha,
        fit_text_bank_file_sha256=fit_sha,
        fit_text_embedding_tensor_sha256=fit_tensor_sha,
    )


def _finite_float_matrix(
    value: torch.Tensor, *, label: str, device: torch.device
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device != device
        or value.ndim != 2
        or not value.is_floating_point()
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be finite floating [N,D] on {device}")
    if bool((torch.linalg.vector_norm(value.detach().float(), dim=-1) <= 1e-12).any()):
        raise ValueError(f"{label} contains a zero-norm row")
    return value


def _teacher_views(
    descriptors: torch.Tensor,
    pair_rows: torch.Tensor,
    *,
    region_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = descriptors.device
    teacher = F.normalize(descriptors.detach().float(), dim=-1)
    rows = torch.as_tensor(pair_rows, device=device)
    if (
        rows.dtype != torch.int64
        or rows.shape != (teacher.shape[0],)
        or bool((rows < 0).any())
        or bool((rows >= region_count).any())
        or (rows.numel() > 1 and bool((rows[1:] < rows[:-1]).any()))
    ):
        raise ValueError("teacher pair rows must be sorted complete scene rows")
    counts = torch.bincount(rows, minlength=region_count)
    if bool((counts <= 0).any()) or int(counts.max()) > 4:
        raise ValueError("every scene row requires one to four teacher views")
    offsets = torch.zeros(region_count + 1, dtype=torch.int64, device=device)
    offsets[1:] = counts.cumsum(0)
    slots = torch.arange(rows.numel(), device=device) - offsets[rows]
    maximum = int(counts.max())
    dense = torch.zeros(
        (region_count, maximum, teacher.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros(
        (region_count, maximum), dtype=torch.bool, device=device
    )
    dense[rows, slots] = teacher
    mask[rows, slots] = True
    consensus = F.normalize((dense * mask[..., None]).sum(dim=1), dim=-1)
    return dense, mask, consensus


def _teacher_response_chunk(
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    text_chunk: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    scores = torch.einsum("rvd,qd->rvq", teacher_views, text_chunk)
    scores = scores.masked_fill(~teacher_mask[..., None], -torch.inf)
    counts = teacher_mask.sum(dim=1).float().log()[:, None]
    response = float(temperature) * (
        torch.logsumexp(scores / float(temperature), dim=1) - counts
    )
    if not bool(torch.isfinite(response).all()):
        raise ValueError("official teacher response is nonfinite")
    return response


def source_global_response_listwise_loss(
    base_loss: torch.Tensor,
    student_descriptors: torch.Tensor,
    teacher_pair_descriptors: torch.Tensor,
    teacher_pair_region_indices: torch.Tensor,
    fit_text_embeddings: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    authority: FrozenSourceResponseAuthority,
    *,
    accepted_v2_file_sha256: str,
    teacher_file_sha256: str,
    teacher_pair_descriptors_sha256: str,
    fit_text_bank_file_sha256: str,
    config: SourceGlobalResponseLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Add three source-only scene-global losses to an existing scalar loss.

    ``auxiliary_weight == 0`` is an early identity branch: it returns
    ``base_loss`` itself, without validating or evaluating auxiliary inputs.
    """

    chosen = recommended_v2_config() if config is None else config
    if not isinstance(chosen, SourceGlobalResponseLossConfig):
        raise TypeError("config must be SourceGlobalResponseLossConfig")
    if (
        not isinstance(base_loss, torch.Tensor)
        or base_loss.ndim != 0
        or not base_loss.is_floating_point()
        or not bool(torch.isfinite(base_loss.detach()))
    ):
        raise ValueError("base_loss must be one finite floating scalar")
    if chosen.auxiliary_weight == 0.0:
        zero = base_loss.detach().new_zeros(())
        return base_loss, {
            "centered_response_profile_loss": zero,
            "pairwise_ordering_loss": zero,
            "hard_negative_triplet_loss": zero,
            "auxiliary_loss": zero,
            "valid_profile_queries": 0,
            "valid_pair_query_units": 0,
            "hard_negative_pairs": 0,
        }
    if type(authority) is not FrozenSourceResponseAuthority:
        raise TypeError("authority must be a fail-closed frozen authority")
    bindings = {
        "accepted_v2": (
            accepted_v2_file_sha256,
            authority.accepted_v2_file_sha256,
        ),
        "teacher": (teacher_file_sha256, authority.teacher_file_sha256),
        "teacher_pair_descriptors": (
            teacher_pair_descriptors_sha256,
            authority.teacher_pair_descriptors_sha256,
        ),
        "fit_text": (
            fit_text_bank_file_sha256,
            authority.fit_text_bank_file_sha256,
        ),
    }
    for label, (actual, expected) in bindings.items():
        if _lower_sha256(actual, label=label) != expected:
            raise ValueError(f"runtime {label} binding differs")

    device = student_descriptors.device
    student_raw = _finite_float_matrix(
        student_descriptors, label="student_descriptors", device=device
    )
    teacher_raw = _finite_float_matrix(
        teacher_pair_descriptors,
        label="teacher_pair_descriptors",
        device=device,
    )
    text_raw = _finite_float_matrix(
        fit_text_embeddings, label="fit_text_embeddings", device=device
    )
    if student_raw.shape[1] != teacher_raw.shape[1] or student_raw.shape[1] != text_raw.shape[1]:
        raise ValueError("student, teacher, and text descriptor dimensions differ")
    if tensor_sha256(text_raw.detach().cpu()) != authority.fit_text_embedding_tensor_sha256:
        raise ValueError("runtime fit text embedding tensor differs")
    canonical = torch.as_tensor(canonical_region_indices).detach().long().cpu()
    authority_canonical = authority.payload["canonical_region_indices"]
    if (
        canonical.shape != authority_canonical.shape
        or not torch.equal(canonical, authority_canonical)
        or student_raw.shape[0] != canonical.numel()
    ):
        raise ValueError("loss requires the complete canonical scene row order")

    student = F.normalize(student_raw.float(), dim=-1)
    text = F.normalize(text_raw.detach().float(), dim=-1)
    teacher_views, teacher_mask, teacher_consensus = _teacher_views(
        teacher_raw,
        teacher_pair_region_indices,
        region_count=int(student.shape[0]),
    )
    channels = authority.payload["channels"]
    anchors = channels["anchor_region_indices"].to(device)
    negatives = channels["negative_region_indices"].to(device)
    declared_teacher_cosine = channels["teacher_cosines"].to(
        device=device, dtype=torch.float32
    )

    profile_sum = student.sum() * 0.0
    pair_sum = student.sum() * 0.0
    profile_count = 0
    pair_count = 0
    for query_start in range(0, text.shape[0], chosen.query_chunk_rows):
        query_stop = min(query_start + chosen.query_chunk_rows, text.shape[0])
        query = text[query_start:query_stop]
        student_response = student @ query.T
        teacher_response = _teacher_response_chunk(
            teacher_views,
            teacher_mask,
            query,
            temperature=chosen.response_temperature,
        )
        teacher_centered = teacher_response - teacher_response.mean(
            dim=0, keepdim=True
        )
        student_centered = student_response - student_response.mean(
            dim=0, keepdim=True
        )
        teacher_norm = torch.linalg.vector_norm(teacher_centered, dim=0)
        profile_valid = teacher_norm > 1e-6
        if bool(profile_valid.any()):
            profile = F.cosine_similarity(
                student_centered[:, profile_valid],
                teacher_centered[:, profile_valid],
                dim=0,
                eps=1e-6,
            )
            profile_sum = profile_sum + (1.0 - profile).sum()
            profile_count += int(profile.numel())

        teacher_span = teacher_response.amax(dim=0) - teacher_response.amin(dim=0)
        span_valid = teacher_span > 1e-6
        for pair_start in range(0, anchors.numel(), chosen.pair_chunk_rows):
            pair_stop = min(pair_start + chosen.pair_chunk_rows, anchors.numel())
            a = anchors[pair_start:pair_stop]
            n = negatives[pair_start:pair_stop]
            teacher_gap = teacher_response[a] - teacher_response[n]
            student_gap = student_response[a] - student_response[n]
            scale = teacher_span.clamp_min(1e-6)[None, :]
            normalized_teacher = teacher_gap / scale
            normalized_student = student_gap / scale
            valid = span_valid[None, :] & (
                normalized_teacher.abs()
                >= float(chosen.normalized_pair_gap_floor)
            )
            if bool(valid.any()):
                pair_sum = pair_sum + F.smooth_l1_loss(
                    normalized_student[valid],
                    normalized_teacher[valid],
                    reduction="sum",
                )
                pair_count += int(valid.sum())
    if profile_count <= 0 or pair_count <= 0:
        raise ValueError("scene has no valid response-profile or pairwise units")
    profile_loss = profile_sum / profile_count
    pairwise_loss = pair_sum / pair_count

    positive = (student[anchors] * teacher_consensus[anchors]).sum(dim=-1)
    negative = (student[anchors] * teacher_consensus[negatives]).sum(dim=-1)
    margin = (
        float(chosen.triplet_margin_scale)
        * (1.0 - declared_teacher_cosine).clamp_min(0.0)
    ).clamp_max(float(chosen.triplet_margin_ceiling))
    triplet_loss = F.relu(margin - positive + negative).mean()
    auxiliary = (profile_loss + pairwise_loss + triplet_loss) / 3.0
    total = base_loss + float(chosen.auxiliary_weight) * auxiliary
    if not bool(torch.isfinite(total.detach())):
        raise RuntimeError("source-global response objective is nonfinite")
    return total, {
        "centered_response_profile_loss": profile_loss,
        "pairwise_ordering_loss": pairwise_loss,
        "hard_negative_triplet_loss": triplet_loss,
        "auxiliary_loss": auxiliary,
        "valid_profile_queries": profile_count,
        "valid_pair_query_units": pair_count,
        "hard_negative_pairs": int(anchors.numel()),
    }


__all__ = [
    "FrozenSourceResponseAuthority",
    "SourceGlobalResponseLossConfig",
    "load_frozen_source_response_authority",
    "recommended_v2_config",
    "source_global_response_listwise_loss",
    "zero_weight_config",
]
