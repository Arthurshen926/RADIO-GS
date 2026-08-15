"""Target-blind text-response preservation for genuine region summaries.

The field trainer owns rendering and source-view selection.  This module owns
only a small, frozen semantic contract: reduce a predicted and a genuine
source region-summary map to a fixed regional grid, then preserve their
generic SigLIP2 response ordering and typed lexical relations.  It never
opens benchmark vocabulary, masks, or metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.semantic_alignment import align_full_extent_feature_grid
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    FrozenCompositionalGenericBank,
    load_frozen_compositional_generic_bank,
)
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    FrozenTypedTextRelationAuthority,
    load_frozen_typed_text_relation_authority,
)


REGION_GRID = (8, 8)
LISTWISE_TEMPERATURE = 0.05
SMOOTH_L1_BETA = 0.05
COMPONENT_WEIGHTS = {
    "profile": 0.35,
    "listwise": 0.35,
    "sibling": 0.15,
    "synonym": 0.15,
}


@dataclass(frozen=True)
class FrozenGenericRegionTextBundle:
    """SHA-bound target-blind text embeddings and typed pair indices."""

    primary_text: torch.Tensor
    synonym_text: torch.Tensor
    synonym_left_primary_indices: torch.Tensor
    synonym_right_indices: torch.Tensor
    sibling_left_primary_indices: torch.Tensor
    sibling_right_primary_indices: torch.Tensor
    relation_authority_sha256: str
    relation_content_authority_sha256: str
    primary_file_sha256: str
    primary_embedding_sha256: str
    synonym_file_sha256: str
    synonym_embedding_sha256: str

    def to(self, device: torch.device | str) -> "FrozenGenericRegionTextBundle":
        """Return the same immutable authority with tensors on ``device``."""

        return replace(
            self,
            primary_text=self.primary_text.to(device),
            synonym_text=self.synonym_text.to(device),
            synonym_left_primary_indices=self.synonym_left_primary_indices.to(device),
            synonym_right_indices=self.synonym_right_indices.to(device),
            sibling_left_primary_indices=self.sibling_left_primary_indices.to(device),
            sibling_right_primary_indices=self.sibling_right_primary_indices.to(device),
        )


def _verified_component(
    authority: FrozenTypedTextRelationAuthority,
    component_id: str,
) -> FrozenCompositionalGenericBank:
    record = authority.components[component_id]
    bank = load_frozen_compositional_generic_bank(
        Path(str(record["path"])),
        expected_file_sha256=str(record["sha256"]),
        component_id=component_id,
        loss_weight=1.0,
    )
    if (
        bank.query_rows != int(record["query_rows"])
        or bank.embedding_tensor_sha256 != str(record["embedding_tensor_sha256"])
        or tensor_sha256(bank.embeddings) != str(record["embedding_tensor_sha256"])
    ):
        raise ValueError(f"typed relation component differs: {component_id}")
    return bank


def load_frozen_generic_region_text_bundle(
    relation_authority_path: str | Path,
    *,
    expected_relation_authority_sha256: str,
) -> FrozenGenericRegionTextBundle:
    """Load all generic text inputs through one fail-closed authority."""

    authority = load_frozen_typed_text_relation_authority(
        relation_authority_path,
        expected_file_sha256=expected_relation_authority_sha256,
    )
    primary = _verified_component(authority, "primary")
    synonym = _verified_component(authority, "synonym_relation")
    if primary.embeddings.shape[1] != synonym.embeddings.shape[1]:
        raise ValueError("primary and synonym SigLIP2 dimensions differ")
    return FrozenGenericRegionTextBundle(
        primary_text=primary.embeddings,
        synonym_text=synonym.embeddings,
        synonym_left_primary_indices=authority.synonym_left_primary_indices,
        synonym_right_indices=authority.synonym_right_component_indices,
        sibling_left_primary_indices=authority.sibling_left_primary_indices,
        sibling_right_primary_indices=authority.sibling_right_primary_indices,
        relation_authority_sha256=authority.file_sha256,
        relation_content_authority_sha256=authority.content_authority_sha256,
        primary_file_sha256=primary.file_sha256,
        primary_embedding_sha256=primary.embedding_tensor_sha256,
        synonym_file_sha256=synonym.file_sha256,
        synonym_embedding_sha256=synonym.embedding_tensor_sha256,
    )


def _finite_summary_map(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.ndim != 4
        or value.shape[0] != 1
        or min(value.shape[1:]) <= 0
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be finite floating [1,C,H,W]")
    return value.float()


def _regional_descriptors(
    value: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = valid[None, None].to(dtype=value.dtype)
    pooled_mask = F.adaptive_avg_pool2d(mask, REGION_GRID)
    pooled = F.adaptive_avg_pool2d(value * mask, REGION_GRID)
    pooled = pooled / pooled_mask.clamp_min(1e-6)
    descriptors = pooled[0].permute(1, 2, 0).reshape(-1, value.shape[1])
    occupied = pooled_mask.reshape(-1) > 1e-6
    return F.normalize(descriptors[occupied], dim=-1, eps=1e-8), occupied


def _profile_loss(
    student_response: torch.Tensor,
    teacher_response: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    student_centered = student_response - student_response.mean(dim=0, keepdim=True)
    teacher_centered = teacher_response - teacher_response.mean(dim=0, keepdim=True)
    informative = torch.linalg.vector_norm(teacher_centered, dim=0) > 1e-6
    if not bool(informative.any()):
        zero = student_response.sum() * 0.0
        return zero, zero.detach(), 0
    cosine = F.cosine_similarity(
        student_centered[:, informative].T,
        teacher_centered[:, informative].T,
        dim=-1,
        eps=1e-8,
    )
    return 1.0 - cosine.mean(), cosine.mean().detach(), int(informative.sum())


def generic_region_text_response_loss(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    alpha_map: torch.Tensor,
    bundle: FrozenGenericRegionTextBundle,
    *,
    alpha_threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Preserve generic response profiles, rankings, and typed pair gaps.

    ``teacher`` is detached by construction.  Both maps are pooled through the
    same fixed regional operator before any text response is computed.
    """

    predicted = _finite_summary_map(predicted, label="predicted region summaries")
    teacher = _finite_summary_map(teacher, label="teacher region summaries")
    if teacher.shape[:2] != predicted.shape[:2]:
        raise ValueError("predicted and teacher region-summary maps must match")
    teacher = align_full_extent_feature_grid(
        teacher,
        tuple(int(size) for size in predicted.shape[-2:]),
        label="predicted and teacher region-summary maps must match",
    )
    if alpha_map.shape == (1, *predicted.shape[-2:]):
        alpha_map = alpha_map[0]
    if alpha_map.shape != predicted.shape[-2:]:
        raise ValueError("alpha map must match the region-summary spatial grid")
    if not bool(torch.isfinite(alpha_map).all()):
        raise ValueError("alpha map must be finite")
    if not 0.0 <= float(alpha_threshold) <= 1.0:
        raise ValueError("alpha_threshold must lie in [0,1]")
    if (
        bundle.primary_text.device != predicted.device
        or bundle.synonym_text.device != predicted.device
    ):
        raise ValueError("generic text bundle and predictions must share a device")
    if (
        bundle.primary_text.shape[1] != predicted.shape[1]
        or bundle.synonym_text.shape[1] != predicted.shape[1]
    ):
        raise ValueError("generic text and region-summary dimensions differ")

    valid = alpha_map >= float(alpha_threshold)
    student, occupied = _regional_descriptors(predicted, valid)
    target, teacher_occupied = _regional_descriptors(teacher.detach(), valid)
    if not torch.equal(occupied, teacher_occupied):
        raise RuntimeError("student and teacher regional visibility differs")
    if student.shape[0] < 2:
        zero = predicted.sum() * 0.0
        return zero, {
            "profile": zero.detach(),
            "profile_cosine": zero.detach(),
            "listwise": zero.detach(),
            "sibling": zero.detach(),
            "synonym": zero.detach(),
            "regions": int(student.shape[0]),
            "informative_queries": 0,
        }

    primary = F.normalize(bundle.primary_text.float(), dim=-1, eps=1e-8)
    synonym = F.normalize(bundle.synonym_text.float(), dim=-1, eps=1e-8)
    student_primary = student @ primary.T
    teacher_primary = target @ primary.T
    profile, profile_cosine, informative_queries = _profile_loss(
        student_primary, teacher_primary
    )

    temperature = float(LISTWISE_TEMPERATURE)
    teacher_distribution = F.softmax(teacher_primary.T / temperature, dim=-1)
    student_log_distribution = F.log_softmax(student_primary.T / temperature, dim=-1)
    listwise = F.kl_div(
        student_log_distribution,
        teacher_distribution,
        reduction="batchmean",
    )

    sibling_left = bundle.sibling_left_primary_indices
    sibling_right = bundle.sibling_right_primary_indices
    synonym_left = bundle.synonym_left_primary_indices
    synonym_right = bundle.synonym_right_indices
    if (
        sibling_left.dtype != torch.int64
        or sibling_right.dtype != torch.int64
        or synonym_left.dtype != torch.int64
        or synonym_right.dtype != torch.int64
        or sibling_left.shape != sibling_right.shape
        or synonym_left.shape != synonym_right.shape
        or sibling_left.numel() <= 0
        or synonym_left.numel() <= 0
        or bool(
            torch.cat(
                (sibling_left, sibling_right, synonym_left, synonym_right)
            ).lt(0).any()
        )
        or int(torch.cat((sibling_left, sibling_right, synonym_left)).max()) >= primary.shape[0]
        or int(synonym_right.max()) >= synonym.shape[0]
    ):
        raise ValueError("generic typed relation indices differ")
    student_sibling_gap = student_primary[:, sibling_left] - student_primary[:, sibling_right]
    teacher_sibling_gap = teacher_primary[:, sibling_left] - teacher_primary[:, sibling_right]
    sibling = F.smooth_l1_loss(
        student_sibling_gap,
        teacher_sibling_gap,
        beta=float(SMOOTH_L1_BETA),
    )
    student_synonym = student @ synonym[synonym_right].T
    teacher_synonym = target @ synonym[synonym_right].T
    student_synonym_gap = student_primary[:, synonym_left] - student_synonym
    teacher_synonym_gap = teacher_primary[:, synonym_left] - teacher_synonym
    synonym_loss = F.smooth_l1_loss(
        student_synonym_gap,
        teacher_synonym_gap,
        beta=float(SMOOTH_L1_BETA),
    )

    total = (
        COMPONENT_WEIGHTS["profile"] * profile
        + COMPONENT_WEIGHTS["listwise"] * listwise
        + COMPONENT_WEIGHTS["sibling"] * sibling
        + COMPONENT_WEIGHTS["synonym"] * synonym_loss
    )
    return total, {
        "profile": profile.detach(),
        "profile_cosine": profile_cosine,
        "listwise": listwise.detach(),
        "sibling": sibling.detach(),
        "synonym": synonym_loss.detach(),
        "regions": int(student.shape[0]),
        "informative_queries": informative_queries,
    }
