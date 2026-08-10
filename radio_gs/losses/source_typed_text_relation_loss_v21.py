"""SHA-bound source-only synonym and sibling relation supervision for V2.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.losses import source_global_response_listwise_loss as v2
from radio_gs.querying.unified_query import (
    cosine_relevancy_torch,
    margin_to_relevancy_torch,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_torch_mapping,
)


SCHEMA = "radio_gs.target_blind_typed_text_relation_authority.v1"
EXPECTED_COUNTS = {"synonym_pairs": 657, "sibling_pairs": 167}
RELATION_COMPONENT_WEIGHTS = {"synonym": 0.20, "sibling": 0.20}
TYPED_RELATION_AUXILIARY_WEIGHT = 0.25


@dataclass(frozen=True)
class FrozenTypedTextRelationAuthority:
    file_sha256: str
    content_authority_sha256: str
    source_sha256: str
    components: Mapping[str, Mapping[str, Any]]
    synonym_left_primary_indices: torch.Tensor
    synonym_right_component_indices: torch.Tensor
    sibling_left_primary_indices: torch.Tensor
    sibling_right_primary_indices: torch.Tensor


def _index_vector(value: object, *, count: int, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if (
        tensor.dtype != torch.int64
        or tensor.shape != (count,)
        or bool((tensor < 0).any())
    ):
        raise ValueError(f"{label} must be nonnegative int64 [{count}]")
    return tensor.contiguous()


def load_frozen_typed_text_relation_authority(
    path: str | Path, *, expected_file_sha256: str
) -> FrozenTypedTextRelationAuthority:
    payload, digest, _ = load_torch_mapping(
        path,
        expected_sha256=v2._lower_sha256(
            expected_file_sha256, label="typed relation authority"
        ),
        map_location="cpu",
        label="target-blind typed text relation authority",
    )
    tensor_keys = {
        "synonym_left_primary_indices",
        "synonym_right_component_indices",
        "sibling_left_primary_indices",
        "sibling_right_primary_indices",
    }
    record_keys = {"synonym_record_ids", "sibling_record_ids"}
    identity_keys = {
        "schema",
        "schema_version",
        "split",
        "source",
        "components",
        "counts",
        "index_semantics",
        "source_access",
    }
    if set(payload) != identity_keys | tensor_keys | record_keys | {
        "content_authority_sha256"
    }:
        raise ValueError("typed relation authority fields differ")
    identity = {key: payload[key] for key in identity_keys}
    content = payload["content_authority_sha256"]
    if (
        payload["schema"] != SCHEMA
        or payload["schema_version"] != 1
        or payload["split"] != "fit"
        or payload["counts"] != EXPECTED_COUNTS
        or content != canonical_json_sha256(identity)
        or payload["source_access"].get("benchmark_vocabulary_opened") is not False
        or payload["source_access"].get("target_metrics_computed") is not False
    ):
        raise ValueError("typed relation identity contract differs")
    components = payload["components"]
    expected_components = {
        "primary",
        "synonym_relation",
        "lexical_sibling_relation",
        "counterfactual_attributes",
        "high_precision_part_of",
    }
    if not isinstance(components, Mapping) or set(components) != expected_components:
        raise ValueError("typed relation component authority differs")
    for name, record in components.items():
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {"path", "sha256", "embedding_tensor_sha256", "query_rows"}
            or int(record["query_rows"]) <= 0
        ):
            raise ValueError(f"typed relation component record differs: {name}")
        v2._lower_sha256(record["sha256"], label=f"{name} file")
        v2._lower_sha256(record["embedding_tensor_sha256"], label=f"{name} embedding")
    source = payload["source"]
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise ValueError("typed relation source record differs")
    source_sha = v2._lower_sha256(source["sha256"], label="relation source")

    synonym_count = EXPECTED_COUNTS["synonym_pairs"]
    sibling_count = EXPECTED_COUNTS["sibling_pairs"]
    synonym_ids = payload["synonym_record_ids"]
    sibling_ids = payload["sibling_record_ids"]
    if (
        not isinstance(synonym_ids, list)
        or len(synonym_ids) != synonym_count
        or len(set(synonym_ids)) != synonym_count
        or not isinstance(sibling_ids, list)
        or len(sibling_ids) != sibling_count
        or len(set(sibling_ids)) != sibling_count
    ):
        raise ValueError("typed relation record ids differ")
    synonym_left = _index_vector(
        payload["synonym_left_primary_indices"],
        count=synonym_count,
        label="synonym left indices",
    )
    synonym_right = _index_vector(
        payload["synonym_right_component_indices"],
        count=synonym_count,
        label="synonym right indices",
    )
    sibling_left = _index_vector(
        payload["sibling_left_primary_indices"],
        count=sibling_count,
        label="sibling left indices",
    )
    sibling_right = _index_vector(
        payload["sibling_right_primary_indices"],
        count=sibling_count,
        label="sibling right indices",
    )
    if (
        int(synonym_left.max()) >= int(components["primary"]["query_rows"])
        or int(synonym_right.max()) >= int(components["synonym_relation"]["query_rows"])
        or int(sibling_left.max()) >= int(components["primary"]["query_rows"])
        or int(sibling_right.max()) >= int(components["primary"]["query_rows"])
    ):
        raise ValueError("typed relation index exceeds a bound component")
    return FrozenTypedTextRelationAuthority(
        file_sha256=digest,
        content_authority_sha256=str(content),
        source_sha256=source_sha,
        components={name: dict(record) for name, record in components.items()},
        synonym_left_primary_indices=synonym_left,
        synonym_right_component_indices=synonym_right,
        sibling_left_primary_indices=sibling_left,
        sibling_right_primary_indices=sibling_right,
    )


def _teacher_relevance(
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    text: torch.Tensor,
    teacher_negative_response: torch.Tensor,
    *,
    response_temperature: float,
    logit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    response = v2._teacher_response_chunk(
        teacher_views,
        teacher_mask,
        text,
        temperature=response_temperature,
    )
    relevance = margin_to_relevancy_torch(
        response - teacher_negative_response.amax(dim=-1, keepdim=True),
        logit_scale=logit_scale,
    )
    return response, relevance


def _student_relevance(
    student: torch.Tensor,
    text: torch.Tensor,
    negative_text: torch.Tensor,
    *,
    logit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    response = student @ text.T
    relevance = cosine_relevancy_torch(
        student,
        text,
        negative_text,
        logit_scale=logit_scale,
        assume_normalized=True,
    )
    return response, relevance


def _relevance_std(
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    text: torch.Tensor,
    negative_text: torch.Tensor,
    *,
    logit_scale: float,
) -> torch.Tensor:
    positive = torch.einsum("rvd,qd->rvq", teacher_views, text)
    negative = torch.einsum("rvd,kd->rvk", teacher_views, negative_text).amax(dim=-1)
    relevance = margin_to_relevancy_torch(
        positive - negative[..., None], logit_scale=logit_scale
    )
    mask = teacher_mask[..., None]
    count = mask.sum(dim=1).clamp_min(1)
    mean = (relevance * mask).sum(dim=1) / count
    variance = (((relevance - mean[:, None]) ** 2) * mask).sum(dim=1) / count
    return torch.sqrt(variance.clamp_min(0)).detach()


def source_typed_text_relation_loss_v21(
    student: torch.Tensor,
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    primary_text: torch.Tensor,
    synonym_text: torch.Tensor,
    negative_text: torch.Tensor,
    teacher_negative_response: torch.Tensor,
    authority: FrozenTypedTextRelationAuthority,
    *,
    response_temperature: float,
    logit_scale: float,
    smooth_l1_beta: float,
    continuous_gap_sigma: float,
    stability_std_scale: float,
    pair_chunk_rows: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Return teacher-calibrated synonym and bidirectional sibling losses."""

    if type(authority) is not FrozenTypedTextRelationAuthority:
        raise TypeError("authority must be FrozenTypedTextRelationAuthority")
    device = student.device
    synonym_left = authority.synonym_left_primary_indices.to(device)
    synonym_right = authority.synonym_right_component_indices.to(device)
    sibling_left = authority.sibling_left_primary_indices.to(device)
    sibling_right = authority.sibling_right_primary_indices.to(device)
    zero = student.sum() * 0.0

    synonym_response_sum = zero
    synonym_relevance_sum = zero
    synonym_units = 0
    for start in range(0, synonym_left.numel(), pair_chunk_rows):
        stop = min(start + pair_chunk_rows, synonym_left.numel())
        left_text = primary_text[synonym_left[start:stop]]
        right_text = synonym_text[synonym_right[start:stop]]
        student_left_response, student_left_relevance = _student_relevance(
            student, left_text, negative_text, logit_scale=logit_scale
        )
        student_right_response, student_right_relevance = _student_relevance(
            student, right_text, negative_text, logit_scale=logit_scale
        )
        teacher_left_response, teacher_left_relevance = _teacher_relevance(
            teacher_views,
            teacher_mask,
            left_text,
            teacher_negative_response,
            response_temperature=response_temperature,
            logit_scale=logit_scale,
        )
        teacher_right_response, teacher_right_relevance = _teacher_relevance(
            teacher_views,
            teacher_mask,
            right_text,
            teacher_negative_response,
            response_temperature=response_temperature,
            logit_scale=logit_scale,
        )
        synonym_response_sum = synonym_response_sum + F.smooth_l1_loss(
            student_left_response - student_right_response,
            teacher_left_response - teacher_right_response,
            beta=smooth_l1_beta,
            reduction="sum",
        )
        synonym_relevance_sum = synonym_relevance_sum + F.smooth_l1_loss(
            student_left_relevance - student_right_relevance,
            teacher_left_relevance - teacher_right_relevance,
            beta=smooth_l1_beta,
            reduction="sum",
        )
        synonym_units += int(student_left_response.numel())
    if synonym_units <= 0:
        raise ValueError("typed synonym authority produced no units")
    synonym_response_loss = synonym_response_sum / synonym_units
    synonym_relevance_loss = synonym_relevance_sum / synonym_units
    synonym_loss = 0.5 * (synonym_response_loss + synonym_relevance_loss)

    left_weighted_sum = zero
    right_weighted_sum = zero
    left_weight_sum = zero.detach()
    right_weight_sum = zero.detach()
    left_units = 0
    right_units = 0
    for start in range(0, sibling_left.numel(), pair_chunk_rows):
        stop = min(start + pair_chunk_rows, sibling_left.numel())
        left_text = primary_text[sibling_left[start:stop]]
        right_text = primary_text[sibling_right[start:stop]]
        _, student_left = _student_relevance(
            student, left_text, negative_text, logit_scale=logit_scale
        )
        _, student_right = _student_relevance(
            student, right_text, negative_text, logit_scale=logit_scale
        )
        _, teacher_left = _teacher_relevance(
            teacher_views,
            teacher_mask,
            left_text,
            teacher_negative_response,
            response_temperature=response_temperature,
            logit_scale=logit_scale,
        )
        _, teacher_right = _teacher_relevance(
            teacher_views,
            teacher_mask,
            right_text,
            teacher_negative_response,
            response_temperature=response_temperature,
            logit_scale=logit_scale,
        )
        teacher_gap = teacher_left - teacher_right
        student_gap = student_left - student_right
        gap = teacher_gap.abs()
        stability = torch.exp(
            -(
                _relevance_std(
                    teacher_views,
                    teacher_mask,
                    left_text,
                    negative_text,
                    logit_scale=logit_scale,
                )
                + _relevance_std(
                    teacher_views,
                    teacher_mask,
                    right_text,
                    negative_text,
                    logit_scale=logit_scale,
                )
            )
            / stability_std_scale
        )
        weight = stability * gap / (gap + continuous_gap_sigma)
        unit = F.smooth_l1_loss(
            student_gap,
            teacher_gap,
            beta=smooth_l1_beta,
            reduction="none",
        )
        left = teacher_gap > 0
        right = teacher_gap < 0
        left_weighted_sum = left_weighted_sum + (weight[left] * unit[left]).sum()
        right_weighted_sum = right_weighted_sum + (weight[right] * unit[right]).sum()
        left_weight_sum = left_weight_sum + weight[left].sum().detach()
        right_weight_sum = right_weight_sum + weight[right].sum().detach()
        left_units += int(left.sum())
        right_units += int(right.sum())
    directions: list[torch.Tensor] = []
    if float(left_weight_sum) > 0:
        directions.append(left_weighted_sum / left_weight_sum)
    if float(right_weight_sum) > 0:
        directions.append(right_weighted_sum / right_weight_sum)
    if not directions:
        raise ValueError("typed sibling authority has no nonzero teacher gaps")
    sibling_loss = torch.stack(directions).mean()
    relation_weights = torch.tensor(
        [RELATION_COMPONENT_WEIGHTS["synonym"], RELATION_COMPONENT_WEIGHTS["sibling"]],
        device=device,
    )
    relation_weights = relation_weights / relation_weights.sum()
    total = relation_weights[0] * synonym_loss + relation_weights[1] * sibling_loss
    return total, {
        "synonym_response_gap_loss": synonym_response_loss,
        "synonym_relevance_gap_loss": synonym_relevance_loss,
        "synonym_loss": synonym_loss,
        "sibling_bidirectional_relevance_gap_loss": sibling_loss,
        "typed_relation_loss": total,
        "synonym_pair_region_units": synonym_units,
        "sibling_left_dominant_units": left_units,
        "sibling_right_dominant_units": right_units,
    }


__all__ = [
    "FrozenTypedTextRelationAuthority",
    "TYPED_RELATION_AUXILIARY_WEIGHT",
    "load_frozen_typed_text_relation_authority",
    "source_typed_text_relation_loss_v21",
]
