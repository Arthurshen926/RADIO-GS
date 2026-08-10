"""Absolute-relevance source supervision for bounded V2.1 residuals.

V2.1 is an additive source-only loss.  It leaves the frozen V2 module
byte-identical, consumes no query strings, and delegates the student
canonical-negative probability to the same pure relevance function used by
inference.  Optional target-blind compositional banks extend only the query
axis and carry independent file and tensor authorities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.losses import source_global_response_listwise_loss as v2
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    FrozenTypedTextRelationAuthority,
    TYPED_RELATION_AUXILIARY_WEIGHT,
    source_typed_text_relation_loss_v21,
)
from radio_gs.querying.unified_query import (
    cosine_relevancy_torch,
    margin_to_relevancy_torch,
)
from radio_gs.utils.immutable_artifacts import load_torch_mapping


INFERENCE_LOGIT_SCALE = 10.0
O4_TEACHER_RESPONSE_TEMPERATURE = 0.1
SMOOTH_L1_BETA = 0.05
CONTINUOUS_GAP_SIGMA = 0.05
STABILITY_STD_SCALE = 0.10
RECOMMENDED_AUXILIARY_WEIGHT = 0.25
PRIMARY_COMPONENT_WEIGHT = 0.25
QUERY_CHUNK_ROWS = 128
PAIR_CHUNK_ROWS = 4096
CANONICAL_NEGATIVE_ROWS = 4
CANONICAL_NEGATIVE_MODEL = "google/siglip2-giant-opt-patch16-384"


@dataclass(frozen=True)
class SourceGlobalResponseLossV21Config:
    auxiliary_weight: float = RECOMMENDED_AUXILIARY_WEIGHT
    primary_component_weight: float = PRIMARY_COMPONENT_WEIGHT
    response_temperature: float = O4_TEACHER_RESPONSE_TEMPERATURE
    inference_logit_scale: float = INFERENCE_LOGIT_SCALE
    smooth_l1_beta: float = SMOOTH_L1_BETA
    continuous_gap_sigma: float = CONTINUOUS_GAP_SIGMA
    stability_std_scale: float = STABILITY_STD_SCALE
    triplet_margin_ceiling: float = v2.TRIPLET_MARGIN_CEILING
    triplet_margin_scale: float = v2.TRIPLET_MARGIN_SCALE
    query_chunk_rows: int = QUERY_CHUNK_ROWS
    pair_chunk_rows: int = PAIR_CHUNK_ROWS

    def __post_init__(self) -> None:
        values = {
            "auxiliary_weight": self.auxiliary_weight,
            "primary_component_weight": self.primary_component_weight,
            "response_temperature": self.response_temperature,
            "inference_logit_scale": self.inference_logit_scale,
            "smooth_l1_beta": self.smooth_l1_beta,
            "continuous_gap_sigma": self.continuous_gap_sigma,
            "stability_std_scale": self.stability_std_scale,
            "triplet_margin_ceiling": self.triplet_margin_ceiling,
            "triplet_margin_scale": self.triplet_margin_scale,
        }
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("V2.1 response-loss scalars must be finite")
        if self.auxiliary_weight < 0:
            raise ValueError("auxiliary_weight must be nonnegative")
        if self.primary_component_weight <= 0:
            raise ValueError("primary_component_weight must be positive")
        positive = (
            self.response_temperature,
            self.inference_logit_scale,
            self.smooth_l1_beta,
            self.continuous_gap_sigma,
            self.stability_std_scale,
        )
        if any(float(value) <= 0 for value in positive):
            raise ValueError("V2.1 temperatures and scales must be positive")
        if self.triplet_margin_ceiling < 0 or self.triplet_margin_scale < 0:
            raise ValueError("triplet margin constants must be nonnegative")
        if self.query_chunk_rows <= 0 or self.pair_chunk_rows <= 0:
            raise ValueError("query and pair chunk sizes must be positive")


@dataclass(frozen=True)
class FrozenCanonicalNegativeBank:
    embeddings: torch.Tensor
    file_sha256: str
    embedding_tensor_sha256: str
    model_id: str


@dataclass(frozen=True)
class FrozenCompositionalGenericBank:
    component_id: str
    embeddings: torch.Tensor
    file_sha256: str
    embedding_tensor_sha256: str
    model_id: str
    query_rows: int
    loss_weight: float


def recommended_v21_config() -> SourceGlobalResponseLossV21Config:
    """Return the sole preregistered V2.1 configuration."""

    return SourceGlobalResponseLossV21Config()


def zero_weight_v21_config() -> SourceGlobalResponseLossV21Config:
    """Return the exact base-objective compatibility configuration."""

    return SourceGlobalResponseLossV21Config(auxiliary_weight=0.0)


def _normalized_cpu_embeddings(value: object, *, label: str) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or value.ndim != 2
        or min(value.shape) <= 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be finite CPU float32 [N,D]")
    norms = torch.linalg.vector_norm(value, dim=-1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=0)):
        raise ValueError(f"{label} rows must be L2 normalized")
    return value.detach().contiguous()


def load_frozen_canonical_negative_bank(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> FrozenCanonicalNegativeBank:
    """Load the protocol-generic canonical-negative cache fail closed."""

    payload, digest, _ = load_torch_mapping(
        path,
        expected_sha256=v2._lower_sha256(
            expected_file_sha256, label="canonical-negative bank"
        ),
        map_location="cpu",
        label="canonical-negative text bank",
    )
    required = {
        "embeddings",
        "model_name",
        "prompt_templates",
        "queries",
        "text_encoder",
    }
    if set(payload) != required:
        raise ValueError("canonical-negative bank fields differ")
    queries = payload["queries"]
    if (
        payload["text_encoder"] != "siglip2"
        or payload["model_name"] != CANONICAL_NEGATIVE_MODEL
        or payload["prompt_templates"] != ["{query}"]
        or not isinstance(queries, list)
        or len(queries) != CANONICAL_NEGATIVE_ROWS
        or any(not isinstance(item, str) or not item.strip() for item in queries)
        or len(set(queries)) != len(queries)
    ):
        raise ValueError("canonical-negative text semantics differ")
    embeddings = _normalized_cpu_embeddings(
        payload["embeddings"], label="canonical-negative embeddings"
    )
    if embeddings.shape[0] != CANONICAL_NEGATIVE_ROWS:
        raise ValueError("canonical-negative row count differs")
    return FrozenCanonicalNegativeBank(
        embeddings=embeddings,
        file_sha256=digest,
        embedding_tensor_sha256=tensor_sha256(embeddings),
        model_id=CANONICAL_NEGATIVE_MODEL,
    )


def load_frozen_compositional_generic_bank(
    path: str | Path,
    *,
    expected_file_sha256: str,
    component_id: str,
    loss_weight: float,
) -> FrozenCompositionalGenericBank:
    """Load one independently authorized target-blind fit-bank component."""

    identifier = str(component_id).strip()
    if not identifier or any(
        ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in identifier
    ):
        raise ValueError("compositional component_id must be lowercase snake case")
    if not math.isfinite(float(loss_weight)) or float(loss_weight) <= 0:
        raise ValueError("compositional loss_weight must be finite and positive")
    payload, digest, _ = load_torch_mapping(
        path,
        expected_sha256=v2._lower_sha256(
            expected_file_sha256, label="compositional generic bank"
        ),
        map_location="cpu",
        label=f"target-blind compositional bank {identifier}",
    )
    queries = payload.get("queries")
    text_encoder = payload.get("text_encoder")
    model_id = (
        text_encoder.get("model_id") if isinstance(text_encoder, Mapping) else None
    )
    if (
        payload.get("artifact_type")
        not in {
            "target_blind_text_embedding_cache",
            "target_blind_compositional_text_embedding_cache",
        }
        or payload.get("split") != "fit"
        or payload.get("benchmark_vocabulary_opened") is not False
        or payload.get("uses_benchmark_vocabulary_for_construction") is not False
        or payload.get("prompt_templates") != ["{query}"]
        or model_id != CANONICAL_NEGATIVE_MODEL
        or not isinstance(queries, list)
        or not queries
        or any(not isinstance(item, str) or not item.strip() for item in queries)
        or len(set(queries)) != len(queries)
    ):
        raise ValueError("compositional generic bank source contract differs")
    embeddings = _normalized_cpu_embeddings(
        payload.get("embeddings"), label="compositional generic embeddings"
    )
    if embeddings.shape[0] != len(queries):
        raise ValueError("compositional query and embedding rows differ")
    embedding_sha = tensor_sha256(embeddings)
    if payload.get("embedding_tensor_sha256") != embedding_sha:
        raise ValueError("compositional embedding tensor authority differs")
    return FrozenCompositionalGenericBank(
        component_id=identifier,
        embeddings=embeddings,
        file_sha256=digest,
        embedding_tensor_sha256=embedding_sha,
        model_id=str(model_id),
        query_rows=int(embeddings.shape[0]),
        loss_weight=float(loss_weight),
    )


def _weighted_component_mean(
    values: Sequence[torch.Tensor], weights: Sequence[float]
) -> torch.Tensor:
    if not values or len(values) != len(weights):
        raise ValueError("component losses and weights must be nonempty and aligned")
    reference = values[0]
    if any(value.ndim != 0 or value.device != reference.device for value in values):
        raise ValueError("component losses must be scalar tensors on one device")
    weight = torch.as_tensor(weights, dtype=torch.float32, device=reference.device)
    if not bool(torch.isfinite(weight).all()) or bool((weight <= 0).any()):
        raise ValueError("component weights must be finite and positive")
    return (torch.stack(tuple(values)) * (weight / weight.sum())).sum()


def _multiview_relevance_std(
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    positive_text: torch.Tensor,
    canonical_negative_text: torch.Tensor,
    *,
    logit_scale: float,
) -> torch.Tensor:
    positive = torch.einsum("rvd,qd->rvq", teacher_views, positive_text)
    negative = torch.einsum("rvd,kd->rvk", teacher_views, canonical_negative_text).amax(
        dim=-1
    )
    per_view = margin_to_relevancy_torch(
        positive - negative[..., None], logit_scale=logit_scale
    )
    mask = teacher_mask[..., None]
    count = mask.sum(dim=1).clamp_min(1)
    mean = (per_view * mask).sum(dim=1) / count
    variance = (((per_view - mean[:, None, :]) ** 2) * mask).sum(dim=1) / count
    return torch.sqrt(variance.clamp_min(0.0)).detach()


def source_global_response_listwise_loss_v21(
    base_loss: torch.Tensor,
    student_descriptors: torch.Tensor,
    teacher_pair_descriptors: torch.Tensor,
    teacher_pair_region_indices: torch.Tensor,
    fit_text_embeddings: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    authority: v2.FrozenSourceResponseAuthority,
    canonical_negative_bank: FrozenCanonicalNegativeBank,
    *,
    accepted_v2_file_sha256: str,
    teacher_file_sha256: str,
    teacher_pair_descriptors_sha256: str,
    fit_text_bank_file_sha256: str,
    compositional_banks: Sequence[FrozenCompositionalGenericBank] = (),
    relation_authority: FrozenTypedTextRelationAuthority | None = None,
    trainable_region_mask: torch.Tensor | None = None,
    exclude_both_immutable_pairs: bool = False,
    config: SourceGlobalResponseLossV21Config | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Add absolute response/relevance and continuously weighted ordering.

    The positive query axis is the SHA-bound primary fit bank followed by zero
    or more independently SHA-bound compositional components.  Query strings
    are never passed to this function.
    """

    chosen = recommended_v21_config() if config is None else config
    if not isinstance(chosen, SourceGlobalResponseLossV21Config):
        raise TypeError("config must be SourceGlobalResponseLossV21Config")
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
            "absolute_response_loss": zero,
            "response_fidelity_loss": zero,
            "absolute_relevance_loss": zero,
            "continuous_pairwise_relevance_loss": zero,
            "hard_negative_triplet_loss": zero,
            "auxiliary_loss": zero,
            "valid_profile_queries": 0,
            "positive_weight_pair_query_units": 0,
            "small_margin_positive_weight_units": 0,
            "hard_negative_pairs": 0,
            "authority_hard_negative_pairs": 0,
            "objective_hard_negative_pairs": 0,
            "both_immutable_pairs_excluded": 0,
            "pair_trainable_endpoint_coverage": zero,
            "generic_query_rows": 0,
            "compositional_query_rows": 0,
            "generic_bank_components": 0,
            "generic_bank_component_weight_sum": zero,
            "typed_relation_loss": zero,
            "typed_relation_active": 0,
        }
    if type(authority) is not v2.FrozenSourceResponseAuthority:
        raise TypeError("authority must be a fail-closed frozen V2 authority")
    if type(canonical_negative_bank) is not FrozenCanonicalNegativeBank:
        raise TypeError("canonical_negative_bank must be frozen and SHA-bound")

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
        if v2._lower_sha256(actual, label=label) != expected:
            raise ValueError(f"runtime {label} binding differs")

    device = student_descriptors.device
    student_raw = v2._finite_float_matrix(
        student_descriptors, label="student_descriptors", device=device
    )
    teacher_raw = v2._finite_float_matrix(
        teacher_pair_descriptors,
        label="teacher_pair_descriptors",
        device=device,
    )
    primary_raw = v2._finite_float_matrix(
        fit_text_embeddings, label="fit_text_embeddings", device=device
    )
    if (
        tensor_sha256(primary_raw.detach().cpu())
        != authority.fit_text_embedding_tensor_sha256
    ):
        raise ValueError("runtime fit text embedding tensor differs")
    canonical = torch.as_tensor(canonical_region_indices).detach().long().cpu()
    if (
        canonical.shape != authority.payload["canonical_region_indices"].shape
        or not torch.equal(canonical, authority.payload["canonical_region_indices"])
        or student_raw.shape[0] != canonical.numel()
    ):
        raise ValueError("loss requires the complete canonical scene row order")

    optional: list[torch.Tensor] = []
    optional_weights: list[float] = []
    optional_by_id: dict[str, FrozenCompositionalGenericBank] = {}
    optional_rows = 0
    seen_components: set[tuple[str, str]] = set()
    for component in compositional_banks:
        if type(component) is not FrozenCompositionalGenericBank:
            raise TypeError("compositional banks must be frozen and SHA-bound")
        key = (component.component_id, component.file_sha256)
        if key in seen_components:
            raise ValueError("duplicate compositional bank component")
        seen_components.add(key)
        if component.model_id != canonical_negative_bank.model_id:
            raise ValueError("compositional and canonical-negative encoders differ")
        values = component.embeddings.to(device)
        if values.shape[1] != student_raw.shape[1]:
            raise ValueError("compositional and descriptor dimensions differ")
        if tensor_sha256(values.detach().cpu()) != component.embedding_tensor_sha256:
            raise ValueError("runtime compositional embedding tensor differs")
        optional.append(values)
        optional_weights.append(float(component.loss_weight))
        optional_by_id[component.component_id] = component
        optional_rows += int(values.shape[0])

    if (
        student_raw.shape[1] != teacher_raw.shape[1]
        or student_raw.shape[1] != primary_raw.shape[1]
        or canonical_negative_bank.embeddings.shape[1] != student_raw.shape[1]
    ):
        raise ValueError("student, teacher, positive, and negative dimensions differ")
    student = F.normalize(student_raw.float(), dim=-1)
    primary_text = F.normalize(primary_raw.detach().float(), dim=-1)
    normalized_optional = [F.normalize(values.float(), dim=-1) for values in optional]
    text_components = [primary_text, *normalized_optional]
    text_by_id = {
        component.component_id: text
        for component, text in zip(compositional_banks, normalized_optional)
    }
    component_weights = [
        float(chosen.primary_component_weight),
        *optional_weights,
    ]
    negative_text = F.normalize(
        canonical_negative_bank.embeddings.to(device).float(), dim=-1
    )
    teacher_views, teacher_mask, teacher_consensus = v2._teacher_views(
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
    if trainable_region_mask is None:
        if exclude_both_immutable_pairs:
            raise ValueError(
                "exclude_both_immutable_pairs requires a trainable region mask"
            )
        pair_has_trainable_endpoint = torch.ones_like(anchors, dtype=torch.bool)
    else:
        trainable = torch.as_tensor(trainable_region_mask).detach()
        if (
            trainable.dtype != torch.bool
            or trainable.ndim != 1
            or trainable.numel() != student.shape[0]
        ):
            raise ValueError("trainable_region_mask must be bool [canonical rows]")
        trainable = trainable.to(device)
        pair_has_trainable_endpoint = trainable[anchors] | trainable[negatives]
    objective_pair_mask = (
        pair_has_trainable_endpoint
        if exclude_both_immutable_pairs
        else torch.ones_like(pair_has_trainable_endpoint)
    )
    objective_pair_count = int(objective_pair_mask.sum())
    if objective_pair_count <= 0:
        raise ValueError(
            "V2.1 objective has no hard-negative pair with a trainable endpoint"
        )
    authority_pair_count = int(anchors.numel())
    trainable_endpoint_pair_count = int(pair_has_trainable_endpoint.sum())
    teacher_negative_response = v2._teacher_response_chunk(
        teacher_views,
        teacher_mask,
        negative_text,
        temperature=chosen.response_temperature,
    )

    zero = student.sum() * 0.0
    component_profile_losses: list[torch.Tensor] = []
    component_absolute_response_losses: list[torch.Tensor] = []
    component_absolute_relevance_losses: list[torch.Tensor] = []
    component_pairwise_losses: list[torch.Tensor] = []
    profile_count = 0
    pair_count = 0
    small_margin_count = 0

    for text in text_components:
        component_profile_sum = zero
        component_absolute_response_sum = zero
        component_absolute_relevance_sum = zero
        component_pair_sum = zero
        component_pair_weight_sum = zero.detach()
        component_profile_count = 0
        component_response_count = 0
        component_pair_count = 0
        for query_start in range(0, text.shape[0], chosen.query_chunk_rows):
            query_stop = min(query_start + chosen.query_chunk_rows, text.shape[0])
            query = text[query_start:query_stop]
            student_response = student @ query.T
            teacher_response = v2._teacher_response_chunk(
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
                component_profile_sum = component_profile_sum + (1.0 - profile).sum()
                component_profile_count += int(profile.numel())

            component_absolute_response_sum = (
                component_absolute_response_sum
                + F.smooth_l1_loss(
                    student_response,
                    teacher_response,
                    beta=float(chosen.smooth_l1_beta),
                    reduction="sum",
                )
            )
            student_relevance = cosine_relevancy_torch(
                student,
                query,
                negative_text,
                logit_scale=float(chosen.inference_logit_scale),
                assume_normalized=True,
            )
            teacher_relevance = margin_to_relevancy_torch(
                teacher_response - teacher_negative_response.amax(dim=-1, keepdim=True),
                logit_scale=float(chosen.inference_logit_scale),
            )
            component_absolute_relevance_sum = (
                component_absolute_relevance_sum
                + F.smooth_l1_loss(
                    student_relevance,
                    teacher_relevance,
                    beta=float(chosen.smooth_l1_beta),
                    reduction="sum",
                )
            )
            component_response_count += int(student_response.numel())

            relevance_std = _multiview_relevance_std(
                teacher_views,
                teacher_mask,
                query,
                negative_text,
                logit_scale=float(chosen.inference_logit_scale),
            )
            for pair_start in range(0, anchors.numel(), chosen.pair_chunk_rows):
                pair_stop = min(pair_start + chosen.pair_chunk_rows, anchors.numel())
                retained = objective_pair_mask[pair_start:pair_stop]
                if not bool(retained.any()):
                    continue
                a = anchors[pair_start:pair_stop][retained]
                n = negatives[pair_start:pair_stop][retained]
                teacher_gap = teacher_relevance[a] - teacher_relevance[n]
                student_gap = student_relevance[a] - student_relevance[n]
                absolute_gap = teacher_gap.abs()
                stability = torch.exp(
                    -(relevance_std[a] + relevance_std[n])
                    / float(chosen.stability_std_scale)
                )
                weight = (
                    stability
                    * absolute_gap
                    / (absolute_gap + float(chosen.continuous_gap_sigma))
                )
                unit = F.smooth_l1_loss(
                    student_gap,
                    teacher_gap,
                    beta=float(chosen.smooth_l1_beta),
                    reduction="none",
                )
                component_pair_sum = component_pair_sum + (weight * unit).sum()
                component_pair_weight_sum = (
                    component_pair_weight_sum + weight.sum().detach()
                )
                positive_weight = weight > 0
                units = int(positive_weight.sum())
                component_pair_count += units
                pair_count += units
                small_margin_count += int(
                    (positive_weight & (absolute_gap < CONTINUOUS_GAP_SIGMA)).sum()
                )

        if (
            component_profile_count <= 0
            or component_response_count <= 0
            or component_pair_count <= 0
        ):
            raise ValueError(
                "each V2.1 query-bank component requires valid response and pair units"
            )
        if (
            not bool(torch.isfinite(component_pair_weight_sum))
            or float(component_pair_weight_sum) <= 0
        ):
            raise ValueError(
                "a V2.1 query-bank component has zero pairwise authority weight"
            )
        component_profile_losses.append(component_profile_sum / component_profile_count)
        component_absolute_response_losses.append(
            component_absolute_response_sum / component_response_count
        )
        component_absolute_relevance_losses.append(
            component_absolute_relevance_sum / component_response_count
        )
        component_pairwise_losses.append(component_pair_sum / component_pair_weight_sum)
        profile_count += component_profile_count

    profile_loss = _weighted_component_mean(component_profile_losses, component_weights)
    absolute_response_loss = _weighted_component_mean(
        component_absolute_response_losses, component_weights
    )
    response_fidelity_loss = 0.5 * (profile_loss + absolute_response_loss)
    absolute_relevance_loss = _weighted_component_mean(
        component_absolute_relevance_losses, component_weights
    )
    pairwise_loss = _weighted_component_mean(
        component_pairwise_losses, component_weights
    )

    objective_anchors = anchors[objective_pair_mask]
    objective_negatives = negatives[objective_pair_mask]
    objective_teacher_cosine = declared_teacher_cosine[objective_pair_mask]
    positive = (student[objective_anchors] * teacher_consensus[objective_anchors]).sum(
        dim=-1
    )
    negative = (
        student[objective_anchors] * teacher_consensus[objective_negatives]
    ).sum(dim=-1)
    margin = (
        float(chosen.triplet_margin_scale)
        * (1.0 - objective_teacher_cosine).clamp_min(0.0)
    ).clamp_max(float(chosen.triplet_margin_ceiling))
    triplet_loss = F.relu(margin - positive + negative).mean()
    base_auxiliary = (
        response_fidelity_loss + absolute_relevance_loss + pairwise_loss + triplet_loss
    ) / 4.0
    relation_loss = zero
    relation_metrics: dict[str, torch.Tensor | int] = {
        "synonym_response_gap_loss": zero,
        "synonym_relevance_gap_loss": zero,
        "synonym_loss": zero,
        "sibling_bidirectional_relevance_gap_loss": zero,
        "typed_relation_loss": zero,
        "synonym_pair_region_units": 0,
        "sibling_left_dominant_units": 0,
        "sibling_right_dominant_units": 0,
    }
    if relation_authority is not None:
        if type(relation_authority) is not FrozenTypedTextRelationAuthority:
            raise TypeError(
                "relation_authority must be FrozenTypedTextRelationAuthority"
            )
        required_optional = {
            "synonym_relation",
            "lexical_sibling_relation",
            "counterfactual_attributes",
            "high_precision_part_of",
        }
        if set(optional_by_id) != required_optional:
            raise ValueError(
                "typed relation V2.1 requires all four compositional components"
            )
        runtime_components = {
            "primary": {
                "sha256": authority.fit_text_bank_file_sha256,
                "embedding_tensor_sha256": authority.fit_text_embedding_tensor_sha256,
                "query_rows": int(primary_text.shape[0]),
            },
            **{
                name: {
                    "sha256": component.file_sha256,
                    "embedding_tensor_sha256": component.embedding_tensor_sha256,
                    "query_rows": component.query_rows,
                }
                for name, component in optional_by_id.items()
            },
        }
        for name, runtime in runtime_components.items():
            sealed = relation_authority.components[name]
            if any(runtime[field] != sealed[field] for field in runtime):
                raise ValueError(f"typed relation runtime component differs: {name}")
        relation_loss, relation_metrics = source_typed_text_relation_loss_v21(
            student,
            teacher_views,
            teacher_mask,
            primary_text,
            text_by_id["synonym_relation"],
            negative_text,
            teacher_negative_response,
            relation_authority,
            response_temperature=float(chosen.response_temperature),
            logit_scale=float(chosen.inference_logit_scale),
            smooth_l1_beta=float(chosen.smooth_l1_beta),
            continuous_gap_sigma=float(chosen.continuous_gap_sigma),
            stability_std_scale=float(chosen.stability_std_scale),
            pair_chunk_rows=int(chosen.pair_chunk_rows),
        )
    auxiliary = base_auxiliary + float(TYPED_RELATION_AUXILIARY_WEIGHT) * relation_loss
    total = base_loss + float(chosen.auxiliary_weight) * auxiliary
    if not bool(torch.isfinite(total.detach())):
        raise RuntimeError("source-global V2.1 response objective is nonfinite")
    return total, {
        "centered_response_profile_loss": profile_loss,
        "absolute_response_loss": absolute_response_loss,
        "response_fidelity_loss": response_fidelity_loss,
        "absolute_relevance_loss": absolute_relevance_loss,
        "continuous_pairwise_relevance_loss": pairwise_loss,
        "hard_negative_triplet_loss": triplet_loss,
        "auxiliary_loss": auxiliary,
        "valid_profile_queries": profile_count,
        "positive_weight_pair_query_units": pair_count,
        "small_margin_positive_weight_units": small_margin_count,
        "hard_negative_pairs": authority_pair_count,
        "authority_hard_negative_pairs": authority_pair_count,
        "objective_hard_negative_pairs": objective_pair_count,
        "both_immutable_pairs_excluded": (authority_pair_count - objective_pair_count),
        "pair_trainable_endpoint_coverage": student.new_tensor(
            trainable_endpoint_pair_count / authority_pair_count
        ),
        "generic_query_rows": int(sum(item.shape[0] for item in text_components)),
        "compositional_query_rows": optional_rows,
        "generic_bank_components": len(text_components),
        "generic_bank_component_weight_sum": student.new_tensor(sum(component_weights)),
        **relation_metrics,
        "typed_relation_active": int(relation_authority is not None),
    }


__all__ = [
    "FrozenCanonicalNegativeBank",
    "FrozenCompositionalGenericBank",
    "SourceGlobalResponseLossV21Config",
    "load_frozen_canonical_negative_bank",
    "load_frozen_compositional_generic_bank",
    "recommended_v21_config",
    "source_global_response_listwise_loss_v21",
    "zero_weight_v21_config",
]
