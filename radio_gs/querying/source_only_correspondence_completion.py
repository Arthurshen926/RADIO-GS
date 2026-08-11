"""Bounded source-only completion for registered primitive likelihoods.

The operator in this module is deliberately local.  It transfers signed
reference evidence across exactly one precomputed KNN edge and only writes
rows on which the registered observation abstained.  Appearance, boundary,
and source-view co-visibility must all support the transfer; an unobserved
component is therefore never filled merely because it is connected in XYZ.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .query_spec import PrimitiveUnaryEvidence
from .support_solver import PrimitiveSupportGraph


@dataclass(frozen=True)
class SourceOnlyCorrespondenceDiagnostics:
    num_nodes: int
    observed_rows: int
    positive_anchor_rows: int
    negative_anchor_rows: int
    abstained_rows: int
    completed_rows: int
    completed_confidence_sum: float
    completed_mean_absolute_probability_shift: float


def method_contract() -> dict[str, object]:
    """Return the target-independent scientific contract."""

    return {
        "schema_version": 1,
        "method": "source_only_one_hop_signed_correspondence_completion_v1",
        "write_domain": "registered_observation_confidence_equals_exact_zero",
        "anchor_domain": "abs_registered_signed_value_at_least_hard_seed_threshold",
        "anchor_labels": "sign_of_registered_signed_value",
        "topology": "one_hop_frozen_geometry_knn_edges",
        "relation_affinity": (
            "min(frozen_DINO_appearance_affinity,frozen_SAM_boundary_affinity)"
            "*source_view_covisibility_jaccard"
        ),
        "structural_probability": "positive_anchor_mass/(positive+negative_anchor_mass)",
        "prototype_probability": "frozen_positive_negative_primitive_prototype_field_probability",
        "completion_probability": (
            "normalized_product_of_structural_and_prototype_Bernoulli_likelihoods"
        ),
        "completion_confidence": (
            "anchor_relation_coverage*structural_signed_consensus*"
            "prototype_structural_agreement*query_independent_view_reliability"
        ),
        "query_independent_view_reliability": "source_visible_view_fraction",
        "observed_rows_rewritten": False,
        "multi_hop_diffusion": False,
        "connected_selection": False,
        "uses_target_rgb_mask_or_metric": False,
        "learned_or_scene_specific_constants": False,
    }


def _finite_vector(
    value: torch.Tensor,
    *,
    count: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device).float().reshape(-1)
    if result.shape != (count,) or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be a finite [num_nodes] vector")
    return result


def source_only_one_hop_correspondence_completion(
    graph: PrimitiveSupportGraph,
    observation: PrimitiveUnaryEvidence,
    prototype_probability: torch.Tensor,
    source_view_observations: torch.Tensor,
    *,
    hard_seed_threshold: float,
    query_independent_reliability: torch.Tensor | None = None,
) -> tuple[PrimitiveUnaryEvidence, SourceOnlyCorrespondenceDiagnostics]:
    """Complete exact-abstain rows from signed, source-only correspondence.

    ``observation`` must obey ``u=c(2q-1)``.  Existing ``u`` and ``c`` are
    cloned and never recomputed on rows with ``c>0``.  The returned evidence
    can consequently replace the input evidence in the existing probability
    mixture without weakening a scribble, mask observation, or hard anchor.
    """

    if observation.confidence is None:
        raise ValueError("source-only completion requires explicit confidence")
    threshold = float(hard_seed_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("hard_seed_threshold must be in (0,1]")
    if "appearance" not in graph.edge_channels or "boundary" not in graph.edge_channels:
        raise ValueError("completion requires typed appearance and boundary edges")

    device = graph.edge_index.device
    count = int(graph.num_nodes)
    values = _finite_vector(
        observation.values, count=count, name="observation.values", device=device
    )
    confidence = _finite_vector(
        observation.confidence,
        count=count,
        name="observation.confidence",
        device=device,
    )
    field = _finite_vector(
        prototype_probability,
        count=count,
        name="prototype_probability",
        device=device,
    )
    if bool(((confidence < 0) | (confidence > 1)).any()):
        raise ValueError("observation confidence must be in [0,1]")
    if bool((values.abs() > confidence + 1e-6).any()):
        raise ValueError("observation values must satisfy abs(u)<=c")
    if bool(((field < 0) | (field > 1)).any()):
        raise ValueError("prototype_probability must be in [0,1]")

    visibility = torch.as_tensor(source_view_observations, device=device).bool()
    if visibility.ndim != 2 or visibility.shape[0] != count or visibility.shape[1] <= 0:
        raise ValueError("source_view_observations must be [num_nodes,num_source_views]")
    visible_fraction = visibility.float().mean(dim=1)
    if query_independent_reliability is None:
        reliability = visible_fraction
    else:
        supplied = _finite_vector(
            query_independent_reliability,
            count=count,
            name="query_independent_reliability",
            device=device,
        )
        if bool(((supplied < 0) | (supplied > 1)).any()):
            raise ValueError("query_independent_reliability must be in [0,1]")
        # Source visibility is an authority condition, not a replacement for
        # an optional learned/query-independent precision estimate.
        reliability = supplied * visible_fraction

    observed = confidence > 0
    positive_anchor = observed & (values >= threshold)
    negative_anchor = observed & (values <= -threshold)
    row, col = graph.edge_index
    appearance = torch.as_tensor(
        graph.edge_channels["appearance"], device=device
    ).float()
    boundary = torch.as_tensor(
        graph.edge_channels["boundary"], device=device
    ).float()
    if appearance.shape != row.shape or boundary.shape != row.shape:
        raise ValueError("typed edge affinities must align with edge_index")
    if bool((appearance < 0).any()) or bool((boundary < 0).any()):
        raise ValueError("typed edge affinities must be non-negative")

    shared = (visibility[row] & visibility[col]).sum(dim=1).float()
    union = (visibility[row] | visibility[col]).sum(dim=1).float()
    covisibility = shared / union.clamp_min(1.0)
    # ``min`` is an agreement-limited t-norm: a strong semantic relation
    # cannot erase a SAM boundary, and a boundary-like match alone cannot
    # manufacture instance identity.
    relation = torch.minimum(appearance, boundary) * covisibility

    total_relation = torch.zeros(count, device=device)
    positive_mass = torch.zeros(count, device=device)
    negative_mass = torch.zeros(count, device=device)
    total_relation.index_add_(0, row, relation)
    positive_mass.index_add_(
        0, row, relation * values[col].clamp_min(0.0) * positive_anchor[col]
    )
    negative_mass.index_add_(
        0, row, relation * (-values[col]).clamp_min(0.0) * negative_anchor[col]
    )
    anchor_mass = positive_mass + negative_mass
    supported = anchor_mass > 0
    structural_probability = torch.where(
        supported,
        positive_mass / anchor_mass.clamp_min(1e-30),
        torch.full_like(anchor_mass, 0.5),
    )
    structural_coverage = torch.where(
        total_relation > 0,
        anchor_mass / total_relation.clamp_min(1e-30),
        torch.zeros_like(anchor_mass),
    ).clamp(0.0, 1.0)
    structural_consensus = (2.0 * structural_probability - 1.0).abs()
    prototype_agreement = 1.0 - (field - structural_probability).abs()

    positive_product = field * structural_probability
    negative_product = (1.0 - field) * (1.0 - structural_probability)
    product_normalizer = positive_product + negative_product
    completed_probability = torch.where(
        product_normalizer > 0,
        positive_product / product_normalizer.clamp_min(1e-30),
        torch.full_like(product_normalizer, 0.5),
    ).clamp(0.0, 1.0)
    completed_confidence = (
        structural_coverage
        * structural_consensus
        * prototype_agreement.clamp(0.0, 1.0)
        * reliability
    ).clamp(0.0, 1.0)
    completion_rows = (~observed) & supported & (completed_confidence > 0)

    output_values = values.clone()
    output_confidence = confidence.clone()
    output_confidence[completion_rows] = completed_confidence[completion_rows]
    output_values[completion_rows] = completed_confidence[completion_rows] * (
        2.0 * completed_probability[completion_rows] - 1.0
    )
    if not torch.equal(output_values[observed], values[observed]) or not torch.equal(
        output_confidence[observed], confidence[observed]
    ):
        raise RuntimeError("source-only completion rewrote an observed row")

    mean_shift = (
        float((completed_probability[completion_rows] - field[completion_rows]).abs().mean())
        if bool(completion_rows.any())
        else 0.0
    )
    diagnostics = SourceOnlyCorrespondenceDiagnostics(
        num_nodes=count,
        observed_rows=int(observed.sum()),
        positive_anchor_rows=int(positive_anchor.sum()),
        negative_anchor_rows=int(negative_anchor.sum()),
        abstained_rows=int((~observed).sum()),
        completed_rows=int(completion_rows.sum()),
        completed_confidence_sum=float(completed_confidence[completion_rows].sum()),
        completed_mean_absolute_probability_shift=mean_shift,
    )
    return (
        PrimitiveUnaryEvidence(
            output_values,
            "source_only_one_hop_signed_correspondence_completion_v1",
            output_confidence,
        ),
        diagnostics,
    )
