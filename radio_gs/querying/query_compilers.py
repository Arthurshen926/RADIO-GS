"""Compile text, image, registered-2D, and world-3D inputs into 3-D evidence."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from radio_gs.field.field_signature import FeatureSpaceSignature
from .query_spec import (
    PrototypeSet,
    QueryIntent,
    QueryModality,
    QuerySpec,
    RegistrationMode,
    SelectionMode,
    SoftSeedSet,
)


def _deterministic_prototypes(
    features: torch.Tensor,
    weights: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted farthest-point prototypes; no learned or benchmark state."""

    values = F.normalize(torch.as_tensor(features).float(), dim=-1, eps=1e-8)
    masses = torch.as_tensor(weights).float().reshape(-1).to(values.device)
    valid = masses > 0
    values = values[valid]
    masses = masses[valid]
    if values.shape[0] == 0:
        raise ValueError("cannot build prototypes from empty support")
    count = min(max(1, int(count)), values.shape[0])
    selected = [int(masses.argmax())]
    nearest_distance = 1.0 - values @ values[selected[0]]
    for _ in range(1, count):
        score = nearest_distance * masses
        index = int(score.argmax())
        selected.append(index)
        nearest_distance = torch.minimum(
            nearest_distance, 1.0 - values @ values[index]
        )
    prototypes = values[selected]
    similarities = values @ prototypes.T
    assignment = similarities.argmax(dim=1)
    prototype_masses = torch.stack(
        [masses[assignment == index].sum() for index in range(count)]
    )
    prototype_masses /= prototype_masses.sum().clamp_min(1e-8)
    return prototypes, prototype_masses


def compile_text_query(
    positive_text_embeddings: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
    *,
    signature: FeatureSpaceSignature,
    intent: QueryIntent = QueryIntent.CATEGORY,
    granularity_m: Iterable[float] = (),
) -> QuerySpec:
    return QuerySpec(
        modality=QueryModality.TEXT,
        intent=intent,
        registration=RegistrationMode.NONE,
        semantic_evidence=PrototypeSet(
            positive_text_embeddings,
            signature,
            negatives=negative_text_embeddings,
        ),
        granularity_m=tuple(granularity_m),
        selection_mode=(
            SelectionMode.ALL_COMPONENTS
            if QueryIntent(intent) is QueryIntent.CATEGORY
            else SelectionMode.TOP_COMPONENT
        ),
        field_signature=signature,
    )


def compile_image_query(
    semantic_summary: torch.Tensor,
    appearance_tokens: torch.Tensor,
    *,
    semantic_signature: FeatureSpaceSignature,
    appearance_signature: FeatureSpaceSignature,
    semantic_negatives: torch.Tensor | None = None,
    appearance_negatives: torch.Tensor | None = None,
    foreground_weights: torch.Tensor | None = None,
    prototype_count: int = 4,
) -> QuerySpec:
    tokens = torch.as_tensor(appearance_tokens).float()
    if tokens.ndim != 2:
        raise ValueError("appearance_tokens must be [P,D]")
    weights = (
        torch.ones(tokens.shape[0], dtype=torch.float32, device=tokens.device)
        if foreground_weights is None
        else torch.as_tensor(foreground_weights).float().to(tokens.device)
    )
    prototypes, masses = _deterministic_prototypes(tokens, weights, prototype_count)
    return QuerySpec(
        modality=QueryModality.IMAGE,
        intent=QueryIntent.INSTANCE,
        registration=RegistrationMode.NONE,
        semantic_evidence=PrototypeSet(
            semantic_summary,
            semantic_signature,
            negatives=semantic_negatives,
        ),
        appearance_evidence=PrototypeSet(
            prototypes,
            appearance_signature,
            masses,
            appearance_negatives,
        ),
        selection_mode=SelectionMode.TOP_K,
        field_signature=semantic_signature,
    )


def compile_registered_2d_query(
    responsibilities: torch.Tensor,
    positive_prompt: torch.Tensor,
    negative_prompt: torch.Tensor | None,
    *,
    appearance_features: torch.Tensor,
    boundary_features: torch.Tensor,
    appearance_signature: FeatureSpaceSignature,
    boundary_signature: FeatureSpaceSignature,
    prototype_count: int = 4,
) -> QuerySpec:
    """Lift arbitrary points/scribbles/box/mask through raster responsibilities."""

    matrix = torch.as_tensor(responsibilities).float()
    positive = torch.as_tensor(positive_prompt).float().reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != positive.numel():
        raise ValueError("responsibilities must be [pixels,gaussians]")
    matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1e-8)
    positive_seeds = matrix.transpose(0, 1) @ positive
    negative_seeds = None
    if negative_prompt is not None:
        negative = torch.as_tensor(negative_prompt).float().reshape(-1)
        if negative.shape != positive.shape:
            raise ValueError("positive/negative prompt maps must align")
        negative_seeds = matrix.transpose(0, 1) @ negative

    return compile_registered_primitive_seeds(
        positive_seeds,
        negative_seeds,
        appearance_features=appearance_features,
        boundary_features=boundary_features,
        appearance_signature=appearance_signature,
        boundary_signature=boundary_signature,
        prototype_count=prototype_count,
        seed_source="raster_responsibility",
    )


def compile_registered_primitive_seeds(
    positive_seeds: torch.Tensor,
    negative_seeds: torch.Tensor | None,
    *,
    appearance_features: torch.Tensor,
    boundary_features: torch.Tensor,
    appearance_signature: FeatureSpaceSignature,
    boundary_signature: FeatureSpaceSignature,
    prototype_count: int = 4,
    seed_source: str = "raster_responsibility",
) -> QuerySpec:
    """Compile sparse raster-registered primitive seeds into shared evidence."""

    positive = torch.as_tensor(positive_seeds).float().reshape(-1)
    negative = (
        None
        if negative_seeds is None
        else torch.as_tensor(negative_seeds).float().reshape(-1)
    )
    appearance = torch.as_tensor(appearance_features).float()
    boundary = torch.as_tensor(boundary_features).float()
    if appearance.ndim != 2 or boundary.ndim != 2:
        raise ValueError("appearance/boundary features must be matrices")
    if appearance.shape[0] != positive.numel() or boundary.shape[0] != positive.numel():
        raise ValueError("registered seeds and capability rows must align")
    if negative is not None and negative.shape != positive.shape:
        raise ValueError("positive and negative primitive seeds must align")
    if not bool((positive > 0).any()):
        raise ValueError("registered query has no positive primitive support")

    app_proto, app_mass = _deterministic_prototypes(
        appearance, positive, prototype_count
    )
    bnd_proto, bnd_mass = _deterministic_prototypes(
        boundary, positive, prototype_count
    )
    app_neg = None
    bnd_neg = None
    if negative is not None and bool((negative > 0).any()):
        app_neg, _ = _deterministic_prototypes(
            appearance, negative, prototype_count
        )
        bnd_neg, _ = _deterministic_prototypes(
            boundary, negative, prototype_count
        )
    return QuerySpec(
        modality=QueryModality.REGISTERED_2D,
        intent=QueryIntent.REGION,
        registration=RegistrationMode.CAMERA,
        appearance_evidence=PrototypeSet(
            app_proto, appearance_signature, app_mass, app_neg
        ),
        boundary_evidence=PrototypeSet(
            bnd_proto, boundary_signature, bnd_mass, bnd_neg
        ),
        positive_seeds=SoftSeedSet(positive, seed_source),
        negative_seeds=(
            SoftSeedSet(negative, seed_source)
            if negative is not None
            else None
        ),
        selection_mode=SelectionMode.SEEDED_COMPONENT,
        field_signature=appearance_signature,
    )


def world_point_soft_seeds(
    gaussian_xyz: torch.Tensor,
    gaussian_covariance: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    xyz = torch.as_tensor(gaussian_xyz).float()
    covariance = torch.as_tensor(gaussian_covariance).float()
    queries = torch.as_tensor(points).float()
    if queries.ndim == 1:
        queries = queries[None]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or covariance.shape != (xyz.shape[0], 3, 3):
        raise ValueError("xyz/covariance must be [N,3] and [N,3,3]")
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("points must be [P,3]")
    identity = torch.eye(3, device=covariance.device, dtype=covariance.dtype)
    inverse = torch.linalg.pinv(covariance + 1e-6 * identity)
    delta = xyz[:, None, :] - queries[None, :, :]
    mahalanobis = torch.einsum("npi,nij,npj->np", delta, inverse, delta)
    return torch.exp(-0.5 * mahalanobis).amax(dim=1)


def compile_world_3d_query(
    gaussian_xyz: torch.Tensor,
    gaussian_covariance: torch.Tensor,
    positive_points: torch.Tensor,
    *,
    appearance_features: torch.Tensor,
    boundary_features: torch.Tensor,
    appearance_signature: FeatureSpaceSignature,
    boundary_signature: FeatureSpaceSignature,
    negative_points: torch.Tensor | None = None,
    prototype_count: int = 4,
    scene_mean_negative: bool = True,
) -> QuerySpec:
    positive = world_point_soft_seeds(gaussian_xyz, gaussian_covariance, positive_points)
    negative = (
        world_point_soft_seeds(gaussian_xyz, gaussian_covariance, negative_points)
        if negative_points is not None
        else None
    )
    app_proto, app_mass = _deterministic_prototypes(
        appearance_features, positive, prototype_count
    )
    bnd_proto, bnd_mass = _deterministic_prototypes(
        boundary_features, positive, prototype_count
    )
    app_neg = None
    bnd_neg = None
    if negative is not None:
        app_neg, _ = _deterministic_prototypes(appearance_features, negative, prototype_count)
        bnd_neg, _ = _deterministic_prototypes(boundary_features, negative, prototype_count)
    elif scene_mean_negative:
        # A one-click protocol supplies no explicit background click.  The
        # unlabeled scene mean is a fixed, query-independent negative and may
        # include the target itself; it therefore cannot leak instance GT.
        app_neg = F.normalize(
            torch.as_tensor(appearance_features).float().mean(dim=0), dim=0
        )[None]
        bnd_neg = F.normalize(
            torch.as_tensor(boundary_features).float().mean(dim=0), dim=0
        )[None]
    return QuerySpec(
        modality=QueryModality.WORLD_3D,
        intent=QueryIntent.INSTANCE,
        registration=RegistrationMode.WORLD,
        appearance_evidence=PrototypeSet(
            app_proto, appearance_signature, app_mass, app_neg
        ),
        boundary_evidence=PrototypeSet(
            bnd_proto, boundary_signature, bnd_mass, bnd_neg
        ),
        positive_seeds=SoftSeedSet(positive, "gaussian_mahalanobis"),
        negative_seeds=(SoftSeedSet(negative, "gaussian_mahalanobis") if negative is not None else None),
        selection_mode=SelectionMode.SEEDED_COMPONENT,
        field_signature=appearance_signature,
        metadata={
            "negative_evidence": (
                "explicit_world_points"
                if negative is not None
                else "unlabeled_scene_mean"
                if scene_mean_negative
                else "none"
            )
        },
    )
