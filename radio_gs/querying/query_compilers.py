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
    *,
    chunk_size: int = 8192,
    strategy: str = "weighted_fps",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic weighted prototypes; no learned or benchmark state.

    ``spherical_mean_fps`` anchors the set with the weighted spherical mean
    before retaining diverse FPS representatives.  This makes the dominant
    prototype an aggregate observation instead of a potentially noisy single
    primitive while preserving multi-modal prompt support.
    """

    # Promote only one supported chunk at a time.  A full-reference negative
    # prompt can cover most of a million-row scene, so even selecting all
    # supported fp16 rows before conversion can require many GiB in fp32.
    # This is the same weighted farthest-point algorithm as the dense form:
    # only the normalized rows and assignments are evaluated in chunks.
    raw_values = torch.as_tensor(features)
    raw_masses = torch.as_tensor(weights).reshape(-1)
    if raw_values.ndim != 2 or raw_masses.shape != (raw_values.shape[0],):
        raise ValueError("prototype features and weights must align as [N,D] and [N]")
    if int(chunk_size) <= 0:
        raise ValueError("prototype chunk_size must be positive")
    strategy = str(strategy)
    if strategy not in {"weighted_fps", "spherical_mean_fps"}:
        raise ValueError("prototype strategy must be weighted_fps or spherical_mean_fps")
    valid = raw_masses > 0
    if not bool(valid.any()):
        raise ValueError("cannot build prototypes from empty support")
    active_rows = torch.where(valid)[0]
    device = raw_values.device
    masses = raw_masses[valid].float().to(device)
    count = min(max(1, int(count)), active_rows.numel())

    def normalized_active_chunk(start: int, stop: int) -> torch.Tensor:
        rows = active_rows[start:stop].to(device)
        return F.normalize(
            raw_values.index_select(0, rows).float(), dim=-1, eps=1e-8
        )

    def normalized_active_row(local_index: int) -> torch.Tensor:
        global_index = active_rows[int(local_index)].reshape(1).to(device)
        return F.normalize(
            raw_values.index_select(0, global_index).float()[0], dim=0, eps=1e-8
        )

    if strategy == "spherical_mean_fps":
        mean = torch.zeros(raw_values.shape[1], device=device, dtype=torch.float32)
        for start in range(0, active_rows.numel(), int(chunk_size)):
            stop = min(start + int(chunk_size), active_rows.numel())
            mean.add_((normalized_active_chunk(start, stop) * masses[start:stop, None]).sum(dim=0))
        if float(mean.norm()) <= 1e-8:
            first = normalized_active_row(int(masses.argmax()))
        else:
            first = F.normalize(mean, dim=0, eps=1e-8)
        selected: list[int] = []
        prototypes = [first]
    else:
        selected = [int(masses.argmax())]
        prototypes = [normalized_active_row(selected[0])]
    nearest_distance = torch.empty(active_rows.numel(), device=device)
    for start in range(0, active_rows.numel(), int(chunk_size)):
        stop = min(start + int(chunk_size), active_rows.numel())
        values = normalized_active_chunk(start, stop)
        nearest_distance[start:stop] = 1.0 - values @ prototypes[0]
    for _ in range(1, count):
        score = nearest_distance * masses
        index = int(score.argmax())
        selected.append(index)
        prototype = normalized_active_row(index)
        prototypes.append(prototype)
        for start in range(0, active_rows.numel(), int(chunk_size)):
            stop = min(start + int(chunk_size), active_rows.numel())
            values = normalized_active_chunk(start, stop)
            nearest_distance[start:stop] = torch.minimum(
                nearest_distance[start:stop], 1.0 - values @ prototype
            )
    prototype_matrix = torch.stack(prototypes)
    prototype_masses = torch.zeros(count, device=device)
    for start in range(0, active_rows.numel(), int(chunk_size)):
        stop = min(start + int(chunk_size), active_rows.numel())
        values = normalized_active_chunk(start, stop)
        assignment = (values @ prototype_matrix.T).argmax(dim=1)
        prototype_masses.scatter_add_(0, assignment, masses[start:stop])
    prototype_masses /= prototype_masses.sum().clamp_min(1e-8)
    return prototype_matrix, prototype_masses


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
    prototype_strategy: str = "spherical_mean_fps",
) -> QuerySpec:
    tokens = torch.as_tensor(appearance_tokens).float()
    if tokens.ndim != 2:
        raise ValueError("appearance_tokens must be [P,D]")
    weights = (
        torch.ones(tokens.shape[0], dtype=torch.float32, device=tokens.device)
        if foreground_weights is None
        else torch.as_tensor(foreground_weights).float().to(tokens.device)
    )
    prototypes, masses = _deterministic_prototypes(
        tokens, weights, prototype_count, strategy=prototype_strategy
    )
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
    prototype_strategy: str = "spherical_mean_fps",
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
        prototype_strategy=prototype_strategy,
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
    prototype_strategy: str = "spherical_mean_fps",
    seed_source: str = "raster_responsibility",
) -> QuerySpec:
    """Compile sparse raster-registered primitive seeds into shared evidence."""

    positive = torch.as_tensor(positive_seeds).float().reshape(-1)
    negative = (
        None
        if negative_seeds is None
        else torch.as_tensor(negative_seeds).float().reshape(-1)
    )
    # Keep the full capability banks in their compact storage dtype.  The
    # prototype builder promotes only non-zero seed rows after indexing.
    appearance = torch.as_tensor(appearance_features)
    boundary = torch.as_tensor(boundary_features)
    if appearance.ndim != 2 or boundary.ndim != 2:
        raise ValueError("appearance/boundary features must be matrices")
    if appearance.shape[0] != positive.numel() or boundary.shape[0] != positive.numel():
        raise ValueError("registered seeds and capability rows must align")
    if negative is not None and negative.shape != positive.shape:
        raise ValueError("positive and negative primitive seeds must align")
    if not bool((positive > 0).any()):
        raise ValueError("registered query has no positive primitive support")

    app_proto, app_mass = _deterministic_prototypes(
        appearance, positive, prototype_count, strategy=prototype_strategy
    )
    bnd_proto, bnd_mass = _deterministic_prototypes(
        boundary, positive, prototype_count, strategy=prototype_strategy
    )
    app_neg = None
    bnd_neg = None
    if negative is not None and bool((negative > 0).any()):
        app_neg, _ = _deterministic_prototypes(
            appearance, negative, prototype_count, strategy=prototype_strategy
        )
        bnd_neg, _ = _deterministic_prototypes(
            boundary, negative, prototype_count, strategy=prototype_strategy
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
        metadata={"prototype_strategy": str(prototype_strategy)},
    )


def world_point_soft_seeds(
    gaussian_xyz: torch.Tensor,
    gaussian_covariance: torch.Tensor,
    points: torch.Tensor,
    *,
    gaussian_precision: torch.Tensor | None = None,
    euclidean_candidate_k: int = 0,
    candidate_mask: torch.Tensor | None = None,
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
    if gaussian_precision is None:
        identity = torch.eye(3, device=covariance.device, dtype=covariance.dtype)
        inverse = torch.linalg.pinv(covariance + 1e-6 * identity)
    else:
        inverse = torch.as_tensor(gaussian_precision).to(covariance)
        if inverse.shape != covariance.shape:
            raise ValueError("gaussian_precision must align with covariance [N,3,3]")
    delta = xyz[:, None, :] - queries[None, :, :]
    mahalanobis = torch.einsum("npi,nij,npj->np", delta, inverse, delta)
    weights = torch.exp(-0.5 * mahalanobis)
    candidate_k = int(euclidean_candidate_k)
    if candidate_k < 0:
        raise ValueError("euclidean_candidate_k cannot be negative")
    if 0 < candidate_k < xyz.shape[0]:
        nearest = delta.square().sum(dim=-1).topk(
            candidate_k, dim=0, largest=False
        ).indices
        euclidean_mask = torch.zeros_like(weights, dtype=torch.bool)
        euclidean_mask.scatter_(0, nearest, True)
        weights = weights.masked_fill(~euclidean_mask, 0.0)
    if candidate_mask is not None:
        allowed = torch.as_tensor(candidate_mask, device=weights.device).bool().reshape(-1)
        if allowed.shape != (xyz.shape[0],) or not bool(allowed.any()):
            raise ValueError("candidate_mask must be a non-empty [N] mask")
        weights = weights.masked_fill(~allowed[:, None], 0.0)
    return weights.amax(dim=1)


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
    prototype_strategy: str = "weighted_fps",
    scene_mean_negative: bool = True,
    gaussian_precision: torch.Tensor | None = None,
    euclidean_candidate_k: int = 64,
    seed_topk: int = 0,
    seed_temperature: float = 1.0,
    seed_candidate_mask: torch.Tensor | None = None,
) -> QuerySpec:
    positive = world_point_soft_seeds(
        gaussian_xyz,
        gaussian_covariance,
        positive_points,
        gaussian_precision=gaussian_precision,
        euclidean_candidate_k=euclidean_candidate_k,
        candidate_mask=seed_candidate_mask,
    )
    negative = (
        world_point_soft_seeds(
            gaussian_xyz,
            gaussian_covariance,
            negative_points,
            gaussian_precision=gaussian_precision,
            euclidean_candidate_k=euclidean_candidate_k,
            candidate_mask=seed_candidate_mask,
        )
        if negative_points is not None
        else None
    )
    seed_temperature = float(seed_temperature)
    if seed_temperature <= 0:
        raise ValueError("seed_temperature must be positive")

    def calibrate_seed_weights(values: torch.Tensor) -> torch.Tensor:
        relative = values / values.max().clamp_min(1e-30)
        return relative.pow(1.0 / seed_temperature)

    positive = calibrate_seed_weights(positive)
    if negative is not None:
        negative = calibrate_seed_weights(negative)
    seed_topk = int(seed_topk)
    if seed_topk < 0:
        raise ValueError("seed_topk cannot be negative")
    if 0 < seed_topk < positive.numel():
        keep = positive.topk(seed_topk).indices
        sparse_positive = torch.zeros_like(positive)
        sparse_positive[keep] = positive[keep]
        positive = sparse_positive
    if negative is not None and 0 < seed_topk < negative.numel():
        keep = negative.topk(seed_topk).indices
        sparse_negative = torch.zeros_like(negative)
        sparse_negative[keep] = negative[keep]
        negative = sparse_negative

    app_proto, app_mass = _deterministic_prototypes(
        appearance_features, positive, prototype_count, strategy=prototype_strategy
    )
    bnd_proto, bnd_mass = _deterministic_prototypes(
        boundary_features, positive, prototype_count, strategy=prototype_strategy
    )
    app_neg = None
    bnd_neg = None
    if negative is not None:
        app_neg, _ = _deterministic_prototypes(
            appearance_features, negative, prototype_count, strategy=prototype_strategy
        )
        bnd_neg, _ = _deterministic_prototypes(
            boundary_features, negative, prototype_count, strategy=prototype_strategy
        )
    elif scene_mean_negative:
        # A one-click protocol supplies no explicit background click.  The
        # unlabeled scene mean is a fixed, query-independent negative and may
        # include the target itself; it therefore cannot leak instance GT.
        app_mean = torch.as_tensor(appearance_features).float().mean(dim=0)
        bnd_mean = torch.as_tensor(boundary_features).float().mean(dim=0)
        app_neg = F.normalize(app_mean, dim=0)[None]
        bnd_neg = F.normalize(bnd_mean, dim=0)[None]
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
            "seed_topk": seed_topk,
            "euclidean_candidate_k": int(euclidean_candidate_k),
            "seed_temperature": seed_temperature,
            "seed_candidate_mask_count": (
                int(torch.as_tensor(seed_candidate_mask).bool().sum())
                if seed_candidate_mask is not None
                else 0
            ),
            "prototype_strategy": str(prototype_strategy),
            "negative_evidence": (
                "explicit_world_points"
                if negative is not None
                else "unlabeled_scene_mean"
                if scene_mean_negative
                else "none"
            )
        },
    )
