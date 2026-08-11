"""Compile text, image, registered-2D, and world-3D inputs into 3-D evidence."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from radio_gs.field.field_signature import FeatureSpaceSignature
from .evidence_scorer import registered_seed_observation
from .query_spec import (
    PrimitiveUnaryEvidence,
    PrototypeSet,
    QueryIntent,
    QueryModality,
    QuerySpec,
    RegistrationMode,
    SelectionMode,
    SoftSeedGroups,
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
        # A pose-free exemplar denotes one physical instance.  Retaining a
        # fixed number of high-scoring components conflates instance recovery
        # with multi-instance category retrieval and permits avoidable
        # cross-instance leakage.  Ranking remains an unfiltered unary output;
        # only the full-mask readout uses the strongest connected support.
        selection_mode=SelectionMode.TOP_COMPONENT,
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
    selection_mode: SelectionMode | str = SelectionMode.SEEDED_COMPONENT,
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
        selection_mode=selection_mode,
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
    selection_mode: SelectionMode | str = SelectionMode.SEEDED_COMPONENT,
    positive_prompt_mass: torch.Tensor | None = None,
    negative_prompt_mass: torch.Tensor | None = None,
    primitive_unary_evidence: PrimitiveUnaryEvidence | None = None,
    seed_normalization: str = "independent_max",
    prototype_positive_seeds: torch.Tensor | None = None,
    prototype_negative_seeds: torch.Tensor | None = None,
    prototype_seed_provenance: str | None = None,
    solver_seed_provenance: str | None = None,
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
    prototype_positive = (
        positive
        if prototype_positive_seeds is None
        else torch.as_tensor(prototype_positive_seeds).float().reshape(-1)
    )
    prototype_negative = (
        negative
        if prototype_negative_seeds is None
        else torch.as_tensor(prototype_negative_seeds).float().reshape(-1)
    )
    prototype_decoupled = (
        prototype_positive_seeds is not None
        or prototype_negative_seeds is not None
    )
    if (
        prototype_positive.shape != positive.shape
        or not bool(torch.isfinite(prototype_positive).all())
        or bool((prototype_positive < 0).any())
        or not bool((prototype_positive > 0).any())
    ):
        raise ValueError(
            "prototype positive seeds must be finite, non-negative, aligned, "
            "and non-empty"
        )
    if prototype_negative is not None and (
        prototype_negative.shape != positive.shape
        or not bool(torch.isfinite(prototype_negative).all())
        or bool((prototype_negative < 0).any())
        or (
            prototype_negative_seeds is not None
            and not bool((prototype_negative > 0).any())
        )
    ):
        raise ValueError(
            "prototype negative seeds must be finite, non-negative, aligned, "
            "and explicit overrides must be non-empty"
        )
    if not prototype_decoupled and (
        prototype_seed_provenance is not None
        or solver_seed_provenance is not None
    ):
        raise ValueError(
            "prototype/solver seed provenance requires decoupled prototype seeds"
        )
    if prototype_decoupled and (
        not str(prototype_seed_provenance or "").strip()
        or not str(solver_seed_provenance or "").strip()
    ):
        # Generic callers remain supported, but their provenance is explicit
        # and cannot accidentally satisfy the stricter dual-registration
        # engine contract.
        prototype_seed_provenance = (
            str(prototype_seed_provenance or "explicit_override_unspecified")
        )
        solver_seed_provenance = str(
            solver_seed_provenance or "registered_solver_seed_unspecified"
        )
    if primitive_unary_evidence is not None and (
        positive_prompt_mass is not None or negative_prompt_mass is not None
    ):
        raise ValueError(
            "explicit primitive unary evidence cannot be combined with prompt masses"
        )
    if primitive_unary_evidence is None:
        direct_positive = (
            positive
            if positive_prompt_mass is None
            else torch.as_tensor(positive_prompt_mass).float().reshape(-1)
        )
        direct_negative = (
            negative
            if negative_prompt_mass is None
            else torch.as_tensor(negative_prompt_mass).float().reshape(-1)
        )
        direct_unary, direct_confidence = registered_seed_observation(
            direct_positive, direct_negative
        )
        observation = PrimitiveUnaryEvidence(
            direct_unary,
            f"{seed_source}_joint_mass",
            direct_confidence,
        )
    else:
        observation = primitive_unary_evidence
    if observation.values.shape != positive.shape:
        raise ValueError("registered prompt evidence and primitive seeds must align")

    app_proto, app_mass = _deterministic_prototypes(
        appearance,
        prototype_positive,
        prototype_count,
        strategy=prototype_strategy,
    )
    bnd_proto, bnd_mass = _deterministic_prototypes(
        boundary,
        prototype_positive,
        prototype_count,
        strategy=prototype_strategy,
    )
    app_neg = None
    bnd_neg = None
    if prototype_negative is not None and bool((prototype_negative > 0).any()):
        app_neg, _ = _deterministic_prototypes(
            appearance,
            prototype_negative,
            prototype_count,
            strategy=prototype_strategy,
        )
        bnd_neg, _ = _deterministic_prototypes(
            boundary,
            prototype_negative,
            prototype_count,
            strategy=prototype_strategy,
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
        positive_seeds=SoftSeedSet(
            positive,
            seed_source,
            normalization=seed_normalization,
        ),
        negative_seeds=(
            SoftSeedSet(
                negative,
                seed_source,
                normalization=seed_normalization,
            )
            if negative is not None
            else None
        ),
        positive_seed_groups=SoftSeedGroups(
            positive[:, None],
            seed_source,
            normalization=seed_normalization,
        ),
        negative_seed_groups=(
            SoftSeedGroups(
                negative[:, None],
                seed_source,
                normalization=seed_normalization,
            )
            if negative is not None and bool((negative > 0).any())
            else None
        ),
        primitive_unary_evidence=observation,
        selection_mode=SelectionMode(selection_mode),
        field_signature=appearance_signature,
        metadata={
            "prototype_strategy": str(prototype_strategy),
            "registered_unary_normalization": "joint_prompt_mass",
            "registered_seed_normalization": str(seed_normalization),
            **(
                {
                    "prototype_seed_decoupled": True,
                    "prototype_seed_role": "prototype_construction_only",
                    "solver_seed_role": "soft_seed_and_registered_unary",
                    "prototype_seed_provenance": str(
                        prototype_seed_provenance
                    ),
                    "solver_seed_provenance": str(solver_seed_provenance),
                }
                if prototype_decoupled
                else {}
            ),
        },
    )


def world_point_soft_seed_matrix(
    gaussian_xyz: torch.Tensor,
    gaussian_covariance: torch.Tensor,
    points: torch.Tensor,
    *,
    gaussian_precision: torch.Tensor | None = None,
    euclidean_candidate_k: int = 0,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return covariance-aware relative seeds separately for every world point.

    Columns correspond to the supplied world points.  They are normalized in
    log-kernel space for numerical stability, but candidate membership remains
    exactly the query-independent Gaussian/candidate contract.
    """
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
    candidate_k = int(euclidean_candidate_k)
    if candidate_k < 0:
        raise ValueError("euclidean_candidate_k cannot be negative")
    if 0 < candidate_k < xyz.shape[0]:
        nearest = delta.square().sum(dim=-1).topk(
            candidate_k, dim=0, largest=False
        ).indices
        euclidean_mask = torch.zeros_like(mahalanobis, dtype=torch.bool)
        euclidean_mask.scatter_(0, nearest, True)
        allowed_by_distance = euclidean_mask
    else:
        allowed_by_distance = torch.ones_like(mahalanobis, dtype=torch.bool)
    if candidate_mask is not None:
        allowed = torch.as_tensor(
            candidate_mask, device=mahalanobis.device
        ).bool().reshape(-1)
        if allowed.shape != (xyz.shape[0],) or not bool(allowed.any()):
            raise ValueError("candidate_mask must be a non-empty [N] mask")
        allowed_by_distance &= allowed[:, None]
    # A click is a hard interaction constraint.  Only relative Gaussian
    # responsibility is meaningful for its seed weights: the absolute 3-D
    # kernel density is audited separately by continuous field support.
    # Subtracting the best permitted log-density prevents tiny, valid
    # covariances from underflowing to an all-zero seed at an official 5 cm
    # point.  This is not a radius lift: candidate membership is still fixed
    # by the covariance-aware standard compiler (and optional Euclidean top-K).
    masked_mahalanobis = mahalanobis.masked_fill(~allowed_by_distance, float("inf"))
    best = masked_mahalanobis.amin(dim=0, keepdim=True)
    if not bool(torch.isfinite(best).all()):
        raise ValueError("world point query has no permitted Gaussian candidate")
    weights = torch.exp(-0.5 * (masked_mahalanobis - best))
    weights = weights.masked_fill(~allowed_by_distance, 0.0)
    return weights


def world_point_soft_seeds(
    gaussian_xyz: torch.Tensor,
    gaussian_covariance: torch.Tensor,
    points: torch.Tensor,
    *,
    gaussian_precision: torch.Tensor | None = None,
    euclidean_candidate_k: int = 0,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate covariance-aware soft seeds over one or more world points."""

    return world_point_soft_seed_matrix(
        gaussian_xyz,
        gaussian_covariance,
        points,
        gaussian_precision=gaussian_precision,
        euclidean_candidate_k=euclidean_candidate_k,
        candidate_mask=candidate_mask,
    ).amax(dim=1)


def _world_point_local_columns(
    point_count: int,
    maximum_count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    maximum_count = int(maximum_count)
    if maximum_count < 0:
        raise ValueError("world_point_max_prototypes cannot be negative")
    if 0 < maximum_count < int(point_count):
        # Preserve temporal coverage without using output or benchmark state.
        return torch.linspace(
            0, int(point_count) - 1, maximum_count, device=device
        ).round().long().unique(sorted=True)
    return torch.arange(int(point_count), device=device)


def _world_point_local_prototypes(
    features: torch.Tensor,
    seed_matrix: torch.Tensor,
    *,
    maximum_count: int = 0,
    prototype_weighting: str = "support_mass",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool one feature prototype per registered world point.

    A point prompt is an interaction event, not a generic region.  Keeping a
    local descriptor for each click retains different object facets and
    accumulated corrective evidence without adding any scene-trained head.
    """

    bank = torch.as_tensor(features)
    weights = torch.as_tensor(seed_matrix, device=bank.device).float()
    if bank.ndim != 2 or weights.ndim != 2 or weights.shape[0] != bank.shape[0]:
        raise ValueError("world-point local prototypes require [N,D] features and [N,P] seeds")
    if not bool((weights > 0).any(dim=0).all()):
        raise ValueError("every world point needs non-empty Gaussian support")
    point_count = int(weights.shape[1])
    prototype_weighting = str(prototype_weighting)
    if prototype_weighting not in {"support_mass", "equal_click"}:
        raise ValueError(
            "world-point prototype weighting must be support_mass or equal_click"
        )
    keep = _world_point_local_columns(
        point_count,
        maximum_count,
        device=weights.device,
    )
    prototypes: list[torch.Tensor] = []
    masses: list[torch.Tensor] = []
    for column in keep.tolist():
        local_weights = weights[:, column]
        rows = torch.where(local_weights > 0)[0]
        local_features = F.normalize(
            bank.index_select(0, rows).float(), dim=-1, eps=1e-8
        )
        mass = local_weights.index_select(0, rows)
        prototype = F.normalize(
            (local_features * mass[:, None]).sum(dim=0), dim=0, eps=1e-8
        )
        prototypes.append(prototype)
        # A registered world click is an interaction constraint, not a vote
        # proportional to the size of the Gaussian kernel that happens to
        # support it.  Equal click mass prevents a broad early Gaussian from
        # silencing a later corrective click; support-mass remains available
        # as the historical, reproducible ablation.
        masses.append(
            mass.sum()
            if prototype_weighting == "support_mass"
            else mass.new_tensor(1.0)
        )
    prototype_masses = torch.stack(masses).float()
    prototype_masses /= prototype_masses.sum().clamp_min(1e-8)
    return torch.stack(prototypes), prototype_masses


def continuous_gaussian_readout(
    gaussian_xyz: torch.Tensor,
    gaussian_covariance: torch.Tensor,
    primitive_values: torch.Tensor,
    points: torch.Tensor,
    *,
    gaussian_precision: torch.Tensor | None = None,
    opacity: torch.Tensor | None = None,
    candidate_k: int = 64,
    candidate_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read scalar primitive values at world points through fixed Gaussians.

    The result is a normalized opacity-weighted Gaussian-kernel average and
    its unnormalized support mass.  Callers may supply query-independent
    nearest-candidate indices to avoid materializing an all-points by
    all-primitives distance matrix for dense evaluation domains.
    """

    xyz = torch.as_tensor(gaussian_xyz).float()
    covariance = torch.as_tensor(gaussian_covariance, device=xyz.device).float()
    values = torch.as_tensor(primitive_values, device=xyz.device).float().reshape(-1)
    queries = torch.as_tensor(points, device=xyz.device).float()
    if queries.ndim == 1:
        queries = queries[None]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or covariance.shape != (xyz.shape[0], 3, 3):
        raise ValueError("xyz/covariance must be [N,3] and [N,3,3]")
    if values.shape != (xyz.shape[0],):
        raise ValueError("primitive_values must align with gaussian_xyz")
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("points must be [P,3]")
    if gaussian_precision is None:
        identity = torch.eye(3, device=xyz.device, dtype=covariance.dtype)
        precision = torch.linalg.pinv(covariance + 1e-6 * identity)
    else:
        precision = torch.as_tensor(gaussian_precision, device=xyz.device).float()
        if precision.shape != covariance.shape:
            raise ValueError("gaussian_precision must align with covariance [N,3,3]")
    if opacity is None:
        density = torch.ones(xyz.shape[0], device=xyz.device, dtype=torch.float32)
    else:
        density = torch.as_tensor(opacity, device=xyz.device).float().reshape(-1)
        if density.shape != values.shape or bool((density < 0).any()):
            raise ValueError("opacity must be non-negative and align with primitive_values")

    candidate_k = int(candidate_k)
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if candidate_indices is None:
        count = min(candidate_k, int(xyz.shape[0]))
        indices = torch.cdist(queries, xyz).topk(count, dim=1, largest=False).indices
    else:
        indices = torch.as_tensor(candidate_indices, device=xyz.device).long()
        if indices.ndim != 2 or indices.shape[0] != queries.shape[0] or indices.shape[1] == 0:
            raise ValueError("candidate_indices must be a non-empty [P,K] matrix")
        if bool((indices < 0).any()) or bool((indices >= xyz.shape[0]).any()):
            raise IndexError("candidate_indices contains an invalid Gaussian row")

    centers = xyz[indices]
    selected_precision = precision[indices]
    delta = centers - queries[:, None, :]
    mahalanobis = torch.einsum("pki,pkij,pkj->pk", delta, selected_precision, delta)
    weights = torch.exp(-0.5 * mahalanobis).clamp_min(0.0) * density[indices]
    support = weights.sum(dim=1)
    readout = (weights * values[indices]).sum(dim=1) / support.clamp_min(1e-12)
    return readout, support


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
    world_point_prototype_mode: str = "aggregate_fps",
    world_point_max_prototypes: int = 0,
    world_point_prototype_weighting: str = "support_mass",
    selection_mode: SelectionMode | str = SelectionMode.SEEDED_COMPONENT,
    primitive_unary_evidence: PrimitiveUnaryEvidence | None = None,
) -> QuerySpec:
    positive_matrix = world_point_soft_seed_matrix(
        gaussian_xyz,
        gaussian_covariance,
        positive_points,
        gaussian_precision=gaussian_precision,
        euclidean_candidate_k=euclidean_candidate_k,
        candidate_mask=seed_candidate_mask,
    )
    negative_matrix = (
        world_point_soft_seed_matrix(
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

    def calibrate_seed_groups(values: torch.Tensor) -> torch.Tensor:
        relative = values / values.amax(dim=0, keepdim=True).clamp_min(1e-30)
        return relative.pow(1.0 / seed_temperature)

    positive_groups = calibrate_seed_groups(positive_matrix)
    negative_groups = (
        calibrate_seed_groups(negative_matrix)
        if negative_matrix is not None
        else None
    )
    positive = positive_groups.amax(dim=1)
    negative = negative_groups.amax(dim=1) if negative_groups is not None else None
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

    world_point_prototype_mode = str(world_point_prototype_mode)
    if world_point_prototype_mode not in {"aggregate_fps", "per_click_local"}:
        raise ValueError(
            "world_point_prototype_mode must be aggregate_fps or per_click_local"
        )
    world_point_prototype_weighting = str(world_point_prototype_weighting)
    if world_point_prototype_weighting not in {"support_mass", "equal_click"}:
        raise ValueError(
            "world_point_prototype_weighting must be support_mass or equal_click"
        )
    if world_point_prototype_mode == "per_click_local":
        positive_columns = _world_point_local_columns(
            positive_groups.shape[1],
            int(world_point_max_prototypes),
            device=positive_groups.device,
        )
        positive_spatial_groups = positive_groups.index_select(1, positive_columns)
        app_proto, app_mass = _world_point_local_prototypes(
            appearance_features,
            positive_matrix,
            maximum_count=int(world_point_max_prototypes),
            prototype_weighting=world_point_prototype_weighting,
        )
        bnd_proto, bnd_mass = _world_point_local_prototypes(
            boundary_features,
            positive_matrix,
            maximum_count=int(world_point_max_prototypes),
            prototype_weighting=world_point_prototype_weighting,
        )
    else:
        positive_spatial_groups = positive[:, None]
        app_proto, app_mass = _deterministic_prototypes(
            appearance_features, positive, prototype_count, strategy=prototype_strategy
        )
        bnd_proto, bnd_mass = _deterministic_prototypes(
            boundary_features, positive, prototype_count, strategy=prototype_strategy
        )
    app_neg = None
    bnd_neg = None
    negative_spatial_groups = None
    if negative is not None:
        if world_point_prototype_mode == "per_click_local":
            assert negative_matrix is not None
            assert negative_groups is not None
            negative_columns = _world_point_local_columns(
                negative_groups.shape[1],
                int(world_point_max_prototypes),
                device=negative_groups.device,
            )
            negative_spatial_groups = negative_groups.index_select(
                1, negative_columns
            )
            app_neg, _ = _world_point_local_prototypes(
                appearance_features,
                negative_matrix,
                maximum_count=int(world_point_max_prototypes),
                prototype_weighting=world_point_prototype_weighting,
            )
            bnd_neg, _ = _world_point_local_prototypes(
                boundary_features,
                negative_matrix,
                maximum_count=int(world_point_max_prototypes),
                prototype_weighting=world_point_prototype_weighting,
            )
        else:
            negative_spatial_groups = negative[:, None]
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
        positive_seed_groups=SoftSeedGroups(
            positive_spatial_groups,
            "gaussian_mahalanobis_per_click",
        ),
        negative_seed_groups=(
            SoftSeedGroups(
                negative_spatial_groups,
                "gaussian_mahalanobis_per_click",
            )
            if negative_spatial_groups is not None
            else None
        ),
        primitive_unary_evidence=primitive_unary_evidence,
        selection_mode=SelectionMode(selection_mode),
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
            "world_point_prototype_mode": world_point_prototype_mode,
            "world_point_max_prototypes": int(world_point_max_prototypes),
            "world_point_prototype_weighting": world_point_prototype_weighting,
            "negative_evidence": (
                "explicit_world_points"
                if negative is not None
                else "unlabeled_scene_mean"
                if scene_mean_negative
                else "none"
            )
        },
    )
