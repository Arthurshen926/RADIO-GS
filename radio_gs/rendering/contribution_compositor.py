"""Auditable contribution-space alternatives to ordinary alpha compositing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


MARGINAL_RESPONSIBILITY_CONTRACT = (
    "exact_front_to_back_weight_times_normalized_pixel_marginal_v1"
)
EXACT_CENTER_UNCERTAINTY_CONTRACT = (
    "exact_front_to_back_adjoint_center_plus_marginal_visibility_purity_v1"
)


@dataclass(frozen=True)
class MarginalResponsibilityStatistics:
    """Query-free uncertainty induced by the exact 3DGS compositor.

    ``responsibility`` is the normalized marginal contribution of one hit to
    its pixel. ``target_weight`` is the exact contribution multiplied by that
    responsibility, so a teacher observation is assigned strongly only when
    the primitive both contributes and explains a large fraction of the
    pixel. ``pixel_collision_purity`` is the collision probability
    :math:`sum_i r_i^2`; it is one for an unambiguous pixel and decreases as
    contribution mass is split across primitives.
    """

    responsibility: torch.Tensor
    target_weight: torch.Tensor
    pixel_mass: torch.Tensor
    pixel_collision_purity: torch.Tensor


def marginal_responsibility_statistics(
    pixel_ids: torch.Tensor,
    base_weights: torch.Tensor,
    *,
    num_pixels: int,
    eps: float = 1e-12,
) -> MarginalResponsibilityStatistics:
    """Convert exact front-to-back contributions into target responsibilities.

    This transformation has no scene-specific threshold or learned parameter.
    It preserves continuous visibility while reducing the influence of pixels
    whose feature observation is inherently shared by multiple primitives.
    """

    pids = torch.as_tensor(pixel_ids).long().reshape(-1)
    weights = torch.as_tensor(base_weights).float().reshape(-1)
    if pids.shape != weights.shape:
        raise ValueError("pixel_ids and base_weights must have matching shapes")
    if int(num_pixels) <= 0:
        raise ValueError("num_pixels must be positive")
    if pids.numel() and (
        int(pids.min()) < 0 or int(pids.max()) >= int(num_pixels)
    ):
        raise ValueError("pixel id outside declared image")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("base_weights must be finite and non-negative")
    if float(eps) <= 0:
        raise ValueError("eps must be positive")

    pixel_mass = torch.zeros(
        int(num_pixels), dtype=torch.float32, device=weights.device
    )
    pixel_mass.index_add_(0, pids, weights)
    responsibility = weights / pixel_mass[pids].clamp_min(float(eps))
    target_weight = weights * responsibility
    pixel_collision_purity = torch.zeros_like(pixel_mass)
    pixel_collision_purity.index_add_(0, pids, responsibility.square())
    return MarginalResponsibilityStatistics(
        responsibility=responsibility,
        target_weight=target_weight,
        pixel_mass=pixel_mass,
        pixel_collision_purity=pixel_collision_purity,
    )


def primitive_visibility_purity(
    gaussian_ids: torch.Tensor,
    base_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    num_gaussians: int,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate responsibility-weighted visibility and purity per primitive."""

    gids = torch.as_tensor(gaussian_ids).long().reshape(-1)
    visible = torch.as_tensor(base_weights).float().reshape(-1)
    pure = torch.as_tensor(target_weights).float().reshape(-1)
    if gids.shape != visible.shape or gids.shape != pure.shape:
        raise ValueError("gaussian ids and visibility weights must align")
    if int(num_gaussians) <= 0 or (
        gids.numel()
        and (int(gids.min()) < 0 or int(gids.max()) >= int(num_gaussians))
    ):
        raise ValueError("gaussian id outside declared primitive domain")
    if (
        not bool(torch.isfinite(visible).all())
        or not bool(torch.isfinite(pure).all())
        or bool((visible < 0).any())
        or bool((pure < 0).any())
        or bool((pure > visible + float(eps)).any())
    ):
        raise ValueError("primitive visibility weights are invalid")
    visible_mass = torch.zeros(
        int(num_gaussians), dtype=torch.float32, device=visible.device
    )
    pure_mass = torch.zeros_like(visible_mass)
    visible_mass.index_add_(0, gids, visible)
    pure_mass.index_add_(0, gids, pure)
    purity = torch.where(
        visible_mass > 0,
        pure_mass / visible_mass.clamp_min(float(eps)),
        torch.zeros_like(visible_mass),
    ).clamp(0.0, 1.0)
    return visible_mass, pure_mass, purity


def front_to_back_weights(
    pixel_ids: torch.Tensor,
    alphas: torch.Tensor,
    *,
    num_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stable pixel grouping, contribution weights, and accumulated alpha.

    Input order within each pixel is assumed to be front-to-back, as returned by
    ``gsplat.rasterize_to_indices_in_range``.  A stable pixel sort preserves it.
    The exclusive transmittance is evaluated in float64 to avoid cross-segment
    cancellation when many pixel groups are concatenated.
    """

    pids = torch.as_tensor(pixel_ids).long().reshape(-1)
    alpha = torch.as_tensor(alphas).float().reshape(-1)
    if pids.shape != alpha.shape:
        raise ValueError("pixel_ids and alphas must have matching shapes")
    if num_pixels <= 0 or bool((pids < 0).any()) or bool((pids >= num_pixels).any()):
        raise ValueError("pixel id outside declared image")
    if bool((alpha < 0).any()) or bool((alpha > 1).any()):
        raise ValueError("alphas must be in [0,1]")
    if pids.numel() == 0:
        empty = torch.empty(0, device=pids.device)
        return pids, pids, empty, torch.zeros(num_pixels, device=pids.device)

    order = torch.argsort(pids, stable=True)
    grouped_pids = pids[order]
    grouped_alpha = alpha[order].clamp(max=0.999999)
    starts = torch.ones_like(grouped_pids, dtype=torch.bool)
    starts[1:] = grouped_pids[1:] != grouped_pids[:-1]
    start_indices = torch.nonzero(starts, as_tuple=False).flatten()
    end_indices = torch.cat(
        [start_indices[1:], torch.tensor([pids.numel()], device=pids.device)]
    )
    lengths = end_indices - start_indices

    log_survival = torch.log1p(-grouped_alpha.double())
    inclusive = torch.cumsum(log_survival, dim=0)
    exclusive_global = inclusive - log_survival
    group_bases = exclusive_global[start_indices]
    exclusive = exclusive_global - torch.repeat_interleave(group_bases, lengths)
    weights = grouped_alpha * torch.exp(exclusive).float()
    accumulated = torch.zeros(num_pixels, dtype=torch.float32, device=pids.device)
    accumulated.index_add_(0, grouped_pids, weights)
    return order, grouped_pids, weights, accumulated


def contribution_rank(
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return zero-based descending contribution rank within each pixel."""

    pids = torch.as_tensor(pixel_ids).long().reshape(-1)
    values = torch.as_tensor(weights).float().reshape(-1)
    if pids.shape != values.shape:
        raise ValueError("pixel_ids and weights must have matching shapes")
    if pids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=pids.device)
    weight_order = torch.argsort(values, descending=True, stable=True)
    grouped_order = weight_order[
        torch.argsort(pids[weight_order], stable=True)
    ]
    grouped_pids = pids[grouped_order]
    starts = torch.ones_like(grouped_pids, dtype=torch.bool)
    starts[1:] = grouped_pids[1:] != grouped_pids[:-1]
    start_indices = torch.nonzero(starts, as_tuple=False).flatten()
    end_indices = torch.cat(
        [start_indices[1:], torch.tensor([pids.numel()], device=pids.device)]
    )
    lengths = end_indices - start_indices
    group_starts = torch.repeat_interleave(start_indices, lengths)
    grouped_rank = torch.arange(pids.numel(), device=pids.device) - group_starts
    rank = torch.empty_like(grouped_rank)
    rank[grouped_order] = grouped_rank
    return rank


def build_compositing_variants(
    pixel_ids: torch.Tensor,
    base_weights: torch.Tensor,
    *,
    num_pixels: int,
    depths: torch.Tensor | None = None,
    reference_depth: torch.Tensor | None = None,
    gammas: tuple[float, ...] = (1.5, 2.0, 4.0),
    topk: tuple[int, ...] = (1, 2, 4),
    depth_tolerance: float = 0.08,
    relative_depth_tolerance: float = 0.02,
    uncertainty: torch.Tensor | None = None,
    gaussian_ids: torch.Tensor | None = None,
    uncertainty_strengths: tuple[float, ...] = (),
) -> dict[str, torch.Tensor]:
    """Construct query-free raw contribution weights for controlled variants."""

    pids = torch.as_tensor(pixel_ids).long().reshape(-1)
    weights = torch.as_tensor(base_weights).float().reshape(-1)
    if pids.shape != weights.shape:
        raise ValueError("pixel_ids and weights must have matching shapes")
    variants = {"alpha_mean": weights}
    for gamma in gammas:
        if gamma <= 0:
            raise ValueError("compositing gamma must be positive")
        variants[f"gamma_{gamma:g}"] = weights.pow(float(gamma))
    if topk:
        rank = contribution_rank(pids, weights)
        for count in topk:
            if count <= 0:
                raise ValueError("top-k counts must be positive")
            variants[f"top{count}"] = weights * (rank < int(count))
    if depths is not None:
        depth = torch.as_tensor(depths).float().reshape(-1)
        if depth.shape != weights.shape:
            raise ValueError("depths and weights must have matching shapes")
        nearest = torch.full(
            (num_pixels,), float("inf"), dtype=torch.float32, device=weights.device
        )
        nearest.scatter_reduce_(0, pids, depth, reduce="amin", include_self=True)
        tolerance = torch.maximum(
            torch.full_like(nearest, float(depth_tolerance)),
            nearest.abs() * float(relative_depth_tolerance),
        )
        keep = depth <= nearest[pids] + tolerance[pids]
        variants["front_depth_band"] = weights * keep
        if reference_depth is not None:
            expected = torch.as_tensor(reference_depth).to(weights).float().reshape(-1)
            if expected.shape != (num_pixels,):
                raise ValueError("reference_depth must contain num_pixels values")
            expected_tolerance = torch.maximum(
                torch.full_like(expected, float(depth_tolerance)),
                expected.abs() * float(relative_depth_tolerance),
            )
            expected_keep = (
                (expected[pids] > 0)
                & ((depth - expected[pids]).abs() <= expected_tolerance[pids])
            )
            variants["expected_depth_band"] = weights * expected_keep
    if uncertainty_strengths:
        if uncertainty is None or gaussian_ids is None:
            raise ValueError("uncertainty variants require uncertainty and gaussian ids")
        row_uncertainty = torch.as_tensor(uncertainty).float().reshape(-1)
        gids = torch.as_tensor(gaussian_ids).long().reshape(-1)
        if gids.shape != weights.shape:
            raise ValueError("gaussian_ids and weights must have matching shapes")
        for strength in uncertainty_strengths:
            if strength < 0:
                raise ValueError("uncertainty strength must be non-negative")
            variants[f"uncertainty_{strength:g}"] = weights * torch.exp(
                -float(strength) * row_uncertainty[gids]
            )
    return variants


def gaussian_footprint_alphas(
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    *,
    width: int,
) -> torch.Tensor:
    """Reproduce gsplat's accepted per-hit footprint alpha for one camera.

    ``rasterize_to_indices_in_range`` has already removed footprints below
    1/255 alpha, invalid conics, and the exclusive terminal hit.  This helper
    evaluates the exact remaining footprint formula at half-pixel centres.
    """

    gids = torch.as_tensor(gaussian_ids).long().reshape(-1)
    pids = torch.as_tensor(pixel_ids).long().reshape(-1)
    if gids.shape != pids.shape or int(width) <= 0:
        raise ValueError("gaussian/pixel ids must align and width must be positive")
    means = torch.as_tensor(means2d).float()
    conic = torch.as_tensor(conics).float()
    opacity = torch.as_tensor(opacities).float()
    if means.ndim == 3:
        if means.shape[0] != 1:
            raise ValueError("gaussian_footprint_alphas expects one camera")
        means = means[0]
    if conic.ndim == 3:
        if conic.shape[0] != 1:
            raise ValueError("gaussian_footprint_alphas expects one camera")
        conic = conic[0]
    if opacity.ndim == 2:
        if opacity.shape[0] == 1:
            opacity = opacity[0]
        elif opacity.shape[1] == 1:
            opacity = opacity[:, 0]
    opacity = opacity.reshape(-1)
    if gids.numel() and (
        int(gids.min()) < 0
        or int(gids.max()) >= means.shape[0]
        or int(gids.max()) >= conic.shape[0]
        or int(gids.max()) >= opacity.shape[0]
    ):
        raise ValueError("gaussian id outside projected geometry")
    x = (pids % int(width)).float() + 0.5
    y = torch.div(pids, int(width), rounding_mode="floor").float() + 0.5
    delta_x = means[gids, 0] - x
    delta_y = means[gids, 1] - y
    q = conic[gids]
    sigma = 0.5 * (
        q[:, 0] * delta_x.square() + q[:, 2] * delta_y.square()
    ) + q[:, 1] * delta_x * delta_y
    return (opacity[gids] * torch.exp(-sigma)).clamp(max=0.999)


@torch.no_grad()
def rasterize_single_view_contributions(
    gaussian_model,
    renderer,
    viewmat: torch.Tensor,
    *,
    height: int,
    width: int,
    opacity_scale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return exact accepted 3DGS hits and front-to-back contributions.

    ``opacity_scale`` is an optional query-independent row confidence.  It is
    applied before visibility and transmittance are evaluated, matching the
    opacity semantics of :meth:`FeatureFieldRenderer.render_feature_rows`.
    """

    if bool(getattr(renderer, "use_2dgs", False)):
        raise RuntimeError("controlled contribution audit currently supports 3DGS only")
    from gsplat import rasterization
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range

    device = gaussian_model.get_xyz().device
    means = gaussian_model.get_xyz().float()
    quats = gaussian_model.get_rotation().float()
    scales = gaussian_model.get_scaling().float()
    opacities = gaussian_model.get_opacity().float().reshape(-1)
    if opacity_scale is not None:
        scale = torch.as_tensor(
            opacity_scale, device=device, dtype=opacities.dtype
        ).reshape(-1)
        if scale.shape != opacities.shape:
            raise ValueError("opacity_scale must align with Gaussian rows")
        if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
            raise ValueError("opacity_scale must be finite and non-negative")
        opacities = opacities * scale.clamp(max=1.0)
    view = torch.as_tensor(viewmat, device=device).float()
    if view.shape != (4, 4):
        raise ValueError("viewmat must be [4,4]")
    colors = torch.zeros(means.shape[0], 1, device=device)
    renders, rendered_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=view[None],
        Ks=renderer.scaled_intrinsics(int(width), int(height)).float()[None],
        width=int(width),
        height=int(height),
        near_plane=float(renderer.near_plane),
        far_plane=float(renderer.far_plane),
        backgrounds=torch.zeros(1, 1, device=device),
        render_mode="RGB+ED",
        packed=False,
    )
    total_intersections = int(info["flatten_ids"].numel())
    if total_intersections == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        return {
            "gaussian_ids": empty_long,
            "pixel_ids": empty_long,
            "alphas": empty_float,
            "weights": empty_float,
            "depths": empty_float,
            "accumulated_alpha": torch.zeros(int(height), int(width), device=device),
            "rendered_alpha": rendered_alphas[0, ..., 0].float(),
            "rendered_depth": renders[0, ..., -1].float(),
        }
    transmittance = torch.ones(1, int(height), int(width), device=device)
    gids, pids, camera_ids = rasterize_to_indices_in_range(
        0,
        total_intersections,
        transmittance,
        info["means2d"],
        info["conics"],
        info["opacities"],
        int(width),
        int(height),
        info["tile_size"],
        info["isect_offsets"],
        info["flatten_ids"],
    )
    keep = camera_ids == 0
    gids = gids[keep]
    pids = pids[keep]
    hit_alphas = gaussian_footprint_alphas(
        gids,
        pids,
        info["means2d"],
        info["conics"],
        info["opacities"],
        width=int(width),
    )
    order, grouped_pids, weights, accumulated = front_to_back_weights(
        pids, hit_alphas, num_pixels=int(height) * int(width)
    )
    projected_depths = torch.as_tensor(info["depths"]).float()
    if projected_depths.ndim == 3 and projected_depths.shape[-1] == 1:
        projected_depths = projected_depths[..., 0]
    if projected_depths.ndim == 2:
        projected_depths = projected_depths[0]
    return {
        "gaussian_ids": gids[order],
        "pixel_ids": grouped_pids,
        "alphas": hit_alphas[order],
        "weights": weights,
        "depths": projected_depths[gids[order]],
        "accumulated_alpha": accumulated.reshape(int(height), int(width)),
        "rendered_alpha": rendered_alphas[0, ..., 0].float(),
        "rendered_depth": renders[0, ..., -1].float(),
    }


def composite_feature_variants(
    row_features: torch.Tensor,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    variant_weights: Mapping[str, torch.Tensor],
    *,
    height: int,
    width: int,
    channel_chunk_size: int = 32,
    variant_chunk_size: int = 4,
    eps: float = 1e-8,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Composite multiple normalized variants while sharing sampled row chunks."""

    features = torch.as_tensor(row_features)
    gids = torch.as_tensor(gaussian_ids).long().reshape(-1)
    pids = torch.as_tensor(pixel_ids).long().reshape(-1)
    if features.ndim != 2 or gids.shape != pids.shape:
        raise ValueError("row_features must be [N,C] and hit ids must be matching 1D")
    if gids.numel() and (int(gids.min()) < 0 or int(gids.max()) >= features.shape[0]):
        raise ValueError("gaussian id outside row feature matrix")
    num_pixels = int(height) * int(width)
    names = list(variant_weights)
    if not names:
        raise ValueError("at least one compositing variant is required")
    raw = torch.stack(
        [torch.as_tensor(variant_weights[name]).to(features).float() for name in names],
        dim=1,
    )
    if raw.shape != (gids.numel(), len(names)) or bool((raw < 0).any()):
        raise ValueError("variant weights must be non-negative [num_hits]")
    mass = torch.zeros(num_pixels, len(names), dtype=torch.float32, device=features.device)
    mass.index_add_(0, pids, raw)
    normalized = raw / mass[pids].clamp_min(float(eps))

    channels = int(features.shape[1])
    output = torch.zeros(
        len(names), channels, num_pixels, dtype=torch.float32, device=features.device
    )
    for channel_start in range(0, channels, int(channel_chunk_size)):
        channel_end = min(channel_start + int(channel_chunk_size), channels)
        sampled = features[gids, channel_start:channel_end].float()
        for variant_start in range(0, len(names), int(variant_chunk_size)):
            variant_end = min(variant_start + int(variant_chunk_size), len(names))
            variant_count = variant_end - variant_start
            weighted = (
                normalized[:, variant_start:variant_end, None] * sampled[:, None, :]
            )
            flat_indices = (
                pids[:, None] * variant_count
                + torch.arange(variant_count, device=pids.device)[None]
            ).reshape(-1)
            accumulator = torch.zeros(
                num_pixels * variant_count,
                channel_end - channel_start,
                dtype=torch.float32,
                device=features.device,
            )
            accumulator.index_add_(0, flat_indices, weighted.reshape(-1, channel_end - channel_start))
            values = accumulator.reshape(num_pixels, variant_count, -1).permute(1, 2, 0)
            output[
                variant_start:variant_end, channel_start:channel_end
            ] = values
    maps = {
        name: output[index].reshape(channels, int(height), int(width))
        for index, name in enumerate(names)
    }
    masses = {
        name: mass[:, index].reshape(int(height), int(width))
        for index, name in enumerate(names)
    }
    return maps, masses


@torch.no_grad()
def render_contribution_sharpened_features(
    gaussian_model,
    renderer,
    viewmat: torch.Tensor,
    row_features: torch.Tensor,
    *,
    height: int,
    width: int,
    gamma: float = 2.0,
    opacity_scale: torch.Tensor | None = None,
    contributions: Mapping[str, torch.Tensor] | None = None,
    channel_chunk_size: int | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Render a query-independent sharpened mixture of primitive features.

    Standard alpha compositing gives each accepted hit a front-to-back
    contribution ``w``.  This observation operator replaces only the
    *normalized feature-mixture* weights by ``w**gamma``.  Geometry, depth,
    accumulated alpha, primitive features, and query scores are unchanged.

    The explicit hit path is inference-only and currently supports 3DGS.  A
    precomputed contribution mapping can be reused for multiple feature banks
    from the same geometry, camera, resolution, and opacity scale.
    """

    exponent = float(gamma)
    if not torch.isfinite(torch.tensor(exponent)) or exponent <= 0:
        raise ValueError("contribution gamma must be finite and positive")
    features = torch.as_tensor(row_features)
    if features.ndim != 2 or features.shape[0] != gaussian_model.get_xyz().shape[0]:
        raise ValueError("row_features must align with Gaussian rows [N,C]")
    if features.device != gaussian_model.get_xyz().device:
        raise ValueError("row_features and Gaussian geometry must share a device")
    hits = (
        rasterize_single_view_contributions(
            gaussian_model,
            renderer,
            viewmat,
            height=int(height),
            width=int(width),
            opacity_scale=opacity_scale,
        )
        if contributions is None
        else dict(contributions)
    )
    required = {
        "gaussian_ids",
        "pixel_ids",
        "weights",
        "rendered_alpha",
        "rendered_depth",
    }
    missing = sorted(required - set(hits))
    if missing:
        raise ValueError(f"contribution cache is missing keys: {missing}")
    raw_weights = torch.as_tensor(hits["weights"]).float()
    maps, masses = composite_feature_variants(
        features,
        hits["gaussian_ids"],
        hits["pixel_ids"],
        {"selected": raw_weights.pow(exponent)},
        height=int(height),
        width=int(width),
        channel_chunk_size=(
            int(channel_chunk_size)
            if channel_chunk_size is not None
            else int(getattr(renderer, "max_channels_per_chunk", 32))
        ),
        variant_chunk_size=1,
        eps=float(eps),
    )
    return {
        "feature_map": maps["selected"],
        "depth_map": torch.as_tensor(hits["rendered_depth"]).float(),
        "alpha_map": torch.as_tensor(hits["rendered_alpha"]).float(),
        "compositing_mass_map": masses["selected"],
        "contribution_gamma": torch.tensor(
            exponent, device=features.device, dtype=torch.float32
        ),
    }
