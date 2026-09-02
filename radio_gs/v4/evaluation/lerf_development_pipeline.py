"""Development-only end-to-end LERF evaluation for the v4 scene state.

The method state is frozen from authorized source views before benchmark labels
or text queries are opened.  Diagnostic gates are reported as warnings here;
they deliberately do not block execution or claim frozen-protocol promotion.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.v4.carrier import Camera, SurfaceVoxelCarrier
from radio_gs.v4.completion.lerf_adapter import (
    apply_scannet_spatial_mass_candidate,
    build_real_token_runtime,
)
from radio_gs.v4.completion.scannet import (
    _load_source_rgb,
    _radio_projection_matrix,
)
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.contracts.surface_scene_bundle import load_geometry_binding
from radio_gs.v4.evaluation.lerf_source_mask_gate import _load_sam_records, _masks
from radio_gs.v4.evaluation.real_sam_token_association import _lift_masks
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images
from radio_gs.v4.object_memory import ObservedObjectEvidence, SurfaceTokenBootstrap


GENERIC_NEGATIVES = ("object", "things", "stuff", "texture")


def _validate_source_authority(authority: dict[str, Any], scene_label: str) -> list[int]:
    if authority.get("contract") != "sam3-query-free-source-rgb-authority-v1":
        raise ValueError("source RGB authority contract differs")
    if str(authority.get("scene")) != scene_label:
        raise ValueError("source RGB authority scene differs")
    policy = authority.get("information_policy", {})
    required_false = (
        "benchmark_ground_truth_used", "query_text_used", "target_or_evaluation_rgb_used"
    )
    if any(policy.get(key) is not False for key in required_false):
        raise ValueError("source RGB authority violates the development information policy")
    if policy.get("registered_source_rgb_only") is not True:
        raise ValueError("source RGB authority is not restricted to registered source views")
    frames = [_frame_index(str(item["image_id"])) for item in authority.get("images", [])]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError("source RGB authority frames are empty or duplicated")
    return sorted(frames)


def _validate_semantic_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    authority: dict[str, Any],
    feature_dir: Path,
    source_frames: list[int],
) -> list[int]:
    backbone = manifest.get("features", {}).get("backbone", {})
    grid = backbone.get("grid")
    if (
        backbone.get("dim") != 1280
        or not isinstance(grid, list)
        or len(grid) != 2
        or any(not isinstance(value, int) or value <= 0 for value in grid)
    ):
        raise ValueError("semantic frame manifest lacks the expected RADIO backbone")
    binding = authority.get("construction", {}).get("frame_manifest", {})
    if binding.get("sha256") != sha256_file(manifest_path):
        raise ValueError("semantic frame manifest differs from source authority binding")
    expected_dir = (manifest_path.parent / str(backbone.get("subdir", ""))).resolve()
    if feature_dir != expected_dir:
        raise ValueError("semantic feature directory differs from its manifest")
    records = manifest.get("frames", [])
    frames = [int(record["frame_idx"]) for record in records]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError("semantic frame manifest frames are empty or duplicated")
    excluded = {_frame_index(str(value)) for value in manifest.get("excluded_image_names", [])}
    if excluded & set(frames):
        raise ValueError("semantic frame manifest includes an explicitly excluded frame")
    if not set(source_frames).issubset(frames):
        raise ValueError("semantic frame manifest omits sealed object-memory source frames")
    return sorted(frames)


def _load_dense_feature(
    path: Path,
    *,
    expected_channels: int,
    expected_height: int | None = None,
    expected_width: int | None = None,
) -> torch.Tensor:
    """Load one feature raster and reject malformed or non-finite inputs."""

    value = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"semantic feature must be a [D,H,W] tensor: {path}")
    actual = tuple(int(size) for size in value.shape)
    if actual[0] != expected_channels:
        raise ValueError(f"semantic feature channel mismatch at {path}: {actual[0]}")
    if expected_height is not None and actual[1] != expected_height:
        raise ValueError(f"semantic feature height mismatch at {path}: {actual[1]}")
    if expected_width is not None and actual[2] != expected_width:
        raise ValueError(f"semantic feature width mismatch at {path}: {actual[2]}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"semantic feature contains NaN or Inf: {path}")
    return value


def _load_text_cache(
    path: Path,
    required_queries: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Load an exact frozen SigLIP2 text bank without importing legacy evaluators."""

    payload = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if payload.get("text_encoder") != "siglip2":
        raise ValueError(f"text cache is not a SigLIP2 bank: {path}")
    queries = [str(value) for value in payload.get("queries", [])]
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"text cache embeddings must be a matrix: {path}")
    if len(queries) != embeddings.shape[0] or len(queries) != len(set(queries)):
        raise ValueError(f"text cache queries are misaligned or duplicated: {path}")
    if embeddings.shape[1] != 1536 or not bool(torch.isfinite(embeddings).all()):
        raise ValueError(f"text cache embeddings have the wrong dimension or values: {path}")
    bank = dict(zip(queries, embeddings))
    missing = [query for query in required_queries if query not in bank]
    if missing:
        raise ValueError(f"text cache lacks exact queries: {missing}")
    return F.normalize(
        torch.stack([bank[query] for query in required_queries]).float(), dim=-1
    ).to(device)


def _frame_index(value: str) -> int:
    matches = re.findall(r"\d+", value)
    if not matches:
        raise ValueError(f"no frame index in {value!r}")
    return int(matches[-1])


def _masked_descriptors(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Return normalized dense-feature means for every source proposal."""

    features = torch.as_tensor(features, dtype=torch.float32)
    masks = torch.as_tensor(masks, dtype=torch.float32, device=features.device)
    if features.ndim != 3 or masks.ndim != 3 or features.shape[1:] != masks.shape[1:]:
        raise ValueError("features [D,H,W] and masks [M,H,W] must share a raster")
    flat_masks = masks.flatten(1)
    mass = flat_masks.sum(-1, keepdim=True).clamp_min(1e-8)
    values = flat_masks @ features.flatten(1).T / mass
    return F.normalize(values, dim=-1, eps=1e-8)


def _validated_projection_samples(
    projection: Any,
    *,
    num_elements: int,
    num_pixels: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bounded finite sparse projection samples or fail closed."""

    element_ids = torch.as_tensor(projection.element_ids, dtype=torch.long, device=device)
    pixel_ids = torch.as_tensor(projection.pixel_ids, dtype=torch.long, device=device)
    weights = torch.as_tensor(projection.weights, dtype=torch.float32, device=device)
    if element_ids.ndim != 1 or pixel_ids.shape != element_ids.shape or weights.shape != element_ids.shape:
        raise ValueError("projection samples must be aligned vectors")
    if element_ids.numel():
        if int(element_ids.min()) < 0 or int(element_ids.max()) >= num_elements:
            raise ValueError("projection contains an out-of-range surface element")
        if int(pixel_ids.min()) < 0 or int(pixel_ids.max()) >= num_pixels:
            raise ValueError("projection contains an out-of-range raster pixel")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("projection weights must be finite and non-negative")
    return element_ids, pixel_ids, weights


def accumulate_surface_features(
    feature_sum: torch.Tensor,
    feature_mass: torch.Tensor,
    features: torch.Tensor,
    projection: Any,
    *,
    channel_chunk_size: int = 192,
) -> None:
    """Accumulate query-independent dense visual evidence on carrier elements."""

    if feature_sum.ndim != 2 or feature_mass.shape != (feature_sum.shape[0],):
        raise ValueError("surface feature accumulators must have shapes [E,D] and [E]")
    features = torch.as_tensor(features, dtype=torch.float32, device=feature_sum.device)
    if features.ndim != 3 or features.shape[0] != feature_sum.shape[1]:
        raise ValueError("dense features must have shape [D,H,W]")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("dense features must be finite")
    if channel_chunk_size <= 0:
        raise ValueError("channel chunk size must be positive")
    element_ids, pixel_ids, weights = _validated_projection_samples(
        projection, num_elements=feature_sum.shape[0],
        num_pixels=features.shape[1] * features.shape[2], device=feature_sum.device,
    )
    if not element_ids.numel():
        return
    feature_mass.scatter_add_(0, element_ids, weights)
    pixels = features.flatten(1).T
    for start in range(0, feature_sum.shape[1], channel_chunk_size):
        stop = min(start + channel_chunk_size, feature_sum.shape[1])
        feature_sum[:, start:stop].index_add_(
            0, element_ids, pixels[pixel_ids, start:stop] * weights[:, None]
        )


def update_surface_view_prototypes(
    prototype_descriptors: torch.Tensor,
    prototype_mass: torch.Tensor,
    features: torch.Tensor,
    projection: Any,
    *,
    channel_chunk_size: int = 192,
) -> None:
    """Keep a bounded set of strongest query-independent view observations."""

    if prototype_descriptors.ndim != 3 or prototype_mass.shape != prototype_descriptors.shape[:2]:
        raise ValueError("view prototype tensors must have shapes [E,R,D] and [E,R]")
    features = torch.as_tensor(
        features, dtype=torch.float32, device=prototype_descriptors.device
    )
    if features.ndim != 3 or features.shape[0] != prototype_descriptors.shape[2]:
        raise ValueError("dense features do not align with view prototypes")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("dense features must be finite")
    element_ids, pixel_ids, weights = _validated_projection_samples(
        projection, num_elements=prototype_descriptors.shape[0],
        num_pixels=features.shape[1] * features.shape[2],
        device=prototype_descriptors.device,
    )
    if not element_ids.numel():
        return
    unique_elements, inverse = torch.unique(element_ids, sorted=True, return_inverse=True)
    view_mass = torch.zeros(unique_elements.numel(), device=prototype_descriptors.device)
    view_mass.scatter_add_(0, inverse, weights)
    pixels = features.flatten(1).T
    view_sum = torch.zeros(
        unique_elements.numel(), features.shape[0],
        dtype=torch.float32, device=prototype_descriptors.device,
    )
    for start in range(0, features.shape[0], channel_chunk_size):
        stop = min(start + channel_chunk_size, features.shape[0])
        view_sum[:, start:stop].index_add_(
            0, inverse, pixels[pixel_ids, start:stop] * weights[:, None]
        )
    view_descriptor = F.normalize(
        view_sum / view_mass[:, None].clamp_min(1e-8), dim=-1, eps=1e-8
    ).to(prototype_descriptors.dtype)
    current_mass = prototype_mass[unique_elements]
    weakest_mass, weakest_slot = current_mass.min(-1)
    accepted = view_mass > weakest_mass
    if bool(accepted.any()):
        rows = unique_elements[accepted]
        slots = weakest_slot[accepted]
        prototype_mass[rows, slots] = view_mass[accepted]
        prototype_descriptors[rows, slots] = view_descriptor[accepted]


def accumulate_consistency_weighted_surface_features(
    feature_sum: torch.Tensor,
    feature_mass: torch.Tensor,
    reference_feature_sum: torch.Tensor,
    reference_feature_mass: torch.Tensor,
    features: torch.Tensor,
    projection: Any,
    *,
    agreement_floor: float = 0.3,
    agreement_power: float = 2.0,
    channel_chunk_size: int = 192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate one equal-view observation using source-only consistency."""

    if feature_sum.ndim != 2 or feature_mass.shape != (feature_sum.shape[0],):
        raise ValueError("consistency accumulators must have shapes [E,D] and [E]")
    reference_sum = torch.as_tensor(
        reference_feature_sum, dtype=torch.float32, device=feature_sum.device
    )
    reference_mass = torch.as_tensor(
        reference_feature_mass, dtype=torch.float32, device=feature_sum.device
    )
    features = torch.as_tensor(features, dtype=torch.float32, device=feature_sum.device)
    if (
        reference_sum.shape != feature_sum.shape
        or reference_mass.shape != feature_mass.shape
        or features.ndim != 3
    ):
        raise ValueError("consistency reference or dense features do not align")
    if features.shape[0] != feature_sum.shape[1] or not bool(torch.isfinite(features).all()):
        raise ValueError("consistency features have the wrong dimension or values")
    if not 0 <= agreement_floor < 1 or agreement_power <= 0:
        raise ValueError("consistency agreement parameters are invalid")
    element_ids, pixel_ids, weights = _validated_projection_samples(
        projection, num_elements=feature_sum.shape[0],
        num_pixels=features.shape[1] * features.shape[2], device=feature_sum.device,
    )
    reliability = feature_mass.new_zeros(feature_mass.shape)
    view_observed = torch.zeros(
        feature_mass.shape, dtype=torch.bool, device=feature_mass.device
    )
    if not element_ids.numel():
        return reliability, view_observed
    unique_elements, inverse = torch.unique(element_ids, sorted=True, return_inverse=True)
    view_observed[unique_elements] = True
    view_mass = feature_mass.new_zeros(unique_elements.numel())
    view_mass.scatter_add_(0, inverse, weights)
    view_sum = feature_sum.new_zeros((unique_elements.numel(), feature_sum.shape[1]))
    pixels = features.flatten(1).T
    for start in range(0, feature_sum.shape[1], channel_chunk_size):
        stop = min(start + channel_chunk_size, feature_sum.shape[1])
        view_sum[:, start:stop].index_add_(
            0, inverse, pixels[pixel_ids, start:stop] * weights[:, None]
        )
    view_mean = view_sum / view_mass[:, None].clamp_min(1e-8)
    view_descriptor = F.normalize(view_mean, dim=-1, eps=1e-8)
    peer_mass = reference_mass[unique_elements] - view_mass
    peer_sum = reference_sum[unique_elements] - view_sum
    has_peer = peer_mass > 1e-8
    peer_descriptor = F.normalize(
        peer_sum / peer_mass[:, None].clamp_min(1e-8), dim=-1, eps=1e-8
    )
    agreement = (view_descriptor * peer_descriptor).sum(-1)
    agreement = torch.where(has_peer, agreement, torch.ones_like(agreement))
    view_reliability = (
        (agreement - agreement_floor) / (1.0 - agreement_floor)
    ).clamp(0, 1).pow(agreement_power)
    effective_mass = view_mass * view_reliability
    feature_sum[unique_elements] += view_mean * effective_mass[:, None]
    feature_mass[unique_elements] += effective_mass
    reliability[unique_elements] = view_reliability
    return reliability, view_observed


def conservative_token_geometry_completion(
    centres: torch.Tensor,
    membership: torch.Tensor,
    *,
    observed_threshold: float = 1e-5,
    radius_multiplier: float = 1.5,
    completion_weight: float = 0.25,
    minimum_scale: float = 0.04,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill only currently unobserved elements with one low-weight token guess."""

    centres = torch.as_tensor(centres, dtype=torch.float32)
    membership = torch.as_tensor(membership, dtype=torch.float32, device=centres.device)
    if centres.ndim != 2 or centres.shape[1] != 3 or membership.ndim != 2:
        raise ValueError("centres and membership must have shapes [E,3] and [E,K]")
    if membership.shape[0] != centres.shape[0]:
        raise ValueError("membership does not align with carrier elements")
    completed = membership.clone()
    filled = torch.zeros(centres.shape[0], dtype=torch.bool, device=centres.device)
    if membership.shape[1] == 0:
        return completed, filled
    observed = membership.max(-1).values > observed_threshold
    mass = membership.sum(0).clamp_min(1e-8)
    token_centres = membership.T @ centres / mass[:, None]
    residual = centres[:, None, :] - token_centres[None, :, :]
    variance = (membership[..., None] * residual.square()).sum(0) / mass[:, None]
    token_scales = variance.sqrt().clamp_min(minimum_scale)
    unknown_ids = torch.where(~observed)[0]
    for start in range(0, unknown_ids.numel(), chunk_size):
        ids = unknown_ids[start : start + chunk_size]
        normalized = (centres[ids, None] - token_centres[None]) / token_scales[None]
        distance = normalized.square().sum(-1).sqrt()
        best_distance, best_token = distance.min(-1)
        accepted = best_distance <= radius_multiplier
        if bool(accepted.any()):
            accepted_ids = ids[accepted]
            completed[accepted_ids, best_token[accepted]] = completion_weight
            filled[accepted_ids] = True
    return completed, filled


def token_null_posterior(
    token_descriptors: torch.Tensor,
    query_descriptors: torch.Tensor,
    *,
    temperature: float = 0.03,
    null_similarity: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For every query-token pair, normalize positive evidence against null."""

    tokens = F.normalize(torch.as_tensor(token_descriptors, dtype=torch.float32), dim=-1)
    queries = F.normalize(
        torch.as_tensor(query_descriptors, dtype=torch.float32, device=tokens.device), dim=-1
    )
    if tokens.ndim != 2 or queries.ndim != 2 or tokens.shape[1] != queries.shape[1]:
        raise ValueError("token and query descriptors must be aligned matrices")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    token_logits = queries @ tokens.T / temperature
    null = torch.as_tensor(null_similarity, dtype=torch.float32, device=tokens.device)
    if null.ndim == 0:
        null = null.expand(tokens.shape[0])
    if null.shape != (tokens.shape[0],):
        raise ValueError("null similarity must be scalar or have one value per token")
    null_logits = null[None].expand(queries.shape[0], -1) / temperature
    posterior = torch.softmax(torch.stack([token_logits, null_logits], dim=-1), dim=-1)
    return posterior[..., 0], posterior[..., 1]


def retain_top_query_tokens(probability: torch.Tensor, maximum_tokens: int) -> torch.Tensor:
    """Apply one category-independent capacity bound to every text query."""

    probability = torch.as_tensor(probability, dtype=torch.float32)
    if probability.ndim != 2 or maximum_tokens <= 0:
        raise ValueError("probability must be [Q,K] and maximum_tokens positive")
    if probability.shape[1] <= maximum_tokens:
        return probability
    _, indices = probability.topk(maximum_tokens, dim=-1)
    retained = torch.zeros_like(probability)
    retained.scatter_(1, indices, probability.gather(1, indices))
    return retained


def prototype_max_token_posterior(
    prototypes: torch.Tensor,
    prototype_token_ids: torch.Tensor,
    queries: torch.Tensor,
    negatives: torch.Tensor,
    *,
    num_tokens: int,
    temperature: float = 0.03,
    null_similarity: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool the strongest source observation inside each persistent token."""

    prototypes = F.normalize(torch.as_tensor(prototypes, dtype=torch.float32), dim=-1)
    queries = F.normalize(torch.as_tensor(queries, dtype=torch.float32, device=prototypes.device), dim=-1)
    negatives = F.normalize(
        torch.as_tensor(negatives, dtype=torch.float32, device=prototypes.device), dim=-1
    )
    token_ids = torch.as_tensor(prototype_token_ids, dtype=torch.long, device=prototypes.device)
    if prototypes.ndim != 2 or token_ids.shape != (prototypes.shape[0],):
        raise ValueError("prototypes and their token ids must align")
    if num_tokens <= 0 or temperature <= 0:
        raise ValueError("num_tokens and temperature must be positive")
    positive_similarity = queries @ prototypes.T
    negative_similarity = (prototypes @ negatives.T).max(-1).values
    negative_similarity = torch.maximum(
        negative_similarity, torch.full_like(negative_similarity, null_similarity)
    )
    prototype_probability = torch.sigmoid(
        (positive_similarity - negative_similarity[None]) / temperature
    )
    token_probability = prototypes.new_zeros((queries.shape[0], num_tokens))
    for token_id in range(num_tokens):
        selected = token_ids == token_id
        if bool(selected.any()):
            token_probability[:, token_id] = prototype_probability[:, selected].max(-1).values
    return token_probability, 1.0 - token_probability


def surface_identity_token_posterior(
    surface_descriptors: torch.Tensor,
    semantic_observed: torch.Tensor,
    observed_membership: torch.Tensor,
    queries: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float = 0.03,
    null_similarity: float = 0.0,
    membership_threshold: float = 0.05,
    peak_count: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Localize semantic identity on the surface, then pool it inside tokens."""

    descriptors = F.normalize(torch.as_tensor(surface_descriptors, dtype=torch.float32), dim=-1)
    membership = torch.as_tensor(
        observed_membership, dtype=torch.float32, device=descriptors.device
    )
    observed = torch.as_tensor(semantic_observed, dtype=torch.bool, device=descriptors.device)
    queries = F.normalize(torch.as_tensor(queries, dtype=torch.float32, device=descriptors.device), dim=-1)
    negatives = F.normalize(
        torch.as_tensor(negatives, dtype=torch.float32, device=descriptors.device), dim=-1
    )
    if descriptors.ndim != 2 or membership.ndim != 2:
        raise ValueError("surface descriptors and memberships must be matrices")
    if descriptors.shape[0] != membership.shape[0] or observed.shape != (descriptors.shape[0],):
        raise ValueError("surface semantic evidence does not align with memberships")
    if descriptors.shape[1] != queries.shape[1] or descriptors.shape[1] != negatives.shape[1]:
        raise ValueError("surface and text descriptor dimensions differ")
    if temperature <= 0 or peak_count <= 0:
        raise ValueError("temperature and peak count must be positive")
    local_probability = surface_local_query_posterior(
        descriptors, observed, queries, negatives,
        temperature=temperature, null_similarity=null_similarity,
    )
    token_probability = pool_surface_identity_by_token(
        local_probability, observed, membership,
        membership_threshold=membership_threshold, peak_count=peak_count,
    )
    return token_probability, 1.0 - token_probability


def pool_surface_identity_by_token(
    local_probability: torch.Tensor,
    semantic_observed: torch.Tensor,
    observed_membership: torch.Tensor,
    *,
    membership_threshold: float = 0.05,
    peak_count: int = 8,
) -> torch.Tensor:
    """Pool an already localized identity unary inside observed token support."""

    local_probability = torch.as_tensor(local_probability, dtype=torch.float32)
    observed = torch.as_tensor(
        semantic_observed, dtype=torch.bool, device=local_probability.device
    )
    membership = torch.as_tensor(
        observed_membership, dtype=torch.float32, device=local_probability.device
    )
    if local_probability.ndim != 2 or membership.ndim != 2:
        raise ValueError("local probability and membership must be matrices")
    if local_probability.shape[1] != membership.shape[0] or observed.shape != (membership.shape[0],):
        raise ValueError("local probability does not align with surface membership")
    if peak_count <= 0:
        raise ValueError("peak count must be positive")
    token_probability = local_probability.new_zeros(
        (local_probability.shape[0], membership.shape[1])
    )
    for token_id in range(membership.shape[1]):
        selected = observed & (membership[:, token_id] >= membership_threshold)
        count = min(peak_count, int(selected.sum()))
        if count:
            token_probability[:, token_id] = local_probability[:, selected].topk(
                count, dim=-1
            ).values.mean(-1)
    return token_probability


def surface_local_query_posterior(
    surface_descriptors: torch.Tensor,
    semantic_observed: torch.Tensor,
    queries: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float = 0.03,
    null_similarity: float = 0.0,
) -> torch.Tensor:
    """Compute an identity-only probability without consulting object extent."""

    descriptors = F.normalize(torch.as_tensor(surface_descriptors, dtype=torch.float32), dim=-1)
    observed = torch.as_tensor(semantic_observed, dtype=torch.bool, device=descriptors.device)
    queries = F.normalize(torch.as_tensor(queries, dtype=torch.float32, device=descriptors.device), dim=-1)
    negatives = F.normalize(
        torch.as_tensor(negatives, dtype=torch.float32, device=descriptors.device), dim=-1
    )
    if descriptors.ndim != 2 or observed.shape != (descriptors.shape[0],):
        raise ValueError("surface descriptors and observed mask must align")
    if descriptors.shape[1] != queries.shape[1] or descriptors.shape[1] != negatives.shape[1]:
        raise ValueError("surface and text descriptor dimensions differ")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positive_similarity = queries @ descriptors.T
    negative_similarity = (descriptors @ negatives.T).max(-1).values
    negative_similarity = torch.maximum(
        negative_similarity, torch.full_like(negative_similarity, null_similarity)
    )
    probability = torch.sigmoid(
        (positive_similarity - negative_similarity[None]) / temperature
    )
    probability[:, ~observed] = 0
    return probability


def surface_view_prototype_query_posterior(
    prototype_descriptors: torch.Tensor,
    prototype_mass: torch.Tensor,
    queries: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float = 0.03,
    null_similarity: float = 0.0,
) -> torch.Tensor:
    """Select the strongest retained view-local identity per surface element."""

    prototypes = F.normalize(torch.as_tensor(prototype_descriptors, dtype=torch.float32), dim=-1)
    mass = torch.as_tensor(prototype_mass, dtype=torch.float32, device=prototypes.device)
    queries = F.normalize(torch.as_tensor(queries, dtype=torch.float32, device=prototypes.device), dim=-1)
    negatives = F.normalize(
        torch.as_tensor(negatives, dtype=torch.float32, device=prototypes.device), dim=-1
    )
    if prototypes.ndim != 3 or mass.shape != prototypes.shape[:2]:
        raise ValueError("surface view prototypes and mass must align")
    if prototypes.shape[2] != queries.shape[1] or prototypes.shape[2] != negatives.shape[1]:
        raise ValueError("surface prototype and text dimensions differ")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    flat = prototypes.reshape(-1, prototypes.shape[-1])
    positive_similarity = queries @ flat.T
    negative_similarity = (flat @ negatives.T).max(-1).values
    negative_similarity = torch.maximum(
        negative_similarity, torch.full_like(negative_similarity, null_similarity)
    )
    probability = torch.sigmoid(
        (positive_similarity - negative_similarity[None]) / temperature
    ).reshape(queries.shape[0], prototypes.shape[0], prototypes.shape[1])
    probability[:, mass <= 0] = 0
    return probability.max(-1).values


def select_consistent_surface_view(
    prototype_descriptors: torch.Tensor,
    prototype_mass: torch.Tensor,
    average_descriptors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the retained view most consistent with each cross-view mean."""

    prototypes = F.normalize(
        torch.as_tensor(prototype_descriptors, dtype=torch.float32), dim=-1
    )
    mass = torch.as_tensor(prototype_mass, dtype=torch.float32, device=prototypes.device)
    average = F.normalize(
        torch.as_tensor(average_descriptors, dtype=torch.float32, device=prototypes.device),
        dim=-1,
    )
    if prototypes.ndim != 3 or mass.shape != prototypes.shape[:2]:
        raise ValueError("surface view prototypes and mass must align")
    if average.shape != (prototypes.shape[0], prototypes.shape[2]):
        raise ValueError("cross-view average does not align with surface prototypes")
    observed = mass.gt(0).any(-1)
    agreement = (prototypes * average[:, None]).sum(-1)
    agreement = agreement.masked_fill(mass <= 0, -torch.inf)
    selected_slot = agreement.argmax(-1)
    selected = prototypes[
        torch.arange(prototypes.shape[0], device=prototypes.device), selected_slot
    ]
    selected[~observed] = 0
    return selected, observed


def _load_labels(label_root: Path, scene_label: str) -> tuple[dict[int, list[dict[str, Any]]], list[str], int, int]:
    scene_dir = label_root / scene_label
    annotations: dict[int, list[dict[str, Any]]] = {}
    categories: set[str] = set()
    source_height = source_width = 0
    for path in sorted(scene_dir.glob("frame_*.json")):
        payload = json.loads(path.read_text())
        info = payload.get("info", {})
        source_height = int(info.get("height", source_height))
        source_width = int(info.get("width", source_width))
        objects = []
        for item in payload.get("objects", []):
            category = str(item.get("category", "")).strip()
            segmentation = item.get("segmentation", [])
            if isinstance(segmentation, dict):
                segmentation = segmentation.get("polygons", [])
            if segmentation:
                array = np.asarray(segmentation, dtype=np.float32)
                if array.ndim == 2 and array.shape[1] == 2:
                    segmentation = [array]
                elif isinstance(segmentation[0], (int, float)):
                    segmentation = [segmentation]
            polygons = []
            for value in segmentation:
                array = np.asarray(value, dtype=np.float32).reshape(-1, 2)
                if array.shape[0] >= 3:
                    polygons.append(array)
            if category and polygons:
                categories.add(category)
                objects.append({"category": category, "polygons": polygons})
        annotations[_frame_index(path.stem)] = objects
    if not annotations:
        raise FileNotFoundError(f"no LERF annotations under {scene_dir}")
    if not categories:
        raise ValueError(f"LERF annotations under {scene_dir} contain no valid polygons")
    return annotations, sorted(categories), source_height, source_width


def _ground_truth_masks(
    objects: list[dict[str, Any]], categories: list[str], height: int, width: int,
    source_height: int, source_width: int,
) -> dict[str, torch.Tensor]:
    scale = np.asarray([width / source_width, height / source_height], dtype=np.float32)
    output = {}
    for category in categories:
        mask = np.zeros((height, width), dtype=np.uint8)
        polygons = [
            np.round(polygon * scale).astype(np.int32)
            for item in objects if item["category"] == category for polygon in item["polygons"]
        ]
        if polygons:
            cv2.fillPoly(mask, polygons, 1)
        output[category] = torch.from_numpy(mask.astype(bool))
    return output


def _camera(view: Any, frame_id: int, height: int, width: int) -> Camera:
    intrinsic = view.intrinsic.clone()
    intrinsic[0] *= width / view.width
    intrinsic[1] *= height / view.height
    return Camera(str(frame_id), intrinsic, view.camera_to_world, height, width)


def _binary_iou(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, int, int]:
    prediction, target = prediction.bool(), target.bool()
    intersection = int((prediction & target).sum())
    union = int((prediction | target).sum())
    return (float(intersection / union) if union else float("nan"), intersection, union)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not getattr(args, "allow_deprecated_development_proxy", False):
        raise RuntimeError(
            "the legacy LERF polygon proxy is quarantined; build a cold-loadable "
            "v4 surface scene bundle and use the formal evaluator"
        )
    torch.set_num_threads(args.cpu_threads)
    if args.surface_view_prototype_count <= 0:
        raise ValueError("surface view prototype count must be positive")
    if not args.text_query_cache or not args.negative_text_query_cache:
        raise ValueError("exact positive and negative frozen text caches are required")
    device = torch.device(args.device)
    scene_root = Path(args.scene_root).resolve(strict=True)
    authority_path = Path(args.source_rgb_authority).resolve(strict=True)
    authority = json.loads(authority_path.read_text())
    source_frames = _validate_source_authority(authority, args.scene_label)
    source_image_paths = {
        _frame_index(str(item["image_id"])): Path(item["path"]).resolve(strict=True)
        for item in authority["images"]
    }
    if set(source_image_paths) != set(source_frames):
        raise ValueError("source RGB authority paths do not match its frame inventory")
    sam_paths = [Path(value).resolve(strict=True) for value in args.sam_manifest]
    sam_records = _load_sam_records(sam_paths)
    if set(source_frames) != set(sam_records):
        raise ValueError("source authority and SAM records have different frames")
    sparse = scene_root / "sparse" / "0"
    raw_cameras = _read_cameras_binary(sparse / "cameras.bin")
    views = _read_images(sparse / "images.bin", raw_cameras)

    surface_path = Path(args.surface_carrier).resolve(strict=True)
    geometry_binding, surface_payload = load_geometry_binding(
        args.geometry_authority, surface_path
    )
    carrier = geometry_binding.configuration.build_carrier(
        surface_payload["centres"],
        normals=surface_payload.get("normals"),
        confidence=surface_payload.get(
            "confidence", torch.ones(len(surface_payload["centres"]), dtype=torch.float32)
        ),
    )
    feature_dir = Path(args.siglip_feature_dir).resolve(strict=True)
    bound_manifest_value = authority.get("construction", {}).get("frame_manifest", {}).get("path")
    if not bound_manifest_value:
        raise ValueError("source RGB authority lacks a bound feature manifest")
    bound_manifest_path = Path(bound_manifest_value).resolve(strict=True)
    semantic_manifest_path = (
        Path(args.semantic_frame_manifest).resolve(strict=True)
        if args.semantic_frame_manifest else bound_manifest_path
    )
    semantic_extra_frames: list[int] = []
    semantic_manifest = json.loads(semantic_manifest_path.read_text())
    manifest_frames = _validate_semantic_manifest(
        semantic_manifest, semantic_manifest_path, authority, feature_dir, source_frames
    )
    if args.semantic_frame_manifest:
        semantic_extra_frames = sorted(set(manifest_frames) - set(source_frames))
    missing_views = sorted(set(source_frames + semantic_extra_frames) - set(views))
    if missing_views:
        raise ValueError(f"semantic manifest frames are absent from COLMAP registration: {missing_views}")
    first_feature = _load_dense_feature(
        feature_dir / f"rgb_{source_frames[0]}.pt", expected_channels=1280
    )
    input_feature_dimension, height, width = first_feature.shape
    declared_grid = semantic_manifest["features"]["backbone"]["grid"]
    if [height, width] != declared_grid:
        raise ValueError("semantic feature tensor grid differs from its bound manifest")
    summary_head = None
    if args.semantic_feature_space == "radio_backbone_summary_head":
        if input_feature_dimension != 1280 or not (args.summary_head_weights or args.radio_checkpoint):
            raise ValueError("corrected text-aligned semantics require 1280D backbone and head weights")
        summary_head = (
            SigLIP2SummaryHead.from_extracted_weights(
                str(Path(args.summary_head_weights).resolve(strict=True))
            )
            if args.summary_head_weights
            else SigLIP2SummaryHead.from_radio_checkpoint(
                str(Path(args.radio_checkpoint).resolve(strict=True))
            )
        ).to(device).eval()
        for parameter in summary_head.parameters():
            parameter.requires_grad_(False)
        feature_dimension = 1536
    else:
        raise ValueError("the RADIO SigLIP2 spatial adaptor is not text-aligned")
    model = SurfaceTokenBootstrap(
        carrier.centres.to(device), minimum_overlap=args.minimum_overlap,
        null_logit=args.association_null_logit, temperature=args.association_temperature,
        geometry_weight=args.geometry_weight, appearance_weight=args.appearance_weight,
        batch_birth_overlap=args.batch_birth_overlap,
    )

    surface_feature_sum = torch.zeros(
        carrier.num_elements, feature_dimension, dtype=torch.float32, device=device
    )
    surface_feature_mass = torch.zeros(carrier.num_elements, dtype=torch.float32, device=device)
    learned_completion = args.completion_mode == "scannet_spatial_mass"
    required_completion_paths = (
        args.completion_base_report,
        args.completion_base_checkpoint,
        args.completion_slot_checkpoint,
        args.completion_mass_report,
    )
    if learned_completion and not all(required_completion_paths):
        raise ValueError("learned completion requires all frozen report/checkpoint paths")
    completion_feature_sum = torch.zeros(
        carrier.num_elements,
        67 if learned_completion else 0,
        dtype=torch.float32,
        device=device,
    )
    completion_feature_mass = torch.zeros(
        carrier.num_elements, dtype=torch.float32, device=device
    )
    completion_cameras: list[Camera] = []
    radio_projection = _radio_projection_matrix().to(device) if learned_completion else None
    retained_view_count = (
        args.surface_view_prototype_count
        if args.surface_semantic_evidence in ("view_prototype", "consistent_view") else 0
    )
    surface_view_descriptors = torch.zeros(
        carrier.num_elements, retained_view_count, feature_dimension,
        dtype=torch.float16, device=device,
    )
    surface_view_mass = torch.zeros(
        carrier.num_elements, retained_view_count,
        dtype=torch.float32, device=device,
    )
    evidences, visibilities, parents, descriptors = [], [], [], []
    proposal_count = 0
    for source_index, frame_id in enumerate(source_frames, start=1):
        print(f"building source scene state: {source_index}/{len(source_frames)} frame={frame_id}", flush=True)
        camera = _camera(views[frame_id], frame_id, height, width)
        cache_path = Path(sam_records[frame_id]["output"])
        masks = _masks(cache_path, height, width)
        positive, visible = _lift_masks(carrier, camera, masks, device)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        quality = torch.as_tensor(cache["quality"], dtype=torch.float32, device=device)
        evidences.append(ObservedObjectEvidence.from_positive_visibility(
            positive, visible,
            view_ids=torch.full((masks.shape[0],), frame_id, dtype=torch.long, device=device),
            quality=quality,
        ))
        visibilities.append(visible)
        parents.append(torch.as_tensor(cache["parent_index"], dtype=torch.long, device=device))
        dense = _load_dense_feature(
            feature_dir / f"rgb_{frame_id}.pt",
            expected_channels=input_feature_dimension,
            expected_height=height,
            expected_width=width,
        )
        raw_dense_device = dense.to(device).float()
        projection = carrier.project(camera)
        if learned_completion:
            normalized_backbone = F.normalize(
                raw_dense_device.permute(1, 2, 0), dim=-1, eps=1e-12
            )
            projected_radio = F.normalize(
                normalized_backbone @ radio_projection, dim=-1, eps=1e-12
            ).permute(2, 0, 1)
            source_rgb = _load_source_rgb(
                source_image_paths[frame_id], height, width
            ).to(device).permute(2, 0, 1)
            accumulate_surface_features(
                completion_feature_sum,
                completion_feature_mass,
                torch.cat((source_rgb, projected_radio), dim=0),
                projection,
                channel_chunk_size=args.semantic_channel_chunk_size,
            )
            completion_cameras.append(camera)
        if summary_head is not None:
            mask_backbone = _masked_descriptors(raw_dense_device, masks.to(device))
            pooled_backbone_descriptor = F.normalize(
                summary_head(mask_backbone[None])[0].float(), dim=-1, eps=1e-8
            )
            tokens = raw_dense_device.flatten(1).T[None]
            dense_device = summary_head(tokens)[0].T.reshape(1536, height, width)
        else:
            dense_device = raw_dense_device
            pooled_backbone_descriptor = _masked_descriptors(dense_device, masks.to(device))
        dense_device = F.normalize(dense_device.float(), dim=0, eps=1e-8)
        proposal_descriptor = (
            _masked_descriptors(dense_device, masks.to(device))
            if args.proposal_descriptor_mode == "mean_text_aligned_pixels"
            else pooled_backbone_descriptor
        )
        descriptors.append(proposal_descriptor)
        accumulate_surface_features(
            surface_feature_sum,
            surface_feature_mass,
            dense_device,
            projection,
            channel_chunk_size=args.semantic_channel_chunk_size,
        )
        if retained_view_count:
            update_surface_view_prototypes(
                surface_view_descriptors, surface_view_mass, dense_device,
                projection, channel_chunk_size=args.semantic_channel_chunk_size,
            )
        proposal_count += int(masks.shape[0])
    for semantic_index, frame_id in enumerate(semantic_extra_frames, start=1):
        print(
            f"expanding source semantic coverage: {semantic_index}/{len(semantic_extra_frames)} "
            f"frame={frame_id}", flush=True,
        )
        camera = _camera(views[frame_id], frame_id, height, width)
        dense = _load_dense_feature(
            feature_dir / f"rgb_{frame_id}.pt",
            expected_channels=input_feature_dimension,
            expected_height=height,
            expected_width=width,
        ).to(device).float()
        if summary_head is not None:
            dense = summary_head(dense.flatten(1).T[None])[0].T.reshape(
                1536, height, width
            )
        dense = F.normalize(dense.float(), dim=0, eps=1e-8)
        accumulate_surface_features(
            surface_feature_sum,
            surface_feature_mass,
            dense,
            carrier.project(camera),
            channel_chunk_size=args.semantic_channel_chunk_size,
        )
        if retained_view_count:
            update_surface_view_prototypes(
                surface_view_descriptors, surface_view_mass, dense,
                carrier.project(camera), channel_chunk_size=args.semantic_channel_chunk_size,
            )
    semantic_observed = surface_feature_mass > 0
    surface_average_descriptors = F.normalize(
        surface_feature_sum / surface_feature_mass[:, None].clamp_min(1e-8),
        dim=-1,
        eps=1e-8,
    )
    surface_descriptors_device = surface_average_descriptors
    semantic_reliability_mass = torch.zeros_like(surface_feature_mass)
    reliability_observation_count = 0
    reliability_positive_count = 0
    reliability_value_sum = 0.0
    if args.surface_semantic_evidence == "consistency_weighted":
        reliability_sum = torch.zeros_like(surface_feature_sum)
        semantic_frames = sorted(set(source_frames + semantic_extra_frames))
        for semantic_index, frame_id in enumerate(semantic_frames, start=1):
            print(
                f"building consistency-weighted semantics: {semantic_index}/"
                f"{len(semantic_frames)} frame={frame_id}", flush=True,
            )
            dense = _load_dense_feature(
                feature_dir / f"rgb_{frame_id}.pt",
                expected_channels=input_feature_dimension,
                expected_height=height,
                expected_width=width,
            ).to(device).float()
            dense = summary_head(dense.flatten(1).T[None])[0].T.reshape(
                feature_dimension, height, width
            )
            dense = F.normalize(dense.float(), dim=0, eps=1e-8)
            reliability, view_observed = accumulate_consistency_weighted_surface_features(
                reliability_sum, semantic_reliability_mass,
                surface_feature_sum, surface_feature_mass, dense,
                carrier.project(_camera(views[frame_id], frame_id, height, width)),
                agreement_floor=args.semantic_agreement_floor,
                agreement_power=args.semantic_agreement_power,
                channel_chunk_size=args.semantic_channel_chunk_size,
            )
            reliability_observation_count += int(view_observed.sum())
            reliability_positive_count += int((reliability > 0).sum())
            reliability_value_sum += float(reliability[view_observed].sum())
        reliability_observed = semantic_reliability_mass > 0
        reliability_descriptors = F.normalize(
            reliability_sum / semantic_reliability_mass[:, None].clamp_min(1e-8),
            dim=-1,
            eps=1e-8,
        )
        surface_descriptors_device = torch.where(
            reliability_observed[:, None], reliability_descriptors,
            surface_average_descriptors,
        )
    results = model.process_batch(
        evidences, element_visibilities=visibilities, parent_indices=parents,
        proposal_descriptors=descriptors,
    )
    assigned_proposals = sum(int((item.token_ids >= 0).sum()) for item in results)
    prototype_descriptors = torch.cat(descriptors, dim=0).detach().cpu()
    prototype_token_ids = torch.cat([item.token_ids for item in results], dim=0).detach().cpu()
    observed_membership = model.membership.detach()
    token_descriptors = model.descriptor_sum / model.descriptor_mass[:, None].clamp_min(1e-8)
    token_descriptors = F.normalize(token_descriptors, dim=-1).cpu()
    completion_audit: dict[str, Any]
    categorical_observed_membership = None
    if learned_completion:
        if carrier.normals is None:
            raise RuntimeError("learned completion requires carrier normals")
        completion_available = completion_feature_mass > 0
        completion_average = completion_feature_sum / completion_feature_mass[:, None].clamp_min(1e-8)
        completion_rgb = (
            completion_average[:, :3] * 2.0 - 1.0
        ) * completion_available[:, None]
        completion_radio = F.normalize(
            completion_average[:, 3:], dim=-1, eps=1e-12
        ) * completion_available[:, None]
        completion_local_features = torch.cat(
            (
                completion_rgb,
                completion_available.float()[:, None],
                completion_radio,
                carrier.normals.to(device).float(),
            ),
            dim=-1,
        ).cpu()
        completion_runtime, adapter_audit = build_real_token_runtime(
            carrier=carrier,
            local_features=completion_local_features,
            source_visible=completion_available.cpu(),
            observed_membership=observed_membership.cpu(),
            observation_cameras=completion_cameras,
            view_token_ids=[item.token_ids.cpu() for item in results],
            observed_threshold=args.observed_threshold,
        )
        categorical_active = completion_runtime["partial"].positive.float()
        completed_active, learned_audit = apply_scannet_spatial_mass_candidate(
            completion_runtime,
            base_report_path=args.completion_base_report,
            base_checkpoint_path=args.completion_base_checkpoint,
            slot_checkpoint_path=args.completion_slot_checkpoint,
            mass_report_path=args.completion_mass_report,
            device=device,
            unary_element_batch_size=args.completion_unary_element_batch_size,
            inference_element_batch_size=args.completion_inference_element_batch_size,
            projection_iteration_count=args.completion_projection_iteration_count,
            projection_damping=args.completion_projection_damping,
        )
        active_token_ids = completion_runtime["active_token_ids"].long()
        categorical_observed_membership = torch.zeros_like(observed_membership).cpu()
        categorical_observed_membership[:, active_token_ids] = categorical_active
        # Completion is defined only for tokens with a real categorical seed.
        # Unseeded semantic tokens retain their source association evidence but
        # receive no synthesized support.
        completed_membership = observed_membership.float().cpu().clone()
        completed_membership[:, active_token_ids] = completed_active
        observed_categorical = categorical_observed_membership.max(-1).values > 0
        completion_mask = (
            completed_membership.max(-1).values > 0
        ) & ~observed_categorical
        completion_audit = {"adapter": adapter_audit, "inference": learned_audit}
    else:
        completed_membership, completion_mask = conservative_token_geometry_completion(
            model.element_centres, observed_membership,
            observed_threshold=args.observed_threshold,
            radius_multiplier=args.completion_radius_multiplier,
            completion_weight=args.completion_weight,
            minimum_scale=args.minimum_completion_scale,
        )
        completion_audit = {"method": "legacy_conservative_token_geometry"}
    surface_average_descriptors_cpu = surface_average_descriptors.half().cpu()
    surface_descriptors = surface_descriptors_device.half().cpu()
    surface_view_descriptors_cpu = surface_view_descriptors.cpu()
    surface_view_mass_cpu = surface_view_mass.cpu()
    scene_state = {
        "schema": "radio_gs.surface_object_memory_v4.development_scene_state.v1",
        "development_only": True,
        "scene_label": args.scene_label,
        "centres": carrier.centres,
        "normals": carrier.normals,
        "confidence": carrier.confidence,
        "voxel_size": carrier.voxel_size,
        "observed_membership": observed_membership.cpu(),
        "completed_membership": completed_membership.cpu(),
        "completion_mask": completion_mask.cpu(),
        "categorical_observed_membership": (
            categorical_observed_membership.cpu()
            if categorical_observed_membership is not None else None
        ),
        "completion_audit": completion_audit,
        "token_descriptors": token_descriptors,
        "prototype_descriptors": prototype_descriptors,
        "prototype_token_ids": prototype_token_ids,
        "surface_descriptors": surface_descriptors,
        "surface_average_descriptors": surface_average_descriptors_cpu,
        "semantic_observed": semantic_observed.cpu(),
        "semantic_reliability_mass": semantic_reliability_mass.cpu(),
        "surface_view_descriptors": surface_view_descriptors_cpu,
        "surface_view_mass": surface_view_mass_cpu,
        "source_frames": source_frames,
        "semantic_source_frames": sorted(source_frames + semantic_extra_frames),
        "source_input_digests": {
            "geometry_authority": geometry_binding.authority_sha256,
            "source_rgb_authority": sha256_file(authority_path),
            "surface_carrier": sha256_file(surface_path),
            "semantic_projector_checkpoint": (
                sha256_file(Path(args.summary_head_weights or args.radio_checkpoint).resolve(strict=True))
                if (args.summary_head_weights or args.radio_checkpoint) else None
            ),
            "semantic_frame_manifest": (
                sha256_file(semantic_manifest_path)
            ),
            **(
                {
                    "completion_base_report": sha256_file(
                        Path(args.completion_base_report).resolve(strict=True)
                    ),
                    "completion_base_checkpoint": sha256_file(
                        Path(args.completion_base_checkpoint).resolve(strict=True)
                    ),
                    "completion_slot_checkpoint": sha256_file(
                        Path(args.completion_slot_checkpoint).resolve(strict=True)
                    ),
                    "completion_mass_report": sha256_file(
                        Path(args.completion_mass_report).resolve(strict=True)
                    ),
                }
                if learned_completion else {}
            ),
            **{f"sam_manifest_{index}": sha256_file(path) for index, path in enumerate(sam_paths)},
        },
        "method_configuration": {
            "carrier": {
                "voxel_size": geometry_binding.configuration.voxel_size,
                "maximum_splat_radius": geometry_binding.configuration.maximum_splat_radius,
                "surface_band_voxels": geometry_binding.configuration.surface_band_voxels,
                "maximum_contributors_per_pixel": (
                    geometry_binding.configuration.maximum_contributors_per_pixel
                ),
                "camera_convention": geometry_binding.configuration.camera_convention,
            },
            **{
                key: getattr(args, key) for key in (
                    "minimum_overlap", "association_null_logit", "association_temperature",
                    "geometry_weight", "appearance_weight", "batch_birth_overlap",
                    "completion_radius_multiplier", "completion_weight", "observed_threshold",
                    "completion_mode",
                    "semantic_channel_chunk_size",
                    "semantic_feature_space",
                    "proposal_descriptor_mode",
                    "surface_view_prototype_count",
                    "semantic_agreement_floor", "semantic_agreement_power",
                )
            },
        },
        "information_policy": {
            "target_rgb_opened_during_construction": False,
            "benchmark_labels_opened_during_construction": False,
            "text_queries_opened_during_construction": False,
            "semantic_manifest_excluded_frames_respected": True,
        },
    }
    state_path = Path(args.scene_state_output).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scene_state, state_path)

    # Evaluation boundary: benchmark labels and authorized text queries open only
    # after the persistent, query-independent scene state has been serialized.
    annotations, categories, source_height, source_width = _load_labels(
        Path(args.label_root).resolve(strict=True), args.scene_label
    )
    query_cache_path = Path(args.text_query_cache).resolve(strict=True)
    negative_cache_path = Path(args.negative_text_query_cache).resolve(strict=True)
    query_text = _load_text_cache(query_cache_path, categories, device)
    negative_text = _load_text_cache(
        negative_cache_path, list(GENERIC_NEGATIVES), device
    )
    # A generic-negative similarity supplies a query-independent rejection
    # reference while the final normalization retains one explicit null state.
    if args.semantic_pooling in ("surface_local_peak", "hybrid_surface_prototype"):
        if args.surface_semantic_evidence == "view_prototype":
            local_identity_probability = surface_view_prototype_query_posterior(
                surface_view_descriptors_cpu.to(device), surface_view_mass_cpu.to(device),
                query_text, negative_text, temperature=args.query_temperature,
                null_similarity=args.null_similarity,
            )
            identity_observed = surface_view_mass_cpu.gt(0).any(-1).to(device)
        elif args.surface_semantic_evidence == "consistent_view":
            consistent_descriptor, identity_observed = select_consistent_surface_view(
                surface_view_descriptors_cpu.to(device), surface_view_mass_cpu.to(device),
                surface_descriptors.to(device),
            )
            local_identity_probability = surface_local_query_posterior(
                consistent_descriptor, identity_observed, query_text, negative_text,
                temperature=args.query_temperature,
                null_similarity=args.null_similarity,
            )
        else:
            local_identity_probability = surface_local_query_posterior(
                surface_descriptors.to(device), semantic_observed.to(device),
                query_text, negative_text, temperature=args.query_temperature,
                null_similarity=args.null_similarity,
            )
            identity_observed = semantic_observed.to(device)
        surface_token_probability = pool_surface_identity_by_token(
            local_identity_probability, identity_observed, observed_membership,
            membership_threshold=args.identity_membership_threshold,
            peak_count=args.identity_peak_count,
        )
        if args.semantic_pooling == "hybrid_surface_prototype":
            prototype_token_probability, _ = prototype_max_token_posterior(
                prototype_descriptors.to(device), prototype_token_ids.to(device),
                query_text, negative_text, num_tokens=model.num_tokens,
                temperature=args.query_temperature, null_similarity=args.null_similarity,
            )
            token_probability = 0.5 * (
                surface_token_probability + prototype_token_probability
            )
        else:
            token_probability = surface_token_probability
        null_probability = 1.0 - token_probability
        identity_element_probability = local_identity_probability.T.cpu()
    else:
        token_probability, null_probability = prototype_max_token_posterior(
            prototype_descriptors.to(device), prototype_token_ids.to(device), query_text, negative_text,
            num_tokens=model.num_tokens,
            temperature=args.query_temperature,
            null_similarity=args.null_similarity,
        )
        identity_element_probability = None
    token_probability = retain_top_query_tokens(token_probability, args.maximum_query_tokens)
    null_probability = 1.0 - token_probability
    token_probability_cpu = token_probability.cpu()
    element_probability = torch.stack([
        (completed_membership.cpu() * probability[None]).max(-1).values
        for probability in token_probability_cpu
    ], dim=-1)

    per_observation, sums = [], {"2d": [0, 0], "3d": [0, 0], "identity": [0, 0]}
    category_ious: dict[str, dict[str, list[float]]] = {
        category: {"2d": [], "3d": []} for category in categories
    }
    for frame_id, objects in sorted(annotations.items()):
        camera = _camera(views[frame_id], frame_id, height, width)
        gt = _ground_truth_masks(objects, categories, height, width, source_height, source_width)
        for query_index, category in enumerate(categories):
            if not bool(gt[category].any()):
                continue
            continuous = carrier.render_posterior(element_probability[:, query_index], camera)
            selected_elements = element_probability[:, query_index] >= args.element_threshold
            discrete = carrier.render_posterior(selected_elements.float(), camera)
            pred_2d = continuous >= args.pixel_threshold
            pred_3d = discrete >= args.pixel_threshold
            iou_2d, intersection_2d, union_2d = _binary_iou(pred_2d, gt[category])
            iou_3d, intersection_3d, union_3d = _binary_iou(pred_3d, gt[category])
            identity_iou = None
            identity_peak_in_gt = None
            if identity_element_probability is not None:
                identity_render = carrier.render_posterior(
                    identity_element_probability[:, query_index], camera
                )
                identity_prediction = identity_render >= args.identity_pixel_threshold
                identity_iou, identity_intersection, identity_union = _binary_iou(
                    identity_prediction, gt[category]
                )
                sums["identity"][0] += identity_intersection
                sums["identity"][1] += identity_union
                if bool(torch.isfinite(identity_render).all()) and float(identity_render.max()) > 0:
                    peak = int(identity_render.argmax())
                    identity_peak_in_gt = bool(gt[category].flatten()[peak])
            sums["2d"][0] += intersection_2d; sums["2d"][1] += union_2d
            sums["3d"][0] += intersection_3d; sums["3d"][1] += union_3d
            category_ious[category]["2d"].append(iou_2d)
            category_ious[category]["3d"].append(iou_3d)
            per_observation.append({
                "frame_id": frame_id, "category": category,
                "iou_2d": iou_2d, "iou_3d": iou_3d,
                "identity_only_iou": identity_iou,
                "identity_peak_in_gt": identity_peak_in_gt,
                "selected_element_fraction": float(selected_elements.float().mean()),
            })
    valid_identity_peaks = [
        value["identity_peak_in_gt"] for value in per_observation
        if value["identity_peak_in_gt"] is not None
    ]
    category_metrics = {
        category: {
            "mean_iou_2d": float(np.mean(values["2d"])) if values["2d"] else None,
            "mean_iou_3d": float(np.mean(values["3d"])) if values["3d"] else None,
            "observation_count": len(values["2d"]),
            "best_token_null_probability": float(null_probability[index].min()),
            "best_token_positive_probability": float(token_probability[index].max()),
        }
        for index, (category, values) in enumerate(category_ious.items())
    }
    valid = [value for value in category_metrics.values() if value["observation_count"]]
    report = {
        "schema": "radio_gs.surface_object_memory_v4.lerf_development_evaluation.v1",
        "development_only": True,
        "evaluation_contract_scope": (
            "development 2D polygon proxy for continuous and element-thresholded "
            "surface posterior renderings"
        ),
        "scene_label": args.scene_label,
        "flow_completed": True,
        "source_view_count": len(source_frames),
        "semantic_source_view_count": len(source_frames) + len(semantic_extra_frames),
        "proposal_count": proposal_count,
        "assigned_proposal_count": assigned_proposals,
        "token_count": model.num_tokens,
        "observed_element_fraction": float((observed_membership.max(-1).values > args.observed_threshold).float().mean()),
        "completion_added_fraction": float(completion_mask.float().mean()),
        "post_completion_element_fraction": float((completed_membership.max(-1).values > 0).float().mean()),
        "semantic_observed_element_fraction": float(semantic_observed.float().mean()),
        "semantic_reliability_observed_element_fraction": (
            float((semantic_reliability_mass > 0).float().mean())
            if args.surface_semantic_evidence == "consistency_weighted" else None
        ),
        "semantic_reliability_positive_observation_fraction": (
            reliability_positive_count / max(reliability_observation_count, 1)
            if args.surface_semantic_evidence == "consistency_weighted" else None
        ),
        "semantic_reliability_mean": (
            reliability_value_sum / max(reliability_observation_count, 1)
            if args.surface_semantic_evidence == "consistency_weighted" else None
        ),
        "raster_shape": [height, width],
        "query_configuration": {
            "maximum_query_tokens": args.maximum_query_tokens,
            "temperature": args.query_temperature,
            "element_threshold": args.element_threshold,
            "pixel_threshold": args.pixel_threshold,
            "development_text_query_cache": str(query_cache_path) if query_cache_path else None,
            "development_text_query_cache_sha256": sha256_file(query_cache_path) if query_cache_path else None,
            "development_negative_text_cache": (
                str(negative_cache_path) if negative_cache_path else None
            ),
            "development_negative_text_cache_sha256": (
                sha256_file(negative_cache_path) if negative_cache_path else None
            ),
            "token_semantic_pooling": args.semantic_pooling,
            "identity_membership_threshold": args.identity_membership_threshold,
            "identity_peak_count": args.identity_peak_count,
            "identity_pixel_threshold": args.identity_pixel_threshold,
            "semantic_feature_space": args.semantic_feature_space,
            "proposal_descriptor_mode": args.proposal_descriptor_mode,
            "surface_semantic_evidence": args.surface_semantic_evidence,
            "surface_view_prototype_count": args.surface_view_prototype_count,
            "semantic_agreement_floor": args.semantic_agreement_floor,
            "semantic_agreement_power": args.semantic_agreement_power,
        },
        "metrics": {
            "macro_category_miou_2d": float(np.mean([value["mean_iou_2d"] for value in valid])),
            "macro_category_miou_3d": float(np.mean([value["mean_iou_3d"] for value in valid])),
            "micro_iou_2d": sums["2d"][0] / max(sums["2d"][1], 1),
            "micro_iou_3d": sums["3d"][0] / max(sums["3d"][1], 1),
            "identity_only_micro_iou": (
                sums["identity"][0] / max(sums["identity"][1], 1)
                if identity_element_probability is not None else None
            ),
            "identity_peak_localization_accuracy": float(np.mean([
                value for value in valid_identity_peaks
            ])) if valid_identity_peaks else None,
            "identity_peak_evaluated_count": len(valid_identity_peaks),
        },
        "category_metrics": category_metrics,
        "per_observation": per_observation,
        "scene_state": str(state_path),
        "scene_state_sha256": sha256_file(state_path),
        "completion_method": args.completion_mode,
        "completion_audit": completion_audit,
        "relaxed_gate_policy": {
            "gates_block_execution": False,
            "promotion_claimed": False,
            "warnings": [
                "object organization and completion are development baselines",
                "fixed thresholds are provisional and were not benchmark-tuned",
                "full-cohort confirmation remains pending",
                "pointwise summary-head projection is a development approximation",
            ],
        },
        "information_policy": scene_state["information_policy"],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--source-rgb-authority", required=True)
    parser.add_argument("--sam-manifest", action="append", required=True)
    parser.add_argument("--surface-carrier", required=True)
    parser.add_argument("--geometry-authority", required=True)
    parser.add_argument("--siglip-feature-dir", required=True)
    parser.add_argument("--semantic-frame-manifest")
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--scene-state-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--allow-deprecated-development-proxy",
        action="store_true",
        help="explicitly opt into the quarantined non-promotion polygon proxy",
    )
    parser.add_argument("--minimum-overlap", type=float, default=0.10)
    parser.add_argument("--association-null-logit", type=float, default=0.25)
    parser.add_argument("--association-temperature", type=float, default=0.10)
    parser.add_argument("--geometry-weight", type=float, default=0.25)
    parser.add_argument("--appearance-weight", type=float, default=0.10)
    parser.add_argument("--batch-birth-overlap", type=float, default=0.20)
    parser.add_argument("--observed-threshold", type=float, default=1e-5)
    parser.add_argument("--completion-radius-multiplier", type=float, default=1.5)
    parser.add_argument("--completion-weight", type=float, default=0.25)
    parser.add_argument("--minimum-completion-scale", type=float, default=0.04)
    parser.add_argument(
        "--completion-mode",
        choices=("legacy_conservative_geometry", "scannet_spatial_mass"),
        default="legacy_conservative_geometry",
    )
    parser.add_argument("--completion-base-report")
    parser.add_argument("--completion-base-checkpoint")
    parser.add_argument("--completion-slot-checkpoint")
    parser.add_argument("--completion-mass-report")
    parser.add_argument("--completion-unary-element-batch-size", type=int, default=512)
    parser.add_argument("--completion-inference-element-batch-size", type=int, default=1024)
    parser.add_argument("--completion-projection-iteration-count", type=int, default=256)
    parser.add_argument("--completion-projection-damping", type=float, default=0.5)
    parser.add_argument(
        "--semantic-feature-space",
        choices=("radio_backbone_summary_head",),
        default="radio_backbone_summary_head",
    )
    parser.add_argument("--radio-checkpoint")
    parser.add_argument("--summary-head-weights")
    parser.add_argument(
        "--proposal-descriptor-mode",
        choices=("mean_text_aligned_pixels", "pooled_backbone_summary"),
        default="mean_text_aligned_pixels",
    )
    parser.add_argument("--semantic-channel-chunk-size", type=int, default=192)
    parser.add_argument(
        "--semantic-pooling",
        choices=("surface_local_peak", "prototype_max", "hybrid_surface_prototype"),
        default="surface_local_peak",
    )
    parser.add_argument("--identity-membership-threshold", type=float, default=0.05)
    parser.add_argument("--identity-peak-count", type=int, default=8)
    parser.add_argument(
        "--surface-semantic-evidence",
        choices=("average", "view_prototype", "consistent_view", "consistency_weighted"),
        default="average",
    )
    parser.add_argument("--surface-view-prototype-count", type=int, default=2)
    parser.add_argument("--semantic-agreement-floor", type=float, default=0.3)
    parser.add_argument("--semantic-agreement-power", type=float, default=2.0)
    parser.add_argument("--identity-pixel-threshold", type=float, default=0.5)
    parser.add_argument("--query-temperature", type=float, default=0.03)
    parser.add_argument("--text-query-cache", required=True)
    parser.add_argument("--negative-text-query-cache", required=True)
    parser.add_argument("--maximum-query-tokens", type=int, default=3)
    parser.add_argument("--null-similarity", type=float, default=0.0)
    parser.add_argument("--element-threshold", type=float, default=0.2)
    parser.add_argument("--pixel-threshold", type=float, default=0.2)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"flow_completed": report["flow_completed"], **report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
