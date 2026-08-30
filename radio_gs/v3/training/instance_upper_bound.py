"""Faithful source-mask upper-bound experiment for SUGM-v3 Arm A/B.

The runner consumes immutable source SAM caches, exact front-to-back
contribution shards, exact-MPR initial supports, and ternary proposal
relations. It never opens benchmark RGB, masks, queries, or metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.v3.query.membership import membership_from_prototype, pool_prototype
from radio_gs.v3.training.rendered_mask import render_membership, rendered_mask_loss


@dataclass(frozen=True)
class MaskEpisode:
    proposal_index: int
    view_index: int
    gaussian_ids: torch.Tensor
    pixel_ids: torch.Tensor
    contribution_weights: torch.Tensor
    target: torch.Tensor
    known: torch.Tensor
    boundary: torch.Tensor
    unknown: torch.Tensor
    scale: float
    different_proposals: tuple[int, ...] = ()


def unpack_masks(packed: torch.Tensor, width: int) -> torch.Tensor:
    values = np.unpackbits(
        torch.as_tensor(packed).cpu().numpy(), axis=-1, bitorder="little"
    )[..., : int(width)]
    return torch.from_numpy(values.astype(np.bool_))


def align_masks(masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(
        torch.as_tensor(masks).float().unsqueeze(1),
        size=(int(height), int(width)),
        mode="nearest",
    )[:, 0].bool()


def mask_boundary(mask: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(mask).float()[None, None]
    dilated = F.max_pool2d(value, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-value, 3, stride=1, padding=1)
    return (dilated != eroded)[0, 0]


def proposal_supports(
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    num_proposals: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    rows = torch.as_tensor(row_indices).long()
    proposals = torch.as_tensor(proposal_indices).long()
    mass = torch.as_tensor(weights).float()
    output: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index in range(int(num_proposals)):
        selected = proposals == index
        output.append((rows[selected], mass[selected]))
    return tuple(output)


def build_known_pixel_authority(
    masks: torch.Tensor,
    local_proposal: int,
    global_proposal: int,
    global_indices: torch.Tensor,
    different_left: torch.Tensor,
    different_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask interior plus explicitly mutually-exclusive masks are known."""

    target = masks[int(local_proposal)]
    global_to_local = {int(value): i for i, value in enumerate(global_indices.tolist())}
    peers = [
        global_to_local[value]
        for value in same_view_different_peers(masks, local_proposal, global_indices)
    ]
    for left, right in zip(different_left.tolist(), different_right.tolist()):
        if left == global_proposal and right in global_to_local:
            peers.append(global_to_local[right])
        elif right == global_proposal and left in global_to_local:
            peers.append(global_to_local[left])
    peers = sorted(set(peers))
    known = target.clone()
    if peers:
        known |= masks[torch.tensor(peers)].any(0)
    return target, known


def same_view_different_peers(
    masks: torch.Tensor,
    local_proposal: int,
    global_indices: torch.Tensor,
    *,
    minimum_area_ratio: float = 0.25,
) -> tuple[int, ...]:
    """Return global indices of comparable, pixel-disjoint source masks."""

    values = torch.as_tensor(masks).bool()
    globals_for_view = torch.as_tensor(global_indices).long().reshape(-1)
    if values.ndim != 3 or globals_for_view.shape != (values.shape[0],):
        raise ValueError("same-view mask and proposal axes differ")
    if not 0 <= local_proposal < values.shape[0] or not 0 < minimum_area_ratio <= 1:
        raise ValueError("same-view proposal index or area ratio differs")
    target = values[int(local_proposal)]
    target_area = int(target.sum())
    peers: list[int] = []
    for local, candidate in enumerate(values):
        if local == int(local_proposal):
            continue
        area = int(candidate.sum())
        ratio = min(target_area, area) / max(target_area, area, 1)
        if ratio >= float(minimum_area_ratio) and not bool((target & candidate).any()):
            peers.append(int(globals_for_view[local]))
    return tuple(peers)


def render_episode(
    embedding: torch.Tensor,
    support: tuple[torch.Tensor, torch.Tensor],
    episode: MaskEpisode,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, weights = support
    prototype = pool_prototype(embedding[rows], weights.to(embedding.device))
    hit_embedding = embedding[episode.gaussian_ids.to(embedding.device)]
    hit_posterior = membership_from_prototype(
        hit_embedding, prototype, temperature=temperature
    )
    # render_membership accepts a Gaussian-domain vector. Here the hit domain
    # is deliberately compacted, preserving exactly the same values and hits.
    prediction = render_membership(
        hit_posterior,
        torch.arange(hit_posterior.numel(), device=embedding.device),
        episode.pixel_ids.to(embedding.device),
        episode.contribution_weights.to(embedding.device),
        num_pixels=episode.target.numel(),
    )
    return prediction, prototype


def episode_objective(
    embedding: torch.Tensor,
    support: tuple[torch.Tensor, torch.Tensor],
    episode: MaskEpisode,
    *,
    temperature: float,
    unknown_growth_weight: float = 0.25,
) -> torch.Tensor:
    prediction, _ = render_episode(
        embedding, support, episode, temperature=temperature
    )
    target = episode.target.to(embedding.device).float()
    known = episode.known.to(embedding.device)
    boundary = episode.boundary.to(embedding.device).float()
    # A morphology-derived confidence is used only as a differentiable edge
    # prediction; the persistent boundary head is introduced in Arm C.
    height, width = episode.target.shape
    image = prediction.reshape(height, width)
    gradient = (
        F.max_pool2d(image[None, None], 3, 1, 1)
        + F.max_pool2d(-image[None, None], 3, 1, 1)
    )[0, 0]
    supervised = rendered_mask_loss(
        prediction,
        target.flatten(),
        known=known.flatten(),
        boundary_target=boundary.flatten(),
        boundary_prediction=(gradient * 16.0 - 4.0).flatten(),
    ).total
    # Unknown is not background. This one-sided restraint merely prevents the
    # unobserved posterior from growing above its neutral 0.5 prior.
    unknown = episode.unknown.to(embedding.device).flatten()
    growth = (
        F.relu(prediction[unknown] - 0.5).square().mean()
        if bool(unknown.any()) else prediction.new_zeros(())
    )
    return supervised + float(unknown_growth_weight) * growth


def relation_contrastive_loss(
    embedding: torch.Tensor,
    supports: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    edge_relation: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Proper same/different prototype loss; unknown edges are excluded."""

    left = torch.as_tensor(edge_left).long().reshape(-1)
    right = torch.as_tensor(edge_right).long().reshape(-1)
    relation = torch.as_tensor(edge_relation).to(torch.int8).reshape(-1)
    if not (left.shape == right.shape == relation.shape) or temperature <= 0:
        raise ValueError("relation contrastive axes or temperature differ")
    known = (relation == 0) | (relation == 1)
    if bool((~known).any()):
        left, right, relation = left[known], right[known], relation[known]
    if not left.numel() or not bool((relation == 0).any()) or not bool((relation == 1).any()):
        raise ValueError("relation contrastive batch requires same and different")
    left_prototypes = torch.stack([
        pool_prototype(embedding[supports[index][0]], supports[index][1].to(embedding.device))
        for index in left.tolist()
    ])
    right_prototypes = torch.stack([
        pool_prototype(embedding[supports[index][0]], supports[index][1].to(embedding.device))
        for index in right.tolist()
    ])
    cosine = (left_prototypes * right_prototypes).sum(-1)
    # Zero cosine is the fixed midpoint: same codes move positive and
    # explicitly mutually-exclusive codes move negative.
    logits = cosine / float(temperature)
    return F.binary_cross_entropy_with_logits(logits, relation.to(embedding.device).float())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_only_inputs(
    membership: Mapping[str, object], relation: Mapping[str, object]
) -> None:
    metadata = membership.get("metadata")
    relation_metadata = relation.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(relation_metadata, Mapping):
        raise ValueError("upper-bound input metadata is absent")
    forbidden = ("benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened")
    if any(metadata.get(name) is not False for name in forbidden):
        raise ValueError("membership input is not source-only")
    native_language_v3 = (
        relation.get("schema") == "radio_gs.sugm_v3.native_language_authority.v3"
    )
    relation_forbidden = (
        ("benchmark_images_opened", "benchmark_masks_opened", "benchmark_metrics_opened")
        if native_language_v3
        else ("benchmark_masks_opened", "evaluation_rgb_opened")
    )
    if (
        relation_metadata.get("source_only") is not True
        or any(relation_metadata.get(name) is not False for name in relation_forbidden)
        or (
            native_language_v3
            and (
                relation_metadata.get("historical_field_opened") is not False
                or relation_metadata.get(
                    "dev_and_audit_text_scores_used_for_label_selection"
                )
                is not False
            )
        )
    ):
        raise ValueError("relation input is not source-only")
    if torch.as_tensor(relation["edge_relation"]).to(torch.int8).min() != -1:
        raise ValueError("relation authority does not preserve unknown")


class FrozenProjectionArm(nn.Module):
    """Arm A: one global D512-to-D32 projection, latent strictly frozen."""

    deployment_eligible = True

    def __init__(self, latent: torch.Tensor, output_dim: int = 32) -> None:
        super().__init__()
        value = torch.as_tensor(latent).detach().float()
        if value.ndim != 2 or value.shape[1] != 512:
            raise ValueError("Arm A requires a frozen D512 latent")
        # The field is hash-bound by the runner and must remain the single
        # Gaussian-indexed D512 authority.  Keeping this non-persistent avoids
        # silently serializing a second full latent table into the head-only
        # development checkpoint.
        self.register_buffer("latent", value, persistent=False)
        self.projection = nn.Linear(512, int(output_dim), bias=False)
        self.scale_adapter = nn.Linear(2, 2 * int(output_dim))

    def forward(self, scale: float = 0.5) -> torch.Tensor:
        phase = self.latent.new_tensor([float(scale)]).clamp(0, 1) * torch.pi
        gamma, beta = self.scale_adapter(torch.cat((phase.sin(), phase.cos()))).chunk(2)
        value = self.projection(self.latent) * (1 + 0.1 * gamma.tanh()) + 0.1 * beta
        return F.normalize(value, dim=-1, eps=1e-8)


class ExtraCodeArm(nn.Module):
    """Arm B: explicit temporary D16 per-Gaussian upper bound."""

    deployment_eligible = False

    def __init__(self, num_gaussians: int, output_dim: int = 16, seed: int = 20260826) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.code = nn.Parameter(torch.randn(num_gaussians, output_dim, generator=generator) / output_dim**0.5)
        self.scale_adapter = nn.Linear(2, 2 * int(output_dim))

    def forward(self, scale: float = 0.5) -> torch.Tensor:
        phase = self.code.new_tensor([float(scale)]).clamp(0, 1) * torch.pi
        gamma, beta = self.scale_adapter(torch.cat((phase.sin(), phase.cos()))).chunk(2)
        value = self.code * (1 + 0.1 * gamma.tanh()) + 0.1 * beta
        return F.normalize(value, dim=-1, eps=1e-8)


__all__ = [
    "ExtraCodeArm",
    "FrozenProjectionArm",
    "MaskEpisode",
    "align_masks",
    "build_known_pixel_authority",
    "episode_objective",
    "mask_boundary",
    "proposal_supports",
    "relation_contrastive_loss",
    "render_episode",
    "same_view_different_peers",
    "unpack_masks",
    "validate_source_only_inputs",
]
