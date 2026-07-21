"""Query-free, scale-ordered SAM region supervision for local 3-D edges.

The teacher does not collapse nested SAM masks into a binary same/different
label.  A mask with physical radius ``r`` containing both endpoints supplies
an upper-bound constraint ``mu_ij <= log(r)``; a confident separating mask
supplies ``mu_ij > log(r)``.  Accumulating both kinds of evidence across
views preserves part/object/context hierarchy and leaves conflicts visible.
"""

from __future__ import annotations

from typing import Iterable

import torch


def logarithmic_scale_bin_edges(
    *,
    minimum_radius_m: float = 0.05,
    maximum_radius_m: float = 4.0,
    bins: int = 8,
) -> torch.Tensor:
    if not 0.0 < float(minimum_radius_m) < float(maximum_radius_m):
        raise ValueError("scale-bin radii must satisfy 0 < minimum < maximum")
    if int(bins) <= 0:
        raise ValueError("scale-bin count must be positive")
    return torch.linspace(
        torch.log(torch.tensor(float(minimum_radius_m))),
        torch.log(torch.tensor(float(maximum_radius_m))),
        int(bins) + 1,
        dtype=torch.float32,
    )


def robust_mask_physical_radius(
    xyz: torch.Tensor,
    membership: torch.Tensor,
    *,
    inside_threshold: float = 0.80,
    minimum_primitives: int = 3,
    device: str | torch.device | None = None,
) -> float:
    """Return ``Q0.90 ||X - coordinatewise_median(X)||`` in metres.

    A coordinate-wise median is a deterministic robust medoid surrogate: it
    has no image-scale dependence and avoids the quadratic cost of an exact
    discrete medoid on a dense mask projection.
    """

    compute_device = torch.device(device) if device is not None else torch.as_tensor(xyz).device
    points = torch.as_tensor(xyz).float().to(compute_device)
    probability = torch.as_tensor(membership).float().to(compute_device).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or probability.shape != (len(points),):
        raise ValueError("xyz and membership must align as [N,3] and [N]")
    selected = points[probability >= float(inside_threshold)]
    if len(selected) < int(minimum_primitives):
        return float("nan")
    centre = selected.median(dim=0).values
    radius = torch.linalg.vector_norm(selected - centre, dim=-1).quantile(0.90)
    return float(radius.clamp_min(1e-4))


def accumulate_scale_ordered_votes(
    memberships: Iterable[torch.Tensor],
    observations: Iterable[torch.Tensor],
    radii_m: Iterable[torch.Tensor],
    edge_index: torch.Tensor,
    scale_bin_edges_log: torch.Tensor,
    *,
    quality: Iterable[torch.Tensor | None] | None = None,
    stability: Iterable[torch.Tensor | None] | None = None,
    inside_threshold: float = 0.80,
    outside_threshold: float = 0.20,
    mask_chunk: int = 4,
    include_same: bool = True,
    include_separate: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Accumulate independent soft same/separate evidence in physical bins.

    ``memberships`` contains all masks for every view as soft primitive
    responsibilities ``[M,N]``.  Each corresponding observation tensor may
    be either shared by that view (``[N]``) or supplied per mask (``[M,N]``).
    The latter is needed for a batched set of source-target virtual tracks,
    whose source observations legitimately differ even when their target view
    is the same.  Centre projection is represented as exact 0/1 values only
    for compatibility diagnostics; it is deliberately not converted to a
    smallest-mask assignment.  Same and separate votes remain separate even
    if they arise from the same edge in different views.
    """

    # Keep caller tensors on CPU while validating the public contract, then
    # move only one view/track batch at a time to the requested compute device.
    # This makes 1M-edge relation accumulation practical on a spare GPU without
    # changing the stored cache precision or semantics.
    compute_device = torch.device(device)
    membership_rows = [torch.as_tensor(value).float().cpu() for value in memberships]
    observation_rows = [torch.as_tensor(value).bool().cpu() for value in observations]
    radius_rows = [torch.as_tensor(value).float().cpu().reshape(-1) for value in radii_m]
    if not (len(membership_rows) == len(observation_rows) == len(radius_rows)):
        raise ValueError("memberships, observations, and radii_m must have equal view counts")
    edge = torch.as_tensor(edge_index).long().cpu()
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("edge_index must be [2,E]")
    bins = torch.as_tensor(scale_bin_edges_log).float().cpu().reshape(-1)
    if bins.numel() < 2 or not bool(torch.isfinite(bins).all()) or not bool((bins[1:] > bins[:-1]).all()):
        raise ValueError("scale_bin_edges_log must be strictly increasing")
    if not 0.0 <= float(outside_threshold) < float(inside_threshold) <= 1.0:
        raise ValueError("relation membership thresholds must satisfy 0 <= out < in <= 1")
    if int(mask_chunk) <= 0:
        raise ValueError("mask_chunk must be positive")
    if not bool(include_same) and not bool(include_separate):
        raise ValueError("at least one relation evidence direction must be included")
    edge_count, bin_count = edge.shape[1], bins.numel() - 1
    compute_edge = edge.to(compute_device)
    compute_bins = bins.to(compute_device)
    same_votes = torch.zeros(edge_count, bin_count, dtype=torch.float32, device=compute_device)
    separate_votes = torch.zeros_like(same_votes)
    observed_votes = torch.zeros_like(same_votes)
    same_events = torch.zeros(bin_count, dtype=torch.int64)
    separate_events = torch.zeros(bin_count, dtype=torch.int64)

    # Per-view mask counts differ, so normalize optional metadata inside the
    # loop rather than forcing an artificial rectangular representation.
    quality_rows = list(quality) if quality is not None else [None] * len(membership_rows)
    stability_rows = list(stability) if stability is not None else [None] * len(membership_rows)
    if len(quality_rows) != len(membership_rows) or len(stability_rows) != len(membership_rows):
        raise ValueError("quality/stability must have one entry per view")

    src, dst = compute_edge
    for member, observed, radius, row_quality, row_stability in zip(
        membership_rows, observation_rows, radius_rows, quality_rows, stability_rows
    ):
        if member.ndim != 2 or radius.shape != (member.shape[0],):
            raise ValueError("per-view scale relation inputs do not align")
        if observed.ndim == 1:
            if observed.shape != (member.shape[1],):
                raise ValueError("shared per-view observation rows do not align")
            shared_observation = True
        elif observed.ndim == 2:
            if observed.shape != member.shape:
                raise ValueError("per-mask observation rows do not align")
            shared_observation = False
        else:
            raise ValueError("observations must be [N] or [M,N]")
        if not bool(torch.isfinite(member).all()) or not bool((member >= 0).all()) or not bool((member <= 1).all()):
            raise ValueError("soft memberships must be finite probabilities in [0,1]")
        if row_quality is None:
            row_quality = torch.ones(member.shape[0], dtype=torch.float32)
        else:
            row_quality = torch.as_tensor(row_quality).float().cpu().reshape(-1)
        if row_stability is None:
            row_stability = torch.ones(member.shape[0], dtype=torch.float32)
        else:
            row_stability = torch.as_tensor(row_stability).float().cpu().reshape(-1)
        if row_quality.shape != (member.shape[0],) or row_stability.shape != (member.shape[0],):
            raise ValueError("quality/stability rows do not align with masks")
        member = member.to(compute_device)
        observed = observed.to(compute_device)
        radius = radius.to(compute_device)
        row_quality = row_quality.to(compute_device)
        row_stability = row_stability.to(compute_device)
        coobserved = (
            observed[src] & observed[dst]
            if shared_observation
            else observed[:, src] & observed[:, dst]
        )
        valid_radius = torch.isfinite(radius) & (radius > 0)
        # Clamp out-of-range radii into the declared extreme bins rather than
        # silently discarding a genuine part or room-scale proposal.
        bin_index = torch.bucketize(
            radius.clamp_min(1e-12).log(), compute_bins[1:-1], right=False
        )
        base = row_quality.clamp(0, 1) * row_stability.clamp(0, 1)
        for start in range(0, member.shape[0], int(mask_chunk)):
            stop = min(start + int(mask_chunk), member.shape[0])
            probability = member[start:stop]
            local_base = base[start:stop, None]
            source_probability, destination_probability = probability[:, src], probability[:, dst]
            observation_pair = coobserved[None] if shared_observation else coobserved[start:stop]
            same = bool(include_same) & (
                observation_pair
                & (source_probability >= float(inside_threshold))
                & (destination_probability >= float(inside_threshold))
            )
            separate = bool(include_separate) & observation_pair & (
                ((source_probability >= float(inside_threshold))
                 & (destination_probability <= float(outside_threshold)))
                | ((destination_probability >= float(inside_threshold))
                   & (source_probability <= float(outside_threshold)))
            )
            same_confidence = torch.minimum(
                source_probability, destination_probability
            ) * local_base * same.float()
            forward = torch.minimum(source_probability, 1.0 - destination_probability)
            reverse = torch.minimum(destination_probability, 1.0 - source_probability)
            separate_confidence = torch.maximum(forward, reverse) * local_base * separate.float()
            local_bins = bin_index[start:stop]
            local_valid = valid_radius[start:stop]
            for current_bin in torch.unique(local_bins[local_valid]).tolist():
                selected = (local_bins == int(current_bin)) & local_valid
                same_sum = same_confidence[selected].sum(dim=0)
                separate_sum = separate_confidence[selected].sum(dim=0)
                same_votes[:, int(current_bin)].add_(same_sum)
                separate_votes[:, int(current_bin)].add_(separate_sum)
                observed_votes[:, int(current_bin)].add_(same_sum + separate_sum)
                same_events[int(current_bin)] += int(same[selected].sum())
                separate_events[int(current_bin)] += int(separate[selected].sum())
    return {
        "same_votes": same_votes.cpu(),
        "separate_votes": separate_votes.cpu(),
        "observed_votes": observed_votes.cpu(),
        "same_events": same_events,
        "separate_events": separate_events,
        "scale_bin_edges_log": bins,
    }


def _weighted_median_bin(histogram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return median-bin index and an availability mask for ``[E,B]`` histograms."""

    values = torch.as_tensor(histogram).float().cpu()
    if values.ndim != 2 or not bool((values >= 0).all()):
        raise ValueError("histogram must be non-negative [E,B]")
    total = values.sum(dim=-1)
    available = total > 0
    cumulative = values.cumsum(dim=-1)
    index = (cumulative >= 0.5 * total[:, None]).to(torch.int64).argmax(dim=-1)
    return index, available


def merge_scale_intervals(votes: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Derive transparent non-learning merge-scale intervals from soft votes."""

    same = torch.as_tensor(votes["same_votes"]).float().cpu()
    separate = torch.as_tensor(votes["separate_votes"]).float().cpu()
    edges = torch.as_tensor(votes["scale_bin_edges_log"]).float().cpu().reshape(-1)
    if same.shape != separate.shape or edges.numel() != same.shape[1] + 1:
        raise ValueError("scale vote tensors and bin edges do not align")
    centres = 0.5 * (edges[:-1] + edges[1:])
    lower_index, has_lower = _weighted_median_bin(separate)
    upper_index, has_upper = _weighted_median_bin(same)
    lower = centres[lower_index]
    upper = centres[upper_index]
    lower = torch.where(has_lower, lower, torch.full_like(lower, float("nan")))
    upper = torch.where(has_upper, upper, torch.full_like(upper, float("nan")))
    both = has_lower & has_upper
    consistent = ~both | (lower <= upper)
    merge = torch.where(
        both, 0.5 * (lower + upper), torch.where(has_upper, upper, lower)
    )
    total = same + separate
    probability_same = same / total.clamp_min(1e-12)
    entropy = -(
        probability_same * probability_same.clamp_min(1e-12).log()
        + (1.0 - probability_same) * (1.0 - probability_same).clamp_min(1e-12).log()
    )
    entropy = (entropy * total).sum(dim=-1) / total.sum(dim=-1).clamp_min(1e-12)
    return {
        "merge_log_radius": merge,
        "lower_log_radius": lower,
        "upper_log_radius": upper,
        "has_lower": has_lower,
        "has_upper": has_upper,
        "interval_consistent": consistent,
        "constraint_entropy": entropy,
        "same_mass": same.sum(dim=-1),
        "separate_mass": separate.sum(dim=-1),
        "scale_bin_centres_log": centres,
    }
