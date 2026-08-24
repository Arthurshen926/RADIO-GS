"""Order-invariant query-time candidate marginal for registered RGB prompts.

This module contains no vision model and reads no benchmark data.  It is the
probability-correct seam after independent per-view SAM calls and exact
renderer-adjoint registration have produced one Gaussian probability field per
candidate and view.  Query-time tensors are ephemeral and never become scene
state.
"""

from __future__ import annotations

from dataclasses import dataclass
import string
from typing import Sequence

import numpy as np
import torch


POSITIVE_POINT_SALT = 0xA4093822299F31D0
NEGATIVE_POINT_SALT = 0x082EFA98EC4E6C89
_UINT64_MASK = (1 << 64) - 1


class QueryAbstention(RuntimeError):
    """Raised when the frozen complete-candidate contract cannot be met."""


@dataclass(frozen=True)
class SynchronousMultiviewCandidateMarginal:
    probability: torch.Tensor
    candidate_probability: torch.Tensor
    candidate_field: torch.Tensor
    view_probability: torch.Tensor
    candidate_digests: tuple[str, ...]
    view_digests: tuple[str, ...]


def fuse_positive_unknown_views(
    view_probability: torch.Tensor,
    view_log_precision: torch.Tensor,
    *,
    decision_boundary: float = 0.5,
) -> torch.Tensor:
    """Fuse complementary positive detections while preserving unknown.

    A registered view contributes positive object support only where its SAM
    posterior crosses the frozen binary decision boundary.  A lower value is
    not automatically background evidence: the object can be occluded or the
    proposal can miss a visible part.  Positive detections are independent
    coverage events and are therefore composed by noisy-OR.  Query-independent
    exact-renderer precision is used as relative reliability without forcing
    the views to divide one unit of authority between them.
    """

    fields = torch.as_tensor(view_probability)
    precision = torch.as_tensor(
        view_log_precision, device=fields.device, dtype=fields.dtype
    )
    if fields.ndim != 2 or not fields.is_floating_point() or fields.shape[0] <= 0:
        raise QueryAbstention("view probabilities must be floating [V,N]")
    if precision.shape != (fields.shape[0],):
        raise QueryAbstention("view precision axis differs")
    if not bool(torch.isfinite(fields).all()) or bool(
        ((fields < 0) | (fields > 1)).any()
    ):
        raise QueryAbstention("view probabilities are not finite in [0,1]")
    if not bool(torch.isfinite(precision).all()):
        raise QueryAbstention("view precision is absent or nonfinite")
    if not 0.0 < float(decision_boundary) < 1.0:
        raise ValueError("decision_boundary must lie in (0,1)")
    relative_reliability = torch.exp(precision - precision.max()).clamp(0, 1)
    # Exactly-at-boundary values are the explicit unknown state produced for
    # primitives that are invisible in one registered view.  They must not be
    # converted into a half-strength positive detection: doing so makes every
    # additional view expand foreground over all of its unobserved carrier
    # rows.  Only evidence strictly above the Bernoulli decision boundary is
    # affirmative support.
    positive = torch.where(
        fields > float(decision_boundary), fields, torch.zeros_like(fields)
    )
    positive = positive * relative_reliability[:, None]
    return (1.0 - torch.prod(1.0 - positive, dim=0)).clamp(0, 1)


def _validate_digests(values: Sequence[str], count: int, label: str) -> tuple[str, ...]:
    digests = tuple(str(value).lower() for value in values)
    if len(digests) != count or len(set(digests)) != count:
        raise QueryAbstention(f"{label} digest set is incomplete or non-unique")
    hexdigits = set(string.hexdigits.lower())
    if any(len(value) != 64 or not set(value).issubset(hexdigits) for value in digests):
        raise QueryAbstention(f"{label} digest is not SHA-256")
    return digests


def _splitmix64(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.uint64)
    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def deterministic_visible_signed_points(
    probability: np.ndarray,
    visibility: np.ndarray,
    *,
    positive_authority: np.ndarray | None = None,
    negative_authority: np.ndarray | None = None,
    candidate_digest: str,
    view_digest: str,
    points_per_sign: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Select stable positive/negative SAM points from one projected candidate.

    The sign of a point must come from explicit authorized query evidence.  In
    particular, low candidate posterior is not negative scribble evidence: it
    can equally mean occlusion or unknown support.  Pixel priority depends only
    on row-major id, candidate identity, view identity, and a sign-specific
    frozen salt.  It is independent of process RNG state and view traversal
    order.
    """

    posterior = np.asarray(probability)
    visible = np.asarray(visibility, dtype=bool)
    if posterior.ndim != 2 or visible.shape != posterior.shape:
        raise QueryAbstention("projected probability and visibility axes differ")
    if not np.isfinite(posterior).all() or np.any((posterior < 0) | (posterior > 1)):
        raise QueryAbstention("projected probability is not finite in [0,1]")
    if positive_authority is None or negative_authority is None:
        raise QueryAbstention("explicit signed authority is required")
    positive = np.asarray(positive_authority, dtype=bool)
    negative = np.asarray(negative_authority, dtype=bool)
    if positive.shape != posterior.shape or negative.shape != posterior.shape:
        raise QueryAbstention("signed authority axes differ")
    if bool((positive & negative).any()):
        raise QueryAbstention("positive and negative authority overlap")
    if int(points_per_sign) <= 0:
        raise ValueError("points_per_sign must be positive")
    candidate = _validate_digests((candidate_digest,), 1, "candidate")[0]
    view = _validate_digests((view_digest,), 1, "view")[0]
    identity = int(candidate[-16:], 16) ^ int(view[-16:], 16)

    selected: list[np.ndarray] = []
    for population, salt in (
        (visible & positive, POSITIVE_POINT_SALT),
        (visible & negative, NEGATIVE_POINT_SALT),
    ):
        pixel_ids = np.flatnonzero(population.reshape(-1)).astype(np.uint64)
        if pixel_ids.size < int(points_per_sign):
            raise QueryAbstention("projected candidate lacks signed visible support")
        key = _splitmix64(pixel_ids ^ np.uint64((identity ^ salt) & _UINT64_MASK))
        order = np.lexsort((pixel_ids, key))[: int(points_per_sign)]
        chosen = pixel_ids[order].astype(np.int64)
        rows, columns = np.divmod(chosen, posterior.shape[1])
        selected.append(np.stack((columns, rows), axis=-1))
    points = np.concatenate(selected, axis=0).astype(np.float32)
    labels = np.concatenate(
        (
            np.ones(int(points_per_sign), dtype=np.int64),
            np.zeros(int(points_per_sign), dtype=np.int64),
        )
    )
    return points, labels


def marginalize_synchronous_multiview_candidates(
    candidate_view_probability: torch.Tensor,
    view_log_precision: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    candidate_digests: Sequence[str],
    view_digests: Sequence[str],
    expected_candidates: int = 10,
    view_huber_delta: float = 2.0,
    probability_logit_clip: float = 12.0,
) -> SynchronousMultiviewCandidateMarginal:
    """Marginalize every complete candidate and registered view symmetrically.

    Inputs have shape ``[K,V,N]``, ``[K,V]``, and ``[K]``.  Canonical digest
    sorting makes both candidate and view traversal order semantically inert.
    Every candidate/view is mandatory: missing calls abstain rather than
    changing K or silently falling back to a reference-only method.  Views are
    fused in log-odds space around a precision-weighted median with a bounded
    Huber influence.  This keeps mutually corroborating views decisive while
    preventing one occluded or failed view from linearly diluting the field.
    Candidate uncertainty is then marginalized in probability space.
    """

    fields = torch.as_tensor(candidate_view_probability)
    precision = torch.as_tensor(
        view_log_precision, device=fields.device, dtype=fields.dtype
    )
    logits = torch.as_tensor(candidate_logits, device=fields.device, dtype=fields.dtype)
    if fields.ndim != 3 or not fields.is_floating_point():
        raise QueryAbstention("candidate-view probabilities must be floating [K,V,N]")
    candidates, views, rows = map(int, fields.shape)
    if candidates != int(expected_candidates) or views <= 0 or rows <= 0:
        raise QueryAbstention("candidate/view/row cohort is incomplete")
    if precision.shape != (candidates, views) or logits.shape != (candidates,):
        raise QueryAbstention("candidate likelihood axes differ")
    if not bool(torch.isfinite(fields).all()) or bool(((fields < 0) | (fields > 1)).any()):
        raise QueryAbstention("candidate-view probabilities are not finite in [0,1]")
    if not bool(torch.isfinite(precision).all()) or not bool(torch.isfinite(logits).all()):
        raise QueryAbstention("candidate likelihood contains missing/nonfinite evidence")
    if not np.isfinite(view_huber_delta) or float(view_huber_delta) <= 0:
        raise ValueError("view_huber_delta must be finite and positive")
    if not np.isfinite(probability_logit_clip) or float(probability_logit_clip) <= 0:
        raise ValueError("probability_logit_clip must be finite and positive")

    candidate_ids = _validate_digests(candidate_digests, candidates, "candidate")
    view_ids = _validate_digests(view_digests, views, "view")
    candidate_order = [index for index, _ in sorted(enumerate(candidate_ids), key=lambda x: x[1])]
    view_order = [index for index, _ in sorted(enumerate(view_ids), key=lambda x: x[1])]
    fields = fields[candidate_order][:, view_order]
    precision = precision[candidate_order][:, view_order]
    logits = logits[candidate_order]
    sorted_candidates = tuple(candidate_ids[index] for index in candidate_order)
    sorted_views = tuple(view_ids[index] for index in view_order)

    view_probability = torch.softmax(precision, dim=1)
    clip = float(probability_logit_clip)
    eps = torch.sigmoid(fields.new_tensor(-clip))
    view_log_odds = torch.logit(fields.clamp(eps, 1.0 - eps))
    sorted_log_odds, sorted_indices = torch.sort(view_log_odds, dim=1)
    expanded_weights = view_probability[:, :, None].expand_as(view_log_odds)
    sorted_weights = torch.gather(expanded_weights, 1, sorted_indices)
    cumulative_weights = torch.cumsum(sorted_weights, dim=1)
    median_index = (cumulative_weights < 0.5).sum(dim=1, keepdim=True).clamp_max(views - 1)
    median_log_odds = torch.gather(sorted_log_odds, 1, median_index).squeeze(1)
    bounded_residual = (view_log_odds - median_log_odds[:, None, :]).clamp(
        -float(view_huber_delta), float(view_huber_delta)
    )
    candidate_log_odds = median_log_odds + torch.einsum(
        "kv,kvn->kn", view_probability, bounded_residual
    )
    candidate_field = torch.sigmoid(candidate_log_odds)
    candidate_probability = torch.softmax(logits, dim=0)
    probability = torch.einsum("k,kn->n", candidate_probability, candidate_field)
    tolerance = 16 * torch.finfo(probability.dtype).eps
    if bool(((probability < -tolerance) | (probability > 1 + tolerance)).any()):
        raise RuntimeError("synchronous candidate marginal left probability bounds")
    return SynchronousMultiviewCandidateMarginal(
        probability=probability.clamp(0, 1),
        candidate_probability=candidate_probability,
        candidate_field=candidate_field,
        view_probability=view_probability,
        candidate_digests=sorted_candidates,
        view_digests=sorted_views,
    )


__all__ = [
    "NEGATIVE_POINT_SALT",
    "POSITIVE_POINT_SALT",
    "QueryAbstention",
    "SynchronousMultiviewCandidateMarginal",
    "deterministic_visible_signed_points",
    "fuse_positive_unknown_views",
    "marginalize_synchronous_multiview_candidates",
]
