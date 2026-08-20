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
    candidate_digest: str,
    view_digest: str,
    points_per_sign: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Select stable positive/negative SAM points from one projected candidate.

    Pixel priority depends only on row-major id, candidate identity, view
    identity, and a sign-specific frozen salt.  It is independent of process
    RNG state and view traversal order.
    """

    posterior = np.asarray(probability)
    visible = np.asarray(visibility, dtype=bool)
    if posterior.ndim != 2 or visible.shape != posterior.shape:
        raise QueryAbstention("projected probability and visibility axes differ")
    if not np.isfinite(posterior).all() or np.any((posterior < 0) | (posterior > 1)):
        raise QueryAbstention("projected probability is not finite in [0,1]")
    if int(points_per_sign) <= 0:
        raise ValueError("points_per_sign must be positive")
    candidate = _validate_digests((candidate_digest,), 1, "candidate")[0]
    view = _validate_digests((view_digest,), 1, "view")[0]
    identity = int(candidate[-16:], 16) ^ int(view[-16:], 16)

    selected: list[np.ndarray] = []
    for population, salt in (
        (visible & (posterior >= 0.5), POSITIVE_POINT_SALT),
        (visible & (posterior < 0.5), NEGATIVE_POINT_SALT),
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
) -> SynchronousMultiviewCandidateMarginal:
    """Marginalize every complete candidate and registered view symmetrically.

    Inputs have shape ``[K,V,N]``, ``[K,V]``, and ``[K]``.  Canonical digest
    sorting makes both candidate and view traversal order semantically inert.
    Every candidate/view is mandatory: missing calls abstain rather than
    changing K or silently falling back to a reference-only method.
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
    candidate_field = torch.einsum("kv,kvn->kn", view_probability, fields)
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
    "marginalize_synchronous_multiview_candidates",
]
