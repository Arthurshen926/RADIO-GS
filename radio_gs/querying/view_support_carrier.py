"""Query-independent primitive/view support and edge co-visibility.

The carrier deliberately retains only whether an upstream renderer authority
observed a primitive in a training view.  Pixel identities, RGB, prompts and
labels are not inputs.  This makes the result a reusable physical carrier
rather than query evidence.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def build_compact_view_support(
    *,
    global_rows: torch.Tensor,
    view_global_ids: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Map per-view full-geometry row ids to a compact ``[N,V]`` incidence.

    ``global_rows`` is the sealed compact-to-full row map from the primitive
    bundle.  Repeated pixel hits within a view are intentionally collapsed:
    co-visibility is a view-level event, not a proxy for projected area.
    """

    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    if rows.numel() == 0:
        raise ValueError("global_rows cannot be empty")
    if bool((rows < 0).any()) or not bool((rows[1:] > rows[:-1]).all()):
        raise ValueError("global_rows must be strictly increasing non-negative ids")
    views = list(view_global_ids)
    if not views:
        raise ValueError("at least one training view is required")
    support = torch.zeros((rows.numel(), len(views)), dtype=torch.bool)
    for view_index, raw_ids in enumerate(views):
        ids = torch.as_tensor(raw_ids).detach().long().cpu().reshape(-1)
        if bool((ids < 0).any()):
            raise ValueError("responsibility gaussian ids cannot be negative")
        if ids.numel() == 0:
            continue
        ids = torch.unique(ids, sorted=True)
        positions = torch.searchsorted(rows, ids)
        inside = positions < rows.numel()
        if not bool(inside.any()):
            continue
        positions = positions[inside]
        ids = ids[inside]
        matched = rows[positions] == ids
        support[positions[matched], view_index] = True
    return support


def dense_support_to_csr(
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic row-major CSR for a boolean ``[N,V]`` matrix."""

    matrix = torch.as_tensor(support).detach().bool().cpu()
    if matrix.ndim != 2:
        raise ValueError("support must be [num_primitives,num_views]")
    row, col = matrix.nonzero(as_tuple=True)
    counts = torch.bincount(row, minlength=matrix.shape[0])
    crow = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    # ScanNet uses far fewer than 32767 training views.  int16 keeps this
    # authority compact without changing its exact integer semantics.
    if matrix.shape[1] >= 32768:
        raise ValueError("view axis exceeds the int16 carrier contract")
    return crow, col.to(torch.int16)


@torch.inference_mode()
def edge_covisibility_from_support(
    *,
    support: torch.Tensor,
    edge_index: torch.Tensor,
    chunk_size: int = 131072,
) -> dict[str, torch.Tensor]:
    """Compute binary Jaccard and conditional overlap on candidate edges.

    Directional conditional overlaps are retained because a rarely observed
    endpoint can be wholly supported by a broadly observed endpoint while the
    reverse is false.  ``overlap_coefficient`` is the symmetric
    ``shared/min(count_i,count_j)`` summary.
    """

    matrix = torch.as_tensor(support).detach().bool().cpu()
    edges = torch.as_tensor(edge_index).detach().long().cpu()
    if matrix.ndim != 2:
        raise ValueError("support must be [num_primitives,num_views]")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must be [2,num_edges]")
    if edges.numel() and (
        bool((edges < 0).any()) or int(edges.max()) >= matrix.shape[0]
    ):
        raise ValueError("edge_index references a nonexistent primitive")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")

    count = edges.shape[1]
    shared = torch.empty(count, dtype=torch.uint8)
    union = torch.empty(count, dtype=torch.uint8)
    source_count = torch.empty(count, dtype=torch.uint8)
    target_count = torch.empty(count, dtype=torch.uint8)
    jaccard = torch.empty(count, dtype=torch.float16)
    source_given_target = torch.empty(count, dtype=torch.float16)
    target_given_source = torch.empty(count, dtype=torch.float16)
    overlap = torch.empty(count, dtype=torch.float16)
    for start in range(0, count, int(chunk_size)):
        stop = min(start + int(chunk_size), count)
        source, target = edges[:, start:stop]
        left, right = matrix[source], matrix[target]
        common = (left & right).sum(dim=1).to(torch.int64)
        either = (left | right).sum(dim=1).to(torch.int64)
        left_count = left.sum(dim=1).to(torch.int64)
        right_count = right.sum(dim=1).to(torch.int64)
        if matrix.shape[1] > 255:
            raise ValueError("uint8 count carrier supports at most 255 views")
        shared[start:stop] = common.to(torch.uint8)
        union[start:stop] = either.to(torch.uint8)
        source_count[start:stop] = left_count.to(torch.uint8)
        target_count[start:stop] = right_count.to(torch.uint8)
        common_f = common.float()
        jaccard[start:stop] = (common_f / either.clamp_min(1).float()).half()
        source_given_target[start:stop] = (
            common_f / right_count.clamp_min(1).float()
        ).half()
        target_given_source[start:stop] = (
            common_f / left_count.clamp_min(1).float()
        ).half()
        overlap[start:stop] = (
            common_f / torch.minimum(left_count, right_count).clamp_min(1).float()
        ).half()
    return {
        "shared_view_count": shared,
        "union_view_count": union,
        "source_view_count": source_count,
        "target_view_count": target_count,
        "jaccard": jaccard,
        "source_given_target": source_given_target,
        "target_given_source": target_given_source,
        "overlap_coefficient": overlap,
    }


__all__ = [
    "build_compact_view_support",
    "dense_support_to_csr",
    "edge_covisibility_from_support",
]
