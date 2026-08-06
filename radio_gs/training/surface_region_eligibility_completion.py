"""Deterministic query-free eligibility variants for SurfaceRegion V3 caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json

import numpy as np
from scipy.sparse.csgraph import dijkstra
import torch

from radio_gs.interfaces.surface_region_contract import (
    PreparedSurfaceRegionGraphV3,
    SurfaceRegionContractV3,
)


STRUCTURED_ELIGIBILITY_POLICY = (
    "hash_direction_anchor_connected_shortest_path_tree_with_external_support_v2"
)
_HASH_DOMAIN = "surface-region-v3-eligibility-completion-v2"


@dataclass(frozen=True)
class StructuredEligibilityVariant:
    """One auditable eligibility-induced completion input."""

    mask: torch.Tensor
    mask_sha256: str
    policy: str
    variant_index: int
    orientation_axis: int
    orientation_sign: int
    semantic_domain_tokens: int
    nominal_semantic_keep_tokens: int
    semantic_eligible_tokens: int
    globally_eligible_tokens: int
    extreme_graph_fallback: bool
    extreme_graph_fallback_reason: str


def _mask_sha256(mask: torch.Tensor) -> str:
    value = torch.as_tensor(mask).detach().bool().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def structured_eligibility_variant(
    *,
    contract: SurfaceRegionContractV3,
    prepared_graph: PreparedSurfaceRegionGraphV3,
    anchor: int,
    radius_m: float,
    teacher_region_id: str,
    variant_index: int,
) -> StructuredEligibilityVariant:
    """Construct one scene-independent geometry/hash eligibility pattern.

    All semantic-ball nodes are identified on the full hard graph.  A fixed
    hash chooses an axial direction, then a direction-aware frontier selects a
    connected prefix of the anchor's shortest-path tree.  The normal keep
    count is ``minimum - max(1, minimum // 6)`` (20 of 24).  Nodes outside the
    semantic ball remain eligible recovery support.  Calling V3 ``expand`` on
    this induced graph therefore produces a small, explicit completion tail
    rather than post-hoc token deletion or an anchor-only degenerate region.
    """

    if not isinstance(contract, SurfaceRegionContractV3):
        raise TypeError("structured eligibility requires SurfaceRegion V3")
    if not isinstance(prepared_graph, PreparedSurfaceRegionGraphV3):
        raise TypeError("structured eligibility requires a prepared V3 graph")
    anchor = int(anchor)
    variant = int(variant_index)
    radius = float(radius_m)
    if not 0 <= anchor < prepared_graph.num_nodes or radius <= 0:
        raise ValueError("structured eligibility anchor/radius is invalid")
    if variant < 0 or not str(teacher_region_id):
        raise ValueError("structured eligibility identity is invalid")

    key = hashlib.sha256(
        json.dumps(
            [_HASH_DOMAIN, str(teacher_region_id), variant],
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    orientation = (int(key[0]) + variant) % 6
    axis = orientation % 3
    sign = 1 if orientation < 3 else -1

    distance, predecessor = dijkstra(
        prepared_graph.semantic_csr.copy(),
        directed=False,
        indices=anchor,
        limit=radius * float(contract.context_ratio),
        return_predecessors=True,
    )
    distance = np.asarray(distance, dtype=np.float64).reshape(-1)
    predecessor = np.asarray(predecessor, dtype=np.int64).reshape(-1)
    semantic_domain = np.isfinite(distance)
    semantic_domain[anchor] = True
    semantic_rows = np.flatnonzero(semantic_domain)
    external_rows = np.flatnonzero(~semantic_domain)
    if external_rows.size == 0:
        raise ValueError(
            "structured eligibility cannot induce support fill when the "
            "semantic ball covers the complete scene graph"
        )
    minimum = int(contract.minimum_tokens)
    nominal_keep = minimum - max(1, minimum // 6)
    semantic_keep = min(nominal_keep, int(semantic_rows.size))
    fallback_reasons: list[str] = []
    if semantic_keep < nominal_keep:
        fallback_reasons.append("semantic_component_below_nominal_keep")
    if semantic_keep + int(external_rows.size) < minimum:
        required = minimum - int(external_rows.size)
        if required > int(semantic_rows.size):
            raise ValueError("structured eligibility cannot satisfy minimum support")
        semantic_keep = required
        fallback_reasons.append("external_support_below_nominal_fill")
    if semantic_keep <= 0 or semantic_keep >= minimum:
        raise ValueError("structured eligibility cannot satisfy minimum support")

    points = prepared_graph.xyz
    children: dict[int, list[int]] = {}
    for row in semantic_rows:
        row = int(row)
        if row == anchor:
            continue
        parent = int(predecessor[row])
        if parent < 0 or not semantic_domain[parent]:
            raise RuntimeError("semantic shortest-path tree lost an anchor path")
        children.setdefault(parent, []).append(row)

    def priority(row: int) -> tuple[float, float, int, int]:
        delta = points[row] - points[anchor]
        norm = float(np.linalg.norm(delta))
        alignment = sign * float(delta[axis]) / max(norm, 1e-12)
        tie = int.from_bytes(
            hashlib.sha256(key + int(row).to_bytes(8, "little")).digest()[:8],
            "little",
        )
        return (-alignment, float(distance[row]), tie, int(row))

    selected = [anchor]
    frontier: list[tuple[float, float, int, int]] = []
    for child in children.get(anchor, []):
        heapq.heappush(frontier, priority(child))
    while len(selected) < semantic_keep and frontier:
        *_priority, row = heapq.heappop(frontier)
        selected.append(int(row))
        for child in children.get(int(row), []):
            heapq.heappush(frontier, priority(child))
    if len(selected) != semantic_keep:
        raise RuntimeError("anchor-connected semantic prefix is incomplete")

    eligibility = np.ones(prepared_graph.num_nodes, dtype=np.bool_)
    eligibility[semantic_domain] = False
    eligibility[np.asarray(selected, dtype=np.int64)] = True
    mask = torch.from_numpy(eligibility)
    return StructuredEligibilityVariant(
        mask=mask,
        mask_sha256=_mask_sha256(mask),
        policy=STRUCTURED_ELIGIBILITY_POLICY,
        variant_index=variant,
        orientation_axis=axis,
        orientation_sign=sign,
        semantic_domain_tokens=int(semantic_rows.size),
        nominal_semantic_keep_tokens=nominal_keep,
        semantic_eligible_tokens=int(eligibility[semantic_domain].sum()),
        globally_eligible_tokens=int(eligibility.sum()),
        extreme_graph_fallback=bool(fallback_reasons),
        extreme_graph_fallback_reason="+".join(fallback_reasons),
    )


def completion_region_id(
    *,
    teacher_region_id: str,
    variant: StructuredEligibilityVariant,
) -> str:
    """Bind a unique paired row ID to teacher, policy, index, and mask."""

    payload = {
        "domain": _HASH_DOMAIN,
        "teacher_region_id": str(teacher_region_id),
        "policy": variant.policy,
        "variant_index": int(variant.variant_index),
        "eligibility_sha256": variant.mask_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
