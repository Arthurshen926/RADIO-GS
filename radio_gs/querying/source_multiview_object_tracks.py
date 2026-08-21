"""Query-independent object tracks from source-view exact-MPR SAM masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse


@dataclass(frozen=True)
class SourceObjectTracks:
    """Sparse Gaussian-to-track membership ready for a typed posterior."""

    row_indices: torch.Tensor
    track_indices: torch.Tensor
    membership_weights: torch.Tensor
    proposal_track_indices: torch.Tensor
    track_view_counts: torch.Tensor
    track_confidence: torch.Tensor
    num_tracks: int
    stats: dict[str, int | float | str]


def build_source_multiview_object_tracks(
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    membership_weights: torch.Tensor,
    proposal_view_indices: torch.Tensor,
    *,
    num_rows: int,
    num_proposals: int,
    minimum_soft_cosine: float = 0.5,
) -> SourceObjectTracks:
    """Associate masks across views using only lifted Gaussian membership.

    For every pair of source views, proposals are linked only when they are
    reciprocal soft-cosine nearest neighbours and their overlap reaches the
    fixed majority-overlap authority.  Edges are consumed from strongest to
    weakest while enforcing at most one observation from each source view per
    track.  No semantic score, query, label, camera target, or benchmark metric
    enters association.

    Singleton proposals are retained in ``proposal_track_indices`` as ``-1``
    but do not become object tracks: a track requires independent source-view
    confirmation.  Gaussian membership within a track is the clipped sum of
    its source observations, a conservative approximation to their union.
    """

    rows = torch.as_tensor(row_indices).detach().cpu().long().reshape(-1).numpy()
    proposals = (
        torch.as_tensor(proposal_indices).detach().cpu().long().reshape(-1).numpy()
    )
    weights = (
        torch.as_tensor(membership_weights).detach().cpu().float().reshape(-1).numpy()
    )
    views = (
        torch.as_tensor(proposal_view_indices).detach().cpu().long().reshape(-1).numpy()
    )
    row_count, proposal_count = int(num_rows), int(num_proposals)
    floor = float(minimum_soft_cosine)
    if not (rows.shape == proposals.shape == weights.shape):
        raise ValueError("sparse proposal membership axes differ")
    if row_count <= 0 or proposal_count <= 0 or views.shape != (proposal_count,):
        raise ValueError("row/proposal/view counts differ")
    if not 0.0 < floor <= 1.0:
        raise ValueError("minimum_soft_cosine must be in (0,1]")
    valid = (
        (rows >= 0)
        & (rows < row_count)
        & (proposals >= 0)
        & (proposals < proposal_count)
        & np.isfinite(weights)
        & (weights > 0)
        & (views[proposals] >= 0)
    )
    rows, proposals, weights = rows[valid], proposals[valid], weights[valid]
    matrix = sparse.csr_matrix(
        (weights, (proposals, rows)), shape=(proposal_count, row_count)
    )
    gram = (matrix @ matrix.T).toarray()
    norm = np.sqrt(np.maximum(np.diag(gram), 1.0e-12))
    similarity = gram / norm[:, None] / norm[None, :]

    candidate_edges: list[tuple[int, int, float]] = []
    unique_views = np.unique(views)
    for left_position, left_view in enumerate(unique_views):
        left = np.where(views == left_view)[0]
        for right_view in unique_views[left_position + 1 :]:
            right = np.where(views == right_view)[0]
            pair = similarity[np.ix_(left, right)]
            if not np.any(pair >= floor):
                continue
            left_best = pair.argmax(axis=1)
            right_best = pair.argmax(axis=0)
            for left_local, right_local in enumerate(left_best):
                score = float(pair[left_local, right_local])
                if right_best[right_local] == left_local and score >= floor:
                    candidate_edges.append(
                        (int(left[left_local]), int(right[right_local]), score)
                    )

    parent = np.arange(proposal_count, dtype=np.int64)
    component_views = [{int(view)} for view in views]
    component_edges: list[list[float]] = [[] for _ in range(proposal_count)]

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    accepted_edges = 0
    for left, right, score in sorted(candidate_edges, key=lambda edge: -edge[2]):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        if not component_views[left_root].isdisjoint(component_views[right_root]):
            continue
        parent[right_root] = left_root
        component_views[left_root].update(component_views[right_root])
        component_edges[left_root].extend(component_edges[right_root])
        component_edges[left_root].append(score)
        accepted_edges += 1

    roots = np.asarray([find(index) for index in range(proposal_count)])
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    proposal_counts = np.bincount(inverse)
    linked_components = np.where(proposal_counts > 1)[0]
    component_to_track = np.full(len(unique_roots), -1, dtype=np.int64)
    component_to_track[linked_components] = np.arange(len(linked_components))
    proposal_tracks = component_to_track[inverse]
    track_count = int(len(linked_components))
    if track_count == 0:
        empty_long = torch.empty(0, dtype=torch.long)
        empty_float = torch.empty(0, dtype=torch.float32)
        return SourceObjectTracks(
            empty_long,
            empty_long,
            empty_float,
            torch.from_numpy(proposal_tracks),
            empty_long,
            empty_float,
            0,
            {
                "construction": "cross_view_reciprocal_soft_overlap_tracks_v1",
                "candidate_edges": len(candidate_edges),
                "accepted_edges": accepted_edges,
                "num_tracks": 0,
            },
        )

    retained_proposals = np.where(proposal_tracks >= 0)[0]
    track_assignment = sparse.csr_matrix(
        (
            np.ones(len(retained_proposals), dtype=np.float32),
            (proposal_tracks[retained_proposals], retained_proposals),
        ),
        shape=(track_count, proposal_count),
    )
    track_membership = (track_assignment @ matrix).tocoo()
    track_weights = np.minimum(track_membership.data, 1.0).astype(np.float32)
    track_view_counts = np.zeros(track_count, dtype=np.int64)
    track_confidence = np.zeros(track_count, dtype=np.float32)
    for track, component in enumerate(linked_components):
        root = find(int(unique_roots[component]))
        track_view_counts[track] = len(component_views[root])
        edge_values = component_edges[root]
        track_confidence[track] = float(np.mean(edge_values)) if edge_values else 0.0
    return SourceObjectTracks(
        row_indices=torch.from_numpy(track_membership.col.astype(np.int64)),
        track_indices=torch.from_numpy(track_membership.row.astype(np.int64)),
        membership_weights=torch.from_numpy(track_weights),
        proposal_track_indices=torch.from_numpy(proposal_tracks),
        track_view_counts=torch.from_numpy(track_view_counts),
        track_confidence=torch.from_numpy(track_confidence),
        num_tracks=track_count,
        stats={
            "construction": "cross_view_reciprocal_soft_overlap_tracks_v1",
            "minimum_soft_cosine": floor,
            "candidate_edges": len(candidate_edges),
            "accepted_edges": accepted_edges,
            "num_tracks": track_count,
            "tracked_proposals": int((proposal_tracks >= 0).sum()),
            "maximum_track_views": int(track_view_counts.max()),
            "membership_count": int(track_membership.nnz),
        },
    )


def build_source_learned_object_tracks(
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    membership_weights: torch.Tensor,
    proposal_view_indices: torch.Tensor,
    proposal_embeddings: torch.Tensor,
    *,
    num_rows: int,
    num_proposals: int,
    relation_logit_scale: float,
    minimum_same_probability: float,
) -> SourceObjectTracks:
    """Compile learned proposal affinity into the existing sparse interface.

    Association consumes only the frozen 16-D query-independent proposal code.
    Exact-MPR overlap is used to lift accepted tracks to Gaussians, never to
    decide cross-view identity in this learned path.
    """

    rows = torch.as_tensor(row_indices).detach().cpu().long().reshape(-1).numpy()
    proposals = torch.as_tensor(proposal_indices).detach().cpu().long().reshape(-1).numpy()
    weights = torch.as_tensor(membership_weights).detach().cpu().float().reshape(-1).numpy()
    views = torch.as_tensor(proposal_view_indices).detach().cpu().long().reshape(-1).numpy()
    embedding = torch.as_tensor(proposal_embeddings).detach().cpu().float()
    row_count, proposal_count = int(num_rows), int(num_proposals)
    scale, floor = float(relation_logit_scale), float(minimum_same_probability)
    if not (rows.shape == proposals.shape == weights.shape):
        raise ValueError("sparse proposal membership axes differ")
    if views.shape != (proposal_count,) or embedding.shape[0] != proposal_count or embedding.ndim != 2:
        raise ValueError("learned proposal authority axes differ")
    if scale <= 0 or not 0 < floor < 1:
        raise ValueError("learned relation calibration differs")
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    probability = torch.sigmoid(scale * (embedding @ embedding.T)).numpy()

    candidate_edges: list[tuple[int, int, float]] = []
    unique_views = np.unique(views)
    for left_position, left_view in enumerate(unique_views):
        left = np.where(views == left_view)[0]
        for right_view in unique_views[left_position + 1 :]:
            right = np.where(views == right_view)[0]
            pair = probability[np.ix_(left, right)]
            if pair.size == 0 or not np.any(pair >= floor):
                continue
            left_best = pair.argmax(axis=1)
            right_best = pair.argmax(axis=0)
            for left_local, right_local in enumerate(left_best):
                score = float(pair[left_local, right_local])
                if right_best[right_local] == left_local and score >= floor:
                    candidate_edges.append((int(left[left_local]), int(right[right_local]), score))

    parent = np.arange(proposal_count, dtype=np.int64)
    component_views = [{int(view)} for view in views]
    component_edges: list[list[float]] = [[] for _ in range(proposal_count)]

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    accepted_edges = 0
    for left, right, score in sorted(candidate_edges, key=lambda edge: -edge[2]):
        left_root, right_root = find(left), find(right)
        if left_root == right_root or not component_views[left_root].isdisjoint(component_views[right_root]):
            continue
        parent[right_root] = left_root
        component_views[left_root].update(component_views[right_root])
        component_edges[left_root].extend(component_edges[right_root])
        component_edges[left_root].append(score)
        accepted_edges += 1

    roots = np.asarray([find(index) for index in range(proposal_count)])
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    proposal_counts = np.bincount(inverse)
    linked_components = np.where(proposal_counts > 1)[0]
    component_to_track = np.full(len(unique_roots), -1, dtype=np.int64)
    component_to_track[linked_components] = np.arange(len(linked_components))
    proposal_tracks = component_to_track[inverse]
    track_count = int(len(linked_components))
    valid = (
        (rows >= 0) & (rows < row_count) & (proposals >= 0) & (proposals < proposal_count)
        & np.isfinite(weights) & (weights > 0)
    )
    matrix = sparse.csr_matrix((weights[valid], (proposals[valid], rows[valid])), shape=(proposal_count, row_count))
    if track_count == 0:
        empty_long, empty_float = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
        return SourceObjectTracks(
            empty_long, empty_long, empty_float, torch.from_numpy(proposal_tracks),
            empty_long, empty_float, 0,
            {"construction": "learned_compact_affinity_tracks_v1", "candidate_edges": len(candidate_edges), "accepted_edges": 0, "num_tracks": 0},
        )
    retained = np.where(proposal_tracks >= 0)[0]
    assignment = sparse.csr_matrix(
        (np.ones(len(retained), dtype=np.float32), (proposal_tracks[retained], retained)),
        shape=(track_count, proposal_count),
    )
    track_membership = (assignment @ matrix).tocoo()
    track_views = np.zeros(track_count, dtype=np.int64)
    track_confidence = np.zeros(track_count, dtype=np.float32)
    for track, component in enumerate(linked_components):
        root = find(int(unique_roots[component]))
        track_views[track] = len(component_views[root])
        track_confidence[track] = float(np.mean(component_edges[root]))
    return SourceObjectTracks(
        row_indices=torch.from_numpy(track_membership.col.astype(np.int64)),
        track_indices=torch.from_numpy(track_membership.row.astype(np.int64)),
        membership_weights=torch.from_numpy(np.minimum(track_membership.data, 1.0).astype(np.float32)),
        proposal_track_indices=torch.from_numpy(proposal_tracks),
        track_view_counts=torch.from_numpy(track_views),
        track_confidence=torch.from_numpy(track_confidence),
        num_tracks=track_count,
        stats={
            "construction": "learned_compact_affinity_tracks_v1",
            "relation_logit_scale": scale,
            "minimum_same_probability": floor,
            "candidate_edges": len(candidate_edges),
            "accepted_edges": accepted_edges,
            "num_tracks": track_count,
            "tracked_proposals": int((proposal_tracks >= 0).sum()),
            "maximum_track_views": int(track_views.max()),
            "membership_count": int(track_membership.nnz),
        },
    )


__all__ = [
    "SourceObjectTracks",
    "build_source_learned_object_tracks",
    "build_source_multiview_object_tracks",
]
