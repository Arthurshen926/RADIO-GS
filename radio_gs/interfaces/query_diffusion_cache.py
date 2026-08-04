"""Fail-closed Euclidean kNN caches for query-conditioned diffusion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def tensor_sha256(values: torch.Tensor) -> str:
    array = torch.as_tensor(values).detach().cpu().contiguous().numpy()
    # ``ndarray.tobytes`` duplicates the complete bank.  A 393k-by-4096 fp16
    # official relation cache is already about 3 GiB, so hash its contiguous
    # buffer incrementally without changing the byte-level digest.
    raw = memoryview(array).cast("B")
    digest = hashlib.sha256()
    block_size = 64 * 1024 * 1024
    for start in range(0, len(raw), block_size):
        digest.update(raw[start : start + block_size])
    return digest.hexdigest()


@dataclass(frozen=True)
class QueryDiffusionKnnCache:
    neighbor_indices: torch.Tensor
    global_rows: torch.Tensor
    num_global_rows: int
    xyz_sha256: str
    metadata: Mapping[str, object]

    @property
    def num_nodes(self) -> int:
        return int(self.neighbor_indices.shape[0])

    @property
    def effective_k(self) -> int:
        return int(self.neighbor_indices.shape[1])


@dataclass(frozen=True)
class QueryDiffusionRelationCache:
    features: torch.Tensor
    global_rows: torch.Tensor
    num_global_rows: int
    xyz_sha256: str
    metadata: Mapping[str, object]

    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[1])


def build_exact_euclidean_knn(
    xyz: torch.Tensor,
    *,
    num_neighbors: int = 200,
    include_self: bool = True,
    workers: int = -1,
) -> torch.Tensor:
    """Vectorized equivalent of LUDVIG's per-point ``cKDTree.query`` loop.

    The released helper calls ``query(k=num_neighbors + 1)``.  With its
    default ``include_self=True`` it retains all returned columns, so the
    official parameter ``num_neighbors=200`` has an effective K of 201.
    """

    from scipy.spatial import cKDTree

    points = torch.as_tensor(xyz).detach().float().cpu()
    if points.ndim != 2 or points.shape[1] != 3 or not bool(torch.isfinite(points).all()):
        raise ValueError("xyz must be a finite [num_nodes,3] tensor")
    requested = int(num_neighbors)
    if requested <= 0:
        raise ValueError("num_neighbors must be positive")
    query_k = requested + 1
    if query_k > points.shape[0]:
        raise ValueError("num_neighbors + self exceeds the node count")
    tree = cKDTree(points.numpy())
    try:
        _distances, indices = tree.query(points.numpy(), k=query_k, workers=int(workers))
    except TypeError:  # scipy before the workers keyword
        _distances, indices = tree.query(points.numpy(), k=query_k)
    indices = np.asarray(indices, dtype=np.int64)
    if not include_self:
        indices = indices[:, 1:]
    expected_k = requested + int(bool(include_self))
    if indices.shape != (points.shape[0], expected_k):
        raise RuntimeError("cKDTree returned an unexpected kNN shape")
    if indices.min(initial=0) < 0 or indices.max(initial=0) >= points.shape[0]:
        raise RuntimeError("cKDTree returned an out-of-domain node")
    return torch.from_numpy(indices.astype(np.int32, copy=False))


def load_query_diffusion_knn_cache(
    path: str | Path,
    *,
    expected_global_rows: torch.Tensor | None = None,
    expected_xyz: torch.Tensor | None = None,
    expected_source_graph_sha256: str = "",
    expected_num_neighbors: int | None = None,
) -> QueryDiffusionKnnCache:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported query-diffusion kNN cache")
    if payload.get("artifact_type") != "query_conditioned_diffusion_euclidean_knn":
        raise ValueError("unexpected query-diffusion kNN artifact type")
    neighbors = torch.as_tensor(payload.get("neighbor_indices")).long().cpu()
    rows = torch.as_tensor(payload.get("global_rows")).long().cpu()
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("query-diffusion kNN cache lacks metadata")
    if neighbors.ndim != 2 or rows.shape != (neighbors.shape[0],):
        raise ValueError("query-diffusion kNN rows are malformed")
    if neighbors.numel() and (
        int(neighbors.min()) < 0 or int(neighbors.max()) >= neighbors.shape[0]
    ):
        raise ValueError("query-diffusion kNN neighbor is outside local rows")
    if rows.unique().numel() != rows.numel():
        raise ValueError("query-diffusion global rows are not unique")
    num_global_rows = int(payload.get("num_global_rows", -1))
    if num_global_rows <= 0 or bool((rows < 0).any()) or bool((rows >= num_global_rows).any()):
        raise ValueError("query-diffusion global rows are out of range")
    if metadata.get("query_independent") is not True or any(
        metadata.get(key) is not False
        for key in ("labels_opened", "target_masks_opened", "target_metrics_opened")
    ):
        raise ValueError("query-diffusion kNN cache violates safety metadata")
    if expected_global_rows is not None and not torch.equal(
        rows, torch.as_tensor(expected_global_rows).long().cpu()
    ):
        raise ValueError("query-diffusion kNN rows differ from capability rows")
    xyz_digest = str(payload.get("xyz_sha256", ""))
    if expected_xyz is not None and tensor_sha256(expected_xyz) != xyz_digest:
        raise ValueError("query-diffusion kNN geometry digest differs")
    if expected_source_graph_sha256 and str(
        metadata.get("source_graph_sha256", "")
    ) != str(expected_source_graph_sha256):
        raise ValueError("query-diffusion kNN source graph digest differs")
    if expected_num_neighbors is not None:
        requested = int(metadata.get("official_num_neighbors_parameter", -1))
        if requested != int(expected_num_neighbors):
            raise ValueError("query-diffusion requested neighbor count differs")
        include_self = bool(metadata.get("include_self", False))
        if neighbors.shape[1] != requested + int(include_self):
            raise ValueError("query-diffusion effective K differs from metadata")
    return QueryDiffusionKnnCache(
        neighbor_indices=neighbors,
        global_rows=rows,
        num_global_rows=num_global_rows,
        xyz_sha256=xyz_digest,
        metadata=dict(metadata),
    )


def load_query_diffusion_relation_cache(
    path: str | Path,
    *,
    expected_global_rows: torch.Tensor | None = None,
    expected_xyz: torch.Tensor | None = None,
    expected_source_graph_sha256: str = "",
    expected_field_checkpoint_sha256: str = "",
    expected_source_capability_cache: str | Path | None = None,
) -> QueryDiffusionRelationCache:
    """Load a relation bank while binding rows, geometry, field, and source.

    Features intentionally retain their stored dtype.  In particular, the
    matched-capacity diagnostic stores the official 4096-D rows losslessly as
    fp16 and must not create an avoidable multi-gigabyte fp32 CPU duplicate at
    load time.
    """

    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported query-diffusion relation cache")
    if payload.get("artifact_type") != "query_conditioned_diffusion_relation_features":
        raise ValueError("unexpected query-diffusion relation artifact type")
    features = torch.as_tensor(payload.get("features")).cpu()
    rows = torch.as_tensor(payload.get("global_rows")).long().cpu()
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("query-diffusion relation cache lacks metadata")
    if features.ndim != 2 or rows.shape != (features.shape[0],):
        raise ValueError("query-diffusion relation rows are malformed")
    if not features.is_floating_point():
        raise ValueError("query-diffusion relation features must be floating point")
    for start in range(0, features.shape[0], 4096):
        if not bool(torch.isfinite(features[start : start + 4096]).all()):
            raise ValueError("query-diffusion relation features must be finite")
    if rows.unique().numel() != rows.numel():
        raise ValueError("query-diffusion relation global rows are not unique")
    num_global_rows = int(payload.get("num_global_rows", -1))
    if num_global_rows <= 0 or bool((rows < 0).any()) or bool(
        (rows >= num_global_rows).any()
    ):
        raise ValueError("query-diffusion relation global rows are out of range")
    if metadata.get("query_independent") is not True or any(
        metadata.get(key) is not False
        for key in ("labels_opened", "target_masks_opened", "target_metrics_opened")
    ):
        raise ValueError("query-diffusion relation cache violates safety metadata")
    declared_dimension = int(metadata.get("output_dimension", -1))
    if declared_dimension != int(features.shape[1]):
        raise ValueError("query-diffusion relation feature dimension differs")
    if expected_global_rows is not None and not torch.equal(
        rows, torch.as_tensor(expected_global_rows).long().cpu()
    ):
        raise ValueError("query-diffusion relation rows differ from capability rows")
    xyz_digest = str(payload.get("xyz_sha256", ""))
    if expected_xyz is not None and tensor_sha256(expected_xyz) != xyz_digest:
        raise ValueError("query-diffusion relation geometry digest differs")
    if expected_source_graph_sha256 and str(
        metadata.get("source_graph_sha256", "")
    ) != str(expected_source_graph_sha256):
        raise ValueError("query-diffusion relation source graph digest differs")
    if expected_field_checkpoint_sha256 and str(
        metadata.get("field_checkpoint_sha256", "")
    ) != str(expected_field_checkpoint_sha256):
        raise ValueError("query-diffusion relation canonical-field hash differs")
    if expected_source_capability_cache is not None:
        declared = Path(str(metadata.get("source_capability_cache", ""))).resolve()
        expected = Path(expected_source_capability_cache).resolve()
        if declared != expected:
            raise ValueError("query-diffusion relation capability source differs")
    return QueryDiffusionRelationCache(
        features=features,
        global_rows=rows,
        num_global_rows=num_global_rows,
        xyz_sha256=xyz_digest,
        metadata=dict(metadata),
    )
