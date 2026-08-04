#!/usr/bin/env python3
"""Build the shared query-independent 3D support graph from official views."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.primitive_row_authority import PrimitiveRowAuthority
from radio_gs.querying.support_solver import (
    SupportGraphConfig,
    build_primitive_support_graph,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visibility_from_registration_responsibility(
    payload: object,
    *,
    num_global_rows: int,
    global_rows: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Materialize a compact, label-free primitive-by-view visibility matrix.

    The responsibility sidecar is already produced during query-free MPR.  It
    records which Gaussian won a valid RGB-D raster contribution in each
    selected training view.  Converting that sparse assignment list to a
    boolean matrix lets the support graph distinguish nearby primitives that
    were never actually observed together, without opening a query, object,
    label, mask, or metric.  Visibility is intentionally binary here: MPR's
    continuous weights remain the feature-reconstruction contract, while the
    graph relation asks only whether an edge has real shared observation.
    """

    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported registration responsibility cache")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("registration responsibility cache lacks view assignments")
    rows = torch.as_tensor(global_rows).long().cpu()
    if rows.ndim != 1 or rows.numel() == 0:
        raise ValueError("global_rows must be a non-empty [N] tensor")
    if num_global_rows <= 0 or bool((rows < 0).any()) or bool((rows >= num_global_rows).any()):
        raise ValueError("global_rows are outside the responsibility domain")
    if rows.unique().numel() != rows.numel():
        raise ValueError("global_rows must be unique")

    # Mapping once avoids a dense N_global x V allocation when a capability
    # graph deliberately excludes invalid primitive rows.
    global_to_local = torch.full((int(num_global_rows),), -1, dtype=torch.long)
    global_to_local[rows] = torch.arange(rows.numel(), dtype=torch.long)
    visible = torch.zeros((rows.numel(), len(assignments)), dtype=torch.bool)
    registered_global_rows = 0
    for view_index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict) or "gaussian_ids" not in assignment:
            raise ValueError(f"responsibility assignment {view_index} lacks gaussian_ids")
        identifiers = torch.as_tensor(assignment["gaussian_ids"]).long().cpu().reshape(-1)
        if identifiers.numel() == 0:
            continue
        if bool((identifiers < 0).any()) or bool((identifiers >= num_global_rows).any()):
            raise ValueError(
                f"responsibility assignment {view_index} has out-of-range Gaussian ids"
            )
        identifiers = identifiers.unique()
        registered_global_rows += int(identifiers.numel())
        local = global_to_local[identifiers]
        local = local[local >= 0]
        if local.numel():
            visible[local, view_index] = True
    return visible, {
        "num_views": int(visible.shape[1]),
        "valid_primitives_with_any_view": int(visible.any(dim=1).sum()),
        "registered_global_rows_before_capability_filter": int(registered_global_rows),
    }


def _load_mpr_sidecar_metadata(mpr_path: str | Path) -> dict[str, object]:
    """Read the small MPR report used to bind a graph to its sidecar."""

    path = Path(mpr_path)
    sidecar = Path(str(path) + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(
            "covisibility graph requires the MPR JSON sidecar for provenance: "
            f"{sidecar}"
        )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("MPR JSON sidecar has invalid metadata")
    return metadata


def load_covisibility_observations(
    capability_metadata: object,
    *,
    responsibility_cache: str | Path,
    num_global_rows: int,
    global_rows: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Load a responsibility sidecar only when it is bound to this field.

    A same-sized responsibility file from a different RGB Gaussian geometry
    would silently corrupt a graph.  Require the canonical capability bank's
    raw-MPR report to commit to both the exact responsibility file digest and
    the geometry digest before materializing co-visibility.
    """

    if not isinstance(capability_metadata, dict):
        raise ValueError("capability metadata must be a mapping")
    mpr_path = str(capability_metadata.get("mpr_cache", ""))
    if not mpr_path:
        raise ValueError("capability cache lacks raw MPR provenance")
    mpr_metadata = _load_mpr_sidecar_metadata(mpr_path)
    expected_digest = str(mpr_metadata.get("registration_responsibility_cache_sha256", ""))
    expected_xyz_digest = str(mpr_metadata.get("xyz_sha256", ""))
    if not expected_digest or not expected_xyz_digest:
        raise ValueError("MPR sidecar lacks responsibility/geometry provenance")
    cache_path = Path(responsibility_cache).resolve()
    if _sha256_file(cache_path) != expected_digest:
        raise ValueError("responsibility cache digest differs from canonical MPR provenance")
    payload = torch.load(cache_path, map_location="cpu")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        raise ValueError("registration responsibility metadata must be a mapping")
    if str(metadata.get("xyz_sha256", "")) != expected_xyz_digest:
        raise ValueError("responsibility cache geometry differs from canonical MPR")
    selected = metadata.get("selected_dataset_indices", [])
    assignments = payload.get("assignments", []) if isinstance(payload, dict) else []
    if not isinstance(selected, list) or len(selected) != len(assignments):
        raise ValueError("responsibility cache has an incomplete selected-view audit")
    visibility, audit = visibility_from_registration_responsibility(
        payload,
        num_global_rows=int(num_global_rows),
        global_rows=global_rows,
    )
    return visibility, {
        **audit,
        "responsibility_cache": str(cache_path),
        "responsibility_cache_sha256": expected_digest,
        "raw_mpr_cache": str(Path(mpr_path).resolve()),
        "raw_mpr_xyz_sha256": expected_xyz_digest,
        "query_independent": True,
        "labels_opened": False,
    }


def deterministic_feature_hash(
    features: torch.Tensor,
    output_dim: int,
    *,
    batch_size: int = 8192,
) -> torch.Tensor:
    """Signed feature hashing for query-free approximate cosine affinities."""
    values = torch.as_tensor(features)
    if values.ndim != 2 or output_dim <= 0:
        raise ValueError("features must be [N,D] and output_dim must be positive")
    input_dim = int(values.shape[1])
    index = torch.arange(input_dim, dtype=torch.long)
    hashed = index * 2654435761 + 2246822519
    buckets = torch.remainder(hashed, int(output_dim))
    signs = torch.where(
        torch.bitwise_and(hashed, 1) == 0,
        torch.ones(input_dim),
        -torch.ones(input_dim),
    )
    result = torch.empty(values.shape[0], int(output_dim), dtype=torch.float32)
    expanded_buckets = buckets.unsqueeze(0)
    for start in range(0, values.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), values.shape[0])
        batch = values[start:stop].float()
        projected = torch.zeros(batch.shape[0], int(output_dim), dtype=torch.float32)
        projected.scatter_add_(
            1,
            expanded_buckets.expand(batch.shape[0], -1),
            batch * signs,
        )
        result[start:stop] = F.normalize(projected, dim=-1, eps=1e-8)
    return result


def capability_affinity_features(
    capability: dict[str, object],
    global_rows: torch.Tensor,
    *,
    mode: str,
    affinity_dim: int,
    hash_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Choose a query-free DINO/SAM representation for graph affinities.

    The historic graph compresses each official capability bank through a
    deterministic signed hash so CPU graph construction remains lightweight.
    ``exact_official_capability`` is a separate high-fidelity graph variant:
    it preserves the canonical official adaptor rows verbatim and lets the
    graph builder compute their cosine affinities in chunks (optionally on a
    GPU).  Neither path reads a query, label, object, mask, or metric.
    """

    requested = str(mode)
    if requested not in {"signed_hash", "exact_official_capability"}:
        raise ValueError(f"unsupported capability affinity mode: {requested!r}")
    rows = torch.as_tensor(global_rows).long().cpu()
    if rows.ndim != 1 or rows.numel() == 0:
        raise ValueError("global_rows must be a non-empty [N] vector")
    try:
        appearance_full = torch.as_tensor(capability["appearance_dino_v3"])
        boundary_full = torch.as_tensor(capability["boundary_sam3"])
    except KeyError as error:
        raise ValueError("capability cache lacks official DINO/SAM banks") from error
    if (
        appearance_full.ndim != 2
        or boundary_full.ndim != 2
        or appearance_full.shape[0] != boundary_full.shape[0]
        or bool((rows < 0).any())
        or bool((rows >= appearance_full.shape[0]).any())
    ):
        raise ValueError("capability banks/global_rows are not row aligned")
    appearance = appearance_full.index_select(0, rows)
    boundary = boundary_full.index_select(0, rows)
    if requested == "signed_hash":
        dimension = int(affinity_dim)
        if dimension <= 0:
            raise ValueError("affinity_dim must be positive for signed_hash")
        return (
            deterministic_feature_hash(
                appearance, dimension, batch_size=int(hash_batch_size)
            ),
            deterministic_feature_hash(
                boundary, dimension, batch_size=int(hash_batch_size)
            ),
            {
                "mode": "signed_hash",
                "algorithm": "signed_multiplicative_hash",
                "output_dim": dimension,
                "query_independent": True,
            },
        )
    return (
        appearance,
        boundary,
        {
            "mode": "exact_official_capability",
            "appearance_dim": int(appearance.shape[1]),
            "boundary_dim": int(boundary.shape[1]),
            "query_independent": True,
        },
    )


def estimate_unoriented_local_surface_normals(
    xyz: torch.Tensor,
    *,
    neighbors: int = 24,
    batch_size: int = 8192,
    minimum_planarity: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate label-free local surface relations from canonical centres.

    Normals are deliberately *unoriented*: downstream affinity uses the
    absolute dot product, because a local PCA eigenvector has no stable global
    sign.  The accompanying planarity confidence becomes neutral relation
    weight in edges that are corner-like, linear, or otherwise not reliably
    surface-shaped.  Both outputs depend only on frozen primitive geometry.
    """

    from scipy.spatial import cKDTree

    points = torch.as_tensor(xyz).detach().float().cpu().numpy()
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("local surface normals require finite xyz [N,3]")
    count = len(points)
    if neighbors < 3 or batch_size <= 0 or not 0.0 <= float(minimum_planarity) < 1.0:
        raise ValueError("surface-normal neighbors/batch/planarity arguments are invalid")
    normals = np.zeros((count, 3), dtype=np.float32)
    reliability = np.zeros(count, dtype=np.float32)
    if count < 4:
        return torch.from_numpy(normals), torch.from_numpy(reliability)
    query_count = min(int(neighbors) + 1, count)
    tree = cKDTree(points)
    for start in range(0, count, int(batch_size)):
        stop = min(start + int(batch_size), count)
        _distance, indices = tree.query(points[start:stop], k=query_count, workers=1)
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices[:, None]
        offsets = points[indices] - points[start:stop, None, :]
        covariance = np.einsum("bki,bkj->bij", offsets, offsets, optimize=True)
        covariance /= float(max(1, offsets.shape[1]))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normals[start:stop] = eigenvectors[:, :, 0].astype(np.float32, copy=False)
        # λ0 <= λ1 <= λ2. A plane has a small λ0 and comparable λ1/λ2;
        # line-like or isotropic neighborhoods become neutral rather than a
        # speculative normal boundary.
        planarity = (eigenvalues[:, 1] - eigenvalues[:, 0]) / np.maximum(
            eigenvalues[:, 2], 1e-12
        )
        reliability[start:stop] = np.clip(
            (planarity - float(minimum_planarity))
            / max(1e-8, 1.0 - float(minimum_planarity)),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
    return torch.from_numpy(normals), torch.from_numpy(reliability)


def build(args: argparse.Namespace) -> dict:
    bank = load_canonical_capability_bank(
        args.capability_cache, require_row_authority=True
    )
    capability = {
        "xyz": bank.xyz,
        "valid": bank.valid,
        "appearance_dino_v3": bank.appearance,
        "boundary_sam3": bank.boundary,
        "metadata": bank.metadata,
    }
    capability_valid = bank.valid
    valid = capability_valid
    valid_mask_source = "capability.valid"
    if args.valid_mask_cache:
        mask_payload = torch.load(args.valid_mask_cache, map_location="cpu")
        if args.valid_mask_key not in mask_payload:
            raise ValueError(
                f"valid-mask cache lacks key {args.valid_mask_key!r}"
            )
        valid = torch.as_tensor(mask_payload[args.valid_mask_key]).bool().cpu()
        if valid.shape != capability_valid.shape:
            raise ValueError("override valid mask does not align with capability rows")
        if bool((valid & ~capability_valid).any()):
            raise ValueError("override valid mask must be a capability-valid subset")
        mask_xyz = mask_payload.get("xyz")
        if mask_xyz is not None and not torch.equal(
            torch.as_tensor(mask_xyz).float().cpu(),
            torch.as_tensor(capability["xyz"]).float().cpu(),
        ):
            raise ValueError("override valid-mask geometry does not align")
        valid_mask_source = (
            f"{Path(args.valid_mask_cache).resolve()}:{args.valid_mask_key}"
        )
    global_rows = torch.where(valid)[0]
    xyz = bank.xyz[global_rows]
    if bank.features_are_compact:
        capability_rows = torch.where(capability_valid)[0]
        global_to_capability = torch.full(
            (capability_valid.numel(),), -1, dtype=torch.long
        )
        global_to_capability[capability_rows] = torch.arange(
            capability_rows.numel(), dtype=torch.long
        )
        feature_rows = global_to_capability[global_rows]
        if bool((feature_rows < 0).any()):
            raise ValueError("graph valid rows are absent from compact capability bank")
    else:
        feature_rows = global_rows
    appearance, boundary, affinity_audit = capability_affinity_features(
        capability,
        feature_rows,
        mode=str(args.capability_affinity_mode),
        affinity_dim=int(args.affinity_dim),
        hash_batch_size=int(args.hash_batch_size),
    )
    normals = None
    normal_reliability = None
    surface_relation = str(args.surface_relation)
    if surface_relation in {"local_pca_v1", "local_pca_tangent_v1"}:
        normals, normal_reliability = estimate_unoriented_local_surface_normals(
            xyz,
            neighbors=int(args.surface_normal_neighbors),
            batch_size=int(args.surface_normal_batch_size),
            minimum_planarity=float(args.surface_normal_min_planarity),
        )
    view_observations = None
    covisibility_audit: dict[str, object] = {
        "mode": "none",
        "query_independent": True,
        "labels_opened": False,
    }
    if args.responsibility_cache:
        view_observations, covisibility_audit = load_covisibility_observations(
            capability["metadata"],
            responsibility_cache=args.responsibility_cache,
            num_global_rows=int(capability_valid.numel()),
            global_rows=global_rows,
        )
        covisibility_audit = {"mode": "mpr_top1_view_membership_v1", **covisibility_audit}
    config = SupportGraphConfig(
        neighbors=int(args.neighbors),
        spatial_scale=float(args.spatial_scale),
        appearance_temperature=float(args.appearance_temperature),
        boundary_temperature=float(args.boundary_temperature),
        normal_temperature=float(args.normal_temperature),
        surface_tangent_temperature=float(args.surface_tangent_temperature),
        surface_tangent_relation=surface_relation == "local_pca_tangent_v1",
        surface_topology_min_affinity=float(
            args.surface_topology_min_affinity
        ),
        covisibility_weight=float(args.covisibility_weight),
        require_covisibility_topology=bool(
            args.require_covisibility_topology
        ),
        affinity_chunk_size=int(args.affinity_chunk_size),
        topology_mode=str(args.topology_mode),
    )
    graph = build_primitive_support_graph(
        xyz,
        appearance_features=appearance,
        boundary_features=boundary,
        normals=normals,
        normal_reliability=normal_reliability,
        view_observations=view_observations,
        config=config,
        feature_affinity_device=str(args.affinity_device),
    )
    metadata = {
        "schema_version": 1,
        "source": "canonical_official_dino_sam3_multichannel_support_graph",
        "capability_cache": str(Path(args.capability_cache).resolve()),
        "capability_metadata": capability["metadata"],
        "primitive_row_authority": PrimitiveRowAuthority.from_tensors(
            bank.xyz, valid
        ).to_dict(),
        "valid_mask_source": valid_mask_source,
        # Retain the historic key for readers that audit old graph files, but
        # make a high-fidelity exact route explicit rather than allowing a
        # graph filename to hide a changed affinity representation.
        "feature_hash": (
            affinity_audit
            if affinity_audit["mode"] == "signed_hash"
            else {
                "algorithm": "not_applied",
                "output_dim": 0,
                "query_independent": True,
            }
        ),
        "capability_affinity": affinity_audit,
        "graph_config": asdict(config),
        "surface_relation": {
            "mode": surface_relation,
            "normal_estimator": (
                "unoriented_local_pca_v1"
                if surface_relation in {"local_pca_v1", "local_pca_tangent_v1"}
                else "none"
            ),
            "tangent_continuity": (
                "point_to_local_tangent_plane_v1"
                if surface_relation == "local_pca_tangent_v1"
                else "none"
            ),
            "neighbors": int(args.surface_normal_neighbors),
            "minimum_planarity": float(args.surface_normal_min_planarity),
            "normal_reliability": (
                "local_planarity_blends_uncertain_edges_to_neutral"
                if surface_relation in {"local_pca_v1", "local_pca_tangent_v1"}
                else "not_applicable"
            ),
            "query_independent": True,
            "labels_opened": False,
        },
        "covisibility_relation": covisibility_audit,
        "edge_channels": sorted(graph.edge_channels),
        "legacy_edge_weight": "product_of_all_available_channel_affinities",
        "typed_edge_weight": "arithmetic_mixture_of_row_normalized_channels",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "global_rows": global_rows,
            "num_global_rows": int(valid.numel()),
            "xyz": xyz,
            "edge_index": graph.edge_index,
            "edge_weight": graph.edge_weight.half(),
            "raw_affinity": graph.raw_affinity.half(),
            "edge_channels": {
                name: values.half() for name, values in graph.edge_channels.items()
            },
            "local_sigma": graph.local_sigma,
            "metadata": metadata,
        },
        output,
    )
    report = {
        **metadata,
        "output": str(output),
        "num_nodes": graph.num_nodes,
        "num_edges": int(graph.edge_index.shape[1]),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument(
        "--valid-mask-cache",
        default="",
        help="Optional row-aligned cache supplying a capability-valid subset",
    )
    parser.add_argument(
        "--valid-mask-key",
        default="primary_valid",
        help="Mask key read from --valid-mask-cache",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--spatial-scale", type=float, default=2.0)
    parser.add_argument("--appearance-temperature", type=float, default=0.10)
    parser.add_argument("--boundary-temperature", type=float, default=0.10)
    parser.add_argument("--normal-temperature", type=float, default=0.20)
    parser.add_argument("--surface-tangent-temperature", type=float, default=0.20)
    parser.add_argument(
        "--surface-relation",
        choices=("none", "local_pca_v1", "local_pca_tangent_v1"),
        default="none",
        help=(
            "optional query-independent local surface relation from canonical xyz; "
            "local_pca_tangent_v1 additionally rejects cross-layer shortcuts"
        ),
    )
    parser.add_argument("--surface-normal-neighbors", type=int, default=24)
    parser.add_argument("--surface-normal-batch-size", type=int, default=8192)
    parser.add_argument("--surface-normal-min-planarity", type=float, default=0.0)
    parser.add_argument(
        "--surface-topology-min-affinity",
        type=float,
        default=0.0,
        help=(
            "optional query-free hard topology filter on local tangent "
            "continuity; zero preserves the historical relation-only graph"
        ),
    )
    parser.add_argument(
        "--responsibility-cache",
        default="",
        help=(
            "optional query-free MPR responsibility sidecar; when supplied, "
            "adds an auditable primitive co-visibility relation"
        ),
    )
    parser.add_argument(
        "--covisibility-weight",
        type=float,
        default=0.25,
        help="fixed Jaccard relation strength used only when --responsibility-cache is set",
    )
    parser.add_argument(
        "--require-covisibility-topology",
        action="store_true",
        help=(
            "retain an edge only when its primitives share at least one "
            "registered source view"
        ),
    )
    parser.add_argument("--affinity-dim", type=int, default=256)
    parser.add_argument("--hash-batch-size", type=int, default=8192)
    parser.add_argument(
        "--capability-affinity-mode",
        choices=("signed_hash", "exact_official_capability"),
        default="signed_hash",
        help=(
            "signed_hash preserves the frozen compact graph; "
            "exact_official_capability keeps official DINO/SAM rows for a "
            "separate query-free high-fidelity graph variant"
        ),
    )
    parser.add_argument(
        "--affinity-device",
        default="cpu",
        help=(
            "device used only for capability edge-cosine chunks; topology, "
            "graph artifact, and all query/evaluator data remain unchanged"
        ),
    )
    parser.add_argument("--affinity-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--topology-mode",
        choices=("symmetric_union", "mutual_knn"),
        default="symmetric_union",
    )
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
