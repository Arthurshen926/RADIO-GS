"""Materialize sealed query-independent higher-order edge features for AGILE v10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.scripts.train_agile3d_seed_conditioned_graph_residual_v7 import _load_graph
from radio_gs.scripts.train_query_likelihood_head_fixed import _sha256, _write_json_no_clobber, _write_torch_no_clobber


ARTIFACT = "agile3d-query-independent-higher-order-edge-features-v10"
FEATURE_NAMES = [
    "appearance_similarity",
    "boundary_similarity",
    "geometry_affinity",
    "scaled_distance",
    "endpoint_reliability",
    "endpoint_coverage",
    "principal_normal_alignment",
    "tangent_offset_compatibility",
    "scale_continuity",
    "anisotropy_continuity",
    "opacity_continuity",
    "mutual_knn",
    "shared_neighbor_jaccard",
    "two_hop_context_continuity"
]


def _safe_search(sorted_values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_values, queries)
    safe = np.minimum(positions, max(0, len(sorted_values) - 1))
    return (positions < len(sorted_values)) & (sorted_values[safe] == queries)


@torch.inference_mode()
def build_symmetric_features(
    *,
    graph,
    bundle: dict[str, object],
    device: torch.device,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    row_all, col_all = graph.edge_index
    unique = row_all < col_all
    edge_index = graph.edge_index[:, unique].long().cpu()
    row, col = edge_index
    count = graph.num_nodes
    xyz = torch.as_tensor(bundle["primitive_xyz"]).float().cpu()
    covariance = torch.as_tensor(bundle["primitive_covariance"], device=device).float()
    eigenvalues, eigenvectors = torch.linalg.eigh(
        covariance + 1e-8 * torch.eye(3, device=device)
    )
    eigenvalues = eigenvalues.clamp_min(1e-10)
    normals = eigenvectors[:, :, 0]
    scale = eigenvalues.mean(dim=1).sqrt()
    anisotropy = (eigenvalues[:, -1] / eigenvalues[:, 0]).sqrt().clamp_max(1e6)
    opacity = torch.as_tensor(bundle["primitive_opacity"], device=device).float().reshape(-1).clamp(1e-5, 1 - 1e-5)
    xyz_device = xyz.to(device)
    reliability = torch.as_tensor(bundle["reliability"], device=device).float().reshape(-1)
    coverage = torch.as_tensor(bundle["coverage"], device=device).float().reshape(-1)

    from scipy.spatial import cKDTree

    neighbors = np.asarray(
        cKDTree(xyz.numpy()).query(xyz.numpy(), k=min(17, count))[1][:, 1:],
        dtype=np.int64,
    )
    directed_row = np.repeat(np.arange(count, dtype=np.int64), neighbors.shape[1])
    directed_col = neighbors.reshape(-1)
    codes = np.sort(directed_row * count + directed_col)
    row_np, col_np = row.numpy(), col.numpy()
    mutual = _safe_search(codes, row_np * count + col_np) & _safe_search(
        codes, col_np * count + row_np
    )

    edge_row_all = graph.edge_index[0].cpu()
    app_all = graph.edge_channels["appearance"].float().cpu()
    boundary_all = graph.edge_channels["boundary"].float().cpu()
    degree = torch.zeros(count)
    app_mean = torch.zeros(count)
    boundary_mean = torch.zeros(count)
    degree.index_add_(0, edge_row_all, torch.ones_like(app_all))
    app_mean.index_add_(0, edge_row_all, app_all)
    boundary_mean.index_add_(0, edge_row_all, boundary_all)
    app_mean /= degree.clamp_min(1)
    boundary_mean /= degree.clamp_min(1)
    app_mean_device = app_mean.to(device)
    boundary_mean_device = boundary_mean.to(device)

    output = torch.empty((edge_index.shape[1], len(FEATURE_NAMES)), dtype=torch.float16)
    unique_indices = torch.nonzero(unique, as_tuple=False).flatten()
    for start in range(0, edge_index.shape[1], int(chunk_size)):
        stop = min(start + int(chunk_size), edge_index.shape[1])
        src = row[start:stop].to(device)
        dst = col[start:stop].to(device)
        directed = unique_indices[start:stop]
        appearance = graph.edge_channels["appearance"][directed].float().to(device).clamp(0, 1)
        boundary = graph.edge_channels["boundary"][directed].float().to(device).clamp(0, 1)
        geometry = graph.edge_channels["geometry"][directed].float().to(device).clamp(1e-12, 1)
        distance = torch.sqrt((-2.0 * geometry.log()).clamp_min(0))
        distance = distance / (1.0 + distance)
        endpoint_reliability = torch.sqrt(reliability[src] * reliability[dst])
        endpoint_coverage = torch.sqrt(coverage[src] * coverage[dst])
        normal_alignment = (normals[src] * normals[dst]).sum(dim=1).abs().clamp(0, 1)
        delta = xyz_device[dst] - xyz_device[src]
        direction = delta / delta.norm(dim=1, keepdim=True).clamp_min(1e-8)
        normal_offset = torch.maximum(
            (direction * normals[src]).sum(dim=1).abs(),
            (direction * normals[dst]).sum(dim=1).abs(),
        ).clamp(0, 1)
        tangent = 1.0 - normal_offset
        scale_continuity = torch.exp(-torch.abs(torch.log(scale[src] / scale[dst])))
        anisotropy_continuity = torch.exp(
            -torch.abs(torch.log(anisotropy[src] / anisotropy[dst]))
        )
        opacity_continuity = torch.exp(
            -torch.abs(torch.logit(opacity[src]) - torch.logit(opacity[dst]))
        )
        left = torch.from_numpy(neighbors[row_np[start:stop]]).to(device)
        right = torch.from_numpy(neighbors[col_np[start:stop]]).to(device)
        intersection = (left[:, :, None] == right[:, None, :]).any(dim=2).sum(dim=1).float()
        jaccard = intersection / (2 * neighbors.shape[1] - intersection).clamp_min(1)
        local_app = 1.0 - torch.abs(app_mean_device[src] - app_mean_device[dst])
        local_boundary = 1.0 - torch.abs(
            boundary_mean_device[src] - boundary_mean_device[dst]
        )
        context = 0.5 * (local_app + local_boundary)
        matrix = torch.stack(
            [
                appearance,
                boundary,
                geometry,
                distance,
                endpoint_reliability,
                endpoint_coverage,
                normal_alignment,
                tangent,
                scale_continuity,
                anisotropy_continuity,
                opacity_continuity,
                torch.from_numpy(mutual[start:stop]).to(device).float(),
                jaccard,
                context,
            ],
            dim=1,
        ).clamp(0, 1)
        output[start:stop] = matrix.half().cpu()
    return edge_index.to(torch.int32), output, {
        "node_count": count,
        "unique_edge_count": int(edge_index.shape[1]),
        "knn_neighbors": int(neighbors.shape[1]),
        "view_support_sidecar_present": False,
        "co_visibility_feature_included": False,
    }


def materialize(args: argparse.Namespace) -> tuple[Path, Path]:
    graph_path = Path(args.typed_graph).resolve()
    bundle_path = Path(args.primitive_bundle).resolve()
    if _sha256(graph_path) != args.typed_graph_sha256 or _sha256(bundle_path) != args.primitive_bundle_sha256:
        raise ValueError("v10 feature authority SHA differs")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    scene_id = str(bundle["scene_id"])
    safety = bundle.get("safety", {})
    if (
        safety.get("query_independent") is not True
        or safety.get("gt_labels_opened") is not False
        or safety.get("test_labels_opened") is not False
        or safety.get("point_as_primitive_used") is not False
    ):
        raise PermissionError("v10 bundle violates query-independent contract")
    graph, graph_payload = _load_graph(graph_path, scene_id=scene_id, device=torch.device("cpu"))
    edge_index, features, inventory = build_symmetric_features(
        graph=graph,
        bundle=bundle,
        device=torch.device(args.device),
        chunk_size=args.chunk_size,
    )
    artifact = {
        "schema_version": 10,
        "artifact_type": ARTIFACT,
        "scene_id": scene_id,
        "feature_names": FEATURE_NAMES,
        "edge_index": edge_index,
        "features": features,
        "inventory": inventory,
        "typed_graph": {"path": str(graph_path), "sha256": _sha256(graph_path)},
        "primitive_bundle": {"path": str(bundle_path), "sha256": _sha256(bundle_path)},
        "graph_safety": graph_payload["safety"],
        "safety": {
            "query_independent": True,
            "labels_opened": False,
            "clicks_opened": False,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
        },
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = _write_torch_no_clobber(output_dir / f"{scene_id}.higher_order_edges_v10.pt", artifact)
    receipt = {
        "schema_version": 1,
        "artifact_type": "agile3d-higher-order-edge-feature-materialization-receipt-v10",
        "scene_id": scene_id,
        "artifact": {"path": str(artifact_path), "sha256": _sha256(artifact_path)},
        "feature_names": FEATURE_NAMES,
        "inventory": inventory,
        "safety": artifact["safety"],
    }
    receipt_path = _write_json_no_clobber(output_dir / f"{scene_id}.higher_order_edges_v10.receipt.json", receipt)
    return artifact_path, receipt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--typed-graph", required=True)
    parser.add_argument("--typed-graph-sha256", required=True)
    parser.add_argument("--primitive-bundle", required=True)
    parser.add_argument("--primitive-bundle-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=65536)
    return parser.parse_args()


def main() -> None:
    artifact, receipt = materialize(parse_args())
    print(json.dumps({"artifact": str(artifact), "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
