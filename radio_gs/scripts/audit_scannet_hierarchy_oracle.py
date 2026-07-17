#!/usr/bin/env python3
"""GT-only diagnostic for a query-free maximum-spanning region hierarchy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from radio_gs.interfaces.relation_calibrator import (
    MonotonicRelationCalibrator,
    edge_relation_features,
)
from radio_gs.scripts.eval_scannet_3d_point_query import load_scannet_instances
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply


def _maximum_spanning_tree(
    payload: dict, weights: dict[str, float], relation_calibrator: str = "",
    relation_private_code: str = "", support_graph_path: str = "",
):
    edge = torch.as_tensor(payload["edge_index"]).long().numpy()
    keep = edge[0] < edge[1]
    src, dst = edge[0, keep], edge[1, keep]
    channels = payload.get("edge_channels", {})
    if relation_private_code:
        checkpoint = torch.load(relation_private_code, map_location="cpu", weights_only=False)
        if str(checkpoint.get("scene")) != str(payload.get("scene")):
            raise ValueError("relation-private code belongs to another scene")
        if int(checkpoint.get("num_graph_edges", -1)) != edge.shape[1]:
            raise ValueError("relation-private code edge count differs from graph")
        if support_graph_path and checkpoint.get("scene_graph_sha256") != hashlib.sha256(
            Path(support_graph_path).read_bytes()
        ).hexdigest():
            raise ValueError("relation-private code graph digest mismatch")
        base_path = checkpoint.get("provenance", {}).get("global_calibrator", "")
        if base_path:
            base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
            model = MonotonicRelationCalibrator()
            model.load_state_dict(base_checkpoint["state_dict"]); model.eval()
            with torch.no_grad():
                full_score = model(edge_relation_features(payload))
        else:
            full_score = torch.zeros(edge.shape[1])
        residual = torch.as_tensor(checkpoint.get("edge_residual")).float()
        if residual.shape != full_score.shape:
            raise ValueError("relation-private edge residual is malformed")
        score = (full_score + residual).numpy()[keep]
    elif relation_calibrator:
        checkpoint = torch.load(relation_calibrator, map_location="cpu")
        model = MonotonicRelationCalibrator()
        model.load_state_dict(checkpoint["state_dict"]); model.eval()
        with torch.no_grad():
            score = model(edge_relation_features(payload)).numpy()[keep]
    else:
        score = np.zeros(src.shape[0], dtype=np.float32)
        total_weight = sum(weights.values())
        for name, weight in weights.items():
            if name not in channels:
                raise ValueError(f"support graph lacks hierarchy channel {name!r}")
            values = torch.as_tensor(channels[name]).float().numpy()[keep]
            score += float(weight) / total_weight * np.log(np.maximum(values, 1e-12))
    order = np.argsort(-score, kind="stable")
    count = int(payload["xyz"].shape[0]); maximum = 2 * count
    dsu = np.arange(count, dtype=np.int64)
    rank = np.zeros(count, dtype=np.int8)
    component_node = np.arange(count, dtype=np.int64)
    parent = np.full(maximum, -1, dtype=np.int64)
    left = np.full(maximum, -1, dtype=np.int64)
    right = np.full(maximum, -1, dtype=np.int64)
    merge_score = np.full(maximum, np.nan, dtype=np.float32)

    def find(value: int) -> int:
        root = value
        while dsu[root] != root: root = int(dsu[root])
        while dsu[value] != value:
            nxt = int(dsu[value]); dsu[value] = root; value = nxt
        return root

    next_node = count
    for edge_index in order:
        a, b = find(int(src[edge_index])), find(int(dst[edge_index]))
        if a == b: continue
        node_a, node_b = int(component_node[a]), int(component_node[b])
        node = next_node; next_node += 1
        left[node], right[node] = node_a, node_b
        parent[node_a] = parent[node_b] = node
        merge_score[node] = score[edge_index]
        if rank[a] < rank[b]: a, b = b, a
        dsu[b] = a
        if rank[a] == rank[b]: rank[a] += 1
        component_node[a] = node
    return (
        parent[:next_node], left[:next_node], right[:next_node],
        merge_score[:next_node], count,
    )


def _aggregate(leaves: np.ndarray, leaf_count: int, left, right) -> np.ndarray:
    values = np.bincount(leaves, minlength=leaf_count).astype(np.int64)
    result = np.zeros(len(left), dtype=np.int64); result[:leaf_count] = values
    for node in range(leaf_count, len(left)):
        result[node] = result[left[node]] + result[right[node]]
    return result


def run(args: argparse.Namespace) -> dict:
    payload = torch.load(args.support_graph, map_location="cpu")
    weights = {
        "geometry": float(args.geometry_weight),
        "appearance": float(args.appearance_weight),
        "boundary": float(args.boundary_weight),
    }
    weights = {name: value for name, value in weights.items() if value > 0}
    parent, left, right, merge_score, leaf_count = _maximum_spanning_tree(
        payload, weights, str(args.relation_calibrator),
        str(args.relation_private_code), str(args.support_graph),
    )
    mesh_xyz, _labels = _read_label_ply(args.label_ply)
    instance_ids, metadata = load_scannet_instances(args.aggregation, args.segmentation)
    if len(mesh_xyz) != len(instance_ids):
        raise ValueError("ScanNet mesh and instance annotation rows differ")
    primitive_xyz = torch.as_tensor(payload["xyz"]).float().numpy()
    mesh_leaf = cKDTree(primitive_xyz).query(mesh_xyz, k=1)[1].astype(np.int64)
    total = _aggregate(mesh_leaf, leaf_count, left, right)
    queries = []
    for instance_id in sorted(metadata):
        target_vertices = np.flatnonzero(instance_ids == int(instance_id))
        if target_vertices.size < int(args.minimum_instance_vertices): continue
        target = _aggregate(mesh_leaf[target_vertices], leaf_count, left, right)
        union = total + target_vertices.size - target
        iou = target / np.maximum(union, 1)
        best_node = int(np.argmax(iou))
        rng = np.random.default_rng(int(args.seed) + 1_000_003 * int(instance_id))
        seed_vertex = int(target_vertices[int(rng.integers(0, target_vertices.size))])
        seed_leaf = int(mesh_leaf[seed_vertex]); chain = []
        node = seed_leaf
        while node >= 0:
            chain.append(node); node = int(parent[node])
        chain_array = np.asarray(chain, dtype=np.int64)
        chain_best = int(chain_array[np.argmax(iou[chain_array])])
        queries.append({
            "instance_id": int(instance_id), "label": metadata[instance_id]["label"],
            "vertices": int(target_vertices.size), "seed_vertex": seed_vertex,
            "seed_leaf": seed_leaf, "ancestor_count": len(chain),
            "global_region_oracle_iou": float(iou[best_node]),
            "global_region_oracle_node": best_node,
            "one_click_ancestor_oracle_iou": float(iou[chain_best]),
            "one_click_ancestor_oracle_node": chain_best,
        })
    if not queries: raise RuntimeError("no ScanNet instances satisfy the oracle protocol")
    report = {
        "schema_version": 1, "diagnostic_only_gt_oracle": True,
        "scene": str(args.scene), "support_graph": str(Path(args.support_graph).resolve()),
        "hierarchy": "maximum_spanning_forest_over_query_free_typed_edge_channels",
        "edge_weights": weights, "leaf_primitives": leaf_count,
        "relation_calibrator": str(Path(args.relation_calibrator).resolve()) if args.relation_calibrator else "",
        "relation_private_code": str(Path(args.relation_private_code).resolve()) if args.relation_private_code else "",
        "tree_nodes": len(parent), "forest_roots": int((parent < 0).sum()),
        "cold_storage_bytes_parent_merge": int(parent.nbytes + merge_score.nbytes),
        "mesh_projection": "nearest canonical primitive leaf",
        "minimum_instance_vertices": int(args.minimum_instance_vertices),
        "macro_global_region_oracle_iou": float(np.mean([q["global_region_oracle_iou"] for q in queries])),
        "macro_one_click_ancestor_oracle_iou": float(np.mean([q["one_click_ancestor_oracle_iou"] for q in queries])),
        "queries": queries,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--label-ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-weight", type=float, default=0.2)
    parser.add_argument("--appearance-weight", type=float, default=0.4)
    parser.add_argument("--boundary-weight", type=float, default=0.4)
    parser.add_argument("--minimum-instance-vertices", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relation-calibrator", default="")
    parser.add_argument("--relation-private-code", default="")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__": main()
