#!/usr/bin/env python3
"""GT-only audit of a query-free support graph's instance-boundary topology.

This diagnostic never feeds labels back into graph construction, path costs,
cache generation, or model selection.  It exists to answer a narrower
identifiability question: do the frozen DINO/SAM relation channels separate
same-instance from cross-instance physical neighbours well enough that a
relation-weighted path would be justified at all?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import torch

from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.eval_scannet_3d_point_query import load_scannet_instances
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply


def _support_graph(payload: dict) -> PrimitiveSupportGraph:
    return PrimitiveSupportGraph(
        edge_index=payload["edge_index"], edge_weight=payload["edge_weight"],
        raw_affinity=payload["raw_affinity"], local_sigma=payload["local_sigma"],
        num_nodes=int(torch.as_tensor(payload["xyz"]).shape[0]),
        edge_channels=payload.get("edge_channels", {}),
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: 0.0 for key in ("min", "p05", "p50", "mean", "p95", "p99", "max")}
    return {
        "min": float(values.min()), "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)), "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)), "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _auc_same_vs_cross(same: np.ndarray, cross: np.ndarray) -> float | None:
    """Tie-correct Mann--Whitney AUC without introducing an sklearn dependency."""

    same, cross = np.asarray(same), np.asarray(cross)
    if same.size == 0 or cross.size == 0:
        return None
    ranks = rankdata(np.concatenate([same, cross]), method="average")
    numerator = ranks[:same.size].sum() - same.size * (same.size + 1) / 2.0
    return float(numerator / (same.size * cross.size))


def _majority_primitive_instances(
    xyz: np.ndarray,
    mesh_xyz: np.ndarray,
    mesh_instance_ids: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Transfer GT labels *after* graph construction via mesh-to-primitive NN.

    The majority rule is deliberately only for this audit.  The graph itself
    never observes this array.  Ambiguous primitives are counted explicitly so
    a high cross-edge rate cannot be mistaken for a clean mesh projection.
    """

    mesh_leaf = cKDTree(xyz).query(mesh_xyz, k=1, workers=-1)[1].astype(np.int64)
    positive = np.asarray(mesh_instance_ids, dtype=np.int64) > 0
    labels = np.zeros(len(xyz), dtype=np.int64)
    support = np.zeros(len(xyz), dtype=np.int64)
    ambiguous = np.zeros(len(xyz), dtype=bool)
    leaf, instance = mesh_leaf[positive], np.asarray(mesh_instance_ids, dtype=np.int64)[positive]
    if leaf.size:
        order = np.lexsort((instance, leaf))
        leaf, instance = leaf[order], instance[order]
        unique_leaf, start = np.unique(leaf, return_index=True)
        stop = np.append(start[1:], len(leaf))
        for primitive, left, right in zip(unique_leaf, start, stop):
            candidates, counts = np.unique(instance[left:right], return_counts=True)
            winner = int(counts.argmax())
            labels[int(primitive)] = int(candidates[winner])
            support[int(primitive)] = int(counts[winner])
            ambiguous[int(primitive)] = len(candidates) > 1
    metadata = {
        "mesh_vertices": int(len(mesh_xyz)),
        "positive_instance_mesh_vertices": int(positive.sum()),
        "labeled_primitives": int((labels > 0).sum()),
        "labeled_primitive_fraction": float((labels > 0).mean()),
        "ambiguous_primitives": int(ambiguous.sum()),
        "ambiguous_labeled_primitive_fraction": float(
            ambiguous[labels > 0].mean() if bool((labels > 0).any()) else 0.0
        ),
        "majority_mesh_support": _summary(support[support > 0]),
    }
    return labels, metadata


def _topology(rows: np.ndarray, cols: np.ndarray, count: int) -> dict:
    if rows.size == 0:
        return {
            "undirected_edges": 0, "components": int(count),
            "largest_component_fraction": 1.0 / max(1, int(count)),
            "isolated_node_fraction": 1.0,
            "degree": _summary(np.zeros(count)),
        }
    adjacency = csr_matrix(
        (np.ones(rows.size * 2, dtype=np.uint8),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(count, count),
    )
    component_count, component_id = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(component_id, minlength=component_count)
    degree = np.bincount(
        np.concatenate([rows, cols]), minlength=count
    )
    return {
        "undirected_edges": int(rows.size), "components": int(component_count),
        "largest_component_fraction": float(component_sizes.max() / max(1, count)),
        "isolated_node_fraction": float((degree == 0).mean()),
        "degree": _summary(degree),
    }


def run(args: argparse.Namespace) -> dict:
    payload = torch.load(args.support_graph, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(payload["xyz"]).float().cpu().numpy()
    graph = _support_graph(payload)
    mesh_xyz, _ = _read_label_ply(args.label_ply)
    instance_ids, _metadata = load_scannet_instances(args.aggregation, args.segmentation)
    if len(mesh_xyz) != len(instance_ids):
        raise ValueError("ScanNet label mesh and instance annotations differ in length")
    primitive_instance, projection = _majority_primitive_instances(
        xyz, mesh_xyz, instance_ids
    )

    edge = graph.edge_index.numpy()
    # Support graphs are symmetric; retaining one orientation prevents an
    # artificial factor-of-two in topology and label-pair statistics.
    keep_undirected = edge[0] < edge[1]
    src, dst = edge[0, keep_undirected], edge[1, keep_undirected]
    lengths = np.linalg.norm(xyz[src] - xyz[dst], axis=1)
    src_label, dst_label = primitive_instance[src], primitive_instance[dst]
    known = (src_label > 0) & (dst_label > 0)
    same = known & (src_label == dst_label)
    cross = known & (src_label != dst_label)

    channels = {}
    for name, tensor in graph.edge_channels.items():
        values = torch.as_tensor(tensor).cpu().numpy()[keep_undirected]
        channels[name] = {
            "same_instance": _summary(values[same]),
            "cross_instance": _summary(values[cross]),
            "same_vs_cross_auc": _auc_same_vs_cross(values[same], values[cross]),
        }
    if {"appearance", "boundary"}.issubset(graph.edge_channels):
        appearance = graph.edge_channels["appearance"].numpy()[keep_undirected]
        boundary = graph.edge_channels["boundary"].numpy()[keep_undirected]
        joint = np.sqrt(appearance * boundary)
        channels["appearance_boundary_geometric"] = {
            "same_instance": _summary(joint[same]),
            "cross_instance": _summary(joint[cross]),
            "same_vs_cross_auc": _auc_same_vs_cross(joint[same], joint[cross]),
        }

    gated = np.ones(src.shape[0], dtype=bool)
    if "appearance" in graph.edge_channels:
        gated &= graph.edge_channels["appearance"].numpy()[keep_undirected] >= float(
            args.minimum_appearance_affinity
        )
    if "boundary" in graph.edge_channels:
        gated &= graph.edge_channels["boundary"].numpy()[keep_undirected] >= float(
            args.minimum_boundary_affinity
        )

    def edge_label_stats(mask: np.ndarray) -> dict:
        local_known, local_cross = known[mask], cross[mask]
        return {
            "known_instance_edges": int(local_known.sum()),
            "same_instance_edges": int(same[mask].sum()),
            "cross_instance_edges": int(local_cross.sum()),
            "cross_instance_fraction_among_known": float(
                local_cross.sum() / max(1, local_known.sum())
            ),
            "edge_length_m": _summary(lengths[mask]),
        }

    report = {
        "schema_version": 1,
        "diagnostic_only_gt_audit": True,
        "scene": str(args.scene),
        "support_graph": str(Path(args.support_graph).resolve()),
        "labels_used_only_after_graph_construction": True,
        "mesh_projection": "nearest_support_graph_primitive_then_positive_instance_majority",
        "projection": projection,
        "graph": {
            "nodes": int(graph.num_nodes), "undirected_edges": int(src.size),
            "edge_length_m": _summary(lengths),
            "known_instance_edges": int(known.sum()),
            "same_instance_edges": int(same.sum()),
            "cross_instance_edges": int(cross.sum()),
            "cross_instance_fraction_among_known": float(cross.sum() / max(1, known.sum())),
        },
        "relation_channels": channels,
        "topology": {
            "raw": _topology(src, dst, graph.num_nodes),
            "current_affinity_gate": _topology(src[gated], dst[gated], graph.num_nodes),
        },
        "edge_labels": {
            "raw": edge_label_stats(np.ones(src.shape[0], dtype=bool)),
            "current_affinity_gate": edge_label_stats(gated),
        },
        "current_gate": {
            "minimum_appearance_affinity": float(args.minimum_appearance_affinity),
            "minimum_boundary_affinity": float(args.minimum_boundary_affinity),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--minimum-appearance-affinity", type=float, default=1e-4)
    parser.add_argument("--minimum-boundary-affinity", type=float, default=1e-3)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
