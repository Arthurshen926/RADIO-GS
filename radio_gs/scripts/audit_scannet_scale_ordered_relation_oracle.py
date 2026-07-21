#!/usr/bin/env python3
"""GT-only audit for a query-free scale-ordered relation teacher.

This diagnostic is intentionally downstream of teacher-cache construction.
ScanNet instance labels are opened only after all SAM3 masks, renderer
memberships, and interval constraints have been frozen.  It reports two
separate facts rather than using GT to choose a relation rule:

* scale-binned same/cross-instance edge separation among observed constraints;
* a conservative, monotonic component oracle from *upper* merge bounds.

The latter only joins an edge once a same-mask upper bound certifies it.  A
separate-only lower bound never becomes a fabricated positive edge.  Thus the
oracle is a diagnostic of teacher coverage and relation correctness, not a
post-hoc tuned prompt solver or a proposed final inference backend.
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

from radio_gs.scripts.audit_scannet_relation_topology import _majority_primitive_instances
from radio_gs.scripts.eval_scannet_3d_point_query import load_scannet_instances
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply


def _same_vs_cross_auc(same: np.ndarray, cross: np.ndarray) -> float | None:
    """Tie-correct Mann--Whitney AUC, with higher scores favouring ``same``."""

    same, cross = np.asarray(same, dtype=np.float64), np.asarray(cross, dtype=np.float64)
    if not len(same) or not len(cross):
        return None
    ranks = rankdata(np.concatenate([same, cross]), method="average")
    numerator = ranks[: len(same)].sum() - len(same) * (len(same) + 1) / 2.0
    return float(numerator / (len(same) * len(cross)))


def conservative_join_mask(
    upper_log_radius: torch.Tensor,
    interval_consistent: torch.Tensor,
    *,
    log_radius: float,
) -> torch.Tensor:
    """Return scale-monotonic joins certified by a same-mask upper bound.

    ``mu <= upper`` is the only positive conclusion available from a same-mask
    observation.  A lower bound says only ``mu > lower`` and is therefore
    intentionally excluded here.  This makes the diagnostic fail *closed* on
    sparse evidence rather than pretending absent membership is a merge.
    """

    upper = torch.as_tensor(upper_log_radius).float()
    consistent = torch.as_tensor(interval_consistent).bool()
    if upper.shape != consistent.shape:
        raise ValueError("upper merge bounds and interval consistency do not align")
    return torch.isfinite(upper) & consistent & (upper <= float(log_radius))


def _component_labels(
    edge_index: np.ndarray, join: np.ndarray, node_count: int,
) -> tuple[np.ndarray, dict]:
    src, dst = edge_index[:, join]
    if not len(src):
        return np.arange(node_count, dtype=np.int64), {
            "joined_edges": 0, "components": int(node_count), "largest_component_fraction": 1.0 / max(1, node_count),
        }
    adjacency = csr_matrix(
        (
            np.ones(len(src) * 2, dtype=np.uint8),
            (np.concatenate([src, dst]), np.concatenate([dst, src])),
        ),
        shape=(node_count, node_count),
    )
    component_count, component = connected_components(adjacency, directed=False)
    counts = np.bincount(component, minlength=component_count)
    return component.astype(np.int64), {
        "joined_edges": int(len(src)), "components": int(component_count),
        "largest_component_fraction": float(counts.max() / max(1, node_count)),
    }


def _queries(
    instance_ids: np.ndarray,
    metadata: dict,
    mesh_leaf: np.ndarray,
    *,
    minimum_vertices: int,
    seed: int,
    maximum_instances: int,
) -> list[dict]:
    result: list[dict] = []
    for instance_id in sorted(metadata):
        target = np.flatnonzero(instance_ids == int(instance_id))
        if len(target) < int(minimum_vertices):
            continue
        rng = np.random.default_rng(int(seed) + 1_000_003 * int(instance_id))
        seed_vertex = int(target[int(rng.integers(len(target)))])
        result.append({
            "instance_id": int(instance_id), "label": str(metadata[instance_id]["label"]),
            "target_rows": target, "seed_vertex": seed_vertex,
            "seed_leaf": int(mesh_leaf[seed_vertex]),
        })
    if int(maximum_instances) > 0:
        result = result[: int(maximum_instances)]
    if not result:
        raise RuntimeError("no ScanNet instance satisfies the oracle minimum")
    return result


def _component_metrics(
    component: np.ndarray,
    mesh_leaf: np.ndarray,
    query: dict,
) -> dict:
    predicted = component[mesh_leaf] == component[int(query["seed_leaf"])]
    target = np.zeros(len(mesh_leaf), dtype=bool)
    target[np.asarray(query["target_rows"], dtype=np.int64)] = True
    intersection = int(np.logical_and(predicted, target).sum())
    predicted_count, target_count = int(predicted.sum()), int(target.sum())
    union = int(np.logical_or(predicted, target).sum())
    return {
        "iou": float(intersection / union) if union else 0.0,
        "precision": float(intersection / predicted_count) if predicted_count else 0.0,
        "recall": float(intersection / target_count) if target_count else 0.0,
        "predicted_mesh_vertices": predicted_count,
        "target_mesh_vertices": target_count,
    }


def _edge_audit(
    *,
    same_votes: np.ndarray,
    separate_votes: np.ndarray,
    primitive_instance: np.ndarray,
    edge_index: np.ndarray,
    bin_edges_log: np.ndarray,
) -> list[dict]:
    src, dst = edge_index
    known = (primitive_instance[src] > 0) & (primitive_instance[dst] > 0)
    same_instance = known & (primitive_instance[src] == primitive_instance[dst])
    cross_instance = known & ~same_instance
    rows: list[dict] = []
    for index in range(same_votes.shape[1]):
        same, separate = same_votes[:, index], separate_votes[:, index]
        observed = (same + separate) > 0
        score = same / np.maximum(same + separate, 1e-12)
        rows.append({
            "bin": int(index),
            "radius_interval_m": [float(np.exp(bin_edges_log[index])), float(np.exp(bin_edges_log[index + 1]))],
            "observed_known_edges": int((observed & known).sum()),
            "observed_same_instance_edges": int((observed & same_instance).sum()),
            "observed_cross_instance_edges": int((observed & cross_instance).sum()),
            "same_instance_observation_fraction": float(
                (observed & same_instance).sum() / max(1, same_instance.sum())
            ),
            "cross_instance_observation_fraction": float(
                (observed & cross_instance).sum() / max(1, cross_instance.sum())
            ),
            "same_vs_cross_auc": _same_vs_cross_auc(
                score[observed & same_instance], score[observed & cross_instance],
            ),
        })
    return rows


def run(args: argparse.Namespace) -> dict:
    cache_path = Path(args.relation_cache)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("relation cache must use scale-ordered schema v2")
    if not bool(metadata.get("query_free", False)):
        raise ValueError("relation teacher does not declare query-free provenance")
    if any(bool(metadata.get(key, False)) for key in ("labels_opened", "instances_opened", "text_opened")):
        raise ValueError("relation teacher was built with benchmark annotations")

    edge = torch.as_tensor(payload["edge_index"]).long().cpu().numpy()
    same_votes = torch.as_tensor(payload["same_votes"]).float().cpu().numpy()
    separate_votes = torch.as_tensor(payload["separate_votes"]).float().cpu().numpy()
    upper = torch.as_tensor(payload["upper_log_radius"]).float().cpu()
    consistent = torch.as_tensor(payload["interval_consistent"]).bool().cpu()
    bins = torch.as_tensor(payload["scale_bin_edges_log"]).float().cpu().numpy()
    if edge.ndim != 2 or edge.shape[0] != 2 or same_votes.shape != separate_votes.shape:
        raise ValueError("malformed edge-aligned scale relation cache")
    if same_votes.shape != (edge.shape[1], len(bins) - 1):
        raise ValueError("scale vote bins do not align with relation edges")

    graph_path = Path(str(metadata.get("scene_graph", "")))
    if not graph_path.is_file():
        raise ValueError("teacher cache lacks its source graph for GT-only projection")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(graph["xyz"]).float().cpu().numpy()
    if len(xyz) <= int(edge.max(initial=-1)):
        raise ValueError("relation edges exceed source graph rows")
    mesh_xyz, _ = _read_label_ply(args.label_ply)
    instance_ids, instance_metadata = load_scannet_instances(args.aggregation, args.segmentation)
    if len(mesh_xyz) != len(instance_ids):
        raise ValueError("ScanNet label mesh and instance annotations differ in length")
    mesh_leaf = cKDTree(xyz).query(mesh_xyz, k=1, workers=-1)[1].astype(np.int64)
    primitive_instance, projection = _majority_primitive_instances(xyz, mesh_xyz, instance_ids)
    edge_audit = _edge_audit(
        same_votes=same_votes, separate_votes=separate_votes,
        primitive_instance=primitive_instance, edge_index=edge, bin_edges_log=bins,
    )
    queries = _queries(
        instance_ids, instance_metadata, mesh_leaf,
        minimum_vertices=int(args.minimum_instance_vertices), seed=int(args.seed),
        maximum_instances=int(args.maximum_instances),
    )

    scale_records = []
    per_query = [
        {"instance_id": row["instance_id"], "label": row["label"], "seed_vertex": row["seed_vertex"], "seed_leaf": row["seed_leaf"], "per_scale": {}}
        for row in queries
    ]
    for index, log_radius in enumerate(0.5 * (bins[:-1] + bins[1:])):
        joined = conservative_join_mask(upper, consistent, log_radius=float(log_radius)).numpy()
        component, component_report = _component_labels(edge, joined, len(xyz))
        ious, precision, recall = [], [], []
        seed_nontrivial = []
        for record, query in zip(per_query, queries):
            metrics = _component_metrics(component, mesh_leaf, query)
            ious.append(metrics["iou"])
            precision.append(metrics["precision"])
            recall.append(metrics["recall"])
            nontrivial = bool(component_report["joined_edges"] and np.sum(component == component[query["seed_leaf"]]) > 1)
            seed_nontrivial.append(nontrivial)
            record["per_scale"][str(index)] = {
                "radius_m": float(np.exp(log_radius)), **metrics,
                "nontrivial_seed_component": nontrivial,
            }
        scale_records.append({
            "bin": int(index), "radius_m": float(np.exp(log_radius)),
            **component_report,
            "macro_iou": float(np.mean(ious)),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "nontrivial_seed_component_fraction": float(np.mean(seed_nontrivial)),
        })
    oracle, oracle_precision, oracle_recall = [], [], []
    for record in per_query:
        values = record["per_scale"]
        best_key = max(values, key=lambda key: values[key]["iou"])
        record["scale_oracle_iou"] = float(values[best_key]["iou"])
        record["scale_oracle_precision"] = float(values[best_key]["precision"])
        record["scale_oracle_recall"] = float(values[best_key]["recall"])
        record["scale_oracle_bin"] = int(best_key)
        oracle.append(record["scale_oracle_iou"])
        oracle_precision.append(record["scale_oracle_precision"])
        oracle_recall.append(record["scale_oracle_recall"])

    report = {
        "schema_version": 1, "diagnostic_only_gt_audit": True,
        "labels_used_only_after_teacher_construction": True,
        "scene": str(args.scene), "relation_cache": str(cache_path.resolve()),
        "membership_lifting": metadata.get("membership_lifting"),
        "raster_lifting_semantics": metadata.get("raster_lifting_semantics", ""),
        "component_rule": "join_only_consistent_same_mask_upper_bound_at_or_below_scale",
        "mesh_projection": "nearest_relation_primitive_for_vertices; majority_instance_for_edges",
        "projection": projection,
        "edge_scale_audit": edge_audit,
        "component_oracle": {
            "minimum_instance_vertices": int(args.minimum_instance_vertices),
            "num_instances": len(per_query), "scales": scale_records,
            "macro_scale_oracle_iou": float(np.mean(oracle)),
            "macro_scale_oracle_precision": float(np.mean(oracle_precision)),
            "macro_scale_oracle_recall": float(np.mean(oracle_recall)),
            "instances": per_query,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--relation-cache", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--label-ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-instance-vertices", type=int, default=100)
    parser.add_argument("--maximum-instances", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
