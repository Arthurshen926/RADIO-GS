#!/usr/bin/env python3
"""GT-only audit of frozen SAM3 alpha-adjoint mask purity and coverage.

This opens ScanNet instance annotations only after a query-free membership
sidecar has been written.  It answers a diagnostic question that edge AUC and
component IoU alone cannot: are official 2-D proposals impure after lifting,
or are they locally pure but too incomplete to connect a 3-D object?  The
report is never consumed by cache construction, training, or query inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from radio_gs.scripts.audit_scannet_relation_topology import _majority_primitive_instances
from radio_gs.scripts.eval_scannet_3d_point_query import load_scannet_instances
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)), "min": float(array.min()), "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)), "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()), "mean": float(array.mean()),
    }


def _mask_statistics(
    membership: torch.Tensor,
    radii_m: torch.Tensor,
    primitive_instance: np.ndarray,
    *,
    inside_threshold: float,
    frame: str,
    source_mask_indices: torch.Tensor,
) -> tuple[list[dict], np.ndarray]:
    """Return per-mask purity/completeness and their union support.

    A dominant instance and its completeness are diagnostic labels only.  A
    mask with no known positive instance is retained in the report rather
    than silently treated as a correct object proposal.
    """

    member = torch.as_tensor(membership).float().cpu().numpy()
    radii = torch.as_tensor(radii_m).float().cpu().numpy().reshape(-1)
    indices = torch.as_tensor(source_mask_indices).long().cpu().numpy().reshape(-1)
    labels = np.asarray(primitive_instance, dtype=np.int64).reshape(-1)
    if member.ndim != 2 or member.shape[1] != len(labels) or radii.shape != (member.shape[0],) or indices.shape != (member.shape[0],):
        raise ValueError("membership rows, mask metadata, and primitive labels do not align")
    inside = member >= float(inside_threshold)
    support = inside.any(axis=0)
    instance_size = np.bincount(labels[labels > 0]) if np.any(labels > 0) else np.zeros(0, dtype=np.int64)
    rows: list[dict] = []
    for local_index, selected in enumerate(inside):
        selected_labels = labels[selected]
        known = selected_labels[selected_labels > 0]
        row = {
            "frame": str(frame), "source_mask_index": int(indices[local_index]),
            "physical_radius_m": float(radii[local_index]),
            "inside_primitives": int(selected.sum()), "known_instance_primitives": int(len(known)),
            "dominant_instance_id": 0, "dominant_instance_purity": 0.0,
            "dominant_instance_completeness": 0.0,
        }
        if len(known):
            counts = np.bincount(known)
            dominant = int(counts.argmax())
            dominant_count = int(counts[dominant])
            row.update({
                "dominant_instance_id": dominant,
                "dominant_instance_purity": float(dominant_count / len(known)),
                "dominant_instance_completeness": float(
                    dominant_count / max(1, int(instance_size[dominant]) if dominant < len(instance_size) else 0)
                ),
            })
        rows.append(row)
    return rows, support


def _load_sidecars(paths: list[Path]) -> tuple[list[dict], dict, Path]:
    if not paths:
        raise ValueError("at least one membership sidecar is required")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    reference_metadata: dict | None = None
    records: list[dict] = []
    seen_frames: set[str] = set()
    for path, payload in zip(paths, payloads):
        metadata = dict(payload.get("metadata", {}))
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError(f"{path} must use membership-sidecar schema version 1")
        if not bool(metadata.get("query_free", False)) or not bool(metadata.get("offline_teacher_audit_only", False)):
            raise ValueError(f"{path} lacks query-free offline-audit provenance")
        if not bool(metadata.get("not_an_inference_representation", False)):
            raise ValueError(f"{path} is not explicitly marked non-inference")
        if any(bool(metadata.get(key, False)) for key in ("labels_opened", "instances_opened", "text_opened")):
            raise ValueError(f"{path} was built with benchmark annotations")
        if metadata.get("membership_lifting") != "raster_adjoint" or metadata.get("raster_lifting_semantics") != "true_alpha_compositing_adjoint":
            raise ValueError(f"{path} is not a true raster-adjoint teacher")
        if reference_metadata is None:
            reference_metadata = metadata
        else:
            for key in ("scene_graph", "scene_graph_sha256", "membership_lifting", "raster_lifting_semantics", "inside_threshold", "outside_threshold"):
                if metadata.get(key) != reference_metadata.get(key):
                    raise ValueError(f"membership-sidecar contract differs at {key!r}")
        for record in payload.get("records", []):
            frame = str(record["mask_frame"])
            if frame in seen_frames:
                raise ValueError(f"duplicate membership frame across sidecars: {frame}")
            seen_frames.add(frame); records.append(record)
    assert reference_metadata is not None
    graph_path = Path(str(reference_metadata.get("scene_graph", "")))
    if not graph_path.is_file() or _sha256_file(graph_path) != str(reference_metadata.get("scene_graph_sha256", "")):
        raise ValueError("membership sidecar does not match its frozen scene graph")
    return records, reference_metadata, graph_path


def run(args: argparse.Namespace) -> dict:
    sidecar_paths = [Path(value).resolve() for value in args.membership_sidecars]
    records, metadata, graph_path = _load_sidecars(sidecar_paths)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(graph["xyz"]).float().cpu().numpy()
    mesh_xyz, _ = _read_label_ply(args.label_ply)
    instance_ids, instance_metadata = load_scannet_instances(args.aggregation, args.segmentation)
    if len(mesh_xyz) != len(instance_ids):
        raise ValueError("ScanNet label mesh and instance annotations differ in length")
    primitive_instance, projection = _majority_primitive_instances(xyz, mesh_xyz, instance_ids)

    rows: list[dict] = []
    covered = np.zeros(len(xyz), dtype=bool)
    for record in records:
        local_rows, local_covered = _mask_statistics(
            record["membership"], record["physical_radius_m"], primitive_instance,
            inside_threshold=float(metadata["inside_threshold"]), frame=str(record["mask_frame"]),
            source_mask_indices=record["source_mask_indices"],
        )
        rows.extend(local_rows); covered |= local_covered
    if not rows:
        raise RuntimeError("membership sidecar contains no metric SAM3 masks")

    positive = primitive_instance > 0
    per_instance = []
    for instance_id in sorted(instance_metadata):
        nodes = primitive_instance == int(instance_id)
        if not bool(nodes.any()):
            continue
        candidate_rows = [row for row in rows if row["dominant_instance_id"] == int(instance_id)]
        per_instance.append({
            "instance_id": int(instance_id), "label": str(instance_metadata[instance_id]["label"]),
            "primitive_nodes": int(nodes.sum()), "covered_fraction": float(covered[nodes].mean()),
            "best_mask_completeness": float(max((row["dominant_instance_completeness"] for row in candidate_rows), default=0.0)),
            "best_mask_purity": float(max((row["dominant_instance_purity"] for row in candidate_rows), default=0.0)),
            "dominant_mask_count": len(candidate_rows),
        })
    purity = [row["dominant_instance_purity"] for row in rows if row["known_instance_primitives"]]
    completeness = [row["dominant_instance_completeness"] for row in rows if row["known_instance_primitives"]]
    report = {
        "schema_version": 1, "diagnostic_only_gt_audit": True,
        "labels_used_only_after_teacher_construction": True,
        "membership_sidecars": [str(path) for path in sidecar_paths],
        "membership_lifting": metadata["membership_lifting"],
        "raster_lifting_semantics": metadata["raster_lifting_semantics"],
        "inside_threshold": float(metadata["inside_threshold"]),
        "projection": projection,
        "mask_summary": {
            "masks": len(rows),
            "physical_radius_m": _quantiles([row["physical_radius_m"] for row in rows]),
            "inside_primitives": _quantiles([float(row["inside_primitives"]) for row in rows]),
            "dominant_instance_purity": _quantiles(purity),
            "dominant_instance_completeness": _quantiles(completeness),
            "fraction_purity_ge_0_90": float(np.mean(np.asarray(purity) >= 0.90)) if purity else 0.0,
            "fraction_completeness_ge_0_50": float(np.mean(np.asarray(completeness) >= 0.50)) if completeness else 0.0,
        },
        "primitive_coverage": {
            "labeled_primitive_fraction_covered": float(covered[positive].mean()) if bool(positive.any()) else 0.0,
            "instances": len(per_instance),
            "instance_covered_fraction": _quantiles([row["covered_fraction"] for row in per_instance]),
            "best_mask_completeness": _quantiles([row["best_mask_completeness"] for row in per_instance]),
        },
        "masks": rows, "instances": per_instance,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership-sidecars", nargs="+", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--label-ply", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
