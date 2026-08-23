#!/usr/bin/env python3
"""Audit local semantic-support geometry without opening benchmark outputs.

This is a G0 diagnostic, not geometry optimization.  It measures whether one
frozen Gaussian ellipsoid overlaps incompatible official source semantics and
materializes a query-independent ambiguity/risk authority for later readout
ablation.  It intentionally does not unfreeze or split scene geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch

from radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field import (
    _gaussian_covariances,
    _load_geometry_model,
)
from radio_gs.config import load_config
from radio_gs.data.scannet_source_region_semantics import (
    load_scannet_raw_to_nyu40,
    official_vertex_nyu40_labels,
    sha256_file,
)
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS


SCHEMA = "radio_gs.scannet_semantic_support_geometry_audit.v1"


def support_statistics(
    *,
    gaussian_xyz: np.ndarray,
    gaussian_covariance: np.ndarray,
    gaussian_opacity: np.ndarray,
    mesh_xyz: np.ndarray,
    mesh_labels: np.ndarray,
    class_ids: list[int],
    neighbors: int = 32,
    chunk_size: int = 8192,
    maximum_surface_distance: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compute ellipsoid-aware semantic purity and risk in bounded chunks."""

    xyz = np.asarray(gaussian_xyz, dtype=np.float64)
    covariance = np.asarray(gaussian_covariance, dtype=np.float64)
    opacity = np.asarray(gaussian_opacity, dtype=np.float64).reshape(-1)
    surface = np.asarray(mesh_xyz, dtype=np.float64)
    labels = np.asarray(mesh_labels, dtype=np.int64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("gaussian_xyz must be [N,3]")
    if covariance.shape != (xyz.shape[0], 3, 3):
        raise ValueError("gaussian_covariance must be [N,3,3]")
    if opacity.shape != (xyz.shape[0],):
        raise ValueError("gaussian_opacity must align with gaussian_xyz")
    if surface.ndim != 2 or surface.shape[1] != 3 or labels.shape[0] != surface.shape[0]:
        raise ValueError("mesh coordinates and labels must align")
    if not 1 <= int(neighbors) <= surface.shape[0]:
        raise ValueError("neighbors must fit the mesh")

    class_to_column = {int(value): index for index, value in enumerate(class_ids)}
    class_to_column[0] = len(class_ids)
    column_count = len(class_ids) + 1
    tree = cKDTree(surface)
    outputs = {
        name: np.empty(xyz.shape[0], dtype=np.float32)
        for name in (
            "surface_distance",
            "support_sigma_max",
            "semantic_purity",
            "semantic_entropy",
            "boundary_ambiguity",
            "geometry_risk",
            "evidence_authority",
            "joint_risk",
        )
    }
    support_valid = np.empty(xyz.shape[0], dtype=np.bool_)
    dominant = np.full(xyz.shape[0], -1, dtype=np.int16)

    for start in range(0, xyz.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), xyz.shape[0])
        center = xyz[start:stop]
        distance, indices = tree.query(center, k=int(neighbors), workers=-1)
        if int(neighbors) == 1:
            distance = distance[:, None]
            indices = indices[:, None]
        delta = surface[indices] - center[:, None, :]
        cov = covariance[start:stop]
        inverse = np.linalg.inv(cov + np.eye(3, dtype=np.float64)[None] * 1.0e-10)
        mahalanobis = np.einsum("nki,nij,nkj->nk", delta, inverse, delta)
        inside = (mahalanobis <= 9.0) & (distance <= maximum_surface_distance)
        weight = np.exp(-0.5 * np.clip(mahalanobis, 0.0, 80.0)) * inside
        neighbor_labels = labels[indices]
        distribution = np.zeros((stop - start, column_count), dtype=np.float64)
        for label_id, column in class_to_column.items():
            distribution[:, column] += (weight * (neighbor_labels == label_id)).sum(axis=1)
        # All non-OVS labels are an explicit other/background channel.
        known = np.isin(neighbor_labels, np.asarray(class_ids, dtype=np.int64))
        distribution[:, -1] += (weight * (~known) * (neighbor_labels != 0)).sum(axis=1)
        mass = distribution.sum(axis=1)
        valid = mass > 1.0e-12
        probability = distribution / np.maximum(mass[:, None], 1.0e-12)
        purity = probability.max(axis=1)
        entropy = -(
            probability * np.log(np.maximum(probability, 1.0e-12))
        ).sum(axis=1) / np.log(float(column_count))
        # An unsupported primitive is unknown, not a semantic boundary.  Keep
        # validity separate so zero evidence cannot masquerade as maximal
        # boundary ambiguity or inflate the geometry intervention signal.
        ambiguity = np.where(valid, 1.0 - purity, 0.0)
        eigmax = np.linalg.eigvalsh(cov)[:, -1]
        sigma_max = np.sqrt(np.maximum(eigmax, 1.0e-12))
        nearest = np.asarray(distance[:, 0])
        off_surface = np.clip(nearest / maximum_surface_distance, 0.0, 1.0)
        over_extent = np.clip(sigma_max / maximum_surface_distance, 0.0, 1.0)
        geometry_risk = np.maximum(off_surface, over_extent)
        evidence = np.clip(opacity[start:stop], 0.0, 1.0) * np.clip(
            mass / np.maximum(weight.sum(axis=1), 1.0e-12), 0.0, 1.0
        )
        joint = ambiguity * geometry_risk * evidence

        outputs["surface_distance"][start:stop] = nearest
        outputs["support_sigma_max"][start:stop] = sigma_max
        outputs["semantic_purity"][start:stop] = purity
        outputs["semantic_entropy"][start:stop] = entropy
        outputs["boundary_ambiguity"][start:stop] = ambiguity
        outputs["geometry_risk"][start:stop] = geometry_risk
        outputs["evidence_authority"][start:stop] = evidence
        outputs["joint_risk"][start:stop] = joint
        support_valid[start:stop] = valid
        best = probability.argmax(axis=1)
        dominant[start:stop] = np.asarray(
            [class_ids[value] if value < len(class_ids) else 0 for value in best],
            dtype=np.int16,
        )

    result = {name: torch.from_numpy(value) for name, value in outputs.items()}
    result["support_valid"] = torch.from_numpy(support_valid)
    result["dominant_nyu40_id"] = torch.from_numpy(dominant)
    return result


def _summary(value: torch.Tensor) -> dict[str, float]:
    values = torch.as_tensor(value).float()
    quantiles = torch.quantile(values, torch.tensor([0.5, 0.9, 0.95, 0.99]))
    return {
        "mean": float(values.mean()),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--label-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()

    cohort_path = args.cohort_audit.expanduser().resolve(strict=True)
    cohort = json.loads(cohort_path.read_text())
    matches = [
        record for record in cohort.get("selected_records", [])
        if record.get("scene_id") == args.scene
    ]
    if len(matches) != 1:
        raise ValueError("scene is not uniquely present in selected source cohort")
    assets = {name: Path(record["path"]) for name, record in matches[0]["assets"].items()}
    for name, path in assets.items():
        path.resolve(strict=True)
        expected = matches[0]["assets"][name].get("sha256")
        if expected and sha256_file(path) != expected:
            raise ValueError(f"cohort-bound {name} changed")

    mesh = PlyData.read(str(assets["mesh"]))
    vertex = mesh["vertex"]
    mesh_xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    segmentation = json.loads(assets["segmentation"].read_text())
    aggregation = json.loads(assets["aggregation"].read_text())
    raw_to_nyu40 = load_scannet_raw_to_nyu40(args.label_tsv)
    mesh_labels, label_audit = official_vertex_nyu40_labels(
        scene_id=args.scene,
        vertex_count=mesh_xyz.shape[0],
        segmentation=segmentation,
        aggregation=aggregation,
        raw_to_nyu40=raw_to_nyu40,
    )
    device = torch.device("cpu")
    model = _load_geometry_model(
        load_config(str(assets["render_config"])),
        str(assets["geometry_checkpoint"]),
        device,
    )
    with torch.inference_mode():
        xyz = model.get_xyz().float().cpu().numpy()
        covariance = _gaussian_covariances(model).float().cpu().numpy()
        opacity = model.get_opacity().float().cpu().numpy()
    statistics = support_statistics(
        gaussian_xyz=xyz,
        gaussian_covariance=covariance,
        gaussian_opacity=opacity,
        mesh_xyz=mesh_xyz,
        mesh_labels=mesh_labels,
        class_ids=list(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"]),
        neighbors=args.neighbors,
        chunk_size=args.chunk_size,
    )
    positive_joint = statistics["joint_risk"][statistics["joint_risk"] > 0]
    high_risk_threshold = (
        torch.quantile(positive_joint, 0.95)
        if positive_joint.numel()
        else torch.tensor(float("inf"))
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene_id": args.scene,
        "statistics": statistics,
        "summary": {name: _summary(value) for name, value in statistics.items() if value.dtype.is_floating_point},
        "support_valid_fraction": float(statistics["support_valid"].float().mean()),
        "positive_joint_risk_fraction": float((statistics["joint_risk"] > 0).float().mean()),
        "high_positive_joint_risk_threshold_p95": float(high_risk_threshold),
        "high_positive_joint_risk_fraction_all_gaussians": float(
            (statistics["joint_risk"] >= high_risk_threshold).float().mean()
        ),
        "official_label_audit": label_audit,
        "parameters": {
            "neighbors": args.neighbors,
            "mahalanobis_radius_squared": 9.0,
            "maximum_surface_distance_meters": 0.05,
            "risk_formula": "boundary_ambiguity*max(off_surface,over_extent)*evidence_authority",
        },
        "provenance": {
            "cohort_audit": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
            "benchmark_masks_opened": False,
            "benchmark_predictions_opened": False,
            "geometry_modified": False,
        },
    }
    torch.save(payload, output)
    receipt = output.with_suffix(output.suffix + ".json")
    receipt.write_text(json.dumps({
        "schema": SCHEMA,
        "scene_id": args.scene,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "summary": payload["summary"],
        "official_label_audit": label_audit,
        "geometry_modified": False,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "receipt": str(receipt), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
