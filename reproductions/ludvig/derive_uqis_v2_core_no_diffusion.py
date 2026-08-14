#!/usr/bin/env python3
"""Deterministically read out saved v0.2 CLIP primitives without graph diffusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_c import _import_ludvig_geometry
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_d import gaussian_covariances
from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION_V2_CANDIDATE,
    canonical_json_sha256,
    sha256_file,
)


def _readout(
    primitive: np.ndarray,
    xyz: np.ndarray,
    precision: np.ndarray,
    opacity: np.ndarray,
    tree: cKDTree,
    mesh: np.ndarray,
    *,
    k: int = 64,
    chunk: int = 32768,
) -> np.ndarray:
    result = np.empty(len(mesh), dtype=np.float32)
    for begin in range(0, len(mesh), chunk):
        end = min(begin + chunk, len(mesh))
        _distance, indices = tree.query(mesh[begin:end], k=k, workers=-1)
        indices = np.asarray(indices, dtype=np.int64)
        delta = xyz[indices] - mesh[begin:end, None]
        mahalanobis = np.einsum("vki,vkij,vkj->vk", delta, precision[indices], delta, optimize=True)
        local_opacity = opacity[indices]
        log_weights = np.full_like(mahalanobis, -np.inf)
        positive = local_opacity > 0
        log_weights[positive] = -0.5 * np.maximum(mahalanobis[positive], 0) + np.log(local_opacity[positive])
        maximum = log_weights.max(axis=1)
        weights = np.exp(log_weights - maximum[:, None])
        result[begin:end] = (
            (weights * primitive[indices]).sum(axis=1) / weights.sum(axis=1)
        ).astype(np.float32)
    if not np.isfinite(result).all() or bool(((result < 0) | (result > 1)).any()):
        raise ValueError("direct CLIP readout produced invalid scores")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-inventory", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--diffusion-run-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--ludvig-upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inventory = json.loads(args.workspace_inventory.read_text(encoding="utf-8"))
    if inventory.get("benchmark_version") != BENCHMARK_VERSION_V2_CANDIDATE:
        raise ValueError("workspace inventory changed")
    by_scene: dict[str, list[str]] = {}
    for row in inventory["workspaces"]:
        by_scene.setdefault(str(row["scene_id"]), []).append(str(row["query_id"]))
    args.output_dir.mkdir(parents=True)
    records = []
    GaussianModel, _ = _import_ludvig_geometry(args.ludvig_upstream)
    for scene_id, query_ids in sorted(by_scene.items()):
        field_root = args.clip_root / f"{scene_id}_v1"
        field_path = field_root / "run_manifest.json"
        field = json.loads(field_path.read_text(encoding="utf-8"))
        geometry = field["geometry"]
        geometry_path = Path(geometry["path"])
        if sha256_file(geometry_path) != geometry["sha256"]:
            raise ValueError("CLIP geometry changed")
        gaussian = GaussianModel(sh_degree=0)
        gaussian.load_ply(str(geometry_path))
        xyz = gaussian.get_xyz.detach().cpu().numpy().astype(np.float64, copy=False)
        covariance = gaussian_covariances(gaussian.get_scaling, gaussian.get_rotation).detach().cpu().numpy().astype(np.float64, copy=False)
        opacity = gaussian.get_opacity.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        precision = np.linalg.pinv(covariance + 1e-6 * np.eye(3)[None])
        tree = cKDTree(xyz)
        for query_id in sorted(query_ids):
            run_root = args.diffusion_run_root / query_id
            run = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
            binding = run["artifacts"]["primitive_relevancy"]
            primitive_path = run_root / binding["relative_path"]
            if sha256_file(primitive_path) != binding["sha256"]:
                raise ValueError("saved CLIP primitive changed")
            primitive = np.load(primitive_path, allow_pickle=False).astype(np.float64, copy=False)
            workspace_manifest = json.loads(
                (args.workspace_root / query_id / "query_manifest.json").read_text(encoding="utf-8")
            )
            domain = workspace_manifest["scene_domains"][0]
            mesh_path = Path(domain["mesh_xyz_path"])
            if sha256_file(mesh_path) != domain["mesh_xyz_sha256"]:
                raise ValueError("workspace mesh changed")
            mesh = np.load(mesh_path, allow_pickle=False).astype(np.float64, copy=False)
            probability = _readout(primitive, xyz, precision, opacity, tree, mesh)
            output = args.output_dir / f"{query_id}.npy"
            np.save(output, probability, allow_pickle=False)
            records.append(
                {
                    "query_id": query_id,
                    "scene_id": scene_id,
                    "prediction_sha256": sha256_file(output),
                    "primitive_sha256": binding["sha256"],
                    "clip_field_manifest_sha256": sha256_file(field_path),
                    "shape": [len(probability)],
                }
            )
        del gaussian, xyz, covariance, opacity, precision, tree
    body = {
        "schema_version": "scannet_uqis_v2_core_direct_clip_readout_v1",
        "benchmark_version": BENCHMARK_VERSION_V2_CANDIDATE,
        "status": "complete_posthoc_deterministic_ablation",
        "formal_benchmark_eligible": False,
        "query_count": len(records),
        "protocol": {
            "source": "saved_pre_diffusion_primitive_relevancy",
            "mesh_readout": "K64_continuous_opacity_weighted_gaussian",
            "parameters_changed_after_evaluator_open": False,
            "evaluator_private_inputs_opened_by_derivation_process": False,
        },
        "predictions": records,
    }
    manifest = {**body, "inventory_sha256": canonical_json_sha256(records)}
    path = args.output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "query_count": len(records),
                      "manifest_sha256": sha256_file(path)}, indent=2))


if __name__ == "__main__":
    main()
