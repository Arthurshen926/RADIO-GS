"""Masked-row upper bound for unknown-aware structural reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.build_row_propagation_authority import _voxel_keys
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _masked_split(rows: torch.Tensor, *, modulus: int = 5, residue: int = 0) -> torch.Tensor:
    value = torch.as_tensor(rows).long()
    return ((value * 2654435761 + 1013904223) % modulus) == residue


def _neighbor_reconstruction(
    memory: torch.Tensor, xyz: torch.Tensor, train: torch.Tensor, target: torch.Tensor,
    *, resolution: int, top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = _voxel_keys(xyz, resolution)
    grouped: dict[int, list[int]] = {}
    for key, row in zip(keys[train].tolist(), torch.where(train)[0].tolist()):
        grouped.setdefault(key, []).append(row)
    target_rows = torch.where(target)[0]
    reconstructed = torch.zeros(target_rows.numel(), memory.shape[1])
    covered = torch.zeros(target_rows.numel(), dtype=torch.bool)
    lower, upper = xyz.amin(0), xyz.amax(0)
    scale = float(torch.linalg.vector_norm(upper - lower)) * (3 ** 0.5 / resolution)
    for index, row in enumerate(target_rows.tolist()):
        candidates = grouped.get(int(keys[row]))
        if not candidates:
            continue
        candidate_rows = torch.tensor(candidates)
        distance = torch.linalg.vector_norm(xyz[candidate_rows] - xyz[row], dim=1)
        count = min(top_k, candidate_rows.numel())
        nearest_distance, nearest = torch.topk(distance, count, largest=False)
        weight = torch.softmax(-nearest_distance / max(scale, 1e-8), dim=0)
        reconstructed[index] = (memory[candidate_rows[nearest]] * weight[:, None]).sum(0)
        covered[index] = True
    return reconstructed, covered


def _cosine_report(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    cosine = (F.normalize(prediction, dim=1, eps=1e-8) * F.normalize(target, dim=1, eps=1e-8)).sum(1)
    return {
        "mean": float(cosine.mean()), "p10": float(torch.quantile(cosine, 0.1)),
        "median": float(cosine.median()),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    initialization_path = Path(args.initialization).resolve(strict=True)
    primitive_path = Path(args.primitive_cache).resolve(strict=True)
    policy_path = Path(args.coverage_policy).resolve(strict=True)
    initialization = torch.load(initialization_path, map_location="cpu", weights_only=False)
    primitive = torch.load(primitive_path, map_location="cpu", weights_only=False)
    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    memory = torch.as_tensor(initialization["state_dict"]["memory"]).float()[:, :448]
    xyz = torch.as_tensor(primitive["xyz"]).float()
    authority = torch.as_tensor(policy["visual_write_authority"]).bool()
    if memory.shape != (xyz.shape[0], 448) or authority.shape != (xyz.shape[0],):
        raise ValueError("masked reconstruction row authorities differ")
    validation = authority & _masked_split(torch.arange(xyz.shape[0]))
    train = authority & ~validation
    prediction, covered = _neighbor_reconstruction(
        memory, xyz, train, validation, resolution=args.voxel_resolution, top_k=args.top_k
    )
    validation_rows = torch.where(validation)[0]
    prediction, target = prediction[covered], memory[validation_rows[covered]]
    coverage = float(covered.float().mean())
    shared = _cosine_report(prediction[:, :320], target[:, :320])
    semantic = _cosine_report(prediction[:, 320:], target[:, 320:])
    gate_pass = coverage >= args.minimum_coverage and shared["median"] >= args.minimum_shared_median
    payload = {
        "schema": "radio_gs.sugm_v3.masked_structural_reconstruction.v1",
        "scene": policy["scene"], "validation_rows": int(validation.sum()),
        "covered_rows": int(covered.sum()), "coverage": coverage,
        "shared_d320": shared, "semantic_d128": semantic,
        "gate": {
            "pass": bool(gate_pass), "minimum_coverage": args.minimum_coverage,
            "minimum_shared_median": args.minimum_shared_median,
        },
        "metadata": {
            "split": {"type": "deterministic_authorized_row_mask", "modulus": 5, "residue": 0},
            "voxel_resolution": args.voxel_resolution, "top_k": args.top_k,
            "source_only": True, "target_rgb_opened": False,
            "benchmark_metrics_opened": False, "source_dev_opened": False,
            "inputs": {
                "initialization": {"path": str(initialization_path), "sha256": sha256_file(initialization_path)},
                "primitive_cache": {"path": str(primitive_path), "sha256": sha256_file(primitive_path)},
                "coverage_policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "coverage": coverage, "shared_d320": shared, "semantic_d128": semantic, "gate": payload["gate"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--primitive-cache", required=True)
    parser.add_argument("--coverage-policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel-resolution", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--minimum-coverage", type=float, default=0.80)
    parser.add_argument("--minimum-shared-median", type=float, default=0.80)
    args = parser.parse_args()
    if args.voxel_resolution <= 0 or args.top_k <= 0:
        raise ValueError("masked reconstruction budgets are invalid")
    print(run(args))


if __name__ == "__main__":
    main()
