"""Cross-scene source-only visual-semantic codec for SUGM-v3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file


def apply_codec(values: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != mean.numel() or basis.shape[0] != mean.numel():
        raise ValueError("codec input axes differ")
    return (values - mean.to(values)) @ basis.to(values)


def _balanced_source_train_samples(
    memberships: Sequence[Mapping[str, Any]],
    teacher_roots: Sequence[Path],
    subdirectory: str,
    *,
    samples_per_view: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    samples, lineage = [], []
    for scene_index, (membership, root) in enumerate(zip(memberships, teacher_roots)):
        records = [
            record for record in membership["metadata"]["source_records"]
            if int(record["source_view_index"]) % 4 in (1, 2)
        ]
        for record in records:
            frame_id = int(record["frame_id"])
            path = (root / subdirectory / f"rgb_{frame_id}.pt").resolve(strict=True)
            feature = torch.load(path, map_location="cpu").float()
            flat = feature.permute(1, 2, 0).reshape(-1, feature.shape[0])
            generator = torch.Generator(device="cpu").manual_seed(
                int(seed + scene_index * 1_000_003 + int(record["source_view_index"]))
            )
            count = min(int(samples_per_view), flat.shape[0])
            indices = torch.randperm(flat.shape[0], generator=generator)[:count]
            samples.append(flat[indices])
            lineage.append({
                "source_view_index": int(record["source_view_index"]),
                "frame_id": frame_id,
                "feature_path": str(path),
                "sample_count": count,
            })
    if not samples:
        raise ValueError("codec fitting lacks source-train samples")
    return torch.cat(samples), lineage


def _principal_codec(
    samples: torch.Tensor, output_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if output_dim <= 0 or output_dim > samples.shape[1]:
        raise ValueError("invalid learned codec output dimension")
    mean = samples.mean(0)
    centered = samples - mean
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.to(device))
    basis = eigenvectors[:, -output_dim:].flip(1).cpu()
    retained = float(
        eigenvalues[-output_dim:].sum().div(eigenvalues.clamp_min(0).sum().clamp_min(1e-12))
    )
    return mean, basis, retained


@torch.no_grad()
def fit_codec(args: argparse.Namespace) -> dict[str, Any]:
    if not (len(args.scene) == len(args.membership) == len(args.teacher_root)):
        raise ValueError("codec scene, membership, and teacher-root counts differ")
    membership_paths = [Path(value).resolve(strict=True) for value in args.membership]
    roots = [Path(value).resolve(strict=True) for value in args.teacher_root]
    memberships = [torch.load(path, map_location="cpu") for path in membership_paths]
    radio, radio_lineage = _balanced_source_train_samples(
        memberships, roots, "backbone",
        samples_per_view=args.samples_per_view, seed=args.seed,
    )
    siglip, siglip_lineage = _balanced_source_train_samples(
        memberships, roots, "siglip2",
        samples_per_view=args.samples_per_view, seed=args.seed + 1,
    )
    device = torch.device(args.device)
    radio_mean, radio_basis, radio_retained = _principal_codec(
        radio, args.shared_dim, device
    )
    del radio
    if device.type == "cuda":
        torch.cuda.empty_cache()
    siglip_mean, siglip_basis, siglip_retained = _principal_codec(
        siglip, args.semantic_dim, device
    )
    payload = {
        "schema": "radio_gs.sugm_v3.cross_scene_source_codec.v1",
        "state_dict": {
            "radio_mean": radio_mean,
            "radio_basis": radio_basis,
            "siglip_mean": siglip_mean,
            "siglip_basis": siglip_basis,
        },
        "metadata": {
            "type": "cross_scene_source_train_pca_exact_mpr",
            "source_only": True,
            "source_train_residues": [1, 2],
            "historical_field_opened": False,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "normalization_order": "linear_projection_then_exact_mpr_then_row_normalize",
            "scenes": list(args.scene),
            "memberships": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in membership_paths
            ],
            "teacher_roots": [str(root) for root in roots],
            "samples_per_view": int(args.samples_per_view),
            "seed": int(args.seed),
            "radio": {
                "input_dim": radio_mean.numel(), "output_dim": args.shared_dim,
                "retained_variance_fraction": radio_retained, "lineage": radio_lineage,
            },
            "siglip": {
                "input_dim": siglip_mean.numel(), "output_dim": args.semantic_dim,
                "retained_variance_fraction": siglip_retained, "lineage": siglip_lineage,
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {**payload["metadata"], "output": str(output), "sha256": sha256_file(output)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--scene", action="append", required=True)
    value.add_argument("--membership", action="append", required=True)
    value.add_argument("--teacher-root", action="append", required=True)
    value.add_argument("--shared-dim", type=int, default=320)
    value.add_argument("--semantic-dim", type=int, default=128)
    value.add_argument("--samples-per-view", type=int, default=512)
    value.add_argument("--seed", type=int, default=20260827)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--output", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.samples_per_view <= 0:
        raise ValueError("samples-per-view must be positive")
    print(fit_codec(args))


if __name__ == "__main__":
    main()


__all__ = ["apply_codec", "fit_codec"]
