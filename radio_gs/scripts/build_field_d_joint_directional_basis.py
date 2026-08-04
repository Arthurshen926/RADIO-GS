#!/usr/bin/env python3
"""Fit a compact basis jointly to canonical centers and directional modes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from radio_gs.field import fit_affine_basis
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_torch_mapping, sha256_file


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@torch.no_grad()
def _metrics(decoder, values: torch.Tensor, batch_size: int) -> dict[str, float]:
    device = decoder.basis.device
    parts: list[torch.Tensor] = []
    for start in range(0, values.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), values.shape[0])
        target = values[start:stop].to(device).float()
        reconstruction = decoder(decoder.encode(target))
        parts.append(
            F.cosine_similarity(reconstruction, target, dim=-1, eps=1e-8)
            .float()
            .cpu()
        )
    cosine = torch.cat(parts)
    return {
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(torch.quantile(cosine, 0.05)),
        "minimum_cosine": float(cosine.min()),
    }


@torch.no_grad()
def build(args: argparse.Namespace) -> dict[str, object]:
    raw, raw_sha, raw_path = load_mpr_cache(
        args.raw_mpr_cache,
        expected_sha256=args.expected_raw_mpr_cache_sha256,
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=True,
    )
    raw_metadata = dict(raw["metadata"])
    if raw_metadata.get("aggregation_mode") != "raster_exact_center_uncertainty":
        raise ValueError("Field-D joint basis requires exact-center raw MPR")
    prototype, prototype_sha, prototype_path = load_torch_mapping(
        args.prototype_cache,
        expected_sha256=args.expected_prototype_cache_sha256,
        map_location="cpu",
        label="Field-D prototype cache",
    )
    if prototype.get("contract") != "weighted_spherical_two_prototype_v1":
        raise ValueError("Field-D directional prototype contract differs")
    metadata = dict(prototype["metadata"])
    if metadata.get("source_mpr_sha256") != raw_sha:
        raise ValueError("Field-D prototype cache belongs to another raw MPR")
    if prototype.get("geometry_fingerprint") != raw.get("geometry_fingerprint"):
        raise ValueError("Field-D raw/prototype geometry differs")
    if any(
        item.get(key) is not False
        for item in (raw_metadata, metadata)
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
    ):
        raise ValueError("Field-D joint-basis input is task contaminated")

    raw_features = torch.as_tensor(raw["features"]).float().cpu()
    raw_valid = torch.as_tensor(raw["valid"]).bool().cpu()
    center_values = raw_features[raw_valid]
    prototype_values = torch.as_tensor(prototype["prototypes"]).float().reshape(-1, 1280)
    if center_values.shape[1] != prototype_values.shape[1]:
        raise ValueError("Field-D center/prototype feature dimensions differ")
    fit_values = torch.cat([center_values, prototype_values], dim=0)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    if fit_values.shape[0] > int(args.maximum_fit_samples):
        selected = torch.randperm(
            fit_values.shape[0], generator=generator
        )[: int(args.maximum_fit_samples)]
        fit_values = fit_values[selected]
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    decoder, fit_report = fit_affine_basis(
        fit_values.to(device),
        int(args.coefficient_dim),
        standardize=True,
        max_samples=int(fit_values.shape[0]),
        seed=int(args.seed),
        trainable_basis=False,
    )
    decoder = decoder.to(device).eval()
    center_metrics = _metrics(decoder, center_values, int(args.batch_size))
    prototype_metrics = _metrics(decoder, prototype_values, int(args.batch_size))
    gate = {
        "center_mean_cosine": center_metrics["mean_cosine"]
        >= float(args.minimum_center_mean_cosine),
        "center_p05_cosine": center_metrics["p05_cosine"]
        >= float(args.minimum_center_p05_cosine),
        "prototype_mean_cosine": prototype_metrics["mean_cosine"]
        >= float(args.minimum_prototype_mean_cosine),
        "prototype_p05_cosine": prototype_metrics["p05_cosine"]
        >= float(args.minimum_prototype_p05_cosine),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "contract": "field_d_joint_center_directional_basis_v1",
            "architecture": {
                "feature_dim": decoder.feature_dim,
                "coefficient_dim": decoder.coefficient_dim,
                "prototype_count": 2,
                "trainable_basis": False,
            },
            "decoder_state_dict": {
                key: value.detach().cpu() for key, value in decoder.state_dict().items()
            },
            "geometry_fingerprint": raw["geometry_fingerprint"],
            "metadata": {
                "raw_mpr_path": str(raw_path),
                "raw_mpr_sha256": raw_sha,
                "prototype_cache_path": str(prototype_path),
                "prototype_cache_sha256": prototype_sha,
                "fit_sample_count": int(fit_values.shape[0]),
                "fit_seed": int(args.seed),
                "fit_report": fit_report.__dict__,
                "center_metrics": center_metrics,
                "prototype_metrics": prototype_metrics,
                "gate": gate,
                "benchmark_images_opened": False,
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
            },
        },
        output,
    )
    report: dict[str, object] = {
        "schema_version": "field_d_joint_directional_basis_receipt_v1",
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "raw_mpr": {"path": str(raw_path), "sha256": raw_sha},
        "prototype_cache": {"path": str(prototype_path), "sha256": prototype_sha},
        "fit_sample_count": int(fit_values.shape[0]),
        "fit_report": fit_report.__dict__,
        "center_metrics": center_metrics,
        "prototype_metrics": prototype_metrics,
        "gate": gate,
        "passed": bool(all(gate.values())),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    receipt = output.with_suffix(output.suffix + ".json")
    _atomic_json(receipt, report)
    return {**report, "receipt": str(receipt), "receipt_sha256": sha256_file(receipt)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mpr-cache", required=True)
    parser.add_argument("--expected-raw-mpr-cache-sha256", required=True)
    parser.add_argument("--prototype-cache", required=True)
    parser.add_argument("--expected-prototype-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coefficient-dim", type=int, default=256)
    parser.add_argument("--maximum-fit-samples", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--minimum-center-mean-cosine", type=float, default=0.95)
    parser.add_argument("--minimum-center-p05-cosine", type=float, default=0.90)
    parser.add_argument("--minimum-prototype-mean-cosine", type=float, default=0.95)
    parser.add_argument("--minimum-prototype-p05-cosine", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
