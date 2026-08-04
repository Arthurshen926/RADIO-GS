#!/usr/bin/env python3
"""Fit Field-D compression while preserving the frozen field's amplitude gauge.

Directional observations define semantic directions on the unit sphere, but
the frozen nonlinear SurfaceRegion readout also consumes feature magnitude.
This builder transfers each primitive's label-free base-field norm to both
directional modes before fitting a shared compact basis.  It therefore changes
directional capacity without silently changing the downstream feature gauge.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from radio_gs.field import fit_affine_basis, load_canonical_field_checkpoint
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_torch_mapping, sha256_file


CONTRACT = "field_d_gauge_preserving_joint_basis_v1"


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


def _safe(metadata: dict, label: str) -> None:
    if any(
        metadata.get(key) is not False
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
    ):
        raise ValueError(f"{label} is task contaminated")


@torch.no_grad()
def _fidelity(decoder, values: torch.Tensor, batch_size: int) -> dict[str, float]:
    device = decoder.basis.device
    cosine_parts: list[torch.Tensor] = []
    log_norm_error_parts: list[torch.Tensor] = []
    relative_l2_parts: list[torch.Tensor] = []
    for start in range(0, values.shape[0], int(batch_size)):
        stop = min(values.shape[0], start + int(batch_size))
        target = values[start:stop].to(device=device, dtype=torch.float32)
        reconstructed = decoder(decoder.encode(target))
        target_norm = target.norm(dim=-1).clamp_min(1e-8)
        reconstructed_norm = reconstructed.norm(dim=-1).clamp_min(1e-8)
        cosine_parts.append(
            F.cosine_similarity(reconstructed, target, dim=-1, eps=1e-8)
            .float()
            .cpu()
        )
        log_norm_error_parts.append(
            (reconstructed_norm / target_norm).log().abs().float().cpu()
        )
        relative_l2_parts.append(
            ((reconstructed - target).norm(dim=-1) / target_norm).float().cpu()
        )
    cosine = torch.cat(cosine_parts)
    log_norm_error = torch.cat(log_norm_error_parts)
    relative_l2 = torch.cat(relative_l2_parts)
    return {
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(torch.quantile(cosine, 0.05)),
        "mean_absolute_log_norm_error": float(log_norm_error.mean()),
        "p95_absolute_log_norm_error": float(torch.quantile(log_norm_error, 0.95)),
        "mean_relative_l2": float(relative_l2.mean()),
        "p95_relative_l2": float(torch.quantile(relative_l2, 0.95)),
    }


@torch.no_grad()
def build(args: argparse.Namespace) -> dict[str, object]:
    base, base_payload = load_canonical_field_checkpoint(
        args.base_field_checkpoint,
        expected_sha256=args.expected_base_field_checkpoint_sha256,
        map_location="cpu",
    )
    raw, raw_sha, raw_path = load_mpr_cache(
        args.raw_mpr_cache,
        expected_sha256=args.expected_raw_mpr_cache_sha256,
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=True,
    )
    prototype, prototype_sha, prototype_path = load_torch_mapping(
        args.prototype_cache,
        expected_sha256=args.expected_prototype_cache_sha256,
        map_location="cpu",
        label="Field-D directional prototype cache",
    )
    if prototype.get("contract") != "weighted_spherical_two_prototype_v1":
        raise ValueError("Field-D directional prototype contract differs")
    geometry = raw.get("geometry_fingerprint")
    if base_payload.get("geometry_fingerprint") != geometry or prototype.get(
        "geometry_fingerprint"
    ) != geometry:
        raise ValueError("Field-D gauge inputs use different geometry authorities")
    _safe(dict(base_payload), "base field")
    _safe(dict(raw["metadata"]), "exact-center MPR")
    _safe(dict(prototype["metadata"]), "directional prototypes")
    if dict(prototype["metadata"]).get("source_mpr_sha256") != raw_sha:
        raise ValueError("directional prototypes belong to another MPR")

    valid_rows = torch.where(torch.as_tensor(raw["valid"]).bool().cpu())[0]
    prototype_rows = torch.as_tensor(prototype["global_rows"]).long().cpu()
    if prototype_rows.numel() and not bool(
        torch.isin(prototype_rows, valid_rows).all()
    ):
        raise ValueError("directional prototypes are outside exact-center valid rows")
    modes = torch.as_tensor(prototype["prototypes"]).float().cpu()
    if modes.shape != (prototype_rows.numel(), 2, 1280):
        raise ValueError("directional prototype tensor differs")
    device = torch.device(args.device)
    base = base.to(device).eval()
    base_features = torch.empty(valid_rows.numel(), 1280, dtype=torch.float16)
    batch_size = int(args.batch_size)
    for start in range(0, valid_rows.numel(), batch_size):
        stop = min(valid_rows.numel(), start + batch_size)
        base_features[start:stop] = base.radio_features(
            valid_rows[start:stop].to(device)
        ).half().cpu()
    inverse = torch.full((int(geometry["num_gaussians"]),), -1, dtype=torch.long)
    inverse[valid_rows] = torch.arange(valid_rows.numel())
    local = inverse[prototype_rows]
    if bool((local < 0).any()):
        raise ValueError("prototype/base-field row mapping failed")
    prototype_amplitude = base_features[local].float().norm(dim=-1)
    if not bool(torch.isfinite(prototype_amplitude).all()) or bool(
        (prototype_amplitude <= 1e-8).any()
    ):
        raise ValueError("base-field amplitude authority is non-finite or zero")
    scaled_modes = F.normalize(modes, dim=-1, eps=1e-8) * prototype_amplitude[:, None, None]

    candidates = torch.cat(
        [base_features.float(), scaled_modes.reshape(-1, 1280)], dim=0
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    if candidates.shape[0] > int(args.maximum_fit_samples):
        selected = torch.randperm(
            candidates.shape[0], generator=generator
        )[: int(args.maximum_fit_samples)]
        fit_values = candidates[selected]
    else:
        fit_values = candidates
    torch.manual_seed(int(args.seed))
    decoder, fit_report = fit_affine_basis(
        fit_values.to(device),
        int(args.coefficient_dim),
        standardize=True,
        max_samples=int(fit_values.shape[0]),
        seed=int(args.seed),
        trainable_basis=False,
    )
    decoder = decoder.to(device).eval()
    center_metrics = _fidelity(decoder, base_features, batch_size)
    prototype_metrics = _fidelity(
        decoder, scaled_modes.reshape(-1, 1280), batch_size
    )
    gate = {
        "center_mean_cosine": center_metrics["mean_cosine"]
        >= float(args.minimum_mean_cosine),
        "center_p05_cosine": center_metrics["p05_cosine"]
        >= float(args.minimum_p05_cosine),
        "prototype_mean_cosine": prototype_metrics["mean_cosine"]
        >= float(args.minimum_mean_cosine),
        "prototype_p05_cosine": prototype_metrics["p05_cosine"]
        >= float(args.minimum_p05_cosine),
        "center_mean_log_norm_error": center_metrics[
            "mean_absolute_log_norm_error"
        ] <= float(args.maximum_mean_log_norm_error),
        "center_p95_log_norm_error": center_metrics[
            "p95_absolute_log_norm_error"
        ] <= float(args.maximum_p95_log_norm_error),
        "prototype_mean_log_norm_error": prototype_metrics[
            "mean_absolute_log_norm_error"
        ] <= float(args.maximum_mean_log_norm_error),
        "prototype_p95_log_norm_error": prototype_metrics[
            "p95_absolute_log_norm_error"
        ] <= float(args.maximum_p95_log_norm_error),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable gauge-preserving basis exists: {output}")
    state = {key: value.detach().cpu() for key, value in decoder.state_dict().items()}
    torch.save(
        {
            "schema_version": 1,
            "contract": CONTRACT,
            "architecture": {
                "feature_dim": decoder.feature_dim,
                "coefficient_dim": decoder.coefficient_dim,
                "prototype_count": 2,
                "trainable_basis": False,
            },
            "decoder_state_dict": state,
            "prototype_global_rows": prototype_rows,
            "prototype_amplitude": prototype_amplitude.half(),
            "geometry_fingerprint": geometry,
            "metadata": {
                "raw_mpr_path": str(raw_path),
                "raw_mpr_sha256": raw_sha,
                "prototype_cache_path": str(prototype_path),
                "prototype_cache_sha256": prototype_sha,
                "amplitude_reference_field": {
                    "path": str(Path(args.base_field_checkpoint).expanduser().resolve()),
                    "sha256": args.expected_base_field_checkpoint_sha256,
                },
                "amplitude_semantics": "per_primitive_base_field_l2_norm",
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
    report = {
        "schema_version": "field_d_gauge_preserving_basis_receipt_v1",
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "base_field": {
            "path": str(Path(args.base_field_checkpoint).expanduser().resolve()),
            "sha256": args.expected_base_field_checkpoint_sha256,
        },
        "raw_mpr": {"path": str(raw_path), "sha256": raw_sha},
        "prototype_cache": {"path": str(prototype_path), "sha256": prototype_sha},
        "fit_sample_count": int(fit_values.shape[0]),
        "fit_report": fit_report.__dict__,
        "center_metrics": center_metrics,
        "prototype_metrics": prototype_metrics,
        "amplitude": {
            "mean": float(prototype_amplitude.mean()),
            "p05": float(torch.quantile(prototype_amplitude, 0.05)),
            "p95": float(torch.quantile(prototype_amplitude, 0.95)),
        },
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
    parser.add_argument("--base-field-checkpoint", required=True)
    parser.add_argument("--expected-base-field-checkpoint-sha256", required=True)
    parser.add_argument("--raw-mpr-cache", required=True)
    parser.add_argument("--expected-raw-mpr-cache-sha256", required=True)
    parser.add_argument("--prototype-cache", required=True)
    parser.add_argument("--expected-prototype-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coefficient-dim", type=int, default=256)
    parser.add_argument("--maximum-fit-samples", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--minimum-mean-cosine", type=float, default=0.95)
    parser.add_argument("--minimum-p05-cosine", type=float, default=0.90)
    parser.add_argument("--maximum-mean-log-norm-error", type=float, default=0.05)
    parser.add_argument("--maximum-p95-log-norm-error", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
