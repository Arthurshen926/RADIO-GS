#!/usr/bin/env python3
"""Project a learned field's directions onto a frozen per-row amplitude gauge.

Cosine reconstruction objectives leave feature magnitude unidentified, while
nonlinear downstream readouts are generally not scale invariant.  This
query-free adapter preserves every learned semantic direction and restores the
L2 norm of the corresponding primitive from an immutable reference field.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.utils.immutable_artifacts import sha256_file


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable gauge-projected field exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe(payload: Mapping[str, Any], label: str) -> None:
    for key in (
        "benchmark_images_opened",
        "benchmark_masks_opened",
        "text_queries_opened",
    ):
        if payload.get(key) is False:
            continue
        # One frozen legacy field predates the top-level image flag.  Accept
        # only its explicit nested MPR denial; arbitrary missing values still
        # fail closed.
        metadata = payload.get("mpr_cache_metadata")
        if (
            key == "benchmark_images_opened"
            and key not in payload
            and isinstance(metadata, Mapping)
            and metadata.get(key) is False
        ):
            continue
        raise ValueError(f"{label} is task contaminated")


@torch.no_grad()
def build(args: argparse.Namespace) -> dict[str, Any]:
    learned, learned_payload = load_canonical_field_checkpoint(
        args.learned_field_checkpoint,
        expected_sha256=args.expected_learned_field_checkpoint_sha256,
        map_location="cpu",
    )
    reference, reference_payload = load_canonical_field_checkpoint(
        args.gauge_reference_field_checkpoint,
        expected_sha256=args.expected_gauge_reference_field_checkpoint_sha256,
        map_location="cpu",
    )
    if learned_payload.get("geometry_fingerprint") != reference_payload.get(
        "geometry_fingerprint"
    ):
        raise ValueError("learned/reference gauge geometry differs")
    if learned_payload.get("feature_signature") != reference_payload.get(
        "feature_signature"
    ):
        raise ValueError("learned/reference feature signatures differ")
    _safe(learned_payload, "learned field")
    _safe(reference_payload, "gauge reference field")
    if learned.decoder.feature_dim != reference.decoder.feature_dim:
        raise ValueError("learned/reference feature dimensions differ")

    device = torch.device(args.device)
    learned = learned.to(device).eval()
    reference = reference.to(device).eval()
    decoder = learned.decoder
    num_gaussians = learned.num_gaussians
    coefficients = torch.empty(
        num_gaussians, decoder.coefficient_dim, dtype=torch.float32
    )
    cosine_parts: list[torch.Tensor] = []
    log_norm_error_parts: list[torch.Tensor] = []
    direction_drift_parts: list[torch.Tensor] = []
    amplitude_ratio_parts: list[torch.Tensor] = []
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, num_gaussians, batch_size):
        stop = min(num_gaussians, start + batch_size)
        rows = torch.arange(start, stop, device=device)
        learned_features = learned.radio_features(rows).float()
        reference_features = reference.radio_features(rows).float()
        learned_norm = learned_features.norm(dim=-1).clamp_min(1e-8)
        reference_norm = reference_features.norm(dim=-1).clamp_min(1e-8)
        target = F.normalize(learned_features, dim=-1, eps=1e-8) * reference_norm[:, None]
        encoded = decoder.encode(target)
        reconstructed = decoder(encoded)
        reconstructed_norm = reconstructed.norm(dim=-1).clamp_min(1e-8)
        coefficients[start:stop] = encoded.float().cpu()
        cosine_parts.append(
            F.cosine_similarity(reconstructed, target, dim=-1, eps=1e-8)
            .float()
            .cpu()
        )
        log_norm_error_parts.append(
            (reconstructed_norm / reference_norm).log().abs().float().cpu()
        )
        direction_drift_parts.append(
            (
                1.0
                - F.cosine_similarity(
                    reconstructed, learned_features, dim=-1, eps=1e-8
                )
            )
            .float()
            .cpu()
        )
        amplitude_ratio_parts.append((learned_norm / reference_norm).float().cpu())
    cosine = torch.cat(cosine_parts)
    log_norm_error = torch.cat(log_norm_error_parts)
    direction_drift = torch.cat(direction_drift_parts)
    amplitude_ratio = torch.cat(amplitude_ratio_parts)
    metrics = {
        "target_mean_cosine": float(cosine.mean()),
        "target_p05_cosine": float(torch.quantile(cosine, 0.05)),
        "mean_absolute_log_norm_error": float(log_norm_error.mean()),
        "p95_absolute_log_norm_error": float(torch.quantile(log_norm_error, 0.95)),
        "mean_direction_drift_one_minus_cosine": float(direction_drift.mean()),
        "p95_direction_drift_one_minus_cosine": float(
            torch.quantile(direction_drift, 0.95)
        ),
        "preprojection_amplitude_ratio_mean": float(amplitude_ratio.mean()),
        "preprojection_amplitude_ratio_p05": float(torch.quantile(amplitude_ratio, 0.05)),
        "preprojection_amplitude_ratio_p95": float(torch.quantile(amplitude_ratio, 0.95)),
    }
    gate = {
        "mean_cosine": metrics["target_mean_cosine"]
        >= float(args.minimum_mean_cosine),
        "p05_cosine": metrics["target_p05_cosine"]
        >= float(args.minimum_p05_cosine),
        "mean_log_norm_error": metrics["mean_absolute_log_norm_error"]
        <= float(args.maximum_mean_log_norm_error),
        "p95_log_norm_error": metrics["p95_absolute_log_norm_error"]
        <= float(args.maximum_p95_log_norm_error),
    }
    if not all(gate.values()):
        raise RuntimeError(f"gauge-projected field failed its label-free gate: {gate}")

    learned_state = learned.state_dict()
    reliability = torch.as_tensor(learned_payload["reliability"]).float().cpu()
    state_dict = {
        "local_codes": coefficients.contiguous(),
        "reliability": reliability.contiguous(),
        "decoder.basis": learned_state["decoder.basis"].float().cpu().contiguous(),
        "decoder.mean": learned_state["decoder.mean"].float().cpu().contiguous(),
        "decoder.log_scale": learned_state["decoder.log_scale"].float().cpu().contiguous(),
    }
    output = Path(args.output).expanduser().resolve()
    payload = {
        "schema_version": 1,
        "architecture": {
            "num_gaussians": num_gaussians,
            "feature_dim": decoder.feature_dim,
            "coefficient_dim": decoder.coefficient_dim,
            "local_dim": decoder.coefficient_dim,
            "coarse_dim": 0,
            "spatial_hash": None,
            "position_storage": "none",
            "fusion_reliability": False,
            "hidden_dim": 192,
            "fusion_residual_blocks": 0,
            "use_fusion": False,
            "trainable_basis": False,
            "trainable_statistics": False,
        },
        "feature_signature": dict(learned_payload["feature_signature"]),
        "state_dict": state_dict,
        "reliability": reliability,
        "geometry_fingerprint": dict(learned_payload["geometry_fingerprint"]),
        "mpr_cache": learned_payload["mpr_cache"],
        "mpr_cache_sha256": learned_payload["mpr_cache_sha256"],
        "mpr_cache_storage": learned_payload.get("mpr_cache_storage", {}),
        "mpr_cache_metadata": learned_payload.get("mpr_cache_metadata", {}),
        "gauge_projection": {
            "contract": "per_primitive_frozen_reference_l2_gauge_v1",
            "learned_direction_field": {
                "path": str(Path(args.learned_field_checkpoint).expanduser().resolve()),
                "sha256": args.expected_learned_field_checkpoint_sha256,
            },
            "amplitude_reference_field": {
                "path": str(
                    Path(args.gauge_reference_field_checkpoint).expanduser().resolve()
                ),
                "sha256": args.expected_gauge_reference_field_checkpoint_sha256,
            },
            "metrics": metrics,
            "gate": gate,
        },
        "final_metrics": metrics,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    _atomic_torch(output, payload)
    # Reopen through the canonical loader in the calling process after this
    # script; the receipt binds the immutable bytes published here.
    report = {
        "schema_version": "gauge_projected_canonical_field_receipt_v1",
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "learned_field": payload["gauge_projection"]["learned_direction_field"],
        "gauge_reference_field": payload["gauge_projection"][
            "amplitude_reference_field"
        ],
        "metrics": metrics,
        "gate": gate,
        "passed": True,
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
    parser.add_argument("--learned-field-checkpoint", required=True)
    parser.add_argument("--expected-learned-field-checkpoint-sha256", required=True)
    parser.add_argument("--gauge-reference-field-checkpoint", required=True)
    parser.add_argument(
        "--expected-gauge-reference-field-checkpoint-sha256", required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--minimum-mean-cosine", type=float, default=0.95)
    parser.add_argument("--minimum-p05-cosine", type=float, default=0.90)
    parser.add_argument("--maximum-mean-log-norm-error", type=float, default=0.05)
    parser.add_argument("--maximum-p95-log-norm-error", type=float, default=0.15)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
