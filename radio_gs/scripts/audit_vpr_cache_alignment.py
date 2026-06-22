#!/usr/bin/env python3
"""Audit VPR primitive-feature cache alignment against a GaussFM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.config import load_config
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _build_hybrid_model
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint


def xyz_sha256(xyz: torch.Tensor) -> str:
    """Stable SHA256 hash for an ``[N,3]`` xyz tensor."""
    arr = torch.as_tensor(xyz, dtype=torch.float32).detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def tensor_sha256_float32(tensor: torch.Tensor) -> str:
    """Stable SHA256 hash for a float32 tensor payload."""
    arr = torch.as_tensor(tensor, dtype=torch.float32).detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def compute_xyz_alignment_stats(cache_xyz: torch.Tensor, model_xyz: torch.Tensor) -> dict[str, Any]:
    """Return row-wise xyz alignment statistics."""
    cache = torch.as_tensor(cache_xyz, dtype=torch.float32).detach().cpu()
    model = torch.as_tensor(model_xyz, dtype=torch.float32).detach().cpu()
    count_match = int(cache.shape[0]) == int(model.shape[0])
    stats: dict[str, Any] = {
        "count_match": bool(count_match),
        "cache_count": int(cache.shape[0]),
        "model_count": int(model.shape[0]),
        "cache_shape": list(cache.shape),
        "model_shape": list(model.shape),
        "cache_xyz_sha256": xyz_sha256(cache) if cache.ndim == 2 and cache.shape[-1] == 3 else "",
        "model_xyz_sha256": xyz_sha256(model) if model.ndim == 2 and model.shape[-1] == 3 else "",
    }
    stats["xyz_sha256_match"] = (
        bool(stats["cache_xyz_sha256"])
        and bool(stats["model_xyz_sha256"])
        and stats["cache_xyz_sha256"] == stats["model_xyz_sha256"]
    )
    if cache.ndim != 2 or model.ndim != 2 or cache.shape[-1] != 3 or model.shape[-1] != 3:
        stats.update(
            {
                "max_l2": float("inf"),
                "mean_l2": float("inf"),
                "p95_l2": float("inf"),
                "scene_scale": 0.0,
                "normalized_max_l2": float("inf"),
            }
        )
        return stats
    if not count_match:
        stats.update(
            {
                "max_l2": float("inf"),
                "mean_l2": float("inf"),
                "p95_l2": float("inf"),
                "scene_scale": float(
                    (model.max(dim=0).values - model.min(dim=0).values).norm().item()
                )
                if model.numel()
                else 0.0,
                "normalized_max_l2": float("inf"),
            }
        )
        return stats
    delta = (cache - model).norm(dim=-1)
    scene_scale = float((model.max(dim=0).values - model.min(dim=0).values).norm().item())
    max_l2 = float(delta.max().item()) if delta.numel() else 0.0
    stats.update(
        {
            "max_l2": max_l2,
            "mean_l2": float(delta.mean().item()) if delta.numel() else 0.0,
            "p95_l2": float(torch.quantile(delta, 0.95).item()) if delta.numel() else 0.0,
            "scene_scale": scene_scale,
            "normalized_max_l2": max_l2 / max(scene_scale, 1e-12),
        }
    )
    return stats


def _feature_key(payload: dict[str, Any]) -> str:
    if "summary_features" in payload:
        return "summary_features"
    if "features" in payload:
        return "features"
    return ""


def audit_vpr_cache_payload_alignment(
    payload: dict[str, Any],
    model_xyz: torch.Tensor,
    *,
    model_scales: torch.Tensor | None = None,
    model_rotations: torch.Tensor | None = None,
    model_opacities: torch.Tensor | None = None,
    fail_max_l2: float = 1e-5,
    cache_path: str = "",
) -> dict[str, Any]:
    """Audit an already-loaded VPR cache payload."""
    feature_key = _feature_key(payload)
    report: dict[str, Any] = {
        "cache_path": str(cache_path),
        "feature_key": feature_key,
        "has_valid": isinstance(payload.get("valid"), torch.Tensor),
        "has_view_counts": isinstance(payload.get("view_counts"), torch.Tensor),
        "metadata": payload.get("metadata") or {},
    }
    if feature_key:
        features = torch.as_tensor(payload[feature_key])
        report["feature_shape"] = list(features.shape)
        report["feature_dtype"] = str(features.dtype)
    if "valid" in payload and torch.is_tensor(payload["valid"]):
        valid = payload["valid"].bool()
        report["valid_count"] = int(valid.sum().item())
        report["total_count"] = int(valid.numel())
    fingerprint = payload.get("geometry_fingerprint")
    if isinstance(fingerprint, dict):
        report["geometry_fingerprint"] = {
            "num_gaussians": fingerprint.get("num_gaussians"),
            "xyz_sha256": fingerprint.get("xyz_sha256", ""),
        }
        for key in ("scales", "rotations", "opacities"):
            hash_key = f"{key}_sha256"
            shape_key = f"{key}_shape"
            if hash_key in fingerprint:
                report["geometry_fingerprint"][hash_key] = fingerprint.get(hash_key, "")
                report["geometry_fingerprint"][shape_key] = fingerprint.get(shape_key, [])
    if "xyz" not in payload:
        report.update(
            {
                "status": "missing_xyz",
                "passed": False,
                "message": "VPR cache payload missing required xyz for row-alignment audit",
            }
        )
        return report
    stats = compute_xyz_alignment_stats(torch.as_tensor(payload["xyz"]), model_xyz)
    report.update(stats)
    cached_fingerprint_hash = (
        str(fingerprint.get("xyz_sha256", ""))
        if isinstance(fingerprint, dict)
        else ""
    )
    if cached_fingerprint_hash:
        report["geometry_fingerprint_xyz_sha256_match"] = (
            cached_fingerprint_hash == stats.get("cache_xyz_sha256", "")
        )
    passed = bool(stats["count_match"]) and float(stats["max_l2"]) <= float(fail_max_l2)
    optional_mismatches: list[str] = []
    optional_model_tensors = {
        "scales": model_scales,
        "rotations": model_rotations,
        "opacities": model_opacities,
    }
    if isinstance(fingerprint, dict):
        optional_checks: dict[str, Any] = {}
        for key, tensor in optional_model_tensors.items():
            expected_hash = str(fingerprint.get(f"{key}_sha256", ""))
            if not expected_hash:
                continue
            if tensor is None:
                optional_checks[key] = {
                    "checked": False,
                    "cached_sha256": expected_hash,
                    "message": "model tensor was not provided for audit",
                }
                continue
            model_hash = tensor_sha256_float32(torch.as_tensor(tensor))
            matches = model_hash == expected_hash
            optional_checks[key] = {
                "checked": True,
                "cached_sha256": expected_hash,
                "model_sha256": model_hash,
                "match": bool(matches),
            }
            if not matches:
                optional_mismatches.append(key)
        if optional_checks:
            report["geometry_fingerprint_optional_checks"] = optional_checks
    if optional_mismatches:
        passed = False
    report["fail_max_l2"] = float(fail_max_l2)
    report["passed"] = passed
    report["status"] = "passed" if passed else "failed"
    report["message"] = (
        "cache xyz rows align with model geometry"
        if passed
        else (
            "cache Gaussian footprint tensors do not align with model geometry: "
            + ", ".join(optional_mismatches)
            if optional_mismatches
            else "cache xyz rows do not align with model geometry"
        )
    )
    return report


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec = _build_hybrid_model(config, args.checkpoint, device)
    payload = load_trusted_checkpoint(args.teacher_cache, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"VPR cache must be a dict payload: {args.teacher_cache}")
    report = audit_vpr_cache_payload_alignment(
        payload,
        model.get_xyz().detach().cpu(),
        model_scales=model.get_scaling().detach().cpu(),
        model_rotations=model.get_rotation().detach().cpu(),
        model_opacities=model.get_opacity().detach().cpu(),
        fail_max_l2=args.fail_max_l2,
        cache_path=args.teacher_cache,
    )
    report["config"] = args.config
    report["checkpoint"] = args.checkpoint
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher_cache", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--fail_max_l2", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    report = run_audit(_parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("passed") else 2)


if __name__ == "__main__":
    main()
