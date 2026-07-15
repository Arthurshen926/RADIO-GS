#!/usr/bin/env python3
"""Build a small, query-independent primitive reliability sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.field import (
    canonical_primitive_reliability,
    load_canonical_field_checkpoint,
)
from radio_gs.interfaces.frozen_radio_views import sha256_file


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    rows = torch.as_tensor(values).float().reshape(-1)
    if not rows.numel():
        return {name: 0.0 for name in ("minimum", "p05", "median", "p95", "maximum", "mean")}
    positions = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])
    quantile = torch.quantile(rows, positions)
    return {
        "minimum": float(quantile[0]),
        "p05": float(quantile[1]),
        "median": float(quantile[2]),
        "p95": float(quantile[3]),
        "maximum": float(quantile[4]),
        "mean": float(rows.mean()),
    }


def _count_bin_report(
    counts: torch.Tensor,
    confidence: torch.Tensor,
    reconstruction: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    bins = {
        "one_view": counts == 1,
        "two_views": counts == 2,
        "three_to_four_views": (counts >= 3) & (counts <= 4),
        "five_or_more_views": counts >= 5,
    }
    report: dict[str, dict[str, float | int]] = {}
    for name, selected in bins.items():
        selected = selected & valid
        report[name] = {
            "count": int(selected.sum()),
            "mean_confidence": (
                float(confidence[selected].mean()) if bool(selected.any()) else 0.0
            ),
            "mean_reconstruction_cosine": (
                float(reconstruction[selected].mean())
                if bool(selected.any())
                else 0.0
            ),
        }
    return report


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, Any]:
    field_path = Path(args.field_checkpoint).resolve()
    mpr_path = Path(args.mpr_cache).resolve()
    field, field_payload = load_canonical_field_checkpoint(
        field_path, map_location="cpu"
    )
    mpr = torch.load(mpr_path, map_location="cpu")
    if not isinstance(mpr, dict):
        raise ValueError("MPR cache must be a mapping")
    required = {"xyz", "features", "valid", "view_counts", "reliability", "metadata"}
    if not required.issubset(mpr):
        raise ValueError(f"MPR cache lacks keys: {sorted(required - set(mpr))}")
    mpr_metadata = mpr["metadata"]
    if not isinstance(mpr_metadata, dict):
        raise ValueError("MPR metadata must be a mapping")
    if mpr_metadata.get("benchmark_masks_opened") is not False:
        raise ValueError("reliability source must not open benchmark masks")
    if mpr_metadata.get("text_queries_opened") is not False:
        raise ValueError("reliability source must not open text queries")
    if field_payload.get("benchmark_masks_opened") is not False:
        raise ValueError("canonical field must not open benchmark masks")
    if field_payload.get("text_queries_opened") is not False:
        raise ValueError("canonical field must not open text queries")

    xyz = torch.as_tensor(mpr["xyz"]).float().cpu()
    teacher = torch.as_tensor(mpr["features"]).cpu()
    valid = torch.as_tensor(mpr["valid"]).bool().cpu()
    view_counts = torch.as_tensor(mpr["view_counts"]).long().cpu()
    multiview = torch.as_tensor(mpr["reliability"]).float().cpu()
    count = int(xyz.shape[0]) if xyz.ndim == 2 else -1
    feature_dim = int(field_payload["architecture"]["feature_dim"])
    if xyz.shape != (count, 3) or valid.shape != (count,):
        raise ValueError("MPR geometry/valid rows are malformed")
    if teacher.shape != (count, feature_dim):
        raise ValueError("MPR teacher does not match canonical feature dimension")
    if view_counts.shape != (count,) or multiview.ndim != 2 or multiview.shape[0] != count:
        raise ValueError("MPR reliability rows are malformed")
    if int(field.num_gaussians) != count:
        raise ValueError("canonical field and MPR primitive counts differ")
    field_geometry = dict(field_payload.get("geometry_fingerprint", {}))
    mpr_geometry = dict(mpr.get("geometry_fingerprint", {}))
    if field_geometry != mpr_geometry:
        raise ValueError("canonical field and MPR geometry fingerprints differ")
    if not bool(torch.isfinite(xyz).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("MPR cache contains NaN or infinity")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reliability build requested but CUDA is unavailable")
    field = field.eval().to(device)
    reconstruction_cosine = torch.zeros(count, dtype=torch.float32)
    chunk_size = max(1, int(args.chunk_size))
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        rows = torch.arange(start, stop, device=device, dtype=torch.long)
        prediction = field.radio_features(rows).float()
        target = teacher[start:stop].to(device=device, dtype=torch.float32)
        reconstruction_cosine[start:stop] = F.cosine_similarity(
            prediction, target, dim=-1, eps=1e-8
        ).cpu()

    components = canonical_primitive_reliability(
        view_counts,
        multiview,
        reconstruction_cosine,
        valid=valid,
    )
    field_hash = sha256_file(field_path)
    mpr_hash = sha256_file(mpr_path)
    formula = (
        "confidence=((n/(n+1))*mpr_agreement*"
        "clamp(cos(compact_radio,mpr_radio),0,1))^(1/3)"
    )
    metadata = {
        "schema_version": 1,
        "source": "canonical_primitive_reliability_v1",
        "formula": formula,
        "observation_prior_count": 1,
        "combination": "equal_weight_geometric_mean",
        "field_checkpoint": str(field_path),
        "field_checkpoint_sha256": field_hash,
        "mpr_cache": str(mpr_path),
        "mpr_cache_sha256": mpr_hash,
        "mpr_construction": str(mpr_metadata.get("construction", "")),
        "geometry_fingerprint": field_geometry,
        "query_independent": True,
        "uses_query": False,
        "uses_text": False,
        "uses_target_labels": False,
        "uses_target_masks": False,
        "uses_metric_feedback": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "third_mpr_reliability_channel_used": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "xyz": xyz,
            "valid": valid,
            "confidence": components.confidence.half(),
            "components": {
                "observation_evidence": components.observation_evidence.half(),
                "multiview_agreement": components.multiview_agreement.half(),
                "reconstruction_fidelity": components.reconstruction_fidelity.half(),
            },
            "metadata": metadata,
        },
        output,
    )
    valid_rows = valid
    report = {
        "output": str(output.resolve()),
        "num_gaussians": count,
        "valid_gaussians": int(valid.sum()),
        "confidence": _quantiles(components.confidence[valid_rows]),
        "reconstruction_fidelity": _quantiles(
            components.reconstruction_fidelity[valid_rows]
        ),
        "by_observation_count": _count_bin_report(
            view_counts,
            components.confidence,
            components.reconstruction_fidelity,
            valid,
        ),
        "metadata": metadata,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=8192)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
