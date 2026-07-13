#!/usr/bin/env python3
"""Evaluate direct primitive RADIO fidelity against a row-verified MPR cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.field import load_canonical_field_checkpoint


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    field, payload = load_canonical_field_checkpoint(args.field_checkpoint, map_location="cpu")
    cache_path = str(args.mpr_cache or payload["mpr_cache"])
    cache = torch.load(cache_path, map_location="cpu")
    expected = str(payload.get("geometry_fingerprint", {}).get("xyz_sha256", ""))
    actual = str(cache.get("geometry_fingerprint", {}).get("xyz_sha256", ""))
    if not expected or actual != expected:
        raise ValueError("canonical field and MPR cache geometry rows differ")
    targets = torch.as_tensor(cache["features"]).float()
    valid = torch.as_tensor(cache["valid"]).bool()
    if targets.shape[0] != field.num_gaussians or valid.shape != (field.num_gaussians,):
        raise ValueError("MPR cache rows do not align with canonical field")
    rows = torch.where(valid)[0]
    if args.max_rows > 0 and rows.numel() > int(args.max_rows):
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
        rows = rows[torch.randperm(rows.numel(), generator=generator)[: int(args.max_rows)]]
    field = field.to(device).eval()
    cosines: list[torch.Tensor] = []
    rmses: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, rows.numel(), int(args.batch_size)):
            batch = rows[start : start + int(args.batch_size)]
            predicted = field.radio_features(batch.to(device)).float().cpu()
            target = targets[batch]
            cosines.append(F.cosine_similarity(predicted, target, dim=-1, eps=1e-8))
            rmses.append((predicted - target).square().mean(dim=-1).sqrt())
    cosine = torch.cat(cosines)
    rmse = torch.cat(rmses)
    report = {
        "schema_version": 1,
        "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
        "mpr_cache": str(Path(cache_path).resolve()),
        "geometry_rows_verified": True,
        "evaluated_rows": int(rows.numel()),
        "valid_rows_in_cache": int(valid.sum()),
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(cosine.quantile(0.05)),
        "mean_rmse": float(rmse.mean()),
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--mpr-cache", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
