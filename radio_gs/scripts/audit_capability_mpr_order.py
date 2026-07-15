#!/usr/bin/env python3
"""Audit the non-commutation of official capability projection and MPR.

The comparison is query-free: ``A(MPR(raw view features))`` is measured
against ``MPR(A(raw view features))`` on identical Gaussian rows and matched
visibility responsibilities.  No benchmark mask, category, or prompt is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.frozen_radio_views import FrozenRadioViews, sha256_file
from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.scripts.train_canonical_radio_field import (
    _capability_reconstruction_metrics,
    _load_capability_mpr_target,
)


def audit(args: argparse.Namespace) -> dict:
    raw_path = Path(args.raw_mpr_cache)
    raw = torch.load(raw_path, map_location="cpu")
    if not isinstance(raw, dict) or "features" not in raw:
        raise ValueError("raw MPR cache lacks primitive features")
    raw_metadata = dict(raw.get("metadata", {}))
    if str(raw_metadata.get("feature_space", "")) != "radio":
        raise ValueError("raw cache must be a RADIO MPR cache")
    radio_hash = sha256_file(args.radio_checkpoint)
    target, provenance = _load_capability_mpr_target(
        args.capability_mpr_cache,
        expected_space=args.capability_space,
        raw_cache=raw,
        raw_metadata=raw_metadata,
        radio_checkpoint_sha256=radio_hash,
    )
    device = torch.device(args.device)
    official = FrozenRadioViews.from_radio_checkpoint(args.radio_checkpoint).to(
        device
    ).eval()
    official.requires_grad_(False)
    valid_rows = torch.where(target.valid)[0]
    cosines: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, valid_rows.numel(), int(args.batch_size)):
            rows = valid_rows[start : start + int(args.batch_size)]
            radio = torch.as_tensor(raw["features"])[rows].to(device).float()
            projected_after = (
                official.project_dino_primitives(radio)
                if args.capability_space == "dino_v3"
                else official.project_sam3_primitives(radio)
            )
            projected_before = target.targets[rows].to(device).float()
            cosines.append(
                F.cosine_similarity(
                    projected_after, projected_before, dim=-1, eps=1e-8
                ).cpu()
            )
            counts.append(target.observation_count[rows].cpu())
    cosine = torch.cat(cosines)
    observation_count = torch.cat(counts)

    def summarize(mask: torch.Tensor) -> dict:
        values = cosine[mask]
        return {
            "rows": int(values.numel()),
            "mean_cosine": float(values.mean()) if values.numel() else None,
            "p05_cosine": (
                float(torch.quantile(values, 0.05)) if values.numel() else None
            ),
            "p01_cosine": (
                float(torch.quantile(values, 0.01)) if values.numel() else None
            ),
        }

    bins = {
        "one_view": observation_count == 1,
        "two_views": observation_count == 2,
        "three_to_four_views": (observation_count >= 3) & (observation_count <= 4),
        "five_plus_views": observation_count >= 5,
    }
    canonical_fields: list[dict] = []
    for field_path_value in args.field_checkpoint:
        field_path = Path(field_path_value)
        field, field_payload = load_canonical_field_checkpoint(
            field_path, map_location="cpu"
        )
        if field_payload.get("benchmark_masks_opened", False) or field_payload.get(
            "text_queries_opened", False
        ):
            raise ValueError("audited canonical field used benchmark supervision")
        if str(
            field_payload.get("geometry_fingerprint", {}).get("xyz_sha256", "")
        ) != str(raw.get("geometry_fingerprint", {}).get("xyz_sha256", "")):
            raise ValueError("audited canonical field geometry does not match MPR")
        if field.signature.radio_checkpoint_sha256 != radio_hash:
            raise ValueError("audited canonical field uses another RADIO checkpoint")
        field = field.to(device).eval()
        metrics = _capability_reconstruction_metrics(
            field,
            official,
            {args.capability_space: target},
            valid_rows,
            int(args.batch_size),
        )
        field.cpu()
        canonical_fields.append(
            {
                "path": str(field_path.resolve()),
                "sha256": sha256_file(field_path),
                "capability_target_mode": field_payload.get(
                    "capability_target_mode", "legacy_or_unspecified"
                ),
                "metrics": metrics,
            }
        )
    report = {
        "schema_version": 1,
        "audit": "official_capability_projection_mpr_order_v1",
        "capability_space": args.capability_space,
        "comparison": {
            "left": "official_adaptor(raw_radio_mpr)",
            "right": "matched_mpr(official_adaptor(per_view_raw_radio))",
            "identical_geometry_visibility_and_view_set": True,
        },
        "aggregate": summarize(torch.ones_like(cosine, dtype=torch.bool)),
        "observation_count_bins": {
            name: summarize(mask) for name, mask in bins.items()
        },
        "canonical_field_target_fidelity": canonical_fields,
        "raw_mpr_cache": str(raw_path.resolve()),
        "capability_mpr_target": provenance,
        "radio_checkpoint": str(Path(args.radio_checkpoint).resolve()),
        "radio_checkpoint_sha256": radio_hash,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "metric_feedback_used_for_training": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mpr-cache", required=True)
    parser.add_argument("--capability-mpr-cache", required=True)
    parser.add_argument(
        "--capability-space", choices=("dino_v3", "sam3"), required=True
    )
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--field-checkpoint",
        action="append",
        default=[],
        help=(
            "Optional canonical field to score against the same capability-first "
            "target; repeat for a paired comparison."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
