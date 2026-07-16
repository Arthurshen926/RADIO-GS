#!/usr/bin/env python3
"""Test whether one free RADIO vector can attain raw and capability MPR targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint


def _cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(prediction.float(), target.float(), dim=-1, eps=1e-8)


def audit(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    raw = torch.load(args.raw_mpr_cache, map_location="cpu")
    capability = torch.load(args.capability_mpr_cache, map_location="cpu")
    raw_features = torch.as_tensor(raw["features"]).float()
    cap_features = torch.as_tensor(capability["features"]).float()
    valid = torch.as_tensor(raw["valid"]).bool() & torch.as_tensor(
        capability["valid"]
    ).bool()
    rows = torch.where(valid)[0]
    generator = torch.Generator().manual_seed(int(args.seed))
    if rows.numel() > int(args.sample_count):
        rows = rows[torch.randperm(rows.numel(), generator=generator)[: args.sample_count]]
    raw_target = raw_features[rows].to(device)
    cap_target = cap_features[rows].to(device)
    adaptor = load_radio_adaptor_from_checkpoint(
        args.radio_checkpoint, args.adaptor_name, kind="feature_projection"
    ).to(device).eval()
    adaptor.requires_grad_(False)

    values = torch.nn.Parameter(raw_target.clone())
    optimizer = torch.optim.Adam([values], lr=float(args.learning_rate))

    def losses() -> tuple[torch.Tensor, torch.Tensor]:
        raw_loss = 1.0 - _cosine(values, raw_target).mean()
        projected = adaptor(values[:, None, :])[:, 0]
        cap_loss = 1.0 - _cosine(projected, cap_target).mean()
        return raw_loss, cap_loss

    raw_loss, cap_loss = losses()
    raw_gradient = torch.autograd.grad(raw_loss, values, retain_graph=True)[0]
    cap_gradient = torch.autograd.grad(cap_loss, values)[0]
    gradient_cosine = _cosine(raw_gradient, cap_gradient)
    initial = {
        "raw_mean_cosine": float(1.0 - raw_loss),
        "capability_mean_cosine": float(1.0 - cap_loss),
        "gradient_cosine_mean": float(gradient_cosine.mean()),
        "gradient_conflict_fraction": float((gradient_cosine < 0).float().mean()),
    }
    history = []
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        raw_loss, cap_loss = losses()
        loss = raw_loss + cap_loss
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % int(args.log_every) == 0:
            history.append(
                {
                    "step": step + 1,
                    "raw_mean_cosine": float(1.0 - raw_loss.detach()),
                    "capability_mean_cosine": float(1.0 - cap_loss.detach()),
                }
            )
    raw_loss, cap_loss = losses()
    report = {
        "schema_version": 1,
        "sample_count": int(rows.numel()),
        "adaptor_name": args.adaptor_name,
        "initial": initial,
        "optimized": {
            "raw_mean_cosine": float(1.0 - raw_loss.detach()),
            "capability_mean_cosine": float(1.0 - cap_loss.detach()),
        },
        "history": history,
        "query_free": True,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mpr-cache", required=True)
    parser.add_argument("--capability-mpr-cache", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--adaptor-name", default="sam3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
