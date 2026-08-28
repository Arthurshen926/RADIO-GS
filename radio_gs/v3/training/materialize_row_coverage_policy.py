"""Freeze fail-closed per-row visual authority and coverage confidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _coverage_policy(
    assignments: torch.Tensor, assignment_confidence: torch.Tensor, *, fine_gate_pass: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tier = assignments[:, 2].long()
    observed = tier == 0
    fine = (tier == 1) | (tier == 2)
    visual_authority = observed | (fine & fine_gate_pass)
    confidence = torch.where(
        observed, torch.ones_like(assignment_confidence),
        torch.where(fine & fine_gate_pass, assignment_confidence.clamp(0, 1), torch.zeros_like(assignment_confidence)),
    )
    unknown = 1 - confidence
    return visual_authority, confidence, unknown


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    propagation_path = Path(args.propagation).resolve(strict=True)
    gate_path = Path(args.source_dev_gate).resolve(strict=True)
    propagation = torch.load(propagation_path, map_location="cpu", weights_only=False)
    gate = torch.load(gate_path, map_location="cpu", weights_only=False)
    if gate["metadata"]["inputs"]["propagation"]["sha256"] != sha256_file(propagation_path):
        raise ValueError("source-dev gate and propagation authority differ")
    assignments = torch.as_tensor(propagation["assignments"])
    membership_path = Path(propagation["metadata"]["inputs"]["membership"]["path"]).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    num_rows = int(membership["num_rows"])
    fine_gate_pass = bool(gate["gate"]["pass"])
    assigned_authority, assigned_confidence, _ = _coverage_policy(
        assignments, assignments[:, 3].float(), fine_gate_pass=fine_gate_pass
    )
    assigned_rows = assignments[:, 0].long()
    visual_authority = torch.zeros(num_rows, dtype=torch.bool)
    confidence = torch.zeros(num_rows)
    tier = torch.full((num_rows,), -1, dtype=torch.long)
    visual_authority[assigned_rows] = assigned_authority
    confidence[assigned_rows] = assigned_confidence
    tier[assigned_rows] = assignments[:, 2].long()
    unknown = 1 - confidence
    payload = {
        "schema": "radio_gs.sugm_v3.row_coverage_policy.v1",
        "scene": propagation["scene"],
        "row_indices": torch.arange(num_rows),
        "visual_write_authority": visual_authority,
        "coverage_confidence": confidence,
        "unknown_probability": unknown,
        "propagation_tier": tier,
        "metadata": {
            "fine_gate_pass": fine_gate_pass,
            "authorized_rows": int(visual_authority.sum()),
            "abstained_rows": int((~visual_authority).sum()),
            "fail_closed": True, "source_only": True,
            "target_rgb_opened": False, "benchmark_metrics_opened": False,
            "inputs": {
                "propagation": {"path": str(propagation_path), "sha256": sha256_file(propagation_path)},
                "source_dev_gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "authorized_rows": int(visual_authority.sum()), "abstained_rows": int((~visual_authority).sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--propagation", required=True)
    parser.add_argument("--source-dev-gate", required=True)
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
