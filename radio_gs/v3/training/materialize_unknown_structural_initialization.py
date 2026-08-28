"""Materialize masked-gate-authorized compact-space interpolation into D512."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _interpolate_unknown(
    memory: torch.Tensor, target_rows: torch.Tensor, source_rows: torch.Tensor,
    weights: torch.Tensor, eligible: torch.Tensor, *, start: int = 0, stop: int = 448,
) -> torch.Tensor:
    output = torch.as_tensor(memory).detach().clone()
    targets = torch.as_tensor(target_rows).long()[eligible]
    sources = torch.as_tensor(source_rows).long()[eligible]
    mixture = torch.as_tensor(weights).float()[eligible]
    valid = sources >= 0
    safe = sources.clamp_min(0)
    value = (output[safe, start:stop] * mixture[..., None] * valid[..., None]).sum(1)
    output[targets, start:stop] = value
    return output


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    initialization_path = Path(args.initialization).resolve(strict=True)
    propagation_path = Path(args.propagation).resolve(strict=True)
    policy_path = Path(args.coverage_policy).resolve(strict=True)
    gate_path = Path(args.masked_gate).resolve(strict=True)
    initialization = torch.load(initialization_path, map_location="cpu", weights_only=False)
    propagation = torch.load(propagation_path, map_location="cpu", weights_only=False)
    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    gate = torch.load(gate_path, map_location="cpu", weights_only=False)
    if not bool(gate["gate"]["pass"]):
        raise ValueError("masked structural reconstruction gate did not pass")
    if policy["metadata"]["inputs"]["propagation"]["sha256"] != sha256_file(propagation_path):
        raise ValueError("coverage policy and propagation authority differ")
    memory = torch.as_tensor(initialization["state_dict"]["memory"]).float()
    assignments = torch.as_tensor(propagation["assignments"])
    target_rows = assignments[:, 0].long()
    tier = assignments[:, 2].long()
    authority = torch.as_tensor(policy["visual_write_authority"]).bool()
    eligible = (~authority[target_rows]) & ((tier == 1) | (tier == 2))
    block = {"visual_semantic": (0, 448), "semantic": (320, 448)}[args.block]
    updated = _interpolate_unknown(
        memory, target_rows, torch.as_tensor(propagation["mixture_source_rows"]),
        torch.as_tensor(propagation["mixture_weights"]), eligible,
        start=block[0], stop=block[1],
    )
    state_dict = dict(initialization["state_dict"])
    state_dict["memory"] = updated
    payload = {
        "schema": initialization["schema"], "state_dict": state_dict,
        "metadata": {
            **initialization["metadata"],
            "phase_order": "masked_gate_authorized_unknown_structural_initialization",
            "unknown_structural_initialization": {
                "filled_rows": int(eligible.sum()), "block": args.block,
                "column_start": block[0], "column_stop": block[1],
                "dimensions": block[1] - block[0],
                "private_dimensions_changed": 0, "observed_rows_changed": 0,
                "propagation_top_k": int(propagation["metadata"]["propagation_top_k"]),
                "initialization": {"path": str(initialization_path), "sha256": sha256_file(initialization_path)},
                "propagation": {"path": str(propagation_path), "sha256": sha256_file(propagation_path)},
                "coverage_policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
                "masked_gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
            },
            "target_rgb_opened": False, "benchmark_metrics_opened": False,
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "filled_rows": int(eligible.sum()), "observed_rows_changed": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--propagation", required=True)
    parser.add_argument("--coverage-policy", required=True)
    parser.add_argument("--masked-gate", required=True)
    parser.add_argument("--block", choices=("visual_semantic", "semantic"), default="visual_semantic")
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
