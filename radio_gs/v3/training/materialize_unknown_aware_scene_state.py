"""Seal semantic coverage repair as the sole D512 plus R5 scene state."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.contracts.method import validate_scene_state
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _reliability_scalars(
    policy: dict[str, Any], propagation: dict[str, Any], membership: dict[str, Any]
) -> torch.Tensor:
    rows = int(membership["num_rows"])
    authority = torch.as_tensor(policy["visual_write_authority"]).float()
    coverage = torch.as_tensor(policy["coverage_confidence"]).float()
    unknown = torch.as_tensor(policy["unknown_probability"]).float()
    structural = torch.zeros(rows)
    assignments = torch.as_tensor(propagation["assignments"])
    structural[assignments[:, 0].long()] = assignments[:, 3].float().clamp(0, 1)
    membership_strength = torch.zeros(rows)
    membership_strength.scatter_reduce_(
        0, torch.as_tensor(membership["row_indices"]).long(),
        torch.as_tensor(membership["weights"]).float().clamp(0, 1),
        reduce="amax", include_self=True,
    )
    return torch.stack((authority, coverage, unknown, structural, membership_strength), dim=1)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = Path(args.candidate).resolve(strict=True)
    policy_path = Path(args.coverage_policy).resolve(strict=True)
    propagation_path = Path(args.propagation).resolve(strict=True)
    membership_path = Path(args.membership).resolve(strict=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    propagation = torch.load(propagation_path, map_location="cpu", weights_only=False)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    candidate_metadata = candidate.get("metadata", {})
    if (
        not candidate_metadata.get("source_only")
        or candidate_metadata.get("historical_field_opened")
        or candidate_metadata.get("target_rgb_opened")
        or candidate_metadata.get("benchmark_metrics_opened")
    ):
        raise ValueError("deployment candidate violates source-only authority")
    source_authority_sha256 = sha256_file(membership_path)
    if candidate_metadata.get("membership", {}).get("sha256") != source_authority_sha256:
        raise ValueError("deployment candidate and membership authority differ")
    latent = torch.as_tensor(candidate["state_dict"]["memory"]).float()
    reliability = _reliability_scalars(policy, propagation, membership)
    validate_scene_state(
        latent, reliability, source_authority_sha256=source_authority_sha256
    )
    payload = {
        "schema": "radio_gs.sugm_v3.unknown_aware_scene_state.v1",
        "scene": membership["scene"], "latent": latent,
        "reliability": reliability,
        "reliability_columns": [
            "visual_write_authority", "coverage_confidence", "unknown_probability",
            "structural_candidate_confidence", "semantic_membership_strength",
        ],
        "global_state_dict": {
            name: value for name, value in candidate["state_dict"].items() if name != "memory"
        },
        "metadata": {
            "persistent_gaussian_state": "exactly_one_d512_plus_five_scalars",
            "gaussian_indexed_high_dimensional_sidecars": 0,
            "semantic_unknown_repair": True, "shared_unknown_repair": False,
            "private_architecture": candidate_metadata.get("architecture"),
            "instance_boundary_private_trained": True,
            "source_only": True, "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "inputs": {
                "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
                "coverage_policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
                "propagation": {"path": str(propagation_path), "sha256": sha256_file(propagation_path)},
                "membership": {"path": str(membership_path), "sha256": source_authority_sha256},
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "rows": int(latent.shape[0]), "latent_dim": 512, "reliability_dim": 5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--coverage-policy", required=True)
    parser.add_argument("--propagation", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
