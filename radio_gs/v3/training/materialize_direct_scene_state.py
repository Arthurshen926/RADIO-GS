"""Seal a trained structured candidate when no semantic propagation is selected."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.contracts.method import validate_scene_state
from radio_gs.v3.training.instance_upper_bound import sha256_file


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate_path = Path(args.candidate).resolve(strict=True)
    membership_path = Path(args.membership).resolve(strict=True)
    candidate = torch.load(candidate_path, map_location="cpu")
    membership = torch.load(membership_path, map_location="cpu")
    metadata = candidate.get("metadata", {})
    membership_hash = sha256_file(membership_path)
    if (
        not metadata.get("source_only")
        or metadata.get("historical_field_opened")
        or metadata.get("target_rgb_opened")
        or metadata.get("benchmark_metrics_opened")
        or metadata.get("membership", {}).get("sha256") != membership_hash
    ):
        raise ValueError("direct deployment candidate lineage differs")
    latent = torch.as_tensor(candidate["state_dict"]["memory"]).float()
    rows = latent.shape[0]
    views = torch.as_tensor(membership["view_observed"]).bool()
    train_views = torch.as_tensor(membership["proposal_view_indices"]).new_tensor(
        [index for index in range(views.shape[0]) if index % 4 in (1, 2)]
    )
    observed_fraction = views[train_views].float().mean(0)
    observed = (observed_fraction > 0).float()
    membership_strength = torch.zeros(rows)
    membership_strength.scatter_reduce_(
        0, torch.as_tensor(membership["row_indices"]).long(),
        torch.as_tensor(membership["weights"]).float().clamp(0, 1),
        reduce="amax", include_self=True,
    )
    reliability = torch.stack((
        observed, observed_fraction, 1.0 - observed, observed_fraction,
        membership_strength,
    ), dim=1)
    validate_scene_state(latent, reliability, source_authority_sha256=membership_hash)
    payload = {
        "schema": "radio_gs.sugm_v3.unknown_aware_scene_state.v1",
        "scene": membership["scene"],
        "latent": latent,
        "reliability": reliability,
        "reliability_columns": [
            "visual_write_authority", "coverage_confidence", "unknown_probability",
            "source_observation_fraction", "semantic_membership_strength",
        ],
        "global_state_dict": {
            key: value for key, value in candidate["state_dict"].items() if key != "memory"
        },
        "metadata": {
            "persistent_gaussian_state": "exactly_one_d512_plus_five_scalars",
            "gaussian_indexed_high_dimensional_sidecars": 0,
            "semantic_unknown_repair": False,
            "shared_unknown_repair": False,
            "private_architecture": metadata.get("architecture"),
            "instance_boundary_private_trained": True,
            "source_only": True,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "inputs": {
                "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
                "membership": {"path": str(membership_path), "sha256": membership_hash},
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print({"output": str(output), "sha256": sha256_file(output), "rows": rows})


if __name__ == "__main__":
    main()
