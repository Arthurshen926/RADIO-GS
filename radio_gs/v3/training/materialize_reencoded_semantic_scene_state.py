"""Re-encode only the protected D128 semantic block of a sealed scene state."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.contracts.method import validate_scene_state
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.structured_initialization import fixed_jl_projection


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--propagation")
    parser.add_argument("--coverage-policy")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (args.propagation is None) != (args.coverage_policy is None):
        raise ValueError("semantic propagation authority is incomplete")
    state_path = Path(args.scene_state).resolve(strict=True)
    membership_path = Path(args.membership).resolve(strict=True)
    siglip_path = Path(args.siglip_teacher).resolve(strict=True)
    state = torch.load(state_path, map_location="cpu")
    membership = torch.load(membership_path, map_location="cpu")
    siglip = torch.load(siglip_path, map_location="cpu")
    if (
        state.get("schema") != "radio_gs.sugm_v3.unknown_aware_scene_state.v1"
        or not state.get("metadata", {}).get("source_only")
        or state.get("metadata", {}).get("target_rgb_opened")
        or state.get("metadata", {}).get("benchmark_metrics_opened")
    ):
        raise ValueError("semantic re-encoding parent authority differs")
    rows = int(membership["num_rows"])
    descriptors = torch.as_tensor(siglip["descriptors"]).float()
    if descriptors.shape != (int(membership["num_proposals"]), 1536):
        raise ValueError("semantic proposal descriptor axes differ")
    basis = fixed_jl_projection(1536, 128, args.seed)
    proposal = F.normalize(descriptors @ basis, dim=-1, eps=1e-8)
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    train = (proposal_views % 4 == 1) | (proposal_views % 4 == 2)
    membership_proposals = torch.as_tensor(membership["proposal_indices"]).long()
    selected = train[membership_proposals]
    gaussian_rows = torch.as_tensor(membership["row_indices"]).long()[selected]
    proposal_rows = membership_proposals[selected]
    weights = torch.as_tensor(membership["weights"]).float()[selected]
    semantic_sum = torch.zeros(rows, 128)
    semantic_mass = torch.zeros(rows)
    semantic_sum.index_add_(0, gaussian_rows, proposal[proposal_rows] * weights[:, None])
    semantic_mass.index_add_(0, gaussian_rows, weights)
    observed = semantic_mass > 0
    semantic = semantic_sum / semantic_mass.clamp_min(1e-8)[:, None]
    semantic[observed] = F.normalize(semantic[observed], dim=-1, eps=1e-8)

    propagation_receipt = None
    if args.propagation is not None:
        propagation_path = Path(args.propagation).resolve(strict=True)
        policy_path = Path(args.coverage_policy).resolve(strict=True)
        propagation = torch.load(propagation_path, map_location="cpu")
        policy = torch.load(policy_path, map_location="cpu")
        assignments = torch.as_tensor(propagation["assignments"])
        targets = assignments[:, 0].long()
        tier = assignments[:, 2].long()
        authority = torch.as_tensor(policy["visual_write_authority"]).bool()
        eligible = (~authority[targets]) & ((tier == 1) | (tier == 2))
        source_rows = torch.as_tensor(propagation["mixture_source_rows"]).long()[eligible]
        mixture = torch.as_tensor(propagation["mixture_weights"]).float()[eligible]
        valid = source_rows >= 0
        semantic[targets[eligible]] = (
            semantic[source_rows.clamp_min(0)] * mixture[..., None] * valid[..., None]
        ).sum(1)
        propagation_receipt = {
            "filled_rows": int(eligible.sum()),
            "propagation": {"path": str(propagation_path), "sha256": sha256_file(propagation_path)},
            "coverage_policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        }

    parent = torch.as_tensor(state["latent"]).float()
    latent = parent.clone()
    latent[:, 320:448] = semantic
    reliability = torch.as_tensor(state["reliability"]).float()
    validate_scene_state(
        latent, reliability, source_authority_sha256=sha256_file(membership_path)
    )
    global_state = dict(state["global_state_dict"])
    global_state["codec.siglip_mean"] = torch.zeros(1536)
    global_state["codec.siglip_basis"] = basis
    payload = {
        **state,
        "latent": latent,
        "global_state_dict": global_state,
        "metadata": {
            **state["metadata"],
            "semantic_reencoding": {
                "type": "fixed_jl_cross_modal_distance_control",
                "seed": args.seed,
                "observed_rows": int(observed.sum()),
                "propagation": propagation_receipt,
                "parent": {"path": str(state_path), "sha256": sha256_file(state_path)},
                "siglip_teacher": {"path": str(siglip_path), "sha256": sha256_file(siglip_path)},
            },
            "source_only": True,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print({
        "output": str(output), "sha256": sha256_file(output),
        "observed_rows": int(observed.sum()),
        "d320_max_abs_delta": float((latent[:, :320] - parent[:, :320]).abs().max()),
        "d48_d16_max_abs_delta": float((latent[:, 448:] - parent[:, 448:]).abs().max()),
        "d128_mean_abs_delta": float((latent[:, 320:448] - parent[:, 320:448]).abs().mean()),
    })


if __name__ == "__main__":
    main()
