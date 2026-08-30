"""Seal the sole clean parent with protected D320/D128 and reset child blocks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.contracts.method import validate_scene_state
from radio_gs.v3.memory.structured_memory import LowRankPrivateBranchMemory
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _reset_child_global_state(latent: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    model = LowRankPrivateBranchMemory(latent)
    with torch.no_grad():
        for module in (
            model.visual_to_instance,
            model.context_to_boundary,
            model.scale_adapter,
            model.instance_down,
            model.instance_up,
            model.boundary_down,
            model.boundary_up,
        ):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key != "memory"
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-semantic-state", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("clean_semantic_state", args.clean_semantic_state),
            ("membership", args.membership),
            ("authority", args.authority),
        )
    }
    parent = torch.load(paths["clean_semantic_state"], map_location="cpu")
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    metadata = parent.get("metadata", {})
    if (
        parent.get("schema") != "radio_gs.sugm_v3.unknown_aware_scene_state.v1"
        or not metadata.get("source_only")
        or metadata.get("target_rgb_opened")
        or metadata.get("benchmark_metrics_opened")
        or not metadata.get("clean_semantic_rewrite")
        or authority.get("schema") != "radio_gs.sugm_v3.native_language_authority.v3"
    ):
        raise ValueError("clean parent lineage differs")
    if parent["scene"] != membership["scene"] or parent["scene"] != authority["scene"]:
        raise ValueError("clean parent scene differs")
    membership_hash = sha256_file(paths["membership"])
    if metadata.get("inputs", {}).get("membership", {}).get("sha256") != membership_hash:
        raise ValueError("clean parent membership authority differs")
    original = torch.as_tensor(parent["latent"]).float()
    reliability = torch.as_tensor(parent["reliability"]).float()
    latent = original.clone()
    latent[:, 448:] = 0
    validate_scene_state(latent, reliability, source_authority_sha256=membership_hash)
    reset_state = _reset_child_global_state(latent, args.seed)
    protected_global = {
        key: torch.as_tensor(value).detach().cpu()
        for key, value in parent["global_state_dict"].items()
        if key.startswith("visual_codec.") or key.startswith("codec.")
    }
    boundary_head = nn.Linear(16, 1)
    nn.init.zeros_(boundary_head.weight)
    nn.init.zeros_(boundary_head.bias)
    global_state = {
        **reset_state,
        **protected_global,
        "boundary_head.weight": boundary_head.weight.detach().cpu(),
        "boundary_head.bias": boundary_head.bias.detach().cpu(),
    }
    rewrite = metadata["clean_semantic_rewrite"]
    payload = {
        **parent,
        "latent": latent,
        "global_state_dict": global_state,
        "metadata": {
            **metadata,
            "private_architecture": "clean_parent_children_reset",
            "instance_boundary_private_trained": False,
            "clean_parent_contract": {
                "version": "clean_parent_v1",
                "layout": {"visual": 320, "semantic": 128, "instance": 48, "boundary": 16},
                "d320_tensor_sha256": tensor_sha256(latent[:, :320]),
                "d128_tensor_sha256": tensor_sha256(latent[:, 320:448]),
                "semantic_memory": rewrite["semantic_memory"],
                "semantic_codec": rewrite["semantic_codec"],
                "authority": {
                    "path": str(paths["authority"]),
                    "sha256": sha256_file(paths["authority"]),
                },
                "membership": {
                    "path": str(paths["membership"]),
                    "sha256": membership_hash,
                },
                "source_split": {"train_residues": [1, 2], "dev_residue": 3, "audit_residue": 0},
                "identity_score_gauge": "raw_unit_D128_text_cosine_v1",
                "anchor_policy": "positive_identity_topk_before_null",
                "deployment_order": "gaussian_logit_then_sigmoid_then_render_or_pool",
                "posterior_schema": "identity_plus_bounded_instance_plus_signed_boundary_minus_null_v1",
                "child_initialization": "D48_D16_and_all_child_global_paths_exact_zero",
                "seed": args.seed,
            },
            "source_only": True,
            "historical_language_authority_opened": False,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "inputs": {
                "clean_semantic_state": {
                    "path": str(paths["clean_semantic_state"]),
                    "sha256": sha256_file(paths["clean_semantic_state"]),
                },
                "membership": {"path": str(paths["membership"]), "sha256": membership_hash},
                "authority": {
                    "path": str(paths["authority"]),
                    "sha256": sha256_file(paths["authority"]),
                },
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print(
        {
            "output": str(output),
            "sha256": sha256_file(output),
            "scene": parent["scene"],
            "d320_max_abs_delta": float((latent[:, :320] - original[:, :320]).abs().max()),
            "d128_max_abs_delta": float((latent[:, 320:448] - original[:, 320:448]).abs().max()),
            "d48_max_abs": float(latent[:, 448:496].abs().max()),
            "d16_max_abs": float(latent[:, 496:].abs().max()),
        }
    )


if __name__ == "__main__":
    main()


__all__ = ["_reset_child_global_state"]
