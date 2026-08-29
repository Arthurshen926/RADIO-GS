"""Write a clean D128 semantic memory with robust source-only aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--semantic-codec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("membership", args.membership),
            ("siglip_teacher", args.siglip_teacher),
            ("semantic_codec", args.semantic_codec),
        )
    }
    membership = torch.load(paths["membership"], map_location="cpu")
    teacher = torch.load(paths["siglip_teacher"], map_location="cpu")
    codec = torch.load(paths["semantic_codec"], map_location="cpu")
    if codec.get("schema") != "radio_gs.sugm_v3.query_discriminative_semantic_codec.v1":
        raise ValueError("semantic writer requires query-discriminative codec")
    rows = int(membership["num_rows"])
    proposals = int(membership["num_proposals"])
    state = codec["state_dict"]
    mean = torch.as_tensor(state["siglip_mean"]).float()
    basis = torch.as_tensor(state["siglip_basis"]).float()
    descriptor = torch.as_tensor(teacher["descriptors"]).float()
    if descriptor.shape != (proposals, 1536):
        raise ValueError("semantic writer proposal axes differ")
    encoded = F.normalize((descriptor - mean) @ basis, dim=-1, eps=1e-8)
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    train_proposal = (proposal_views % 4 == 1) | (proposal_views % 4 == 2)
    membership_proposal = torch.as_tensor(membership["proposal_indices"]).long()
    selected = train_proposal[membership_proposal]
    gaussian_rows = torch.as_tensor(membership["row_indices"]).long()[selected]
    proposal_rows = membership_proposal[selected]
    weight = torch.as_tensor(membership["weights"]).float()[selected]
    quality = (
        torch.as_tensor(membership["proposal_scores"]).float()
        * torch.as_tensor(membership["proposal_stability"]).float()
    ).clamp(0, 1)
    weight = weight * quality[proposal_rows]
    initial_sum = torch.zeros(rows, 128)
    initial_mass = torch.zeros(rows)
    initial_sum.index_add_(0, gaussian_rows, encoded[proposal_rows] * weight[:, None])
    initial_mass.index_add_(0, gaussian_rows, weight)
    initial = F.normalize(initial_sum, dim=-1, eps=1e-8)
    agreement = (encoded[proposal_rows] * initial[gaussian_rows]).sum(1)
    robust_factor = ((agreement - 0.25) / 0.75).clamp(0, 1).square()
    robust_weight = weight * robust_factor
    semantic_sum = torch.zeros(rows, 128)
    semantic_mass = torch.zeros(rows)
    semantic_sum.index_add_(
        0, gaussian_rows, encoded[proposal_rows] * robust_weight[:, None]
    )
    semantic_mass.index_add_(0, gaussian_rows, robust_weight)
    observed = semantic_mass > 1e-8
    semantic = torch.zeros(rows, 128)
    semantic[observed] = F.normalize(
        semantic_sum[observed] / semantic_mass[observed, None], dim=-1, eps=1e-8
    )
    high = agreement >= 0.8
    medium = (agreement >= 0.5) & ~high
    weak = agreement < 0.5
    payload = {
        "schema": "radio_gs.sugm_v3.conflict_aware_semantic_memory.v1",
        "scene": membership["scene"], "semantic": semantic,
        "write_mass": semantic_mass,
        "write_confidence": (semantic_mass / initial_mass.clamp_min(1e-8)).clamp(0, 1),
        "metadata": {
            "source_only": True, "source_train_residues": [1, 2],
            "aggregation": "quality_weighted_two_pass_robust_directional_consensus",
            "correspondence_tiers": {
                "high": int(high.sum()), "medium": int(medium.sum()), "weak": int(weak.sum())
            },
            "observed_rows": int(observed.sum()),
            "unknown_rows": int((~observed).sum()),
            "historical_field_opened": False, "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "persistent_target_block": "D128_semantic_inside_canonical_D512",
            "gaussian_indexed_high_dimensional_sidecars": 0,
            "inputs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print({
        "output": str(output), "sha256": sha256_file(output),
        "observed_rows": int(observed.sum()), "unknown_rows": int((~observed).sum()),
        "high": int(high.sum()), "medium": int(medium.sum()), "weak": int(weak.sum()),
    })


if __name__ == "__main__":
    main()
