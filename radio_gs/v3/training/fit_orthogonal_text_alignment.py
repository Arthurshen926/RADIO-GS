"""Fit a shared text-to-image semantic alignment from source proposal authority."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.v3.training.instance_upper_bound import proposal_supports, sha256_file


def _parse(value: str) -> tuple[Path, Path, Path]:
    parts = value.split("::")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("scene must be STATE::MEMBERSHIP::RELATION")
    return tuple(Path(item).resolve(strict=True) for item in parts)  # type: ignore[return-value]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", type=_parse, required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--minimum-probability", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.scene) < 2:
        raise ValueError("text alignment requires multiple source scenes")
    text_path = Path(args.text_embeddings).resolve(strict=True)
    text = torch.load(text_path, map_location="cpu")
    text_lookup = {str(name).casefold(): i for i, name in enumerate(text["queries"])}
    text_values = torch.as_tensor(text["embeddings"]).float()
    all_source, all_target, all_weight, receipts = [], [], [], []
    codec_mean = codec_basis = None
    for state_path, membership_path, relation_path in args.scene:
        state = torch.load(state_path, map_location="cpu")
        membership = torch.load(membership_path, map_location="cpu")
        relation = torch.load(relation_path, map_location="cpu")
        state_dict = state.get("global_state_dict", state.get("state_dict", {}))
        memory = torch.as_tensor(state.get("latent", state_dict.get("memory"))).float()
        mean = torch.as_tensor(state_dict["codec.siglip_mean"]).float()
        basis = torch.as_tensor(state_dict["codec.siglip_basis"]).float()
        if codec_mean is None:
            codec_mean, codec_basis = mean, basis
        elif not torch.equal(codec_mean, mean) or not torch.equal(codec_basis, basis):
            raise ValueError("source scenes do not share one semantic codec")
        supports = proposal_supports(
            membership["row_indices"], membership["proposal_indices"],
            membership["weights"], int(membership["num_proposals"]),
        )
        semantic = F.normalize(memory[:, 320:448], dim=-1, eps=1e-8)
        prototypes = []
        valid = []
        for rows, weights in supports:
            if rows.numel():
                prototypes.append(F.normalize((semantic[rows] * weights[:, None]).sum(0), dim=0, eps=1e-8))
                valid.append(True)
            else:
                prototypes.append(torch.zeros(128)); valid.append(False)
        prototypes = torch.stack(prototypes)
        views = torch.as_tensor(membership["proposal_view_indices"]).long()
        train = ((views % 4 == 1) | (views % 4 == 2)) & torch.tensor(valid)
        probability = torch.as_tensor(relation["proposal_probability"]).float()
        query_names = [str(name) for name in relation["query_names"]]
        pair_count = 0
        for column, name in enumerate(query_names):
            if name.casefold() not in text_lookup:
                continue
            selected = train & (probability[:, column] >= args.minimum_probability)
            for index in torch.where(selected)[0].tolist():
                all_source.append(text_values[text_lookup[name.casefold()]])
                all_target.append(prototypes[index])
                all_weight.append(probability[index, column])
                pair_count += 1
        receipts.append({
            "scene": membership["scene"], "weighted_pairs": pair_count,
            "state": {"path": str(state_path), "sha256": sha256_file(state_path)},
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
        })
    if len(all_source) < 16:
        raise ValueError("source text alignment has insufficient authority")
    source = F.normalize((torch.stack(all_source) - codec_mean) @ codec_basis, dim=-1, eps=1e-8)
    target = F.normalize(torch.stack(all_target), dim=-1, eps=1e-8)
    weight = torch.stack(all_weight).sqrt()[:, None]
    # The identity prior fixes the rank-deficient complement of the observed
    # source query span instead of allowing SVD to rotate unseen text axes.
    cross = (source * weight).T @ (target * weight) + torch.eye(128)
    left, _, right_h = torch.linalg.svd(cross, full_matrices=False)
    matrix = left @ right_h
    before = float((source * target).sum(-1).mean())
    after = float(((source @ matrix) * target).sum(-1).mean())
    payload = {
        "schema": "radio_gs.sugm_v3.orthogonal_text_alignment.v1",
        "matrix": matrix,
        "dimension": 128,
        "fit_pair_count": len(all_source),
        "mean_pair_cosine_before": before,
        "mean_pair_cosine_after": after,
        "scene_receipts": receipts,
        "metadata": {
            "source_only": True, "train_view_residues": [1, 2],
            "target_rgb_opened": False, "benchmark_metrics_opened": False,
            "gaussian_indexed_state_added": 0, "shared_across_scenes": True,
            "orthogonal": True,
        },
    }
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print({"output": str(output), "sha256": sha256_file(output), "pairs": len(all_source), "before": before, "after": after})


if __name__ == "__main__":
    main()
