"""Fit a strongly regularized text-only affine map from source authority."""

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
    parser.add_argument("--ridge", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.scene) < 2 or args.ridge <= 0:
        raise ValueError("affine text alignment scene count or ridge differs")
    text_path = Path(args.text_embeddings).resolve(strict=True)
    text = torch.load(text_path, map_location="cpu")
    text_lookup = {str(name).casefold(): i for i, name in enumerate(text["queries"])}
    text_values = torch.as_tensor(text["embeddings"]).double()
    sources, targets, weights, receipts = [], [], [], []
    codec_mean = codec_basis = None
    for state_path, membership_path, relation_path in args.scene:
        state = torch.load(state_path, map_location="cpu")
        membership = torch.load(membership_path, map_location="cpu")
        relation = torch.load(relation_path, map_location="cpu")
        state_dict = state.get("global_state_dict", state.get("state_dict", {}))
        memory = torch.as_tensor(state.get("latent", state_dict.get("memory"))).double()
        mean = torch.as_tensor(state_dict["codec.siglip_mean"]).double()
        basis = torch.as_tensor(state_dict["codec.siglip_basis"]).double()
        if codec_mean is None:
            codec_mean, codec_basis = mean, basis
        elif not torch.equal(codec_mean, mean) or not torch.equal(codec_basis, basis):
            raise ValueError("source scenes do not share one semantic codec")
        supports = proposal_supports(
            membership["row_indices"], membership["proposal_indices"],
            membership["weights"], int(membership["num_proposals"]),
        )
        semantic = F.normalize(memory[:, 320:448], dim=-1, eps=1e-12)
        views = torch.as_tensor(membership["proposal_view_indices"]).long()
        train = (views % 4 == 1) | (views % 4 == 2)
        probability = torch.as_tensor(relation["proposal_probability"]).double()
        pair_count = 0
        for column, name in enumerate(relation["query_names"]):
            lookup = text_lookup.get(str(name).casefold())
            if lookup is None:
                continue
            selected = torch.where(train & (probability[:, column] >= args.minimum_probability))[0]
            for index in selected.tolist():
                rows, mass = supports[index]
                if not rows.numel():
                    continue
                target = F.normalize(
                    (semantic[rows] * mass.double()[:, None]).sum(0), dim=0, eps=1e-12
                )
                sources.append(text_values[lookup])
                targets.append(target)
                weights.append(probability[index, column])
                pair_count += 1
        receipts.append({
            "scene": membership["scene"], "weighted_pairs": pair_count,
            "state": {"path": str(state_path), "sha256": sha256_file(state_path)},
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
        })
    if len(sources) < 16:
        raise ValueError("affine text alignment has insufficient source authority")
    source = F.normalize((torch.stack(sources) - codec_mean) @ codec_basis, dim=-1, eps=1e-12)
    target = F.normalize(torch.stack(targets), dim=-1, eps=1e-12)
    design = torch.cat((source, torch.ones(source.shape[0], 1, dtype=source.dtype)), dim=1)
    weight = torch.stack(weights).clamp_min(1e-8)
    prior = torch.cat((torch.eye(128, dtype=source.dtype), torch.zeros(1, 128, dtype=source.dtype)))
    regularizer = torch.eye(129, dtype=source.dtype) * float(args.ridge)
    normal = design.T @ (design * weight[:, None]) + regularizer
    right = design.T @ (target * weight[:, None]) + regularizer @ prior
    solution = torch.linalg.solve(normal, right)
    matrix, bias = solution[:128].float(), solution[128].float()
    predicted = F.normalize(source.float() @ matrix + bias, dim=-1, eps=1e-8)
    payload = {
        "schema": "radio_gs.sugm_v3.affine_text_alignment.v1",
        "matrix": matrix,
        "bias": bias,
        "dimension": 128,
        "fit_pair_count": len(sources),
        "ridge": float(args.ridge),
        "mean_pair_cosine_before": float((source * target).sum(-1).mean()),
        "mean_pair_cosine_after": float((predicted.double() * target).sum(-1).mean()),
        "matrix_identity_deviation": float((matrix - torch.eye(128)).norm()),
        "bias_norm": float(bias.norm()),
        "scene_receipts": receipts,
        "metadata": {
            "source_only": True, "train_view_residues": [1, 2],
            "target_rgb_opened": False, "benchmark_metrics_opened": False,
            "gaussian_indexed_state_added": 0, "shared_across_scenes": True,
            "identity_regularized": True,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print({
        "output": str(output), "sha256": sha256_file(output),
        "pairs": len(sources), "ridge": args.ridge,
        "before": payload["mean_pair_cosine_before"],
        "after": payload["mean_pair_cosine_after"],
        "bias_norm": payload["bias_norm"],
    })


if __name__ == "__main__":
    main()
