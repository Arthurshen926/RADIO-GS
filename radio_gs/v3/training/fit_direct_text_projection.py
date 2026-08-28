"""Fit a low-rank correction from raw text D1536 to the sealed image D128."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.v3.training.instance_upper_bound import proposal_supports, sha256_file


def _parse(value: str) -> tuple[Path, Path, Path, Path]:
    parts = value.split("::")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "scene must be STATE::MEMBERSHIP::RELATION::SIGLIP_TEACHER"
        )
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
        raise ValueError("direct text projection scene count or ridge differs")
    text_path = Path(args.text_embeddings).resolve(strict=True)
    text = torch.load(text_path, map_location="cpu")
    lookup = {str(name).casefold(): i for i, name in enumerate(text["queries"])}
    text_values = torch.as_tensor(text["embeddings"]).double()
    sources, targets, weights, receipts = [], [], [], []
    codec_mean = codec_basis = None
    for state_path, membership_path, relation_path, siglip_path in args.scene:
        state = torch.load(state_path, map_location="cpu")
        membership = torch.load(membership_path, map_location="cpu")
        relation = torch.load(relation_path, map_location="cpu")
        siglip = torch.load(siglip_path, map_location="cpu")
        descriptors = torch.as_tensor(siglip["descriptors"]).double()
        global_state = state["global_state_dict"]
        mean = torch.as_tensor(global_state["codec.siglip_mean"]).double()
        basis = torch.as_tensor(global_state["codec.siglip_basis"]).double()
        if codec_mean is None:
            codec_mean, codec_basis = mean, basis
        elif not torch.equal(codec_mean, mean) or not torch.equal(codec_basis, basis):
            raise ValueError("direct text projection requires one sealed image codec")
        supports = proposal_supports(
            membership["row_indices"], membership["proposal_indices"],
            membership["weights"], int(membership["num_proposals"]),
        )
        semantic = F.normalize(torch.as_tensor(state["latent"])[:, 320:448].double(), dim=-1)
        views = torch.as_tensor(membership["proposal_view_indices"]).long()
        train = (views % 4 == 1) | (views % 4 == 2)
        probability = torch.as_tensor(relation["proposal_probability"]).double()
        pair_count = 0
        for column, name in enumerate(relation["query_names"]):
            text_index = lookup.get(str(name).casefold())
            if text_index is None:
                continue
            selected = torch.where(train & (probability[:, column] >= args.minimum_probability))[0]
            for index in selected.tolist():
                rows, mass = supports[index]
                if not rows.numel():
                    continue
                target = F.normalize(
                    (semantic[rows] * mass.double()[:, None]).sum(0), dim=0
                )
                sources.append(text_values[text_index] - mean)
                targets.append(target)
                weights.append(probability[index, column])
                pair_count += 1
        receipts.append({
            "scene": membership["scene"], "weighted_pairs": pair_count,
            "state": {"path": str(state_path), "sha256": sha256_file(state_path)},
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
            "siglip_teacher": {"path": str(siglip_path), "sha256": sha256_file(siglip_path)},
        })
    if len(sources) < 16:
        raise ValueError("direct text projection has insufficient source authority")
    source = torch.stack(sources)
    target = F.normalize(torch.stack(targets), dim=-1)
    prior_prediction = F.normalize(source @ codec_basis, dim=-1)
    weight = torch.stack(weights).clamp_min(1e-8).sqrt()[:, None]
    design = source * weight
    residual = (target - source @ codec_basis) * weight
    gram = design @ design.T + torch.eye(design.shape[0], dtype=design.dtype) * args.ridge
    correction = design.T @ torch.linalg.solve(gram, residual)
    fitted_basis = (codec_basis + correction).float()
    prediction = F.normalize(source.float() @ fitted_basis, dim=-1)
    payload = {
        "schema": "radio_gs.sugm_v3.direct_text_projection.v1",
        "basis": fitted_basis,
        "ridge": float(args.ridge),
        "fit_pair_count": len(sources),
        "effective_correction_rank_upper_bound": len(sources),
        "mean_pair_cosine_before": float((prior_prediction * target).sum(-1).mean()),
        "mean_pair_cosine_after": float((prediction.double() * target).sum(-1).mean()),
        "basis_correction_norm": float(correction.norm()),
        "scene_receipts": receipts,
        "metadata": {
            "source_only": True, "train_view_residues": [1, 2],
            "target_rgb_opened": False, "benchmark_metrics_opened": False,
            "gaussian_indexed_state_added": 0, "shared_across_scenes": True,
            "image_semantic_memory_changed": False,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print({
        "output": str(output), "sha256": sha256_file(output), "ridge": args.ridge,
        "pairs": len(sources), "before": payload["mean_pair_cosine_before"],
        "after": payload["mean_pair_cosine_after"],
    })


if __name__ == "__main__":
    main()
