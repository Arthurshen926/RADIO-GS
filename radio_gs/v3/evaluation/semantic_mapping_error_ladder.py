"""Source-only text semantic ladder: native D1536 -> codec D128 -> 3D D128."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import proposal_supports, sha256_file


def _metrics(scores: torch.Tensor, states: torch.Tensor) -> dict[str, float]:
    recalls1, recalls5, reciprocal, positive, negative, margins = [], [], [], [], [], []
    for column in range(scores.shape[1]):
        known = states[:, column] >= 0
        truth = states[known, column] == 1
        known_scores = scores[known, column]
        if not bool(truth.any()) or not bool((~truth).any()):
            continue
        order = known_scores.argsort(descending=True)
        ranked_truth = truth[order]
        rank = int(torch.where(ranked_truth)[0][0]) + 1
        pos = known_scores[truth].max()
        neg = known_scores[~truth].max()
        recalls1.append(float(rank <= 1))
        recalls5.append(float(rank <= 5))
        reciprocal.append(1.0 / rank)
        positive.append(float(pos))
        negative.append(float(neg))
        margins.append(float(pos - neg))
    if not reciprocal:
        raise ValueError("semantic ladder split has no evaluable source query")
    values = {
        "queries": len(reciprocal),
        "recall_at_1": sum(recalls1) / len(recalls1),
        "recall_at_5": sum(recalls5) / len(recalls5),
        "mrr": sum(reciprocal) / len(reciprocal),
        "positive_similarity": sum(positive) / len(positive),
        "hardest_negative_similarity": sum(negative) / len(negative),
        "margin": sum(margins) / len(margins),
    }
    return values


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--language-authority", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--residue", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.residue not in (0, 1, 2, 3):
        raise ValueError("semantic ladder residue differs")
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("scene_state", args.scene_state), ("membership", args.membership),
            ("language_authority", args.language_authority),
            ("siglip_teacher", args.siglip_teacher),
            ("text_embeddings", args.text_embeddings),
        )
    }
    state = torch.load(paths["scene_state"], map_location="cpu")
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["language_authority"], map_location="cpu")
    if authority.get("schema") != "radio_gs.sugm_v3.native_language_authority.v3":
        raise ValueError("semantic ladder requires fresh native language authority")
    siglip = torch.load(paths["siglip_teacher"], map_location="cpu")
    text = torch.load(paths["text_embeddings"], map_location="cpu")
    descriptors = torch.as_tensor(siglip["descriptors"]).float()
    text_lookup = {str(name).casefold(): index for index, name in enumerate(text["queries"])}
    query_indices = []
    query_columns = []
    for column, name in enumerate(authority["query_names"]):
        index = text_lookup.get(str(name).casefold())
        if index is not None:
            query_indices.append(index)
            query_columns.append(column)
    if not query_indices:
        raise ValueError("semantic ladder has no text embeddings")
    text_values = torch.as_tensor(text["embeddings"])[query_indices].float()
    states = torch.as_tensor(authority["query_state"]).to(torch.int8)[:, query_columns]
    views = torch.as_tensor(membership["proposal_view_indices"]).long()
    supports = proposal_supports(
        membership["row_indices"], membership["proposal_indices"],
        membership["weights"], int(membership["num_proposals"]),
    )
    selected_indices = torch.tensor([
        index for index, (rows, _) in enumerate(supports)
        if rows.numel() and int(views[index] % 4) == args.residue
    ], dtype=torch.long)
    if not selected_indices.numel():
        raise ValueError("semantic ladder proposal split is empty")
    states = states[selected_indices]
    active_queries = (states == 1).any(0) & (states == 0).any(0)
    if not bool(active_queries.any()):
        raise ValueError("semantic ladder split has no query with explicit positive and negative authority")
    states = states[:, active_queries]
    text_values = text_values[active_queries]
    native_proposal = F.normalize(descriptors[selected_indices], dim=-1, eps=1e-8)
    native_text = F.normalize(text_values, dim=-1, eps=1e-8)
    native_scores = native_proposal @ native_text.T

    global_state = state["global_state_dict"]
    mean = torch.as_tensor(global_state["codec.siglip_mean"]).float()
    basis = torch.as_tensor(global_state["codec.siglip_basis"]).float()
    codec_proposal = F.normalize((descriptors[selected_indices] - mean) @ basis, dim=-1, eps=1e-8)
    codec_text = F.normalize((text_values - mean) @ basis, dim=-1, eps=1e-8)
    codec_scores = codec_proposal @ codec_text.T

    semantic = F.normalize(torch.as_tensor(state["latent"])[:, 320:448].float(), dim=-1, eps=1e-8)
    prototypes = []
    for index in selected_indices.tolist():
        rows, weights = supports[index]
        prototypes.append(F.normalize(
            (semantic[rows] * weights.float()[:, None]).sum(0), dim=0, eps=1e-8
        ))
    memory_scores = torch.stack(prototypes) @ codec_text.T
    payload = {
        "schema": "radio_gs.sugm_v3.semantic_mapping_error_ladder.v2",
        "scene": membership["scene"],
        "residue": args.residue,
        "candidate_proposals": int(selected_indices.numel()),
        "evaluable_queries": int(active_queries.sum()),
        "positive_query_proposal_pairs": int((states == 1).sum()),
        "negative_query_proposal_pairs": int((states == 0).sum()),
        "unknown_query_proposal_pairs": int((states == -1).sum()),
        "known_pair_fraction": float((states >= 0).float().mean()),
        "stages": {
            "native_siglip_d1536": _metrics(native_scores, states),
            "image_pca_codec_d128": _metrics(codec_scores, states),
            "written_gaussian_memory_d128": _metrics(memory_scores, states),
        },
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()
