"""Source-only image/text identity-anchor error ladder for SUGM-v3."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.source_heldout import evaluate_source_heldout
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.membership import membership_from_prototype, pool_prototype
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import sha256_file, validate_source_only_inputs
from radio_gs.v3.training.run_instance_upper_bound import load_episodes, training_support_for_heldout


def _lookup(payload: dict) -> dict[str, torch.Tensor]:
    names = [str(value).casefold() for value in payload["queries"]]
    values = torch.as_tensor(payload["embeddings"]).float()
    if values.shape != (len(names), 1536):
        raise ValueError("text embedding axes differ")
    return {name: values[index] for index, name in enumerate(names)}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--identity-adapter")
    parser.add_argument("--text-alignment")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--minimum-probability", type=float, default=0.05)
    parser.add_argument("--evaluation-residue", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.evaluation_residue in (1, 2) or args.max_pairs <= 0:
        raise ValueError("identity diagnostic split or pair budget differs")

    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("scene_state", args.scene_state), ("membership", args.membership),
            ("relation", args.relation), ("siglip_teacher", args.siglip_teacher),
            ("text_embeddings", args.text_embeddings),
        )
    }
    membership = torch.load(paths["membership"], map_location="cpu")
    relation = torch.load(paths["relation"], map_location="cpu")
    siglip = torch.load(paths["siglip_teacher"], map_location="cpu")
    text = _lookup(torch.load(paths["text_embeddings"], map_location="cpu"))
    descriptors = torch.as_tensor(siglip["descriptors"]).float()
    validate_source_only_inputs(membership, relation)
    if descriptors.shape != (int(membership["num_proposals"]), 1536):
        raise ValueError("image descriptor proposal axis differs")
    interface = load_query_interface(
        paths["scene_state"], device=args.device,
        identity_adapter_path=args.identity_adapter,
        text_alignment_path=args.text_alignment,
        center_semantic_identity=True,
    )
    episodes, supports = load_episodes(membership, relation)
    valid = [item for item in episodes if supports[item.proposal_index][0].numel()]
    training = [item for item in valid if item.view_index % 4 in (1, 2)]
    evaluation = [item for item in valid if item.view_index % 4 == args.evaluation_residue]
    train_proposals = {item.proposal_index for item in training}
    probabilities = torch.as_tensor(relation["proposal_probability"]).float()
    query_names = [str(value) for value in relation["query_names"]]
    if probabilities.shape != (len(episodes), len(query_names)):
        raise ValueError("source query probability axes differ")

    totals = {
        modality: {"precision": 0.0, "peak": 0.0, "authority_score": 0.0, "global_score": 0.0}
        for modality in ("image", "text")
    }
    render_metrics = {modality: [] for modality in ("image", "text")}
    pairs = 0
    missing_text_queries: set[str] = set()
    for episode in evaluation:
        support = training_support_for_heldout(
            episode.proposal_index, train_proposals, supports, relation
        )
        if support is None:
            continue
        columns = torch.where(
            probabilities[episode.proposal_index] >= args.minimum_probability
        )[0].tolist()
        authority = torch.unique(support[0]).to(interface.model.memory.device)
        for column in columns:
            name = query_names[column]
            token = text.get(name.casefold())
            if token is None:
                missing_text_queries.add(name)
                continue
            packets = {
                "image": QueryPacket("image", descriptors[episode.proposal_index]),
                "text": QueryPacket("text", token),
            }
            for modality, packet in packets.items():
                rows, anchor_weights, score = interface.compile_identity_anchors(
                    packet, topk=args.topk
                )
                totals[modality]["precision"] += float(torch.isin(rows, authority).float().mean())
                totals[modality]["peak"] += float(bool((score.argmax() == authority).any()))
                totals[modality]["authority_score"] += float(score[authority].max())
                totals[modality]["global_score"] += float(score.max())
                support_embedding = interface.model.instance_view(episode.scale, rows)
                prototype = pool_prototype(support_embedding, anchor_weights)
                unique_hits, hit_inverse = torch.unique(
                    episode.gaussian_ids, sorted=True, return_inverse=True
                )
                unique_embedding = interface.model.instance_view(
                    episode.scale, unique_hits.to(interface.model.memory.device)
                )
                hit_embedding = unique_embedding[
                    hit_inverse.to(unique_embedding.device)
                ]
                hit_probability = membership_from_prototype(
                    hit_embedding, prototype, temperature=0.15
                )
                prediction = interface.render_posterior(
                    hit_probability,
                    torch.arange(hit_probability.numel(), device=hit_probability.device),
                    episode.pixel_ids.to(hit_probability.device),
                    episode.contribution_weights.to(hit_probability.device),
                    num_pixels=episode.target.numel(),
                ).cpu()
                zeros = torch.zeros_like(prediction)
                render_metrics[modality].append(evaluate_source_heldout(
                    prediction, episode.target.flatten(), episode.known.flatten(),
                    episode.unknown.flatten(), zeros, episode.boundary.flatten(),
                ))
            pairs += 1
            if pairs >= args.max_pairs:
                break
        if pairs >= args.max_pairs:
            break
    if not pairs:
        raise ValueError("source text identity cohort is empty")
    summary = {
        modality: {key: value / pairs for key, value in values.items()}
        for modality, values in totals.items()
    }
    render_summary = {
        modality: {
            key: sum(float(getattr(item, key)) for item in values) / len(values)
            for key in ("mask_iou", "brier", "unknown_fp_mass")
        }
        for modality, values in render_metrics.items()
    }
    payload = {
        "schema": "radio_gs.sugm_v3.text_identity_anchor_diagnostic.v1",
        "scene": membership["scene"],
        "evaluation_residue": args.evaluation_residue,
        "evaluated_query_proposal_pairs": pairs,
        "topk": args.topk,
        "minimum_probability": args.minimum_probability,
        "metrics": summary,
        "render_metrics": render_summary,
        "missing_text_queries": sorted(missing_text_queries),
        "center_semantic_identity": True,
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "identity_adapter": str(Path(args.identity_adapter).resolve()) if args.identity_adapter else None,
        "text_alignment": str(Path(args.text_alignment).resolve()) if args.text_alignment else None,
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()
