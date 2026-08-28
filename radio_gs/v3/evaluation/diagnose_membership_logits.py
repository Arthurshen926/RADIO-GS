"""Source-heldout diagnosis for absolute versus relative D48 membership.

This runner opens only sealed source authority.  The relative result is an
upper-bound diagnostic because its negative prototypes come from explicit
source relation edges; it is not a deployable query path by itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.source_heldout import evaluate_source_heldout
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.membership import pool_prototype
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import sha256_file, validate_source_only_inputs
from radio_gs.v3.training.rendered_mask import render_membership
from radio_gs.v3.training.run_instance_upper_bound import (
    load_episodes,
    training_support_for_heldout,
)


def different_training_adjacency(
    train_proposals: set[int], relation: dict
) -> dict[int, tuple[int, ...]]:
    labels = torch.as_tensor(relation["edge_relation"]).to(torch.int8)
    left = torch.as_tensor(relation["edge_left"]).long()
    right = torch.as_tensor(relation["edge_right"]).long()
    peers: dict[int, set[int]] = {}
    for a, b, label in zip(left.tolist(), right.tolist(), labels.tolist()):
        if label != 0:
            continue
        if b in train_proposals:
            peers.setdefault(a, set()).add(b)
        if a in train_proposals:
            peers.setdefault(b, set()).add(a)
    return {key: tuple(sorted(value)) for key, value in peers.items()}


def _mean_metrics(values: list) -> dict[str, float]:
    return {
        key: sum(float(getattr(item, key)) for item in values) / len(values)
        for key in ("mask_iou", "brier", "unknown_fp_mass")
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--evaluation-residue", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature differs")
    if args.evaluation_residue in (1, 2):
        raise ValueError("evaluation residue overlaps source training")
    if args.max_episodes <= 0:
        raise ValueError("maximum episode count differs")

    state_path = Path(args.scene_state).resolve(strict=True)
    membership_path = Path(args.membership).resolve(strict=True)
    relation_path = Path(args.relation).resolve(strict=True)
    siglip_path = Path(args.siglip_teacher).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    relation = torch.load(relation_path, map_location="cpu")
    siglip = torch.load(siglip_path, map_location="cpu")
    descriptors = torch.as_tensor(siglip["descriptors"]).float()
    validate_source_only_inputs(membership, relation)
    if descriptors.shape != (int(membership["num_proposals"]), 1536):
        raise ValueError("image-query descriptor proposal axis differs")
    interface = load_query_interface(state_path, device=args.device)
    episodes, supports = load_episodes(membership, relation)
    valid = [item for item in episodes if supports[item.proposal_index][0].numel()]
    training = [item for item in valid if item.view_index % 4 in (1, 2)]
    evaluation = [
        item for item in valid if item.view_index % 4 == args.evaluation_residue
    ]
    train_proposals = {item.proposal_index for item in training}
    negative_adjacency = different_training_adjacency(train_proposals, relation)

    absolute_metrics = []
    relative_metrics = []
    image_query_metrics = []
    anchor_precisions = []
    prototype_cosines = []
    hit_totals = {
        "foreground_weight": 0.0,
        "background_weight": 0.0,
        "foreground_positive_similarity": 0.0,
        "background_positive_similarity": 0.0,
        "foreground_absolute_probability": 0.0,
        "background_absolute_probability": 0.0,
        "foreground_relative_probability": 0.0,
        "background_relative_probability": 0.0,
        "foreground_absolute_above_half": 0.0,
        "background_absolute_above_half": 0.0,
        "foreground_relative_above_half": 0.0,
        "background_relative_above_half": 0.0,
    }
    for episode in evaluation:
        if len(relative_metrics) >= args.max_episodes:
            break
        positive_support = training_support_for_heldout(
            episode.proposal_index, train_proposals, supports, relation
        )
        negative_peers = negative_adjacency.get(episode.proposal_index, ())
        if (
            positive_support is None
            or not negative_peers
            or not bool((episode.known & ~episode.target).any())
        ):
            continue
        positive_rows, positive_weights = positive_support
        negative_supports = [supports[index] for index in negative_peers]
        packet = QueryPacket("image", token=descriptors[episode.proposal_index])
        query_rows, query_weights, _ = interface.compile_identity_anchors(
            packet, topk=args.topk
        )
        segments = (
            [positive_rows]
            + [value[0] for value in negative_supports]
            + [query_rows.cpu()]
        )
        hit_offset = sum(value.numel() for value in segments)
        combined = torch.cat(segments + [episode.gaussian_ids])
        unique, inverse = torch.unique(combined, sorted=True, return_inverse=True)
        embedding = interface.model.instance_view(
            episode.scale, unique.to(interface.model.memory.device)
        )
        cursor = 0
        positive_count = positive_rows.numel()
        positive = pool_prototype(
            embedding[inverse[cursor : cursor + positive_count].to(embedding.device)],
            positive_weights.to(embedding.device),
        )
        cursor += positive_count
        negatives = []
        for rows, weights in negative_supports:
            count = rows.numel()
            negatives.append(pool_prototype(
                embedding[inverse[cursor : cursor + count].to(embedding.device)],
                weights.to(embedding.device),
            ))
            cursor += count
        query_count = query_rows.numel()
        query_prototype = pool_prototype(
            embedding[inverse[cursor : cursor + query_count].to(embedding.device)],
            query_weights.to(embedding.device),
        )
        cursor += query_count
        hit_embedding = F.normalize(
            embedding[inverse[hit_offset:].to(embedding.device)], dim=-1, eps=1e-8
        )
        positive_similarity = hit_embedding @ positive
        negative_similarity = (hit_embedding @ torch.stack(negatives).T).max(1).values
        absolute = (positive_similarity / args.temperature).sigmoid()
        relative = (
            (positive_similarity - negative_similarity) / args.temperature
        ).sigmoid()
        image_query = ((hit_embedding @ query_prototype) / args.temperature).sigmoid()

        render_args = (
            torch.arange(absolute.numel(), device=absolute.device),
            episode.pixel_ids.to(absolute.device),
            episode.contribution_weights.to(absolute.device),
        )
        absolute_image = render_membership(
            absolute, *render_args, num_pixels=episode.target.numel()
        ).cpu()
        relative_image = render_membership(
            relative, *render_args, num_pixels=episode.target.numel()
        ).cpu()
        image_query_image = render_membership(
            image_query, *render_args, num_pixels=episode.target.numel()
        ).cpu()
        zeros = torch.zeros_like(absolute_image)
        metric_args = (
            episode.target.flatten(), episode.known.flatten(), episode.unknown.flatten(),
            zeros, episode.boundary.flatten(),
        )
        absolute_metrics.append(evaluate_source_heldout(absolute_image, *metric_args))
        relative_metrics.append(evaluate_source_heldout(relative_image, *metric_args))
        image_query_metrics.append(evaluate_source_heldout(image_query_image, *metric_args))
        authority_rows = torch.unique(positive_rows).to(query_rows.device)
        anchor_precisions.append(float(torch.isin(query_rows, authority_rows).float().mean()))
        prototype_cosines.append(float(positive @ query_prototype))

        pixel = episode.pixel_ids.to(absolute.device)
        known = episode.known.flatten().to(absolute.device)[pixel]
        foreground = episode.target.flatten().to(absolute.device)[pixel] & known
        background = ~episode.target.flatten().to(absolute.device)[pixel] & known
        weight = episode.contribution_weights.to(absolute.device)
        for name, selected in (("foreground", foreground), ("background", background)):
            selected_weight = weight[selected]
            total = float(selected_weight.sum())
            hit_totals[f"{name}_weight"] += total
            if total == 0:
                continue
            hit_totals[f"{name}_positive_similarity"] += float(
                (positive_similarity[selected] * selected_weight).sum()
            )
            hit_totals[f"{name}_absolute_probability"] += float(
                (absolute[selected] * selected_weight).sum()
            )
            hit_totals[f"{name}_relative_probability"] += float(
                (relative[selected] * selected_weight).sum()
            )
            hit_totals[f"{name}_absolute_above_half"] += float(
                ((absolute[selected] >= 0.5).float() * selected_weight).sum()
            )
            hit_totals[f"{name}_relative_above_half"] += float(
                ((relative[selected] >= 0.5).float() * selected_weight).sum()
            )
    if not relative_metrics:
        raise ValueError("no heldout episode has both positive and explicit negative authority")

    hit_summary: dict[str, dict[str, float]] = {}
    for name in ("foreground", "background"):
        denominator = hit_totals[f"{name}_weight"]
        hit_summary[name] = {
            key: hit_totals[f"{name}_{key}"] / denominator
            for key in (
                "positive_similarity", "absolute_probability", "relative_probability",
                "absolute_above_half", "relative_above_half",
            )
        }
    payload = {
        "schema": "radio_gs.sugm_v3.membership_logit_diagnostic.v1",
        "scene": membership["scene"],
        "evaluation_residue": args.evaluation_residue,
        "evaluation_proposals": len(relative_metrics),
        "max_episodes": args.max_episodes,
        "temperature": args.temperature,
        "absolute_positive_only": _mean_metrics(absolute_metrics),
        "relative_explicit_negative_upper_bound": _mean_metrics(relative_metrics),
        "image_query_selected_anchors": _mean_metrics(image_query_metrics),
        "identity_anchor_training_support_precision": sum(anchor_precisions) / len(anchor_precisions),
        "oracle_to_query_prototype_cosine": sum(prototype_cosines) / len(prototype_cosines),
        "contribution_weighted_hit_statistics": hit_summary,
        "relative_result_is_deployable": False,
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            "scene_state": {"path": str(state_path), "sha256": sha256_file(state_path)},
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
            "siglip_teacher": {"path": str(siglip_path), "sha256": sha256_file(siglip_path)},
        },
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()
