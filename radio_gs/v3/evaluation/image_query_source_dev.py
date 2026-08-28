"""Source-dev image-token identity to the shared Gaussian posterior."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.source_heldout import evaluate_source_heldout
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import sha256_file, validate_source_only_inputs
from radio_gs.v3.training.run_instance_upper_bound import (
    load_episodes,
    training_support_for_heldout,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--membership-margin", type=float, default=0.0)
    parser.add_argument("--center-semantic-identity", action="store_true")
    parser.add_argument("--identity-extent-weight", type=float, default=0.0)
    parser.add_argument("--identity-adapter")
    parser.add_argument("--evaluation-residue", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
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
    interface = load_query_interface(
        state_path, device=args.device, identity_adapter_path=args.identity_adapter,
        center_semantic_identity=args.center_semantic_identity,
    )
    episodes, supports = load_episodes(membership, relation)
    valid = [item for item in episodes if supports[item.proposal_index][0].numel()]
    training = [item for item in valid if item.view_index % 4 in (1, 2)]
    if args.evaluation_residue in (1, 2):
        raise ValueError("evaluation residue overlaps source training")
    evaluation = [
        item for item in valid if item.view_index % 4 == args.evaluation_residue
    ]
    train_proposals = {item.proposal_index for item in training}
    boundary = interface.boundary_probability()
    metrics = []
    peak_hits = []
    anchor_precisions = []
    evaluated = 0
    for episode in evaluation:
        support = training_support_for_heldout(
            episode.proposal_index, train_proposals, supports, relation
        )
        if support is None or not bool((episode.known & ~episode.target).any()):
            continue
        packet = QueryPacket("image", token=descriptors[episode.proposal_index])
        rows, _weights, identity = interface.compile_identity_anchors(packet, topk=args.topk)
        posterior, _ = interface.posterior_from_packet(
            packet, scale=episode.scale, topk=args.topk, temperature=args.temperature,
            membership_margin=args.membership_margin,
            identity_extent_weight=args.identity_extent_weight,
        )
        prediction = interface.render_posterior(
            posterior,
            episode.gaussian_ids.to(posterior.device),
            episode.pixel_ids.to(posterior.device),
            episode.contribution_weights.to(posterior.device),
            num_pixels=episode.target.numel(),
        )
        edge = interface.render_posterior(
            boundary,
            episode.gaussian_ids.to(boundary.device),
            episode.pixel_ids.to(boundary.device),
            episode.contribution_weights.to(boundary.device),
            num_pixels=episode.target.numel(),
        )
        metrics.append(evaluate_source_heldout(
            prediction.cpu(), episode.target.flatten(), episode.known.flatten(),
            episode.unknown.flatten(), edge.cpu(), episode.boundary.flatten(),
        ))
        authority_rows = torch.unique(support[0]).to(rows.device)
        peak_hits.append(bool((authority_rows == identity.argmax()).any()))
        anchor_precisions.append(float(torch.isin(rows, authority_rows).float().mean()))
        evaluated += 1
    if not metrics:
        raise ValueError("image-query source-dev cohort is empty")
    summary = {
        name: sum(getattr(item, name) for item in metrics) / len(metrics)
        for name in ("mask_iou", "brier", "boundary_f", "unknown_fp_mass")
    }
    payload = {
        "schema": "radio_gs.sugm_v3.image_query_source_dev.v1",
        "scene": membership["scene"],
        "evaluation_proposals": evaluated,
        "topk": args.topk,
        "evaluation_residue": args.evaluation_residue,
        "membership_margin": args.membership_margin,
        "center_semantic_identity": args.center_semantic_identity,
        "identity_extent_weight": args.identity_extent_weight,
        "metrics": summary,
        "identity_peak_training_support_hit_rate": sum(peak_hits) / len(peak_hits),
        "identity_anchor_training_support_precision": sum(anchor_precisions) / len(anchor_precisions),
        "same_gaussian_posterior_for_2d_and_3d": True,
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            "scene_state": {"path": str(state_path), "sha256": sha256_file(state_path)},
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
            "siglip_teacher": {"path": str(siglip_path), "sha256": sha256_file(siglip_path)},
        },
        "identity_adapter": (
            {"path": str(Path(args.identity_adapter).resolve(strict=True)),
             "sha256": sha256_file(Path(args.identity_adapter).resolve(strict=True))}
            if args.identity_adapter else None
        ),
        "status": (
            "source_dev_identity_adapter_evaluation"
            if args.evaluation_residue == 3
            else "source_audit_identity_adapter_evaluation"
        ),
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()
