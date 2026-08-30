"""Pool disentangled clean-query evidence on source proposals."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.query.calibrated_posterior import NullCalibratedPosterior
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _pool_membership(
    value: torch.Tensor,
    *,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    num_proposals: int,
) -> torch.Tensor:
    """Weighted proposal mean without a Python loop over supports."""

    source = torch.as_tensor(value).float()
    rows = torch.as_tensor(row_indices, device=source.device).long()
    proposals = torch.as_tensor(proposal_indices, device=source.device).long()
    mass = torch.as_tensor(weights, device=source.device).float()
    numerator = source.new_zeros((num_proposals, source.shape[1]))
    numerator.index_add_(0, proposals, source[rows] * mass[:, None])
    denominator = source.new_zeros(num_proposals)
    denominator.index_add_(0, proposals, mass)
    return numerator / denominator.clamp_min(1e-8)[:, None]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--text-negatives", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--max-train-hits-per-pair", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("scene_state", args.scene_state),
            ("membership", args.membership),
            ("authority", args.authority),
            ("text_embeddings", args.text_embeddings),
            ("text_negatives", args.text_negatives),
        )
    }
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    text_payload = torch.load(paths["text_embeddings"], map_location="cpu")
    scene_state = torch.load(paths["scene_state"], map_location="cpu")
    signed_gate = scene_state.get("metadata", {}).get("signed_boundary_gate3", {})
    if signed_gate.get("status") != "passed":
        raise ValueError("posterior evidence requires a passed signed D16 parent")
    if authority.get("schema") != "radio_gs.sugm_v3.native_language_authority.v3":
        raise ValueError("posterior evidence requires native language authority v3")
    if membership["scene"] != authority["scene"]:
        raise ValueError("posterior evidence scene differs")
    count = int(membership["num_proposals"])
    states = torch.as_tensor(authority["query_state"]).to(torch.int8)
    if states.shape != (count, len(authority["query_names"])):
        raise ValueError("posterior evidence authority axes differ")
    interface = load_query_interface(
        paths["scene_state"],
        device=args.device,
        text_negative_path=paths["text_negatives"],
        text_logit_scale=10.0,
    )
    lookup = {
        str(value).casefold(): index
        for index, value in enumerate(text_payload["queries"])
    }
    if any(str(name).casefold() not in lookup for name in authority["query_names"]):
        raise ValueError("posterior evidence text cache misses authority query")
    reliability = interface.reliability
    calibrator = NullCalibratedPosterior().to(args.device).eval()
    if args.max_train_hits_per_pair <= 0:
        raise ValueError("posterior evidence train support budget differs")
    proposal_indices_cpu = torch.as_tensor(membership["proposal_indices"]).long()
    row_indices_cpu = torch.as_tensor(membership["row_indices"]).long()
    weights_cpu = torch.as_tensor(membership["weights"]).float()
    proposal_hits = [
        torch.where(proposal_indices_cpu == index)[0] for index in range(count)
    ]
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    train = (views % 4 == 1) | (views % 4 == 2)
    pooled_positive, pooled_negative = [], []
    train_hit_positive, train_hit_negative = [], []
    train_hit_group, train_hit_weight = [], []
    train_pair_state, train_pair_query_group = [], []
    pair_offset = 0
    query_group = 0
    for query_index, name in enumerate(authority["query_names"]):
        token = torch.as_tensor(text_payload["embeddings"])[
            lookup[str(name).casefold()]
        ].float()
        packet = QueryPacket("text", token)
        identity, null, unknown = interface.semantic_text_evidence(packet)
        base_instance, returned_identity = interface.posterior_from_packet(
            packet,
            scale=args.scale,
            topk=args.topk,
            temperature=args.temperature,
            posterior_chunk_size=65536,
            text_anchor_policy="positive",
        )
        if not torch.equal(identity, returned_identity):
            raise RuntimeError("positive identity evidence changed during expansion")
        instance, boundary = interface.refine_instance_with_boundary(
            base_instance,
            maximum_logit_residual=interface.maximum_boundary_logit_residual,
        )
        positive, negative = calibrator.evidence_features(
            identity=identity,
            instance=instance,
            null=null,
            negative=torch.sigmoid((null - identity) * 10.0),
            unknown=unknown,
            boundary=boundary,
            reliability=reliability,
        )
        common = {
            "row_indices": membership["row_indices"],
            "proposal_indices": membership["proposal_indices"],
            "weights": membership["weights"],
            "num_proposals": count,
        }
        pooled_positive.append(_pool_membership(positive, **common).cpu())
        pooled_negative.append(_pool_membership(negative, **common).cpu())
        explicit = train & (states[:, query_index] != -1)
        local_state = states[explicit, query_index]
        if not bool((local_state == 1).any()) or not bool((local_state == 0).any()):
            continue
        sampled_rows, sampled_weights, sampled_groups = [], [], []
        selected_proposals = torch.where(explicit)[0]
        for local_group, proposal in enumerate(selected_proposals.tolist()):
            hits = proposal_hits[proposal]
            mass = weights_cpu[hits]
            if hits.numel() > args.max_train_hits_per_pair:
                cumulative = mass.cumsum(0)
                targets = (
                    torch.arange(args.max_train_hits_per_pair).float() + 0.5
                ) * (cumulative[-1] / args.max_train_hits_per_pair)
                chosen = torch.searchsorted(cumulative, targets).clamp_max(hits.numel() - 1)
                hits = hits[chosen]
                mass = torch.full(
                    (args.max_train_hits_per_pair,),
                    float(cumulative[-1]) / args.max_train_hits_per_pair,
                )
            sampled_rows.append(row_indices_cpu[hits])
            sampled_weights.append(mass)
            sampled_groups.append(
                torch.full((hits.numel(),), pair_offset + local_group, dtype=torch.long)
            )
        rows = torch.cat(sampled_rows).to(positive.device)
        train_hit_positive.append(positive[rows].cpu())
        train_hit_negative.append(negative[rows].cpu())
        train_hit_group.append(torch.cat(sampled_groups))
        train_hit_weight.append(torch.cat(sampled_weights))
        train_pair_state.append(local_state)
        train_pair_query_group.append(
            torch.full((selected_proposals.numel(),), query_group, dtype=torch.long)
        )
        pair_offset += int(selected_proposals.numel())
        query_group += 1
    positive_features = torch.stack(pooled_positive, dim=1)
    negative_features = torch.stack(pooled_negative, dim=1)
    if positive_features.shape != (count, len(authority["query_names"]), 7):
        raise RuntimeError("pooled positive evidence axes differ")
    payload = {
        "schema": "radio_gs.sugm_v3.clean_posterior_evidence.v2",
        "scene": membership["scene"],
        "query_names": list(authority["query_names"]),
        "proposal_view_indices": torch.as_tensor(
            authority["proposal_view_indices"]
        ).long(),
        "query_state": states,
        "positive_features": positive_features,
        "negative_features": negative_features,
        "train_hit_positive_features": torch.cat(train_hit_positive),
        "train_hit_negative_features": torch.cat(train_hit_negative),
        "train_hit_pair_group": torch.cat(train_hit_group),
        "train_hit_weight": torch.cat(train_hit_weight),
        "train_pair_state": torch.cat(train_pair_state),
        "train_pair_query_group": torch.cat(train_pair_query_group),
        "feature_names": {
            "positive": [
                "raw_positive_identity",
                "raw_positive_anchor_d48_instance",
                "signed_D16_magnitude_identity_minus_instance_contrast",
                "r5_visual_write_authority",
                "r5_coverage_confidence",
                "r5_structural_confidence",
                "r5_membership_strength",
            ],
            "negative": [
                "canonical_null_similarity",
                "canonical_null_over_positive_probability",
                "semantic_or_r5_unknown",
            ],
        },
        "metadata": {
            "source_only": True,
            "source_train_residues": [1, 2],
            "source_dev_residue": 3,
            "unknown_pairs_used_as_negative": False,
            "historical_language_authority_opened": False,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "anchor_policy": "raw_positive_text_similarity",
            "null_policy": "independent_final_posterior_evidence",
            "pooling": "membership_weighted_proposal_mean_of_gaussian_logit_features",
            "calibrator_train_order": "gaussian_logit_then_sigmoid_then_membership_weighted_proposal_mean",
            "max_train_hits_per_pair": args.max_train_hits_per_pair,
            "support_sampling": "deterministic_membership_mass_quantiles",
            "method": {
                "topk": args.topk,
                "scale": args.scale,
                "temperature": args.temperature,
            },
            "inputs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print(
        {
            "output": str(output),
            "sha256": sha256_file(output),
            "scene": membership["scene"],
            "shape": list(positive_features.shape),
            "positive_pairs": int((states == 1).sum()),
            "negative_pairs": int((states == 0).sum()),
            "unknown_pairs": int((states == -1).sum()),
            "sampled_train_hits": int(sum(value.shape[0] for value in train_hit_positive)),
            "explicit_train_pairs": pair_offset,
            "complete_train_query_groups": query_group,
        }
    )


if __name__ == "__main__":
    main()


__all__ = ["_pool_membership"]
