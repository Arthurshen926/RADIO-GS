"""Fit the constant-size null-aware posterior on clean source-train pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.query.calibrated_posterior import NullCalibratedPosterior
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _training_examples(
    paths: list[Path], device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict],
]:
    positive_rows, negative_rows = [], []
    hit_groups, hit_weights, labels, query_groups = [], [], [], []
    receipts = []
    pair_offset = 0
    query_offset = 0
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata", {})
        if (
            payload.get("schema") != "radio_gs.sugm_v3.clean_posterior_evidence.v2"
            or not metadata.get("source_only")
            or metadata.get("target_rgb_opened")
            or metadata.get("benchmark_metrics_opened")
            or metadata.get("unknown_pairs_used_as_negative")
        ):
            raise ValueError("posterior calibrator evidence lineage differs")
        local_pair_state = torch.as_tensor(payload["train_pair_state"]).float()
        local_query_group = torch.as_tensor(payload["train_pair_query_group"]).long()
        local_hit_group = torch.as_tensor(payload["train_hit_pair_group"]).long()
        positive_rows.append(torch.as_tensor(payload["train_hit_positive_features"]))
        negative_rows.append(torch.as_tensor(payload["train_hit_negative_features"]))
        hit_groups.append(local_hit_group + pair_offset)
        hit_weights.append(torch.as_tensor(payload["train_hit_weight"]).float())
        labels.append(local_pair_state)
        query_groups.append(local_query_group + query_offset)
        scene_pairs = int(local_pair_state.numel())
        scene_groups = int(torch.unique(local_query_group).numel())
        pair_offset += scene_pairs
        query_offset += scene_groups
        receipts.append(
            {
                "scene": payload["scene"],
                "train_groups": scene_groups,
                "explicit_train_pairs": scene_pairs,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    if not positive_rows:
        raise ValueError("posterior calibrator has no complete source-train group")
    return (
        torch.cat(positive_rows).to(device),
        torch.cat(negative_rows).to(device),
        torch.cat(hit_groups).to(device),
        torch.cat(hit_weights).to(device),
        torch.cat(labels).to(device),
        torch.cat(query_groups).to(device),
        receipts,
    )


def _pool_hit_probability(
    probability: torch.Tensor,
    hit_group: torch.Tensor,
    hit_weight: torch.Tensor,
    num_pairs: int,
) -> torch.Tensor:
    numerator = probability.new_zeros(num_pairs)
    denominator = probability.new_zeros(num_pairs)
    numerator.index_add_(0, hit_group, probability * hit_weight)
    denominator.index_add_(0, hit_group, hit_weight)
    return numerator / denominator.clamp_min(1e-8)


def _balanced_group_loss(
    logit: torch.Tensor, target: torch.Tensor, group: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    response_terms, listwise_terms = [], []
    for index in torch.unique(group).tolist():
        selected = group == index
        positive = logit[selected & (target == 1)]
        negative = logit[selected & (target == 0)]
        response_terms.append(F.softplus(-positive).mean() + F.softplus(negative).mean())
        listwise_terms.append(F.softplus(negative.max() - positive.max() + 0.25))
    return torch.stack(response_terms).mean(), torch.stack(listwise_terms).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--disable-private-structure", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.evidence) < 2 or args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("posterior calibrator training budget differs")
    paths = [Path(value).resolve(strict=True) for value in args.evidence]
    device = torch.device(args.device)
    positive, negative, hit_group, hit_weight, target, group, receipts = _training_examples(
        paths, device
    )
    torch.manual_seed(args.seed)
    model = NullCalibratedPosterior().to(device)
    if args.disable_private_structure:
        model.positive_feature_mask[1:3] = 0
    with torch.no_grad():
        model.raw_positive_weight.add_(0.05 * torch.randn_like(model.raw_positive_weight))
        model.raw_negative_weight.add_(0.05 * torch.randn_like(model.raw_negative_weight))
        model.bias.add_(0.05 * torch.randn_like(model.bias))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        hit_probability = torch.sigmoid(model.logit_from_features(positive, negative))
        prediction = _pool_hit_probability(
            hit_probability, hit_group, hit_weight, int(target.numel())
        )
        logit = torch.logit(prediction.clamp(1e-6, 1 - 1e-6))
        response, listwise = _balanced_group_loss(logit, target, group)
        loss = response + listwise
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == args.epochs:
            prediction = prediction.detach()
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "balanced_response": float(response.detach()),
                    "listwise": float(listwise.detach()),
                    "positive_mean": float(prediction[target == 1].mean()),
                    "negative_mean": float(prediction[target == 0].mean()),
                }
            )
    payload = {
        "schema": "radio_gs.sugm_v3.null_calibrated_posterior.v1",
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "history": history,
        "training_receipts": receipts,
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
            "source_dev_opened_for_fit": False,
            "unknown_pairs_used_as_negative": False,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "objective": "query_balanced_explicit_ternary_response_plus_listwise_margin",
            "training_order": "gaussian_logit_then_sigmoid_then_membership_weighted_proposal_mean",
            "parameter_count": sum(value.numel() for value in model.parameters()),
            "disabled_positive_feature_indices": (
                [1, 2] if args.disable_private_structure else []
            ),
            "private_structure_policy": (
                "semantic_local_control_without_D48_or_D16"
                if args.disable_private_structure
                else "retained_D48_plus_D16"
            ),
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print(
        {
            "output": str(output),
            "sha256": sha256_file(output),
            "explicit_pairs": int(target.numel()),
            "groups": int(torch.unique(group).numel()),
            "history": history[-1],
        }
    )


if __name__ == "__main__":
    main()


__all__ = ["_balanced_group_loss", "_training_examples"]
