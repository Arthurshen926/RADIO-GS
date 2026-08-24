#!/usr/bin/env python3
"""Source-heldout Query-Native image-to-Gaussian posterior sentinel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.interfaces.query_packet import QueryPacket
from radio_gs.models.query_native_gaussian_memory import (
    ModalityQueryAdapter,
    QueryNativeGaussianPosteriorDecoder,
)
from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import (
    _calibrate_threshold,
    _iou_from_scores,
    _load_mapping,
    _proposal_soft_support,
    _proposal_support,
    _sample_without_replacement,
    _similarity_scores_for_proposal,
    compose_membership_query_features,
    visible_membership_target,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber


def _proposal_pairs(
    proposal: int,
    supports: list[torch.Tensor],
    probabilities: list[torch.Tensor],
    proposal_views: torch.Tensor,
    observed: torch.Tensor,
    positive_cap: int,
    negative_cap: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = supports[proposal]
    target = probabilities[proposal]
    if positive.numel() > positive_cap:
        order = torch.randperm(positive.numel(), generator=generator)[:positive_cap]
        positive, target = positive[order], target[order]
    visible = torch.where(observed[int(proposal_views[proposal])])[0]
    lookup = torch.zeros(observed.shape[1], dtype=torch.bool)
    lookup[positive] = True
    negative = _sample_without_replacement(visible[~lookup[visible]], negative_cap, generator)
    return torch.cat((positive, negative)), torch.cat((target, torch.zeros(negative.numel())))


@torch.inference_mode()
def _posterior(
    adapter: ModalityQueryAdapter,
    decoder: QueryNativeGaussianPosteriorDecoder,
    latent: torch.Tensor,
    reliability: torch.Tensor,
    query: torch.Tensor,
    identity_prior: torch.Tensor,
    rows: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    token = adapter(query[None].to(device))
    packet = QueryPacket(token, "image")
    logits, _identity = decoder(
        latent[rows].to(device), reliability[rows].to(device), packet,
        identity_prior=identity_prior.to(device),
    )
    return torch.sigmoid(logits).cpu()


def run(args: argparse.Namespace) -> dict[str, Any]:
    membership, membership_record = _load_mapping(
        args.membership, args.expected_membership_sha256, "source SAM membership"
    )
    teacher, teacher_record = _load_mapping(
        args.language_teacher, args.expected_language_teacher_sha256,
        "source mask language teacher",
    )
    query_cache, query_record = _load_mapping(
        args.query_cache, args.expected_query_cache_sha256, "primitive query cache"
    )
    universal, universal_record = _load_mapping(
        args.universal_field, args.expected_universal_field_sha256,
        "Universal Field reliability authority",
    )
    appearance = appearance_record = None
    if args.appearance_teacher:
        appearance, appearance_record = _load_mapping(
            args.appearance_teacher, args.expected_appearance_teacher_sha256,
            "source mask appearance teacher",
        )
    field_path = Path(args.field).expanduser().resolve(strict=True)
    field, _payload, _signature = load_factorized_canonical_field_checkpoint(
        field_path, map_location="cpu", expected_sha256=args.expected_field_sha256
    )
    if universal.get("universal_field_migration", {}).get("source_field_sha256") != args.expected_field_sha256:
        raise ValueError("query-native membership reliability binding differs")
    if (
        membership.get("metadata", {}).get("query_independent_proposal_set") is not True
        or teacher.get("metadata", {}).get("source_only") is not True
        or teacher.get("metadata", {}).get("benchmark_masks_opened") is not False
    ):
        raise ValueError("query-native membership source authority differs")

    rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float()
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    observed = torch.as_tensor(membership["view_observed"]).bool()
    proposal_count = int(membership["num_proposals"])
    hard = weights >= float(args.evaluation_membership_threshold)
    hard_support = _proposal_support(rows[hard], proposals[hard], proposal_count)
    soft_support, soft_values = _proposal_soft_support(rows, proposals, weights, proposal_count)
    valid = torch.tensor([value.numel() > 0 for value in hard_support], dtype=torch.bool)
    heldout = (
        torch.remainder(proposal_views, int(args.holdout_stride)) == int(args.holdout_residue)
    ) & valid
    training = (~heldout) & valid
    if int(training.sum()) < 2 or not bool(heldout.any()):
        raise ValueError("query-native membership source split differs")

    descriptors = F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
    contexts = F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1)
    semantic = F.normalize(0.75 * descriptors + 0.25 * contexts, dim=-1)
    appearance_descriptor = None
    if appearance is not None:
        appearance_descriptor = torch.as_tensor(appearance["descriptors"]).float()
        if appearance_descriptor.shape[0] != proposal_count:
            raise ValueError("query-native appearance proposal domain differs")
    query = compose_membership_query_features(semantic, appearance_descriptor)
    baseline = F.normalize(torch.as_tensor(
        query_cache.get("features", query_cache.get("summary_features"))
    ).float(), dim=-1)
    latent = field.local_codes.detach().cpu().float().contiguous()
    reliability = torch.as_tensor(universal.get("reliability")).float().contiguous()
    if (
        latent.shape != (field.num_gaussians, 512)
        or reliability.shape != (field.num_gaussians, 5)
        or observed.shape[1] != field.num_gaussians
    ):
        raise ValueError("query-native membership Gaussian domain differs")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    adapter = ModalityQueryAdapter(query.shape[1], int(args.query_dim)).to(device)
    decoder = QueryNativeGaussianPosteriorDecoder(
        query_dim=int(args.query_dim), hidden_dim=int(args.hidden_dim)
    ).to(device)
    parameters = list(adapter.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    training_indices = torch.where(training)[0]
    best_loss = float("inf")
    best_adapter = best_decoder = None
    for step in range(int(args.steps)):
        proposal = int(training_indices[step % training_indices.numel()])
        local_rows, target = _proposal_pairs(
            proposal, soft_support, soft_values, proposal_views, observed,
            int(args.positive_cap), int(args.negative_cap), generator,
        )
        token = adapter(query[proposal:proposal + 1].to(device))
        identity_prior = baseline[local_rows] @ semantic[proposal]
        logits, identity = decoder(
            latent[local_rows].to(device), reliability[local_rows].to(device),
            QueryPacket(token, "image"), identity_prior=identity_prior.to(device),
        )
        target_device = target.to(device)
        probability = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target_device)
        intersection = (probability * target_device).sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (
            probability.sum() + target_device.sum() + 1.0
        )
        brier = F.mse_loss(probability, target_device)
        positive = target_device >= float(args.evaluation_membership_threshold)
        negative = target_device == 0
        ranking = torch.zeros((), device=device)
        if bool(positive.any()) and bool(negative.any()):
            ranking = F.relu(
                float(args.identity_margin) - identity[positive].mean() + identity[negative].mean()
            )
        loss = bce + float(args.dice_weight) * dice + float(args.brier_weight) * brier + float(args.rank_weight) * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_adapter = {key: tensor.detach().cpu().clone() for key, tensor in adapter.state_dict().items()}
            best_decoder = {key: tensor.detach().cpu().clone() for key, tensor in decoder.state_dict().items()}
    if best_adapter is None or best_decoder is None:
        raise RuntimeError("query-native membership optimization did not complete")
    adapter.load_state_dict(best_adapter)
    decoder.load_state_dict(best_decoder)
    adapter.eval(); decoder.eval()

    calibration_indices = torch.where(training)[0]
    if calibration_indices.numel() > int(args.calibration_proposals):
        positions = torch.linspace(0, calibration_indices.numel() - 1, int(args.calibration_proposals)).round().long()
        calibration_indices = calibration_indices[positions]
    query_native_calibration: list[tuple[torch.Tensor, torch.Tensor]] = []
    baseline_calibration: list[tuple[torch.Tensor, torch.Tensor]] = []
    for proposal in calibration_indices.tolist():
        visible = torch.where(observed[int(proposal_views[proposal])])[0]
        truth = visible_membership_target(visible, hard_support[proposal], num_rows=field.num_gaussians)
        query_native_calibration.append((
            _posterior(
                adapter, decoder, latent, reliability, query[proposal],
                baseline[visible] @ semantic[proposal], visible, device,
            ), truth
        ))
        baseline_calibration.append((
            _similarity_scores_for_proposal(
                baseline, semantic[proposal], visible, device=device,
                chunk_size=int(args.eval_chunk_size), features_are_normalized=True,
            ), truth,
        ))
    threshold, train_iou = _calibrate_threshold(query_native_calibration, int(args.threshold_candidates))
    baseline_threshold, baseline_train_iou = _calibrate_threshold(baseline_calibration, int(args.threshold_candidates))
    query_native_ious: list[float] = []
    baseline_ious: list[float] = []
    for proposal in torch.where(heldout)[0].tolist():
        visible = torch.where(observed[int(proposal_views[proposal])])[0]
        truth = visible_membership_target(visible, hard_support[proposal], num_rows=field.num_gaussians)
        score = _posterior(
            adapter, decoder, latent, reliability, query[proposal],
            baseline[visible] @ semantic[proposal], visible, device,
        )
        baseline_score = _similarity_scores_for_proposal(
            baseline, semantic[proposal], visible, device=device,
            chunk_size=int(args.eval_chunk_size), features_are_normalized=True,
        )
        query_native_ious.append(_iou_from_scores(score, truth, threshold))
        baseline_ious.append(_iou_from_scores(baseline_score, truth, baseline_threshold))
    posterior_iou = float(torch.tensor(query_native_ious).mean())
    baseline_iou = float(torch.tensor(baseline_ious).mean())
    passed = posterior_iou >= float(args.minimum_heldout_iou) and posterior_iou >= baseline_iou + float(args.minimum_gain)

    output = Path(args.output).expanduser().resolve()
    inputs = {
        "field": file_record(field_path), "universal_field": universal_record,
        "membership": membership_record, "language_teacher": teacher_record,
        "query_cache": query_record,
    }
    if appearance_record is not None:
        inputs["appearance_teacher"] = appearance_record
    write_torch_noclobber(output, {
        "schema": "radio_gs.query_native_membership_decoder.v1", "schema_version": 1,
        "scene": str(args.scene), "adapter_state_dict": best_adapter,
        "decoder_state_dict": best_decoder, "threshold": threshold,
        "metadata": {
            "field_frozen": True, "per_gaussian_parameters_added": False,
            "teacher_features_decoded": False, "direct_gaussian_posterior": True,
            "query_modality": "image", "source_only": True,
            "benchmark_images_opened": False, "benchmark_masks_opened": False,
            "inputs": inputs,
        },
    })
    report = {
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene), "training_proposals": int(training.sum()),
        "heldout_proposals": int(heldout.sum()), "best_train_loss": best_loss,
        "query_dim": int(args.query_dim), "appearance_teacher": appearance is not None,
        "source_train_calibration": {
            "query_native_threshold": threshold, "query_native_iou": train_iou,
            "primitive_threshold": baseline_threshold, "primitive_iou": baseline_train_iou,
        },
        "source_heldout_macro_iou": {
            "primitive_similarity": baseline_iou, "query_native_posterior": posterior_iou,
            "delta": posterior_iou - baseline_iou,
        },
        "gate": {"minimum_heldout_iou": float(args.minimum_heldout_iou),
                 "minimum_gain": float(args.minimum_gain), "passed": passed},
        "output": file_record(output),
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--expected-field-sha256", required=True)
    parser.add_argument("--universal-field", required=True)
    parser.add_argument("--expected-universal-field-sha256", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expected-membership-sha256", required=True)
    parser.add_argument("--language-teacher", required=True)
    parser.add_argument("--expected-language-teacher-sha256", required=True)
    parser.add_argument("--appearance-teacher", default="")
    parser.add_argument("--expected-appearance-teacher-sha256", default="")
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-cap", type=int, default=1024)
    parser.add_argument("--negative-cap", type=int, default=2048)
    parser.add_argument("--calibration-proposals", type=int, default=32)
    parser.add_argument("--threshold-candidates", type=int, default=64)
    parser.add_argument("--eval-chunk-size", type=int, default=32768)
    parser.add_argument("--holdout-stride", type=int, default=4)
    parser.add_argument("--holdout-residue", type=int, default=3)
    parser.add_argument("--evaluation-membership-threshold", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--brier-weight", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--identity-margin", type=float, default=0.05)
    parser.add_argument("--minimum-heldout-iou", type=float, default=0.16)
    parser.add_argument("--minimum-gain", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260824)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
