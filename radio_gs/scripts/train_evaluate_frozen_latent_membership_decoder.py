#!/usr/bin/env python3
"""Source-holdout gate for direct query-to-Gaussian membership.

Official source SAM proposals provide positive support only where their exact
renderer marginal reaches a Gaussian.  Other Gaussians visible in that source
view are negatives; invisible rows are unknown and never enter the loss.  A
single decoder consumes a source-mask SigLIP2 descriptor and frozen L512 code.
It adds no per-Gaussian parameters and opens no benchmark query, image, mask,
or metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.models.frozen_latent_membership_decoder import (
    FrozenLatentMembershipDecoder,
)
from radio_gs.querying.source_multiview_object_tracks import (
    build_source_multiview_object_tracks,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


def _load_mapping(path: str, digest: str, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    value, actual, source = load_sha_bound_project_checkpoint_mapping(
        path,
        expected_sha256=digest,
        map_location="cpu",
        label=label,
    )
    return dict(value), {"path": str(source), "sha256": actual}


def _proposal_support(
    rows: torch.Tensor, proposals: torch.Tensor, count: int
) -> list[torch.Tensor]:
    buckets: list[list[int]] = [[] for _ in range(int(count))]
    for row, proposal in zip(rows.tolist(), proposals.tolist()):
        buckets[int(proposal)].append(int(row))
    return [torch.tensor(sorted(set(values)), dtype=torch.long) for values in buckets]


def _proposal_soft_support(
    rows: torch.Tensor,
    proposals: torch.Tensor,
    probabilities: torch.Tensor,
    count: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    buckets: list[dict[int, float]] = [{} for _ in range(int(count))]
    for row, proposal, probability in zip(
        rows.tolist(), proposals.tolist(), probabilities.tolist()
    ):
        key = int(row)
        value = float(probability)
        buckets[int(proposal)][key] = max(value, buckets[int(proposal)].get(key, 0.0))
    support_rows: list[torch.Tensor] = []
    support_values: list[torch.Tensor] = []
    for bucket in buckets:
        ordered = sorted(bucket)
        support_rows.append(torch.tensor(ordered, dtype=torch.long))
        support_values.append(
            torch.tensor([bucket[row] for row in ordered], dtype=torch.float32)
        )
    return support_rows, support_values


def _sample_without_replacement(
    values: torch.Tensor, maximum: int, generator: torch.Generator
) -> torch.Tensor:
    if values.numel() <= int(maximum):
        return values
    order = torch.randperm(values.numel(), generator=generator)[: int(maximum)]
    return values[order]


def build_training_pairs(
    *,
    supports: list[torch.Tensor],
    proposal_views: torch.Tensor,
    view_observed: torch.Tensor,
    selected: torch.Tensor,
    positive_cap: int,
    negative_cap: int,
    seed: int,
    support_probabilities: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return proposal,row,label,weight with unknown visibility excluded."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    pair_proposal: list[torch.Tensor] = []
    pair_row: list[torch.Tensor] = []
    pair_label: list[torch.Tensor] = []
    pair_weight: list[torch.Tensor] = []
    for proposal in torch.where(selected)[0].tolist():
        positive = supports[proposal]
        positive_probability = (
            torch.ones(positive.numel())
            if support_probabilities is None
            else support_probabilities[proposal]
        )
        if positive_probability.shape != (positive.numel(),):
            raise ValueError("soft membership and support rows differ")
        visible = torch.where(view_observed[int(proposal_views[proposal])])[0]
        if not positive.numel() or not visible.numel():
            continue
        if positive.numel() > int(positive_cap):
            order = torch.randperm(positive.numel(), generator=generator)[: int(positive_cap)]
            positive = positive[order]
            positive_probability = positive_probability[order]
        positive_mask = torch.zeros(view_observed.shape[1], dtype=torch.bool)
        positive_mask[supports[proposal]] = True
        negative = visible[~positive_mask[visible]]
        if not negative.numel():
            continue
        negative = _sample_without_replacement(negative, negative_cap, generator)
        rows = torch.cat((positive, negative))
        labels = torch.cat(
            (positive_probability.float(), torch.zeros(negative.numel()))
        )
        # Every proposal and sign has equal authority.  The score is therefore
        # an equal-cost membership likelihood ratio, not a foreground-frequency
        # prior learned from proposal area.
        weights = torch.cat(
            (
                torch.full((positive.numel(),), 0.5 / positive.numel()),
                torch.full((negative.numel(),), 0.5 / negative.numel()),
            )
        )
        pair_proposal.append(torch.full((rows.numel(),), proposal, dtype=torch.long))
        pair_row.append(rows)
        pair_label.append(labels)
        pair_weight.append(weights)
    if not pair_row:
        raise ValueError("source split has no visible membership training pair")
    return tuple(torch.cat(values) for values in (pair_proposal, pair_row, pair_label, pair_weight))


def track_augmented_training_targets(
    *,
    rows: torch.Tensor,
    proposals: torch.Tensor,
    weights: torch.Tensor,
    supports: list[torch.Tensor],
    proposal_views: torch.Tensor,
    view_observed: torch.Tensor,
    training: torch.Tensor,
    minimum_soft_cosine: float,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Compile training-view tracks into full-object positive/unknown targets.

    Tracks are built from training views only.  Every tracked proposal is then
    supervised by the union of its independently observed track members, and
    negatives are drawn only from rows visible in at least one track view.
    Held-out proposal masks never enter either the track graph or its targets.
    """

    proposal_count = len(supports)
    row_count = int(view_observed.shape[1])
    selected_entries = torch.as_tensor(training).bool()[proposals]
    tracks = build_source_multiview_object_tracks(
        rows[selected_entries],
        proposals[selected_entries],
        weights[selected_entries],
        proposal_views,
        num_rows=row_count,
        num_proposals=proposal_count,
        minimum_soft_cosine=float(minimum_soft_cosine),
    )
    augmented = [value.clone() for value in supports]
    proposal_observed = view_observed[proposal_views].clone()
    if tracks.num_tracks == 0:
        return augmented, proposal_observed, dict(tracks.stats)
    track_supports = _proposal_support(
        tracks.row_indices, tracks.track_indices, tracks.num_tracks
    )
    for track in range(tracks.num_tracks):
        members = torch.where(tracks.proposal_track_indices == track)[0]
        if not members.numel() or bool((~training[members]).any()):
            raise RuntimeError("training-only track contains a held-out proposal")
        observed = view_observed[torch.unique(proposal_views[members])].any(dim=0)
        for proposal in members.tolist():
            augmented[int(proposal)] = track_supports[track]
            proposal_observed[int(proposal)] = observed
    return augmented, proposal_observed, dict(tracks.stats)


@torch.inference_mode()
def _scores_for_proposal(
    model: FrozenLatentMembershipDecoder,
    latent: torch.Tensor,
    query: torch.Tensor,
    rows: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    output = []
    identity = query[None].to(device)
    for chunk in rows.split(int(chunk_size)):
        local = latent[chunk].to(device)
        output.append(torch.sigmoid(model(local, identity.expand(local.shape[0], -1))).cpu())
    return torch.cat(output)


def _iou_from_scores(score: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    prediction = score >= float(threshold)
    truth = torch.as_tensor(target).bool()
    intersection = int((prediction & truth).sum())
    union = int((prediction | truth).sum())
    return float(intersection / union) if union else 1.0


def visible_membership_target(
    visible: torch.Tensor, support: torch.Tensor, *, num_rows: int
) -> torch.Tensor:
    """Mark visible proposal members with a dense linear-time lookup.

    The Gaussian carrier has a fixed dense row domain.  Building one boolean
    lookup is exactly equivalent to ``torch.isin(visible, support)`` but avoids
    the prohibitively expensive large-set membership kernel on LERF scenes.
    """

    visible_rows = torch.as_tensor(visible).long().reshape(-1)
    support_rows = torch.as_tensor(support).long().reshape(-1)
    if int(num_rows) <= 0:
        raise ValueError("num_rows must be positive")
    if visible_rows.numel() and (
        int(visible_rows.min()) < 0 or int(visible_rows.max()) >= int(num_rows)
    ):
        raise ValueError("visible Gaussian row falls outside the carrier")
    if support_rows.numel() and (
        int(support_rows.min()) < 0 or int(support_rows.max()) >= int(num_rows)
    ):
        raise ValueError("support Gaussian row falls outside the carrier")
    lookup = torch.zeros(int(num_rows), dtype=torch.bool)
    lookup[support_rows] = True
    return lookup[visible_rows]


def _calibrate_threshold(
    score_and_target: list[tuple[torch.Tensor, torch.Tensor]], candidates: int
) -> tuple[float, float]:
    values = torch.cat([score for score, _target in score_and_target])
    quantiles = torch.linspace(0.01, 0.99, int(candidates))
    thresholds = torch.unique(torch.quantile(values.float(), quantiles)).tolist()
    best_threshold, best_iou = 0.5, -1.0
    for threshold in thresholds:
        mean_iou = sum(
            _iou_from_scores(score, target, threshold)
            for score, target in score_and_target
        ) / len(score_and_target)
        if mean_iou > best_iou:
            best_threshold, best_iou = float(threshold), float(mean_iou)
    return best_threshold, best_iou


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    membership, membership_record = _load_mapping(
        args.membership, args.expected_membership_sha256, "source SAM membership"
    )
    teacher, teacher_record = _load_mapping(
        args.language_teacher,
        args.expected_language_teacher_sha256,
        "source mask language teacher",
    )
    query_cache, query_record = _load_mapping(
        args.query_cache, args.expected_query_cache_sha256, "primitive query cache"
    )
    field_path = Path(args.field).expanduser().resolve(strict=True)
    field, _payload, _signature = load_factorized_canonical_field_checkpoint(
        field_path,
        map_location="cpu",
        expected_sha256=args.expected_field_sha256,
    )
    if membership.get("metadata", {}).get("query_independent_proposal_set") is not True:
        raise ValueError("proposal membership is not query-independent")
    if teacher.get("metadata", {}).get("source_only") is not True:
        raise ValueError("mask language teacher is not source-only")
    if teacher.get("metadata", {}).get("benchmark_masks_opened") is not False:
        raise ValueError("mask language teacher opened benchmark masks")

    rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    membership_weights = torch.as_tensor(membership["weights"]).float()
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    view_observed = torch.as_tensor(membership["view_observed"]).bool()
    proposal_count = int(membership["num_proposals"])
    if view_observed.shape[1] != field.num_gaussians:
        raise ValueError("visibility and frozen field row domains differ")
    evaluation_selected = membership_weights >= float(args.evaluation_membership_threshold)
    support = _proposal_support(
        rows[evaluation_selected], proposals[evaluation_selected], proposal_count
    )
    valid = torch.tensor([value.numel() > 0 for value in support], dtype=torch.bool)
    heldout = (
        torch.remainder(proposal_views, int(args.holdout_stride))
        == int(args.holdout_residue)
    ) & valid
    training = (~heldout) & valid
    if int(training.sum()) < 2 or not bool(heldout.any()):
        raise ValueError("fixed source split lacks train or heldout proposals")

    descriptors = F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
    contexts = F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1)
    language = F.normalize(0.75 * descriptors + 0.25 * contexts, dim=-1)
    baseline_feature = F.normalize(
        torch.as_tensor(query_cache.get("features", query_cache.get("summary_features"))).float(),
        dim=-1,
    )
    if language.shape != (proposal_count, 1536):
        raise ValueError("language teacher proposal domain differs")
    if baseline_feature.shape != (field.num_gaussians, 1536):
        raise ValueError("primitive query feature domain differs")
    latent = field.local_codes.detach().cpu().float()

    training_support = support
    training_support_probabilities: list[torch.Tensor] | None = None
    training_views = proposal_views
    training_observed = view_observed
    track_stats: dict[str, Any] | None = None
    if bool(args.soft_membership_targets):
        if bool(args.track_augmented_training):
            raise ValueError("soft membership and track augmentation are separate sentinels")
        training_support, training_support_probabilities = _proposal_soft_support(
            rows, proposals, membership_weights, proposal_count
        )
    if bool(args.track_augmented_training):
        training_support, training_observed, track_stats = track_augmented_training_targets(
            rows=rows,
            proposals=proposals,
            weights=membership_weights,
            supports=support,
            proposal_views=proposal_views,
            view_observed=view_observed,
            training=training,
            minimum_soft_cosine=float(args.track_minimum_soft_cosine),
        )
        training_views = torch.arange(proposal_count)
    pair_proposal, pair_row, pair_label, pair_weight = build_training_pairs(
        supports=training_support,
        proposal_views=training_views,
        view_observed=training_observed,
        selected=training,
        positive_cap=int(args.positive_cap),
        negative_cap=int(args.negative_cap),
        seed=int(args.seed),
        support_probabilities=training_support_probabilities,
    )
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    model = FrozenLatentMembershipDecoder(hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1)
    best_loss = float("inf")
    best_state: Mapping[str, torch.Tensor] | None = None
    for _step in range(int(args.steps)):
        index = torch.randint(
            pair_row.numel(),
            (min(int(args.batch_size), pair_row.numel()),),
            generator=generator,
        )
        logits = model(
            latent[pair_row[index]].to(device),
            language[pair_proposal[index]].to(device),
        )
        losses = F.binary_cross_entropy_with_logits(
            logits, pair_label[index].to(device), reduction="none"
        )
        weights = pair_weight[index].to(device)
        loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("membership decoder did not complete one optimization step")
    model.load_state_dict(best_state)
    model.eval()

    calibration_indices = torch.where(training)[0]
    if calibration_indices.numel() > int(args.calibration_proposals):
        positions = torch.linspace(
            0, calibration_indices.numel() - 1, int(args.calibration_proposals)
        ).round().long()
        calibration_indices = calibration_indices[positions]
    decoder_calibration: list[tuple[torch.Tensor, torch.Tensor]] = []
    baseline_calibration: list[tuple[torch.Tensor, torch.Tensor]] = []
    for proposal in calibration_indices.tolist():
        visible = torch.where(view_observed[int(proposal_views[proposal])])[0]
        truth = visible_membership_target(
            visible, support[proposal], num_rows=field.num_gaussians
        )
        decoder_calibration.append(
            (
                _scores_for_proposal(
                    model,
                    latent,
                    language[proposal],
                    visible,
                    device=device,
                    chunk_size=int(args.eval_chunk_size),
                ),
                truth,
            )
        )
        baseline_calibration.append(
            ((baseline_feature[visible] @ language[proposal]).float(), truth)
        )
    decoder_threshold, decoder_train_iou = _calibrate_threshold(
        decoder_calibration, int(args.threshold_candidates)
    )
    baseline_threshold, baseline_train_iou = _calibrate_threshold(
        baseline_calibration, int(args.threshold_candidates)
    )

    decoder_ious: list[float] = []
    baseline_ious: list[float] = []
    for proposal in torch.where(heldout)[0].tolist():
        visible = torch.where(view_observed[int(proposal_views[proposal])])[0]
        truth = visible_membership_target(
            visible, support[proposal], num_rows=field.num_gaussians
        )
        decoder_score = _scores_for_proposal(
            model,
            latent,
            language[proposal],
            visible,
            device=device,
            chunk_size=int(args.eval_chunk_size),
        )
        baseline_score = baseline_feature[visible] @ language[proposal]
        decoder_ious.append(_iou_from_scores(decoder_score, truth, decoder_threshold))
        baseline_ious.append(_iou_from_scores(baseline_score, truth, baseline_threshold))
    decoder_iou = float(torch.tensor(decoder_ious).mean())
    baseline_iou = float(torch.tensor(baseline_ious).mean())
    passed = (
        decoder_iou >= float(args.minimum_heldout_iou)
        and decoder_iou >= baseline_iou + float(args.minimum_gain)
    )

    output = Path(args.output).expanduser().resolve()
    inputs = {
        "field": file_record(field_path),
        "membership": membership_record,
        "language_teacher": teacher_record,
        "query_cache": query_record,
    }
    write_torch_noclobber(
        output,
        {
            "schema": "radio_gs.frozen_l512_text_membership_decoder.v1",
            "schema_version": 1,
            "scene": str(args.scene),
            "state_dict": best_state,
            "threshold": decoder_threshold,
            "metadata": {
                "field_frozen": True,
                "query_independent_training": True,
                "per_gaussian_parameters_added": False,
                "source_label_semantics": "inside_sam_positive_outside_visible_negative_invisible_unknown",
                "track_augmented_training": bool(args.track_augmented_training),
                "soft_membership_targets": bool(args.soft_membership_targets),
                "evaluation_membership_threshold": float(
                    args.evaluation_membership_threshold
                ),
                "training_track_stats": track_stats,
                "score_semantics": "equal_cost_membership_likelihood_ratio",
                "view_split": f"source_view_mod_{args.holdout_stride}_eq_{args.holdout_residue}",
                "benchmark_images_opened": False,
                "benchmark_masks_opened": False,
                "benchmark_metrics_opened": False,
                "text_queries_opened": False,
                "inputs": inputs,
            },
        },
    )
    report = {
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene),
        "training_proposals": int(training.sum()),
        "heldout_proposals": int(heldout.sum()),
        "training_pairs": int(pair_row.numel()),
        "track_augmented_training": bool(args.track_augmented_training),
        "soft_membership_targets": bool(args.soft_membership_targets),
        "evaluation_membership_threshold": float(args.evaluation_membership_threshold),
        "training_track_stats": track_stats,
        "best_train_loss": best_loss,
        "source_train_calibration": {
            "decoder_threshold": decoder_threshold,
            "decoder_iou": decoder_train_iou,
            "primitive_similarity_threshold": baseline_threshold,
            "primitive_similarity_iou": baseline_train_iou,
        },
        "source_heldout_macro_iou": {
            "primitive_similarity": baseline_iou,
            "direct_membership_decoder": decoder_iou,
            "delta": decoder_iou - baseline_iou,
        },
        "gate": {
            "minimum_heldout_iou": float(args.minimum_heldout_iou),
            "minimum_gain": float(args.minimum_gain),
            "passed": passed,
        },
        "output": file_record(output),
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--expected-field-sha256", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expected-membership-sha256", required=True)
    parser.add_argument("--language-teacher", required=True)
    parser.add_argument("--expected-language-teacher-sha256", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-cap", type=int, default=512)
    parser.add_argument("--negative-cap", type=int, default=2048)
    parser.add_argument("--calibration-proposals", type=int, default=32)
    parser.add_argument("--threshold-candidates", type=int, default=64)
    parser.add_argument("--eval-chunk-size", type=int, default=32768)
    parser.add_argument("--holdout-stride", type=int, default=4)
    parser.add_argument("--holdout-residue", type=int, default=3)
    parser.add_argument("--minimum-heldout-iou", type=float, default=0.20)
    parser.add_argument("--minimum-gain", type=float, default=0.02)
    parser.add_argument("--track-augmented-training", action="store_true")
    parser.add_argument("--track-minimum-soft-cosine", type=float, default=0.5)
    parser.add_argument("--soft-membership-targets", action="store_true")
    parser.add_argument("--evaluation-membership-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260823)
    print(json.dumps(train_and_evaluate(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
