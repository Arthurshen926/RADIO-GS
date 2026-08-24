#!/usr/bin/env python3
"""Source-heldout Query-Native image-to-Gaussian posterior sentinel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.interfaces.query_packet import QueryPacket
from radio_gs.models.query_native_gaussian_memory import (
    GaussianGeometry,
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
    negative_supports: list[torch.Tensor],
    positive_cap: int,
    negative_cap: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = supports[proposal]
    target = probabilities[proposal]
    if positive.numel() > positive_cap:
        order = torch.randperm(positive.numel(), generator=generator)[:positive_cap]
        positive, target = positive[order], target[order]
    negative = _sample_without_replacement(
        negative_supports[proposal], negative_cap, generator
    )
    return torch.cat((positive, negative)), torch.cat((target, torch.zeros(negative.numel())))


def _mutually_exclusive_negative_supports(
    hard_supports: list[torch.Tensor],
    proposal_views: torch.Tensor,
    semantic: torch.Tensor,
    *,
    negative_semantic_max: float,
    negative_max_support_iou: float,
) -> list[torch.Tensor]:
    """Precompute explicit negatives once; all remaining rows stay unknown."""

    support_sets = [set(value.tolist()) for value in hard_supports]
    result: list[torch.Tensor] = []
    for proposal, positive in enumerate(support_sets):
        view = int(proposal_views[proposal])
        candidates = torch.where(proposal_views == view)[0]
        candidates = candidates[
            (semantic[candidates] @ semantic[proposal]) <= float(negative_semantic_max)
        ]
        negative: set[int] = set()
        for candidate in candidates.tolist():
            if candidate == proposal:
                continue
            other = support_sets[int(candidate)]
            intersection = len(positive & other)
            if intersection / max(len(positive | other), 1) <= float(negative_max_support_iou):
                negative.update(other - positive)
        result.append(torch.tensor(sorted(negative), dtype=torch.long))
    return result


def _cross_view_episodes(
    authority: dict[str, Any],
    proposal_views: torch.Tensor,
    hard_supports: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Build directed query-view -> target-view same-object episodes.

    Negatives require an explicit DINO-cycle ``different`` edge from the query
    proposal to a proposal in the target view. All unsupported rows stay
    unknown.
    """

    count = int(proposal_views.numel())
    if torch.as_tensor(authority.get("proposal_views")).long().shape != (count,):
        raise ValueError("cross-view authority proposal domain differs")
    left = torch.as_tensor(authority["edge_left"]).long()
    right = torch.as_tensor(authority["edge_right"]).long()
    label = torch.as_tensor(authority["edge_label"]).long()
    matrix = torch.full((count, count), -1, dtype=torch.int8)
    matrix[left, right] = matrix[right, left] = label.to(torch.int8)
    same = torch.where(label == 1)[0]
    query = torch.cat((left[same], right[same]))
    target = torch.cat((right[same], left[same]))
    negative_supports: list[torch.Tensor] = []
    support_sets = [set(value.tolist()) for value in hard_supports]
    for query_proposal, target_proposal in zip(query.tolist(), target.tolist()):
        target_view = int(proposal_views[target_proposal])
        candidates = torch.where(
            (proposal_views == target_view) & (matrix[query_proposal] == 0)
        )[0]
        negative: set[int] = set()
        positive = support_sets[target_proposal]
        for candidate in candidates.tolist():
            negative.update(support_sets[candidate] - positive)
        negative_supports.append(torch.tensor(sorted(negative), dtype=torch.long))
    return query, target, negative_supports


@torch.inference_mode()
def _posterior(
    adapter: ModalityQueryAdapter,
    decoder: QueryNativeGaussianPosteriorDecoder,
    latent: torch.Tensor,
    reliability: torch.Tensor,
    query: torch.Tensor,
    identity_prior: torch.Tensor,
    xyz: torch.Tensor,
    rows: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    token = adapter(query[None].to(device))
    packet = QueryPacket(token, "image")
    logits, _identity = decoder(
        latent[rows].to(device), reliability[rows].to(device), packet,
        identity_prior=identity_prior.to(device),
        geometry=GaussianGeometry(xyz[rows].to(device)),
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
    cross_view_authority = cross_view_record = None
    if args.cross_view_authority:
        cross_view_authority, cross_view_record = _load_mapping(
            args.cross_view_authority, args.expected_cross_view_authority_sha256,
            "source DINO physical-track authority",
        )
    if args.appearance_teacher:
        appearance, appearance_record = _load_mapping(
            args.appearance_teacher, args.expected_appearance_teacher_sha256,
            "source mask appearance teacher",
        )
    field_path = Path(args.field).expanduser().resolve(strict=True)
    field, field_payload, _signature = load_factorized_canonical_field_checkpoint(
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
    if cross_view_authority is not None:
        metadata = cross_view_authority.get("metadata", {})
        if (
            metadata.get("source_only") is not True
            or metadata.get("benchmark_masks_opened") is not False
            or metadata.get("evaluation_rgb_opened") is not False
        ):
            raise ValueError("cross-view physical-track information contract differs")
        episode_query, episode_target, episode_negative_supports = _cross_view_episodes(
            cross_view_authority, proposal_views, hard_support,
        )
        sample_views = proposal_views[episode_target]
        sample_valid = valid[episode_target] & torch.tensor(
            [value.numel() > 0 for value in episode_negative_supports], dtype=torch.bool
        )
    else:
        episode_query = episode_target = torch.arange(proposal_count)
        episode_negative_supports = []
        sample_views = proposal_views
        sample_valid = valid
    heldout = (
        torch.remainder(sample_views, int(args.holdout_stride)) == int(args.holdout_residue)
    ) & sample_valid
    validation = (
        torch.remainder(sample_views, int(args.holdout_stride))
        == int(args.validation_residue)
    ) & sample_valid
    training = (~heldout) & (~validation) & sample_valid
    if int(training.sum()) < 2 or not bool(validation.any()) or not bool(heldout.any()):
        raise ValueError("query-native membership source split differs")

    descriptors = F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
    contexts = F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1)
    semantic = F.normalize(0.75 * descriptors + 0.25 * contexts, dim=-1)
    negative_supports = _mutually_exclusive_negative_supports(
        hard_support, proposal_views, semantic,
        negative_semantic_max=float(args.negative_semantic_max),
        negative_max_support_iou=float(args.negative_max_support_iou),
    )
    appearance_descriptor = None
    if appearance is not None:
        appearance_descriptor = torch.as_tensor(appearance["descriptors"]).float()
        if appearance_descriptor.shape[0] != proposal_count:
            raise ValueError("query-native appearance proposal domain differs")
    query = compose_membership_query_features(semantic, appearance_descriptor)

    def example(sample: int, local_generator: torch.Generator) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        query_proposal = int(episode_query[sample])
        target_proposal = int(episode_target[sample])
        negatives = (
            episode_negative_supports[sample]
            if cross_view_authority is not None else negative_supports[target_proposal]
        )
        positive = soft_support[target_proposal]
        target = soft_values[target_proposal]
        if positive.numel() > int(args.positive_cap):
            order = torch.randperm(positive.numel(), generator=local_generator)[:int(args.positive_cap)]
            positive, target = positive[order], target[order]
        negative = _sample_without_replacement(
            negatives, int(args.negative_cap), local_generator
        )
        local_rows = torch.cat((positive, negative))
        target = torch.cat((target, torch.zeros(negative.numel())))
        return query_proposal, target_proposal, local_rows, target
    baseline = F.normalize(torch.as_tensor(
        query_cache.get("features", query_cache.get("summary_features"))
    ).float(), dim=-1)
    device = torch.device(args.device)
    field = field.to(device).eval()
    baseline_device = baseline.to(device)
    semantic_device = semantic.to(device)
    representation = str(args.memory_representation)
    if representation == "radio_projected":
        projection_generator = torch.Generator(device=device).manual_seed(int(args.projection_seed))
        projection = torch.empty(
            field.decoder.feature_dim, int(args.projected_dim), dtype=torch.float32,
            device=device,
        )
        projection.bernoulli_(0.5, generator=projection_generator).mul_(2).sub_(1)
        projection.div_(float(args.projected_dim) ** 0.5)
        chunks = []
        with torch.inference_mode():
            for start in range(0, field.num_gaussians, int(args.memory_chunk_size)):
                selected = torch.arange(
                    start, min(start + int(args.memory_chunk_size), field.num_gaussians),
                    device=device,
                )
                chunks.append(field.query_memory(
                    selected, representation=representation, radio_projection=projection
                ).cpu())
        latent = torch.cat(chunks).float().contiguous()
    else:
        with torch.inference_mode():
            latent = field.query_memory(representation=representation).detach().cpu().float().contiguous()
    xyz = torch.as_tensor(query_cache["xyz"]).float().contiguous()
    xyz_sha256 = hashlib.sha256(
        xyz.numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()
    if xyz_sha256 != field_payload["geometry_fingerprint"]["xyz_sha256"]:
        raise ValueError("query-native query-cache geometry differs from field")
    reliability = torch.as_tensor(universal.get("reliability")).float().contiguous()
    if (
        latent.ndim != 2 or latent.shape[0] != field.num_gaussians
        or reliability.shape != (field.num_gaussians, 5)
        or observed.shape[1] != field.num_gaussians
        or xyz.shape != (field.num_gaussians, 3)
    ):
        raise ValueError("query-native membership Gaussian domain differs")

    torch.manual_seed(int(args.seed))
    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    adapter = ModalityQueryAdapter(query.shape[1], int(args.query_dim)).to(device)
    decoder = QueryNativeGaussianPosteriorDecoder(
        latent_dim=latent.shape[1], query_dim=int(args.query_dim),
        hidden_dim=int(args.hidden_dim), topk_anchors=int(args.topk_anchors),
    ).to(device)
    initialization_record = None
    if args.initial_model:
        initialization, initialization_record = _load_mapping(
            args.initial_model, args.expected_initial_model_sha256,
            "source-only query-native initialization",
        )
        init_metadata = initialization.get("metadata", {})
        if (
            init_metadata.get("source_only") is not True
            or not (
                init_metadata.get("cross_view_same_object_authority") is True
                or init_metadata.get("evaluation_semantics")
                == "query_view_to_target_view_DINO_cycle_confirmed_same_object"
            )
        ):
            raise ValueError("cross-scene initialization information contract differs")
        adapter.load_state_dict(initialization["adapter_state_dict"])
        decoder.load_state_dict(initialization["decoder_state_dict"])
    if bool(args.freeze_adapter):
        adapter.requires_grad_(False)
    parameters = [value for value in adapter.parameters() if value.requires_grad]
    parameters += list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    training_indices = torch.where(training)[0]
    best_validation_loss = float("inf")
    best_adapter = best_decoder = None
    for step in range(int(args.steps)):
        sample = int(training_indices[step % training_indices.numel()])
        proposal, _target_proposal, local_rows, target = example(sample, generator)
        if not bool((target == 0).any()):
            continue
        token = adapter(query[proposal:proposal + 1].to(device))
        local_rows_device = local_rows.to(device)
        identity_prior = baseline_device[local_rows_device] @ semantic_device[proposal]
        logits, identity = decoder(
            latent[local_rows].to(device), reliability[local_rows].to(device),
            QueryPacket(token, "image"), identity_prior=identity_prior.to(device),
            geometry=GaussianGeometry(xyz[local_rows].to(device)),
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
        if (step + 1) % int(args.validation_interval) == 0 or step + 1 == int(args.steps):
            adapter.eval(); decoder.eval()
            validation_generator = torch.Generator().manual_seed(int(args.seed) + 2)
            validation_losses = []
            selected_validation = torch.where(validation)[0]
            with torch.no_grad():
                for validation_sample in selected_validation.tolist():
                    validation_proposal, _validation_target_proposal, validation_rows, validation_target = example(
                        validation_sample, validation_generator,
                    )
                    if not bool((validation_target == 0).any()):
                        continue
                    validation_token = adapter(query[validation_proposal:validation_proposal + 1].to(device))
                    validation_rows_device = validation_rows.to(device)
                    validation_prior = (
                        baseline_device[validation_rows_device]
                        @ semantic_device[validation_proposal]
                    )
                    validation_logits, _ = decoder(
                        latent[validation_rows].to(device), reliability[validation_rows].to(device),
                        QueryPacket(validation_token, "image"),
                        identity_prior=validation_prior.to(device),
                        geometry=GaussianGeometry(xyz[validation_rows].to(device)),
                    )
                    validation_losses.append(float(F.binary_cross_entropy_with_logits(
                        validation_logits, validation_target.to(device)
                    )))
                    if len(validation_losses) >= int(args.validation_proposals):
                        break
            adapter.train(); decoder.train()
            if validation_losses:
                validation_loss = sum(validation_losses) / len(validation_losses)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_adapter = {key: tensor.detach().cpu().clone() for key, tensor in adapter.state_dict().items()}
                    best_decoder = {key: tensor.detach().cpu().clone() for key, tensor in decoder.state_dict().items()}
    if best_adapter is None or best_decoder is None:
        raise RuntimeError("query-native membership optimization did not complete")
    adapter.load_state_dict(best_adapter)
    decoder.load_state_dict(best_decoder)
    adapter.eval(); decoder.eval()

    calibration_indices = torch.where(validation)[0]
    if calibration_indices.numel() > int(args.calibration_proposals):
        positions = torch.linspace(0, calibration_indices.numel() - 1, int(args.calibration_proposals)).round().long()
        calibration_indices = calibration_indices[positions]
    query_native_calibration: list[tuple[torch.Tensor, torch.Tensor]] = []
    baseline_calibration: list[tuple[torch.Tensor, torch.Tensor]] = []
    for sample in calibration_indices.tolist():
        proposal, target_proposal = int(episode_query[sample]), int(episode_target[sample])
        visible = torch.where(observed[int(proposal_views[target_proposal])])[0]
        truth = visible_membership_target(visible, hard_support[target_proposal], num_rows=field.num_gaussians)
        query_native_calibration.append((
            _posterior(
                adapter, decoder, latent, reliability, query[proposal],
                baseline_device[visible.to(device)] @ semantic_device[proposal],
                xyz, visible, device,
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
    for sample in torch.where(heldout)[0].tolist():
        proposal, target_proposal = int(episode_query[sample]), int(episode_target[sample])
        visible = torch.where(observed[int(proposal_views[target_proposal])])[0]
        truth = visible_membership_target(visible, hard_support[target_proposal], num_rows=field.num_gaussians)
        score = _posterior(
            adapter, decoder, latent, reliability, query[proposal],
            baseline_device[visible.to(device)] @ semantic_device[proposal],
            xyz, visible, device,
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
    if cross_view_record is not None:
        inputs["cross_view_authority"] = cross_view_record
    if initialization_record is not None:
        inputs["initial_model"] = initialization_record
    write_torch_noclobber(output, {
        "schema": "radio_gs.query_native_membership_decoder.v2", "schema_version": 2,
        "scene": str(args.scene), "adapter_state_dict": best_adapter,
        "decoder_state_dict": best_decoder, "threshold": threshold,
        "metadata": {
            "field_frozen": True, "per_gaussian_parameters_added": False,
            "teacher_features_decoded": False, "direct_gaussian_posterior": True,
            "query_modality": "image", "source_only": True,
            "memory_representation": representation,
            "negative_semantics": "mutually_exclusive_dissimilar_proposals_only_else_unknown",
            "checkpoint_selection": "independent_source_validation_loss",
            "evaluation_semantics": (
                "query_view_to_target_view_DINO_cycle_confirmed_same_object"
                if cross_view_authority is not None
                else "unseen_proposal_not_same_object_cross_view"
            ),
            "benchmark_images_opened": False, "benchmark_masks_opened": False,
            "cross_view_same_object_authority": cross_view_authority is not None,
            "cross_scene_source_initialization": initialization_record is not None,
            "adapter_frozen_after_source_initialization": bool(args.freeze_adapter),
            "inputs": inputs,
        },
    })
    report = {
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene), "training_proposals": int(training.sum()),
        "validation_proposals": int(validation.sum()),
        "heldout_proposals": int(heldout.sum()),
        "best_validation_loss": best_validation_loss,
        "memory_representation": representation,
        "query_dim": int(args.query_dim), "appearance_teacher": appearance is not None,
        "cross_view_same_object_authority": cross_view_authority is not None,
        "source_validation_calibration": {
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
    parser.add_argument("--cross-view-authority", default="")
    parser.add_argument("--expected-cross-view-authority-sha256", default="")
    parser.add_argument("--initial-model", default="")
    parser.add_argument("--expected-initial-model-sha256", default="")
    parser.add_argument("--freeze-adapter", action="store_true")
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--topk-anchors", type=int, default=6)
    parser.add_argument(
        "--memory-representation", choices=("local_codes", "coefficients", "radio_projected"),
        default="coefficients",
    )
    parser.add_argument("--projected-dim", type=int, default=512)
    parser.add_argument("--projection-seed", type=int, default=20260824)
    parser.add_argument("--memory-chunk-size", type=int, default=2048)
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
    parser.add_argument("--validation-residue", type=int, default=2)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-proposals", type=int, default=8)
    parser.add_argument("--negative-semantic-max", type=float, default=0.2)
    parser.add_argument("--negative-max-support-iou", type=float, default=0.05)
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
