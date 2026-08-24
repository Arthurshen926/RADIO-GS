#!/usr/bin/env python3
"""Distill the consumed native ScanNet categorical score variable into L512.

Unlike descriptor-space distillation, this stage reproduces the exact native
region readout after per-view class scoring, class-symmetric centering,
agreement gating, and structural replay.  The protocol vocabulary is opened,
but benchmark RGB, labels, masks, and metrics are not.  The learned state is a
single scene-global decoder; no per-Gaussian parameter is added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.models.frozen_latent_split_score_decoder import (
    FrozenReliabilityEligibilityGate,
    FrozenLatentSplitScoreDecoder,
)
from radio_gs.scannet_constants import NYU40_ID_TO_NAME, OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts.evaluate_scannet_native_sam_siglip_region_vote import (
    STRUCTURAL_IDS,
    _region_per_view,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


SPLITS = ("19", "15", "10")
SPLIT_DIMS = (19, 15, 10)


def _load(path: str, digest: str, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    value, actual, source = load_sha_bound_project_checkpoint_mapping(
        path, expected_sha256=digest, map_location="cpu", label=label
    )
    return dict(value), {"path": str(source), "sha256": actual}


def _embedding(path: str, digest: str, expected_names: list[str]) -> tuple[torch.Tensor, dict[str, str]]:
    payload, record = _load(path, digest, "ScanNet categorical text bank")
    names = payload.get("queries", payload.get("class_names"))
    values = F.normalize(torch.as_tensor(payload.get("embeddings")).float(), dim=-1)
    if list(names or []) != expected_names or values.shape != (len(expected_names), 1536):
        raise ValueError("ScanNet categorical text bank domain differs")
    return values, record


def _block_metrics(
    prediction: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor,
    eligible: torch.Tensor,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    cursor = 0
    for split, width in zip(SPLITS, SPLIT_DIMS):
        selected = eligible[:, cursor:cursor + width].any(dim=1)
        unchanged = ~selected
        p = prediction[:, cursor:cursor + width]
        t = target[:, cursor:cursor + width]
        b = baseline[:, cursor:cursor + width]
        if not bool(selected.any()):
            raise ValueError(f"split{split} source validation has no eligible row")
        replay_agreement = (
            float((p[unchanged].argmax(1) == b[unchanged].argmax(1)).float().mean())
            if bool(unchanged.any()) else 1.0
        )
        output[split] = {
            "baseline_eligible_mae": float((b[selected] - t[selected]).abs().mean()),
            "decoder_eligible_mae": float((p[selected] - t[selected]).abs().mean()),
            "decoder_teacher_top1_agreement": float((p[selected].argmax(1) == t[selected].argmax(1)).float().mean()),
            "unchanged_baseline_top1_agreement": replay_agreement,
        }
        cursor += width
    return output


@torch.inference_mode()
def _decode(
    model: FrozenLatentSplitScoreDecoder, latent: torch.Tensor,
    baseline: torch.Tensor, device: torch.device, chunk: int,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for start in range(0, latent.shape[0], int(chunk)):
        stop = min(start + int(chunk), latent.shape[0])
        # The categorical cache has only 44 channels.  Preserve FP32 so an
        # abstaining gate replays baseline decisions exactly; FP16 rounding can
        # otherwise swap near-tied classes even when no residual is authorized.
        values.append(
            model(
                latent[start:stop].to(device), baseline[start:stop].to(device)
            ).float().cpu()
        )
    return torch.cat(values).float()


def run(args: argparse.Namespace) -> dict[str, Any]:
    membership, membership_record = _load(
        args.membership, args.expected_membership_sha256, "native source SAM membership"
    )
    teacher, teacher_record = _load(
        args.proposal_teacher, args.expected_proposal_teacher_sha256,
        "native source SigLIP region teacher",
    )
    baseline_payload, baseline_record = _load(
        args.baseline_query_cache, args.expected_baseline_query_cache_sha256,
        "restored categorical descriptor baseline",
    )
    field_path = Path(args.field).expanduser().resolve(strict=True)
    field, _payload, _signature = load_factorized_canonical_field_checkpoint(
        field_path, map_location="cpu", expected_sha256=args.expected_field_sha256
    )
    reliability: torch.Tensor | None = None
    universal_record: dict[str, str] | None = None
    if args.universal_field:
        universal, universal_record = _load(
            args.universal_field, args.expected_universal_field_sha256,
            "Universal Field reliability authority",
        )
        migration = universal.get("universal_field_migration", {})
        if migration.get("source_field_sha256") != args.expected_field_sha256:
            raise ValueError("Universal Field reliability is bound to another L512 field")
        reliability = torch.as_tensor(universal.get("reliability")).float().contiguous()
    if (
        membership.get("metadata", {}).get("benchmark_masks_opened") is not False
        or teacher.get("metadata", {}).get("source_only") is not True
        or teacher.get("metadata", {}).get("benchmark_masks_opened") is not False
    ):
        raise ValueError("native categorical source authority differs")
    rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float().clamp_min(0)
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    descriptor = F.normalize(
        0.75 * F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
        + 0.25 * F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1),
        dim=-1,
    )
    xyz = torch.as_tensor(baseline_payload["xyz"]).float().contiguous()
    baseline_descriptor = F.normalize(
        torch.as_tensor(baseline_payload.get("summary_features", baseline_payload.get("features"))).float(), dim=-1
    )
    latent = field.local_codes.detach().cpu().float().contiguous()
    num_rows = int(membership["num_rows"])
    if (
        xyz.shape != (num_rows, 3) or baseline_descriptor.shape != (num_rows, 1536)
        or latent.shape != (num_rows, 512) or descriptor.shape[0] != int(membership["num_proposals"])
    ):
        raise ValueError("categorical teacher and frozen field row domains differ")
    if bool(args.predict_teacher_eligibility) and (
        reliability is None or reliability.shape != (num_rows, 5)
    ):
        raise ValueError("eligibility prediction requires aligned Universal Field reliability")
    if bool(args.predict_teacher_eligibility) and bool(args.replay_teacher_eligibility):
        raise ValueError("select predicted or teacher-replayed eligibility, not both")

    primitive_paths = args.primitive_text_banks.split(",")
    region_paths = args.region_text_banks.split(",")
    primitive_hashes = args.expected_primitive_text_sha256.split(",")
    region_hashes = args.expected_region_text_sha256.split(",")
    if not all(len(values) == 3 for values in (primitive_paths, region_paths, primitive_hashes, region_hashes)):
        raise ValueError("three split text-bank paths and hashes are required")
    baseline_blocks: list[torch.Tensor] = []
    target_blocks: list[torch.Tensor] = []
    eligible_blocks: list[torch.Tensor] = []
    text_records: dict[str, Any] = {}
    for position, split in enumerate(SPLITS):
        class_ids = tuple(int(value) for value in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split])
        names = [NYU40_ID_TO_NAME[value] for value in class_ids]
        primitive_text, primitive_record = _embedding(
            primitive_paths[position], primitive_hashes[position], names
        )
        region_text, region_record = _embedding(
            region_paths[position], region_hashes[position], names
        )
        primitive = baseline_descriptor @ primitive_text.T
        primitive = F.normalize(primitive - primitive.mean(1, keepdim=True), dim=-1)
        proposal_scores = descriptor @ region_text.T
        region, view_count, agreement = _region_per_view(
            row_indices=rows, proposal_indices=proposals, weights=weights,
            proposal_views=proposal_views, proposal_scores=proposal_scores,
            num_rows=num_rows,
        )
        region = F.normalize(region - region.mean(1, keepdim=True), dim=-1)
        primitive_labels = torch.tensor(class_ids)[primitive.argmax(1)]
        eligible = (
            (view_count >= int(args.minimum_views))
            & (agreement >= float(args.minimum_view_agreement))
            & ~torch.isin(primitive_labels, torch.tensor(sorted(STRUCTURAL_IDS)))
        )
        target = primitive.clone()
        target[eligible] = F.normalize(
            (1.0 - float(args.alpha)) * primitive[eligible]
            + float(args.alpha) * agreement[eligible, None] * region[eligible], dim=-1,
        )
        baseline_blocks.append(primitive.contiguous())
        target_blocks.append(target.contiguous())
        eligible_blocks.append(eligible[:, None].expand(-1, len(class_ids)))
        text_records[split] = {"primitive": primitive_record, "region": region_record}
    baseline = torch.cat(baseline_blocks, dim=1)
    target = torch.cat(target_blocks, dim=1)
    eligible_channels = torch.cat(eligible_blocks, dim=1)
    eligible_rows = eligible_channels.any(1)
    row_ids = torch.arange(num_rows)
    validation_rows = row_ids % int(args.holdout_stride) == int(args.holdout_residue)
    validation_rows &= eligible_rows
    training_changed = torch.where(eligible_rows & ~validation_rows)[0]
    training_replay = torch.where(~eligible_rows)[0]
    validation_indices = torch.where(validation_rows)[0]
    if min(training_changed.numel(), training_replay.numel(), validation_indices.numel()) < 128:
        raise ValueError("categorical source-row split is too small")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    model = FrozenLatentSplitScoreDecoder(hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    best_objective = float("inf")
    best_step = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    for step in range(1, int(args.steps) + 1):
        half = min(int(args.batch_size) // 2, training_changed.numel(), training_replay.numel())
        changed = training_changed[torch.randint(training_changed.numel(), (half,), generator=generator)]
        replay = training_replay[torch.randint(training_replay.numel(), (half,), generator=generator)]
        sampled = torch.cat((changed, replay))
        prediction = model(latent[sampled].to(device), baseline[sampled].to(device))
        local_target = target[sampled].to(device)
        local_eligible = eligible_channels[sampled].to(device)
        weight = 1.0 + float(args.changed_channel_weight) * local_eligible.float()
        coordinate = F.smooth_l1_loss(prediction, local_target, reduction="none")
        loss = (coordinate * weight).sum() / weight.sum().clamp_min(1)
        cursor = 0
        margin_loss = torch.zeros((), device=device)
        for width in SPLIT_DIMS:
            p, t = prediction[:, cursor:cursor + width], local_target[:, cursor:cursor + width]
            top2 = t.topk(k=2, dim=1).indices
            margin_loss = margin_loss + F.smooth_l1_loss(
                p.gather(1, top2)[:, 0] - p.gather(1, top2)[:, 1],
                t.gather(1, top2)[:, 0] - t.gather(1, top2)[:, 1],
            )
            cursor += width
        loss = loss + float(args.margin_weight) * margin_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % int(args.validation_interval) != 0 and step != int(args.steps):
            continue
        candidate = _decode(model, latent[validation_indices], baseline[validation_indices], device, int(args.eval_chunk_size))
        metrics = _block_metrics(
            candidate, target[validation_indices], baseline[validation_indices],
            eligible_channels[validation_indices],
        )
        objective = sum(value["decoder_eligible_mae"] for value in metrics.values())
        history.append({"step": step, "loss": float(loss.detach()), "objective": objective})
        if objective < best_objective:
            best_objective, best_step = objective, step
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    restored = _decode(
        model, latent, baseline, device, int(args.eval_chunk_size)
    ).clone()
    eligibility = torch.stack(
        [value[:, 0] for value in eligible_blocks], dim=1
    ).bool()
    gate_state: dict[str, torch.Tensor] | None = None
    gate_thresholds: list[float] | None = None
    gate_metrics: dict[str, Any] | None = None
    if bool(args.predict_teacher_eligibility):
        assert reliability is not None
        gate = FrozenReliabilityEligibilityGate(hidden_dim=int(args.gate_hidden_dim)).to(device)
        gate_optimizer = torch.optim.AdamW(
            gate.parameters(), lr=float(args.gate_learning_rate), weight_decay=float(args.weight_decay)
        )
        gate_training = torch.where(row_ids % int(args.holdout_stride) != int(args.holdout_residue))[0]
        gate_validation = torch.where(row_ids % int(args.holdout_stride) == int(args.holdout_residue))[0]
        positives = eligibility[gate_training].float().sum(0)
        pos_weight = (gate_training.numel() - positives) / positives.clamp_min(1)
        best_gate_loss = float("inf")
        for step in range(1, int(args.gate_steps) + 1):
            sampled = gate_training[torch.randint(
                gate_training.numel(),
                (min(int(args.batch_size), gate_training.numel()),),
                generator=generator,
            )]
            logits = gate(
                latent[sampled].to(device), reliability[sampled].to(device),
                baseline[sampled].to(device),
            )
            gate_loss = F.binary_cross_entropy_with_logits(
                logits, eligibility[sampled].float().to(device),
                pos_weight=pos_weight.to(device),
            )
            gate_optimizer.zero_grad(set_to_none=True)
            gate_loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            gate_optimizer.step()
            if step % int(args.validation_interval) == 0 or step == int(args.gate_steps):
                with torch.inference_mode():
                    validation_logits = gate(
                        latent[gate_validation].to(device), reliability[gate_validation].to(device),
                        baseline[gate_validation].to(device),
                    )
                    value = float(F.binary_cross_entropy_with_logits(
                        validation_logits, eligibility[gate_validation].float().to(device),
                        pos_weight=pos_weight.to(device),
                    ))
                if value < best_gate_loss:
                    best_gate_loss = value
                    gate_state = {key: tensor.detach().cpu().clone() for key, tensor in gate.state_dict().items()}
        if gate_state is None:
            raise RuntimeError("eligibility gate did not complete")
        gate.load_state_dict(gate_state)
        with torch.inference_mode():
            validation_probability = torch.sigmoid(gate(
                latent[gate_validation].to(device), reliability[gate_validation].to(device),
                baseline[gate_validation].to(device),
            )).cpu()
        gate_thresholds = []
        gate_metrics = {}
        cursor = 0
        for index, (split, width) in enumerate(zip(SPLITS, SPLIT_DIMS)):
            target_block = target[gate_validation, cursor:cursor + width]
            baseline_block = baseline[gate_validation, cursor:cursor + width]
            restored_block = restored[gate_validation, cursor:cursor + width]
            candidates: list[tuple[float, float, float]] = []
            truth = eligibility[gate_validation, index]
            for threshold in (
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                0.95, 0.975, 0.99, 1.01,
            ):
                selected = validation_probability[:, index] >= threshold
                candidate = torch.where(selected[:, None], restored_block, baseline_block)
                # Match the score-decoder gate: replay safety is measured on
                # rows eligible for at least one split, where cross-split false
                # positives can actually perturb a categorical decision.
                replay = (~truth) & eligible_rows[gate_validation]
                if not bool(replay.any()):
                    replay = ~truth
                replay_agreement = (
                    float((candidate[replay].argmax(1) == baseline_block[replay].argmax(1)).float().mean())
                    if bool(replay.any()) else 1.0
                )
                candidates.append((float((candidate - target_block).abs().mean()), threshold, replay_agreement))
            safe = [value for value in candidates if value[2] >= float(args.minimum_replay_agreement)]
            mae, threshold, replay_agreement = (
                min(safe) if safe else min(candidates, key=lambda value: (-value[2], value[0], value[1]))
            )
            gate_thresholds.append(threshold)
            prediction = validation_probability[:, index] >= threshold
            tp = int((prediction & truth).sum())
            fp = int((prediction & ~truth).sum())
            fn = int((~prediction & truth).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            gate_metrics[split] = {
                "threshold": threshold, "teacher_score_mae": mae,
                "replay_top1_agreement": replay_agreement,
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            }
            cursor += width
        probability_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, num_rows, int(args.eval_chunk_size)):
                stop = min(start + int(args.eval_chunk_size), num_rows)
                probability_parts.append(torch.sigmoid(gate(
                    latent[start:stop].to(device), reliability[start:stop].to(device),
                    baseline[start:stop].to(device),
                )).cpu())
        predicted_gate = torch.cat(probability_parts)
        cursor = 0
        for index, width in enumerate(SPLIT_DIMS):
            selected = predicted_gate[:, index] >= gate_thresholds[index]
            restored[:, cursor:cursor + width] = torch.where(
                selected[:, None], restored[:, cursor:cursor + width],
                baseline[:, cursor:cursor + width],
            )
            cursor += width
    elif bool(args.replay_teacher_eligibility):
        # Capacity diagnostic: preserve the native compiler's compact 3-bit
        # eligibility gate while testing whether the global score decoder can
        # absorb its high-dimensional region teacher.  This gate must be
        # distilled or bound to existing reliability before final promotion.
        restored[~eligible_channels] = baseline[~eligible_channels]
    validation_metrics = _block_metrics(
        restored[validation_indices], target[validation_indices], baseline[validation_indices],
        eligible_channels[validation_indices],
    )
    if bool(args.predict_teacher_eligibility):
        noninferior = all(
            values["decoder_eligible_mae"]
            <= values["baseline_eligible_mae"] + 1e-8
            and values["unchanged_baseline_top1_agreement"]
            >= float(args.minimum_replay_agreement)
            for values in validation_metrics.values()
        )
        aggregate_improvement = sum(
            values["decoder_eligible_mae"] for values in validation_metrics.values()
        ) < sum(
            values["baseline_eligible_mae"] for values in validation_metrics.values()
        ) - 1e-8
        passed = best_step > 0 and noninferior and aggregate_improvement
    else:
        passed = best_step > 0 and all(
            values["decoder_eligible_mae"] < values["baseline_eligible_mae"]
            and values["unchanged_baseline_top1_agreement"]
            >= float(args.minimum_replay_agreement)
            for values in validation_metrics.values()
        )
    output_model = Path(args.output_model).expanduser().resolve()
    inputs = {
        "field": file_record(field_path), "membership": membership_record,
        "proposal_teacher": teacher_record, "baseline_query_cache": baseline_record,
        "text_banks": text_records,
    }
    if universal_record is not None:
        inputs["universal_field"] = universal_record
    write_torch_noclobber(output_model, {
        "schema": "radio_gs.frozen_l512_native_categorical_score_decoder.v1",
        "schema_version": 1, "scene": str(args.scene), "state_dict": best_state,
        "eligibility_gate_state_dict": gate_state,
        "eligibility_gate_thresholds": gate_thresholds,
        "metadata": {"field_frozen": True, "per_gaussian_parameters_added": False,
            "protocol_vocabulary_opened": True, "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "teacher_eligibility_replayed": bool(args.replay_teacher_eligibility),
            "teacher_eligibility_predicted": bool(args.predict_teacher_eligibility),
            "predicted_gate_source_rule": (
                "per_split_noninferior_replay_safe_and_aggregate_strict_gain"
                if bool(args.predict_teacher_eligibility) else None
            ),
            "inputs": inputs},
    })
    output_cache = Path(args.output_score_cache).expanduser().resolve()
    values = restored.split(SPLIT_DIMS, dim=1)
    write_torch_noclobber(output_cache, {
        "schema": "radio_gs.scannet_source_distilled_score_cache.v1",
        "schema_version": 1, "xyz": xyz, "valid": torch.ones(num_rows, dtype=torch.bool),
        "direct_observed": eligible_rows,
        **{
            f"scores_split_{split}": value.float().contiguous()
            for split, value in zip(SPLITS, values)
        },
        "metadata": {"artifact_type": "radio_gs_scannet_source_distilled_score_cache",
            "query_independent": False, "source_only": True, "evaluation_diagnostic_only": False,
            "benchmark_images_opened": False, "benchmark_masks_opened": False,
            "benchmark_labels_opened": False, "text_queries_opened": True,
            "postprocessing": "none", "source_gate_passed": passed,
            "teacher_eligibility_replayed": bool(args.replay_teacher_eligibility),
            "teacher_eligibility_predicted": bool(args.predict_teacher_eligibility),
            "final_promotion_requires_gate_compression": bool(args.replay_teacher_eligibility),
            "decoder": file_record(output_model), "inputs": inputs},
    })
    report = {"status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene), "eligible_rows": int(eligible_rows.sum()),
        "validation_rows": int(validation_indices.numel()), "best_step": best_step,
        "teacher_eligibility_replayed": bool(args.replay_teacher_eligibility),
        "teacher_eligibility_predicted": bool(args.predict_teacher_eligibility),
        "eligibility_gate_validation": gate_metrics,
        "validation": validation_metrics, "history": history,
        "model": file_record(output_model), "score_cache": file_record(output_cache)}
    write_frozen_json(output_cache.with_suffix(output_cache.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--expected-field-sha256", required=True)
    parser.add_argument("--universal-field", default="")
    parser.add_argument("--expected-universal-field-sha256", default="")
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expected-membership-sha256", required=True)
    parser.add_argument("--proposal-teacher", required=True)
    parser.add_argument("--expected-proposal-teacher-sha256", required=True)
    parser.add_argument("--baseline-query-cache", required=True)
    parser.add_argument("--expected-baseline-query-cache-sha256", required=True)
    parser.add_argument("--primitive-text-banks", required=True)
    parser.add_argument("--expected-primitive-text-sha256", required=True)
    parser.add_argument("--region-text-banks", required=True)
    parser.add_argument("--expected-region-text-sha256", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-score-cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--changed-channel-weight", type=float, default=3.0)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--minimum-views", type=int, default=2)
    parser.add_argument("--minimum-view-agreement", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--holdout-stride", type=int, default=8)
    parser.add_argument("--holdout-residue", type=int, default=7)
    parser.add_argument("--minimum-replay-agreement", type=float, default=0.995)
    parser.add_argument("--replay-teacher-eligibility", action="store_true")
    parser.add_argument("--predict-teacher-eligibility", action="store_true")
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--gate-steps", type=int, default=600)
    parser.add_argument("--gate-learning-rate", type=float, default=2e-3)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--eval-chunk-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260824)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
