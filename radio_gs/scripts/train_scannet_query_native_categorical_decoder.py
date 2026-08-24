#!/usr/bin/env python3
"""Train an open-set-equivariant ScanNet query-native score compiler.

The frozen L512 field is queried directly with text tokens.  No descriptor is
decoded and no class-indexed parameter exists.  Native source RGB teachers
supervise only the categorical score variable consumed by evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.models.query_native_gaussian_memory import (
    QuerySetCategoricalDecoder,
    QuerySetEligibilityGate,
)
from radio_gs.scannet_constants import NYU40_ID_TO_NAME, OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts.evaluate_scannet_native_sam_siglip_region_vote import (
    STRUCTURAL_IDS,
    _region_per_view,
)
from radio_gs.scripts.train_scannet_frozen_l512_native_categorical_score_decoder import (
    _embedding,
    _load,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    write_frozen_json,
    write_torch_noclobber,
)


SPLITS = ("19", "15", "10")


def _query_holdout(name: str, modulus: int, residue: int) -> bool:
    if modulus <= 0:
        return False
    value = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
    return value % modulus == residue


@torch.inference_mode()
def _decode_block(
    model: QuerySetCategoricalDecoder,
    latent: torch.Tensor,
    reliability: torch.Tensor,
    query: torch.Tensor,
    baseline: torch.Tensor,
    device: torch.device,
    chunk: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    local_query = query.to(device)
    for start in range(0, latent.shape[0], int(chunk)):
        stop = min(start + int(chunk), latent.shape[0])
        output.append(model(
            latent[start:stop].to(device), reliability[start:stop].to(device),
            local_query, baseline[start:stop].to(device),
        ).float().cpu())
    return torch.cat(output)


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
    eligible: torch.Tensor,
    query_holdout: torch.Tensor,
) -> dict[str, float | None]:
    selected = eligible
    replay = ~eligible
    result: dict[str, float | None] = {
        "baseline_eligible_mae": float((baseline[selected] - target[selected]).abs().mean()),
        "decoder_eligible_mae": float((prediction[selected] - target[selected]).abs().mean()),
        "teacher_top1_agreement": float(
            (prediction[selected].argmax(1) == target[selected].argmax(1)).float().mean()
        ),
        "replay_top1_agreement": (
            float((prediction[replay].argmax(1) == baseline[replay].argmax(1)).float().mean())
            if bool(replay.any()) else 1.0
        ),
        "heldout_query_count": int(query_holdout.sum()),
        "heldout_query_baseline_mae": None,
        "heldout_query_decoder_mae": None,
    }
    if bool(query_holdout.any()):
        result["heldout_query_baseline_mae"] = float(
            (baseline[selected][:, query_holdout] - target[selected][:, query_holdout]).abs().mean()
        )
        result["heldout_query_decoder_mae"] = float(
            (prediction[selected][:, query_holdout] - target[selected][:, query_holdout]).abs().mean()
        )
    return result


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
    universal, universal_record = _load(
        args.universal_field, args.expected_universal_field_sha256,
        "Universal Field reliability authority",
    )
    field_path = Path(args.field).expanduser().resolve(strict=True)
    field, _payload, _signature = load_factorized_canonical_field_checkpoint(
        field_path, map_location="cpu", expected_sha256=args.expected_field_sha256
    )
    migration = universal.get("universal_field_migration", {})
    if migration.get("source_field_sha256") != args.expected_field_sha256:
        raise ValueError("Universal Field reliability is bound to another L512 field")
    if (
        membership.get("metadata", {}).get("benchmark_masks_opened") is not False
        or teacher.get("metadata", {}).get("source_only") is not True
        or teacher.get("metadata", {}).get("benchmark_masks_opened") is not False
    ):
        raise ValueError("query-native categorical source authority differs")

    rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float().clamp_min(0)
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    descriptor = F.normalize(
        0.75 * F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
        + 0.25 * F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1),
        dim=-1,
    )
    baseline_descriptor = F.normalize(torch.as_tensor(
        baseline_payload.get("summary_features", baseline_payload.get("features"))
    ).float(), dim=-1)
    xyz = torch.as_tensor(baseline_payload["xyz"]).float().contiguous()
    latent = field.local_codes.detach().cpu().float().contiguous()
    reliability = torch.as_tensor(universal.get("reliability")).float().contiguous()
    num_rows = int(membership["num_rows"])
    if (
        xyz.shape != (num_rows, 3) or latent.shape != (num_rows, 512)
        or reliability.shape != (num_rows, 5)
        or baseline_descriptor.shape != (num_rows, 1536)
        or descriptor.shape[0] != int(membership["num_proposals"])
    ):
        raise ValueError("query-native categorical row domain differs")

    primitive_paths = args.primitive_text_banks.split(",")
    region_paths = args.region_text_banks.split(",")
    primitive_hashes = args.expected_primitive_text_sha256.split(",")
    region_hashes = args.expected_region_text_sha256.split(",")
    if not all(len(value) == 3 for value in (
        primitive_paths, region_paths, primitive_hashes, region_hashes
    )):
        raise ValueError("three split text-bank paths and hashes are required")

    queries: list[torch.Tensor] = []
    baselines: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    eligible: list[torch.Tensor] = []
    query_holdouts: list[torch.Tensor] = []
    text_records: dict[str, Any] = {}
    split_names: list[list[str]] = []
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
        local_eligible = (
            (view_count >= int(args.minimum_views))
            & (agreement >= float(args.minimum_view_agreement))
            & ~torch.isin(primitive_labels, torch.tensor(sorted(STRUCTURAL_IDS)))
        )
        target = primitive.clone()
        target[local_eligible] = F.normalize(
            (1.0 - float(args.alpha)) * primitive[local_eligible]
            + float(args.alpha) * agreement[local_eligible, None] * region[local_eligible],
            dim=-1,
        )
        queries.append(primitive_text.contiguous())
        baselines.append(primitive.contiguous())
        targets.append(target.contiguous())
        eligible.append(local_eligible.contiguous())
        query_holdouts.append(torch.tensor([
            _query_holdout(name, int(args.query_holdout_modulus), int(args.query_holdout_residue))
            for name in names
        ], dtype=torch.bool))
        split_names.append(names)
        text_records[split] = {"primitive": primitive_record, "region": region_record}

    row_ids = torch.arange(num_rows)
    validation_rows = row_ids % int(args.holdout_stride) == int(args.holdout_residue)
    train_rows = ~validation_rows
    if min(int(train_rows.sum()), int(validation_rows.sum())) < 128:
        raise ValueError("query-native source-row split is too small")
    device = torch.device(args.device)
    split_sequence = [value.strip() for value in args.split_training_sequence.split(",") if value.strip()]
    if not split_sequence or any(value not in SPLITS for value in split_sequence):
        raise ValueError("split training sequence differs")
    torch.manual_seed(int(args.seed))
    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    model = QuerySetCategoricalDecoder(
        hidden_dim=int(args.hidden_dim), pair_hidden_dim=int(args.pair_hidden_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_objective, best_step = float("inf"), 0
    history: list[dict[str, Any]] = []
    train_indices = torch.where(train_rows)[0]
    validation_indices = torch.where(validation_rows)[0]
    for step in range(1, int(args.steps) + 1):
        split_index = SPLITS.index(split_sequence[(step - 1) % len(split_sequence)])
        local_eligible = eligible[split_index]
        changed = torch.where(train_rows & local_eligible)[0]
        replay = torch.where(train_rows & ~local_eligible)[0]
        half = min(int(args.batch_size) // 2, changed.numel(), replay.numel())
        sampled = torch.cat((
            changed[torch.randint(changed.numel(), (half,), generator=generator)],
            replay[torch.randint(replay.numel(), (half,), generator=generator)],
        ))
        prediction = model(
            latent[sampled].to(device), reliability[sampled].to(device),
            queries[split_index].to(device), baselines[split_index][sampled].to(device),
        )
        local_target = targets[split_index][sampled].to(device)
        coordinate = F.smooth_l1_loss(prediction, local_target, reduction="none")
        channel_mask = ~query_holdouts[split_index].to(device)
        coordinate = coordinate[:, channel_mask]
        row_weight = 1.0 + float(args.changed_row_weight) * local_eligible[sampled].float().to(device)
        loss = (coordinate.mean(1) * row_weight).mean()
        top2 = local_target[:, channel_mask].topk(k=2, dim=1).indices
        local_prediction = prediction[:, channel_mask]
        local_target = local_target[:, channel_mask]
        loss = loss + float(args.margin_weight) * F.smooth_l1_loss(
            local_prediction.gather(1, top2)[:, 0] - local_prediction.gather(1, top2)[:, 1],
            local_target.gather(1, top2)[:, 0] - local_target.gather(1, top2)[:, 1],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % int(args.validation_interval) != 0 and step != int(args.steps):
            continue
        validation: dict[str, Any] = {}
        objective = 0.0
        for index, split in enumerate(SPLITS):
            candidate = _decode_block(
                model, latent[validation_indices], reliability[validation_indices],
                queries[index], baselines[index][validation_indices], device,
                int(args.eval_chunk_size),
            )
            values = _metrics(
                candidate, targets[index][validation_indices],
                baselines[index][validation_indices], eligible[index][validation_indices],
                query_holdouts[index],
            )
            validation[split] = values
            objective += float(values["decoder_eligible_mae"])
        history.append({"step": step, "loss": float(loss.detach()), "objective": objective})
        if objective < best_objective:
            best_objective, best_step = objective, step
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)

    restored = [
        _decode_block(model, latent, reliability, queries[index], baselines[index], device,
                      int(args.eval_chunk_size))
        for index in range(len(SPLITS))
    ]
    gate = QuerySetEligibilityGate(hidden_dim=int(args.gate_hidden_dim)).to(device)
    gate_optimizer = torch.optim.AdamW(
        gate.parameters(), lr=float(args.gate_learning_rate), weight_decay=float(args.weight_decay)
    )
    positives = sum(int(value[train_rows].sum()) for value in eligible)
    total = int(train_rows.sum()) * len(SPLITS)
    pos_weight = torch.tensor((total - positives) / max(positives, 1), device=device)
    best_gate_state = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
    best_gate_loss = float("inf")
    for step in range(1, int(args.gate_steps) + 1):
        split_index = SPLITS.index(split_sequence[(step - 1) % len(split_sequence)])
        sampled = train_indices[torch.randint(
            train_indices.numel(), (min(int(args.batch_size), train_indices.numel()),),
            generator=generator,
        )]
        logits = gate(
            latent[sampled].to(device), reliability[sampled].to(device),
            queries[split_index].to(device), baselines[split_index][sampled].to(device),
        )
        gate_loss = F.binary_cross_entropy_with_logits(
            logits, eligible[split_index][sampled].float().to(device), pos_weight=pos_weight
        )
        gate_optimizer.zero_grad(set_to_none=True)
        gate_loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
        gate_optimizer.step()
        if step % int(args.validation_interval) == 0 or step == int(args.gate_steps):
            value = 0.0
            with torch.inference_mode():
                for index in range(len(SPLITS)):
                    logits = gate(
                        latent[validation_indices].to(device), reliability[validation_indices].to(device),
                        queries[index].to(device), baselines[index][validation_indices].to(device),
                    )
                    value += float(F.binary_cross_entropy_with_logits(
                        logits, eligible[index][validation_indices].float().to(device),
                        pos_weight=pos_weight,
                    ))
            if value < best_gate_loss:
                best_gate_loss = value
                best_gate_state = {key: tensor.detach().cpu().clone() for key, tensor in gate.state_dict().items()}
    gate.load_state_dict(best_gate_state)

    thresholds: list[float] = []
    gate_validation: dict[str, Any] = {}
    for index, split in enumerate(SPLITS):
        with torch.inference_mode():
            probability = torch.sigmoid(gate(
                latent[validation_indices].to(device), reliability[validation_indices].to(device),
                queries[index].to(device), baselines[index][validation_indices].to(device),
            )).cpu()
        candidates: list[tuple[float, float, float]] = []
        truth = eligible[index][validation_indices]
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1.01):
            selected = probability >= threshold
            candidate = torch.where(
                selected[:, None], restored[index][validation_indices],
                baselines[index][validation_indices],
            )
            replay = ~truth
            agreement = (
                float((candidate[replay].argmax(1) == baselines[index][validation_indices][replay].argmax(1)).float().mean())
                if bool(replay.any()) else 1.0
            )
            candidates.append((float((candidate - targets[index][validation_indices]).abs().mean()), threshold, agreement))
        safe = [value for value in candidates if value[2] >= float(args.minimum_replay_agreement)]
        mae, threshold, agreement = min(safe) if safe else min(candidates, key=lambda value: (-value[2], value[0]))
        thresholds.append(threshold)
        gate_validation[split] = {"threshold": threshold, "teacher_score_mae": mae,
                                  "replay_top1_agreement": agreement}
        probabilities: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, num_rows, int(args.eval_chunk_size)):
                stop = min(start + int(args.eval_chunk_size), num_rows)
                probabilities.append(torch.sigmoid(gate(
                    latent[start:stop].to(device), reliability[start:stop].to(device),
                    queries[index].to(device), baselines[index][start:stop].to(device),
                )).cpu())
        selected = torch.cat(probabilities) >= threshold
        restored[index] = torch.where(selected[:, None], restored[index], baselines[index])

    validation_metrics = {
        split: _metrics(
            restored[index][validation_indices], targets[index][validation_indices],
            baselines[index][validation_indices], eligible[index][validation_indices],
            query_holdouts[index],
        ) for index, split in enumerate(SPLITS)
    }
    noninferior = all(
        float(value["decoder_eligible_mae"]) <= float(value["baseline_eligible_mae"]) + 1e-8
        and float(value["replay_top1_agreement"]) >= float(args.minimum_replay_agreement)
        for value in validation_metrics.values()
    )
    strict = sum(float(value["decoder_eligible_mae"]) for value in validation_metrics.values()) < sum(
        float(value["baseline_eligible_mae"]) for value in validation_metrics.values()
    ) - 1e-8
    heldout_values = [value for value in validation_metrics.values() if value["heldout_query_count"]]
    heldout_noninferior = all(
        float(value["heldout_query_decoder_mae"]) <= float(value["heldout_query_baseline_mae"]) + 1e-8
        for value in heldout_values
    )
    passed = best_step > 0 and noninferior and strict and heldout_noninferior

    output_model = Path(args.output_model).expanduser().resolve()
    inputs = {
        "field": file_record(field_path), "universal_field": universal_record,
        "membership": membership_record, "proposal_teacher": teacher_record,
        "baseline_query_cache": baseline_record, "text_banks": text_records,
    }
    write_torch_noclobber(output_model, {
        "schema": "radio_gs.query_native_categorical_decoder.v1", "schema_version": 1,
        "scene": str(args.scene), "decoder_state_dict": best_state,
        "gate_state_dict": best_gate_state, "gate_thresholds": thresholds,
        "metadata": {
            "field_frozen": True, "per_gaussian_parameters_added": False,
            "class_indexed_parameters": False, "query_set_permutation_equivariant": True,
            "query_cardinality_dynamic": True, "teacher_features_decoded": False,
            "benchmark_labels_opened": False, "benchmark_masks_opened": False,
            "source_only": True, "inputs": inputs,
        },
    })
    output_cache = Path(args.output_score_cache).expanduser().resolve()
    write_torch_noclobber(output_cache, {
        "schema": "radio_gs.scannet_source_distilled_score_cache.v1", "schema_version": 1,
        "xyz": xyz, "valid": torch.ones(num_rows, dtype=torch.bool),
        "direct_observed": torch.stack(eligible).any(0),
        **{f"scores_split_{split}": restored[index].float().contiguous()
           for index, split in enumerate(SPLITS)},
        "metadata": {
            "artifact_type": "radio_gs_scannet_source_distilled_score_cache",
            "query_independent": False, "source_only": True,
            "evaluation_diagnostic_only": False, "benchmark_images_opened": False,
            "benchmark_masks_opened": False, "benchmark_labels_opened": False,
            "text_queries_opened": True, "postprocessing": "none",
            "source_gate_passed": passed, "query_native": True,
            "decoder": file_record(output_model), "inputs": inputs,
        },
    })
    report = {
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene), "best_step": best_step,
        "query_holdout": {
            "modulus": int(args.query_holdout_modulus),
            "residue": int(args.query_holdout_residue),
            "names": {split: [name for name, held in zip(split_names[index], query_holdouts[index]) if bool(held)]
                      for index, split in enumerate(SPLITS)},
        },
        "validation": validation_metrics, "gate_validation": gate_validation,
        "history": history, "model": file_record(output_model),
        "score_cache": file_record(output_cache),
    }
    write_frozen_json(output_cache.with_suffix(output_cache.suffix + ".json"), report)
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
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--pair-hidden-dim", type=int, default=48)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--changed-row-weight", type=float, default=3.0)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--split-training-sequence", default="19,15,10")
    parser.add_argument("--minimum-views", type=int, default=2)
    parser.add_argument("--minimum-view-agreement", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--holdout-stride", type=int, default=8)
    parser.add_argument("--holdout-residue", type=int, default=7)
    parser.add_argument("--query-holdout-modulus", type=int, default=0)
    parser.add_argument("--query-holdout-residue", type=int, default=0)
    parser.add_argument("--minimum-replay-agreement", type=float, default=0.995)
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--gate-steps", type=int, default=900)
    parser.add_argument("--gate-learning-rate", type=float, default=2e-3)
    parser.add_argument("--validation-interval", type=int, default=30)
    parser.add_argument("--eval-chunk-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260824)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
