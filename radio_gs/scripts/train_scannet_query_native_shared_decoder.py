#!/usr/bin/env python3
"""Cross-scene Query-Native categorical decoder over frozen L512 fields.

The source-distilled compact score caches are used as mapping-time teachers.
The student has one parameter set across every scene and no class-indexed or
per-Gaussian state.  A scene/query bipartite holdout tests adapter-free query
transfer without withholding a word from every training scene.
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
from radio_gs.scripts.train_scannet_frozen_l512_native_categorical_score_decoder import (
    _embedding,
    _load,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SPLITS = ("19", "15", "10")


def _pair_holdout(scene: str, query: str, modulus: int, residue: int) -> bool:
    if modulus <= 0:
        return False
    digest = hashlib.sha256(f"{scene}\0{query}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus == residue


def _trusted(path: Path, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    digest = sha256_file(path)
    return _load(str(path), digest, label)


def _load_scene(
    scene: str,
    root: Path,
    primitive_text: list[torch.Tensor],
    query_names: list[list[str]],
    holdout_modulus: int,
    holdout_residue: int,
) -> dict[str, Any]:
    core = root / "optimization_20260815/core_method_v1" / scene
    field_path = core / "generic_text_response_w005_s0_64.pth"
    universal_path = root / "optimization_20260816/universal_field_v1" / scene / "universal_field_v1.pth"
    baseline_path = root / "optimization_20260823/scannet_semantic_ladder/restored_direct_capability" / scene / "primitive_query_restored_direct_capability.pt"
    teacher_path = root / "optimization_20260824/native_multiteacher_v1" / scene / "native_categorical_score_l512_v10_fp32_predicted_gate/source_distilled_scores.pt"
    for path in (field_path, universal_path, baseline_path, teacher_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    field_sha = sha256_file(field_path)
    field, _payload, _signature = load_factorized_canonical_field_checkpoint(
        field_path, map_location="cpu", expected_sha256=field_sha
    )
    universal, universal_record = _trusted(universal_path, "Universal Field reliability authority")
    baseline_payload, baseline_record = _trusted(baseline_path, "restored categorical baseline")
    teacher, teacher_record = _trusted(teacher_path, "source-distilled categorical teacher")
    if universal.get("universal_field_migration", {}).get("source_field_sha256") != field_sha:
        raise ValueError(f"{scene}: reliability field binding differs")
    metadata = teacher.get("metadata", {})
    if (
        metadata.get("source_only") is not True
        or metadata.get("source_gate_passed") is not True
        or bool(metadata.get("benchmark_labels_opened", True))
        or bool(metadata.get("benchmark_masks_opened", True))
    ):
        raise ValueError(f"{scene}: shared teacher access contract differs")
    latent = field.local_codes.detach().cpu().float().contiguous()
    reliability = torch.as_tensor(universal.get("reliability")).float().contiguous()
    descriptor = F.normalize(torch.as_tensor(
        baseline_payload.get("summary_features", baseline_payload.get("features"))
    ).float(), dim=-1)
    xyz = torch.as_tensor(baseline_payload["xyz"]).float().contiguous()
    if latent.shape != (xyz.shape[0], 512) or reliability.shape != (xyz.shape[0], 5):
        raise ValueError(f"{scene}: shared query-native row domain differs")
    baseline: list[torch.Tensor] = []
    target: list[torch.Tensor] = []
    changed: list[torch.Tensor] = []
    query_holdout: list[torch.Tensor] = []
    for index, split in enumerate(SPLITS):
        local_baseline = descriptor @ primitive_text[index].T
        local_baseline = F.normalize(local_baseline - local_baseline.mean(1, keepdim=True), dim=-1)
        local_target = torch.as_tensor(teacher[f"scores_split_{split}"]).float().contiguous()
        if local_target.shape != local_baseline.shape:
            raise ValueError(f"{scene}: split{split} teacher shape differs")
        baseline.append(local_baseline.contiguous())
        target.append(local_target)
        changed.append((local_target - local_baseline).abs().max(1).values > 1e-7)
        query_holdout.append(torch.tensor([
            _pair_holdout(scene, name, holdout_modulus, holdout_residue)
            for name in query_names[index]
        ], dtype=torch.bool))
    del descriptor
    return {
        "scene": scene, "latent": latent, "reliability": reliability, "xyz": xyz,
        "baseline": baseline, "target": target, "changed": changed,
        "query_holdout": query_holdout,
        "inputs": {
            "field": file_record(field_path), "universal_field": universal_record,
            "baseline_query_cache": baseline_record, "teacher_score_cache": teacher_record,
        },
    }


@torch.inference_mode()
def _decode(
    model: QuerySetCategoricalDecoder,
    data: dict[str, Any],
    query: torch.Tensor,
    split_index: int,
    device: torch.device,
    chunk: int,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for start in range(0, data["latent"].shape[0], chunk):
        stop = min(start + chunk, data["latent"].shape[0])
        values.append(model(
            data["latent"][start:stop].to(device), data["reliability"][start:stop].to(device),
            query.to(device), data["baseline"][split_index][start:stop].to(device),
        ).float().cpu())
    return torch.cat(values)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).expanduser().resolve(strict=True)
    scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    if len(scenes) < 2 or len(set(scenes)) != len(scenes):
        raise ValueError("shared query-native training requires distinct scenes")
    primitive_paths = [Path(value).expanduser().resolve(strict=True) for value in args.primitive_text_banks.split(",")]
    primitive_hashes = args.expected_primitive_text_sha256.split(",")
    if len(primitive_paths) != 3 or len(primitive_hashes) != 3:
        raise ValueError("three primitive text banks are required")
    queries: list[torch.Tensor] = []
    query_names: list[list[str]] = []
    text_records: dict[str, Any] = {}
    for index, split in enumerate(SPLITS):
        ids = tuple(int(value) for value in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split])
        names = [NYU40_ID_TO_NAME[value] for value in ids]
        value, record = _embedding(str(primitive_paths[index]), primitive_hashes[index], names)
        queries.append(value.contiguous())
        query_names.append(names)
        text_records[split] = record
    datasets = [
        _load_scene(
            scene, root, queries, query_names,
            int(args.query_holdout_modulus), int(args.query_holdout_residue),
        ) for scene in scenes
    ]
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    model = QuerySetCategoricalDecoder(
        hidden_dim=int(args.hidden_dim), pair_hidden_dim=int(args.pair_hidden_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    for step in range(1, int(args.steps) + 1):
        scene_index = (step - 1) % len(datasets)
        split_index = ((step - 1) // len(datasets)) % len(SPLITS)
        data = datasets[scene_index]
        row_count = data["latent"].shape[0]
        validation = torch.arange(row_count) % int(args.holdout_stride) == int(args.holdout_residue)
        changed = data["changed"][split_index]
        positive = torch.where(~validation & changed)[0]
        negative = torch.where(~validation & ~changed)[0]
        if positive.numel() and negative.numel():
            half = min(int(args.batch_size) // 2, positive.numel(), negative.numel())
            sampled = torch.cat((
                positive[torch.randint(positive.numel(), (half,), generator=generator)],
                negative[torch.randint(negative.numel(), (half,), generator=generator)],
            ))
        else:
            pool = positive if positive.numel() else negative
            if not pool.numel():
                raise ValueError(f"{data['scene']}: split{SPLITS[split_index]} has no training row")
            sampled = pool[torch.randint(
                pool.numel(), (min(int(args.batch_size), pool.numel()),), generator=generator
            )]
        prediction = model(
            data["latent"][sampled].to(device), data["reliability"][sampled].to(device),
            queries[split_index].to(device), data["baseline"][split_index][sampled].to(device),
        )
        target = data["target"][split_index][sampled].to(device)
        query_mask = ~data["query_holdout"][split_index].to(device)
        coordinate = F.smooth_l1_loss(
            prediction[:, query_mask], target[:, query_mask], reduction="none"
        ).mean(1)
        row_weight = 1.0 + float(args.changed_row_weight) * changed[sampled].float().to(device)
        loss = (coordinate * row_weight).mean()
        top2 = target[:, query_mask].topk(k=2, dim=1).indices
        local_prediction, local_target = prediction[:, query_mask], target[:, query_mask]
        loss = loss + float(args.margin_weight) * F.smooth_l1_loss(
            local_prediction.gather(1, top2)[:, 0] - local_prediction.gather(1, top2)[:, 1],
            local_target.gather(1, top2)[:, 0] - local_target.gather(1, top2)[:, 1],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % int(args.validation_interval) == 0:
            history.append({"step": step, "loss": float(loss.detach())})
            if float(loss.detach()) < best_loss:
                best_loss = float(loss.detach())
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)

    gate = QuerySetEligibilityGate(hidden_dim=int(args.gate_hidden_dim)).to(device)
    gate_optimizer = torch.optim.AdamW(
        gate.parameters(), lr=float(args.gate_learning_rate), weight_decay=float(args.weight_decay)
    )
    best_gate_state = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
    best_gate_loss = float("inf")
    for step in range(1, int(args.gate_steps) + 1):
        scene_index = (step - 1) % len(datasets)
        split_index = ((step - 1) // len(datasets)) % len(SPLITS)
        data = datasets[scene_index]
        rows = torch.arange(data["latent"].shape[0])
        training = rows % int(args.holdout_stride) != int(args.holdout_residue)
        sampled_pool = rows[training]
        sampled = sampled_pool[torch.randint(
            sampled_pool.numel(), (min(int(args.batch_size), sampled_pool.numel()),),
            generator=generator,
        )]
        truth = data["changed"][split_index][sampled].float().to(device)
        positive = float(data["changed"][split_index][training].float().mean())
        pos_weight = torch.tensor((1.0 - positive) / max(positive, 1e-6), device=device)
        logits = gate(
            data["latent"][sampled].to(device), data["reliability"][sampled].to(device),
            queries[split_index].to(device), data["baseline"][split_index][sampled].to(device),
        )
        loss = F.binary_cross_entropy_with_logits(logits, truth, pos_weight=pos_weight)
        gate_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
        gate_optimizer.step()
        if step % int(args.validation_interval) == 0 and float(loss.detach()) < best_gate_loss:
            best_gate_loss = float(loss.detach())
            best_gate_state = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
    gate.load_state_dict(best_gate_state)

    decoded: dict[tuple[int, int], torch.Tensor] = {}
    validation_records: dict[str, Any] = {}
    threshold_candidates = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1.01)
    threshold_objective = {value: 0.0 for value in threshold_candidates}
    threshold_replay = {value: [] for value in threshold_candidates}
    for scene_index, data in enumerate(datasets):
        rows = torch.arange(data["latent"].shape[0])
        validation = rows % int(args.holdout_stride) == int(args.holdout_residue)
        for split_index, split in enumerate(SPLITS):
            prediction = _decode(
                model, data, queries[split_index], split_index, device, int(args.eval_chunk_size)
            )
            decoded[(scene_index, split_index)] = prediction
            with torch.inference_mode():
                probability_parts: list[torch.Tensor] = []
                for start in range(0, rows.numel(), int(args.eval_chunk_size)):
                    stop = min(start + int(args.eval_chunk_size), rows.numel())
                    probability_parts.append(torch.sigmoid(gate(
                        data["latent"][start:stop].to(device), data["reliability"][start:stop].to(device),
                        queries[split_index].to(device), data["baseline"][split_index][start:stop].to(device),
                    )).cpu())
            probability = torch.cat(probability_parts)
            for threshold in threshold_candidates:
                selected = probability[validation] >= threshold
                candidate = torch.where(
                    selected[:, None], prediction[validation], data["baseline"][split_index][validation]
                )
                target = data["target"][split_index][validation]
                threshold_objective[threshold] += float((candidate - target).abs().mean())
                replay = ~data["changed"][split_index][validation]
                if bool(replay.any()):
                    threshold_replay[threshold].append(float(
                        (candidate[replay].argmax(1) == data["baseline"][split_index][validation][replay].argmax(1)).float().mean()
                    ))
    safe = [value for value in threshold_candidates if min(threshold_replay[value], default=1.0) >= float(args.minimum_replay_agreement)]
    threshold = min(safe or threshold_candidates, key=lambda value: threshold_objective[value])

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "shared_decoder.pt"
    write_torch_noclobber(model_path, {
        "schema": "radio_gs.query_native_shared_categorical_decoder.v1", "schema_version": 1,
        "decoder_state_dict": best_state, "gate_state_dict": best_gate_state,
        "global_gate_threshold": threshold,
        "metadata": {
            "scenes": scenes, "field_frozen": True, "per_gaussian_parameters_added": False,
            "scene_specific_parameters": False, "class_indexed_parameters": False,
            "query_set_permutation_equivariant": True, "query_cardinality_dynamic": True,
            "teacher_features_decoded": False, "source_only": True,
            "benchmark_labels_opened": False, "benchmark_masks_opened": False,
            "text_banks": text_records,
        },
    })
    all_noninferior = True
    heldout_noninferior = True
    cache_payloads: dict[str, tuple[Path, dict[str, Any]]] = {}
    for scene_index, data in enumerate(datasets):
        scene_dir = output_dir / data["scene"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_validation: dict[str, Any] = {}
        score_values: dict[str, torch.Tensor] = {}
        rows = torch.arange(data["latent"].shape[0])
        validation = rows % int(args.holdout_stride) == int(args.holdout_residue)
        for split_index, split in enumerate(SPLITS):
            prediction = decoded[(scene_index, split_index)]
            with torch.inference_mode():
                probability_parts = []
                for start in range(0, rows.numel(), int(args.eval_chunk_size)):
                    stop = min(start + int(args.eval_chunk_size), rows.numel())
                    probability_parts.append(torch.sigmoid(gate(
                        data["latent"][start:stop].to(device), data["reliability"][start:stop].to(device),
                        queries[split_index].to(device), data["baseline"][split_index][start:stop].to(device),
                    )).cpu())
            selected = torch.cat(probability_parts) >= threshold
            candidate = torch.where(selected[:, None], prediction, data["baseline"][split_index])
            score_values[f"scores_split_{split}"] = candidate.float().contiguous()
            changed = data["changed"][split_index][validation]
            baseline_mae = float((data["baseline"][split_index][validation][changed] - data["target"][split_index][validation][changed]).abs().mean()) if bool(changed.any()) else 0.0
            decoder_mae = float((candidate[validation][changed] - data["target"][split_index][validation][changed]).abs().mean()) if bool(changed.any()) else 0.0
            holdout = data["query_holdout"][split_index]
            heldout_baseline = heldout_decoder = None
            if bool(holdout.any()) and bool(changed.any()):
                heldout_baseline = float((data["baseline"][split_index][validation][changed][:, holdout] - data["target"][split_index][validation][changed][:, holdout]).abs().mean())
                heldout_decoder = float((candidate[validation][changed][:, holdout] - data["target"][split_index][validation][changed][:, holdout]).abs().mean())
                heldout_noninferior &= heldout_decoder <= heldout_baseline + 1e-8
            all_noninferior &= decoder_mae <= baseline_mae + 1e-8
            scene_validation[split] = {
                "baseline_changed_mae": baseline_mae, "decoder_changed_mae": decoder_mae,
                "heldout_query_count": int(holdout.sum()),
                "heldout_query_baseline_mae": heldout_baseline,
                "heldout_query_decoder_mae": heldout_decoder,
            }
        cache_path = scene_dir / "source_distilled_scores.pt"
        cache_payloads[data["scene"]] = (cache_path, {
            "schema": "radio_gs.scannet_source_distilled_score_cache.v1", "schema_version": 1,
            "xyz": data["xyz"], "valid": torch.ones(rows.numel(), dtype=torch.bool),
            "direct_observed": torch.stack(data["changed"]).any(0), **score_values,
            "metadata": {
                "artifact_type": "radio_gs_scannet_source_distilled_score_cache",
                "query_independent": False, "source_only": True,
                "evaluation_diagnostic_only": False, "benchmark_images_opened": False,
                "benchmark_masks_opened": False, "benchmark_labels_opened": False,
                "text_queries_opened": True, "postprocessing": "none",
                "source_gate_passed": False, "query_native": True,
                "shared_cross_scene_decoder": file_record(model_path), "inputs": data["inputs"],
            },
        })
        validation_records[data["scene"]] = scene_validation
    passed = all_noninferior and heldout_noninferior
    for _scene, (cache_path, payload) in cache_payloads.items():
        payload["metadata"]["source_gate_passed"] = passed
        payload["metadata"]["evaluation_diagnostic_only"] = not passed
        write_torch_noclobber(cache_path, payload)
    report = {
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scenes": scenes, "global_gate_threshold": threshold,
        "query_pair_holdout": {"modulus": int(args.query_holdout_modulus),
                               "residue": int(args.query_holdout_residue)},
        "all_scene_split_noninferior": all_noninferior,
        "heldout_scene_query_noninferior": heldout_noninferior,
        "validation": validation_records, "history": history,
        "model": file_record(model_path),
    }
    write_frozen_json(output_dir / "source_gate.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output-root", default="/mnt/pool/sqy/results/RADIO-GS/output")
    parser.add_argument("--primitive-text-banks", required=True)
    parser.add_argument("--expected-primitive-text-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--pair-hidden-dim", type=int, default=48)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--gate-steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--gate-learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--changed-row-weight", type=float, default=3.0)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--holdout-stride", type=int, default=8)
    parser.add_argument("--holdout-residue", type=int, default=7)
    parser.add_argument("--query-holdout-modulus", type=int, default=0)
    parser.add_argument("--query-holdout-residue", type=int, default=0)
    parser.add_argument("--minimum-replay-agreement", type=float, default=0.995)
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--eval-chunk-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260824)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
