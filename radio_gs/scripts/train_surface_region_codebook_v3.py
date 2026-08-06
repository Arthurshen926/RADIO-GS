#!/usr/bin/env python3
"""Train the scene-disjoint multi-hypothesis SurfaceRegion V3 readout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryCodebookV3,
)
from radio_gs.losses.surface_region_codebook_loss import (
    balanced_latent_relation_loss,
    latent_query_responses,
    permutation_set_matching_loss,
    scene_listwise_and_hard_negative_loss,
    uniform_slot_prior_loss,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.train_surface_region_summary_readout import (
    _load as load_surface_caches,
    _paths as cache_paths,
    _seed_training,
)
from radio_gs.utils.immutable_artifacts import (
    load_surface_region_summary_readout_v2,
    load_sha_bound_project_checkpoint_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


def _load_text_bank(
    path: Path,
    *,
    expected_sha256: str,
    expected_split: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    payload, digest, source = load_sha_bound_project_checkpoint_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label=f"target-blind {expected_split} text bank",
    )
    embeddings = payload.get("embeddings")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "target_blind_text_embedding_cache"
        or payload.get("split") != expected_split
        or payload.get("benchmark_vocabulary_opened") is not False
        or payload.get("uses_benchmark_vocabulary_for_construction") is not False
        or not isinstance(embeddings, torch.Tensor)
        or tuple(embeddings.shape[1:]) != (1536,)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("target-blind text-bank contract differs")
    normalized = F.normalize(embeddings.float(), dim=-1, eps=1e-8)
    return normalized, {
        "path": str(source),
        "sha256": digest,
        "split": expected_split,
        "queries": int(normalized.shape[0]),
        "embedding_tensor_sha256": str(
            payload.get("embedding_tensor_sha256", "")
        ),
    }


def _multiview_targets(
    data: Mapping[str, object],
    rows: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.as_tensor(data["official_summary_tokens"])[rows].to(
        device=device, dtype=torch.float32
    )
    descriptors = F.normalize(
        torch.as_tensor(data["official_crop_summaries"])[rows].to(
            device=device, dtype=torch.float32
        ),
        dim=-1,
        eps=1e-8,
    )
    mask = torch.as_tensor(data["teacher_mask"])[rows].to(device).bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every region must retain an official teacher view")
    return tokens, descriptors, mask


def _canonical_targets(
    tokens: torch.Tensor,
    descriptors: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    similarity = torch.einsum("bvd,bwd->bvw", descriptors, descriptors)
    similarity = similarity.masked_fill(~mask[:, None, :], 0.0)
    medoid = similarity.sum(-1).masked_fill(~mask, -1e9).argmax(-1)
    batch = torch.arange(tokens.shape[0], device=tokens.device)
    target_token = tokens[batch, medoid]
    weights = mask.float() / mask.sum(dim=1, keepdim=True)
    target_descriptor = F.normalize(
        torch.einsum("bv,bvd->bd", weights, descriptors), dim=-1, eps=1e-8
    )
    return target_token, target_descriptor


def _complete_scene_batches(
    scene_ids: Sequence[str],
    *,
    target_rows: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    grouped: dict[str, list[int]] = {}
    for row, raw_scene in enumerate(scene_ids):
        grouped.setdefault(str(raw_scene), []).append(row)
    if not grouped or any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("every training scene needs at least two complete rows")
    order = torch.randperm(len(grouped), generator=generator).tolist()
    names = sorted(grouped)
    batches: list[torch.Tensor] = []
    pending: list[int] = []
    for index in order:
        pending.extend(grouped[names[index]])
        if len(pending) >= int(target_rows):
            batches.append(torch.tensor(pending, dtype=torch.long))
            pending = []
    if pending:
        batches.append(torch.tensor(pending, dtype=torch.long))
    return batches


def _gradient_norm(
    loss: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=False,
    )
    value = torch.stack(
        [gradient.detach().float().square().sum() for gradient in gradients]
    ).sum().sqrt()
    result = float(value.cpu())
    if not math.isfinite(result) or result <= 1e-12:
        raise ValueError("V3 gradient calibration is degenerate")
    return result


def _loss_terms(
    model: SurfaceRegionSummaryCodebookV3,
    head: SigLIP2SummaryHead,
    data: Mapping[str, object],
    rows: torch.Tensor,
    scene_ids: Sequence[str],
    text_bank: torch.Tensor,
    device: torch.device,
    *,
    token_weight: float,
    response_temperature: float,
) -> dict[str, torch.Tensor]:
    features = torch.as_tensor(data["radio_features"])[rows].to(
        device=device, dtype=torch.float32
    )
    geometry = torch.as_tensor(data["geometry"])[rows].to(
        device=device, dtype=torch.float32
    )
    token_mask = torch.as_tensor(data["token_mask"])[rows].to(device).bool()
    anchor = torch.as_tensor(data["anchor_index"])[rows].to(device).long()
    output = model.forward_codebook(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
    )
    predicted_descriptors = F.normalize(
        head(output.slot_tokens).float(), dim=-1, eps=1e-8
    )
    canonical_descriptor = F.normalize(
        head(output.canonical_token[:, None])[:, 0].float(), dim=-1, eps=1e-8
    )
    teacher_tokens, teacher_descriptors, teacher_mask = _multiview_targets(
        data, rows, device
    )
    target_token, target_descriptor = _canonical_targets(
        teacher_tokens, teacher_descriptors, teacher_mask
    )
    set_loss, _assignments = permutation_set_matching_loss(
        output.slot_tokens,
        predicted_descriptors,
        teacher_tokens,
        teacher_descriptors,
        teacher_mask,
        token_weight=float(token_weight),
    )
    canonical_loss = (
        1.0
        - F.cosine_similarity(
            output.canonical_token, target_token, dim=-1
        )
    ).mean() * float(token_weight) + (
        1.0
        - F.cosine_similarity(
            canonical_descriptor, target_descriptor, dim=-1
        )
    ).mean()
    main = 0.5 * (set_loss + canonical_loss)
    student_response = latent_query_responses(
        predicted_descriptors,
        text_bank,
        priors=output.slot_priors,
        temperature=float(response_temperature),
    )
    teacher_response = latent_query_responses(
        teacher_descriptors,
        text_bank,
        mask=teacher_mask,
        temperature=float(response_temperature),
    )
    independent = F.smooth_l1_loss(student_response, teacher_response)
    listwise, hard_negative = scene_listwise_and_hard_negative_loss(
        student_response, teacher_response, scene_ids
    )
    relation = balanced_latent_relation_loss(
        predicted_descriptors,
        teacher_descriptors,
        teacher_mask,
        scene_ids,
    )
    prior = uniform_slot_prior_loss(output.slot_priors)
    scene = (listwise + hard_negative + relation + prior) / 4.0
    return {
        "main": main,
        "set": set_loss,
        "canonical": canonical_loss,
        "independent": independent,
        "scene": scene,
        "listwise": listwise,
        "hard_negative": hard_negative,
        "relation_balanced": relation,
        "prior": prior,
    }


def _ranking_metrics(
    student: torch.Tensor,
    teacher: torch.Tensor,
    scene_ids: Sequence[str],
) -> dict[str, float]:
    spearman: list[torch.Tensor] = []
    overlap: list[torch.Tensor] = []
    top1: list[torch.Tensor] = []
    for scene in sorted(set(str(value) for value in scene_ids)):
        rows = torch.tensor(
            [index for index, value in enumerate(scene_ids) if str(value) == scene]
        )
        if rows.numel() < 2:
            continue
        predicted = student[rows]
        target = teacher[rows]
        predicted_rank = predicted.argsort(dim=0).argsort(dim=0).float()
        target_rank = target.argsort(dim=0).argsort(dim=0).float()
        predicted_rank -= predicted_rank.mean(dim=0)
        target_rank -= target_rank.mean(dim=0)
        correlation = (predicted_rank * target_rank).sum(dim=0) / (
            predicted_rank.square().sum(dim=0).sqrt()
            * target_rank.square().sum(dim=0).sqrt()
        ).clamp_min(1e-8)
        spearman.append(correlation)
        keep = max(1, int(math.ceil(0.1 * len(rows))))
        predicted_top = predicted.topk(keep, dim=0).indices
        target_top = target.topk(keep, dim=0).indices
        overlap.append(
            (predicted_top[:, None, :] == target_top[None, :, :])
            .any(dim=1)
            .float()
            .mean(dim=0)
        )
        top1.append((predicted.argmax(0) == target.argmax(0)).float())
    if not spearman:
        raise ValueError("validation needs complete multi-row scenes")
    s = torch.cat(spearman)
    o = torch.cat(overlap)
    t = torch.cat(top1)
    return {
        "ranking_spearman_mean": float(s.mean()),
        "ranking_spearman_p05": float(torch.quantile(s, 0.05)),
        "top_decile_overlap_mean": float(o.mean()),
        "top_decile_overlap_p05": float(torch.quantile(o, 0.05)),
        "support_top1_agreement": float(t.mean()),
    }


@torch.no_grad()
def _evaluate(
    model: SurfaceRegionSummaryCodebookV3,
    head: SigLIP2SummaryHead,
    data: Mapping[str, object],
    text_bank: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    response_temperature: float,
) -> dict[str, float | list[float]]:
    model.eval()
    slot_descriptors: list[torch.Tensor] = []
    canonical_descriptors: list[torch.Tensor] = []
    canonical_tokens: list[torch.Tensor] = []
    slot_priors: list[torch.Tensor] = []
    for start in range(0, len(torch.as_tensor(data["radio_features"])), batch_size):
        rows = torch.arange(
            start,
            min(start + batch_size, len(torch.as_tensor(data["radio_features"]))),
        )
        output = model.forward_codebook(
            torch.as_tensor(data["radio_features"])[rows].to(device).float(),
            torch.as_tensor(data["geometry"])[rows].to(device).float(),
            anchor_index=torch.as_tensor(data["anchor_index"])[rows].to(device),
            token_mask=torch.as_tensor(data["token_mask"])[rows].to(device),
        )
        slot_descriptors.append(
            F.normalize(head(output.slot_tokens).float(), dim=-1).cpu()
        )
        canonical_descriptors.append(
            F.normalize(
                head(output.canonical_token[:, None])[:, 0].float(), dim=-1
            ).cpu()
        )
        canonical_tokens.append(output.canonical_token.cpu())
        slot_priors.append(output.slot_priors.cpu())
    predicted = torch.cat(slot_descriptors)
    canonical_descriptor = torch.cat(canonical_descriptors)
    canonical_token = torch.cat(canonical_tokens)
    priors = torch.cat(slot_priors)
    rows = torch.arange(len(predicted))
    teacher_tokens, teacher_descriptors, teacher_mask = _multiview_targets(
        data, rows, torch.device("cpu")
    )
    target_token, target_descriptor = _canonical_targets(
        teacher_tokens, teacher_descriptors, teacher_mask
    )
    teacher_to_slot = torch.einsum(
        "bvd,bkd->bvk", teacher_descriptors, predicted
    ).amax(dim=-1)
    student_response = latent_query_responses(
        predicted,
        text_bank,
        priors=priors,
        temperature=response_temperature,
    )
    teacher_response = latent_query_responses(
        teacher_descriptors,
        text_bank,
        mask=teacher_mask,
        temperature=response_temperature,
    )
    valid_cosine = teacher_to_slot[teacher_mask]
    response_profile = F.cosine_similarity(
        student_response, teacher_response, dim=-1
    )
    effective_slots = torch.exp(
        -(priors * priors.clamp_min(1e-12).log()).sum(-1)
    )
    slot_argmax = torch.einsum("bkd,qd->bkq", predicted, text_bank).argmax(dim=1)
    usage = torch.bincount(slot_argmax.flatten(), minlength=predicted.shape[1]).float()
    usage /= usage.sum().clamp_min(1)
    metrics: dict[str, float | list[float]] = {
        "canonical_summary_token_cosine": float(
            F.cosine_similarity(canonical_token, target_token, dim=-1).mean()
        ),
        "canonical_descriptor_cosine": float(
            F.cosine_similarity(
                canonical_descriptor, target_descriptor, dim=-1
            ).mean()
        ),
        "teacher_best_slot_cosine_mean": float(valid_cosine.mean()),
        "teacher_best_slot_cosine_p05": float(torch.quantile(valid_cosine, 0.05)),
        "text_response_smooth_l1": float(
            F.smooth_l1_loss(student_response, teacher_response)
        ),
        "response_profile_cosine_mean": float(response_profile.mean()),
        "response_profile_cosine_p05": float(
            torch.quantile(response_profile, 0.05)
        ),
        "effective_slots_mean": float(effective_slots.mean()),
        "query_slot_usage": [float(value) for value in usage],
    }
    metrics.update(
        _ranking_metrics(
            student_response,
            teacher_response,
            list(data["scene_ids"]),
        )
    )
    return metrics


@torch.no_grad()
def _evaluate_v2_control(
    model: torch.nn.Module,
    head: SigLIP2SummaryHead,
    data: Mapping[str, object],
    text_bank: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    response_temperature: float,
) -> dict[str, float | list[float]]:
    model.eval()
    tokens: list[torch.Tensor] = []
    descriptors: list[torch.Tensor] = []
    for start in range(0, len(torch.as_tensor(data["radio_features"])), batch_size):
        rows = torch.arange(
            start,
            min(start + batch_size, len(torch.as_tensor(data["radio_features"]))),
        )
        predicted = model(
            torch.as_tensor(data["radio_features"])[rows].to(device).float(),
            torch.as_tensor(data["geometry"])[rows].to(device).float(),
            anchor_index=torch.as_tensor(data["anchor_index"])[rows].to(device),
            token_mask=torch.as_tensor(data["token_mask"])[rows].to(device),
            reliability=torch.as_tensor(data["reliability"])[rows].to(device),
        )
        tokens.append(predicted.cpu())
        descriptors.append(
            F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1).cpu()
        )
    canonical_token = torch.cat(tokens)
    canonical_descriptor = torch.cat(descriptors)
    rows = torch.arange(len(canonical_token))
    teacher_tokens, teacher_descriptors, teacher_mask = _multiview_targets(
        data, rows, torch.device("cpu")
    )
    target_token, target_descriptor = _canonical_targets(
        teacher_tokens, teacher_descriptors, teacher_mask
    )
    predicted = canonical_descriptor[:, None, :]
    teacher_to_slot = torch.einsum(
        "bvd,bkd->bvk", teacher_descriptors, predicted
    ).amax(dim=-1)
    student_response = latent_query_responses(
        predicted,
        text_bank,
        priors=torch.ones(len(predicted), 1),
        temperature=response_temperature,
    )
    teacher_response = latent_query_responses(
        teacher_descriptors,
        text_bank,
        mask=teacher_mask,
        temperature=response_temperature,
    )
    valid_cosine = teacher_to_slot[teacher_mask]
    response_profile = F.cosine_similarity(
        student_response, teacher_response, dim=-1
    )
    metrics: dict[str, float | list[float]] = {
        "canonical_summary_token_cosine": float(
            F.cosine_similarity(canonical_token, target_token, dim=-1).mean()
        ),
        "canonical_descriptor_cosine": float(
            F.cosine_similarity(
                canonical_descriptor, target_descriptor, dim=-1
            ).mean()
        ),
        "teacher_best_slot_cosine_mean": float(valid_cosine.mean()),
        "teacher_best_slot_cosine_p05": float(torch.quantile(valid_cosine, 0.05)),
        "text_response_smooth_l1": float(
            F.smooth_l1_loss(student_response, teacher_response)
        ),
        "response_profile_cosine_mean": float(response_profile.mean()),
        "response_profile_cosine_p05": float(
            torch.quantile(response_profile, 0.05)
        ),
        "effective_slots_mean": 1.0,
        "query_slot_usage": [1.0],
    }
    metrics.update(
        _ranking_metrics(
            student_response,
            teacher_response,
            list(data["scene_ids"]),
        )
    )
    return metrics


def train(args: argparse.Namespace) -> dict[str, object]:
    train_paths = cache_paths(args.train_caches)
    validation_paths = cache_paths(args.validation_caches)
    train_data, train_meta = load_surface_caches(train_paths, "train")
    validation_data, validation_meta = load_surface_caches(
        validation_paths, "validation"
    )
    overlap = set(train_meta["scenes"]) & set(validation_meta["scenes"])
    if overlap:
        raise ValueError(f"V3 train/validation scene leakage: {sorted(overlap)}")
    for field in (
        "region_contract_sha256",
        "region_contract",
        "teacher_region",
        "radio_checkpoint_sha256",
        "excluded_physical_spaces",
    ):
        if train_meta[field] != validation_meta[field]:
            raise ValueError(f"V3 train/validation {field} differs")
    if "scene_ids" not in train_data or "scene_ids" not in validation_data:
        raise ValueError("V3 requires exact cache row-to-scene bindings")
    radio_path = Path(args.radio_checkpoint)
    if sha256_file(radio_path) != train_meta["radio_checkpoint_sha256"]:
        raise ValueError("V3 RADIO checkpoint differs from cache provenance")
    fit_text, fit_record = _load_text_bank(
        Path(args.fit_text_bank),
        expected_sha256=args.fit_text_bank_sha256,
        expected_split="fit",
    )
    validation_text, validation_text_record = _load_text_bank(
        Path(args.validation_text_bank),
        expected_sha256=args.validation_text_bank_sha256,
        expected_split="dev",
    )
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"V3 output already exists: {output}")
    device = torch.device(args.device)
    generator = _seed_training(int(args.seed), device=device)
    model = SurfaceRegionSummaryCodebookV3(
        hidden_dim=int(args.hidden_dim), slots=int(args.slots)
    ).to(device)
    head = SigLIP2SummaryHead.from_radio_checkpoint(radio_path).to(device).eval()
    head.requires_grad_(False)
    control, control_payload, control_sha256, control_path = (
        load_surface_region_summary_readout_v2(
            args.control_readout,
            expected_sha256=args.control_readout_sha256,
            map_location=device,
        )
    )
    if (
        control_payload.get("provenance", {}).get("uses_benchmark_scenes")
        is not False
        or control_payload.get("provenance", {}).get(
            "uses_benchmark_test_vocabulary"
        )
        is not False
    ):
        raise ValueError("V3 control readout provenance is benchmark contaminated")
    control = control.to(device)
    fit_text = fit_text.to(device)
    validation_text = validation_text.cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    initial = _evaluate(
        model,
        head,
        validation_data,
        validation_text,
        device,
        batch_size=int(args.eval_batch_size),
        response_temperature=float(args.response_temperature),
    )
    control_metrics = _evaluate_v2_control(
        control,
        head,
        validation_data,
        validation_text,
        device,
        batch_size=int(args.eval_batch_size),
        response_temperature=float(args.response_temperature),
    )
    print(
        json.dumps(
            {"untrained": initial, "frozen_v2_control": control_metrics},
            sort_keys=True,
        ),
        flush=True,
    )
    calibration: dict[str, float] | None = None
    history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_key: tuple[float, ...] | None = None
    stale = 0
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        batches = _complete_scene_batches(
            list(train_data["scene_ids"]),
            target_rows=int(args.batch_size),
            generator=generator,
        )
        epoch_losses: list[float] = []
        component_sums: dict[str, float] = {}
        for rows in batches:
            scenes = [str(train_data["scene_ids"][int(row)]) for row in rows]
            terms = _loss_terms(
                model,
                head,
                train_data,
                rows,
                scenes,
                fit_text,
                device,
                token_weight=float(args.token_weight),
                response_temperature=float(args.response_temperature),
            )
            if epoch <= int(args.warmup_epochs):
                loss = terms["main"]
            else:
                if calibration is None:
                    main_norm = _gradient_norm(
                        terms["main"], parameters, retain_graph=True
                    )
                    independent_norm = _gradient_norm(
                        terms["independent"], parameters, retain_graph=True
                    )
                    scene_norm = _gradient_norm(
                        terms["scene"], parameters, retain_graph=True
                    )
                    ratio = float(args.auxiliary_gradient_ratio)
                    calibration = {
                        "main_gradient_l2": main_norm,
                        "independent_gradient_l2": independent_norm,
                        "scene_gradient_l2": scene_norm,
                        "auxiliary_gradient_ratio_each": ratio,
                        "independent_lambda": ratio * main_norm / independent_norm,
                        "scene_lambda": ratio * main_norm / scene_norm,
                    }
                    print(json.dumps({"gradient_calibration": calibration}), flush=True)
                loss = (
                    terms["main"]
                    + calibration["independent_lambda"] * terms["independent"]
                    + calibration["scene_lambda"] * terms["scene"]
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            for name, value in terms.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(
                    value.detach().cpu()
                )
        metrics = _evaluate(
            model,
            head,
            validation_data,
            validation_text,
            device,
            batch_size=int(args.eval_batch_size),
            response_temperature=float(args.response_temperature),
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "loss": sum(epoch_losses) / len(epoch_losses),
            "train_components": {
                name: value / len(batches) for name, value in component_sums.items()
            },
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        # Scene-disjoint and query-disjoint lexicographic selection avoids a
        # hand-tuned scalar combination of ranking, tail, and set fidelity.
        key = (
            float(metrics["support_top1_agreement"]),
            float(metrics["ranking_spearman_p05"]),
            float(metrics["response_profile_cosine_p05"]),
            float(metrics["teacher_best_slot_cosine_mean"]),
            float(metrics["canonical_descriptor_cosine"]),
            -float(metrics["text_response_smooth_l1"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if int(args.patience) > 0 and stale >= int(args.patience):
            break
    if best_state is None or calibration is None:
        raise RuntimeError("V3 training did not reach its calibrated stage")
    model.load_state_dict(best_state)
    final = _evaluate(
        model,
        head,
        validation_data,
        validation_text,
        device,
        batch_size=int(args.eval_batch_size),
        response_temperature=float(args.response_temperature),
    )
    architecture = model.architecture(train_meta["region_contract_sha256"])
    provenance = {
        "training_scope": "global_cross_scene_surface_region_codebook_v3",
        "frozen": True,
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "scene_disjoint": True,
        "query_split_disjoint": True,
        "train": train_meta,
        "validation": validation_meta,
        "fit_text_bank": fit_record,
        "validation_text_bank": validation_text_record,
        "control_readout": {
            "path": str(control_path),
            "sha256": control_sha256,
        },
        "region_contract": train_meta["region_contract"],
        "region_contract_sha256": train_meta["region_contract_sha256"],
        "official_summary_head": "c-radio_v4 siglip2-g",
        "custom_text_projection": False,
        "semantic_boundary_proxy": "balanced_teacher_set_relation_terciles",
        "physical_sam_boundary_claimed": False,
        "feature_gauge": "l2_direction_inside_model",
    }
    payload = {
        "schema_version": 4,
        "architecture": architecture,
        "state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
        "provenance": provenance,
        "history": history,
        "best_epoch": best_epoch,
        "best_selection_key": list(best_key or ()),
        "untrained_baseline": initial,
        "frozen_v2_control": control_metrics,
        "validation": final,
        "gradient_calibration": calibration,
        "training_config": {
            "seed": int(args.seed),
            "hidden_dim": int(args.hidden_dim),
            "slots": int(args.slots),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "warmup_epochs": int(args.warmup_epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "token_weight": float(args.token_weight),
            "response_temperature": float(args.response_temperature),
            "auxiliary_gradient_ratio": float(args.auxiliary_gradient_ratio),
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema_version": 1,
        "status": "complete",
        "output": str(output),
        "checkpoint_sha256": sha256_file(output),
        "best_epoch": best_epoch,
        "validation": final,
        "untrained_baseline": initial,
        "frozen_v2_control": control_metrics,
        "gradient_calibration": calibration,
        "train_scenes": len(train_meta["scenes"]),
        "validation_scenes": len(validation_meta["scenes"]),
        "scene_overlap": [],
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--fit-text-bank", required=True)
    parser.add_argument("--fit-text-bank-sha256", required=True)
    parser.add_argument("--validation-text-bank", required=True)
    parser.add_argument("--validation-text-bank-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-readout", required=True)
    parser.add_argument("--control-readout-sha256", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--response-temperature", type=float, default=0.05)
    parser.add_argument("--auxiliary-gradient-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
