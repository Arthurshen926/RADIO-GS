#!/usr/bin/env python3
"""Train the frozen-V2 residual SurfaceRegion codebook on disjoint scenes."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryResidualCodebookV1,
)
from radio_gs.losses.surface_region_codebook_loss import (
    balanced_latent_relation_loss,
    gauge_aware_permutation_set_matching_loss,
    latent_query_max_responses,
    scene_listwise_and_hard_negative_loss,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.train_surface_region_codebook_v3 import (
    _canonical_targets,
    _complete_scene_batches,
    _gradient_norm,
    _load_text_bank,
    _multiview_targets,
    _ranking_metrics,
)
from radio_gs.scripts.train_surface_region_summary_readout import (
    _load as load_surface_caches,
    _paths as cache_paths,
    _seed_training,
)
from radio_gs.utils.immutable_artifacts import (
    load_surface_region_summary_readout_v2,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


def _assert_checkpoint_training_contract(
    payload: Mapping[str, object],
    *,
    expected_contract_sha256: str,
    expected_radio_sha256: str,
    label: str,
) -> None:
    """Fail closed when a frozen model was trained under another contract."""

    architecture = payload.get("architecture")
    provenance = payload.get("provenance")
    if not isinstance(architecture, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError(f"{label} lacks architecture/provenance bindings")
    train = provenance.get("train")
    validation = provenance.get("validation")
    if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
        raise ValueError(f"{label} lacks train/validation provenance bindings")
    contract_bindings = (
        architecture.get("contract_sha256"),
        provenance.get("region_contract_sha256"),
        train.get("region_contract_sha256"),
        validation.get("region_contract_sha256"),
    )
    if any(str(value) != expected_contract_sha256 for value in contract_bindings):
        raise ValueError(f"{label} region contract differs from the input caches")
    radio_bindings = (
        train.get("radio_checkpoint_sha256"),
        validation.get("radio_checkpoint_sha256"),
    )
    if any(str(value) != expected_radio_sha256 for value in radio_bindings):
        raise ValueError(f"{label} RADIO checkpoint differs from the input caches")


def _head_descriptors(
    head: SigLIP2SummaryHead,
    canonical: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project fallback separately so its GEMM path matches the V2 control."""

    # Every slot uses the identical [B,1,D] matrix path.  With duplicate
    # tokens, changing N from one to three can otherwise introduce tiny GEMM
    # differences that a hard maximum amplifies into a different rank order.
    projected = [head(canonical.contiguous()[:, None])[:, 0].float()]
    projected.extend(
        head(residual[:, slot].contiguous()[:, None])[:, 0].float()
        for slot in range(residual.shape[1])
    )
    descriptors = F.normalize(torch.stack(projected, dim=1), dim=-1, eps=1e-8)
    return descriptors[:, 0], descriptors


def _padded_teacher_set(
    teacher_tokens: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    fallback_token: torch.Tensor,
    fallback_descriptor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bind otherwise unmatched learned slots to the exact frozen fallback."""

    if teacher_tokens.shape[1] != 3:
        raise ValueError("residual codebook authority must expose three teacher columns")
    padded_tokens = torch.where(
        teacher_mask[..., None],
        teacher_tokens,
        fallback_token[:, None, :],
    )
    padded_descriptors = torch.where(
        teacher_mask[..., None],
        teacher_descriptors,
        fallback_descriptor[:, None, :],
    )
    return (
        padded_tokens,
        padded_descriptors,
        torch.ones_like(teacher_mask),
    )


def _loss_terms(
    model: SurfaceRegionSummaryResidualCodebookV1,
    head: SigLIP2SummaryHead,
    data: Mapping[str, object],
    rows: torch.Tensor,
    scene_ids: Sequence[str],
    text_bank: torch.Tensor,
    device: torch.device,
    *,
    token_direction_weight: float,
    token_log_norm_weight: float,
) -> dict[str, torch.Tensor]:
    features = torch.as_tensor(data["radio_features"])[rows].to(
        device=device, dtype=torch.float32
    )
    geometry = torch.as_tensor(data["geometry"])[rows].to(
        device=device, dtype=torch.float32
    )
    token_mask = torch.as_tensor(data["token_mask"])[rows].to(device).bool()
    anchor = torch.as_tensor(data["anchor_index"])[rows].to(device).long()
    reliability = torch.as_tensor(data["reliability"])[rows].to(
        device=device, dtype=torch.float32
    )
    output = model.forward_codebook(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
        reliability=reliability,
    )
    fallback_descriptor, predicted_descriptors = _head_descriptors(
        head, output.canonical_token, output.slot_tokens[:, 1:]
    )
    teacher_tokens, teacher_descriptors, teacher_mask = _multiview_targets(
        data, rows, device
    )
    padded_tokens, padded_descriptors, padded_mask = _padded_teacher_set(
        teacher_tokens,
        teacher_descriptors,
        teacher_mask,
        output.canonical_token,
        fallback_descriptor,
    )
    set_loss, _assignments = gauge_aware_permutation_set_matching_loss(
        output.slot_tokens[:, 1:],
        predicted_descriptors[:, 1:],
        padded_tokens,
        padded_descriptors,
        padded_mask,
        token_direction_weight=float(token_direction_weight),
        token_log_norm_weight=float(token_log_norm_weight),
    )
    student_response = latent_query_max_responses(
        predicted_descriptors, text_bank
    )
    teacher_response = latent_query_max_responses(
        teacher_descriptors, text_bank, mask=teacher_mask
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
    return {
        "main": set_loss,
        "set_gauge_aware": set_loss,
        "independent": independent,
        "listwise": listwise,
        "hard_negative": hard_negative,
        "relation_balanced": relation,
    }


@torch.no_grad()
def _evaluate(
    model: SurfaceRegionSummaryResidualCodebookV1,
    head: SigLIP2SummaryHead,
    data: Mapping[str, object],
    text_bank: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, float | list[float] | bool]:
    model.eval()
    slot_descriptors: list[torch.Tensor] = []
    canonical_descriptors: list[torch.Tensor] = []
    canonical_tokens: list[torch.Tensor] = []
    exact_canonical = True
    for start in range(0, len(torch.as_tensor(data["radio_features"])), batch_size):
        rows = torch.arange(
            start,
            min(start + batch_size, len(torch.as_tensor(data["radio_features"]))),
        )
        features = torch.as_tensor(data["radio_features"])[rows].to(device).float()
        geometry = torch.as_tensor(data["geometry"])[rows].to(device).float()
        anchor = torch.as_tensor(data["anchor_index"])[rows].to(device)
        token_mask = torch.as_tensor(data["token_mask"])[rows].to(device)
        reliability = torch.as_tensor(data["reliability"])[rows].to(device).float()
        output = model.forward_codebook(
            features,
            geometry,
            anchor_index=anchor,
            token_mask=token_mask,
            reliability=reliability,
        )
        direct = model.base(
            features,
            geometry,
            anchor_index=anchor,
            token_mask=token_mask,
            reliability=reliability,
        )
        exact_canonical = exact_canonical and torch.equal(
            output.canonical_token, direct
        ) and torch.equal(output.slot_tokens[:, 0], direct)
        fallback, descriptors = _head_descriptors(
            head, output.canonical_token, output.slot_tokens[:, 1:]
        )
        slot_descriptors.append(descriptors.cpu())
        canonical_descriptors.append(fallback.cpu())
        canonical_tokens.append(output.canonical_token.cpu())
    predicted = torch.cat(slot_descriptors)
    canonical_descriptor = torch.cat(canonical_descriptors)
    canonical_token = torch.cat(canonical_tokens)
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
    fallback_teacher = torch.einsum(
        "bvd,bd->bv", teacher_descriptors, canonical_descriptor
    )
    elementwise_gain = (teacher_to_slot - fallback_teacher)[teacher_mask]
    student_response = latent_query_max_responses(predicted, text_bank)
    teacher_response = latent_query_max_responses(
        teacher_descriptors, text_bank, mask=teacher_mask
    )
    valid_cosine = teacher_to_slot[teacher_mask]
    response_profile = F.cosine_similarity(
        student_response, teacher_response, dim=-1
    )
    slot_argmax = torch.einsum("bkd,qd->bkq", predicted, text_bank).argmax(dim=1)
    usage = torch.bincount(
        slot_argmax.flatten(), minlength=predicted.shape[1]
    ).float()
    usage /= usage.sum().clamp_min(1)
    slot_pair = torch.einsum("bkd,bld->bkl", predicted, predicted)
    off_diagonal = ~torch.eye(
        predicted.shape[1], dtype=torch.bool
    )[None].expand(len(predicted), -1, -1)
    pair_values = slot_pair[off_diagonal]
    metrics: dict[str, float | list[float] | bool] = {
        "canonical_fallback_bitwise_equal": bool(exact_canonical),
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
        "teacher_best_slot_elementwise_gain_min": float(elementwise_gain.min()),
        "text_response_smooth_l1": float(
            F.smooth_l1_loss(student_response, teacher_response)
        ),
        "response_profile_cosine_mean": float(response_profile.mean()),
        "response_profile_cosine_p05": float(
            torch.quantile(response_profile, 0.05)
        ),
        "query_slot_usage": [float(value) for value in usage],
        "learned_query_slot_usage_min": float(usage[1:].min()),
        "slot_descriptor_pair_cosine_mean": float(pair_values.mean()),
        "slot_descriptor_pair_cosine_p05": float(
            torch.quantile(pair_values, 0.05)
        ),
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
def _evaluate_v2_control_max(
    model: torch.nn.Module,
    head: SigLIP2SummaryHead,
    data: Mapping[str, object],
    text_bank: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, float | list[float]]:
    """Evaluate V2 against the identical hard-max multiview authority."""

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
            F.normalize(
                head(predicted.contiguous()[:, None])[:, 0].float(),
                dim=-1,
                eps=1e-8,
            ).cpu()
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
    student_response = latent_query_max_responses(predicted, text_bank)
    teacher_response = latent_query_max_responses(
        teacher_descriptors, text_bank, mask=teacher_mask
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


def _generic_gate(
    candidate: Mapping[str, object],
    control: Mapping[str, object],
) -> dict[str, object]:
    checks = {
        "canonical_fallback_bitwise_equal": bool(
            candidate["canonical_fallback_bitwise_equal"]
        ),
        "teacher_best_slot_elementwise_nondecreasing": float(
            candidate["teacher_best_slot_elementwise_gain_min"]
        ) >= -1e-7,
        "teacher_best_slot_gain": float(
            candidate["teacher_best_slot_cosine_mean"]
        ) - float(control["teacher_best_slot_cosine_mean"]) >= 0.005,
        "response_error_strictly_lower": float(
            candidate["text_response_smooth_l1"]
        ) < float(control["text_response_smooth_l1"]),
        "support_top1_nondecreasing": float(
            candidate["support_top1_agreement"]
        ) >= float(control["support_top1_agreement"]),
        "ranking_p05_noninferior": float(candidate["ranking_spearman_p05"])
        - float(control["ranking_spearman_p05"]) >= -0.002,
        "profile_p05_noninferior": float(
            candidate["response_profile_cosine_p05"]
        ) - float(control["response_profile_cosine_p05"]) >= -0.002,
        "learned_slot_usage": float(candidate["learned_query_slot_usage_min"])
        >= 0.02,
    }
    return {
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "failed": sum(not bool(value) for value in checks.values()),
        "overall_pass": all(checks.values()),
        "deltas": {
            "teacher_best_slot_cosine_mean": float(
                candidate["teacher_best_slot_cosine_mean"]
            ) - float(control["teacher_best_slot_cosine_mean"]),
            "text_response_smooth_l1": float(
                candidate["text_response_smooth_l1"]
            ) - float(control["text_response_smooth_l1"]),
            "support_top1_agreement": float(
                candidate["support_top1_agreement"]
            ) - float(control["support_top1_agreement"]),
            "ranking_spearman_p05": float(candidate["ranking_spearman_p05"])
            - float(control["ranking_spearman_p05"]),
            "response_profile_cosine_p05": float(
                candidate["response_profile_cosine_p05"]
            ) - float(control["response_profile_cosine_p05"]),
        },
    }


def _selection_feasible(
    candidate: Mapping[str, object],
    control: Mapping[str, object],
) -> bool:
    return (
        float(candidate["support_top1_agreement"])
        >= float(control["support_top1_agreement"])
        and float(candidate["ranking_spearman_p05"])
        - float(control["ranking_spearman_p05"])
        >= -0.002
        and float(candidate["response_profile_cosine_p05"])
        - float(control["response_profile_cosine_p05"])
        >= -0.002
        and float(candidate["text_response_smooth_l1"])
        < float(control["text_response_smooth_l1"])
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    train_paths = cache_paths(args.train_caches)
    validation_paths = cache_paths(args.validation_caches)
    train_data, train_meta = load_surface_caches(train_paths, "train")
    validation_data, validation_meta = load_surface_caches(
        validation_paths, "validation"
    )
    overlap = set(train_meta["scenes"]) & set(validation_meta["scenes"])
    if overlap:
        raise ValueError(f"residual codebook scene leakage: {sorted(overlap)}")
    for field in (
        "region_contract_sha256",
        "region_contract",
        "teacher_region",
        "radio_checkpoint_sha256",
        "excluded_physical_spaces",
    ):
        if train_meta[field] != validation_meta[field]:
            raise ValueError(f"train/validation {field} differs")
    if "scene_ids" not in train_data or "scene_ids" not in validation_data:
        raise ValueError("residual codebook requires row-to-scene bindings")
    radio_path = Path(args.radio_checkpoint)
    if sha256_file(radio_path) != train_meta["radio_checkpoint_sha256"]:
        raise ValueError("RADIO checkpoint differs from cache provenance")
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
        raise FileExistsError(f"output already exists: {output}")
    device = torch.device(args.device)
    generator = _seed_training(int(args.seed), device=device)
    control, control_payload, control_sha256, control_path = (
        load_surface_region_summary_readout_v2(
            args.control_readout,
            expected_sha256=args.control_readout_sha256,
            map_location=device,
        )
    )
    _assert_checkpoint_training_contract(
        control_payload,
        expected_contract_sha256=str(train_meta["region_contract_sha256"]),
        expected_radio_sha256=str(train_meta["radio_checkpoint_sha256"]),
        label="control readout",
    )
    if (
        control_payload.get("provenance", {}).get("uses_benchmark_scenes")
        is not False
        or control_payload.get("provenance", {}).get(
            "uses_benchmark_test_vocabulary"
        )
        is not False
    ):
        raise ValueError("control readout provenance is benchmark contaminated")
    control = control.to(device)
    model = SurfaceRegionSummaryResidualCodebookV1(
        feature_dim=control.feature_dim,
        hidden_dim=control.hidden_dim,
        reliability_attention_mode=control.reliability_attention_mode,
        context_pooling_mode=control.context_pooling_mode,
        control_sha256=control_sha256,
    ).to(device)
    model.load_frozen_base_state_dict(control.state_dict())
    head = SigLIP2SummaryHead.from_radio_checkpoint(radio_path).to(device).eval()
    head.requires_grad_(False)
    fit_text = fit_text.to(device)
    validation_text = validation_text.cpu()
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    control_metrics = _evaluate_v2_control_max(
        control,
        head,
        validation_data,
        validation_text,
        device,
        batch_size=int(args.eval_batch_size),
    )
    initial = _evaluate(
        model,
        head,
        validation_data,
        validation_text,
        device,
        batch_size=int(args.eval_batch_size),
    )
    print(
        json.dumps(
            {
                "untrained": initial,
                "frozen_v2_control": control_metrics,
                "untrained_gate": _generic_gate(initial, control_metrics),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    auxiliary_names = (
        "independent",
        "listwise",
        "hard_negative",
        "relation_balanced",
    )
    calibration: dict[str, float] | None = None
    history: list[dict[str, object]] = []
    feasible_state: dict[str, torch.Tensor] | None = None
    feasible_metrics: dict[str, object] | None = None
    feasible_epoch = 0
    feasible_key: tuple[float, float] | None = None
    stale = 0
    last_state: dict[str, torch.Tensor] | None = None
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
                token_direction_weight=float(args.token_direction_weight),
                token_log_norm_weight=float(args.token_log_norm_weight),
            )
            if epoch <= int(args.warmup_epochs):
                loss = terms["main"]
            else:
                if calibration is None:
                    main_norm = _gradient_norm(
                        terms["main"], parameters, retain_graph=True
                    )
                    calibration = {
                        "main_gradient_l2": main_norm,
                        "auxiliary_gradient_ratio_each": float(
                            args.auxiliary_gradient_ratio_each
                        ),
                    }
                    for name in auxiliary_names:
                        norm = _gradient_norm(
                            terms[name], parameters, retain_graph=True
                        )
                        calibration[f"{name}_gradient_l2"] = norm
                        calibration[f"{name}_lambda"] = (
                            float(args.auxiliary_gradient_ratio_each)
                            * main_norm
                            / norm
                        )
                    print(json.dumps({"gradient_calibration": calibration}), flush=True)
                loss = terms["main"] + sum(
                    calibration[f"{name}_lambda"] * terms[name]
                    for name in auxiliary_names
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
        )
        feasible = _selection_feasible(metrics, control_metrics)
        record: dict[str, object] = {
            "epoch": epoch,
            "loss": sum(epoch_losses) / len(epoch_losses),
            "selection_feasible": feasible,
            "train_components": {
                name: value / len(batches) for name, value in component_sums.items()
            },
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        last_state = copy.deepcopy(model.state_dict())
        if feasible:
            key = (
                float(metrics["teacher_best_slot_cosine_mean"]),
                -float(metrics["text_response_smooth_l1"]),
            )
            if feasible_key is None or key > feasible_key:
                feasible_key = key
                feasible_epoch = epoch
                feasible_state = copy.deepcopy(model.state_dict())
                feasible_metrics = dict(metrics)
                stale = 0
            else:
                stale += 1
            if int(args.patience) > 0 and stale >= int(args.patience):
                break
    if last_state is None or calibration is None:
        raise RuntimeError("residual codebook training did not reach calibrated stage")
    selection_status = (
        "control_referenced_feasible_epoch_selected"
        if feasible_state is not None
        else "no_feasible_epoch_diagnostic_final_state_only"
    )
    selected_state = feasible_state if feasible_state is not None else last_state
    selected_epoch = feasible_epoch if feasible_state is not None else int(history[-1]["epoch"])
    model.load_state_dict(selected_state)
    final = _evaluate(
        model,
        head,
        validation_data,
        validation_text,
        device,
        batch_size=int(args.eval_batch_size),
    )
    if feasible_metrics is not None and final != feasible_metrics:
        raise RuntimeError("selected feasible metrics are not exactly reproducible")
    gate = _generic_gate(final, control_metrics)
    architecture = model.architecture(train_meta["region_contract_sha256"])
    provenance = {
        "training_scope": "global_cross_scene_frozen_v2_residual_codebook_v1",
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
        "canonical_gauge": "caller_provided_exact_frozen_v2",
        "residual_gauge": "l2_direction_tangent_plus_log_norm",
        "physical_sam_boundary_claimed": False,
    }
    payload = {
        "schema_version": 5,
        "architecture": architecture,
        "state_dict": {
            key: value.detach().cpu() for key, value in selected_state.items()
        },
        "provenance": provenance,
        "history": history,
        "selected_epoch": selected_epoch,
        "selection_status": selection_status,
        "untrained_baseline": initial,
        "frozen_v2_control": control_metrics,
        "validation": final,
        "generic_gate": gate,
        "gradient_calibration": calibration,
        "training_config": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "warmup_epochs": int(args.warmup_epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "token_direction_weight": float(args.token_direction_weight),
            "token_log_norm_weight": float(args.token_log_norm_weight),
            "auxiliary_gradient_ratio_each": float(
                args.auxiliary_gradient_ratio_each
            ),
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema_version": 1,
        "status": "complete" if gate["overall_pass"] else "generic_gate_failed",
        "output": str(output),
        "checkpoint_sha256": sha256_file(output),
        "selected_epoch": selected_epoch,
        "selection_status": selection_status,
        "validation": final,
        "frozen_v2_control": control_metrics,
        "generic_gate": gate,
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-direction-weight", type=float, default=0.25)
    parser.add_argument("--token-log-norm-weight", type=float, default=0.25)
    parser.add_argument(
        "--auxiliary-gradient-ratio-each", type=float, default=0.125
    )
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
