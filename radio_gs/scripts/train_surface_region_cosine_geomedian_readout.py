#!/usr/bin/env python3
"""Train one global SurfaceRegion readout with a robust semantic target.

This treatment changes only how the already frozen, query-free multi-view
teacher observations are summarized.  It follows VALA's released streaming
cosine-tangent update rather than averaging descriptors, and deliberately
does not add another relation loss or a persistent language sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_summary import (
    JOINT_CONTEXT_POOLING,
    SurfaceRegionSummaryReadoutV2,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.train_surface_region_summary_readout import (
    _load,
    _paths,
    _seed_training,
    inject_tangent_direction_noise,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


AGGREGATION = "released_vala_streaming_cosine_tangent_v1"


def streaming_cosine_geometric_median(
    descriptors: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Aggregate ordered unit descriptors with VALA's streaming update.

    For each valid observation ``f_t`` and current unit estimate ``g_t``, the
    update is the released VALA cosine-tangent step::

        g <- normalize(g + (w_t / W_t) * (f_t - <f_t,g> g)).

    The cache order is part of the frozen teacher authority.  Invalid padded
    views have exactly zero weight and cannot change either the estimate or
    cumulative mass.
    """

    values = F.normalize(torch.as_tensor(descriptors).float(), dim=-1, eps=eps)
    valid = torch.as_tensor(mask, device=values.device).bool()
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("descriptors/mask must align as [B,V,D] and [B,V]")
    if not bool(valid.any(dim=1).all()):
        raise ValueError("each row needs at least one valid teacher view")
    if weights is None:
        mass = valid.to(values.dtype)
    else:
        mass = torch.as_tensor(weights, device=values.device).float()
        if mass.shape != valid.shape or not bool(torch.isfinite(mass).all()):
            raise ValueError("weights must be finite and align with the mask")
        if bool((mass < 0).any()):
            raise ValueError("weights cannot be negative")
        mass = mass * valid.to(mass.dtype)
        if not bool((mass.sum(dim=1) > 0).all()):
            raise ValueError("each row needs positive teacher weight")

    estimate = torch.zeros(
        values.shape[0], values.shape[2], dtype=values.dtype, device=values.device
    )
    cumulative = torch.zeros(values.shape[0], 1, dtype=values.dtype, device=values.device)
    for view_index in range(values.shape[1]):
        weight = mass[:, view_index : view_index + 1]
        active = weight > 0
        observation = values[:, view_index]
        next_cumulative = cumulative + weight
        step = torch.where(
            active,
            weight / next_cumulative.clamp_min(eps),
            torch.zeros_like(weight),
        )
        dot = (observation * estimate).sum(dim=-1, keepdim=True)
        tangent = observation - dot * estimate
        candidate = F.normalize(estimate + step * tangent, dim=-1, eps=eps)
        estimate = torch.where(active, candidate, estimate)
        cumulative = next_cumulative
    return F.normalize(estimate, dim=-1, eps=eps)


def robust_targets(
    data: dict,
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return robust target token/descriptor and the mean audit target."""

    tokens = data["official_summary_tokens"][rows].float()
    descriptors = F.normalize(
        data["official_crop_summaries"][rows].float(), dim=-1, eps=1e-8
    )
    mask = data["teacher_mask"][rows].bool()
    robust = streaming_cosine_geometric_median(descriptors, mask)
    similarity = torch.einsum("bvd,bd->bv", descriptors, robust)
    nearest = similarity.masked_fill(~mask, -torch.inf).argmax(dim=-1)
    batch = torch.arange(len(rows))
    target_token = tokens[batch, nearest]
    mean_weights = mask.float() / mask.sum(dim=1, keepdim=True)
    arithmetic_mean = F.normalize(
        (descriptors * mean_weights[..., None]).sum(dim=1), dim=-1, eps=1e-8
    )
    return target_token, robust, arithmetic_mean, descriptors, mask


@torch.no_grad()
def evaluate(
    model: SurfaceRegionSummaryReadoutV2,
    head: SigLIP2SummaryHead,
    data: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    token_cos: list[float] = []
    robust_cos: list[float] = []
    mean_cos: list[float] = []
    view_cos: list[float] = []
    for start in range(0, len(data["radio_features"]), int(batch_size)):
        rows = torch.arange(
            start, min(start + int(batch_size), len(data["radio_features"]))
        )
        token, robust, mean, all_descriptors, teacher_mask = robust_targets(data, rows)
        predicted = model(
            data["radio_features"][rows].to(device),
            data["geometry"][rows].to(device),
            anchor_index=data["anchor_index"][rows].to(device),
            token_mask=data["token_mask"][rows].to(device),
            reliability=data["reliability"][rows].to(device),
        )
        projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1)
        predicted_cpu = predicted.cpu()
        projected_cpu = projected.cpu()
        token_cos.extend(F.cosine_similarity(predicted_cpu, token, dim=-1).tolist())
        robust_cos.extend(F.cosine_similarity(projected_cpu, robust, dim=-1).tolist())
        mean_cos.extend(F.cosine_similarity(projected_cpu, mean, dim=-1).tolist())
        pair = torch.einsum("bd,bvd->bv", projected_cpu, all_descriptors)
        view_cos.extend(pair[teacher_mask].tolist())
    return {
        "summary_token_cosine": sum(token_cos) / len(token_cos),
        "robust_target_descriptor_cosine": sum(robust_cos) / len(robust_cos),
        "arithmetic_mean_descriptor_cosine": sum(mean_cos) / len(mean_cos),
        "all_view_descriptor_cosine": sum(view_cos) / len(view_cos),
    }


def selection_score(metrics: dict[str, float]) -> float:
    return 0.5 * (
        metrics["robust_target_descriptor_cosine"]
        + metrics["all_view_descriptor_cosine"]
    )


def train(args: argparse.Namespace) -> dict:
    registration = Path(args.experiment_registration).resolve()
    if not registration.is_file():
        raise FileNotFoundError("the robust-target experiment must be preregistered")
    expected_registration = str(args.experiment_registration_sha256).strip()
    if not expected_registration or sha256_file(registration) != expected_registration:
        raise ValueError("experiment registration changed after preregistration")

    train_data, train_meta = _load(_paths(args.train_caches), "train")
    validation_data, validation_meta = _load(
        _paths(args.validation_caches), "validation"
    )
    overlap = set(train_meta["scenes"]) & set(validation_meta["scenes"])
    if overlap:
        raise ValueError(f"train/validation scene leakage: {sorted(overlap)}")
    for key in (
        "region_contract_sha256",
        "excluded_physical_spaces",
        "teacher_region",
        "radio_checkpoint_sha256",
    ):
        if train_meta[key] != validation_meta[key]:
            raise ValueError(f"train/validation {key} differs")
    if sha256_file(args.radio_checkpoint) != train_meta["radio_checkpoint_sha256"]:
        raise ValueError("RADIO checkpoint differs from cache provenance")

    device = torch.device(args.device)
    generator = _seed_training(int(args.seed), device=device)
    model = SurfaceRegionSummaryReadoutV2(
        hidden_dim=int(args.hidden_dim),
        reliability_attention_mode="log_prior",
        context_pooling_mode=JOINT_CONTEXT_POOLING,
    ).to(device)
    head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint).to(device).eval()
    head.requires_grad_(False)

    model.eval()
    untrained = evaluate(model, head, validation_data, device, args.batch_size)
    print(json.dumps({"untrained": untrained}), flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history: list[dict] = []
    for epoch in range(int(args.epochs)):
        order = torch.randperm(len(train_data["radio_features"]), generator=generator)
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), int(args.batch_size)):
            rows = order[start : start + int(args.batch_size)]
            token, robust, _mean, all_descriptors, teacher_mask = robust_targets(
                train_data, rows
            )
            token_mask = train_data["token_mask"][rows].to(device)
            radio_features = inject_tangent_direction_noise(
                train_data["radio_features"][rows].to(device),
                token_mask,
                angle_degrees=float(args.canonical_noise_degrees),
            )
            predicted = model(
                radio_features,
                train_data["geometry"][rows].to(device),
                anchor_index=train_data["anchor_index"][rows].to(device),
                token_mask=token_mask,
                reliability=train_data["reliability"][rows].to(device),
            )
            projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1)
            token = token.to(device)
            robust = robust.to(device)
            all_descriptors = all_descriptors.to(device)
            teacher_mask = teacher_mask.to(device)
            token_loss = (
                1.0 - F.cosine_similarity(predicted, token, dim=-1)
            ).mean()
            robust_loss = (
                1.0 - F.cosine_similarity(projected, robust, dim=-1)
            ).mean()
            view_cosine = torch.einsum("bd,bvd->bv", projected, all_descriptors)
            all_view_loss = (1.0 - view_cosine)[teacher_mask].mean()
            loss = (
                float(args.robust_target_weight) * robust_loss
                + float(args.all_view_weight) * all_view_loss
                + float(args.token_weight) * token_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        metrics = evaluate(model, head, validation_data, device, args.batch_size)
        score = selection_score(metrics)
        record = {
            "epoch": epoch + 1,
            "loss": sum(losses) / len(losses),
            "selection_score": score,
            **metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if int(args.patience) > 0 and stale >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation = evaluate(model, head, validation_data, device, args.batch_size)
    architecture = model.architecture(train_meta["region_contract_sha256"])
    provenance = {
        "training_scope": "global_cross_scene_3d_surface_v2_cosine_geomedian",
        "frozen": True,
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "scene_disjoint": True,
        "official_summary_head": "c-radio_v4 siglip2-g",
        "custom_text_projection": False,
        "region_contract_sha256": train_meta["region_contract_sha256"],
        "region_contract": train_meta["region_contract"],
        "teacher_aggregation": AGGREGATION,
        "teacher_region": train_meta["teacher_region"],
        "train": train_meta,
        "validation": validation_meta,
        "experiment_registration": file_record(registration),
        "canonical_direction_noise_degrees": float(args.canonical_noise_degrees),
        "random_seed_contract": {
            "seed": int(args.seed),
            "model_initialization": True,
            "data_order": True,
            "canonical_noise": True,
        },
    }
    payload = {
        "schema_version": 3,
        "architecture": architecture,
        "state_dict": best_state,
        "provenance": provenance,
        "history": history,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "untrained_baseline": untrained,
        "training_config": vars(args),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    report = {
        "schema_version": 1,
        "artifact_type": "surface_region_cosine_geomedian_readout_result",
        "output": str(output),
        "checkpoint_sha256": sha256_file(output),
        "experiment_registration": file_record(registration),
        "teacher_aggregation": AGGREGATION,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "validation": validation,
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-registration", required=True)
    parser.add_argument("--experiment-registration-sha256", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--robust-target-weight", type=float, default=1.0)
    parser.add_argument("--all-view-weight", type=float, default=0.25)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--canonical-noise-degrees", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
