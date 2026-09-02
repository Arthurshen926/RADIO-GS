"""Train source-only token spatial support slots over the frozen v10 unary."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.v4.completion.scannet import load_scene_cache
from radio_gs.v4.completion.spatial_slots import TokenSpatialSupportSlots
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.training.diagnose_scannet_learned_mass_calibration import (
    build_mass_features,
)
from radio_gs.v4.training.diagnose_scannet_oracle_mass_projection import METRIC_KEYS
from radio_gs.v4.training.train_scannet_completion_message_passing import (
    _clamp_contract,
    _frozen_unary_probabilities,
    _load_frozen_unary_model,
    _membership_metrics,
    _posterior_to_membership,
    _prepare_runtime,
)


REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.scannet_spatial_slots.v1"
CHECKPOINT_SCHEMA = (
    "radio_gs.surface_object_memory_v4.scannet_spatial_slots_checkpoint.v1"
)
SlotMode = Literal["spatial_slots", "spatial_only", "bias_only"]


def build_observed_pca_geometry(
    runtime: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return source-observed token centres, PCA frames, and PCA scales."""

    positive = runtime["partial"].positive.bool()
    centres = runtime["centres"].float()
    minimum_scale = float(runtime["minimum_scale"])
    token_centres = []
    token_frames = []
    token_scales = []
    for token_id in range(positive.shape[1]):
        points = centres[positive[:, token_id]]
        if points.shape[0] == 0:
            raise RuntimeError("spatial slots require an observed support per token")
        centre = points.mean(0)
        offset = points - centre
        covariance = offset.T @ offset / float(points.shape[0])
        eigenvalues, frame = torch.linalg.eigh(covariance)
        scale = eigenvalues.clamp_min(0).sqrt().clamp_min(minimum_scale)
        token_centres.append(centre)
        token_frames.append(frame)
        token_scales.append(scale)
    return (
        torch.stack(token_centres),
        torch.stack(token_frames),
        torch.stack(token_scales),
    )


def _sample_training_indices(
    runtime: dict[str, Any],
    *,
    maximum_positive_per_token: int,
    maximum_null: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if maximum_positive_per_token <= 0 or maximum_null <= 0:
        raise ValueError("spatial slot sample caps must be positive")
    partial = runtime["partial"]
    labels = torch.as_tensor(runtime["labels"], dtype=torch.long)
    candidate = partial.unknown.any(-1) & partial.eligible_elements
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pieces = []
    for token_id in range(partial.positive.shape[1]):
        indices = torch.where(candidate & (labels == token_id))[0]
        if indices.numel() > maximum_positive_per_token:
            order = torch.randperm(indices.numel(), generator=generator)
            indices = indices[order[:maximum_positive_per_token]]
        if indices.numel():
            pieces.append(indices)
    positive_count = sum(int(value.numel()) for value in pieces)
    null_indices = torch.where(candidate & (labels < 0))[0]
    null_count = min(int(null_indices.numel()), maximum_null, max(positive_count, 1))
    if null_indices.numel() > null_count:
        order = torch.randperm(null_indices.numel(), generator=generator)
        null_indices = null_indices[order[:null_count]]
    if null_indices.numel():
        pieces.append(null_indices)
    if not pieces:
        raise RuntimeError("spatial slot training scene has no unknown samples")
    indices = torch.cat(pieces)
    indices = indices[torch.randperm(indices.numel(), generator=generator)]
    token_count = int(partial.positive.shape[1])
    targets = labels[indices].clone()
    targets[targets < 0] = token_count
    return indices, targets


def _object_equal_sample_iou_loss(
    probabilities: torch.Tensor, targets: torch.Tensor, token_count: int
) -> torch.Tensor:
    token_prediction = probabilities[:, :token_count]
    one_hot = F.one_hot(targets.clamp_max(token_count), token_count + 1)[
        :, :token_count
    ].to(token_prediction.dtype)
    target_mass = one_hot.sum(0)
    present = target_mass > 0
    if not bool(present.any()):
        return token_prediction.sum() * 0
    intersection = (token_prediction * one_hot).sum(0)
    prediction_mass = token_prediction.sum(0)
    union = prediction_mass + target_mass - intersection
    return 1 - (intersection[present] / union[present].clamp_min(1e-8)).mean()


def _forward_indices(
    model: TokenSpatialSupportSlots,
    record: dict[str, Any],
    indices: torch.Tensor,
    *,
    device: torch.device,
):
    runtime = record["runtime"]
    clamp_mask, clamp_probabilities = _clamp_contract(runtime)
    token_centres, token_frames, token_scales = record["pca_geometry"]
    return model(
        record["unary"][indices].to(device),
        runtime["centres"][indices].to(device),
        record["token_features"].to(device),
        token_centres.to(device),
        token_frames.to(device),
        token_scales.to(device),
        clamp_mask[indices].to(device),
        clamp_probabilities[indices].to(device),
    )


def _fit(
    model: TokenSpatialSupportSlots,
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    order_rng = random.Random(args.seed + 719)
    history = []
    model.train()
    for epoch in range(args.epoch_count):
        order = list(range(len(records)))
        order_rng.shuffle(order)
        losses = []
        categorical_losses = []
        iou_losses = []
        for record_index in order:
            record = records[record_index]
            indices, targets = _sample_training_indices(
                record["runtime"],
                maximum_positive_per_token=args.maximum_positive_per_token,
                maximum_null=args.maximum_null_samples,
                seed=args.seed + epoch * 1009 + record_index,
            )
            output = _forward_indices(model, record, indices, device=device)
            target_value = targets.to(device)
            categorical = F.nll_loss(
                output.probabilities.clamp_min(1e-8).log(), target_value
            )
            token_count = int(record["runtime"]["partial"].positive.shape[1])
            iou = _object_equal_sample_iou_loss(
                output.probabilities, target_value, token_count
            )
            loss = categorical + args.object_equal_iou_loss_weight * iou
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
            categorical_losses.append(float(categorical.detach()))
            iou_losses.append(float(iou.detach()))
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": float(np.mean(losses)),
                "mean_categorical_loss": float(np.mean(categorical_losses)),
                "mean_object_equal_sample_iou_loss": float(np.mean(iou_losses)),
            }
        )
    return history


@torch.no_grad()
def _full_posterior(
    model: TokenSpatialSupportSlots,
    record: dict[str, Any],
    *,
    device: torch.device,
    element_batch_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if element_batch_size <= 0:
        raise ValueError("spatial slot element batch size must be positive")
    element_count = int(record["runtime"]["centres"].shape[0])
    pieces = []
    fusion_strength = None
    bias_absolute_mean = None
    scale_multiplier_minimum = float("inf")
    scale_multiplier_maximum = float("-inf")
    base_scales = record["pca_geometry"][2].to(device)
    for start in range(0, element_count, element_batch_size):
        stop = min(start + element_batch_size, element_count)
        indices = torch.arange(start, stop)
        output = _forward_indices(model, record, indices, device=device)
        pieces.append(output.probabilities.cpu())
        fusion_strength = float(output.fusion_strength)
        bias_absolute_mean = float(output.token_bias.abs().mean())
        multiplier = output.slot_scales_local / base_scales[:, None, :]
        scale_multiplier_minimum = min(scale_multiplier_minimum, float(multiplier.min()))
        scale_multiplier_maximum = max(scale_multiplier_maximum, float(multiplier.max()))
    return torch.cat(pieces), {
        "fusion_strength": float(fusion_strength),
        "token_bias_absolute_mean": float(bias_absolute_mean),
        "slot_scale_multiplier_minimum": scale_multiplier_minimum,
        "slot_scale_multiplier_maximum": scale_multiplier_maximum,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode not in ("spatial_slots", "spatial_only", "bias_only"):
        raise ValueError("unsupported spatial slot mode")
    base_report_path = Path(args.base_report).resolve(strict=True)
    base_checkpoint_path = Path(args.base_checkpoint).resolve(strict=True)
    base_report = json.loads(base_report_path.read_text())
    try:
        base_checkpoint = torch.load(
            base_checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError as error:
        raise RuntimeError("safe weights-only checkpoint loading is required") from error
    training_ids = set(map(str, args.training_scene))
    validation_ids = set(map(str, args.validation_scene))
    if training_ids & validation_ids:
        raise ValueError("spatial slot training and validation scenes must be disjoint")
    split = base_report["split"]
    frozen_training_ids = set(map(str, split["training_scene_ids"]))
    frozen_validation_ids = set(map(str, split["validation_scene_ids"]))
    if not frozen_training_ids.issubset(training_ids):
        raise ValueError("spatial slot training must retain the frozen training split")
    if validation_ids != frozen_validation_ids:
        raise ValueError("spatial slots require the frozen validation split")
    payloads = [load_scene_cache(Path(value)) for value in args.scene_cache]
    by_id = {str(value["scene_id"]): value for value in payloads}
    if set(by_id) != training_ids | validation_ids:
        raise ValueError("spatial slot caches do not equal the declared cohort")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    frozen_model = _load_frozen_unary_model(base_checkpoint, device=device)
    temperature = float(base_report["training_configuration"]["temperature"])
    confidence_cap = float(
        base_report["training_configuration"]["completion_confidence_cap"]
    )
    training = []
    for scene_id in sorted(training_ids):
        runtime = _prepare_runtime(by_id[scene_id])
        unary = _frozen_unary_probabilities(
            frozen_model,
            runtime,
            device=device,
            element_batch_size=args.unary_element_batch_size,
            temperature=temperature,
        )
        source = build_mass_features(runtime, unary, feature_mode="summary_f71")
        training.append(
            {
                "scene_id": scene_id,
                "runtime": runtime,
                "unary": unary,
                "token_features": source["features"],
                "pca_geometry": build_observed_pca_geometry(runtime),
            }
        )
    input_dimension = int(training[0]["token_features"].shape[1])
    model = TokenSpatialSupportSlots(
        input_dimension=input_dimension,
        hidden_dimension=args.hidden_dimension,
        dropout=args.dropout,
        use_token_bias=args.mode != "spatial_only",
    ).to(device)
    if args.mode == "bias_only":
        with torch.no_grad():
            model.fusion_parameter.zero_()
        model.fusion_parameter.requires_grad_(False)
    history = _fit(model, training, args=args, device=device)

    # Validation runtimes and target metrics are constructed only after fit.
    del training
    model.eval()
    records = []
    for scene_id in sorted(validation_ids):
        runtime = _prepare_runtime(by_id[scene_id])
        unary = _frozen_unary_probabilities(
            frozen_model,
            runtime,
            device=device,
            element_batch_size=args.unary_element_batch_size,
            temperature=temperature,
        )
        source = build_mass_features(runtime, unary, feature_mode="summary_f71")
        record = {
            "runtime": runtime,
            "unary": unary,
            "token_features": source["features"],
            "pca_geometry": build_observed_pca_geometry(runtime),
        }
        posterior, model_audit = _full_posterior(
            model,
            record,
            device=device,
            element_batch_size=args.inference_element_batch_size,
        )
        frozen_membership, frozen_null = _posterior_to_membership(
            unary, runtime, completion_confidence_cap=confidence_cap
        )
        slot_membership, slot_null = _posterior_to_membership(
            posterior, runtime, completion_confidence_cap=confidence_cap
        )
        records.append(
            {
                "scene_id": scene_id,
                "element_count": int(runtime["centres"].shape[0]),
                "token_count": int(runtime["partial"].positive.shape[1]),
                "model_audit": model_audit,
                "frozen_aligned_pointwise": _membership_metrics(
                    runtime, frozen_membership, frozen_null
                ),
                "spatial_slot_completion": _membership_metrics(
                    runtime, slot_membership, slot_null
                ),
            }
        )

    scene_macro = {}
    for method in ("frozen_aligned_pointwise", "spatial_slot_completion"):
        scene_macro[method] = {
            key: float(np.mean([record[method][key] for record in records]))
            for key in METRIC_KEYS
        }
    difference = {
        key: scene_macro["spatial_slot_completion"][key]
        - scene_macro["frozen_aligned_pointwise"][key]
        for key in METRIC_KEYS
    }
    source_path = Path(__file__).resolve()
    core_path = Path(__file__).resolve().parents[1] / "completion" / "spatial_slots.py"
    output_path = Path(args.output).resolve()
    checkpoint_path = output_path.with_suffix(".pt")
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "mode": args.mode,
        "model_configuration": {
            "input_dimension": input_dimension,
            "hidden_dimension": args.hidden_dimension,
            "dropout": args.dropout,
            "use_token_bias": args.mode != "spatial_only",
        },
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": REPORT_SCHEMA,
        "role": "scene_disjoint_source_only_structured_spatial_support_diagnostic",
        "mode": args.mode,
        "target_membership_is_model_input": False,
        "validation_constructed_after_last_optimizer_step": True,
        "validation_used_for_model_selection": False,
        "training_scene_ids": sorted(training_ids),
        "validation_scene_ids": sorted(validation_ids),
        "additional_training_scene_ids": sorted(training_ids - frozen_training_ids),
        "architecture": model.architecture_receipt(),
        "training_configuration": {
            "seed": args.seed,
            "epoch_count": args.epoch_count,
            "optimizer_step_count": args.epoch_count * len(training_ids),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "object_equal_iou_loss_weight": args.object_equal_iou_loss_weight,
            "maximum_positive_per_token": args.maximum_positive_per_token,
            "maximum_null_samples": args.maximum_null_samples,
            "input_dimension": input_dimension,
            "hidden_dimension": args.hidden_dimension,
        },
        "training_history": history,
        "source": {
            "trainer_path": str(source_path),
            "trainer_sha256": sha256_file(source_path),
            "core_path": str(core_path),
            "core_sha256": sha256_file(core_path),
        },
        "base_report": {
            "path": str(base_report_path),
            "sha256": sha256_file(base_report_path),
        },
        "base_checkpoint": {
            "path": str(base_checkpoint_path),
            "sha256": sha256_file(base_checkpoint_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "per_validation_scene": records,
        "scene_macro": scene_macro,
        "spatial_slots_minus_frozen_scene_macro": difference,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-cache", action="append", required=True)
    parser.add_argument("--training-scene", action="append", required=True)
    parser.add_argument("--validation-scene", action="append", required=True)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--mode",
        choices=("spatial_slots", "spatial_only", "bias_only"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--epoch-count", type=int, default=80)
    parser.add_argument("--hidden-dimension", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--object-equal-iou-loss-weight", type=float, default=0.5)
    parser.add_argument("--maximum-positive-per-token", type=int, default=64)
    parser.add_argument("--maximum-null-samples", type=int, default=2048)
    parser.add_argument("--unary-element-batch-size", type=int, default=4096)
    parser.add_argument("--inference-element-batch-size", type=int, default=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["spatial_slots_minus_frozen_scene_macro"], indent=2))


if __name__ == "__main__":
    main()
