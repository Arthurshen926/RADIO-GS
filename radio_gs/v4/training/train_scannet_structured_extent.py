"""Train token-conditioned structured extent over the frozen v10 unary.

The structured model is an independent v4 arm.  It reuses the strict formal16
cohort, frozen aligned RADIO unary, rendering protocol, and losslessly poolable
evaluation statistics without modifying either frozen completion trainer.
ScanNet instance identities and held-out mesh rasters are target-only and are
passed only to the training-loss functions for the selected training scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.v4.completion.structured_extent import (
    STRUCTURED_EXTENT_MODES,
    TokenConditionedStructuredExtent,
)
from radio_gs.v4.completion.scannet import load_scene_cache
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.training.train_scannet_completion_message_passing import (
    AGGREGATE_KEYS,
    PROTOCOLS,
    RGB_RADIO_GEOMETRY_LAYOUT,
    _base_replay_receipt,
    _cache_receipts,
    _canonical_json_sha256,
    _clamp_contract,
    _edge_same_instance_bce,
    _frozen_unary_probabilities,
    _load_base_authority,
    _load_cohort_manifest,
    _load_frozen_unary_model,
    _membership_metrics,
    _message_passing_bypass_control,
    _posterior_to_membership,
    _prepare_runtime,
    _render_differentiable,
    _target_absent_mass_comparison,
    _target_absent_prediction_mass,
    _unknown_categorical_loss,
)
from radio_gs.v4.training.train_scannet_completion_oracle import (
    SOFT_IOU_COHORTS,
    _mean_scene_metric,
    _pool_soft_iou_sufficient_statistics,
    _pooled_categorical_confusion,
)


REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.scannet_structured_extent.v2"
CHECKPOINT_SCHEMA = (
    "radio_gs.surface_object_memory_v4.scannet_structured_extent_checkpoint.v2"
)
METHODS = (
    "baseline_observed_only",
    "frozen_aligned_pointwise",
    "structured_extent_bypass_control",
    "structured_extent_completion",
)
FIXED_ITERATION_COUNT = 2


def _source_tree_receipt() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "radio_gs/v4/training/train_scannet_structured_extent.py",
        "radio_gs/v4/completion/structured_extent.py",
        "radio_gs/v4/training/train_scannet_completion_message_passing.py",
        "radio_gs/v4/completion/message_passing.py",
        "radio_gs/v4/training/train_scannet_completion_oracle.py",
        "radio_gs/v4/completion/oracle.py",
        "radio_gs/v4/completion/scannet.py",
        "radio_gs/v4/carrier/sparse_voxel.py",
        "radio_gs/v4/carrier/base.py",
    )
    records = [
        {
            "path": relative_path,
            "sha256": sha256_file((repo_root / relative_path).resolve(strict=True)),
        }
        for relative_path in relative_paths
    ]
    return {
        "schema": "radio_gs.structured_extent_source_closure.v1",
        "repository_root": str(repo_root),
        "files": records,
        "source_tree_sha256": _canonical_json_sha256(records),
    }


def _target_membership(
    runtime: dict[str, Any], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    labels = torch.as_tensor(runtime["labels"], dtype=torch.long, device=device)
    eligible = runtime["partial"].eligible_elements.to(device)
    token_count = int(runtime["partial"].positive.shape[1])
    target = torch.zeros(labels.numel(), token_count, device=device, dtype=dtype)
    object_surface = eligible & (labels >= 0)
    if not bool(object_surface.any()):
        raise ValueError("structured extent requires retained object targets")
    if int(labels[object_surface].max()) >= token_count:
        raise ValueError("structured extent target exceeds the token domain")
    target[object_surface, labels[object_surface]] = 1
    return target


def _object_equal_3d_soft_iou_loss(
    membership: torch.Tensor, runtime: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, float | int]]:
    target = _target_membership(
        runtime, device=membership.device, dtype=membership.dtype
    )
    eligible = runtime["partial"].eligible_elements.to(membership.device)
    prediction = membership * eligible[:, None]
    target = target * eligible[:, None]
    intersection = (prediction * target).sum(0)
    prediction_mass = prediction.sum(0)
    target_mass = target.sum(0)
    union = prediction_mass + target_mass - intersection
    present = target_mass > 0
    if not bool(present.any()):
        raise ValueError("object-equal 3D loss has no target-present token")
    soft_iou = intersection[present] / union[present].clamp_min(1e-12)
    return 1 - soft_iou.mean(), {
        "target_present_token_count": int(present.sum()),
        "object_equal_soft_iou": float(soft_iou.detach().mean()),
        "prediction_mass": float(prediction_mass.detach().sum()),
        "target_mass": float(target_mass.detach().sum()),
    }


def _log_full_mass_smooth_l1_loss(
    predicted_full_mass: torch.Tensor, runtime: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    predicted = torch.as_tensor(predicted_full_mass)
    token_count = int(runtime["partial"].positive.shape[1])
    if predicted.shape != (token_count,):
        raise ValueError("predicted_full_mass must have shape [K]")
    if not torch.isfinite(predicted).all() or bool((predicted < 0).any()):
        raise ValueError("predicted full mass must be finite and non-negative")
    target_membership = _target_membership(
        runtime, device=predicted.device, dtype=predicted.dtype
    )
    target = target_membership.sum(0)
    loss = F.smooth_l1_loss(
        predicted.clamp_min(torch.finfo(predicted.dtype).tiny).log(),
        target.clamp_min(torch.finfo(target.dtype).tiny).log(),
    )
    return loss, {
        "token_count": token_count,
        "predicted_full_mass_mean": float(predicted.detach().mean()),
        "target_full_mass_mean": float(target.detach().mean()),
        "log_full_mass_smooth_l1": float(loss.detach()),
    }


def _heldout_present_and_absent_losses(
    membership: torch.Tensor,
    runtime: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    projections = runtime["heldout_projections"]
    targets = runtime["payload"]["heldout_mesh_target_rasters"]
    if not projections or len(projections) != len(targets):
        raise ValueError("structured held-out losses require aligned views")
    token_count = membership.shape[1]
    intersection = torch.zeros(token_count, device=membership.device)
    prediction_mass = torch.zeros_like(intersection)
    prediction_square_mass = torch.zeros_like(intersection)
    target_mass = torch.zeros_like(intersection)
    pixel_count = 0
    for projection, target_value in zip(projections, targets):
        prediction = _render_differentiable(membership, projection).reshape(
            -1, token_count
        )
        target = torch.as_tensor(
            target_value, device=prediction.device, dtype=prediction.dtype
        ).reshape(-1, token_count)
        if target.shape != prediction.shape:
            raise ValueError("structured held-out render and mesh target do not align")
        intersection = intersection + (prediction * target).sum(0)
        prediction_mass = prediction_mass + prediction.sum(0)
        prediction_square_mass = prediction_square_mass + prediction.square().sum(0)
        target_mass = target_mass + target.sum(0)
        pixel_count += prediction.shape[0]
    present = target_mass > 0
    absent = ~present
    if not bool(present.any()):
        raise ValueError("structured held-out loss has no target-present token")
    union = prediction_mass + target_mass - intersection
    present_iou = intersection[present] / union[present].clamp_min(1e-12)
    present_macro_loss = 1 - present_iou.mean()
    if bool(absent.any()):
        absent_probability_mass = prediction_mass[absent] / float(pixel_count)
        absent_mean_mass = absent_probability_mass.mean()
        absent_rms_mass = torch.sqrt(
            prediction_square_mass[absent].sum()
            / float(pixel_count * int(absent.sum()))
            + 1e-12
        )
        # The old mean divided by every pixel and absent token before applying
        # its weight, leaving parameter gradients around 1e-8.  A per-token
        # log total-mass penalty is resolution-auditable and retains useful
        # gradients without a hard support threshold.
        absent_log_total_mass = torch.log1p(prediction_mass[absent]).mean()
    else:
        zero = prediction_mass.sum() * 0
        absent_mean_mass = zero
        absent_rms_mass = zero
        absent_log_total_mass = zero
    return present_macro_loss, absent_rms_mass, absent_log_total_mass, {
        "heldout_view_count": len(projections),
        "heldout_pixel_count": pixel_count,
        "target_present_token_count": int(present.sum()),
        "target_absent_token_count": int(absent.sum()),
        "present_macro_soft_iou": float(present_iou.detach().mean()),
        "continuous_absent_rms_mass": float(absent_rms_mass.detach()),
        "continuous_absent_mean_probability": float(absent_mean_mass.detach()),
        "continuous_absent_log_total_mass_loss": float(
            absent_log_total_mass.detach()
        ),
        "continuous_absent_total_prediction_mass": float(
            prediction_mass[absent].detach().sum()
        )
        if bool(absent.any())
        else 0.0,
    }


def _sample_token_conditioned_edge_pairs(
    runtime: dict[str, Any],
    *,
    maximum_pairs_per_class: int,
    seed: int,
) -> dict[str, Any]:
    if maximum_pairs_per_class <= 0:
        raise ValueError("maximum_pairs_per_class must be positive")
    edges = torch.as_tensor(runtime["edge_index"], dtype=torch.long).cpu()
    labels = torch.as_tensor(runtime["labels"], dtype=torch.long).cpu()
    eligible = runtime["partial"].eligible_elements.cpu()
    token_count = int(runtime["partial"].positive.shape[1])
    source, destination = edges
    valid_edge = eligible[source] & eligible[destination]
    same_instance = (
        valid_edge
        & (labels[source] >= 0)
        & (labels[source] == labels[destination])
    )
    positive_edge_ids = torch.where(same_instance)[0]
    positive_token_ids = labels[source[positive_edge_ids]]
    candidate_edge_ids = torch.where(valid_edge)[0]
    scene_seed = int(
        hashlib.sha256(
            f"{runtime['payload']['scene_id']}\0{int(seed)}".encode("utf-8")
        ).hexdigest()[:16],
        16,
    )
    candidate_token_ids = (
        torch.arange(candidate_edge_ids.numel(), dtype=torch.long)
        + (scene_seed % token_count)
    ) % token_count
    candidate_is_positive = (
        labels[source[candidate_edge_ids]] == candidate_token_ids
    ) & (labels[destination[candidate_edge_ids]] == candidate_token_ids)
    negative_edge_ids = candidate_edge_ids[~candidate_is_positive]
    negative_token_ids = candidate_token_ids[~candidate_is_positive]
    if not positive_edge_ids.numel() or not negative_edge_ids.numel():
        raise ValueError("token-conditioned edge BCE requires both target classes")
    generator = torch.Generator().manual_seed(scene_seed % (2**63 - 1))

    positive_count = min(maximum_pairs_per_class, positive_edge_ids.numel())
    negative_count = min(maximum_pairs_per_class, negative_edge_ids.numel())
    positive_order = torch.randperm(
        positive_edge_ids.numel(), generator=generator
    )[:positive_count]
    negative_order = torch.randperm(
        negative_edge_ids.numel(), generator=generator
    )[:negative_count]
    selected_edge_ids = torch.cat(
        (positive_edge_ids[positive_order], negative_edge_ids[negative_order])
    )
    selected_token_ids = torch.cat(
        (positive_token_ids[positive_order], negative_token_ids[negative_order])
    )
    target = torch.cat(
        (
            torch.ones(positive_count, dtype=torch.float32),
            torch.zeros(negative_count, dtype=torch.float32),
        )
    )
    return {
        "edge_ids": selected_edge_ids,
        "token_ids": selected_token_ids,
        "target": target,
        "audit": {
            "scene_id": str(runtime["payload"]["scene_id"]),
            "positive_pair_count": positive_count,
            "negative_pair_count": negative_count,
            "maximum_pairs_per_class": maximum_pairs_per_class,
            "edge_ids_sha256": hashlib.sha256(
                selected_edge_ids.numpy().astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "token_ids_sha256": hashlib.sha256(
                selected_token_ids.numpy().astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "target_definition": (
                "both_edge_endpoints_belong_to_the_sampled_retained_token"
            ),
        },
    }


def _balanced_token_edge_bce(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    target = torch.as_tensor(target, dtype=logits.dtype, device=logits.device)
    if logits.shape != target.shape or logits.ndim != 1:
        raise ValueError("sampled token-edge logits and targets must align")
    positive = target == 1
    negative = target == 0
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("sampled token-edge BCE requires both target classes")
    return 0.5 * (
        F.binary_cross_entropy_with_logits(
            logits[positive], torch.ones_like(logits[positive])
        )
        + F.binary_cross_entropy_with_logits(
            logits[negative], torch.zeros_like(logits[negative])
        )
    )


def _structured_forward(
    model: TokenConditionedStructuredExtent,
    runtime: dict[str, Any],
    *,
    device: torch.device,
    completion_confidence_cap: float,
) -> Any:
    clamp_mask, clamp_probabilities = _clamp_contract(runtime)
    return model(
        unary_probabilities=runtime["frozen_unary"].to(device, non_blocking=True),
        edge_index=runtime["edge_index"].to(device, non_blocking=True),
        centres=runtime["centres"].to(device, non_blocking=True),
        normals=torch.as_tensor(
            runtime["payload"]["normals"], dtype=torch.float32
        ).to(device, non_blocking=True),
        local_features=runtime["local_features"].to(device, non_blocking=True),
        source_visible=runtime["source_visible"].to(device, non_blocking=True),
        observed_positive=runtime["partial"].positive.to(
            device, non_blocking=True
        ),
        clamp_mask=clamp_mask.to(device, non_blocking=True),
        clamp_probabilities=clamp_probabilities.to(device, non_blocking=True),
        voxel_size=float(runtime["minimum_scale"]),
        completion_confidence_cap=completion_confidence_cap,
        return_token_edge_logits=False,
    )


def _sampled_token_edge_logits(
    model: TokenConditionedStructuredExtent,
    output: Any,
    runtime: dict[str, Any],
    supervision: dict[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    edge_ids = supervision["edge_ids"].to(device, non_blocking=True)
    token_ids = supervision["token_ids"].to(device, non_blocking=True)
    _, token_logits = model.score_token_edges(
        output.edge_features,
        runtime["edge_index"].to(device, non_blocking=True),
        output.node_token_affinity,
        edge_ids=edge_ids,
        token_ids=token_ids,
    )
    if token_logits.shape != edge_ids.shape:
        raise RuntimeError("sampled token-conditioned edge logits changed shape")
    return token_logits


def _structured_edge_supervision_loss(
    model: TokenConditionedStructuredExtent,
    output: Any,
    runtime: dict[str, Any],
    supervision: dict[str, Any] | None,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if model.mode == "shared_edge_plus_mass":
        loss, audit = _edge_same_instance_bce(
            output.base_edge_logits,
            runtime["edge_index"],
            runtime,
        )
        return loss, {
            "policy": "query_free_same_retained_instance_edge_bce",
            **audit,
        }
    if supervision is None:
        raise ValueError("token-conditioned edge supervision is required")
    sampled_edge_logits = _sampled_token_edge_logits(
        model,
        output,
        runtime,
        supervision,
        device=device,
    )
    loss = _balanced_token_edge_bce(sampled_edge_logits, supervision["target"])
    return loss, {
        "policy": "sampled_token_conditioned_pair_bce",
        **supervision["audit"],
    }


def _clip_structured_gradient_groups(
    model: TokenConditionedStructuredExtent, maximum_norm: float
) -> dict[str, torch.Tensor]:
    """Clip causal posterior and auxiliary-only mass parameters separately."""

    if not np.isfinite(maximum_norm) or maximum_norm <= 0:
        raise ValueError("maximum structured gradient norm must be positive")
    mass_parameters = list(model.mass_encoder.parameters()) + list(
        model.mass_head.parameters()
    )
    mass_parameter_ids = {id(parameter) for parameter in mass_parameters}
    posterior_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in mass_parameter_ids
    ]
    if not mass_parameters or not posterior_parameters:
        raise RuntimeError("structured gradient parameter groups are empty")
    return {
        "posterior": torch.nn.utils.clip_grad_norm_(
            posterior_parameters, maximum_norm
        ),
        "mass": torch.nn.utils.clip_grad_norm_(mass_parameters, maximum_norm),
    }


def _fit(
    model: TokenConditionedStructuredExtent,
    training_runtimes: list[dict[str, Any]],
    *,
    device: torch.device,
    epoch_count: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    completion_confidence_cap: float,
    loss_weights: dict[str, float],
    maximum_edge_pairs_per_class: int,
    seed: int,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    uses_token_pair_supervision = model.mode != "shared_edge_plus_mass"
    edge_supervision = (
        {
            str(runtime["payload"]["scene_id"]): _sample_token_conditioned_edge_pairs(
                runtime,
                maximum_pairs_per_class=maximum_edge_pairs_per_class,
                seed=seed,
            )
            for runtime in training_runtimes
        }
        if uses_token_pair_supervision
        else {}
    )
    shared_edge_audit: dict[str, dict[str, Any]] = {}
    python_rng = random.Random(seed + 101)
    history = []
    model.train()
    loss_names = tuple(loss_weights)
    for epoch in range(epoch_count):
        order = list(range(len(training_runtimes)))
        python_rng.shuffle(order)
        values = {
            "total": [],
            "posterior_gradient_norm": [],
            "mass_gradient_norm": [],
            "mass_dual_log_residual": [],
        } | {name: [] for name in loss_names}
        for runtime_index in order:
            runtime = training_runtimes[runtime_index]
            scene_id = str(runtime["payload"]["scene_id"])
            output = _structured_forward(
                model,
                runtime,
                device=device,
                completion_confidence_cap=completion_confidence_cap,
            )
            membership, _ = _posterior_to_membership(
                output.probabilities,
                runtime,
                completion_confidence_cap=completion_confidence_cap,
            )
            categorical, _ = _unknown_categorical_loss(
                output.probabilities, runtime
            )
            object_equal_3d, _ = _object_equal_3d_soft_iou_loss(
                membership, runtime
            )
            token_edge, current_edge_audit = _structured_edge_supervision_loss(
                model,
                output,
                runtime,
                edge_supervision.get(scene_id),
                device=device,
            )
            if not uses_token_pair_supervision:
                if (
                    scene_id in shared_edge_audit
                    and shared_edge_audit[scene_id] != current_edge_audit
                ):
                    raise RuntimeError("shared edge supervision changed across epochs")
                shared_edge_audit[scene_id] = current_edge_audit
            log_full_mass, _ = _log_full_mass_smooth_l1_loss(
                output.predicted_full_mass, runtime
            )
            (
                heldout_present,
                continuous_absent_rms,
                continuous_absent_mass,
                _,
            ) = _heldout_present_and_absent_losses(membership, runtime)
            losses = {
                "class_balanced_categorical": categorical,
                "object_equal_3d_soft_iou": object_equal_3d,
                "token_conditioned_edge_bce": token_edge,
                "log_full_mass_smooth_l1": log_full_mass,
                "heldout_present_macro_soft_iou": heldout_present,
                "continuous_absent_rms": continuous_absent_rms,
                "continuous_absent_mass": continuous_absent_mass,
            }
            if set(losses) != set(loss_weights):
                raise RuntimeError("structured extent loss contract changed")
            total = sum(loss_weights[name] * losses[name] for name in loss_names)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norms = _clip_structured_gradient_groups(
                model, gradient_clip_norm
            )
            if any(
                not torch.isfinite(torch.as_tensor(value))
                for value in gradient_norms.values()
            ):
                raise RuntimeError("structured extent gradient norm is non-finite")
            optimizer.step()
            values["total"].append(float(total.detach()))
            values["posterior_gradient_norm"].append(
                float(gradient_norms["posterior"].detach())
            )
            values["mass_gradient_norm"].append(
                float(gradient_norms["mass"].detach())
            )
            values["mass_dual_log_residual"].append(
                float(
                    (
                        output.predicted_dual_posterior_mass.detach().log()
                        - output.realized_posterior_mass.detach().clamp_min(1e-12).log()
                    )
                    .abs()
                    .mean()
                )
            )
            for name, loss in losses.items():
                values[name].append(float(loss.detach()))
        history.append(
            {
                "epoch": epoch + 1,
                **{
                    f"mean_{name}_loss": float(np.mean(current))
                    for name, current in values.items()
                },
                "transport_step_strengths": [
                    float(value)
                    for value in model._bounded_step_strength(
                        model.transport_step_parameters.detach()
                    ).cpu()
                ],
                "dual_step_strengths": [
                    float(value)
                    for value in model._bounded_step_strength(
                        model.dual_step_parameters.detach()
                    ).cpu()
                ],
            }
        )
    return {
        "epoch_history": history,
        "optimizer_step_count": epoch_count * len(training_runtimes),
        "final_epoch": history[-1],
        "edge_supervision_policy": (
            "sampled_token_conditioned_pair_bce"
            if uses_token_pair_supervision
            else "query_free_same_retained_instance_edge_bce"
        ),
        "token_conditioned_edge_supervision": {
            scene_id: value["audit"]
            for scene_id, value in edge_supervision.items()
        },
        "query_free_shared_edge_supervision": shared_edge_audit,
        "edge_supervision_is_fixed_across_epochs": True,
        "gradient_clipping_policy": (
            "separate_posterior_and_auxiliary_mass_parameter_groups"
        ),
    }


@torch.no_grad()
def _evaluate_scene(
    model: TokenConditionedStructuredExtent,
    runtime: dict[str, Any],
    *,
    device: torch.device,
    completion_confidence_cap: float,
) -> dict[str, Any]:
    observed_membership = runtime["partial"].positive.float()
    observed_null = 1 - observed_membership.sum(-1)
    frozen_membership, frozen_null = _posterior_to_membership(
        runtime["frozen_unary"],
        runtime,
        completion_confidence_cap=completion_confidence_cap,
    )
    bypass = _message_passing_bypass_control(runtime["frozen_unary"])
    bypass_membership, bypass_null = _posterior_to_membership(
        bypass, runtime, completion_confidence_cap=completion_confidence_cap
    )
    output = _structured_forward(
        model,
        runtime,
        device=device,
        completion_confidence_cap=completion_confidence_cap,
    )
    structured_membership, structured_null = _posterior_to_membership(
        output.probabilities,
        runtime,
        completion_confidence_cap=completion_confidence_cap,
    )
    return {
        "scene_id": str(runtime["payload"]["scene_id"]),
        "element_count": int(runtime["centres"].shape[0]),
        "token_count": int(runtime["partial"].positive.shape[1]),
        "heldout_pixel_count": int(
            sum(
                projection.height * projection.width
                for projection in runtime["heldout_projections"]
            )
        ),
        "baseline_observed_only": _membership_metrics(
            runtime, observed_membership, observed_null
        ),
        "frozen_aligned_pointwise": _membership_metrics(
            runtime, frozen_membership, frozen_null
        ),
        "structured_extent_bypass_control": _membership_metrics(
            runtime, bypass_membership, bypass_null
        ),
        "structured_extent_completion": _membership_metrics(
            runtime, structured_membership, structured_null
        ),
        "structured_extent": {
            "iteration_count": len(output.step_probabilities),
            "clamp_max_error": float(output.clamp_max_error),
            "predicted_physical_support_mass_mean": float(
                output.predicted_full_mass.mean()
            ),
            "predicted_completed_membership_mass_mean": float(
                output.predicted_completed_membership_mass.mean()
            ),
            "realized_completed_membership_mass_mean": float(
                output.realized_full_mass.mean()
            ),
            "realized_posterior_support_mass_mean": float(
                output.realized_posterior_mass.mean()
            ),
            "dual_bias_absolute_mean": float(output.dual_bias.abs().mean()),
            "node_token_affinity_mean": float(output.node_token_affinity.mean()),
            "base_edge_logit_mean": float(output.base_edge_logits.mean())
            if output.base_edge_logits.numel()
            else 0.0,
            "bypass_max_probability_error_from_frozen_unary": float(
                (bypass - runtime["frozen_unary"]).abs().max()
            ),
        },
    }


def _aggregate_evaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
    scene_macro = {
        method: {
            key: _mean_scene_metric(records, method, key)
            for key in AGGREGATE_KEYS
        }
        for method in METHODS
    }
    pooled_soft_iou = {
        method: _pool_soft_iou_sufficient_statistics(records, method)
        for method in METHODS
    }
    pooled_unknown = {
        method: _pooled_categorical_confusion(records, method, "unknown")
        for method in METHODS
    }
    pooled_strata = {
        stratum: {
            method: _pooled_categorical_confusion(records, method, stratum)
            for method in METHODS
        }
        for stratum in ("visible_but_unmasked", "never_visible")
    }
    absent_mass = {
        method: _target_absent_prediction_mass(records, method)
        for method in METHODS
    }
    return {
        "scene_macro": scene_macro,
        "pooled_soft_iou_sufficient_statistics": pooled_soft_iou,
        "pooled_unknown_categorical_confusion": pooled_unknown,
        "pooled_unknown_strata_categorical_confusion": pooled_strata,
        "target_absent_heldout_prediction_mass": {
            "by_method": absent_mass,
            "final_vs_frozen": _target_absent_mass_comparison(
                absent_mass,
                final_method="structured_extent_completion",
            ),
            "false_positive_only_count_role": (
                "support_diagnostic_only_not_a_promotion_direction_under_"
                "strict_positive_soft_posteriors"
            ),
        },
    }


def _paired_differences(aggregate: dict[str, Any]) -> dict[str, Any]:
    frozen = "frozen_aligned_pointwise"
    final = "structured_extent_completion"
    scene_macro = aggregate["scene_macro"]
    pooled = aggregate["pooled_soft_iou_sufficient_statistics"]
    scene_keys = (
        "soft_3d_miou",
        "unknown_only_soft_3d_miou",
        "heldout_2d_soft_miou",
        "visible_but_unmasked_soft_3d_miou",
        "visible_but_unmasked_assignment_precision",
        "visible_but_unmasked_retained_object_coverage",
        "visible_but_unmasked_correct_assignment_recall",
        "never_visible_soft_3d_miou",
    )
    scene = {
        key: scene_macro[final][key] - scene_macro[frozen][key]
        for key in scene_keys
    }
    pooled_difference = {}
    for cohort in SOFT_IOU_COHORTS:
        for metric in (
            "scene_token_macro_soft_iou",
            "union_summed_element_or_pixel_token_micro_soft_iou",
        ):
            pooled_difference[f"{cohort}_{metric}"] = (
                pooled[final][cohort][metric] - pooled[frozen][cohort][metric]
            )
    fp_frozen = pooled[frozen]["heldout_2d"]["counts"][
        "false_positive_only_scene_token_count"
    ]
    fp_final = pooled[final]["heldout_2d"]["counts"][
        "false_positive_only_scene_token_count"
    ]
    absent = aggregate["target_absent_heldout_prediction_mass"]["final_vs_frozen"]
    normalized_absent = absent["metrics"][
        "prediction_mass_per_target_absent_heldout_pixel_token"
    ]
    return {
        "structured_extent_minus_frozen_aligned_pointwise_scene_macro": scene,
        "structured_extent_minus_frozen_aligned_pointwise_pooled": pooled_difference,
        "target_absent_heldout_prediction_mass": absent,
        "heldout_2d_false_positive_only_scene_token_count": {
            "frozen_aligned_pointwise": fp_frozen,
            "structured_extent_completion": fp_final,
            "difference": fp_final - fp_frozen,
            "role": "support_diagnostic_only_not_a_promotion_direction",
        },
        "directional_diagnostics_not_hard_gates": {
            "object_equal_heldout_2d_improves": pooled_difference[
                "heldout_2d_scene_token_macro_soft_iou"
            ]
            > 0,
            "target_absent_normalized_prediction_mass_reduces": (
                normalized_absent["final_minus_frozen"] < 0
            ),
            "scene_macro_3d_does_not_regress": scene["soft_3d_miou"] >= 0,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_instance_oracle_training:
        raise PermissionError(
            "structured extent requires explicit supervised instance-oracle authorization"
        )
    if args.ablation_mode not in STRUCTURED_EXTENT_MODES:
        raise ValueError("unsupported structured extent ablation mode")
    for name in (
        "epoch_count",
        "unary_element_batch_size",
        "embedding_dimension",
        "edge_hidden_dimension",
        "edge_chunk_size",
        "maximum_edge_pairs_per_class",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        args.learning_rate <= 0
        or args.weight_decay < 0
        or args.gradient_clip_norm <= 0
        or not 0 <= args.dropout < 1
    ):
        raise ValueError("structured extent optimizer/dropout parameters are invalid")
    loss_weights = {
        "class_balanced_categorical": args.categorical_loss_weight,
        "object_equal_3d_soft_iou": args.object_equal_3d_loss_weight,
        "token_conditioned_edge_bce": args.token_edge_loss_weight,
        "log_full_mass_smooth_l1": args.log_full_mass_loss_weight,
        "heldout_present_macro_soft_iou": args.heldout_present_loss_weight,
        "continuous_absent_rms": args.continuous_absent_rms_loss_weight,
        "continuous_absent_mass": args.continuous_absent_mass_loss_weight,
    }
    if any(not np.isfinite(value) or value <= 0 for value in loss_weights.values()):
        raise ValueError("all structured extent loss weights must be positive")

    cache_payloads = [load_scene_cache(Path(value)) for value in args.scene_cache]
    by_id = {str(payload["scene_id"]): payload for payload in cache_payloads}
    if len(by_id) != len(cache_payloads):
        raise ValueError("duplicate structured extent scene-cache identity")
    training_scene_ids = set(map(str, args.training_scene))
    validation_scene_ids = set(map(str, args.validation_scene))
    if training_scene_ids | validation_scene_ids != set(by_id):
        raise ValueError("scene caches must exactly equal the structured protocol scenes")
    cohort_manifest = _load_cohort_manifest(
        args.cohort_manifest,
        protocol=args.protocol,
        training_scene_ids=training_scene_ids,
        validation_scene_ids=validation_scene_ids,
        epoch_count=args.epoch_count,
        allow_bounded_dev_protocol=args.allow_bounded_dev_protocol,
    )
    cache_receipts = _cache_receipts(cache_payloads)
    base_report, base_checkpoint, base_authority = _load_base_authority(
        report_path_value=args.base_report,
        checkpoint_path_value=args.base_checkpoint,
        cohort_manifest=cohort_manifest,
        selected_cache_receipts=cache_receipts,
        protocol=args.protocol,
        training_scene_ids=training_scene_ids,
        validation_scene_ids=validation_scene_ids,
    )
    configurations = [payload["configuration"] for payload in cache_payloads]
    if any(value != configurations[0] for value in configurations[1:]):
        raise ValueError("structured extent caches use different configurations")
    if tuple(configurations[0].get("local_feature_layout", ())) != (
        RGB_RADIO_GEOMETRY_LAYOUT
    ):
        raise ValueError("structured extent requires the sealed aligned F71 cache")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA structured training but CUDA is unavailable")
    source_tree = _source_tree_receipt()
    frozen_unary_model = _load_frozen_unary_model(base_checkpoint, device=device)
    temperature = float(base_report["training_configuration"]["temperature"])
    completion_confidence_cap = float(
        base_report["training_configuration"]["completion_confidence_cap"]
    )

    training_runtimes = [
        _prepare_runtime(by_id[scene_id]) for scene_id in sorted(training_scene_ids)
    ]
    for runtime in training_runtimes:
        runtime["frozen_unary"] = _frozen_unary_probabilities(
            frozen_unary_model,
            runtime,
            device=device,
            element_batch_size=args.unary_element_batch_size,
            temperature=temperature,
        )
    model = TokenConditionedStructuredExtent(
        feature_dimension=len(RGB_RADIO_GEOMETRY_LAYOUT),
        embedding_dimension=args.embedding_dimension,
        edge_hidden_dimension=args.edge_hidden_dimension,
        dropout=args.dropout,
        mode=args.ablation_mode,
        edge_chunk_size=args.edge_chunk_size,
    ).to(device)
    architecture_receipt = model.architecture_receipt()
    if int(architecture_receipt.get("iteration_count", -1)) != FIXED_ITERATION_COUNT:
        raise RuntimeError("structured extent architecture is not fixed to two iterations")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training = _fit(
        model,
        training_runtimes,
        device=device,
        epoch_count=args.epoch_count,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        completion_confidence_cap=completion_confidence_cap,
        loss_weights=loss_weights,
        maximum_edge_pairs_per_class=args.maximum_edge_pairs_per_class,
        seed=args.seed,
    )
    training_peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    if _source_tree_receipt()["source_tree_sha256"] != source_tree["source_tree_sha256"]:
        raise RuntimeError("structured extent source tree changed during training")

    # Validation runtimes are constructed only after the last optimizer step;
    # they are never passed to _fit or used for model choice.
    validation_runtimes = [
        _prepare_runtime(by_id[scene_id]) for scene_id in sorted(validation_scene_ids)
    ]
    for runtime in validation_runtimes:
        runtime["frozen_unary"] = _frozen_unary_probabilities(
            frozen_unary_model,
            runtime,
            device=device,
            element_batch_size=args.unary_element_batch_size,
            temperature=temperature,
        )
    del frozen_unary_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    per_scene = [
        _evaluate_scene(
            model,
            runtime,
            device=device,
            completion_confidence_cap=completion_confidence_cap,
        )
        for runtime in validation_runtimes
    ]
    evaluation_peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    if _source_tree_receipt()["source_tree_sha256"] != source_tree["source_tree_sha256"]:
        raise RuntimeError("structured extent source tree changed during evaluation")
    aggregate = _aggregate_evaluation(per_scene)
    paired = _paired_differences(aggregate)
    base_replay = _base_replay_receipt(
        per_scene, base_report, tolerance=args.base_replay_tolerance
    )
    bypass_error = max(
        record["structured_extent"][
            "bypass_max_probability_error_from_frozen_unary"
        ]
        for record in per_scene
    )
    if bypass_error != 0:
        raise RuntimeError("structured extent bypass changed the frozen unary")

    checkpoint_path = Path(args.output_checkpoint).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_configuration": {
            "feature_dimension": len(RGB_RADIO_GEOMETRY_LAYOUT),
            "embedding_dimension": args.embedding_dimension,
            "edge_hidden_dimension": args.edge_hidden_dimension,
            "dropout": args.dropout,
            "mode": args.ablation_mode,
            "edge_chunk_size": args.edge_chunk_size,
            "iteration_count": FIXED_ITERATION_COUNT,
            "parameter_count": parameter_count,
            "architecture_receipt": architecture_receipt,
        },
        "protocol": args.protocol,
        "training_scene_ids": sorted(training_scene_ids),
        "validation_scene_ids": sorted(validation_scene_ids),
        "validation_used_for_model_selection": False,
        "validation_targets_used_during_training": False,
        "integer_instance_ids_are_model_inputs": False,
        "training_target_authority": (
            "train_scene_labels_and_heldout_mesh_rasters_are_loss_targets_only"
        ),
        "loss_weights": loss_weights,
        "edge_supervision_policy": training["edge_supervision_policy"],
        "gradient_clipping_policy": training["gradient_clipping_policy"],
        "base_unary_authority": base_authority,
        "cohort_manifest": cohort_manifest,
        "scene_cache_receipts": cache_receipts,
        "source_tree": source_tree,
        "completion_confidence_cap": completion_confidence_cap,
        "training_peak_cuda_memory_bytes": training_peak_cuda_memory_bytes,
        "evaluation_peak_cuda_memory_bytes": evaluation_peak_cuda_memory_bytes,
        "evaluation_contract": {
            "methods": list(METHODS),
            "bypass": "direct_frozen_unary_no_structured_residual",
            "base_unary_replay": base_replay,
            "soft_iou_cohorts": list(SOFT_IOU_COHORTS),
            "target_absent_primary_diagnostic": (
                "continuous_prediction_mass_and_rms_not_strict_positive_support_count"
            ),
            "assignment": "threshold_free_token_plus_null_argmax",
        },
    }
    temporary_checkpoint = checkpoint_path.with_name(
        checkpoint_path.name
        + ".temporary."
        + hashlib.sha256(
            f"{args.seed}\0{args.ablation_mode}\0{source_tree['source_tree_sha256']}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
    )
    torch.save(checkpoint_payload, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)

    report = {
        "schema": REPORT_SCHEMA,
        "stage": "scene_disjoint_token_conditioned_structured_extent",
        "protocol": args.protocol,
        "formal16_result": args.protocol == "formal16",
        "ablation_mode": args.ablation_mode,
        "available_ablation_modes": list(STRUCTURED_EXTENT_MODES),
        "ablation_arms_are_separate_same_contract_runs": True,
        "oracle_identity_diagnostic_only": True,
        "association_isolated": True,
        "pointwise_unary_frozen": True,
        "fixed_iteration_count": FIXED_ITERATION_COUNT,
        "validation_used_for_model_selection": False,
        "validation_targets_used_during_training": False,
        "integer_instance_ids_are_model_inputs": False,
        "training_target_authority": (
            "train_scene_labels_and_heldout_mesh_rasters_are_loss_targets_only"
        ),
        "primary_assignment_decision": "threshold_free_token_plus_null_argmax",
        "benchmark_threshold_used_by_method": False,
        "hard_promotion_gate_applied": False,
        "false_positive_only_scene_token_count_is_promotion_direction": False,
        "target_absent_primary_diagnostic": (
            "continuous_prediction_mass_total_mean_rms_and_pixel_token_normalized"
        ),
        "absent_loss_contract": {
            "rms": "per_rendered_pixel_token_probability_rms",
            "mass": "mean_log1p_total_prediction_mass_per_target_absent_token",
            "strict_support_count_used_for_optimization": False,
        },
        "base_unary_authority": base_authority,
        "base_unary_replay": base_replay,
        "structured_extent_bypass_control": {
            "definition": "direct_frozen_unary_no_structured_residual",
            "maximum_probability_error": bypass_error,
        },
        "cohort_manifest": cohort_manifest,
        "scene_cache_receipts": cache_receipts,
        "scene_cache_record_list_sha256": _canonical_json_sha256(cache_receipts),
        "source_tree": source_tree,
        "structured_extent_architecture": architecture_receipt,
        "split": {
            "training_scene_ids": sorted(training_scene_ids),
            "validation_scene_ids": sorted(validation_scene_ids),
            "overlap": [],
            "physical_family_disjoint": True,
        },
        "carrier_and_observation_configuration": configurations[0],
        "training_configuration": {
            "seed": args.seed,
            "epoch_count": args.epoch_count,
            "optimizer_step_count": training["optimizer_step_count"],
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "gradient_clipping_policy": training["gradient_clipping_policy"],
            "device": str(device),
            "unary_element_batch_size": args.unary_element_batch_size,
            "unary_temperature_inherited_from_v10": temperature,
            "completion_confidence_cap_inherited_from_v10": completion_confidence_cap,
            "embedding_dimension": args.embedding_dimension,
            "edge_hidden_dimension": args.edge_hidden_dimension,
            "dropout": args.dropout,
            "edge_chunk_size": args.edge_chunk_size,
            "maximum_edge_pairs_per_class": args.maximum_edge_pairs_per_class,
            "iteration_count": FIXED_ITERATION_COUNT,
            "ablation_mode": args.ablation_mode,
            "model_parameter_count": parameter_count,
            "loss_weights": loss_weights,
            "edge_supervision_policy": training["edge_supervision_policy"],
            "mass_target_space": "physical_object_support_element_count",
            "dual_target_space": "pre_cap_K_plus_null_posterior",
            "training_peak_cuda_memory_bytes": training_peak_cuda_memory_bytes,
            "evaluation_peak_cuda_memory_bytes": evaluation_peak_cuda_memory_bytes,
        },
        "training": training,
        "per_validation_scene": per_scene,
        **aggregate,
        "paired_differences": paired,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "schema": CHECKPOINT_SCHEMA,
        },
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-cache", action="append", required=True)
    parser.add_argument("--training-scene", action="append", required=True)
    parser.add_argument("--validation-scene", action="append", required=True)
    parser.add_argument("--cohort-manifest", required=True)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="formal16")
    parser.add_argument("--allow-bounded-dev-protocol", action="store_true")
    parser.add_argument("--allow-instance-oracle-training", action="store_true")
    parser.add_argument("--ablation-mode", choices=STRUCTURED_EXTENT_MODES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--epoch-count", type=int, default=40)
    parser.add_argument("--unary-element-batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dimension", type=int, default=32)
    parser.add_argument("--edge-hidden-dimension", type=int, default=64)
    parser.add_argument("--edge-chunk-size", type=int, default=32768)
    parser.add_argument("--maximum-edge-pairs-per-class", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--categorical-loss-weight", type=float, default=1.0)
    parser.add_argument("--object-equal-3d-loss-weight", type=float, default=1.0)
    parser.add_argument("--token-edge-loss-weight", type=float, default=0.25)
    parser.add_argument("--log-full-mass-loss-weight", type=float, default=0.25)
    parser.add_argument("--heldout-present-loss-weight", type=float, default=1.0)
    parser.add_argument("--continuous-absent-rms-loss-weight", type=float, default=0.25)
    parser.add_argument("--continuous-absent-mass-loss-weight", type=float, default=0.1)
    parser.add_argument("--base-replay-tolerance", type=float, default=2e-5)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "ablation_mode": report["ablation_mode"],
                "scene_macro": report["scene_macro"],
                "paired_differences": report["paired_differences"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
