"""Train bounded surface message passing over a frozen v10 completion unary.

This is an independent v4 experiment.  It deliberately does not modify or
retrain the frozen pointwise completion implementation.  Integer ScanNet
instance identities are used only to form supervised targets on the training
scenes; the model receives a scene-specific token simplex, source-only local
features, geometry, and immutable observed membership facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.v4.completion import OracleIdentityCompletionMLP, completion_metrics
from radio_gs.v4.completion.message_passing import SurfaceMessagePassing
from radio_gs.v4.completion.scannet import camera_from_record, load_scene_cache
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.training.train_scannet_completion_oracle import (
    CHECKPOINT_SCHEMA as BASE_CHECKPOINT_SCHEMA,
    REPORT_SCHEMA as BASE_REPORT_SCHEMA,
    RGB_RADIO_GEOMETRY_LAYOUT,
    SOFT_IOU_COHORTS,
    _element_soft_iou_sufficient_statistics,
    _heldout_2d_metrics,
    _mean_scene_metric,
    _physical_scene_family,
    _pool_soft_iou_sufficient_statistics,
    _pooled_categorical_confusion,
    _runtime,
)


REPORT_SCHEMA = (
    "radio_gs.surface_object_memory_v4.scannet_completion_message_passing.v2"
)
CHECKPOINT_SCHEMA = (
    "radio_gs.surface_object_memory_v4.completion_message_passing_checkpoint.v2"
)
COHORT_SCHEMA = "radio_gs.surface_object_memory_v4.scannet_pfir_completion_cohort.v3"
PROTOCOLS = ("formal16", "bounded_dev")
MESSAGE_PASSING_STEPS = (2, 3)
METHODS = (
    "baseline_observed_only",
    "frozen_aligned_pointwise",
    "message_passing_bypass_control",
    "message_passing_completion",
)


@dataclass(frozen=True)
class DifferentiableProjection:
    """Frozen carrier-to-pixel map used by the held-out training loss."""

    numerator_element_ids: torch.Tensor
    numerator_pixel_ids: torch.Tensor
    numerator_weights: torch.Tensor
    denominator: torch.Tensor
    height: int
    width: int

    def __post_init__(self) -> None:
        element_ids = torch.as_tensor(self.numerator_element_ids, dtype=torch.long).cpu()
        pixel_ids = torch.as_tensor(self.numerator_pixel_ids, dtype=torch.long).cpu()
        weights = torch.as_tensor(self.numerator_weights, dtype=torch.float32).cpu()
        denominator = torch.as_tensor(self.denominator, dtype=torch.float32).cpu()
        if (
            element_ids.ndim != 1
            or pixel_ids.shape != element_ids.shape
            or weights.shape != element_ids.shape
        ):
            raise ValueError("projection numerator triplets must be aligned vectors")
        if denominator.shape != (int(self.height) * int(self.width),):
            raise ValueError("projection denominator must cover every pixel")
        if int(self.height) <= 0 or int(self.width) <= 0:
            raise ValueError("projection dimensions must be positive")
        if element_ids.numel() and (
            int(element_ids.min()) < 0
            or int(pixel_ids.min()) < 0
            or int(pixel_ids.max()) >= denominator.numel()
        ):
            raise ValueError("projection triplets are outside their domains")
        if not torch.isfinite(weights).all() or not torch.isfinite(denominator).all():
            raise ValueError("projection weights must be finite")
        if bool((weights < 0).any()) or bool((denominator < 0).any()):
            raise ValueError("projection weights must be non-negative")
        object.__setattr__(self, "numerator_element_ids", element_ids)
        object.__setattr__(self, "numerator_pixel_ids", pixel_ids)
        object.__setattr__(self, "numerator_weights", weights)
        object.__setattr__(self, "denominator", denominator)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _source_tree_receipt() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "radio_gs/v4/training/train_scannet_completion_message_passing.py",
        "radio_gs/v4/completion/message_passing.py",
        "radio_gs/v4/training/train_scannet_completion_oracle.py",
        "radio_gs/v4/completion/oracle.py",
        "radio_gs/v4/completion/scannet.py",
        "radio_gs/v4/carrier/sparse_voxel.py",
        "radio_gs/v4/carrier/base.py",
    )
    records = []
    for relative_path in relative_paths:
        path = (repo_root / relative_path).resolve(strict=True)
        records.append({"path": relative_path, "sha256": sha256_file(path)})
    return {
        "schema": "radio_gs.source_tree_explicit_closure.v1",
        "repository_root": str(repo_root),
        "files": records,
        "source_tree_sha256": _canonical_json_sha256(records),
    }


def _validate_protocol_split(
    *,
    protocol: str,
    training_scene_ids: set[str],
    validation_scene_ids: set[str],
    cohort_training_scene_ids: set[str],
    cohort_validation_scene_ids: set[str],
    epoch_count: int,
    allow_bounded_dev_protocol: bool,
) -> None:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported message-passing protocol {protocol!r}")
    if not training_scene_ids or not validation_scene_ids:
        raise ValueError("message passing requires training and validation scenes")
    if training_scene_ids & validation_scene_ids:
        raise ValueError("message-passing training and validation scenes overlap")
    training_families = {_physical_scene_family(value) for value in training_scene_ids}
    validation_families = {
        _physical_scene_family(value) for value in validation_scene_ids
    }
    if training_families & validation_families:
        raise ValueError("message-passing physical scene families overlap")
    if protocol == "formal16":
        if allow_bounded_dev_protocol:
            raise ValueError("formal16 must not carry bounded-development authorization")
        if (
            len(cohort_training_scene_ids) != 12
            or len(cohort_validation_scene_ids) != 4
            or training_scene_ids != cohort_training_scene_ids
            or validation_scene_ids != cohort_validation_scene_ids
        ):
            raise ValueError("formal16 requires the complete frozen 12/4 cohort split")
        return
    if not allow_bounded_dev_protocol:
        raise PermissionError(
            "bounded_dev requires explicit --allow-bounded-dev-protocol"
        )
    if (
        not training_scene_ids <= cohort_training_scene_ids
        or not validation_scene_ids <= cohort_validation_scene_ids
    ):
        raise ValueError("bounded_dev scenes must remain on their frozen cohort side")
    if len(training_scene_ids) > 2 or len(validation_scene_ids) > 1 or epoch_count > 2:
        raise ValueError("bounded_dev is capped at 2 train, 1 validation, and 2 epochs")


def _load_cohort_manifest(
    path_value: str,
    *,
    protocol: str,
    training_scene_ids: set[str],
    validation_scene_ids: set[str],
    epoch_count: int,
    allow_bounded_dev_protocol: bool,
) -> dict[str, Any]:
    path = Path(path_value).resolve(strict=True)
    payload = json.loads(path.read_text())
    if payload.get("schema") != COHORT_SCHEMA:
        raise ValueError("message passing requires the strict completion cohort v3")
    scene_ids = list(map(str, payload.get("scene_ids", ())))
    split = payload.get("split", {})
    cohort_training = set(map(str, split.get("training_scene_ids", ())))
    cohort_validation = set(map(str, split.get("validation_scene_ids", ())))
    if (
        len(scene_ids) != 16
        or len(scene_ids) != len(set(scene_ids))
        or set(scene_ids) != cohort_training | cohort_validation
        or cohort_training & cohort_validation
        or split.get("physical_family_disjoint") is not True
    ):
        raise ValueError("completion cohort manifest is not the frozen formal16 split")
    _validate_protocol_split(
        protocol=protocol,
        training_scene_ids=training_scene_ids,
        validation_scene_ids=validation_scene_ids,
        cohort_training_scene_ids=cohort_training,
        cohort_validation_scene_ids=cohort_validation,
        epoch_count=epoch_count,
        allow_bounded_dev_protocol=allow_bounded_dev_protocol,
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": COHORT_SCHEMA,
        "scene_ids": sorted(scene_ids),
        "training_scene_ids": sorted(cohort_training),
        "validation_scene_ids": sorted(cohort_validation),
        "selection_policy": payload.get("selection", {}).get("policy"),
        "selection_salt": payload.get("selection", {}).get("salt"),
        "validation_selection_salt": split.get("validation_selection_salt"),
    }


def _cache_receipts(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for payload in sorted(payloads, key=lambda value: str(value["scene_id"])):
        geometry_receipt = payload.get("geometry_receipt", {})
        if geometry_receipt.get("target_rgb_opened") is not False:
            raise ValueError("message-passing cache does not prove target-RGB isolation")
        records.append(
            {
                "scene_id": str(payload["scene_id"]),
                "cache_path": str(Path(payload["cache_path"]).resolve(strict=True)),
                "cache_sha256": str(payload["cache_sha256"]),
                "sealed_inputs": payload["input_receipt"],
                "target_rgb_opened": False,
            }
        )
    return records


def _load_base_authority(
    *,
    report_path_value: str,
    checkpoint_path_value: str,
    cohort_manifest: dict[str, Any],
    selected_cache_receipts: list[dict[str, Any]],
    protocol: str,
    training_scene_ids: set[str],
    validation_scene_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report_path = Path(report_path_value).resolve(strict=True)
    checkpoint_path = Path(checkpoint_path_value).resolve(strict=True)
    report = json.loads(report_path.read_text())
    if report.get("schema") != BASE_REPORT_SCHEMA:
        raise ValueError("base unary report is not the frozen v10 schema")
    if (
        report.get("radio_alignment_control") != "aligned"
        or report.get("local_feature_mode") != "rgb_radio_geometry"
        or report.get("unknown_sampling_mode") != "token_uniform"
        or report.get("scoring_mode") != "mlp"
    ):
        raise ValueError("base unary must be the aligned F71 pointwise reference arm")
    if report.get("cohort_manifest", {}).get("sha256") != cohort_manifest["sha256"]:
        raise ValueError("base unary and message passing use different cohort manifests")
    report_split = report.get("split", {})
    base_training = set(map(str, report_split.get("training_scene_ids", ())))
    base_validation = set(map(str, report_split.get("validation_scene_ids", ())))
    cohort_training = set(cohort_manifest["training_scene_ids"])
    cohort_validation = set(cohort_manifest["validation_scene_ids"])
    if base_training != cohort_training or base_validation != cohort_validation:
        raise ValueError("base unary report does not use the frozen formal16 split")
    if protocol == "formal16" and (
        training_scene_ids != base_training or validation_scene_ids != base_validation
    ):
        raise ValueError("formal message passing must exactly reuse the base split")
    if not training_scene_ids <= base_training or not validation_scene_ids <= base_validation:
        raise ValueError("development subset crosses the base unary split")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    report_checkpoint = report.get("checkpoint", {})
    if report_checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("base unary checkpoint SHA differs from its report")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("safe weights-only checkpoint loading is required") from error
    if checkpoint.get("schema") != BASE_CHECKPOINT_SCHEMA:
        raise ValueError("base unary checkpoint is not the frozen v10 schema")
    if (
        checkpoint.get("radio_alignment_control") != "aligned"
        or checkpoint.get("local_feature_mode") != "rgb_radio_geometry"
        or checkpoint.get("scoring_mode") != "mlp"
        or checkpoint.get("unknown_sampling_mode") != "token_uniform"
        or checkpoint.get("cohort_manifest", {}).get("sha256")
        != cohort_manifest["sha256"]
    ):
        raise ValueError("base unary checkpoint is not the aligned F71 reference")
    model_configuration = checkpoint.get("model_configuration", {})
    if (
        tuple(model_configuration.get("selected_local_feature_layout", ()))
        != RGB_RADIO_GEOMETRY_LAYOUT
        or model_configuration.get("radio_alignment_control") != "aligned"
    ):
        raise ValueError("base unary checkpoint has the wrong feature contract")
    repo_root = Path(__file__).resolve().parents[3]
    frozen_trainer_sha = sha256_file(
        repo_root / "radio_gs/v4/training/train_scannet_completion_oracle.py"
    )
    frozen_oracle_sha = sha256_file(repo_root / "radio_gs/v4/completion/oracle.py")
    if (
        report.get("implementation_sha256") != frozen_trainer_sha
        or report.get("completion_implementation_sha256") != frozen_oracle_sha
        or checkpoint.get("implementation_sha256") != frozen_trainer_sha
        or checkpoint.get("completion_implementation_sha256") != frozen_oracle_sha
    ):
        raise ValueError("frozen unary source hashes no longer match its artifacts")
    selected_by_id = {row["scene_id"]: row for row in selected_cache_receipts}
    base_by_id = {
        str(row["scene_id"]): row for row in report.get("scene_cache_receipts", ())
    }
    if not selected_by_id.keys() <= base_by_id.keys():
        raise ValueError("base report is missing selected scene-cache receipts")
    for scene_id, selected in selected_by_id.items():
        base = base_by_id[scene_id]
        if (
            str(base.get("cache_sha256")) != selected["cache_sha256"]
            or Path(base.get("cache_path", "")).resolve()
            != Path(selected["cache_path"]).resolve()
        ):
            raise ValueError(f"base unary cache differs for {scene_id}")
    training_configuration = report.get("training_configuration", {})
    for field, expected in (
        ("temperature", 0.5),
        ("completion_confidence_cap", 0.95),
    ):
        if float(training_configuration.get(field, -1)) != expected:
            raise ValueError(f"frozen base unary changed {field}")
    receipt = {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "report_schema": BASE_REPORT_SCHEMA,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_schema": BASE_CHECKPOINT_SCHEMA,
        "implementation_sha256": frozen_trainer_sha,
        "completion_implementation_sha256": frozen_oracle_sha,
        "only_role": "frozen_aligned_radio_pointwise_unary",
        "parameters_trainable_in_message_passing": False,
    }
    return report, checkpoint, receipt


def _load_frozen_unary_model(
    checkpoint: dict[str, Any], *, device: torch.device
) -> OracleIdentityCompletionMLP:
    configuration = checkpoint["model_configuration"]
    model = OracleIdentityCompletionMLP(
        int(configuration["input_dimension"]),
        int(configuration["hidden_dimension"]),
        float(configuration["dropout"]),
        explicit_similarity_residual=False,
        availability_conditioned_experts=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def _clamp_contract(runtime: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    partial = runtime["partial"]
    token_count = int(partial.positive.shape[1])
    clamp_mask = partial.element_is_observed | ~partial.eligible_elements
    probabilities = torch.zeros(
        partial.positive.shape[0], token_count + 1, dtype=torch.float32
    )
    probabilities[:, token_count] = 1.0
    positive_count = partial.positive.sum(-1)
    if bool((positive_count > 1).any()):
        raise ValueError("an observed carrier row cannot clamp multiple object tokens")
    observed_object = clamp_mask & partial.eligible_elements & (positive_count == 1)
    if bool(observed_object.any()):
        probabilities[observed_object, token_count] = 0.0
        probabilities[observed_object, :token_count] = partial.positive[
            observed_object
        ].float()
    if not torch.equal(
        probabilities[clamp_mask].sum(-1),
        torch.ones(int(clamp_mask.sum()), dtype=torch.float32),
    ):
        raise RuntimeError("observed clamp is not a categorical simplex")
    return clamp_mask, probabilities


@torch.no_grad()
def _frozen_unary_probabilities(
    model: OracleIdentityCompletionMLP,
    runtime: dict[str, Any],
    *,
    device: torch.device,
    element_batch_size: int,
    temperature: float,
) -> torch.Tensor:
    if element_batch_size <= 0 or temperature <= 0:
        raise ValueError("unary batch size and temperature must be positive")
    partial = runtime["partial"]
    token_count = int(partial.positive.shape[1])
    clamp_mask, clamp_probabilities = _clamp_contract(runtime)
    probabilities = clamp_probabilities.clone()
    unknown_indices = torch.where(partial.unknown.any(-1))[0]
    from radio_gs.v4.completion import build_pair_features

    for start in range(0, unknown_indices.numel(), element_batch_size):
        indices = unknown_indices[start : start + element_batch_size]
        pair = build_pair_features(
            runtime["centres"],
            runtime["local_features"],
            runtime["context"],
            indices,
            minimum_scale=runtime["minimum_scale"],
        ).to(device, non_blocking=True)
        categorical = torch.softmax(model.categorical_logits(pair) / temperature, -1)
        probabilities[indices] = categorical.cpu()
    if not torch.allclose(
        probabilities.sum(-1), torch.ones(probabilities.shape[0]), atol=1e-5
    ):
        raise RuntimeError("frozen unary does not form a K-plus-null simplex")
    if not torch.equal(probabilities[clamp_mask], clamp_probabilities[clamp_mask]):
        raise RuntimeError("frozen unary changed observed facts")
    return probabilities


def _posterior_to_membership(
    posterior: torch.Tensor,
    runtime: dict[str, Any],
    *,
    completion_confidence_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < completion_confidence_cap < 1:
        raise ValueError("completion confidence cap must be in (0, 1)")
    probabilities = torch.as_tensor(posterior)
    partial = runtime["partial"]
    token_count = int(partial.positive.shape[1])
    if probabilities.shape != (partial.positive.shape[0], token_count + 1):
        raise ValueError("message posterior must have shape [N, K+1]")
    if not torch.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise ValueError("message posterior must be finite and non-negative")
    if not torch.allclose(
        probabilities.sum(-1),
        torch.ones(probabilities.shape[0], device=probabilities.device),
        atol=1e-5,
    ):
        raise ValueError("message posterior must form a K-plus-null simplex")
    device = probabilities.device
    unknown = partial.unknown.any(-1).to(device)
    eligible = partial.eligible_elements.to(device)
    positive = partial.positive.to(device)
    membership = torch.zeros_like(probabilities[:, :token_count])
    membership = torch.where(
        unknown[:, None],
        probabilities[:, :token_count] * completion_confidence_cap,
        membership,
    )
    membership = torch.where(positive, torch.ones_like(membership), membership)
    membership = torch.where(eligible[:, None], membership, torch.zeros_like(membership))
    null = torch.ones(probabilities.shape[0], device=device, dtype=probabilities.dtype)
    null[eligible] = 1.0 - membership[eligible].sum(-1)
    if not torch.allclose(
        membership[eligible].sum(-1) + null[eligible],
        torch.ones(int(eligible.sum()), device=device),
        atol=1e-5,
    ):
        raise RuntimeError("completed membership lost its K-plus-null simplex")
    return membership, null


def _unknown_categorical_loss(
    posterior: torch.Tensor, runtime: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, int]]:
    labels = torch.as_tensor(runtime["labels"], dtype=torch.long, device=posterior.device)
    unknown = runtime["partial"].unknown.any(-1).to(posterior.device)
    token_count = posterior.shape[1] - 1
    target = labels[unknown].clone()
    target[target < 0] = token_count
    if target.numel() == 0:
        raise ValueError("categorical training requires unknown eligible elements")
    log_probability = posterior[unknown].clamp_min(1e-12).log()
    per_element = F.nll_loss(log_probability, target, reduction="none")
    present_classes = torch.unique(target, sorted=True)
    class_losses = torch.stack(
        [per_element[target == class_index].mean() for class_index in present_classes]
    )
    return class_losses.mean(), {
        "unknown_element_count": int(target.numel()),
        "present_categorical_class_count": int(present_classes.numel()),
        "unknown_object_element_count": int((target < token_count).sum()),
        "unknown_null_element_count": int((target == token_count).sum()),
    }


def _edge_same_instance_bce(
    edge_logits: torch.Tensor,
    edge_index: torch.Tensor,
    runtime: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, int]]:
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=edge_logits.device)
    if edges.ndim != 2 or edges.shape[0] != 2 or edge_logits.shape != (edges.shape[1],):
        raise ValueError("edge logits must align with directed carrier edges")
    labels = torch.as_tensor(runtime["labels"], dtype=torch.long, device=edge_logits.device)
    eligible = runtime["partial"].eligible_elements.to(edge_logits.device)
    source, destination = edges
    used = eligible[source] & eligible[destination]
    logits = edge_logits[used]
    target = (
        (labels[source[used]] >= 0)
        & (labels[source[used]] == labels[destination[used]])
    )
    positive = target
    negative = ~target
    pieces = []
    if bool(positive.any()):
        pieces.append(F.binary_cross_entropy_with_logits(logits[positive], torch.ones_like(logits[positive])))
    if bool(negative.any()):
        pieces.append(F.binary_cross_entropy_with_logits(logits[negative], torch.zeros_like(logits[negative])))
    if not pieces:
        raise ValueError("edge supervision requires eligible directed edges")
    return torch.stack(pieces).mean(), {
        "directed_edge_count": int(edge_logits.numel()),
        "supervised_directed_edge_count": int(used.sum()),
        "same_retained_instance_edge_count": int(positive.sum()),
        "different_or_null_edge_count": int(negative.sum()),
    }


def _projection_from_runtime(
    runtime: dict[str, Any], camera_record: dict[str, Any]
) -> DifferentiableProjection:
    projection = runtime["carrier"].project(camera_from_record(camera_record))
    eligible = runtime["partial"].eligible_elements[projection.element_ids]
    denominator = torch.zeros(projection.num_pixels, dtype=torch.float32)
    denominator.index_add_(0, projection.pixel_ids, projection.weights)
    return DifferentiableProjection(
        numerator_element_ids=projection.element_ids[eligible],
        numerator_pixel_ids=projection.pixel_ids[eligible],
        numerator_weights=projection.weights[eligible],
        denominator=denominator,
        height=projection.height,
        width=projection.width,
    )


def _render_differentiable(
    membership: torch.Tensor, projection: DifferentiableProjection
) -> torch.Tensor:
    device = membership.device
    element_ids = projection.numerator_element_ids.to(device, non_blocking=True)
    pixel_ids = projection.numerator_pixel_ids.to(device, non_blocking=True)
    weights = projection.numerator_weights.to(device, non_blocking=True)
    denominator = projection.denominator.to(device, non_blocking=True)
    numerator = torch.zeros(
        denominator.numel(), membership.shape[1], device=device, dtype=membership.dtype
    )
    if element_ids.numel():
        numerator.index_add_(
            0, pixel_ids, membership[element_ids] * weights[:, None]
        )
    return (numerator / denominator.clamp_min(1e-12)[:, None]).reshape(
        projection.height, projection.width, membership.shape[1]
    )


def _heldout_render_losses(
    membership: torch.Tensor,
    projections: list[DifferentiableProjection],
    targets: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
    if not projections or len(projections) != len(targets):
        raise ValueError("held-out render loss requires aligned views and targets")
    token_count = membership.shape[1]
    intersection = torch.zeros(token_count, device=membership.device)
    prediction_mass = torch.zeros_like(intersection)
    target_mass = torch.zeros_like(intersection)
    pixel_count = 0
    for projection, target_value in zip(projections, targets):
        prediction = _render_differentiable(membership, projection).reshape(-1, token_count)
        target = torch.as_tensor(
            target_value, dtype=prediction.dtype, device=prediction.device
        ).reshape(-1, token_count)
        if prediction.shape != target.shape:
            raise ValueError("held-out rendered prediction and mesh target do not align")
        intersection = intersection + (prediction * target).sum(0)
        prediction_mass = prediction_mass + prediction.sum(0)
        target_mass = target_mass + target.sum(0)
        pixel_count += prediction.shape[0]
    present = target_mass > 0
    if not bool(present.any()):
        raise ValueError("held-out render supervision contains no target-present token")
    union = prediction_mass + target_mass - intersection
    soft_iou = intersection[present] / union[present].clamp_min(1e-12)
    soft_iou_loss = 1.0 - soft_iou.mean()
    absent = ~present
    if bool(absent.any()):
        absent_fp_mass = prediction_mass[absent].sum() / (
            float(pixel_count) * float(absent.sum())
        )
    else:
        absent_fp_mass = prediction_mass.sum() * 0.0
    return soft_iou_loss, absent_fp_mass, {
        "heldout_view_count": len(projections),
        "heldout_pixel_count": int(pixel_count),
        "target_present_token_count": int(present.sum()),
        "target_absent_token_count": int(absent.sum()),
        "false_positive_mass_normalizer": float(pixel_count * max(int(absent.sum()), 1)),
    }


def _prepare_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = _runtime(payload, "rgb_radio_geometry")
    runtime["edge_index"] = runtime["carrier"].neighbors().edge_index
    runtime["heldout_projections"] = [
        _projection_from_runtime(runtime, record)
        for record in payload["heldout_cameras"]
    ]
    return runtime


def _message_forward(
    model: SurfaceMessagePassing,
    runtime: dict[str, Any],
    unary_probabilities: torch.Tensor,
    *,
    device: torch.device,
) -> Any:
    clamp_mask, clamp_probabilities = _clamp_contract(runtime)
    return model(
        unary_probabilities=unary_probabilities.to(device, non_blocking=True),
        edge_index=runtime["edge_index"].to(device, non_blocking=True),
        centres=runtime["centres"].to(device, non_blocking=True),
        normals=torch.as_tensor(runtime["payload"]["normals"], dtype=torch.float32).to(
            device, non_blocking=True
        ),
        local_features=runtime["local_features"].to(device, non_blocking=True),
        source_visible=runtime["source_visible"].to(device, non_blocking=True),
        clamp_mask=clamp_mask.to(device, non_blocking=True),
        clamp_probabilities=clamp_probabilities.to(device, non_blocking=True),
        voxel_size=float(runtime["minimum_scale"]),
    )


def _message_passing_bypass_control(unary_probabilities: torch.Tensor) -> torch.Tensor:
    """Bypass every learned residual while retaining an auditable tensor copy."""

    unary = torch.as_tensor(unary_probabilities)
    if unary.ndim != 2 or unary.shape[1] < 2 or not unary.is_floating_point():
        raise ValueError("bypass unary must have shape [N, K+1]")
    if not torch.isfinite(unary).all() or bool((unary < 0).any()):
        raise ValueError("bypass unary must be finite and non-negative")
    if not torch.allclose(unary.sum(-1), torch.ones(unary.shape[0]), atol=1e-5):
        raise ValueError("bypass unary must lie on the K-plus-null simplex")
    return unary.detach().clone()


def _fit(
    model: SurfaceMessagePassing,
    training_runtimes: list[dict[str, Any]],
    *,
    device: torch.device,
    epoch_count: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    categorical_loss_weight: float,
    edge_loss_weight: float,
    heldout_soft_iou_loss_weight: float,
    absent_token_fp_loss_weight: float,
    completion_confidence_cap: float,
    seed: int,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    python_rng = random.Random(seed + 17)
    history = []
    edge_audit_by_scene: dict[str, dict[str, int]] = {}
    categorical_audit_by_scene: dict[str, dict[str, int]] = {}
    model.train()
    for epoch in range(epoch_count):
        order = list(range(len(training_runtimes)))
        python_rng.shuffle(order)
        epoch_values: dict[str, list[float]] = {
            "total": [],
            "unknown_categorical": [],
            "edge_same_instance_bce": [],
            "heldout_soft_iou": [],
            "target_absent_fp_mass": [],
        }
        for runtime_index in order:
            runtime = training_runtimes[runtime_index]
            output = _message_forward(
                model, runtime, runtime["frozen_unary"], device=device
            )
            categorical_loss, categorical_audit = _unknown_categorical_loss(
                output.probabilities, runtime
            )
            edge_loss, edge_audit = _edge_same_instance_bce(
                output.edge_logits, runtime["edge_index"], runtime
            )
            scene_id = str(runtime["payload"]["scene_id"])
            if scene_id in edge_audit_by_scene and edge_audit_by_scene[scene_id] != edge_audit:
                raise RuntimeError("edge supervision counts changed across epochs")
            if (
                scene_id in categorical_audit_by_scene
                and categorical_audit_by_scene[scene_id] != categorical_audit
            ):
                raise RuntimeError("categorical supervision counts changed across epochs")
            edge_audit_by_scene[scene_id] = edge_audit
            categorical_audit_by_scene[scene_id] = categorical_audit
            membership, _ = _posterior_to_membership(
                output.probabilities,
                runtime,
                completion_confidence_cap=completion_confidence_cap,
            )
            heldout_loss, absent_fp_loss, _ = _heldout_render_losses(
                membership,
                runtime["heldout_projections"],
                runtime["payload"]["heldout_mesh_target_rasters"],
            )
            total = (
                categorical_loss_weight * categorical_loss
                + edge_loss_weight * edge_loss
                + heldout_soft_iou_loss_weight * heldout_loss
                + absent_token_fp_loss_weight * absent_fp_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                raise RuntimeError("message-passing gradient norm is non-finite")
            optimizer.step()
            for name, value in (
                ("total", total),
                ("unknown_categorical", categorical_loss),
                ("edge_same_instance_bce", edge_loss),
                ("heldout_soft_iou", heldout_loss),
                ("target_absent_fp_mass", absent_fp_loss),
            ):
                epoch_values[name].append(float(value.detach()))
        history.append(
            {
                "epoch": epoch + 1,
                **{
                    f"mean_{name}_loss": float(np.mean(values))
                    for name, values in epoch_values.items()
                },
            }
        )
    return {
        "epoch_history": history,
        "optimizer_step_count": epoch_count * len(training_runtimes),
        "final_epoch": history[-1],
        "edge_supervision_audit_by_training_scene": edge_audit_by_scene,
        "edge_supervision_counts_constant_across_epochs": True,
        "edge_supervision_total_counts": {
            key: sum(row[key] for row in edge_audit_by_scene.values())
            for key in next(iter(edge_audit_by_scene.values()))
        },
        "categorical_supervision_audit_by_training_scene": (
            categorical_audit_by_scene
        ),
    }


def _membership_metrics(
    runtime: dict[str, Any], membership: torch.Tensor, null: torch.Tensor
) -> dict[str, Any]:
    membership = membership.detach().float().cpu()
    null = null.detach().float().cpu()
    metrics = completion_metrics(
        membership,
        runtime["partial"],
        runtime["labels"],
        null_probability=null,
        assignment_threshold=0.5,
        unknown_strata=runtime.get("unknown_strata"),
    )
    statistics = _element_soft_iou_sufficient_statistics(runtime, membership)
    metrics.update(_heldout_2d_metrics(runtime, membership))
    statistics["heldout_2d"] = metrics.pop(
        "heldout_2d_soft_iou_sufficient_statistics"
    )
    metrics["soft_iou_sufficient_statistics"] = statistics
    return metrics


@torch.no_grad()
def _evaluate_scene(
    model: SurfaceMessagePassing,
    runtime: dict[str, Any],
    *,
    device: torch.device,
    completion_confidence_cap: float,
) -> dict[str, Any]:
    observed_membership = runtime["partial"].positive.float()
    observed_null = 1.0 - observed_membership.sum(-1)
    frozen_membership, frozen_null = _posterior_to_membership(
        runtime["frozen_unary"],
        runtime,
        completion_confidence_cap=completion_confidence_cap,
    )
    output = _message_forward(
        model, runtime, runtime["frozen_unary"], device=device
    )
    bypass_probabilities = _message_passing_bypass_control(runtime["frozen_unary"])
    message_membership, message_null = _posterior_to_membership(
        output.probabilities,
        runtime,
        completion_confidence_cap=completion_confidence_cap,
    )
    bypass_membership, bypass_null = _posterior_to_membership(
        bypass_probabilities,
        runtime,
        completion_confidence_cap=completion_confidence_cap,
    )
    clamp_mask, clamp_probabilities = _clamp_contract(runtime)
    clamp_error = (
        output.probabilities[clamp_mask.to(device)]
        - clamp_probabilities[clamp_mask].to(device)
    ).abs().max()
    extent_cohorts = {
        "unknown": runtime["partial"].unknown.any(-1),
        "visible_but_unmasked": runtime["unknown_strata"]["visible_but_unmasked"],
        "never_visible": runtime["unknown_strata"]["never_visible"],
    }
    extent_diagnostics = {}
    for name, mask_value in extent_cohorts.items():
        mask = mask_value.to(device)
        extent_diagnostics[name] = {
            "element_count": int(mask.sum()),
            "mean_seed_reachability_over_element_tokens": (
                float(output.seed_reachability[mask].mean()) if bool(mask.any()) else 0.0
            ),
            "mean_soft_extent_weight_over_element_tokens": (
                float(output.extent_weights[mask].mean()) if bool(mask.any()) else 0.0
            ),
        }
    return {
        "scene_id": str(runtime["payload"]["scene_id"]),
        "element_count": int(runtime["centres"].shape[0]),
        "token_count": int(runtime["partial"].positive.shape[1]),
        "heldout_pixel_count": int(
            sum(
                camera_from_record(record).height
                * camera_from_record(record).width
                for record in runtime["payload"]["heldout_cameras"]
            )
        ),
        "baseline_observed_only": _membership_metrics(
            runtime, observed_membership, observed_null
        ),
        "frozen_aligned_pointwise": _membership_metrics(
            runtime, frozen_membership, frozen_null
        ),
        "message_passing_bypass_control": _membership_metrics(
            runtime, bypass_membership, bypass_null
        ),
        "message_passing_completion": _membership_metrics(
            runtime, message_membership, message_null
        ),
        "message_passing": {
            "step_count": len(output.step_probabilities),
            "observed_clamp_max_error": float(clamp_error),
            "edge_weight_mean": float(output.edge_weights.mean())
            if output.edge_weights.numel()
            else 0.0,
            "edge_weight_min": float(output.edge_weights.min())
            if output.edge_weights.numel()
            else 0.0,
            "edge_weight_max": float(output.edge_weights.max())
            if output.edge_weights.numel()
            else 0.0,
            "step_strengths": list(map(float, output.step_strengths)),
            "extent_gate_strengths": list(map(float, output.extent_gate_strengths)),
            "soft_extent_diagnostics": extent_diagnostics,
            "bypass_control_max_probability_error_from_frozen_unary": float(
                (bypass_probabilities - runtime["frozen_unary"]).abs().max()
            ),
        },
    }


AGGREGATE_KEYS = (
    "soft_3d_miou",
    "unknown_only_soft_3d_miou",
    "heldout_2d_soft_miou",
    "heldout_2d_cross_view_token_soft_miou",
    "full_object_token_top1_accuracy",
    "full_k_plus_null_categorical_accuracy",
    "token_probability_concentration",
    "target_aware_token_mass_precision",
    "unknown_target_aware_token_mass_precision",
    "unknown_assignment_precision",
    "unknown_retained_object_coverage",
    "unknown_correct_assignment_recall",
    "assigned_unknown_object_top1_accuracy",
    "unknown_retained_set_null_recall",
    "known_element_fraction",
    "retained_set_false_positive_mean",
    "visible_but_unmasked_soft_3d_miou",
    "visible_but_unmasked_assignment_precision",
    "visible_but_unmasked_retained_object_coverage",
    "visible_but_unmasked_correct_assignment_recall",
    "visible_but_unmasked_assigned_object_top1_accuracy",
    "visible_but_unmasked_retained_set_null_recall",
    "never_visible_soft_3d_miou",
    "never_visible_assignment_precision",
    "never_visible_retained_object_coverage",
    "never_visible_correct_assignment_recall",
    "never_visible_assigned_object_top1_accuracy",
    "never_visible_retained_set_null_recall",
)


def _target_absent_prediction_mass(
    records: list[dict[str, Any]], method: str
) -> dict[str, Any]:
    """Aggregate continuous false-positive mass on target-absent scene tokens."""

    if not records:
        raise ValueError("target-absent prediction mass requires validation scenes")
    prediction_mass = 0.0
    absent_scene_token_count = 0
    positive_support_count = 0
    heldout_pixel_token_normalizer = 0
    for record in records:
        pixel_count = int(record.get("heldout_pixel_count", 0))
        if pixel_count <= 0:
            raise ValueError("heldout_pixel_count must be positive for every scene")
        try:
            statistics = record[method]["soft_iou_sufficient_statistics"][
                "heldout_2d"
            ]
        except KeyError as error:
            raise KeyError(
                f"missing {method}/heldout_2d sufficient statistics"
            ) from error
        if statistics.get("numeric_dtype") != "float64":
            raise ValueError("target-absent mass requires float64 sufficient statistics")
        token_rows = statistics.get("token_statistics", ())
        if int(statistics.get("token_count", -1)) != len(token_rows):
            raise ValueError("heldout token statistics have an inconsistent token count")
        for token in token_rows:
            current_prediction_mass = float(token["prediction_mass"])
            target_mass = float(token["target_mass"])
            if (
                not np.isfinite(current_prediction_mass)
                or not np.isfinite(target_mass)
                or current_prediction_mass < 0
                or target_mass < 0
            ):
                raise ValueError("heldout prediction and target masses must be finite")
            if target_mass != 0:
                continue
            prediction_mass += current_prediction_mass
            absent_scene_token_count += 1
            positive_support_count += int(current_prediction_mass > 0)
            heldout_pixel_token_normalizer += pixel_count
    mean_per_absent = (
        prediction_mass / absent_scene_token_count
        if absent_scene_token_count
        else 0.0
    )
    normalized_mass = (
        prediction_mass / heldout_pixel_token_normalizer
        if heldout_pixel_token_normalizer
        else 0.0
    )
    return {
        "schema": (
            "radio_gs.surface_object_memory_v4."
            "target_absent_heldout_prediction_mass.v1"
        ),
        "numeric_dtype": "float64",
        "target_absent_definition": "heldout_cross_view_target_mass_exactly_zero",
        "total_prediction_mass": prediction_mass,
        "target_absent_scene_token_count": absent_scene_token_count,
        "mean_prediction_mass_per_target_absent_scene_token": mean_per_absent,
        "heldout_pixel_token_normalizer": heldout_pixel_token_normalizer,
        "prediction_mass_per_target_absent_heldout_pixel_token": normalized_mass,
        "strict_positive_support_scene_token_count": positive_support_count,
        "strict_positive_support_fraction": (
            positive_support_count / absent_scene_token_count
            if absent_scene_token_count
            else 0.0
        ),
        "promotion_direction": "lower_continuous_prediction_mass",
    }


def _target_absent_mass_comparison(
    by_method: dict[str, dict[str, Any]],
    *,
    frozen_method: str = "frozen_aligned_pointwise",
    final_method: str = "message_passing_completion",
) -> dict[str, Any]:
    frozen = by_method[frozen_method]
    final = by_method[final_method]
    for field in (
        "target_absent_scene_token_count",
        "heldout_pixel_token_normalizer",
    ):
        if int(frozen[field]) != int(final[field]):
            raise ValueError(
                "frozen and final target-absent mass use different evaluation domains"
            )
    metrics = (
        "total_prediction_mass",
        "mean_prediction_mass_per_target_absent_scene_token",
        "prediction_mass_per_target_absent_heldout_pixel_token",
    )
    comparison = {}
    for metric in metrics:
        frozen_value = float(frozen[metric])
        final_value = float(final[metric])
        comparison[metric] = {
            "frozen_aligned_pointwise": frozen_value,
            "message_passing_completion": final_value,
            "final_minus_frozen": final_value - frozen_value,
            "relative_reduction_fraction": (
                (frozen_value - final_value) / frozen_value
                if frozen_value > 0
                else None
            ),
        }
    return {
        "promotion_direction": "positive_relative_reduction_fraction",
        "metrics": comparison,
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
    target_absent_mass = {
        method: _target_absent_prediction_mass(records, method)
        for method in METHODS
    }
    return {
        "scene_macro": scene_macro,
        "pooled_soft_iou_sufficient_statistics": pooled_soft_iou,
        "pooled_unknown_categorical_confusion": pooled_unknown,
        "pooled_unknown_strata_categorical_confusion": pooled_strata,
        "target_absent_heldout_prediction_mass": {
            "by_method": target_absent_mass,
            "final_vs_frozen": _target_absent_mass_comparison(
                target_absent_mass
            ),
            "false_positive_only_count_role": (
                "support_diagnostic_only_not_a_promotion_direction_under_"
                "strict_positive_soft_posteriors"
            ),
        },
    }


def _base_replay_receipt(
    records: list[dict[str, Any]], base_report: dict[str, Any], *, tolerance: float
) -> dict[str, Any]:
    base_by_id = {
        str(record["scene_id"]): record
        for record in base_report.get("per_validation_scene", ())
    }
    maximum_error = 0.0
    compared_value_count = 0
    keys = (
        "soft_3d_miou",
        "unknown_only_soft_3d_miou",
        "heldout_2d_soft_miou",
        "visible_but_unmasked_soft_3d_miou",
        "never_visible_soft_3d_miou",
    )
    for record in records:
        scene_id = str(record["scene_id"])
        if scene_id not in base_by_id:
            raise ValueError(f"base unary report lacks validation scene {scene_id}")
        expected = base_by_id[scene_id]["learned_completion"]
        observed = record["frozen_aligned_pointwise"]
        for key in keys:
            error = abs(float(observed[key]) - float(expected[key]))
            maximum_error = max(maximum_error, error)
            compared_value_count += 1
    if maximum_error > tolerance:
        raise RuntimeError(
            f"frozen aligned unary replay differs from v10 by {maximum_error}"
        )
    return {
        "policy": "per_scene_frozen_pointwise_metric_replay",
        "compared_value_count": compared_value_count,
        "absolute_tolerance": tolerance,
        "maximum_absolute_error": maximum_error,
        "passes": True,
    }


def _paired_differences(aggregate: dict[str, Any]) -> dict[str, Any]:
    scene_macro = aggregate["scene_macro"]
    pooled = aggregate["pooled_soft_iou_sufficient_statistics"]
    pointwise = "frozen_aligned_pointwise"
    message = "message_passing_completion"
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
    scene_differences = {
        key: scene_macro[message][key] - scene_macro[pointwise][key]
        for key in scene_keys
    }
    pooled_differences = {}
    for cohort in SOFT_IOU_COHORTS:
        for metric in (
            "scene_token_macro_soft_iou",
            "union_summed_element_or_pixel_token_micro_soft_iou",
        ):
            pooled_differences[f"{cohort}_{metric}"] = (
                pooled[message][cohort][metric] - pooled[pointwise][cohort][metric]
            )
    pointwise_fp = pooled[pointwise]["heldout_2d"]["counts"][
        "false_positive_only_scene_token_count"
    ]
    message_fp = pooled[message]["heldout_2d"]["counts"][
        "false_positive_only_scene_token_count"
    ]
    absent_mass = aggregate["target_absent_heldout_prediction_mass"]
    absent_mass_comparison = absent_mass["final_vs_frozen"]
    normalized_absent = absent_mass_comparison["metrics"][
        "prediction_mass_per_target_absent_heldout_pixel_token"
    ]
    return {
        "message_passing_minus_frozen_aligned_pointwise_scene_macro": scene_differences,
        "message_passing_minus_frozen_aligned_pointwise_pooled": pooled_differences,
        "heldout_2d_false_positive_only_scene_token_count": {
            "frozen_aligned_pointwise": pointwise_fp,
            "message_passing_completion": message_fp,
            "difference": message_fp - pointwise_fp,
            "role": (
                "support_diagnostic_only_not_a_promotion_direction_under_"
                "strict_positive_soft_posteriors"
            ),
        },
        "target_absent_heldout_prediction_mass": absent_mass_comparison,
        "directional_diagnostics_not_hard_gates": {
            "object_equal_heldout_2d_improves": pooled_differences[
                "heldout_2d_scene_token_macro_soft_iou"
            ]
            > 0,
            "target_absent_normalized_prediction_mass_reduces": (
                normalized_absent["final_minus_frozen"] < 0
            ),
            "scene_macro_3d_does_not_regress": scene_differences["soft_3d_miou"] >= 0,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_instance_oracle_training:
        raise PermissionError(
            "message passing requires explicit supervised instance-oracle authorization"
        )
    if args.message_passing_step_count not in MESSAGE_PASSING_STEPS:
        raise ValueError("message passing is bounded to exactly 2 or 3 steps")
    for name in (
        "epoch_count",
        "unary_element_batch_size",
        "edge_hidden_dimension",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip_norm <= 0:
        raise ValueError("message-passing optimizer parameters are invalid")
    loss_weights = {
        "unknown_categorical": args.categorical_loss_weight,
        "edge_same_instance_bce": args.edge_loss_weight,
        "heldout_render_soft_iou": args.heldout_soft_iou_loss_weight,
        "target_absent_token_fp_mass": args.absent_token_fp_loss_weight,
    }
    if any(not np.isfinite(value) or value <= 0 for value in loss_weights.values()):
        raise ValueError("all four preregistered message-passing losses need positive weights")
    cache_payloads = [load_scene_cache(Path(value)) for value in args.scene_cache]
    by_id = {str(payload["scene_id"]): payload for payload in cache_payloads}
    if len(by_id) != len(cache_payloads):
        raise ValueError("duplicate message-passing scene cache identity")
    training_scene_ids = set(map(str, args.training_scene))
    validation_scene_ids = set(map(str, args.validation_scene))
    selected_scene_ids = training_scene_ids | validation_scene_ids
    if selected_scene_ids != set(by_id):
        raise ValueError("scene-cache inputs must equal the requested protocol scenes")
    cohort_manifest = _load_cohort_manifest(
        args.cohort_manifest,
        protocol=args.protocol,
        training_scene_ids=training_scene_ids,
        validation_scene_ids=validation_scene_ids,
        epoch_count=args.epoch_count,
        allow_bounded_dev_protocol=args.allow_bounded_dev_protocol,
    )
    selected_cache_receipts = _cache_receipts(cache_payloads)
    base_report, base_checkpoint, base_authority = _load_base_authority(
        report_path_value=args.base_report,
        checkpoint_path_value=args.base_checkpoint,
        cohort_manifest=cohort_manifest,
        selected_cache_receipts=selected_cache_receipts,
        protocol=args.protocol,
        training_scene_ids=training_scene_ids,
        validation_scene_ids=validation_scene_ids,
    )
    configurations = [payload["configuration"] for payload in cache_payloads]
    if any(value != configurations[0] for value in configurations[1:]):
        raise ValueError("message-passing caches use different carrier configurations")
    if tuple(configurations[0].get("local_feature_layout", ())) != RGB_RADIO_GEOMETRY_LAYOUT:
        raise ValueError("message passing requires the sealed aligned F71 cache")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA message-passing training but CUDA is unavailable")
    source_tree = _source_tree_receipt()
    frozen_unary_model = _load_frozen_unary_model(base_checkpoint, device=device)
    temperature = float(base_report["training_configuration"]["temperature"])
    completion_confidence_cap = float(
        base_report["training_configuration"]["completion_confidence_cap"]
    )
    training_runtimes = [
        _prepare_runtime(by_id[scene_id]) for scene_id in sorted(training_scene_ids)
    ]
    validation_runtimes = [
        _prepare_runtime(by_id[scene_id]) for scene_id in sorted(validation_scene_ids)
    ]
    for runtime in training_runtimes + validation_runtimes:
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
    message_model = SurfaceMessagePassing(
        feature_dimension=len(RGB_RADIO_GEOMETRY_LAYOUT),
        edge_hidden_dimension=args.edge_hidden_dimension,
        step_count=args.message_passing_step_count,
        dropout=args.dropout,
    ).to(device)
    architecture_receipt = message_model.architecture_receipt()
    parameter_count = sum(parameter.numel() for parameter in message_model.parameters())
    training = _fit(
        message_model,
        training_runtimes,
        device=device,
        epoch_count=args.epoch_count,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        categorical_loss_weight=args.categorical_loss_weight,
        edge_loss_weight=args.edge_loss_weight,
        heldout_soft_iou_loss_weight=args.heldout_soft_iou_loss_weight,
        absent_token_fp_loss_weight=args.absent_token_fp_loss_weight,
        completion_confidence_cap=completion_confidence_cap,
        seed=args.seed,
    )
    if _source_tree_receipt()["source_tree_sha256"] != source_tree["source_tree_sha256"]:
        raise RuntimeError("message-passing source tree changed during training")
    checkpoint_path = Path(args.output_checkpoint).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in message_model.state_dict().items()
        },
        "model_configuration": {
            "feature_dimension": len(RGB_RADIO_GEOMETRY_LAYOUT),
            "hidden_dimension": args.edge_hidden_dimension,
            "step_count": args.message_passing_step_count,
            "dropout": args.dropout,
            "parameter_count": parameter_count,
            "posterior_layout": "N_by_K_plus_null",
            "observed_fact_policy": "exact_hard_clamp_every_step",
            "edge_compatibility": "query_independent_geometry_normal_source_radio",
            "architecture_receipt": architecture_receipt,
        },
        "protocol": args.protocol,
        "training_scene_ids": sorted(training_scene_ids),
        "validation_scene_ids": sorted(validation_scene_ids),
        "validation_used_for_model_selection": False,
        "integer_instance_ids_are_model_inputs": False,
        "training_target_authority": (
            "training_scene_unknown_membership_labels_and_heldout_mesh_rasters_only"
        ),
        "validation_targets_used_during_training": False,
        "base_unary_authority": base_authority,
        "cohort_manifest": cohort_manifest,
        "scene_cache_receipts": selected_cache_receipts,
        "source_tree": source_tree,
        "loss_weights": loss_weights,
        "completion_confidence_cap": completion_confidence_cap,
    }
    message_model.eval()
    per_scene = [
        _evaluate_scene(
            message_model,
            runtime,
            device=device,
            completion_confidence_cap=completion_confidence_cap,
        )
        for runtime in validation_runtimes
    ]
    if _source_tree_receipt()["source_tree_sha256"] != source_tree["source_tree_sha256"]:
        raise RuntimeError("message-passing source tree changed during evaluation")
    aggregate = _aggregate_evaluation(per_scene)
    base_replay = _base_replay_receipt(
        per_scene, base_report, tolerance=args.base_replay_tolerance
    )
    paired = _paired_differences(aggregate)
    bypass_error = max(
        row["message_passing"][
            "bypass_control_max_probability_error_from_frozen_unary"
        ]
        for row in per_scene
    )
    if bypass_error != 0:
        raise RuntimeError("message-passing bypass changed the frozen unary")
    checkpoint_payload["evaluation_contract"] = {
        "methods": list(METHODS),
        "message_passing_bypass_control": "direct_frozen_unary_no_learned_residual",
        "bypass_max_probability_error": bypass_error,
        "heldout_renderer": (
            "eligible_token_numerator_with_all_projected_elements_in_denominator"
        ),
        "reported_soft_iou_cohorts": list(SOFT_IOU_COHORTS),
        "target_absent_primary_fp_diagnostic": (
            "continuous_prediction_mass_total_per_absent_scene_token_and_"
            "per_absent_heldout_pixel_token"
        ),
        "false_positive_only_count_role": (
            "support_diagnostic_only_not_a_promotion_direction_under_"
            "strict_positive_soft_posteriors"
        ),
        "assignment": "threshold_free_token_plus_null_argmax",
        "base_unary_replay": base_replay,
    }
    temporary_checkpoint = checkpoint_path.with_name(
        checkpoint_path.name
        + ".temporary."
        + hashlib.sha256(
            f"{args.seed}\0{source_tree['source_tree_sha256']}".encode("utf-8")
        ).hexdigest()[:16]
    )
    torch.save(checkpoint_payload, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)
    report = {
        "schema": REPORT_SCHEMA,
        "stage": "bounded_token_conditioned_surface_message_passing",
        "protocol": args.protocol,
        "formal16_result": args.protocol == "formal16",
        "oracle_identity_diagnostic_only": True,
        "association_isolated": True,
        "pointwise_unary_frozen": True,
        "pointwise_unary_parameters_trainable": False,
        "method_scope": (
            "local_boundary_and_extent_correction_around_a_frozen_pointwise_unary;"
            "not_a_claim_of_whole_object_coverage_from_sparse_seeds"
        ),
        "completion_writes_unknown_only": True,
        "observed_positive_negative_and_null_facts_hard_clamped_every_step": True,
        "primary_assignment_decision": "threshold_free_token_plus_null_argmax",
        "benchmark_threshold_used_by_method": False,
        "target_absent_primary_fp_diagnostic": (
            "continuous_prediction_mass_total_per_absent_scene_token_and_"
            "per_absent_heldout_pixel_token"
        ),
        "false_positive_only_scene_token_count_is_promotion_direction": False,
        "false_positive_only_count_limitation": (
            "strictly_positive_softmax_and_soft_extent_outputs_give_positive_"
            "support_to_almost_every_target_absent_scene_token"
        ),
        "validation_used_for_model_selection": False,
        "validation_targets_used_during_training": False,
        "training_target_authority": (
            "training_scene_unknown_membership_labels_and_heldout_mesh_rasters_only"
        ),
        "integer_instance_ids_are_model_inputs": False,
        "edge_target_definition": (
            "directed_carrier_edge_is_positive_only_when_both_endpoints_share_one_retained_instance"
        ),
        "base_unary_authority": base_authority,
        "base_unary_replay": base_replay,
        "message_passing_bypass_control": {
            "definition": "direct_frozen_unary_no_learned_residual",
            "maximum_probability_error": bypass_error,
        },
        "cohort_manifest": cohort_manifest,
        "scene_cache_receipts": selected_cache_receipts,
        "scene_cache_record_list_sha256": _canonical_json_sha256(
            selected_cache_receipts
        ),
        "source_tree": source_tree,
        "message_passing_architecture": architecture_receipt,
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
            "device": str(device),
            "unary_element_batch_size": args.unary_element_batch_size,
            "unary_temperature_inherited_from_v10": temperature,
            "completion_confidence_cap_inherited_from_v10": completion_confidence_cap,
            "message_passing_step_count": args.message_passing_step_count,
            "edge_hidden_dimension": args.edge_hidden_dimension,
            "dropout": args.dropout,
            "model_parameter_count": parameter_count,
            "message_passing_architecture": architecture_receipt,
            "loss_weights": loss_weights,
            "soft_extent_training_signal": (
                "joint_unknown_categorical_heldout_soft_iou_and_target_absent_fp_mass"
            ),
            "all_four_heldout_mesh_views_used_per_training_scene_step": True,
            "directed_edge_class_balancing": "equal_mean_of_present_positive_and_negative_classes",
            "unknown_categorical_class_balancing": "equal_mean_over_present_token_plus_null_classes",
        },
        "training": training,
        "per_validation_scene": per_scene,
        **aggregate,
        "paired_differences": paired,
        "hard_promotion_gate_applied": False,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--epoch-count", type=int, default=40)
    parser.add_argument("--unary-element-batch-size", type=int, default=4096)
    parser.add_argument(
        "--message-passing-step-count", type=int, choices=MESSAGE_PASSING_STEPS, default=2
    )
    parser.add_argument("--edge-hidden-dimension", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--categorical-loss-weight", type=float, default=1.0)
    parser.add_argument("--edge-loss-weight", type=float, default=0.25)
    parser.add_argument("--heldout-soft-iou-loss-weight", type=float, default=1.0)
    parser.add_argument("--absent-token-fp-loss-weight", type=float, default=0.5)
    parser.add_argument("--base-replay-tolerance", type=float, default=2e-5)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "scene_macro": report["scene_macro"],
                "paired_differences": report["paired_differences"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
