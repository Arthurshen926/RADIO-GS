"""Scene-disjoint learned calibration of frozen-unary object support mass.

This bounded diagnostic trains only a token-level support-size regressor on the
formal training scenes.  Validation targets are loaded after the last optimizer
step and are never used for model selection.  The predicted support masses are
then imposed by the same continuous global projection used by the oracle upper
bound.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.v4.completion.scannet import camera_from_record, load_scene_cache
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.training.diagnose_scannet_oracle_mass_projection import (
    METRIC_KEYS,
    oracle_mass_project,
)
from radio_gs.v4.training.train_scannet_completion_message_passing import (
    _clamp_contract,
    _frozen_unary_probabilities,
    _load_frozen_unary_model,
    _membership_metrics,
    _posterior_to_membership,
    _prepare_runtime,
)


REPORT_SCHEMA = (
    "radio_gs.surface_object_memory_v4.learned_unary_mass_calibration_diagnostic.v1"
)
FeatureMode = Literal[
    "summary",
    "summary_f71",
    "coverage_geometry",
    "source_view_coverage",
    "source_view_coverage_f71",
]


class UnaryMassCalibrator(nn.Module):
    """Predict a continuous correction to frozen-unary missing support mass."""

    def __init__(self, input_dimension: int, hidden_dimension: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        observed_mass: torch.Tensor,
        frozen_unary_mass: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.network(features).squeeze(-1)
        # Smoothly bound only the multiplicative correction for numerical
        # stability; there is no support threshold or discrete promotion gate.
        log_ratio = 4.0 * torch.tanh(residual / 4.0)
        baseline_missing = (frozen_unary_mass - observed_mass).clamp_min(1e-4)
        return observed_mass + baseline_missing * torch.exp(log_ratio)


def build_mass_features(
    runtime: dict[str, Any], unary: torch.Tensor, *, feature_mode: FeatureMode
) -> dict[str, torch.Tensor]:
    partial = runtime["partial"]
    token_count = int(partial.positive.shape[1])
    centres = runtime["centres"].float()
    probabilities = torch.as_tensor(unary, dtype=torch.float32)[:, :token_count]
    observed_mass = partial.positive.sum(0).float()
    frozen_mass = probabilities.sum(0)
    unknown = partial.unknown.any(-1)
    visible = runtime["source_visible"].bool()
    visible_unknown = unknown & visible
    never_visible = unknown & ~visible
    epsilon = 1e-8

    probability_mass = frozen_mass.clamp_min(epsilon)
    weighted_centre = probabilities.T @ centres / probability_mass[:, None]
    second = probabilities.T @ centres.square() / probability_mass[:, None]
    weighted_scale = (second - weighted_centre.square()).clamp_min(0).sqrt()
    observed_scale = runtime["context"].scale.float().clamp_min(
        float(runtime["minimum_scale"])
    )
    entropy = -(
        probabilities.clamp_min(epsilon).log() * probabilities
    ).sum(0) / probability_mass

    scene_values = torch.tensor(
        [
            np.log1p(float(centres.shape[0])),
            np.log1p(float(partial.eligible_elements.sum())),
            np.log1p(float(unknown.sum())),
            np.log1p(float(token_count)),
        ],
        dtype=torch.float32,
    )[None, :].expand(token_count, -1)
    token_values = torch.stack(
        (
            torch.log1p(observed_mass),
            torch.log1p(frozen_mass),
            torch.log1p(probabilities[unknown].sum(0)),
            torch.log1p(probabilities[visible_unknown].sum(0)),
            torch.log1p(probabilities[never_visible].sum(0)),
            torch.log1p(probabilities.square().sum(0)),
            entropy,
        ),
        dim=-1,
    )
    geometric = torch.cat(
        (
            torch.log(observed_scale / float(runtime["minimum_scale"])),
            torch.log(
                weighted_scale.clamp_min(float(runtime["minimum_scale"]))
                / float(runtime["minimum_scale"])
            ),
            torch.log(
                weighted_scale.clamp_min(float(runtime["minimum_scale"]))
                / observed_scale
            ),
        ),
        dim=-1,
    )
    pieces = [scene_values, token_values, geometric]
    if feature_mode == "summary_f71":
        pieces.append(runtime["context"].feature_prototype.float())
    elif feature_mode == "coverage_geometry":
        pieces.append(_source_coverage_geometry_features(runtime))
    elif feature_mode in ("source_view_coverage", "source_view_coverage_f71"):
        pieces.extend(
            (
                _source_coverage_geometry_features(runtime),
                _source_view_coverage_features(runtime),
            )
        )
        if feature_mode == "source_view_coverage_f71":
            pieces.append(runtime["context"].feature_prototype.float())
    elif feature_mode != "summary":
        raise ValueError("unsupported mass feature mode")
    features = torch.cat(pieces, dim=-1)
    if not torch.isfinite(features).all():
        raise RuntimeError("mass calibration features became non-finite")
    return {
        "features": features,
        "observed_mass": observed_mass,
        "frozen_mass": frozen_mass,
    }


def _source_coverage_geometry_features(runtime: dict[str, Any]) -> torch.Tensor:
    """Summarize source-only shape and camera coverage without target access."""

    partial = runtime["partial"]
    centres = runtime["centres"].float()
    token_count = int(partial.positive.shape[1])
    minimum_scale = float(runtime["minimum_scale"])
    payload = runtime["payload"]
    cameras = payload["observation_cameras"]
    camera_centres = torch.stack(
        [torch.as_tensor(camera["camera_to_world"][:3, 3]).float() for camera in cameras]
    )
    object_ids = list(map(int, payload["object_ids"]))
    kept_count_by_object = {object_id: 0 for object_id in object_ids}
    for record in payload["mask_dropout_receipt"]["records"]:
        object_id = int(record["object_id"])
        if object_id in kept_count_by_object and bool(record["kept"]):
            kept_count_by_object[object_id] += 1

    rows = []
    epsilon = 1e-8
    for token_id, object_id in enumerate(object_ids):
        observed = partial.positive[:, token_id].bool()
        points = centres[observed]
        if points.shape[0] == 0:
            raise RuntimeError("coverage geometry requires an observed token support")
        centre = points.mean(0)
        offset = points - centre
        covariance = offset.T @ offset / float(max(points.shape[0], 1))
        shape_scale = torch.linalg.eigvalsh(covariance).clamp_min(0).sqrt().sort().values
        extent = (points.max(0).values - points.min(0).values).sort().values

        camera_offset = camera_centres - centre
        distance = camera_offset.norm(dim=-1).clamp_min(minimum_scale)
        direction = camera_offset / distance[:, None]
        angular_second_moment = direction.T @ direction / float(direction.shape[0])
        angular_spectrum = torch.linalg.eigvalsh(angular_second_moment).clamp_min(0)
        kept_count = float(kept_count_by_object[object_id])
        kept_fraction = kept_count / float(max(len(cameras), 1))
        rows.append(
            torch.cat(
                (
                    torch.log1p(shape_scale / minimum_scale),
                    torch.log1p(extent / minimum_scale),
                    torch.log1p(
                        torch.stack((distance.min(), distance.mean(), distance.max()))
                        / minimum_scale
                    ),
                    angular_spectrum,
                    torch.tensor(
                        [
                            np.log1p(kept_count),
                            kept_fraction,
                            np.log1p(float(points.shape[0]) / max(kept_count, epsilon)),
                        ],
                        dtype=torch.float32,
                    ),
                )
            )
        )
    result = torch.stack(rows)
    if not torch.isfinite(result).all():
        raise RuntimeError("source coverage geometry became non-finite")
    return result


def _source_view_coverage_features(runtime: dict[str, Any]) -> torch.Tensor:
    """Measure observed-token coverage from source cameras only.

    The carrier projection is geometry-only and ``partial.positive`` contains
    only source-observed membership.  In particular, this function never reads
    the complete token labels or held-out camera rasters.  Every statistic has
    fixed dimension even when the number of source cameras changes.
    """

    partial = runtime["partial"]
    positive = partial.positive.bool()
    token_count = int(positive.shape[1])
    payload = runtime["payload"]
    camera_records = payload["observation_cameras"]
    if not camera_records:
        raise RuntimeError("source-view coverage requires observation cameras")
    camera_keys = [str(record["key"]) for record in camera_records]
    if len(set(camera_keys)) != len(camera_keys):
        raise RuntimeError("source observation camera keys must be unique")

    visible_by_view = []
    for record in camera_records:
        projection = runtime["carrier"].project(camera_from_record(record))
        visible = torch.zeros(positive.shape[0], dtype=torch.bool)
        visible[projection.element_ids.unique()] = True
        visible_by_view.append(visible)
    visible = torch.stack(visible_by_view, dim=-1)
    view_count = int(visible.shape[1])
    observed_mass = positive.sum(0).float().clamp_min(1.0)
    visible_observed_count = positive.float().T @ visible.float()
    visible_fraction = visible_observed_count / observed_mass[:, None]
    visible_carrier_count = visible.sum(0).float().clamp_min(1.0)
    carrier_occupancy = visible_observed_count / visible_carrier_count[None, :]

    object_ids = list(map(int, payload["object_ids"]))
    if len(object_ids) != token_count:
        raise RuntimeError("source-view coverage object order does not match tokens")
    receipt_lookup: dict[tuple[str, int], bool] = {}
    for record in payload["mask_dropout_receipt"]["records"]:
        key = (str(record["frame_id"]), int(record["object_id"]))
        if key in receipt_lookup:
            raise RuntimeError("duplicate source mask-dropout receipt record")
        receipt_lookup[key] = bool(record["kept"])
    kept = torch.empty(token_count, view_count, dtype=torch.bool)
    for token_id, object_id in enumerate(object_ids):
        for view_id, camera_key in enumerate(camera_keys):
            key = (camera_key, object_id)
            if key not in receipt_lookup:
                raise RuntimeError("incomplete source mask-dropout receipt")
            kept[token_id, view_id] = receipt_lookup[key]
    if bool((kept.sum(-1) == 0).any()):
        raise RuntimeError("retained source token must have at least one kept mask view")

    def view_statistics(values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                values.min(-1).values,
                values.mean(-1),
                values.max(-1).values,
                values.std(-1, unbiased=False),
                values.sum(-1),
            ),
            dim=-1,
        )

    def masked_statistics(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(-1).float()
        denominator = count.clamp_min(1.0)
        mean = (values * mask).sum(-1) / denominator
        maximum = values.masked_fill(~mask, -torch.inf).max(-1).values
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        return torch.stack((count / float(view_count), mean, maximum), dim=-1)

    # Visibility multiplicity captures whether the observed surface is seen
    # repeatedly or only in a narrow sliver of the source trajectory.
    visibility_multiplicity = visible.sum(-1).float()
    observed_visibility_mean = (
        positive.float().T @ (visibility_multiplicity / float(view_count))
    ) / observed_mass
    observed_seen_once = (
        positive.float().T @ (visibility_multiplicity > 0).float()
    ) / observed_mass
    observed_seen_all = (
        positive.float().T @ (visibility_multiplicity == view_count).float()
    ) / observed_mass

    kept_visible = visible[:, None, :] & kept[None, :, :]
    dropped = ~kept
    dropped_visible = visible[:, None, :] & dropped[None, :, :]
    positive_by_token = positive.T[:, :, None]
    kept_union = (positive_by_token & kept_visible.permute(1, 0, 2)).any(-1).sum(-1)
    dropped_union = (
        (positive_by_token & dropped_visible.permute(1, 0, 2)).any(-1).sum(-1)
    )
    union_statistics = torch.stack(
        (
            kept_union.float() / observed_mass,
            dropped_union.float() / observed_mass,
            observed_seen_once,
            observed_seen_all,
            observed_visibility_mean,
        ),
        dim=-1,
    )
    result = torch.cat(
        (
            view_statistics(visible_fraction),
            view_statistics(carrier_occupancy),
            masked_statistics(visible_fraction, kept),
            masked_statistics(visible_fraction, dropped),
            union_statistics,
        ),
        dim=-1,
    )
    if result.shape != (token_count, 21) or not torch.isfinite(result).all():
        raise RuntimeError("source-view coverage features violated their contract")
    return result


def target_physical_mass(runtime: dict[str, Any]) -> torch.Tensor:
    labels = torch.as_tensor(runtime["labels"], dtype=torch.long)
    eligible = runtime["partial"].eligible_elements
    token_count = int(runtime["partial"].positive.shape[1])
    return torch.stack(
        [((labels == token_id) & eligible).sum() for token_id in range(token_count)]
    ).float()


def _fit_calibrator(
    records: list[dict[str, Any]],
    *,
    input_dimension: int,
    hidden_dimension: int,
    learning_rate: float,
    weight_decay: float,
    epoch_count: int,
    seed: int,
    device: torch.device,
) -> tuple[UnaryMassCalibrator, list[dict[str, float]]]:
    torch.manual_seed(seed)
    calibrator = UnaryMassCalibrator(input_dimension, hidden_dimension).to(device)
    optimizer = torch.optim.AdamW(
        calibrator.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    order_rng = random.Random(seed + 211)
    history = []
    calibrator.train()
    for epoch in range(epoch_count):
        order = list(range(len(records)))
        order_rng.shuffle(order)
        losses = []
        for index in order:
            record = records[index]
            source = record["source"]
            predicted = calibrator(
                source["features"].to(device),
                source["observed_mass"].to(device),
                source["frozen_mass"].to(device),
            )
            target = record["target"].to(device)
            loss = F.smooth_l1_loss(torch.log1p(predicted), torch.log1p(target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(
            {"epoch": epoch + 1, "mean_log_mass_loss": float(np.mean(losses))}
        )
    return calibrator.eval(), history


def _crossfit_blend_fraction(
    records: list[dict[str, Any]],
    *,
    input_dimension: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    predictions = []
    for heldout_index, heldout in enumerate(records):
        fold_model, _ = _fit_calibrator(
            [record for index, record in enumerate(records) if index != heldout_index],
            input_dimension=input_dimension,
            hidden_dimension=args.hidden_dimension,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epoch_count=args.epoch_count,
            seed=args.seed + 1000 + heldout_index,
            device=device,
        )
        source = heldout["source"]
        with torch.no_grad():
            raw = fold_model(
                source["features"].to(device),
                source["observed_mass"].to(device),
                source["frozen_mass"].to(device),
            ).cpu()
        predictions.append(raw)

    candidates = [index / 20.0 for index in range(21)]
    curve = []
    for fraction in candidates:
        scene_losses = []
        for record, raw in zip(records, predictions):
            source = record["source"]
            blended = (
                source["frozen_mass"]
                + fraction * (raw - source["frozen_mass"])
            ).clamp_min(source["observed_mass"])
            scene_losses.append(
                float(
                    F.smooth_l1_loss(
                        torch.log1p(blended), torch.log1p(record["target"])
                    )
                )
            )
        curve.append(
            {"blend_fraction": fraction, "scene_macro_log_mass_loss": float(np.mean(scene_losses))}
        )
    selected = min(curve, key=lambda value: (value["scene_macro_log_mass_loss"], value["blend_fraction"]))
    return float(selected["blend_fraction"]), {
        "method": "leave_one_training_scene_out_scene_macro_log_mass_loss",
        "fold_count": len(records),
        "candidate_curve": curve,
        "selected_blend_fraction": float(selected["blend_fraction"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.feature_mode not in ("summary", "summary_f71", "coverage_geometry"):
        raise ValueError("unsupported mass feature mode")
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
        raise ValueError("mass calibration training and validation scenes must be disjoint")
    expected_split = base_report["split"]
    frozen_training_ids = set(map(str, expected_split["training_scene_ids"]))
    if args.allow_training_superset:
        if not frozen_training_ids.issubset(training_ids):
            raise ValueError("expanded mass training must retain the frozen training split")
    elif training_ids != frozen_training_ids:
        raise ValueError("mass calibration requires the frozen training split")
    if validation_ids != set(map(str, expected_split["validation_scene_ids"])):
        raise ValueError("mass calibration requires the frozen validation split")
    payloads = [load_scene_cache(Path(value)) for value in args.scene_cache]
    by_id = {str(value["scene_id"]): value for value in payloads}
    if set(by_id) != training_ids | validation_ids:
        raise ValueError("mass calibration caches do not equal the formal cohort")

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
        training.append(
            {
                "scene_id": scene_id,
                "runtime": runtime,
                "unary": unary,
                "source": build_mass_features(
                    runtime, unary, feature_mode=args.feature_mode
                ),
                "target": target_physical_mass(runtime),
            }
        )
    input_dimension = int(training[0]["source"]["features"].shape[1])
    blend_fraction = 1.0
    crossfit = None
    if args.crossfit_blend:
        blend_fraction, crossfit = _crossfit_blend_fraction(
            training, input_dimension=input_dimension, args=args, device=device
        )
    calibrator, history = _fit_calibrator(
        training,
        input_dimension=input_dimension,
        hidden_dimension=args.hidden_dimension,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epoch_count=args.epoch_count,
        seed=args.seed,
        device=device,
    )

    # Validation construction is intentionally delayed until training ends.
    del frozen_model
    calibrator.eval()
    records = []
    with torch.no_grad():
        for scene_id in sorted(validation_ids):
            runtime = _prepare_runtime(by_id[scene_id])
            checkpoint_model = _load_frozen_unary_model(base_checkpoint, device=device)
            unary = _frozen_unary_probabilities(
                checkpoint_model,
                runtime,
                device=device,
                element_batch_size=args.unary_element_batch_size,
                temperature=temperature,
            )
            del checkpoint_model
            source = build_mass_features(runtime, unary, feature_mode=args.feature_mode)
            raw_predicted = calibrator(
                source["features"].to(device),
                source["observed_mass"].to(device),
                source["frozen_mass"].to(device),
            )
            predicted = (
                source["frozen_mass"].to(device)
                + blend_fraction
                * (raw_predicted - source["frozen_mass"].to(device))
            ).clamp_min(source["observed_mass"].to(device))
            target = target_physical_mass(runtime).to(device)
            clamp_mask, clamp_probabilities = _clamp_contract(runtime)
            projected, projection = oracle_mass_project(
                unary.to(device),
                clamp_mask.to(device),
                clamp_probabilities.to(device),
                predicted,
                iteration_count=args.projection_iteration_count,
                damping=args.projection_damping,
            )
            frozen_membership, frozen_null = _posterior_to_membership(
                unary, runtime, completion_confidence_cap=confidence_cap
            )
            projected_membership, projected_null = _posterior_to_membership(
                projected.cpu(), runtime, completion_confidence_cap=confidence_cap
            )
            relative_error = (predicted - target).abs() / target
            records.append(
                {
                    "scene_id": scene_id,
                    "token_count": int(target.numel()),
                    "mass_prediction": {
                        "predicted_mean": float(predicted.mean()),
                        "target_mean": float(target.mean()),
                        "mean_absolute_relative_error": float(relative_error.mean()),
                        "median_absolute_relative_error": float(relative_error.median()),
                    },
                    "projection": projection,
                    "frozen_aligned_pointwise": _membership_metrics(
                        runtime, frozen_membership, frozen_null
                    ),
                    "learned_mass_projection": _membership_metrics(
                        runtime, projected_membership, projected_null
                    ),
                }
            )

    scene_macro = {}
    for method in ("frozen_aligned_pointwise", "learned_mass_projection"):
        scene_macro[method] = {
            key: float(np.mean([record[method][key] for record in records]))
            for key in METRIC_KEYS
        }
    difference = {
        key: scene_macro["learned_mass_projection"][key]
        - scene_macro["frozen_aligned_pointwise"][key]
        for key in METRIC_KEYS
    }
    source_path = Path(__file__).resolve()
    report = {
        "schema": REPORT_SCHEMA,
        "role": "scene_disjoint_train_only_mass_calibration_diagnostic",
        "feature_mode": args.feature_mode,
        "target_membership_is_model_input": False,
        "target_physical_mass_is_training_loss_only": True,
        "validation_constructed_after_last_optimizer_step": True,
        "validation_used_for_model_selection": False,
        "training_scene_ids": sorted(training_ids),
        "frozen_unary_training_scene_ids": sorted(frozen_training_ids),
        "additional_mass_training_scene_ids": sorted(training_ids - frozen_training_ids),
        "validation_scene_ids": sorted(validation_ids),
        "training_configuration": {
            "seed": args.seed,
            "epoch_count": args.epoch_count,
            "optimizer_step_count": args.epoch_count * len(training),
            "hidden_dimension": args.hidden_dimension,
            "input_dimension": input_dimension,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "projection_iteration_count": args.projection_iteration_count,
            "projection_damping": args.projection_damping,
            "crossfit_blend": args.crossfit_blend,
            "applied_blend_fraction": blend_fraction,
            "allow_training_superset": args.allow_training_superset,
        },
        "training_only_crossfit_blend_selection": crossfit,
        "training_history": history,
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "base_report": {"path": str(base_report_path), "sha256": sha256_file(base_report_path)},
        "base_checkpoint": {"path": str(base_checkpoint_path), "sha256": sha256_file(base_checkpoint_path)},
        "per_validation_scene": records,
        "scene_macro": scene_macro,
        "learned_minus_frozen_scene_macro": difference,
        "model_state_dict": {
            key: value.detach().cpu().tolist() for key, value in calibrator.state_dict().items()
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-cache", action="append", required=True)
    parser.add_argument("--training-scene", action="append", required=True)
    parser.add_argument("--validation-scene", action="append", required=True)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--feature-mode",
        choices=("summary", "summary_f71", "coverage_geometry"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--epoch-count", type=int, default=40)
    parser.add_argument("--hidden-dimension", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unary-element-batch-size", type=int, default=4096)
    parser.add_argument("--projection-iteration-count", type=int, default=256)
    parser.add_argument("--projection-damping", type=float, default=0.5)
    parser.add_argument("--crossfit-blend", action="store_true")
    parser.add_argument("--allow-training-superset", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["learned_minus_frozen_scene_macro"], indent=2))


if __name__ == "__main__":
    main()
