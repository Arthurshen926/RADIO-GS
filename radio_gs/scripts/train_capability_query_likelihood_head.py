"""Train the generic multichannel likelihood head under the frozen v2 recipe."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.build_capability_likelihood_training_dataset import (
    CAPABILITY_CHANNELS,
    DATASET_SCHEMA_V2,
    iter_capability_training_examples,
)
from radio_gs.querying.query_compilers import continuous_gaussian_readout
from radio_gs.querying.query_likelihood_head import MonotoneQueryLikelihoodHead
from radio_gs.scripts.train_query_likelihood_head_fixed import (
    _binary_metrics,
    _mean_metric_rows,
    _parameter_summary,
    _sha256,
    _write_json_no_clobber,
    _write_torch_no_clobber,
)


CHECKPOINT_SCHEMA_V2 = "monotone-query-likelihood-head-checkpoint-v2"
RECIPE_ID = "capability-likelihood-balanced-bce-dice-adam-seed0-e200-lr0.05-v2"


def balanced_bce_soft_dice_loss(
    probability: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q = torch.as_tensor(probability).float().reshape(-1).clamp(1e-6, 1 - 1e-6)
    y = torch.as_tensor(target).bool().reshape(-1)
    if q.shape != y.shape or not bool(y.any()) or bool(y.all()):
        raise ValueError("balanced likelihood objective requires a nontrivial target")
    positive_bce = -torch.log(q[y]).mean()
    negative_bce = -torch.log1p(-q[~y]).mean()
    balanced_bce = 0.5 * (positive_bce + negative_bce)
    y_float = y.float()
    soft_dice = 1.0 - (2.0 * (q * y_float).sum() + 1.0) / (
        q.sum() + y_float.sum() + 1.0
    )
    total = balanced_bce + soft_dice
    return total, {
        "balanced_bce": balanced_bce,
        "positive_bce": positive_bce,
        "negative_bce": negative_bce,
        "soft_dice": soft_dice,
    }


def _load_inputs(
    manifest_path: Path, primitive_bundle: Path, preregistration: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, torch.Tensor]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != DATASET_SCHEMA_V2 or manifest.get("scene_count") != 1:
        raise ValueError("v2 trainer requires one sealed capability dataset")
    safety = manifest.get("safety", {})
    if (
        safety.get("labels_opened") is not True
        or safety.get("label_scope") != "official_source_train_scene_only"
        or safety.get("development_labels_opened") is not False
        or safety.get("test_labels_opened") is not False
        or safety.get("full312_evaluation_authorized") is not False
    ):
        raise PermissionError("v2 dataset crosses the source-train label boundary")
    record = manifest["records"][0]
    if record.get("scene_id") != "scene0000_00" or record.get("partition") != "fit":
        raise PermissionError("v2 Stage-A trainer is sealed to fit scene0000_00")
    shard = record["shard"]
    if _sha256(shard["path"]) != shard["sha256"]:
        raise ValueError("sealed v2 shard changed")
    payload = torch.load(shard["path"], map_location="cpu", weights_only=True)
    if payload.get("affinity_channels") != list(CAPABILITY_CHANNELS):
        raise ValueError("v2 capability channel order differs")
    if payload.get("safety", {}).get("spatial_kernel_used_as_instance_likelihood") is not False:
        raise ValueError("v2 shard allows a spatial kernel to impersonate likelihood")
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    recipe = prereg.get("training_recipe", {})
    if recipe.get("recipe_id") != RECIPE_ID:
        raise ValueError("v2 training recipe differs from preregistration")
    source = payload.get("source_authority", {}).get("primitive_bundle")
    if not isinstance(source, Mapping) or source.get("sha256") != _sha256(primitive_bundle):
        raise ValueError("v2 primitive bundle differs from its shard authority")
    bundle = torch.load(primitive_bundle, map_location="cpu", weights_only=True)
    covariance = torch.as_tensor(bundle["primitive_covariance"]).float()
    identity = torch.eye(3, dtype=torch.float32)
    context = {
        "primitive_xyz": torch.as_tensor(bundle["primitive_xyz"]).float(),
        "primitive_covariance": covariance,
        "primitive_precision": torch.linalg.pinv(covariance + 1e-6 * identity),
        "primitive_opacity": torch.as_tensor(bundle["primitive_opacity"]).float(),
        "official_point_xyz": torch.as_tensor(bundle["official_point_xyz"]).float(),
        "point_candidate_indices": torch.as_tensor(
            bundle["point_candidate_indices"]
        ).long(),
    }
    return manifest, payload, context


@torch.inference_mode()
def _evaluate(
    head: MonotoneQueryLikelihoodHead,
    payload: Mapping[str, object],
    context: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    primitive_rows = []
    point_rows = []
    by_click = defaultdict(lambda: {"primitive": [], "official_point": []})
    point_target = torch.as_tensor(payload["point_target"]).bool()
    for observations, target, step in iter_capability_training_examples(payload):
        probability = head(
            observations, source="registered_capability_source_train"
        ).foreground_probability
        primitive_metric = _binary_metrics(probability, target)
        point_probability, support = continuous_gaussian_readout(
            context["primitive_xyz"],
            context["primitive_covariance"],
            probability,
            context["official_point_xyz"],
            gaussian_precision=context["primitive_precision"],
            opacity=context["primitive_opacity"],
            candidate_indices=context["point_candidate_indices"],
        )
        point_metric = _binary_metrics(point_probability, point_target)
        point_metric["mean_gaussian_support"] = float(support.mean())
        click = int(step["click_count"])
        primitive_rows.append(primitive_metric)
        point_rows.append(point_metric)
        by_click[click]["primitive"].append(primitive_metric)
        by_click[click]["official_point"].append(point_metric)
    return {
        "mean": {
            "primitive": _mean_metric_rows(primitive_rows),
            "official_point": _mean_metric_rows(point_rows),
        },
        "by_click_count": {
            str(click): {
                domain: _mean_metric_rows(rows) for domain, rows in domains.items()
            }
            for click, domains in sorted(by_click.items())
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    if torch.cuda.is_initialized():
        raise RuntimeError("v2 fit sentinel must start before CUDA initialization")
    manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    bundle_path = Path(args.primitive_bundle).expanduser().resolve()
    prereg_path = Path(args.preregistration).expanduser().resolve()
    manifest, payload, context = _load_inputs(manifest_path, bundle_path, prereg_path)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    recipe = prereg["training_recipe"]
    examples = list(iter_capability_training_examples(payload))
    torch.manual_seed(int(recipe["seed"]))
    head = MonotoneQueryLikelihoodHead(
        affinity_channel_count=len(CAPABILITY_CHANNELS)
    ).cpu()
    initial = _evaluate(head, payload, context)
    optimizer = torch.optim.Adam(head.parameters(), lr=float(recipe["learning_rate"]))
    epoch_trace = []
    for epoch in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        components = []
        for observations, target, _step in examples:
            probability = head(
                observations, source="registered_capability_source_train"
            ).foreground_probability
            loss, detail = balanced_bce_soft_dice_loss(probability, target)
            step_losses.append(loss)
            components.append(detail)
        objective = torch.stack(step_losses).mean()
        objective.backward()
        optimizer.step()
        if epoch in {0, 1, 2, 4, 9, 19, 49, 99, 149, 199}:
            epoch_trace.append(
                {
                    "epoch": epoch + 1,
                    "objective": float(objective.detach()),
                    "balanced_bce": float(
                        torch.stack([row["balanced_bce"] for row in components]).mean()
                    ),
                    "soft_dice": float(
                        torch.stack([row["soft_dice"] for row in components]).mean()
                    ),
                }
            )
    trained = _evaluate(head, payload, context)
    mean_iou = float(trained["mean"]["official_point"]["iou_at_0.5"])
    click1 = float(trained["by_click_count"]["1"]["official_point"]["iou_at_0.5"])
    click10 = float(trained["by_click_count"]["10"]["official_point"]["iou_at_0.5"])
    gate = prereg["fit_sentinel_gate"]
    mean_pass = mean_iou > float(gate["required_mean_iou_strictly_greater_than"])
    response_pass = (click10 - click1) > float(
        gate["required_click10_iou_minus_click1_iou_strictly_greater_than"]
    )
    checkpoint_payload = {
        "schema_version": 2,
        "artifact_type": CHECKPOINT_SCHEMA_V2,
        "head_class": "MonotoneQueryLikelihoodHead",
        "head_schema_version": head.schema_version,
        "affinity_channels": list(CAPABILITY_CHANNELS),
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "recipe": recipe,
        "preregistration_sha256": _sha256(prereg_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "source_scene_ids": ["scene0000_00"],
        "safety": {
            "labels_opened": True,
            "label_scope": "official_source_train_scene_only",
            "development_labels_opened": False,
            "test_labels_opened": False,
            "full312_evaluation_run": False,
            "cuda_initialized": torch.cuda.is_initialized(),
        },
    }
    checkpoint = _write_torch_no_clobber(args.checkpoint, checkpoint_payload)
    receipt = {
        "schema_version": 2,
        "artifact_type": "capability-query-likelihood-fit-sentinel-v2",
        "status": "pass_ready_for_frozen_development" if mean_pass and response_pass else "fail_do_not_run_development",
        "dataset_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "primitive_bundle": {"path": str(bundle_path), "sha256": _sha256(bundle_path)},
        "preregistration": {"path": str(prereg_path), "sha256": _sha256(prereg_path)},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "head_schema_version": head.schema_version,
        "affinity_channels": list(CAPABILITY_CHANNELS),
        "trainable_parameter_count": sum(p.numel() for p in head.parameters()),
        "initial": initial,
        "trained": trained,
        "trained_parameters": _parameter_summary(head),
        "epoch_trace": epoch_trace,
        "gate": {
            "mean_iou_pass": mean_pass,
            "click_response_pass": response_pass,
            "mean_official_point_iou": mean_iou,
            "click1_official_point_iou": click1,
            "click10_official_point_iou": click10,
            "click10_minus_click1": click10 - click1,
            "development_authorized": bool(mean_pass and response_pass),
        },
        "safety": checkpoint_payload["safety"],
    }
    receipt_path = _write_json_no_clobber(args.receipt, receipt)
    return receipt_path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--primitive-bundle", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receipt", required=True)
    path, receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
