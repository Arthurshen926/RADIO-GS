"""Fit and source-gate the preregistered per-channel density-ratio v5 head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.build_capability_likelihood_training_dataset import (
    CAPABILITY_CHANNELS,
)
from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_ratio_training_dataset import (
    FIXED_FIT_SCENES,
)
from radio_gs.querying.query_likelihood_head import (
    MonotoneChannelDensityRatioHead,
    MonotoneOneSidedDensityRatioHead,
    QueryLikelihoodInputs,
)
from radio_gs.scripts.train_capability_likelihood_ratio_head import (
    _aggregate_scenes,
    _evaluate_scene,
    _load_manifests,
    _load_shard,
    _load_v2,
    _metric_row,
    _scene_readout_context,
)
from radio_gs.scripts.train_query_likelihood_head_fixed import (
    _sha256,
    _write_json_no_clobber,
    _write_torch_no_clobber,
)


RECIPE_ID = "capability-channel-density-ratio-adam-seed0-e100-lr0.02-v5"
CHECKPOINT_SCHEMA = "monotone-channel-density-ratio-head-checkpoint-v5"
RECIPE_ID_V6 = "capability-one-sided-channel-density-ratio-adam-seed0-e100-lr0.02-v6"
CHECKPOINT_SCHEMA_V6 = "monotone-one-sided-channel-density-ratio-head-checkpoint-v6"


def density_ratio_posterior_loss(
    positive_signed_cosine: torch.Tensor,
    negative_signed_cosine: torch.Tensor,
    *,
    prevalence: float,
    raw_slopes: torch.Tensor,
    intercepts: torch.Tensor,
) -> torch.Tensor:
    """Population posterior BCE for one click-prototype example per channel."""

    positive = torch.as_tensor(positive_signed_cosine).float()
    negative = torch.as_tensor(negative_signed_cosine).float()
    if (
        positive.ndim != 2
        or negative.ndim != 2
        or positive.shape[1] != negative.shape[1]
        or positive.shape[1] != int(torch.as_tensor(raw_slopes).numel())
    ):
        raise ValueError("density-ratio class samples must be [rows,channels]")
    pi = float(prevalence)
    if not positive.shape[0] or not negative.shape[0] or not 0 < pi < 1:
        raise ValueError("density-ratio example requires two classes and pi in (0,1)")
    prior_logit = torch.as_tensor(pi, device=positive.device).logit()
    slopes = F.softplus(raw_slopes).reshape(1, -1)
    bias = torch.as_tensor(intercepts).reshape(1, -1)
    positive_logit = positive * slopes + bias + prior_logit
    negative_logit = negative * slopes + bias + prior_logit
    return (
        pi * F.softplus(-positive_logit).mean(dim=0)
        + (1.0 - pi) * F.softplus(negative_logit).mean(dim=0)
    ).mean()


def _compile_scene_calibration(
    manifest: Mapping[str, object],
) -> tuple[dict[str, torch.Tensor | int], dict[str, object]]:
    positive_rows = []
    negative_rows = []
    positive_prior = []
    negative_prior = []
    positive_coefficient = []
    negative_coefficient = []
    included_objects = []
    excluded_objects = []
    example_count = 0
    for record in manifest["records"]:
        payload = _load_shard(record)
        pi = float(payload["primitive_foreground_prevalence"])
        inventory = {
            "scene_id": str(payload["scene_id"]),
            "object_id": int(payload["object_id"]),
            "primitive_foreground_prevalence": pi,
        }
        if not 0 < pi < 1:
            excluded_objects.append(inventory)
            continue
        included_objects.append(inventory)
        final_step = payload["steps"][-1]
        positive_index = torch.as_tensor(final_step["positive_training_rows"]).long()
        negative_index = torch.as_tensor(final_step["negative_training_rows"]).long()
        affinity = torch.as_tensor(payload["capability_click_affinity"])
        positive_columns = [
            index
            for index, click in enumerate(payload["clicks"])
            if bool(click["is_positive"])
        ]
        for column in positive_columns:
            positive = 2.0 * affinity[positive_index, column, :].float() - 1.0
            negative = 2.0 * affinity[negative_index, column, :].float() - 1.0
            prior = torch.tensor(pi).logit().item()
            positive_rows.append(positive)
            negative_rows.append(negative)
            positive_prior.append(torch.full((len(positive),), prior))
            negative_prior.append(torch.full((len(negative),), prior))
            positive_coefficient.append(torch.full((len(positive),), pi / len(positive)))
            negative_coefficient.append(
                torch.full((len(negative),), (1.0 - pi) / len(negative))
            )
            example_count += 1
    if not example_count:
        raise ValueError("density-ratio scene has no supported positive clicks")
    return {
        "positive_signed_cosine": torch.cat(positive_rows),
        "negative_signed_cosine": torch.cat(negative_rows),
        "positive_prior": torch.cat(positive_prior),
        "negative_prior": torch.cat(negative_prior),
        "positive_coefficient": torch.cat(positive_coefficient),
        "negative_coefficient": torch.cat(negative_coefficient),
        "example_count": example_count,
    }, {
        "included_object_count": len(included_objects),
        "excluded_zero_or_full_primitive_support": excluded_objects,
        "positive_click_calibration_example_count": example_count,
        "metric_object_subselection": False,
    }


def _fit(
    compiled: Mapping[str, Mapping[str, torch.Tensor | int]],
    inventories: Mapping[str, Mapping[str, object]],
    *,
    train_scenes: list[str],
    recipe: Mapping[str, object],
    device: torch.device,
    head_class: type[MonotoneChannelDensityRatioHead] = MonotoneChannelDensityRatioHead,
) -> tuple[MonotoneChannelDensityRatioHead, list[dict[str, float]], dict[str, object]]:
    torch.manual_seed(int(recipe["seed"]))
    head = head_class(
        affinity_channel_count=len(CAPABILITY_CHANNELS)
    ).to(device)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    keys = (
        "positive_signed_cosine",
        "negative_signed_cosine",
        "positive_prior",
        "negative_prior",
        "positive_coefficient",
        "negative_coefficient",
    )
    data = {
        key: torch.cat(
            [torch.as_tensor(compiled[scene][key]).to(device) for scene in train_scenes]
        )
        for key in keys
    }
    count = sum(int(compiled[scene]["example_count"]) for scene in train_scenes)
    trace = []
    for epoch in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        slopes = F.softplus(head.raw_slopes)[None, :]
        positive_logit = (
            data["positive_signed_cosine"] * slopes
            + head.intercepts[None, :]
            + data["positive_prior"][:, None]
        )
        negative_logit = (
            data["negative_signed_cosine"] * slopes
            + head.intercepts[None, :]
            + data["negative_prior"][:, None]
        )
        positive_loss = (
            data["positive_coefficient"][:, None] * F.softplus(-positive_logit)
        ).sum(dim=0) / count
        negative_loss = (
            data["negative_coefficient"][:, None] * F.softplus(negative_logit)
        ).sum(dim=0) / count
        per_channel = positive_loss + negative_loss
        objective = per_channel.mean()
        objective.backward()
        optimizer.step()
        if epoch in {0, 1, 2, 4, 9, 19, 49, 99}:
            slope = F.softplus(head.raw_slopes).detach()
            midpoint = -head.intercepts.detach() / slope
            trace.append(
                {
                    "epoch": epoch + 1,
                    "objective": float(objective.detach()),
                    "per_channel_bce": per_channel.detach().cpu().tolist(),
                    "slopes": slope.cpu().tolist(),
                    "intercepts": head.intercepts.detach().cpu().tolist(),
                    "null_midpoints": midpoint.cpu().tolist(),
                }
            )
    inventory = {
        "included_object_count": sum(
            int(inventories[scene]["included_object_count"]) for scene in train_scenes
        ),
        "excluded_zero_or_full_primitive_support": sum(
            (
                list(inventories[scene]["excluded_zero_or_full_primitive_support"])
                for scene in train_scenes
            ),
            [],
        ),
        "positive_click_calibration_example_count": count,
        "metric_object_subselection": False,
    }
    return head, trace, inventory


@torch.inference_mode()
def _evaluate_scene_density(
    head: MonotoneChannelDensityRatioHead,
    manifest: Mapping[str, object],
    *,
    device: torch.device,
    context: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    rows = []
    by_click: dict[int, list[dict[str, float]]] = {}
    by_object = []
    for record in manifest["records"]:
        payload = _load_shard(record)
        affinity = torch.as_tensor(payload["capability_click_affinity"], device=device)
        point_target = torch.as_tensor(payload["point_target"], device=device).bool()
        prior = torch.as_tensor(payload["prior_probability"], device=device).float()
        coverage = torch.as_tensor(payload["coverage"], device=device).float()
        reliability = torch.as_tensor(payload["reliability"], device=device).float()
        object_rows = []
        for step in payload["steps"]:
            positive = affinity[:, list(step["positive_columns"]), :]
            negative = affinity[:, list(step["negative_columns"]), :]
            observations = QueryLikelihoodInputs(
                positive_affinity=positive,
                negative_affinity=negative,
                prior_probability=prior,
                coverage=coverage,
                reliability=reliability,
            )
            primitive_probability = torch.sigmoid(head.log_likelihood_ratio(observations))
            point_probability = (
                context["weights"] * primitive_probability[context["indices"]]
            ).sum(dim=1)
            metric = _metric_row(point_probability, point_target)
            click = int(step["click_count"])
            metric.update(
                {
                    "scene_id": str(payload["scene_id"]),
                    "object_id": int(payload["object_id"]),
                    "click_count": click,
                }
            )
            rows.append(metric)
            object_rows.append(metric)
            by_click.setdefault(click, []).append(metric)
        by_object.append(
            {
                "object_id": int(payload["object_id"]),
                "mean": _numeric_mean(object_rows),
                "click10_minus_click1_iou": (
                    object_rows[-1]["iou_at_0.5"] - object_rows[0]["iou_at_0.5"]
                ),
            }
        )
    click_mean = {str(key): _numeric_mean(value) for key, value in sorted(by_click.items())}
    return {
        "scene_id": str(manifest["scene_id"]),
        "object_count": len(manifest["records"]),
        "example_count": len(rows),
        "mean": _numeric_mean(rows),
        "by_click_count": click_mean,
        "click10_minus_click1_iou": (
            click_mean["10"]["iou_at_0.5"] - click_mean["1"]["iou_at_0.5"]
        ),
        "by_object": by_object,
    }


def _numeric_mean(rows: list[Mapping[str, object]]) -> dict[str, float]:
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
        if isinstance(rows[0][key], (float, int))
    }


def _checkpoint(
    head: MonotoneChannelDensityRatioHead,
    *,
    train_scenes: list[str],
    manifests: Mapping[str, Mapping[str, object]],
    preregistration: Path,
    recipe: Mapping[str, object],
) -> dict[str, object]:
    slopes = F.softplus(head.raw_slopes.detach())
    one_sided = isinstance(head, MonotoneOneSidedDensityRatioHead)
    return {
        "schema_version": 6 if one_sided else 5,
        "artifact_type": CHECKPOINT_SCHEMA_V6 if one_sided else CHECKPOINT_SCHEMA,
        "head_class": (
            "MonotoneOneSidedDensityRatioHead"
            if one_sided
            else "MonotoneChannelDensityRatioHead"
        ),
        "head_schema_version": head.schema_version,
        "affinity_channels": list(CAPABILITY_CHANNELS),
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "slopes": slopes.cpu(),
        "intercepts": head.intercepts.detach().cpu(),
        "null_midpoints": (-head.intercepts.detach() / slopes).cpu(),
        "aggregate_bias": 0.0,
        "recipe": dict(recipe),
        "train_scene_ids": train_scenes,
        "dataset_manifests": {
            scene: {"path": manifests[scene]["_path"], "sha256": manifests[scene]["_sha256"]}
            for scene in train_scenes
        },
        "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
        "safety": {
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    preregistration = Path(args.preregistration).resolve()
    prereg = json.loads(preregistration.read_text())
    recipe = prereg["training_recipe"]
    one_sided = bool(getattr(args, "one_sided_negative", False))
    expected_recipe = RECIPE_ID_V6 if one_sided else RECIPE_ID
    if recipe.get("recipe_id") != expected_recipe:
        raise ValueError("density-ratio recipe differs from preregistration")
    head_class = (
        MonotoneOneSidedDensityRatioHead
        if one_sided
        else MonotoneChannelDensityRatioHead
    )
    candidate_key = "v6" if one_sided else "v5"
    manifests = _load_manifests(args.dataset_manifest)
    device = torch.device(args.device)
    v2_path = Path(args.v2_checkpoint).resolve()
    v2 = _load_v2(v2_path, device=device)
    contexts = {
        scene: _scene_readout_context(manifest, device=device)
        for scene, manifest in manifests.items()
    }
    compiled = {}
    inventories = {}
    for scene, manifest in manifests.items():
        compiled[scene], inventories[scene] = _compile_scene_calibration(manifest)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for heldout in FIXED_FIT_SCENES:
        train_scenes = [scene for scene in FIXED_FIT_SCENES if scene != heldout]
        head, trace, inventory = _fit(
            compiled,
            inventories,
            train_scenes=train_scenes,
            recipe=recipe,
            device=device,
            head_class=head_class,
        )
        checkpoint = _write_torch_no_clobber(
            output_dir / f"loo_holdout_{heldout}.pt",
            _checkpoint(
                head,
                train_scenes=train_scenes,
                manifests=manifests,
                preregistration=preregistration,
                recipe=recipe,
            ),
        )
        v2_metric = _evaluate_scene(
            v2, manifests[heldout], device=device, context=contexts[heldout]
        )
        candidate_metric = _evaluate_scene_density(
            head.eval(), manifests[heldout], device=device, context=contexts[heldout]
        )
        folds.append(
            {
                "heldout_scene": heldout,
                "train_scenes": train_scenes,
                "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
                "epoch_trace": trace,
                "training_inventory": inventory,
                "v2": v2_metric,
                candidate_key: candidate_metric,
                "mean_iou_gain": candidate_metric["mean"]["iou_at_0.5"] - v2_metric["mean"]["iou_at_0.5"],
                "absolute_log_mass_error_gain": v2_metric["mean"]["absolute_log_probability_mass_ratio_error"] - candidate_metric["mean"]["absolute_log_probability_mass_ratio_error"],
            }
        )
    all_head, all_trace, all_inventory = _fit(
        compiled,
        inventories,
        train_scenes=list(FIXED_FIT_SCENES),
        recipe=recipe,
        device=device,
        head_class=head_class,
    )
    all_checkpoint = _write_torch_no_clobber(
        output_dir / "all_fit_scene0000_0002_0005.pt",
        _checkpoint(
            all_head,
            train_scenes=list(FIXED_FIT_SCENES),
            manifests=manifests,
            preregistration=preregistration,
            recipe=recipe,
        ),
    )
    all_v2 = _aggregate_scenes(
        [
            _evaluate_scene(v2, manifests[s], device=device, context=contexts[s])
            for s in FIXED_FIT_SCENES
        ]
    )
    all_candidate = _aggregate_scenes(
        [_evaluate_scene_density(all_head.eval(), manifests[s], device=device, context=contexts[s]) for s in FIXED_FIT_SCENES]
    )
    loo_gain = sum(float(row["mean_iou_gain"]) for row in folds) / 3
    scene_passes = sum(float(row["mean_iou_gain"]) > 0.02 for row in folds)
    gates = {
        "loo_macro_mean_iou_gain": loo_gain,
        "loo_macro_mean_iou_gain_pass": loo_gain > 0.02,
        "loo_scenes_with_iou_gain_over_0.02": scene_passes,
        "loo_scene_count_pass": scene_passes >= 2,
        "loo_each_scene_click_response_over_0.02_pass": all(row[candidate_key]["click10_minus_click1_iou"] > 0.02 for row in folds),
        "loo_mass_calibration_improves_pass": sum(row[candidate_key]["mean"]["absolute_log_probability_mass_ratio_error"] for row in folds) < sum(row["v2"]["mean"]["absolute_log_probability_mass_ratio_error"] for row in folds),
        "all_fit_mean_iou_gain": all_candidate["scene_macro_mean"]["iou_at_0.5"] - all_v2["scene_macro_mean"]["iou_at_0.5"],
        "all_fit_mean_iou_gain_pass": all_candidate["scene_macro_mean"]["iou_at_0.5"] - all_v2["scene_macro_mean"]["iou_at_0.5"] > 0.02,
        "all_fit_precision_gain": all_candidate["scene_macro_mean"]["precision_at_0.5"] - all_v2["scene_macro_mean"]["precision_at_0.5"],
        "all_fit_precision_gain_pass": all_candidate["scene_macro_mean"]["precision_at_0.5"] - all_v2["scene_macro_mean"]["precision_at_0.5"] > 0.02,
        "all_fit_click10_minus_click1_iou": all_candidate["scene_macro_click10_minus_click1_iou"],
        "all_fit_click_response_pass": all_candidate["scene_macro_click10_minus_click1_iou"] > 0.05,
    }
    passed = all(value for key, value in gates.items() if key.endswith("_pass"))
    receipt = {
        "schema_version": 6 if one_sided else 5,
        "artifact_type": (
            "capability-one-sided-channel-density-ratio-source-gate-v6"
            if one_sided
            else "capability-channel-density-ratio-source-gate-v5"
        ),
        "status": "pass_source_gates_development_authorized" if passed else "fail_source_gates_do_not_open_development",
        "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
        "dataset_manifests": {scene: {"path": row["_path"], "sha256": row["_sha256"]} for scene, row in manifests.items()},
        "reference_v2_checkpoint": {"path": str(v2_path), "sha256": _sha256(v2_path)},
        "leave_one_scene_out": folds,
        "all_fit": {
            "checkpoint": {"path": str(all_checkpoint), "sha256": _sha256(all_checkpoint)},
            "epoch_trace": all_trace,
            "training_inventory": all_inventory,
            "v2": all_v2,
            candidate_key: all_candidate,
        },
        "gates": {**gates, "all_pass": passed, "development_authorized": passed},
        "safety": {
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
        },
    }
    path = _write_json_no_clobber(args.receipt, receipt)
    return path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", action="append", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--v2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--one-sided-negative", action="store_true")
    path, receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
