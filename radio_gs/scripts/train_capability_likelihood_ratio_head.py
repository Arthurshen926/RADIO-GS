"""Train and source-gate the frozen multi-scene capability LLR v3 head."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.build_capability_likelihood_training_dataset import (
    CAPABILITY_CHANNELS,
)
from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_ratio_training_dataset import (
    DATASET_SCHEMA_V3,
    FIXED_FIT_SCENES,
    SHARD_SCHEMA_V3,
)
from radio_gs.querying.query_likelihood_head import (
    MonotoneLikelihoodRatioHead,
    MonotoneQueryLikelihoodHead,
    MonotoneSignedLikelihoodRatioHead,
)
from radio_gs.scripts.train_query_likelihood_head_fixed import (
    _sha256,
    _write_json_no_clobber,
    _write_torch_no_clobber,
)


CHECKPOINT_SCHEMA_V3 = "monotone-query-likelihood-ratio-head-checkpoint-v3"
RECEIPT_SCHEMA_V3 = "capability-query-likelihood-ratio-source-gate-v3"
RECIPE_ID = (
    "capability-likelihood-ratio-bce-prior-correction-rank025-"
    "adam-seed0-e100-lr0.02-v3"
)


def prevalence_weighted_posterior_ranking_loss(
    positive_log_likelihood_ratio: torch.Tensor,
    negative_log_likelihood_ratio: torch.Tensor,
    *,
    prevalence: float,
    ranking_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stratified unbiased population BCE plus prior-invariant ranking.

    Equal-count positive/negative rows are only a variance-reduction device.
    Multiplication by the actual population prevalence makes the BCE estimate
    identical in expectation to ordinary per-primitive posterior BCE, rather
    than the foreground-heavy 0.5/0.5 objective used by v2.
    """

    positive = torch.as_tensor(positive_log_likelihood_ratio).float().reshape(-1)
    negative = torch.as_tensor(negative_log_likelihood_ratio).float().reshape(-1)
    pi = float(prevalence)
    if not positive.numel() or not negative.numel() or not 0.0 < pi < 1.0:
        raise ValueError("LLR loss requires nonempty class strata and prevalence in (0,1)")
    prior_logit = math.log(pi / (1.0 - pi))
    positive_bce = F.softplus(-(positive + prior_logit)).mean()
    negative_bce = F.softplus(negative + prior_logit).mean()
    posterior_bce = pi * positive_bce + (1.0 - pi) * negative_bce
    pair_count = min(int(positive.numel()), int(negative.numel()))
    ranking = F.softplus(-(positive[:pair_count] - negative[:pair_count])).mean()
    total = posterior_bce + float(ranking_weight) * ranking
    return total, {
        "posterior_bce": posterior_bce,
        "positive_bce": positive_bce,
        "negative_bce": negative_bce,
        "ranking": ranking,
    }


def _set_design(
    affinity: torch.Tensor,
    columns: list[int],
    *,
    null_centered: bool = False,
) -> torch.Tensor:
    """Return [N,C,peak/mean] sufficient statistics on the current device."""

    values = torch.as_tensor(affinity)
    if values.ndim != 3 or values.shape[2] != len(CAPABILITY_CHANNELS):
        raise ValueError("v3 affinity must be [N,K,2]")
    if not columns:
        return torch.zeros(
            (values.shape[0], values.shape[2], 2),
            device=values.device,
            dtype=torch.float32,
        )
    selected = values[:, columns, :].float()
    if null_centered:
        selected = 2.0 * selected - 1.0
    return torch.stack((selected.amax(dim=1), selected.mean(dim=1)), dim=-1)


def _score_design(
    head: torch.nn.Module, positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    return (
        head.bias
        + (positive * F.softplus(head.raw_positive_weights)).sum(dim=(-2, -1))
        - (negative * F.softplus(head.raw_negative_weights)).sum(dim=(-2, -1))
    )


def _load_manifests(paths: Iterable[str | Path]) -> dict[str, dict[str, object]]:
    loaded: dict[str, dict[str, object]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        scene_id = str(manifest.get("scene_id"))
        safety = manifest.get("safety", {})
        if (
            manifest.get("artifact_type") != DATASET_SCHEMA_V3
            or scene_id not in FIXED_FIT_SCENES
            or manifest.get("partition") != "fit"
            or safety.get("labels_opened") is not True
            or safety.get("development_labels_opened") is not False
            or safety.get("test_labels_opened") is not False
            or safety.get("test312_run") is not False
        ):
            raise PermissionError("v3 manifest violates the fixed source-fit boundary")
        if scene_id in loaded:
            raise ValueError(f"duplicate v3 scene manifest: {scene_id}")
        records = manifest.get("records", [])
        if len(records) != int(manifest.get("object_count", -1)) or not records:
            raise ValueError("v3 manifest object inventory is incomplete")
        for record in records:
            shard = record["shard"]
            if _sha256(shard["path"]) != shard["sha256"]:
                raise ValueError("sealed v3 training shard changed")
        manifest["_path"] = str(path)
        manifest["_sha256"] = _sha256(path)
        loaded[scene_id] = manifest
    if tuple(sorted(loaded)) != tuple(sorted(FIXED_FIT_SCENES)):
        raise PermissionError("v3 requires exactly fit scenes 0000/0002/0005")
    return loaded


def _load_shard(record: Mapping[str, object]) -> dict[str, object]:
    payload = torch.load(record["shard"]["path"], map_location="cpu", weights_only=True)
    safety = payload.get("safety", {})
    if (
        payload.get("artifact_type") != SHARD_SCHEMA_V3
        or payload.get("partition") != "fit"
        or payload.get("affinity_channels") != list(CAPABILITY_CHANNELS)
        or safety.get("labels_opened") is not True
        or safety.get("development_labels_opened") is not False
        or safety.get("test_labels_opened") is not False
        or safety.get("soft_dice_target_materialized") is not False
        or safety.get("spatial_kernel_used_as_instance_likelihood") is not False
    ):
        raise PermissionError("v3 shard violates the likelihood-ratio contract")
    return payload


def _sample_design(
    payload: Mapping[str, object],
    step: Mapping[str, object],
    *,
    device: torch.device,
    null_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    affinity = torch.as_tensor(payload["capability_click_affinity"])
    positive_rows = torch.as_tensor(step["positive_training_rows"]).long()
    negative_rows = torch.as_tensor(step["negative_training_rows"]).long()
    all_rows = torch.cat((positive_rows, negative_rows))
    selected = affinity.index_select(0, all_rows).to(device)
    positive_design = _set_design(
        selected,
        list(step["positive_columns"]),
        null_centered=null_centered,
    )
    negative_design = _set_design(
        selected,
        list(step["negative_columns"]),
        null_centered=null_centered,
    )
    split = int(positive_rows.numel())
    return (
        positive_design[:split],
        negative_design[:split],
        positive_design[split:],
        negative_design[split:],
    )


def _compile_scene_training(
    manifest: Mapping[str, object],
    *,
    null_centered: bool = False,
) -> tuple[dict[str, torch.Tensor | int], dict[str, object]]:
    positive_designs = []
    positive_priors = []
    positive_coefficients = []
    negative_designs = []
    negative_priors = []
    negative_coefficients = []
    ranking_differences = []
    ranking_coefficients = []
    included_objects = []
    excluded_objects = []
    example_count = 0
    scene_id = str(manifest["scene_id"])
    for record in manifest["records"]:
        payload = _load_shard(record)
        prevalence = float(payload["primitive_foreground_prevalence"])
        inventory_row = {
            "scene_id": scene_id,
            "object_id": int(payload["object_id"]),
            "primitive_foreground_prevalence": prevalence,
        }
        if not 0.0 < prevalence < 1.0:
            excluded_objects.append(inventory_row)
            continue
        included_objects.append(inventory_row)
        prior_logit = math.log(prevalence / (1.0 - prevalence))
        for step in payload["steps"]:
            pp, pn, np_, nn = _sample_design(
                payload,
                step,
                device=torch.device("cpu"),
                null_centered=null_centered,
            )
            positive = torch.cat((pp.flatten(1), -pn.flatten(1)), dim=1)
            negative = torch.cat((np_.flatten(1), -nn.flatten(1)), dim=1)
            pair_count = min(len(positive), len(negative))
            positive_designs.append(positive)
            positive_priors.append(torch.full((len(positive),), prior_logit))
            positive_coefficients.append(
                torch.full((len(positive),), prevalence / len(positive))
            )
            negative_designs.append(negative)
            negative_priors.append(torch.full((len(negative),), prior_logit))
            negative_coefficients.append(
                torch.full((len(negative),), (1.0 - prevalence) / len(negative))
            )
            ranking_differences.append(positive[:pair_count] - negative[:pair_count])
            ranking_coefficients.append(
                torch.full((pair_count,), 1.0 / pair_count)
            )
            example_count += 1
    if not example_count:
        raise ValueError("v3 fold has no class-conditionally supported examples")
    compiled: dict[str, torch.Tensor | int] = {
        "positive_design": torch.cat(positive_designs),
        "positive_prior": torch.cat(positive_priors),
        "positive_coefficient": torch.cat(positive_coefficients),
        "negative_design": torch.cat(negative_designs),
        "negative_prior": torch.cat(negative_priors),
        "negative_coefficient": torch.cat(negative_coefficients),
        "ranking_difference": torch.cat(ranking_differences),
        "ranking_coefficient": torch.cat(ranking_coefficients),
        "example_count": example_count,
    }
    return compiled, {
        "included_object_count": len(included_objects),
        "excluded_zero_or_full_primitive_support": excluded_objects,
        "metric_object_subselection": False,
    }


def _train_fold(
    compiled_scenes: Mapping[str, Mapping[str, torch.Tensor | int]],
    inventories: Mapping[str, Mapping[str, object]],
    *,
    train_scenes: list[str],
    recipe: Mapping[str, object],
    device: torch.device,
    head_class: type[MonotoneLikelihoodRatioHead] = MonotoneLikelihoodRatioHead,
) -> tuple[MonotoneLikelihoodRatioHead, list[dict[str, float]], dict[str, object]]:
    torch.manual_seed(int(recipe["seed"]))
    head = head_class(
        affinity_channel_count=len(CAPABILITY_CHANNELS)
    ).to(device)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    tensor_keys = (
        "positive_design",
        "positive_prior",
        "positive_coefficient",
        "negative_design",
        "negative_prior",
        "negative_coefficient",
        "ranking_difference",
        "ranking_coefficient",
    )
    tensors = {
        key: torch.cat(
            [torch.as_tensor(compiled_scenes[scene][key]).to(device) for scene in train_scenes]
        )
        for key in tensor_keys
    }
    example_count = sum(
        int(compiled_scenes[scene]["example_count"]) for scene in train_scenes
    )
    trace = []
    for epoch in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        weight = torch.cat(
            (
                F.softplus(head.raw_positive_weights).flatten(),
                F.softplus(head.raw_negative_weights).flatten(),
            )
        )
        positive_score = head.bias + tensors["positive_design"] @ weight
        negative_score = head.bias + tensors["negative_design"] @ weight
        positive_bce = (
            tensors["positive_coefficient"]
            * F.softplus(-(positive_score + tensors["positive_prior"]))
        ).sum() / example_count
        negative_bce = (
            tensors["negative_coefficient"]
            * F.softplus(negative_score + tensors["negative_prior"])
        ).sum() / example_count
        posterior_bce = positive_bce + negative_bce
        ranking = (
            tensors["ranking_coefficient"]
            * F.softplus(-(tensors["ranking_difference"] @ weight))
        ).sum() / example_count
        objective = posterior_bce + float(recipe["ranking_weight"]) * ranking
        objective.backward()
        optimizer.step()
        if epoch in {0, 1, 2, 4, 9, 19, 49, 99}:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "objective": float(objective.detach()),
                    "posterior_bce": float(posterior_bce.detach()),
                    "positive_bce_contribution": float(positive_bce.detach()),
                    "negative_bce_contribution": float(negative_bce.detach()),
                    "ranking": float(ranking.detach()),
                }
            )
    return head, trace, {
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
        "training_example_count": example_count,
        "metric_object_subselection": False,
    }


def _load_v2(path: Path, *, device: torch.device) -> MonotoneQueryLikelihoodHead:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("artifact_type")
        != "monotone-query-likelihood-head-checkpoint-v2"
        or payload.get("head_schema_version")
        != "monotone-query-likelihood-multichannel-v2"
    ):
        raise ValueError("reference is not the frozen v2 capability head")
    head = MonotoneQueryLikelihoodHead(
        affinity_channel_count=len(CAPABILITY_CHANNELS)
    ).to(device)
    head.load_state_dict(payload["state_dict"], strict=True)
    return head.eval()


def _scene_readout_context(
    manifest: Mapping[str, object], *, device: torch.device
) -> dict[str, torch.Tensor]:
    source = manifest["source_authority"]["primitive_bundle"]
    if _sha256(source["path"]) != source["sha256"]:
        raise ValueError("sealed primitive bundle changed")
    bundle = torch.load(source["path"], map_location="cpu", weights_only=True)
    xyz = torch.as_tensor(bundle["primitive_xyz"], device=device).float()
    covariance = torch.as_tensor(bundle["primitive_covariance"], device=device).float()
    opacity = torch.as_tensor(bundle["primitive_opacity"], device=device).float().reshape(-1)
    points = torch.as_tensor(bundle["official_point_xyz"], device=device).float()
    indices = torch.as_tensor(bundle["point_candidate_indices"], device=device).long()
    identity = torch.eye(3, device=device)
    precision = torch.linalg.pinv(covariance + 1e-6 * identity)
    delta = xyz[indices] - points[:, None]
    mahalanobis = torch.einsum(
        "pki,pkij,pkj->pk", delta, precision[indices], delta
    )
    weights = torch.exp(-0.5 * mahalanobis).clamp_min(0) * opacity[indices]
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return {"indices": indices, "weights": weights}


def _metric_row(probability: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    q = torch.as_tensor(probability).float().reshape(-1)
    y = torch.as_tensor(target, device=q.device).bool().reshape(-1)
    prediction = q >= 0.5
    intersection = int((prediction & y).sum())
    union = int((prediction | y).sum())
    predicted_fraction = float(prediction.float().mean())
    probability_mass = float(q.mean())
    target_fraction = float(y.float().mean())
    return {
        "iou_at_0.5": intersection / max(1, union),
        "precision_at_0.5": intersection / max(1, int(prediction.sum())),
        "recall_at_0.5": intersection / max(1, int(y.sum())),
        "thresholded_foreground_fraction": predicted_fraction,
        "probability_foreground_mass": probability_mass,
        "target_foreground_fraction": target_fraction,
        "absolute_log_probability_mass_ratio_error": abs(
            math.log((probability_mass + 1e-8) / (target_fraction + 1e-8))
        ),
    }


def _mean(rows: list[Mapping[str, float]]) -> dict[str, float]:
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
        if isinstance(rows[0][key], (float, int))
    }


@torch.inference_mode()
def _evaluate_scene(
    head: torch.nn.Module,
    manifest: Mapping[str, object],
    *,
    device: torch.device,
    context: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    if context is None:
        context = _scene_readout_context(manifest, device=device)
    rows = []
    by_click: dict[int, list[dict[str, float]]] = defaultdict(list)
    by_object = []
    for record in manifest["records"]:
        payload = _load_shard(record)
        affinity = torch.as_tensor(
            payload["capability_click_affinity"], device=device
        )
        point_target = torch.as_tensor(payload["point_target"], device=device).bool()
        object_rows = []
        for step in payload["steps"]:
            null_centered = isinstance(head, MonotoneSignedLikelihoodRatioHead)
            positive = _set_design(
                affinity,
                list(step["positive_columns"]),
                null_centered=null_centered,
            )
            negative = _set_design(
                affinity,
                list(step["negative_columns"]),
                null_centered=null_centered,
            )
            if isinstance(head, MonotoneLikelihoodRatioHead):
                primitive_probability = torch.sigmoid(
                    _score_design(head, positive, negative)
                )
            else:
                prior = torch.as_tensor(payload["prior_probability"], device=device).float()
                prior_logit = torch.logit(prior.clamp(1e-4, 1 - 1e-4))
                primitive_probability = torch.sigmoid(
                    head.bias
                    + (positive * F.softplus(head.raw_positive_weights)).sum(dim=(-2, -1))
                    - (negative * F.softplus(head.raw_negative_weights)).sum(dim=(-2, -1))
                    + F.softplus(head.raw_prior_weight) * prior_logit
                )
            point_probability = (
                context["weights"]
                * primitive_probability[context["indices"]]
            ).sum(dim=1)
            metric = _metric_row(point_probability, point_target)
            metric.update(
                {
                    "scene_id": str(payload["scene_id"]),
                    "object_id": int(payload["object_id"]),
                    "click_count": int(step["click_count"]),
                }
            )
            rows.append(metric)
            object_rows.append(metric)
            by_click[int(step["click_count"])].append(metric)
        by_object.append(
            {
                "object_id": int(payload["object_id"]),
                "mean": _mean(object_rows),
                "click10_minus_click1_iou": (
                    float(object_rows[-1]["iou_at_0.5"])
                    - float(object_rows[0]["iou_at_0.5"])
                ),
            }
        )
    by_click_mean = {str(key): _mean(value) for key, value in sorted(by_click.items())}
    return {
        "scene_id": str(manifest["scene_id"]),
        "object_count": len(manifest["records"]),
        "example_count": len(rows),
        "mean": _mean(rows),
        "by_click_count": by_click_mean,
        "click10_minus_click1_iou": (
            by_click_mean["10"]["iou_at_0.5"]
            - by_click_mean["1"]["iou_at_0.5"]
        ),
        "by_object": by_object,
    }


def _aggregate_scenes(rows: list[Mapping[str, object]]) -> dict[str, object]:
    return {
        "scene_macro_mean": _mean([row["mean"] for row in rows]),
        "scene_macro_click10_minus_click1_iou": sum(
            float(row["click10_minus_click1_iou"]) for row in rows
        )
        / len(rows),
        "scenes": rows,
    }


def _checkpoint_payload(
    head: MonotoneLikelihoodRatioHead,
    *,
    train_scenes: list[str],
    manifests: Mapping[str, Mapping[str, object]],
    preregistration: Path,
    recipe: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "artifact_type": CHECKPOINT_SCHEMA_V3,
        "head_class": "MonotoneLikelihoodRatioHead",
        "head_schema_version": head.schema_version,
        "affinity_channels": list(CAPABILITY_CHANNELS),
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "recipe": dict(recipe),
        "train_scene_ids": list(train_scenes),
        "dataset_manifests": {
            scene: {
                "path": manifests[scene]["_path"],
                "sha256": manifests[scene]["_sha256"],
            }
            for scene in train_scenes
        },
        "preregistration": {
            "path": str(preregistration),
            "sha256": _sha256(preregistration),
        },
        "safety": {
            "fit_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    preregistration = Path(args.preregistration).expanduser().resolve()
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    recipe = prereg["training_recipe"]
    if recipe.get("recipe_id") != RECIPE_ID:
        raise ValueError("v3 recipe differs from sealed preregistration")
    manifests = _load_manifests(args.dataset_manifest)
    device = torch.device(args.device)
    v2_path = Path(args.v2_checkpoint).expanduser().resolve()
    v2 = _load_v2(v2_path, device=device)
    readout_contexts = {
        scene: _scene_readout_context(manifest, device=device)
        for scene, manifest in manifests.items()
    }
    compiled_scenes = {}
    inventories = {}
    for scene, manifest in manifests.items():
        compiled_scenes[scene], inventories[scene] = _compile_scene_training(manifest)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = []
    for heldout in FIXED_FIT_SCENES:
        train_scenes = [scene for scene in FIXED_FIT_SCENES if scene != heldout]
        head, trace, training_inventory = _train_fold(
            compiled_scenes,
            inventories,
            train_scenes=train_scenes,
            recipe=recipe,
            device=device,
        )
        checkpoint = _write_torch_no_clobber(
            output_dir / f"loo_holdout_{heldout}.pt",
            _checkpoint_payload(
                head,
                train_scenes=train_scenes,
                manifests=manifests,
                preregistration=preregistration,
                recipe=recipe,
            ),
        )
        v2_metric = _evaluate_scene(
            v2,
            manifests[heldout],
            device=device,
            context=readout_contexts[heldout],
        )
        v3_metric = _evaluate_scene(
            head.eval(),
            manifests[heldout],
            device=device,
            context=readout_contexts[heldout],
        )
        folds.append(
            {
                "heldout_scene": heldout,
                "train_scenes": train_scenes,
                "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
                "epoch_trace": trace,
                "training_inventory": training_inventory,
                "v2": v2_metric,
                "v3": v3_metric,
                "mean_iou_gain": (
                    v3_metric["mean"]["iou_at_0.5"]
                    - v2_metric["mean"]["iou_at_0.5"]
                ),
                "absolute_log_mass_error_gain": (
                    v2_metric["mean"]["absolute_log_probability_mass_ratio_error"]
                    - v3_metric["mean"]["absolute_log_probability_mass_ratio_error"]
                ),
            }
        )

    all_head, all_trace, all_training_inventory = _train_fold(
        compiled_scenes,
        inventories,
        train_scenes=list(FIXED_FIT_SCENES),
        recipe=recipe,
        device=device,
    )
    all_checkpoint = _write_torch_no_clobber(
        output_dir / "all_fit_scene0000_0002_0005.pt",
        _checkpoint_payload(
            all_head,
            train_scenes=list(FIXED_FIT_SCENES),
            manifests=manifests,
            preregistration=preregistration,
            recipe=recipe,
        ),
    )
    all_v2 = _aggregate_scenes(
        [
            _evaluate_scene(
                v2,
                manifests[scene],
                device=device,
                context=readout_contexts[scene],
            )
            for scene in FIXED_FIT_SCENES
        ]
    )
    all_v3 = _aggregate_scenes(
        [
            _evaluate_scene(
                all_head.eval(),
                manifests[scene],
                device=device,
                context=readout_contexts[scene],
            )
            for scene in FIXED_FIT_SCENES
        ]
    )
    loo_mean_gain = sum(float(row["mean_iou_gain"]) for row in folds) / len(folds)
    loo_scene_passes = sum(float(row["mean_iou_gain"]) > 0.02 for row in folds)
    loo_click_pass = all(
        float(row["v3"]["click10_minus_click1_iou"]) > 0.02 for row in folds
    )
    loo_mass_pass = (
        sum(
            float(row["v3"]["mean"]["absolute_log_probability_mass_ratio_error"])
            for row in folds
        )
        < sum(
            float(row["v2"]["mean"]["absolute_log_probability_mass_ratio_error"])
            for row in folds
        )
    )
    all_iou_gain = (
        all_v3["scene_macro_mean"]["iou_at_0.5"]
        - all_v2["scene_macro_mean"]["iou_at_0.5"]
    )
    all_precision_gain = (
        all_v3["scene_macro_mean"]["precision_at_0.5"]
        - all_v2["scene_macro_mean"]["precision_at_0.5"]
    )
    gates = {
        "loo_macro_mean_iou_gain": loo_mean_gain,
        "loo_macro_mean_iou_gain_pass": loo_mean_gain > 0.02,
        "loo_scenes_with_iou_gain_over_0.02": loo_scene_passes,
        "loo_scene_count_pass": loo_scene_passes >= 2,
        "loo_each_scene_click_response_over_0.02_pass": loo_click_pass,
        "loo_mass_calibration_improves_pass": loo_mass_pass,
        "all_fit_mean_iou_gain": all_iou_gain,
        "all_fit_mean_iou_gain_pass": all_iou_gain > 0.02,
        "all_fit_precision_gain": all_precision_gain,
        "all_fit_precision_gain_pass": all_precision_gain > 0.02,
        "all_fit_click10_minus_click1_iou": all_v3[
            "scene_macro_click10_minus_click1_iou"
        ],
        "all_fit_click_response_pass": all_v3[
            "scene_macro_click10_minus_click1_iou"
        ]
        > 0.05,
    }
    passed = all(value for key, value in gates.items() if key.endswith("_pass"))
    receipt = {
        "schema_version": 3,
        "artifact_type": RECEIPT_SCHEMA_V3,
        "status": (
            "pass_source_gates_development_authorized"
            if passed
            else "fail_source_gates_do_not_open_development"
        ),
        "preregistration": {
            "path": str(preregistration),
            "sha256": _sha256(preregistration),
        },
        "dataset_manifests": {
            scene: {"path": row["_path"], "sha256": row["_sha256"]}
            for scene, row in manifests.items()
        },
        "reference_v2_checkpoint": {"path": str(v2_path), "sha256": _sha256(v2_path)},
        "leave_one_scene_out": folds,
        "all_fit": {
            "checkpoint": {"path": str(all_checkpoint), "sha256": _sha256(all_checkpoint)},
            "epoch_trace": all_trace,
            "training_inventory": all_training_inventory,
            "v2": all_v2,
            "v3": all_v3,
        },
        "gates": {**gates, "all_pass": passed, "development_authorized": passed},
        "safety": {
            "fit_scene_ids": list(FIXED_FIT_SCENES),
            "development_scene_id": "scene0003_00",
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
        },
    }
    receipt_path = _write_json_no_clobber(args.receipt, receipt)
    return receipt_path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", action="append", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--v2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--device", default="cpu")
    path, receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
