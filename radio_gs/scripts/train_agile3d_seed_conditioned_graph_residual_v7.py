"""Build source trajectory graph features and gate the structured AGILE v7 head."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.build_capability_likelihood_training_dataset import (
    CAPABILITY_CHANNELS,
    click_gaussian_mixture_weights,
)
from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_ratio_training_dataset import (
    FIXED_FIT_SCENES,
)
from radio_gs.querying.query_likelihood_head import (
    MonotoneOneSidedDensityRatioHead,
)
from radio_gs.querying.seed_conditioned_graph_residual import (
    SeedConditionedGraphResidualHead,
    nonnegative_seed_hop_stack,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.train_capability_likelihood_ratio_head import (
    _aggregate_scenes,
    _load_manifests,
    _load_shard,
    _metric_row,
    _scene_readout_context,
)
from radio_gs.scripts.train_capability_density_ratio_head import _numeric_mean
from radio_gs.scripts.train_query_likelihood_head_fixed import (
    _sha256,
    _write_json_no_clobber,
    _write_torch_no_clobber,
)


GRAPH_ARTIFACT = "agile3d-query-independent-typed-primitive-graph-v7"
CACHE_ARTIFACT = "agile3d-seed-conditioned-graph-hop-cache-v7"
CHECKPOINT_ARTIFACT = "agile3d-seed-conditioned-graph-residual-head-v7"
RECEIPT_ARTIFACT = "agile3d-seed-conditioned-graph-residual-source-gate-v7"
RECIPE_ID = (
    "seed-conditioned-typed-graph-residual-population-bce-rank025-"
    "adam-seed0-e100-lr0.02-v7"
)
V6_SHA = {
    "scene0000_00": "ceed9b63aa456b05bc19462049c0da72c00235adceec9b7a22f02a7fa0e3b8b5",
    "scene0002_00": "ec513b1e3974c7d623608b298c1bc1d91dc96782fda5ff93e96caf0f31daa54d",
    "scene0005_00": "d4325e731121256cc91fb4ec52a6e2ac8ac649d57e74e98dcf970098a00139d1",
    "all": "296f5aea618bb37bc0c65141b14e8389c214f021bc84ad3d05f738e426124bbe",
}


def _scene_map(values: list[str], *, required: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or key in result:
            raise ValueError("scene paths must be unique key=/absolute/path entries")
        result[key] = Path(raw_path).resolve()
    if set(result) != required:
        raise ValueError(f"scene path keys differ: {sorted(result)}")
    return result


def _load_graph(path: Path, *, scene_id: str, device: torch.device) -> tuple[PrimitiveSupportGraph, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    safety = payload.get("safety", {})
    if (
        payload.get("artifact_type") != GRAPH_ARTIFACT
        or payload.get("scene_id") != scene_id
        or safety.get("query_independent") is not True
        or safety.get("labels_opened") is not False
        or safety.get("clicks_opened") is not False
        or safety.get("development_labels_opened") is not False
        or safety.get("test_labels_opened") is not False
        or safety.get("point_as_primitive_used") is not False
    ):
        raise PermissionError("v7 graph violates query-independent source contract")
    graph = PrimitiveSupportGraph(
        edge_index=torch.as_tensor(payload["edge_index"]).long(),
        edge_weight=torch.as_tensor(payload["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(payload["raw_affinity"]).float(),
        local_sigma=torch.as_tensor(payload["local_sigma"]).float(),
        num_nodes=int(payload["num_nodes"]),
        edge_channels={
            name: torch.as_tensor(values).float()
            for name, values in payload["edge_channels"].items()
        },
    ).to(device)
    return graph, payload


def _load_bundle(path: Path, *, scene_id: str, expected_sha: str, device: torch.device) -> dict[str, object]:
    if _sha256(path) != expected_sha:
        raise ValueError("v7 canonical bundle differs from preregistration")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    safety = payload.get("safety", {})
    if (
        payload.get("scene_id") != scene_id
        or safety.get("query_independent") is not True
        or safety.get("gt_labels_opened") is not False
        or safety.get("test_labels_opened") is not False
        or safety.get("point_as_primitive_used") is not False
    ):
        raise PermissionError("v7 canonical bundle violates safety contract")
    return {
        "primitive_xyz": torch.as_tensor(payload["primitive_xyz"], device=device).float(),
        "primitive_covariance": torch.as_tensor(
            payload["primitive_covariance"], device=device
        ).float(),
        "primitive_opacity": torch.as_tensor(
            payload["primitive_opacity"], device=device
        ).float(),
        "official_point_xyz": torch.as_tensor(
            payload["official_point_xyz"], device=device
        ).float(),
        "point_candidate_indices": torch.as_tensor(
            payload["point_candidate_indices"], device=device
        ).long(),
    }


def _cumulative_seed_matrices(
    payload: Mapping[str, object],
    bundle: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    clicks = list(payload["clicks"])
    click_rows = torch.tensor(
        [int(click["point_index"]) for click in clicks],
        device=device,
        dtype=torch.long,
    )
    candidates = bundle["point_candidate_indices"].index_select(0, click_rows)
    mixture = click_gaussian_mixture_weights(
        primitive_xyz=bundle["primitive_xyz"],
        primitive_covariance=bundle["primitive_covariance"],
        primitive_opacity=bundle["primitive_opacity"],
        click_xyz=bundle["official_point_xyz"].index_select(0, click_rows),
        click_candidate_indices=candidates,
    ).to(device)
    node_count = int(bundle["primitive_xyz"].shape[0])
    click_seeds = torch.zeros((node_count, len(clicks)), device=device)
    for column in range(len(clicks)):
        click_seeds[:, column].scatter_reduce_(
            0,
            candidates[column],
            mixture[column],
            reduce="amax",
            include_self=True,
        )
    steps = list(payload["steps"])
    positive = torch.zeros((node_count, len(steps)), device=device)
    negative = torch.zeros_like(positive)
    for index, step in enumerate(steps):
        positive_columns = list(step["positive_columns"])
        negative_columns = list(step["negative_columns"])
        if positive_columns:
            positive[:, index] = click_seeds[:, positive_columns].amax(dim=1)
        if negative_columns:
            negative[:, index] = click_seeds[:, negative_columns].amax(dim=1)
    return positive, negative


@torch.inference_mode()
def _materialize_scene_cache(
    manifest: Mapping[str, object],
    *,
    graph_path: Path,
    bundle_path: Path,
    bundle_sha: str,
    output_dir: Path,
    device: torch.device,
    preregistration: Path,
) -> dict[str, object]:
    scene_id = str(manifest["scene_id"])
    graph, graph_payload = _load_graph(graph_path, scene_id=scene_id, device=device)
    bundle = _load_bundle(
        bundle_path, scene_id=scene_id, expected_sha=bundle_sha, device=device
    )
    records = []
    for record in manifest["records"]:
        shard_path = Path(record["shard"]["path"]).resolve()
        payload = _load_shard(record)
        object_id = int(payload["object_id"])
        cache_path = output_dir / scene_id / f"object_{object_id:04d}.v7.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (
                cached.get("artifact_type") != CACHE_ARTIFACT
                or cached.get("source_shard", {}).get("sha256") != record["shard"]["sha256"]
                or cached.get("typed_graph", {}).get("sha256") != _sha256(graph_path)
            ):
                raise ValueError("existing v7 hop cache differs from sealed inputs")
        else:
            positive, negative = _cumulative_seed_matrices(
                payload, bundle, device=device
            )
            positive_hops = nonnegative_seed_hop_stack(
                graph, positive, steps=4, decay=0.85
            )
            negative_hops = nonnegative_seed_hop_stack(
                graph, negative, steps=4, decay=0.85
            )
            hard_positive = positive >= 0.20
            hard_negative = (negative >= 0.20) & ~hard_positive
            cached = {
                "schema_version": 7,
                "artifact_type": CACHE_ARTIFACT,
                "scene_id": scene_id,
                "object_id": object_id,
                "hop_contrast": (positive_hops - negative_hops).half().cpu(),
                "hard_positive": hard_positive.cpu(),
                "hard_negative": hard_negative.cpu(),
                "source_shard": {
                    "path": str(shard_path),
                    "sha256": record["shard"]["sha256"],
                },
                "typed_graph": {"path": str(graph_path), "sha256": _sha256(graph_path)},
                "primitive_bundle": {"path": str(bundle_path), "sha256": bundle_sha},
                "preregistration": {
                    "path": str(preregistration),
                    "sha256": _sha256(preregistration),
                },
                "safety": {
                    "fit_labels_opened": True,
                    "labels_used_for_graph_or_hops": False,
                    "clicks_opened": True,
                    "development_labels_opened": False,
                    "test_labels_opened": False,
                    "test312_run": False,
                    "point_as_primitive_used": False,
                },
            }
            _write_torch_no_clobber(cache_path, cached)
        records.append(
            {
                "object_id": object_id,
                "path": str(cache_path.resolve()),
                "sha256": _sha256(cache_path),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "agile3d-seed-conditioned-graph-hop-cache-manifest-v7",
        "scene_id": scene_id,
        "object_count": len(records),
        "records": records,
        "typed_graph": {"path": str(graph_path), "sha256": _sha256(graph_path)},
        "primitive_bundle": {"path": str(bundle_path), "sha256": bundle_sha},
        "graph_safety": graph_payload["safety"],
        "safety": {
            "fit_labels_opened": True,
            "labels_used_for_graph_or_hops": False,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
        },
    }


def _load_v6(path: Path, *, key: str, device: torch.device) -> MonotoneOneSidedDensityRatioHead:
    if _sha256(path) != V6_SHA[key]:
        raise ValueError("frozen v6 checkpoint SHA differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("head_class") != "MonotoneOneSidedDensityRatioHead":
        raise ValueError("v7 base checkpoint is not the frozen v6 head")
    head = MonotoneOneSidedDensityRatioHead(
        affinity_channel_count=len(CAPABILITY_CHANNELS)
    ).to(device)
    head.load_state_dict(payload["state_dict"], strict=True)
    return head.eval()


def _base_llr(
    head: MonotoneOneSidedDensityRatioHead,
    affinity: torch.Tensor,
    rows: torch.Tensor,
    step: Mapping[str, object],
) -> torch.Tensor:
    selected = affinity.index_select(0, rows).float()
    per_observation = head.per_observation_log_likelihood_ratio(selected)
    positive_columns = list(step["positive_columns"])
    negative_columns = list(step["negative_columns"])
    positive = (
        per_observation[:, positive_columns, :].sum(dim=(1, 2))
        if positive_columns
        else torch.zeros(len(rows), device=affinity.device)
    )
    negative = (
        F.relu(per_observation[:, negative_columns, :]).sum(dim=(1, 2))
        if negative_columns
        else torch.zeros(len(rows), device=affinity.device)
    )
    return positive - negative


@torch.inference_mode()
def _compile_training(
    manifests: Mapping[str, Mapping[str, object]],
    cache_manifests: Mapping[str, Mapping[str, object]],
    *,
    train_scenes: list[str],
    base_head: MonotoneOneSidedDensityRatioHead,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor | int], dict[str, object]]:
    output: dict[str, list[torch.Tensor]] = defaultdict(list)
    included = []
    excluded = []
    example_count = 0
    for scene_id in train_scenes:
        cache_by_object = {
            int(row["object_id"]): row for row in cache_manifests[scene_id]["records"]
        }
        for record in manifests[scene_id]["records"]:
            payload = _load_shard(record)
            pi = float(payload["primitive_foreground_prevalence"])
            inventory = {"scene_id": scene_id, "object_id": int(payload["object_id"]), "prevalence": pi}
            if not 0 < pi < 1:
                excluded.append(inventory)
                continue
            included.append(inventory)
            cache_record = cache_by_object[int(payload["object_id"])]
            if _sha256(cache_record["path"]) != cache_record["sha256"]:
                raise ValueError("v7 hop cache changed")
            cache = torch.load(cache_record["path"], map_location="cpu", weights_only=True)
            affinity = torch.as_tensor(payload["capability_click_affinity"], device=device)
            contrast = torch.as_tensor(cache["hop_contrast"], device=device).float()
            hard_positive = torch.as_tensor(cache["hard_positive"], device=device).bool()
            hard_negative = torch.as_tensor(cache["hard_negative"], device=device).bool()
            prior_logit = math.log(pi / (1.0 - pi))
            for step_index, step in enumerate(payload["steps"]):
                positive_rows = torch.as_tensor(
                    step["positive_training_rows"], device=device
                ).long()
                negative_rows = torch.as_tensor(
                    step["negative_training_rows"], device=device
                ).long()
                # Anchors are exact constraints, not trainable residual targets.
                positive_rows = positive_rows[
                    ~(hard_positive[positive_rows, step_index] | hard_negative[positive_rows, step_index])
                ]
                negative_rows = negative_rows[
                    ~(hard_positive[negative_rows, step_index] | hard_negative[negative_rows, step_index])
                ]
                if not len(positive_rows) or not len(negative_rows):
                    continue
                positive_base = _base_llr(base_head, affinity, positive_rows, step)
                negative_base = _base_llr(base_head, affinity, negative_rows, step)
                positive_design = contrast[positive_rows, step_index]
                negative_design = contrast[negative_rows, step_index]
                pair_count = min(len(positive_rows), len(negative_rows))
                output["positive_base"].append(positive_base.cpu())
                output["negative_base"].append(negative_base.cpu())
                output["positive_design"].append(positive_design.cpu())
                output["negative_design"].append(negative_design.cpu())
                output["positive_prior"].append(torch.full((len(positive_rows),), prior_logit))
                output["negative_prior"].append(torch.full((len(negative_rows),), prior_logit))
                output["positive_coefficient"].append(torch.full((len(positive_rows),), pi / len(positive_rows)))
                output["negative_coefficient"].append(torch.full((len(negative_rows),), (1.0 - pi) / len(negative_rows)))
                output["ranking_base"].append((positive_base[:pair_count] - negative_base[:pair_count]).cpu())
                output["ranking_design"].append((positive_design[:pair_count] - negative_design[:pair_count]).cpu())
                output["ranking_coefficient"].append(torch.full((pair_count,), 1.0 / pair_count))
                example_count += 1
            del affinity, contrast, hard_positive, hard_negative, cache
    if not example_count:
        raise ValueError("v7 training fold has no non-anchor class pairs")
    compiled = {key: torch.cat(value) for key, value in output.items()}
    compiled["example_count"] = example_count
    return compiled, {
        "included_object_count": len(included),
        "excluded_zero_or_full_primitive_support": excluded,
        "training_step_count": example_count,
        "anchor_rows_excluded_from_trainable_objective": True,
        "metric_object_subselection": False,
    }


def _train_residual(
    base_head: MonotoneOneSidedDensityRatioHead,
    compiled: Mapping[str, torch.Tensor | int],
    *,
    recipe: Mapping[str, object],
    device: torch.device,
) -> tuple[SeedConditionedGraphResidualHead, list[dict[str, object]]]:
    torch.manual_seed(int(recipe["seed"]))
    head = SeedConditionedGraphResidualHead(
        base_head,
        propagation_steps=4,
        propagation_decay=0.85,
        max_logit_residual=4.0,
        hard_seed_threshold=0.20,
    ).to(device)
    optimizer = torch.optim.Adam(
        [head.raw_residual_gate, head.raw_hop_weights],
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    data = {
        key: torch.as_tensor(value, device=device)
        for key, value in compiled.items()
        if key != "example_count"
    }
    count = int(compiled["example_count"])
    trace = []
    for epoch in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        hop_weight = head.hop_weights
        gate = head.residual_gate
        positive_score = data["positive_base"] + gate * (data["positive_design"] @ hop_weight)
        negative_score = data["negative_base"] + gate * (data["negative_design"] @ hop_weight)
        positive_bce = (
            data["positive_coefficient"] * F.softplus(-(positive_score + data["positive_prior"]))
        ).sum() / count
        negative_bce = (
            data["negative_coefficient"] * F.softplus(negative_score + data["negative_prior"])
        ).sum() / count
        ranking_difference = data["ranking_base"] + gate * (data["ranking_design"] @ hop_weight)
        ranking = (
            data["ranking_coefficient"] * F.softplus(-ranking_difference)
        ).sum() / count
        objective = positive_bce + negative_bce + 0.25 * ranking
        objective.backward()
        optimizer.step()
        if epoch in {0, 1, 2, 4, 9, 19, 49, 99}:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "objective": float(objective.detach()),
                    "posterior_bce": float((positive_bce + negative_bce).detach()),
                    "ranking": float(ranking.detach()),
                    "residual_gate": float(head.residual_gate.detach()),
                    "hop_weights": head.hop_weights.detach().cpu().tolist(),
                }
            )
    return head.eval(), trace


@torch.inference_mode()
def _evaluate_scene_v7(
    head: SeedConditionedGraphResidualHead,
    manifest: Mapping[str, object],
    cache_manifest: Mapping[str, object],
    *,
    device: torch.device,
) -> dict[str, object]:
    context = _scene_readout_context(manifest, device=device)
    cache_by_object = {
        int(row["object_id"]): row for row in cache_manifest["records"]
    }
    rows = []
    by_click: dict[int, list[dict[str, float]]] = defaultdict(list)
    by_object = []
    for record in manifest["records"]:
        payload = _load_shard(record)
        cache_record = cache_by_object[int(payload["object_id"])]
        if _sha256(cache_record["path"]) != cache_record["sha256"]:
            raise ValueError("v7 evaluation hop cache changed")
        cache = torch.load(cache_record["path"], map_location="cpu", weights_only=True)
        affinity = torch.as_tensor(payload["capability_click_affinity"], device=device)
        contrast = torch.as_tensor(cache["hop_contrast"], device=device).float()
        hard_positive = torch.as_tensor(cache["hard_positive"], device=device).bool()
        hard_negative = torch.as_tensor(cache["hard_negative"], device=device).bool()
        point_target = torch.as_tensor(payload["point_target"], device=device).bool()
        object_rows = []
        for step_index, step in enumerate(payload["steps"]):
            all_rows = torch.arange(affinity.shape[0], device=device)
            base = _base_llr(head.base_head, affinity, all_rows, step)
            score = base + head.residual_gate * (
                contrast[:, step_index] @ head.hop_weights
            )
            probability = torch.sigmoid(score)
            probability = torch.where(
                hard_negative[:, step_index], torch.zeros_like(probability), probability
            )
            probability = torch.where(
                hard_positive[:, step_index], torch.ones_like(probability), probability
            )
            point_probability = (
                context["weights"] * probability[context["indices"]]
            ).sum(dim=1)
            metric = _metric_row(point_probability, point_target)
            click_count = int(step["click_count"])
            metric.update(
                {
                    "scene_id": str(payload["scene_id"]),
                    "object_id": int(payload["object_id"]),
                    "click_count": click_count,
                }
            )
            rows.append(metric)
            object_rows.append(metric)
            by_click[click_count].append(metric)
        by_object.append(
            {
                "object_id": int(payload["object_id"]),
                "mean": _numeric_mean(object_rows),
                "click10_minus_click1_iou": object_rows[-1]["iou_at_0.5"] - object_rows[0]["iou_at_0.5"],
            }
        )
    click_mean = {str(key): _numeric_mean(value) for key, value in sorted(by_click.items())}
    return {
        "scene_id": str(manifest["scene_id"]),
        "object_count": len(manifest["records"]),
        "example_count": len(rows),
        "mean": _numeric_mean(rows),
        "by_click_count": click_mean,
        "click10_minus_click1_iou": click_mean["10"]["iou_at_0.5"] - click_mean["1"]["iou_at_0.5"],
        "by_object": by_object,
    }


def _checkpoint(
    head: SeedConditionedGraphResidualHead,
    *,
    base_checkpoint: Path,
    train_scenes: list[str],
    cache_manifests: Mapping[str, Mapping[str, object]],
    preregistration: Path,
    recipe: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 7,
        "artifact_type": CHECKPOINT_ARTIFACT,
        "head_class": "SeedConditionedGraphResidualHead",
        "head_schema_version": head.schema_version,
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "residual_gate": head.residual_gate.detach().cpu(),
        "hop_weights": head.hop_weights.detach().cpu(),
        "base_v6_checkpoint": {"path": str(base_checkpoint), "sha256": _sha256(base_checkpoint)},
        "train_scene_ids": train_scenes,
        "cache_manifests": {
            scene: {
                "path": str(cache_manifests[scene]["_path"]),
                "sha256": cache_manifests[scene]["_sha256"],
            }
            for scene in train_scenes
        },
        "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
        "recipe": dict(recipe),
        "safety": {
            "fit_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    preregistration = Path(args.preregistration).resolve()
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    recipe = prereg["training_recipe"]
    if recipe.get("recipe_id") != RECIPE_ID:
        raise ValueError("v7 training recipe differs from sealed preregistration")
    manifests = _load_manifests(args.dataset_manifest)
    graph_paths = _scene_map(args.typed_graph, required=set(FIXED_FIT_SCENES))
    bundle_paths = _scene_map(args.primitive_bundle, required=set(FIXED_FIT_SCENES))
    v6_paths = _scene_map(
        args.v6_checkpoint, required=set(FIXED_FIT_SCENES) | {"all"}
    )
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_manifests = {}
    for scene_id in FIXED_FIT_SCENES:
        cache = _materialize_scene_cache(
            manifests[scene_id],
            graph_path=graph_paths[scene_id],
            bundle_path=bundle_paths[scene_id],
            bundle_sha=prereg["data_contract"]["canonical_gaussian_bundle_sha256"][scene_id],
            output_dir=output_dir / "hop_cache",
            device=device,
            preregistration=preregistration,
        )
        cache_path = _write_json_no_clobber(
            output_dir / "hop_cache" / scene_id / "manifest.json", cache
        )
        cache["_path"] = str(cache_path)
        cache["_sha256"] = _sha256(cache_path)
        cache_manifests[scene_id] = cache
        print(json.dumps({"cache_complete": scene_id, "object_count": cache["object_count"]}), flush=True)

    v6_authority_path = Path(args.v6_source_receipt).resolve()
    if _sha256(v6_authority_path) != "a2c3ea6845eafcd4674d00415c005ef73213017c593b2bc4dd398d0edf3b5a02":
        raise ValueError("frozen v6 source authority differs")
    v6_authority = json.loads(v6_authority_path.read_text(encoding="utf-8"))
    v2_loo = {row["heldout_scene"]: row["v2"] for row in v6_authority["leave_one_scene_out"]}
    v2_all = v6_authority["all_fit"]["v2"]

    folds = []
    for heldout in FIXED_FIT_SCENES:
        train_scenes = [scene for scene in FIXED_FIT_SCENES if scene != heldout]
        base = _load_v6(v6_paths[heldout], key=heldout, device=device)
        compiled, inventory = _compile_training(
            manifests,
            cache_manifests,
            train_scenes=train_scenes,
            base_head=base,
            device=device,
        )
        head, trace = _train_residual(base, compiled, recipe=recipe, device=device)
        checkpoint = _write_torch_no_clobber(
            output_dir / f"loo_holdout_{heldout}.pt",
            _checkpoint(
                head,
                base_checkpoint=v6_paths[heldout],
                train_scenes=train_scenes,
                cache_manifests=cache_manifests,
                preregistration=preregistration,
                recipe=recipe,
            ),
        )
        metric = _evaluate_scene_v7(
            head, manifests[heldout], cache_manifests[heldout], device=device
        )
        reference = v2_loo[heldout]
        folds.append(
            {
                "heldout_scene": heldout,
                "train_scenes": train_scenes,
                "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
                "epoch_trace": trace,
                "training_inventory": inventory,
                "v2": reference,
                "v7": metric,
                "iou_gain": metric["mean"]["iou_at_0.5"] - reference["mean"]["iou_at_0.5"],
                "precision_gain": metric["mean"]["precision_at_0.5"] - reference["mean"]["precision_at_0.5"],
                "absolute_log_mass_error_gain": reference["mean"]["absolute_log_probability_mass_ratio_error"] - metric["mean"]["absolute_log_probability_mass_ratio_error"],
            }
        )
        print(json.dumps({"fold_complete": heldout, "iou_gain": folds[-1]["iou_gain"]}), flush=True)
        del compiled, head, base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base = _load_v6(v6_paths["all"], key="all", device=device)
    compiled, inventory = _compile_training(
        manifests,
        cache_manifests,
        train_scenes=list(FIXED_FIT_SCENES),
        base_head=base,
        device=device,
    )
    head, trace = _train_residual(base, compiled, recipe=recipe, device=device)
    checkpoint = _write_torch_no_clobber(
        output_dir / "all_fit_scene0000_0002_0005.pt",
        _checkpoint(
            head,
            base_checkpoint=v6_paths["all"],
            train_scenes=list(FIXED_FIT_SCENES),
            cache_manifests=cache_manifests,
            preregistration=preregistration,
            recipe=recipe,
        ),
    )
    all_v7 = _aggregate_scenes(
        [
            _evaluate_scene_v7(head, manifests[scene], cache_manifests[scene], device=device)
            for scene in FIXED_FIT_SCENES
        ]
    )
    loo_iou_gain = sum(row["iou_gain"] for row in folds) / len(folds)
    loo_precision_gain = sum(row["precision_gain"] for row in folds) / len(folds)
    loo_mass_gain = sum(row["absolute_log_mass_error_gain"] for row in folds) / len(folds)
    all_iou_gain = all_v7["scene_macro_mean"]["iou_at_0.5"] - v2_all["scene_macro_mean"]["iou_at_0.5"]
    all_precision_gain = all_v7["scene_macro_mean"]["precision_at_0.5"] - v2_all["scene_macro_mean"]["precision_at_0.5"]
    all_mass_gain = v2_all["scene_macro_mean"]["absolute_log_probability_mass_ratio_error"] - all_v7["scene_macro_mean"]["absolute_log_probability_mass_ratio_error"]
    gates = {
        "loo_macro_iou_gain": loo_iou_gain,
        "loo_macro_iou_pass": loo_iou_gain > 0,
        "loo_macro_precision_gain": loo_precision_gain,
        "loo_macro_precision_pass": loo_precision_gain > 0,
        "loo_macro_absolute_log_mass_error_gain": loo_mass_gain,
        "loo_macro_mass_pass": loo_mass_gain > 0,
        "loo_each_scene_click_response_positive_pass": all(row["v7"]["click10_minus_click1_iou"] > 0 for row in folds),
        "all_fit_iou_gain": all_iou_gain,
        "all_fit_iou_pass": all_iou_gain > 0,
        "all_fit_precision_gain": all_precision_gain,
        "all_fit_precision_pass": all_precision_gain > 0,
        "all_fit_absolute_log_mass_error_gain": all_mass_gain,
        "all_fit_mass_pass": all_mass_gain > 0,
        "all_fit_click_response": all_v7["scene_macro_click10_minus_click1_iou"],
        "all_fit_click_response_pass": all_v7["scene_macro_click10_minus_click1_iou"] > 0,
    }
    passed = all(value for key, value in gates.items() if key.endswith("_pass"))
    receipt = {
        "schema_version": 7,
        "artifact_type": RECEIPT_ARTIFACT,
        "status": "source_gate_pass" if passed else "source_gate_failed_stop_before_development",
        "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
        "dataset_manifests": {
            scene: {"path": manifests[scene]["_path"], "sha256": manifests[scene]["_sha256"]}
            for scene in FIXED_FIT_SCENES
        },
        "cache_manifests": {
            scene: {"path": cache_manifests[scene]["_path"], "sha256": cache_manifests[scene]["_sha256"]}
            for scene in FIXED_FIT_SCENES
        },
        "leave_one_scene_out": folds,
        "all_fit": {
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "epoch_trace": trace,
            "training_inventory": inventory,
            "v2": v2_all,
            "v7": all_v7,
        },
        "gates": gates,
        "development_authorized": passed,
        "safety": {
            "fit_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
        },
    }
    receipt_path = _write_json_no_clobber(output_dir / "source_gate_receipt.json", receipt)
    return receipt_path, receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", action="append", required=True)
    parser.add_argument("--typed-graph", action="append", required=True)
    parser.add_argument("--primitive-bundle", action="append", required=True)
    parser.add_argument("--v6-checkpoint", action="append", required=True)
    parser.add_argument("--v6-source-receipt", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    path, receipt = run(parse_args())
    print(json.dumps({"receipt": str(path), "status": receipt["status"], "gates": receipt["gates"]}, indent=2))


if __name__ == "__main__":
    main()
