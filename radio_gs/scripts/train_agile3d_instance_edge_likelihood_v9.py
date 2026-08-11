"""Train source-only instance edge LRs and gate AGILE v9 trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_ratio_training_dataset import FIXED_FIT_SCENES
from radio_gs.querying.absorbing_extent_head import finite_absorbing_seed_reach
from radio_gs.querying.instance_edge_likelihood import (
    MonotoneInstanceEdgeLikelihood,
    gate_graph_by_instance_edge_likelihood,
    instance_edge_features_from_graph,
)
from radio_gs.scripts.evaluate_agile3d_absorbing_extent_v8_source import _evaluate_scene
from radio_gs.scripts.train_agile3d_seed_conditioned_graph_residual_v7 import (
    _cumulative_seed_matrices,
    _load_bundle,
    _load_graph,
    _load_v6,
    _scene_map,
)
from radio_gs.scripts.train_capability_likelihood_ratio_head import _aggregate_scenes, _load_manifests, _load_shard
from radio_gs.scripts.train_query_likelihood_head_fixed import _sha256, _write_json_no_clobber, _write_torch_no_clobber


RECIPE_ID = "query-independent-instance-edge-lr-population-bce-rank025-adam-seed0-e100-lr0.02-v9"
EDGE_CHECKPOINT = "agile3d-query-independent-instance-edge-likelihood-checkpoint-v9"
EXTENT_CACHE = "agile3d-instance-edge-absorbing-extent-cache-v9"
SOURCE_RECEIPT = "agile3d-instance-edge-likelihood-source-gate-v9"


def _primitive_instance_labels(
    manifest: Mapping[str, object], *, node_count: int
) -> tuple[torch.Tensor, dict[str, object]]:
    labels = torch.zeros(node_count, dtype=torch.long)
    conflicts = torch.zeros(node_count, dtype=torch.bool)
    for record in manifest["records"]:
        payload = _load_shard(record)
        target = torch.as_tensor(payload["primitive_target"]).bool()
        if target.shape != (node_count,):
            raise ValueError("source primitive targets do not align with graph")
        object_id = int(payload["object_id"])
        overlap = target & (labels > 0) & (labels != object_id)
        conflicts |= overlap
        labels[target & ~overlap] = object_id
    labels[conflicts] = -1
    return labels, {
        "assigned_instance_primitive_count": int((labels > 0).sum()),
        "background_or_unknown_primitive_count": int((labels == 0).sum()),
        "conflicting_primitive_count": int((labels < 0).sum()),
    }


def _load_authority_bundle(path: Path, *, expected_sha: str) -> dict[str, object]:
    if _sha256(path) != expected_sha:
        raise ValueError("v9 canonical bundle SHA differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    safety = payload.get("safety", {})
    if (
        safety.get("query_independent") is not True
        or safety.get("gt_labels_opened") is not False
        or safety.get("test_labels_opened") is not False
        or safety.get("point_as_primitive_used") is not False
    ):
        raise PermissionError("v9 bundle violates query-independent contract")
    return payload


@torch.inference_mode()
def _compile_scene_edges(
    manifest: Mapping[str, object],
    *,
    graph_path: Path,
    bundle_path: Path,
    bundle_sha: str,
) -> tuple[dict[str, torch.Tensor | float | int], dict[str, object]]:
    scene_id = str(manifest["scene_id"])
    graph, _payload = _load_graph(graph_path, scene_id=scene_id, device=torch.device("cpu"))
    bundle = _load_authority_bundle(bundle_path, expected_sha=bundle_sha)
    labels, inventory = _primitive_instance_labels(manifest, node_count=graph.num_nodes)
    features = instance_edge_features_from_graph(
        graph,
        reliability=torch.as_tensor(bundle["reliability"]),
        coverage=torch.as_tensor(bundle["coverage"]),
    ).matrix()
    row, col = graph.edge_index
    unique = row < col
    authoritative = (labels[row] > 0) & (labels[col] > 0)
    keep = unique & authoritative
    selected = features[keep]
    target = labels[row[keep]] == labels[col[keep]]
    positive = int(target.sum())
    negative = int((~target).sum())
    if not positive or not negative:
        raise ValueError("source scene lacks both authoritative edge classes")
    prevalence = positive / len(target)
    inventory.update(
        {
            "unique_candidate_edge_count": int(unique.sum()),
            "authoritative_edge_count": int(keep.sum()),
            "omitted_unknown_edge_count": int(unique.sum() - keep.sum()),
            "same_instance_edge_count": positive,
            "different_instance_edge_count": negative,
            "same_instance_prevalence": prevalence,
        }
    )
    return {
        "features": selected,
        "target": target,
        "prevalence": prevalence,
        "positive_count": positive,
        "negative_count": negative,
    }, inventory


def _train_head(
    compiled: Mapping[str, Mapping[str, torch.Tensor | float | int]],
    *,
    train_scenes: list[str],
    recipe: Mapping[str, object],
    device: torch.device,
) -> tuple[MonotoneInstanceEdgeLikelihood, list[dict[str, object]]]:
    torch.manual_seed(int(recipe["seed"]))
    head = MonotoneInstanceEdgeLikelihood().to(device)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=float(recipe["learning_rate"]), weight_decay=float(recipe["weight_decay"])
    )
    data = {}
    for scene in train_scenes:
        features = torch.as_tensor(compiled[scene]["features"], device=device)
        target = torch.as_tensor(compiled[scene]["target"], device=device).bool()
        data[scene] = (features[target], features[~target], float(compiled[scene]["prevalence"]))
    trace = []
    for epoch in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        posterior_terms = []
        ranking_terms = []
        for scene in train_scenes:
            positive, negative, prevalence = data[scene]
            positive_score = head.bias + positive @ head.signed_weights
            negative_score = head.bias + negative @ head.signed_weights
            prior = math.log(prevalence / (1.0 - prevalence))
            posterior_terms.append(
                prevalence * F.softplus(-(positive_score + prior)).mean()
                + (1.0 - prevalence) * F.softplus(negative_score + prior).mean()
            )
            pair_count = min(len(positive_score), len(negative_score))
            ranking_terms.append(
                F.softplus(-(positive_score[:pair_count] - negative_score[:pair_count])).mean()
            )
        posterior = torch.stack(posterior_terms).mean()
        ranking = torch.stack(ranking_terms).mean()
        objective = posterior + 0.25 * ranking
        objective.backward()
        optimizer.step()
        if epoch in {0, 1, 2, 4, 9, 19, 49, 99}:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "objective": float(objective.detach()),
                    "posterior_bce": float(posterior.detach()),
                    "ranking": float(ranking.detach()),
                    "bias": float(head.bias.detach()),
                    "signed_weights": head.signed_weights.detach().cpu().tolist(),
                }
            )
    return head.eval(), trace


@torch.inference_mode()
def _edge_metrics(
    head: MonotoneInstanceEdgeLikelihood,
    compiled: Mapping[str, torch.Tensor | float | int],
    *,
    device: torch.device,
) -> dict[str, float]:
    features = torch.as_tensor(compiled["features"], device=device)
    target = torch.as_tensor(compiled["target"], device=device).bool()
    score = head.bias + features @ head.signed_weights
    prediction = score >= 0
    tp = int((prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    positive_recall = tp / max(1, tp + fn)
    negative_recall = tn / max(1, tn + fp)
    return {
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
        "same_instance_precision": tp / max(1, tp + fp),
        "same_instance_recall": positive_recall,
        "different_instance_recall": negative_recall,
        "eligible_edge_keep_fraction": float(prediction.float().mean()),
        "mean_positive_llr": float(score[target].mean()),
        "mean_negative_llr": float(score[~target].mean()),
    }


def _component_labels(graph) -> torch.Tensor:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    row = graph.edge_index[0].detach().cpu().numpy()
    col = graph.edge_index[1].detach().cpu().numpy()
    keep = graph.edge_weight.detach().cpu().numpy() > 0
    adjacency = coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.uint8), (row[keep], col[keep])),
        shape=(graph.num_nodes, graph.num_nodes),
    ).tocsr()
    count, labels = connected_components(adjacency, directed=False, return_labels=True)
    return torch.from_numpy(labels).long(), int(count)


def _positive_component_support(labels: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    seeds_cpu = torch.as_tensor(seeds).bool().cpu()
    support = torch.zeros_like(seeds_cpu)
    for column in range(seeds_cpu.shape[1]):
        rows = torch.nonzero(seeds_cpu[:, column], as_tuple=False).flatten()
        if rows.numel():
            support[:, column] = torch.isin(labels, labels[rows].unique())
    return support


@torch.inference_mode()
def _materialize_extent_cache(
    manifest: Mapping[str, object],
    *,
    gated_graph,
    edge_checkpoint: Path,
    bundle_path: Path,
    bundle_sha: str,
    output_dir: Path,
    device: torch.device,
    preregistration: Path,
) -> dict[str, object]:
    scene_id = str(manifest["scene_id"])
    labels, component_count = _component_labels(gated_graph)
    bundle = _load_bundle(bundle_path, scene_id=scene_id, expected_sha=bundle_sha, device=device)
    graph_device = gated_graph if gated_graph.edge_index.device == device else gated_graph.to(device)
    records = []
    for record in manifest["records"]:
        payload = _load_shard(record)
        object_id = int(payload["object_id"])
        cache_path = output_dir / scene_id / f"object_{object_id:04d}.v9.pt"
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (
                cache.get("artifact_type") != EXTENT_CACHE
                or cache.get("edge_checkpoint", {}).get("sha256") != _sha256(edge_checkpoint)
                or cache.get("source_shard", {}).get("sha256") != record["shard"]["sha256"]
            ):
                raise ValueError("existing v9 extent cache differs")
        else:
            positive, negative = _cumulative_seed_matrices(payload, bundle, device=device)
            hard_positive = positive >= 0.20
            hard_negative = (negative >= 0.20) & ~hard_positive
            support = _positive_component_support(labels, hard_positive).to(device)
            negative_reach = finite_absorbing_seed_reach(
                graph_device, hard_negative.float(), steps=12
            )
            cache = {
                "schema_version": 9,
                "artifact_type": EXTENT_CACHE,
                "scene_id": scene_id,
                "object_id": object_id,
                "positive_extent_support": support.cpu(),
                "negative_reach": negative_reach.half().cpu(),
                "hard_positive": hard_positive.cpu(),
                "hard_negative": hard_negative.cpu(),
                "retained_component_count": component_count,
                "source_shard": {"path": record["shard"]["path"], "sha256": record["shard"]["sha256"]},
                "edge_checkpoint": {"path": str(edge_checkpoint), "sha256": _sha256(edge_checkpoint)},
                "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
                "safety": {
                    "fit_labels_opened": True,
                    "labels_used_for_query_extent": False,
                    "development_labels_opened": False,
                    "test_labels_opened": False,
                    "test312_run": False,
                    "point_as_primitive_used": False,
                },
            }
            _write_torch_no_clobber(cache_path, cache)
        records.append({"object_id": object_id, "path": str(cache_path.resolve()), "sha256": _sha256(cache_path)})
    return {
        "schema_version": 1,
        "artifact_type": "agile3d-instance-edge-extent-cache-manifest-v9",
        "scene_id": scene_id,
        "object_count": len(records),
        "retained_component_count": component_count,
        "records": records,
        "edge_checkpoint": {"path": str(edge_checkpoint), "sha256": _sha256(edge_checkpoint)},
        "safety": {
            "fit_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
        },
    }


def _checkpoint(
    head: MonotoneInstanceEdgeLikelihood,
    *,
    train_scenes: list[str],
    preregistration: Path,
    trace: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 9,
        "artifact_type": EDGE_CHECKPOINT,
        "head_class": "MonotoneInstanceEdgeLikelihood",
        "head_schema_version": head.schema_version,
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "signed_weights": head.signed_weights.detach().cpu(),
        "train_scene_ids": train_scenes,
        "epoch_trace": trace,
        "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
        "safety": {
            "source_fit_edge_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
        },
    }


def _gated_graph(head, graph_path, bundle_path, scene_id, bundle_sha, device):
    graph, _ = _load_graph(graph_path, scene_id=scene_id, device=device)
    bundle = _load_authority_bundle(bundle_path, expected_sha=bundle_sha)
    features = instance_edge_features_from_graph(
        graph,
        reliability=torch.as_tensor(bundle["reliability"], device=device),
        coverage=torch.as_tensor(bundle["coverage"], device=device),
    )
    return gate_graph_by_instance_edge_likelihood(
        graph, features, head, apply_edge_likelihood=True
    )


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    preregistration = Path(args.preregistration).resolve()
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    recipe = prereg["training_recipe"]
    if recipe.get("recipe_id") != RECIPE_ID:
        raise ValueError("v9 recipe differs from preregistration")
    manifests = _load_manifests(args.dataset_manifest)
    graph_paths = _scene_map(args.typed_graph, required=set(FIXED_FIT_SCENES))
    bundle_paths = _scene_map(args.primitive_bundle, required=set(FIXED_FIT_SCENES))
    v6_paths = _scene_map(args.v6_checkpoint, required=set(FIXED_FIT_SCENES) | {"all"})
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled = {}
    edge_inventory = {}
    for scene in FIXED_FIT_SCENES:
        if (
            manifests[scene]["_sha256"] != prereg["data_contract"]["dataset_manifest_sha256"][scene]
            or _sha256(graph_paths[scene]) != prereg["data_contract"]["typed_graph_sha256"][scene]
        ):
            raise ValueError("v9 source authority differs")
        compiled[scene], edge_inventory[scene] = _compile_scene_edges(
            manifests[scene],
            graph_path=graph_paths[scene],
            bundle_path=bundle_paths[scene],
            bundle_sha=prereg["data_contract"]["canonical_bundle_sha256"][scene],
        )
        print(json.dumps({"edge_data": scene, **edge_inventory[scene]}), flush=True)

    v8_receipt_path = Path(args.v8_source_receipt).resolve()
    if _sha256(v8_receipt_path) != "fb0cda2700de11d32df999aea2ba0ef698a60bd7c42046f3ae121d2caee18b9a":
        raise ValueError("v8 source authority differs")
    v8_receipt = json.loads(v8_receipt_path.read_text(encoding="utf-8"))
    v8_loo = {row["heldout_scene"]: row["v8"] for row in v8_receipt["leave_one_scene_out"]}
    v2_loo = {row["heldout_scene"]: row["v2"] for row in v8_receipt["leave_one_scene_out"]}
    v8_all = v8_receipt["all_fit"]["v8"]
    v2_all = v8_receipt["all_fit"]["v2"]

    folds = []
    for heldout in FIXED_FIT_SCENES:
        train_scenes = [scene for scene in FIXED_FIT_SCENES if scene != heldout]
        head, trace = _train_head(compiled, train_scenes=train_scenes, recipe=recipe, device=device)
        checkpoint = _write_torch_no_clobber(
            output_dir / f"loo_holdout_{heldout}.edge_head.pt",
            _checkpoint(head, train_scenes=train_scenes, preregistration=preregistration, trace=trace),
        )
        edge_metric = _edge_metrics(head, compiled[heldout], device=device)
        graph = _gated_graph(
            head,
            graph_paths[heldout],
            bundle_paths[heldout],
            heldout,
            prereg["data_contract"]["canonical_bundle_sha256"][heldout],
            device,
        )
        cache = _materialize_extent_cache(
            manifests[heldout],
            gated_graph=graph,
            edge_checkpoint=checkpoint,
            bundle_path=bundle_paths[heldout],
            bundle_sha=prereg["data_contract"]["canonical_bundle_sha256"][heldout],
            output_dir=output_dir / "extent_cache" / f"loo_{heldout}",
            device=device,
            preregistration=preregistration,
        )
        cache_path = _write_json_no_clobber(
            output_dir / "extent_cache" / f"loo_{heldout}" / heldout / "manifest.json", cache
        )
        cache["_path"] = str(cache_path)
        cache["_sha256"] = _sha256(cache_path)
        base = _load_v6(v6_paths[heldout], key=heldout, device=device)
        metric = _evaluate_scene(base, manifests[heldout], cache, device=device)
        folds.append(
            {
                "heldout_scene": heldout,
                "train_scenes": train_scenes,
                "edge_checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
                "edge_metrics": edge_metric,
                "retained_component_count": cache["retained_component_count"],
                "v2": v2_loo[heldout],
                "v8": v8_loo[heldout],
                "v9": metric,
            }
        )
        print(json.dumps({"fold_complete": heldout, "edge": edge_metric, "v9_iou": metric["mean"]["iou_at_0.5"]}), flush=True)

    head, trace = _train_head(compiled, train_scenes=list(FIXED_FIT_SCENES), recipe=recipe, device=device)
    checkpoint = _write_torch_no_clobber(
        output_dir / "all_fit.edge_head.pt",
        _checkpoint(head, train_scenes=list(FIXED_FIT_SCENES), preregistration=preregistration, trace=trace),
    )
    all_scene_metrics = []
    all_edge_metrics = {}
    all_cache_manifests = {}
    for scene in FIXED_FIT_SCENES:
        all_edge_metrics[scene] = _edge_metrics(head, compiled[scene], device=device)
        graph = _gated_graph(
            head, graph_paths[scene], bundle_paths[scene], scene,
            prereg["data_contract"]["canonical_bundle_sha256"][scene], device,
        )
        cache = _materialize_extent_cache(
            manifests[scene],
            gated_graph=graph,
            edge_checkpoint=checkpoint,
            bundle_path=bundle_paths[scene],
            bundle_sha=prereg["data_contract"]["canonical_bundle_sha256"][scene],
            output_dir=output_dir / "extent_cache" / "all_fit",
            device=device,
            preregistration=preregistration,
        )
        cache_path = _write_json_no_clobber(
            output_dir / "extent_cache" / "all_fit" / scene / "manifest.json", cache
        )
        cache["_path"] = str(cache_path)
        cache["_sha256"] = _sha256(cache_path)
        all_cache_manifests[scene] = {"path": str(cache_path), "sha256": _sha256(cache_path)}
        base = _load_v6(v6_paths["all"], key="all", device=device)
        all_scene_metrics.append(_evaluate_scene(base, manifests[scene], cache, device=device))
    all_v9 = _aggregate_scenes(all_scene_metrics)

    def macro_metric(rows, version, key):
        return sum(float(row[version]["mean"][key]) for row in rows) / len(rows)

    loo_v9_iou = macro_metric(folds, "v9", "iou_at_0.5")
    loo_v8_iou = macro_metric(folds, "v8", "iou_at_0.5")
    loo_v2_iou = macro_metric(folds, "v2", "iou_at_0.5")
    loo_v9_precision = macro_metric(folds, "v9", "precision_at_0.5")
    loo_v8_precision = macro_metric(folds, "v8", "precision_at_0.5")
    loo_v9_recall = macro_metric(folds, "v9", "recall_at_0.5")
    loo_v8_recall = macro_metric(folds, "v8", "recall_at_0.5")
    loo_v9_mass = macro_metric(folds, "v9", "absolute_log_probability_mass_ratio_error")
    loo_v8_mass = macro_metric(folds, "v8", "absolute_log_probability_mass_ratio_error")
    v9_mean = all_v9["scene_macro_mean"]
    v8_mean = v8_all["scene_macro_mean"]
    v2_mean = v2_all["scene_macro_mean"]
    gates = {
        "loo_iou_gain_over_max_v8_v2": loo_v9_iou - max(loo_v8_iou, loo_v2_iou),
        "loo_iou_pass": loo_v9_iou > max(loo_v8_iou, loo_v2_iou),
        "loo_precision_gain_over_v8": loo_v9_precision - loo_v8_precision,
        "loo_precision_pass": loo_v9_precision >= loo_v8_precision,
        "loo_recall_gain_over_v8": loo_v9_recall - loo_v8_recall,
        "loo_recall_pass": loo_v9_recall > loo_v8_recall,
        "loo_mass_error_reduction_over_v8": loo_v8_mass - loo_v9_mass,
        "loo_mass_pass": loo_v9_mass < loo_v8_mass,
        "loo_each_scene_click_response_positive_pass": all(row["v9"]["click10_minus_click1_iou"] > 0 for row in folds),
        "all_fit_iou_gain_over_max_v8_v2": v9_mean["iou_at_0.5"] - max(v8_mean["iou_at_0.5"], v2_mean["iou_at_0.5"]),
        "all_fit_iou_pass": v9_mean["iou_at_0.5"] > max(v8_mean["iou_at_0.5"], v2_mean["iou_at_0.5"]),
        "all_fit_precision_gain_over_v8": v9_mean["precision_at_0.5"] - v8_mean["precision_at_0.5"],
        "all_fit_precision_pass": v9_mean["precision_at_0.5"] >= v8_mean["precision_at_0.5"],
        "all_fit_recall_gain_over_v8": v9_mean["recall_at_0.5"] - v8_mean["recall_at_0.5"],
        "all_fit_recall_pass": v9_mean["recall_at_0.5"] > v8_mean["recall_at_0.5"],
        "all_fit_mass_error_reduction_over_v8": v8_mean["absolute_log_probability_mass_ratio_error"] - v9_mean["absolute_log_probability_mass_ratio_error"],
        "all_fit_mass_pass": v9_mean["absolute_log_probability_mass_ratio_error"] < v8_mean["absolute_log_probability_mass_ratio_error"],
        "all_fit_click_response": all_v9["scene_macro_click10_minus_click1_iou"],
        "all_fit_click_response_pass": all_v9["scene_macro_click10_minus_click1_iou"] > 0,
    }
    passed = all(value for key, value in gates.items() if key.endswith("_pass"))
    receipt = {
        "schema_version": 9,
        "artifact_type": SOURCE_RECEIPT,
        "status": "source_gate_pass" if passed else "source_gate_failed_stop_instance_edge_carrier",
        "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
        "edge_training_inventory": edge_inventory,
        "leave_one_scene_out": folds,
        "all_fit": {
            "edge_checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "edge_metrics": all_edge_metrics,
            "extent_cache_manifests": all_cache_manifests,
            "v2": v2_all,
            "v8": v8_all,
            "v9": all_v9,
        },
        "gates": gates,
        "development_authorized": passed,
        "safety": {
            "source_fit_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
            "old_solver_invoked_after_extent": False,
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
    parser.add_argument("--v8-source-receipt", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    path, receipt = run(parse_args())
    print(json.dumps({"receipt": str(path), "status": receipt["status"], "gates": receipt["gates"]}, indent=2))


if __name__ == "__main__":
    main()
