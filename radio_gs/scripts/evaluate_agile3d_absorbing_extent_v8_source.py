"""Materialize and source-gate the preregistered AGILE v8 absorbing extent."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_ratio_training_dataset import (
    FIXED_FIT_SCENES,
)
from radio_gs.querying.absorbing_extent_head import finite_absorbing_seed_reach
from radio_gs.scripts.train_agile3d_seed_conditioned_graph_residual_v7 import (
    _base_llr,
    _cumulative_seed_matrices,
    _load_bundle,
    _load_graph,
    _load_v6,
    _scene_map,
)
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


CACHE_ARTIFACT = "agile3d-absorbing-extent-cache-v8"
CHECKPOINT_ARTIFACT = "agile3d-absorbing-extent-structured-final-v8"
RECEIPT_ARTIFACT = "agile3d-absorbing-extent-source-gate-v8"
V7_RESULT_SHA = "6683aad616dca2f28459318c531420587cfc4317daab14682b9bb54ed22742df"


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
        payload = _load_shard(record)
        object_id = int(payload["object_id"])
        cache_path = output_dir / scene_id / f"object_{object_id:04d}.v8.pt"
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (
                cache.get("artifact_type") != CACHE_ARTIFACT
                or cache.get("source_shard", {}).get("sha256") != record["shard"]["sha256"]
                or cache.get("typed_graph", {}).get("sha256") != _sha256(graph_path)
            ):
                raise ValueError("existing v8 extent cache differs from sealed inputs")
        else:
            positive, negative = _cumulative_seed_matrices(
                payload, bundle, device=device
            )
            hard_positive = positive >= 0.20
            hard_negative = (negative >= 0.20) & ~hard_positive
            positive_reach = finite_absorbing_seed_reach(
                graph, hard_positive.float(), steps=12
            )
            negative_reach = finite_absorbing_seed_reach(
                graph, hard_negative.float(), steps=12
            )
            cache = {
                "schema_version": 8,
                "artifact_type": CACHE_ARTIFACT,
                "scene_id": scene_id,
                "object_id": object_id,
                "positive_extent_support": (positive_reach > 0).cpu(),
                "negative_reach": negative_reach.half().cpu(),
                "hard_positive": hard_positive.cpu(),
                "hard_negative": hard_negative.cpu(),
                "source_shard": {
                    "path": record["shard"]["path"],
                    "sha256": record["shard"]["sha256"],
                },
                "typed_graph": {"path": str(graph_path), "sha256": _sha256(graph_path)},
                "primitive_bundle": {"path": str(bundle_path), "sha256": bundle_sha},
                "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
                "operator": {
                    "absorbing_steps": 12,
                    "hard_seed_threshold": 0.20,
                    "unreached_confidence": 0.0,
                    "solver_policy": "bypass_existing_graph_solver_with_anchor_and_extent_contract",
                },
                "safety": {
                    "fit_labels_opened": True,
                    "labels_used_for_extent": False,
                    "development_labels_opened": False,
                    "test_labels_opened": False,
                    "test312_run": False,
                    "point_as_primitive_used": False,
                },
            }
            _write_torch_no_clobber(cache_path, cache)
        records.append(
            {"object_id": object_id, "path": str(cache_path.resolve()), "sha256": _sha256(cache_path)}
        )
    return {
        "schema_version": 1,
        "artifact_type": "agile3d-absorbing-extent-cache-manifest-v8",
        "scene_id": scene_id,
        "object_count": len(records),
        "records": records,
        "typed_graph": {"path": str(graph_path), "sha256": _sha256(graph_path)},
        "primitive_bundle": {"path": str(bundle_path), "sha256": bundle_sha},
        "graph_safety": graph_payload["safety"],
        "safety": {
            "fit_labels_opened": True,
            "labels_used_for_extent": False,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
        },
    }


@torch.inference_mode()
def _evaluate_scene(
    base_head: torch.nn.Module,
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
            raise ValueError("v8 extent cache changed")
        cache = torch.load(cache_record["path"], map_location="cpu", weights_only=True)
        affinity = torch.as_tensor(payload["capability_click_affinity"], device=device)
        support = torch.as_tensor(cache["positive_extent_support"], device=device).bool()
        negative_reach = torch.as_tensor(cache["negative_reach"], device=device).float()
        hard_positive = torch.as_tensor(cache["hard_positive"], device=device).bool()
        hard_negative = torch.as_tensor(cache["hard_negative"], device=device).bool()
        point_target = torch.as_tensor(payload["point_target"], device=device).bool()
        all_rows = torch.arange(affinity.shape[0], device=device)
        object_rows = []
        for step_index, step in enumerate(payload["steps"]):
            base_probability = torch.sigmoid(
                _base_llr(base_head, affinity, all_rows, step)
            )
            probability = base_probability * (
                1.0 - negative_reach[:, step_index].clamp(0.0, 1.0)
            )
            # Structured final-probability readout: exact zero is abstention,
            # and the already consumed graph is never invoked a second time.
            probability = torch.where(
                support[:, step_index], probability, torch.zeros_like(probability)
            )
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
                {"scene_id": str(payload["scene_id"]), "object_id": int(payload["object_id"]), "click_count": click_count}
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
    *,
    base_checkpoint: Path,
    train_scenes: list[str],
    cache_manifests: Mapping[str, Mapping[str, object]],
    preregistration: Path,
) -> dict[str, object]:
    return {
        "schema_version": 8,
        "artifact_type": CHECKPOINT_ARTIFACT,
        "head_class": "AbsorbingExtentInteractionHead",
        "head_schema_version": "absorbing-seed-extent-structured-final-v8",
        "trainable_parameter_count": 0,
        "absorbing_steps": 12,
        "hard_seed_threshold": 0.20,
        "solver_policy": "bypass_existing_graph_solver_with_anchor_and_extent_contract",
        "base_v6_checkpoint": {"path": str(base_checkpoint), "sha256": _sha256(base_checkpoint)},
        "base_train_scene_ids": train_scenes,
        "cache_manifests": {
            scene: {"path": cache_manifests[scene]["_path"], "sha256": cache_manifests[scene]["_sha256"]}
            for scene in FIXED_FIT_SCENES
        },
        "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
        "safety": {
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    preregistration = Path(args.preregistration).resolve()
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    v7_result = prereg["design_provenance"]["v7_authority"]
    if _sha256(v7_result["path"]) != V7_RESULT_SHA:
        raise ValueError("v8 design authority differs")
    manifests = _load_manifests(args.dataset_manifest)
    graph_paths = _scene_map(args.typed_graph, required=set(FIXED_FIT_SCENES))
    bundle_paths = _scene_map(args.primitive_bundle, required=set(FIXED_FIT_SCENES))
    v6_paths = _scene_map(args.v6_checkpoint, required=set(FIXED_FIT_SCENES) | {"all"})
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifests = {}
    for scene_id in FIXED_FIT_SCENES:
        if (
            manifests[scene_id]["_sha256"]
            != prereg["data_contract"]["dataset_manifest_sha256"][scene_id]
            or _sha256(graph_paths[scene_id])
            != prereg["data_contract"]["typed_reliability_graph_sha256"][scene_id]
        ):
            raise ValueError("v8 scene authority differs from preregistration")
        cache = _materialize_scene_cache(
            manifests[scene_id],
            graph_path=graph_paths[scene_id],
            bundle_path=bundle_paths[scene_id],
            bundle_sha=prereg["data_contract"]["canonical_gaussian_bundle_sha256"][scene_id],
            output_dir=output_dir / "extent_cache",
            device=device,
            preregistration=preregistration,
        )
        cache_path = _write_json_no_clobber(
            output_dir / "extent_cache" / scene_id / "manifest.json", cache
        )
        cache["_path"] = str(cache_path)
        cache["_sha256"] = _sha256(cache_path)
        cache_manifests[scene_id] = cache
        print(json.dumps({"cache_complete": scene_id, "object_count": cache["object_count"]}), flush=True)

    v6_receipt_path = Path(args.v6_source_receipt).resolve()
    if _sha256(v6_receipt_path) != "a2c3ea6845eafcd4674d00415c005ef73213017c593b2bc4dd398d0edf3b5a02":
        raise ValueError("frozen v6 source authority differs")
    v6_receipt = json.loads(v6_receipt_path.read_text(encoding="utf-8"))
    v2_loo = {row["heldout_scene"]: row["v2"] for row in v6_receipt["leave_one_scene_out"]}
    v2_all = v6_receipt["all_fit"]["v2"]

    folds = []
    for heldout in FIXED_FIT_SCENES:
        train_scenes = [scene for scene in FIXED_FIT_SCENES if scene != heldout]
        base = _load_v6(v6_paths[heldout], key=heldout, device=device)
        metric = _evaluate_scene(base, manifests[heldout], cache_manifests[heldout], device=device)
        checkpoint = _write_torch_no_clobber(
            output_dir / f"loo_holdout_{heldout}.pt",
            _checkpoint(
                base_checkpoint=v6_paths[heldout],
                train_scenes=train_scenes,
                cache_manifests=cache_manifests,
                preregistration=preregistration,
            ),
        )
        reference = v2_loo[heldout]
        folds.append(
            {
                "heldout_scene": heldout,
                "base_train_scenes": train_scenes,
                "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
                "v2": reference,
                "v8": metric,
                "iou_gain": metric["mean"]["iou_at_0.5"] - reference["mean"]["iou_at_0.5"],
                "precision_gain": metric["mean"]["precision_at_0.5"] - reference["mean"]["precision_at_0.5"],
                "absolute_log_mass_error_gain": reference["mean"]["absolute_log_probability_mass_ratio_error"] - metric["mean"]["absolute_log_probability_mass_ratio_error"],
            }
        )
        print(json.dumps({"fold_complete": heldout, "iou_gain": folds[-1]["iou_gain"]}), flush=True)

    base = _load_v6(v6_paths["all"], key="all", device=device)
    all_v8 = _aggregate_scenes(
        [_evaluate_scene(base, manifests[scene], cache_manifests[scene], device=device) for scene in FIXED_FIT_SCENES]
    )
    checkpoint = _write_torch_no_clobber(
        output_dir / "all_fit_scene0000_0002_0005.pt",
        _checkpoint(
            base_checkpoint=v6_paths["all"],
            train_scenes=list(FIXED_FIT_SCENES),
            cache_manifests=cache_manifests,
            preregistration=preregistration,
        ),
    )
    loo_iou_gain = sum(row["iou_gain"] for row in folds) / 3
    loo_precision_gain = sum(row["precision_gain"] for row in folds) / 3
    loo_mass_gain = sum(row["absolute_log_mass_error_gain"] for row in folds) / 3
    all_iou_gain = all_v8["scene_macro_mean"]["iou_at_0.5"] - v2_all["scene_macro_mean"]["iou_at_0.5"]
    all_precision_gain = all_v8["scene_macro_mean"]["precision_at_0.5"] - v2_all["scene_macro_mean"]["precision_at_0.5"]
    all_mass_gain = v2_all["scene_macro_mean"]["absolute_log_probability_mass_ratio_error"] - all_v8["scene_macro_mean"]["absolute_log_probability_mass_ratio_error"]
    gates = {
        "loo_macro_iou_gain": loo_iou_gain,
        "loo_macro_iou_pass": loo_iou_gain > 0,
        "loo_macro_precision_gain": loo_precision_gain,
        "loo_macro_precision_pass": loo_precision_gain > 0,
        "loo_macro_absolute_log_mass_error_gain": loo_mass_gain,
        "loo_macro_mass_pass": loo_mass_gain > 0,
        "loo_each_scene_click_response_positive_pass": all(row["v8"]["click10_minus_click1_iou"] > 0 for row in folds),
        "all_fit_iou_gain": all_iou_gain,
        "all_fit_iou_pass": all_iou_gain > 0,
        "all_fit_precision_gain": all_precision_gain,
        "all_fit_precision_pass": all_precision_gain > 0,
        "all_fit_absolute_log_mass_error_gain": all_mass_gain,
        "all_fit_mass_pass": all_mass_gain > 0,
        "all_fit_click_response": all_v8["scene_macro_click10_minus_click1_iou"],
        "all_fit_click_response_pass": all_v8["scene_macro_click10_minus_click1_iou"] > 0,
    }
    passed = all(value for key, value in gates.items() if key.endswith("_pass"))
    receipt = {
        "schema_version": 8,
        "artifact_type": RECEIPT_ARTIFACT,
        "status": "source_gate_pass" if passed else "source_gate_failed_stop_canonical_graph_family",
        "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
        "leave_one_scene_out": folds,
        "all_fit": {
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "v2": v2_all,
            "v8": all_v8,
        },
        "gates": gates,
        "development_authorized": passed,
        "safety": {
            "fit_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
            "old_graph_solver_invoked_after_extent": False,
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
