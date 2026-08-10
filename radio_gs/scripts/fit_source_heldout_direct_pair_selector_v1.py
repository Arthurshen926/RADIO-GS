#!/usr/bin/env python3
"""Fit and cross-scene audit the source-heldout direct-pair selector."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.source_heldout_direct_pair_selector import (
    DIRECT_FEATURE_NAMES,
    SELECTED_INDICES,
    SELECTOR_FEATURE_NAMES,
    DirectPairMonotoneRanker,
    direct_pair_ranker_probability,
    fit_direct_pair_monotone_ranker,
)
from radio_gs.querying.source_heldout_missing_support import FEATURE_NAMES
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


AUTHORITY_SCHEMA = "radio_gs.source_heldout_direct_pair_selector_authority.v1"
MODEL_SCHEMA = "radio_gs.source_heldout_direct_pair_selector.v1"
REPORT_SCHEMA = "radio_gs.source_heldout_direct_pair_selector_report.v1"
TRAIN_SCENES = ("scene0001_00", "scene0002_00", "scene0003_00")
VALIDATION_SCENES = ("scene0004_00",)


def source_access() -> dict[str, bool]:
    return {
        "source_train_heldout_tables_opened": True,
        "source_validation_heldout_tables_opened": True,
        "source_native_v3_pair_features_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_metrics_opened": False,
        "target_metrics_computed": False,
        "gpu_used": False,
    }


def fixed_fit() -> dict[str, Any]:
    return {
        "model": "direct_pair_monotone_additive_logistic",
        "direct_feature_names": list(DIRECT_FEATURE_NAMES),
        "selector_feature_names": list(SELECTOR_FEATURE_NAMES),
        "selected_indices": list(SELECTED_INDICES),
        "normalization": "training_median_and_1p4826_MAD",
        "loss": "equal_scene_equal_query_target_group_BCE_plus_L2",
        "l2_strength": 0.01,
        "maximum_LBFGS_iterations": 100,
        "positive_weight_parameterization": "softplus_nonnegative",
        "source_train_scenes": list(TRAIN_SCENES),
        "source_validation_scenes": list(VALIDATION_SCENES),
        "cross_scene_audit": "leave_one_source_train_scene_out",
        "promotion_primary": "query_macro_and_query_p05_ranking_not_global_prevalence_ranking",
        "target_threshold_selected": False,
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} record differs")
    path = validate_file_record(value, label=label)
    return {"path": str(path), "sha256": str(value["sha256"])}


def validate_execution_authority(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="source-heldout direct-pair selector authority",
    )
    required = {
        "schema", "schema_version", "status", "producer", "selector_interface",
        "source_train", "source_validation", "fixed_fit", "outputs",
        "source_access", "benchmark_execution_authorized",
    }
    authority = dict(raw)
    if (
        set(authority) != required
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_source_only_direct_pair_cross_scene_audit"
        or authority.get("fixed_fit")
        != {"sha256": canonical_json_sha256(fixed_fit())}
        or authority.get("source_access") != source_access()
        or authority.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("source-heldout direct-pair selector authority differs")
    for name in ("producer", "selector_interface"):
        authority[name] = _record(authority[name], label=name)
    if Path(authority["producer"]["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("source-heldout selector producer path differs")
    expected = (("source_train", TRAIN_SCENES), ("source_validation", VALIDATION_SCENES))
    for split, scenes in expected:
        rows = authority[split]
        if not isinstance(rows, list) or len(rows) != len(scenes):
            raise ValueError(f"source-heldout selector {split} differs")
        normalized = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {"scene_id", "table", "native_v3_source"}:
                raise ValueError(f"source-heldout selector {split} row differs")
            if row.get("scene_id") != scenes[index]:
                raise ValueError(f"source-heldout selector {split} scene differs")
            normalized.append({
                "scene_id": scenes[index],
                "table": _record(row["table"], label=f"{split} table"),
                "native_v3_source": _record(row["native_v3_source"], label=f"{split} native"),
            })
        authority[split] = normalized
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"model", "report"}:
        raise ValueError("source-heldout selector outputs differ")
    authority["outputs"] = {
        name: str(Path(value).expanduser().resolve()) for name, value in outputs.items()
    }
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    return authority


def _load_scene(row: Mapping[str, Any]) -> dict[str, torch.Tensor | str]:
    table, _, _ = load_torch_mapping(
        row["table"]["path"], expected_sha256=row["table"]["sha256"],
        map_location="cpu", label="source-heldout direct table",
    )
    native, _, _ = load_torch_mapping(
        row["native_v3_source"]["path"],
        expected_sha256=row["native_v3_source"]["sha256"],
        map_location="cpu", label="source-heldout direct native",
    )
    scene_id = str(row["scene_id"])
    base = torch.as_tensor(table["features"]).float().cpu()
    labels = torch.as_tensor(table["hard_labels"]).bool().cpu()
    queries = torch.as_tensor(table["query_indices"]).long().cpu()
    targets = torch.as_tensor(table["target_region_indices"]).long().cpu()
    seeds = torch.as_tensor(table["seed_region_indices"]).long().cpu()
    regions = int(torch.as_tensor(native["region_rows"]).shape[0])
    pairs = torch.as_tensor(native["pair_indices"]).long().cpu()
    pair_features = torch.as_tensor(native["pair_features"]).float().cpu()
    keys = pairs[0] * regions + pairs[1]
    proposal_keys = torch.minimum(seeds, targets) * regions + torch.maximum(seeds, targets)
    indices = torch.searchsorted(keys, proposal_keys)
    safe = indices.clamp_max(keys.numel() - 1)
    direct = (indices < keys.numel()) & (keys[safe] == proposal_keys)
    if (
        table.get("scene_id") != scene_id
        or native.get("scene_id") != scene_id
        or base.shape != (labels.numel(), len(FEATURE_NAMES))
        or queries.shape != labels.shape
        or targets.shape != labels.shape
        or seeds.shape != labels.shape
        or not bool(direct.any())
    ):
        raise ValueError("source-heldout direct scene axes differ")
    direct_features = torch.cat((base[direct], pair_features[indices[direct]]), dim=1)
    return {
        "scene_id": scene_id,
        "features": direct_features.contiguous(),
        "labels": labels[direct].contiguous(),
        "queries": queries[direct].contiguous(),
        "targets": targets[direct].contiguous(),
        "direct_fraction": direct.float().mean().reshape(()),
    }


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    value = torch.as_tensor(scores).double().cpu().reshape(-1)
    truth = torch.as_tensor(labels).bool().cpu().reshape(-1)
    if int(truth.sum()) <= 0 or int((~truth).sum()) <= 0:
        return float("nan")
    return float(
        (
            (value[truth, None] > value[~truth][None, :]).double()
            + 0.5 * (value[truth, None] == value[~truth][None, :]).double()
        ).mean()
    )


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    value = torch.as_tensor(scores).double().cpu().reshape(-1)
    truth = torch.as_tensor(labels).bool().cpu().reshape(-1)
    positives = int(truth.sum())
    if positives <= 0:
        return float("nan")
    order = torch.argsort(value, descending=True, stable=True)
    sorted_value = value[order]
    ranked = truth[order].double()
    _, counts = torch.unique_consecutive(sorted_value, return_counts=True)
    stops = counts.cumsum(0)
    cumulative = ranked.cumsum(0)[stops - 1]
    precision = cumulative / stops
    recall = cumulative / positives
    previous = torch.cat((torch.zeros(1, dtype=recall.dtype), recall[:-1]))
    return float(((recall - previous) * precision).sum())


def _calibration(probability: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    p = torch.as_tensor(probability).double().cpu().reshape(-1).clamp(1e-7, 1.0 - 1e-7)
    y = torch.as_tensor(labels).bool().cpu().reshape(-1)
    brier = float((p - y.double()).square().mean())
    log_loss = float(-(y.double() * p.log() + (~y).double() * (1.0 - p).log()).mean())
    ece = 0.0
    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        mask = (p >= lower) & (p < upper if index < 9 else p <= upper)
        if bool(mask.any()):
            ece += float(mask.float().mean()) * abs(
                float(p[mask].mean()) - float(y[mask].float().mean())
            )
    return {"brier": brier, "log_loss": log_loss, "ece10": ece}


def _query_metrics(scores: torch.Tensor, labels: torch.Tensor, queries: torch.Tensor) -> dict[str, Any]:
    aucs, aps = [], []
    for query in torch.unique(queries, sorted=True).tolist():
        local = queries == int(query)
        if bool(labels[local].any()) and bool((~labels[local]).any()):
            aucs.append(_auc(scores[local], labels[local]))
            aps.append(_average_precision(scores[local], labels[local]))
    if not aucs:
        return {"evaluable_queries": 0, "auroc_macro": 0.0, "auroc_p05": 0.0, "average_precision_macro": 0.0, "average_precision_p05": 0.0}
    auc_tensor = torch.tensor(aucs)
    ap_tensor = torch.tensor(aps)
    return {
        "evaluable_queries": len(aucs),
        "auroc_macro": float(auc_tensor.mean()),
        "auroc_p05": float(torch.quantile(auc_tensor, 0.05)),
        "average_precision_macro": float(ap_tensor.mean()),
        "average_precision_p05": float(torch.quantile(ap_tensor, 0.05)),
    }


def _evaluation(probability: torch.Tensor, scene: Mapping[str, Any], constant: float) -> dict[str, Any]:
    labels = scene["labels"]
    edge = scene["features"][:, 0]
    constant_probability = torch.full_like(probability, float(constant))
    return {
        "units": int(labels.numel()),
        "positives": int(labels.sum()),
        "positive_fraction": float(labels.float().mean()),
        "selector": {
            "auroc": _auc(probability, labels),
            "average_precision": _average_precision(probability, labels),
            "query": _query_metrics(probability, labels, scene["queries"]),
            "calibration": _calibration(probability, labels),
        },
        "edge_probability_baseline": {
            "auroc": _auc(edge, labels),
            "average_precision": _average_precision(edge, labels),
            "query": _query_metrics(edge, labels, scene["queries"]),
            "calibration": _calibration(edge, labels),
        },
        "training_prevalence_constant": {
            "probability": float(constant),
            "calibration": _calibration(constant_probability, labels),
        },
    }


def _model_payload(model: DirectPairMonotoneRanker) -> dict[str, torch.Tensor]:
    return {"location": model.location, "scale": model.scale, "positive_weights": model.positive_weights, "bias": model.bias}


def _fit(scenes: list[Mapping[str, Any]]) -> tuple[DirectPairMonotoneRanker, float]:
    features = torch.cat([scene["features"] for scene in scenes])
    labels = torch.cat([scene["labels"] for scene in scenes])
    queries = torch.cat([scene["queries"] for scene in scenes])
    targets = torch.cat([scene["targets"] for scene in scenes])
    scene_axis = torch.cat([
        torch.full((scene["labels"].numel(),), index, dtype=torch.long)
        for index, scene in enumerate(scenes)
    ])
    return (
        fit_direct_pair_monotone_ranker(
            features, labels, scene_axis, queries, targets,
            l2_strength=fixed_fit()["l2_strength"],
            maximum_iterations=fixed_fit()["maximum_LBFGS_iterations"],
        ),
        float(labels.float().mean()),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    model_path = Path(authority["outputs"]["model"])
    report_path = Path(authority["outputs"]["report"])
    if model_path.exists() or model_path.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("source-heldout direct selector output exists")
    train = [_load_scene(row) for row in authority["source_train"]]
    validation = [_load_scene(row) for row in authority["source_validation"]]
    loo = []
    for heldout in range(len(train)):
        fitting = [scene for index, scene in enumerate(train) if index != heldout]
        model, prevalence = _fit(fitting)
        held = train[heldout]
        probability = direct_pair_ranker_probability(model, held["features"])
        loo.append({
            "scene_id": held["scene_id"],
            "training_scenes": [scene["scene_id"] for scene in fitting],
            "evaluation": _evaluation(probability, held, prevalence),
            "positive_weights": model.positive_weights.tolist(),
        })
    final_model, prevalence = _fit(train)
    validation_rows = []
    for scene in validation:
        probability = direct_pair_ranker_probability(final_model, scene["features"])
        validation_rows.append({
            "scene_id": scene["scene_id"],
            "evaluation": _evaluation(probability, scene, prevalence),
        })
    loo_selector_auc = sum(row["evaluation"]["selector"]["query"]["auroc_macro"] for row in loo) / len(loo)
    loo_edge_auc = sum(row["evaluation"]["edge_probability_baseline"]["query"]["auroc_macro"] for row in loo) / len(loo)
    loo_selector_ap = sum(row["evaluation"]["selector"]["query"]["average_precision_macro"] for row in loo) / len(loo)
    loo_edge_ap = sum(row["evaluation"]["edge_probability_baseline"]["query"]["average_precision_macro"] for row in loo) / len(loo)
    validation_gate = []
    for row in validation_rows:
        selected = row["evaluation"]["selector"]
        edge = row["evaluation"]["edge_probability_baseline"]
        constant = row["evaluation"]["training_prevalence_constant"]
        validation_gate.append({
            "scene_id": row["scene_id"],
            "query_macro_auroc_strictly_improves_edge": selected["query"]["auroc_macro"] > edge["query"]["auroc_macro"],
            "query_macro_average_precision_strictly_improves_edge": selected["query"]["average_precision_macro"] > edge["query"]["average_precision_macro"],
            "query_p05_auroc_non_regression": selected["query"]["auroc_p05"] >= edge["query"]["auroc_p05"],
            "query_p05_average_precision_non_regression": selected["query"]["average_precision_p05"] >= edge["query"]["average_precision_p05"],
            "brier_improves_training_prevalence_constant": selected["calibration"]["brier"] < constant["calibration"]["brier"],
            "log_loss_improves_training_prevalence_constant": selected["calibration"]["log_loss"] < constant["calibration"]["log_loss"],
        })
    gate = {
        "LOO_query_macro_AUROC_strictly_improves_edge": loo_selector_auc > loo_edge_auc,
        "LOO_query_macro_AP_strictly_improves_edge": loo_selector_ap > loo_edge_ap,
        "every_validation_scene_passes_query_tail_and_calibration": all(all(value for key, value in row.items() if key != "scene_id") for row in validation_gate),
    }
    gate["passed"] = all(gate.values())
    model_payload = {
        "schema": MODEL_SCHEMA,
        "schema_version": 1,
        "feature_names": list(DIRECT_FEATURE_NAMES),
        "selector_feature_names": list(SELECTOR_FEATURE_NAMES),
        "selected_indices": list(SELECTED_INDICES),
        "model": _model_payload(final_model),
        "training_prevalence": prevalence,
        "execution_authority": authority["verified_record"],
        "channel_sha256": {},
    }
    model_payload["channel_sha256"] = {
        name: tensor_sha256(value)
        for name, value in model_payload["model"].items()
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(model_path, model_payload)
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "status": "source_heldout_direct_pair_selector_gate_passed" if gate["passed"] else "source_heldout_direct_pair_selector_gate_failed",
        "execution_authority": authority["verified_record"],
        "model": file_record(model_path),
        "fixed_fit": fixed_fit(),
        "direct_pair_retention": {
            scene["scene_id"]: float(scene["direct_fraction"])
            for scene in train + validation
        },
        "cross_scene_LOO": loo,
        "cross_scene_LOO_macro": {
            "selector_query_AUROC": loo_selector_auc,
            "edge_query_AUROC": loo_edge_auc,
            "selector_query_AP": loo_selector_ap,
            "edge_query_AP": loo_edge_ap,
        },
        "source_validation": validation_rows,
        "source_validation_gate": validation_gate,
        "promotion_gate": gate,
        "label_leakage_audit": {
            "selector_fit_uses_only_source_train_heldout_labels": True,
            "source_validation_labels_used_for_checkpoint_or_weight_fit": False,
            "target_labels_or_metrics_opened": False,
            "native_v3_pair_probability_has_prior_aggregate_source_supervision": True,
            "consequence": "promotion_requires_query_grouped_cross_scene_evidence_and_cannot_rely_on_global_prevalence ranking",
        },
        "source_access": source_access(),
        "benchmark_execution_authorized": False,
    }
    report["content_authority_sha256"] = canonical_json_sha256(report)
    write_frozen_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
