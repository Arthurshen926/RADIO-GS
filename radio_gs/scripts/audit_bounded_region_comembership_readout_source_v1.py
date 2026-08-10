#!/usr/bin/env python3
"""Select bounded readout rules on scene0001 and audit them on scene0002."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from radio_gs.querying.bounded_region_comembership_readout import (
    bounded_regions_for_seed,
    bridge_free_component_ids,
    thresholded_adjacency,
)
from radio_gs.scripts.train_source_region_comembership_v1 import (
    LEARNING_RATE,
    SEED,
    THRESHOLDS,
    WEIGHT_DECAY,
    SceneAuthority,
    balanced_scene_loss,
    component_scene_metrics,
    load_scene_authority,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json


PREREGISTRATION = Path(
    "paper/artifacts/source_only_region_comembership_bounded_readout_source_audit_preregistration_20260807.json"
)
METHODS = (
    "maximum_product",
    "widest_path",
    "dual_path_widest",
    "multipoint_consistency",
)
MAXIMUM_REGIONS = (2, 4, 8)
EPOCHS = 100


def _fit_full_linear_probabilities(
    fit_scene: SceneAuthority, heldout_scene: SceneAuthority
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    active = fit_scene.evidence_weights > 0
    fit = fit_scene.pair_features[active]
    median = fit.median(dim=0).values
    mad = (fit - median).abs().median(dim=0).values
    scale = torch.where(mad > 0, mad * 1.4826, torch.ones_like(mad))
    torch.manual_seed(SEED)
    model = torch.nn.Linear(fit.shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    normalized_fit = (fit_scene.pair_features - median) / scale
    history = []
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(normalized_fit).squeeze(-1)
        loss = balanced_scene_loss(
            logits, fit_scene.targets, fit_scene.evidence_weights
        )
        loss.backward()
        optimizer.step()
        if epoch in {1, 25, 50, 75, 100}:
            history.append({"epoch": epoch, "fit_balanced_bce": float(loss.detach())})
    result = {}
    with torch.no_grad():
        for scene in (fit_scene, heldout_scene):
            normalized = (scene.pair_features - median) / scale
            result[scene.scene_id] = torch.sigmoid(
                model(normalized).squeeze(-1)
            ).contiguous()
    return result, {
        "median": median.tolist(),
        "robust_scale": scale.tolist(),
        "history": history,
        "heldout_contribution": False,
    }


def _bounded_seed_metrics(
    *,
    scene: SceneAuthority,
    pair_probability: torch.Tensor,
    threshold: float,
    method: str,
) -> dict[int, dict[str, float]]:
    adjacency = thresholded_adjacency(
        region_count=scene.region_count,
        pair_indices=scene.pair_indices,
        pair_probabilities=pair_probability,
        threshold=threshold,
    )
    components = (
        bridge_free_component_ids(adjacency) if method == "dual_path_widest" else None
    )
    dominant = scene.dominant_instance_ids
    evidence = scene.instance_purity * scene.instance_label_coverage
    eligible = scene.instance_observed & (dominant > 0) & (evidence > 0)
    rows = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    instances = sorted(set(int(dominant[row]) for row in rows))
    target_total = {
        instance: float(evidence[dominant == instance].sum()) for instance in instances
    }
    scene_total = sum(target_total.values())
    sums = {
        maximum: {
            instance: {
                "seed_weight": 0.0,
                "iou": 0.0,
                "f1": 0.0,
                "contamination": 0.0,
                "giant_excess": 0.0,
                "selected_regions": 0.0,
            }
            for instance in instances
        }
        for maximum in MAXIMUM_REGIONS
    }
    for seed in rows:
        ordered = bounded_regions_for_seed(
            method=method,
            seed_region_index=seed,
            adjacency=adjacency,
            maximum_regions=max(MAXIMUM_REGIONS),
            bridge_free_components=components,
        )
        instance = int(dominant[seed])
        seed_weight = float(evidence[seed])
        for maximum in MAXIMUM_REGIONS:
            selected = torch.tensor(ordered[:maximum], dtype=torch.long)
            selected_eligible = selected[eligible[selected]]
            selected_weight = float(evidence[selected_eligible].sum())
            correct_weight = float(
                evidence[
                    selected_eligible[dominant[selected_eligible] == instance]
                ].sum()
            )
            target_weight = target_total[instance]
            union = target_weight + selected_weight - correct_weight
            values = {
                "iou": correct_weight / max(union, 1e-12),
                "f1": 2 * correct_weight / max(target_weight + selected_weight, 1e-12),
                "contamination": (selected_weight - correct_weight)
                / max(selected_weight, 1e-12),
                "giant_excess": max(
                    0.0,
                    selected_weight / scene_total - target_weight / scene_total,
                ),
                "selected_regions": float(len(selected)),
            }
            row = sums[maximum][instance]
            row["seed_weight"] += seed_weight
            for name, value in values.items():
                row[name] += seed_weight * value
    result = {}
    for maximum in MAXIMUM_REGIONS:
        per_instance = {}
        for instance, values in sums[maximum].items():
            denominator = max(values["seed_weight"], 1e-12)
            per_instance[instance] = {
                name: values[name] / denominator
                for name in (
                    "iou",
                    "f1",
                    "contamination",
                    "giant_excess",
                    "selected_regions",
                )
            }
        macro = {
            name: sum(row[name] for row in per_instance.values()) / len(per_instance)
            for name in (
                "iou",
                "f1",
                "contamination",
                "giant_excess",
                "selected_regions",
            )
        }
        macro["topology_score"] = (
            macro["iou"] - macro["contamination"] - macro["giant_excess"]
        )
        result[maximum] = macro
    return result


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    metric = row["metrics"]
    return (
        float(metric["topology_score"]),
        float(metric["iou"]),
        float(metric["f1"]),
        -float(metric["contamination"]),
        -float(metric["giant_excess"]),
        -float(row["threshold"]),
    )


def _singleton_metrics(scene: SceneAuthority) -> dict[str, float]:
    probability = torch.zeros(scene.pair_indices.shape[1])
    return _bounded_seed_metrics(
        scene=scene,
        pair_probability=probability,
        threshold=0.5,
        method="widest_path",
    )[2]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"bounded source audit already exists: {output}")
    fit_record = {
        "path": str(Path(args.scene0001_authority).resolve()),
        "sha256": str(args.expected_scene0001_authority_sha256),
    }
    heldout_record = {
        "path": str(Path(args.scene0002_authority).resolve()),
        "sha256": str(args.expected_scene0002_authority_sha256),
    }
    fit_scene = load_scene_authority(
        fit_record, expected_scene_id="scene0001_00", expected_split="source_train"
    )
    heldout_scene = load_scene_authority(
        heldout_record,
        expected_scene_id="scene0002_00",
        expected_split="source_train",
    )
    probabilities, fit_audit = _fit_full_linear_probabilities(fit_scene, heldout_scene)
    selection_curves: dict[str, dict[int, list[dict[str, Any]]]] = {
        method: {maximum: [] for maximum in MAXIMUM_REGIONS} for method in METHODS
    }
    for threshold in THRESHOLDS:
        for method in METHODS:
            metrics = _bounded_seed_metrics(
                scene=fit_scene,
                pair_probability=probabilities[fit_scene.scene_id],
                threshold=threshold,
                method=method,
            )
            for maximum in MAXIMUM_REGIONS:
                selection_curves[method][maximum].append(
                    {"threshold": threshold, "metrics": metrics[maximum]}
                )
    selected = {
        method: {
            maximum: max(selection_curves[method][maximum], key=_selection_key)
            for maximum in MAXIMUM_REGIONS
        }
        for method in METHODS
    }
    heldout = {method: {} for method in METHODS}
    for method in METHODS:
        by_threshold: dict[float, dict[int, dict[str, float]]] = {}
        for maximum in MAXIMUM_REGIONS:
            threshold = float(selected[method][maximum]["threshold"])
            if threshold not in by_threshold:
                by_threshold[threshold] = _bounded_seed_metrics(
                    scene=heldout_scene,
                    pair_probability=probabilities[heldout_scene.scene_id],
                    threshold=threshold,
                    method=method,
                )
            heldout[method][maximum] = {
                "source_selected_threshold": threshold,
                "metrics": by_threshold[threshold][maximum],
            }
    unbounded_curve = []
    for threshold in THRESHOLDS:
        metric = component_scene_metrics(
            pair_indices=fit_scene.pair_indices,
            pair_probabilities=probabilities[fit_scene.scene_id],
            dominant_instance_ids=fit_scene.dominant_instance_ids,
            instance_purity=fit_scene.instance_purity,
            instance_label_coverage=fit_scene.instance_label_coverage,
            instance_observed=fit_scene.instance_observed,
            threshold=threshold,
        )["instance_macro"]
        unbounded_curve.append({"threshold": threshold, "metrics": metric})
    unbounded_selected = max(unbounded_curve, key=_selection_key)
    unbounded_heldout = component_scene_metrics(
        pair_indices=heldout_scene.pair_indices,
        pair_probabilities=probabilities[heldout_scene.scene_id],
        dominant_instance_ids=heldout_scene.dominant_instance_ids,
        instance_purity=heldout_scene.instance_purity,
        instance_label_coverage=heldout_scene.instance_label_coverage,
        instance_observed=heldout_scene.instance_observed,
        threshold=float(unbounded_selected["threshold"]),
    )
    report = {
        "schema": "radio_gs.bounded_region_comembership_source_audit.v1",
        "schema_version": 1,
        "status": "scene0001_selected_scene0002_heldout_bounded_readout_audit_complete",
        "preregistration": file_record(
            Path(__file__).resolve().parents[2] / PREREGISTRATION
        ),
        "producer": file_record(Path(__file__).resolve()),
        "fit_authority": fit_record,
        "heldout_authority": heldout_record,
        "probability_fit": fit_audit,
        "selection_curves_scene0001": selection_curves,
        "selected_on_scene0001": selected,
        "heldout_scene0002": heldout,
        "baselines": {
            "singleton_scene0002": _singleton_metrics(heldout_scene),
            "unbounded_selected_on_scene0001": unbounded_selected,
            "unbounded_heldout_scene0002": unbounded_heldout,
        },
        "metric_limitation": "This engineering audit weights canonical region evidence; a V2 promotion audit should instead deduplicate selected semantic cores and score primitive-instance evidence.",
        "used_for_formal_selection": False,
        "benchmark_opened": False,
        "target_metric_computed": False,
    }
    write_frozen_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene0001-authority", required=True)
    parser.add_argument("--expected-scene0001-authority-sha256", required=True)
    parser.add_argument("--scene0002-authority", required=True)
    parser.add_argument("--expected-scene0002-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    result = run(parser.parse_args())
    print(
        json.dumps(
            {
                "heldout_scene0002": result["heldout_scene0002"],
                "baselines": result["baselines"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
