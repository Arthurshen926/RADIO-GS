#!/usr/bin/env python3
"""Audit scene0001-fit to scene0002-heldout RegionCoMembership learnability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from radio_gs.scripts.train_source_region_comembership_v1 import (
    LEARNING_RATE,
    SEED,
    THRESHOLDS,
    WEIGHT_DECAY,
    SceneAuthority,
    _weighted_f1,
    balanced_scene_loss,
    component_scene_metrics,
    load_scene_authority,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    write_frozen_json,
)


PREREGISTRATION = Path(
    "paper/artifacts/source_only_region_comembership_v1_cross_scene_learnability_audit_preregistration_20260807.json"
)
EPOCHS = 100


def _fit_probability(
    train: SceneAuthority,
    heldout: SceneAuthority,
    feature_columns: Sequence[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    columns = torch.as_tensor(feature_columns).long()
    active = train.evidence_weights > 0
    fit = train.pair_features[active][:, columns]
    median = fit.median(dim=0).values
    mad = (fit - median).abs().median(dim=0).values
    scale = torch.where(mad > 0, mad * 1.4826, torch.ones_like(mad))
    model = torch.nn.Linear(int(columns.numel()), 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    torch.manual_seed(SEED)
    history = []
    train_values = (train.pair_features[:, columns] - median) / scale
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_values).squeeze(-1)
        loss = balanced_scene_loss(logits, train.targets, train.evidence_weights)
        loss.backward()
        optimizer.step()
        if epoch in {1, 25, 50, 75, 100}:
            history.append({"epoch": epoch, "fit_balanced_bce": float(loss.detach())})
    heldout_values = (heldout.pair_features[:, columns] - median) / scale
    with torch.no_grad():
        probability = torch.sigmoid(model(heldout_values).squeeze(-1)).contiguous()
    return probability, {
        "feature_columns": columns.tolist(),
        "fit_active_pairs": int(active.sum()),
        "median": median.tolist(),
        "robust_scale": scale.tolist(),
        "history": history,
    }


def _audit_probability(
    name: str, probability: torch.Tensor, heldout: SceneAuthority
) -> dict[str, Any]:
    active = heldout.evidence_weights > 0
    target = heldout.targets[active].numpy()
    score = probability[active].numpy()
    weight = heldout.evidence_weights[active].numpy()
    curve = []
    for threshold in THRESHOLDS:
        component = component_scene_metrics(
            pair_indices=heldout.pair_indices,
            pair_probabilities=probability,
            dominant_instance_ids=heldout.dominant_instance_ids,
            instance_purity=heldout.instance_purity,
            instance_label_coverage=heldout.instance_label_coverage,
            instance_observed=heldout.instance_observed,
            threshold=threshold,
        )
        curve.append(
            {
                "threshold": threshold,
                "edge_weighted_f1": _weighted_f1(
                    probability,
                    heldout.targets,
                    heldout.evidence_weights,
                    threshold,
                ),
                "component_instance_macro": component["instance_macro"],
                "largest_component_scene_evidence_fraction": component[
                    "largest_predicted_component_scene_evidence_fraction"
                ],
            }
        )
    best = max(
        curve,
        key=lambda row: (
            row["component_instance_macro"]["topology_score"],
            row["component_instance_macro"]["iou"],
            -row["threshold"],
        ),
    )
    return {
        "name": name,
        "heldout_edge_evidence_weighted_roc_auc": float(
            roc_auc_score(target, score, sample_weight=weight)
        ),
        "heldout_edge_evidence_weighted_average_precision": float(
            average_precision_score(target, score, sample_weight=weight)
        ),
        "descriptive_heldout_best_topology": best,
        "threshold_curve": curve,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"cross-scene audit already exists: {output}")
    train_record = {
        "path": str(Path(args.scene0001_authority).resolve()),
        "sha256": str(args.expected_scene0001_authority_sha256),
    }
    heldout_record = {
        "path": str(Path(args.scene0002_authority).resolve()),
        "sha256": str(args.expected_scene0002_authority_sha256),
    }
    train = load_scene_authority(
        train_record, expected_scene_id="scene0001_00", expected_split="source_train"
    )
    heldout = load_scene_authority(
        heldout_record,
        expected_scene_id="scene0002_00",
        expected_split="source_train",
    )
    full_probability, full_fit = _fit_probability(train, heldout, tuple(range(15)))
    descriptor_probability, descriptor_fit = _fit_probability(train, heldout, (0,))
    epoch_zero_probability = torch.full_like(full_probability, 0.5)
    report = {
        "schema": "radio_gs.region_comembership_cross_scene_learnability_audit.v1",
        "schema_version": 1,
        "status": "scene0001_fit_scene0002_heldout_diagnostic_complete",
        "preregistration": file_record(
            Path(__file__).resolve().parents[2] / PREREGISTRATION
        ),
        "producer": file_record(Path(__file__).resolve()),
        "fit_authority": train_record,
        "heldout_authority": heldout_record,
        "fit_contract": {
            "epochs": EPOCHS,
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "heldout_contribution_to_fit_or_stopping": False,
            "role": "diagnostic_only",
        },
        "models": {
            "full_linear": {
                "fit": full_fit,
                "audit": _audit_probability("full_linear", full_probability, heldout),
            },
            "descriptor_only": {
                "fit": descriptor_fit,
                "audit": _audit_probability(
                    "descriptor_only", descriptor_probability, heldout
                ),
            },
            "epoch_zero": {
                "fit": None,
                "audit": _audit_probability(
                    "epoch_zero", epoch_zero_probability, heldout
                ),
            },
        },
        "benchmark_opened": False,
        "target_metric_computed": False,
        "used_for_4plus2_selection": False,
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
    summary = {
        name: {
            "auc": row["audit"]["heldout_edge_evidence_weighted_roc_auc"],
            "ap": row["audit"]["heldout_edge_evidence_weighted_average_precision"],
            "best_topology": row["audit"]["descriptive_heldout_best_topology"],
        }
        for name, row in result["models"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
