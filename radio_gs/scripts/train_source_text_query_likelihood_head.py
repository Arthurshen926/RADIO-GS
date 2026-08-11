#!/usr/bin/env python3
"""Fit the shared monotone text-likelihood head on sealed source scenes only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.querying.query_likelihood_head import MonotoneQueryLikelihoodHead
from radio_gs.querying.source_text_query_likelihood import (
    SOURCE_TEXT_CHECKPOINT_SCHEMA,
    confidence_weighted_balanced_bce,
    initialize_source_text_head,
    iter_source_text_likelihood_examples,
    sha256_file,
    source_text_likelihood_contract,
)
from radio_gs.scripts.build_source_text_query_likelihood_dataset import (
    _write_json_noclobber,
    _write_torch_noclobber,
    validate_dataset_manifest,
)


RECIPE = {
    "recipe_id": "source-text-monotone-balanced-adam-seed0-e128-lr0.05-v1",
    "seed": 0,
    "optimizer": "Adam",
    "epochs": 128,
    "learning_rate": 0.05,
    "weight_decay": 0.0,
    "example_order": "sealed_scene_then_class_order_no_shuffle",
    "objective": (
        "scene_query_macro_confidence_weighted_balanced_binary_cross_entropy"
    ),
    "initial_affinity_weight": 0.05,
    "initial_prior_weight": 1.0,
    "probability_clamp": [1e-6, 1.0 - 1e-6],
}
RECEIPT_SCHEMA = "radio_gs.source_text_query_likelihood_fit_receipt.v1"


def _parameter_summary(head: MonotoneQueryLikelihoodHead) -> dict[str, Any]:
    return {
        "bias": float(head.bias.detach()),
        "positive_weights": F.softplus(head.raw_positive_weights.detach()).tolist(),
        "negative_weights": F.softplus(head.raw_negative_weights.detach()).tolist(),
        "prior_weight": float(F.softplus(head.raw_prior_weight.detach())),
    }


def _examples(payloads: Sequence[Mapping[str, Any]]) -> list[Any]:
    values = [
        example
        for payload in payloads
        for example in iter_source_text_likelihood_examples(payload)
    ]
    if not values:
        raise ValueError("sealed source text dataset has no balanced class example")
    return values


@torch.inference_mode()
def evaluate_source_objective(
    head: MonotoneQueryLikelihoodHead,
    examples: Sequence[Any],
) -> dict[str, Any]:
    rows = []
    for example in examples:
        evidence = head(example.observations, source="source_text_calibration")
        loss, details = confidence_weighted_balanced_bce(
            evidence.foreground_probability,
            example.target,
            example.training_weight,
        )
        target = example.target.float()
        weight = example.training_weight
        positive = weight * target
        negative = weight * (1.0 - target)
        rows.append(
            {
                "scene_id": example.scene_id,
                "class_id": example.class_id,
                "class_name": example.class_name,
                "balanced_bce": float(loss),
                "positive_probability": float(
                    (evidence.foreground_probability * positive).sum()
                    / positive.sum()
                ),
                "negative_probability": float(
                    (evidence.foreground_probability * negative).sum()
                    / negative.sum()
                ),
                "mean_confidence": float(evidence.confidence.mean()),
                "positive_weight": float(details["positive_weight"]),
                "negative_weight": float(details["negative_weight"]),
            }
        )
    return {
        "example_count": len(rows),
        "scene_count": len({row["scene_id"] for row in rows}),
        "macro_balanced_bce": float(
            sum(row["balanced_bce"] for row in rows) / len(rows)
        ),
        "macro_positive_probability": float(
            sum(row["positive_probability"] for row in rows) / len(rows)
        ),
        "macro_negative_probability": float(
            sum(row["negative_probability"] for row in rows) / len(rows)
        ),
        "macro_positive_minus_negative_probability": float(
            sum(
                row["positive_probability"] - row["negative_probability"]
                for row in rows
            )
            / len(rows)
        ),
        "rows": rows,
    }


def fit_source_text_head(
    payloads: Sequence[Mapping[str, Any]],
    *,
    recipe: Mapping[str, Any] = RECIPE,
) -> tuple[MonotoneQueryLikelihoodHead, dict[str, Any]]:
    if torch.cuda.is_initialized():
        raise RuntimeError("source text calibrator must not initialize CUDA")
    if dict(recipe) != RECIPE:
        raise ValueError("source text calibration recipe differs from frozen v1")
    examples = _examples(payloads)
    torch.manual_seed(int(recipe["seed"]))
    head = MonotoneQueryLikelihoodHead(affinity_channel_count=1).cpu()
    initialize_source_text_head(
        head,
        affinity_weight=float(recipe["initial_affinity_weight"]),
        prior_weight=float(recipe["initial_prior_weight"]),
    )
    initial = evaluate_source_objective(head, examples)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    trace = []
    checkpoints = {0, 1, 2, 3, 7, 15, 31, 63, 95, 127}
    for epoch in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for example in examples:
            evidence = head(example.observations, source="source_text_calibration")
            loss, _details = confidence_weighted_balanced_bce(
                evidence.foreground_probability,
                example.target,
                example.training_weight,
            )
            losses.append(loss)
        objective = torch.stack(losses).mean()
        objective.backward()
        optimizer.step()
        if epoch in checkpoints:
            trace.append({"epoch": epoch + 1, "macro_balanced_bce": float(objective.detach())})
    head.eval()
    final = evaluate_source_objective(head, examples)
    if not final["macro_balanced_bce"] < initial["macro_balanced_bce"]:
        raise RuntimeError("source text calibrator did not improve source balanced BCE")
    trace_sha = hashlib.sha256(
        json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return head, {
        "recipe": dict(recipe),
        "initial": initial,
        "final": final,
        "trace": trace,
        "trace_sha256": trace_sha,
        "parameters": _parameter_summary(head),
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if torch.cuda.is_initialized():
        raise RuntimeError("source text calibrator must start before CUDA initialization")
    manifest_path = Path(args.dataset_manifest).expanduser().resolve(strict=True)
    manifest, payloads = validate_dataset_manifest(manifest_path)
    head, diagnostics = fit_source_text_head(payloads)
    checkpoint_payload = {
        "schema": SOURCE_TEXT_CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "head_class": "MonotoneQueryLikelihoodHead",
        "head_schema_version": head.schema_version,
        "state_dict": {
            key: value.detach().cpu() for key, value in head.state_dict().items()
        },
        "contract": source_text_likelihood_contract(),
        "recipe": dict(RECIPE),
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "source_scene_ids": [record["scene_id"] for record in manifest["records"]],
        "class_ids": list(manifest["class_ids"]),
        "class_names": list(manifest["class_names"]),
        "source_access": dict(manifest["source_access"]),
    }
    checkpoint = _write_torch_noclobber(args.checkpoint, checkpoint_payload)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_train_only_text_calibration",
        "dataset_manifest": checkpoint_payload["dataset_manifest"],
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "head_schema_version": head.schema_version,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in head.parameters()
        ),
        "diagnostics": diagnostics,
        "source_access": dict(manifest["source_access"]),
        "evaluator_integration": {
            "enabled": False,
            "scannet_exact_default_changed": False,
            "lerf_metric_run": False,
            "scannet_metric_run": False,
        },
    }
    receipt_path = _write_json_noclobber(args.receipt, receipt)
    return receipt_path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receipt", required=True)
    path, receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
