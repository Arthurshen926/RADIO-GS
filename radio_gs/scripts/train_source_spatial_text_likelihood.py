#!/usr/bin/env python3
"""Fit the scene-shared bounded spatial likelihood residual on source scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.querying.source_spatial_text_likelihood import (
    BoundedSourceSpatialLikelihoodHead,
    SOURCE_SPATIAL_CHECKPOINT_SCHEMA,
    SourceSpatialLikelihoodInputs,
    sha256_file,
    state_dict_sha256,
    validate_source_spatial_shard,
)
from radio_gs.querying.source_text_query_likelihood import (
    confidence_weighted_balanced_bce,
)


RECIPE = {
    "recipe_id": "source-spatial-bounded-residual-adam-seed17-e128-lr0.02-v1",
    "seed": 17,
    "optimizer": "Adam",
    "epochs": 128,
    "learning_rate": 0.02,
    "weight_decay": 0.0001,
    "loss_balancing": "equal_mean_of_legacy_normalized_bce_and_local_relation",
    "example_order": "sealed_scene_order_full_batch_no_shuffle",
    "fit_scenes": ["scene0001_00", "scene0002_00", "scene0005_00"],
}
RECEIPT_SCHEMA = "radio_gs.source_spatial_text_likelihood_fit_receipt.v1"


def _write_torch_noclobber(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    torch.save(dict(payload), output)
    return output


def _write_json_noclobber(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    return output


def load_source_spatial_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads(source.read_text())
    if (
        manifest.get("schema")
        != "radio_gs.source_spatial_text_likelihood_dataset_manifest.v1"
        or manifest.get("schema_version") != 1
        or manifest.get("scene_ids") != RECIPE["fit_scenes"]
    ):
        raise ValueError("source spatial dataset manifest differs")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(RECIPE["fit_scenes"]):
        raise ValueError("source spatial manifest records differ")
    payloads = []
    for expected_scene, record in zip(RECIPE["fit_scenes"], records):
        if not isinstance(record, Mapping) or record.get("scene_id") != expected_scene:
            raise ValueError("source spatial manifest scene order differs")
        shard = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
        if sha256_file(shard) != record.get("sha256"):
            raise ValueError("source spatial manifest shard changed")
        payload = validate_source_spatial_shard(
            torch.load(shard, map_location="cpu", weights_only=False)
        )
        if payload["scene_id"] != expected_scene:
            raise ValueError("source spatial shard scene differs")
        payloads.append(payload)
    if len({str(payload["physical_space_id"]) for payload in payloads}) != len(payloads):
        raise ValueError("source spatial fit must be physical-space disjoint")
    return manifest, payloads


def _inputs(payload: Mapping[str, Any]) -> SourceSpatialLikelihoodInputs:
    return SourceSpatialLikelihoodInputs(
        raw_logit=payload["raw_logit"],
        neighbor_mean_logit=payload["neighbor_mean_logit"],
        neighbor_max_logit=payload["neighbor_max_logit"],
        neighbor_contrast_logit=payload["neighbor_contrast_logit"],
        coverage=payload["coverage"],
        reliability=payload["reliability"],
    )


def _training_weight(payload: Mapping[str, Any]) -> torch.Tensor:
    return (
        payload["valid"].float()
        * payload["training_label_weight"].float()
        * payload["coverage"].float()
        * payload["reliability"].float()
    )


def _present_class_indices(payload: Mapping[str, Any]) -> list[int]:
    target = payload["semantic_class_distribution"]
    weight = _training_weight(payload)
    return [
        index
        for index in range(int(target.shape[1]))
        if float((weight * target[:, index]).sum()) > 0
        and float((weight * (1.0 - target[:, index])).sum()) > 0
    ]


def _local_relation_loss(
    probability: torch.Tensor,
    payload: Mapping[str, Any],
) -> torch.Tensor:
    values = probability[:, 0, :]
    target = payload["semantic_class_distribution"].float()
    neighbors = payload["neighbor_indices"][:, 1:]
    weight = _training_weight(payload)
    edge_weight = torch.sqrt(
        (weight[:, None] * weight[neighbors]).clamp_min(0.0)
    )
    pred_relation = values[:, None, :] - values[neighbors]
    target_relation = target[:, None, :] - target[neighbors]
    error = F.smooth_l1_loss(pred_relation, target_relation, reduction="none")
    return (error * edge_weight[:, :, None]).sum() / (
        edge_weight.sum().clamp_min(1.0) * target.shape[1]
    )


def _objective(
    head: BoundedSourceSpatialLikelihoodHead,
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce_losses = []
    legacy_bce_scales = []
    local_losses = []
    legacy_local_scales = []
    for payload in payloads:
        probability, _residual = head(_inputs(payload))
        values = probability[:, 0, :]
        weight = _training_weight(payload)
        target = payload["semantic_class_distribution"]
        for class_index in _present_class_indices(payload):
            loss, _ = confidence_weighted_balanced_bce(
                values[:, class_index], target[:, class_index], weight
            )
            bce_losses.append(loss)
            legacy_loss, _ = confidence_weighted_balanced_bce(
                torch.sigmoid(payload["raw_logit"][:, 0, class_index]),
                target[:, class_index],
                weight,
            )
            legacy_bce_scales.append(legacy_loss.detach())
        local_losses.append(_local_relation_loss(probability, payload))
        legacy_local_scales.append(
            _local_relation_loss(torch.sigmoid(payload["raw_logit"]), payload)
            .detach()
            .clamp_min(1.0e-6)
        )
    bce = torch.stack(bce_losses).mean()
    legacy_bce = torch.stack(legacy_bce_scales).mean().clamp_min(1.0e-6)
    local = torch.stack(local_losses).mean()
    normalized_local = torch.stack(
        [value / scale for value, scale in zip(local_losses, legacy_local_scales)]
    ).mean()
    total = 0.5 * (bce / legacy_bce + normalized_local)
    return total, bce, local


@torch.inference_mode()
def evaluate_source_spatial_objective(
    head: BoundedSourceSpatialLikelihoodHead,
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    local_rows = []
    for payload in payloads:
        learned, residual = head(_inputs(payload))
        legacy = torch.sigmoid(payload["raw_logit"])
        weight = _training_weight(payload)
        target = payload["semantic_class_distribution"]
        if not torch.equal(
            torch.where(residual == 0, learned, legacy)[residual == 0],
            legacy[residual == 0],
        ):
            raise RuntimeError("source spatial zero residual identity changed")
        for class_index in _present_class_indices(payload):
            class_target = target[:, class_index]
            positive = weight * class_target
            negative = weight * (1.0 - class_target)
            for method, probability in (
                ("legacy", legacy[:, 0, class_index]),
                ("learned", learned[:, 0, class_index]),
            ):
                loss, _ = confidence_weighted_balanced_bce(
                    probability, class_target, weight
                )
                pos = (probability * positive).sum() / positive.sum()
                neg = (probability * negative).sum() / negative.sum()
                rows.append(
                    {
                        "scene_id": payload["scene_id"],
                        "class_id": int(payload["class_ids"][class_index]),
                        "method": method,
                        "balanced_bce": float(loss),
                        "positive_probability": float(pos),
                        "negative_probability": float(neg),
                        "positive_minus_negative_probability": float(pos - neg),
                    }
                )
        for method, probability in (("legacy", legacy), ("learned", learned)):
            local_rows.append(
                {
                    "scene_id": payload["scene_id"],
                    "method": method,
                    "local_relation_loss": float(
                        _local_relation_loss(probability, payload)
                    ),
                }
            )

    def macro(method: str, key: str) -> float:
        values = [row[key] for row in rows if row["method"] == method]
        return float(sum(values) / len(values))

    def local_macro(method: str) -> float:
        values = [
            row["local_relation_loss"]
            for row in local_rows
            if row["method"] == method
        ]
        return float(sum(values) / len(values))

    legacy_bce = macro("legacy", "balanced_bce")
    learned_bce = macro("learned", "balanced_bce")
    legacy_gap = macro("legacy", "positive_minus_negative_probability")
    learned_gap = macro("learned", "positive_minus_negative_probability")
    legacy_local = local_macro("legacy")
    learned_local = local_macro("learned")
    return {
        "present_scene_class_count": len(rows) // 2,
        "legacy_macro_balanced_bce": legacy_bce,
        "learned_macro_balanced_bce": learned_bce,
        "balanced_bce_delta": learned_bce - legacy_bce,
        "legacy_macro_positive_minus_negative_probability": legacy_gap,
        "learned_macro_positive_minus_negative_probability": learned_gap,
        "positive_negative_gap_delta": learned_gap - legacy_gap,
        "legacy_macro_local_relation_loss": legacy_local,
        "learned_macro_local_relation_loss": learned_local,
        "local_relation_loss_delta": learned_local - legacy_local,
        "rows": rows,
        "local_rows": local_rows,
    }


def fit_source_spatial_head(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[BoundedSourceSpatialLikelihoodHead, dict[str, Any]]:
    if torch.cuda.is_initialized():
        raise RuntimeError("source spatial likelihood fit must remain CPU-only")
    torch.manual_seed(int(RECIPE["seed"]))
    head = BoundedSourceSpatialLikelihoodHead().cpu()
    head.reset_parameters_deterministic(seed=int(RECIPE["seed"]))
    initial = evaluate_source_spatial_objective(head, payloads)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(RECIPE["learning_rate"]),
        weight_decay=float(RECIPE["weight_decay"]),
    )
    trace = []
    checkpoints = {0, 1, 3, 7, 15, 31, 63, 95, 127}
    for epoch in range(int(RECIPE["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        total, bce, local = _objective(head, payloads)
        total.backward()
        optimizer.step()
        if epoch in checkpoints:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "total": float(total.detach()),
                    "macro_balanced_bce": float(bce.detach()),
                    "macro_local_relation_loss": float(local.detach()),
                }
            )
    head.eval()
    final = evaluate_source_spatial_objective(head, payloads)
    gate = {
        "balanced_bce_improved": final["balanced_bce_delta"] < 0,
        "positive_negative_gap_improved": final["positive_negative_gap_delta"] > 0,
        "local_relation_improved": final["local_relation_loss_delta"] < 0,
    }
    gate["all_passed"] = all(gate.values())
    if not gate["all_passed"]:
        summary = {
            key: final[key]
            for key in (
                "balanced_bce_delta",
                "positive_negative_gap_delta",
                "local_relation_loss_delta",
                "legacy_macro_local_relation_loss",
                "learned_macro_local_relation_loss",
            )
        }
        raise RuntimeError(f"source spatial fit gates failed: {gate}; {summary}")
    return head, {"initial": initial, "final": final, "gate": gate, "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.dataset_manifest).expanduser().resolve(strict=True)
    manifest, payloads = load_source_spatial_manifest(manifest_path)
    head, diagnostics = fit_source_spatial_head(payloads)
    checkpoint_payload = {
        "schema": SOURCE_SPATIAL_CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "head_class": "BoundedSourceSpatialLikelihoodHead",
        "head_schema_version": head.schema_version,
        "state_dict": {
            name: value.detach().cpu() for name, value in head.state_dict().items()
        },
        "state_dict_sha256": state_dict_sha256(head.state_dict()),
        "recipe": dict(RECIPE),
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "source_scene_ids": list(RECIPE["fit_scenes"]),
        "class_ids": list(payloads[0]["class_ids"]),
        "class_names": list(payloads[0]["class_names"]),
        "source_access": dict(manifest["source_access"]),
    }
    checkpoint = _write_torch_noclobber(args.checkpoint, checkpoint_payload)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_train_only_spatial_likelihood_fit",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "state_dict_sha256": checkpoint_payload["state_dict_sha256"],
        "dataset_manifest": checkpoint_payload["dataset_manifest"],
        "diagnostics": diagnostics,
        "source_access": dict(manifest["source_access"]),
        "heldout_development_opened": False,
        "evaluator_integration": {
            "enabled": False,
            "scannet_exact_default_changed": False,
            "lerf_metric_run": False,
            "scannet_metric_run": False
        }
    }
    receipt_path = _write_json_noclobber(args.receipt, receipt)
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
