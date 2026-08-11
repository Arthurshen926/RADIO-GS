#!/usr/bin/env python3
"""Run sealed LOSO gates and fit the typed-MPR source spatial head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.querying.source_spatial_text_likelihood import (
    sha256_file,
    state_dict_sha256,
)
from radio_gs.querying.source_text_query_likelihood import (
    confidence_weighted_balanced_bce,
)
from radio_gs.querying.source_typed_spatial_text_likelihood import (
    BoundedTypedSourceSpatialLikelihoodHead,
    SOURCE_TYPED_SPATIAL_CHECKPOINT_SCHEMA,
    SourceTypedSpatialLikelihoodInputs,
    validate_source_typed_spatial_shard,
)


FIT_SCENES = ["scene0001_00", "scene0002_00", "scene0005_00"]
RECIPE = {
    "recipe_id": "source-typed-mpr-spatial-bounded-residual-adam-seed17-e128-lr0.02-v1",
    "seed": 17,
    "optimizer": "Adam",
    "epochs": 128,
    "learning_rate": 0.02,
    "weight_decay": 0.0001,
    "loss_balancing": "equal_mean_of_legacy_normalized_bce_and_typed_local_relation",
    "example_order": "sealed_scene_order_full_batch_no_shuffle",
    "fit_scenes": FIT_SCENES,
    "loso_folds": [
        {"fit": ["scene0002_00", "scene0005_00"], "heldout": "scene0001_00"},
        {"fit": ["scene0001_00", "scene0005_00"], "heldout": "scene0002_00"},
        {"fit": ["scene0001_00", "scene0002_00"], "heldout": "scene0005_00"},
    ],
}
MANIFEST_SCHEMA = "radio_gs.source_typed_spatial_likelihood_dataset_manifest.v1"
LOSO_SCHEMA = "radio_gs.source_typed_spatial_likelihood_loso_result.v1"
FIT_RECEIPT_SCHEMA = "radio_gs.source_typed_spatial_likelihood_fit_receipt.v1"


def _write_json_noclobber(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    return output


def _write_torch_noclobber(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    torch.save(dict(payload), output)
    return output


def load_typed_source_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads(source.read_text())
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("scene_ids") != FIT_SCENES
    ):
        raise ValueError("typed spatial dataset manifest differs")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(FIT_SCENES):
        raise ValueError("typed spatial manifest records differ")
    payloads = []
    for expected_scene, record in zip(FIT_SCENES, records):
        if not isinstance(record, Mapping) or record.get("scene_id") != expected_scene:
            raise ValueError("typed spatial manifest scene order differs")
        shard = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
        if sha256_file(shard) != record.get("sha256"):
            raise ValueError("typed spatial manifest shard changed")
        payload = validate_source_typed_spatial_shard(
            torch.load(shard, map_location="cpu", weights_only=False)
        )
        if payload["scene_id"] != expected_scene:
            raise ValueError("typed spatial shard scene differs")
        payloads.append(payload)
    if len({str(payload["physical_space_id"]) for payload in payloads}) != len(payloads):
        raise ValueError("typed source fit must be physical-space disjoint")
    if any(payload["edge_types"] != payloads[0]["edge_types"] for payload in payloads):
        raise ValueError("typed source fit edge channels differ")
    return manifest, payloads


def _inputs(payload: Mapping[str, Any]) -> SourceTypedSpatialLikelihoodInputs:
    return SourceTypedSpatialLikelihoodInputs(
        raw_logit=payload["raw_logit"],
        typed_statistics=payload["typed_statistics"],
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


def _typed_local_relation_loss(
    probability: torch.Tensor,
    payload: Mapping[str, Any],
) -> torch.Tensor:
    values = probability[:, 0, :]
    target = payload["semantic_class_distribution"].float()
    row_weight = _training_weight(payload)
    losses = []
    for edge_type in payload["edge_types"]:
        record = payload["typed_region_edges"][edge_type]
        receiver, neighbor = record["edge_index"]
        edge_weight = record["edge_weight"] * torch.sqrt(
            (row_weight[receiver] * row_weight[neighbor]).clamp_min(0.0)
        )
        pred_relation = values[receiver] - values[neighbor]
        target_relation = target[receiver] - target[neighbor]
        error = F.smooth_l1_loss(pred_relation, target_relation, reduction="none")
        losses.append(
            (error * edge_weight[:, None]).sum()
            / (edge_weight.sum().clamp_min(1.0) * target.shape[1])
        )
    return torch.stack(losses).mean()


def _objective(
    head: BoundedTypedSourceSpatialLikelihoodHead,
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce_losses = []
    legacy_bce_scales = []
    local_losses = []
    legacy_local_scales = []
    for payload in payloads:
        learned, _ = head(_inputs(payload))
        legacy = torch.sigmoid(payload["raw_logit"])
        weight = _training_weight(payload)
        target = payload["semantic_class_distribution"]
        for class_index in _present_class_indices(payload):
            loss, _ = confidence_weighted_balanced_bce(
                learned[:, 0, class_index], target[:, class_index], weight
            )
            legacy_loss, _ = confidence_weighted_balanced_bce(
                legacy[:, 0, class_index], target[:, class_index], weight
            )
            bce_losses.append(loss)
            legacy_bce_scales.append(legacy_loss.detach())
        local_losses.append(_typed_local_relation_loss(learned, payload))
        legacy_local_scales.append(
            _typed_local_relation_loss(legacy, payload).detach().clamp_min(1.0e-6)
        )
    bce = torch.stack(bce_losses).mean()
    legacy_bce = torch.stack(legacy_bce_scales).mean().clamp_min(1.0e-6)
    local = torch.stack(local_losses).mean()
    normalized_local = torch.stack(
        [value / scale for value, scale in zip(local_losses, legacy_local_scales)]
    ).mean()
    return 0.5 * (bce / legacy_bce + normalized_local), bce, local


@torch.inference_mode()
def evaluate_typed_source_objective(
    head: BoundedTypedSourceSpatialLikelihoodHead,
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    local_rows = []
    for payload in payloads:
        learned, residual = head(_inputs(payload))
        legacy = torch.sigmoid(payload["raw_logit"])
        if not torch.equal(learned[residual == 0], legacy[residual == 0]):
            raise RuntimeError("typed source zero residual identity changed")
        weight = _training_weight(payload)
        target = payload["semantic_class_distribution"]
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
                pos = (probability * positive).sum() / positive.sum().clamp_min(1e-12)
                neg = (probability * negative).sum() / negative.sum().clamp_min(1e-12)
                rows.append(
                    {
                        "scene_id": payload["scene_id"],
                        "class_id": int(payload["class_ids"][class_index]),
                        "method": method,
                        "balanced_bce": float(loss),
                        "positive_minus_negative_probability": float(pos - neg),
                    }
                )
        for method, probability in (("legacy", legacy), ("learned", learned)):
            local_rows.append(
                {
                    "scene_id": payload["scene_id"],
                    "method": method,
                    "typed_local_relation_loss": float(
                        _typed_local_relation_loss(probability, payload)
                    ),
                }
            )

    def macro(method: str, key: str) -> float:
        values = [row[key] for row in rows if row["method"] == method]
        return float(sum(values) / len(values))

    def local_macro(method: str) -> float:
        values = [
            row["typed_local_relation_loss"]
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
        "legacy_macro_typed_local_relation_loss": legacy_local,
        "learned_macro_typed_local_relation_loss": learned_local,
        "typed_local_relation_loss_delta": learned_local - legacy_local,
        "rows": rows,
        "local_rows": local_rows,
    }


def _fit(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[BoundedTypedSourceSpatialLikelihoodHead, dict[str, Any]]:
    if torch.cuda.is_initialized():
        raise RuntimeError("typed spatial likelihood fit must remain CPU-only")
    torch.manual_seed(int(RECIPE["seed"]))
    head = BoundedTypedSourceSpatialLikelihoodHead(payloads[0]["edge_types"]).cpu()
    head.reset_parameters_deterministic(seed=int(RECIPE["seed"]))
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
                    "macro_typed_local_relation_loss": float(local.detach()),
                }
            )
    head.eval()
    return head, {"trace": trace}


def _gate(metrics: Mapping[str, Any]) -> dict[str, bool]:
    gate = {
        "balanced_bce_improved": float(metrics["balanced_bce_delta"]) < 0,
        "positive_negative_gap_improved": float(metrics["positive_negative_gap_delta"]) > 0,
        "typed_local_relation_improved": float(metrics["typed_local_relation_loss_delta"]) < 0,
    }
    gate["all_passed"] = all(gate.values())
    return gate


def run_loso(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scene = {payload["scene_id"]: payload for payload in payloads}
    folds = []
    for fold in RECIPE["loso_folds"]:
        head, training = _fit([by_scene[scene] for scene in fold["fit"]])
        heldout_metrics = evaluate_typed_source_objective(
            head, [by_scene[fold["heldout"]]]
        )
        folds.append(
            {
                **fold,
                "heldout_metrics": heldout_metrics,
                "gate": _gate(heldout_metrics),
                "training_trace": training["trace"],
                "state_dict_sha256": state_dict_sha256(head.state_dict()),
            }
        )
    macro = {}
    for key in (
        "balanced_bce_delta",
        "positive_negative_gap_delta",
        "typed_local_relation_loss_delta",
    ):
        macro[key] = float(sum(fold["heldout_metrics"][key] for fold in folds) / len(folds))
    macro_gate = _gate(macro)
    all_passed = all(fold["gate"]["all_passed"] for fold in folds) and macro_gate["all_passed"]
    return {"folds": folds, "macro_deltas": macro, "macro_gate": macro_gate, "all_passed": all_passed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loso", "fit3"), required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--result")
    parser.add_argument("--loso-authority")
    parser.add_argument("--checkpoint")
    parser.add_argument("--receipt")
    args = parser.parse_args()
    manifest_path = Path(args.dataset_manifest).expanduser().resolve(strict=True)
    manifest, payloads = load_typed_source_manifest(manifest_path)
    if args.mode == "loso":
        if not args.result or any((args.loso_authority, args.checkpoint, args.receipt)):
            raise ValueError("LOSO mode requires only --result")
        diagnostics = run_loso(payloads)
        result = {
            "schema": LOSO_SCHEMA,
            "schema_version": 1,
            "status": "complete_three_fold_source_only_loso",
            "recipe": dict(RECIPE),
            "dataset_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "diagnostics": diagnostics,
            "development_scene0003_opened": False,
            "lerf_queries_or_ground_truth_opened": False,
            "source_access": dict(manifest["source_access"]),
        }
        output = _write_json_noclobber(args.result, result)
        print(json.dumps({"result": str(output), "sha256": sha256_file(output), "all_passed": diagnostics["all_passed"]}, sort_keys=True))
        return

    if not all((args.loso_authority, args.checkpoint, args.receipt)) or args.result:
        raise ValueError("fit3 mode requires --loso-authority --checkpoint --receipt")
    loso_path = Path(args.loso_authority).expanduser().resolve(strict=True)
    loso = json.loads(loso_path.read_text())
    if (
        loso.get("schema") != LOSO_SCHEMA
        or loso.get("dataset_manifest", {}).get("sha256") != sha256_file(manifest_path)
        or loso.get("diagnostics", {}).get("all_passed") is not True
    ):
        raise PermissionError("fit3 is closed until all source-only LOSO gates pass")
    head, training = _fit(payloads)
    fit_metrics = evaluate_typed_source_objective(head, payloads)
    fit_gate = _gate(fit_metrics)
    if not fit_gate["all_passed"]:
        raise RuntimeError(f"typed fit3 gates failed: {fit_gate}")
    checkpoint_payload = {
        "schema": SOURCE_TYPED_SPATIAL_CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "head_class": "BoundedTypedSourceSpatialLikelihoodHead",
        "head_schema_version": head.schema_version,
        "edge_types": list(head.edge_types),
        "state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()},
        "state_dict_sha256": state_dict_sha256(head.state_dict()),
        "recipe": dict(RECIPE),
        "dataset_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "loso_authority": {"path": str(loso_path), "sha256": sha256_file(loso_path)},
        "source_scene_ids": FIT_SCENES,
        "class_ids": list(payloads[0]["class_ids"]),
        "class_names": list(payloads[0]["class_names"]),
        "source_access": dict(manifest["source_access"]),
    }
    checkpoint = _write_torch_noclobber(args.checkpoint, checkpoint_payload)
    receipt = {
        "schema": FIT_RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_train_only_typed_mpr_spatial_fit",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "state_dict_sha256": checkpoint_payload["state_dict_sha256"],
        "dataset_manifest": checkpoint_payload["dataset_manifest"],
        "loso_authority": checkpoint_payload["loso_authority"],
        "fit3_metrics": fit_metrics,
        "fit3_gate": fit_gate,
        "training_trace": training["trace"],
        "scene0003_opened": False,
        "lerf_metric_run": False,
        "source_access": dict(manifest["source_access"]),
    }
    receipt_path = _write_json_noclobber(args.receipt, receipt)
    print(json.dumps({"receipt": str(receipt_path), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "state_dict_sha256": checkpoint_payload["state_dict_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
