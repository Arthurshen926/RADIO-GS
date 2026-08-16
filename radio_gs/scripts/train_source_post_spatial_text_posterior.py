#!/usr/bin/env python3
"""Run source-only LOSO gates and fit the post-spatial TextPosteriorV2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.querying.source_post_spatial_text_posterior import (
    EXTENT_FEATURE_NAMES,
    SOURCE_POST_SPATIAL_TEXT_CHECKPOINT_SCHEMA,
    validate_source_post_spatial_shard,
)
from radio_gs.querying.source_spatial_text_likelihood import (
    sha256_file,
    state_dict_sha256,
)
from radio_gs.querying.source_text_query_likelihood import (
    confidence_weighted_balanced_bce,
)
from radio_gs.querying.typed_posteriors import TextPosteriorV2


FIT_SCENES = ["scene0001_00", "scene0002_00", "scene0005_00"]
RECIPE = {
    "recipe_id": "post-spatial-text-posterior-v2-adam-seed17-e192-lr0.01-shrink0.6-deterministic-v4",
    "seed": 17,
    "optimizer": "Adam",
    "epochs": 192,
    "learning_rate": 0.01,
    "weight_decay": 0.0001,
    "hidden_dim": 32,
    "extent_feature_names": list(EXTENT_FEATURE_NAMES),
    "base_readout": "positive_cosine_knn10_per_query_scene_minmax_clip_2u_minus_1",
    "posterior_position": "after_complete_spatial_readout_before_fixed_threshold",
    "fixed_threshold": 0.6,
    "deployment_residual_scale": 0.6,
    "cpu_threads": 1,
    "deterministic_algorithms": True,
    "objective_weights": {
        "balanced_bce_over_legacy": 0.45,
        "local_relation_over_legacy": 0.25,
        "soft_jaccard_over_legacy": 0.30,
    },
    "fit_scenes": FIT_SCENES,
    "loso_folds": [
        {"fit": ["scene0002_00", "scene0005_00"], "heldout": "scene0001_00"},
        {"fit": ["scene0001_00", "scene0005_00"], "heldout": "scene0002_00"},
        {"fit": ["scene0001_00", "scene0002_00"], "heldout": "scene0005_00"},
    ],
}
MANIFEST_SCHEMA = "radio_gs.source_post_spatial_text_posterior_dataset_manifest.v1"
LOSO_SCHEMA = "radio_gs.source_post_spatial_text_posterior_loso_result.v1"
FIT_RECEIPT_SCHEMA = "radio_gs.source_post_spatial_text_posterior_fit_receipt.v1"


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


def load_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads(source.read_text())
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("scene_ids") != FIT_SCENES
    ):
        raise ValueError("post-spatial source dataset manifest differs")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(FIT_SCENES):
        raise ValueError("post-spatial source manifest records differ")
    payloads = []
    for scene_id, record in zip(FIT_SCENES, records):
        if not isinstance(record, Mapping) or record.get("scene_id") != scene_id:
            raise ValueError("post-spatial source manifest order differs")
        shard = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
        if sha256_file(shard) != record.get("sha256"):
            raise ValueError("post-spatial source shard changed")
        payload = validate_source_post_spatial_shard(
            torch.load(shard, map_location="cpu", weights_only=False)
        )
        if payload["scene_id"] != scene_id:
            raise ValueError("post-spatial source scene differs")
        payloads.append(payload)
    if len({payload["physical_space_id"] for payload in payloads}) != len(payloads):
        raise ValueError("post-spatial source scenes must be physical-space disjoint")
    return manifest, payloads


def _training_weight(payload: Mapping[str, Any]) -> torch.Tensor:
    reliability = payload["reliability"]
    authority = torch.sqrt(
        (reliability[:, 0] * reliability[:, 3]).clamp_min(0.0)
    )
    return (
        payload["valid"].float()
        * payload["training_label_weight"].float()
        * payload["coverage"].float()
        * authority
    )


def _present_class_indices(payload: Mapping[str, Any]) -> list[int]:
    target = payload["semantic_class_distribution"]
    weight = _training_weight(payload)
    return [
        index
        for index in range(target.shape[1])
        if float((weight * target[:, index]).sum()) > 0
        and float((weight * (1.0 - target[:, index])).sum()) > 0
    ]


def _predict(
    head: TextPosteriorV2,
    payload: Mapping[str, Any],
    *,
    residual_scale: float | None = None,
) -> torch.Tensor:
    return head.forward_post_spatial(
        payload["base_probability"],
        reliability=payload["reliability"],
        valid=payload["valid"],
        extent_features=payload["extent_features"],
        residual_scale=(
            float(RECIPE["deployment_residual_scale"])
            if residual_scale is None
            else float(residual_scale)
        ),
    ).probability


def _local_relation_loss(
    probability: torch.Tensor, payload: Mapping[str, Any]
) -> torch.Tensor:
    target = payload["semantic_class_distribution"].float()
    neighbors = payload["neighbor_indices"][:, 1:]
    weight = _training_weight(payload)
    edge_weight = torch.sqrt((weight[:, None] * weight[neighbors]).clamp_min(0.0))
    pred_relation = probability[:, None, :] - probability[neighbors]
    target_relation = target[:, None, :] - target[neighbors]
    error = F.smooth_l1_loss(pred_relation, target_relation, reduction="none")
    return (error * edge_weight[:, :, None]).sum() / (
        edge_weight.sum().clamp_min(1.0) * target.shape[1]
    )


def _soft_jaccard_loss(
    probability: torch.Tensor, payload: Mapping[str, Any]
) -> torch.Tensor:
    target = payload["semantic_class_distribution"].float()
    weight = _training_weight(payload)
    losses = []
    for class_index in _present_class_indices(payload):
        prediction = probability[:, class_index]
        truth = target[:, class_index]
        intersection = (weight * prediction * truth).sum()
        union = (weight * (prediction + truth - prediction * truth)).sum()
        losses.append(1.0 - intersection / union.clamp_min(1.0e-12))
    return torch.stack(losses).mean()


def _balanced_bce(
    probability: torch.Tensor, payload: Mapping[str, Any]
) -> torch.Tensor:
    target = payload["semantic_class_distribution"]
    weight = _training_weight(payload)
    losses = []
    for class_index in _present_class_indices(payload):
        loss, _ = confidence_weighted_balanced_bce(
            probability[:, class_index], target[:, class_index], weight
        )
        losses.append(loss)
    return torch.stack(losses).mean()


def _objective(
    head: TextPosteriorV2, payloads: Sequence[Mapping[str, Any]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bce, base_bce, local, base_local, jaccard, base_jaccard = ([] for _ in range(6))
    for payload in payloads:
        learned = _predict(head, payload, residual_scale=1.0)
        base = payload["base_probability"]
        bce.append(_balanced_bce(learned, payload))
        base_bce.append(_balanced_bce(base, payload).detach().clamp_min(1.0e-6))
        local.append(_local_relation_loss(learned, payload))
        base_local.append(
            _local_relation_loss(base, payload).detach().clamp_min(1.0e-6)
        )
        jaccard.append(_soft_jaccard_loss(learned, payload))
        base_jaccard.append(
            _soft_jaccard_loss(base, payload).detach().clamp_min(1.0e-6)
        )
    bce_value = torch.stack(bce).mean()
    local_value = torch.stack(local).mean()
    jaccard_value = torch.stack(jaccard).mean()
    total = (
        0.45 * torch.stack([x / y for x, y in zip(bce, base_bce)]).mean()
        + 0.25 * torch.stack([x / y for x, y in zip(local, base_local)]).mean()
        + 0.30
        * torch.stack([x / y for x, y in zip(jaccard, base_jaccard)]).mean()
    )
    return total, bce_value, local_value, jaccard_value


def _weighted_hard_iou(
    probability: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    threshold: float = 0.6,
) -> float:
    prediction = probability >= float(threshold)
    truth = target >= 0.5
    intersection = (weight * (prediction & truth).float()).sum()
    union = (weight * (prediction | truth).float()).sum()
    return float(intersection / union.clamp_min(1.0e-12))


@torch.inference_mode()
def evaluate(
    head: TextPosteriorV2, payloads: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = []
    local_rows = []
    for payload in payloads:
        learned = _predict(head, payload)
        base = payload["base_probability"]
        target = payload["semantic_class_distribution"]
        weight = _training_weight(payload)
        for class_index in _present_class_indices(payload):
            class_target = target[:, class_index]
            positive = weight * class_target
            negative = weight * (1.0 - class_target)
            for method, probability in (
                ("base", base[:, class_index]),
                ("learned", learned[:, class_index]),
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
                        "hard_iou_at_0.6": _weighted_hard_iou(
                            probability, class_target, weight
                        ),
                    }
                )
        for method, probability in (("base", base), ("learned", learned)):
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

    base_bce = macro("base", "balanced_bce")
    learned_bce = macro("learned", "balanced_bce")
    base_gap = macro("base", "positive_minus_negative_probability")
    learned_gap = macro("learned", "positive_minus_negative_probability")
    base_iou = macro("base", "hard_iou_at_0.6")
    learned_iou = macro("learned", "hard_iou_at_0.6")
    base_local = local_macro("base")
    learned_local = local_macro("learned")
    return {
        "present_scene_class_count": len(rows) // 2,
        "base_macro_balanced_bce": base_bce,
        "learned_macro_balanced_bce": learned_bce,
        "balanced_bce_delta": learned_bce - base_bce,
        "base_macro_positive_minus_negative_probability": base_gap,
        "learned_macro_positive_minus_negative_probability": learned_gap,
        "positive_negative_gap_delta": learned_gap - base_gap,
        "base_macro_hard_iou_at_0.6": base_iou,
        "learned_macro_hard_iou_at_0.6": learned_iou,
        "hard_iou_delta": learned_iou - base_iou,
        "base_macro_local_relation_loss": base_local,
        "learned_macro_local_relation_loss": learned_local,
        "local_relation_loss_delta": learned_local - base_local,
        "rows": rows,
        "local_rows": local_rows,
    }


def _fit(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[TextPosteriorV2, dict[str, Any]]:
    if torch.cuda.is_initialized():
        raise RuntimeError("post-spatial source fit must remain CPU-only")
    torch.set_num_threads(int(RECIPE["cpu_threads"]))
    torch.use_deterministic_algorithms(bool(RECIPE["deterministic_algorithms"]))
    torch.manual_seed(int(RECIPE["seed"]))
    head = TextPosteriorV2(
        extent_feature_dim=len(EXTENT_FEATURE_NAMES),
        hidden_dim=int(RECIPE["hidden_dim"]),
    ).cpu()
    initial = evaluate(head, payloads)
    if any(
        initial[key] != 0.0
        for key in (
            "balanced_bce_delta",
            "positive_negative_gap_delta",
            "hard_iou_delta",
            "local_relation_loss_delta",
        )
    ):
        raise RuntimeError("post-spatial zero initialization is not end-to-end identity")
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(RECIPE["learning_rate"]),
        weight_decay=float(RECIPE["weight_decay"]),
    )
    trace = []
    checkpoints = {0, 1, 3, 7, 15, 31, 63, 95, 127, 159, 191}
    for epoch in range(int(RECIPE["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        total, bce, local, jaccard = _objective(head, payloads)
        total.backward()
        optimizer.step()
        if epoch in checkpoints:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "total": float(total.detach()),
                    "macro_balanced_bce": float(bce.detach()),
                    "macro_local_relation_loss": float(local.detach()),
                    "macro_soft_jaccard_loss": float(jaccard.detach()),
                }
            )
    head.eval()
    return head, {"initial": initial, "trace": trace}


def _fold_admissible(metrics: Mapping[str, Any]) -> dict[str, bool]:
    base_local = float(metrics["base_macro_local_relation_loss"])
    checks = {
        "hard_iou_regression_within_0.01": float(metrics["hard_iou_delta"]) >= -0.01,
        "local_relation_regression_within_max_2pct_or_1e-5": float(
            metrics["local_relation_loss_delta"]
        )
        <= max(1.0e-5, 0.02 * base_local),
    }
    checks["all_passed"] = all(checks.values())
    return checks


def _macro_gate(metrics: Mapping[str, float]) -> dict[str, bool]:
    checks = {
        "hard_iou_improved": metrics["hard_iou_delta"] > 0,
        "balanced_bce_improved": metrics["balanced_bce_delta"] < 0,
        "positive_negative_gap_improved": metrics["positive_negative_gap_delta"] > 0,
        "local_relation_not_regressed_more_than_2pct": metrics[
            "local_relation_loss_delta"
        ]
        <= max(1.0e-5, 0.02 * metrics["base_macro_local_relation_loss"]),
    }
    checks["all_passed"] = all(checks.values())
    return checks


def run_loso(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scene = {payload["scene_id"]: payload for payload in payloads}
    folds = []
    for fold in RECIPE["loso_folds"]:
        head, training = _fit([by_scene[scene] for scene in fold["fit"]])
        metrics = evaluate(head, [by_scene[fold["heldout"]]])
        folds.append(
            {
                **fold,
                "heldout_metrics": metrics,
                "admissibility": _fold_admissible(metrics),
                "training_trace": training["trace"],
                "state_dict_sha256": state_dict_sha256(head.state_dict()),
            }
        )
    keys = (
        "balanced_bce_delta",
        "positive_negative_gap_delta",
        "hard_iou_delta",
        "local_relation_loss_delta",
        "base_macro_local_relation_loss",
    )
    macro = {
        key: float(sum(fold["heldout_metrics"][key] for fold in folds) / len(folds))
        for key in keys
    }
    gate = _macro_gate(macro)
    all_passed = gate["all_passed"] and all(
        fold["admissibility"]["all_passed"] for fold in folds
    )
    return {"folds": folds, "macro_deltas": macro, "macro_gate": gate, "all_passed": all_passed}


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
    manifest, payloads = load_manifest(manifest_path)
    if args.mode == "loso":
        if not args.result or any((args.loso_authority, args.checkpoint, args.receipt)):
            raise ValueError("LOSO mode requires only --result")
        diagnostics = run_loso(payloads)
        result = {
            "schema": LOSO_SCHEMA,
            "schema_version": 1,
            "status": "complete_three_fold_source_only_post_spatial_loso",
            "recipe": dict(RECIPE),
            "dataset_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "diagnostics": diagnostics,
            "development_scene0003_opened": False,
            "lerf_queries_or_ground_truth_opened": False,
            "source_access": dict(manifest["source_access"]),
        }
        output = _write_json_noclobber(args.result, result)
        print(
            json.dumps(
                {
                    "result": str(output),
                    "sha256": sha256_file(output),
                    "all_passed": diagnostics["all_passed"],
                    "macro_deltas": diagnostics["macro_deltas"],
                },
                sort_keys=True,
            )
        )
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
        raise PermissionError("post-spatial fit3 is closed until source-only LOSO passes")
    head, training = _fit(payloads)
    fit_metrics = evaluate(head, payloads)
    fit_gate = _macro_gate(fit_metrics)
    if not fit_gate["all_passed"]:
        raise RuntimeError(f"post-spatial fit3 gates failed: {fit_gate}")
    checkpoint_payload = {
        "schema": SOURCE_POST_SPATIAL_TEXT_CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "head_class": "TextPosteriorV2",
        "head_schema": head.schema,
        "extent_feature_names": list(EXTENT_FEATURE_NAMES),
        "state_dict": {
            name: value.detach().cpu() for name, value in head.state_dict().items()
        },
        "state_dict_sha256": state_dict_sha256(head.state_dict()),
        "recipe": dict(RECIPE),
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "loso_authority": {"path": str(loso_path), "sha256": sha256_file(loso_path)},
        "source_scene_ids": FIT_SCENES,
        "source_access": dict(manifest["source_access"]),
    }
    checkpoint = _write_torch_noclobber(args.checkpoint, checkpoint_payload)
    receipt = {
        "schema": FIT_RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_only_post_spatial_text_posterior_fit3",
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
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
