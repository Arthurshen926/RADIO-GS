#!/usr/bin/env python3
"""Source-only LOSO and fit for paper-eligible CategoricalPosteriorV2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.querying.source_post_spatial_text_posterior import (
    validate_source_post_spatial_shard,
)
from radio_gs.querying.source_spatial_text_likelihood import (
    sha256_file,
    state_dict_sha256,
)
from radio_gs.querying.typed_posteriors import CategoricalPosteriorV2
from radio_gs.scripts.train_source_post_spatial_text_posterior import (
    FIT_SCENES,
    MANIFEST_SCHEMA,
)


RECIPE = {
    "recipe_id": "source-categorical-posterior-v2-adamw-seed20260816-s300-v1",
    "seed": 20260816,
    "optimizer": "AdamW",
    "steps": 300,
    "learning_rate": 0.03,
    "regularization": 0.01,
    "cpu_threads": 1,
    "deterministic_algorithms": True,
    "fit_scenes": FIT_SCENES,
    "loso_folds": [
        {"fit": ["scene0002_00", "scene0005_00"], "heldout": "scene0001_00"},
        {"fit": ["scene0001_00", "scene0005_00"], "heldout": "scene0002_00"},
        {"fit": ["scene0001_00", "scene0002_00"], "heldout": "scene0005_00"},
    ],
}
LOSO_SCHEMA = "radio_gs.source_categorical_posterior_v2_loso_result.v1"
CHECKPOINT_SCHEMA = "radio_gs.source_categorical_posterior_v2_checkpoint.v1"
FIT_RECEIPT_SCHEMA = "radio_gs.source_categorical_posterior_v2_fit_receipt.v1"


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
        raise ValueError("source categorical manifest differs")
    payloads = []
    for scene_id, record in zip(FIT_SCENES, manifest.get("records", [])):
        if record.get("scene_id") != scene_id:
            raise ValueError("source categorical manifest order differs")
        path_value = Path(record["path"]).expanduser().resolve(strict=True)
        if sha256_file(path_value) != record.get("sha256"):
            raise ValueError("source categorical shard changed")
        payload = validate_source_post_spatial_shard(
            torch.load(path_value, map_location="cpu", weights_only=False)
        )
        if payload["scene_id"] != scene_id:
            raise ValueError("source categorical scene differs")
        payloads.append(payload)
    if len(payloads) != len(FIT_SCENES):
        raise ValueError("source categorical shard count differs")
    return manifest, payloads


def soft_targets(payload: Mapping[str, Any]) -> torch.Tensor:
    foreground = payload["semantic_class_distribution"].float()
    background = (1.0 - foreground.sum(dim=-1, keepdim=True)).clamp(0.0, 1.0)
    target = torch.cat((foreground, background), dim=-1)
    denominator = target.sum(dim=-1, keepdim=True)
    if bool((denominator <= 0).any()):
        raise ValueError("source categorical target has zero mass")
    return (target / denominator).contiguous()


def _row_weights(payload: Mapping[str, Any], target: torch.Tensor) -> torch.Tensor:
    reliability = payload["reliability"]
    authority = torch.sqrt(
        (reliability[:, 0] * reliability[:, 3]).clamp_min(0.0)
    )
    base = (
        payload["valid"].float()
        * payload["training_label_weight"].float()
        * payload["coverage"].float()
        * authority
    )
    class_mass = (target.double() * base.double()[:, None]).sum(dim=0)
    present = class_mass > 0
    inverse = torch.zeros_like(class_mass)
    inverse[present] = 1.0 / class_mass[present]
    balanced = base.double() * (target.double() * inverse[None, :]).sum(dim=-1)
    active = balanced > 0
    balanced[active] = balanced[active] / balanced[active].mean()
    return balanced.float()


def weighted_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, float]:
    selected = torch.as_tensor(prediction).long().reshape(-1)
    truth = torch.as_tensor(target).argmax(dim=-1).long()
    values = []
    accuracies = []
    foreground_classes = target.shape[1] - 1
    for class_index in range(foreground_classes):
        positive = truth == class_index
        if float(weight[positive].sum()) <= 0:
            continue
        predicted = selected == class_index
        intersection = weight[predicted & positive].sum()
        union = weight[predicted | positive].sum()
        values.append(intersection / union.clamp_min(1.0e-12))
        accuracies.append(intersection / weight[positive].sum().clamp_min(1.0e-12))
    if not values:
        raise ValueError("source categorical scene has no foreground classes")
    return {
        "miou": float(torch.stack(values).mean()),
        "macc": float(torch.stack(accuracies).mean()),
        "present_foreground_classes": len(values),
    }


def _predict(model: CategoricalPosteriorV2, payload: Mapping[str, Any]) -> torch.Tensor:
    with torch.inference_mode():
        return model(
            payload["raw_positive_cosine"],
            reliability=payload["reliability"],
            valid=payload["valid"],
        ).prediction


def evaluate(
    model: CategoricalPosteriorV2, payloads: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = []
    for payload in payloads:
        target = soft_targets(payload)
        weight = _row_weights(payload, target)
        baseline = payload["raw_positive_cosine"].argmax(dim=-1)
        baseline = torch.where(
            payload["valid"], baseline, torch.full_like(baseline, -1)
        )
        calibrated = _predict(model, payload)
        rows.append(
            {
                "scene_id": payload["scene_id"],
                "baseline": weighted_metrics(baseline, target, weight),
                "calibrated": weighted_metrics(calibrated, target, weight),
            }
        )
    base_iou = sum(row["baseline"]["miou"] for row in rows) / len(rows)
    calibrated_iou = sum(row["calibrated"]["miou"] for row in rows) / len(rows)
    base_acc = sum(row["baseline"]["macc"] for row in rows) / len(rows)
    calibrated_acc = sum(row["calibrated"]["macc"] for row in rows) / len(rows)
    return {
        "baseline_scene_macro_miou": base_iou,
        "calibrated_scene_macro_miou": calibrated_iou,
        "miou_delta": calibrated_iou - base_iou,
        "baseline_scene_macro_macc": base_acc,
        "calibrated_scene_macro_macc": calibrated_acc,
        "macc_delta": calibrated_acc - base_acc,
        "rows": rows,
    }


def _fit(payloads: Sequence[Mapping[str, Any]]) -> tuple[CategoricalPosteriorV2, list[float]]:
    torch.set_num_threads(int(RECIPE["cpu_threads"]))
    torch.use_deterministic_algorithms(bool(RECIPE["deterministic_algorithms"]))
    torch.manual_seed(int(RECIPE["seed"]))
    model = CategoricalPosteriorV2(num_classes=19).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(RECIPE["learning_rate"]), weight_decay=0.0
    )
    prepared = []
    for payload in payloads:
        target = soft_targets(payload)
        prepared.append((payload, target, _row_weights(payload, target)))
    history = []
    for step in range(int(RECIPE["steps"])):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for payload, target, weight in prepared:
            output = model(
                payload["raw_positive_cosine"],
                reliability=payload["reliability"],
                valid=payload["valid"],
            )
            ce = -(target * F.log_softmax(output.logits, dim=-1)).sum(dim=-1)
            losses.append((ce * weight).sum() / weight.sum().clamp_min(1.0))
        calibration = torch.stack(losses).mean()
        regularization = sum(parameter.square().mean() for parameter in model.parameters())
        loss = calibration + float(RECIPE["regularization"]) * regularization
        loss.backward()
        optimizer.step()
        if step in {0, int(RECIPE["steps"]) - 1} or (step + 1) % 50 == 0:
            history.append(float(loss.detach()))
    model.eval()
    return model, history


def _fold_gate(metrics: Mapping[str, float]) -> dict[str, bool]:
    checks = {
        "miou_regression_within_0.01": float(metrics["miou_delta"]) >= -0.01,
        "macc_regression_within_0.01": float(metrics["macc_delta"]) >= -0.01,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def run_loso(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scene = {payload["scene_id"]: payload for payload in payloads}
    folds = []
    for fold in RECIPE["loso_folds"]:
        model, history = _fit([by_scene[scene] for scene in fold["fit"]])
        metrics = evaluate(model, [by_scene[fold["heldout"]]])
        folds.append(
            {
                **fold,
                "heldout_metrics": metrics,
                "gate": _fold_gate(metrics),
                "loss_history": history,
                "state_dict_sha256": state_dict_sha256(model.state_dict()),
            }
        )
    macro = {
        key: sum(fold["heldout_metrics"][key] for fold in folds) / len(folds)
        for key in ("miou_delta", "macc_delta")
    }
    macro_gate = {
        "miou_improved": macro["miou_delta"] > 0,
        "macc_regression_within_0.01": macro["macc_delta"] >= -0.01,
    }
    macro_gate["all_passed"] = all(macro_gate.values())
    passed = macro_gate["all_passed"] and all(
        fold["gate"]["all_passed"] for fold in folds
    )
    return {"folds": folds, "macro_delta": macro, "macro_gate": macro_gate, "all_passed": passed}


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
            raise ValueError("categorical LOSO mode requires only --result")
        diagnostics = run_loso(payloads)
        result = {
            "schema": LOSO_SCHEMA,
            "schema_version": 1,
            "status": "complete_three_fold_source_only_categorical_loso",
            "recipe": dict(RECIPE),
            "dataset_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "diagnostics": diagnostics,
            "paper_eight_labels_opened": False,
            "source_access": dict(manifest["source_access"]),
        }
        output = _write_json_noclobber(args.result, result)
        print(json.dumps({"result": str(output), "sha256": sha256_file(output), "diagnostics": diagnostics}, sort_keys=True))
        return
    if not all((args.loso_authority, args.checkpoint, args.receipt)) or args.result:
        raise ValueError("categorical fit3 requires --loso-authority --checkpoint --receipt")
    loso_path = Path(args.loso_authority).expanduser().resolve(strict=True)
    loso = json.loads(loso_path.read_text())
    if (
        loso.get("schema") != LOSO_SCHEMA
        or loso.get("dataset_manifest", {}).get("sha256") != sha256_file(manifest_path)
        or loso.get("diagnostics", {}).get("all_passed") is not True
    ):
        raise PermissionError("categorical fit3 is closed until source LOSO passes")
    model, history = _fit(payloads)
    fit_metrics = evaluate(model, payloads)
    if fit_metrics["miou_delta"] <= 0 or fit_metrics["macc_delta"] < -0.01:
        raise RuntimeError("categorical fit3 source gate failed")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "model_schema": model.schema,
        "num_classes": 19,
        "class_ids": list(payloads[0]["class_ids"]),
        "class_names": list(payloads[0]["class_names"]),
        "state_dict": state,
        "state_dict_sha256": state_dict_sha256(state),
        "recipe": dict(RECIPE),
        "dataset_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "loso_authority": {"path": str(loso_path), "sha256": sha256_file(loso_path)},
        "paper_eight_labels_opened": False,
        "source_access": dict(manifest["source_access"]),
    }
    checkpoint = _write_torch_noclobber(args.checkpoint, checkpoint_payload)
    receipt = {
        "schema": FIT_RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_only_categorical_posterior_v2_fit3",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "state_dict_sha256": checkpoint_payload["state_dict_sha256"],
        "fit_metrics": fit_metrics,
        "loss_history": history,
        "paper_eight_labels_opened": False,
    }
    receipt_path = _write_json_noclobber(args.receipt, receipt)
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
