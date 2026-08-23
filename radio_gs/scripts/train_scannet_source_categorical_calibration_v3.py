#!/usr/bin/env python3
"""Fit a small class-complete ScanNet categorical calibration on source scenes.

The model has only a positive per-class scale and a per-class bias.  It cannot
invent spatial support or mix class identities; it corrects global categorical
competition learned from independent, scene-disjoint official annotations.
Every leave-one-scene-out fold must pass before a full-source checkpoint is
materialized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.scannet_source_categorical_calibration_v3.v1"


def calibrated_logits(
    scores: torch.Tensor, raw_scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return scores * F.softplus(raw_scale)[None] + bias[None]


def weighted_miou(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, classes: int) -> float:
    values = []
    for index in range(classes):
        truth = target == index
        if float(weight[truth].sum()) <= 0:
            continue
        predicted = prediction == index
        intersection = weight[truth & predicted].sum()
        union = weight[truth | predicted].sum()
        values.append(intersection / union.clamp_min(1.0e-12))
    if not values:
        raise ValueError("no source foreground class has positive authority")
    return float(torch.stack(values).mean())


def _load_scene(score_path: Path, geometry_path: Path, *, max_per_class: int) -> dict[str, Any]:
    score = torch.load(score_path, map_location="cpu", weights_only=False)
    geometry = torch.load(geometry_path, map_location="cpu", weights_only=False)
    if score.get("scene_id") != geometry.get("scene_id"):
        raise ValueError("source score and geometry scene differ")
    class_ids = list(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"])
    if list(score.get("class_ids", [])) != class_ids:
        raise ValueError("source score class order differs")
    statistics = geometry["statistics"]
    dominant = torch.as_tensor(statistics["dominant_nyu40_id"]).long()
    lookup = torch.full((41,), -1, dtype=torch.long)
    lookup[torch.tensor(class_ids)] = torch.arange(len(class_ids))
    target = lookup[dominant.clamp(0, 40)]
    reliability = torch.as_tensor(score["reliability"]).float()
    authority = (
        torch.as_tensor(statistics["semantic_purity"]).float()
        * torch.as_tensor(statistics["evidence_authority"]).float()
        * torch.sqrt((reliability[:, 0] * reliability[:, 1]).clamp_min(0.0))
    )
    valid = (
        torch.as_tensor(score["valid"]).bool()
        & torch.as_tensor(statistics["support_valid"]).bool()
        & (target >= 0)
        & (authority > 0)
    )
    # Deterministic, class-balanced cap bounds fit cost without selecting on a
    # metric.  Highest-authority rows are the least label-ambiguous source rows.
    chosen = []
    for index in range(len(class_ids)):
        rows = torch.where(valid & (target == index))[0]
        if rows.numel() > max_per_class:
            order = torch.argsort(authority[rows], descending=True, stable=True)
            rows = rows[order[:max_per_class]]
        chosen.append(rows)
    rows = torch.cat(chosen)
    if not rows.numel():
        raise ValueError("source scene has no class-supervised Gaussian")
    return {
        "scene_id": score["scene_id"],
        "scores": torch.as_tensor(score["semantic_scores"])[rows].float(),
        "target": target[rows],
        "weight": authority[rows],
        "score_path": score_path,
        "geometry_path": geometry_path,
    }


def _balance(scene: dict[str, Any], classes: int) -> torch.Tensor:
    target, weight = scene["target"], scene["weight"].double()
    mass = torch.stack([weight[target == index].sum() for index in range(classes)])
    inverse = torch.where(mass > 0, mass.reciprocal(), torch.zeros_like(mass))
    balanced = weight * inverse[target]
    return (balanced / balanced[balanced > 0].mean()).float()


def fit(
    scenes: Sequence[dict[str, Any]], *, steps: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    classes = len(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"])
    raw_scale = torch.nn.Parameter(torch.full((classes,), 0.5413248546, device=device))  # softplus ~= 1
    bias = torch.nn.Parameter(torch.zeros(classes, device=device))
    optimizer = torch.optim.AdamW([raw_scale, bias], lr=0.03, weight_decay=0.0)
    # Concatenate once.  Repeated per-scene Python dispatch and default
    # many-core CPU kernels made the original 38-parameter fit pathologically
    # slower without changing the objective.
    scores = torch.cat([scene["scores"] for scene in scenes]).to(device)
    target = torch.cat([scene["target"] for scene in scenes]).to(device)
    weight = torch.cat([_balance(scene, classes) for scene in scenes]).to(device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = calibrated_logits(scores, raw_scale, bias)
        ce = F.cross_entropy(logits, target, reduction="none")
        loss = (ce * weight).sum() / weight.sum().clamp_min(1.0) + 1.0e-3 * (
            (F.softplus(raw_scale) - 1).square().mean() + bias.square().mean()
        )
        loss.backward()
        optimizer.step()
    return raw_scale.detach().cpu(), bias.detach().cpu()


def evaluate(raw_scale: torch.Tensor, bias: torch.Tensor, scenes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    classes = len(raw_scale)
    for scene in scenes:
        baseline = weighted_miou(scene["scores"].argmax(1), scene["target"], scene["weight"], classes)
        calibrated = weighted_miou(
            calibrated_logits(scene["scores"], raw_scale, bias).argmax(1),
            scene["target"], scene["weight"], classes,
        )
        rows.append({"scene_id": scene["scene_id"], "baseline_miou": baseline, "calibrated_miou": calibrated, "delta": calibrated - baseline})
    return {
        "rows": rows,
        "baseline_scene_macro_miou": sum(row["baseline_miou"] for row in rows) / len(rows),
        "calibrated_scene_macro_miou": sum(row["calibrated_miou"] for row in rows) / len(rows),
        "delta": sum(row["delta"] for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--geometry-audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-per-class", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if len(args.score) != len(args.geometry_audit) or len(args.score) < 3:
        raise ValueError("at least three paired source scenes are required")
    scenes = [_load_scene(score.resolve(strict=True), geometry.resolve(strict=True), max_per_class=args.max_per_class) for score, geometry in zip(args.score, args.geometry_audit)]
    if len({scene["scene_id"] for scene in scenes}) != len(scenes):
        raise ValueError("source scenes must be distinct")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    folds = []
    for heldout in range(len(scenes)):
        train = [scene for index, scene in enumerate(scenes) if index != heldout]
        raw_scale, bias = fit(train, steps=args.steps, seed=args.seed, device=device)
        metrics = evaluate(raw_scale, bias, [scenes[heldout]])
        folds.append({"heldout": scenes[heldout]["scene_id"], "metrics": metrics, "passed": metrics["delta"] >= -0.01})
    all_folds_passed = all(fold["passed"] for fold in folds)
    if not all_folds_passed:
        failure_path = args.output.expanduser().resolve().with_suffix(
            args.output.suffix + ".source_gate_failure.json"
        )
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        if failure_path.exists():
            raise FileExistsError(f"refusing to overwrite {failure_path}")
        failure_path.write_text(json.dumps({
            "schema": SCHEMA,
            "schema_version": 1,
            "status": "rejected_source_loso_gate",
            "loso_folds": folds,
            "all_loso_folds_passed": False,
            "recipe": {"steps": args.steps, "seed": args.seed, "max_per_class": args.max_per_class, "model": "positive_diagonal_scale_plus_bias", "execution_device": str(device)},
            "sources": [{"scene_id": scene["scene_id"], "score": {"path": str(scene["score_path"]), "sha256": sha256_file(scene["score_path"])}, "geometry_audit": {"path": str(scene["geometry_path"]), "sha256": sha256_file(scene["geometry_path"])}} for scene in scenes],
            "access_contract": {"paper8_labels_opened": False, "paper8_metrics_opened": False},
        }, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"source LOSO gate failed: {folds}")
    raw_scale, bias = fit(scenes, steps=args.steps, seed=args.seed, device=device)
    source_metrics = evaluate(raw_scale, bias, scenes)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "class_ids": list(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"]),
        "raw_scale": raw_scale,
        "positive_scale": F.softplus(raw_scale),
        "bias": bias,
        "loso_folds": folds,
        "all_loso_folds_passed": all_folds_passed,
        "source_fit_metrics": source_metrics,
        "recipe": {"steps": args.steps, "seed": args.seed, "max_per_class": args.max_per_class, "model": "positive_diagonal_scale_plus_bias", "execution_device": str(device)},
        "sources": [{"scene_id": scene["scene_id"], "score": {"path": str(scene["score_path"]), "sha256": sha256_file(scene["score_path"])}, "geometry_audit": {"path": str(scene["geometry_path"]), "sha256": sha256_file(scene["geometry_path"])}} for scene in scenes],
        "access_contract": {"paper8_labels_opened": False, "paper8_metrics_opened": False},
    }
    torch.save(payload, output)
    print(json.dumps({"output": str(output), "loso_folds": folds, "source_fit_metrics": source_metrics}, indent=2))


if __name__ == "__main__":
    main()
