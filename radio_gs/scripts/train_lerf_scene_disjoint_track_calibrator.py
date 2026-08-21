#!/usr/bin/env python3
"""Train a scene-shared Bernoulli track-association calibrator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_scene_disjoint_track_calibrator.v1"


def _auc(logit: torch.Tensor, label: torch.Tensor) -> float:
    order = torch.argsort(logit)
    sorted_score = logit[order]
    sorted_label = label[order]
    ranks = torch.arange(1, logit.numel() + 1, dtype=torch.float64)
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    offset = 0
    for count in counts.tolist():
        if count > 1:
            ranks[offset : offset + count] = ranks[offset : offset + count].mean()
        offset += count
    positive_count = int((sorted_label == 1).sum())
    negative_count = int((sorted_label == 0).sum())
    rank_sum = ranks[sorted_label == 1].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def _fit(features: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, steps: int) -> tuple[dict, dict]:
    mean, std = features.mean(0), features.std(0).clamp_min(1e-6)
    x = (features - mean) / std
    linear = torch.nn.Linear(features.shape[1], 1)
    torch.nn.init.zeros_(linear.weight); torch.nn.init.zeros_(linear.bias)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=0.03, weight_decay=1e-4)
    positive, negative = labels == 1, labels == 0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True); logit = linear(x).squeeze(1)
        pos = (F.binary_cross_entropy_with_logits(logit[positive], torch.ones_like(logit[positive]), reduction="none") * weights[positive]).sum() / weights[positive].sum().clamp_min(1e-8)
        neg = (F.binary_cross_entropy_with_logits(logit[negative], torch.zeros_like(logit[negative]), reduction="none") * weights[negative]).sum() / weights[negative].sum().clamp_min(1e-8)
        loss = pos + neg; loss.backward(); optimizer.step()
    state = {"mean": mean, "std": std, "weight": linear.weight.detach().squeeze(0), "bias": linear.bias.detach().squeeze(0)}
    return state, {"train_balanced_log_score": float(loss.detach()), "train_auc": _auc(linear(x).detach().squeeze(1), labels)}


def _predict(state: dict, features: torch.Tensor) -> torch.Tensor:
    return ((features - state["mean"]) / state["std"]) @ state["weight"] + state["bias"]


def build(args: argparse.Namespace) -> dict:
    torch.set_num_threads(min(4, max(1, int(args.cpu_threads))))
    scene_paths = {Path(value).stem.removesuffix("_v2"): Path(value).resolve() for value in args.authority}
    output = Path(args.output).resolve(); report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists(): raise FileExistsError(f"calibrator exists: {output}")
    datasets = {}
    feature_names = None
    for scene, path in scene_paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        names = [str(value) for value in payload["feature_names"]]
        if feature_names is None: feature_names = names
        if names != feature_names or payload.get("metadata", {}).get("benchmark_masks_opened") is not False: raise ValueError("track authority contract differs")
        label = torch.as_tensor(payload["edge_label"]).long(); known = label >= 0
        datasets[scene] = (torch.as_tensor(payload["edge_features"]).float()[known], label[known], torch.as_tensor(payload["edge_weight"]).float()[known])
    loso = {}
    for heldout in sorted(datasets):
        train = [value for scene, value in datasets.items() if scene != heldout]
        state, _ = _fit(torch.cat([v[0] for v in train]), torch.cat([v[1] for v in train]), torch.cat([v[2] for v in train]), int(args.steps))
        x, y, _ = datasets[heldout]; logit = _predict(state, x)
        pos, neg = y == 1, y == 0
        score = 0.5 * (F.binary_cross_entropy_with_logits(logit[pos], torch.ones_like(logit[pos])) + F.binary_cross_entropy_with_logits(logit[neg], torch.zeros_like(logit[neg])))
        loso[heldout] = {"auc": _auc(logit, y), "balanced_log_score": float(score), "epoch_zero_balanced_log_score": float(torch.log(torch.tensor(2.0))), "known_edges": int(y.numel())}
    all_x = torch.cat([v[0] for v in datasets.values()]); all_y = torch.cat([v[1] for v in datasets.values()]); all_w = torch.cat([v[2] for v in datasets.values()])
    state, train_metrics = _fit(all_x, all_y, all_w, int(args.steps))
    passed = all(row["balanced_log_score"] < row["epoch_zero_balanced_log_score"] and row["auc"] > 0.5 for row in loso.values())
    checkpoint = {"schema": SCHEMA, "schema_version": 1, "feature_names": feature_names, **state,
                  "metadata": {"source_scenes": sorted(datasets), "benchmark_masks_opened": False, "evaluation_rgb_opened": False, "figurines_opened": False, "proper_score": "class_balanced_Bernoulli_log_score", "formal_stage_a_complete": False, "null_calibration": "unavailable_no_explicit_null_labels", "authorities": {scene: {"path": str(path), "sha256": sha256_file(path)} for scene, path in scene_paths.items()}}}
    output.parent.mkdir(parents=True, exist_ok=True); temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp"); torch.save(checkpoint, temporary); os.replace(temporary, output)
    report = {"schema": SCHEMA, "status": "source_loso_pass" if passed else "source_loso_fail", "formal_stage_a_complete": False, "train": train_metrics, "loso": loso, "output": str(output), "output_sha256": sha256_file(output)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n"); return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--authority", action="append", required=True); parser.add_argument("--steps", type=int, default=200); parser.add_argument("--cpu-threads", type=int, default=4); parser.add_argument("--output", required=True); print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__": main()
