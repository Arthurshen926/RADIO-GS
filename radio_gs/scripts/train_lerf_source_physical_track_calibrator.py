#!/usr/bin/env python3
"""Fit and gate a global source-only physical-track association calibrator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_source_physical_track_calibrator.v1"


def rank_auc(score: torch.Tensor, label: torch.Tensor) -> float:
    positive, negative = score[label == 1], score[label == 0]
    if not positive.numel() or not negative.numel():
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=torch.float64)
    return float((ranks[label == 1].sum() - positive.numel() * (positive.numel() + 1) / 2) /
                 (positive.numel() * negative.numel()))


def balanced_log_score(logit: torch.Tensor, label: torch.Tensor) -> float:
    losses = F.binary_cross_entropy_with_logits(logit, label.float(), reduction="none")
    return float(0.5 * losses[label == 1].mean() + 0.5 * losses[label == 0].mean())


def build(args: argparse.Namespace) -> dict:
    paths = [Path(value).resolve() for value in args.authority]
    scenes = []
    train_x, train_y = [], []
    heldout = []
    feature_names = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "radio_gs.lerf_source_dino_physical_track_authority.v1":
            raise ValueError("physical-track authority schema differs")
        if payload.get("metadata", {}).get("figurines_opened") is not False:
            raise ValueError("source authority opened heldout figurines")
        names = list(payload["feature_names"])
        if feature_names is None: feature_names = names
        if names != feature_names: raise ValueError("feature schemas differ")
        x = torch.as_tensor(payload["edge_features"]).float()
        y = torch.as_tensor(payload["edge_label"]).long()
        left = torch.as_tensor(payload["edge_left"]).long()
        right = torch.as_tensor(payload["edge_right"]).long()
        views = torch.as_tensor(payload["proposal_views"]).long()
        known = y >= 0
        is_heldout = (views[left] % 4 == 3) | (views[right] % 4 == 3)
        training = known & ~is_heldout
        testing = known & is_heldout
        if not bool(training.any()) or not bool(testing.any()):
            raise ValueError(f"{payload['scene']} lacks train/heldout known edges")
        for split in (training, testing):
            labels = y[split]
            if not bool((labels == 0).any()) or not bool((labels == 1).any()):
                raise ValueError(f"{payload['scene']} split lacks both classes")
        train_x.append(x[training]); train_y.append(y[training])
        heldout.append((str(payload["scene"]), x[testing], y[testing]))
        scenes.append({"scene": payload["scene"], "path": str(path), "sha256": sha256_file(path)})
    x_train, y_train = torch.cat(train_x), torch.cat(train_y)
    mean, scale = x_train.mean(0), x_train.std(0).clamp_min(1e-4)
    weight = torch.zeros(x_train.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logit = ((x_train - mean) / scale) @ weight + bias
        loss_parts = F.binary_cross_entropy_with_logits(logit, y_train.float(), reduction="none")
        loss = 0.5 * loss_parts[y_train == 1].mean() + 0.5 * loss_parts[y_train == 0].mean()
        loss = loss + 1e-3 * weight.square().sum()
        loss.backward(); return loss
    optimizer.step(closure)
    reports = {}
    passed = True
    with torch.no_grad():
        for scene, x, y in heldout:
            logit = ((x - mean) / scale) @ weight + bias
            auc = rank_auc(logit, y)
            proper = balanced_log_score(logit, y)
            scene_pass = auc > 0.5 and proper < 0.6931471805599453
            passed &= scene_pass
            reports[scene] = {"known_edges": int(y.numel()), "same_edges": int((y == 1).sum()),
                              "different_edges": int((y == 0).sum()), "auc": auc,
                              "balanced_log_score": proper, "epoch0_balanced_log_score": 0.6931471805599453,
                              "pass": scene_pass}
    output = Path(args.output).resolve(); report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists(): raise FileExistsError(f"calibrator exists: {output}")
    payload = {"schema": SCHEMA, "schema_version": 1, "feature_names": feature_names,
               "feature_mean": mean, "feature_scale": scale, "weight": weight.detach(), "bias": bias.detach(),
               "metadata": {"source_only": True, "figurines_opened": False,
                            "label_authority": "independent_DINO_three_view_transport_cycle",
                            "proper_gate": "every_scene_heldout_view_AUC_gt_0.5_and_balanced_log_score_lt_log2",
                            "source_assets": scenes}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp"); torch.save(payload, temporary); os.replace(temporary, output)
    report = {"schema": SCHEMA, "status": "source_heldout_gate_pass" if passed else "source_heldout_gate_fail",
              "formal_stage_a_complete": passed, "figurines_opened": False, "scenes": reports,
              "output": str(output), "output_sha256": sha256_file(output)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__": main()
