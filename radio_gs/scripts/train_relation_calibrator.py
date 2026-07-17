#!/usr/bin/env python3
"""Train a global monotonic relation calibrator from official SAM3 regions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.relation_calibrator import MonotonicRelationCalibrator


def _paths(raw: str) -> list[Path]:
    result = []
    for value in str(raw).replace(",", " ").split():
        result.extend(sorted(Path().glob(value)) if "*" in value else [Path(value)])
    if not result or any(not path.is_file() for path in result):
        raise FileNotFoundError("relation cache list is empty or missing")
    return result


def _load(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    features, labels, scenes = [], [], []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata", {})
        if metadata.get("teacher") != "official_sam3_query_free_automatic_masks":
            raise ValueError(f"{path} is not an official query-free relation cache")
        if any(metadata.get(key, True) for key in ("labels_opened", "instances_opened", "text_opened")):
            raise ValueError(f"{path} violates relation-calibration provenance")
        features.append(torch.as_tensor(payload["features"]).float())
        labels.append(torch.as_tensor(payload["labels"]).float())
        scenes.append(str(payload["scene"]))
    return torch.cat(features), torch.cat(labels), scenes


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    score = torch.as_tensor(logits).float().cpu(); target = torch.as_tensor(labels).bool().cpu()
    order = torch.argsort(score, descending=True); sorted_target = target[order]
    positives = int(target.sum()); negatives = int((~target).sum())
    if not positives or not negatives: raise ValueError("AUC needs both relation classes")
    true_positive = sorted_target.cumsum(0).float()
    false_positive = (~sorted_target).cumsum(0).float()
    auc = torch.trapz(
        torch.cat([torch.zeros(1), true_positive / positives]),
        torch.cat([torch.zeros(1), false_positive / negatives]),
    ).item()
    prediction = score.sigmoid() >= 0.5
    tp = int((prediction & target).sum()); fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {"auc": float(auc), "f1_at_0.5": float(f1),
            "positive": positives, "negative": negatives}


@torch.no_grad()
def _evaluate(model, features, labels, device, batch_size) -> dict:
    logits = []
    for start in range(0, len(features), int(batch_size)):
        logits.append(model(features[start:start + int(batch_size)].to(device)).cpu())
    return binary_metrics(torch.cat(logits), labels)


def train(args: argparse.Namespace) -> dict:
    train_x, train_y, train_scenes = _load(_paths(args.train_caches))
    val_x, val_y, val_scenes = _load(_paths(args.validation_caches))
    overlap = set(train_scenes) & set(val_scenes)
    if overlap: raise ValueError(f"relation train/validation scene overlap: {sorted(overlap)}")
    device = torch.device(args.device); torch.manual_seed(int(args.seed))
    model = MonotonicRelationCalibrator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    positive_weight = float((train_y == 0).sum() / (train_y == 1).sum().clamp_min(1))
    generator = torch.Generator().manual_seed(int(args.seed))
    best_auc, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(int(args.epochs)):
        order = torch.randperm(len(train_x), generator=generator); losses = []
        model.train()
        for start in range(0, len(order), int(args.batch_size)):
            rows = order[start:start + int(args.batch_size)]
            logits = model(train_x[rows].to(device)); target = train_y[rows].to(device)
            loss = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=torch.tensor(positive_weight, device=device)
            )
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval(); metrics = _evaluate(model, val_x, val_y, device, args.batch_size)
        record = {"epoch": epoch + 1, "loss": sum(losses)/len(losses), **metrics}
        history.append(record); print(json.dumps(record), flush=True)
        if metrics["auc"] > best_auc:
            best_auc, best_epoch = metrics["auc"], epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None; model.load_state_dict(best_state)
    payload = {
        "schema_version": 1, "architecture": model.architecture(),
        "state_dict": best_state, "best_epoch": best_epoch, "history": history,
        "provenance": {"training_scope": "global_scene_disjoint_query_free_relation",
                       "teacher": "official_sam3_automatic_masks",
                       "train_scenes": train_scenes, "validation_scenes": val_scenes,
                       "labels_opened": False, "instances_opened": False,
                       "text_opened": False},
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, output)
    report = {"output": str(output.resolve()), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "best_epoch": best_epoch,
              "validation": _evaluate(model.to(device), val_x, val_y, device, args.batch_size),
              "train_edges": len(train_y), "validation_edges": len(val_y)}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); print(json.dumps(train(args), indent=2))


if __name__ == "__main__": main()
