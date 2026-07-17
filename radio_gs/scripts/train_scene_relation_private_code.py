#!/usr/bin/env python3
"""Fit a tiny query-free per-primitive relation code from official SAM3 regions.

The code is deliberately relation-private: it changes only edge merge scores
and never enters RADIO, text, image, or prompt descriptors.  Checkpoint
selection uses held-out RGB frames from the same scene and never benchmark
labels, queries, or masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from radio_gs.interfaces.relation_calibrator import MonotonicRelationCalibrator
from radio_gs.scripts.train_relation_calibrator import binary_metrics


class SceneRelationPrivateCode(nn.Module):
    def __init__(self, num_nodes: int, dimension: int = 8) -> None:
        super().__init__()
        if num_nodes <= 0 or dimension <= 0:
            raise ValueError("num_nodes and dimension must be positive")
        self.code = nn.Embedding(num_nodes, dimension)
        nn.init.normal_(self.code.weight, std=0.01)
        self.raw_scale = nn.Parameter(torch.tensor(0.0))
        self.margin = nn.Parameter(torch.tensor(0.5))

    def residual(self, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = torch.as_tensor(edge_index).long()
        squared_distance = (self.code(src) - self.code(dst)).square().mean(-1)
        return F.softplus(self.raw_scale) * (self.margin - squared_distance)


def _load_cache(path: str | Path, *, graph: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("scene")) != str(graph.get("scene")):
        raise ValueError("relation cache and graph scene differ")
    metadata = payload.get("metadata", {})
    if any(metadata.get(key, True) for key in ("labels_opened", "instances_opened", "text_opened")):
        raise ValueError("relation-private supervision violates query-free provenance")
    rows = torch.as_tensor(payload.get("edge_rows")).long()
    labels = torch.as_tensor(payload.get("labels")).float()
    if rows.ndim != 1 or labels.shape != rows.shape:
        raise ValueError("relation cache must contain aligned edge_rows/labels")
    if int(payload.get("num_nodes", -1)) != int(graph["xyz"].shape[0]):
        raise ValueError("relation cache and graph node counts differ")
    return {**payload, "edge_rows": rows, "labels": labels}


def _base_logits(
    graph: dict, cache: dict, calibrator: MonotonicRelationCalibrator | None,
) -> torch.Tensor:
    if calibrator is None:
        return torch.zeros(len(cache["labels"]))
    with torch.no_grad():
        return calibrator(torch.as_tensor(cache["features"]).float()).cpu()


@torch.no_grad()
def _metrics(model, graph_edge, cache, base, device) -> dict:
    rows = cache["edge_rows"]
    residual = model.residual(graph_edge[:, rows].to(device)).cpu()
    return binary_metrics(base + residual, cache["labels"])


def train(args: argparse.Namespace) -> dict:
    graph = torch.load(args.scene_graph, map_location="cpu", weights_only=False)
    edge = torch.as_tensor(graph["edge_index"]).long().cpu()
    train_cache = _load_cache(args.train_cache, graph=graph)
    val_cache = _load_cache(args.validation_cache, graph=graph)
    train_frames = set(train_cache["metadata"].get("mask_frames", []))
    val_frames = set(val_cache["metadata"].get("mask_frames", []))
    if train_frames & val_frames:
        raise ValueError("relation-private train/validation frames overlap")

    calibrator = None
    if args.global_calibrator:
        checkpoint = torch.load(args.global_calibrator, map_location="cpu", weights_only=False)
        calibrator = MonotonicRelationCalibrator()
        calibrator.load_state_dict(checkpoint["state_dict"]); calibrator.eval()
    train_base = _base_logits(graph, train_cache, calibrator)
    val_base = _base_logits(graph, val_cache, calibrator)

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    model = SceneRelationPrivateCode(
        int(graph["xyz"].shape[0]), int(args.dimension)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    positive_weight = float(
        (train_cache["labels"] == 0).sum()
        / (train_cache["labels"] == 1).sum().clamp_min(1)
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    best_auc, best_epoch, best_state, history = -1.0, 0, None, []
    rows_all = train_cache["edge_rows"]
    labels_all = train_cache["labels"]
    for epoch in range(int(args.epochs)):
        order = torch.randperm(len(rows_all), generator=generator)
        losses = []
        model.train()
        for start in range(0, len(order), int(args.batch_size)):
            selected = order[start:start + int(args.batch_size)]
            edge_rows = rows_all[selected]
            logits = train_base[selected].to(device) + model.residual(
                edge[:, edge_rows].to(device)
            )
            targets = labels_all[selected].to(device)
            relation_loss = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=torch.tensor(positive_weight, device=device),
            )
            used_nodes = torch.unique(edge[:, edge_rows]).to(device)
            regularizer = model.code(used_nodes).square().mean()
            loss = relation_loss + float(args.code_regularization) * regularizer
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        metrics = _metrics(model, edge, val_cache, val_base, device)
        record = {"epoch": epoch + 1, "loss": sum(losses) / len(losses), **metrics}
        history.append(record); print(json.dumps(record), flush=True)
        if metrics["auc"] > best_auc:
            best_auc, best_epoch = metrics["auc"], epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    # A single scalar per graph edge is enough for the downstream maximum
    # spanning tree; the 8D code is retained only for method/audit inspection.
    all_residual = torch.empty(edge.shape[1], dtype=torch.float16)
    for start in range(0, edge.shape[1], int(args.batch_size)):
        stop = min(edge.shape[1], start + int(args.batch_size))
        all_residual[start:stop] = model.residual(edge[:, start:stop].to(device)).half().cpu()
    payload = {
        "schema_version": 1,
        "scene": str(graph["scene"]),
        "scene_graph_sha256": hashlib.sha256(Path(args.scene_graph).read_bytes()).hexdigest(),
        "num_graph_edges": int(edge.shape[1]),
        "architecture": {
            "type": "scene_relation_private_code",
            "dimension": int(args.dimension),
            "parameters_per_primitive": int(args.dimension),
            "edge_readout": "positive_margin_minus_squared_code_distance",
        },
        "state_dict": {**best_state, "code.weight": best_state["code.weight"].half()},
        "edge_residual": all_residual,
        "best_epoch": int(best_epoch),
        "history": history,
        "provenance": {
            "training_scope": "scene_specific_query_free_relation_only",
            "teacher": "official_sam3_automatic_masks",
            "train_frames": sorted(train_frames),
            "validation_frames": sorted(val_frames),
            "labels_opened": False,
            "instances_opened": False,
            "text_opened": False,
            "global_calibrator": str(Path(args.global_calibrator).resolve()) if args.global_calibrator else "",
        },
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    validation = _metrics(model, edge, val_cache, val_base, device)
    report = {
        "output": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "scene": str(graph["scene"]),
        "best_epoch": int(best_epoch),
        "validation": validation,
        "base_validation": binary_metrics(val_base, val_cache["labels"]),
        "train_edges": int(len(labels_all)),
        "validation_edges": int(len(val_cache["labels"])),
        "persistent_code_bytes_fp16": int(best_state["code.weight"].numel() * 2),
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-graph", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--global-calibrator", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--code-regularization", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    print(json.dumps(train(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
