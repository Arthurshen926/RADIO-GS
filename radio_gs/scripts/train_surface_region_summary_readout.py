#!/usr/bin/env python3
"""Train the global query-free 3-D surface-region RADIO summary readout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadout
from radio_gs.models.siglip_projection import SigLIP2SummaryHead


def _paths(raw: str) -> list[Path]:
    result = []
    for value in str(raw).replace(",", " ").split():
        matches = sorted(Path().glob(value)) if any(c in value for c in "*?[") else [Path(value)]
        result.extend(matches)
    if not result or any(not path.is_file() for path in result):
        raise FileNotFoundError("surface-region cache list is empty or missing")
    return result


def _load(paths: list[Path], expected_role: str) -> tuple[dict, dict]:
    keys = (
        "radio_features", "geometry", "token_mask", "reliability",
        "official_summary_tokens", "official_crop_summaries", "teacher_mask",
    )
    parts = {key: [] for key in keys}; scenes = set(); hashes = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata", {})
        if metadata.get("schema_version") != 2 or metadata.get("split_role") != expected_role:
            raise ValueError(f"{path} has wrong 3-D cache schema/split")
        if any(metadata.get(key, True) for key in (
            "uses_benchmark_scenes", "uses_benchmark_test_vocabulary",
            "annotations_opened", "labels_opened", "instances_opened", "text_opened",
        )):
            raise ValueError(f"{path} violates the query-free scene-disjoint contract")
        scenes.update(str(value) for value in metadata["scene_names"])
        hashes.append(str(metadata["split_file_sha256"]))
        for key in keys:
            parts[key].append(torch.as_tensor(payload[key]))
    merged = {key: torch.cat(value, dim=0) for key, value in parts.items()}
    return merged, {"scenes": sorted(scenes), "split_hashes": sorted(set(hashes)),
                    "cache_paths": [str(path.resolve()) for path in paths]}


def _targets(data: dict, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = data["official_summary_tokens"][rows].float()
    descriptors = F.normalize(data["official_crop_summaries"][rows].float(), dim=-1)
    mask = data["teacher_mask"][rows].bool()
    # The medoid is selected in the official SigLIP2 descriptor space, not
    # by averaging or comparing backbone summary tokens.
    normalized = descriptors
    similarity = torch.einsum("bvd,bwd->bvw", normalized, normalized)
    similarity = similarity.masked_fill(~mask[:, None, :], 0.0)
    medoid = similarity.sum(-1).masked_fill(~mask, -1e9).argmax(-1)
    batch = torch.arange(len(rows))
    target_token = tokens[batch, medoid]
    weights = mask.float() / mask.sum(1, keepdim=True)
    target_descriptor = F.normalize(
        (descriptors * weights[..., None]).sum(1), dim=-1, eps=1e-8
    )
    return target_token, target_descriptor, descriptors, mask


@torch.no_grad()
def _evaluate(model, head, data, device, batch_size: int) -> dict:
    token_cos, descriptor_cos, multiview_cos = [], [], []
    for start in range(0, len(data["radio_features"]), int(batch_size)):
        rows = torch.arange(start, min(start + int(batch_size), len(data["radio_features"])))
        token, descriptor, all_descriptors, teacher_mask = _targets(data, rows)
        predicted = model(
            data["radio_features"][rows].to(device), data["geometry"][rows].to(device),
            token_mask=data["token_mask"][rows].to(device),
            reliability=data["reliability"][rows].to(device),
        )
        projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1)
        token_cos.extend(F.cosine_similarity(predicted.cpu(), token, dim=-1).tolist())
        descriptor_cos.extend(F.cosine_similarity(projected.cpu(), descriptor, dim=-1).tolist())
        pair = torch.einsum("bd,bvd->bv", projected.cpu(), all_descriptors)
        multiview_cos.extend(pair[teacher_mask].tolist())
    return {
        "summary_token_cosine": sum(token_cos) / len(token_cos),
        "mean_descriptor_cosine": sum(descriptor_cos) / len(descriptor_cos),
        "all_view_descriptor_cosine": sum(multiview_cos) / len(multiview_cos),
    }


def train(args: argparse.Namespace) -> dict:
    train_data, train_meta = _load(_paths(args.train_caches), "train")
    val_data, val_meta = _load(_paths(args.validation_caches), "validation")
    overlap = set(train_meta["scenes"]) & set(val_meta["scenes"])
    if overlap:
        raise ValueError(f"train/validation scene leakage: {sorted(overlap)}")
    device = torch.device(args.device)
    model = SurfaceRegionSummaryReadout(hidden_dim=int(args.hidden_dim)).to(device)
    head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint).to(device).eval()
    for parameter in head.parameters(): parameter.requires_grad_(False)
    model.eval()
    baseline = _evaluate(model, head, val_data, device, int(args.batch_size))
    baseline_score = 0.5 * (
        baseline["mean_descriptor_cosine"] + baseline["all_view_descriptor_cosine"]
    )
    print(json.dumps({"untrained_baseline": baseline, "selection_score": baseline_score}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate),
                                  weight_decay=float(args.weight_decay))
    generator = torch.Generator().manual_seed(int(args.seed))
    best_score, best_epoch, best_state, history, stale = -1.0, 0, None, [], 0
    for epoch in range(int(args.epochs)):
        order = torch.randperm(len(train_data["radio_features"]), generator=generator)
        losses = []
        model.train()
        for start in range(0, len(order), int(args.batch_size)):
            rows = order[start:start + int(args.batch_size)]
            target_token, target_descriptor, all_descriptors, teacher_mask = _targets(train_data, rows)
            predicted = model(
                train_data["radio_features"][rows].to(device),
                train_data["geometry"][rows].to(device),
                token_mask=train_data["token_mask"][rows].to(device),
                reliability=train_data["reliability"][rows].to(device),
            )
            projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1)
            target_token, target_descriptor = target_token.to(device), target_descriptor.to(device)
            token_loss = (1 - F.cosine_similarity(predicted, target_token, dim=-1)).mean()
            all_descriptors = all_descriptors.to(device)
            teacher_mask = teacher_mask.to(device)
            all_view_cosine = torch.einsum("bd,bvd->bv", projected, all_descriptors)
            descriptor_loss = (1 - all_view_cosine)[teacher_mask].mean()
            teacher_rel = target_descriptor @ target_descriptor.T
            predicted_rel = projected @ projected.T
            relation_loss = F.smooth_l1_loss(predicted_rel, teacher_rel)
            loss = (float(args.token_weight) * token_loss + descriptor_loss
                    + float(args.relation_weight) * relation_loss)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval(); metrics = _evaluate(model, head, val_data, device, int(args.batch_size))
        score = 0.5 * (metrics["mean_descriptor_cosine"] + metrics["all_view_descriptor_cosine"])
        record = {"epoch": epoch + 1, "loss": sum(losses) / len(losses),
                  "selection_score": score, **metrics}
        history.append(record); print(json.dumps(record), flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if int(args.patience) and stale >= int(args.patience): break
    assert best_state is not None
    model.load_state_dict(best_state)
    architecture = model.architecture()
    provenance = {
        "training_scope": "global_cross_scene_3d_surface", "frozen": True,
        "uses_benchmark_scenes": False, "uses_benchmark_test_vocabulary": False,
        "train": train_meta, "validation": val_meta,
        "scene_disjoint": True, "official_summary_head": "c-radio_v4 siglip2-g",
        "custom_text_projection": False,
    }
    payload = {"schema_version": 2, "architecture": architecture,
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
               "provenance": provenance, "history": history, "best_epoch": best_epoch,
               "best_selection_score": best_score, "untrained_baseline": baseline,
               "untrained_baseline_score": baseline_score, "training_config": vars(args)}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {"output": str(output.resolve()), "checkpoint_sha256": digest,
              "architecture": architecture, "best_epoch": best_epoch,
              "best_selection_score": best_score,
              "untrained_baseline": baseline,
              "selection_score_delta": best_score - baseline_score,
              "validation": _evaluate(model.to(device), head, val_data, device, int(args.batch_size)),
              "train_scenes": len(train_meta["scenes"]),
              "validation_scenes": len(val_meta["scenes"]), "scene_overlap": []}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radio-checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    args = parser.parse_args(); print(json.dumps(train(args), indent=2))


if __name__ == "__main__": main()
