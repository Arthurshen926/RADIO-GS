#!/usr/bin/env python3
"""Train the optional global region aligner on generic crop pairs only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.semantic_alignment import (
    GlobalRegionSummaryBridge,
    GlobalSemanticBridgeManifest,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _metrics(bridge, summary_head, tokens, target_token, target_descriptor, rows, device):
    predicted_token = bridge(tokens[rows].to(device))
    predicted_descriptor = F.normalize(
        summary_head(predicted_token[:, None])[:, 0].float(), dim=-1, eps=1e-8
    )
    predicted_centered = predicted_descriptor - predicted_descriptor.mean(
        dim=0, keepdim=True
    )
    target_centered = target_descriptor[rows].to(device) - target_descriptor[
        rows
    ].to(device).mean(dim=0, keepdim=True)
    return {
        "summary_token_cosine": float(
            F.cosine_similarity(predicted_token.cpu(), target_token[rows], dim=-1).mean()
        ),
        "semantic_descriptor_cosine": float(
            F.cosine_similarity(
                predicted_descriptor.cpu(), target_descriptor[rows], dim=-1
            ).mean()
        ),
        "semantic_descriptor_centered_cosine": float(
            F.cosine_similarity(
                predicted_centered, target_centered, dim=-1, eps=1e-8
            ).mean()
        ),
    }


def train(args: argparse.Namespace) -> dict:
    cache_path = Path(args.training_cache)
    cache = torch.load(cache_path, map_location="cpu")
    required = {
        "radio_region_tokens",
        "official_summary_tokens",
        "official_crop_summaries",
        "metadata",
    }
    if not isinstance(cache, dict) or not required.issubset(cache):
        raise ValueError(f"generic crop cache must contain {sorted(required)}")
    metadata = dict(cache["metadata"])
    if metadata.get("uses_benchmark_test_vocabulary", True):
        raise ValueError("global bridge cache cannot use benchmark test vocabulary")
    if metadata.get("uses_benchmark_scenes", True):
        raise ValueError("global bridge cache cannot use benchmark scenes")
    if metadata.get("training_scope") != "global_cross_scene":
        raise ValueError("global bridge cache must declare global_cross_scene")

    tokens = torch.as_tensor(cache["radio_region_tokens"]).float().cpu()
    target_token = torch.as_tensor(cache["official_summary_tokens"]).float().cpu()
    target_descriptor = F.normalize(
        torch.as_tensor(cache["official_crop_summaries"]).float().cpu(), dim=-1
    )
    if tokens.ndim != 3 or tokens.shape[-1] != 1280:
        raise ValueError("radio_region_tokens must be [N,T,1280]")
    if target_token.shape != (tokens.shape[0], 1280):
        raise ValueError("official_summary_tokens must be [N,1280]")
    if target_descriptor.ndim != 2 or target_descriptor.shape[0] != tokens.shape[0]:
        raise ValueError("official_crop_summaries must align as [N,D]")

    device = torch.device(args.device)
    bridge = GlobalRegionSummaryBridge(
        input_dim=1280,
        output_dim=1280,
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    summary_head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint).to(device)
    summary_head.eval()
    for parameter in summary_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    order = torch.randperm(tokens.shape[0], generator=generator)
    validation_count = max(1, int(round(tokens.shape[0] * float(args.validation_fraction))))
    validation_rows = order[:validation_count]
    training_rows = order[validation_count:]
    if training_rows.numel() == 0:
        raise ValueError("generic crop cache is too small for a train/validation split")
    history = []
    best_descriptor_score = -1.0
    best_state = None
    for epoch in range(int(args.epochs)):
        epoch_rows = training_rows[
            torch.randperm(training_rows.numel(), generator=generator)
        ]
        losses = []
        bridge.train()
        for start in range(0, epoch_rows.numel(), int(args.batch_size)):
            rows = epoch_rows[start : start + int(args.batch_size)]
            inputs = tokens[rows].to(device)
            teacher_token = target_token[rows].to(device)
            teacher_descriptor = target_descriptor[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_token = bridge(inputs)
            predicted_descriptor = F.normalize(
                summary_head(predicted_token[:, None])[:, 0].float(), dim=-1, eps=1e-8
            )
            token_cosine = (
                1.0 - F.cosine_similarity(predicted_token, teacher_token, dim=-1)
            ).mean()
            token_huber = F.huber_loss(predicted_token, teacher_token, delta=0.1)
            semantic_cosine = (1.0 - (predicted_descriptor * teacher_descriptor).sum(dim=-1)).mean()
            predicted_centered = predicted_descriptor - predicted_descriptor.mean(
                dim=0, keepdim=True
            )
            teacher_centered = teacher_descriptor - teacher_descriptor.mean(
                dim=0, keepdim=True
            )
            semantic_centered = (
                1.0
                - F.cosine_similarity(
                    predicted_centered, teacher_centered, dim=-1, eps=1e-8
                )
            ).mean()
            sample_count = min(predicted_descriptor.shape[0], int(args.relation_samples))
            pred_relation = predicted_descriptor[:sample_count] @ predicted_descriptor[:sample_count].T
            teacher_relation = teacher_descriptor[:sample_count] @ teacher_descriptor[:sample_count].T
            relation = F.smooth_l1_loss(pred_relation, teacher_relation)
            loss = (
                float(args.token_cosine_weight) * token_cosine
                + float(args.token_huber_weight) * token_huber
                + float(args.semantic_weight) * semantic_cosine
                + float(args.semantic_centered_weight) * semantic_centered
                + float(args.relation_weight) * relation
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        bridge.eval()
        validation = _metrics(
            bridge,
            summary_head,
            tokens,
            target_token,
            target_descriptor,
            validation_rows,
            device,
        )
        record = {
            "epoch": epoch + 1,
            "loss": sum(losses) / max(1, len(losses)),
            **validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        descriptor_score = 0.5 * (
            validation["semantic_descriptor_cosine"]
            + validation["semantic_descriptor_centered_cosine"]
        )
        record["semantic_descriptor_selection_score"] = descriptor_score
        if descriptor_score > best_descriptor_score:
            best_descriptor_score = descriptor_score
            best_state = {
                key: value.detach().cpu().clone() for key, value in bridge.state_dict().items()
            }
    assert best_state is not None
    bridge.load_state_dict(best_state, strict=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    training_manifest_hash = str(metadata.get("dataset_manifest_sha256", "")) or _sha256(cache_path)
    provisional_manifest = GlobalSemanticBridgeManifest(
        checkpoint_sha256="pending",
        training_scope="global_cross_scene",
        frozen=True,
        uses_benchmark_test_vocabulary=False,
        uses_benchmark_scenes=False,
        training_dataset_manifest_sha256=training_manifest_hash,
    )
    payload = {
        "schema_version": 1,
        "architecture": {
            "input_dim": bridge.input_dim,
            "output_dim": bridge.output_dim,
            "hidden_dim": bridge.hidden_dim,
        },
        "state_dict": bridge.cpu().eval().state_dict(),
        "official_summary_head": {
            "name": "siglip2-g visual summary head",
            "radio_checkpoint_sha256": metadata.get("radio_checkpoint_sha256", ""),
            "custom_text_projection": False,
        },
        "manifest": asdict(provisional_manifest),
        "training_cache": str(cache_path.resolve()),
        "training_cache_metadata": metadata,
        "history": history,
        "best_semantic_descriptor_score": best_descriptor_score,
    }
    torch.save(payload, output)
    checkpoint_hash = _sha256(output)
    manifest = GlobalSemanticBridgeManifest(
        checkpoint_sha256=checkpoint_hash,
        training_scope="global_cross_scene",
        frozen=True,
        uses_benchmark_test_vocabulary=False,
        uses_benchmark_scenes=False,
        training_dataset_manifest_sha256=training_manifest_hash,
    )
    manifest.validate()
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    sidecar.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    report = {
        "output": str(output),
        "manifest": str(sidecar),
        "checkpoint_sha256": checkpoint_hash,
        "best_semantic_descriptor_score": best_descriptor_score,
        "num_pairs": tokens.shape[0],
        "validation_pairs": validation_rows.numel(),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-cosine-weight", type=float, default=0.25)
    parser.add_argument("--token-huber-weight", type=float, default=0.05)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--semantic-centered-weight", type=float, default=1.0)
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument("--relation-samples", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
