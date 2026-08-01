#!/usr/bin/env python3
"""Measure Surface and text-response gradient scales at a frozen warm start."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_independent_normalized_cosine_response_smooth_l1_loss,
    compute_scene_wise_text_response_profile_ranking_loss,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.train_surface_region_text_response_distill import (
    _descriptor_loss,
    _load,
    _targets,
    load_fit_text_embedding_bank,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_surface_region_summary_readout_v2,
    write_frozen_json,
)


def _gradient_norm(loss: torch.Tensor, parameters, *, retain_graph: bool) -> float:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squares = [
        value.detach().float().square().sum()
        for value in gradients
        if value is not None
    ]
    if not squares:
        raise ValueError("diagnostic loss is disconnected from the readout")
    result = float(torch.stack(squares).sum().sqrt().cpu())
    if not math.isfinite(result) or result <= 0:
        raise ValueError("diagnostic gradient norm must be finite and positive")
    return result


def diagnose(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    cache_paths = [Path(value).resolve() for value in args.train_cache]
    data, metadata = _load(cache_paths, "train")
    checkpoint = Path(args.surface_control_checkpoint).resolve()
    model, payload, checkpoint_sha, _ = load_surface_region_summary_readout_v2(
        checkpoint,
        expected_sha256=args.surface_control_checkpoint_sha256,
        map_location="cpu",
    )
    if checkpoint_sha != args.surface_control_checkpoint_sha256:
        raise ValueError("Surface control SHA256 differs")
    if payload.get("provenance", {}).get("train") != metadata:
        raise ValueError("Surface control binds different training caches")
    model = model.to(device).train().requires_grad_(True)
    radio = Path(args.radio_checkpoint).resolve()
    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio)).to(device).eval()
    head.requires_grad_(False)
    fit = load_fit_text_embedding_bank(
        Path(args.fit_text_bank), Path(args.fit_text_bank_manifest)
    )
    text = fit["embeddings"].to(device)

    scene_ids = data.get("scene_ids")
    if not isinstance(scene_ids, list) or len(scene_ids) != len(data["radio_features"]):
        raise ValueError("training caches lack exact scene row identities")
    selected_scenes = sorted(set(scene_ids))[: int(args.scenes)]
    if len(selected_scenes) != int(args.scenes):
        raise ValueError("not enough complete scenes for gradient diagnostic")
    rows = torch.tensor(
        [index for index, scene in enumerate(scene_ids) if scene in selected_scenes],
        dtype=torch.long,
    )
    selected_ids = [scene_ids[index] for index in rows.tolist()]
    target_token, target_descriptor, all_descriptors, teacher_mask = _targets(
        data, rows
    )
    predicted = model(
        data["radio_features"][rows].to(device),
        data["geometry"][rows].to(device),
        anchor_index=data["anchor_index"][rows].to(device),
        token_mask=data["token_mask"][rows].to(device),
        reliability=data["reliability"][rows].to(device),
    )
    projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1, eps=1e-8)
    target_token = target_token.to(device)
    target_descriptor = target_descriptor.to(device)
    all_descriptors = all_descriptors.to(device)
    teacher_mask = teacher_mask.to(device)
    token_loss = (
        1.0 - F.cosine_similarity(predicted, target_token, dim=-1)
    ).mean()
    descriptor_loss = _descriptor_loss(projected, all_descriptors, teacher_mask)
    relation_loss = F.smooth_l1_loss(
        projected @ projected.T,
        target_descriptor @ target_descriptor.T,
    )
    surface_loss = 0.25 * token_loss + descriptor_loss + 0.1 * relation_loss
    independent_loss = (
        compute_independent_normalized_cosine_response_smooth_l1_loss(
            projected, target_descriptor, text
        )
    )
    scene_loss, scene_stats = compute_scene_wise_text_response_profile_ranking_loss(
        projected,
        target_descriptor,
        text,
        selected_ids,
        ranking_temperature=float(args.ranking_temperature),
    )
    parameters = tuple(model.parameters())
    surface_norm = _gradient_norm(surface_loss, parameters, retain_graph=True)
    independent_norm = _gradient_norm(
        independent_loss, parameters, retain_graph=True
    )
    scene_norm = _gradient_norm(scene_loss, parameters, retain_graph=False)
    result = {
        "schema_version": 1,
        "artifact_type": "warmstart_surface_text_response_gradient_diagnostic",
        "device": str(device),
        "rows": int(len(rows)),
        "scenes": selected_scenes,
        "ranking_temperature": float(args.ranking_temperature),
        "losses": {
            "surface": float(surface_loss.detach().cpu()),
            "token": float(token_loss.detach().cpu()),
            "descriptor": float(descriptor_loss.detach().cpu()),
            "relation": float(relation_loss.detach().cpu()),
            "independent_response": float(independent_loss.detach().cpu()),
            "scene_response": float(scene_loss.detach().cpu()),
            "scene_profile": float(scene_stats["profile_loss"].cpu()),
            "scene_ranking": float(scene_stats["ranking_loss"].cpu()),
        },
        "gradient_l2": {
            "surface": surface_norm,
            "independent_response": independent_norm,
            "scene_response": scene_norm,
        },
        "equal_surface_gradient_lambdas": {
            "independent_response": surface_norm / independent_norm,
            "scene_response": surface_norm / scene_norm,
        },
        "bindings": {
            "surface_control": file_record(checkpoint),
            "radio_checkpoint": file_record(radio),
            "train_caches": [file_record(path) for path in cache_paths],
            "fit_text_bank": file_record(Path(args.fit_text_bank)),
            "fit_text_bank_manifest": file_record(Path(args.fit_text_bank_manifest)),
            "implementation": file_record(Path(__file__).resolve()),
        },
    }
    write_frozen_json(Path(args.output), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", action="append", required=True)
    parser.add_argument("--surface-control-checkpoint", required=True)
    parser.add_argument("--surface-control-checkpoint-sha256", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--fit-text-bank", required=True)
    parser.add_argument("--fit-text-bank-manifest", required=True)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--ranking-temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    result = diagnose(parser.parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
