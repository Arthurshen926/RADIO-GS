"""Fit a bounded cross-scene render metric on top of the source PCA codec."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.learned_source_codec import apply_codec


def _episode(
    membership: dict[str, Any],
    shared: torch.Tensor,
    teacher_root: Path,
    record: dict[str, Any],
    mean: torch.Tensor,
    basis: torch.Tensor,
    *,
    pixels_per_view: int,
    seed: int,
) -> dict[str, torch.Tensor | int]:
    shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
    pixel_ids = torch.as_tensor(shard["pixel_ids"]).long()
    gaussian_ids = torch.as_tensor(shard["gaussian_ids"]).long()
    weights = torch.as_tensor(shard["base_weights"]).float()
    available = torch.unique_consecutive(pixel_ids)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = min(int(pixels_per_view), available.numel())
    chosen = available[torch.randperm(available.numel(), generator=generator)[:count]].sort().values
    left = torch.searchsorted(pixel_ids, chosen, right=False)
    right = torch.searchsorted(pixel_ids, chosen, right=True)
    hit_rows, hit_pixels, hit_weights = [], [], []
    for local, (start, stop) in enumerate(zip(left.tolist(), right.tolist())):
        if stop <= start:
            continue
        hit_rows.append(gaussian_ids[start:stop])
        hit_pixels.append(torch.full((stop - start,), local, dtype=torch.long))
        hit_weights.append(weights[start:stop])
    rows = torch.cat(hit_rows)
    unique, inverse = torch.unique(rows, sorted=True, return_inverse=True)
    teacher = torch.load(
        teacher_root / "backbone" / f"rgb_{int(record['frame_id'])}.pt",
        map_location="cpu",
    ).float()
    flat = teacher.permute(1, 2, 0).reshape(-1, teacher.shape[0])
    target = apply_codec(flat[chosen], mean, basis)
    return {
        "features": shared[unique].half(),
        "inverse": inverse,
        "pixel_ids": torch.cat(hit_pixels),
        "weights": torch.cat(hit_weights).half(),
        "target": target.half(),
        "num_pixels": count,
        "source_view_index": int(record["source_view_index"]),
    }


def _loss(episode: dict[str, Any], log_scale: torch.Tensor, temperature: float) -> torch.Tensor:
    device = log_scale.device
    scale = log_scale.exp()
    features = F.normalize(
        episode["features"].to(device).float() * scale, dim=-1, eps=1e-8
    )
    rendered = torch.zeros(
        int(episode["num_pixels"]), features.shape[1], device=device
    )
    rendered.index_add_(
        0,
        episode["pixel_ids"].to(device),
        features[episode["inverse"].to(device)]
        * episode["weights"].to(device).float()[:, None],
    )
    rendered = F.normalize(rendered, dim=-1, eps=1e-8)
    target = F.normalize(
        episode["target"].to(device).float() * scale, dim=-1, eps=1e-8
    )
    cosine = 1.0 - (rendered * target).sum(-1).mean()
    labels = torch.arange(target.shape[0], device=device)
    correspondence = F.cross_entropy(rendered @ target.T / float(temperature), labels)
    return cosine + correspondence


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not (
        len(args.scene) == len(args.membership) == len(args.candidate)
        == len(args.teacher_root)
    ):
        raise ValueError("render-metric scene inputs differ")
    parent_path = Path(args.codec).resolve(strict=True)
    parent = torch.load(parent_path, map_location="cpu")
    state = parent["state_dict"]
    mean = torch.as_tensor(state["radio_mean"]).float()
    basis = torch.as_tensor(state["radio_basis"]).float()
    episodes = []
    input_lineage = []
    for scene_index, (scene, membership_value, candidate_value, root_value) in enumerate(
        zip(args.scene, args.membership, args.candidate, args.teacher_root)
    ):
        membership_path = Path(membership_value).resolve(strict=True)
        candidate_path = Path(candidate_value).resolve(strict=True)
        root = Path(root_value).resolve(strict=True)
        membership = torch.load(membership_path, map_location="cpu")
        candidate = torch.load(candidate_path, map_location="cpu")
        if candidate["metadata"]["initialization"]["codec"]["sha256"] != sha256_file(parent_path):
            raise ValueError("render-metric candidate was not initialized by parent codec")
        shared_dim = int(candidate["metadata"]["layout"]["shared"])
        shared = torch.as_tensor(candidate["state_dict"]["memory"][:, :shared_dim]).float()
        records = [
            record for record in membership["metadata"]["source_records"]
            if int(record["source_view_index"]) % 4 in (1, 2)
        ]
        for record in records:
            episodes.append(_episode(
                membership, shared, root, record, mean, basis,
                pixels_per_view=args.pixels_per_view,
                seed=args.seed + scene_index * 1_000_003 + int(record["source_view_index"]),
            ))
        input_lineage.append({
            "scene": scene,
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "teacher_root": str(root),
            "source_train_views": [int(record["source_view_index"]) for record in records],
        })
    device = torch.device(args.device)
    log_scale = torch.zeros(basis.shape[1], device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([log_scale], lr=args.learning_rate, weight_decay=0.0)
    rng = random.Random(args.seed)
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        selected = rng.sample(episodes, k=min(args.views_per_step, len(episodes)))
        fit_loss = torch.stack([
            _loss(episode, log_scale, args.temperature) for episode in selected
        ]).mean()
        regularizer = log_scale.square().mean()
        loss = fit_loss + float(args.regularization) * regularizer
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_scale.sub_(log_scale.mean()).clamp_(-args.max_log_scale, args.max_log_scale)
        if step == 0 or (step + 1) % args.snapshot_interval == 0:
            history.append({
                "step": step + 1,
                "loss": float(loss.detach()),
                "fit_loss": float(fit_loss.detach()),
                "scale_min": float(log_scale.exp().min()),
                "scale_max": float(log_scale.exp().max()),
            })
            print(history[-1], flush=True)
    scale = log_scale.detach().exp().cpu()
    output_payload = {
        "schema": parent["schema"],
        "state_dict": {
            **state,
            "radio_basis": basis * scale,
            "radio_render_metric_scale": scale,
        },
        "metadata": {
            **parent["metadata"],
            "type": "cross_scene_source_train_pca_render_metric_exact_mpr",
            "parent_codec": {"path": str(parent_path), "sha256": sha256_file(parent_path)},
            "render_metric": {
                "parameterization": "bounded_diagonal_positive_global_d320",
                "steps": args.steps,
                "pixels_per_view": args.pixels_per_view,
                "views_per_step": args.views_per_step,
                "temperature": args.temperature,
                "regularization": args.regularization,
                "max_log_scale": args.max_log_scale,
                "seed": args.seed,
                "objective": "render_cosine_plus_in_view_correspondence_cross_entropy",
                "history": history,
                "inputs": input_lineage,
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, output_payload)
    return {"output": str(output), "sha256": sha256_file(output), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--membership", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--teacher-root", action="append", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--pixels-per-view", type=int, default=64)
    parser.add_argument("--views-per-step", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--max-log-scale", type=float, default=0.7)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.steps, args.pixels_per_view, args.views_per_step, args.snapshot_interval) <= 0:
        raise ValueError("render-metric budgets must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
