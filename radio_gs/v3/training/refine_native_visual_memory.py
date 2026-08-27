"""Renderer-aware source-train refinement of the SUGM-v3.1 visual block."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_frozen_json, write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.native_visual_codec import GatedResidualVisualCodec, _load_dino


@torch.no_grad()
def _encoded_teacher(model, radio_path, dino_path, device, chunk):
    radio = torch.as_tensor(torch.load(radio_path, map_location="cpu")).float()
    dino = _load_dino(dino_path).float()
    radio = radio.permute(1, 2, 0).reshape(-1, 1280)
    dino = dino.permute(1, 2, 0).reshape(-1, 768)
    if radio.shape[0] != dino.shape[0]:
        raise ValueError("RADIO and DINO native grids differ")
    parts = []
    for start in range(0, radio.shape[0], chunk):
        stop = min(start + chunk, radio.shape[0])
        parts.append(model.encode(
            radio[start:stop].to(device), dino[start:stop].to(device)
        ).cpu().half())
    return torch.cat(parts)


def _episode(record, target, *, pixels_per_view: int, seed: int):
    shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
    pixel_ids = torch.as_tensor(shard["pixel_ids"]).long()
    gaussian_ids = torch.as_tensor(shard["gaussian_ids"]).long()
    weights = torch.as_tensor(shard["base_weights"]).float()
    available = torch.unique_consecutive(pixel_ids)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = min(int(pixels_per_view), int(available.numel()))
    chosen = available[
        torch.randperm(available.numel(), generator=generator)[:count]
    ].sort().values
    left = torch.searchsorted(pixel_ids, chosen, right=False)
    right = torch.searchsorted(pixel_ids, chosen, right=True)
    hit_rows, hit_pixels, hit_weights = [], [], []
    for local, (start, stop) in enumerate(zip(left.tolist(), right.tolist())):
        if stop <= start:
            continue
        hit_rows.append(gaussian_ids[start:stop])
        hit_pixels.append(torch.full((stop - start,), local, dtype=torch.long))
        hit_weights.append(weights[start:stop])
    if not hit_rows:
        raise ValueError("source-train render episode has no exact compositor hits")
    rows = torch.cat(hit_rows)
    unique, inverse = torch.unique(rows, sorted=True, return_inverse=True)
    return {
        "rows": unique,
        "inverse": inverse,
        "pixels": torch.cat(hit_pixels),
        "weights": torch.cat(hit_weights).half(),
        "target": target[chosen].half(),
        "num_pixels": count,
        "source_view_index": int(record["source_view_index"]),
    }


def _render_loss(visual, initial, episode, *, temperature: float, anchor_weight: float):
    device = visual.device
    rows = episode["rows"].to(device)
    features = F.normalize(visual[rows], dim=-1, eps=1e-8)
    rendered = torch.zeros(int(episode["num_pixels"]), features.shape[1], device=device)
    rendered.index_add_(
        0, episode["pixels"].to(device),
        features[episode["inverse"].to(device)]
        * episode["weights"].to(device).float()[:, None],
    )
    rendered = F.normalize(rendered, dim=-1, eps=1e-8)
    target = F.normalize(episode["target"].to(device).float(), dim=-1, eps=1e-8)
    cosine = 1.0 - (rendered * target).sum(-1).mean()
    labels = torch.arange(target.shape[0], device=device)
    logits = rendered @ target.T / float(temperature)
    correspondence = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
    anchor = 1.0 - (
        features * F.normalize(initial[rows], dim=-1, eps=1e-8)
    ).sum(-1).mean()
    return cosine + correspondence + float(anchor_weight) * anchor, cosine, correspondence, anchor


def run(args: argparse.Namespace) -> dict[str, Any]:
    membership_path = Path(args.membership).resolve(strict=True)
    candidate_path = Path(args.candidate).resolve(strict=True)
    radio_root = Path(args.radio_root).resolve(strict=True)
    dino_root = Path(args.dino_root).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    candidate = torch.load(candidate_path, map_location="cpu")
    metadata = candidate["metadata"]
    source_metadata = membership["metadata"]
    if (
        metadata.get("historical_field_opened") is not False
        or metadata.get("source_only") is not True
        or metadata.get("gaussian_indexed_sidecars") != 0
        or metadata["membership"]["sha256"] != sha256_file(membership_path)
        or source_metadata.get("benchmark_images_opened") is not False
        or source_metadata.get("benchmark_masks_opened") is not False
        or source_metadata.get("evaluation_rgb_opened") is not False
        or int(source_metadata.get("source_view_count", -1)) != 32
        or Path(metadata["radio_teacher_root"]).resolve(strict=True) != radio_root
        or Path(metadata["dino_teacher_root"]).resolve(strict=True) != dino_root
        or metadata["initialization"]["radio_projection"]["type"]
        != "native_gated_residual_radio_dino"
    ):
        raise ValueError("renderer refinement requires a fresh native source-only D512")
    device = torch.device(args.device)
    model = GatedResidualVisualCodec().to(device).eval()
    model.load_state_dict({
        name.removeprefix("visual_codec."): torch.as_tensor(value)
        for name, value in candidate["state_dict"].items()
        if name.startswith("visual_codec.")
    }, strict=True)
    model.requires_grad_(False)
    records = [
        record for record in membership["metadata"]["source_records"]
        if int(record["source_view_index"]) % 4 in (1, 2)
    ]
    episodes = []
    for record in records:
        frame = int(record["frame_id"])
        target = _encoded_teacher(
            model, radio_root / "backbone" / f"rgb_{frame}.pt",
            dino_root / f"frame_{frame:05d}.pt", device, args.teacher_pixel_chunk,
        )
        episodes.append(_episode(
            record, target, pixels_per_view=args.pixels_per_view,
            seed=args.seed + int(record["source_view_index"]),
        ))
    memory = torch.as_tensor(candidate["state_dict"]["memory"]).float()
    shared_dim = int(metadata["layout"]["shared"])
    initial = memory[:, :shared_dim].to(device)
    visual = nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([visual], lr=args.learning_rate, weight_decay=0.0)
    rng = random.Random(args.seed)
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        selected = rng.sample(episodes, k=min(args.views_per_step, len(episodes)))
        values = [
            _render_loss(
                visual, initial, episode, temperature=args.temperature,
                anchor_weight=args.anchor_weight,
            )
            for episode in selected
        ]
        loss = torch.stack([value[0] for value in values]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([visual], args.max_grad_norm)
        optimizer.step()
        if step == 0 or (step + 1) % args.snapshot_interval == 0:
            item = {
                "step": step + 1, "loss": float(loss.detach()),
                "render_cosine_loss": float(torch.stack([v[1] for v in values]).mean().detach()),
                "correspondence": float(torch.stack([v[2] for v in values]).mean().detach()),
                "initial_anchor": float(torch.stack([v[3] for v in values]).mean().detach()),
            }
            history.append(item)
            print(item, flush=True)
    refined = visual.detach().cpu()
    observed = initial.detach().cpu().norm(dim=-1) > 0
    refined[observed] = F.normalize(refined[observed], dim=-1, eps=1e-8)
    output_memory = memory.clone()
    output_memory[:, :shared_dim] = refined
    output_state = dict(candidate["state_dict"])
    output_state["memory"] = output_memory
    payload = {
        **candidate,
        "state_dict": output_state,
        "metadata": {
            **metadata,
            "phase_order": "native_visual_semantic_render_refinement_before_private_training",
            "render_refinement": {
                "type": "source_train_exact_compositor_visual_only",
                "steps": args.steps, "pixels_per_view": args.pixels_per_view,
                "views_per_step": args.views_per_step, "temperature": args.temperature,
                "anchor_weight": args.anchor_weight, "seed": args.seed,
                "source_train_views": [int(r["source_view_index"]) for r in records],
                "history": history,
                "parent": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    report = {
        "schema": "radio_gs.sugm_v3.native_visual_render_refinement.report.v1",
        "status": "visual_semantic_gate_only_private_training_not_opened",
        "checkpoint": {"path": str(output), "sha256": sha256_file(output)},
        "parent": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        "history": history,
        "protected_block_max_abs_delta": {
            "semantic_instance_boundary": float(
                (output_memory[:, shared_dim:] - memory[:, shared_dim:]).abs().max()
            ),
        },
        "historical_field_opened": False,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--radio-root", required=True)
    parser.add_argument("--dino-root", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--pixels-per-view", type=int, default=512)
    parser.add_argument("--views-per-step", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--teacher-pixel-chunk", type=int, default=1024)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(
        args.steps, args.pixels_per_view, args.views_per_step,
        args.teacher_pixel_chunk, args.snapshot_interval,
    ) <= 0:
        raise ValueError("native visual render-refinement budgets must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
