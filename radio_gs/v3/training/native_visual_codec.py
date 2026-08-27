"""One registered RADIO+DINO gated residual visual codec for SUGM-v3.1."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file


class GatedResidualVisualCodec(nn.Module):
    """Fixed v3.1 visual writer: low-rank projections, gate, residual MLP."""

    def __init__(
        self,
        radio_dim: int = 1280,
        dino_dim: int = 768,
        output_dim: int = 320,
        radio_rank: int = 160,
        dino_rank: int = 96,
        hidden_dim: int = 640,
    ) -> None:
        super().__init__()
        self.radio_dim = int(radio_dim)
        self.dino_dim = int(dino_dim)
        self.output_dim = int(output_dim)
        self.radio_norm = nn.LayerNorm(radio_dim, elementwise_affine=False)
        self.dino_norm = nn.LayerNorm(dino_dim, elementwise_affine=False)
        self.radio_down = nn.Linear(radio_dim, radio_rank, bias=False)
        self.radio_up = nn.Linear(radio_rank, output_dim, bias=False)
        self.dino_down = nn.Linear(dino_dim, dino_rank, bias=False)
        self.dino_up = nn.Linear(dino_rank, output_dim, bias=False)
        self.gate = nn.Linear(output_dim * 2, output_dim)
        self.residual = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim)
        )
        self.radio_decoder = nn.Linear(output_dim, radio_dim, bias=False)
        self.dino_decoder = nn.Linear(output_dim, dino_dim, bias=False)

    def encode(self, radio: torch.Tensor, dino: torch.Tensor) -> torch.Tensor:
        radio_value = self.radio_up(self.radio_down(self.radio_norm(radio)))
        dino_value = self.dino_up(self.dino_down(self.dino_norm(dino)))
        gate = torch.sigmoid(self.gate(torch.cat((radio_value, dino_value), dim=-1)))
        fused = gate * radio_value + (1.0 - gate) * dino_value
        return F.normalize(fused + self.residual(fused), dim=-1, eps=1e-8)

    def forward(
        self, radio: torch.Tensor, dino: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encode(radio, dino)
        return embedding, self.radio_decoder(embedding), self.dino_decoder(embedding)


def _load_dino(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if value.get("schema") != "radio_gs.native_dinov2_exact_mpr_teacher.v1":
        raise ValueError("native DINOv2 frame contract differs")
    return torch.as_tensor(value["feature"]).half()


def _best_pixel_per_gaussian(record: Mapping[str, Any], rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
    gaussian_ids = torch.as_tensor(shard["gaussian_ids"]).long()
    pixel_ids = torch.as_tensor(shard["pixel_ids"]).long()
    weights = torch.as_tensor(shard["base_weights"]).float()
    best = torch.full((rows,), -torch.inf)
    best.scatter_reduce_(0, gaussian_ids, weights, reduce="amax", include_self=True)
    selected = weights == best[gaussian_ids]
    selected_ids = gaussian_ids[selected]
    selected_pixels = pixel_ids[selected]
    order = torch.argsort(selected_ids, stable=True)
    selected_ids = selected_ids[order]
    selected_pixels = selected_pixels[order]
    first = torch.ones(selected_ids.numel(), dtype=torch.bool)
    first[1:] = selected_ids[1:] != selected_ids[:-1]
    return selected_ids[first], selected_pixels[first]


def _matched_pairs(
    observations: Sequence[tuple[torch.Tensor, torch.Tensor]],
    view_indices: Sequence[int],
    *,
    pairs_per_view_pair: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = []
    left_views = [index for index, view in enumerate(view_indices) if view % 4 == 1]
    right_views = [index for index, view in enumerate(view_indices) if view % 4 == 2]
    for left_view in left_views:
        left_ids, left_pixels = observations[left_view]
        for right_view in right_views:
            right_ids, right_pixels = observations[right_view]
            position = torch.searchsorted(right_ids, left_ids)
            valid = position < right_ids.numel()
            matched = torch.zeros_like(valid)
            matched[valid] = right_ids[position[valid]] == left_ids[valid]
            indices = torch.where(matched)[0]
            if not indices.numel():
                continue
            indices = indices[
                torch.randperm(indices.numel(), generator=generator)[
                    : min(pairs_per_view_pair, indices.numel())
                ]
            ]
            output.append(torch.stack((
                torch.full_like(indices, left_view),
                left_pixels[indices],
                torch.full_like(indices, right_view),
                right_pixels[position[indices]],
                left_ids[indices],
            ), dim=-1))
    if not output:
        raise ValueError("source-train exact-MPR has no cross-view Gaussian pairs")
    return torch.cat(output)


def _load_scene(
    membership_path: Path,
    radio_root: Path,
    dino_root: Path,
    *,
    pairs_per_view_pair: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, dict[str, Any]]:
    membership = torch.load(membership_path, map_location="cpu")
    records = [
        record for record in membership["metadata"]["source_records"]
        if int(record["source_view_index"]) % 4 in (1, 2)
    ]
    rows = int(membership["num_rows"])
    radio_frames, dino_frames, observations, view_indices = [], [], [], []
    for record in records:
        frame = int(record["frame_id"])
        radio = torch.load(radio_root / "backbone" / f"rgb_{frame}.pt", map_location="cpu")
        radio_frames.append(torch.as_tensor(radio).half().permute(1, 2, 0).reshape(-1, 1280))
        dino = _load_dino(dino_root / f"frame_{frame:05d}.pt")
        dino_frames.append(dino.permute(1, 2, 0).reshape(-1, 768))
        observations.append(_best_pixel_per_gaussian(record, rows))
        view_indices.append(int(record["source_view_index"]))
    pairs = _matched_pairs(
        observations, view_indices, pairs_per_view_pair=pairs_per_view_pair, seed=seed
    )
    return radio_frames, dino_frames, pairs, {
        "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
        "radio_root": str(radio_root),
        "dino_root": str(dino_root),
        "source_train_views": view_indices,
        "cross_view_pairs": int(pairs.shape[0]),
    }


def _gather(
    frames: Sequence[torch.Tensor], scene: torch.Tensor, view: torch.Tensor, pixel: torch.Tensor
) -> torch.Tensor:
    values = []
    for scene_index, local_frames in enumerate(frames):
        selected = torch.where(scene == scene_index)[0]
        if not selected.numel():
            continue
        for view_index in torch.unique(view[selected]).tolist():
            local = selected[view[selected] == view_index]
            values.append((local, local_frames[int(view_index)][pixel[local]]))
    output = torch.empty(scene.numel(), frames[0][0].shape[1], dtype=frames[0][0].dtype)
    for indices, value in values:
        output[indices] = value
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not (
        len(args.scene) == len(args.membership) == len(args.radio_root) == len(args.dino_root)
    ):
        raise ValueError("native visual codec scene inputs differ")
    all_radio, all_dino, pair_blocks, lineage = [], [], [], []
    for scene_index, values in enumerate(
        zip(args.membership, args.radio_root, args.dino_root)
    ):
        membership_path = Path(values[0]).resolve(strict=True)
        radio_root = Path(values[1]).resolve(strict=True)
        dino_root = Path(values[2]).resolve(strict=True)
        radio, dino, pairs, scene_lineage = _load_scene(
            membership_path, radio_root, dino_root,
            pairs_per_view_pair=args.pairs_per_view_pair,
            seed=args.seed + scene_index * 1_000_003,
        )
        all_radio.append(radio)
        all_dino.append(dino)
        pair_blocks.append(torch.cat((
            torch.full((pairs.shape[0], 1), scene_index, dtype=torch.long), pairs
        ), dim=-1))
        lineage.append({"scene": args.scene[scene_index], **scene_lineage})
    pairs = torch.cat(pair_blocks)
    device_ids = [int(value) for value in args.device_ids.split(",")]
    primary = torch.device(f"cuda:{device_ids[0]}")
    torch.manual_seed(args.seed)
    base_model = GatedResidualVisualCodec().to(primary)
    model: nn.Module = (
        nn.DataParallel(base_model, device_ids=device_ids) if len(device_ids) > 1 else base_model
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = random.Random(args.seed)
    history = []
    for step in range(args.steps):
        order = torch.randperm(pairs.shape[0])
        chosen, seen = [], set()
        for index in order.tolist():
            identity = (int(pairs[index, 0]), int(pairs[index, 5]))
            if identity in seen:
                continue
            seen.add(identity)
            chosen.append(index)
            if len(chosen) == args.batch_size:
                break
        batch = pairs[chosen]
        scene = batch[:, 0]
        left_radio = _gather(all_radio, scene, batch[:, 1], batch[:, 2]).float()
        right_radio = _gather(all_radio, scene, batch[:, 3], batch[:, 4]).float()
        left_dino = _gather(all_dino, scene, batch[:, 1], batch[:, 2]).float()
        right_dino = _gather(all_dino, scene, batch[:, 3], batch[:, 4]).float()
        radio = torch.cat((left_radio, right_radio)).to(primary, non_blocking=True)
        dino = torch.cat((left_dino, right_dino)).to(primary, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        embedding, radio_reconstruction, dino_reconstruction = model(radio, dino)
        left_embedding, right_embedding = embedding.chunk(2)
        labels = torch.arange(left_embedding.shape[0], device=primary)
        logits = left_embedding @ right_embedding.T / args.temperature
        correspondence = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
        positive = 1.0 - (left_embedding * right_embedding).sum(-1).mean()
        radio_target = F.normalize(radio, dim=-1, eps=1e-8)
        dino_target = F.normalize(dino, dim=-1, eps=1e-8)
        reconstruction = (
            1.0 - F.cosine_similarity(radio_reconstruction, radio_target, dim=-1).mean()
            + 1.0 - F.cosine_similarity(dino_reconstruction, dino_target, dim=-1).mean()
        )
        teacher_similarity = 0.5 * (
            F.normalize(left_radio.to(primary), dim=-1) @ F.normalize(right_radio.to(primary), dim=-1).T
            + F.normalize(left_dino.to(primary), dim=-1) @ F.normalize(right_dino.to(primary), dim=-1).T
        )
        neighborhood = F.smooth_l1_loss(
            left_embedding @ right_embedding.T, teacher_similarity
        )
        loss = (
            args.correspondence_weight * correspondence
            + args.positive_weight * positive
            + args.reconstruction_weight * reconstruction
            + args.neighborhood_weight * neighborhood
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % args.snapshot_interval == 0:
            value = {
                "step": step + 1, "loss": float(loss.detach()),
                "correspondence": float(correspondence.detach()),
                "positive": float(positive.detach()),
                "reconstruction": float(reconstruction.detach()),
                "neighborhood": float(neighborhood.detach()),
            }
            history.append(value)
            print(value, flush=True)
    output = Path(args.output).resolve()
    payload = {
        "schema": "radio_gs.sugm_v3.native_gated_residual_visual_codec.v1",
        "state_dict": base_model.state_dict(),
        "metadata": {
            "architecture": "layernorm_low_rank_radio_dino_gated_residual_mlp_d320",
            "dimensions": {
                "radio": 1280, "dino": 768, "output": 320,
                "radio_rank": 160, "dino_rank": 96, "hidden": 640,
            },
            "source_only": True,
            "source_train_residues": [1, 2],
            "historical_field_opened": False,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "objectives": {
                "cross_view_correspondence": args.correspondence_weight,
                "positive_pair": args.positive_weight,
                "native_feature_reconstruction": args.reconstruction_weight,
                "teacher_neighborhood": args.neighborhood_weight,
                "temperature": args.temperature,
            },
            "training": {
                "steps": args.steps, "batch_size": args.batch_size,
                "seed": args.seed, "history": history, "scenes": lineage,
            },
        },
    }
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--membership", action="append", required=True)
    parser.add_argument("--radio-root", action="append", required=True)
    parser.add_argument("--dino-root", action="append", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-view-pair", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--correspondence-weight", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=0.5)
    parser.add_argument("--reconstruction-weight", type=float, default=0.25)
    parser.add_argument("--neighborhood-weight", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device-ids", default="0,1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.pairs_per_view_pair, args.snapshot_interval) <= 0:
        raise ValueError("native visual codec budgets must be positive")
    print(run(args))


if __name__ == "__main__":
    main()


__all__ = ["GatedResidualVisualCodec"]
