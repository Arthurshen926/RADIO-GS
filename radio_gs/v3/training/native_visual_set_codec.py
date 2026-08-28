"""Train the single registered top-K observation-set visual writer."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.native_visual_codec import GatedResidualVisualCodec, _load_dino


class ObservationSetVisualCodec(nn.Module):
    """Frozen-layout pixel codec plus a top-K DeepSets observation writer."""

    def __init__(self, pixel_codec: GatedResidualVisualCodec | None = None) -> None:
        super().__init__()
        self.pixel_codec = pixel_codec or GatedResidualVisualCodec()
        self.phi = nn.Sequential(nn.Linear(321, 320), nn.GELU(), nn.Linear(320, 320))
        self.rho = nn.Sequential(nn.Linear(320, 640), nn.GELU(), nn.Linear(640, 320))
        self.view_confidence = nn.Linear(320, 1)
        nn.init.zeros_(self.phi[-1].weight)
        nn.init.zeros_(self.phi[-1].bias)
        nn.init.zeros_(self.rho[-1].weight)
        nn.init.zeros_(self.rho[-1].bias)
        nn.init.zeros_(self.view_confidence.weight)
        nn.init.zeros_(self.view_confidence.bias)

    def encode_set(
        self, radio: torch.Tensor, dino: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count = weight.shape
        pixel = self.pixel_codec.encode(radio.reshape(-1, 1280), dino.reshape(-1, 768))
        pixel = pixel.reshape(batch, count, 320)
        valid = weight > 0
        normalized = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        log_weight = weight.clamp_min(1e-8).log().clamp_min(-16.0)
        update = self.phi(torch.cat((pixel, log_weight[..., None]), dim=-1))
        pooled = ((pixel + update) * normalized[..., None]).sum(1)
        value = F.normalize(pooled + self.rho(pooled), dim=-1, eps=1e-8)
        confidence = self.view_confidence(value).squeeze(-1)
        confidence = confidence.masked_fill(~valid.any(-1), -torch.inf)
        return value, confidence

    def forward(
        self, radio: torch.Tensor, dino: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_set(radio, dino, weight)


def _topk_sets(record: dict[str, Any], requested: torch.Tensor, top_k: int):
    shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
    gaussian = torch.as_tensor(shard["gaussian_ids"]).long()
    pixel = torch.as_tensor(shard["pixel_ids"]).long()
    weight = torch.as_tensor(shard["base_weights"]).float()
    order = torch.argsort(gaussian, stable=True)
    gaussian, pixel, weight = gaussian[order], pixel[order], weight[order]
    output = {}
    for identity in torch.unique(requested).tolist():
        left = int(torch.searchsorted(gaussian, identity, right=False))
        right = int(torch.searchsorted(gaussian, identity, right=True))
        if right <= left:
            continue
        selected = torch.argsort(weight[left:right], descending=True)[:top_k]
        pixels = pixel[left:right][selected]
        weights = weight[left:right][selected]
        output[int(identity)] = (pixels, weights)
    return output


def _load_scene(membership_path: Path, authority_path: Path, radio_root: Path, dino_root: Path, top_k: int):
    membership = torch.load(membership_path, map_location="cpu")
    authority = torch.load(authority_path, map_location="cpu")
    if authority.get("schema") != "radio_gs.sugm_v3.multisource_correspondence_authority.v1":
        raise ValueError("multisource correspondence authority differs")
    high = torch.as_tensor(authority["high_confidence_pairs"]).long()
    medium = torch.as_tensor(authority["medium_confidence_pairs"]).long()
    pairs = torch.cat((high, medium))
    confidence = torch.cat((torch.ones(high.shape[0]), torch.full((medium.shape[0],), 0.5)))
    negatives = torch.as_tensor(authority["hard_support_disjoint_negatives"]).long()
    negative_map = {(int(row[0]), int(row[1])): row for row in negatives}
    keep = torch.tensor([(int(row[0]), int(row[1])) in negative_map for row in pairs])
    pairs, confidence = pairs[keep], confidence[keep]
    negative = torch.stack([negative_map[(int(row[0]), int(row[1]))] for row in pairs])
    records = {int(r["source_view_index"]): r for r in membership["metadata"]["source_records"]}
    requests: dict[int, list[int]] = {}
    for block in (pairs, negative):
        for row in block:
            requests.setdefault(int(row[0]), []).append(int(row[4]))
            requests.setdefault(int(row[2]), []).append(int(row[5]))
    sets, radio, dino = {}, {}, {}
    for view, identities in requests.items():
        record = records[view]
        sets[view] = _topk_sets(record, torch.tensor(identities), top_k)
        frame = int(record["frame_id"])
        radio_value = torch.load(radio_root / "backbone" / f"rgb_{frame}.pt", map_location="cpu")
        radio[view] = torch.as_tensor(radio_value).half().permute(1, 2, 0).reshape(-1, 1280)
        dino[view] = _load_dino(dino_root / f"frame_{frame:05d}.pt").permute(1, 2, 0).reshape(-1, 768)
    return pairs, negative, confidence, sets, radio, dino, {
        "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
        "authority": {"path": str(authority_path), "sha256": sha256_file(authority_path)},
        "usable_pairs": int(pairs.shape[0]), "high_pairs": int(high.shape[0]),
        "medium_pairs": int(medium.shape[0]),
    }


def _gather(rows, sets, radio, dino, top_k):
    radio_out = torch.zeros(rows.shape[0], top_k, 1280, dtype=torch.float16)
    dino_out = torch.zeros(rows.shape[0], top_k, 768, dtype=torch.float16)
    weight_out = torch.zeros(rows.shape[0], top_k)
    for index, row in enumerate(rows):
        view, identity = int(row[0]), int(row[2])
        value = sets[view].get(identity)
        if value is None:
            continue
        pixels, weights = value
        count = pixels.numel()
        radio_out[index, :count] = radio[view][pixels]
        dino_out[index, :count] = dino[view][pixels]
        weight_out[index, :count] = weights
    return radio_out.float(), dino_out.float(), weight_out


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not (len(args.scene) == len(args.membership) == len(args.authority) == len(args.radio_root) == len(args.dino_root)):
        raise ValueError("set-codec scene inputs differ")
    pair_blocks, negative_blocks, confidence_blocks, scene_data, lineage = [], [], [], [], []
    for scene_index, values in enumerate(zip(args.membership, args.authority, args.radio_root, args.dino_root)):
        loaded = _load_scene(*(Path(v).resolve(strict=True) for v in values), args.top_k)
        pairs, negatives, confidence, sets, radio, dino, info = loaded
        pair_blocks.append(torch.cat((torch.full((pairs.shape[0], 1), scene_index), pairs), dim=1))
        negative_blocks.append(torch.cat((torch.full((negatives.shape[0], 1), scene_index), negatives), dim=1))
        confidence_blocks.append(confidence)
        scene_data.append((sets, radio, dino))
        lineage.append({"scene": args.scene[scene_index], **info})
    pairs, negatives = torch.cat(pair_blocks).long(), torch.cat(negative_blocks).long()
    confidence = torch.cat(confidence_blocks)
    def gather_block(block, side):
        output = []
        for scene_index, data in enumerate(scene_data):
            local = torch.where(block[:, 0] == scene_index)[0]
            if local.numel():
                columns = (1, 2, 5) if side == 0 else (3, 4, 6)
                rows = torch.stack((
                    block[local, columns[0]], block[local, columns[1]],
                    block[local, columns[2]],
                ), 1)
                output.append((local, _gather(rows, *data, args.top_k)))
        result = [torch.zeros(block.shape[0], args.top_k, dim) for dim in (1280, 768)]
        result.append(torch.zeros(block.shape[0], args.top_k))
        for local, values in output:
            for target, value in zip(result, values):
                target[local] = value
        return tuple(result)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(args.preprocess_threads)
    try:
        left_all = gather_block(pairs, 0)
        right_all = gather_block(pairs, 1)
        hard_all = gather_block(negatives, 1)
    finally:
        torch.set_num_threads(previous_threads)
    device_ids = [int(value) for value in args.device_ids.split(",")]
    primary = torch.device(f"cuda:{device_ids[0]}")
    parent_path = Path(args.parent_codec).resolve(strict=True)
    parent = torch.load(parent_path, map_location="cpu")
    torch.manual_seed(args.seed)
    pixel_codec = GatedResidualVisualCodec()
    pixel_codec.load_state_dict(parent["state_dict"], strict=True)
    base_model = ObservationSetVisualCodec(pixel_codec).to(primary)
    model: nn.Module = nn.DataParallel(base_model, device_ids=device_ids) if len(device_ids) > 1 else base_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    for step in range(args.steps):
        chosen = torch.randperm(pairs.shape[0])[: min(args.batch_size, pairs.shape[0])]
        pair_weight = confidence[chosen].to(primary)
        left = tuple(value[chosen].to(primary) for value in left_all)
        right = tuple(value[chosen].to(primary) for value in right_all)
        hard = tuple(value[chosen].to(primary) for value in hard_all)
        optimizer.zero_grad(set_to_none=True)
        left_z, left_confidence = model(*left)
        right_z, right_confidence = model(*right)
        hard_z, _ = model(*hard)
        positive = (left_z * right_z).sum(-1)
        negative_score = (left_z * hard_z).sum(-1)
        margin = (pair_weight * F.relu(args.margin - positive + negative_score)).mean()
        teacher_left = F.normalize(left[0].mean(1), dim=-1)
        teacher_right = F.normalize(right[0].mean(1), dim=-1)
        teacher_left_dino = F.normalize(left[1].mean(1), dim=-1)
        teacher_right_dino = F.normalize(right[1].mean(1), dim=-1)
        teacher_similarity = 0.5 * (
            teacher_left @ teacher_right.T + teacher_left_dino @ teacher_right_dino.T
        )
        teacher_distribution = F.softmax(teacher_similarity / args.teacher_temperature, dim=-1)
        student_log = F.log_softmax(left_z @ right_z.T / args.temperature, dim=-1)
        soft_corr = F.kl_div(student_log, teacher_distribution, reduction="batchmean")
        neighborhood = F.smooth_l1_loss(left_z @ right_z.T, teacher_similarity)
        confidence_loss = F.mse_loss(
            torch.sigmoid(0.5 * (left_confidence + right_confidence)), pair_weight
        )
        radio_target = F.normalize(torch.cat((left[0], right[0]), 0).mean(1), dim=-1)
        dino_target = F.normalize(torch.cat((left[1], right[1]), 0).mean(1), dim=-1)
        embeddings = torch.cat((left_z, right_z))
        reconstruction = (
            1 - F.cosine_similarity(base_model.pixel_codec.radio_decoder(embeddings), radio_target).mean()
            + 1 - F.cosine_similarity(base_model.pixel_codec.dino_decoder(embeddings), dino_target).mean()
        )
        loss = (
            soft_corr + margin + confidence_loss
            + args.neighborhood_weight * neighborhood
            + args.reconstruction_weight * reconstruction
        )
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 0 or (step + 1) % args.snapshot_interval == 0:
            item = {"step": step + 1, "loss": float(loss.detach()), "soft_corr": float(soft_corr.detach()),
                    "hard_margin": float(margin.detach()), "positive": float(positive.mean().detach()),
                    "explicit_negative": float(negative_score.mean().detach()),
                    "confidence": float(confidence_loss.detach()),
                    "reconstruction": float(reconstruction.detach())}
            history.append(item); print(item, flush=True)
    output = Path(args.output).resolve()
    payload = {
        "schema": "radio_gs.sugm_v3.native_observation_set_visual_codec.v1",
        "state_dict": base_model.state_dict(),
        "metadata": {"architecture": "native_radio_dino_gated_residual_top4_deepsets",
                     "source_only": True, "historical_field_opened": False, "target_rgb_opened": False,
                     "benchmark_metrics_opened": False, "top_k": args.top_k,
                     "parent_codec": {"path": str(parent_path), "sha256": sha256_file(parent_path)},
                     "training": {"steps": args.steps, "batch_size": args.batch_size, "seed": args.seed,
                                  "history": history, "scenes": lineage},
                     "objectives": "soft_teacher_distribution+explicit_support_disjoint_margin+neighborhood+dual_reconstruction"},
    }
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("scene", "membership", "authority", "radio_root", "dino_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", action="append", required=True)
    parser.add_argument("--parent-codec", required=True); parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.1); parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--neighborhood-weight", type=float, default=0.25)
    parser.add_argument("--reconstruction-weight", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--snapshot-interval", type=int, default=50); parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--preprocess-threads", type=int, default=1)
    parser.add_argument("--device-ids", default="1,3"); parser.add_argument("--output", required=True)
    args = parser.parse_args(); print(run(args))


if __name__ == "__main__": main()


__all__ = ["ObservationSetVisualCodec"]
