"""Materialize the learned top-K observation-set writer into one structured D512."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_frozen_json, write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.native_visual_codec import _load_dino
from radio_gs.v3.training.native_visual_set_codec import ObservationSetVisualCodec


def _topk_all(shard: dict[str, Any], rows: int, top_k: int):
    gaussian = torch.as_tensor(shard["gaussian_ids"]).long()
    pixel = torch.as_tensor(shard["pixel_ids"]).long()
    weight = torch.as_tensor(shard["base_weights"]).float()
    weight_order = torch.argsort(weight, descending=True, stable=True)
    gaussian_order = torch.argsort(gaussian[weight_order], stable=True)
    order = weight_order[gaussian_order]
    gaussian, pixel, weight = gaussian[order], pixel[order], weight[order]
    first = torch.ones(gaussian.numel(), dtype=torch.bool)
    first[1:] = gaussian[1:] != gaussian[:-1]
    starts = torch.where(first)[0]
    group = first.cumsum(0) - 1
    rank = torch.arange(gaussian.numel()) - starts[group]
    selected = rank < top_k
    unique = gaussian[starts]
    pixels = torch.zeros(unique.numel(), top_k, dtype=torch.long)
    weights = torch.zeros(unique.numel(), top_k)
    pixels[group[selected], rank[selected]] = pixel[selected]
    weights[group[selected], rank[selected]] = weight[selected]
    if unique.numel() and (int(unique.min()) < 0 or int(unique.max()) >= rows):
        raise ValueError("set observation Gaussian index differs")
    return unique, pixels, weights


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    membership_path = Path(args.membership).resolve(strict=True)
    parent_path = Path(args.parent_candidate).resolve(strict=True)
    codec_path = Path(args.set_codec).resolve(strict=True)
    radio_root = Path(args.radio_root).resolve(strict=True)
    dino_root = Path(args.dino_root).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    parent = torch.load(parent_path, map_location="cpu")
    codec_payload = torch.load(codec_path, map_location="cpu")
    if (
        codec_payload.get("schema") != "radio_gs.sugm_v3.native_observation_set_visual_codec.v1"
        or parent["metadata"].get("source_only") is not True
        or parent["metadata"]["membership"]["sha256"] != sha256_file(membership_path)
    ):
        raise ValueError("set materialization lineage differs")
    device = torch.device(args.device)
    model = ObservationSetVisualCodec().to(device).eval()
    model.load_state_dict(codec_payload["state_dict"], strict=True)
    rows = int(membership["num_rows"])
    top_k = int(codec_payload["metadata"]["top_k"])
    shared_sum = torch.zeros(rows, 320)
    confidence_mass = torch.zeros(rows)
    records = [
        record for record in membership["metadata"]["source_records"]
        if int(record["source_view_index"]) % 4 in (1, 2)
    ]
    for record in records:
        frame = int(record["frame_id"])
        radio_value = torch.load(radio_root / "backbone" / f"rgb_{frame}.pt", map_location="cpu")
        radio = torch.as_tensor(radio_value).float().permute(1, 2, 0).reshape(-1, 1280)
        dino = _load_dino(dino_root / f"frame_{frame:05d}.pt").float().permute(1, 2, 0).reshape(-1, 768)
        shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
        identities, pixels, weights = _topk_all(shard, rows, top_k)
        for start in range(0, identities.numel(), args.set_chunk):
            stop = min(start + args.set_chunk, identities.numel())
            local_pixels = pixels[start:stop]
            embedding, confidence = model(
                radio[local_pixels].to(device), dino[local_pixels].to(device),
                weights[start:stop].to(device),
            )
            view_weight = torch.sigmoid(confidence).cpu()
            shared_sum.index_add_(0, identities[start:stop], embedding.cpu() * view_weight[:, None])
            confidence_mass.index_add_(0, identities[start:stop], view_weight)
    observed = confidence_mass > 0
    shared = shared_sum / confidence_mass.clamp_min(1e-8)[:, None]
    shared[observed] = F.normalize(shared[observed], dim=-1, eps=1e-8)
    memory = torch.as_tensor(parent["state_dict"]["memory"]).clone()
    memory[:, :320] = shared
    state_dict = dict(parent["state_dict"])
    state_dict["memory"] = memory
    state_dict.update({f"set_visual_codec.{name}": value.cpu() for name, value in model.state_dict().items()})
    payload = {
        **parent, "state_dict": state_dict,
        "metadata": {
            **parent["metadata"],
            "phase_order": "native_observation_set_visual_semantic_before_private_training",
            "initialization": {
                **parent["metadata"]["initialization"],
                "radio_projection": {"type": "native_observation_set_radio_dino", "dim": 320, "top_k": top_k},
                "set_visual_codec": {"path": str(codec_path), "sha256": sha256_file(codec_path)},
                "shared_observed_rows": int(observed.sum()),
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    report = {
        "schema": "radio_gs.sugm_v3.native_visual_set_materialization.report.v1",
        "status": "visual_semantic_gate_only_private_training_not_opened",
        "checkpoint": {"path": str(output), "sha256": sha256_file(output)},
        "protected_semantic_instance_boundary_max_abs_delta": float((memory[:, 320:] - parent["state_dict"]["memory"][:, 320:]).abs().max()),
        "historical_field_opened": False, "target_rgb_opened": False, "benchmark_metrics_opened": False,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True); parser.add_argument("--parent-candidate", required=True)
    parser.add_argument("--set-codec", required=True); parser.add_argument("--radio-root", required=True)
    parser.add_argument("--dino-root", required=True); parser.add_argument("--set-chunk", type=int, default=512)
    parser.add_argument("--device", default="cuda:1"); parser.add_argument("--output", required=True)
    args = parser.parse_args(); print(run(args))


if __name__ == "__main__": main()
