"""Write the native nonlinear visual codec into a fresh structured D512."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_frozen_json, write_torch_noclobber
from radio_gs.v3.memory.structured_memory import SharedPrivateLayout, StructuredSharedPrivateMemory
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.learned_source_codec import apply_codec
from radio_gs.v3.training.native_visual_codec import GatedResidualVisualCodec, _load_dino


def _encode_frame(
    model: GatedResidualVisualCodec,
    radio_path: Path,
    dino_path: Path,
    device: torch.device,
    chunk: int,
) -> torch.Tensor:
    radio = torch.as_tensor(torch.load(radio_path, map_location="cpu")).float()
    dino = _load_dino(dino_path).float()
    radio = radio.permute(1, 2, 0).reshape(-1, 1280)
    dino = dino.permute(1, 2, 0).reshape(-1, 768)
    if radio.shape[0] != dino.shape[0]:
        raise ValueError("RADIO and DINO native grids differ")
    output = []
    with torch.no_grad():
        for start in range(0, radio.shape[0], chunk):
            stop = min(start + chunk, radio.shape[0])
            output.append(model.encode(
                radio[start:stop].to(device), dino[start:stop].to(device)
            ).cpu())
    return torch.cat(output)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    membership_path = Path(args.membership).resolve(strict=True)
    visual_codec_path = Path(args.visual_codec).resolve(strict=True)
    semantic_codec_path = Path(args.semantic_codec).resolve(strict=True)
    siglip_path = Path(args.siglip_teacher).resolve(strict=True)
    radio_root = Path(args.radio_root).resolve(strict=True)
    dino_root = Path(args.dino_root).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    visual_payload = torch.load(visual_codec_path, map_location="cpu")
    if (
        visual_payload.get("schema")
        != "radio_gs.sugm_v3.native_gated_residual_visual_codec.v1"
        or visual_payload["metadata"].get("source_only") is not True
        or visual_payload["metadata"].get("historical_field_opened") is not False
    ):
        raise ValueError("native visual codec lineage differs")
    device = torch.device(args.device)
    visual_model = GatedResidualVisualCodec().to(device).eval()
    visual_model.load_state_dict(visual_payload["state_dict"], strict=True)
    layout = SharedPrivateLayout()
    rows = int(membership["num_rows"])
    records = [
        record for record in membership["metadata"]["source_records"]
        if int(record["source_view_index"]) % 4 in (1, 2)
    ]
    shared_sum = torch.zeros(rows, layout.shared)
    shared_mass = torch.zeros(rows)
    for record in records:
        frame = int(record["frame_id"])
        encoded = _encode_frame(
            visual_model,
            radio_root / "backbone" / f"rgb_{frame}.pt",
            dino_root / f"frame_{frame:05d}.pt",
            device,
            args.pixel_chunk,
        )
        shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
        gaussian_ids = torch.as_tensor(shard["gaussian_ids"]).long()
        pixel_ids = torch.as_tensor(shard["pixel_ids"]).long()
        weights = torch.as_tensor(shard["base_weights"]).float()
        for start in range(0, gaussian_ids.numel(), args.hit_chunk):
            stop = min(start + args.hit_chunk, gaussian_ids.numel())
            ids = gaussian_ids[start:stop]
            weight = weights[start:stop]
            shared_sum.index_add_(
                0, ids, encoded[pixel_ids[start:stop]] * weight[:, None]
            )
            shared_mass.index_add_(0, ids, weight)
    observed_shared = shared_mass > 0
    shared = shared_sum / shared_mass.clamp_min(1e-8)[:, None]
    shared[observed_shared] = F.normalize(shared[observed_shared], dim=-1, eps=1e-8)

    semantic_payload = torch.load(semantic_codec_path, map_location="cpu")
    semantic_state = semantic_payload["state_dict"]
    siglip = torch.load(siglip_path, map_location="cpu")
    descriptors = torch.as_tensor(siglip["descriptors"]).float()
    proposal_semantic = apply_codec(
        descriptors,
        torch.as_tensor(semantic_state["siglip_mean"]).float(),
        torch.as_tensor(semantic_state["siglip_basis"]).float(),
    )
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    train_proposals = (proposal_views % 4 == 1) | (proposal_views % 4 == 2)
    proposal_indices = torch.as_tensor(membership["proposal_indices"]).long()
    selected = train_proposals[proposal_indices]
    membership_rows = torch.as_tensor(membership["row_indices"]).long()[selected]
    proposal_indices = proposal_indices[selected]
    membership_weights = torch.as_tensor(membership["weights"]).float()[selected]
    semantic_sum = torch.zeros(rows, layout.semantic)
    semantic_mass = torch.zeros(rows)
    semantic_sum.index_add_(
        0, membership_rows,
        proposal_semantic[proposal_indices] * membership_weights[:, None],
    )
    semantic_mass.index_add_(0, membership_rows, membership_weights)
    observed_semantic = semantic_mass > 0
    semantic = semantic_sum / semantic_mass.clamp_min(1e-8)[:, None]
    semantic[observed_semantic] = F.normalize(
        semantic[observed_semantic], dim=-1, eps=1e-8
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    instance = torch.randn(rows, layout.instance, generator=generator) / math.sqrt(layout.instance)
    boundary = torch.zeros(rows, layout.boundary)
    memory = torch.cat((shared, semantic, instance, boundary), dim=-1)
    structured = StructuredSharedPrivateMemory(memory, layout=layout)
    state_dict = dict(structured.state_dict())
    state_dict.update({
        f"visual_codec.{name}": value.cpu()
        for name, value in visual_model.state_dict().items()
    })
    state_dict.update({
        "codec.siglip_mean": semantic_state["siglip_mean"],
        "codec.siglip_basis": semantic_state["siglip_basis"],
    })
    output = Path(args.output).resolve()
    payload = {
        "schema": "radio_gs.sugm_v3.structured_source_mapping.v1",
        "state_dict": state_dict,
        "metadata": {
            "state_representation": "single_structured_shared_private_d512_plus_global_heads",
            "architecture": structured.architecture,
            "layout": dict(layout.__dict__),
            "partition_owned_writes": True,
            "phase_order": "native_visual_semantic_initialization_before_private_training",
            "cross_block_bridges_enabled": False,
            "gaussian_indexed_sidecars": 0,
            "historical_field_opened": False,
            "source_only": True,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "view_split": {"train_residues": [1, 2], "dev_residue": 3, "audit_residue": 0},
            "initialization": {
                "source_train_residues": [1, 2],
                "historical_field_opened": False,
                "radio_projection": {"type": "native_gated_residual_radio_dino", "dim": 320},
                "siglip_projection": {"type": "learned_cross_scene_pca", "dim": 128},
                "shared_observed_rows": int(observed_shared.sum()),
                "semantic_observed_rows": int(observed_semantic.sum()),
                "num_rows": rows,
                "visual_codec": {"path": str(visual_codec_path), "sha256": sha256_file(visual_codec_path)},
                "semantic_codec": {"path": str(semantic_codec_path), "sha256": sha256_file(semantic_codec_path)},
            },
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "siglip_teacher": {"path": str(siglip_path), "sha256": sha256_file(siglip_path)},
            "radio_teacher_root": str(radio_root),
            "dino_teacher_root": str(dino_root),
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema": "radio_gs.sugm_v3.native_visual_initialization.report.v1",
        "status": "visual_semantic_gate_only_private_training_not_opened",
        "checkpoint": {"path": str(output), "sha256": sha256_file(output)},
        "historical_field_opened": False,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--radio-root", required=True)
    parser.add_argument("--dino-root", required=True)
    parser.add_argument("--visual-codec", required=True)
    parser.add_argument("--semantic-codec", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pixel-chunk", type=int, default=1024)
    parser.add_argument("--hit-chunk", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
