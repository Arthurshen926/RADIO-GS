"""Materialize a fresh codec-initialized D512 before private-branch training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from radio_gs.utils.immutable_artifacts import write_frozen_json, write_torch_noclobber
from radio_gs.v3.memory.structured_memory import SharedPrivateLayout, StructuredSharedPrivateMemory
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.structured_initialization import initialize_structured_memory


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    membership_path = Path(args.membership).resolve(strict=True)
    siglip_path = Path(args.siglip_teacher).resolve(strict=True)
    codec_path = Path(args.codec).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    layout = SharedPrivateLayout()
    memory, initialization = initialize_structured_memory(
        membership,
        radio_teacher_root=args.radio_teacher_root,
        siglip_teacher_path=siglip_path,
        layout=layout,
        seed=args.seed,
        hit_chunk=args.hit_chunk,
        codec_path=codec_path,
    )
    model = StructuredSharedPrivateMemory(memory, layout=layout)
    boundary_head = nn.Linear(layout.boundary, 1)
    nn.init.zeros_(boundary_head.weight)
    nn.init.zeros_(boundary_head.bias)
    codec = torch.load(codec_path, map_location="cpu")["state_dict"]
    state_dict = dict(model.state_dict())
    state_dict.update({
        "codec.radio_mean": codec["radio_mean"],
        "codec.radio_basis": codec["radio_basis"],
        "codec.siglip_mean": codec["siglip_mean"],
        "codec.siglip_basis": codec["siglip_basis"],
        "boundary_head.weight": boundary_head.weight.detach(),
        "boundary_head.bias": boundary_head.bias.detach(),
    })
    output = Path(args.output).resolve()
    payload = {
        "schema": "radio_gs.sugm_v3.structured_source_mapping.v1",
        "state_dict": state_dict,
        "metadata": {
            "state_representation": "single_structured_shared_private_d512_plus_global_heads",
            "architecture": model.architecture,
            "layout": dict(layout.__dict__),
            "partition_owned_writes": True,
            "phase_order": "learned_codec_initialization_before_private_training",
            "cross_block_bridges_enabled": False,
            "gaussian_indexed_sidecars": 0,
            "historical_field_opened": False,
            "source_only": True,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "view_split": {"train_residues": [1, 2], "dev_residue": 3, "audit_residue": 0},
            "initialization": initialization,
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "siglip_teacher": {"path": str(siglip_path), "sha256": sha256_file(siglip_path)},
            "radio_teacher_root": str(Path(args.radio_teacher_root).resolve(strict=True)),
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema": "radio_gs.sugm_v3.codec_initialization.report.v1",
        "status": "visual_semantic_gate_only_private_training_not_opened",
        "layout": dict(layout.__dict__),
        "checkpoint": {"path": str(output), "sha256": sha256_file(output)},
        "codec": {"path": str(codec_path), "sha256": sha256_file(codec_path)},
        "historical_field_opened": False,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--radio-teacher-root", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--hit-chunk", type=int, default=32768)
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
