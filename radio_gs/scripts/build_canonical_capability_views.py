#!/usr/bin/env python3
"""Derive frozen official DINOv3/SAM3 primitive views from one canonical field."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.field import FeatureSpaceSignature, load_canonical_field_checkpoint
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def build(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    field, payload = load_canonical_field_checkpoint(
        args.field_checkpoint, map_location="cpu"
    )
    mpr_path = Path(args.mpr_cache or payload["mpr_cache"])
    mpr = torch.load(mpr_path, map_location="cpu")
    xyz = torch.as_tensor(mpr["xyz"]).float().cpu()
    valid = torch.as_tensor(mpr["valid"]).bool().cpu()
    if field.num_gaussians != xyz.shape[0] or valid.shape != (xyz.shape[0],):
        raise ValueError("canonical field and MPR rows do not align")

    adaptors = {
        "appearance_dino_v3": load_radio_adaptor_from_checkpoint(
            args.radio_checkpoint, "dino_v3_7b", kind="feature_projection"
        ).to(device).eval(),
        "boundary_sam3": load_radio_adaptor_from_checkpoint(
            args.radio_checkpoint, "sam3", kind="feature_projection"
        ).to(device).eval(),
    }
    field = field.to(device).eval()
    for module in (field, *adaptors.values()):
        module.requires_grad_(False)

    rows = torch.where(valid)[0]
    outputs = {
        name: torch.zeros(
            xyz.shape[0], adaptor.output_dim, dtype=torch.float16
        )
        for name, adaptor in adaptors.items()
    }
    for start in range(0, rows.numel(), int(args.batch_size)):
        selected_cpu = rows[start : start + int(args.batch_size)]
        selected = selected_cpu.to(device)
        radio = field.radio_features(selected).float()
        for name, adaptor in adaptors.items():
            projected = F.normalize(adaptor(radio).float(), dim=-1, eps=1e-8)
            outputs[name][selected_cpu] = projected.half().cpu()

    radio_checkpoint_sha256 = _sha256_file(args.radio_checkpoint)
    field_checkpoint_sha256 = _sha256_file(args.field_checkpoint)
    base_signature = field.signature.to_dict()
    raw_capability_targets = payload.get("capability_mpr_targets", {})
    if not isinstance(raw_capability_targets, dict):
        raise ValueError("canonical field capability MPR provenance must be a mapping")
    capability_teacher_sources: dict[str, dict[str, object]] = {}
    for target_name, output_name in (
        ("dino_v3", "appearance"),
        ("sam3", "boundary"),
    ):
        target = raw_capability_targets.get(target_name, {})
        if not isinstance(target, dict):
            raise ValueError(
                f"canonical field {target_name} capability provenance must be a mapping"
            )
        native_grid = target.get("capability_native_map_grid", [])
        if not isinstance(native_grid, (list, tuple)):
            raise ValueError(
                f"canonical field {target_name} native-map grid must be a sequence"
            )
        capability_teacher_sources[output_name] = {
            "capability_map_source": str(
                target.get("capability_map_source", "project_raw")
            ),
            "capability_native_map_manifest": str(
                target.get("capability_native_map_manifest", "")
            ),
            "capability_native_map_manifest_sha256": str(
                target.get("capability_native_map_manifest_sha256", "")
            ),
            "capability_native_map_grid": list(native_grid),
            "capability_adaptor_execution": str(
                target.get("capability_adaptor_execution", "")
            ),
        }
    render_optimization = payload.get("render_optimization", {})
    if not isinstance(render_optimization, dict):
        raise ValueError("canonical field render optimization provenance must be a mapping")
    render_capability = render_optimization.get("official_render_capability", {})
    if not isinstance(render_capability, dict):
        raise ValueError("canonical field render capability provenance must be a mapping")
    render_teacher_provenance = render_capability.get("teacher_map_provenance", {})
    if not isinstance(render_teacher_provenance, dict):
        raise ValueError("canonical field render teacher provenance must be a mapping")

    def capability_signature(name: str, output_dim: int) -> dict:
        return FeatureSpaceSignature(
            **{
                **base_signature,
                "adaptor_name": f"{name}.feature_projection",
                "adaptor_sha256": radio_checkpoint_sha256,
                "adaptor_output_dim": int(output_dim),
                "token_type": "primitive",
                "normalization": "l2",
                "field_checkpoint_sha256": field_checkpoint_sha256,
                "semantic_alignment": "none",
                "semantic_alignment_sha256": "",
            }
        ).to_dict()

    metadata = {
        "schema_version": 1,
        "source": "canonical_radio_field_official_frozen_capability_views",
        "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
        "field_checkpoint_sha256": field_checkpoint_sha256,
        "mpr_cache": str(mpr_path.resolve()),
        "radio_checkpoint": str(Path(args.radio_checkpoint).resolve()),
        "radio_checkpoint_sha256": radio_checkpoint_sha256,
        "appearance_view": "official C-RADIOv4 dino_v3_7b feature_projection",
        "boundary_view": "official C-RADIOv4 sam3 feature_projection",
        "custom_adaptor_head": False,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "capability_training_mpr_sources": capability_teacher_sources,
        "render_capability_teacher_source": str(
            render_capability.get("teacher_map_source", "project_raw")
        ),
        "render_capability_teacher_provenance": dict(render_teacher_provenance),
        "capability_signatures": {
            "appearance": capability_signature(
                "dino_v3_7b", outputs["appearance_dino_v3"].shape[1]
            ),
            "boundary": capability_signature(
                "sam3", outputs["boundary_sam3"].shape[1]
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "xyz": xyz,
            "valid": valid,
            **outputs,
            "metadata": metadata,
        },
        output,
    )
    report = {
        **metadata,
        "output": str(output),
        "num_gaussians": int(xyz.shape[0]),
        "valid_gaussians": int(valid.sum()),
        "appearance_dim": int(outputs["appearance_dino_v3"].shape[1]),
        "boundary_dim": int(outputs["boundary_sam3"].shape[1]),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mpr-cache", default="")
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
