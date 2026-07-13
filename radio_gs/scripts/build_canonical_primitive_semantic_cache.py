#!/usr/bin/env python3
"""Build primitive-first text-space descriptors from one canonical RADIO field.

Each descriptor is computed from density-adaptive 3-D neighbourhoods, the
frozen global region-to-summary bridge, and the frozen official SigLIP2
summary head.  No image, query string, benchmark mask, or evaluation frame is
opened by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.semantic_alignment import GlobalRegionSummaryBridge
from radio_gs.models.siglip_projection import SigLIP2SummaryHead


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _parse_sizes(raw: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(value) for value in raw.replace(",", " ").split()}))
    if not sizes or sizes[0] <= 0:
        raise ValueError("neighborhood sizes must be positive")
    return sizes


@torch.no_grad()
def build(args: argparse.Namespace) -> dict:
    from scipy.spatial import cKDTree

    device = torch.device(args.device)
    field, field_payload = load_canonical_field_checkpoint(
        args.field_checkpoint, map_location="cpu"
    )
    cache_path = Path(args.mpr_cache or field_payload["mpr_cache"])
    mpr = torch.load(cache_path, map_location="cpu")
    xyz = torch.as_tensor(mpr["xyz"]).float().cpu()
    valid = torch.as_tensor(mpr["valid"]).bool().cpu()
    expected_hash = str(field_payload.get("geometry_fingerprint", {}).get("xyz_sha256", ""))
    actual_hash = _sha256_tensor_rows(xyz)
    if expected_hash != actual_hash:
        raise ValueError("canonical field and MPR geometry rows differ")
    if field.num_gaussians != xyz.shape[0] or valid.shape != (xyz.shape[0],):
        raise ValueError("canonical field, geometry, and valid rows do not align")

    bridge, bridge_manifest = GlobalRegionSummaryBridge.from_checkpoint(
        args.bridge_checkpoint, map_location="cpu"
    )
    bridge = bridge.to(device).eval()
    summary_head = SigLIP2SummaryHead.from_radio_checkpoint(
        args.radio_checkpoint
    ).to(device).eval()
    for module in (field, bridge, summary_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    field = field.to(device).eval()

    count = xyz.shape[0]
    radio = torch.empty(
        count, field.decoder.feature_dim, dtype=torch.float16, device=device
    )
    if args.radio_source == "mpr":
        mpr_radio = torch.as_tensor(mpr["features"])
        if mpr_radio.shape != (count, field.decoder.feature_dim):
            raise ValueError("MPR RADIO features do not align with the canonical field")
        for start in range(0, count, int(args.radio_batch_size)):
            stop = min(count, start + int(args.radio_batch_size))
            radio[start:stop] = mpr_radio[start:stop].to(device).half()
        del mpr_radio
    else:
        for start in range(0, count, int(args.radio_batch_size)):
            stop = min(count, start + int(args.radio_batch_size))
            rows = torch.arange(start, stop, device=device)
            radio[start:stop] = field.radio_features(rows).half()
    encoded = torch.empty(
        count, bridge.hidden_dim, dtype=torch.float16, device=device
    )
    attention_logits = torch.empty(count, dtype=torch.float16, device=device)
    for start in range(0, count, int(args.radio_batch_size)):
        stop = min(count, start + int(args.radio_batch_size))
        token_hidden, token_logits = bridge.encode_region_tokens(radio[start:stop])
        encoded[start:stop] = token_hidden.half()
        attention_logits[start:stop] = token_logits.half()
    del field
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sizes = _parse_sizes(args.neighborhood_sizes)
    valid_rows = torch.where(valid)[0]
    maximum = min(max(sizes), count)
    _distances, neighbor_np = cKDTree(xyz.numpy()).query(
        xyz[valid_rows].numpy(),
        k=maximum,
        workers=int(args.knn_workers),
    )
    neighbor_np = np.asarray(neighbor_np, dtype=np.int64)
    if neighbor_np.ndim == 1:
        neighbor_np = neighbor_np[:, None]

    descriptors = torch.zeros(count, int(args.output_dim), dtype=torch.float16)
    valid_device = valid.to(device)
    for start in range(0, valid_rows.numel(), int(args.semantic_batch_size)):
        stop = min(valid_rows.numel(), start + int(args.semantic_batch_size))
        neighborhood = torch.from_numpy(neighbor_np[start:stop]).to(device)
        scale_descriptors: list[torch.Tensor] = []
        for requested in sizes:
            scale = min(int(requested), neighborhood.shape[1])
            rows = neighborhood[:, :scale]
            summary = bridge.summarize_preencoded_region(
                radio[rows],
                encoded[rows],
                attention_logits[rows],
                token_mask=valid_device[rows],
            )
            projected = summary_head(summary[:, None])[:, 0]
            scale_descriptors.append(F.normalize(projected.float(), dim=-1, eps=1e-8))
        fused = F.normalize(
            torch.stack(scale_descriptors, dim=1).mean(dim=1), dim=-1, eps=1e-8
        )
        descriptors[valid_rows[start:stop]] = fused.half().cpu()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "source": f"{args.radio_source}_radio_primitive_neighborhood",
        "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
        "field_checkpoint_sha256": _sha256_file(args.field_checkpoint),
        "mpr_cache": str(cache_path.resolve()),
        "bridge_checkpoint": str(Path(args.bridge_checkpoint).resolve()),
        "bridge_checkpoint_sha256": bridge_manifest.checkpoint_sha256,
        "bridge_training_scope": bridge_manifest.training_scope,
        "radio_checkpoint": str(Path(args.radio_checkpoint).resolve()),
        "radio_checkpoint_sha256": _sha256_file(args.radio_checkpoint),
        "neighborhood_type": "density_adaptive_3d_knn",
        "neighborhood_sizes": list(sizes),
        "scale_fusion": "normalized_official_descriptor_mean",
        "official_summary_head": True,
        "custom_text_projection": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    torch.save(
        {
            "schema_version": 1,
            "summary_features": descriptors,
            "features": descriptors,
            "valid": valid,
            "xyz": xyz,
            "geometry_fingerprint": mpr.get("geometry_fingerprint", {}),
            "metadata": metadata,
        },
        output,
    )
    report = {
        **metadata,
        "output": str(output),
        "num_gaussians": count,
        "valid_gaussians": int(valid.sum()),
        "feature_dim": descriptors.shape[1],
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mpr-cache", default="")
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--neighborhood-sizes", default="8,32,64")
    parser.add_argument(
        "--radio-source",
        choices=["canonical", "mpr"],
        default="canonical",
        help="Use the learned field or its query-free multiview RADIO target (oracle audit).",
    )
    parser.add_argument("--radio-batch-size", type=int, default=16384)
    parser.add_argument("--semantic-batch-size", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=1536)
    parser.add_argument("--knn-workers", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
