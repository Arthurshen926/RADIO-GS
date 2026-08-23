#!/usr/bin/env python3
"""Materialize ScanNet split19 scores on independent source-only scenes.

The score function is exactly the frozen canonical-mpr-v3 region readout used
by the paper8 path: h128 physical regions at fixed radii, the official RADIO
SigLIP2 summary head, and the exact split19 text bank.  Only official source
annotations are joined later; this stage never opens semantic labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field import (
    _load_geometry_model,
)
from radio_gs.config import load_config
from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadout
from radio_gs.models.siglip_projection import (
    OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
    SigLIP2SummaryHead,
)
from radio_gs.scripts.materialize_ours_scannet_gaussian_semantic_score_cache import (
    compute_canonical_mpr_v3_semantic_scores,
    load_frozen_split19_text_bank,
)
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.scannet_independent_source_semantic_scores.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--field-checkpoint", type=Path, required=True)
    parser.add_argument("--mpr-cache", type=Path, required=True)
    parser.add_argument("--support-graph", type=Path, required=True)
    parser.add_argument("--readout-checkpoint", type=Path, required=True)
    parser.add_argument("--radio-checkpoint", type=Path, required=True)
    parser.add_argument("--text-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radio-batch-size", type=int, default=4096)
    parser.add_argument("--semantic-batch-size", type=int, default=128)
    args = parser.parse_args()

    paths = {
        name: Path(getattr(args, name)).expanduser().resolve(strict=True)
        for name in (
            "config",
            "geometry_checkpoint",
            "field_checkpoint",
            "mpr_cache",
            "support_graph",
            "readout_checkpoint",
            "radio_checkpoint",
            "text_bank",
        )
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    geometry = _load_geometry_model(
        load_config(str(paths["config"])), str(paths["geometry_checkpoint"]), torch.device("cpu")
    )
    reference_xyz = geometry.get_xyz().detach().cpu().float().contiguous()
    field, field_payload = load_canonical_field_checkpoint(
        paths["field_checkpoint"], map_location=device
    )
    mpr = torch.load(paths["mpr_cache"], map_location="cpu", weights_only=False)
    graph = torch.load(paths["support_graph"], map_location="cpu", weights_only=False)
    mpr_xyz = torch.as_tensor(mpr.get("xyz")).float().contiguous()
    graph_rows = torch.as_tensor(graph.get("global_rows")).long()
    if not torch.equal(mpr_xyz, reference_xyz):
        raise ValueError("source MPR rows differ from frozen geometry")
    if not torch.equal(graph_rows, torch.where(torch.as_tensor(mpr.get("valid")).bool())[0]):
        raise ValueError("source graph rows differ from MPR valid authority")
    if int(field.num_gaussians) != int(reference_xyz.shape[0]):
        raise ValueError("canonical field row count differs from frozen geometry")
    readout, _ = SurfaceRegionSummaryReadout.from_checkpoint(
        paths["readout_checkpoint"], map_location="cpu"
    )
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        str(paths["radio_checkpoint"]),
        expected_sha256=OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
    )
    text, text_sha, _ = load_frozen_split19_text_bank(paths["text_bank"], device=device)
    for module in (field, readout, head):
        module.to(device).eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    scores, region_observed = compute_canonical_mpr_v3_semantic_scores(
        field,
        mpr,
        graph,
        readout,
        head,
        text,
        device=device,
        radio_batch_size=args.radio_batch_size,
        semantic_batch_size=args.semantic_batch_size,
    )
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene_id": args.scene,
        "semantic_scores": scores,
        "valid": torch.as_tensor(mpr["valid"]).bool().contiguous(),
        "region_observed": region_observed,
        "reliability": torch.as_tensor(mpr["reliability"]).half().contiguous(),
        "geometry_xyz": reference_xyz,
        "class_ids": list(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"]),
        "source_paths": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "text_bank_sha256": text_sha,
        "field_architecture": dict(field_payload["architecture"]),
        "readout_formula": (
            "frozen_h128_physical_regions_radii_0.20_0.40_0.70_max_cosine_"
            "official_siglip2_summary_head"
        ),
        "access_contract": {
            "official_source_labels_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_predictions_opened": False,
            "benchmark_metrics_opened": False,
        },
    }
    torch.save(payload, output)
    receipt = output.with_suffix(output.suffix + ".json")
    receipt.write_text(json.dumps({
        "schema": SCHEMA,
        "scene_id": args.scene,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "num_gaussians": int(scores.shape[0]),
        "region_observed_fraction": float(region_observed.float().mean()),
        "benchmark_masks_opened": False,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "receipt": str(receipt)}, indent=2))


if __name__ == "__main__":
    main()
