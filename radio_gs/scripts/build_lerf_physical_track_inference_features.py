#!/usr/bin/env python3
"""Build label-free proposal-pair features for a heldout LERF scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_physical_track_inference_features.v1"


def build(args: argparse.Namespace) -> dict:
    proposal_dir = Path(args.proposal_dir).resolve(); teacher_path = Path(args.teacher).resolve()
    output = Path(args.output).resolve(); report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists(): raise FileExistsError(f"features exist: {output}")
    manifest = json.loads((proposal_dir / "manifest.json").read_text())
    qualities, areas, views = [], [], []
    for view, item in enumerate(manifest["images"]):
        payload = torch.load(Path(item["output"]), map_location="cpu", weights_only=False)
        count = int(torch.as_tensor(payload["quality"]).numel())
        qualities.append(torch.as_tensor(payload["quality"]).float())
        areas.append(torch.as_tensor(payload["proposal_area_fraction"]).float())
        views.append(torch.full((count,), view, dtype=torch.long))
    quality, area, view = torch.cat(qualities), torch.cat(areas), torch.cat(views)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    descriptor = F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
    context = F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1)
    if descriptor.shape[0] != view.numel(): raise ValueError("teacher/proposal axes differ")
    left, right = torch.triu_indices(view.numel(), view.numel(), offset=1)
    cross = view[left] != view[right]; left, right = left[cross], right[cross]
    features = torch.stack([
        (descriptor[left] * descriptor[right]).sum(-1),
        (context[left] * context[right]).sum(-1),
        -torch.abs(torch.log2(area[left].clamp_min(1e-8) / area[right].clamp_min(1e-8))),
        torch.minimum(quality[left], quality[right]),
    ], dim=1)
    payload = {"schema": SCHEMA, "schema_version": 1, "scene": args.scene,
               "edge_left": left, "edge_right": right, "edge_features": features,
               "feature_names": ["masked_descriptor_cosine", "context_descriptor_cosine", "negative_absolute_log2_area_ratio", "minimum_sam_quality"],
               "metadata": {"label_free": True, "benchmark_masks_opened": False,
                            "evaluation_rgb_opened": False, "target_metrics_opened": False,
                            "proposal_manifest": {"path": str(proposal_dir / 'manifest.json'), "sha256": sha256_file(proposal_dir / 'manifest.json')},
                            "teacher": {"path": str(teacher_path), "sha256": sha256_file(teacher_path)}}}
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary); os.replace(temporary, output)
    report = {"schema": SCHEMA, "status": "complete", "scene": args.scene,
              "proposal_count": int(view.numel()), "edge_count": int(left.numel()),
              "benchmark_masks_opened": False, "output": str(output), "output_sha256": sha256_file(output)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--scene", required=True); parser.add_argument("--proposal-dir", required=True); parser.add_argument("--teacher", required=True); parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__": main()
