#!/usr/bin/env python3
"""Lift official query-free SAM3 automatic masks into 3-D edge relations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from radio_gs.interfaces.relation_calibrator import edge_relation_features
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks


def accumulate_relation_votes(
    memberships: list[torch.Tensor], observations: list[torch.Tensor],
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge = torch.as_tensor(edge_index).long().cpu(); src, dst = edge
    same_votes = torch.zeros(edge.shape[1], dtype=torch.int16)
    cannot_votes = torch.zeros_like(same_votes); observed_votes = torch.zeros_like(same_votes)
    for membership, observed in zip(memberships, observations):
        member = torch.as_tensor(membership).bool().cpu()
        seen = torch.as_tensor(observed).bool().cpu()
        if member.ndim != 2 or member.shape[1] != seen.numel():
            raise ValueError("membership must be [M,N] aligned with observation [N]")
        coobserved = seen[src] & seen[dst]
        covered = member.any(0)
        # Rows are ordered from the smallest (most specific) automatic region
        # to the largest.  Assign each primitive to its most specific covering
        # region; otherwise a broad room/background mask would incorrectly
        # turn nearly every local edge into a must-link.
        assignment = member.float().argmax(0) if len(member) else torch.zeros_like(seen, dtype=torch.long)
        same = covered[src] & covered[dst] & (assignment[src] == assignment[dst])
        cannot = coobserved & covered[src] & covered[dst] & ~same
        observed_votes += coobserved.to(torch.int16)
        same_votes += (coobserved & same).to(torch.int16)
        cannot_votes += cannot.to(torch.int16)
    return same_votes, cannot_votes, observed_votes


def _project_membership(graph: dict, frame: dict, mask_payload: dict) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = torch.as_tensor(graph["xyz"]).float()
    pose = torch.as_tensor(frame["pose"]).float()
    camera = torch.cat([xyz, torch.ones(len(xyz), 1)], 1) @ torch.linalg.inv(pose).T
    z = camera[:, 2]
    kd = torch.as_tensor(graph["depth_intrinsic"]).float()
    kc = torch.as_tensor(graph["color_intrinsic"]).float()
    depth = torch.from_numpy(np.asarray(Image.open(frame["depth"]), dtype=np.float32)) / 1000.0
    ud = kd[0, 0] * camera[:, 0] / z.clamp_min(1e-6) + kd[0, 2]
    vd = kd[1, 1] * camera[:, 1] / z.clamp_min(1e-6) + kd[1, 2]
    ix, iy = ud.round().long(), vd.round().long()
    depth_inside = (ix >= 0) & (iy >= 0) & (ix < depth.shape[1]) & (iy < depth.shape[0])
    observed = (z > 0.15) & depth_inside
    valid_rows = torch.where(observed)[0]
    if len(valid_rows):
        measured = depth[iy[valid_rows], ix[valid_rows]]
        observed[valid_rows] &= (measured > 0) & ((measured - z[valid_rows]).abs() < 0.10)
    height, width = (int(value) for value in mask_payload["mask_shape"])
    u = kc[0, 0] * camera[:, 0] / z.clamp_min(1e-6) + kc[0, 2]
    v = kc[1, 1] * camera[:, 1] / z.clamp_min(1e-6) + kc[1, 2]
    ui, vi = u.round().long(), v.round().long()
    color_inside = (ui >= 0) & (vi >= 0) & (ui < width) & (vi < height)
    observed &= color_inside
    masks = torch.from_numpy(unpack_masks(mask_payload["packed_masks"], width))
    if len(masks):
        masks = masks[torch.argsort(masks.flatten(1).sum(1), stable=True)]
    membership = torch.zeros(len(masks), len(xyz), dtype=torch.bool)
    rows = torch.where(observed)[0]
    if len(rows) and len(masks): membership[:, rows] = masks[:, vi[rows], ui[rows]]
    return membership, observed


def run(args: argparse.Namespace) -> dict:
    graph_path = Path(args.scene_graph); graph = torch.load(graph_path, map_location="cpu")
    mask_root = Path(args.mask_root); by_stem = {Path(f["color"]).stem: f for f in graph["frames"]}
    requested_stems = {
        value.strip() for value in str(args.mask_stems).replace(",", " ").split()
        if value.strip()
    }
    memberships, observations, used = [], [], []
    for mask_path in sorted(mask_root.glob("*.pt")):
        if requested_stems and mask_path.stem not in requested_stems:
            continue
        frame = by_stem.get(mask_path.stem)
        if frame is None: continue
        payload = torch.load(mask_path, map_location="cpu")
        member, observed = _project_membership(graph, frame, payload)
        memberships.append(member); observations.append(observed); used.append(mask_path.name)
    if not memberships: raise RuntimeError("no automatic-mask frame aligns with scene graph")
    same, cannot, observed = accumulate_relation_votes(
        memberships, observations, graph["edge_index"]
    )
    edge = torch.as_tensor(graph["edge_index"]).long(); unique = edge[0] < edge[1]
    positive = same > 0
    negative = (same == 0) & (cannot > 0)
    keep = unique & (positive | negative)
    labels = positive[keep].float()
    edge_rows = torch.where(keep)[0]
    payload = {
        "schema_version": 1, "scene": graph["scene"],
        "features": edge_relation_features(graph)[keep], "labels": labels,
        "edge_rows": edge_rows, "num_nodes": int(graph["xyz"].shape[0]),
        "same_votes": same[keep], "cannot_votes": cannot[keep],
        "observed_votes": observed[keep],
        "metadata": {"teacher": "official_sam3_query_free_automatic_masks",
                     "mask_frames": used, "labels_opened": False,
                     "instances_opened": False, "text_opened": False},
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {"output": str(output.resolve()), "scene": graph["scene"],
              "edges": int(keep.sum()), "positive": int(labels.sum()),
              "negative": int((1-labels).sum()), "mask_frames": used}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-graph", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument(
        "--mask-stems", default="",
        help="Optional comma/space-separated frame stems for a deterministic split.",
    )
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__": main()
