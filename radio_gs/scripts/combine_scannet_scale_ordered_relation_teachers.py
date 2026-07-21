#!/usr/bin/env python3
"""Add a base relation teacher and same-only MPR-confirmed track teacher.

This is deliberately narrower than the disjoint-shard merger.  The base cache
contains local same *and* separate official-SAM3 evidence; the confirmed cache
contains only independently re-segmented source/target track positives.  Both
must be float32, share every frozen geometry/raster/scale contract, and are
added before intervals are recomputed.  No interval is copied or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.interfaces.scale_ordered_relation import merge_scale_intervals


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError(f"{path} is not a schema-v2 scale relation cache")
    metadata = dict(payload.get("metadata", {}))
    if not bool(metadata.get("query_free", False)):
        raise ValueError(f"{path} lacks query-free provenance")
    if any(bool(metadata.get(key, False)) for key in ("labels_opened", "instances_opened", "text_opened")):
        raise ValueError(f"{path} was built with benchmark annotations")
    if metadata.get("membership_lifting") != "raster_adjoint" or metadata.get("raster_lifting_semantics") != "true_alpha_compositing_adjoint":
        raise ValueError(f"{path} lacks promoted alpha-adjoint semantics")
    if not str(metadata.get("vote_storage", "")).startswith("fp32_"):
        raise ValueError(f"{path} must retain unquantized fp32 votes before teacher addition")
    edge = torch.as_tensor(payload.get("edge_index")).long().cpu()
    bins = torch.as_tensor(payload.get("scale_bin_edges_log")).float().cpu().reshape(-1)
    if edge.ndim != 2 or edge.shape[0] != 2 or bins.numel() < 2:
        raise ValueError(f"{path} has malformed static relation tensors")
    expected = (edge.shape[1], bins.numel() - 1)
    for key in ("same_votes", "separate_votes", "observed_votes"):
        values = torch.as_tensor(payload.get(key)).cpu()
        if values.dtype != torch.float32 or values.shape != expected:
            raise ValueError(f"{path} lacks aligned fp32 {key}")
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError(f"{path} has invalid {key}")
    return payload


def _assert_shared_contract(base: dict, confirmed: dict) -> None:
    if str(base.get("scene", "")) != str(confirmed.get("scene", "")):
        raise ValueError("base and confirmed relation scenes differ")
    for key in ("edge_rows", "edge_index", "features", "scale_bin_edges_log"):
        if not torch.equal(torch.as_tensor(base[key]).cpu(), torch.as_tensor(confirmed[key]).cpu()):
            raise ValueError(f"base and confirmed relation {key} differ")
    first, second = dict(base["metadata"]), dict(confirmed["metadata"])
    for key in (
        "scene_graph", "scene_graph_sha256", "membership_lifting",
        "raster_lifting_semantics", "responsibility_cache",
        "responsibility_metadata", "inside_threshold", "outside_threshold",
        "minimum_primitives_per_mask", "minimum_stability", "scale_definition",
        "scale_bins", "scale_minimum_radius_m", "scale_maximum_radius_m",
    ):
        if first.get(key) != second.get(key):
            raise ValueError(f"base and confirmed frozen contract differs at {key!r}")


def combine(base: dict, confirmed: dict, *, base_path: Path, confirmed_path: Path) -> dict:
    _assert_shared_contract(base, confirmed)
    confirmed_metadata = dict(confirmed["metadata"])
    if confirmed_metadata.get("teacher") != "official_sam3_mpr_confirmed_source_target_track_same_only":
        raise ValueError("second cache is not a same-only MPR-confirmed track teacher")
    if confirmed_metadata.get("confirmed_track_exterior") != "not_used_same_only_positive_constraints":
        raise ValueError("confirmed cache does not explicitly suppress virtual exterior negatives")
    if bool(torch.as_tensor(confirmed["separate_votes"]).any()) or bool(torch.as_tensor(confirmed["separate_events"]).any()):
        raise ValueError("same-only confirmed teacher contains unexpected separate evidence")
    votes = {
        key: torch.as_tensor(base[key]).float().cpu() + torch.as_tensor(confirmed[key]).float().cpu()
        for key in ("same_votes", "separate_votes", "observed_votes")
    }
    votes["scale_bin_edges_log"] = torch.as_tensor(base["scale_bin_edges_log"]).float().cpu()
    intervals = merge_scale_intervals(votes)
    metadata = dict(base["metadata"])
    metadata.update({
        "teacher": "official_sam3_multimask_plus_mpr_confirmed_tracks",
        "confirmed_track_teacher": "same_only_positive_source_target_union",
        "same_vote_merge": "base_local_same_and_separate_plus_confirmed_track_same_before_interval_derivation",
        "base_relation_cache": {"path": str(base_path.resolve()), "sha256": _sha256_file(base_path)},
        "confirmed_relation_cache": {"path": str(confirmed_path.resolve()), "sha256": _sha256_file(confirmed_path)},
        "vote_storage": "fp32_soft_same_and_separate_mass_no_overwrite",
        "mask_frames": list(base["metadata"].get("mask_frames", [])) + list(confirmed["metadata"].get("mask_frames", [])),
        "skipped_mask_frames": list(base["metadata"].get("skipped_mask_frames", [])) + list(confirmed["metadata"].get("skipped_mask_frames", [])),
    })
    return {
        "schema_version": 2,
        "scene": base["scene"],
        "edge_rows": torch.as_tensor(base["edge_rows"]).long().cpu(),
        "edge_index": torch.as_tensor(base["edge_index"]).long().cpu(),
        "features": torch.as_tensor(base["features"]).cpu(),
        "same_votes": votes["same_votes"],
        "separate_votes": votes["separate_votes"],
        "observed_votes": votes["observed_votes"],
        "same_events": torch.as_tensor(base["same_events"]).long().cpu() + torch.as_tensor(confirmed["same_events"]).long().cpu(),
        "separate_events": torch.as_tensor(base["separate_events"]).long().cpu() + torch.as_tensor(confirmed["separate_events"]).long().cpu(),
        "scale_bin_edges_log": votes["scale_bin_edges_log"],
        "merge_log_radius": intervals["merge_log_radius"].float(),
        "lower_log_radius": intervals["lower_log_radius"].float(),
        "upper_log_radius": intervals["upper_log_radius"].float(),
        "has_lower": intervals["has_lower"],
        "has_upper": intervals["has_upper"],
        "interval_consistent": intervals["interval_consistent"],
        "constraint_entropy": intervals["constraint_entropy"].float(),
        "same_mass": intervals["same_mass"].float(),
        "separate_mass": intervals["separate_mass"].float(),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--confirmed-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base_path, confirmed_path = Path(args.base_cache).resolve(), Path(args.confirmed_cache).resolve()
    payload = combine(_load(base_path), _load(confirmed_path), base_path=base_path, confirmed_path=confirmed_path)
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    constrained = (payload["same_mass"] + payload["separate_mass"]) > 0
    report = {
        "output": str(output), "scene": str(payload["scene"]),
        "edges": int(payload["edge_index"].shape[1]),
        "constrained_edges": int(constrained.sum()),
        "constrained_edge_fraction": float(constrained.float().mean()),
        "base_relation_cache": payload["metadata"]["base_relation_cache"],
        "confirmed_relation_cache": payload["metadata"]["confirmed_relation_cache"],
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
