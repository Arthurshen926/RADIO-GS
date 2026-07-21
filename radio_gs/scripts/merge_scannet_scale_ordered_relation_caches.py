#!/usr/bin/env python3
"""Merge disjoint, query-free raster-adjoint relation-cache shards exactly.

Raster-adjoint mask lifting is independent across registered source views.  A
large scene can therefore render disjoint view shards on several GPUs, then
add their *unquantized* same/separate vote histograms before deriving merge
intervals.  This utility deliberately refuses fp16 shards, duplicated mask
frames, changed geometry/provenance, or any cache without the explicit
query-free contract.  It is not an ensemble or a model-selection operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.scale_ordered_relation import merge_scale_intervals


STATIC_KEYS = (
    "edge_rows", "edge_index", "features", "scale_bin_edges_log",
)
VOTE_KEYS = (
    "same_votes", "separate_votes", "observed_votes",
)
CONTRACT_METADATA_KEYS = (
    "query_free", "labels_opened", "instances_opened", "text_opened",
    "scene_graph", "scene_graph_sha256", "membership_lifting",
    "raster_lifting_semantics", "raster_responsibility_used",
    "responsibility_cache", "responsibility_metadata",
    "raster_adjoint_provenance", "mask_raster_alignment",
    "inside_threshold", "outside_threshold", "minimum_primitives_per_mask",
    "minimum_stability", "scale_definition", "scale_bins",
    "scale_minimum_radius_m", "scale_maximum_radius_m",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _numeric_stem_key(value: str) -> tuple[int, int | str, str]:
    stem = Path(str(value)).stem
    return (0, int(stem), str(value)) if stem.isdigit() else (1, stem, str(value))


def _load_relation_cache(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError(f"{path} is not a schema-v2 scale-ordered relation cache")
    metadata = dict(payload.get("metadata", {}))
    if not bool(metadata.get("query_free", False)):
        raise ValueError(f"{path} does not declare query-free construction")
    if any(bool(metadata.get(key, False)) for key in ("labels_opened", "instances_opened", "text_opened")):
        raise ValueError(f"{path} was constructed with benchmark annotations")
    if metadata.get("membership_lifting") != "raster_adjoint":
        raise ValueError(f"{path} does not use promoted raster-adjoint lifting")
    if metadata.get("raster_lifting_semantics") != "true_alpha_compositing_adjoint":
        raise ValueError(f"{path} lacks true alpha-compositing adjoint semantics")
    if str(metadata.get("vote_storage", "")).split("_", 1)[0] != "fp32":
        raise ValueError(
            f"{path} stores quantized votes; rebuild the shard with "
            "--vote-storage-dtype float32 before exact merging"
        )
    for key in (*STATIC_KEYS, *VOTE_KEYS, "same_events", "separate_events"):
        if key not in payload:
            raise ValueError(f"{path} lacks {key}")
    edge = torch.as_tensor(payload["edge_index"]).long().cpu()
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError(f"{path} has malformed edge_index")
    bins = torch.as_tensor(payload["scale_bin_edges_log"]).float().cpu().reshape(-1)
    if bins.numel() < 2 or not bool(torch.isfinite(bins).all()) or not bool((bins[1:] > bins[:-1]).all()):
        raise ValueError(f"{path} has invalid scale bins")
    expected = (edge.shape[1], bins.numel() - 1)
    for key in VOTE_KEYS:
        values = torch.as_tensor(payload[key]).cpu()
        if values.dtype != torch.float32 or values.shape != expected:
            raise ValueError(f"{path} must retain unquantized float32 {key} with shape {expected}")
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError(f"{path} has invalid {key}")
    for key in ("same_events", "separate_events"):
        values = torch.as_tensor(payload[key]).long().cpu()
        if values.shape != (expected[1],) or bool((values < 0).any()):
            raise ValueError(f"{path} has invalid {key}")
    return payload


def _assert_same_static_contract(reference: dict, candidate: dict, *, reference_path: Path, candidate_path: Path) -> None:
    if str(reference.get("scene", "")) != str(candidate.get("scene", "")):
        raise ValueError(f"scene differs between {reference_path} and {candidate_path}")
    for key in STATIC_KEYS:
        if not torch.equal(torch.as_tensor(reference[key]).cpu(), torch.as_tensor(candidate[key]).cpu()):
            raise ValueError(f"{key} differs between {reference_path} and {candidate_path}")
    first_metadata, second_metadata = dict(reference["metadata"]), dict(candidate["metadata"])
    for key in CONTRACT_METADATA_KEYS:
        if key not in first_metadata or key not in second_metadata:
            raise ValueError(f"missing immutable relation contract key {key!r}")
        if _canonical_json(first_metadata[key]) != _canonical_json(second_metadata[key]):
            raise ValueError(f"relation contract {key!r} differs between {reference_path} and {candidate_path}")


def merge_relation_caches(paths: list[Path]) -> dict:
    """Add independently rendered float32 vote histograms and rebuild intervals."""

    if len(paths) < 2:
        raise ValueError("need at least two disjoint relation-cache shards to merge")
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate relation-cache input path")
    payloads = [_load_relation_cache(path) for path in resolved]
    reference = payloads[0]
    for path, payload in zip(resolved[1:], payloads[1:]):
        _assert_same_static_contract(reference, payload, reference_path=resolved[0], candidate_path=path)

    used_frames: set[str] = set()
    skipped_frames: list[dict] = []
    resized_frames: list[str] = []
    roots: set[str] = set()
    schemas: set[int] = set()
    for path, payload in zip(resolved, payloads):
        metadata = dict(payload["metadata"])
        local_frames = [str(value) for value in metadata.get("mask_frames", [])]
        duplicate = used_frames.intersection(local_frames)
        if duplicate:
            raise ValueError(
                f"mask frame(s) appear in more than one shard: {sorted(duplicate, key=_numeric_stem_key)}"
            )
        used_frames.update(local_frames)
        skipped_frames.extend(dict(value) for value in metadata.get("skipped_mask_frames", []))
        resized_frames.extend(str(value) for value in metadata.get("resized_mask_frames", []))
        roots.update(str(value) for value in metadata.get("mask_roots", []))
        schemas.update(int(value) for value in metadata.get("mask_schema_versions", []))

    votes = {
        key: sum((torch.as_tensor(payload[key]).float().cpu() for payload in payloads), torch.zeros_like(torch.as_tensor(reference[key]).float().cpu()))
        for key in VOTE_KEYS
    }
    votes["scale_bin_edges_log"] = torch.as_tensor(reference["scale_bin_edges_log"]).float().cpu()
    intervals = merge_scale_intervals(votes)
    metadata = dict(reference["metadata"])
    metadata.update({
        "vote_storage": "fp32_soft_same_and_separate_mass_no_overwrite",
        "mask_roots": sorted(roots),
        "mask_frames": sorted(used_frames, key=_numeric_stem_key),
        "skipped_mask_frames": skipped_frames,
        "mask_schema_versions": sorted(schemas),
        "resized_mask_frames": sorted(resized_frames, key=_numeric_stem_key),
        "resized_mask_count": int(sum(int(dict(payload["metadata"]).get("resized_mask_count", 0)) for payload in payloads)),
        "shard_merge": "additive_frozen_query_free_relation_vote_merge",
        "source_relation_cache_count": len(resolved),
        "source_relation_caches": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in resolved
        ],
    })
    return {
        "schema_version": 2,
        "scene": reference["scene"],
        "edge_rows": torch.as_tensor(reference["edge_rows"]).long().cpu(),
        "edge_index": torch.as_tensor(reference["edge_index"]).long().cpu(),
        "features": torch.as_tensor(reference["features"]).cpu(),
        "same_votes": votes["same_votes"],
        "separate_votes": votes["separate_votes"],
        "observed_votes": votes["observed_votes"],
        "same_events": sum((torch.as_tensor(payload["same_events"]).long().cpu() for payload in payloads), torch.zeros_like(torch.as_tensor(reference["same_events"]).long().cpu())),
        "separate_events": sum((torch.as_tensor(payload["separate_events"]).long().cpu() for payload in payloads), torch.zeros_like(torch.as_tensor(reference["separate_events"]).long().cpu())),
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


def _report(payload: dict, output: Path) -> dict:
    both = torch.as_tensor(payload["has_lower"]).bool() & torch.as_tensor(payload["has_upper"]).bool()
    constrained = (torch.as_tensor(payload["same_mass"]) + torch.as_tensor(payload["separate_mass"])) > 0
    return {
        "output": str(output.resolve()), "scene": str(payload["scene"]),
        "source_relation_cache_count": int(payload["metadata"]["source_relation_cache_count"]),
        "mask_frames": len(payload["metadata"]["mask_frames"]),
        "edges": int(torch.as_tensor(payload["edge_index"]).shape[1]),
        "constrained_edges": int(constrained.sum()),
        "constrained_edge_fraction": float(constrained.float().mean()),
        "edges_with_both_bounds": int(both.sum()),
        "interval_consistent_fraction_among_both": float(
            torch.as_tensor(payload["interval_consistent"])[both].float().mean() if bool(both.any()) else 0.0
        ),
        "raster_lifting_semantics": payload["metadata"]["raster_lifting_semantics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Disjoint float32 raster-adjoint shard caches")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    payload = merge_relation_caches([Path(value) for value in args.inputs])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = _report(payload, output)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
