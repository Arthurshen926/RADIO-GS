#!/usr/bin/env python3
"""Extract the exact descriptor-only semantic sidecar used by image queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_torch_save(payload: object, output: Path) -> None:
    """Write a sidecar privately, then publish it as one complete archive."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _query_metadata(metadata: object) -> dict:
    """Normalize old semantic-cache provenance without touching descriptors.

    Early v3 cache writers recorded the official RADIO and readout hashes under
    descriptive names, but the pose-free compiler expected the standardized
    bridge fields.  Derive those fields only after validating the referenced
    frozen readout checkpoint and its cross-scene provenance.
    """

    if not isinstance(metadata, dict):
        raise ValueError("semantic cache metadata must be a mapping")
    result = dict(metadata)
    if not result.get("radio_checkpoint_sha256"):
        radio_sha = str(result.get("official_radio_checkpoint_sha256", ""))
        if not radio_sha:
            raise ValueError("semantic cache lacks official RADIO checkpoint provenance")
        result["radio_checkpoint_sha256"] = radio_sha
    required = (
        "bridge_training_scope",
        "bridge_checkpoint_sha256",
    )
    if all(result.get(key) for key in required):
        return result
    readout_path = Path(str(result.get("readout_checkpoint", "")))
    expected_sha = str(result.get("readout_checkpoint_sha256", ""))
    if not readout_path.is_file() or not expected_sha:
        raise ValueError("semantic cache lacks a verifiable readout checkpoint")
    if _sha256(readout_path) != expected_sha:
        raise ValueError("semantic cache readout checkpoint hash mismatch")
    checkpoint = torch.load(readout_path, map_location="cpu")
    provenance = checkpoint.get("provenance", {}) if isinstance(checkpoint, dict) else {}
    scope = str(provenance.get("training_scope", ""))
    if (
        not scope.startswith("global_cross_scene")
        or provenance.get("uses_benchmark_scenes", True)
        or provenance.get("uses_benchmark_test_vocabulary", True)
        or provenance.get("scene_disjoint") is not True
    ):
        raise ValueError("readout checkpoint is not frozen global cross-scene training")
    result["bridge_training_scope"] = "global_cross_scene"
    result["bridge_training_scope_detail"] = scope
    result["bridge_checkpoint_sha256"] = expected_sha
    return result


def run(args: argparse.Namespace) -> dict:
    source = Path(args.semantic_cache)
    output = Path(args.output)
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("semantic cache must be a mapping")
    required = ("xyz", "features", "valid", "metadata")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"semantic cache misses required keys: {missing}")
    features = torch.as_tensor(payload["features"])
    xyz = torch.as_tensor(payload["xyz"])
    valid = torch.as_tensor(payload["valid"]).bool()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or valid.shape != (xyz.shape[0],):
        raise ValueError("semantic cache global geometry is malformed")
    global_rows = payload.get("global_rows")
    if global_rows is None:
        if features.ndim != 2 or features.shape[0] != xyz.shape[0]:
            raise ValueError("dense semantic cache descriptor geometry is malformed")
    else:
        global_rows = torch.as_tensor(global_rows).long().cpu()
        if (
            features.ndim != 2
            or global_rows.ndim != 1
            or features.shape[0] != global_rows.numel()
            or not torch.equal(torch.where(valid)[0], global_rows)
        ):
            raise ValueError("sparse semantic cache descriptor rows are malformed")
    sidecar = {
        "xyz": xyz,
        "features": features,
        "valid": valid,
        "metadata": _query_metadata(payload["metadata"]),
    }
    if global_rows is not None:
        sidecar["global_rows"] = global_rows
    _atomic_torch_save(sidecar, output)
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "num_rows": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "exact_descriptor_sidecar": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-cache", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
