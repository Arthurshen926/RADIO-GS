#!/usr/bin/env python3
"""Restrict canonical reliability to a query-free descriptor-valid row set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align(
    reliability_path: str | Path,
    descriptor_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    reliability_path = Path(reliability_path).resolve()
    descriptor_path = Path(descriptor_path).resolve()
    output_path = Path(output_path).resolve()
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or report_path.exists():
        raise FileExistsError(output_path if output_path.exists() else report_path)

    reliability = torch.load(reliability_path, map_location="cpu")
    descriptor = torch.load(descriptor_path, map_location="cpu")
    if not isinstance(reliability, Mapping) or not isinstance(descriptor, Mapping):
        raise ValueError("reliability and descriptor caches must be mappings")
    required_reliability = {"xyz", "valid", "confidence", "components", "metadata"}
    required_descriptor = {"xyz", "valid"}
    if not required_reliability.issubset(reliability):
        raise ValueError("canonical reliability cache is incomplete")
    if not required_descriptor.issubset(descriptor):
        raise ValueError("query-free descriptor cache is incomplete")

    xyz = torch.as_tensor(reliability["xyz"]).float().cpu()
    descriptor_xyz = torch.as_tensor(descriptor["xyz"]).float().cpu()
    source_valid = torch.as_tensor(reliability["valid"]).bool().cpu()
    target_valid = torch.as_tensor(descriptor["valid"]).bool().cpu()
    count = int(xyz.shape[0]) if xyz.ndim == 2 else -1
    if xyz.shape != (count, 3) or descriptor_xyz.shape != xyz.shape:
        raise ValueError("reliability/descriptor geometry is malformed")
    if not torch.equal(xyz, descriptor_xyz):
        raise ValueError("reliability/descriptor xyz authority differs")
    if source_valid.shape != (count,) or target_valid.shape != (count,):
        raise ValueError("reliability/descriptor valid rows are malformed")
    if bool((target_valid & ~source_valid).any()):
        raise ValueError("descriptor-valid rows are not a subset of reliability rows")

    confidence = torch.as_tensor(reliability["confidence"]).clone().cpu()
    if confidence.shape != (count,):
        raise ValueError("reliability confidence is malformed")
    confidence[~target_valid] = 0
    raw_components = reliability["components"]
    if not isinstance(raw_components, Mapping):
        raise ValueError("reliability components must be a mapping")
    components: dict[str, torch.Tensor] = {}
    for name, value in raw_components.items():
        tensor = torch.as_tensor(value).clone().cpu()
        if tensor.shape != (count,):
            raise ValueError(f"reliability component {name!r} is malformed")
        tensor[~target_valid] = 0
        components[str(name)] = tensor

    metadata = dict(reliability["metadata"])
    metadata["validity_alignment"] = {
        "contract": "query_free_descriptor_valid_subset_v1",
        "descriptor_cache": str(descriptor_path),
        "descriptor_cache_sha256": sha256_file(descriptor_path),
        "source_reliability_cache": str(reliability_path),
        "source_reliability_cache_sha256": sha256_file(reliability_path),
        "source_valid_rows": int(source_valid.sum()),
        "aligned_valid_rows": int(target_valid.sum()),
        "values_on_retained_rows_changed": False,
    }
    payload = {
        "schema_version": 1,
        "xyz": xyz,
        "valid": target_valid,
        "confidence": confidence,
        "components": components,
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    report = {
        "schema_version": 1,
        "artifact_type": "canonical_primitive_reliability_validity_alignment",
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "source": metadata["validity_alignment"],
        "query_independent": True,
        "benchmark_masks_opened": False,
        "benchmark_metrics_opened": False,
        "target_rgb_opened": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reliability", required=True)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(align(args.reliability, args.descriptor, args.output), indent=2))


if __name__ == "__main__":
    main()
