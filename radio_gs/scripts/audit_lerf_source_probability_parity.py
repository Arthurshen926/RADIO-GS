#!/usr/bin/env python3
"""Fail-closed, label-free parity gate for LERF streamed source probabilities."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: str | Path, expected_sha256: str, label: str) -> Mapping:
    source = Path(path).expanduser().resolve()
    observed = _sha256(source)
    if observed != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 differs")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} is not a mapping")
    return payload


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.float().reshape(-1)
    y = right.float().reshape(-1)
    if x.shape != y.shape or x.numel() == 0:
        raise ValueError("parity tensors are not aligned and nonempty")
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().mean().sqrt() * y.square().mean().sqrt()
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(x, y) else 0.0
    return float((x * y).mean() / denominator)


def audit_source_probability_parity(
    *,
    positive: Mapping,
    negative: Mapping,
    control: Mapping,
    minimum_pearson: float = 0.9999,
    maximum_mean_absolute_error: float = 5e-4,
    maximum_absolute_error: float = 2e-3,
    logit_scale: float = 10.0,
) -> dict[str, object]:
    for key in ("query_scores", "query_ids", "scale_ids", "scale_radii_m", "xyz", "valid"):
        if key not in positive or key not in negative:
            raise ValueError(f"accepted score cache lacks {key}")
    for key in ("features", "xyz", "valid", "metadata"):
        if key not in control:
            raise ValueError(f"streamed control lacks {key}")
    if not torch.equal(torch.as_tensor(positive["xyz"]), torch.as_tensor(negative["xyz"])):
        raise ValueError("accepted positive/negative geometry differs")
    if not torch.equal(torch.as_tensor(positive["xyz"]), torch.as_tensor(control["xyz"])):
        raise ValueError("accepted/control geometry differs")
    positive_valid = torch.as_tensor(positive["valid"]).bool()
    if not torch.equal(positive_valid, torch.as_tensor(negative["valid"]).bool()):
        raise ValueError("accepted positive/negative valid mask differs")
    if not torch.equal(positive_valid, torch.as_tensor(control["valid"]).bool()):
        raise ValueError("accepted/control valid mask differs")
    if list(positive["scale_ids"]) != list(negative["scale_ids"]):
        raise ValueError("accepted positive/negative scale IDs differ")
    if list(positive["scale_radii_m"]) != list(negative["scale_radii_m"]):
        raise ValueError("accepted positive/negative scale radii differ")
    metadata = control["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("streamed control metadata differs")
    if list(metadata.get("query_names", [])) != list(positive["query_ids"]):
        raise ValueError("accepted/control query axis differs")
    if list(metadata.get("scale_radii_m", [])) != list(positive["scale_radii_m"]):
        raise ValueError("accepted/control scale axis differs")

    positive_scores = torch.as_tensor(positive["query_scores"]).float()
    negative_scores = torch.as_tensor(negative["query_scores"]).float()
    observed = torch.as_tensor(control["features"]).float()
    expected = torch.sigmoid(
        float(logit_scale)
        * (positive_scores - negative_scores.amax(dim=-1, keepdim=True))
    )
    if observed.shape != expected.shape:
        raise ValueError("accepted/control probability shape differs")
    if bool(observed[~positive_valid].ne(0).any()):
        raise ValueError("streamed control is nonzero outside its valid mask")
    expected_valid = expected[positive_valid]
    observed_valid = observed[positive_valid]
    absolute = (observed_valid - expected_valid).abs()
    pearson = _pearson(observed_valid, expected_valid)
    mean_absolute_error = float(absolute.mean())
    maximum_error = float(absolute.max())
    passed = (
        pearson >= float(minimum_pearson)
        and mean_absolute_error <= float(maximum_mean_absolute_error)
        and maximum_error <= float(maximum_absolute_error)
    )
    return {
        "status": "pass" if passed else "fail_closed",
        "passed": passed,
        "valid_primitives": int(positive_valid.sum()),
        "scales": len(positive["scale_ids"]),
        "queries": len(positive["query_ids"]),
        "metrics": {
            "pearson": pearson,
            "mean_absolute_error": mean_absolute_error,
            "maximum_absolute_error": maximum_error,
        },
        "thresholds": {
            "minimum_pearson": float(minimum_pearson),
            "maximum_mean_absolute_error": float(maximum_mean_absolute_error),
            "maximum_absolute_error": float(maximum_absolute_error),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("positive", "negative", "control"):
        parser.add_argument(f"--{name}", required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-pearson", type=float, default=0.9999)
    parser.add_argument("--maximum-mean-absolute-error", type=float, default=5e-4)
    parser.add_argument("--maximum-absolute-error", type=float, default=2e-3)
    args = parser.parse_args()
    inputs = {
        name: {
            "path": str(Path(getattr(args, name)).expanduser().resolve()),
            "sha256": str(getattr(args, f"{name}_sha256")),
        }
        for name in ("positive", "negative", "control")
    }
    payloads = {
        name: _load(record["path"], record["sha256"], name)
        for name, record in inputs.items()
    }
    result = audit_source_probability_parity(
        positive=payloads["positive"],
        negative=payloads["negative"],
        control=payloads["control"],
        minimum_pearson=args.minimum_pearson,
        maximum_mean_absolute_error=args.maximum_mean_absolute_error,
        maximum_absolute_error=args.maximum_absolute_error,
    )
    report = {
        "schema_version": 1,
        "artifact_type": "lerf_source_probability_parity_gate_v1",
        "inputs": inputs,
        "formula": "sigmoid(10*(accepted_positive_cosine-max_accepted_negative_cosine))",
        "alignment": "exact_xyz_valid_query_and_scale_axes",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_metrics_opened": False,
        **result,
    }
    write_frozen_json(args.output, report)
    if not bool(result["passed"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
