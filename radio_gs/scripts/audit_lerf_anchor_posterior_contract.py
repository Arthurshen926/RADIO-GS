#!/usr/bin/env python3
"""Audit LERF identity/extent posterior gauge and spatial authority contracts."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_payload,
    write_frozen_json,
)


def _quantiles(value: torch.Tensor, *, maximum_samples: int = 1_000_000) -> dict[str, float]:
    value = value.detach().float().reshape(-1)
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError("posterior audit requires finite non-empty scores")
    # Exact quantiles sort the complete full4 Gaussian-by-query tensor and can
    # exceed the memory budget of a read-only audit.  A fixed regular stride is
    # deterministic, query-independent, and keeps this diagnostic CPU-only.
    if value.numel() > maximum_samples:
        stride = (value.numel() + maximum_samples - 1) // maximum_samples
        sample = value[::stride]
    else:
        sample = value
    points = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])
    result = torch.quantile(sample, points)
    return dict(zip(("min", "p05", "median", "p95", "max"), map(float, result)))


def audit_payload(
    payload: Mapping[str, Any], *, authority_radius_multiplier: float = 4.0,
    maximum_authority_fraction: float = 0.8,
) -> dict[str, Any]:
    """Return a deterministic contract audit for one posterior payload."""

    if authority_radius_multiplier <= 0:
        raise ValueError("authority radius multiplier must be positive")
    score = torch.as_tensor(payload["query_scores"]).float()
    identity = torch.as_tensor(payload["identity_query_scores"]).float()
    xyz = torch.as_tensor(payload["xyz"]).float()
    peaks = torch.as_tensor(payload["peak_rows"]).long()
    radii = torch.as_tensor(payload["local_radii"]).float()
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("posterior metadata differs")
    if score.ndim != 2 or identity.shape != score.shape:
        raise ValueError("posterior and identity score domains differ")
    if xyz.shape != (score.shape[0], 3):
        raise ValueError("posterior xyz row domain differs")
    if peaks.shape != radii.shape or peaks.shape != (score.shape[1],):
        raise ValueError("posterior anchor query domain differs")
    if peaks.numel() and (int(peaks.min()) < 0 or int(peaks.max()) >= xyz.shape[0]):
        raise ValueError("posterior peak row is out of range")
    if not all(bool(torch.isfinite(value).all()) for value in (score, identity, xyz, radii)):
        raise ValueError("posterior audit inputs must be finite")

    threshold = float(metadata["score_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("posterior score threshold is not probabilistic")
    foreground = score >= threshold
    identity_foreground = identity >= threshold
    authority_fractions: list[float] = []
    outside_authority_fractions: list[float] = []
    foreground_counts: list[int] = []
    identity_counts: list[int] = []
    per_query: list[dict[str, Any]] = []
    for column in range(score.shape[1]):
        distance = torch.linalg.vector_norm(xyz - xyz[peaks[column]], dim=1)
        authority = distance <= radii[column] * authority_radius_multiplier
        selected = foreground[:, column]
        selected_count = int(selected.sum())
        outside_count = int((selected & ~authority).sum())
        authority_fraction = float(authority.float().mean())
        outside_fraction = outside_count / max(selected_count, 1)
        identity_count = int(identity_foreground[:, column].sum())
        foreground_counts.append(selected_count)
        identity_counts.append(identity_count)
        authority_fractions.append(authority_fraction)
        outside_authority_fractions.append(outside_fraction)
        per_query.append({
            "query_index": column,
            "peak_row": int(peaks[column]),
            "local_radius": float(radii[column]),
            "authority_fraction": authority_fraction,
            "foreground_count": selected_count,
            "foreground_fraction": selected_count / score.shape[0],
            "identity_foreground_count": identity_count,
            "foreground_outside_authority_count": outside_count,
            "foreground_outside_authority_fraction": outside_fraction,
        })

    warnings: list[str] = []
    if max(authority_fractions, default=0.0) > maximum_authority_fraction:
        warnings.append("extent_authority_is_effectively_scene_global")
    if sum(identity_counts) == 0:
        warnings.append("identity_never_crosses_posterior_decision_threshold")
    if threshold > _quantiles(identity)["max"]:
        warnings.append("posterior_threshold_exceeds_all_identity_scores")
    scene_index = int(metadata.get("scene_canonicalizer_index", -1))
    if scene_index < 0:
        warnings.append("scene_canonicalizer_is_skipped_for_unseen_scene")
    if max(outside_authority_fractions, default=0.0) > 0:
        warnings.append("selected_posterior_exists_outside_extent_authority")

    return {
        "scene": str(payload.get("scene", "unknown")),
        "rows": int(score.shape[0]),
        "queries": int(score.shape[1]),
        "score_threshold": threshold,
        "scene_canonicalizer_index": scene_index,
        "authority_radius_multiplier": authority_radius_multiplier,
        "maximum_allowed_authority_fraction": maximum_authority_fraction,
        "posterior_quantiles": _quantiles(score),
        "identity_quantiles": _quantiles(identity),
        "foreground_count": _quantiles(torch.tensor(foreground_counts, dtype=torch.float32)),
        "identity_foreground_count": _quantiles(torch.tensor(identity_counts, dtype=torch.float32)),
        "authority_fraction": _quantiles(torch.tensor(authority_fractions)),
        "foreground_outside_authority_fraction": _quantiles(
            torch.tensor(outside_authority_fractions)
        ),
        "near_empty_query_count": sum(value <= 1 for value in foreground_counts),
        "warnings": warnings,
        "passed": not warnings,
        "per_query": per_query,
    }


def _posterior_argument(value: str) -> tuple[str, str, str | None]:
    parts = value.split("=", 2)
    if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("posterior must be SCENE=PATH[=SHA256]")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for scene, path, expected in args.posterior:
        if scene in scenes:
            raise ValueError(f"duplicate posterior scene: {scene}")
        payload, digest, source = load_torch_payload(
            path, expected_sha256=expected, label=f"{scene} posterior",
        )
        if not isinstance(payload, Mapping):
            raise ValueError(f"{scene} posterior payload differs")
        if str(payload.get("scene")) != scene:
            raise ValueError(f"{scene} posterior scene identity differs")
        scenes[scene] = audit_payload(
            payload,
            authority_radius_multiplier=args.authority_radius_multiplier,
            maximum_authority_fraction=args.maximum_authority_fraction,
        )
        sources[scene] = {"path": str(source), "sha256": digest}
        del payload
        gc.collect()
    result = {
        "schema": "radio_gs.lerf_anchor_posterior_contract_audit.v1",
        "schema_version": 1,
        "status": "passed" if all(item["passed"] for item in scenes.values()) else "contract_failed",
        "purpose": "diagnostic_only_no_checkpoint_or_threshold_selection",
        "sources": sources,
        "scenes": scenes,
        "summary": {
            "scene_count": len(scenes),
            "passed_scene_count": sum(item["passed"] for item in scenes.values()),
            "warning_counts": {
                warning: sum(warning in item["warnings"] for item in scenes.values())
                for warning in sorted({warning for item in scenes.values() for warning in item["warnings"]})
            },
        },
    }
    output = Path(args.output).resolve()
    write_frozen_json(output, result)
    return {"status": result["status"], "output": file_record(output), "summary": result["summary"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", action="append", required=True, type=_posterior_argument)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authority-radius-multiplier", type=float, default=4.0)
    parser.add_argument("--maximum-authority-fraction", type=float, default=0.8)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
