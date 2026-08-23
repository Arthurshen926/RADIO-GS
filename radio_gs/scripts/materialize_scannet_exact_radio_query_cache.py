#!/usr/bin/env python3
"""Materialize an observed-exact ScanNet RADIO teacher under the D512 readout.

The exact marginal cache has no semantic value on unobserved Gaussian rows.
For a same-domain comparison, those rows inherit the frozen Method-v1 D512
descriptor bit-for-bit.  Observed rows alone are replaced by the official
SigLIP2 summary of the exact 1280-D RADIO MPR teacher.  This makes the arm a
causal ``exact where observed, D512 totality elsewhere`` intervention rather
than silently turning missing observations into a background class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.five_benchmark_method_v1 import METHOD_ID
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


@torch.inference_mode()
def project_observed_exact_rows(
    exact_radio: torch.Tensor,
    observed: torch.Tensor,
    fallback: torch.Tensor,
    summary_head: torch.nn.Module,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    """Replace fallback descriptors only on rows with exact observations."""

    raw = torch.as_tensor(exact_radio).detach().cpu()
    valid = torch.as_tensor(observed).detach().cpu().bool().reshape(-1)
    output = torch.as_tensor(fallback).detach().cpu().float().clone()
    if raw.ndim != 2 or raw.shape[1] != 1280:
        raise ValueError("exact RADIO rows must have shape [N,1280]")
    if valid.shape != (raw.shape[0],) or output.shape != (raw.shape[0], 1536):
        raise ValueError("exact, observed, and fallback row domains differ")
    if int(chunk_size) <= 0 or not bool(torch.isfinite(raw[valid].float()).all()):
        raise ValueError("exact RADIO rows or chunk size are invalid")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("fallback descriptors contain NaN or infinity")

    indices = torch.nonzero(valid, as_tuple=False).flatten()
    head = summary_head.to(device).eval()
    try:
        dtype = next(head.parameters()).dtype
    except StopIteration:
        dtype = torch.float32
    for start in range(0, int(indices.numel()), int(chunk_size)):
        selected = indices[start : start + int(chunk_size)]
        projected = head(raw[selected].to(device=device, dtype=dtype).unsqueeze(0))
        projected = F.normalize(projected.squeeze(0).float(), dim=-1, eps=1e-8)
        output[selected] = projected.cpu()
    return F.normalize(output, dim=-1, eps=1e-8).half().contiguous()


def materialize(args: argparse.Namespace) -> dict[str, object]:
    exact, exact_sha, exact_path = load_sha_bound_project_checkpoint_mapping(
        args.exact_radio_mpr,
        expected_sha256=args.expected_exact_radio_mpr_sha256,
        map_location="cpu",
        label="ScanNet exact RADIO MPR teacher",
    )
    fallback, fallback_sha, fallback_path = load_sha_bound_project_checkpoint_mapping(
        args.d512_query_cache,
        expected_sha256=args.expected_d512_query_cache_sha256,
        map_location="cpu",
        label="ScanNet D512 query-cache fallback",
    )
    xyz = torch.as_tensor(exact.get("xyz")).detach().cpu().float().contiguous()
    fallback_xyz = torch.as_tensor(fallback.get("xyz")).detach().cpu().float().contiguous()
    raw = torch.as_tensor(exact.get("features")).detach().cpu()
    observed = torch.as_tensor(exact.get("valid")).detach().cpu().bool().reshape(-1)
    fallback_features = torch.as_tensor(
        fallback.get("summary_features", fallback.get("features"))
    ).detach().cpu()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not torch.equal(xyz, fallback_xyz):
        raise ValueError("exact teacher and D512 fallback xyz rows differ")
    if raw.shape != (xyz.shape[0], 1280) or observed.shape != (xyz.shape[0],):
        raise ValueError("exact teacher tensor domain differs")
    if fallback_features.shape != (xyz.shape[0], 1536):
        raise ValueError("D512 fallback descriptor domain differs")
    if not bool(observed.any()):
        raise ValueError("exact teacher has no observed Gaussian row")

    head_path = Path(args.summary_head_weights).expanduser().resolve(strict=True)
    head_record = file_record(head_path)
    if head_record["sha256"] != args.expected_summary_head_sha256:
        raise ValueError("official SigLIP2 summary-head SHA256 differs")
    head = SigLIP2SummaryHead.from_extracted_weights(str(head_path))
    features = project_observed_exact_rows(
        raw,
        observed,
        fallback_features,
        head,
        device=torch.device(args.device),
        chunk_size=int(args.chunk_size),
    )
    output = Path(args.output).expanduser().resolve()
    payload = {
        "xyz": xyz,
        "summary_features": features,
        "features": features,
        # The cache is total because unobserved rows explicitly inherit D512.
        "valid": torch.ones(xyz.shape[0], dtype=torch.bool),
        "exact_observed": observed,
        "metadata": {
            "schema_version": 1,
            "artifact_type": "radio_gs_method_v1_primitive_query_cache",
            "method_id": METHOD_ID,
            "feature_space": "official_siglip2_summary_descriptor_per_primitive",
            "construction": (
                "observed_exact_radio_mpr_then_frozen_official_summary_head_"
                "with_d512_totality_fallback_then_l2"
            ),
            "exact_radio_mpr": {"path": str(exact_path), "sha256": exact_sha},
            "d512_query_fallback": {
                "path": str(fallback_path),
                "sha256": fallback_sha,
            },
            "summary_head": head_record,
            "query_independent": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "text_queries_opened": False,
            "postprocessing": "none",
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "status": "complete",
        "cache": file_record(output),
        "rows": int(xyz.shape[0]),
        "exact_observed_rows": int(observed.sum()),
        "exact_observed_ratio": float(observed.float().mean()),
        "totality_fallback_rows": int((~observed).sum()),
        "exact_radio_mpr": {"path": str(exact_path), "sha256": exact_sha},
        "d512_query_fallback": {"path": str(fallback_path), "sha256": fallback_sha},
        "summary_head": head_record,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-radio-mpr", required=True)
    parser.add_argument("--expected-exact-radio-mpr-sha256", required=True)
    parser.add_argument("--d512-query-cache", required=True)
    parser.add_argument("--expected-d512-query-cache-sha256", required=True)
    parser.add_argument("--summary-head-weights", required=True)
    parser.add_argument("--expected-summary-head-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(materialize(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
