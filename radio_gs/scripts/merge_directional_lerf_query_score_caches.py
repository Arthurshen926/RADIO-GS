#!/usr/bin/env python3
"""Pool two frozen LERF score caches over a query-free Field-D mixture.

Both branches must have identical geometry, queries, native scales, readout,
and renderer authorities.  The only permitted reduction is elementwise maximum
of the two raw normalized-cosine tensors, matching existential set retrieval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.utils.immutable_artifacts import (
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


CONTRACT = "radio_gs.ours_lerf_direct3d_multiscale_query_scores.v2"
AUTHORITY_CONTRACT = "radio_gs.lerf_multiscale_query_score_authority.v2"


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    if tensor.ndim == 0:
        digest.update(tensor.contiguous().numpy().tobytes(order="C"))
    else:
        for start in range(0, int(tensor.shape[0]), 4096):
            digest.update(
                tensor[start : start + 4096]
                .contiguous()
                .numpy()
                .tobytes(order="C")
            )
    return digest.hexdigest()


def _file_record(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest}


def _validate_source(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("version") != 2 or payload.get("contract") != CONTRACT:
        raise ValueError(f"{label} is not a frozen LERF multiscale cache")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or authority.get(
        "contract"
    ) != AUTHORITY_CONTRACT:
        raise ValueError(f"{label} lacks the shared frozen authority")
    constraints = authority.get("calibration_constraints")
    if not isinstance(constraints, Mapping) or any(
        constraints.get(key) is not False
        for key in (
            "softmax_applied",
            "temperature_applied",
            "peak_normalization_applied",
            "threshold_applied",
            "scale_reduction_applied",
            "benchmark_images_opened",
            "benchmark_annotations_opened",
            "benchmark_masks_opened",
            "benchmark_metrics_opened",
        )
    ):
        raise ValueError(f"{label} is calibrated or benchmark contaminated")
    scores = payload.get("query_scores")
    valid = payload.get("valid")
    xyz = payload.get("xyz")
    if (
        not isinstance(scores, torch.Tensor)
        or scores.dtype != torch.float16
        or scores.ndim != 3
        or scores.shape[1] != 3
        or not bool(torch.isfinite(scores).all())
    ):
        raise ValueError(f"{label} score tensor differs")
    if (
        not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
    ):
        raise ValueError(f"{label} valid rows differ")
    if (
        not isinstance(xyz, torch.Tensor)
        or not xyz.is_floating_point()
        or xyz.shape != (scores.shape[0], 3)
        or not bool(torch.isfinite(xyz).all())
    ):
        raise ValueError(f"{label} geometry rows differ")


def _require_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    scalar_keys = (
        "query_ids",
        "scale_ids",
        "scale_radii_m",
        "geometry_fingerprint",
        "readout_checkpoint_sha256",
        "renderer_geometry_checkpoint_sha256",
    )
    for key in scalar_keys:
        if left.get(key) != right.get(key):
            raise ValueError(f"directional score sources disagree on {key}")
    for key in ("xyz", "valid"):
        if not torch.equal(torch.as_tensor(left[key]), torch.as_tensor(right[key])):
            raise ValueError(f"directional score sources disagree on {key}")
    for key in ("scale_axis", "query_axis"):
        if left["authority"].get(key) != right["authority"].get(key):
            raise ValueError(f"directional score authorities disagree on {key}")


def merge(args: argparse.Namespace) -> dict[str, Any]:
    mode0, mode0_sha, mode0_path = load_torch_mapping(
        args.mode_0_cache,
        expected_sha256=args.expected_mode_0_cache_sha256,
        map_location="cpu",
        label="Field-D LERF mode-0 score cache",
    )
    mode1, mode1_sha, mode1_path = load_torch_mapping(
        args.mode_1_cache,
        expected_sha256=args.expected_mode_1_cache_sha256,
        map_location="cpu",
        label="Field-D LERF mode-1 score cache",
    )
    _validate_source(mode0, label="mode 0")
    _validate_source(mode1, label="mode 1")
    _require_equal(mode0, mode1)
    bundle_path = Path(args.field_bundle).expanduser().resolve()
    if sha256_file(bundle_path) != args.expected_field_bundle_sha256:
        raise ValueError("Field-D bundle SHA-256 differs")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("contract") != "query_free_two_mode_field_then_frozen_readout_max_v1":
        raise ValueError("Field-D bundle contract differs")
    if bundle.get("pooling") != "elementwise_max_raw_cosine_after_identical_frozen_readout":
        raise ValueError("Field-D bundle pooling differs")
    if bundle.get("passed") is not True:
        raise ValueError("Field-D bundle did not pass its label-free gate")
    expected_mode_shas = [item["sha256"] for item in bundle.get("modes", [])]
    observed_mode_shas = [
        mode0.get("field_checkpoint_sha256"),
        mode1.get("field_checkpoint_sha256"),
    ]
    if expected_mode_shas != observed_mode_shas:
        raise ValueError("score cache fields do not match the Field-D bundle")

    scores = torch.maximum(mode0["query_scores"], mode1["query_scores"]).contiguous()
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("immutable pooled Field-D score output already exists")
    authority = copy.deepcopy(dict(mode0["authority"]))
    bundle_sha = args.expected_field_bundle_sha256
    authority["score_semantics"] = (
        "raw_independent_normalized_cosine_then_max_over_query_free_field_modes"
    )
    authority["score_formula"] = (
        "max_mode(l2_normalize(descriptor_mode) @ l2_normalize(text).T)"
    )
    authority["query_scores_sha256"] = _tensor_sha256(scores)
    authority["geometry_axis"]["field_checkpoint_sha256"] = bundle_sha
    authority["source_artifacts"]["field_checkpoint"] = _file_record(
        bundle_path, bundle_sha
    )
    authority["source_artifacts"]["directional_mode_0_score_cache"] = _file_record(
        mode0_path, mode0_sha
    )
    authority["source_artifacts"]["directional_mode_1_score_cache"] = _file_record(
        mode1_path, mode1_sha
    )
    authority["source_artifacts"]["directional_pooler_source"] = _file_record(
        Path(__file__).resolve(), sha256_file(Path(__file__).resolve())
    )
    authority["directional_pooling"] = {
        "contract": bundle["contract"],
        "operator": "elementwise_max",
        "axis": "query_free_field_mode",
        "mode_count": 2,
        "mode_field_checkpoint_sha256": expected_mode_shas,
        "changes_query_order": False,
        "changes_native_scales": False,
        "uses_threshold": False,
    }
    payload = {
        **{
            key: mode0[key]
            for key in (
                "version",
                "contract",
                "query_ids",
                "scale_ids",
                "scale_radii_m",
                "xyz",
                "valid",
                "geometry_fingerprint",
                "readout_checkpoint_sha256",
                "renderer_geometry_checkpoint_sha256",
            )
        },
        "query_scores": scores,
        "field_checkpoint_sha256": bundle_sha,
        "authority": authority,
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema_version": "field_d_lerf_score_pool_receipt_v1",
        "status": "complete_parameter_free_directional_pooling",
        "output": _file_record(output, sha256_file(output)),
        "field_bundle": _file_record(bundle_path, bundle_sha),
        "mode_score_caches": [
            _file_record(mode0_path, mode0_sha),
            _file_record(mode1_path, mode1_sha),
        ],
        "score_shape": list(scores.shape),
        "score_sha256": authority["query_scores_sha256"],
        "pooling": "elementwise_max_raw_cosine",
        "benchmark_images_opened": False,
        "benchmark_annotations_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_metrics_opened": False,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_frozen_json(report_path, report)
    return {**report, "receipt": str(report_path), "receipt_sha256": sha256_file(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-0-cache", required=True)
    parser.add_argument("--expected-mode-0-cache-sha256", required=True)
    parser.add_argument("--mode-1-cache", required=True)
    parser.add_argument("--expected-mode-1-cache-sha256", required=True)
    parser.add_argument("--field-bundle", required=True)
    parser.add_argument("--expected-field-bundle-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
