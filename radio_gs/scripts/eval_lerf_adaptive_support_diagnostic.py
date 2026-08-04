#!/usr/bin/env python3
"""Opt-in target-blind LERF3D support calibration diagnostic.

This wrapper leaves ``eval_lerf_direct_3d_selection.py`` unchanged and invokes
its frozen ``vala_repo_3d`` preset. The only opt-in substitution is primitive
membership selection from the already frozen, authority-validated score field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

from radio_gs.querying.adaptive_support import select_adaptive_otsu_support
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen_evaluator
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


MODES = {
    "otsu2": 1,
    "recursive_upper_otsu3": 3,
}

_EXPECTED_CALIBRATION_CONSTRAINTS = {
    "softmax_applied": False,
    "temperature_applied": False,
    "peak_normalization_applied": False,
    "threshold_applied": False,
    "scale_reduction_applied": False,
    "benchmark_images_opened": False,
    "benchmark_annotations_opened": False,
    "benchmark_masks_opened": False,
    "benchmark_metrics_opened": False,
}
_EXPECTED_SCORE_SEMANTICS = "raw_independent_normalized_cosine"
_EXPECTED_SCORE_DTYPE = "torch.float16"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authority_tensor_sha256(value: torch.Tensor) -> str:
    """Match the frozen multiscale cache's dtype/shape-aware tensor hash."""

    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.layout != torch.strided:
        raise ValueError("authority tensor hashing requires strided tensors")
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


def build_frozen_evaluator_argv(args: argparse.Namespace) -> list[str]:
    """Construct the closed diagnostic invocation of the frozen evaluator."""

    return [
        "eval_lerf_direct_3d_selection.py",
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--scene",
        args.scene,
        "--protocol_preset",
        "vala_repo_3d",
        "--label_dir",
        args.label_dir,
        "--output_dir",
        args.output_dir,
        "--summary_head_weights",
        args.summary_head_weights,
        "--text_embedding_cache",
        args.text_embedding_cache,
        "--canonical_embedding_cache",
        args.canonical_embedding_cache,
        "--ours_multiscale_query_score_cache",
        args.ours_multiscale_query_score_cache,
        "--gpu",
        str(int(args.gpu)),
    ]


def _load_cache_inputs(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Ours multiscale query-score cache must be a mapping")
    if payload.get("contract") != frozen_evaluator.OURS_MULTISCALE_QUERY_SCORE_CACHE_CONTRACT:
        raise ValueError("Ours multiscale query-score cache contract mismatch")
    valid = payload.get("valid")
    scores = payload.get("query_scores")
    xyz = payload.get("xyz")
    query_ids = tuple(str(value) for value in payload.get("query_ids", []))
    if (
        not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or not isinstance(scores, torch.Tensor)
        or scores.ndim != 3
        or not isinstance(xyz, torch.Tensor)
        or xyz.shape != (int(scores.shape[0]), 3)
        or valid.shape != (int(scores.shape[0]),)
        or not bool(valid.any())
    ):
        raise ValueError("Ours multiscale cache has invalid row-aligned score inputs")
    if len(query_ids) != int(scores.shape[2]) or len(set(query_ids)) != len(query_ids):
        raise ValueError("Ours multiscale cache has an invalid query axis")
    renderer_geometry_checkpoint_sha256 = str(
        payload.get("renderer_geometry_checkpoint_sha256", "")
    )
    # Reuse the frozen evaluator's complete structural validator before any
    # benchmark label can be opened.  Passing the payload xyz as the expected
    # geometry still checks it against the independently stored geometry
    # fingerprint, authority axes, row count, scales, and checkpoint bindings;
    # run() separately binds the renderer checkpoint file bytes below.
    frozen_evaluator.validate_ours_multiscale_query_score_cache(
        payload,
        expected_xyz=xyz,
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=(
            renderer_geometry_checkpoint_sha256
        ),
    )
    authority = payload.get("authority")
    if (
        not isinstance(authority, dict)
        or authority.get("contract")
        != frozen_evaluator.OURS_MULTISCALE_QUERY_SCORE_AUTHORITY_CONTRACT
    ):
        raise ValueError("Ours multiscale cache authority contract mismatch")
    query_scores_sha256 = authority_tensor_sha256(scores)
    valid_sha256 = authority_tensor_sha256(valid)
    if authority.get("query_scores_sha256") != query_scores_sha256:
        raise ValueError("Ours multiscale cache query-score authority hash mismatch")
    geometry_axis = authority.get("geometry_axis")
    query_axis = authority.get("query_axis")
    if not isinstance(geometry_axis, dict) or geometry_axis.get("valid_sha256") != valid_sha256:
        raise ValueError("Ours multiscale cache valid-mask authority hash mismatch")
    if not isinstance(query_axis, dict) or tuple(query_axis.get("ids", [])) != query_ids:
        raise ValueError("Ours multiscale cache authority query order mismatch")
    if query_axis.get("order_sha256") != canonical_json_sha256(list(query_ids)):
        raise ValueError("Ours multiscale cache authority query-order hash mismatch")
    if authority.get("calibration_constraints") != _EXPECTED_CALIBRATION_CONSTRAINTS:
        raise ValueError("Ours multiscale cache calibration constraints differ")
    if authority.get("score_semantics") != _EXPECTED_SCORE_SEMANTICS:
        raise ValueError("Ours multiscale cache score semantics differ")
    if authority.get("score_dtype") != _EXPECTED_SCORE_DTYPE:
        raise ValueError("Ours multiscale cache declared score dtype differs")
    if scores.dtype != torch.float16:
        raise ValueError("Ours multiscale cache score tensor dtype differs")
    return {
        "query_scores": scores.detach().float().cpu(),
        "xyz": xyz.detach().float().cpu(),
        "valid": valid.detach().cpu(),
        "query_ids": query_ids,
        "query_scores_sha256": query_scores_sha256,
        "valid_sha256": valid_sha256,
        "renderer_geometry_checkpoint_sha256": (
            renderer_geometry_checkpoint_sha256
        ),
        "field_checkpoint_sha256": str(payload.get("field_checkpoint_sha256", "")),
        "readout_checkpoint_sha256": str(payload.get("readout_checkpoint_sha256", "")),
    }


def precompute_adaptive_membership(cache: dict, *, otsu_stages: int) -> dict:
    """Freeze adaptive membership before the evaluator can open labels or RGB."""

    readout = frozen_evaluator.vala_multiscale_knn_peak_select_scores(
        cache["query_scores"],
        cache["xyz"],
        k=10,
        valid_mask=cache["valid"],
    )
    selection = select_adaptive_otsu_support(
        readout.scores,
        cache["valid"],
        otsu_stages=int(otsu_stages),
    )
    membership = selection.selected.bool()
    return {
        "selection": selection,
        "processed_scores_sha256": frozen_evaluator.tensor_sha256_float32(
            readout.scores
        ),
        "membership_sha256": authority_tensor_sha256(membership),
    }


def _result_path(output_dir: str | Path, scene: str) -> Path:
    return Path(output_dir) / scene / "lerf_direct_3d_selection_results.json"


def run(args: argparse.Namespace) -> Path:
    stages = MODES[args.calibration_mode]
    # This happens before frozen_evaluator.main(), which is the first call that
    # opens benchmark labels, scene cameras, RGB, or model/render state.
    cache = _load_cache_inputs(args.ours_multiscale_query_score_cache)
    renderer_checkpoint_sha256 = sha256_file(args.checkpoint)
    if renderer_checkpoint_sha256 != cache["renderer_geometry_checkpoint_sha256"]:
        raise ValueError(
            "renderer checkpoint differs from the query-score cache authority"
        )
    precomputed = precompute_adaptive_membership(cache, otsu_stages=stages)
    selection = precomputed["selection"]
    calls = []
    original_selector = frozen_evaluator.select_gaussians_from_scores

    def adaptive_selector(scores, spec, *, min_select=1):
        if spec != frozen_evaluator.SelectionSpec(
            "score_threshold", frozen_evaluator.OURS_VALA_MASK_THRESHOLD
        ):
            raise ValueError(
                "adaptive support requires the frozen singleton score-threshold spec"
            )
        if int(min_select) != 0:
            raise ValueError("adaptive support requires frozen min_select=0")
        if calls:
            raise RuntimeError("adaptive support selector must be called exactly once")
        actual_scores_sha256 = frozen_evaluator.tensor_sha256_float32(scores)
        if actual_scores_sha256 != precomputed["processed_scores_sha256"]:
            raise ValueError(
                "frozen evaluator processed scores differ from precomputed adaptive inputs"
            )
        if tuple(scores.shape) != tuple(selection.selected.shape):
            raise ValueError("precomputed adaptive membership shape differs")
        calls.append(actual_scores_sha256)
        return selection.selected

    previous_argv = sys.argv
    frozen_evaluator.select_gaussians_from_scores = adaptive_selector
    try:
        sys.argv = build_frozen_evaluator_argv(args)
        frozen_evaluator.main()
    finally:
        frozen_evaluator.select_gaussians_from_scores = original_selector
        sys.argv = previous_argv

    if len(calls) != 1:
        raise RuntimeError(f"adaptive selector call count differs: {len(calls)}")
    path = _result_path(args.output_dir, args.scene)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adaptive_support_diagnostic"] = {
        "status": "diagnostic_only_not_formal",
        "calibration_mode": args.calibration_mode,
        "otsu_stages": stages,
        "score_inputs": "frozen post-kNN10, per-level minmax, 2x-1 clipped primitive scores",
        "allowed_inputs": ["canonical primitive query scores", "authority-bound valid mask"],
        "forbidden_inputs": ["target RGB", "benchmark GT", "rendered target masks"],
        "membership_precomputed_before_frozen_evaluator": True,
        "processed_scores_sha256": precomputed["processed_scores_sha256"],
        "membership_sha256": precomputed["membership_sha256"],
        "thresholds": [float(value) for value in selection.thresholds.tolist()],
        "selected_counts": [int(value) for value in selection.selected_counts.tolist()],
        "valid_primitives": int(selection.valid_count),
        "query_score_cache": str(Path(args.ours_multiscale_query_score_cache).resolve()),
        "query_score_cache_sha256": sha256_file(args.ours_multiscale_query_score_cache),
        "query_scores_authority_sha256": cache["query_scores_sha256"],
        "valid_authority_sha256": cache["valid_sha256"],
        "query_ids": list(cache["query_ids"]),
        "renderer_geometry_checkpoint_sha256": renderer_checkpoint_sha256,
        "frozen_evaluator_source": str(Path(frozen_evaluator.__file__).resolve()),
        "frozen_evaluator_source_sha256": sha256_file(frozen_evaluator.__file__),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--scene", required=True, choices=list(frozen_evaluator.LERF_OVS_SCENES)
    )
    parser.add_argument("--ours_multiscale_query_score_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_mode", choices=sorted(MODES), required=True)
    parser.add_argument(
        "--label_dir", default="/mnt/pool/sqy/3d_understanding/lerf_ovs/label"
    )
    parser.add_argument(
        "--summary_head_weights", default="checkpoints/siglip2_summary_head.pth"
    )
    parser.add_argument(
        "--text_embedding_cache",
        default="checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt",
    )
    parser.add_argument(
        "--canonical_embedding_cache",
        default="checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt",
    )
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = run(args)
    print(path)


if __name__ == "__main__":
    main()
