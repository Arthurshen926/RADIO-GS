#!/usr/bin/env python3
"""Fuse frozen dense DINO completion with exact prompt-responsibility anchors.

Only primitive rows directly and exclusively touched by one official scribble
are changed.  Positive-exclusive rows become one, negative-exclusive rows
become zero, and both unlabeled and conflicting rows preserve the dense source
bit-for-bit.  The construction is parameter-free and opens no target asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
    sha256_file,
    tensor_sha256,
)
from radio_gs.scripts.analyze_nvos_dense_prompt_adjoint_cycle import (
    validate_dino_completion_payload,
)


METHOD_CONTRACT = {
    "dense_source": "frozen_compact_dino_exact_adjoint_primitive_probability",
    "positive_mass": "exact_W_transpose_times_official_positive_scribble",
    "negative_mass": "exact_W_transpose_times_official_negative_scribble",
    "positive_anchor": "positive_mass>0_and_negative_mass==0_sets_probability_1",
    "negative_anchor": "negative_mass>0_and_positive_mass==0_sets_probability_0",
    "conflict_rule": "both_signs_preserve_dense_source_bitwise",
    "unlabeled_rule": "no_sign_preserves_dense_source_bitwise",
    "anchor_threshold": "none_exact_positive_mass_support",
    "graph": "none",
    "connected_selection": "none",
    "threshold": 0.5,
    "target_dependent_tuning": False,
}

TENSOR_KEYS = {
    "source_primitive_probability",
    "primitive_probability",
    "positive_mass",
    "negative_mass",
    "positive_exclusive",
    "negative_exclusive",
    "conflict",
}

ARTIFACT_KEYS = {
    "schema_version",
    "artifact_type",
    "scene_id",
    "method_contract",
    "method_contract_sha256",
    "responsibility_authority_sha256",
    "responsibility_file_sha256",
    "responsibility_tensor_bundle_sha256",
    "source_completion_sha256",
    "source_completion_method_contract_sha256",
    "source_completion_tensor_bundle_sha256",
    "source_primitive_probability_sha256",
    "positive_scribble_sha256",
    "negative_scribble_sha256",
    "tensors",
    "tensor_sha256",
    "tensor_bundle_sha256",
    "target_rgb_opened",
    "target_mask_opened",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def exact_exclusive_anchor_fusion(
    source_probability: torch.Tensor,
    positive_mass: torch.Tensor,
    negative_mass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact exclusive-anchor selector and its disjoint row masks."""

    source = torch.as_tensor(source_probability, device="cpu")
    positive = torch.as_tensor(positive_mass, device="cpu")
    negative = torch.as_tensor(negative_mass, device="cpu")
    if source.ndim != 1 or positive.shape != source.shape or negative.shape != source.shape:
        raise ValueError("source probability and signed masses must be aligned vectors")
    if not source.dtype.is_floating_point or not bool(torch.isfinite(source).all()):
        raise ValueError("source probability must be a finite floating-point vector")
    if bool(((source < 0) | (source > 1)).any()):
        raise ValueError("source probability must be in [0,1]")
    for name, value in (("positive", positive), ("negative", negative)):
        if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} mass must be finite floating point")
        if bool((value < 0).any()):
            raise ValueError(f"{name} mass must be non-negative")
    positive_observed = positive > 0
    negative_observed = negative > 0
    positive_exclusive = positive_observed & ~negative_observed
    negative_exclusive = negative_observed & ~positive_observed
    conflict = positive_observed & negative_observed
    fused = source.clone()
    fused[positive_exclusive] = 1.0
    fused[negative_exclusive] = 0.0
    return fused, positive_exclusive, negative_exclusive, conflict


def validate_exact_anchor_selector_payload(
    payload: object,
    *,
    authority: PromptResponsibilityAuthority,
    expected_responsibility_file_sha256: str,
    expected_responsibility_tensor_bundle_sha256: str,
    expected_source_completion_sha256: str,
    expected_source_primitive_sha256: str,
    expected_primitive_sha256: str | None = None,
) -> torch.Tensor:
    """Fail closed on the authority, derivation, and tensors of a selector."""

    if not isinstance(payload, dict) or set(payload) != ARTIFACT_KEYS:
        raise ValueError("exact-anchor selector artifact schema differs")
    if (
        payload["schema_version"] != 1
        or isinstance(payload["schema_version"], bool)
        or payload["artifact_type"] != "nvos_dino_dense_exact_exclusive_anchor_selector"
        or payload["scene_id"] != authority.scene_id
        or payload["method_contract"] != METHOD_CONTRACT
        or payload["method_contract_sha256"] != _json_sha256(METHOD_CONTRACT)
        or payload["responsibility_authority_sha256"] != authority.digest
        or payload["responsibility_file_sha256"]
        != expected_responsibility_file_sha256
        or payload["responsibility_tensor_bundle_sha256"]
        != expected_responsibility_tensor_bundle_sha256
        or payload["source_completion_sha256"] != expected_source_completion_sha256
        or payload["source_primitive_probability_sha256"]
        != expected_source_primitive_sha256
        or payload["positive_scribble_sha256"]
        != authority.source_sha256["positive_scribble"]
        or payload["negative_scribble_sha256"]
        != authority.source_sha256["negative_scribble"]
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
    ):
        raise ValueError("exact-anchor selector method or authority differs")
    for name in (
        "source_completion_method_contract_sha256",
        "source_completion_tensor_bundle_sha256",
    ):
        value = payload[name]
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise ValueError(f"exact-anchor selector {name} is not a SHA-256")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != TENSOR_KEYS:
        raise ValueError("exact-anchor selector tensor schema differs")
    count = authority.num_gaussians
    float_specs = {
        "source_primitive_probability": torch.float32,
        "primitive_probability": torch.float32,
        "positive_mass": torch.float64,
        "negative_mass": torch.float64,
    }
    for name, dtype in float_specs.items():
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != dtype
            or tuple(value.shape) != (count,)
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"exact-anchor selector tensor {name} is malformed")
    for name in ("source_primitive_probability", "primitive_probability"):
        if bool(((tensors[name] < 0) | (tensors[name] > 1)).any()):
            raise ValueError(f"exact-anchor selector probability {name} is outside [0,1]")
    if bool((tensors["positive_mass"] < 0).any()) or bool(
        (tensors["negative_mass"] < 0).any()
    ):
        raise ValueError("exact-anchor selector masses must be non-negative")
    for name in ("positive_exclusive", "negative_exclusive", "conflict"):
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.bool
            or tuple(value.shape) != (count,)
            or not value.is_contiguous()
        ):
            raise ValueError(f"exact-anchor selector mask {name} is malformed")
    expected, positive, negative, conflict = exact_exclusive_anchor_fusion(
        tensors["source_primitive_probability"],
        tensors["positive_mass"],
        tensors["negative_mass"],
    )
    if (
        not torch.equal(tensors["positive_exclusive"], positive)
        or not torch.equal(tensors["negative_exclusive"], negative)
        or not torch.equal(tensors["conflict"], conflict)
        or not torch.equal(tensors["primitive_probability"], expected)
    ):
        raise ValueError("exact-anchor selector derivation differs")
    actual_digests = {
        name: tensor_sha256(value) for name, value in sorted(tensors.items())
    }
    if (
        payload["tensor_sha256"] != actual_digests
        or payload["tensor_bundle_sha256"] != _json_sha256(actual_digests)
        or actual_digests["source_primitive_probability"]
        != expected_source_primitive_sha256
        or (
            expected_primitive_sha256 is not None
            and actual_digests["primitive_probability"] != expected_primitive_sha256
        )
    ):
        raise ValueError("exact-anchor selector tensor digest differs")
    return tensors["primitive_probability"]


@torch.inference_mode()
def fuse(args: argparse.Namespace) -> dict[str, object]:
    cache_report_path = Path(args.cache_report).resolve()
    cache_report_sha256 = sha256_file(cache_report_path)
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    authority = PromptResponsibilityAuthority.from_dict(cache_report["authority"])
    if authority.scene_id != args.scene_id:
        raise ValueError("scene differs from responsibility authority")
    cache_path = Path(args.cache).resolve()
    if Path(str(cache_report["artifact_path"])).resolve() != cache_path:
        raise ValueError("cache path differs from responsibility report")
    cache = load_prompt_responsibility_cache(
        cache_path,
        expected_authority=authority,
        expected_file_sha256=str(cache_report["file_sha256"]),
    )
    if cache.tensor_bundle_sha256 != str(cache_report["tensor_bundle_sha256"]):
        raise ValueError("cache tensor bundle differs from responsibility report")

    completion_path = Path(args.completion).resolve()
    completion_sha256 = sha256_file(completion_path)
    if completion_sha256 != str(args.expected_completion_sha256):
        raise ValueError("source completion SHA-256 differs")
    completion = torch.load(completion_path, map_location="cpu", weights_only=True)
    source_tensors = validate_dino_completion_payload(
        completion,
        authority=authority,
        expected_responsibility_file_sha256=str(cache_report["file_sha256"]),
        expected_primitive_sha256=str(args.expected_source_primitive_sha256),
    )
    if sha256_file(completion_path) != completion_sha256:
        raise ValueError("source completion changed across trusted load")
    source = source_tensors["primitive_probability"].clone().contiguous()

    positive_path = Path(args.positive_scribble).resolve()
    negative_path = Path(args.negative_scribble).resolve()
    positive_sha256 = sha256_file(positive_path)
    negative_sha256 = sha256_file(negative_path)
    if (
        positive_sha256 != authority.source_sha256["positive_scribble"]
        or negative_sha256 != authority.source_sha256["negative_scribble"]
    ):
        raise ValueError("official scribbles differ from responsibility authority")
    positive_pixels = torch.from_numpy(load_ground_truth_mask(positive_path))
    negative_pixels = torch.from_numpy(load_ground_truth_mask(negative_path))
    shape = (authority.height, authority.width)
    if tuple(positive_pixels.shape) != shape or tuple(negative_pixels.shape) != shape:
        raise ValueError("official scribbles differ from responsibility native shape")
    if bool((positive_pixels & negative_pixels).any()):
        raise ValueError("official positive and negative scribbles overlap")
    positive_mass = cache.adjoint(positive_pixels).weighted_sum.contiguous()
    negative_mass = cache.adjoint(negative_pixels).weighted_sum.contiguous()
    labeled_mass = positive_mass + negative_mass
    tolerance = 1e-12 * torch.maximum(
        torch.ones_like(cache.visible_mass), cache.visible_mass
    )
    if bool((labeled_mass > cache.visible_mass + tolerance).any()):
        raise ValueError("exclusive scribble mass exceeds exact visible mass")
    fused, positive_exclusive, negative_exclusive, conflict = (
        exact_exclusive_anchor_fusion(source, positive_mass, negative_mass)
    )

    # The previous hard_seed_anchor_only_probability path uses the shared 0.20
    # threshold on |(m_pos-m_neg)/visible_mass|.  Compute it for semantics-only
    # comparison; it cannot influence this selector.
    visible = cache.visible_mass
    signed = torch.zeros_like(visible)
    observed = visible > 0
    signed[observed] = (
        (positive_mass[observed] - negative_mass[observed]) / visible[observed]
    )
    prior_anchor = (labeled_mass > 0) & (signed.abs() > 0) & (signed.abs() >= 0.20)

    tensors = {
        "source_primitive_probability": source,
        "primitive_probability": fused.contiguous(),
        "positive_mass": positive_mass,
        "negative_mass": negative_mass,
        "positive_exclusive": positive_exclusive.contiguous(),
        "negative_exclusive": negative_exclusive.contiguous(),
        "conflict": conflict.contiguous(),
    }
    tensor_digests = {
        name: tensor_sha256(value) for name, value in sorted(tensors.items())
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": "nvos_dino_dense_exact_exclusive_anchor_selector",
        "scene_id": args.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": _json_sha256(METHOD_CONTRACT),
        "responsibility_authority_sha256": authority.digest,
        "responsibility_file_sha256": str(cache_report["file_sha256"]),
        "responsibility_tensor_bundle_sha256": str(
            cache_report["tensor_bundle_sha256"]
        ),
        "source_completion_sha256": completion_sha256,
        "source_completion_method_contract_sha256": completion[
            "method_contract_sha256"
        ],
        "source_completion_tensor_bundle_sha256": completion[
            "tensor_bundle_sha256"
        ],
        "source_primitive_probability_sha256": str(
            args.expected_source_primitive_sha256
        ),
        "positive_scribble_sha256": positive_sha256,
        "negative_scribble_sha256": negative_sha256,
        "tensors": tensors,
        "tensor_sha256": tensor_digests,
        "tensor_bundle_sha256": _json_sha256(tensor_digests),
        "target_rgb_opened": False,
        "target_mask_opened": False,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    torch.save(artifact, output_path)
    output_sha256 = sha256_file(output_path)
    frozen = torch.load(output_path, map_location="cpu", weights_only=True)
    validate_exact_anchor_selector_payload(
        frozen,
        authority=authority,
        expected_responsibility_file_sha256=str(cache_report["file_sha256"]),
        expected_responsibility_tensor_bundle_sha256=str(
            cache_report["tensor_bundle_sha256"]
        ),
        expected_source_completion_sha256=completion_sha256,
        expected_source_primitive_sha256=str(args.expected_source_primitive_sha256),
        expected_primitive_sha256=tensor_digests["primitive_probability"],
    )
    if sha256_file(output_path) != output_sha256:
        raise ValueError("exact-anchor selector changed across freeze/reload")

    changed = positive_exclusive | negative_exclusive
    report = {
        "scene_id": args.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": artifact["method_contract_sha256"],
        "cache_report_path": str(cache_report_path),
        "cache_report_sha256": cache_report_sha256,
        "responsibility_file_sha256": str(cache_report["file_sha256"]),
        "source_completion": str(completion_path),
        "source_completion_sha256": completion_sha256,
        "source_primitive_probability_sha256": str(
            args.expected_source_primitive_sha256
        ),
        "output": str(output_path),
        "output_sha256": output_sha256,
        "primitive_probability_sha256": tensor_digests["primitive_probability"],
        "positive_exclusive_rows": int(positive_exclusive.sum()),
        "negative_exclusive_rows": int(negative_exclusive.sum()),
        "conflict_rows_preserving_dense": int(conflict.sum()),
        "unlabeled_rows_preserving_dense": int(
            (~(positive_exclusive | negative_exclusive | conflict)).sum()
        ),
        "changed_rows": int(changed.sum()),
        "changed_fraction": float(changed.double().mean()),
        "mean_absolute_probability_change": float((fused - source).abs().mean()),
        "existing_hard_seed_anchor_only_probability_semantics": {
            "anchor_rule": "abs((positive_mass-negative_mass)/visible_mass)>=0.20",
            "anchor_rows": int(prior_anchor.sum()),
            "intersection_with_exact_exclusive_rows": int((prior_anchor & changed).sum()),
            "exact_exclusive_only_rows": int((changed & ~prior_anchor).sum()),
            "prior_anchor_only_rows": int((prior_anchor & ~changed).sum()),
        },
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-report", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--expected-source-primitive-sha256", required=True)
    parser.add_argument("--positive-scribble", required=True)
    parser.add_argument("--negative-scribble", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(fuse(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
