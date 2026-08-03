#!/usr/bin/env python3
"""Target-blind DINO dense prompt completion and exact W.T/W cycle audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.evaluation.promptable_segmentation import (
    load_ground_truth_mask,
    resize_mask_nearest,
)
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
    sha256_file,
    tensor_sha256,
)
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint


METHOD_CONTRACT = {
    "semantic_space": "official_C-RADIOv4_dino_v3_7b_feature_projection",
    "scribble_resampling": "nearest_native_to_canonical_prompt_grid",
    "dense_rule": "class_count_normalized_all_scribble_logmeanexp_cosine",
    "temperature": 0.07,
    "native_resampling": "bilinear_align_corners_false",
    "strong_unary": "overwrite_official_positive_pixels_to_1_negative_pixels_to_0",
    "adjoint": "u=(W.T@y)/(W.T@1)_visible",
    "cycle": "y_cycle=(W@u)/(W@1)_supported",
    "threshold": 0.5,
    "target_dependent_tuning": False,
}

BASE_TENSOR_KEYS = {
    "low_posterior",
    "dense_native",
    "anchored_native",
    "primitive_probability",
    "primitive_visible",
    "cycle_native",
    "cycle_supported",
}

COMMON_SELECTOR_KEYS = {
    "schema_version",
    "artifact_type",
    "scene_id",
    "method_contract",
    "method_contract_sha256",
    "responsibility_authority_sha256",
    "responsibility_file_sha256",
    "prompt_feature_sha256",
    "radio_checkpoint_sha256",
    "tensors",
    "target_rgb_opened",
    "target_mask_opened",
    "tensor_sha256",
    "tensor_bundle_sha256",
}


def validate_exact_adjoint_selector_payload(
    payload: object,
    *,
    authority: PromptResponsibilityAuthority,
    expected_responsibility_file_sha256: str,
    artifact_type: str,
    method_contract: dict[str, object],
    extra_keys: frozenset[str] = frozenset(),
    expected_primitive_sha256: str | None = None,
) -> dict[str, torch.Tensor]:
    """Validate the common fail-closed exact-adjoint selector envelope."""

    expected_keys = COMMON_SELECTOR_KEYS | set(extra_keys)
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("exact-adjoint selector artifact schema differs")
    if (
        payload["schema_version"] != 1
        or isinstance(payload["schema_version"], bool)
        or payload["artifact_type"] != artifact_type
        or payload["scene_id"] != authority.scene_id
        or payload["method_contract"] != method_contract
        or payload["method_contract_sha256"] != _json_sha256(method_contract)
        or payload["responsibility_authority_sha256"] != authority.digest
        or payload["responsibility_file_sha256"]
        != expected_responsibility_file_sha256
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
    ):
        raise ValueError("exact-adjoint selector method or authority differs")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != BASE_TENSOR_KEYS:
        raise ValueError("exact-adjoint selector tensor schema differs")
    height, width = int(authority.height), int(authority.width)
    num_gaussians = int(authority.num_gaussians)
    expected_specs = {
        "dense_native": (torch.float32, (height, width)),
        "anchored_native": (torch.float32, (height, width)),
        "primitive_probability": (torch.float32, (num_gaussians,)),
        "primitive_visible": (torch.bool, (num_gaussians,)),
        "cycle_native": (torch.float32, (height, width)),
        "cycle_supported": (torch.bool, (height, width)),
    }
    low = tensors["low_posterior"]
    if (
        not torch.is_tensor(low)
        or low.device.type != "cpu"
        or low.dtype != torch.float32
        or low.ndim != 2
        or min(map(int, low.shape)) <= 0
        or not low.is_contiguous()
    ):
        raise ValueError("exact-adjoint low posterior tensor is malformed")
    for name, (dtype, shape) in expected_specs.items():
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != dtype
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise ValueError(f"exact-adjoint selector tensor {name} is malformed")
    for name in (
        "low_posterior",
        "dense_native",
        "anchored_native",
        "primitive_probability",
        "cycle_native",
    ):
        value = tensors[name]
        if not bool(torch.isfinite(value).all()) or bool(
            ((value < 0.0) | (value > 1.0)).any()
        ):
            raise ValueError(f"exact-adjoint probability {name} is outside [0,1]")
    stored_digests = payload["tensor_sha256"]
    if not isinstance(stored_digests, dict) or set(stored_digests) != BASE_TENSOR_KEYS:
        raise ValueError("exact-adjoint selector tensor digest schema differs")
    actual_digests = {
        name: tensor_sha256(value) for name, value in sorted(tensors.items())
    }
    if stored_digests != actual_digests:
        raise ValueError("exact-adjoint selector tensor digest differs")
    if payload["tensor_bundle_sha256"] != _json_sha256(actual_digests):
        raise ValueError("exact-adjoint selector tensor bundle digest differs")
    if (
        expected_primitive_sha256 is not None
        and actual_digests["primitive_probability"] != expected_primitive_sha256
    ):
        raise ValueError("exact-adjoint primitive probability differs")
    return tensors


def validate_dino_completion_payload(
    payload: object,
    *,
    authority: PromptResponsibilityAuthority,
    expected_responsibility_file_sha256: str,
    expected_primitive_sha256: str | None = None,
) -> dict[str, torch.Tensor]:
    """Fail closed on every scientific field in a base DINO selector."""

    return validate_exact_adjoint_selector_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256=expected_responsibility_file_sha256,
        artifact_type="nvos_dino_dense_prompt_exact_adjoint_cycle",
        method_contract=METHOD_CONTRACT,
        expected_primitive_sha256=expected_primitive_sha256,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _find_feature_record(render_manifest: dict, *, frame_id: str) -> dict:
    matches = [
        record
        for record in render_manifest.get("outputs", [])
        if str(record.get("frame_id")) == str(frame_id) and record.get("role") == "prompt"
    ]
    if len(matches) != 1:
        raise ValueError("render manifest must contain exactly one bound prompt feature")
    return matches[0]


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> torch.Tensor:
    # Reuse the frozen prompt protocol's Pillow-NEAREST coordinate convention;
    # torch.interpolate has different half-pixel sampling for this downscale.
    return torch.from_numpy(resize_mask_nearest(mask, shape).astype(bool, copy=False))


def _dense_logmeanexp_posterior(
    features: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    channels, height, width = features.shape
    rows = F.normalize(features.permute(1, 2, 0).reshape(-1, channels).float(), dim=1)
    pos_rows = rows[positive.reshape(-1)]
    neg_rows = rows[negative.reshape(-1)]
    if pos_rows.numel() == 0 or neg_rows.numel() == 0:
        raise ValueError("both positive and negative scribbles need grid support")
    pos_logit = torch.logsumexp(rows @ pos_rows.T / temperature, dim=1) - math.log(pos_rows.shape[0])
    neg_logit = torch.logsumexp(rows @ neg_rows.T / temperature, dim=1) - math.log(neg_rows.shape[0])
    return torch.sigmoid(pos_logit - neg_logit).reshape(height, width)


def _binary_nll(probability: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> float:
    values = torch.cat([probability[positive], probability[negative]])
    labels = torch.cat(
        [torch.ones(int(positive.sum())), torch.zeros(int(negative.sum()))]
    ).to(values)
    return float(F.binary_cross_entropy(values.clamp(1e-7, 1 - 1e-7), labels))


def _cycle_metrics(
    source: torch.Tensor,
    cycle: torch.Tensor,
    supported: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> dict[str, object]:
    x = source[supported].double()
    z = cycle[supported].double()
    centered_x = x - x.mean()
    centered_z = z - z.mean()
    denominator = torch.linalg.vector_norm(centered_x) * torch.linalg.vector_norm(centered_z)
    correlation = float((centered_x @ centered_z) / denominator) if float(denominator) > 0 else 0.0
    return {
        "supported_pixel_fraction": float(supported.double().mean()),
        "mae_supported": float((x - z).abs().mean()),
        "rmse_supported": float(torch.sqrt((x - z).square().mean())),
        "pearson_supported": correlation,
        "source_mean": float(source.mean()),
        "source_foreground_fraction_at_0_5": float((source >= 0.5).double().mean()),
        "cycle_mean_supported": float(z.mean()),
        "cycle_foreground_fraction_at_0_5_supported": float((z >= 0.5).double().mean()),
        "cycle_positive_scribble_mean": float(cycle[positive].mean()),
        "cycle_negative_scribble_mean": float(cycle[negative].mean()),
        "cycle_scribble_nll": _binary_nll(cycle, positive, negative),
    }


@torch.inference_mode()
def analyze(args: argparse.Namespace) -> dict[str, object]:
    cache_report_path = Path(args.cache_report).resolve()
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    authority = PromptResponsibilityAuthority.from_dict(cache_report["authority"])
    cache_path = Path(args.cache).resolve()
    if str(Path(cache_report["artifact_path"]).resolve()) != str(cache_path):
        raise ValueError("cache path differs from export report authority")

    render_manifest_path = Path(args.render_manifest).resolve()
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    if render_manifest.get("scene_id") != authority.scene_id:
        raise ValueError("render manifest scene differs from responsibility authority")
    if render_manifest.get("safety") != {
        "camera_mapping": "queue_locked_rgb_to_colmap_map",
        "evaluation_ground_truth_opened": False,
        "rgb_files_opened": False,
        "rgb_refiner_used": False,
        "segmentation_masks_opened": False,
    }:
        raise ValueError("prompt feature render lacks exact target-blind safety authority")
    geometry = render_manifest.get("canonical_field_geometry_fingerprint", {})
    if (
        geometry.get("xyz_sha256") != authority.geometry_xyz_sha256
        or int(geometry.get("num_gaussians", -1)) != authority.num_gaussians
        or render_manifest.get("checkpoint_sha256") != authority.geometry_checkpoint_sha256
    ):
        raise ValueError("prompt feature and responsibility geometry authorities differ")
    feature_record = _find_feature_record(render_manifest, frame_id=authority.frame_id)
    if (
        feature_record.get("camera_name") != authority.camera_name
        or feature_record.get("colmap_camera_name") != authority.colmap_camera_name
    ):
        raise ValueError("prompt feature camera authority differs")
    feature_path = Path(feature_record["feature_path"]).resolve()
    expected_feature_sha256 = str(args.feature_sha256)
    if sha256_file(feature_path) != expected_feature_sha256:
        raise ValueError("prompt feature SHA-256 differs from independent authority")

    positive_path = Path(args.positive_scribble).resolve()
    negative_path = Path(args.negative_scribble).resolve()
    if (
        sha256_file(positive_path) != authority.source_sha256["positive_scribble"]
        or sha256_file(negative_path) != authority.source_sha256["negative_scribble"]
    ):
        raise ValueError("prompt scribbles differ from responsibility source authority")
    positive_native = torch.from_numpy(load_ground_truth_mask(positive_path))
    negative_native = torch.from_numpy(load_ground_truth_mask(negative_path))
    native_shape = (authority.height, authority.width)
    if tuple(positive_native.shape) != native_shape or tuple(negative_native.shape) != native_shape:
        raise ValueError("prompt scribbles differ from responsibility native shape")
    if bool((positive_native & negative_native).any()):
        raise ValueError("positive and negative prompt scribbles overlap")

    radio_checkpoint = Path(args.radio_checkpoint).resolve()
    radio_sha256 = sha256_file(radio_checkpoint)
    if radio_sha256 != str(args.radio_checkpoint_sha256):
        raise ValueError("frozen RADIO checkpoint SHA-256 differs")
    prompt_radio = torch.load(feature_path, map_location="cpu", weights_only=True).float()
    if tuple(prompt_radio.shape) != tuple(feature_record["shape"]) or prompt_radio.shape[0] != 1280:
        raise ValueError("prompt canonical feature tensor shape differs")
    adaptor = load_radio_adaptor_from_checkpoint(
        radio_checkpoint,
        "dino_v3_7b",
        kind="feature_projection",
        expected_sha256=radio_sha256,
    ).eval().requires_grad_(False)
    height, width = map(int, prompt_radio.shape[1:])
    tokens = prompt_radio.permute(1, 2, 0).reshape(1, height * width, 1280)
    dino = adaptor(tokens).reshape(height, width, -1).permute(2, 0, 1)
    dino = F.normalize(dino.float(), dim=0)
    positive_grid = _resize_mask(positive_native.numpy(), (height, width))
    negative_grid = _resize_mask(negative_native.numpy(), (height, width))
    if bool((positive_grid & negative_grid).any()):
        raise ValueError("prompt scribbles overlap after fixed nearest resampling")
    low_posterior = _dense_logmeanexp_posterior(
        dino, positive_grid, negative_grid, temperature=float(METHOD_CONTRACT["temperature"])
    )
    dense_native = F.interpolate(
        low_posterior[None, None], size=native_shape, mode="bilinear", align_corners=False
    )[0, 0]
    anchored_native = dense_native.clone()
    anchored_native[positive_native] = 1.0
    anchored_native[negative_native] = 0.0

    del adaptor, tokens, dino, prompt_radio
    cache = load_prompt_responsibility_cache(
        cache_path,
        expected_authority=authority,
        expected_file_sha256=str(cache_report["file_sha256"]),
    )
    dense_cycle = cache.cycle(dense_native)
    anchored_adjoint = cache.adjoint(anchored_native)
    anchored_cycle = cache.forward(anchored_adjoint.primitive_probability)

    artifact_payload = {
        "schema_version": 1,
        "artifact_type": "nvos_dino_dense_prompt_exact_adjoint_cycle",
        "scene_id": authority.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": _json_sha256(METHOD_CONTRACT),
        "responsibility_authority_sha256": authority.digest,
        "responsibility_file_sha256": str(cache_report["file_sha256"]),
        "prompt_feature_sha256": expected_feature_sha256,
        "radio_checkpoint_sha256": radio_sha256,
        "tensors": {
            "low_posterior": low_posterior.float().contiguous(),
            "dense_native": dense_native.float().contiguous(),
            "anchored_native": anchored_native.float().contiguous(),
            "primitive_probability": anchored_adjoint.primitive_probability.float().contiguous(),
            "primitive_visible": anchored_adjoint.visible.contiguous(),
            "cycle_native": anchored_cycle.normalized_probability.float().contiguous(),
            "cycle_supported": anchored_cycle.supported.contiguous(),
        },
        "target_rgb_opened": False,
        "target_mask_opened": False,
    }
    tensor_digests = {
        name: tensor_sha256(value)
        for name, value in sorted(artifact_payload["tensors"].items())
    }
    artifact_payload["tensor_sha256"] = tensor_digests
    artifact_payload["tensor_bundle_sha256"] = _json_sha256(tensor_digests)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    torch.save(artifact_payload, output_path)
    output_sha256 = sha256_file(output_path)
    frozen_payload = torch.load(output_path, map_location="cpu", weights_only=True)
    validate_dino_completion_payload(
        frozen_payload,
        authority=authority,
        expected_responsibility_file_sha256=str(cache_report["file_sha256"]),
        expected_primitive_sha256=tensor_digests["primitive_probability"],
    )
    if sha256_file(output_path) != output_sha256:
        raise ValueError("DINO completion artifact changed across freeze/reload")

    report = {
        "scene_id": authority.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": artifact_payload["method_contract_sha256"],
        "responsibility_authority_sha256": authority.digest,
        "responsibility_file_sha256": str(cache_report["file_sha256"]),
        "prompt_feature_sha256": expected_feature_sha256,
        "render_manifest_sha256": sha256_file(render_manifest_path),
        "radio_checkpoint_sha256": radio_sha256,
        "output": str(output_path),
        "output_sha256": output_sha256,
        "tensor_bundle_sha256": artifact_payload["tensor_bundle_sha256"],
        "grid_shape": [height, width],
        "native_shape": list(native_shape),
        "positive_grid_tokens": int(positive_grid.sum()),
        "negative_grid_tokens": int(negative_grid.sum()),
        "dense_grid_positive_mean": float(low_posterior[positive_grid].mean()),
        "dense_grid_negative_mean": float(low_posterior[negative_grid].mean()),
        "dense_grid_scribble_nll": _binary_nll(low_posterior, positive_grid, negative_grid),
        "unanchored_cycle": _cycle_metrics(
            dense_native,
            dense_cycle.normalized_probability,
            dense_cycle.supported,
            positive_native,
            negative_native,
        ),
        "anchored_cycle": _cycle_metrics(
            anchored_native,
            anchored_cycle.normalized_probability,
            anchored_cycle.supported,
            positive_native,
            negative_native,
        ),
        "visible_primitive_fraction": float(anchored_adjoint.visible.double().mean()),
        "primitive_probability_mean_visible": float(
            anchored_adjoint.primitive_probability[anchored_adjoint.visible].mean()
        ),
        "prompt_only_inputs_opened": True,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-report", required=True)
    parser.add_argument("--render-manifest", required=True)
    parser.add_argument("--feature-sha256", required=True)
    parser.add_argument("--positive-scribble", required=True)
    parser.add_argument("--negative-scribble", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--radio-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
