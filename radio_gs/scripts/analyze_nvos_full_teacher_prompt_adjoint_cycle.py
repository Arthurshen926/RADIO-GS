#!/usr/bin/env python3
"""Build a target-blind NVOS selector from the full reference-RGB DINO teacher.

This diagnostic differs from ``analyze_nvos_dense_prompt_adjoint_cycle`` only
at the prompt feature source.  The compact rendered RADIO map is replaced by
the official C-RADIOv4-H ``dino_v3_7b`` spatial adaptor evaluated on the frozen
reference RGB.  The same dense log-mean-exp rule, scribble anchors, and exact
prompt-view raster adjoint are then used.  No target RGB or mask path is
accepted by this entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from radio_gs.evaluation.promptable_segmentation import (
    load_ground_truth_mask,
)
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
    sha256_file,
    tensor_sha256,
)
from radio_gs.scripts.analyze_nvos_dense_prompt_adjoint_cycle import (
    _binary_nll,
    _cycle_metrics,
    _dense_logmeanexp_posterior,
    _resize_mask,
)
from radio_gs.scripts.extract_radio_features import (
    _compute_scaled_radio_resolution,
    _load_and_preprocess,
    _load_radio_model,
    _python_source_tree_fingerprint,
    _run_radio_batch,
)


METHOD_CONTRACT = {
    "semantic_space": "official_C-RADIOv4-H_dino_v3_7b_spatial_adaptor",
    "prompt_feature_source": "frozen_reference_rgb_full_2d_teacher",
    "reference_rgb_role": "protocol_permitted_query_input",
    "target_rgb": "never_opened",
    "source_preprocess": "PIL_RGB_LANCZOS_scale_0.25_snap_each_axis_to_nearest_16",
    "source_resolution_scale": 0.25,
    "radio_feature_format": "NCHW",
    "radio_amp": True,
    "scribble_resampling": "nearest_native_to_full_teacher_prompt_grid",
    "dense_rule": "class_count_normalized_all_scribble_logmeanexp_cosine",
    "temperature": 0.07,
    "native_resampling": "bilinear_align_corners_false",
    "strong_unary": "overwrite_official_positive_pixels_to_1_negative_pixels_to_0",
    "adjoint": "u=(W.T@y)/(W.T@1)_visible",
    "cycle": "y_cycle=(W@u)/(W@1)_supported",
    "threshold": 0.5,
    "graph": "none",
    "connected_selection": "none",
    "target_dependent_tuning": False,
}

OFFICIAL_RADIO_CHECKPOINT_SHA256 = (
    "bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9"
)
FULL_TEACHER_TENSOR_KEYS = {
    "low_posterior",
    "dense_native",
    "anchored_native",
    "primitive_probability",
    "primitive_visible",
    "cycle_native",
    "cycle_supported",
}
FULL_TEACHER_ARTIFACT_KEYS = {
    "schema_version",
    "artifact_type",
    "scene_id",
    "method_contract",
    "method_contract_sha256",
    "responsibility_authority_sha256",
    "responsibility_file_sha256",
    "prompt_feature_sha256",
    "radio_checkpoint_sha256",
    "benchmark_manifest_sha256",
    "source_rgb_path",
    "source_rgb_sha256",
    "source_rgb_shape",
    "source_frame_id",
    "source_feature_grid_shape",
    "radio_source_tree_sha256",
    "tensors",
    "target_rgb_opened",
    "target_mask_opened",
    "tensor_sha256",
    "tensor_bundle_sha256",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def validate_full_teacher_completion_payload(
    payload: object,
    *,
    authority: PromptResponsibilityAuthority,
    expected_responsibility_file_sha256: str,
    expected_primitive_sha256: str,
    expected_source_rgb_path: str | Path,
    expected_benchmark_manifest_sha256: str,
) -> dict[str, torch.Tensor]:
    """Fail closed on the full-reference teacher selector and its source RGB."""

    if not isinstance(payload, dict) or set(payload) != FULL_TEACHER_ARTIFACT_KEYS:
        raise ValueError("full-teacher completion artifact schema differs")
    source_rgb = Path(expected_source_rgb_path).resolve()
    source_height_width = payload.get("source_rgb_shape")
    source_grid = payload.get("source_feature_grid_shape")
    if (
        payload["schema_version"] != 1
        or isinstance(payload["schema_version"], bool)
        or payload["artifact_type"]
        != "nvos_full_teacher_dino_dense_prompt_exact_adjoint_cycle"
        or payload["scene_id"] != authority.scene_id
        or payload["source_frame_id"] != authority.frame_id
        or payload["method_contract"] != METHOD_CONTRACT
        or payload["method_contract_sha256"] != _json_sha256(METHOD_CONTRACT)
        or payload["responsibility_authority_sha256"] != authority.digest
        or payload["responsibility_file_sha256"]
        != expected_responsibility_file_sha256
        or payload["benchmark_manifest_sha256"]
        != expected_benchmark_manifest_sha256
        or payload["benchmark_manifest_sha256"]
        != authority.source_sha256.get("benchmark_manifest")
        or Path(str(payload["source_rgb_path"])).resolve() != source_rgb
        or payload["radio_checkpoint_sha256"]
        != OFFICIAL_RADIO_CHECKPOINT_SHA256
        or not _is_sha256(payload["prompt_feature_sha256"])
        or not _is_sha256(payload["source_rgb_sha256"])
        or not _is_sha256(payload["radio_source_tree_sha256"])
        or not isinstance(source_height_width, list)
        or len(source_height_width) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in source_height_width)
        or not isinstance(source_grid, list)
        or len(source_grid) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in source_grid)
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
    ):
        raise ValueError("full-teacher completion method or authority differs")
    if sha256_file(source_rgb) != payload["source_rgb_sha256"]:
        raise ValueError("full-teacher source RGB digest differs")
    source_height, source_width = map(int, source_height_width)
    expected_height, expected_width = _compute_scaled_radio_resolution(
        source_height,
        source_width,
        float(METHOD_CONTRACT["source_resolution_scale"]),
        patch_size=16,
    )
    if source_grid != [expected_height // 16, expected_width // 16]:
        raise ValueError("full-teacher source feature grid differs")

    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != FULL_TEACHER_TENSOR_KEYS:
        raise ValueError("full-teacher completion tensor schema differs")
    native_shape = (int(authority.height), int(authority.width))
    primitive_shape = (int(authority.num_gaussians),)
    expected_specs = {
        "low_posterior": (torch.float32, tuple(source_grid)),
        "dense_native": (torch.float32, native_shape),
        "anchored_native": (torch.float32, native_shape),
        "primitive_probability": (torch.float32, primitive_shape),
        "primitive_visible": (torch.bool, primitive_shape),
        "cycle_native": (torch.float32, native_shape),
        "cycle_supported": (torch.bool, native_shape),
    }
    for name, (dtype, shape) in expected_specs.items():
        value = tensors[name]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != dtype
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise ValueError(f"full-teacher completion tensor {name} is malformed")
    for name in {
        "low_posterior",
        "dense_native",
        "anchored_native",
        "primitive_probability",
        "cycle_native",
    }:
        value = tensors[name]
        if not bool(torch.isfinite(value).all()) or bool(
            ((value < 0.0) | (value > 1.0)).any()
        ):
            raise ValueError(
                f"full-teacher completion probability {name} is outside [0,1]"
            )
    stored_digests = payload["tensor_sha256"]
    actual_digests = {
        name: tensor_sha256(value) for name, value in sorted(tensors.items())
    }
    if stored_digests != actual_digests:
        raise ValueError("full-teacher completion tensor digest differs")
    if payload["tensor_bundle_sha256"] != _json_sha256(actual_digests):
        raise ValueError("full-teacher completion tensor bundle digest differs")
    if actual_digests["primitive_probability"] != expected_primitive_sha256:
        raise ValueError("full-teacher primitive probability digest differs")
    return tensors


def _scene_record(manifest: dict, scene_id: str) -> dict:
    matches = [
        value
        for value in manifest.get("scenes", [])
        if str(value.get("scene_id")) == str(scene_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one scene {scene_id!r}")
    return matches[0]


def _frame_records(scene: dict) -> list[dict]:
    raw = scene.get("frames", [])
    records = list(raw.values()) if isinstance(raw, dict) else list(raw)
    if not all(isinstance(value, dict) for value in records):
        raise ValueError("benchmark frame records are invalid")
    return records


def _validate_reference_rgb_authority(
    *,
    manifest: dict,
    manifest_sha256: str,
    authority: PromptResponsibilityAuthority,
    source_rgb: Path,
    expected_source_rgb_sha256: str,
) -> dict[str, object]:
    """Validate that *source_rgb* is the frozen prompt input, never a target."""

    if manifest_sha256 != authority.source_sha256.get("benchmark_manifest"):
        raise ValueError("benchmark manifest differs from responsibility authority")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or any(
        protocol.get(key) != value
        for key, value in {
            "benchmark": "NVOS",
            "prompt_type": "fixed_positive_negative_scribbles",
            "target_rgb_at_query": "forbidden",
            "target_rgb_during_field_training": "forbidden",
            "target_mask_use": "scoring_only",
        }.items()
    ):
        raise ValueError("benchmark protocol does not match frozen strict-unseen NVOS")
    scene = _scene_record(manifest, authority.scene_id)
    prompt = scene.get("prompt")
    if not isinstance(prompt, dict) or any(
        prompt.get(key) != value
        for key, value in {
            "frame_id": authority.frame_id,
            "type": "positive_negative_scribbles",
        }.items()
    ):
        raise ValueError("responsibility frame is not the frozen reference prompt")
    if list(map(str, scene.get("prompt_frame_ids", []))) != [authority.frame_id]:
        raise ValueError("scene prompt-frame authority differs")
    evaluation_frame_ids = list(map(str, scene.get("evaluation_frame_ids", [])))
    if not evaluation_frame_ids or authority.frame_id in evaluation_frame_ids:
        raise ValueError("reference prompt overlaps a frozen evaluation target")
    if scene.get("target_rgb_policy") != "excluded_from_field_training_and_query":
        raise ValueError("scene target-RGB exclusion policy differs")
    excluded = set(map(str, scene.get("excluded_training_frame_ids", [])))
    if not set(evaluation_frame_ids).issubset(excluded):
        raise ValueError("frozen target is not excluded from field training")

    matches = [
        record
        for record in _frame_records(scene)
        if str(record.get("frame_id")) == authority.frame_id
    ]
    if len(matches) != 1:
        raise ValueError("benchmark manifest must bind one reference RGB record")
    record = matches[0]
    declared_path = Path(str(record.get("rgb_path", ""))).resolve()
    if declared_path != source_rgb.resolve():
        raise ValueError("source RGB path differs from frozen reference record")
    if record.get("ground_truth") is not None or record.get("gt_mask_path") is not None:
        raise ValueError("reference RGB record unexpectedly contains scoring ground truth")
    target_paths = {
        Path(str(value.get("rgb_path", ""))).resolve()
        for value in _frame_records(scene)
        if str(value.get("frame_id")) in evaluation_frame_ids
    }
    if declared_path in target_paths:
        raise ValueError("source RGB path aliases a frozen target RGB")
    observed_sha256 = sha256_file(declared_path)
    if observed_sha256 != str(expected_source_rgb_sha256):
        raise ValueError("reference RGB SHA-256 differs from independent authority")
    return {
        "scene_id": authority.scene_id,
        "source_frame_id": authority.frame_id,
        "source_camera_name": str(record.get("camera_name")),
        "source_rgb_path": str(declared_path),
        "source_rgb_sha256": observed_sha256,
        "evaluation_frame_ids": evaluation_frame_ids,
        "target_rgb_policy": str(scene["target_rgb_policy"]),
        "reference_rgb_protocol_permitted": True,
    }


@torch.inference_mode()
def analyze(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("official full-reference teacher extraction requires CUDA")

    cache_report_path = Path(args.cache_report).resolve()
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    authority = PromptResponsibilityAuthority.from_dict(cache_report["authority"])
    cache_path = Path(args.cache).resolve()
    if str(Path(cache_report["artifact_path"]).resolve()) != str(cache_path):
        raise ValueError("cache path differs from export report authority")

    manifest_path = Path(args.manifest).resolve()
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_rgb = Path(args.source_rgb).resolve()
    source_authority = _validate_reference_rgb_authority(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        authority=authority,
        source_rgb=source_rgb,
        expected_source_rgb_sha256=str(args.source_rgb_sha256),
    )

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
    radio_checkpoint_sha256 = sha256_file(radio_checkpoint)
    if radio_checkpoint_sha256 != str(args.radio_checkpoint_sha256):
        raise ValueError("frozen RADIO checkpoint SHA-256 differs")
    radio_source_before = _python_source_tree_fingerprint(args.radio_repo)
    with Image.open(source_rgb) as handle:
        source_width, source_height = map(int, handle.size)
        if handle.mode != "RGB":
            # The frozen extractor explicitly converts to RGB; the source file
            # itself may use another Pillow mode without changing the contract.
            pass
    target_height, target_width = _compute_scaled_radio_resolution(
        source_height,
        source_width,
        float(METHOD_CONTRACT["source_resolution_scale"]),
        patch_size=16,
    )
    image = _load_and_preprocess(
        [source_rgb], target_height, target_width, device
    )
    model, conditioner = _load_radio_model(
        str(Path(args.radio_repo).resolve()),
        str(radio_checkpoint),
        ["dino_v3_7b"],
        device,
        expected_checkpoint_sha256=radio_checkpoint_sha256,
    )
    _summary, _backbone, adaptors = _run_radio_batch(
        model,
        conditioner,
        image,
        True,
        target_height // 16,
        target_width // 16,
        adaptor_names=["dino_v3_7b"],
    )
    if set(adaptors) != {"dino_v3_7b"}:
        raise RuntimeError("official runtime did not return exactly the DINO adaptor")
    dino = F.normalize(adaptors["dino_v3_7b"][0].float(), dim=0).cpu().contiguous()
    if dino.ndim != 3 or tuple(dino.shape[1:]) != (
        target_height // 16,
        target_width // 16,
    ) or not bool(torch.isfinite(dino).all()):
        raise ValueError("official DINO spatial tensor is invalid")
    prompt_feature_sha256 = tensor_sha256(dino)
    del adaptors, _summary, _backbone, image, conditioner, model
    torch.cuda.empty_cache()

    height, width = map(int, dino.shape[1:])
    positive_grid = _resize_mask(positive_native.numpy(), (height, width))
    negative_grid = _resize_mask(negative_native.numpy(), (height, width))
    if bool((positive_grid & negative_grid).any()):
        raise ValueError("prompt scribbles overlap after fixed nearest resampling")
    low_posterior = _dense_logmeanexp_posterior(
        dino,
        positive_grid,
        negative_grid,
        temperature=float(METHOD_CONTRACT["temperature"]),
    )
    dense_native = F.interpolate(
        low_posterior[None, None],
        size=native_shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    anchored_native = dense_native.clone()
    anchored_native[positive_native] = 1.0
    anchored_native[negative_native] = 0.0

    cache = load_prompt_responsibility_cache(
        cache_path,
        expected_authority=authority,
        expected_file_sha256=str(cache_report["file_sha256"]),
    )
    dense_cycle = cache.cycle(dense_native)
    anchored_adjoint = cache.adjoint(anchored_native)
    anchored_cycle = cache.forward(anchored_adjoint.primitive_probability)

    if sha256_file(source_rgb) != source_authority["source_rgb_sha256"]:
        raise RuntimeError("reference RGB changed during selector construction")
    if sha256_file(radio_checkpoint) != radio_checkpoint_sha256:
        raise RuntimeError("RADIO checkpoint changed during selector construction")
    radio_source_after = _python_source_tree_fingerprint(args.radio_repo)
    if radio_source_after != radio_source_before:
        raise RuntimeError("RADIO source tree changed during selector construction")

    artifact_payload = {
        "schema_version": 1,
        "artifact_type": "nvos_full_teacher_dino_dense_prompt_exact_adjoint_cycle",
        "scene_id": authority.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": _json_sha256(METHOD_CONTRACT),
        "responsibility_authority_sha256": authority.digest,
        "responsibility_file_sha256": str(cache_report["file_sha256"]),
        "prompt_feature_sha256": prompt_feature_sha256,
        "radio_checkpoint_sha256": radio_checkpoint_sha256,
        "benchmark_manifest_sha256": manifest_sha256,
        "source_rgb_path": source_authority["source_rgb_path"],
        "source_rgb_sha256": source_authority["source_rgb_sha256"],
        "source_rgb_shape": [source_height, source_width],
        "source_frame_id": authority.frame_id,
        "source_feature_grid_shape": [height, width],
        "radio_source_tree_sha256": radio_source_before["tree_sha256"],
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

    report = {
        "scene_id": authority.scene_id,
        "method_contract": METHOD_CONTRACT,
        "method_contract_sha256": artifact_payload["method_contract_sha256"],
        "responsibility_authority_sha256": authority.digest,
        "responsibility_file_sha256": str(cache_report["file_sha256"]),
        "benchmark_manifest_sha256": manifest_sha256,
        "source_authority": source_authority,
        "source_rgb_shape": [source_height, source_width],
        "radio_input_shape": [target_height, target_width],
        "grid_shape": [height, width],
        "prompt_feature_sha256": prompt_feature_sha256,
        "radio_checkpoint_sha256": radio_checkpoint_sha256,
        "radio_source_tree_sha256": radio_source_before["tree_sha256"],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "tensor_bundle_sha256": artifact_payload["tensor_bundle_sha256"],
        "positive_grid_tokens": int(positive_grid.sum()),
        "negative_grid_tokens": int(negative_grid.sum()),
        "dense_grid_positive_mean": float(low_posterior[positive_grid].mean()),
        "dense_grid_negative_mean": float(low_posterior[negative_grid].mean()),
        "dense_grid_scribble_nll": _binary_nll(
            low_posterior, positive_grid, negative_grid
        ),
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
        "reference_rgb_opened": True,
        "reference_rgb_protocol_permitted": True,
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-report", required=True)
    parser.add_argument("--source-rgb", required=True)
    parser.add_argument("--source-rgb-sha256", required=True)
    parser.add_argument("--positive-scribble", required=True)
    parser.add_argument("--negative-scribble", required=True)
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--radio-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
