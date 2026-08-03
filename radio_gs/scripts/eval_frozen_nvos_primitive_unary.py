#!/usr/bin/env python3
"""Render and score a frozen primitive selector under the locked NVOS protocol.

The primitive tensor and target-view score are independently hashed and
reloaded before the target mask is opened.  Threshold 0.5, no graph, and no
connected-component selection are fixed by the method contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.evaluation.promptable_segmentation import (
    compute_binary_metrics,
    load_ground_truth_mask,
)
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    sha256_file,
    tensor_sha256,
)
from radio_gs.rendering.contribution_compositor import (
    rasterize_single_view_contributions,
)
from radio_gs.scripts.analyze_nvos_dense_prompt_adjoint_cycle import (
    validate_dino_completion_payload,
)
from radio_gs.scripts.analyze_nvos_full_teacher_prompt_adjoint_cycle import (
    validate_full_teacher_completion_payload,
)
from radio_gs.scripts.propagate_nvos_dense_selector_fixed_graph import (
    GRAPH_ARTIFACT_KEYS,
    validate_graph_selector_payload,
)
from radio_gs.scripts.fuse_nvos_dense_exact_anchors import (
    ARTIFACT_KEYS as EXACT_ANCHOR_ARTIFACT_KEYS,
    validate_exact_anchor_selector_payload,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _resize_nvos_score_for_evaluation,
)
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views


METHOD_CONTRACT = {
    "selector": "frozen_dino_dense_prompt_exact_adjoint_primitive_probability",
    "target_readout": "exact_front_to_back_Wu_over_W1",
    "numeric_probability_clamp": "clip_float32_roundoff_to_[0,1]_before_score_freeze",
    "threshold": 0.5,
    "threshold_comparator": "greater_or_equal",
    "registered_forward_unary": "none",
    "evaluation_resize": "cv2.INTER_LINEAR_continuous_score_to_target_shape",
    "graph": "none",
    "connected_selection": "none",
    "target_rgb": "never_opened",
    "target_mask": "opened_only_after_selector_and_score_freeze_reload",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _float32_rows_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _scene_record(manifest: dict, scene_id: str) -> dict:
    matches = [scene for scene in manifest.get("scenes", []) if scene.get("scene_id") == scene_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one scene {scene_id!r}")
    return matches[0]


def _view_by_frame(views: list[dict], frame_id: str) -> dict:
    matches = [view for view in views if str(view.get("frame_id")) == str(frame_id)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one view {frame_id!r}")
    return matches[0]


def _target_frame(scene: dict, frame_id: str) -> dict:
    raw = scene.get("frames", [])
    records = raw.values() if isinstance(raw, dict) else raw
    matches = [record for record in records if str(record.get("frame_id")) == frame_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one target frame record {frame_id!r}")
    return matches[0]


def _load_frozen_selector(
    args: argparse.Namespace,
    authority: PromptResponsibilityAuthority,
    *,
    expected_responsibility_file_sha256: str,
    expected_responsibility_tensor_bundle_sha256: str,
    expected_benchmark_manifest_sha256: str,
    expected_source_rgb_path: Path,
):
    completion_path = Path(args.completion).resolve()
    before_sha256 = sha256_file(completion_path)
    if before_sha256 != str(args.expected_completion_sha256):
        raise ValueError("completion artifact SHA-256 differs from frozen selector authority")
    payload = torch.load(completion_path, map_location="cpu", weights_only=True)
    base_expected_keys = {
        "schema_version", "artifact_type", "scene_id", "method_contract",
        "method_contract_sha256", "responsibility_authority_sha256",
        "responsibility_file_sha256", "prompt_feature_sha256",
        "radio_checkpoint_sha256", "tensors", "target_rgb_opened",
        "target_mask_opened", "tensor_sha256", "tensor_bundle_sha256",
    }
    graph_expected_keys = GRAPH_ARTIFACT_KEYS
    exact_anchor_expected_keys = EXACT_ANCHOR_ARTIFACT_KEYS
    full_teacher_extra_keys = frozenset({
        "benchmark_manifest_sha256", "source_rgb_path", "source_rgb_sha256",
        "source_rgb_shape", "source_frame_id", "source_feature_grid_shape",
        "radio_source_tree_sha256",
    })
    full_teacher_expected_keys = base_expected_keys | set(full_teacher_extra_keys)
    if not isinstance(payload, dict) or frozenset(payload) not in {
        frozenset(base_expected_keys), frozenset(graph_expected_keys),
        frozenset(full_teacher_expected_keys), frozenset(exact_anchor_expected_keys),
    }:
        raise ValueError("completion artifact schema differs")
    if payload.get("artifact_type") == "nvos_dino_dense_prompt_exact_adjoint_cycle":
        tensors = validate_dino_completion_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256=expected_responsibility_file_sha256,
            expected_primitive_sha256=str(args.expected_primitive_sha256),
        )
        if sha256_file(completion_path) != before_sha256:
            raise ValueError("completion artifact changed across trusted load")
        return tensors["primitive_probability"], payload
    if payload.get("artifact_type") == (
        "nvos_full_teacher_dino_dense_prompt_exact_adjoint_cycle"
    ):
        tensors = validate_full_teacher_completion_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256=expected_responsibility_file_sha256,
            expected_primitive_sha256=str(args.expected_primitive_sha256),
            expected_source_rgb_path=expected_source_rgb_path,
            expected_benchmark_manifest_sha256=(
                expected_benchmark_manifest_sha256
            ),
        )
        if sha256_file(completion_path) != before_sha256:
            raise ValueError("completion artifact changed across trusted load")
        return tensors["primitive_probability"], payload
    if payload.get("artifact_type") == "nvos_dino_dense_prompt_fixed_graph_selector":
        if args.source_completion_sha256 is None or args.source_primitive_sha256 is None:
            raise ValueError(
                "graph selector requires frozen upstream completion and primitive SHA-256"
            )
        primitive = validate_graph_selector_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256=expected_responsibility_file_sha256,
            expected_completion_sha256=str(args.source_completion_sha256),
            expected_source_primitive_sha256=str(args.source_primitive_sha256),
            expected_primitive_sha256=str(args.expected_primitive_sha256),
        )
        if sha256_file(completion_path) != before_sha256:
            raise ValueError("completion artifact changed across trusted load")
        return primitive, payload
    if payload.get("artifact_type") == (
        "nvos_dino_dense_exact_exclusive_anchor_selector"
    ):
        if args.source_completion_sha256 is None or args.source_primitive_sha256 is None:
            raise ValueError(
                "exact-anchor selector requires frozen upstream completion and primitive SHA-256"
            )
        primitive = validate_exact_anchor_selector_payload(
            payload,
            authority=authority,
            expected_responsibility_file_sha256=expected_responsibility_file_sha256,
            expected_responsibility_tensor_bundle_sha256=(
                expected_responsibility_tensor_bundle_sha256
            ),
            expected_source_completion_sha256=str(args.source_completion_sha256),
            expected_source_primitive_sha256=str(args.source_primitive_sha256),
            expected_primitive_sha256=str(args.expected_primitive_sha256),
        )
        if sha256_file(completion_path) != before_sha256:
            raise ValueError("completion artifact changed across trusted load")
        return primitive, payload
    if (
        payload["schema_version"] != 1
        or isinstance(payload["schema_version"], bool)
        or payload["artifact_type"] != "nvos_dino_dense_prompt_fixed_graph_selector"
        or payload["scene_id"] != authority.scene_id
        or payload["responsibility_authority_sha256"] != authority.digest
        or payload["method_contract_sha256"] != _json_sha256(payload["method_contract"])
        or (
            args.source_primitive_sha256 is not None
            and payload["source_primitive_probability_sha256"]
            != str(args.source_primitive_sha256)
        )
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
    ):
        raise ValueError("completion selector authority differs")
    primitive = payload["tensors"].get("primitive_probability")
    if (
        not torch.is_tensor(primitive)
        or primitive.device.type != "cpu"
        or primitive.dtype != torch.float32
        or primitive.shape != (authority.num_gaussians,)
        or not bool(torch.isfinite(primitive).all())
        or bool(((primitive < 0) | (primitive > 1)).any())
    ):
        raise ValueError("primitive selector tensor is invalid")
    actual_hash = tensor_sha256(primitive)
    if (
        actual_hash != str(args.expected_primitive_sha256)
        or payload["tensor_sha256"].get("primitive_probability") != actual_hash
        or payload["tensor_sha256"] != {"primitive_probability": actual_hash}
        or payload["tensor_bundle_sha256"] != _json_sha256(payload["tensor_sha256"])
    ):
        raise ValueError("primitive selector tensor SHA-256 differs")
    if sha256_file(completion_path) != before_sha256:
        raise ValueError("completion artifact changed across trusted load")
    return primitive, payload


def _validate_target_score_payload(
    payload: object,
    *,
    contract: dict[str, object],
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_keys = {
        "schema_version", "artifact_type", "contract", "contract_sha256",
        "score", "supported", "score_sha256", "supported_sha256",
        "roundoff_diagnostics", "target_rgb_opened", "target_mask_opened",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("target score artifact schema differs")
    if (
        payload["schema_version"] != 1
        or isinstance(payload["schema_version"], bool)
        or payload["artifact_type"] != "nvos_frozen_primitive_unary_target_score"
        or payload["contract"] != contract
        or payload["contract_sha256"] != _json_sha256(contract)
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
    ):
        raise ValueError("target score method or authority differs")
    score, supported = payload["score"], payload["supported"]
    expected_shape = (int(height), int(width))
    if (
        not torch.is_tensor(score)
        or score.device.type != "cpu"
        or score.dtype != torch.float32
        or tuple(score.shape) != expected_shape
        or not score.is_contiguous()
        or not bool(torch.isfinite(score).all())
        or bool(((score < 0.0) | (score > 1.0)).any())
    ):
        raise ValueError("target score tensor is malformed")
    if (
        not torch.is_tensor(supported)
        or supported.device.type != "cpu"
        or supported.dtype != torch.bool
        or tuple(supported.shape) != expected_shape
        or not supported.is_contiguous()
    ):
        raise ValueError("target supported tensor is malformed")
    if (
        payload["score_sha256"] != tensor_sha256(score)
        or payload["supported_sha256"] != tensor_sha256(supported)
    ):
        raise ValueError("target score tensor digest differs")
    diagnostics = payload["roundoff_diagnostics"]
    diagnostic_keys = {
        "tolerance", "raw_minimum", "raw_maximum", "below_zero_count",
        "above_one_count", "maximum_below_zero_deviation",
        "maximum_above_one_deviation", "clamp_applied",
    }
    if not isinstance(diagnostics, dict) or set(diagnostics) != diagnostic_keys:
        raise ValueError("target score roundoff diagnostics schema differs")
    tolerance = diagnostics["tolerance"]
    numeric_keys = {
        "tolerance", "raw_minimum", "raw_maximum",
        "maximum_below_zero_deviation", "maximum_above_one_deviation",
    }
    if (
        any(
            isinstance(diagnostics[name], bool)
            or not isinstance(diagnostics[name], (int, float))
            or not np.isfinite(float(diagnostics[name]))
            for name in numeric_keys
        )
        or not (0.0 < float(tolerance) <= 1e-5)
        or diagnostics["below_zero_count"] < 0
        or diagnostics["above_one_count"] < 0
        or not isinstance(diagnostics["below_zero_count"], int)
        or not isinstance(diagnostics["above_one_count"], int)
        or not isinstance(diagnostics["clamp_applied"], bool)
        or float(diagnostics["maximum_below_zero_deviation"]) > float(tolerance)
        or float(diagnostics["maximum_above_one_deviation"]) > float(tolerance)
        or diagnostics["clamp_applied"]
        != bool(diagnostics["below_zero_count"] or diagnostics["above_one_count"])
    ):
        raise ValueError("target score roundoff diagnostics are invalid")
    return score, supported


def _readout_contract(selector_payload: dict) -> dict[str, object]:
    contract = dict(METHOD_CONTRACT)
    if selector_payload["artifact_type"] == "nvos_dino_dense_prompt_fixed_graph_selector":
        contract["selector"] = "frozen_dino_dense_prompt_fixed_graph_primitive_probability"
        contract["graph"] = "frozen_canonical_mpr_v3_shared_support_graph_k16"
        contract["selector_method_contract_sha256"] = selector_payload[
            "method_contract_sha256"
        ]
        contract["support_graph_sha256"] = selector_payload["support_graph_sha256"]
    elif selector_payload["artifact_type"] == (
        "nvos_full_teacher_dino_dense_prompt_exact_adjoint_cycle"
    ):
        contract["selector"] = (
            "frozen_full_reference_teacher_dino_exact_adjoint_primitive_probability"
        )
        contract["selector_method_contract_sha256"] = selector_payload[
            "method_contract_sha256"
        ]
    elif selector_payload["artifact_type"] == (
        "nvos_dino_dense_exact_exclusive_anchor_selector"
    ):
        contract["selector"] = (
            "frozen_compact_dino_dense_probability_plus_exact_exclusive_scribble_anchors"
        )
        contract["selector_method_contract_sha256"] = selector_payload[
            "method_contract_sha256"
        ]
        contract["selector_source_completion_sha256"] = selector_payload[
            "source_completion_sha256"
        ]
    return contract


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("exact target unary rendering requires CUDA")
    cache_report_path = Path(args.cache_report).resolve()
    cache_report_sha256 = sha256_file(cache_report_path)
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    authority = PromptResponsibilityAuthority.from_dict(cache_report["authority"])
    if authority.scene_id != args.scene_id:
        raise ValueError("cache authority scene differs")
    responsibility_path = Path(str(cache_report["artifact_path"])).resolve()
    if sha256_file(responsibility_path) != str(cache_report["file_sha256"]):
        raise ValueError("prompt responsibility artifact differs from export receipt")
    manifest_path = Path(args.manifest).resolve()
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_sha256 != authority.source_sha256["benchmark_manifest"]:
        raise ValueError("benchmark manifest differs from selector authority")
    scene = _scene_record(manifest, args.scene_id)
    source_record = _target_frame(scene, authority.frame_id)
    source_rgb_path = Path(str(source_record.get("rgb_path") or "")).resolve()
    if not source_rgb_path.is_file():
        raise ValueError("benchmark manifest does not bind a readable prompt-frame RGB")
    primitive_cpu, completion_payload = _load_frozen_selector(
        args,
        authority,
        expected_responsibility_file_sha256=str(cache_report["file_sha256"]),
        expected_responsibility_tensor_bundle_sha256=str(
            cache_report["tensor_bundle_sha256"]
        ),
        expected_benchmark_manifest_sha256=manifest_sha256,
        expected_source_rgb_path=source_rgb_path,
    )
    method_contract = _readout_contract(completion_payload)
    evaluation_frames = list(map(str, scene.get("evaluation_frame_ids", [])))
    if len(evaluation_frames) != 1:
        raise ValueError("sentinel evaluator requires one frozen target frame")
    target_frame_id = evaluation_frames[0]
    target_record = _target_frame(scene, target_frame_id)
    base_scene_id = str(scene.get("base_scene_id") or args.scene_id)
    queue_scene = Path(args.queue_root).resolve() / "scenes" / args.scene_id
    if not queue_scene.is_dir():
        queue_scene = Path(args.queue_root).resolve() / "scenes" / base_scene_id
    config_path = queue_scene / "gaussfm_main_track.yaml"
    checkpoint_path = queue_scene / "feature_field" / "checkpoints" / "best.pth"
    camera_map_path = queue_scene / "rgb_to_colmap_camera_mapping.json"
    config_sha256 = sha256_file(config_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    camera_map_sha256 = sha256_file(camera_map_path)
    if (
        config_sha256 != authority.source_sha256["gaussfm_config"]
        or checkpoint_sha256 != authority.geometry_checkpoint_sha256
        or camera_map_sha256 != authority.source_sha256["camera_mapping"]
    ):
        raise ValueError("target renderer source differs from selector geometry authority")
    config = load_config(str(config_path))
    camera_mapping = json.loads(camera_map_path.read_text(encoding="utf-8"))
    views = resolve_protocol_views(
        manifest,
        scene_id=args.scene_id,
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    target_view = _view_by_frame(views, target_frame_id)
    model, _codec, renderer, _sharpener, refiner, _field_config, _is_hybrid = (
        load_render_pipeline(
            str(config_path), str(checkpoint_path), device,
            strict_checkpoint_contract=True, load_ply_rgb_features=False,
        )
    )
    if refiner is not None:
        raise ValueError("frozen primitive unary forbids RGB screen refiners")
    if (
        int(model.get_xyz().shape[0]) != authority.num_gaussians
        or _float32_rows_sha256(model.get_xyz()) != authority.geometry_xyz_sha256
    ):
        raise ValueError("target renderer Gaussian rows differ from selector authority")
    primitive = primitive_cpu.to(device)
    pose_cpu = torch.from_numpy(target_view["w2c"]).float().cpu().contiguous()
    pose = pose_cpu.to(device=device)
    # NVOS LLFF cameras in a scene share native dimensions.  Using the prompt
    # authority avoids opening the target mask even for its header before score freeze.
    height, width = authority.height, authority.width
    intrinsics_cpu = (
        renderer.scaled_intrinsics(width, height).detach().float().cpu().contiguous()
    )
    hits = rasterize_single_view_contributions(model, renderer, pose, height=height, width=width)
    gids, pids, weights = hits["gaussian_ids"], hits["pixel_ids"], hits["weights"].float()
    pixels = height * width
    numerator = torch.zeros(pixels, dtype=torch.float32, device=device)
    mass = torch.zeros(pixels, dtype=torch.float32, device=device)
    numerator.index_add_(0, pids, weights * primitive[gids])
    mass.index_add_(0, pids, weights)
    supported = mass > 0
    raw_score = torch.zeros_like(numerator)
    raw_score[supported] = numerator[supported] / mass[supported]
    if not bool(torch.isfinite(raw_score).all()):
        raise ValueError("target score contains non-finite values before roundoff clamp")
    tolerance = 2e-6
    below_zero = raw_score < 0.0
    above_one = raw_score > 1.0
    max_below = float((-raw_score[below_zero]).max()) if bool(below_zero.any()) else 0.0
    max_above = float((raw_score[above_one] - 1.0).max()) if bool(above_one.any()) else 0.0
    if max_below > tolerance or max_above > tolerance:
        raise ValueError("target score violates probability range beyond float32 roundoff")
    roundoff_diagnostics = {
        "tolerance": tolerance,
        "raw_minimum": float(raw_score.min()),
        "raw_maximum": float(raw_score.max()),
        "below_zero_count": int(below_zero.sum()),
        "above_one_count": int(above_one.sum()),
        "maximum_below_zero_deviation": max_below,
        "maximum_above_one_deviation": max_above,
        "clamp_applied": bool(below_zero.any() or above_one.any()),
    }
    score = raw_score.clamp(0.0, 1.0).reshape(height, width).cpu().contiguous()
    supported = supported.reshape(height, width).cpu().contiguous()
    del hits, gids, pids, weights, primitive, model
    torch.cuda.empty_cache()

    score_contract = {
        **method_contract,
        "scene_id": args.scene_id,
        "target_frame_id": target_frame_id,
        "target_camera_name": str(target_view["camera_name"]),
        "target_colmap_camera_name": str(target_view["colmap_camera_name"]),
        "native_shape": [height, width],
        "responsibility_authority_sha256": authority.digest,
        "responsibility_path": str(responsibility_path),
        "responsibility_file_sha256": str(cache_report["file_sha256"]),
        "responsibility_report_path": str(cache_report_path),
        "responsibility_report_sha256": cache_report_sha256,
        "benchmark_manifest_path": str(manifest_path),
        "benchmark_manifest_sha256": manifest_sha256,
        "renderer_config_path": str(config_path.resolve()),
        "renderer_config_sha256": config_sha256,
        "geometry_checkpoint_path": str(checkpoint_path.resolve()),
        "geometry_checkpoint_sha256": checkpoint_sha256,
        "camera_mapping_path": str(camera_map_path.resolve()),
        "camera_mapping_sha256": camera_map_sha256,
        "target_pose_sha256": tensor_sha256(pose_cpu),
        "target_intrinsics_sha256": tensor_sha256(intrinsics_cpu),
        "geometry_xyz_sha256": authority.geometry_xyz_sha256,
        "completion_file_sha256": str(args.expected_completion_sha256),
        "primitive_probability_sha256": str(args.expected_primitive_sha256),
    }
    score_payload = {
        "schema_version": 1,
        "artifact_type": "nvos_frozen_primitive_unary_target_score",
        "contract": score_contract,
        "contract_sha256": _json_sha256(score_contract),
        "score": score,
        "supported": supported,
        "score_sha256": tensor_sha256(score),
        "supported_sha256": tensor_sha256(supported),
        "roundoff_diagnostics": roundoff_diagnostics,
        "target_rgb_opened": False,
        "target_mask_opened": False,
    }
    score_path = Path(args.score_output).resolve()
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if score_path.exists() and not args.overwrite:
        raise FileExistsError(score_path)
    torch.save(score_payload, score_path)
    score_file_sha256 = sha256_file(score_path)

    # Independent freeze/reload boundary before any target-mask bytes are read.
    frozen_file_sha256_before = sha256_file(score_path)
    if frozen_file_sha256_before != score_file_sha256:
        raise ValueError("target score file changed before freeze/reload")
    frozen = torch.load(score_path, map_location="cpu", weights_only=True)
    frozen_score, frozen_supported = _validate_target_score_payload(
        frozen, contract=score_contract, height=height, width=width
    )
    if sha256_file(score_path) != frozen_file_sha256_before:
        raise ValueError("target score file changed across freeze/reload")
    # Only verified frozen tensors may cross the scoring boundary.
    score, supported = frozen_score, frozen_supported

    # Scoring-only phase begins here.  No method value is changed afterwards.
    target_path = Path(str(target_record.get("ground_truth") or target_record.get("gt_mask_path"))).resolve()
    declared_target_sha256 = str(target_record.get("ground_truth_sha256") or "")
    if sha256_file(target_path) != declared_target_sha256:
        raise ValueError("target mask SHA-256 differs from frozen benchmark manifest")
    ground_truth = load_ground_truth_mask(target_path)
    # Reuse the exact current frozen Gaussian-first ordinary path.  In
    # particular, registered_forward_unary=none is cv2.INTER_LINEAR; only the
    # separately registered beta-v1/v2 methods use nearest-neighbor.
    resized_score = _resize_nvos_score_for_evaluation(
        score.numpy(),
        tuple(map(int, ground_truth.shape)),
        registered_forward_unary="none",
    )
    prediction = resized_score >= float(method_contract["threshold"])
    metrics = compute_binary_metrics(prediction, ground_truth)
    report = {
        "scene_id": args.scene_id,
        "target_frame_id": target_frame_id,
        "method_contract": method_contract,
        "method_contract_sha256": _json_sha256(method_contract),
        "completion_file_sha256": str(args.expected_completion_sha256),
        "completion_method_contract_sha256": completion_payload["method_contract_sha256"],
        "primitive_probability_sha256": str(args.expected_primitive_sha256),
        "score_path": str(score_path),
        "score_file_sha256": score_file_sha256,
        "score_tensor_sha256": frozen["score_sha256"],
        "score_contract_sha256": frozen["contract_sha256"],
        "roundoff_diagnostics": frozen["roundoff_diagnostics"],
        "score_minimum": float(score.min()),
        "score_maximum": float(score.max()),
        "score_mean": float(score.mean()),
        "foreground_fraction_at_0_5": float(prediction.mean()),
        "target_shape": list(ground_truth.shape),
        "score_to_target_resize": "cv2.INTER_LINEAR_before_threshold",
        "supported_pixel_fraction": float(supported.double().mean()),
        "metrics": metrics,
        "target_mask_sha256": declared_target_sha256,
        "selector_frozen_before_target_mask": True,
        "score_frozen_and_reloaded_before_target_mask": True,
        "target_rgb_opened": False,
        "target_mask_opened_for_scoring_only": True,
        "target_dependent_selection_or_tuning": False,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--cache-report", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--expected-primitive-sha256", required=True)
    parser.add_argument("--source-completion-sha256")
    parser.add_argument("--source-primitive-sha256")
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
