#!/usr/bin/env python3
"""Render and score a sealed observation-clamped NVOS selector.

The prediction authority and primitive selector are strictly reloaded before
target rendering.  The exact front-to-back target score is then frozen and
strictly reloaded before any target-mask byte is opened.  This entrypoint is
independent of the shared frozen NVOS evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.evaluation.promptable_segmentation import (
    compute_binary_metrics,
    load_ground_truth_mask,
)
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    tensor_sha256,
)
from radio_gs.rendering.contribution_compositor import (
    rasterize_single_view_contributions,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _resize_nvos_score_for_evaluation,
)
from radio_gs.scripts.eval_frozen_nvos_primitive_unary import (
    _float32_rows_sha256,
    _scene_record,
    _target_frame,
    _view_by_frame,
)
from radio_gs.scripts.render_promptable_nvs_features import (
    resolve_protocol_views,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


AUTHORITY_ARTIFACT_TYPE = (
    "radio_gs.nvos_observation_clamped_prediction_authority"
)
SCORE_ARTIFACT_TYPE = (
    "radio_gs.nvos_observation_clamped_frozen_exact_target_score"
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_prediction_authority(
    args: argparse.Namespace,
) -> tuple[dict, str, Path, PromptResponsibilityAuthority]:
    authority, digest, path = load_json_object(
        args.prediction_authority,
        expected_sha256=args.prediction_authority_sha256,
        label="sealed observation-clamped prediction authority",
    )
    expected = {
        "schema_version",
        "artifact_type",
        "scene_id",
        "method",
        "selector",
        "source_audit",
        "support_graph",
        "exact_w",
        "geometry_authority",
        "pre_gt_source_access",
        "real_graph_invariants",
        "sealed_before_target_score_render",
        "sealed_before_target_mask_open",
        "target_rgb_opened",
        "target_mask_opened",
        "target_metric_computed",
    }
    if set(authority) != expected:
        raise ValueError("prediction authority schema differs")
    method = _mapping(authority["method"], label="prediction method")
    source_access = _mapping(
        authority["pre_gt_source_access"], label="pre-GT source access"
    )
    invariants = _mapping(
        authority["real_graph_invariants"], label="real-graph invariants"
    )
    if (
        authority["schema_version"] != 1
        or authority["artifact_type"] != AUTHORITY_ARTIFACT_TYPE
        or authority["scene_id"] != args.scene_id
        or method.get("target_readout") != "exact_front_to_back_Wu_over_W1"
        or float(method.get("threshold", -1)) != 0.5
        or method.get("threshold_comparator") != "greater_or_equal"
        or method.get("connected_selection") is not False
        or method.get("target_dependent_routing_or_calibration") is not False
        or source_access.get("source_only") is not True
        or source_access.get("target_rgb_opened") is not False
        or source_access.get("target_mask_opened") is not False
        or source_access.get("target_metric_computed") is not False
        or invariants.get("observed_rows_bitwise_equal") is not True
        or invariants.get("unreachable_no_boundary_rows_bitwise_equal") is not True
        or int(invariants.get("unknown_changed_rows", 0)) <= 0
        or authority["sealed_before_target_score_render"] is not True
        or authority["sealed_before_target_mask_open"] is not True
        or authority["target_rgb_opened"] is not False
        or authority["target_mask_opened"] is not False
        or authority["target_metric_computed"] is not False
    ):
        raise ValueError("prediction authority is not eligible for target rendering")
    exact_w = _mapping(authority["exact_w"], label="exact-W authority")
    report_path = Path(str(exact_w["report_path"])).resolve()
    if (
        sha256_file(report_path) != exact_w["report_sha256"]
        or sha256_file(exact_w["cache_path"]) != exact_w["cache_sha256"]
    ):
        raise ValueError("exact-W files changed after prediction-authority seal")
    report, _report_digest, _report_path = load_json_object(
        report_path,
        expected_sha256=str(exact_w["report_sha256"]),
        label="sealed exact-W report",
    )
    responsibility = PromptResponsibilityAuthority.from_dict(report["authority"])
    if (
        responsibility.digest != exact_w["authority_sha256"]
        or responsibility.scene_id != args.scene_id
        or responsibility.target_rgb_opened is not False
        or responsibility.target_mask_opened is not False
    ):
        raise ValueError("exact-W geometry authority differs")
    # A second content read proves the JSON authority itself stayed immutable
    # across semantic validation.
    frozen, frozen_digest, frozen_path = load_json_object(
        path,
        expected_sha256=digest,
        label="strictly reloaded prediction authority",
    )
    if frozen != authority or frozen_digest != digest or frozen_path != path:
        raise RuntimeError("prediction authority changed across strict reload")
    return authority, digest, path, responsibility


def _load_selector(
    authority: Mapping[str, object],
    responsibility: PromptResponsibilityAuthority,
) -> tuple[torch.Tensor, dict]:
    record = _mapping(authority["selector"], label="selector authority")
    payload, digest, path = load_torch_mapping(
        record["path"],
        expected_sha256=str(record["sha256"]),
        label="sealed observation-clamped selector",
    )
    tensors = _mapping(payload.get("tensors"), label="selector tensors")
    primitive = torch.as_tensor(tensors.get("primitive_probability"))
    tensor_digests = _mapping(
        payload.get("tensor_sha256"), label="selector tensor digests"
    )
    if (
        payload.get("artifact_type")
        != "radio_gs.nvos_observation_clamped_harmonic_selector"
        or payload.get("scene_id") != responsibility.scene_id
        or payload.get("target_rgb_opened") is not False
        or payload.get("target_mask_opened") is not False
        or payload.get("target_metric_computed") is not False
        or primitive.device.type != "cpu"
        or primitive.dtype != torch.float32
        or primitive.shape != (responsibility.num_gaussians,)
        or not primitive.is_contiguous()
        or not bool(torch.isfinite(primitive).all())
        or bool(((primitive < 0) | (primitive > 1)).any())
        or tensor_sha256(primitive) != record["primitive_probability_sha256"]
        or tensor_digests.get("primitive_probability")
        != record["primitive_probability_sha256"]
        or payload.get("tensor_bundle_sha256") != record["tensor_bundle_sha256"]
        or digest != record["sha256"]
        or sha256_file(path) != digest
    ):
        raise ValueError("sealed selector differs from prediction authority")
    return primitive, payload


def _validate_score(
    payload: object,
    *,
    contract: Mapping[str, object],
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "artifact_type",
        "contract",
        "contract_sha256",
        "score",
        "supported",
        "score_sha256",
        "supported_sha256",
        "roundoff_diagnostics",
        "target_rgb_opened",
        "target_mask_opened",
    }:
        raise ValueError("target score artifact schema differs")
    score = payload["score"]
    supported = payload["supported"]
    if (
        payload["schema_version"] != 1
        or payload["artifact_type"] != SCORE_ARTIFACT_TYPE
        or payload["contract"] != contract
        or payload["contract_sha256"] != canonical_json_sha256(contract)
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
        or not torch.is_tensor(score)
        or score.device.type != "cpu"
        or score.dtype != torch.float32
        or tuple(score.shape) != (height, width)
        or not score.is_contiguous()
        or not bool(torch.isfinite(score).all())
        or bool(((score < 0) | (score > 1)).any())
        or not torch.is_tensor(supported)
        or supported.device.type != "cpu"
        or supported.dtype != torch.bool
        or tuple(supported.shape) != (height, width)
        or not supported.is_contiguous()
        or payload["score_sha256"] != tensor_sha256(score)
        or payload["supported_sha256"] != tensor_sha256(supported)
    ):
        raise ValueError("target score artifact authority differs")
    diagnostics = _mapping(
        payload["roundoff_diagnostics"], label="roundoff diagnostics"
    )
    if (
        float(diagnostics.get("tolerance", 0)) != 2e-6
        or float(diagnostics.get("maximum_below_zero_deviation", np.inf)) > 2e-6
        or float(diagnostics.get("maximum_above_one_deviation", np.inf)) > 2e-6
    ):
        raise ValueError("target score roundoff diagnostics differ")
    return score, supported


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("exact target rendering requires CUDA")
    authority, authority_sha256, authority_path, responsibility = (
        _load_prediction_authority(args)
    )
    primitive_cpu, selector = _load_selector(authority, responsibility)

    manifest_path = Path(args.manifest).resolve()
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_sha256 != responsibility.source_sha256["benchmark_manifest"]:
        raise ValueError("benchmark manifest differs from sealed geometry authority")
    scene = _scene_record(manifest, args.scene_id)
    evaluation_frames = list(map(str, scene.get("evaluation_frame_ids", [])))
    if len(evaluation_frames) != 1:
        raise ValueError("exact sentinel evaluator requires one target frame")
    target_frame_id = evaluation_frames[0]
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
        config_sha256 != responsibility.source_sha256["gaussfm_config"]
        or checkpoint_sha256 != responsibility.geometry_checkpoint_sha256
        or camera_map_sha256 != responsibility.source_sha256["camera_mapping"]
    ):
        raise ValueError("target renderer differs from sealed geometry authority")
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
            str(config_path),
            str(checkpoint_path),
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
        )
    )
    if refiner is not None:
        raise ValueError("sealed primitive selector forbids RGB screen refiners")
    if (
        int(model.get_xyz().shape[0]) != responsibility.num_gaussians
        or _float32_rows_sha256(model.get_xyz())
        != responsibility.geometry_xyz_sha256
    ):
        raise ValueError("target renderer Gaussian rows differ from exact-W authority")

    height, width = responsibility.height, responsibility.width
    pose_cpu = torch.from_numpy(target_view["w2c"]).float().cpu().contiguous()
    pose = pose_cpu.to(device)
    intrinsics_cpu = (
        renderer.scaled_intrinsics(width, height).detach().float().cpu().contiguous()
    )
    primitive = primitive_cpu.to(device)
    hits = rasterize_single_view_contributions(
        model, renderer, pose, height=height, width=width
    )
    gids = hits["gaussian_ids"]
    pids = hits["pixel_ids"]
    weights = hits["weights"].float()
    pixel_count = height * width
    numerator = torch.zeros(pixel_count, dtype=torch.float32, device=device)
    mass = torch.zeros(pixel_count, dtype=torch.float32, device=device)
    numerator.index_add_(0, pids, weights * primitive[gids])
    mass.index_add_(0, pids, weights)
    supported_gpu = mass > 0
    raw_score = torch.zeros_like(numerator)
    raw_score[supported_gpu] = numerator[supported_gpu] / mass[supported_gpu]
    if not bool(torch.isfinite(raw_score).all()):
        raise ValueError("target score contains non-finite values")
    tolerance = 2e-6
    below = raw_score < 0
    above = raw_score > 1
    max_below = float((-raw_score[below]).max()) if bool(below.any()) else 0.0
    max_above = float((raw_score[above] - 1).max()) if bool(above.any()) else 0.0
    if max_below > tolerance or max_above > tolerance:
        raise ValueError("target score exceeds float32 probability roundoff")
    diagnostics = {
        "tolerance": tolerance,
        "raw_minimum": float(raw_score.min()),
        "raw_maximum": float(raw_score.max()),
        "below_zero_count": int(below.sum()),
        "above_one_count": int(above.sum()),
        "maximum_below_zero_deviation": max_below,
        "maximum_above_one_deviation": max_above,
        "clamp_applied": bool(below.any() or above.any()),
    }
    score = raw_score.clamp(0, 1).reshape(height, width).cpu().contiguous()
    supported = supported_gpu.reshape(height, width).cpu().contiguous()
    del hits, gids, pids, weights, primitive, model, numerator, mass, raw_score
    torch.cuda.empty_cache()

    selector_record = _mapping(authority["selector"], label="selector authority")
    support_graph = _mapping(authority["support_graph"], label="support graph")
    exact_w = _mapping(authority["exact_w"], label="exact-W authority")
    contract = {
        "method": "observation_clamped_harmonic_extension_v1",
        "selector_method_contract_sha256": selector["method_contract_sha256"],
        "prediction_authority_path": str(authority_path),
        "prediction_authority_sha256": authority_sha256,
        "selector_sha256": selector_record["sha256"],
        "primitive_probability_sha256": selector_record[
            "primitive_probability_sha256"
        ],
        "source_audit_sha256": authority["source_audit"]["sha256"],
        "support_graph_sha256": support_graph["sha256"],
        "responsibility_authority_sha256": responsibility.digest,
        "responsibility_cache_sha256": exact_w["cache_sha256"],
        "target_readout": "exact_front_to_back_Wu_over_W1",
        "numeric_probability_clamp": "float32_roundoff_only",
        "threshold": 0.5,
        "threshold_comparator": "greater_or_equal",
        "evaluation_resize": "cv2.INTER_LINEAR_continuous_score_to_target_shape",
        "connected_selection": "none",
        "target_dependent_routing_or_calibration": False,
        "scene_id": args.scene_id,
        "target_frame_id": target_frame_id,
        "target_camera_name": str(target_view["camera_name"]),
        "target_colmap_camera_name": str(target_view["colmap_camera_name"]),
        "native_shape": [height, width],
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
        "geometry_xyz_sha256": responsibility.geometry_xyz_sha256,
    }
    score_payload = {
        "schema_version": 1,
        "artifact_type": SCORE_ARTIFACT_TYPE,
        "contract": contract,
        "contract_sha256": canonical_json_sha256(contract),
        "score": score,
        "supported": supported,
        "score_sha256": tensor_sha256(score),
        "supported_sha256": tensor_sha256(supported),
        "roundoff_diagnostics": diagnostics,
        "target_rgb_opened": False,
        "target_mask_opened": False,
    }
    score_path = write_torch_noclobber(args.score_output, score_payload)
    score_file_sha256 = sha256_file(score_path)
    frozen, frozen_digest, frozen_path = load_torch_mapping(
        score_path,
        expected_sha256=score_file_sha256,
        label="strictly reloaded target score",
    )
    frozen_score, frozen_supported = _validate_score(
        frozen, contract=contract, height=height, width=width
    )
    if frozen_digest != score_file_sha256 or frozen_path != score_path:
        raise RuntimeError("target score changed across strict reload")

    # Scoring-only boundary: this is the first target-mask byte access.
    target_record = _target_frame(scene, target_frame_id)
    target_path = Path(
        str(target_record.get("ground_truth") or target_record.get("gt_mask_path"))
    ).resolve()
    target_sha256 = str(target_record.get("ground_truth_sha256") or "")
    if sha256_file(target_path) != target_sha256:
        raise ValueError("target mask differs from frozen benchmark manifest")
    ground_truth = load_ground_truth_mask(target_path)
    resized_score = _resize_nvos_score_for_evaluation(
        frozen_score.numpy(),
        tuple(map(int, ground_truth.shape)),
        registered_forward_unary="none",
    )
    prediction = resized_score >= 0.5
    metrics = compute_binary_metrics(prediction, ground_truth)
    report = {
        "schema_version": 1,
        "artifact_type": "radio_gs.nvos_observation_clamped_frozen_exact_report",
        "scene_id": args.scene_id,
        "target_frame_id": target_frame_id,
        "method_contract": contract,
        "method_contract_sha256": canonical_json_sha256(contract),
        "prediction_authority_path": str(authority_path),
        "prediction_authority_sha256": authority_sha256,
        "selector_sha256": selector_record["sha256"],
        "source_audit_sha256": authority["source_audit"]["sha256"],
        "support_graph_sha256": support_graph["sha256"],
        "responsibility_cache_sha256": exact_w["cache_sha256"],
        "score_path": str(score_path),
        "score_file_sha256": score_file_sha256,
        "score_tensor_sha256": frozen["score_sha256"],
        "score_contract_sha256": frozen["contract_sha256"],
        "roundoff_diagnostics": frozen["roundoff_diagnostics"],
        "score_minimum": float(frozen_score.min()),
        "score_maximum": float(frozen_score.max()),
        "score_mean": float(frozen_score.mean()),
        "supported_pixel_fraction": float(frozen_supported.double().mean()),
        "foreground_fraction_at_0_5": float(prediction.mean()),
        "target_shape": list(ground_truth.shape),
        "score_to_target_resize": "cv2.INTER_LINEAR_before_threshold",
        "metrics": metrics,
        "target_mask_sha256": target_sha256,
        "prediction_authority_strictly_reloaded_before_target_render": True,
        "score_frozen_and_strictly_reloaded_before_target_mask": True,
        "target_rgb_opened": False,
        "target_mask_opened_for_scoring_only": True,
        "target_dependent_selection_or_tuning": False,
    }
    report_path = write_frozen_json(args.report, report)
    return {
        **report,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-authority", required=True)
    parser.add_argument("--prediction-authority-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
