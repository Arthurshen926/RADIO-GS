#!/usr/bin/env python3
"""Seal RegisteredEvidenceToUnary V1/V2 target scores before benchmark GT.

The entrypoint is authority-driven and intentionally has no target-mask,
metric, graph, connected-selection, threshold, or per-scene-fit argument.
It reuses the frozen Gaussian scalar renderer without modifying it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import cv2
import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
    tensor_sha256,
)
from radio_gs.querying.global_prompt_logit_calibrator import (
    GlobalPromptLogitCalibratorV2,
)
from radio_gs.querying.registered_evidence_external_adapter import (
    ADAPTER_SCHEMA,
    SupportedCalibration,
    build_external_registered_features,
    infer_registered_primitive_unary,
    render_then_calibrate,
    validate_external_execution_authority,
    validate_external_promotion_documents,
)
from radio_gs.querying.registered_evidence_to_unary import (
    FEATURE_NAMES,
    RegisteredEvidenceToUnaryV1,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.eval_nvos_gaussian_first import _scene_record, _view_by_frame
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _record_path(records: Mapping[str, object], name: str) -> Path:
    return validate_file_record(records[name], label=f"external authority {name}")


def _load_binary_mask(path: Path, *, shape: tuple[int, int]) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"cannot read declared source mask: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2 or tuple(image.shape) != shape:
        raise ValueError("declared source mask shape differs from exact-W authority")
    return torch.from_numpy(np.asarray(image > 0)).bool().contiguous()


def _state_dict(
    payload: Mapping[str, object], *, label: str
) -> Mapping[str, torch.Tensor]:
    candidate = payload.get("state_dict", payload)
    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError(f"{label} lacks a state_dict")
    state = dict(candidate)
    if not all(
        isinstance(name, str) and torch.is_tensor(value)
        for name, value in state.items()
    ):
        raise ValueError(f"{label} state_dict is malformed")
    return state


def _load_v1(path: Path, digest: str) -> RegisteredEvidenceToUnaryV1:
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=digest,
        label="registered-evidence V1 checkpoint",
    )
    module = RegisteredEvidenceToUnaryV1(hidden_dim=32, max_delta_logit=4.0)
    module.load_state_dict(
        _state_dict(payload, label="registered-evidence V1 checkpoint"), strict=True
    )
    return module.eval().requires_grad_(False)


def _load_v2(path: Path, digest: str) -> GlobalPromptLogitCalibratorV2:
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=digest,
        label="global prompt V2 checkpoint",
    )
    module = GlobalPromptLogitCalibratorV2()
    module.load_state_dict(
        _state_dict(payload, label="global prompt V2 checkpoint"), strict=True
    )
    return module.eval().requires_grad_(False)


def _write_npy_noclobber(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_supported_calibration(
    output_root: Path,
    frame_id: str,
    calibrated: SupportedCalibration,
) -> dict[str, object]:
    """Atomically write one validated frame using the dataclass contract."""

    frame_root = output_root / "scores" / str(frame_id)
    raw_path = frame_root / "raw_v1_probability.npy"
    support_path = frame_root / "supported.npy"
    calibrated_path = frame_root / "calibrated_v2_probability.npy"
    _write_npy_noclobber(
        raw_path, calibrated.raw_probability.numpy().astype(np.float32)
    )
    _write_npy_noclobber(support_path, calibrated.supported.numpy().astype(np.uint8))
    _write_npy_noclobber(
        calibrated_path,
        calibrated.calibrated_probability.numpy().astype(np.float32),
    )
    return {
        "raw_v1_probability": file_record(raw_path),
        "supported": file_record(support_path),
        "calibrated_v2_probability": file_record(calibrated_path),
        "shape": list(calibrated.raw_probability.shape),
        "supported_pixels": int(calibrated.supported.sum()),
        "strict_domain_audit": dict(calibrated.strict_domain_audit),
    }


def _load_authorized_inputs(
    args: argparse.Namespace,
) -> tuple[dict, dict, dict[str, Path]]:
    authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.execution_authority_sha256,
        label="registered-evidence external execution authority",
    )
    authority = validate_external_execution_authority(authority)
    records = _mapping(authority["records"], label="external execution records")
    paths = {name: _record_path(records, name) for name in records}
    promotion = _mapping(authority["promotion"], label="external promotion records")
    promotion_paths = {
        name: validate_file_record(promotion[name], label=f"promotion {name}")
        for name in (
            "v1_result",
            "v2_result",
            "cross_scene_confirmation_receipt",
            "cross_scene_confirmation_result",
        )
    }
    v2_source_result, _, _ = load_json_object(
        promotion_paths["v2_result"],
        expected_sha256=str(promotion["v2_result"]["sha256"]),
        label="clean V2 source result",
    )
    cross_scene_receipt, _, _ = load_json_object(
        promotion_paths["cross_scene_confirmation_receipt"],
        expected_sha256=str(promotion["cross_scene_confirmation_receipt"]["sha256"]),
        label="cross-scene promotion receipt",
    )
    cross_scene_result, _, _ = load_json_object(
        promotion_paths["cross_scene_confirmation_result"],
        expected_sha256=str(promotion["cross_scene_confirmation_result"]["sha256"]),
        label="cross-scene promotion result",
    )
    validate_external_promotion_documents(
        v2_source_result=v2_source_result,
        cross_scene_receipt=cross_scene_receipt,
        cross_scene_result=cross_scene_result,
        expected_cross_scene_result_sha256=str(
            promotion["cross_scene_confirmation_result"]["sha256"]
        ),
    )
    manifest, _, _ = load_json_object(
        paths["manifest"],
        expected_sha256=str(records["manifest"]["sha256"]),
        label="frozen benchmark manifest",
    )
    if manifest.get("protocol_hash") != authority["protocol_hash"]:
        raise ValueError("manifest and execution-authority protocol hashes differ")
    scene = _scene_record(manifest, str(authority["scene_id"]))
    if str(scene["prompt"]["frame_id"]) != str(authority["source_frame_id"]):
        raise ValueError("manifest and execution-authority source frames differ")
    evaluation = [str(value) for value in scene["evaluation_frame_ids"]]
    if evaluation != [str(value) for value in authority["target_frame_ids"]]:
        raise ValueError("manifest and execution-authority target frames differ")
    authority["verified_path"] = str(authority_path)
    authority["verified_sha256"] = authority_sha
    return authority, manifest, paths


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, object]:
    authority, manifest, paths = _load_authorized_inputs(args)
    records = _mapping(authority["records"], label="external execution records")
    report, _, _ = load_json_object(
        paths["source_responsibility_report"],
        expected_sha256=str(records["source_responsibility_report"]["sha256"]),
        label="source exact-W report",
    )
    if any(
        report.get(name, False) is not False
        for name in (
            "target_rgb_opened",
            "target_mask_opened",
            "target_metric_computed",
        )
    ):
        raise ValueError("source exact-W report is target-contaminated")
    source_w_sha = str(records["source_responsibility_cache"]["sha256"])
    if report.get("file_sha256") != source_w_sha:
        raise ValueError("source exact-W report binds a different cache")
    source_w_authority = PromptResponsibilityAuthority.from_dict(report["authority"])
    if (
        source_w_authority.scene_id != authority["scene_id"]
        or source_w_authority.frame_id != authority["source_frame_id"]
        or report.get("authority_sha256") != source_w_authority.digest
    ):
        raise ValueError("source exact-W authority differs from execution authority")
    source_w = load_prompt_responsibility_cache(
        paths["source_responsibility_cache"],
        expected_authority=source_w_authority,
        expected_file_sha256=source_w_sha,
    )

    native_shape = (source_w_authority.height, source_w_authority.width)
    positive = _load_binary_mask(paths["source_positive_mask"], shape=native_shape)
    if authority["prompt_mode"] == "signed_scribble":
        negative = _load_binary_mask(paths["source_negative_mask"], shape=native_shape)
    else:
        negative = ~positive

    capability = load_canonical_capability_bank(
        paths["capability_bank"],
        expected_source=str(authority["capability_source"]),
        require_row_authority=True,
        require_formal_projection_order=True,
    )
    state = load_factorized_primitive_state(
        paths["factorized_primitive_state"],
        expected_sha256=str(records["factorized_primitive_state"]["sha256"]),
        expected_xyz=capability.xyz,
        expected_valid=capability.valid,
    )
    features, statistics = build_external_registered_features(
        source_responsibility=source_w,
        positive_mask=positive,
        negative_mask=negative,
        prompt_mode=str(authority["prompt_mode"]),
        capability=capability,
        factorized_state=state,
    )
    v1 = _load_v1(paths["v1_checkpoint"], str(records["v1_checkpoint"]["sha256"]))
    primitive = infer_registered_primitive_unary(
        v1,
        features,
        device=args.device,
        chunk_size=args.chunk_size,
    )

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(
            f"refusing to reuse external prediction root: {output_root}"
        )
    output_root.mkdir(parents=True)
    primitive_path = output_root / "primitive_unary.pt"
    primitive_payload = {
        "schema": ADAPTER_SCHEMA,
        "schema_version": 1,
        "artifact_type": "registered_evidence_v1_primitive_unary",
        "scene_id": str(authority["scene_id"]),
        "protocol_hash": str(authority["protocol_hash"]),
        "prompt_mode": str(authority["prompt_mode"]),
        "feature_names": list(FEATURE_NAMES),
        "feature_values_sha256": tensor_sha256(features.values.float().cpu()),
        "primitive_probability": primitive.probability,
        "primitive_confidence": primitive.confidence,
        "analytic_probability": primitive.analytic_probability,
        "bounded_logit_residual": primitive.bounded_logit_residual,
        "source_statistics": statistics,
        "authorities": {
            "execution": {
                "path": str(authority["verified_path"]),
                "sha256": str(authority["verified_sha256"]),
            },
            "source_exact_w_authority_sha256": source_w_authority.digest,
            "source_exact_w_tensor_bundle_sha256": source_w.tensor_bundle_sha256,
            "capability_bank": dict(records["capability_bank"]),
            "factorized_primitive_state": dict(records["factorized_primitive_state"]),
            "v1_checkpoint": dict(records["v1_checkpoint"]),
        },
        "graph_constructed": False,
        "connected_selection_applied": False,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
        "written_before_target_ground_truth_open": True,
    }
    write_torch_noclobber(primitive_path, primitive_payload)
    primitive_record = file_record(primitive_path)

    camera_map, _, _ = load_json_object(
        paths["camera_map"],
        expected_sha256=str(records["camera_map"]["sha256"]),
        label="frozen camera map",
    )
    config = load_config(str(paths["carrier_config"]))
    views = resolve_protocol_views(
        manifest,
        scene_id=str(authority["scene_id"]),
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_map,
    )
    device = torch.device(args.device)
    model, _codec, scalar_renderer, _sharpener, _refiner, _field_config, _is_hybrid = (
        load_render_pipeline(
            str(paths["carrier_config"]),
            str(paths["carrier_checkpoint"]),
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
            expected_checkpoint_sha256=str(records["carrier_checkpoint"]["sha256"]),
        )
    )
    geometry_xyz = model.get_xyz().detach().float().cpu()
    if geometry_xyz.shape != capability.xyz.shape or not torch.allclose(
        geometry_xyz, capability.xyz.float().cpu(), atol=1e-6, rtol=0.0
    ):
        raise ValueError("carrier geometry and capability rows differ")
    v2 = _load_v2(paths["v2_checkpoint"], str(records["v2_checkpoint"]["sha256"]))
    primitive_device = primitive.probability.to(device)
    render_rows = torch.stack(
        (primitive_device, torch.ones_like(primitive_device)), dim=1
    )

    frame_records: dict[str, object] = {}
    for frame_id in authority["target_frame_ids"]:
        view = _view_by_frame(views, str(frame_id))
        pose = torch.from_numpy(view["w2c"].copy()).float().to(device)

        def render_callback(
            _probability: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if not torch.equal(_probability.float().cpu(), primitive.probability):
                raise ValueError(
                    "renderer callback received a different primitive unary"
                )
            result = (
                scalar_renderer.render_feature_rows(
                    model,
                    pose,
                    render_rows,
                    feature_height=int(scalar_renderer.image_height),
                    feature_width=int(scalar_renderer.image_width),
                    alpha_normalize=True,
                    contribution_gamma=1.0,
                    row_confidence=None,
                )["feature_map"]
                .detach()
                .float()
                .cpu()
            )
            if result.shape[0] != 2:
                raise ValueError(
                    "frozen scalar renderer returned the wrong channel count"
                )
            support_value = result[1]
            if not bool(torch.isfinite(support_value).all()) or bool(
                ((support_value < 0) | (support_value > 1.00001)).any()
            ):
                raise ValueError("frozen scalar renderer support channel is invalid")
            supported = support_value > 0
            if bool(supported.any()) and not torch.allclose(
                support_value[supported],
                torch.ones_like(support_value[supported]),
                atol=1e-5,
                rtol=0.0,
            ):
                raise ValueError(
                    "alpha-normalized unit support channel differs from one"
                )
            raw = result[0]
            if bool(raw[~supported].ne(0).any()):
                raise ValueError(
                    "frozen scalar renderer produced nonzero unsupported probability"
                )
            return raw, supported

        calibrated = render_then_calibrate(
            primitive.probability,
            renderer=render_callback,
            calibrator=v2,
            calibration_device=device,
        )
        frame_records[str(frame_id)] = _write_supported_calibration(
            output_root, str(frame_id), calibrated
        )

    receipt = {
        "schema": ADAPTER_SCHEMA,
        "schema_version": 1,
        "artifact_type": "registered_evidence_v2_pre_metric_prediction_receipt",
        "status": "sealed_before_target_ground_truth_open",
        "scene_id": str(authority["scene_id"]),
        "protocol_hash": str(authority["protocol_hash"]),
        "execution_authority": {
            "path": str(authority["verified_path"]),
            "sha256": str(authority["verified_sha256"]),
        },
        "primitive_unary": primitive_record,
        "frames": frame_records,
        "frame_count": len(frame_records),
        "method_contract": {
            "primitive": "RegisteredEvidenceToUnaryV1 graph-off full global rows",
            "renderer": "frozen alpha-normalized scalar Gaussian compositor",
            "calibration": "GlobalPromptLogitCalibratorV2 after render on supported pixels only",
            "unsupported_policy": "exact zero before and after calibration",
            "graph": False,
            "connected_selection": False,
            "per_scene_parameters": False,
            "threshold_scan": False,
        },
        "carrier": {
            "config": dict(records["carrier_config"]),
            "checkpoint": dict(records["carrier_checkpoint"]),
            "camera_map": dict(records["camera_map"]),
        },
        "checkpoints": {
            "v1": dict(records["v1_checkpoint"]),
            "v2": dict(records["v2_checkpoint"]),
        },
        "implementation": {
            "core": file_record(
                Path(__file__).resolve().parents[1]
                / "querying"
                / "registered_evidence_external_adapter.py"
            ),
            "materializer": file_record(Path(__file__).resolve()),
        },
        "sealed_before_target_ground_truth_open": True,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    receipt["content_sha256"] = canonical_json_sha256(receipt)
    receipt_path = output_root / "pre_metric_prediction_receipt.json"
    write_frozen_json(receipt_path, receipt)
    return {**receipt, "receipt": file_record(receipt_path)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execution-authority", required=True)
    result.add_argument("--execution-authority-sha256", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--chunk-size", type=int, default=65536)
    return result


def main() -> None:
    print(json.dumps(materialize(parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
