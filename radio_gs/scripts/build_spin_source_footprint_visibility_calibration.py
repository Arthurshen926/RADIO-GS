#!/usr/bin/env python3
"""Seal the source-only SPIn visibility-calibration authority from three folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.spin_source_footprint_visibility_calibration import (
    CALIBRATION_ARTIFACT_TYPE,
    MAX_FOLD_THRESHOLD_SPAN,
    build_crossfit_calibration,
    matched_oof_method_contract,
)
from radio_gs.scripts.build_spin_source_footprint_matched_oof import (
    file_sha256,
    json_sha256,
)


def _require_file_sha(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    actual = file_sha256(resolved)
    if actual != str(expected):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return resolved


def _load_fold(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("matched OOF fold is not a mapping")
    hashes = payload.get("tensor_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("matched OOF fold lacks tensor hashes")
    for name, expected in hashes.items():
        if name not in payload or tensor_sha256(torch.as_tensor(payload[name])) != expected:
            raise ValueError(f"matched OOF tensor changed: {name}")
    contract = payload.get("method_contract")
    if not isinstance(contract, Mapping) or dict(contract) != matched_oof_method_contract():
        raise ValueError("matched OOF method contract differs")
    if payload.get("method_contract_sha256") != json_sha256(dict(contract)):
        raise ValueError("matched OOF method-contract digest differs")
    return payload


def _load_seen_threshold(
    path: Path,
    *,
    scene_id: str,
    protocol_hash: str,
) -> tuple[float, dict[str, object]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("artifact_type") != (
        "nvos_pre_metric_prediction_receipt_v1"
    ):
        raise ValueError("unexpected full-fit pre-metric receipt")
    if receipt.get("scene_id") != scene_id or receipt.get("protocol_hash") != protocol_hash:
        raise ValueError("full-fit receipt scene/protocol differs from OOF folds")
    if receipt.get("sealed_before_target_ground_truth_open") is not True or any(
        receipt.get(key) is not False
        for key in ("target_rgb_opened", "target_mask_opened", "target_metric_opened")
    ):
        raise ValueError("full-fit threshold receipt is not target blind")
    method = receipt.get("method_contract")
    if not isinstance(method, Mapping):
        raise ValueError("full-fit receipt lacks method contract")
    threshold = float(method.get("score_threshold", float("nan")))
    if not 0 < threshold < 1:
        raise ValueError("full-fit receipt source threshold is invalid")
    query = method.get("query_conditioned_diffusion")
    if not isinstance(query, Mapping) or query.get("kernel") != "ludvig_release_compat":
        raise ValueError("full-fit receipt is not the matched K201 interface")
    return threshold, receipt


def build(args: argparse.Namespace) -> dict[str, object]:
    preregistration = _require_file_sha(
        args.preregistration, args.preregistration_sha256, "preregistration"
    )
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    if not isinstance(prereg, Mapping) or prereg.get("registration") != (
        "spin_source_footprint_crossfit_visibility_calibration_v1"
    ):
        raise ValueError("unexpected SPIn visibility preregistration")
    fold_paths = {
        fold: _require_file_sha(
            getattr(args, f"fold_{fold}"),
            getattr(args, f"fold_{fold}_sha256"),
            f"matched OOF fold {fold}",
        )
        for fold in range(3)
    }
    folds = {fold: _load_fold(path) for fold, path in fold_paths.items()}
    calibration = build_crossfit_calibration(folds)
    scene_id = str(folds[0].get("scene_id", ""))
    protocol_hash = str(folds[0].get("protocol_hash", ""))
    premetric_path = _require_file_sha(
        args.full_fit_pre_metric_receipt,
        args.full_fit_pre_metric_receipt_sha256,
        "full-fit pre-metric receipt",
    )
    t_seen, _receipt = _load_seen_threshold(
        premetric_path,
        scene_id=scene_id,
        protocol_hash=protocol_hash,
    )
    tensors = {
        "source_visible": calibration.source_visible,
        "pooled_oof_probability": calibration.pooled_probability,
        "pooled_oof_eligible": calibration.pooled_eligible,
    }
    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    status = "pass_stable_source_only_calibration" if calibration.stable else (
        "stop_unstable_completion_thresholds"
    )
    authority = {
        "schema_version": 1,
        "artifact_type": CALIBRATION_ARTIFACT_TYPE,
        "status": status,
        "scene_id": scene_id,
        "protocol_hash": protocol_hash,
        "preregistration": str(preregistration),
        "preregistration_sha256": str(args.preregistration_sha256),
        "matched_oof_folds": {
            str(fold): {
                "path": str(path),
                "sha256": getattr(args, f"fold_{fold}_sha256"),
                "content_sha256": folds[fold].get("content_sha256"),
            }
            for fold, path in fold_paths.items()
        },
        "full_fit_pre_metric_receipt": str(premetric_path),
        "full_fit_pre_metric_receipt_sha256": str(
            args.full_fit_pre_metric_receipt_sha256
        ),
        "t_seen": float(t_seen),
        "t_completion": float(calibration.t_completion),
        "threshold_grid": matched_oof_method_contract()["threshold_grid"],
        "threshold_tie_break": "descending_grid_strict_improvement_first_maximizer",
        "weighted_soft_iou_formula": (
            "sum(selected*positive_weight)/(sum(positive_weight)+sum(selected*negative_weight))"
        ),
        "pooled_weighted_soft_iou": float(
            calibration.pooled_weighted_soft_iou
        ),
        "fold_thresholds": list(calibration.fold_thresholds),
        "fold_weighted_soft_iou": list(calibration.fold_weighted_soft_iou),
        "threshold_span": float(calibration.threshold_span),
        "maximum_registered_threshold_span": MAX_FOLD_THRESHOLD_SPAN,
        "stable": bool(calibration.stable),
        "source_visible_definition": "valid and immutable full-source reference_weight > 0",
        "target_coverage_formula": (
            "alpha_normalized_render_of_binary_source_visible_rows_using_target_pose_only"
        ),
        "spatial_threshold_formula": "c*t_seen+(1-c)*t_completion",
        "deployment_eligible": bool(calibration.stable),
        "tensor_sha256": tensor_hashes,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    content_sha256 = json_sha256(authority)
    payload = {**authority, "content_sha256": content_sha256, **tensors}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        if not isinstance(existing, Mapping) or existing.get("content_sha256") != content_sha256:
            raise FileExistsError(
                f"refusing to overwrite different visibility calibration: {output}"
            )
    else:
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
    receipt = {
        **authority,
        "content_sha256": content_sha256,
        "artifact_path": str(output),
        "artifact_sha256": file_sha256(output),
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if receipt_path.exists() and receipt_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(
            f"refusing to overwrite different visibility receipt: {receipt_path}"
        )
    if not receipt_path.exists():
        receipt_path.write_text(encoded, encoding="utf-8")
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    for fold in range(3):
        result.add_argument(f"--fold-{fold}", required=True)
        result.add_argument(f"--fold-{fold}-sha256", required=True)
    result.add_argument("--full-fit-pre-metric-receipt", required=True)
    result.add_argument("--full-fit-pre-metric-receipt-sha256", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> None:
    receipt = build(parser().parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
