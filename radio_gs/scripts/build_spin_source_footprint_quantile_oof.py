#!/usr/bin/env python3
"""Seal the independent CPU-only SPIn source quantile-OOF authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.spin_source_footprint_quantile_calibration import (
    MAX_FOLD_QUANTILE_THRESHOLD_SPAN,
    QUANTILE_OOF_ARTIFACT_TYPE,
    build_quantile_oof_calibration,
    quantile_method_contract,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_file(path: str | Path, expected: str, label: str) -> Path:
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
    return payload


def build(args: argparse.Namespace) -> dict[str, object]:
    preregistration = _require_file(
        args.preregistration, args.preregistration_sha256, "v2 preregistration"
    )
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    if not isinstance(prereg, Mapping) or prereg.get("registration") != (
        "spin_source_footprint_crossfit_quantile_calibration_v2"
    ):
        raise ValueError("unexpected SPIn quantile preregistration")
    stopped_v1 = _require_file(
        args.stopped_v1_result,
        args.stopped_v1_result_sha256,
        "stopped v1 result",
    )
    fold_paths = {
        fold: _require_file(
            getattr(args, f"fold_{fold}"),
            getattr(args, f"fold_{fold}_sha256"),
            f"matched OOF fold {fold}",
        )
        for fold in range(3)
    }
    folds = {fold: _load_fold(path) for fold, path in fold_paths.items()}
    result = build_quantile_oof_calibration(folds)
    diagnostics = []
    tensors: dict[str, torch.Tensor] = {
        "source_visible": result.source_visible,
        "pooled_oof_quantile": result.pooled_oof_quantile,
        "pooled_oof_eligible": result.pooled_oof_eligible,
    }
    for diagnostic in result.fold_diagnostics:
        prefix = f"fold_{diagnostic.fold}_ecdf"
        tensors[f"{prefix}_support"] = diagnostic.ecdf.support
        tensors[f"{prefix}_cumulative"] = diagnostic.ecdf.cumulative
        diagnostics.append(
            {
                "fold": diagnostic.fold,
                "quantile_threshold": diagnostic.threshold,
                "weighted_soft_iou": diagnostic.weighted_soft_iou,
                "positive_quantile_weighted_mean": (
                    diagnostic.positive_quantile_mean
                ),
                "negative_quantile_weighted_mean": (
                    diagnostic.negative_quantile_mean
                ),
                "training_source_rows": diagnostic.training_source_rows,
                "training_source_weight": diagnostic.training_source_weight,
                "training_zero_score_weight_fraction": (
                    diagnostic.training_zero_score_weight_fraction
                ),
                "ecdf_unique_support_values": int(
                    diagnostic.ecdf.support.numel()
                ),
                "ecdf_total_weight": diagnostic.ecdf.total_weight,
            }
        )
    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    authority = {
        "schema_version": 1,
        "artifact_type": QUANTILE_OOF_ARTIFACT_TYPE,
        "status": (
            "pass_stable_source_only_quantile_gauge"
            if result.stable
            else "stop_unstable_quantile_thresholds"
        ),
        "scene_id": folds[0].get("scene_id"),
        "protocol_hash": folds[0].get("protocol_hash"),
        "preregistration": str(preregistration),
        "preregistration_sha256": str(args.preregistration_sha256),
        "stopped_v1_result": str(stopped_v1),
        "stopped_v1_result_sha256": str(args.stopped_v1_result_sha256),
        "matched_oof_folds": {
            str(fold): {
                "path": str(path),
                "file_sha256": getattr(args, f"fold_{fold}_sha256"),
                "content_sha256": folds[fold].get("content_sha256"),
            }
            for fold, path in fold_paths.items()
        },
        "method_contract": quantile_method_contract(),
        "method_contract_sha256": json_sha256(quantile_method_contract()),
        "fold_diagnostics": diagnostics,
        "fold_quantile_thresholds": [
            diagnostic.threshold for diagnostic in result.fold_diagnostics
        ],
        "fold_quantile_threshold_span": result.threshold_span,
        "maximum_fold_quantile_threshold_span": (
            MAX_FOLD_QUANTILE_THRESHOLD_SPAN
        ),
        "stable": result.stable,
        "t_completion_quantile": result.t_completion_quantile,
        "pooled_weighted_soft_iou": result.pooled_weighted_soft_iou,
        "pooled_eligible_rows": int(result.pooled_oof_eligible.sum()),
        "deployment_eligible_for_full_fit_source_gauge": result.stable,
        "target_distribution_opened": False,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
        "tensor_sha256": tensor_hashes,
    }
    content_sha256 = json_sha256(authority)
    payload = {**authority, "content_sha256": content_sha256, **tensors}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        if not isinstance(existing, Mapping) or existing.get("content_sha256") != content_sha256:
            raise FileExistsError(f"refusing to overwrite different v2 authority: {output}")
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
        raise FileExistsError(f"refusing to overwrite different v2 receipt: {receipt_path}")
    if not receipt_path.exists():
        receipt_path.write_text(encoded, encoding="utf-8")
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--stopped-v1-result", required=True)
    result.add_argument("--stopped-v1-result-sha256", required=True)
    for fold in range(3):
        result.add_argument(f"--fold-{fold}", required=True)
        result.add_argument(f"--fold-{fold}-sha256", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> None:
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
