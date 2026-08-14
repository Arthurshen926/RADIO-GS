"""One-shot evaluator-controlled scoring for a sealed construction release.

This path produces valid benchmark-effectiveness evidence while the public
formal-release authority remains deliberately disabled.  It never labels the
result as a leaderboard/formal row.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .construction_authority import audit_construction_authority
from .evaluate_predictions import (
    MODALITIES,
    _SEALED_BATCH_KEYS,
    _bind_method_field_inventory,
    _load_evaluation_inputs,
    _load_manifest,
    _load_npy,
    _selected_query_manifests,
    _sha256_file,
    evaluate_predictions,
)
from .protocol import (
    BENCHMARK_VERSION,
    UQISProtocolConfig,
    audit_release,
    canonical_json_sha256,
)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _replace(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_public_seal(
    root: Path, predictions_root: Path, seal_path: Path
) -> tuple[dict[str, np.ndarray], Mapping[str, Any], dict[str, Any] | None, str]:
    """Load the complete immutable prediction snapshot before private access."""

    release_path = root / "release.json"
    release = _load_manifest(release_path, required_key="manifest_sha256")
    seal = _load_manifest(seal_path, required_key="predictions")
    if set(seal) != _SEALED_BATCH_KEYS:
        raise ValueError("sealed prediction batch fields changed")
    if (
        seal.get("schema_version") != "scannet_uqis_prediction_batch_v1"
        or seal.get("status") != "sealed_before_private_evaluation"
        or seal.get("benchmark_version") != BENCHMARK_VERSION
        or seal.get("release_json_sha256") != _sha256_file(release_path)
        or seal.get("row_scope") != "universal_complete"
        or tuple(seal.get("modalities", [])) != MODALITIES
    ):
        raise ValueError("controlled evaluation requires one complete sealed system")
    expected, query_bindings = _selected_query_manifests(root, MODALITIES)
    if seal.get("query_manifests") != query_bindings:
        raise ValueError("sealed query-manifest bindings changed")
    records = seal.get("predictions")
    if not isinstance(records, list) or seal.get(
        "prediction_inventory_sha256"
    ) != canonical_json_sha256(records):
        raise ValueError("sealed prediction inventory digest changed")
    by_query = {
        str(row.get("query_id", "")): row for row in records if isinstance(row, Mapping)
    }
    if set(by_query) != expected or len(by_query) != len(records):
        raise ValueError("sealed prediction coverage changed")
    snapshot: dict[str, np.ndarray] = {}
    for query_id in sorted(expected):
        row = by_query[query_id]
        if set(row) != {"query_id", "relative_path", "bytes", "sha256", "dtype", "shape"}:
            raise ValueError(f"{query_id}: sealed prediction fields changed")
        if row.get("relative_path") != f"{query_id}.npy":
            raise ValueError(f"{query_id}: sealed prediction path changed")
        path = predictions_root / f"{query_id}.npy"
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("bytes", -1))
            or _sha256_file(path) != row.get("sha256")
        ):
            raise ValueError(f"{query_id}: sealed prediction binding changed")
        array = _load_npy(path, label=f"{query_id} sealed prediction")
        if (
            array.dtype != np.float32
            or list(array.shape) != row.get("shape")
            or str(array.dtype) != row.get("dtype")
            or array.ndim != 1
            or not np.isfinite(array).all()
            or bool(((array < 0) | (array > 1)).any())
        ):
            raise ValueError(f"{query_id}: prediction array changed")
        snapshot[query_id] = array
    method_binding = seal.get("method_run_manifest")
    if not isinstance(method_binding, Mapping) or set(method_binding) != {
        "path", "sha256", "schema_version", "status"
    }:
        raise ValueError("sealed method binding changed")
    method_path = Path(str(method_binding["path"])).resolve()
    if not method_path.is_file() or _sha256_file(method_path) != method_binding["sha256"]:
        raise ValueError("sealed method manifest changed")
    method = _load_manifest(method_path, required_key="status")
    if (
        method.get("benchmark_version") != BENCHMARK_VERSION
        or method.get("result_eligible") is not True
        or method.get("formal_benchmark_row_eligible") is not False
        or method.get("all_predictions_completed_before_private_evaluation") is not True
        or method.get("private_evaluator_inputs_opened") is not False
    ):
        raise ValueError("method is not eligible for controlled construction evaluation")
    field_binding, representation_scope = _bind_method_field_inventory(method, root)
    if (
        field_binding != seal.get("method_field_inventory")
        or representation_scope != seal.get("method_representation_scope")
    ):
        raise ValueError("sealed method field inventory changed")
    return snapshot, seal, field_binding, representation_scope


def evaluate_construction_authority_once(
    authority_path: str | Path,
    benchmark_dir: str | Path,
    prediction_dir: str | Path,
    sealed_batch_path: str | Path,
    ledger_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Score one sealed batch once, without minting a public formal row."""

    authority_file = Path(authority_path).resolve()
    root = Path(benchmark_dir).resolve()
    predictions = Path(prediction_dir).resolve()
    seal_path = Path(sealed_batch_path).resolve()
    ledger = Path(ledger_path).resolve()
    report_destination = Path(report_path).resolve()
    if ledger.exists() or report_destination.exists():
        raise FileExistsError("one-shot controlled evaluation was already claimed")

    # All 268 arrays and the complete method identity are loaded and verified
    # before construction authority or evaluator-private files are opened.
    snapshot, seal, field_binding, representation_scope = _verify_public_seal(
        root, predictions, seal_path
    )
    claim = {
        "schema_version": "scannet_uqis_controlled_evaluation_ledger_v1",
        "status": "claimed_predictions_verified_private_not_opened",
        "benchmark_version": BENCHMARK_VERSION,
        "sealed_prediction_batch_sha256": _sha256_file(seal_path),
        "prediction_count": len(snapshot),
        "private_authority_opened": False,
        "evaluation_completed": False,
    }
    _write_exclusive(ledger, claim)
    try:
        authority_audit = audit_construction_authority(authority_file, check_files=True)
        if not authority_audit.get("valid"):
            raise ValueError("construction authority audit failed")
        authority = _load_manifest(authority_file, required_key="candidate_release")
        candidate = authority["candidate_release"]
        if (
            Path(str(candidate["path"])).resolve() != (root / "release.json")
            or candidate["sha256"] != _sha256_file(root / "release.json")
            or authority.get("construction_formal_eligible") is not True
            or authority.get("public_formal_evaluation_enabled") is not False
        ):
            raise ValueError("construction authority/release binding changed")
        release_audit = audit_release(root, check_files=True)
        if not release_audit.get("valid"):
            raise ValueError("candidate release audit failed")
        release = _load_manifest(root / "release.json", required_key="protocol_config")
        config = UQISProtocolConfig(**release["protocol_config"])
        evaluator, scenes, xyz_by_scene, ids_by_scene = _load_evaluation_inputs(
            root / "target_manifest.evaluator.json", root / "scene_manifest.json"
        )
        report = evaluate_predictions(
            evaluator,
            scenes,
            snapshot,
            xyz_by_scene,
            ids_by_scene,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
            confidence=0.95,
            modalities=MODALITIES,
        )
        report.update({
            "evaluation_mode": "evaluator_controlled_construction_authority_one_shot",
            "result_valid": True,
            "formal_benchmark_eligible": False,
            "public_leaderboard_row": False,
            "validity_scope": (
                "sealed_complete_method_effectiveness_evidence; public formal release "
                "authority remains disabled"
            ),
            "construction_authority_sha256": authority_audit["authority_sha256"],
            "release_json_sha256": _sha256_file(root / "release.json"),
            "sealed_prediction_batch_sha256": _sha256_file(seal_path),
            "method_result_eligible": bool(seal.get("method_result_eligible")),
            "method_formal_row_eligible": bool(seal.get("method_formal_row_eligible")),
            "method_field_inventory": field_binding,
            "method_representation_scope": representation_scope,
            "single_universal_field_claim_eligible": False,
        })
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        report_destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _replace(ledger, {
            **claim,
            "status": "consumed_complete",
            "private_authority_opened": True,
            "evaluation_completed": True,
            "construction_authority_sha256": authority_audit["authority_sha256"],
            "report_path": str(report_destination),
            "report_sha256": _sha256_file(report_destination),
        })
        return report
    except Exception:
        _replace(ledger, {
            **claim,
            "status": "consumed_failed_after_private_authority_claim",
            "private_authority_opened": True,
            "evaluation_completed": False,
        })
        raise
