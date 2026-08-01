"""Isolated evaluator-only closeout for the LUDVIG PFPR adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from radio_gs.benchmarks.scannet_pfpr.evaluate_predictions import evaluate
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import sha256_file
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_c import _validate_file
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_d import (
    PHASE_D_SCHEMA_VERSION,
    PHASE_D_STATUS,
)
from radio_gs.benchmarks.scannet_pfpr.protocol import canonical_json_sha256


PHASE_E_SCHEMA_VERSION = "ludvig_pfpr_evaluation_closeout_v1"
PHASE_E_STATUS = "complete_custom_pfpr_adapter_one_scene"


class LudvigPFPRPhaseEError(RuntimeError):
    """Raised when predictions are not frozen before private evaluation."""


@dataclass(frozen=True)
class PhaseEConfig:
    phase_d_dir: Path
    expected_phase_d_manifest_sha256: str
    benchmark_dir: Path
    output_dir: Path
    scene_id: str = "scene0050_02"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LudvigPFPRPhaseEError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LudvigPFPRPhaseEError(f"Invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise LudvigPFPRPhaseEError(f"{label} must be a JSON object")
    return value


def audit_phase_d_predictions(config: PhaseEConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate all public scores before any evaluator-private file is opened."""

    root = config.phase_d_dir.resolve()
    manifest_path = root / "run_manifest.json"
    _validate_file(
        manifest_path,
        config.expected_phase_d_manifest_sha256,
        "Phase-D run manifest",
    )
    manifest = _load_json(manifest_path, "Phase-D run manifest")
    if manifest.get("schema_version") != PHASE_D_SCHEMA_VERSION:
        raise LudvigPFPRPhaseEError("Phase-D schema changed")
    if manifest.get("status") != PHASE_D_STATUS or manifest.get("result_eligible") is not False:
        raise LudvigPFPRPhaseEError("Phase-D status changed")
    if str(manifest.get("scene_id", "")) != str(config.scene_id):
        raise LudvigPFPRPhaseEError("Phase-D scene differs from requested scene")
    queries = manifest.get("queries")
    if not isinstance(queries, list) or len(queries) != 10:
        raise LudvigPFPRPhaseEError("Phase-D query inventory changed")
    if manifest.get("queries_sha256") != canonical_json_sha256(queries):
        raise LudvigPFPRPhaseEError("Phase-D query binding changed")
    validated: list[dict[str, Any]] = []
    expected_names: set[str] = set()
    for record in queries:
        if not isinstance(record, Mapping):
            raise LudvigPFPRPhaseEError("Phase-D query record is invalid")
        query_id = str(record.get("query_id", ""))
        binding = record.get("scores")
        if not query_id or query_id in expected_names or not isinstance(binding, Mapping):
            raise LudvigPFPRPhaseEError("Phase-D query score binding is invalid")
        expected_names.add(query_id)
        relative = Path(str(binding.get("relative_path", "")))
        if relative != Path("predictions") / f"{query_id}.npy":
            raise LudvigPFPRPhaseEError("Phase-D prediction path changed")
        path = root / relative
        file_binding = _validate_file(
            path, str(binding.get("sha256", "")), f"Phase-D score {query_id}"
        )
        scores = np.load(path, allow_pickle=False)
        if list(scores.shape) != binding.get("shape") or str(scores.dtype) != binding.get("dtype"):
            raise LudvigPFPRPhaseEError("Phase-D prediction shape/dtype changed")
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise LudvigPFPRPhaseEError("Phase-D prediction is not a finite vector")
        validated.append({"query_id": query_id, **file_binding, "shape": list(scores.shape)})
    actual_names = {path.stem for path in (root / "predictions").glob("*.npy")}
    if actual_names != expected_names:
        raise LudvigPFPRPhaseEError("Phase-D prediction directory has extra/missing vectors")
    return manifest, validated


def run_phase_e(config: PhaseEConfig, *, argv: Sequence[str] = ()) -> dict[str, Any]:
    """Open private anchors only after the Phase-D prediction audit succeeds."""

    output = config.output_dir.resolve()
    if output.exists():
        raise LudvigPFPRPhaseEError(f"Refusing to overwrite Phase-E output: {output}")
    phase_d, predictions = audit_phase_d_predictions(config)
    evaluator_path = config.benchmark_dir.resolve() / "manifest.evaluator.json"
    if not evaluator_path.is_file():
        raise LudvigPFPRPhaseEError(f"Missing evaluator manifest: {evaluator_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.phase_e_tmp_", dir=output.parent))
    try:
        # This call is the first evaluator-private read in the Phase-D/E chain.
        evaluation = evaluate(
            config.benchmark_dir,
            config.phase_d_dir / "predictions",
            temporary / "evaluation.json",
            scene_names=(str(config.scene_id),),
        )
        evaluator_binding = {
            "path": str(evaluator_path),
            "bytes": evaluator_path.stat().st_size,
            "sha256": sha256_file(evaluator_path),
        }
        evaluation_path = temporary / "evaluation.json"
        manifest: dict[str, Any] = {
            "schema_version": PHASE_E_SCHEMA_VERSION,
            "status": PHASE_E_STATUS,
            "result_eligible": True,
            "official_ludvig_reproduction": False,
            "paper_metric_comparable": False,
            "scene_id": str(config.scene_id),
            "attempt_dir": str(output),
            "argv": list(argv),
            "phase_d": {
                "root": str(config.phase_d_dir.resolve()),
                "manifest_sha256": str(config.expected_phase_d_manifest_sha256),
                "predictions": predictions,
                "predictions_sha256": canonical_json_sha256(predictions),
            },
            "privacy_boundary": {
                "method_visible_scoring_complete_before_private_open": True,
                "phase_d_evaluator_private_manifest_opened": False,
                "phase_e_evaluator_private_manifest_opened": True,
                "evaluator_manifest": evaluator_binding,
            },
            "evaluation": {
                "relative_path": "evaluation.json",
                "bytes": evaluation_path.stat().st_size,
                "sha256": sha256_file(evaluation_path),
                "query_count": int(evaluation["protocol"]["query_count"]),
                "scene_count": int(evaluation["protocol"]["scene_count"]),
                "metrics_query_micro": evaluation["metrics_query_micro"],
                "metrics_scene_macro": evaluation["metrics_scene_macro"],
            },
            "interpretation": (
                "One-scene custom PFPR adapter sanity result. LUDVIG publishes no PFPR "
                "task head or PFPR paper number, so these metrics are benchmark-local only."
            ),
            "phase_status": {
                "phase_a_cpu_staging": "bound_complete",
                "phase_b_dino_scene_features_and_pca": "bound_complete",
                "phase_c_inverse_render_uplift": "bound_complete",
                "phase_d_pfpr_crop_scoring": "bound_complete",
                "phase_e_pfpr_evaluation": "complete",
            },
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest_sha256 = sha256_file(manifest_path)
        (temporary / "run_manifest.sha256").write_text(manifest_sha256 + "\n", encoding="ascii")
        if output.exists():
            raise LudvigPFPRPhaseEError(f"Refusing concurrent overwrite: {output}")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**manifest, "run_manifest_sha256": manifest_sha256}

