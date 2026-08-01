#!/usr/bin/env python3
"""One-shot audit90 confirmation for a frozen readout interpolation.

The audit bank is inaccessible until the frozen query-free selection has been
independently reconstructed from its externally SHA-bound diagnostic and the
selected endpoint interpolation has been replayed on the Surface validation
caches.  An immutable opening receipt is then published *before* the audit
bank is loaded.  Consequently, a crash after opening is fail-closed and cannot
silently turn the confirmation split into a reusable tuning split.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation import text_response_fidelity as fidelity
from radio_gs.evaluation.text_response_fidelity import (
    aggregate_paired_seed_gate_from_same_process_metrics,
    evaluate_response_fidelity,
    row_identity_sha256,
    tensor_sha256,
)
from radio_gs.interfaces import surface_region_contract
from radio_gs.interfaces import surface_region_summary
from radio_gs.losses import direct_point_query_logit_distill_loss
from radio_gs.models import siglip_projection
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import build_target_blind_siglip2_embedding_artifact as text_bank_builder
from radio_gs.scripts import diagnose_surface_readout_weight_interpolation as diagnostic_module
from radio_gs.scripts import eval_text_response_fidelity_gate as eval_gate
from radio_gs.scripts import freeze_surface_readout_weight_interpolation_selection as selector
from radio_gs.scripts import materialize_surface_text_response_descriptors as materializer
from radio_gs.scripts import train_surface_region_summary_readout as summary_trainer
from radio_gs.scripts import train_surface_region_text_response_distill as distill_trainer
from radio_gs.scripts.eval_text_response_fidelity_gate import (
    FORMAL_HISTORICAL_TEXT_BANKS,
    load_text_embedding_bank,
)
from radio_gs.scripts.train_surface_region_summary_readout import _targets
from radio_gs.scripts.train_surface_region_text_response_distill import (
    compute_query_free_response_selection_metrics,
    load_fit_text_embedding_bank,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_bytes,
    load_json_object,
    sha256_file,
    validate_file_record,
    write_bytes_noclobber,
    write_frozen_json,
)
from radio_gs.utils import immutable_artifacts


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_readout_weight_interpolation_audit90_confirmation"
OPENING_ARTIFACT_TYPE = "surface_readout_weight_interpolation_audit90_opening_receipt"
REQUIRED_SEEDS = (0, 1, 2)
SURFACE_METRICS = selector.SURFACE_METRICS
SURFACE_NONINFERIORITY_TOLERANCE = selector.SURFACE_NONINFERIORITY_TOLERANCE
MINIMUM_IMPROVED_SEEDS = 2
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260731
QUALITY_NONINFERIORITY_TOLERANCE = 0.0
METRIC_RECOMPUTE_TOLERANCE = 1e-7
AUDIT_BANK_AUTHORITY = FORMAL_HISTORICAL_TEXT_BANKS["audit"]
IMPLEMENTATION_CLOSURE = (
    ("audit_confirmation_executor", None),
    ("text_response_fidelity", fidelity),
    ("surface_region_contract", surface_region_contract),
    ("surface_region_summary", surface_region_summary),
    ("direct_point_query_logit_distill_loss", direct_point_query_logit_distill_loss),
    ("siglip_projection", siglip_projection),
    ("target_blind_text_bank_builder", text_bank_builder),
    ("interpolation_diagnostic", diagnostic_module),
    ("text_response_fidelity_gate", eval_gate),
    ("interpolation_selector", selector),
    ("descriptor_materializer", materializer),
    ("surface_summary_trainer", summary_trainer),
    ("surface_text_distill_trainer", distill_trainer),
    ("immutable_artifacts", immutable_artifacts),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _cpu_only_preflight() -> None:
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"},
        "set CUDA_VISIBLE_DEVICES='' (or -1) for this CPU-only audit confirmation",
    )


def _declared_regular_file(raw: str | Path, *, label: str) -> Path:
    path = Path(raw)
    _require(path.is_absolute(), f"{label} must be an absolute path")
    resolved = path.resolve(strict=True)
    _require(resolved == path and resolved.is_file(), f"{label} must be canonical and non-symlinked")
    return resolved


def _fresh_output(raw: str | Path, *, label: str) -> Path:
    path = Path(raw)
    _require(path.is_absolute(), f"{label} must be an absolute path")
    _require(not path.exists() and not path.is_symlink(), f"{label} already exists")
    parent = path.parent.resolve(strict=True)
    _require(parent.is_dir(), f"{label} parent must be a directory")
    canonical = parent / path.name
    _require(
        not canonical.exists() and not canonical.is_symlink(),
        f"{label} canonical target already exists",
    )
    return canonical


def _file_record(path: Path) -> dict[str, str]:
    resolved = _declared_regular_file(path, label="bound file")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _implementation_closure_records() -> list[dict[str, str]]:
    """Rehash the fixed repo-source closure used by formal confirmation."""

    records = []
    observed_roles = set()
    for role, module in IMPLEMENTATION_CLOSURE:
        _require(role not in observed_roles, "implementation closure roles must be unique")
        observed_roles.add(role)
        path = Path(__file__) if module is None else Path(module.__file__)
        records.append({"role": role, **_file_record(path.resolve())})
    return records


def _finite(value: object, *, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite numeric",
    )
    return float(value)


def _metric_mapping_close(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    names: Sequence[str],
    label: str,
) -> None:
    for name in names:
        left = _finite(observed.get(name), label=f"{label} observed {name}")
        right = _finite(expected.get(name), label=f"{label} expected {name}")
        _require(
            math.isclose(
                left,
                right,
                rel_tol=METRIC_RECOMPUTE_TOLERANCE,
                abs_tol=METRIC_RECOMPUTE_TOLERANCE,
            ),
            f"{label} {name} differs from CPU recomputation",
        )


def _expected_selection_payload(
    diagnostic: Mapping[str, Any],
    *,
    diagnostic_path: Path,
) -> dict[str, Any]:
    """Independently reconstruct the selector's exact frozen payload."""

    view = selector._selection_view(diagnostic)
    decision = selector.select_from_view(view)
    alpha = decision["selected_alpha"]
    per_seed = []
    for seed in REQUIRED_SEEDS:
        record = view["per_seed"][seed]
        baseline = record["points"][0.0]["surface"]
        selected = record["points"][alpha]["surface"]
        deltas = {
            metric: selected[metric] - baseline[metric] for metric in SURFACE_METRICS
        }
        checks = {
            metric: selected[metric]
            >= baseline[metric] - SURFACE_NONINFERIORITY_TOLERANCE
            for metric in SURFACE_METRICS
        }
        per_seed.append(
            {
                "seed": seed,
                "selected_alpha": alpha,
                "control_checkpoint": record["control"],
                "candidate_checkpoint": record["candidate"],
                "interpolation": {
                    "formula": "theta_alpha=(1-alpha)*theta_control+alpha*theta_candidate",
                    "pairing": "same_seed_only",
                    "materialized_checkpoint": None,
                },
                "alpha0_surface": baseline,
                "selected_surface": selected,
                "selected_minus_alpha0": deltas,
                "minimum_allowed_surface": {
                    metric: baseline[metric] - SURFACE_NONINFERIORITY_TOLERANCE
                    for metric in SURFACE_METRICS
                },
                "surface_checks": checks,
                "surface_passes": all(checks.values()),
            }
        )
    base_fit = view["aggregate_fit"][0.0]
    selected_fit = view["aggregate_fit"][alpha]
    aggregate_fit = {
        "scope": "aggregate_seed_mean",
        "alpha0": base_fit,
        "selected": selected_fit,
        "selected_minus_alpha0": {
            metric: selected_fit[metric] - base_fit[metric]
            for metric in (selector.FIT_SUPPORT_METRIC, selector.FIT_ERROR_METRIC)
        },
        "checks": {
            "support_top1_strictly_improves": (
                selected_fit[selector.FIT_SUPPORT_METRIC]
                > base_fit[selector.FIT_SUPPORT_METRIC]
            ),
            "smooth_l1_noninferior": (
                selected_fit[selector.FIT_ERROR_METRIC]
                <= base_fit[selector.FIT_ERROR_METRIC]
            ),
        },
    }
    selector_path = Path(selector.__file__).resolve()
    return {
        "schema_version": selector.SCHEMA_VERSION,
        "artifact_type": selector.ARTIFACT_TYPE,
        "status": "frozen_query_free_selection_audit_unopened",
        "diagnostic": {
            "path": str(diagnostic_path),
            "sha256": selector.FORMAL_DIAGNOSTIC_SHA256,
            "artifact_type": selector.FORMAL_DIAGNOSTIC_ARTIFACT_TYPE,
            "diagnostic_contract_sha256": selector.FORMAL_DIAGNOSTIC_CONTRACT_SHA256,
        },
        "selection_contract": {
            "policy": selector.SELECTION_POLICY,
            "fixed_alphas": list(selector.FIXED_ALPHAS),
            "surface_scope": "every_seed",
            "surface_metrics": list(SURFACE_METRICS),
            "surface_noninferiority_tolerance": SURFACE_NONINFERIORITY_TOLERANCE,
            "surface_rule": "each_metric_gte_same_seed_alpha0_minus_tolerance",
            "fit_scope": "aggregate_seed_mean",
            "fit_support_rule": "strictly_greater_than_alpha0",
            "fit_smooth_l1_rule": "less_than_or_equal_to_alpha0",
            "selection_rule": "minimum_strictly_positive_feasible_alpha",
            "formal_unique_positive_alpha_required": selector.EXPECTED_FORMAL_SELECTED_ALPHA,
            "dev_fields_read": False,
            "dev_values_copied": False,
            "audit_opened": False,
        },
        "feasible_alphas": decision["feasible_alphas"],
        "positive_feasible_alphas": decision["positive_feasible_alphas"],
        "selected_alpha": alpha,
        "per_alpha": decision["per_alpha"],
        "per_seed": per_seed,
        "aggregate_fit": aggregate_fit,
        "selected_interpolation": {
            "alpha": alpha,
            "formula": "theta_alpha=(1-alpha)*theta_control+alpha*theta_candidate",
            "pairing": "same_seed_only",
            "checkpoint_materialized": False,
        },
        "audit": {"opened": False, "status": "unopened", "artifact": None},
        "selector_implementation": {
            "path": str(selector_path),
            "sha256": sha256_file(selector_path),
        },
    }


def validate_frozen_selection(
    selection_path: Path,
    selection_sha256: str,
    diagnostic_path: Path,
    diagnostic_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate external SHA authorities and replay the no-audit decision."""

    _require(_is_sha256(selection_sha256), "selection SHA256 is invalid")
    _require(_is_sha256(diagnostic_sha256), "diagnostic SHA256 is invalid")
    selection_path = _declared_regular_file(selection_path, label="frozen selection")
    diagnostic_path = _declared_regular_file(diagnostic_path, label="interpolation diagnostic")
    diagnostic = selector._validate_formal_diagnostic_identity(
        diagnostic_path,
        diagnostic_sha256,
    )
    selection, observed_sha, source = load_json_object(
        selection_path,
        expected_sha256=selection_sha256,
        label="frozen interpolation selection",
    )
    _require(source == selection_path and observed_sha == selection_sha256, "selection SHA drifted")
    expected = _expected_selection_payload(diagnostic, diagnostic_path=diagnostic_path)
    _require(selection == expected, "frozen selection differs from independent recomputation")
    return selection, diagnostic


@torch.inference_mode()
def _evaluate_readout(
    model: torch.nn.Module,
    head: torch.nn.Module,
    data: Mapping[str, Any],
    *,
    fit_embeddings: torch.Tensor,
    scene_ids: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    token_cosines: list[float] = []
    descriptor_cosines: list[float] = []
    all_view_cosines: list[float] = []
    students: list[torch.Tensor] = []
    teachers: list[torch.Tensor] = []
    row_count = len(data["radio_features"])
    for start in range(0, row_count, int(batch_size)):
        rows = torch.arange(start, min(start + int(batch_size), row_count))
        target_token, teacher, all_descriptors, teacher_mask = _targets(data, rows)
        predicted_token = model(
            data["radio_features"][rows],
            data["geometry"][rows],
            anchor_index=data["anchor_index"][rows],
            token_mask=data["token_mask"][rows],
            reliability=data["reliability"][rows],
        )
        student = F.normalize(
            head(predicted_token[:, None])[:, 0].float(),
            dim=-1,
            eps=1e-8,
        ).cpu()
        token_cosines.extend(
            F.cosine_similarity(predicted_token.cpu(), target_token, dim=-1).tolist()
        )
        descriptor_cosines.extend(F.cosine_similarity(student, teacher, dim=-1).tolist())
        all_view_cosines.extend(
            torch.einsum("bd,bvd->bv", student, all_descriptors)[teacher_mask].tolist()
        )
        students.append(student)
        teachers.append(teacher)
    student = torch.cat(students).contiguous()
    teacher = torch.cat(teachers).contiguous()
    relation_fidelity = 1.0 - diagnostic_module._relation_smooth_l1(student, teacher)
    surface = {
        "summary_token_cosine": sum(token_cosines) / len(token_cosines),
        "mean_descriptor_cosine": sum(descriptor_cosines) / len(descriptor_cosines),
        "all_view_descriptor_cosine": sum(all_view_cosines) / len(all_view_cosines),
        "relation_fidelity": relation_fidelity,
    }
    fit = compute_query_free_response_selection_metrics(
        student,
        teacher,
        fit_embeddings,
        scene_ids=scene_ids,
    )
    return {"surface": surface, "fit": fit, "student": student, "teacher": teacher}


def _load_query_free_runtime(
    selection: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Replay endpoint provenance and selected metrics before audit opening."""

    _require(batch_size > 0, "batch_size must be positive")
    bindings = diagnostic.get("input_bindings")
    _require(isinstance(bindings, Mapping), "diagnostic input bindings are missing")
    validation_records = bindings.get("validation_caches")
    _require(isinstance(validation_records, list) and validation_records, "validation caches missing")
    validation_paths = []
    for index, record in enumerate(validation_records):
        _require(isinstance(record, Mapping), f"validation cache {index} binding is invalid")
        path = _declared_regular_file(record.get("path", ""), label=f"validation cache {index}")
        _require(record.get("sha256") == sha256_file(path), f"validation cache {index} SHA drifted")
        validation_paths.append(path)
    radio_binding = bindings.get("radio_checkpoint")
    control_manifest_binding = bindings.get("control_binding_manifest")
    candidate_manifest_binding = bindings.get("candidate_run_manifest")
    fit_binding = bindings.get("fit_text_bank")
    for label, value in (
        ("RADIO checkpoint", radio_binding),
        ("control binding manifest", control_manifest_binding),
        ("candidate run manifest", candidate_manifest_binding),
    ):
        _require(isinstance(value, Mapping), f"{label} binding is missing")
        validate_file_record(
            {"path": value.get("path"), "sha256": value.get("sha256")},
            label=label,
        )
    _require(isinstance(fit_binding, Mapping), "fit text-bank binding is missing")
    radio_path = _declared_regular_file(radio_binding["path"], label="RADIO checkpoint")
    control_manifest = _declared_regular_file(
        control_manifest_binding["path"], label="control binding manifest"
    )
    candidate_manifest = _declared_regular_file(
        candidate_manifest_binding["path"], label="candidate run manifest"
    )
    fit_path = _declared_regular_file(fit_binding["path"], label="fit text bank")
    fit_manifest = _declared_regular_file(
        fit_binding["manifest_path"], label="fit text-bank manifest"
    )
    _require(fit_binding.get("sha256") == sha256_file(fit_path), "fit text-bank SHA drifted")
    _require(
        fit_binding.get("manifest_sha256") == sha256_file(fit_manifest),
        "fit text-bank manifest SHA drifted",
    )

    data, cache_meta = materializer._load_validation_caches(
        validation_paths,
        include_summary_tokens=True,
    )
    _require(cache_meta["caches"] == validation_records, "diagnostic validation-cache index drifted")
    radio_sha = sha256_file(radio_path)
    _require(
        radio_sha == radio_binding["sha256"] == cache_meta["radio_checkpoint_sha256"],
        "RADIO checkpoint/cache binding drifted",
    )
    fit_bank = load_fit_text_embedding_bank(fit_path, fit_manifest)
    fit_bank = {
        **fit_bank,
        "path": fit_path,
        "artifact_sha256": sha256_file(fit_path),
        "manifest_path": fit_manifest,
        "manifest_sha256": sha256_file(fit_manifest),
    }
    control_authority = diagnostic_module._control_checkpoint_authority_index(
        control_manifest
    )
    _, candidate_authority = diagnostic_module._candidate_checkpoint_authority_index(
        candidate_manifest
    )
    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).cpu().eval()
    head.requires_grad_(False)

    diagnostic_rows = {int(row["seed"]): row for row in diagnostic["per_seed"]}
    selection_rows = {int(row["seed"]): row for row in selection["per_seed"]}
    alpha = float(selection["selected_alpha"])
    results = []
    for seed in REQUIRED_SEEDS:
        selected_row = selection_rows[seed]
        control_path = _declared_regular_file(
            selected_row["control_checkpoint"]["path"],
            label=f"seed {seed} control checkpoint",
        )
        candidate_path = _declared_regular_file(
            selected_row["candidate_checkpoint"]["path"],
            label=f"seed {seed} candidate checkpoint",
        )
        _require(
            control_path in control_authority and candidate_path in candidate_authority,
            f"seed {seed} endpoint is absent from external checkpoint authority",
        )
        control_external = control_authority[control_path]
        candidate_external = candidate_authority[candidate_path]
        control = diagnostic_module._validate_control_checkpoint(
            control_path,
            binding_manifest=control_manifest,
            expected_sha256=control_external["sha256"],
            expected_seed=seed,
            cache_meta=cache_meta,
            radio_path=radio_path,
            radio_sha256=radio_sha,
        )
        candidate = diagnostic_module._validate_candidate_checkpoint(
            candidate_path,
            run_manifest=candidate_manifest,
            expected_sha256=candidate_external["sha256"],
            expected_seed=seed,
            cache_meta=cache_meta,
            radio_path=radio_path,
            radio_sha256=radio_sha,
            fit_bank=fit_bank,
        )
        diagnostic_module.assert_interpolable(control["payload"], candidate["payload"])
        model = control["model"]
        model.load_state_dict(control["payload"]["state_dict"], strict=True)
        control_eval = _evaluate_readout(
            model,
            head,
            data,
            fit_embeddings=fit_bank["embeddings"],
            scene_ids=cache_meta["scene_ids"],
            batch_size=batch_size,
        )
        interpolated_state = diagnostic_module.interpolate_state_dict(
            control["payload"]["state_dict"],
            candidate["payload"]["state_dict"],
            alpha,
        )
        model.load_state_dict(interpolated_state, strict=True)
        selected_eval = _evaluate_readout(
            model,
            head,
            data,
            fit_embeddings=fit_bank["embeddings"],
            scene_ids=cache_meta["scene_ids"],
            batch_size=batch_size,
        )
        points = {float(point["alpha"]): point for point in diagnostic_rows[seed]["points"]}
        _metric_mapping_close(
            control_eval["surface"], points[0.0]["surface"], names=SURFACE_METRICS,
            label=f"seed {seed} alpha0 Surface",
        )
        _metric_mapping_close(
            selected_eval["surface"], points[alpha]["surface"], names=SURFACE_METRICS,
            label=f"seed {seed} selected Surface",
        )
        fit_names = (
            "text_support_top1_agreement",
            "text_support_valid_query_ratio",
            "text_response_smooth_l1",
            "descriptor_relation_smooth_l1",
        )
        _metric_mapping_close(
            control_eval["fit"], points[0.0]["fit"], names=fit_names,
            label=f"seed {seed} alpha0 fit",
        )
        _metric_mapping_close(
            selected_eval["fit"], points[alpha]["fit"], names=fit_names,
            label=f"seed {seed} selected fit",
        )
        _require(
            tensor_sha256(control_eval["teacher"]) == tensor_sha256(selected_eval["teacher"]),
            f"seed {seed} teacher changed across interpolation",
        )
        results.append(
            {
                "seed": seed,
                "control_checkpoint": _file_record(control_path),
                "candidate_checkpoint": _file_record(candidate_path),
                "control": control_eval,
                "interpolated": selected_eval,
            }
        )
    expected_scenes = list(fidelity._load_frozen_validation_scenes())
    _require(cache_meta["scenes"] == expected_scenes, "validation scenes differ from preregistration")
    return {
        "per_seed": results,
        "scene_ids": cache_meta["scene_ids"],
        "region_ids": cache_meta["region_ids"],
        "scenes": cache_meta["scenes"],
        "rows_sha256": row_identity_sha256(cache_meta["scene_ids"], cache_meta["region_ids"]),
        "input_bindings": bindings,
    }


def _opening_receipt_payload(
    *,
    selection_path: Path,
    selection_sha256: str,
    diagnostic_path: Path,
    diagnostic_sha256: str,
    output: Path,
    implementation_closure: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    closure = [dict(record) for record in implementation_closure]
    _require(
        closure and closure[0].get("role") == "audit_confirmation_executor",
        "implementation closure must begin with the audit executor",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": OPENING_ARTIFACT_TYPE,
        "status": "one_shot_opening_authorization_committed",
        "opening_count": 1,
        "audit_bank_loads_authorized": 1,
        "selection_validation_completed": True,
        "query_free_recomputation_completed": True,
        "selection": {"path": str(selection_path), "sha256": selection_sha256},
        "diagnostic": {"path": str(diagnostic_path), "sha256": diagnostic_sha256},
        "declared_audit_bank": {
            "path": AUDIT_BANK_AUTHORITY["artifact_path"],
            "sha256": AUDIT_BANK_AUTHORITY["artifact_sha256"],
            "manifest_path": AUDIT_BANK_AUTHORITY["manifest_path"],
            "manifest_sha256": AUDIT_BANK_AUTHORITY["manifest_sha256"],
            "split": "audit",
            "expected_query_count": 90,
        },
        "intended_confirmation_output": str(output),
        "implementation": {
            "path": closure[0]["path"],
            "sha256": closure[0]["sha256"],
        },
        "implementation_closure": closure,
    }


def _load_audit_bank_after_receipt(receipt: Path) -> dict[str, Any]:
    _require(receipt.is_file(), "audit opening receipt was not durably published")
    artifact = _declared_regular_file(
        AUDIT_BANK_AUTHORITY["artifact_path"], label="formal audit90 bank"
    )
    manifest = _declared_regular_file(
        AUDIT_BANK_AUTHORITY["manifest_path"], label="formal audit90 bank manifest"
    )
    bank = load_text_embedding_bank(artifact, manifest, "audit")
    _require(
        bank["file_sha256"] == AUDIT_BANK_AUTHORITY["artifact_sha256"]
        and bank["manifest_sha256"] == AUDIT_BANK_AUTHORITY["manifest_sha256"]
        and len(bank["query_ids"]) == 90,
        "formal audit90 bank authority differs after one-shot opening",
    )
    return bank


def _response_report(
    *,
    method_id: str,
    seed: int,
    student: torch.Tensor,
    teacher: torch.Tensor,
    runtime: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = evaluate_response_fidelity(
        student,
        teacher,
        bank["embeddings"],
        scene_ids=runtime["scene_ids"],
        region_ids=runtime["region_ids"],
        query_ids=bank["query_ids"],
    )
    return {
        "method_id": method_id,
        "seed": seed,
        "query_split": "audit",
        "descriptor_rows_sha256": runtime["rows_sha256"],
        "teacher_descriptors_sha256": tensor_sha256(teacher),
        "query_bank": {
            "path": str(bank["path"]),
            "sha256": bank["file_sha256"],
            "manifest_path": str(bank["manifest_path"]),
            "manifest_sha256": bank["manifest_sha256"],
            "vocabulary_sha256": bank["vocabulary_sha256"],
            "query_split": "audit",
            "selected_queries": len(bank["selected_records"]),
            "selected_records_sha256": bank["selected_records_sha256"],
            "ordered_records_sha256": bank["ordered_records_sha256"],
            "embedding_tensor_sha256": bank["embedding_tensor_sha256"],
            "embedding_semantic_sha256": bank["embedding_semantic_sha256"],
            "text_encoder": bank["text_encoder"],
        },
        "metrics": metrics,
    }


def _paired_gate_in_memory(
    controls: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    source_authority: Mapping[str, Any],
) -> dict[str, Any]:
    return aggregate_paired_seed_gate_from_same_process_metrics(
        controls,
        candidates,
        source_authority=source_authority,
        required_seeds=REQUIRED_SEEDS,
        minimum_improved_seeds=MINIMUM_IMPROVED_SEEDS,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        quality_noninferiority_tolerance=QUALITY_NONINFERIORITY_TOLERANCE,
        phase="audit",
    )


def _same_process_source_authority(
    *,
    selection_path: Path,
    selection_sha256: str,
    diagnostic_path: Path,
    diagnostic_sha256: str,
    runtime: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    cache_records = [
        {"path": record["path"], "sha256": record["sha256"]}
        for record in runtime["input_bindings"]["validation_caches"]
    ]
    return {
        "schema_version": 1,
        "authority_type": "same_process_interpolation_audit_response_metrics_v1",
        "selection": {"path": str(selection_path), "sha256": selection_sha256},
        "diagnostic": {"path": str(diagnostic_path), "sha256": diagnostic_sha256},
        "validation_caches": cache_records,
        "endpoints": [
            {
                "seed": row["seed"],
                "control": row["control_checkpoint"],
                "candidate": row["candidate_checkpoint"],
            }
            for row in runtime["per_seed"]
        ],
        "audit_bank": {
            "artifact": {"path": str(bank["path"]), "sha256": bank["file_sha256"]},
            "manifest": {
                "path": str(bank["manifest_path"]),
                "sha256": bank["manifest_sha256"],
            },
            "split": "audit",
            "query_count": len(bank["query_ids"]),
        },
        "construction": (
            "evaluate_response_fidelity_from_cpu_descriptors_"
            "recomputed_after_selection_validation"
        ),
    }


def _surface_retention(runtime: Mapping[str, Any]) -> dict[str, Any]:
    per_seed = []
    for row in runtime["per_seed"]:
        control = row["control"]["surface"]
        candidate = row["interpolated"]["surface"]
        delta = {metric: candidate[metric] - control[metric] for metric in SURFACE_METRICS}
        checks = {
            metric: delta[metric] >= -SURFACE_NONINFERIORITY_TOLERANCE
            for metric in SURFACE_METRICS
        }
        per_seed.append(
            {
                "seed": row["seed"],
                "alpha0": control,
                "interpolated": candidate,
                "interpolated_minus_alpha0": delta,
                "checks": checks,
                "passes": all(checks.values()),
            }
        )
    return {
        "definition": {
            "metrics": list(SURFACE_METRICS),
            "scope": "every_seed",
            "minimum_interpolated_minus_alpha0": -SURFACE_NONINFERIORITY_TOLERANCE,
            "source": "cpu_recomputed_from_bound_validation_caches",
        },
        "per_seed": per_seed,
        "passes": all(row["passes"] for row in per_seed),
    }


def confirm(args: argparse.Namespace) -> dict[str, Any]:
    _cpu_only_preflight()
    initial_implementation_closure = _implementation_closure_records()
    selection_path = _declared_regular_file(args.selection, label="frozen selection")
    diagnostic_path = _declared_regular_file(args.diagnostic, label="interpolation diagnostic")
    output = _fresh_output(args.output, label="confirmation output")
    receipt = _fresh_output(args.opening_receipt, label="audit opening receipt")
    _require(output != receipt, "confirmation output and opening receipt must differ")
    selection_sha = str(args.selection_sha256)
    diagnostic_sha = str(args.diagnostic_sha256)
    selection, diagnostic = validate_frozen_selection(
        selection_path,
        selection_sha,
        diagnostic_path,
        diagnostic_sha,
    )
    runtime = _load_query_free_runtime(
        selection,
        diagnostic,
        batch_size=int(args.batch_size),
    )
    opening_implementation_closure = _implementation_closure_records()
    _require(
        opening_implementation_closure == initial_implementation_closure,
        "implementation closure changed during query-free recomputation",
    )

    # This durable no-clobber receipt is the point of no return.  No audit
    # artifact path is resolved, hashed, or loaded before it exists.
    opening = _opening_receipt_payload(
        selection_path=selection_path,
        selection_sha256=selection_sha,
        diagnostic_path=diagnostic_path,
        diagnostic_sha256=diagnostic_sha,
        output=output,
        implementation_closure=opening_implementation_closure,
    )
    write_bytes_noclobber(receipt, canonical_json_bytes(opening) + b"\n")
    bank = _load_audit_bank_after_receipt(receipt)

    control_reports = []
    candidate_reports = []
    per_seed = []
    alpha = float(selection["selected_alpha"])
    for row in runtime["per_seed"]:
        seed = int(row["seed"])
        control_report = _response_report(
            method_id="surface_joint_attention_alpha0",
            seed=seed,
            student=row["control"]["student"],
            teacher=row["control"]["teacher"],
            runtime=runtime,
            bank=bank,
        )
        candidate_report = _response_report(
            method_id=f"surface_weight_interpolation_alpha_{alpha:g}",
            seed=seed,
            student=row["interpolated"]["student"],
            teacher=row["interpolated"]["teacher"],
            runtime=runtime,
            bank=bank,
        )
        control_reports.append(control_report)
        candidate_reports.append(candidate_report)
        per_seed.append(
            {
                "seed": seed,
                "selected_alpha": alpha,
                "control_checkpoint": row["control_checkpoint"],
                "candidate_checkpoint": row["candidate_checkpoint"],
                "descriptor_sha256": {
                    "alpha0_student": tensor_sha256(row["control"]["student"]),
                    "interpolated_student": tensor_sha256(row["interpolated"]["student"]),
                    "teacher": tensor_sha256(row["control"]["teacher"]),
                    "rows": runtime["rows_sha256"],
                },
                "control_embedded_metrics": control_report,
                "interpolated_embedded_metrics": candidate_report,
            }
        )
    source_authority = _same_process_source_authority(
        selection_path=selection_path,
        selection_sha256=selection_sha,
        diagnostic_path=diagnostic_path,
        diagnostic_sha256=diagnostic_sha,
        runtime=runtime,
        bank=bank,
    )
    gate = _paired_gate_in_memory(
        control_reports,
        candidate_reports,
        source_authority=source_authority,
    )
    retention = _surface_retention(runtime)
    confirmed = gate["decision"] == "promote" and retention["passes"]
    final_implementation_closure = _implementation_closure_records()
    _require(
        final_implementation_closure == opening_implementation_closure,
        "implementation closure changed after audit opening",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "complete_one_shot_audit90_confirmation",
        "decision": "promote_confirmed" if confirmed else "audit_reject_no_retuning",
        "main_result_eligible": confirmed,
        "device": "cpu",
        "selected_alpha": alpha,
        "selection_contract": {
            "selection_frozen_before_audit": True,
            "selection_recomputed_before_audit": True,
            "query_free_metrics_recomputed_before_audit": True,
            "audit_role": "confirmation_only_no_retuning",
            "audit_opening_count": 1,
            "audit_query_count": 90,
            "post_audit_alpha_change_forbidden": True,
        },
        "audit": {
            "opened": True,
            "opening_count": 1,
            "status": "consumed_once_for_confirmation",
            "bank": {
                "path": str(bank["path"]),
                "sha256": bank["file_sha256"],
                "manifest_path": str(bank["manifest_path"]),
                "manifest_sha256": bank["manifest_sha256"],
                "query_count": len(bank["query_ids"]),
                "embedding_tensor_sha256": bank["embedding_tensor_sha256"],
            },
            "opening_receipt": _file_record(receipt),
        },
        "surface_retention": retention,
        "paired_text_response_gate": gate,
        "per_seed": per_seed,
        "provenance": {
            "selection": {"path": str(selection_path), "sha256": selection_sha},
            "diagnostic": {"path": str(diagnostic_path), "sha256": diagnostic_sha},
            "query_free_input_bindings": runtime["input_bindings"],
            "implementation": {
                "path": final_implementation_closure[0]["path"],
                "sha256": final_implementation_closure[0]["sha256"],
            },
            "evaluation_implementation": _file_record(
                Path(fidelity.__file__).resolve()
            ),
            "implementation_closure": final_implementation_closure,
            "benchmark_vocabulary_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "audit_reports_aggregated_in_memory_from_same_process_tensors": True,
        },
    }
    write_frozen_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--diagnostic-sha256", required=True)
    parser.add_argument("--opening-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = confirm(args)
    print(
        {
            "output": str(Path(args.output).resolve()),
            "status": payload["status"],
            "decision": payload["decision"],
            "selected_alpha": payload["selected_alpha"],
        }
    )


if __name__ == "__main__":
    main()
