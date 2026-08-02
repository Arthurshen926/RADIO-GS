#!/usr/bin/env python3
"""Train the fit-only seed-0 v2 adapter with matched mean/CVaR risk.

This additive pilot keeps every v1 data, architecture, angular, protocol, and
selection boundary.  Its only method change is the differentiable primary
risk optimized on each complete scene:

``0.5 * mean(scene-query unit loss) + 0.5 * fractional upper-CVaR10``.

The frozen target-blind fit bank is the only text vocabulary opened.  No
external benchmark task is selected or evaluated by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_independent_normalized_cosine_response_smooth_l1_loss,
)
from radio_gs.losses.uncertainty_response_risk import (
    compute_equal_scene_mean_fractional_cvar_risk,
    compute_uncertainty_weighted_pairwise_mean_cvar_risk,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.models.surface_text_response_adapter import (
    LowRankTangentSummaryAdapter,
)
from radio_gs.scripts import (
    train_surface_region_uncertainty_response_adapter_pilot as v1,
)
from radio_gs.scripts.bind_evaluation_protocol_freeze import (
    UNOPENED_SCOPE,
    build_binding,
)
from radio_gs.scripts.train_surface_region_summary_readout import (
    _load,
    _paths,
    _seed_training,
    _targets,
)
from radio_gs.scripts.train_surface_region_text_response_distill import (
    _cache_binding,
    _fit_bank_binding,
    _validate_train_validation_contracts,
    _verify_radio_checkpoint,
    complete_scene_batches,
    load_fit_text_embedding_bank,
    load_surface_control_checkpoint,
    state_dict_sha256,
)
from radio_gs.utils.immutable_artifacts import (
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_region_uncertainty_response_adapter_cvar_seed0_pilot"
ALGORITHM_VERSION = (
    "frozen_surface_low_rank_tangent_adapter_multiview_"
    "equal_scene_mean05_fractional_cvar10_05_v2"
)
RISK_MEAN_WEIGHT = 0.5
RISK_CVAR_WEIGHT = 0.5
RISK_CVAR_TAIL_FRACTION = 0.10


def _objective_contract_v2() -> dict[str, Any]:
    return {
        "total": (
            "equal_scene_uncertainty_pairwise_mean05_cvar10_05+"
            "0.25*independent_response+0.25*all_view_surface_descriptor"
        ),
        "primary_pairwise_risk": {
            "scene_query_unit": (
                "uncertainty_weighted_normalized_pairwise_gap_smooth_l1"
            ),
            "within_scene_mean_weight": RISK_MEAN_WEIGHT,
            "within_scene_fractional_upper_cvar_weight": RISK_CVAR_WEIGHT,
            "within_scene_fractional_upper_cvar_tail": (
                RISK_CVAR_TAIL_FRACTION
            ),
            "across_scene_reduction": "equal_scene_mean",
            "teacher_variance_weights_text_bank_autograd": "detached",
            "student_scene_query_unit_autograd": "retained",
        },
        "primary_pairwise_weight": v1.PAIRWISE_WEIGHT,
        "independent_response_weight": v1.INDEPENDENT_RESPONSE_WEIGHT,
        "surface_descriptor_weight": v1.SURFACE_DESCRIPTOR_WEIGHT,
        "standard_error_multiplier": v1.STANDARD_ERROR_MULTIPLIER,
        "tie_tolerance": v1.TIE_TOLERANCE,
        "vocabulary": "target_blind_fit_only",
    }


def training_contract_v2() -> dict[str, Any]:
    contract = v1._training_contract()
    contract["objective"] = _objective_contract_v2()
    contract["optimizer"] = "persistent_adamw_adapter_parameters_only"
    contract["v1_boundaries_preserved"] = {
        "seed": v1.PILOT_SEED,
        "adapter_rank": v1.ADAPTER_RANK,
        "adapter_max_angle_degrees": v1.ADAPTER_MAX_ANGLE_DEGREES,
        "surface_selector": v1._selector_contract(),
        "evaluation_protocol_freeze_id": v1.EXPECTED_FREEZE_ID,
        "evaluation_protocol_freeze_sha256": v1.EXPECTED_FREEZE_SHA256,
        "scope": UNOPENED_SCOPE,
    }
    return contract


@torch.no_grad()
def _evaluate_v2(
    adapter: LowRankTangentSummaryAdapter,
    head: torch.nn.Module,
    base_tokens: torch.Tensor,
    data: Mapping[str, Any],
    text_bank: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    metrics, units, validity = v1._evaluate(
        adapter, head, base_tokens, data, text_bank
    )
    risk, risk_stats = compute_equal_scene_mean_fractional_cvar_risk(
        units,
        validity,
        mean_weight=RISK_MEAN_WEIGHT,
        cvar_weight=RISK_CVAR_WEIGHT,
        cvar_tail_fraction=RISK_CVAR_TAIL_FRACTION,
    )
    metrics = {
        **metrics,
        "primary_equal_scene_mean_cvar_risk": float(risk.cpu()),
        "primary_scene_mean": [
            float(value) for value in risk_stats["scene_mean"].cpu()
        ],
        "primary_scene_upper_fractional_cvar10": [
            float(value)
            for value in risk_stats["scene_upper_fractional_cvar"].cpu()
        ],
        "primary_scene_risk": [
            float(value) for value in risk_stats["scene_risk"].cpu()
        ],
    }
    return metrics, units, validity


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if (
        output.exists()
        or output.is_symlink()
        or report_path.exists()
        or report_path.is_symlink()
    ):
        raise FileExistsError("v2 pilot checkpoint/report output must be new")
    repo_root = Path(__file__).resolve().parents[2]
    protocol_binding = build_binding(
        Path(args.evaluation_protocol_freeze),
        scope=UNOPENED_SCOPE,
        repo_root=repo_root,
    )
    if (
        protocol_binding["scope"] != UNOPENED_SCOPE
        or protocol_binding["task"] is not None
        or protocol_binding["freeze"]["freeze_id"] != v1.EXPECTED_FREEZE_ID
        or protocol_binding["freeze"]["sha256"] != v1.EXPECTED_FREEZE_SHA256
    ):
        raise ValueError("v2 pilot evaluation-protocol freeze binding differs")

    train_paths = _paths(args.train_caches)
    validation_paths = _paths(args.validation_caches)
    train_data, train_meta = _load(train_paths, "train")
    validation_data, validation_meta = _load(validation_paths, "validation")
    _validate_train_validation_contracts(train_meta, validation_meta)
    radio_path = Path(args.radio_checkpoint).resolve()
    radio_sha = _verify_radio_checkpoint(radio_path, train_meta)
    fit_bank = load_fit_text_embedding_bank(
        Path(args.fit_text_bank), Path(args.fit_text_bank_manifest)
    )
    base_model, surface_control = load_surface_control_checkpoint(
        Path(args.surface_control_checkpoint),
        expected_sha256=str(args.surface_control_checkpoint_sha256),
        seed=v1.PILOT_SEED,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        hidden_dim=256,
        reliability_attention_mode="log_prior",
        context_pooling_mode="joint_attention_v1",
    )
    if set(train_meta["scenes"]) & set(validation_meta["scenes"]):
        raise ValueError("v2 pilot train/validation scenes overlap")
    if "scene_ids" not in train_data or "scene_ids" not in validation_data:
        raise ValueError("v2 pilot caches require exact row-to-scene bindings")

    device = torch.device(args.device)
    generator = _seed_training(v1.PILOT_SEED, device=device)
    base_model = base_model.to(device).eval().requires_grad_(False)
    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).to(device).eval()
    head.requires_grad_(False)
    adapter = LowRankTangentSummaryAdapter(
        feature_dim=1280,
        rank=v1.ADAPTER_RANK,
        max_angle_degrees=v1.ADAPTER_MAX_ANGLE_DEGREES,
    ).to(device)
    text_bank = fit_bank["embeddings"].to(device)
    base_train = v1._precompute_base_tokens(base_model, train_data, device)
    base_validation = v1._precompute_base_tokens(
        base_model, validation_data, device
    )
    base_state = v1._clone_state(base_model)
    base_state_sha = state_dict_sha256(base_state)
    if any(parameter.requires_grad for parameter in base_model.parameters()) or any(
        parameter.requires_grad for parameter in head.parameters()
    ):
        raise RuntimeError("v2 pilot base readout/head freeze failed")

    train_teacher, train_views, train_teacher_mask = v1._all_teacher_targets(
        train_data
    )
    validation_teacher, validation_views, validation_teacher_mask = (
        v1._all_teacher_targets(validation_data)
    )
    uncertainty_statistics = {
        "train": v1.uncertainty_weight_statistics(
            train_teacher,
            train_views,
            train_teacher_mask,
            fit_bank["embeddings"],
            train_data["scene_ids"],
        ),
        "validation": v1.uncertainty_weight_statistics(
            validation_teacher,
            validation_views,
            validation_teacher_mask,
            fit_bank["embeddings"],
            validation_data["scene_ids"],
        ),
    }
    validation_scene_names = v1._scene_order(validation_data["scene_ids"])
    if len(validation_scene_names) != len(validation_meta["scenes"]):
        raise ValueError("v2 validation scene order/count differs")

    adapter.eval()
    control_metrics, control_units, control_valid = _evaluate_v2(
        adapter, head, base_validation, validation_data, text_bank
    )
    control_selector, _ = v1.continuous_selector_metrics(
        control_units,
        control_valid,
        control_units,
        control_valid,
        validation_scene_names,
    )
    control_record = v1.annotate_selection_record(
        {
            "epoch": 0,
            "initialization": "zero_up_identity_adapter",
            **control_metrics,
        },
        control_record={**control_metrics},
        selector=control_selector,
    )
    control_record["selection_updated_best"] = True
    initial_adapter_state = v1._clone_state(adapter)
    initial_adapter_sha = state_dict_sha256(initial_adapter_state)
    architecture = adapter.architecture()
    control_record.update(
        {
            "base_surface_state_dict_sha256": base_state_sha,
            "response_adapter_state_dict_sha256": initial_adapter_sha,
            "combined_state_sha256": v1.combined_state_sha256(
                base_state_sha, initial_adapter_sha, architecture["digest"]
            ),
            "scene_query_unit_loss_sha256": tensor_sha256(control_units.float()),
            "scene_query_unit_valid_sha256": tensor_sha256(control_valid),
        }
    )
    history: list[dict[str, Any]] = [control_record]
    selector_units: list[dict[str, torch.Tensor | int]] = [
        {"epoch": 0, "loss": control_units.float(), "valid": control_valid}
    ]
    best_epoch = 0
    best_state = initial_adapter_state
    stale = 0
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=v1.LEARNING_RATE, weight_decay=v1.WEIGHT_DECAY
    )

    for epoch in range(1, v1.EPOCHS + 1):
        adapter.train()
        term_values = {
            "total": [],
            "primary_equal_scene_mean_cvar_risk": [],
            "primary_scene_mean": [],
            "primary_scene_upper_fractional_cvar10": [],
            "independent_response": [],
            "surface_descriptor": [],
        }
        batches = complete_scene_batches(
            train_data["scene_ids"],
            row_count=len(train_data["radio_features"]),
            target_batch_rows=v1.TARGET_BATCH_ROWS,
            generator=generator,
        )
        for rows in batches:
            _target_token, teacher, all_descriptors, teacher_mask = _targets(
                train_data, rows
            )
            adapted = adapter(base_train[rows.to(base_train.device)])
            student = F.normalize(
                head(adapted[:, None])[:, 0].float(), dim=-1, eps=1e-8
            )
            teacher = teacher.to(device)
            all_descriptors = all_descriptors.to(device)
            teacher_mask = teacher_mask.to(device)
            primary_risk, primary_stats = (
                compute_uncertainty_weighted_pairwise_mean_cvar_risk(
                    student,
                    teacher,
                    all_descriptors,
                    teacher_mask,
                    text_bank,
                    [train_data["scene_ids"][row] for row in rows.tolist()],
                    standard_error_multiplier=v1.STANDARD_ERROR_MULTIPLIER,
                    tie_tolerance=v1.TIE_TOLERANCE,
                    eps=v1.EPS,
                    mean_weight=RISK_MEAN_WEIGHT,
                    cvar_weight=RISK_CVAR_WEIGHT,
                    cvar_tail_fraction=RISK_CVAR_TAIL_FRACTION,
                )
            )
            independent_loss = (
                compute_independent_normalized_cosine_response_smooth_l1_loss(
                    student, teacher, text_bank
                )
            )
            all_view_cosine = torch.einsum(
                "bd,bvd->bv", student, all_descriptors
            )
            surface_descriptor_loss = (1.0 - all_view_cosine)[
                teacher_mask
            ].mean()
            total = (
                v1.PAIRWISE_WEIGHT * primary_risk
                + v1.INDEPENDENT_RESPONSE_WEIGHT * independent_loss
                + v1.SURFACE_DESCRIPTOR_WEIGHT * surface_descriptor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            measured = {
                "total": total,
                "primary_equal_scene_mean_cvar_risk": primary_risk,
                "primary_scene_mean": primary_stats["scene_mean"].mean(),
                "primary_scene_upper_fractional_cvar10": primary_stats[
                    "scene_upper_fractional_cvar"
                ].mean(),
                "independent_response": independent_loss,
                "surface_descriptor": surface_descriptor_loss,
            }
            for name, value in measured.items():
                term_values[name].append(float(value.detach().cpu()))

        adapter.eval()
        metrics, units, valid = _evaluate_v2(
            adapter, head, base_validation, validation_data, text_bank
        )
        selector, _delta = v1.continuous_selector_metrics(
            units,
            valid,
            control_units,
            control_valid,
            validation_scene_names,
        )
        state = v1._clone_state(adapter)
        adapter_sha = state_dict_sha256(state)
        record = v1.annotate_selection_record(
            {
                "epoch": epoch,
                "training_losses": {
                    name: sum(values) / len(values)
                    for name, values in term_values.items()
                },
                **metrics,
            },
            control_record=history[0],
            selector=selector,
        )
        record.update(
            {
                "base_surface_state_dict_sha256": base_state_sha,
                "response_adapter_state_dict_sha256": adapter_sha,
                "combined_state_sha256": v1.combined_state_sha256(
                    base_state_sha, adapter_sha, architecture["digest"]
                ),
                "scene_query_unit_loss_sha256": tensor_sha256(units.float()),
                "scene_query_unit_valid_sha256": tensor_sha256(valid),
            }
        )
        selected_epoch = v1.select_best_epoch([*history, record])
        best_updated = selected_epoch == epoch
        record["selection_updated_best"] = best_updated
        if best_updated:
            best_epoch = epoch
            best_state = state
            stale = 0
        else:
            if selected_epoch != best_epoch:
                raise RuntimeError("v2 pilot best epoch changed retroactively")
            stale += 1
        record["patience_stale"] = stale
        record["patience_stop"] = stale >= v1.PATIENCE
        history.append(record)
        selector_units.append(
            {"epoch": epoch, "loss": units.float(), "valid": valid}
        )
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= v1.PATIENCE:
            break

    if v1.select_best_epoch(history) != best_epoch:
        raise RuntimeError("v2 pilot online/final selection differs")
    adapter.load_state_dict(best_state, strict=True)
    final_metrics, final_units, final_valid = _evaluate_v2(
        adapter, head, base_validation, validation_data, text_bank
    )
    if (
        tensor_sha256(final_units.float())
        != history[best_epoch]["scene_query_unit_loss_sha256"]
        or tensor_sha256(final_valid)
        != history[best_epoch]["scene_query_unit_valid_sha256"]
    ):
        raise RuntimeError("selected v2 adapter replay differs")
    best_adapter_sha = state_dict_sha256(best_state)
    best_combined_sha = v1.combined_state_sha256(
        base_state_sha, best_adapter_sha, architecture["digest"]
    )
    selected_selector = history[best_epoch]["continuous_selector"]
    pilot_advance = (
        best_epoch > 0
        and history[best_epoch]["surface_control_feasible"] is True
        and float(selected_selector["normalized_mean_delta"])
        <= -v1.PILOT_REQUIRED_MEAN_IMPROVEMENT
        and float(selected_selector["normalized_upper_cvar10_delta"])
        <= v1.PILOT_GLOBAL_CVAR_TOLERANCE
        and float(selected_selector["worst_scene_upper_cvar10_delta"])
        <= v1.PILOT_PER_SCENE_CVAR_TOLERANCE
        and float(history[best_epoch]["adapter_angle"]["max_degrees"])
        <= v1.ADAPTER_MAX_ANGLE_DEGREES
        + v1.ANGLE_AUDIT_ABSOLUTE_TOLERANCE_DEGREES
    )
    implementation_paths = (
        Path(__file__),
        repo_root / "radio_gs/losses/uncertainty_response_risk.py",
        repo_root
        / "radio_gs/scripts/train_surface_region_uncertainty_response_adapter_pilot.py",
        repo_root / "radio_gs/models/surface_text_response_adapter.py",
        repo_root / "radio_gs/losses/direct_point_query_logit_distill_loss.py",
        repo_root / "radio_gs/scripts/train_surface_region_summary_readout.py",
        repo_root / "radio_gs/scripts/train_surface_region_text_response_distill.py",
        repo_root / "radio_gs/scripts/bind_evaluation_protocol_freeze.py",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "base_surface_state_dict": base_state,
        "base_surface_state_dict_sha256": base_state_sha,
        "response_adapter_architecture": architecture,
        "response_adapter_state_dict": best_state,
        "response_adapter_state_dict_sha256": best_adapter_sha,
        "combined_state_sha256": best_combined_sha,
        "best_epoch": best_epoch,
        "history": history,
        "selector_unit_losses": selector_units,
        "pilot_advance_gate_passed": pilot_advance,
        "provenance": {
            "evaluation_protocol": protocol_binding,
            "scope": UNOPENED_SCOPE,
            "external_benchmarks_opened": False,
            "formal_authority": False,
            "pilot_only": True,
            "benchmark_queries_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_images_opened": False,
            "fit_text_bank_opened": True,
            "benchmark_vocabulary_opened": False,
            "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False,
            "custom_text_projection": False,
            "official_siglip_summary_head_frozen": True,
            "fit_split_only": True,
            "surface_control": surface_control,
            "train_caches": _cache_binding(train_paths),
            "validation_caches": _cache_binding(validation_paths),
            "fit_text_bank": _fit_bank_binding(fit_bank),
            "radio_checkpoint": {"path": str(radio_path), "sha256": radio_sha},
            "train_contract": train_meta,
            "validation_contract": validation_meta,
            "implementation_sources": [
                v1._file_record(path) for path in implementation_paths
            ],
        },
        "training_contract": training_contract_v2(),
        "uncertainty_weight_statistics": uncertainty_statistics,
        "final_validation": final_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    checkpoint_sha = sha256_file(output)
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": f"{ARTIFACT_TYPE}_report",
        "algorithm_version": ALGORITHM_VERSION,
        "output": str(output),
        "checkpoint_sha256": checkpoint_sha,
        "evaluation_protocol": protocol_binding,
        "scope": UNOPENED_SCOPE,
        "external_benchmarks_opened": False,
        "formal_authority": False,
        "pilot_only": True,
        "base_surface_state_dict_sha256": base_state_sha,
        "response_adapter_architecture": architecture,
        "response_adapter_state_dict_sha256": best_adapter_sha,
        "combined_state_sha256": best_combined_sha,
        "best_epoch": best_epoch,
        "selected_history_record": history[best_epoch],
        "pilot_advance_gate_passed": pilot_advance,
        "pilot_advance_gate": {
            "required_mean_improvement": v1.PILOT_REQUIRED_MEAN_IMPROVEMENT,
            "global_cvar_tolerance": v1.PILOT_GLOBAL_CVAR_TOLERANCE,
            "per_scene_cvar_tolerance": v1.PILOT_PER_SCENE_CVAR_TOLERANCE,
            "adapter_max_angle_degrees": v1.ADAPTER_MAX_ANGLE_DEGREES,
            "adapter_angle_audit_absolute_tolerance_degrees": (
                v1.ANGLE_AUDIT_ABSOLUTE_TOLERANCE_DEGREES
            ),
        },
        "uncertainty_weight_statistics": uncertainty_statistics,
        "history_length": len(history),
        "training_contract": training_contract_v2(),
    }
    write_frozen_json(report_path, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--fit-text-bank", type=Path, required=True)
    parser.add_argument("--fit-text-bank-manifest", type=Path, required=True)
    parser.add_argument("--surface-control-checkpoint", type=Path, required=True)
    parser.add_argument("--surface-control-checkpoint-sha256", required=True)
    parser.add_argument("--radio-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--evaluation-protocol-freeze",
        type=Path,
        default=Path("paper/artifacts/evaluation_protocol_freeze_20260801.yaml"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = train(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

