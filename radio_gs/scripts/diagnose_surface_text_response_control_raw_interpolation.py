#!/usr/bin/env python3
"""Non-authority single-seed control-to-proposal interpolation diagnostic.

This standalone screen intentionally does not participate in the formal
distillation authority.  It reuses the frozen trainer's exact loaders, one
epoch proposal objective, target-blind validation evaluator, and robust epoch
selector, but writes only a JSON diagnostic.  No interpolated checkpoint is
published for inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.scripts import train_surface_region_text_response_distill as trainer


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_text_response_control_raw_interpolation_diagnostic"
ALGORITHM_VERSION = "single_seed_single_epoch_control_raw_power2_micro_ray_v2"
REQUIRED_SEEDS = tuple(trainer.SHARED_TRAINING_SEEDS)
ALPHA_GRID = (
    0.0,
    1.0 / 4096.0,
    1.0 / 2048.0,
    1.0 / 1024.0,
    1.0 / 512.0,
    0.0025,
)
IMPLEMENTATION_SOURCES = (
    "radio_gs/scripts/diagnose_surface_text_response_control_raw_interpolation.py",
    "radio_gs/scripts/train_surface_region_text_response_distill.py",
    "radio_gs/scripts/train_surface_region_summary_readout.py",
    "radio_gs/losses/direct_point_query_logit_distill_loss.py",
    "radio_gs/interfaces/surface_region_summary.py",
    "radio_gs/models/siglip_projection.py",
    "radio_gs/utils/immutable_artifacts.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"diagnostic input is not a file: {source}")
    return {"path": str(source), "sha256": _sha256_file(source)}


def _implementation_source_bindings() -> dict[str, dict[str, str]]:
    root = Path(__file__).resolve().parents[2]
    return {
        relative: _file_record(root / relative)
        for relative in IMPLEMENTATION_SOURCES
    }


def validated_seed(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value not in REQUIRED_SEEDS
    ):
        raise ValueError("diagnostic seed must be exactly one of 0/1/2")
    return int(value)


def state_dict_binding(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Return a deterministic semantic digest for a complete tensor state."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("state_dict binding requires a non-empty mapping")
    tensors: list[dict[str, Any]] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name or not torch.is_tensor(value):
            raise ValueError("state_dict binding fields differ")
        tensor = value.detach().cpu().contiguous()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"state_dict tensor is non-finite: {name}")
        tensors.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor_sha256": trainer.tensor_sha256(tensor),
            }
        )
    return {"tensor_count": len(tensors), "sha256": _canonical_json_sha256(tensors)}


def interpolate_state_dict(
    control: Mapping[str, torch.Tensor],
    raw: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Interpolate every floating tensor and fail on mutable discrete state."""

    value = float(alpha)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("interpolation alpha must be finite within [0,1]")
    if not isinstance(control, Mapping) or set(control) != set(raw) or not control:
        raise ValueError("control/raw state_dict fields differ")
    result: dict[str, torch.Tensor] = {}
    for name in sorted(control):
        control_tensor = control[name]
        raw_tensor = raw[name]
        if (
            not torch.is_tensor(control_tensor)
            or not torch.is_tensor(raw_tensor)
            or control_tensor.shape != raw_tensor.shape
            or control_tensor.dtype != raw_tensor.dtype
        ):
            raise ValueError(f"control/raw state tensor differs: {name}")
        left = control_tensor.detach().cpu().contiguous()
        right = raw_tensor.detach().cpu().contiguous()
        if left.is_floating_point() or left.is_complex():
            if not bool(torch.isfinite(left).all()) or not bool(
                torch.isfinite(right).all()
            ):
                raise ValueError(f"control/raw state tensor is non-finite: {name}")
            result[name] = torch.lerp(left, right, value)
        else:
            if not torch.equal(left, right):
                raise ValueError(
                    f"control/raw non-floating state tensor changed: {name}"
                )
            result[name] = left.clone()
    return result


def parameter_displacement_binding(
    control: Mapping[str, torch.Tensor],
    raw: Mapping[str, torch.Tensor],
    parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    """Measure the complete control-to-raw parameter displacement."""

    if not parameter_names or len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter displacement inventory differs")
    if any(name not in control or name not in raw for name in parameter_names):
        raise ValueError("parameter displacement state fields differ")
    accumulators: dict[str, dict[str, float | int]] = {}
    for name in parameter_names:
        left = control[name].detach().cpu()
        right = raw[name].detach().cpu()
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not left.is_floating_point()
        ):
            raise ValueError(f"parameter displacement tensor differs: {name}")
        left64 = left.to(torch.float64)
        delta64 = right.to(torch.float64) - left64
        group = name.split(".", 1)[0]
        values = accumulators.setdefault(
            group,
            {
                "parameter_tensor_count": 0,
                "parameter_count": 0,
                "control_squared_l2": 0.0,
                "delta_squared_l2": 0.0,
                "max_abs": 0.0,
            },
        )
        values["parameter_tensor_count"] += 1
        values["parameter_count"] += left.numel()
        values["control_squared_l2"] += float(left64.square().sum())
        values["delta_squared_l2"] += float(delta64.square().sum())
        values["max_abs"] = max(
            float(values["max_abs"]),
            float(delta64.abs().max()) if delta64.numel() else 0.0,
        )

    def finalize(values: Mapping[str, float | int]) -> dict[str, float | int]:
        control_l2 = math.sqrt(float(values["control_squared_l2"]))
        delta_l2 = math.sqrt(float(values["delta_squared_l2"]))
        return {
            "parameter_tensor_count": int(values["parameter_tensor_count"]),
            "parameter_count": int(values["parameter_count"]),
            "control_l2": control_l2,
            "delta_l2": delta_l2,
            "relative_delta_l2": delta_l2 / max(control_l2, 1e-12),
            "max_abs": float(values["max_abs"]),
        }

    total = {
        key: sum(float(values[key]) for values in accumulators.values())
        for key in (
            "parameter_tensor_count",
            "parameter_count",
            "control_squared_l2",
            "delta_squared_l2",
        )
    }
    total["max_abs"] = max(
        float(values["max_abs"]) for values in accumulators.values()
    )
    return {
        "definition": "l2_over_all_trainable_parameters_float64_accumulation",
        "relative_l2_denominator_floor": 1e-12,
        "overall": finalize(total),
        "top_level_modules": {
            group: finalize(values) for group, values in sorted(accumulators.items())
        },
    }


def train_single_raw_proposal(
    model: torch.nn.Module,
    head: torch.nn.Module,
    train_data: Mapping[str, Any],
    *,
    device: torch.device,
    text_bank: torch.Tensor,
    response_lambdas: Mapping[str, float],
    generator: torch.Generator,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    token_weight: float,
    relation_weight: float,
    canonical_noise_degrees: float,
) -> dict[str, Any]:
    """Generate exactly one full AdamW epoch from the frozen control state."""

    batches = trainer.complete_scene_batches(
        train_data.get("scene_ids"),
        row_count=len(train_data["radio_features"]),
        target_batch_rows=int(batch_size),
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    names = (
        "total",
        "token",
        "descriptor",
        "relation",
        "independent_response",
        "scene_response",
        "scene_profile",
        "scene_ranking",
    )
    epoch_terms: dict[str, list[float]] = {name: [] for name in names}
    model.train()
    for rows in batches:
        target_token, target_descriptor, all_descriptors, teacher_mask = (
            trainer._targets(train_data, rows)
        )
        token_mask = train_data["token_mask"][rows].to(device)
        radio_features = trainer.inject_tangent_direction_noise(
            train_data["radio_features"][rows].to(device),
            token_mask,
            angle_degrees=float(canonical_noise_degrees),
        )
        predicted = model(
            radio_features,
            train_data["geometry"][rows].to(device),
            anchor_index=train_data["anchor_index"][rows].to(device),
            token_mask=token_mask,
            reliability=train_data["reliability"][rows].to(device),
        )
        projected = F.normalize(
            head(predicted[:, None])[:, 0].float(), dim=-1, eps=1e-8
        )
        terms = trainer.compute_training_losses(
            predicted,
            projected,
            target_token.to(device),
            target_descriptor.to(device),
            all_descriptors.to(device),
            teacher_mask.to(device),
            text_bank,
            [train_data["scene_ids"][row] for row in rows.tolist()],
            token_weight=float(token_weight),
            relation_weight=float(relation_weight),
            independent_response_lambda=float(
                response_lambdas["independent_response"]
            ),
            scene_response_lambda=float(response_lambdas["scene_response"]),
        )
        if set(terms) != set(names):
            raise ValueError("proposal training loss topology differs")
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        optimizer.step()
        for name in names:
            scalar = float(terms[name].detach().cpu())
            if not math.isfinite(scalar) or scalar < 0.0:
                raise ValueError(f"proposal training loss is invalid: {name}")
            epoch_terms[name].append(scalar)
    return {
        "optimizer": "AdamW",
        "epoch_count": 1,
        "complete_scene_batch_count": len(batches),
        "max_complete_scene_batch_rows": max(len(rows) for rows in batches),
        "mean_losses": {
            name: sum(values) / len(values) for name, values in epoch_terms.items()
        },
    }


def evaluate_interpolation_grid(
    model: torch.nn.Module,
    head: torch.nn.Module,
    validation_data: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    text_bank: torch.Tensor,
    control_state: Mapping[str, torch.Tensor],
    raw_state: Mapping[str, torch.Tensor],
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
) -> tuple[list[dict[str, Any]], int, float]:
    """Really evaluate every declared alpha, then reuse the robust selector."""

    if tuple(float(value) for value in alpha_grid) != ALPHA_GRID:
        raise ValueError("diagnostic alpha grid differs from the frozen contract")
    raw_parameter_displacement = parameter_displacement_binding(
        control_state,
        raw_state,
        tuple(name for name, _ in model.named_parameters()),
    )
    raw_radius = raw_parameter_displacement["overall"]
    history: list[dict[str, Any]] = []
    model.eval()
    for candidate_index, alpha in enumerate(ALPHA_GRID):
        candidate_state = interpolate_state_dict(control_state, raw_state, alpha)
        model.load_state_dict(candidate_state, strict=True)
        surface, response = trainer._evaluate_response_aware(
            model,
            head,
            validation_data,
            device,
            int(batch_size),
            text_bank,
        )
        surface_score = 0.5 * (
            float(surface["mean_descriptor_cosine"])
            + float(surface["all_view_descriptor_cosine"])
        )
        history.append(
            {
                "epoch": candidate_index,
                "candidate_index": candidate_index,
                "proposal_epoch": 1,
                "alpha": float(alpha),
                "state_dict": state_dict_binding(candidate_state),
                "control_to_candidate_parameter_radius": {
                    "delta_l2": float(alpha) * float(raw_radius["delta_l2"]),
                    "relative_delta_l2": float(alpha)
                    * float(raw_radius["relative_delta_l2"]),
                    "max_abs": float(alpha) * float(raw_radius["max_abs"]),
                },
                "scene_response_objective": (
                    trainer._scene_response_objective_contract()
                ),
                "surface_selection_score": surface_score,
                "selection_score": surface_score,
                **surface,
                **response,
            }
        )
    finalized, best_index, best_score = (
        trainer.finalize_response_primary_epoch_selection(history)
    )
    if best_index not in range(len(ALPHA_GRID)):
        raise RuntimeError("robust selector returned an invalid interpolation index")
    return finalized, best_index, best_score


def _clone_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Run the non-authority diagnostic without opening dev/audit vocabularies."""

    seed = validated_seed(args.seed)
    for label, value, strictly_positive in (
        ("batch size", args.batch_size, True),
        ("learning rate", args.learning_rate, True),
        ("weight decay", args.weight_decay, False),
        ("token weight", args.token_weight, False),
        ("relation weight", args.relation_weight, False),
        ("canonical noise degrees", args.canonical_noise_degrees, False),
    ):
        numeric = float(value)
        if not math.isfinite(numeric) or (numeric <= 0 if strictly_positive else numeric < 0):
            raise ValueError(f"diagnostic {label} is invalid")

    train_paths = trainer._paths(args.train_caches)
    validation_paths = trainer._paths(args.validation_caches)
    train_data, train_meta = trainer._load(train_paths, "train")
    validation_data, validation_meta = trainer._load(
        validation_paths, "validation"
    )
    trainer._validate_train_validation_contracts(train_meta, validation_meta)
    radio_path = Path(args.radio_checkpoint).resolve()
    trainer._verify_radio_checkpoint(radio_path, train_meta)
    fit_bank = trainer.load_fit_text_embedding_bank(
        Path(args.fit_text_bank), Path(args.fit_text_bank_manifest)
    )
    model, surface_control = trainer.load_surface_control_checkpoint(
        Path(args.surface_control_checkpoint),
        expected_sha256=str(args.surface_control_checkpoint_sha256),
        seed=seed,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        hidden_dim=int(args.hidden_dim),
        reliability_attention_mode=str(args.reliability_attention_mode),
        context_pooling_mode=str(args.context_pooling_mode),
    )
    calibration = trainer.load_calibration_manifest(
        Path(args.calibration_manifest),
        seed=seed,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        train_scene_ids=train_data.get("scene_ids"),
        train_row_count=len(train_data["radio_features"]),
        radio_path=radio_path,
        fit_bank=fit_bank,
        surface_control=surface_control,
        trainable_parameters=(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        token_weight=float(args.token_weight),
        relation_weight=float(args.relation_weight),
    )
    response_lambdas = dict(calibration["response_lambdas"])
    device = torch.device(args.device)
    generator = trainer._seed_training(seed, device=device)
    model = model.to(device)
    head = trainer.SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).to(
        device
    ).eval()
    head.requires_grad_(False)
    text_bank = fit_bank["embeddings"].to(device)
    control_state = _clone_cpu_state(model)
    proposal = train_single_raw_proposal(
        model,
        head,
        train_data,
        device=device,
        text_bank=text_bank,
        response_lambdas=response_lambdas,
        generator=generator,
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        token_weight=float(args.token_weight),
        relation_weight=float(args.relation_weight),
        canonical_noise_degrees=float(args.canonical_noise_degrees),
    )
    raw_state = _clone_cpu_state(model)
    parameter_displacement = parameter_displacement_binding(
        control_state,
        raw_state,
        tuple(name for name, _ in model.named_parameters()),
    )
    candidates, best_index, best_score = evaluate_interpolation_grid(
        model,
        head,
        validation_data,
        device=device,
        batch_size=int(args.batch_size),
        text_bank=text_bank,
        control_state=control_state,
        raw_state=raw_state,
    )
    selected = candidates[best_index]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "status": "complete_non_authority_single_seed_single_epoch_target_blind",
        "authority_eligible": False,
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "seed": seed,
        "algorithm_contract": {
            "proposal": "one_complete_scene_batched_adamw_epoch_from_control",
            "seed_scope": "explicit_one_of_0_1_2",
            "interpolation": "theta_control+alpha*(theta_raw-theta_control)",
            "alpha_grid": list(ALPHA_GRID),
            "evaluation": "real_target_blind_validation_at_every_alpha",
            "selection": trainer.RESPONSE_EPOCH_SELECTION,
        },
        "training_config": {
            "hidden_dim": int(args.hidden_dim),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "token_weight": float(args.token_weight),
            "relation_weight": float(args.relation_weight),
            "reliability_attention_mode": str(args.reliability_attention_mode),
            "context_pooling_mode": str(args.context_pooling_mode),
            "canonical_noise_degrees": float(args.canonical_noise_degrees),
        },
        "inputs": {
            "train_caches": trainer._cache_binding(train_paths),
            "validation_caches": trainer._cache_binding(validation_paths),
            "surface_control": dict(surface_control),
            "fit_text_bank": trainer._fit_bank_binding(fit_bank),
            "calibration_manifest": _file_record(
                Path(args.calibration_manifest)
            ),
            "radio_checkpoint": _file_record(radio_path),
        },
        "implementation_sources": _implementation_source_bindings(),
        "response_lambdas": response_lambdas,
        "scene_response_objective": trainer._scene_response_objective_contract(),
        "proposal": {
            **proposal,
            "control_state_dict": state_dict_binding(control_state),
            "raw_state_dict": state_dict_binding(raw_state),
            "control_to_raw_parameter_displacement": parameter_displacement,
        },
        "candidates": candidates,
        "selected": {
            "candidate_index": best_index,
            "alpha": float(selected["alpha"]),
            "state_dict": dict(selected["state_dict"]),
            "best_selection_score": float(best_score),
            "response_selection_feasible": selected[
                "response_selection_feasible"
            ],
            "text_response_smooth_l1": selected["text_response_smooth_l1"],
            "text_response_mae": selected["text_response_mae"],
        },
    }
    output = Path(args.output).resolve()
    trainer.write_frozen_json(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--fit-text-bank", type=Path, required=True)
    parser.add_argument("--fit-text-bank-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--surface-control-checkpoint", type=Path, required=True)
    parser.add_argument("--surface-control-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, choices=REQUIRED_SEEDS, required=True
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument("--canonical-noise-degrees", type=float, default=0.0)
    parser.add_argument(
        "--reliability-attention-mode",
        choices=("log_prior", "input_only"),
        default="log_prior",
    )
    parser.add_argument(
        "--context-pooling-mode",
        choices=("joint_attention_v1", "core_context_separate_attention_v1"),
        default="joint_attention_v1",
    )
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = run_diagnostic(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(Path(args.output).resolve()),
                "seed": payload["seed"],
                "candidate_count": len(payload["candidates"]),
                "selected": payload["selected"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
