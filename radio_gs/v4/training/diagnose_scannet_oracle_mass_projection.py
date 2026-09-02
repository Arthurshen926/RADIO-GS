"""Target-only upper bound for global token-mass projection.

This diagnostic asks whether perfect knowledge of each retained object's full
surface support can improve the frozen pointwise posterior using only a global
token bias.  Target support enters the iterative projection directly, so this
module is never a deployable method or a model-selection arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.v4.completion.scannet import load_scene_cache
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.training.train_scannet_completion_message_passing import (
    _clamp_contract,
    _frozen_unary_probabilities,
    _load_frozen_unary_model,
    _membership_metrics,
    _posterior_to_membership,
    _prepare_runtime,
)


REPORT_SCHEMA = (
    "radio_gs.surface_object_memory_v4."
    "oracle_physical_mass_projection_diagnostic.v1"
)
METRIC_KEYS = (
    "soft_3d_miou",
    "unknown_only_soft_3d_miou",
    "visible_but_unmasked_soft_3d_miou",
    "never_visible_soft_3d_miou",
    "heldout_2d_soft_miou",
    "unknown_assignment_precision",
    "unknown_retained_object_coverage",
)


def oracle_mass_project(
    unary: torch.Tensor,
    clamp_mask: torch.Tensor,
    clamp_probabilities: torch.Tensor,
    target_physical_support_mass: torch.Tensor,
    *,
    iteration_count: int = 256,
    damping: float = 0.5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Iteratively fit token biases to target pre-cap posterior support mass."""

    if iteration_count <= 0 or not 0 < damping <= 1:
        raise ValueError("oracle projection iteration configuration is invalid")
    probabilities = torch.as_tensor(unary)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("unary must have shape [N, K+1]")
    token_count = probabilities.shape[1] - 1
    target = torch.as_tensor(
        target_physical_support_mass,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    if target.shape != (token_count,) or bool((target <= 0).any()):
        raise ValueError("every oracle token requires positive physical support")
    mask = torch.as_tensor(clamp_mask, device=probabilities.device, dtype=torch.bool)
    clamp = torch.as_tensor(
        clamp_probabilities,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    epsilon = torch.finfo(probabilities.dtype).tiny
    token_log = probabilities[:, :token_count].clamp_min(epsilon).log()
    null_log = probabilities[:, token_count:].clamp_min(epsilon).log()
    bias = probabilities.new_zeros(token_count)
    maximum_relative_error = float("inf")
    result = probabilities
    for _ in range(iteration_count):
        result = torch.softmax(
            torch.cat((token_log + bias[None, :], null_log), dim=-1), dim=-1
        )
        result = torch.where(mask[:, None], clamp, result)
        mass = result[:, :token_count].sum(0).clamp_min(epsilon)
        log_error = target.log() - mass.log()
        bias = bias + float(damping) * log_error
        maximum_relative_error = float(((mass - target).abs() / target).max())
    result = torch.softmax(
        torch.cat((token_log + bias[None, :], null_log), dim=-1), dim=-1
    )
    result = torch.where(mask[:, None], clamp, result)
    final_mass = result[:, :token_count].sum(0)
    maximum_relative_error = float(((final_mass - target).abs() / target).max())
    if not torch.equal(result[mask], clamp[mask]):
        raise RuntimeError("oracle mass projection changed an exact clamp")
    return result, {
        "iteration_count": iteration_count,
        "damping": damping,
        "target_mass_mean": float(target.mean()),
        "initial_mass_mean": float(probabilities[:, :token_count].sum(0).mean()),
        "final_mass_mean": float(final_mass.mean()),
        "maximum_relative_mass_error": maximum_relative_error,
        "bias_absolute_mean": float(bias.abs().mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_report_path = Path(args.base_report).resolve(strict=True)
    base_checkpoint_path = Path(args.base_checkpoint).resolve(strict=True)
    base_report = json.loads(base_report_path.read_text())
    try:
        base_checkpoint = torch.load(
            base_checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError as error:
        raise RuntimeError("safe weights-only checkpoint loading is required") from error
    validation_scene_ids = set(map(str, args.validation_scene))
    expected_validation = set(
        map(str, base_report.get("split", {}).get("validation_scene_ids", ()))
    )
    if validation_scene_ids != expected_validation:
        raise ValueError("oracle mass diagnostic requires the frozen validation split")
    payloads = [load_scene_cache(Path(value)) for value in args.scene_cache]
    if {str(value["scene_id"]) for value in payloads} != validation_scene_ids:
        raise ValueError("scene caches must exactly equal the validation split")

    device = torch.device(args.device)
    model = _load_frozen_unary_model(base_checkpoint, device=device)
    temperature = float(base_report["training_configuration"]["temperature"])
    confidence_cap = float(
        base_report["training_configuration"]["completion_confidence_cap"]
    )
    records = []
    for payload in sorted(payloads, key=lambda value: str(value["scene_id"])):
        runtime = _prepare_runtime(payload)
        unary = _frozen_unary_probabilities(
            model,
            runtime,
            device=device,
            element_batch_size=args.unary_element_batch_size,
            temperature=temperature,
        )
        labels = torch.as_tensor(runtime["labels"], dtype=torch.long)
        eligible = runtime["partial"].eligible_elements
        token_count = int(runtime["partial"].positive.shape[1])
        target_mass = torch.stack(
            [((labels == token_id) & eligible).sum() for token_id in range(token_count)]
        ).float()
        clamp_mask, clamp_probabilities = _clamp_contract(runtime)
        projected, projection = oracle_mass_project(
            unary.to(device),
            clamp_mask.to(device),
            clamp_probabilities.to(device),
            target_mass.to(device),
            iteration_count=args.iteration_count,
            damping=args.damping,
        )
        frozen_membership, frozen_null = _posterior_to_membership(
            unary, runtime, completion_confidence_cap=confidence_cap
        )
        projected_membership, projected_null = _posterior_to_membership(
            projected.cpu(), runtime, completion_confidence_cap=confidence_cap
        )
        records.append(
            {
                "scene_id": str(payload["scene_id"]),
                "element_count": int(labels.numel()),
                "token_count": token_count,
                "projection": projection,
                "frozen_aligned_pointwise": _membership_metrics(
                    runtime, frozen_membership, frozen_null
                ),
                "oracle_physical_mass_projection": _membership_metrics(
                    runtime, projected_membership, projected_null
                ),
            }
        )

    scene_macro = {}
    difference = {}
    for method in ("frozen_aligned_pointwise", "oracle_physical_mass_projection"):
        scene_macro[method] = {
            key: float(np.mean([record[method][key] for record in records]))
            for key in METRIC_KEYS
        }
    for key in METRIC_KEYS:
        difference[key] = (
            scene_macro["oracle_physical_mass_projection"][key]
            - scene_macro["frozen_aligned_pointwise"][key]
        )
    source_path = Path(__file__).resolve()
    report = {
        "schema": REPORT_SCHEMA,
        "role": "target_only_non_deployable_upper_bound_diagnostic",
        "target_membership_is_projection_input": True,
        "validation_used_for_model_selection": False,
        "hard_threshold": False,
        "projection_target_space": "physical_support_equals_pre_cap_posterior_mass",
        "completion_confidence_cap": confidence_cap,
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "base_report": {
            "path": str(base_report_path),
            "sha256": sha256_file(base_report_path),
        },
        "base_checkpoint": {
            "path": str(base_checkpoint_path),
            "sha256": sha256_file(base_checkpoint_path),
        },
        "validation_scene_ids": sorted(validation_scene_ids),
        "per_validation_scene": records,
        "scene_macro": scene_macro,
        "oracle_minus_frozen_scene_macro": difference,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-cache", action="append", required=True)
    parser.add_argument("--validation-scene", action="append", required=True)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unary-element-batch-size", type=int, default=4096)
    parser.add_argument("--iteration-count", type=int, default=256)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["oracle_minus_frozen_scene_macro"], indent=2))


if __name__ == "__main__":
    main()
