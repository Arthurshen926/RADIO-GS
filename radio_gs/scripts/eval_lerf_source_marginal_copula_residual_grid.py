#!/usr/bin/env python3
"""Evaluate a target-blind bounded copula-residual grid on source views.

This is a sibling of the frozen descriptor-first source evaluator.  It opens
only its hash-bound source authority, generic query bank, and source-heldout
teacher maps.  Accepted and genuine-MPR descriptors are rendered once per
frame.  Candidate response rankings are injected into the accepted response
through the exact-marginal interface; benchmark queries, masks, labels, and
target metrics are never opened.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

from radio_gs.evaluation.lerf_source_text_response_ranking import (
    build_scene_summary,
    evaluate_source_response_frame,
)
from radio_gs.evaluation.render_ceiling import normalize_premultiplied
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
import radio_gs.interfaces.lerf_marginal_preserving_copula_residual as copula
from radio_gs.interfaces.lerf_marginal_preserving_copula_residual import (
    CONTRACT,
    marginal_preserving_copula_residual,
)
import radio_gs.scripts.materialize_lerf_source_text_response_confirmation as audit_base
import radio_gs.scripts.materialize_lerf_source_text_response_summaries as dev_base
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
)


PREREGISTRATION_SCHEMA = "radio_gs.lerf_source_marginal_copula_grid_preregistration.v1"
RESULT_SCHEMA = "radio_gs.lerf_source_marginal_copula_grid_result.v1"
IMPLEMENTATION = file_record(Path(__file__).resolve())
INTERFACE_IMPLEMENTATION = file_record(Path(copula.__file__).resolve())


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA-256 record")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def validate_preregistration(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("marginal-copula preregistration must be an object")
    prereg = dict(value)
    if (
        prereg.get("schema") != PREREGISTRATION_SCHEMA
        or prereg.get("schema_version") != 1
        or prereg.get("status") != "sealed_source_only_before_execution"
        or prereg.get("implementation") != IMPLEMENTATION
        or prereg.get("interface_implementation") != INTERFACE_IMPLEMENTATION
        or prereg.get("interface_contract") != CONTRACT
        or prereg.get("split") not in {"dev", "audit"}
    ):
        raise ValueError("marginal-copula preregistration schema differs")
    policies = prereg.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("marginal-copula policy grid is empty")
    ids: set[str] = set()
    for policy in policies:
        if not isinstance(policy, Mapping) or set(policy) != {
            "policy_id",
            "strength",
            "maximum_rank_fraction",
        }:
            raise ValueError("marginal-copula policy schema differs")
        policy_id = str(policy["policy_id"])
        strength = float(policy["strength"])
        fraction = float(policy["maximum_rank_fraction"])
        if (
            not policy_id
            or policy_id in ids
            or not 0.0 <= strength <= 1.0
            or not 0.0 <= fraction <= 1.0
        ):
            raise ValueError("marginal-copula policy value differs")
        ids.add(policy_id)
    authorities = prereg.get("source_authorities")
    if (
        not isinstance(authorities, Mapping)
        or list(authorities) != ["ramen", "teatime"]
    ):
        raise ValueError("marginal-copula source cohort differs")
    for scene, record in authorities.items():
        _record(record, label=f"{scene} source authority")
    access = prereg.get("access_audit")
    if not isinstance(access, Mapping) or any(
        access.get(key) is not False
        for key in (
            "benchmark_queries_opened",
            "benchmark_masks_or_labels_opened",
            "target_metric_execution_authorized",
        )
    ):
        raise ValueError("marginal-copula preregistration is not target blind")
    return prereg


def _load_preregistration(path: str | Path, digest: str) -> tuple[dict[str, Any], dict[str, str]]:
    value, actual, source = load_json_object(
        path,
        expected_sha256=digest,
        label="source marginal-copula preregistration",
    )
    return validate_preregistration(value), {"path": str(source), "sha256": actual}


def _load_source_authority(
    record: Mapping[str, str], *, split: str
) -> tuple[dict[str, Any], dict[str, str], Any]:
    if split == "dev":
        authority, authority_record = dev_base.load_authority(
            record["path"], record["sha256"]
        )
        base = dev_base
    else:
        authority, authority_record = audit_base.load_authority(
            record["path"], record["sha256"]
        )
        base = audit_base
    if authority_record != dict(record):
        raise ValueError("source authority path binding differs")
    return authority, authority_record, base


def _responses(descriptor_map: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    descriptors = F.normalize(descriptor_map.float(), dim=0, eps=1e-12)
    embeddings = F.normalize(text.float(), dim=-1, eps=1e-12)
    return torch.einsum("qd,dhw->qhw", embeddings, descriptors)


def _top_decile_support_metrics(
    scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float | int]:
    """Target-blind topology metrics for generic-query top-decile support."""

    values = torch.as_tensor(scores).detach().float().cpu()
    teacher = torch.as_tensor(teacher_scores).detach().float().cpu()
    valid = torch.as_tensor(valid_mask).detach().bool().cpu()
    if (
        values.ndim != 3
        or teacher.shape != values.shape
        or valid.shape != values.shape[1:]
        or int(valid.sum()) < 10
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(teacher).all())
    ):
        raise ValueError("source top-decile support metric inputs differ")
    count = int(valid.sum())
    selected = int(np.ceil(0.1 * count))
    structure = ndimage.generate_binary_structure(2, 1)
    boundary_f_sum = 0.0
    component_abs_error_sum = 0.0
    selected_count_error_sum = 0
    for query_index in range(values.shape[0]):
        predicted_values = values[query_index, valid]
        teacher_values = teacher[query_index, valid]
        predicted_order = torch.argsort(predicted_values, descending=True, stable=True)
        teacher_order = torch.argsort(teacher_values, descending=True, stable=True)
        predicted_flat = torch.zeros(count, dtype=torch.bool)
        teacher_flat = torch.zeros(count, dtype=torch.bool)
        predicted_flat[predicted_order[:selected]] = True
        teacher_flat[teacher_order[:selected]] = True
        predicted_map = torch.zeros_like(valid)
        teacher_map = torch.zeros_like(valid)
        predicted_map[valid] = predicted_flat
        teacher_map[valid] = teacher_flat
        predicted_np = predicted_map.numpy()
        teacher_np = teacher_map.numpy()
        predicted_boundary = predicted_np & ~ndimage.binary_erosion(
            predicted_np, structure=structure, border_value=0
        )
        teacher_boundary = teacher_np & ~ndimage.binary_erosion(
            teacher_np, structure=structure, border_value=0
        )
        intersection = int(np.logical_and(predicted_boundary, teacher_boundary).sum())
        denominator = int(predicted_boundary.sum()) + int(teacher_boundary.sum())
        boundary_f_sum += 1.0 if denominator == 0 else 2.0 * intersection / denominator
        predicted_components = int(ndimage.label(predicted_np, structure=structure)[1])
        teacher_components = int(ndimage.label(teacher_np, structure=structure)[1])
        component_abs_error_sum += abs(predicted_components - teacher_components)
        selected_count_error_sum += abs(int(predicted_np.sum()) - selected)
    return {
        "support_units": int(values.shape[0]),
        "top_decile_boundary_f_sum": boundary_f_sum,
        "top_decile_component_abs_error_sum": component_abs_error_sum,
        "selected_count_error_sum": selected_count_error_sum,
    }


@torch.inference_mode()
def evaluate_scene(
    preregistration_path: str | Path,
    preregistration_sha256: str,
    *,
    scene_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    prereg, prereg_record = _load_preregistration(
        preregistration_path, preregistration_sha256
    )
    if scene_id not in prereg["source_authorities"]:
        raise ValueError("requested source scene is outside the preregistration")
    authority, authority_record, base = _load_source_authority(
        prereg["source_authorities"][scene_id], split=str(prereg["split"])
    )
    required_visible = str(authority["execution"]["required_cuda_visible_devices"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != required_visible:
        raise RuntimeError("marginal-copula source evaluator CUDA visibility differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("marginal-copula source evaluator requires one visible GPU")
    output = Path(output_path).expanduser().resolve()
    if str(output) != str(output_path) or output.exists() or output.is_symlink():
        raise FileExistsError("marginal-copula source output must be new and canonical")

    inputs = authority["inputs"]
    for key in ("source_view_preregistration", "scene_config", "geometry_checkpoint"):
        validate_file_record(inputs[key], label=key)
    source_view_prereg = dev_base._record(
        inputs["source_view_preregistration"], label="source-view preregistration"
    )
    reseal_records, teacher_root, reseal_record = dev_base._load_source_reseal(
        authority, source_view_prereg
    )
    text, query_ids, query_bank = base._load_query_bank(authority)

    device = torch.device("cuda:0")
    config_record = dev_base._record(inputs["scene_config"], label="scene config")
    geometry_record = dev_base._record(
        inputs["geometry_checkpoint"], label="geometry checkpoint"
    )
    model, _codec, renderer, _sharpener, _refiner, config, _hybrid = load_render_pipeline(
        config_record["path"],
        geometry_record["path"],
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
        expected_checkpoint_sha256=geometry_record["sha256"],
    )
    if (
        int(model.get_xyz().shape[0]) != int(authority["geometry"]["num_gaussians"])
        or dev_base._geometry_sha256(model) != authority["geometry"]["xyz_sha256"]
    ):
        raise ValueError("marginal-copula source geometry differs")
    frame_ids = list(authority["source_heldout_frame_ids"])
    dataset = dev_base._dataset(config, frame_ids)
    frame_to_index = {
        int(frame): index for index, frame in enumerate(dataset.frame_indices)
    }
    if sorted(frame_to_index) != frame_ids:
        raise ValueError("marginal-copula source pose inventory differs")

    methods = {str(method["role"]): method for method in authority["methods"]}
    if set(methods) != {"control", "candidate"}:
        raise ValueError("marginal-copula requires accepted and genuine-MPR methods")
    descriptor_rows: dict[str, torch.Tensor] = {}
    for role in ("control", "candidate"):
        payload = dev_base.load_descriptor_payload(
            methods[role],
            scene_id=scene_id,
            num_gaussians=int(model.get_xyz().shape[0]),
            xyz_sha256=str(authority["geometry"]["xyz_sha256"]),
        )
        descriptor_rows[role] = dev_base.build_primitive_descriptor_rows(
            payload,
            num_gaussians=int(model.get_xyz().shape[0]),
            device=device,
        )
        del payload
        gc.collect()

    text_device = text.to(device)
    control_frames: list[dict[str, object]] = []
    candidate_frames: dict[str, list[dict[str, object]]] = {
        str(policy["policy_id"]): [] for policy in prereg["policies"]
    }
    diagnostics: dict[str, dict[str, float | int | bool]] = {
        str(policy["policy_id"]): {
            "marginal_exact_every_frame": True,
            "maximum_rank_displacement": 0,
            "changed_valid_count": 0,
            "valid_count": 0,
            "support_units": 0,
            "top_decile_boundary_f_sum": 0.0,
            "top_decile_component_abs_error_sum": 0.0,
            "selected_count_error_sum": 0,
        }
        for policy in prereg["policies"]
    }
    control_support = {
        "support_units": 0,
        "top_decile_boundary_f_sum": 0.0,
        "top_decile_component_abs_error_sum": 0.0,
        "selected_count_error_sum": 0,
    }
    for frame_id in frame_ids:
        record = reseal_records[frame_id]
        teacher = dev_base._load_teacher_tensor(teacher_root, record)
        teacher_float = teacher.float().to(device)
        teacher_response = _responses(teacher_float, text_device).cpu()
        teacher_nonzero = torch.linalg.vector_norm(teacher.float(), dim=0) > 0
        pose = torch.from_numpy(dataset.poses_w2c[frame_to_index[frame_id]]).float().to(device)
        response_maps: dict[str, torch.Tensor] = {}
        alpha_cpu: torch.Tensor | None = None
        for role in ("control", "candidate"):
            rendered = renderer.render_feature_rows(
                model,
                pose,
                descriptor_rows[role],
                feature_height=46,
                feature_width=62,
                alpha_normalize=False,
            )
            descriptor_map = normalize_premultiplied(
                rendered["feature_map"].float(), rendered["alpha_map"].float()
            )
            response_maps[role] = _responses(descriptor_map, text_device).cpu()
            current_alpha = rendered["alpha_map"].float().cpu()
            if alpha_cpu is None:
                alpha_cpu = current_alpha
            elif not torch.equal(alpha_cpu, current_alpha):
                raise ValueError("accepted/candidate geometry alpha differs")
            del rendered, descriptor_map
        assert alpha_cpu is not None
        valid_mask = (alpha_cpu >= dev_base.ALPHA_THRESHOLD) & teacher_nonzero
        accepted_selected = response_maps["control"][:, valid_mask]
        genuine_selected = response_maps["candidate"][:, valid_mask]
        control_frames.append(
            evaluate_source_response_frame(
                response_maps["control"],
                teacher_response,
                valid_mask,
                scene_id=scene_id,
                frame_id=frame_id,
                method_id="accepted_o2_source_response_anchor",
                query_ids=query_ids,
                query_bank=query_bank,
                method_input_sha256=tensor_sha256(accepted_selected),
                teacher_input_sha256=str(record["sha256"]),
            )
        )
        frame_control_support = _top_decile_support_metrics(
            response_maps["control"], teacher_response, valid_mask
        )
        for key, value in frame_control_support.items():
            control_support[key] += value
        for policy in prereg["policies"]:
            policy_id = str(policy["policy_id"])
            fused = marginal_preserving_copula_residual(
                accepted_selected,
                genuine_selected,
                torch.ones(accepted_selected.shape[-1], dtype=torch.bool),
                strength=float(policy["strength"]),
                maximum_rank_fraction=float(policy["maximum_rank_fraction"]),
            )
            fused_map = response_maps["control"].clone()
            fused_map[:, valid_mask] = fused.scores
            candidate_frames[policy_id].append(
                evaluate_source_response_frame(
                    fused_map,
                    teacher_response,
                    valid_mask,
                    scene_id=scene_id,
                    frame_id=frame_id,
                    method_id=f"marginal_copula_{policy_id}",
                    query_ids=query_ids,
                    query_bank=query_bank,
                    method_input_sha256=tensor_sha256(fused.scores),
                    teacher_input_sha256=str(record["sha256"]),
                )
            )
            frame_support = _top_decile_support_metrics(
                fused_map, teacher_response, valid_mask
            )
            diag = diagnostics[policy_id]
            diag["marginal_exact_every_frame"] = bool(
                diag["marginal_exact_every_frame"] and fused.marginal_exact
            )
            diag["maximum_rank_displacement"] = max(
                int(diag["maximum_rank_displacement"]),
                fused.maximum_rank_displacement,
            )
            diag["changed_valid_count"] = int(diag["changed_valid_count"]) + fused.changed_valid_count
            diag["valid_count"] = int(diag["valid_count"]) + fused.valid_count
            for key, value in frame_support.items():
                diag[key] = diag[key] + value
        del teacher, teacher_float, teacher_response, response_maps

    control_summary = build_scene_summary(
        control_frames,
        scene_id=scene_id,
        method_id="accepted_o2_source_response_anchor",
        required_frame_ids=frame_ids,
    )
    candidate_summaries: dict[str, dict[str, object]] = {}
    control_support["top_decile_boundary_f_mean"] = float(
        control_support["top_decile_boundary_f_sum"]
    ) / max(int(control_support["support_units"]), 1)
    control_support["top_decile_component_abs_error_mean"] = float(
        control_support["top_decile_component_abs_error_sum"]
    ) / max(int(control_support["support_units"]), 1)
    for policy in prereg["policies"]:
        policy_id = str(policy["policy_id"])
        summary = build_scene_summary(
            candidate_frames[policy_id],
            scene_id=scene_id,
            method_id=f"marginal_copula_{policy_id}",
            required_frame_ids=frame_ids,
        )
        diag = diagnostics[policy_id]
        diag["changed_valid_fraction"] = float(diag["changed_valid_count"]) / max(
            int(diag["valid_count"]), 1
        )
        diag["top_decile_boundary_f_mean"] = float(
            diag["top_decile_boundary_f_sum"]
        ) / max(int(diag["support_units"]), 1)
        diag["top_decile_component_abs_error_mean"] = float(
            diag["top_decile_component_abs_error_sum"]
        ) / max(int(diag["support_units"]), 1)
        candidate_summaries[policy_id] = summary
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_only_no_target_access",
        "scene_id": scene_id,
        "split": prereg["split"],
        "implementation": IMPLEMENTATION,
        "interface_contract": CONTRACT,
        "preregistration": prereg_record,
        "source_authority": authority_record,
        "source_reseal": reseal_record,
        "control_summary": control_summary,
        "control_support_diagnostics": control_support,
        "candidate_summaries": candidate_summaries,
        "diagnostics": diagnostics,
        "access_audit": {
            "source_heldout_views_opened": True,
            "generic_query_bank_opened": True,
            "benchmark_queries_opened": False,
            "benchmark_masks_or_labels_opened": False,
            "target_metric_executed": False,
            "target_metric_execution_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_frozen_json(output, result)
    return {**result, "output": file_record(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--scene-id", choices=("ramen", "teatime"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_scene(
        args.preregistration,
        args.preregistration_sha256,
        scene_id=args.scene_id,
        output_path=args.output,
    )
    print(json.dumps({"output": result["output"], "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
