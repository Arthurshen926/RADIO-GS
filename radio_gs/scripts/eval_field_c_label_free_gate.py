#!/usr/bin/env python3
"""Evaluate the preregistered, query-free Field-C observation-lifting gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values.float(), float(probability)))


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    registration, registration_sha256, registration_path = load_json_object(
        args.registration,
        expected_sha256=args.expected_registration_sha256,
        label="Field-C marginal gate registration",
    )
    schema = str(registration.get("schema_version", ""))
    if schema not in {
        "canonical_field_c_marginal_gate_registration_v1",
        "canonical_field_c_marginal_gate_registration_v2",
    }:
        raise ValueError("Field-C gate registration schema differs")
    safety = dict(registration.get("safety", {}))
    if set(safety.values()) != {False}:
        raise ValueError("Field-C gate safety registration is not target blind")

    authorities = dict(registration.get("authorities", {}))
    for name in ("config", "geometry_checkpoint", "feature_manifest"):
        authority = dict(authorities.get(name, {}))
        if sha256_file(authority["path"]) != authority.get("sha256"):
            raise ValueError(f"Field-C gate {name} authority differs")
    implementation = dict(authorities.get("implementation_sha256", {}))
    implementation_root = Path(__file__).resolve().parents[1]
    implementation_paths = {
        "build_gaussian_multiview_teacher_cache.py": (
            implementation_root / "scripts" / "build_gaussian_multiview_teacher_cache.py"
        ),
        "contribution_compositor.py": (
            implementation_root / "rendering" / "contribution_compositor.py"
        ),
        "tensor_cache_io.py": (
            implementation_root / "training" / "tensor_cache_io.py"
        ),
    }
    if set(implementation) != set(implementation_paths):
        raise ValueError("Field-C implementation authority keys differ")
    for name, path in implementation_paths.items():
        if sha256_file(path) != implementation[name]:
            raise ValueError(f"Field-C implementation changed after registration: {name}")

    control_record = dict(registration["control"])
    candidate_record = dict(registration["candidate"])
    expected_control_sha256 = str(
        control_record.get("sha256", "") or args.expected_control_sha256
    )
    if not expected_control_sha256:
        raise ValueError("Field-C gate requires a trusted control SHA-256")
    control, control_sha256, _ = load_mpr_cache(
        control_record["path"],
        expected_sha256=expected_control_sha256,
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=False,
    )
    candidate, candidate_sha256, _ = load_mpr_cache(
        candidate_record["path"],
        expected_sha256=args.expected_candidate_sha256,
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=True,
    )
    control_metadata = dict(control["metadata"])
    candidate_metadata = dict(candidate["metadata"])
    if schema.endswith("_v2"):
        for key in (
            "aggregation_mode",
            "alpha_threshold",
            "raster_view_fusion",
            "raster_reliability_mode",
            "normalize_each_view",
        ):
            if control_metadata.get(key) != control_record.get(key):
                raise ValueError(f"Field-C corrected control policy differs: {key}")
    for key in (
        "aggregation_mode",
        "marginal_responsibility_contract",
        "registration_weight_mode",
        "alpha_threshold",
        "raster_view_fusion",
        "raster_reliability_mode",
        "normalize_each_view",
    ):
        if candidate_metadata.get(key) != candidate_record.get(key):
            raise ValueError(f"Field-C candidate policy differs: {key}")
    if candidate_metadata.get("selected_frame_indices") != control_record.get(
        "selected_frame_indices"
    ) or control_metadata.get("selected_frame_indices") != control_record.get(
        "selected_frame_indices"
    ):
        raise ValueError("Field-C control/candidate frame authority differs")
    if candidate.get("geometry_fingerprint") != control.get(
        "geometry_fingerprint"
    ):
        raise ValueError("Field-C control/candidate geometry differs")

    control_valid = torch.as_tensor(control["valid"]).bool()
    candidate_valid = torch.as_tensor(candidate["valid"]).bool()
    common = control_valid & candidate_valid
    if not bool(common.any()):
        raise ValueError("Field-C gate has no common valid primitive rows")
    control_features = torch.as_tensor(control["features"])[common].float()
    candidate_features = torch.as_tensor(candidate["features"])[common].float()
    control_resultant = control_features.norm(dim=-1).clamp(0.0, 1.0)
    candidate_resultant = candidate_features.norm(dim=-1).clamp(0.0, 1.0)
    resultant_delta = float(candidate_resultant.mean() - control_resultant.mean())
    support_ratio = float(candidate_valid.sum() / control_valid.sum().clamp_min(1))
    purity = torch.as_tensor(candidate["visibility_purity"])[candidate_valid].float()
    purity_p10 = _quantile(purity, 0.10)
    target_cosine = F.cosine_similarity(
        candidate_features, control_features, dim=-1, eps=1e-8
    )

    gate = dict(registration["gate"])
    conditions = {
        "common_valid_mean_resultant_delta": (
            resultant_delta
            >= float(gate["common_valid_mean_resultant_delta_minimum"])
        ),
        "candidate_valid_count_over_control": (
            support_ratio
            >= float(gate["candidate_valid_count_over_control_minimum"])
        ),
        "candidate_visibility_purity_p10": (
            purity_p10
            <= float(gate["candidate_visibility_purity_p10_maximum"])
        ),
    }
    if "candidate_control_target_cosine_mean_minimum" in gate:
        conditions["candidate_control_target_cosine_mean"] = (
            float(target_cosine.mean())
            >= float(gate["candidate_control_target_cosine_mean_minimum"])
        )
    passed = bool(all(conditions.values()))
    report: dict[str, object] = {
        "schema_version": "canonical_field_c_marginal_gate_result_v1",
        "registration": {
            "path": str(registration_path),
            "sha256": registration_sha256,
        },
        "control": {
            "path": control_record["path"],
            "sha256": control_sha256,
            "valid_count": int(control_valid.sum()),
        },
        "candidate": {
            "path": candidate_record["path"],
            "sha256": candidate_sha256,
            "valid_count": int(candidate_valid.sum()),
        },
        "measurements": {
            "common_valid_count": int(common.sum()),
            "control_common_mean_resultant": float(control_resultant.mean()),
            "candidate_common_mean_resultant": float(candidate_resultant.mean()),
            "common_valid_mean_resultant_delta": resultant_delta,
            "candidate_valid_count_over_control": support_ratio,
            "candidate_visibility_purity_mean": float(purity.mean()),
            "candidate_visibility_purity_p10": purity_p10,
            "candidate_visibility_purity_p50": _quantile(purity, 0.50),
            "candidate_visibility_purity_p90": _quantile(purity, 0.90),
            "candidate_control_target_cosine_mean": float(target_cosine.mean()),
            "candidate_control_target_cosine_p05": _quantile(target_cosine, 0.05),
        },
        "conditions": conditions,
        "passed": passed,
        "decision": (
            "promote_to_capability_targets_and_training"
            if passed
            else "stop_field_c_marginal_branch_before_frozen_benchmarks"
        ),
        "safety": safety,
    }
    _atomic_json(Path(args.output), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--expected-control-sha256", default="")
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
