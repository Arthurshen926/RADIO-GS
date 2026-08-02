#!/usr/bin/env python3
"""Select one scalar contribution compositor without benchmark supervision.

The selector consumes scene-local feature-compositing audits produced on at
least two distinct, non-benchmark development scenes.  Every audit is bound to
an explicit run manifest.  Both files must state that benchmark scenes,
queries, masks, and labels were not opened; missing declarations fail closed.

Selection is deliberately worst-case rather than average-case.  A candidate
must satisfy every dense-fidelity, support, DINO-relation, and SAM-boundary
gate in every scene.  This prevents an improvement in one head or scene from
hiding a regression in another one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from radio_gs.utils.immutable_artifacts import write_frozen_json


SCREEN_NAME = "query-free-scalar-compositor-v1"
AUDIT_NAME = "query_free_feature_compositing_v1"
BASELINE_VARIANT = "alpha_mean"
CANDIDATE_VARIANTS = (
    BASELINE_VARIANT,
    "gamma_1.25",
    "gamma_1.5",
    "gamma_2",
    "top4",
)
DENSE_SPACES = ("raw_radio", "official_dino_v3", "official_sam3")
RELATION_SPACES = ("official_dino_v3", "official_sam3")
QUERY_FREE_FLAGS = (
    "uses_benchmark_scenes",
    "queries_opened",
    "masks_opened",
    "labels_opened",
)

FIXED_SELECTION_CONTRACT = {
    "baseline_variant": BASELINE_VARIANT,
    "candidate_variants": list(CANDIDATE_VARIANTS),
    "required_distinct_development_scenes": 2,
    "max_mean_dense_drop": 0.005,
    "max_p05_dense_drop": 0.01,
    "max_unsupported_fraction": 0.005,
    "min_affinity_pearson_gain_per_head_per_scene": 0.0,
    "min_boundary_margin_retention_gain_per_head_per_scene": 0.005,
    "objective": (
        "lexicographically_maximize_worst_scene_head_boundary_gain_then_"
        "worst_scene_head_affinity_gain_under_per_metric_hard_guards"
    ),
}

FIXED_SCALAR_OPERATOR_CONTRACT = {
    "primitive_scalar_source": "same_rows_as_direct_point_query",
    "base_weight": "front_to_back_alpha_contribution",
    "pixel_normalization": "sum_of_selected_contribution_weights",
    "candidate_rules": {
        "alpha_mean": "a_i=w_i/sum_j(w_j)",
        "gamma_1.25": "a_i=w_i**1.25/sum_j(w_j**1.25)",
        "gamma_1.5": "a_i=w_i**1.5/sum_j(w_j**1.5)",
        "gamma_2": "a_i=w_i**2/sum_j(w_j**2)",
        "top4": "a_i=w_i*1[rank_desc(w_i)<4]/sum_selected_j(w_j)",
    },
    "rendered_scalar": "S_q(p)=sum_i(a_i(p)*s_q(i))",
    "query_dependent_weights": False,
    "changes_primitive_scores": False,
    "changes_geometry_alpha_or_depth": False,
}

PREEXCLUDED_VARIANTS = {
    "top1": "first-surface hard selection failed historical dense-fidelity guards",
    "top2": "hard two-surface selection failed historical dense-fidelity guards",
    "front_depth_band": "nearest-depth gating failed historical dense-fidelity guards",
    "expected_depth_band": (
        "expected-depth gating failed historical dense-fidelity and visible-support guards"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_false_flags(payload: dict, *, label: str) -> None:
    for key in QUERY_FREE_FLAGS:
        if key not in payload:
            raise ValueError(f"{label} does not explicitly declare {key}=false")
        if payload[key] is not False:
            raise ValueError(f"{label} is not query-free: {key} must be false")


def _resolve_manifest(audit_path: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{audit_path}: run_manifest is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = audit_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{audit_path}: run_manifest is not readable")
    return path


def _validate_metric_report(
    report: dict,
    *,
    scene: str,
    variant: str,
) -> None:
    support = _finite(
        report.get("support_fraction_on_visible"),
        label=f"{scene}/{variant}/support_fraction_on_visible",
    )
    if not 0.0 <= support <= 1.0:
        raise ValueError(f"{scene}/{variant}: visible support is outside [0,1]")
    for space in DENSE_SPACES:
        values = report.get(space)
        if not isinstance(values, dict):
            raise ValueError(f"{scene}/{variant}: missing {space}")
        pixels = values.get("pixels")
        if not isinstance(pixels, int) or isinstance(pixels, bool) or pixels <= 0:
            raise ValueError(f"{scene}/{variant}/{space}: pixels must be positive")
        for key in ("mean_cosine", "p05_cosine"):
            value = _finite(values.get(key), label=f"{scene}/{variant}/{space}/{key}")
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{scene}/{variant}/{space}/{key} is outside [-1,1]")
        relation = values.get("local_relation")
        if not isinstance(relation, dict):
            raise ValueError(f"{scene}/{variant}/{space}: local_relation is missing")
        pairs = relation.get("pairs")
        if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs <= 0:
            raise ValueError(
                f"{scene}/{variant}/{space}: relation pairs must be positive"
            )
        pearson = _finite(
            relation.get("affinity_pearson"),
            label=f"{scene}/{variant}/{space}/affinity_pearson",
        )
        if not -1.0 <= pearson <= 1.0:
            raise ValueError(
                f"{scene}/{variant}/{space}/affinity_pearson is outside [-1,1]"
            )
        _finite(
            relation.get("boundary_margin_retention"),
            label=f"{scene}/{variant}/{space}/boundary_margin_retention",
        )
        teacher_margin = _finite(
            relation.get("teacher_boundary_margin"),
            label=f"{scene}/{variant}/{space}/teacher_boundary_margin",
        )
        if teacher_margin <= 0.0:
            raise ValueError(
                f"{scene}/{variant}/{space}: teacher boundary margin must be positive"
            )


def _validate_scene_audit(path: Path) -> dict:
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: unreadable scalar-compositor audit") from exc
    if audit.get("schema_version") != 1 or audit.get("audit") != AUDIT_NAME:
        raise ValueError(f"{path}: unsupported compositor audit schema")
    scene = audit.get("scene_id")
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError(f"{path}: scene_id is missing")
    protocol = audit.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path}: protocol is missing")
    _require_false_flags(protocol, label=f"{path} audit protocol")
    if (
        protocol.get("frame_role") != "development"
        or protocol.get("held_out_from_mpr") is not True
        or protocol.get("mpr_training_overlap") != []
        or protocol.get("official_adaptors_frozen") is not True
        or protocol.get("same_geometry_visible_pixels_for_all_variants") is not True
    ):
        raise ValueError(f"{path}: development/held-out audit contract is invalid")

    manifest_path = _resolve_manifest(path, audit.get("run_manifest"))
    manifest_sha256 = _sha256(manifest_path)
    if audit.get("run_manifest_sha256") != manifest_sha256:
        raise ValueError(f"{path}: run-manifest digest is stale")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{manifest_path}: unreadable run manifest") from exc
    if manifest.get("schema_version") != 1 or manifest.get("screen") != SCREEN_NAME:
        raise ValueError(f"{manifest_path}: unsupported scalar-compositor manifest")
    _require_false_flags(manifest, label=f"{manifest_path} run manifest")
    if manifest.get("split_role") != "development":
        raise ValueError(f"{manifest_path}: split_role must be development")
    if manifest.get("scene_id") != scene:
        raise ValueError(f"{path}: audit/manifest scene mismatch")
    if manifest.get("selection_contract") != FIXED_SELECTION_CONTRACT:
        raise ValueError(f"{manifest_path}: fixed selection contract drifted")
    if manifest.get("scalar_operator_contract") != FIXED_SCALAR_OPERATOR_CONTRACT:
        raise ValueError(f"{manifest_path}: scalar operator contract drifted")

    aggregate = audit.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != set(CANDIDATE_VARIANTS):
        raise ValueError(
            f"{path}: candidates must be exactly {list(CANDIDATE_VARIANTS)}"
        )
    for variant, report in aggregate.items():
        if not isinstance(report, dict):
            raise ValueError(f"{path}: invalid report for {variant}")
        _validate_metric_report(report, scene=scene, variant=variant)

    baseline = aggregate[BASELINE_VARIANT]
    baseline_support = float(baseline["support_fraction_on_visible"])
    if 1.0 - baseline_support > FIXED_SELECTION_CONTRACT["max_unsupported_fraction"]:
        raise ValueError(f"{path}: alpha baseline fails its visible-support contract")
    for variant, report in aggregate.items():
        for space in DENSE_SPACES:
            if report[space]["pixels"] != baseline[space]["pixels"]:
                raise ValueError(
                    f"{path}: {variant}/{space} uses different visible pixels"
                )
            relation = report[space]["local_relation"]
            baseline_relation = baseline[space]["local_relation"]
            if relation["pairs"] != baseline_relation["pairs"]:
                raise ValueError(
                    f"{path}: {variant}/{space} uses different relation pairs"
                )
            if not math.isclose(
                float(relation["teacher_boundary_margin"]),
                float(baseline_relation["teacher_boundary_margin"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{path}: {variant}/{space} changed the teacher boundary target"
                )
    return {
        "scene_id": scene,
        "audit_path": path.resolve(),
        "audit_sha256": _sha256(path),
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "aggregate": aggregate,
    }


def select_scalar_compositor(audit_paths: list[Path]) -> dict:
    """Return a frozen cross-scene scalar-compositor decision."""

    minimum_scenes = FIXED_SELECTION_CONTRACT["required_distinct_development_scenes"]
    if len(audit_paths) < minimum_scenes:
        raise ValueError(
            f"scalar-compositor selection requires at least {minimum_scenes} audits"
        )
    scenes = [_validate_scene_audit(Path(path).resolve()) for path in audit_paths]
    scene_ids = [scene["scene_id"] for scene in scenes]
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("scalar-compositor audits contain a repeated scene")

    candidate_reports: dict[str, dict] = {}
    eligible: list[str] = []
    for variant in CANDIDATE_VARIANTS:
        per_scene: dict[str, dict] = {}
        all_dense_passed = True
        all_support_passed = True
        all_relation_passed = True
        worst_mean_drop = -math.inf
        worst_p05_drop = -math.inf
        worst_unsupported = -math.inf
        worst_affinity_gain = math.inf
        worst_boundary_gain = math.inf
        for scene in scenes:
            baseline = scene["aggregate"][BASELINE_VARIANT]
            report = scene["aggregate"][variant]
            mean_drops = {
                space: float(baseline[space]["mean_cosine"])
                - float(report[space]["mean_cosine"])
                for space in DENSE_SPACES
            }
            p05_drops = {
                space: float(baseline[space]["p05_cosine"])
                - float(report[space]["p05_cosine"])
                for space in DENSE_SPACES
            }
            unsupported = 1.0 - float(report["support_fraction_on_visible"])
            affinity_gains = {
                space: float(report[space]["local_relation"]["affinity_pearson"])
                - float(baseline[space]["local_relation"]["affinity_pearson"])
                for space in RELATION_SPACES
            }
            boundary_gains = {
                space: float(
                    report[space]["local_relation"]["boundary_margin_retention"]
                )
                - float(baseline[space]["local_relation"]["boundary_margin_retention"])
                for space in RELATION_SPACES
            }
            dense_passed = (
                max(mean_drops.values())
                <= FIXED_SELECTION_CONTRACT["max_mean_dense_drop"]
                and max(p05_drops.values())
                <= FIXED_SELECTION_CONTRACT["max_p05_dense_drop"]
            )
            support_passed = (
                unsupported <= FIXED_SELECTION_CONTRACT["max_unsupported_fraction"]
            )
            if variant == BASELINE_VARIANT:
                relation_passed = True
            else:
                relation_passed = (
                    min(affinity_gains.values())
                    >= FIXED_SELECTION_CONTRACT[
                        "min_affinity_pearson_gain_per_head_per_scene"
                    ]
                    and min(boundary_gains.values())
                    >= FIXED_SELECTION_CONTRACT[
                        "min_boundary_margin_retention_gain_per_head_per_scene"
                    ]
                )
            all_dense_passed &= dense_passed
            all_support_passed &= support_passed
            all_relation_passed &= relation_passed
            worst_mean_drop = max(worst_mean_drop, *mean_drops.values())
            worst_p05_drop = max(worst_p05_drop, *p05_drops.values())
            worst_unsupported = max(worst_unsupported, unsupported)
            worst_affinity_gain = min(worst_affinity_gain, *affinity_gains.values())
            worst_boundary_gain = min(worst_boundary_gain, *boundary_gains.values())
            per_scene[scene["scene_id"]] = {
                "dense_guard_passed": dense_passed,
                "support_guard_passed": support_passed,
                "relation_guard_passed": relation_passed,
                "mean_dense_drop": mean_drops,
                "p05_dense_drop": p05_drops,
                "unsupported_fraction": unsupported,
                "affinity_pearson_gain": affinity_gains,
                "boundary_margin_retention_gain": boundary_gains,
            }
        promotion_eligible = bool(
            variant != BASELINE_VARIANT
            and all_dense_passed
            and all_support_passed
            and all_relation_passed
        )
        if promotion_eligible:
            eligible.append(variant)
        candidate_reports[variant] = {
            "promotion_eligible": promotion_eligible,
            "all_scene_dense_guard_passed": all_dense_passed,
            "all_scene_support_guard_passed": all_support_passed,
            "all_scene_per_head_relation_guard_passed": all_relation_passed,
            "worst_mean_dense_drop": worst_mean_drop,
            "worst_p05_dense_drop": worst_p05_drop,
            "worst_unsupported_fraction": worst_unsupported,
            "worst_scene_head_affinity_pearson_gain": worst_affinity_gain,
            "worst_scene_head_boundary_margin_retention_gain": worst_boundary_gain,
            "per_scene": per_scene,
        }

    if eligible:
        selected = max(
            eligible,
            key=lambda variant: (
                candidate_reports[variant][
                    "worst_scene_head_boundary_margin_retention_gain"
                ],
                candidate_reports[variant]["worst_scene_head_affinity_pearson_gain"],
                -candidate_reports[variant]["worst_p05_dense_drop"],
                -candidate_reports[variant]["worst_mean_dense_drop"],
                -CANDIDATE_VARIANTS.index(variant),
            ),
        )
        status = "cross_scene_query_free_scalar_compositor_selected"
    else:
        selected = BASELINE_VARIANT
        status = "cross_scene_alpha_mean_retained_no_uniform_candidate"

    return {
        "schema_version": 1,
        "screen": SCREEN_NAME,
        "selection_status": status,
        "selected_variant": selected,
        "promotion_allowed": selected != BASELINE_VARIANT,
        "selection_uses_task_labels": False,
        "selection_uses_benchmark_scenes": False,
        "queries_opened": False,
        "masks_opened": False,
        "labels_opened": False,
        "fixed_selection_contract": FIXED_SELECTION_CONTRACT,
        "scalar_operator_contract": FIXED_SCALAR_OPERATOR_CONTRACT,
        "preexcluded_variants": PREEXCLUDED_VARIANTS,
        "historical_risk_evidence_is_selection_input": False,
        "scenes": scene_ids,
        "scene_audits": [
            {
                "scene_id": scene["scene_id"],
                "audit": str(scene["audit_path"]),
                "audit_sha256": scene["audit_sha256"],
                "run_manifest": str(scene["manifest_path"]),
                "run_manifest_sha256": scene["manifest_sha256"],
            }
            for scene in scenes
        ],
        "candidates": candidate_reports,
        "next_gate": (
            "first freeze canonical-v5 on its independent query-free capacity gate; "
            "then run this scalar-compositor screen on that exact frozen field; "
            "next run target-blind text-response fidelity on the frozen field and "
            "compositor; open benchmark protocols only after all three gates"
        ),
    }


def _parse_paths(raw: str) -> list[Path]:
    paths = [Path(value) for value in raw.replace(",", " ").split() if value]
    if not paths:
        raise ValueError("--scene-audits is empty")
    return paths


def _write_atomic(path: Path, payload: dict) -> None:
    # A compositor decision is a frozen authority, not a mutable convenience
    # report.  Identical recomputation is allowed; replacement with a different
    # decision is rejected and the first writer always wins.
    write_frozen_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-audits", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = select_scalar_compositor(_parse_paths(args.scene_audits))
    _write_atomic(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
