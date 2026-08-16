#!/usr/bin/env python3
"""Fail-closed validator for the frozen NVOS RGB-assisted full8 contract.

This validator never opens target masks, target metrics, RGB images, or tensor
artifacts. It verifies only JSON authorities and their byte hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.data.promptable_nvs_manifest import validate_manifest
from radio_gs.five_benchmark_method_v1 import validate_method_authority
from radio_gs.utils.immutable_artifacts import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO_ROOT / (
    "paper/artifacts/"
    "nvos_rgb_assisted_full8_method_v1_evaluation_contract_20260816.json"
)
EXPECTED_COHORT = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)


def _load(path: str | Path, label: str) -> tuple[dict[str, Any], Path, str]:
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = REPO_ROOT / source
    source = source.resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, source, sha256_file(source)


def _authority_path(value: Mapping[str, Any], contract_path: Path) -> Path:
    source = Path(str(value.get("path", ""))).expanduser()
    if not source.is_absolute():
        source = REPO_ROOT / source
    return source.resolve(strict=True)


def validate_contract(path: str | Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract, contract_path, contract_sha = _load(path, "NVOS RGB contract")
    if (
        contract.get("schema_version") != 1
        or contract.get("artifact_type")
        != "radio_gs.nvos_rgb_assisted_full8_evaluation_contract"
        or contract.get("contract_id") != "nvos-rgb-assisted-full8-method-v1-v1"
        or contract.get("status") != "frozen_development_evaluation_contract"
        or tuple(map(str, contract.get("cohort", ()))) != EXPECTED_COHORT
    ):
        raise ValueError("NVOS RGB-assisted contract identity differs")

    method_ref = contract.get("method_authority")
    dataset_ref = contract.get("dataset_index_prompt_and_metric_authority")
    if not isinstance(method_ref, Mapping) or not isinstance(dataset_ref, Mapping):
        raise ValueError("NVOS RGB-assisted contract authorities are absent")
    method_path = _authority_path(method_ref, contract_path)
    dataset_path = _authority_path(dataset_ref, contract_path)
    if sha256_file(method_path) != method_ref.get("sha256"):
        raise ValueError("Method-v1 authority SHA-256 differs")
    if sha256_file(dataset_path) != dataset_ref.get("sha256"):
        raise ValueError("NVOS dataset authority SHA-256 differs")
    method, _, _ = _load(method_path, "Method-v1 authority")
    dataset, _, _ = _load(dataset_path, "NVOS dataset authority")
    validate_method_authority(method)
    normalized = validate_manifest(dataset, check_files=False)
    if (
        tuple(map(str, method["frozen_cohorts"]["nvos"])) != EXPECTED_COHORT
        or tuple(str(row["scene_id"]) for row in normalized["scenes"])
        != EXPECTED_COHORT
        or normalized["protocol_hash"] != dataset_ref.get("protocol_hash")
    ):
        raise ValueError("NVOS RGB-assisted cohort or protocol binding differs")
    target_access = method.get("target_access", {})
    readout = method.get("readouts", {}).get("nvos", {})
    legacy_protocol = dataset.get("protocol", {})
    if (
        target_access.get("field_construction_uses_target_rgb") is not False
        or target_access.get("field_construction_uses_target_masks") is not False
        or "nvos"
        not in target_access.get("query_transient_target_rgb_allowed_for", ())
        or readout.get("operator")
        != "signed_field_prompt_to_query_transient_target_rgb_frozen_sam"
        or readout.get("trials") != 10
        or readout.get("positive_points") != 3
        or readout.get("negative_points") != 3
        or legacy_protocol.get("target_rgb_at_query") != "forbidden"
        or legacy_protocol.get("target_rgb_during_field_training") != "forbidden"
    ):
        raise ValueError("Method, RGB contract, and strict ablation boundary differ")

    construction = contract.get("field_construction", {})
    query = contract.get("query_workspace", {})
    barrier = contract.get("pre_gt_barrier", {})
    claims = contract.get("claim_eligibility", {})
    strict = contract.get("strict_unseen_ablation", {})
    evaluation = contract.get("evaluation", {})
    if any(
        construction.get(key) is not False
        for key in (
            "target_rgb_opened",
            "target_mask_opened",
            "target_metric_opened",
            "benchmark_query_used",
            "persistent_state_updated_at_query",
        )
    ) or construction.get("source_registered_rgb_only") is not True:
        raise ValueError("NVOS field-construction information boundary differs")
    if (
        query.get("target_rgb_access")
        != "allowed_only_after_all_full8_signed_field_prompts_are_sealed"
        or query.get("target_mask_access")
        != "forbidden_until_complete_full8_prediction_batch_is_sealed"
        or query.get("adapter") != "frozen_official_sam3_point_interface"
        or query.get("adapter_checkpoint_sha256")
        != "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
        or query.get("trials") != 10
        or query.get("positive_points_per_trial") != 3
        or query.get("negative_points_per_trial") != 3
        or query.get("candidate_selection")
        != "none_exactly_one_candidate_per_trial"
        or query.get("reference_mask_selection") is not False
        or query.get("graph_or_connected_component") is not False
    ):
        raise ValueError("NVOS query-transient RGB/SAM policy differs")
    if (
        evaluation.get("prediction_representation")
        != "continuous_margin_mean_binary_sam_vote_minus_0.5"
        or evaluation.get("threshold")
        != {"mode": "fixed", "value": 0.0, "comparison": "greater_or_equal"}
        or evaluation.get("metrics") != ["foreground_iou", "pixel_accuracy"]
        or evaluation.get("dataset_aggregation")
        != "equal_weight_macro_over_ordered_full8"
    ):
        raise ValueError("NVOS frozen metric or aggregation differs")
    if not all(
        barrier.get(key) is True
        for key in (
            "required",
            "ordered_full8_fields_hash_verified",
            "ordered_full8_signed_prompts_sealed_before_first_target_rgb_open",
            "ordered_full8_predictions_and_receipts_hash_verified_before_first_target_mask_open",
        )
    ):
        raise ValueError("NVOS pre-GT barrier differs")
    if (
        claims.get("rgb_assisted_main_method") is not True
        or claims.get("strict_unseen_protocol") is not False
        or claims.get("blind_confirmation") is not False
        or claims.get("sota_claim") is not False
        or claims.get("result_classification") != "development_evidence"
        or strict.get("retained") is not True
        or strict.get("target_rgb_at_query") != "forbidden"
    ):
        raise ValueError("NVOS claim or strict-unseen ablation boundary differs")
    return {
        "contract": str(contract_path),
        "contract_sha256": contract_sha,
        "contract_id": contract["contract_id"],
        "method_authority": str(method_path),
        "method_authority_sha256": method_ref["sha256"],
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": dataset_ref["sha256"],
        "protocol_hash": normalized["protocol_hash"],
        "scene_order": list(EXPECTED_COHORT),
        "target_or_metric_bytes_opened": False,
        "rgb_assisted_main_method": True,
        "strict_unseen_retained_as_ablation": True,
        "result_classification": "development_evidence",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    report = validate_contract(parser.parse_args(argv).contract)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
