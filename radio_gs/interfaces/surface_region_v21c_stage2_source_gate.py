"""Complete source-chain gate for independently authorized V2.1C Stage-II."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from radio_gs.interfaces import surface_region_v21c_source_gate as frozen_gate
from radio_gs.interfaces import surface_region_v21b_source_gate as v21b_gate
from radio_gs.scripts import (
    train_surface_region_v21c_stage2_pair_constrained_adamw as trainer,
)
from radio_gs.scripts import (
    train_surface_region_v21c_two_stage_constrained_adamw as frozen_stage_i,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


PROMOTION_CHAIN_SCHEMA = (
    "radio_gs.surface_region_v21c_stage2_pair_constrained_promotion_chain.v1"
)


def _actual(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest}


def validate_source_pilot_chain(
    path: str | Path,
    *,
    expected_sha256: str,
    require_promotion: bool = True,
) -> dict[str, Any]:
    raw, result_sha, result_path = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1C Stage-II source result",
    )
    promotion = frozen_gate.validate_source_promotion_evidence(raw)
    result = promotion.pop("normalized_result")
    history = promotion.pop("normalized_history")
    execution_record = result["execution_authority"]
    inputs = trainer.prepare_inputs(
        validate_file_record(
            execution_record, label="V2.1C Stage-II execution authority"
        ),
        expected_sha256=execution_record["sha256"],
    )
    if result["stage_i_audit_result"] != inputs.execution["stage_i_audit_result"]:
        raise ValueError("V2.1C Stage-II result/audit lineage differs")

    normalization_raw, normalization_sha, normalization_path = load_torch_mapping(
        result["normalization_authority"]["path"],
        expected_sha256=result["normalization_authority"]["sha256"],
        map_location="cpu",
        label="V2.1C Stage-II normalization",
    )
    normalization = frozen_gate.validate_normalization(normalization_raw)
    normalization_record = _actual(normalization_path, normalization_sha)
    archive_raw, archive_sha, archive_path = load_torch_mapping(
        result["state_archive"]["path"],
        expected_sha256=result["state_archive"]["sha256"],
        map_location="cpu",
        label="V2.1C Stage-II state archive",
    )
    archive = frozen_gate.validate_state_archive(
        archive_raw, normalization=normalization, history=history
    )
    archive_record = _actual(archive_path, archive_sha)
    if (
        result["normalization_authority"] != normalization_record
        or result["state_archive"] != archive_record
        or archive["execution_authority"] != execution_record
        or archive["stage_i_audit_result"] != result["stage_i_audit_result"]
        or archive["normalization_authority"] != normalization_record
    ):
        raise ValueError("V2.1C Stage-II result/archive chain differs")

    checkpoint_record = None
    certificate_record = None
    selected_sha = None
    if promotion["passed"]:
        certificate_raw, certificate_sha, certificate_path = load_json_object(
            result["certificate"]["path"],
            expected_sha256=result["certificate"]["sha256"],
            label="V2.1C Stage-II certificate",
        )
        if not isinstance(certificate_raw, Mapping):
            raise ValueError("V2.1C Stage-II certificate must be a mapping")
        certificate = dict(certificate_raw)
        declared = certificate.pop("content_sha256", None)
        if (
            certificate.get("schema") != frozen_stage_i.STAGE_II_CERTIFICATE_SCHEMA
            or declared is None
            or canonical_json_sha256(certificate) != declared
            or certificate.get("benchmark_opened") is not False
        ):
            raise ValueError("V2.1C Stage-II certificate identity differs")
        certificate["content_sha256"] = declared
        certificate_record = _actual(certificate_path, certificate_sha)
        checkpoint_raw, checkpoint_sha, checkpoint_path = load_torch_mapping(
            result["checkpoint"]["path"],
            expected_sha256=result["checkpoint"]["sha256"],
            map_location="cpu",
            label="V2.1C Stage-II checkpoint",
        )
        if not isinstance(checkpoint_raw, Mapping):
            raise ValueError("V2.1C Stage-II checkpoint must be a mapping")
        checkpoint = dict(checkpoint_raw)
        if (
            checkpoint.get("schema") != frozen_stage_i.STAGE_II_CHECKPOINT_SCHEMA
            or checkpoint.get("source_access") != frozen_stage_i.source_access()
        ):
            raise ValueError("V2.1C Stage-II checkpoint identity differs")
        _state, selected_sha = v21b_gate._validate_model_state(
            checkpoint.get("model_state_dict"), normalization=normalization
        )
        selected_step = int(promotion["selected_step"])
        checkpoint_record = _actual(checkpoint_path, checkpoint_sha)
        if (
            selected_sha != checkpoint.get("model_state_dict_sha256")
            or selected_sha != history[selected_step]["model_state_dict_sha256"]
            or checkpoint.get("selected_step") != selected_step
            or checkpoint.get("normalization_authority") != normalization_record
            or checkpoint.get("certificate")
            != _actual(certificate_path, certificate_sha)
            or checkpoint.get("state_archive") != archive_record
            or certificate.get("execution_authority") != execution_record
            or certificate.get("stage_i_audit_result")
            != result["stage_i_audit_result"]
            or certificate.get("selected_step") != selected_step
            or certificate.get("selected_validation")
            != history[selected_step]["validation"]
            or certificate.get("model_state_dict_sha256") != selected_sha
            or certificate.get("normalization_authority") != normalization_record
            or certificate.get("state_archive") != archive_record
            or result["checkpoint"] != checkpoint_record
            or result["certificate"] != _actual(certificate_path, certificate_sha)
        ):
            raise ValueError("V2.1C Stage-II promoted chain differs")
    if require_promotion and promotion["passed"] is not True:
        raise ValueError("V2.1C Stage-II source promotion has no eligible state")
    return {
        "schema": PROMOTION_CHAIN_SCHEMA,
        "schema_version": 1,
        "source_result": _actual(result_path, result_sha),
        "execution_authority": execution_record,
        "stage_i_audit_result": result["stage_i_audit_result"],
        "pair_trigger_evidence": inputs.pair_trigger_evidence,
        "normalization_authority": normalization_record,
        "state_archive": archive_record,
        "certificate": certificate_record,
        "checkpoint": checkpoint_record,
        "selected_step": promotion["selected_step"],
        "model_state_dict_sha256": selected_sha,
        "promotion": promotion,
        "source_promotion_authorized": promotion["passed"] is True,
        "target_execution_authorized": False,
        "benchmark_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--expected-source-result-sha256", required=True)
    parser.add_argument("--allow-failed-diagnostic", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate_source_pilot_chain(
        args.source_result,
        expected_sha256=args.expected_source_result_sha256,
        require_promotion=not args.allow_failed_diagnostic,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["PROMOTION_CHAIN_SCHEMA", "build_parser", "validate_source_pilot_chain"]
