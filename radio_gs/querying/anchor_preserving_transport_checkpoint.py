"""Fail-closed loader for promoted anchor-preserving prompt checkpoints."""

from __future__ import annotations

from pathlib import Path

from radio_gs.utils.immutable_artifacts import (
    load_torch_payload,
    sha256_file,
)

from .registered_evidence_to_unary import RegisteredEvidenceToUnaryV2


CHECKPOINT_SCHEMA = "radio_gs.anchor_preserving_prompt_transport.checkpoint.v1"
CHECKPOINT_SCHEMA_V21 = "radio_gs.anchor_preserving_prompt_transport.checkpoint.v2_1"


def load_anchor_preserving_prompt_head(
    checkpoint_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_transport_contract_sha256: str,
    expected_trainer_sha256: str,
    expected_preregistration_sha256: str,
    expected_fit_authority_sha256: str,
    expected_confirmation_authority_sha256: str,
    require_promoted_result: bool = True,
) -> RegisteredEvidenceToUnaryV2:
    """Load only a fully bound V2 checkpoint and its promoted source result."""

    payload, _, _ = load_torch_payload(
        checkpoint_path,
        expected_sha256=expected_checkpoint_sha256,
        map_location="cpu",
        label="anchor-preserving prompt checkpoint",
    )
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("anchor-preserving prompt checkpoint schema differs")
    architecture = payload.get("architecture")
    expected_architecture = {
        "hidden_dim": 32,
        "max_delta_logit": 4.0,
        "fully_observed_tolerance": 1e-5,
    }
    if architecture != expected_architecture:
        raise ValueError("anchor-preserving prompt checkpoint architecture differs")
    lineage = payload.get("lineage")
    expected_lineage = {
        "transport_contract_sha256": expected_transport_contract_sha256,
        "trainer_sha256": expected_trainer_sha256,
        "preregistration_sha256": expected_preregistration_sha256,
        "fit_authority_sha256": expected_fit_authority_sha256,
        "confirmation_authority_sha256": expected_confirmation_authority_sha256,
    }
    if lineage != expected_lineage:
        raise ValueError("anchor-preserving prompt checkpoint lineage differs")
    result_path = Path(str(payload.get("result_path", ""))).expanduser().resolve()
    result_sha = str(payload.get("result_sha256", ""))
    if not result_path.is_file() or sha256_file(result_path) != result_sha:
        raise ValueError("anchor-preserving prompt source result lineage differs")
    if require_promoted_result:
        import json

        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if (
            result.get("schema")
            != "radio_gs.anchor_preserving_prompt_transport.clean_gate_result.v1"
            or result.get("promotion_gate_passed") is not True
            or result.get("decision")
            != "eligible_for_one_preregistered_target_sentinel"
        ):
            raise ValueError("anchor-preserving prompt source result was not promoted")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("anchor-preserving prompt checkpoint state differs")
    head = RegisteredEvidenceToUnaryV2(
        hidden_dim=int(architecture["hidden_dim"]),
        max_delta_logit=float(architecture["max_delta_logit"]),
    )
    if float(head.fully_observed_tolerance) != float(
        architecture["fully_observed_tolerance"]
    ):
        raise ValueError("anchor-preserving prompt tolerance differs")
    head.load_state_dict(state_dict, strict=True)
    return head.eval().requires_grad_(False)


def load_anchor_preserving_prompt_head_v21(
    checkpoint_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_transport_contract_sha256: str,
    expected_trainer_sha256: str,
    expected_base_asset_loader_sha256: str,
    expected_preregistration_sha256: str,
    expected_fit_authority_sha256: str,
    expected_confirmation_authority_sha256: str,
    require_promoted_result: bool = True,
) -> RegisteredEvidenceToUnaryV2:
    """Load only the risk-sensitive V2.1 checkpoint and exact lineage."""

    payload, _, _ = load_torch_payload(
        checkpoint_path,
        expected_sha256=expected_checkpoint_sha256,
        map_location="cpu",
        label="anchor-preserving prompt V2.1 checkpoint",
    )
    if payload.get("schema") != CHECKPOINT_SCHEMA_V21:
        raise ValueError("anchor-preserving prompt V2.1 checkpoint schema differs")
    architecture = payload.get("architecture")
    if architecture != {
        "hidden_dim": 32,
        "max_delta_logit": 4.0,
        "fully_observed_tolerance": 1e-5,
    }:
        raise ValueError("anchor-preserving prompt V2.1 architecture differs")
    if payload.get("risk_sensitive_training") != {
        "objective": "0.5_prompt_mean_plus_0.5_worst_quartile_cvar",
        "cvar_fraction": 0.25,
        "uniform_mixture": 0.5,
    }:
        raise ValueError("anchor-preserving prompt V2.1 risk contract differs")
    expected_lineage = {
        "fit_authority_sha256": expected_fit_authority_sha256,
        "confirmation_authority_sha256": expected_confirmation_authority_sha256,
        "trainer_sha256": expected_trainer_sha256,
        "base_asset_loader_sha256": expected_base_asset_loader_sha256,
        "transport_contract_sha256": expected_transport_contract_sha256,
        "preregistration_sha256": expected_preregistration_sha256,
    }
    if payload.get("lineage") != expected_lineage:
        raise ValueError("anchor-preserving prompt V2.1 lineage differs")
    result_path = Path(str(payload.get("result_path", ""))).expanduser().resolve()
    result_sha = str(payload.get("result_sha256", ""))
    if not result_path.is_file() or sha256_file(result_path) != result_sha:
        raise ValueError("anchor-preserving prompt V2.1 source result lineage differs")
    if require_promoted_result:
        import json

        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if (
            result.get("schema")
            != "radio_gs.anchor_preserving_prompt_transport.clean_gate_result.v2_1"
            or result.get("promotion_gate_passed") is not True
            or result.get("decision")
            != "eligible_for_one_preregistered_target_sentinel"
        ):
            raise ValueError("anchor-preserving prompt V2.1 source result was not promoted")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("anchor-preserving prompt V2.1 state differs")
    head = RegisteredEvidenceToUnaryV2(hidden_dim=32, max_delta_logit=4.0)
    head.load_state_dict(state_dict, strict=True)
    return head.eval().requires_grad_(False)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_V21",
    "load_anchor_preserving_prompt_head",
    "load_anchor_preserving_prompt_head_v21",
]
