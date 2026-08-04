#!/usr/bin/env python3
"""Score a frozen reference-only CV-selected NVOS primitive unary."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    sha256_file,
)
from radio_gs.scripts import eval_frozen_nvos_primitive_unary as frozen_evaluator
from radio_gs.scripts.build_nvos_signed_scribble_cv_selector import (
    ARTIFACT_TYPE,
    validate_nvos_cv_selector_payload,
)


def _load_cv_selector(
    args,
    authority: PromptResponsibilityAuthority,
    *,
    expected_responsibility_file_sha256: str,
    expected_responsibility_tensor_bundle_sha256: str,
    expected_benchmark_manifest_sha256: str,
    expected_source_rgb_path: Path,
):
    del expected_benchmark_manifest_sha256, expected_source_rgb_path
    path = Path(args.completion).resolve()
    before = sha256_file(path)
    if before != str(args.expected_completion_sha256):
        raise ValueError("signed-scribble CV selector artifact SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    primitive = validate_nvos_cv_selector_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256=expected_responsibility_file_sha256,
        expected_primitive_sha256=str(args.expected_primitive_sha256),
    )
    if (
        payload["responsibility_tensor_bundle_sha256"]
        != expected_responsibility_tensor_bundle_sha256
    ):
        raise ValueError("signed-scribble CV selector responsibility bundle differs")
    if sha256_file(path) != before:
        raise ValueError("signed-scribble CV selector changed across trusted load")
    return primitive, payload


def _cv_readout_contract(selector_payload: dict) -> dict[str, object]:
    if selector_payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("signed-scribble CV readout received another selector type")
    contract = dict(frozen_evaluator.METHOD_CONTRACT)
    contract.update(
        {
            "selector": "frozen_NVOS_signed_scribble_3fold_OOF_selected_support",
            "selected_action": selector_payload["selected_action"],
            "selection": (
                "reference_only_responsibility_balanced_logloss_then_weighted_AUC_"
                "then_strong_unary_tie_break"
            ),
            "selector_method_contract_sha256": selector_payload[
                "method_contract_sha256"
            ],
            "cv_contract_sha256": selector_payload["cv_contract_sha256"],
            "experiment_registration_sha256": selector_payload[
                "experiment_registration_sha256"
            ],
            "support_graph_sha256": selector_payload["support_graph_sha256"],
            "knn_cache_sha256": selector_payload["knn_cache_sha256"],
            "feature_hash_sha256": selector_payload["feature_hash_sha256"],
            "compatibility_boundary": (
                "C_RADIO_hashed_relation_diagnostic_not_native_DINO_or_"
                "exact_LUDVIG_feature_match"
            ),
            "target_used_for_selection": False,
        }
    )
    return contract


def evaluate(args):
    frozen_evaluator._load_frozen_selector = _load_cv_selector
    frozen_evaluator._readout_contract = _cv_readout_contract
    return frozen_evaluator.evaluate(args)


def main() -> None:
    args = frozen_evaluator.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
