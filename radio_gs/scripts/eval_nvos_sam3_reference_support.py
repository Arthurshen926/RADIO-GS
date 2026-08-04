#!/usr/bin/env python3
"""Score a frozen source-reference SAM3 NVOS support artifact."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    sha256_file,
)
from radio_gs.scripts import eval_frozen_nvos_primitive_unary as frozen_evaluator
from radio_gs.scripts.build_nvos_sam3_reference_support import (
    ARTIFACT_TYPE,
    validate_nvos_sam3_support_payload,
)


def _load_sam3_reference_selector(
    args,
    authority: PromptResponsibilityAuthority,
    *,
    expected_responsibility_file_sha256: str,
    expected_responsibility_tensor_bundle_sha256: str,
    expected_benchmark_manifest_sha256: str,
    expected_source_rgb_path: Path,
):
    del expected_benchmark_manifest_sha256
    path = Path(args.completion).resolve()
    before = sha256_file(path)
    if before != str(args.expected_completion_sha256):
        raise ValueError("SAM3 reference support artifact SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    primitive = validate_nvos_sam3_support_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256=expected_responsibility_file_sha256,
        expected_completion_sha256=str(payload["reference_completion_sha256"]),
        expected_primitive_sha256=str(args.expected_primitive_sha256),
    )
    if (
        payload["responsibility_tensor_bundle_sha256"]
        != expected_responsibility_tensor_bundle_sha256
    ):
        raise ValueError("SAM3 reference support responsibility bundle differs")
    if Path(payload["source_rgb_path"]).resolve() != expected_source_rgb_path.resolve():
        raise ValueError("SAM3 reference support source RGB path differs")
    reference_completion = Path(payload["reference_completion_path"]).resolve()
    reference_receipt = Path(payload["reference_completion_receipt_path"]).resolve()
    if sha256_file(reference_completion) != payload["reference_completion_sha256"]:
        raise ValueError("frozen SAM3 reference completion changed")
    if (
        sha256_file(reference_receipt)
        != payload["reference_completion_receipt_sha256"]
    ):
        raise ValueError("frozen SAM3 reference completion receipt changed")
    if sha256_file(path) != before:
        raise ValueError("SAM3 reference support changed across trusted load")
    return primitive, payload


def _sam3_reference_readout_contract(selector_payload: dict) -> dict[str, object]:
    if selector_payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("SAM3 reference readout received another selector type")
    contract = dict(frozen_evaluator.METHOD_CONTRACT)
    contract.update(
        {
            "selector": "frozen_source_reference_official_SAM3_completed_support",
            "source_completion": (
                "ten_deterministic_positive_triplets_multimask_false_mean_Li_"
                "with_raw_negative_supervision"
            ),
            "graph": "exact_k200_plus_self_C_RADIO_hashed_relation_query_conditioned_diffusion",
            "selector_method_contract_sha256": selector_payload[
                "method_contract_sha256"
            ],
            "experiment_registration_sha256": selector_payload[
                "experiment_registration_sha256"
            ],
            "reference_completion_sha256": selector_payload[
                "reference_completion_sha256"
            ],
            "reference_completion_receipt_sha256": selector_payload[
                "reference_completion_receipt_sha256"
            ],
            "support_graph_sha256": selector_payload["support_graph_sha256"],
            "knn_cache_sha256": selector_payload["knn_cache_sha256"],
            "feature_hash_sha256": selector_payload["feature_hash_sha256"],
            "compatibility_boundary": (
                "source_only_LUDVIG_role_compatible_not_model_or_protocol_exact"
            ),
            "target_used_for_selection": False,
        }
    )
    return contract


def evaluate(args):
    frozen_evaluator._load_frozen_selector = _load_sam3_reference_selector
    frozen_evaluator._readout_contract = _sam3_reference_readout_contract
    return frozen_evaluator.evaluate(args)


def main() -> None:
    args = frozen_evaluator.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
