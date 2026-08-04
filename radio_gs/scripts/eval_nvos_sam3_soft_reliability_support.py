#!/usr/bin/env python3
"""Score the frozen posthoc SAM3 entropy-reliability support."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    sha256_file,
)
from radio_gs.scripts import eval_frozen_nvos_primitive_unary as frozen_evaluator
from radio_gs.scripts.build_nvos_sam3_soft_reliability_support import (
    ARTIFACT_TYPE,
    validate_nvos_sam3_soft_support_payload,
)


def _load_soft_reliability_selector(
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
        raise ValueError("SAM3 soft reliability artifact SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    primitive = validate_nvos_sam3_soft_support_payload(
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
        raise ValueError("SAM3 soft reliability responsibility bundle differs")
    if Path(payload["source_rgb_path"]).resolve() != expected_source_rgb_path.resolve():
        raise ValueError("SAM3 soft reliability source RGB differs")
    reference_completion = Path(payload["reference_completion_path"]).resolve()
    reference_receipt = Path(payload["reference_completion_receipt_path"]).resolve()
    if sha256_file(reference_completion) != payload["reference_completion_sha256"]:
        raise ValueError("frozen SAM3 completion changed")
    if (
        sha256_file(reference_receipt)
        != payload["reference_completion_receipt_sha256"]
    ):
        raise ValueError("frozen SAM3 completion receipt changed")
    if sha256_file(path) != before:
        raise ValueError("SAM3 soft reliability artifact changed across load")
    return primitive, payload


def _soft_reliability_readout_contract(selector_payload: dict) -> dict[str, object]:
    if selector_payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("soft reliability readout received another selector type")
    contract = dict(frozen_evaluator.METHOD_CONTRACT)
    contract.update(
        {
            "selector": "frozen_posthoc_SAM3_entropy_reliability_soft_support",
            "source_completion": "frozen_ten_mask_mean_q_without_Li",
            "reliability": "c(q)=1-H2(q)/ln2_parameter_free",
            "positive_observation": "q*c(q)_with_raw_signed_scribble_overwrite",
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
            "pixel_tensor_bundle_sha256": selector_payload[
                "pixel_tensor_bundle_sha256"
            ],
            "support_graph_sha256": selector_payload["support_graph_sha256"],
            "knn_cache_sha256": selector_payload["knn_cache_sha256"],
            "feature_hash_sha256": selector_payload["feature_hash_sha256"],
            "claim_boundary": "posthoc_registered_not_independent_validation",
            "target_used_for_numeric_parameters": False,
        }
    )
    return contract


def evaluate(args):
    frozen_evaluator._load_frozen_selector = _load_soft_reliability_selector
    frozen_evaluator._readout_contract = _soft_reliability_readout_contract
    return frozen_evaluator.evaluate(args)


def main() -> None:
    args = frozen_evaluator.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
