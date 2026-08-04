#!/usr/bin/env python3
"""Score a sealed NVOS E2S-v2 selector with the untouched frozen evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    sha256_file,
)
from radio_gs.scripts import eval_frozen_nvos_primitive_unary as frozen_evaluator
from radio_gs.scripts.build_nvos_evidence_to_support_v2 import (
    ARTIFACT_TYPE,
    validate_nvos_e2s_v2_payload,
)


def _load_selector(
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
        raise ValueError("NVOS E2S-v2 selector SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    primitive = validate_nvos_e2s_v2_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256=expected_responsibility_file_sha256,
        expected_primitive_sha256=str(args.expected_primitive_sha256),
    )
    if (
        payload["responsibility_tensor_bundle_sha256"]
        != expected_responsibility_tensor_bundle_sha256
    ):
        raise ValueError("NVOS E2S-v2 responsibility tensor bundle differs")
    if sha256_file(path) != before:
        raise ValueError("NVOS E2S-v2 selector changed across trusted load")
    return primitive, payload


def _readout_contract(selector_payload: dict) -> dict[str, object]:
    if selector_payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("NVOS E2S-v2 readout received another selector type")
    contract = dict(frozen_evaluator.METHOD_CONTRACT)
    contract.update(
        {
            "selector": "frozen_NVOS_strict_Evidence_to_Support_v2",
            "unary": "all_scribble_exact_DINOv3_diagonal_shrinkage_LDA",
            "graph": "continuous_hard_seeded_query_gated_symmetric_Laplacian",
            "connected_selection": "none",
            "selector_method_contract_sha256": selector_payload[
                "method_contract_sha256"
            ],
            "experiment_registration_sha256": selector_payload[
                "experiment_registration_sha256"
            ],
            "support_graph_sha256": selector_payload["support_graph_sha256"],
            "strict_source_only": True,
        }
    )
    return contract


def evaluate(args):
    frozen_evaluator._load_frozen_selector = _load_selector
    frozen_evaluator._readout_contract = _readout_contract
    return frozen_evaluator.evaluate(args)


def main() -> None:
    args = frozen_evaluator.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
