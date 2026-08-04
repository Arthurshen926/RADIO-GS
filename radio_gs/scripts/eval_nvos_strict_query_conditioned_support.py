#!/usr/bin/env python3
"""Score a frozen strict NVOS query-conditioned selector.

The frozen target renderer, score freeze/reload boundary, resize, threshold,
and metric implementation remain byte-for-byte in
``eval_frozen_nvos_primitive_unary``.  This module only supplies a fail-closed
adapter for the new selector artifact and its truthful readout receipt; it
does not modify the frozen evaluator source.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    sha256_file,
)
from radio_gs.scripts import eval_frozen_nvos_primitive_unary as frozen_evaluator
from radio_gs.scripts.build_nvos_strict_query_conditioned_support import (
    ARTIFACT_TYPE,
    validate_nvos_strict_support_payload,
)


def _load_strict_selector(
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
        raise ValueError("strict support artifact SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    primitive = validate_nvos_strict_support_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256=expected_responsibility_file_sha256,
        expected_primitive_sha256=str(args.expected_primitive_sha256),
    )
    if (
        payload["responsibility_tensor_bundle_sha256"]
        != expected_responsibility_tensor_bundle_sha256
    ):
        raise ValueError("strict support responsibility tensor bundle differs")
    if sha256_file(path) != before:
        raise ValueError("strict support artifact changed across trusted load")
    return primitive, payload


def _strict_readout_contract(selector_payload: dict) -> dict[str, object]:
    if selector_payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("strict readout received another selector type")
    contract = dict(frozen_evaluator.METHOD_CONTRACT)
    contract.update(
        {
            "selector": "frozen_NVOS_strict_raw_scribble_query_conditioned_support",
            "graph": "exact_k200_plus_self_C_RADIO_hashed_relation_query_conditioned_diffusion",
            "selector_method_contract_sha256": selector_payload[
                "method_contract_sha256"
            ],
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
        }
    )
    return contract


def evaluate(args):
    # A dedicated process runs this adapter, so installing the two selector
    # hooks cannot affect another evaluator invocation.  Everything after the
    # selector boundary executes the already frozen implementation.
    frozen_evaluator._load_frozen_selector = _load_strict_selector
    frozen_evaluator._readout_contract = _strict_readout_contract
    return frozen_evaluator.evaluate(args)


def main() -> None:
    args = frozen_evaluator.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
