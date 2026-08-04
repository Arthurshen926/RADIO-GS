#!/usr/bin/env python3
"""Score a frozen strict NVOS C-RADIO primitive-PCA40 selector."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    sha256_file,
)
from radio_gs.scripts import eval_frozen_nvos_primitive_unary as frozen_evaluator
from radio_gs.scripts.build_nvos_strict_pca40_query_conditioned_support import (
    ARTIFACT_TYPE,
    validate_nvos_strict_pca40_support_payload,
)


def _load_strict_pca40_selector(
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
        raise ValueError("strict PCA40 support artifact SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    primitive = validate_nvos_strict_pca40_support_payload(
        payload,
        authority=authority,
        expected_responsibility_file_sha256=expected_responsibility_file_sha256,
        expected_primitive_sha256=str(args.expected_primitive_sha256),
    )
    if (
        payload["responsibility_tensor_bundle_sha256"]
        != expected_responsibility_tensor_bundle_sha256
    ):
        raise ValueError("strict PCA40 responsibility tensor bundle differs")
    if sha256_file(path) != before:
        raise ValueError("strict PCA40 support artifact changed across trusted load")
    return primitive, payload


def _strict_pca40_readout_contract(selector_payload: dict) -> dict[str, object]:
    if selector_payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("strict PCA40 readout received another selector type")
    contract = dict(frozen_evaluator.METHOD_CONTRACT)
    contract.update(
        {
            "selector": "frozen_NVOS_strict_raw_scribble_query_conditioned_support",
            "graph": "exact_k200_plus_self_C_RADIO_DINOv3_primitive_PCA40_relation_diffusion",
            "selector_method_contract_sha256": selector_payload[
                "method_contract_sha256"
            ],
            "experiment_registration_sha256": selector_payload[
                "experiment_registration_sha256"
            ],
            "support_graph_sha256": selector_payload["support_graph_sha256"],
            "knn_cache_sha256": selector_payload["knn_cache_sha256"],
            "relation_cache_sha256": selector_payload["relation_cache_sha256"],
            "relation_feature_sha256": selector_payload["relation_feature_sha256"],
            "source_feature_sha256": selector_payload["source_feature_sha256"],
            "compatibility_boundary": (
                "C_RADIO_DINOv3_primitive_PCA40_adaptation_not_native_"
                "LUDVIG_DINOv2_or_exact_feature_match"
            ),
        }
    )
    return contract


def evaluate(args):
    frozen_evaluator._load_frozen_selector = _load_strict_pca40_selector
    frozen_evaluator._readout_contract = _strict_pca40_readout_contract
    return frozen_evaluator.evaluate(args)


def main() -> None:
    args = frozen_evaluator.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
