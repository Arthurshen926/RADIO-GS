#!/usr/bin/env python3
"""Compile a direct-language exact-MPR cache into a typed LERF posterior."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from radio_gs.scripts.eval_lerf_direct_3d_selection import canonical_negative_relevancy_query_scores
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_payload,
    write_frozen_json,
    write_torch_noclobber,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    payload, digest, source = load_torch_payload(
        args.direct_language_cache,
        expected_sha256=args.expected_direct_language_cache_sha256,
        label="LERF direct-language cache",
    )
    if not isinstance(payload, Mapping) or payload.get("schema") != "radio_gs.lerf_direct_language_score_cache.v1":
        raise ValueError("LERF direct-language cache contract differs")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("capability_before_mpr") is not True:
        raise ValueError("LERF direct-language construction order differs")
    if any(bool(metadata.get(key, False)) for key in (
        "benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened",
        "target_metrics_opened",
    )):
        raise ValueError("LERF direct-language cache opened forbidden target information")
    positive = torch.as_tensor(payload["positive_query_scores"]).float()
    negative = torch.as_tensor(payload["negative_query_scores"]).float()
    queries = list(map(str, payload["positive_query_ids"]))
    probability = canonical_negative_relevancy_query_scores(
        positive, negative, logit_scale=args.logit_scale,
    )
    output = Path(args.output).resolve()
    posterior = {
        "schema": "radio_gs.lerf_direct_language_posterior.v1",
        "schema_version": 1,
        "scene": str(payload["scene"]),
        "query_scores": probability,
        "identity_query_scores": probability.clone(),
        "valid": torch.as_tensor(payload["valid"]).bool(),
        "xyz": torch.as_tensor(payload["xyz"]).float(),
        "metadata": {
            "query_names": queries,
            "query_family": "text_object_extent",
            "typed_posterior": "object_aware_universal_field_v2_text_object_posterior_direct_language_ceiling_v1",
            "score_threshold": 0.5,
            "separate_identity_localization": True,
            "localization_authority": "direct_source_view_language_response_exact_mpr",
            "segmentation_authority": "direct_source_view_language_response_exact_mpr_no_extent",
            "persistent_second_semantic_field": False,
            "query_dependent_diagnostic_cache": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "query_text_opened_at_readout": True,
            "canonical_negative_formula": "sigmoid(10*(positive-max(canonical_negatives)))",
            "canonical_negative_logit_scale": args.logit_scale,
            "direct_language_cache": {"path": str(source), "sha256": digest},
        },
    }
    write_torch_noclobber(output, posterior)
    report = {
        "status": "complete", "scene": posterior["scene"],
        "queries": len(queries), "output": file_record(output),
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-language-cache", required=True)
    parser.add_argument("--expected-direct-language-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--logit-scale", type=float, default=10.0)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
