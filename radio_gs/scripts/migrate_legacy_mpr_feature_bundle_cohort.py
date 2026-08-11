"""Seal legacy raw/DINO/SAM MPR caches into one formal feature-bundle cohort.

The migration changes metadata only.  It is intended for early canonical-MPR
paper scenes whose three query-free caches already share exact Gaussian rows,
view support, and responsibility, but predate the mandatory
``feature_output_bundle_sha256`` field.  No tensor is recomputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.field.observation_lifting_contract import (
    canonical_observation_contract,
    observation_contract_sha256,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.legacy_mpr_feature_bundle_cohort_migration.v2"
TENSOR_KEYS = ("xyz", "features", "valid", "view_counts", "reliability")
SHARED_POLICY_KEYS = (
    "config", "checkpoint", "selected_frame_indices", "excluded_frame_ids",
    "aggregation_mode", "registration_weight_mode", "raster_view_fusion",
    "raster_topk", "depth_tolerance", "relative_depth_tolerance",
    "alpha_threshold", "normalize_each_view",
    "registration_responsibility_cache_sha256",
)


def _load(path: str, expected: str, *, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    value, digest, source = load_torch_mapping(
        path, expected_sha256=expected, map_location="cpu", label=label
    )
    return value, {"path": str(source), "sha256": digest}


def _validate_cache(
    payload: Mapping[str, Any], *, space: str, radio_sha256: str,
) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("feature_space") != space:
        raise ValueError(f"legacy {space} MPR metadata differs")
    if metadata.get("feature_output_bundle_sha256"):
        raise ValueError("legacy migration input already has a feature bundle")
    for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened"):
        if metadata.get(key) is not False:
            raise ValueError(f"legacy {space} MPR safety declaration differs: {key}")
    lifting = metadata.get("observation_lifting_contract")
    expected_lifting = canonical_observation_contract("canonical-mpr-v1")
    if not isinstance(lifting, Mapping) or lifting.get("name") != "canonical-mpr-v1":
        raise ValueError(f"legacy {space} MPR lifting contract differs")
    lifting_mismatches = [
        key for key, expected in expected_lifting.items()
        if key in lifting and lifting.get(key) != expected
    ]
    missing_lifting = [
        key for key in expected_lifting
        if key not in lifting and key != "requires_full_observation_source_contract"
    ]
    if lifting_mismatches or missing_lifting:
        raise ValueError(f"legacy {space} MPR lifting policy differs")
    if metadata.get("shared_registration_responsibility") is not True:
        raise ValueError(f"legacy {space} MPR lacks shared responsibility")
    if space == "radio":
        if metadata.get("capability_projection_before_mpr") is not False:
            raise ValueError("legacy raw RADIO MPR projection contract differs")
    elif (
        metadata.get("capability_projection_before_mpr") is not True
        or metadata.get("custom_adaptor_head") is not False
        or metadata.get("official_adaptor_checkpoint_sha256") != radio_sha256
    ):
        raise ValueError(f"legacy {space} capability provenance differs")
    for key in TENSOR_KEYS:
        if not torch.is_tensor(payload.get(key)):
            raise ValueError(f"legacy {space} MPR lacks tensor {key}")
    return dict(metadata)


def migrate(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.radio_checkpoint) != args.expected_radio_checkpoint_sha256:
        raise ValueError("official RADIO checkpoint SHA-256 differs")
    source: dict[str, dict[str, str]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for space, path, digest in (
        ("radio", args.raw_cache, args.expected_raw_cache_sha256),
        ("dino_v3", args.dino_cache, args.expected_dino_cache_sha256),
        ("sam3", args.sam_cache, args.expected_sam_cache_sha256),
    ):
        payloads[space], source[space] = _load(
            path, digest, label=f"legacy {space} MPR cache"
        )
        metadata[space] = _validate_cache(
            payloads[space], space=space,
            radio_sha256=args.expected_radio_checkpoint_sha256,
        )
    raw = payloads["radio"]
    for space in ("dino_v3", "sam3"):
        for key in ("xyz", "valid", "view_counts", "reliability"):
            if not torch.equal(torch.as_tensor(raw[key]), torch.as_tensor(payloads[space][key])):
                raise ValueError(f"legacy {space} MPR tensor cohort differs: {key}")
        mismatched = [
            key for key in SHARED_POLICY_KEYS
            if metadata[space].get(key) != metadata["radio"].get(key)
        ]
        if mismatched:
            raise ValueError(f"legacy {space} MPR policy cohort differs: {mismatched}")
    responsibility_sha = str(metadata["radio"].get("registration_responsibility_cache_sha256", ""))
    if responsibility_sha != args.expected_responsibility_cache_sha256:
        raise ValueError("legacy cohort responsibility SHA-256 differs")
    if sha256_file(args.responsibility_cache) != responsibility_sha:
        raise ValueError("legacy cohort responsibility file differs")
    bundle_authority = {
        "schema": SCHEMA,
        "scene_id": str(args.scene_id),
        "source_caches": source,
        "responsibility_cache": {
            "path": str(Path(args.responsibility_cache).resolve()),
            "sha256": responsibility_sha,
        },
        "official_radio_checkpoint_sha256": args.expected_radio_checkpoint_sha256,
        "selected_frame_indices": metadata["radio"]["selected_frame_indices"],
        "excluded_frame_ids": metadata["radio"]["excluded_frame_ids"],
        "shared_policy": {key: metadata["radio"].get(key) for key in SHARED_POLICY_KEYS},
        "tensor_mutation": False,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    bundle_sha = canonical_json_sha256(bundle_authority)
    output_paths = {
        "radio": Path(args.raw_output).resolve(),
        "dino_v3": Path(args.dino_output).resolve(),
        "sam3": Path(args.sam_output).resolve(),
    }
    for space, output in output_paths.items():
        migrated = dict(payloads[space])
        migrated_metadata = dict(metadata[space])
        migrated_metadata["feature_output_bundle_sha256"] = bundle_sha
        completed_lifting = canonical_observation_contract("canonical-mpr-v1")
        migrated_metadata["observation_lifting_contract"] = completed_lifting
        migrated_metadata["observation_lifting_contract_sha256"] = (
            observation_contract_sha256(completed_lifting)
        )
        migrated_metadata["legacy_feature_bundle_migration"] = {
            "contract": SCHEMA,
            "source_cache": source[space],
            "bundle_authority_sha256": bundle_sha,
            "tensor_mutation": False,
            "observation_contract_completion": (
                "added_requires_full_observation_source_contract_false_and_current_digest"
            ),
        }
        migrated["metadata"] = migrated_metadata
        for key in TENSOR_KEYS:
            if not torch.equal(torch.as_tensor(migrated[key]), torch.as_tensor(payloads[space][key])):
                raise AssertionError("legacy MPR migration changed a tensor")
        write_torch_noclobber(output, migrated)
    outputs = {space: file_record(path) for space, path in output_paths.items()}
    result = {
        **bundle_authority,
        "status": "sealed_metadata_only_formal_feature_bundle_migration",
        "feature_output_bundle_sha256": bundle_sha,
        "migrated_caches": outputs,
        "access_audit": {
            "source_rgb_opened": False,
            "benchmark_query_gt_or_metric_opened": False,
        },
    }
    write_frozen_json(args.authority_output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--raw-cache", required=True)
    parser.add_argument("--expected-raw-cache-sha256", required=True)
    parser.add_argument("--dino-cache", required=True)
    parser.add_argument("--expected-dino-cache-sha256", required=True)
    parser.add_argument("--sam-cache", required=True)
    parser.add_argument("--expected-sam-cache-sha256", required=True)
    parser.add_argument("--responsibility-cache", required=True)
    parser.add_argument("--expected-responsibility-cache-sha256", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--expected-radio-checkpoint-sha256", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--dino-output", required=True)
    parser.add_argument("--sam-output", required=True)
    parser.add_argument("--authority-output", required=True)
    args = parser.parse_args()
    result = migrate(args)
    print(json.dumps({
        "status": result["status"], "scene_id": result["scene_id"],
        "feature_output_bundle_sha256": result["feature_output_bundle_sha256"],
        "authority_output": str(Path(args.authority_output).resolve()),
        "authority_sha256": sha256_file(args.authority_output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
