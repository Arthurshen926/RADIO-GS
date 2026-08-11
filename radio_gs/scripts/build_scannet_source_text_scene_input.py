#!/usr/bin/env python3
"""Adapt one sealed source semantic sidecar to text-likelihood scene input."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.data.scannet_source_region_semantics import (
    PREREGISTERED_SOURCE_FIT_SCENES,
    sha256_file,
    validate_source_region_semantic_sidecar,
)
from radio_gs.interfaces.surface_region_full_scalar_contract import (
    SURFACE_REGION_FULL_SCALAR_NAMES,
)
from radio_gs.querying.source_text_query_likelihood import (
    SOURCE_TEXT_SCENE_INPUT_SCHEMA,
    build_source_text_training_shard,
)
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)


RECEIPT_SCHEMA = "radio_gs.scannet_source_text_scene_input_receipt.v1"
_MEAN_DISPERSION = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_directional_dispersion"
)
_MEAN_EVIDENCE = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_observation_evidence"
)
_MEAN_PURITY = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_visibility_purity_value"
)
_MEAN_PURITY_KNOWN = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_visibility_purity_known"
)


def _canonical_output(path: str | Path) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    raw.parent.mkdir(parents=True, exist_ok=True)
    return raw.parent.resolve(strict=True) / raw.name


def _write_torch_noclobber(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = _canonical_output(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_json_noclobber(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = _canonical_output(path)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError:
        if output.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different artifact: {output}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _record(path: Path) -> dict[str, str]:
    source = path.expanduser().resolve(strict=True)
    return {"path": str(source), "sha256": sha256_file(source)}


def _field_coverage_reliability(
    raw_full_scalar_summary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scalar = torch.as_tensor(raw_full_scalar_summary).detach().cpu().float()
    if scalar.ndim != 2 or scalar.shape[1] != len(SURFACE_REGION_FULL_SCALAR_NAMES):
        raise ValueError("source full-scalar summary must be [R,18]")
    if not bool(torch.isfinite(scalar).all()):
        raise ValueError("source full-scalar summary contains NaN or infinity")
    dispersion = scalar[:, _MEAN_DISPERSION].clamp(0, 1)
    evidence = scalar[:, _MEAN_EVIDENCE].clamp(0, 1)
    purity = scalar[:, _MEAN_PURITY].clamp(0, 1)
    purity_known = scalar[:, _MEAN_PURITY_KNOWN].clamp(0, 1)
    reliability = torch.stack(
        (1.0 - dispersion, evidence, purity * purity_known), dim=1
    ).mean(dim=1)
    return evidence.contiguous(), reliability.clamp(0, 1).contiguous()


def build_scene_input(
    *,
    scene_id: str,
    accepted_region_authority: str | Path,
    semantic_sidecar: str | Path,
    full_scalar_training_shard: str | Path,
    class_text_cache: str | Path,
    canonical_negative_text_cache: str | Path,
) -> dict[str, Any]:
    if scene_id not in PREREGISTERED_SOURCE_FIT_SCENES:
        raise PermissionError("scene is not a preregistered source-fit scene")
    paths = {
        "descriptor_source": Path(accepted_region_authority),
        "semantic_label_source": Path(semantic_sidecar),
        "field_state_source": Path(full_scalar_training_shard),
        "class_text_source": Path(class_text_cache),
        "canonical_negative_text_source": Path(canonical_negative_text_cache),
    }
    paths = {key: value.expanduser().resolve(strict=True) for key, value in paths.items()}
    accepted = torch.load(paths["descriptor_source"], map_location="cpu", weights_only=True)
    sidecar = validate_source_region_semantic_sidecar(
        torch.load(paths["semantic_label_source"], map_location="cpu", weights_only=True)
    )
    scalar_shard = torch.load(
        paths["field_state_source"], map_location="cpu", weights_only=True
    )
    class_text = torch.load(
        paths["class_text_source"], map_location="cpu", weights_only=True
    )
    negative_text = torch.load(
        paths["canonical_negative_text_source"], map_location="cpu", weights_only=True
    )
    if accepted.get("scene_id") != scene_id or sidecar.get("scene_id") != scene_id:
        raise ValueError("source scene authorities differ")
    descriptor = torch.as_tensor(accepted.get("accepted_v2_e0")).detach().cpu().float()
    rows = int(descriptor.shape[0]) if descriptor.ndim == 2 else -1
    if descriptor.ndim != 2 or descriptor.shape[1] != 1536:
        raise ValueError("accepted source descriptor must be [R,1536]")
    if sidecar["nyu40_class_distribution"].shape[0] != rows:
        raise ValueError("semantic sidecar and descriptor row count differ")
    if not torch.equal(
        torch.as_tensor(accepted.get("canonical_region_indices")).long(),
        sidecar["canonical_region_indices"].long(),
    ) or [str(value) for value in accepted.get("region_fingerprints", [])] != [
        str(value) for value in sidecar.get("region_fingerprints", [])
    ]:
        raise ValueError("semantic sidecar and accepted canonical row order differ")
    scalar_descriptor = torch.as_tensor(scalar_shard.get("accepted_v2_e0")).float()
    if scalar_descriptor.shape != descriptor.shape or not torch.equal(
        scalar_descriptor, descriptor
    ):
        raise ValueError("full-scalar source shard and descriptor authority differ")
    eligible = torch.as_tensor(scalar_shard.get("eligible"))
    if eligible.shape != (rows,) or eligible.dtype != torch.bool:
        raise ValueError("full-scalar source eligibility differs")
    region_row_ids = scalar_shard.get("region_row_ids")
    expected_row_ids = [
        f"{scene_id}:accepted-v2-canonical-v1:{fingerprint}"
        for fingerprint in accepted.get("region_fingerprints", [])
    ]
    if list(region_row_ids or []) != expected_row_ids:
        raise ValueError("full-scalar source shard canonical row authority differs")

    class_ids = list(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"])
    class_names = [NYU40_ID_TO_NAME[class_id] for class_id in class_ids]
    if class_text.get("queries") != class_names or class_text.get(
        "exact_scannet_nyu40"
    ) is not True:
        raise ValueError("ScanNet class text cache is not the frozen split19 authority")
    class_embeddings = torch.as_tensor(class_text.get("embeddings")).float()
    negative_embeddings = torch.as_tensor(negative_text.get("embeddings")).float()
    if class_embeddings.shape != (len(class_ids), 1536) or (
        negative_embeddings.ndim != 2 or negative_embeddings.shape[1] != 1536
    ):
        raise ValueError("source text cache dimensions differ")
    coverage, reliability = _field_coverage_reliability(
        scalar_shard.get("raw_full_scalar_summary")
    )
    target = sidecar["nyu40_class_distribution"][:, class_ids].float().contiguous()
    source = {
        "schema": SOURCE_TEXT_SCENE_INPUT_SCHEMA,
        "schema_version": 1,
        "scene_id": scene_id,
        "physical_space_id": sidecar["physical_space_id"],
        "partition": "source_train",
        "descriptors": descriptor.contiguous(),
        "semantic_class_distribution": target,
        "class_ids": class_ids,
        "class_names": class_names,
        "class_text_embeddings": class_embeddings.contiguous(),
        "canonical_negative_text_embeddings": negative_embeddings.contiguous(),
        "valid": (sidecar["valid"] & eligible).contiguous(),
        "coverage": coverage,
        "reliability": reliability,
        "training_label_weight": sidecar["semantic_coverage"].float().contiguous(),
        "lineage": {key: _record(path) for key, path in paths.items()},
        "source_access": {
            "official_scannet_train_scene": True,
            "source_train_semantic_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "lerf_queries_or_ground_truth_opened": False,
            "target_rgb_or_mask_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
            "per_scene_or_per_query_metric_tuning": False,
        },
    }
    # This is a fail-closed validation call, not a second materialization.
    build_source_text_training_shard(source)
    return source


def run(
    *,
    scene_id: str,
    accepted_region_authority: str | Path,
    semantic_sidecar: str | Path,
    full_scalar_training_shard: str | Path,
    class_text_cache: str | Path,
    canonical_negative_text_cache: str | Path,
    output: str | Path,
    receipt: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    source = build_scene_input(
        scene_id=scene_id,
        accepted_region_authority=accepted_region_authority,
        semantic_sidecar=semantic_sidecar,
        full_scalar_training_shard=full_scalar_training_shard,
        class_text_cache=class_text_cache,
        canonical_negative_text_cache=canonical_negative_text_cache,
    )
    output_path = _write_torch_noclobber(output, source)
    valid = torch.as_tensor(source["valid"])
    target = torch.as_tensor(source["semantic_class_distribution"])
    receipt_payload = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_source_fit_text_scene_input",
        "scene_id": scene_id,
        "scene_input": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "row_count": int(valid.numel()),
        "valid_row_count": int(valid.sum()),
        "present_class_ids": [
            int(class_id)
            for class_index, class_id in enumerate(source["class_ids"])
            if float(target[valid, class_index].sum()) > 0
        ],
        "source_access": dict(source["source_access"]),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    receipt_path = _write_json_noclobber(receipt, receipt_payload)
    return output_path, receipt_path, receipt_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True, choices=PREREGISTERED_SOURCE_FIT_SCENES)
    parser.add_argument("--accepted-region-authority", required=True)
    parser.add_argument("--semantic-sidecar", required=True)
    parser.add_argument("--full-scalar-training-shard", required=True)
    parser.add_argument("--class-text-cache", required=True)
    parser.add_argument("--canonical-negative-text-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    _output, _receipt, payload = run(
        scene_id=args.scene_id,
        accepted_region_authority=args.accepted_region_authority,
        semantic_sidecar=args.semantic_sidecar,
        full_scalar_training_shard=args.full_scalar_training_shard,
        class_text_cache=args.class_text_cache,
        canonical_negative_text_cache=args.canonical_negative_text_cache,
        output=args.output,
        receipt=args.receipt,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
