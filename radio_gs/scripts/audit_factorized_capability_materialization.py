#!/usr/bin/env python3
"""Audit a factorized-v2 capability bank without opening benchmark targets."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.scripts.audit_factorized_radio_label_free_gate import (
    _CenteredVariation,
    _cache_valid,
    _feature_rows,
    _validate_mpr_lineage,
    select_common_valid_rows,
)
from radio_gs.training.tensor_cache_io import ShardedMPRCache, load_mpr_cache
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
    write_frozen_json,
)


EXPERIMENT = "canonical_factorized_radio_v1_lego_capability_materialization"
CAPABILITIES = {
    "dino_v3": ("dino_v3", "appearance", 4096),
    "sam3": ("sam3", "boundary", 1024),
}


def compare_materialized_rows(
    predicted: torch.Tensor,
    exact: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Measure source-only fidelity and centered variation on aligned rows."""

    predicted = torch.as_tensor(predicted).detach().cpu()
    exact = torch.as_tensor(exact).detach().cpu()
    if (
        predicted.ndim != 2
        or predicted.shape != exact.shape
        or predicted.shape[0] <= 0
        or batch_size <= 0
    ):
        raise ValueError("materialized and exact capability rows do not align")
    dimension = int(predicted.shape[1])
    variations = {
        "materialized": _CenteredVariation(dimension),
        "exact": _CenteredVariation(dimension),
    }
    cosines: list[torch.Tensor] = []
    for start in range(0, int(predicted.shape[0]), int(batch_size)):
        stop = min(start + int(batch_size), int(predicted.shape[0]))
        materialized_rows = F.normalize(
            predicted[start:stop].float(), dim=-1, eps=1e-8
        )
        exact_rows = F.normalize(exact[start:stop].float(), dim=-1, eps=1e-8)
        cosines.append((materialized_rows * exact_rows).sum(dim=-1))
        variations["materialized"].update(materialized_rows)
        variations["exact"].update(exact_rows)
    exact_rms = variations["exact"].rms()
    if exact_rms <= 0:
        raise ValueError("exact source capability has zero centered variation")
    materialized_rms = variations["materialized"].rms()
    cosine = torch.cat(cosines).float()
    return {
        "rows": int(predicted.shape[0]),
        "dimension": dimension,
        "cosine_mean": float(cosine.mean()),
        "cosine_p05": float(torch.quantile(cosine, 0.05)),
        "centered_row_variation_rms": float(materialized_rms),
        "exact_centered_row_variation_rms": float(exact_rms),
        "centered_row_variation_ratio_to_exact": float(
            materialized_rms / exact_rms
        ),
    }


@torch.inference_mode()
def audit_full_fp16_parity(
    *,
    field: torch.nn.Module,
    adaptor: torch.nn.Module,
    global_rows: torch.Tensor,
    stored: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Independently rematerialize every valid row and require exact fp16 bytes."""

    rows = torch.as_tensor(global_rows).long().cpu()
    stored = torch.as_tensor(stored).detach().cpu()
    if stored.ndim != 2 or stored.shape[0] != rows.numel() or batch_size <= 0:
        raise ValueError("full materialization parity rows do not align")
    field = field.to(device).eval().requires_grad_(False)
    adaptor = adaptor.to(device).eval().requires_grad_(False)
    unequal = 0
    maximum_absolute_error = 0.0
    for start in range(0, int(rows.numel()), int(batch_size)):
        stop = min(start + int(batch_size), int(rows.numel()))
        selected = rows[start:stop].to(device)
        raw_radio = field.radio_features(selected).float()
        projected = F.normalize(adaptor(raw_radio).float(), dim=-1, eps=1e-8)
        materialized = projected.half().cpu()
        reference = stored[start:stop]
        unequal += int(materialized.ne(reference).sum())
        maximum_absolute_error = max(
            maximum_absolute_error,
            float((materialized.float() - reference.float()).abs().max()),
        )
    return {
        "rows": int(rows.numel()),
        "values": int(stored.numel()),
        "unequal_fp16_values": unequal,
        "maximum_absolute_error": maximum_absolute_error,
        "exact_fp16_parity": unequal == 0,
    }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _validate_registration_identity(
    preregistration: Mapping[str, Any], expected_registration: object
) -> str:
    """Require an explicit, exact preregistration identity.

    The default remains the original Lego experiment for command-line
    compatibility.  Other scenes must opt in with their full registration
    string; accepting a prefix or filename-derived identity would weaken the
    frozen protocol boundary.
    """

    expected = str(expected_registration)
    if not expected or preregistration.get("registration") != expected:
        raise ValueError("factorized capability preregistration differs")
    return expected


def run(args: argparse.Namespace) -> dict[str, Any]:
    prereg, prereg_sha256, prereg_path = load_json_object(
        args.preregistration, label="factorized capability preregistration"
    )
    experiment = _validate_registration_identity(
        prereg, getattr(args, "expected_registration", EXPERIMENT)
    )
    prereg_inputs = _mapping(prereg.get("inputs"), "preregistration inputs")
    field_authority = _mapping(prereg_inputs.get("field_checkpoint"), "field authority")
    radio_authority = _mapping(
        prereg_inputs.get("official_radio_checkpoint"), "RADIO authority"
    )
    expected_field_sha256 = str(field_authority.get("sha256", ""))
    expected_radio_sha256 = str(radio_authority.get("sha256", ""))
    expected_responsibility_sha256 = str(
        prereg_inputs.get("registration_responsibility_cache_sha256", "")
    )

    capability_path = Path(args.capability_cache).expanduser().resolve()
    capability_sha256 = sha256_file(capability_path)
    if capability_sha256 != str(args.expected_capability_cache_sha256):
        raise ValueError("materialized capability cache SHA-256 differs")
    bank = load_canonical_capability_bank(
        capability_path,
        expected_field_checkpoint_sha256=expected_field_sha256,
        require_row_authority=True,
        require_formal_projection_order=True,
    )
    metadata = bank.metadata
    if (
        metadata.get("field_checkpoint_schema_version") != 2
        or metadata.get("factorized_radio_field_signature_sha256")
        != field_authority.get("factorized_field_signature_sha256")
        or metadata.get("registration_responsibility_cache_sha256")
        != expected_responsibility_sha256
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("text_queries_opened") is not False
    ):
        raise ValueError("materialized capability factorized lineage differs")

    field_path = Path(args.field_checkpoint).expanduser().resolve()
    if sha256_file(field_path) != expected_field_sha256:
        raise ValueError("factorized field checkpoint SHA-256 differs")
    field, _payload, signature = load_factorized_canonical_field_checkpoint(
        field_path,
        expected_sha256=expected_field_sha256,
        map_location="cpu",
    )
    if signature.digest != metadata.get("factorized_radio_field_signature_sha256"):
        raise ValueError("factorized field signature differs from capability bank")

    radio_path = Path(args.radio_checkpoint).expanduser().resolve()
    if sha256_file(radio_path) != expected_radio_sha256:
        raise ValueError("official RADIO checkpoint SHA-256 differs")
    device = torch.device(args.device)
    global_rows = bank.global_rows
    stored_banks = {
        "dino_v3": bank.appearance,
        "sam3": bank.boundary,
    }
    parity: dict[str, dict[str, Any]] = {}
    for space, (adaptor_name, _bank_name, _dimension) in CAPABILITIES.items():
        adaptor = load_radio_adaptor_from_checkpoint(
            radio_path,
            adaptor_name,
            kind="feature_projection",
            expected_sha256=expected_radio_sha256,
        )
        parity[space] = audit_full_fp16_parity(
            field=field,
            adaptor=adaptor,
            global_rows=global_rows,
            stored=stored_banks[space],
            device=device,
            batch_size=int(args.batch_size),
        )
        del adaptor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    exact_caches: dict[str, Mapping[str, Any] | ShardedMPRCache] = {}
    exact_records: dict[str, dict[str, str]] = {}
    geometry = _mapping(metadata.get("mpr_geometry_fingerprint"), "geometry")
    for space, expected_sha256 in (
        ("dino_v3", str(args.expected_dino_mpr_sha256)),
        ("sam3", str(args.expected_sam3_mpr_sha256)),
    ):
        path = Path(
            args.dino_mpr_cache if space == "dino_v3" else args.sam3_mpr_cache
        ).expanduser().resolve()
        cache, digest, source = load_mpr_cache(
            path,
            expected_sha256=expected_sha256,
            expected_feature_space=space,
            require_reliability=True,
            require_formal_safety=False,
        )
        _validate_mpr_lineage(
            cache=cache,
            expected_space=space,
            expected_geometry=geometry,
            expected_responsibility_sha256=expected_responsibility_sha256,
            expected_radio_checkpoint_sha256=expected_radio_sha256,
        )
        exact_caches[space] = cache
        exact_records[space] = {"path": str(source), "sha256": digest}

    selected_rows, common_valid_rows = select_common_valid_rows(
        [bank.valid, *(_cache_valid(exact_caches[name]) for name in CAPABILITIES)],
        maximum_rows=int(args.maximum_rows),
    )
    compact_indices = torch.searchsorted(global_rows, selected_rows)
    if not torch.equal(global_rows[compact_indices], selected_rows):
        raise ValueError("selected source rows are absent from compact capability bank")
    comparisons: dict[str, dict[str, Any]] = {}
    for space in CAPABILITIES:
        comparisons[space] = compare_materialized_rows(
            stored_banks[space][compact_indices],
            _feature_rows(exact_caches[space], selected_rows),
            batch_size=int(args.metric_batch_size),
        )

    threshold = float(
        _mapping(prereg.get("promotion_gate"), "promotion gate")[
            "dino_centered_row_variation_ratio_to_exact_minimum"
        ]
    )
    sam_threshold = float(
        _mapping(prereg.get("promotion_gate"), "promotion gate")[
            "sam3_centered_row_variation_ratio_to_exact_minimum"
        ]
    )
    checks = {
        "dino_full_fp16_parity": parity["dino_v3"]["exact_fp16_parity"],
        "sam3_full_fp16_parity": parity["sam3"]["exact_fp16_parity"],
        "dino_centered_variation": comparisons["dino_v3"][
            "centered_row_variation_ratio_to_exact"
        ]
        >= threshold,
        "sam3_centered_variation": comparisons["sam3"][
            "centered_row_variation_ratio_to_exact"
        ]
        >= sam_threshold,
        "strict_factorized_lineage": True,
        "formal_projection_contract": True,
        "source_only": True,
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1,
        "experiment": experiment,
        "status": "pass" if passed else "fail_closed",
        "preregistration": {"path": str(prereg_path), "sha256": prereg_sha256},
        "inputs": {
            "capability_cache": {
                "path": str(capability_path),
                "sha256": capability_sha256,
            },
            "field_checkpoint": {
                "path": str(field_path),
                "sha256": expected_field_sha256,
            },
            "official_radio_checkpoint": {
                "path": str(radio_path),
                "sha256": expected_radio_sha256,
            },
            "exact_source_capabilities": exact_records,
        },
        "sampling": {
            "rule": "first_rows_after_ascending_common_valid_global_row_order",
            "common_valid_rows": common_valid_rows,
            "selected_rows": int(selected_rows.numel()),
            "first_global_row": int(selected_rows[0]),
            "last_global_row": int(selected_rows[-1]),
        },
        "full_fp16_parity": parity,
        "source_capability_comparison": comparisons,
        "promotion_gate": {
            "passed": passed,
            "checks": checks,
            "decision": (
                "freeze_capability_sha_and_allow_graph_free_source_sentinel"
                if passed
                else "do_not_enter_graph_or_benchmark"
            ),
        },
        "target_access": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_metric_computed": False,
        },
    }
    write_frozen_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-registration", default=EXPERIMENT)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--expected-capability-cache-sha256", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--dino-mpr-cache", required=True)
    parser.add_argument("--expected-dino-mpr-sha256", required=True)
    parser.add_argument("--sam3-mpr-cache", required=True)
    parser.add_argument("--expected-sam3-mpr-sha256", required=True)
    parser.add_argument("--maximum-rows", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--metric-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.maximum_rows <= 16384:
        parser.error("--maximum-rows must be in [1,16384]")
    if args.batch_size <= 0 or args.metric_batch_size <= 0:
        parser.error("batch sizes must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
