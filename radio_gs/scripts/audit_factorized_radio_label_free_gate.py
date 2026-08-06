#!/usr/bin/env python3
"""Run the preregistered, source-only factorized RADIO gauge gate.

The audit compares two nonlinear-adaptor inputs on the same primitive rows:
the legacy normalized raw MPR and ``canonical-factorized-radio-v1``.  Their
outputs are scored only against query-free, adaptor-before-MPR source caches.
No benchmark mask, text query, prompt, or target metric is accepted by this
entry point.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    validate_factorized_radio_builder_payload,
)
from radio_gs.training.tensor_cache_io import ShardedMPRCache, load_mpr_cache
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
)


EXPERIMENT = "canonical-factorized-radio-v1-label-free-source-gate"
TARGET_ACCESS_FALSE = {
    "benchmark_targets_opened": False,
    "benchmark_masks_opened": False,
    "text_queries_opened": False,
    "target_metrics_used_for_gate": False,
}
_CAPABILITY_SPECS = {
    "dino_v3": {"dimension": 4096, "adaptor": "dino_v3"},
    "sam3": {"dimension": 1024, "adaptor": "sam3"},
}


def _resolved(path: str | Path, *, repo_root: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else repo_root / value).resolve()


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _feature_rows(
    cache: Mapping[str, Any] | ShardedMPRCache,
    rows: torch.Tensor,
) -> torch.Tensor:
    if isinstance(cache, ShardedMPRCache):
        return cache.fetch_rows(rows)
    features = cache.get("features")
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError("MPR cache lacks a feature matrix")
    return features[rows]


def _cache_valid(cache: Mapping[str, Any] | ShardedMPRCache) -> torch.Tensor:
    valid = cache.get("valid")
    if not torch.is_tensor(valid) or valid.ndim != 1 or valid.dtype != torch.bool:
        raise ValueError("MPR cache validity differs")
    return valid.detach().cpu()


def _cache_metadata(
    cache: Mapping[str, Any] | ShardedMPRCache,
) -> Mapping[str, Any]:
    return _require_mapping(cache.get("metadata"), label="MPR metadata")


def select_common_valid_rows(
    masks: list[torch.Tensor], *, maximum_rows: int
) -> tuple[torch.Tensor, int]:
    if not masks or maximum_rows <= 0:
        raise ValueError("common-row selection needs masks and a positive limit")
    normalized = [torch.as_tensor(mask).detach().bool().cpu() for mask in masks]
    shape = normalized[0].shape
    if len(shape) != 1 or any(mask.shape != shape for mask in normalized):
        raise ValueError("common-row validity masks do not align")
    common = normalized[0].clone()
    for mask in normalized[1:]:
        common &= mask
    all_rows = torch.where(common)[0]
    if all_rows.numel() == 0:
        raise ValueError("common valid primitive set is empty")
    return all_rows[: int(maximum_rows)], int(all_rows.numel())


def summarize_norms(features: torch.Tensor) -> dict[str, float | int]:
    values = torch.linalg.vector_norm(
        torch.as_tensor(features).detach().float().cpu(), dim=-1
    )
    if values.ndim != 1 or values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("canonical norm population must be finite and nonempty")
    return {
        "rows": int(values.numel()),
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.5)),
        "p05": float(torch.quantile(values, 0.05)),
        "p95": float(torch.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


class _CenteredVariation:
    """Accumulate row-centered RMS without retaining projected feature rows."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("variation dimension must be positive")
        self.dimension = int(dimension)
        self.rows = 0
        self.column_sum = torch.zeros(self.dimension, dtype=torch.float64)
        self.square_sum = 0.0

    def update(self, values: torch.Tensor) -> None:
        matrix = torch.as_tensor(values).detach().double().cpu()
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError("variation rows have the wrong feature dimension")
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("variation rows must be finite")
        self.rows += int(matrix.shape[0])
        self.column_sum += matrix.sum(dim=0)
        self.square_sum += float(matrix.square().sum())

    def rms(self) -> float:
        if self.rows <= 0:
            raise ValueError("variation population is empty")
        centered_square_sum = self.square_sum - float(
            self.column_sum.square().sum() / float(self.rows)
        )
        centered_square_sum = max(centered_square_sum, 0.0)
        return math.sqrt(centered_square_sum / float(self.rows * self.dimension))


@torch.inference_mode()
def compare_capability_rows(
    *,
    adaptor: torch.nn.Module,
    legacy_radio: torch.Tensor,
    factorized_radio: torch.Tensor,
    exact_capability: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Compare official adaptor outputs with a query-free exact source teacher."""

    legacy = torch.as_tensor(legacy_radio).detach().cpu()
    factorized = torch.as_tensor(factorized_radio).detach().cpu()
    exact = torch.as_tensor(exact_capability).detach().cpu()
    if (
        legacy.ndim != 2
        or legacy.shape != factorized.shape
        or legacy.shape[1] != 1280
        or exact.ndim != 2
        or exact.shape[0] != legacy.shape[0]
        or legacy.shape[0] <= 0
        or batch_size <= 0
    ):
        raise ValueError("capability comparison rows do not align")
    dimension = int(exact.shape[1])
    adaptor = adaptor.to(device).eval().requires_grad_(False)
    legacy_cosine: list[torch.Tensor] = []
    factorized_cosine: list[torch.Tensor] = []
    variations = {
        "exact": _CenteredVariation(dimension),
        "legacy": _CenteredVariation(dimension),
        "factorized": _CenteredVariation(dimension),
    }
    for start in range(0, int(legacy.shape[0]), int(batch_size)):
        stop = min(start + int(batch_size), int(legacy.shape[0]))
        exact_rows = F.normalize(
            exact[start:stop].to(device=device, dtype=torch.float32),
            dim=-1,
            eps=1e-8,
        )
        legacy_rows = F.normalize(
            adaptor(
                legacy[start:stop].to(device=device, dtype=torch.float32)[None]
            )[0].float(),
            dim=-1,
            eps=1e-8,
        )
        factorized_rows = F.normalize(
            adaptor(
                factorized[start:stop].to(device=device, dtype=torch.float32)[None]
            )[0].float(),
            dim=-1,
            eps=1e-8,
        )
        legacy_cosine.append((legacy_rows * exact_rows).sum(dim=-1).cpu())
        factorized_cosine.append((factorized_rows * exact_rows).sum(dim=-1).cpu())
        variations["exact"].update(exact_rows)
        variations["legacy"].update(legacy_rows)
        variations["factorized"].update(factorized_rows)

    cosine = {
        "legacy": torch.cat(legacy_cosine).float(),
        "factorized": torch.cat(factorized_cosine).float(),
    }
    exact_rms = variations["exact"].rms()
    if exact_rms <= 0:
        raise ValueError("exact capability rows have zero centered variation")

    def summarize(name: str) -> dict[str, float]:
        values = cosine[name]
        rms = variations[name].rms()
        return {
            "cosine_mean": float(values.mean()),
            "cosine_p05": float(torch.quantile(values, 0.05)),
            "centered_row_variation_rms": float(rms),
            "centered_row_variation_ratio_to_exact": float(rms / exact_rms),
        }

    legacy_metrics = summarize("legacy")
    factorized_metrics = summarize("factorized")
    return {
        "rows": int(legacy.shape[0]),
        "dimension": dimension,
        "normalization": "independent_fp32_l2_for_prediction_and_exact_source_rows",
        "centered_variation_definition": (
            "sqrt(mean((row_l2_normalized_embedding - column_mean)^2))"
        ),
        "exact": {"centered_row_variation_rms": float(exact_rms)},
        "legacy": legacy_metrics,
        "factorized": factorized_metrics,
        "factorized_minus_legacy": {
            "cosine_mean": float(
                factorized_metrics["cosine_mean"] - legacy_metrics["cosine_mean"]
            ),
            "cosine_p05": float(
                factorized_metrics["cosine_p05"] - legacy_metrics["cosine_p05"]
            ),
        },
    }


def evaluate_promotion_gate(
    *,
    factorized_norms: Mapping[str, Any],
    capabilities: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    cache_and_lineage_invariants: bool,
) -> dict[str, Any]:
    required_capabilities = set(_CAPABILITY_SPECS)
    if set(capabilities) != required_capabilities:
        raise ValueError("promotion gate capability cohort differs")
    checks: dict[str, bool] = {
        "factorized_active_median_norm_lower": float(factorized_norms["median"])
        >= float(thresholds["factorized_active_median_norm_lower"]),
        "factorized_active_median_norm_upper": float(factorized_norms["median"])
        <= float(thresholds["factorized_active_median_norm_upper"]),
        "all_cache_and_lineage_invariants": bool(cache_and_lineage_invariants)
        and thresholds.get("all_cache_and_lineage_invariants") is True,
    }
    for capability, metrics in capabilities.items():
        prefix = "dino" if capability == "dino_v3" else "sam3"
        delta = _require_mapping(
            metrics.get("factorized_minus_legacy"), label=f"{capability} delta"
        )
        factorized = _require_mapping(
            metrics.get("factorized"), label=f"{capability} factorized metrics"
        )
        checks[f"{capability}_mean_cosine_delta"] = float(delta["cosine_mean"]) >= float(
            thresholds[f"{prefix}_mean_cosine_delta_vs_legacy_minimum"]
        )
        checks[f"{capability}_p05_cosine_delta"] = float(delta["cosine_p05"]) >= float(
            thresholds[f"{prefix}_p05_cosine_delta_vs_legacy_minimum"]
        )
        checks[f"{capability}_centered_row_variation_ratio"] = float(
            factorized["centered_row_variation_ratio_to_exact"]
        ) >= float(thresholds["centered_row_variation_ratio_to_exact_minimum"])
    passed = all(checks.values())
    return {
        "status": "pass" if passed else "fail_closed",
        "passed": passed,
        "checks": checks,
        "thresholds": dict(thresholds),
        "decision": (
            "proceed_to_factorized_field_training"
            if passed
            else "do_not_train_or_open_benchmark_targets"
        ),
    }


def _validate_mpr_lineage(
    *,
    cache: Mapping[str, Any] | ShardedMPRCache,
    expected_space: str,
    expected_geometry: Mapping[str, Any],
    expected_responsibility_sha256: str,
    expected_radio_checkpoint_sha256: str | None,
) -> None:
    metadata = _cache_metadata(cache)
    geometry = _require_mapping(
        cache.get("geometry_fingerprint"), label=f"{expected_space} geometry"
    )
    if dict(geometry) != dict(expected_geometry):
        raise ValueError(f"{expected_space} geometry lineage differs")
    for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened"):
        if metadata.get(key) is not False:
            raise ValueError(f"{expected_space} source safety declaration differs: {key}")
    if str(metadata.get("registration_responsibility_cache_sha256", "")) != str(
        expected_responsibility_sha256
    ):
        raise ValueError(f"{expected_space} responsibility lineage differs")
    lifting = _require_mapping(
        metadata.get("observation_lifting_contract"),
        label=f"{expected_space} observation lifting contract",
    )
    if (
        lifting.get("name") != "canonical-mpr-v1"
        or lifting.get("feature_projection_order") != "per_view_before_mpr"
        or lifting.get("query_independent") is not True
        or lifting.get("responsibility_sharing")
        != "exact_sidecar_across_feature_spaces"
    ):
        raise ValueError(f"{expected_space} observation lifting lineage differs")
    if expected_space in _CAPABILITY_SPECS:
        if (
            metadata.get("capability_projection_before_mpr") is not True
            or metadata.get("capability_map_source") != "project_raw"
            or metadata.get("shared_registration_responsibility") is not True
            or str(metadata.get("official_adaptor_checkpoint_sha256", ""))
            != str(expected_radio_checkpoint_sha256)
        ):
            raise ValueError(f"{expected_space} exact capability lineage differs")


def _implementation_records(repo_root: Path) -> dict[str, dict[str, str]]:
    relative_paths = (
        "radio_gs/scripts/audit_factorized_radio_label_free_gate.py",
        "radio_gs/scripts/build_gaussian_multiview_teacher_cache.py",
        "radio_gs/field/factorized_radio_contract.py",
        "radio_gs/interfaces/frozen_radio_views.py",
        "radio_gs/models/radio_adaptors.py",
        "radio_gs/training/tensor_cache_io.py",
    )
    return {
        Path(relative).name: {
            "path": relative,
            "sha256": sha256_file(repo_root / relative),
        }
        for relative in relative_paths
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    prereg, prereg_sha256, prereg_path = load_json_object(
        _resolved(args.preregistration, repo_root=repo_root),
        label="factorized label-free preregistration",
    )
    if (
        prereg.get("experiment") != EXPERIMENT
        or prereg.get("target_access") != TARGET_ACCESS_FALSE
    ):
        raise ValueError("label-free preregistration contract differs")
    correction, correction_sha256, correction_path = load_json_object(
        _resolved(args.legacy_receipt_correction, repo_root=repo_root),
        label="legacy receipt correction addendum",
    )
    if (
        correction.get("experiment") != EXPERIMENT
        or correction.get("scientific_contract_unchanged", {}).get("target_access")
        is not False
    ):
        raise ValueError("legacy receipt correction contract differs")
    corrected_authorities = _require_mapping(
        correction.get("frozen_cache_authorities"),
        label="corrected frozen cache authorities",
    )
    authorities = _require_mapping(
        prereg.get("source_authorities"), label="source authorities"
    )
    raw_authority = _require_mapping(
        authorities.get("raw_radio_control"), label="raw RADIO authority"
    )
    corrected_raw = _require_mapping(
        corrected_authorities.get("radio"), label="corrected raw authority"
    )
    if (
        _resolved(str(corrected_raw.get("path", "")), repo_root=repo_root)
        != _resolved(str(raw_authority.get("path", "")), repo_root=repo_root)
        or str(corrected_raw.get("sha256", ""))
        != str(raw_authority.get("sha256", ""))
    ):
        raise ValueError("corrected raw authority differs from preregistration")
    responsibility_authority = _require_mapping(
        authorities.get("responsibility_sidecar"), label="responsibility authority"
    )
    probe_authority = _require_mapping(
        authorities.get("raw_source_amplitude_probe"), label="amplitude probe authority"
    )
    factorized_path = _resolved(args.factorized_cache, repo_root=repo_root)
    factorized_expected = str(args.expected_factorized_cache_sha256)
    factorized, factorized_sha256, factorized_source = load_torch_mapping(
        factorized_path,
        expected_sha256=factorized_expected,
        map_location="cpu",
        label="factorized RADIO builder cache",
    )
    validate_factorized_radio_builder_payload(factorized)
    factorized_core = _require_mapping(
        factorized.get("factorized_radio"), label="factorized RADIO core"
    )
    factorized_valid = torch.as_tensor(factorized_core["valid"]).bool().cpu()
    factorized_canonical = torch.as_tensor(factorized_core["canonical_feature"])
    factorized_metadata = _require_mapping(
        factorized.get("metadata"), label="factorized RADIO metadata"
    )
    geometry = _require_mapping(
        factorized.get("geometry_fingerprint"), label="factorized geometry"
    )

    responsibility_path = _resolved(
        str(responsibility_authority["path"]), repo_root=repo_root
    )
    responsibility_sha256 = sha256_file(responsibility_path)
    if responsibility_sha256 != str(responsibility_authority["sha256"]):
        raise ValueError("responsibility-sidecar SHA-256 differs")
    if str(factorized_metadata.get("registration_responsibility_cache_sha256", "")) != (
        responsibility_sha256
    ):
        raise ValueError("factorized cache uses another responsibility sidecar")
    probe_path = _resolved(str(probe_authority["path"]), repo_root=repo_root)
    probe_sha256 = sha256_file(probe_path)
    if probe_sha256 != str(probe_authority["sha256"]):
        raise ValueError("raw source amplitude probe SHA-256 differs")

    raw_path = _resolved(str(raw_authority["path"]), repo_root=repo_root)
    raw, raw_sha256, raw_source = load_mpr_cache(
        raw_path,
        expected_sha256=str(raw_authority["sha256"]),
        expected_feature_space="radio",
        require_reliability=True,
        # This preregistered July dense control predates the later feature-
        # bundle receipt.  Its immutable SHA is the source authority here;
        # the source-only flags, geometry, and responsibility lineage remain
        # mandatory immediately below.
        require_formal_safety=False,
    )
    if raw_sha256 != str(raw_authority["sha256"]):
        raise ValueError("legacy raw MPR digest differs")
    _validate_mpr_lineage(
        cache=raw,
        expected_space="radio",
        expected_geometry=geometry,
        expected_responsibility_sha256=responsibility_sha256,
        expected_radio_checkpoint_sha256=None,
    )

    radio_checkpoint = _resolved(args.radio_checkpoint, repo_root=repo_root)
    radio_checkpoint_sha256 = sha256_file(radio_checkpoint)
    if radio_checkpoint_sha256 != str(args.expected_radio_checkpoint_sha256):
        raise ValueError("official RADIO checkpoint SHA-256 differs")

    capability_caches: dict[str, Mapping[str, Any] | ShardedMPRCache] = {}
    capability_records: dict[str, dict[str, str]] = {}
    for space, authority_key in (("dino_v3", "exact_dino_mpr"), ("sam3", "exact_sam3_mpr")):
        corrected = _require_mapping(
            corrected_authorities.get(space), label=f"{space} corrected authority"
        )
        corrected_path = _resolved(str(corrected.get("path", "")), repo_root=repo_root)
        prereg_path_value = _resolved(
            str(authorities[authority_key]), repo_root=repo_root
        )
        if corrected_path != prereg_path_value:
            raise ValueError(f"{space} corrected authority path differs from preregistration")
        cache, digest, source = load_mpr_cache(
            prereg_path_value,
            expected_sha256=str(corrected.get("sha256", "")),
            expected_feature_space=space,
            require_reliability=True,
            # These two preregistered July caches predate the later feature-
            # bundle receipt as well.  The pre-metric correction addendum
            # freezes both complete-file digests, while the exact projection
            # lineage is checked field by field below.
            require_formal_safety=False,
        )
        _validate_mpr_lineage(
            cache=cache,
            expected_space=space,
            expected_geometry=geometry,
            expected_responsibility_sha256=responsibility_sha256,
            expected_radio_checkpoint_sha256=radio_checkpoint_sha256,
        )
        capability_caches[space] = cache
        capability_records[space] = {"path": str(source), "sha256": digest}

    selected_rows, common_valid_count = select_common_valid_rows(
        [
            factorized_valid,
            _cache_valid(raw),
            _cache_valid(capability_caches["dino_v3"]),
            _cache_valid(capability_caches["sam3"]),
        ],
        maximum_rows=int(args.maximum_rows),
    )
    factorized_selected = factorized_canonical[selected_rows].clone()
    legacy_selected = _feature_rows(raw, selected_rows).clone()
    exact_selected = {
        space: _feature_rows(cache, selected_rows).clone()
        for space, cache in capability_caches.items()
    }
    selected_norms = summarize_norms(factorized_selected)
    legacy_norms = summarize_norms(legacy_selected)
    full_active_norms = summarize_norms(factorized_canonical[factorized_valid])

    del factorized, factorized_core, factorized_canonical, raw, capability_caches
    gc.collect()

    capabilities: dict[str, dict[str, Any]] = {}
    for space, spec in _CAPABILITY_SPECS.items():
        adaptor = load_radio_adaptor_from_checkpoint(
            radio_checkpoint,
            str(spec["adaptor"]),
            kind="feature_projection",
            expected_sha256=radio_checkpoint_sha256,
        )
        capabilities[space] = compare_capability_rows(
            adaptor=adaptor,
            legacy_radio=legacy_selected,
            factorized_radio=factorized_selected,
            exact_capability=exact_selected.pop(space),
            device=torch.device(args.device),
            batch_size=int(args.batch_size),
        )
        del adaptor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    thresholds = _require_mapping(
        prereg.get("promotion_gates"), label="promotion thresholds"
    )
    gate = evaluate_promotion_gate(
        factorized_norms=selected_norms,
        capabilities=capabilities,
        thresholds=thresholds,
        cache_and_lineage_invariants=True,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": gate["status"],
        "scene": prereg.get("development_scene"),
        "source_only": True,
        "preregistration": {
            "path": str(prereg_path),
            "sha256": prereg_sha256,
            "status": prereg.get("status"),
        },
        "legacy_receipt_correction": {
            "path": str(correction_path),
            "sha256": correction_sha256,
        },
        "inputs": {
            "factorized_radio": {
                "path": str(factorized_source),
                "sha256": factorized_sha256,
                "builder_envelope_parser_validated": True,
            },
            "legacy_raw_radio": {"path": str(raw_source), "sha256": raw_sha256},
            "exact_capability_mpr": capability_records,
            "responsibility_sidecar": {
                "path": str(responsibility_path),
                "sha256": responsibility_sha256,
            },
            "raw_source_amplitude_probe": {
                "path": str(probe_path),
                "sha256": probe_sha256,
            },
            "official_radio_checkpoint": {
                "path": str(radio_checkpoint),
                "sha256": radio_checkpoint_sha256,
            },
            "geometry_fingerprint": dict(geometry),
        },
        "implementation": _implementation_records(repo_root),
        "sampling": {
            "rule": "first_rows_after_ascending_common_valid_global_row_order",
            "maximum_rows": int(args.maximum_rows),
            "common_valid_rows": common_valid_count,
            "selected_rows": int(selected_rows.numel()),
            "first_global_row": int(selected_rows[0]),
            "last_global_row": int(selected_rows[-1]),
        },
        "input_norms": {
            "legacy_selected_common_rows": legacy_norms,
            "factorized_selected_common_rows": selected_norms,
            "factorized_all_active_rows": full_active_norms,
        },
        "capabilities": capabilities,
        "promotion_gate": gate,
        "target_access": dict(TARGET_ACCESS_FALSE),
        "execution": {
            "device": str(args.device),
            "projection_batch_size": int(args.batch_size),
            "field_trained": False,
            "benchmark_evaluated": False,
        },
    }
    output = _resolved(args.output, repo_root=repo_root)
    write_frozen_json(output, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--preregistration",
        default="paper/artifacts/canonical_factorized_radio_label_free_gate_preregistration_20260805.json",
    )
    parser.add_argument(
        "--legacy-receipt-correction",
        default="paper/artifacts/canonical_factorized_radio_label_free_gate_legacy_receipt_correction_20260805.json",
    )
    parser.add_argument("--factorized-cache", required=True)
    parser.add_argument("--expected-factorized-cache-sha256", required=True)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--expected-radio-checkpoint-sha256", required=True)
    parser.add_argument("--maximum-rows", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.maximum_rows <= 0 or args.maximum_rows > 16384:
        parser.error("--maximum-rows must be in [1,16384]")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
