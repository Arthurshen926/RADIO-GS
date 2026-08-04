#!/usr/bin/env python3
"""Materialize a two-mode Field-D carrier as standard canonical checkpoints.

The frozen downstream stack consumes one canonical field at a time.  This
adapter therefore writes two schema-v1 direct fields with a shared decoder and
row authority.  Unsupported rows use the same exact-center/base fallback in
both modes; supported rows receive the two query-free directional prototypes.
No benchmark image, mask, query, or metric is opened.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.field import AffineBasisDecoder, load_canonical_field_checkpoint
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_torch_mapping, sha256_file


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable directional field already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable directional receipt already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_metadata(metadata: Mapping[str, Any], *, label: str) -> None:
    for key in (
        "benchmark_images_opened",
        "benchmark_masks_opened",
        "text_queries_opened",
    ):
        if metadata.get(key) is not False:
            raise ValueError(f"{label} does not explicitly deny {key}")


def _direct_payload(
    *,
    base_payload: Mapping[str, Any],
    decoder_state: Mapping[str, torch.Tensor],
    coefficients: torch.Tensor,
    reliability: torch.Tensor,
    raw: Mapping[str, Any],
    raw_path: Path,
    raw_sha256: str,
    mode_index: int,
    provenance: Mapping[str, Any],
    reconstruction_metrics: Mapping[str, float],
) -> dict[str, Any]:
    feature_dim, coefficient_dim = decoder_state["basis"].shape
    num_gaussians = int(coefficients.shape[0])
    if coefficients.shape != (num_gaussians, coefficient_dim):
        raise ValueError("directional direct coefficients have the wrong shape")
    if reliability.ndim != 2 or reliability.shape[0] != num_gaussians:
        raise ValueError("directional reliability is not row aligned")
    state_dict = {
        "local_codes": coefficients.float().contiguous(),
        "reliability": reliability.float().contiguous(),
        "decoder.basis": decoder_state["basis"].float().contiguous(),
        "decoder.mean": decoder_state["mean"].float().contiguous(),
        "decoder.log_scale": decoder_state["log_scale"].float().contiguous(),
    }
    return {
        "schema_version": 1,
        "architecture": {
            "num_gaussians": num_gaussians,
            "feature_dim": int(feature_dim),
            "coefficient_dim": int(coefficient_dim),
            "local_dim": int(coefficient_dim),
            "coarse_dim": 0,
            "spatial_hash": None,
            "position_storage": "none",
            "fusion_reliability": False,
            "hidden_dim": 192,
            "fusion_residual_blocks": 0,
            "use_fusion": False,
            "trainable_basis": False,
            "trainable_statistics": False,
        },
        "feature_signature": dict(base_payload["feature_signature"]),
        "state_dict": state_dict,
        # Keep the redundant reliability authority exactly representable in
        # float32 so the fail-closed canonical loader can compare both copies.
        "reliability": reliability.float().contiguous(),
        "geometry_fingerprint": dict(raw["geometry_fingerprint"]),
        "mpr_cache": str(raw_path),
        "mpr_cache_sha256": raw_sha256,
        "mpr_cache_storage": raw.get("storage_provenance", {}),
        "mpr_cache_metadata": dict(raw["metadata"]),
        "directional_mode": {
            "contract": "field_d_standard_mode_checkpoint_v1",
            "mode_index": int(mode_index),
            "unsupported_row_policy": (
                "shared_base_field_center"
                if provenance.get("feature_gauge")
                == "per_primitive_base_field_l2_norm"
                else "shared_exact_center_then_base_fallback"
            ),
            "query_pooling": "maximum_after_frozen_per_mode_readout",
            **dict(provenance),
        },
        "final_metrics": dict(reconstruction_metrics),
        "capability_target_mode": "query_free_two_directional_prototypes",
        "capability_target_contract": "weighted_spherical_two_prototype_v1",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }


@torch.no_grad()
def build(args: argparse.Namespace) -> dict[str, Any]:
    base, base_payload = load_canonical_field_checkpoint(
        args.base_field_checkpoint,
        expected_sha256=args.expected_base_field_checkpoint_sha256,
        map_location="cpu",
    )
    raw, raw_sha, raw_path = load_mpr_cache(
        args.raw_mpr_cache,
        expected_sha256=args.expected_raw_mpr_cache_sha256,
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=True,
    )
    compact, compact_sha, compact_path = load_torch_mapping(
        args.compact_prototype_cache,
        expected_sha256=args.expected_compact_prototype_cache_sha256,
        map_location="cpu",
        label="compact Field-D prototype cache",
    )
    joint, joint_sha, joint_path = load_torch_mapping(
        args.joint_basis_checkpoint,
        expected_sha256=args.expected_joint_basis_checkpoint_sha256,
        map_location="cpu",
        label="Field-D joint basis",
    )
    if compact.get("contract") != "compact_directional_prototype_field_v1":
        raise ValueError("compact Field-D prototype contract differs")
    joint_contract = joint.get("contract")
    if joint_contract not in {
        "field_d_joint_center_directional_basis_v1",
        "field_d_gauge_preserving_joint_basis_v1",
    }:
        raise ValueError("Field-D joint basis contract differs")
    gauge_preserving = joint_contract == "field_d_gauge_preserving_joint_basis_v1"
    geometry = dict(raw["geometry_fingerprint"])
    if any(
        item.get("geometry_fingerprint") != geometry
        for item in (base_payload, compact, joint)
    ):
        raise ValueError("Field-D carrier inputs use different geometry authorities")
    for metadata, label in (
        (base_payload, "base field"),
        (raw["metadata"], "exact-center MPR"),
        (compact["metadata"], "compact prototypes"),
        (joint["metadata"], "joint basis"),
    ):
        _safe_metadata(metadata, label=label)
    if dict(compact["metadata"]).get("basis_authority") != {
        "path": str(joint_path),
        "sha256": joint_sha,
    }:
        raise ValueError("compact prototypes are encoded in another basis")
    if dict(joint["metadata"]).get("raw_mpr_sha256") != raw_sha:
        raise ValueError("joint basis belongs to another exact-center MPR")

    architecture = dict(joint["architecture"])
    decoder_state = {
        key: torch.as_tensor(value).float().cpu()
        for key, value in dict(joint["decoder_state_dict"]).items()
    }
    decoder = AffineBasisDecoder(
        feature_dim=int(architecture["feature_dim"]),
        coefficient_dim=int(architecture["coefficient_dim"]),
        mean=decoder_state["mean"],
        scale=decoder_state["log_scale"].exp(),
        basis=decoder_state["basis"],
        trainable_basis=False,
    )
    num_gaussians = int(geometry["num_gaussians"])
    raw_features = torch.as_tensor(raw["features"])
    raw_valid = torch.as_tensor(raw["valid"]).bool().cpu()
    if raw_features.shape != (num_gaussians, decoder.feature_dim):
        raise ValueError("exact-center MPR feature rows differ")
    rows = torch.as_tensor(compact["global_rows"]).long().cpu()
    compact_coefficients = torch.as_tensor(compact["coefficients"]).float().cpu()
    if compact_coefficients.shape != (rows.numel(), 2, decoder.coefficient_dim):
        raise ValueError("compact prototype coefficients differ")
    if rows.numel() and (
        int(rows.min()) < 0
        or int(rows.max()) >= num_gaussians
        or not bool((rows[1:] > rows[:-1]).all())
    ):
        raise ValueError("compact prototype row authority differs")

    device = torch.device(args.device)
    base = base.to(device).eval()
    decoder = decoder.to(device).eval()
    center_coefficients = torch.empty(
        num_gaussians, decoder.coefficient_dim, dtype=torch.float32
    )
    cosine_parts: list[torch.Tensor] = []
    valid_cosine_parts: list[torch.Tensor] = []
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, num_gaussians, batch_size):
        stop = min(num_gaussians, start + batch_size)
        batch_rows = torch.arange(start, stop, device=device)
        base_target = base.radio_features(batch_rows).float()
        batch_valid = raw_valid[start:stop]
        target = base_target
        if bool(batch_valid.any()) and not gauge_preserving:
            target = target.clone()
            target[batch_valid.to(device)] = raw_features[start:stop][
                batch_valid
            ].to(device=device, dtype=torch.float32)
        encoded = decoder.encode(target)
        reconstructed = decoder(encoded)
        cosine = F.cosine_similarity(reconstructed, target, dim=-1, eps=1e-8)
        center_coefficients[start:stop] = encoded.float().cpu()
        cosine_parts.append(cosine.float().cpu())
        if bool(batch_valid.any()):
            valid_cosine_parts.append(cosine[batch_valid.to(device)].float().cpu())
    cosine = torch.cat(cosine_parts)
    valid_cosine = torch.cat(valid_cosine_parts)
    reconstruction_metrics = {
        "all_rows_mean_cosine": float(cosine.mean()),
        "all_rows_p05_cosine": float(torch.quantile(cosine, 0.05)),
        "carrier_valid_mean_cosine": float(valid_cosine.mean()),
        "carrier_valid_p05_cosine": float(torch.quantile(valid_cosine, 0.05)),
    }
    minimum_mean = float(args.minimum_mean_cosine)
    minimum_p05 = float(args.minimum_p05_cosine)
    gate = {
        "carrier_center_mean_cosine": (
            reconstruction_metrics["carrier_valid_mean_cosine"] >= minimum_mean
        ),
        "carrier_center_p05_cosine": (
            reconstruction_metrics["carrier_valid_p05_cosine"] >= minimum_p05
        ),
        "prototype_mean_cosine": float(
            compact["metadata"]["compression_metrics"]["mean_cosine"]
        ) >= minimum_mean,
        "prototype_p05_cosine": float(
            compact["metadata"]["compression_metrics"]["p05_cosine"]
        ) >= minimum_p05,
    }
    if not all(gate.values()):
        raise RuntimeError(f"Field-D standard carrier failed label-free gate: {gate}")

    reliability = torch.as_tensor(base_payload["reliability"]).float().cpu()
    mode_coefficients = []
    for mode_index in range(2):
        values = center_coefficients.clone()
        values[rows] = compact_coefficients[:, mode_index]
        mode_coefficients.append(values)
    provenance = {
        "base_field_checkpoint": {
            "path": str(Path(args.base_field_checkpoint).expanduser().resolve()),
            "sha256": args.expected_base_field_checkpoint_sha256,
        },
        "compact_prototype_cache": {
            "path": str(compact_path),
            "sha256": compact_sha,
        },
        "joint_basis_checkpoint": {"path": str(joint_path), "sha256": joint_sha},
        "directional_rows": int(rows.numel()),
        "total_rows": num_gaussians,
        "feature_gauge": dict(compact["metadata"]).get(
            "feature_gauge", "source_prototype_native"
        ),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    outputs = [
        Path(args.output_mode_0).expanduser().resolve(),
        Path(args.output_mode_1).expanduser().resolve(),
    ]
    mode_records = []
    for mode_index, (output, coefficients) in enumerate(
        zip(outputs, mode_coefficients)
    ):
        payload = _direct_payload(
            base_payload=base_payload,
            decoder_state=decoder_state,
            coefficients=coefficients,
            reliability=reliability,
            raw=raw,
            raw_path=raw_path,
            raw_sha256=raw_sha,
            mode_index=mode_index,
            provenance=provenance,
            reconstruction_metrics=reconstruction_metrics,
        )
        _atomic_torch(output, payload)
        mode_records.append({"path": str(output), "sha256": sha256_file(output)})

    bundle = Path(args.output_bundle).expanduser().resolve()
    bundle_payload = {
        "schema_version": "field_d_standard_mode_bundle_v1",
        "contract": "query_free_two_mode_field_then_frozen_readout_max_v1",
        "modes": mode_records,
        "geometry_fingerprint": geometry,
        "directional_rows": int(rows.numel()),
        "total_rows": num_gaussians,
        "pooling": "elementwise_max_raw_cosine_after_identical_frozen_readout",
        "reconstruction_metrics": reconstruction_metrics,
        "gate": gate,
        "passed": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(bundle, bundle_payload)
    report = {
        **bundle_payload,
        "bundle": {"path": str(bundle), "sha256": sha256_file(bundle)},
    }
    report_path = bundle.with_suffix(bundle.suffix + ".receipt.json")
    _atomic_json(report_path, report)
    return {**report, "receipt": str(report_path), "receipt_sha256": sha256_file(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-field-checkpoint", required=True)
    parser.add_argument("--expected-base-field-checkpoint-sha256", required=True)
    parser.add_argument("--raw-mpr-cache", required=True)
    parser.add_argument("--expected-raw-mpr-cache-sha256", required=True)
    parser.add_argument("--compact-prototype-cache", required=True)
    parser.add_argument("--expected-compact-prototype-cache-sha256", required=True)
    parser.add_argument("--joint-basis-checkpoint", required=True)
    parser.add_argument("--expected-joint-basis-checkpoint-sha256", required=True)
    parser.add_argument("--output-mode-0", required=True)
    parser.add_argument("--output-mode-1", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--minimum-mean-cosine", type=float, default=0.95)
    parser.add_argument("--minimum-p05-cosine", type=float, default=0.90)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
