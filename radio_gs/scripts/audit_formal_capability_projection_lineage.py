#!/usr/bin/env python3
"""Audit formal capability projection lineage without opening benchmark labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.interfaces.capability_projection_contract import (
    CANONICAL_FIELD_CAPABILITY_SOURCE,
    FORMAL_TARGET_MODES,
    LEGACY_MATCHED_TOP1_CONTRACT,
    LEGACY_PROJECTION_AUTHORITY_CONTRACT,
    sha256_file,
)


def _split_binding(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError("bindings must be LABEL=/absolute/path")
    return label.strip(), Path(path).expanduser().resolve()


def _json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _capability_entry(benchmark: str, sidecar: Path) -> dict[str, Any]:
    cache_metadata = _json_object(sidecar)
    cache_path = Path(str(cache_metadata.get("output", ""))).expanduser().resolve()
    if cache_path.with_suffix(cache_path.suffix + ".json") != sidecar.resolve():
        raise ValueError(f"capability output/sidecar path mismatch: {sidecar}")
    if cache_metadata.get("source") != CANONICAL_FIELD_CAPABILITY_SOURCE:
        raise ValueError(f"canonical capability source differs: {sidecar}")
    field_path = Path(str(cache_metadata.get("field_checkpoint", ""))).resolve()
    field_sidecar = field_path.with_suffix(field_path.suffix + ".json")
    if not cache_path.is_file() or not field_path.is_file() or not field_sidecar.is_file():
        raise FileNotFoundError(f"capability lineage artifact is absent: {sidecar}")
    field_report = _json_object(field_sidecar)
    target_mode = str(field_report.get("capability_target_mode", ""))
    raw_targets = field_report.get("capability_mpr_targets")
    if not isinstance(raw_targets, Mapping):
        raw_targets = {}
    teacher_orders = {
        role: str(
            raw_targets.get(target_name, {}).get("projection_order", "")
            if isinstance(raw_targets.get(target_name), Mapping)
            else ""
        )
        for target_name, role in (("dino_v3", "appearance"), ("sam3", "boundary"))
    }
    field_sha256 = sha256_file(field_path)
    declared_field_sha256 = str(cache_metadata.get("field_checkpoint_sha256", ""))
    digest_matches = field_sha256 == declared_field_sha256
    query_independent = all(
        isinstance(raw_targets.get(name), Mapping)
        and raw_targets[name].get("uses_query_or_benchmark_supervision") is False
        for name in ("dino_v3", "sam3")
    )
    formal_eligible = (
        digest_matches
        and target_mode in FORMAL_TARGET_MODES
        and set(teacher_orders.values()).issubset(FORMAL_TARGET_MODES)
        and len(teacher_orders) == 2
        and query_independent
    )
    return {
        "benchmark": benchmark,
        "scene": sidecar.parent.name,
        "capability_cache": str(cache_path),
        "capability_cache_bytes": cache_path.stat().st_size,
        "capability_cache_sidecar": str(sidecar.resolve()),
        "capability_cache_sidecar_sha256": sha256_file(sidecar),
        "source": CANONICAL_FIELD_CAPABILITY_SOURCE,
        "field_checkpoint": str(field_path),
        "field_checkpoint_sha256": field_sha256,
        "field_digest_matches_capability_metadata": digest_matches,
        "field_checkpoint_sidecar": str(field_sidecar),
        "field_checkpoint_sidecar_sha256": sha256_file(field_sidecar),
        "capability_target_mode": target_mode,
        "capability_target_contract": str(
            field_report.get(
                "capability_target_contract", LEGACY_MATCHED_TOP1_CONTRACT
            )
        ),
        "teacher_projection_orders": teacher_orders,
        "nonlinear_adaptor_after_raw_mpr": False,
        "query_independent_teacher": query_independent,
        "inline_cache_projection_contract_present": isinstance(
            cache_metadata.get("capability_projection_contract"), Mapping
        ),
        "formal_one_field_eligible": formal_eligible,
        "authority_scope": "exact_cache_path_and_sidecar_plus_field_digest",
    }


def _legacy_field_entry(label: str, path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"legacy field or sidecar is absent: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"legacy field payload must be a mapping: {path}")
    targets = payload.get("capability_mpr_targets")
    target_orders = {}
    if isinstance(targets, Mapping):
        target_orders = {
            name: str(value.get("projection_order", ""))
            for name, value in targets.items()
            if isinstance(value, Mapping)
        }
    return {
        "label": label,
        "field_checkpoint": str(path),
        "field_checkpoint_sha256": sha256_file(path),
        "field_checkpoint_sidecar": str(sidecar),
        "field_checkpoint_sidecar_sha256": sha256_file(sidecar),
        "capability_target_mode": payload.get("capability_target_mode"),
        "capability_target_contract": payload.get("capability_target_contract"),
        "teacher_projection_orders": target_orders,
        "mpr_cache": payload.get("mpr_cache"),
        "formal_one_field_eligible": False,
        "compatibility_class": "legacy_raw_mpr_compact_field_only",
        "reason": (
            "checkpoint does not publish an adaptor-before-MPR capability target "
            "contract and therefore cannot claim exact formal capability lineage"
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    capability_entries: list[dict[str, Any]] = []
    missing_scene_assets: list[dict[str, str]] = []
    for raw_binding in args.capability_root:
        benchmark, root = _split_binding(raw_binding)
        for scene in sorted(path for path in root.iterdir() if path.is_dir()):
            if scene.name == "queue_logs":
                continue
            sidecar = scene / "official_dino_sam3_views.pt.json"
            if not sidecar.is_file():
                missing_scene_assets.append(
                    {"benchmark": benchmark, "scene": scene.name, "root": str(root)}
                )
                continue
            capability_entries.append(_capability_entry(benchmark, sidecar))
    legacy_fields = [
        _legacy_field_entry(*_split_binding(value)) for value in args.legacy_field
    ]
    authority_entries = [
        {
            key: entry[key]
            for key in (
                "benchmark",
                "scene",
                "capability_cache",
                "capability_cache_sidecar_sha256",
                "source",
                "field_checkpoint",
                "field_checkpoint_sha256",
                "capability_target_mode",
                "capability_target_contract",
                "teacher_projection_orders",
                "nonlinear_adaptor_after_raw_mpr",
                "formal_one_field_eligible",
                "authority_scope",
            )
        }
        for entry in capability_entries
        if entry["formal_one_field_eligible"]
    ]
    report = {
        "schema_version": 1,
        "kind": "formal_capability_projection_lineage_closure",
        "status": (
            "audited_with_legacy_authority"
            if authority_entries
            else "no_formal_legacy_assets"
        ),
        "principle": {
            "formal_teacher": "official adaptor per view before MPR",
            "formal_compact_field": "supervised by that capability-first teacher",
            "diagnostic_only": "raw RADIO MPR then nonlinear official adaptor",
        },
        "capability_assets": capability_entries,
        "missing_scene_assets": missing_scene_assets,
        "legacy_fields": legacy_fields,
        "legacy_compatibility_authority": {
            "schema_version": 1,
            "contract": LEGACY_PROJECTION_AUTHORITY_CONTRACT,
            "entries": authority_entries,
        },
        "summary": {
            "audited_capability_caches": len(capability_entries),
            "formal_eligible_via_legacy_authority": len(authority_entries),
            "missing_scene_capability_caches": len(missing_scene_assets),
            "legacy_fields_without_formal_target_contract": len(legacy_fields),
        },
        "implementation_closure": {
            "formal_compact_cache": (
                "inline capability_projection_contract or exact legacy authority"
            ),
            "future_compact_builder": (
                "fails before GPU projection unless the field publishes a "
                "capability-first target mode, target contract, and both teacher orders"
            ),
            "exact_raw_mpr": "explicit projection-order diagnostic only",
            "exact_capability_mpr": (
                "formal exact teacher only with per_view_before_mpr authority"
            ),
            "formal_nvos_spin_evaluator": (
                "requires formal projection order and uses this legacy authority"
            ),
        },
        "remaining_scope": {
            "missing_spin_assets": missing_scene_assets,
            "lerf_legacy_field": (
                "remains usable only as a named legacy compact-field lineage; "
                "it cannot be relabeled as exact/formal capability without a new "
                "capability-first field"
            ),
            "large_asset_rebuild_performed": False,
        },
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "gpu_used": False,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capability-root",
        action="append",
        default=[],
        help="Repeat LABEL=/root/containing/scene/directories.",
    )
    parser.add_argument(
        "--legacy-field",
        action="append",
        default=[],
        help="Repeat LABEL=/absolute/field/checkpoint.pth.",
    )
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(parser.parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
