#!/usr/bin/env python3
"""Bind the complete NVOS Beta-v2 reliability sidecar cohort.

This authority is intentionally independent of the Beta scorer and evaluator.
It opens no benchmark prompt, query, RGB, mask, or metric artifact.  Instead it
binds the fixed eight-scene cohort to the exact canonical-field and MPR source
bytes used by the query-independent reliability builder, and to the resulting
``.pt``/``.json`` sidecars.  Publication is immutable and no-clobber.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    stable_descriptor_load,
    write_frozen_json,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "nvos-beta-v2-query-independent-reliability-manifest-v1"
ORDERED_SCENES = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)
FIELD_NAME = "canonical_d256_l128_capability_first.pth"
MPR_NAME = "raw_radio.pt"
CACHE_NAME = "canonical_reliability.pt"
REPORT_NAME = "canonical_reliability.pt.json"
RELIABILITY_SOURCE = "canonical_primitive_reliability_v1"
FORMULA = (
    "confidence=((n/(n+1))*mpr_agreement*"
    "clamp(cos(compact_radio,mpr_radio),0,1))^(1/3)"
)
SAFETY_CONTRACT = {
    "query_independent": True,
    "uses_query": False,
    "uses_text": False,
    "uses_target_labels": False,
    "uses_target_masks": False,
    "uses_metric_feedback": False,
    "benchmark_masks_opened": False,
    "text_queries_opened": False,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ReliabilityManifestError(ValueError):
    """Raised when reliability provenance or immutable bytes drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReliabilityManifestError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReliabilityManifestError(f"{label} must be a mapping")
    return value


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _file_record(path: str | Path, *, label: str) -> dict[str, object]:
    size, digest, source = stable_descriptor_load(
        path,
        lambda handle: int(os.fstat(handle.fileno()).st_size),
        label=label,
    )
    return {"path": str(source), "bytes": size, "sha256": digest}


def _validate_file_record(record: object, *, label: str) -> Path:
    value = _mapping(record, f"{label} file record")
    _require(
        set(value) == {"path", "bytes", "sha256"},
        f"{label} file-record fields differ",
    )
    size = value.get("bytes")
    digest = value.get("sha256")
    _require(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0,
        f"{label} byte count differs",
    )
    _require(
        isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
        f"{label} SHA-256 differs",
    )
    observed = _file_record(str(value.get("path", "")), label=label)
    _require(observed == dict(value), f"{label} immutable bytes differ")
    return Path(str(observed["path"]))


def _xyz_sha256(values: torch.Tensor) -> str:
    array = (
        torch.as_tensor(values)
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_metadata(
    metadata: object,
    *,
    scene: str,
    field_record: Mapping[str, Any],
    mpr_record: Mapping[str, Any],
    num_gaussians: int,
    geometry_xyz_sha256: str,
) -> dict[str, Any]:
    value = dict(_mapping(metadata, f"{scene}: reliability metadata"))
    _require(value.get("schema_version") == 1, f"{scene}: metadata schema differs")
    _require(value.get("source") == RELIABILITY_SOURCE, f"{scene}: source differs")
    _require(value.get("formula") == FORMULA, f"{scene}: formula differs")
    _require(
        value.get("observation_prior_count") == 1
        and value.get("combination") == "equal_weight_geometric_mean"
        and value.get("third_mpr_reliability_channel_used") is False,
        f"{scene}: reliability construction differs",
    )
    for key, expected in SAFETY_CONTRACT.items():
        _require(value.get(key) is expected, f"{scene}: safety flag {key} differs")
    _require(
        _absolute(str(value.get("field_checkpoint", "")))
        == Path(str(field_record["path"]))
        and value.get("field_checkpoint_sha256") == field_record["sha256"],
        f"{scene}: canonical-field source binding differs",
    )
    _require(
        _absolute(str(value.get("mpr_cache", ""))) == Path(str(mpr_record["path"]))
        and value.get("mpr_cache_sha256") == mpr_record["sha256"],
        f"{scene}: MPR source binding differs",
    )
    _require(
        isinstance(value.get("mpr_construction"), str)
        and bool(value["mpr_construction"]),
        f"{scene}: MPR construction is absent",
    )
    geometry = _mapping(value.get("geometry_fingerprint"), f"{scene}: geometry")
    _require(
        geometry
        == {
            "num_gaussians": num_gaussians,
            "xyz_sha256": geometry_xyz_sha256,
        },
        f"{scene}: geometry fingerprint differs",
    )
    return value


def _validate_cache_and_report(
    *,
    scene: str,
    cache_record: Mapping[str, Any],
    report_record: Mapping[str, Any],
    field_record: Mapping[str, Any],
    mpr_record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_path = _validate_file_record(cache_record, label=f"{scene}: reliability cache")
    report_path = _validate_file_record(report_record, label=f"{scene}: build report")
    cache, cache_sha256, loaded_cache_path = load_torch_mapping(
        cache_path,
        expected_sha256=str(cache_record["sha256"]),
        label=f"{scene}: reliability cache payload",
    )
    _require(
        cache_sha256 == cache_record["sha256"] and loaded_cache_path == cache_path,
        f"{scene}: cache descriptor identity differs",
    )
    _require(cache.get("schema_version") == 1, f"{scene}: cache schema differs")
    _require(
        set(cache) == {"schema_version", "xyz", "valid", "confidence", "components", "metadata"},
        f"{scene}: cache fields differ",
    )
    xyz = torch.as_tensor(cache["xyz"]).float().cpu()
    valid = torch.as_tensor(cache["valid"]).bool().cpu()
    confidence = torch.as_tensor(cache["confidence"]).float().cpu()
    count = int(xyz.shape[0]) if xyz.ndim == 2 else -1
    _require(
        xyz.shape == (count, 3)
        and valid.shape == (count,)
        and confidence.shape == (count,),
        f"{scene}: cache row alignment differs",
    )
    _require(
        bool(torch.isfinite(xyz).all())
        and bool(torch.isfinite(confidence).all())
        and not bool((confidence < 0).any())
        and not bool((confidence > 1).any())
        and not bool((confidence[~valid] != 0).any()),
        f"{scene}: cache confidence differs",
    )
    components = _mapping(cache["components"], f"{scene}: reliability components")
    _require(
        set(components)
        == {"observation_evidence", "multiview_agreement", "reconstruction_fidelity"},
        f"{scene}: reliability component fields differ",
    )
    for name, raw in components.items():
        values = torch.as_tensor(raw).float().cpu()
        _require(
            values.shape == (count,)
            and bool(torch.isfinite(values).all())
            and not bool((values < 0).any())
            and not bool((values > 1).any()),
            f"{scene}: reliability component {name} differs",
        )
    xyz_sha256 = _xyz_sha256(xyz)
    metadata = _validate_metadata(
        cache["metadata"],
        scene=scene,
        field_record=field_record,
        mpr_record=mpr_record,
        num_gaussians=count,
        geometry_xyz_sha256=xyz_sha256,
    )
    report, report_sha256, loaded_report_path = load_json_object(
        report_path,
        expected_sha256=str(report_record["sha256"]),
        label=f"{scene}: reliability build report",
    )
    _require(
        report_sha256 == report_record["sha256"] and loaded_report_path == report_path,
        f"{scene}: report descriptor identity differs",
    )
    _require(
        _absolute(str(report.get("output", ""))) == cache_path
        and report.get("num_gaussians") == count
        and report.get("valid_gaussians") == int(valid.sum())
        and report.get("metadata") == metadata,
        f"{scene}: report/cache binding differs",
    )
    return {
        "num_gaussians": count,
        "valid_gaussians": int(valid.sum()),
        "geometry_fingerprint": {
            "num_gaussians": count,
            "xyz_sha256": xyz_sha256,
        },
        "metadata_canonical_json_sha256": canonical_json_sha256(metadata),
    }


def _parent_field_record(
    parent: Mapping[str, Any], *, scene: str
) -> Mapping[str, Any]:
    sources = _mapping(parent.get("source_artifacts"), "parent source artifacts")
    row = _mapping(sources.get(scene), f"{scene}: parent source row")
    record = _mapping(row.get(FIELD_NAME), f"{scene}: parent canonical field")
    _require(
        isinstance(record.get("bytes"), int)
        and isinstance(record.get("sha256"), str),
        f"{scene}: parent canonical-field record differs",
    )
    return record


def build_manifest(
    *,
    source_root: str | Path,
    cache_root: str | Path,
    parent_asset_manifest: str | Path,
    builder_source: str | Path,
) -> dict[str, Any]:
    """Construct and fully revalidate one complete eight-scene manifest."""

    source = _absolute(source_root).resolve(strict=True)
    caches = _absolute(cache_root).resolve(strict=True)
    _require(source.is_dir() and not source.is_symlink(), "source root is unsafe")
    _require(caches.is_dir() and not caches.is_symlink(), "cache root is unsafe")
    parent_record = _file_record(parent_asset_manifest, label="parent asset manifest")
    parent, _, parent_path = load_json_object(
        parent_record["path"],
        expected_sha256=str(parent_record["sha256"]),
        label="parent asset manifest",
    )
    _require(parent_path == Path(str(parent_record["path"])), "parent identity differs")
    _require(parent.get("scenes") == list(ORDERED_SCENES), "parent cohort differs")
    _require(
        _absolute(str(parent.get("source_root", ""))).resolve(strict=True) == source,
        "parent source root differs",
    )
    builder_record = _file_record(builder_source, label="reliability builder source")

    rows: dict[str, Any] = {}
    for scene in ORDERED_SCENES:
        field_record = _file_record(source / scene / FIELD_NAME, label=f"{scene}: canonical field")
        parent_field = _parent_field_record(parent, scene=scene)
        _require(
            field_record["bytes"] == parent_field.get("bytes")
            and field_record["sha256"] == parent_field.get("sha256")
            and Path(str(field_record["path"]))
            == _absolute(str(parent_field.get("path", ""))),
            f"{scene}: canonical field differs from parent asset authority",
        )
        mpr_record = _file_record(source / scene / MPR_NAME, label=f"{scene}: MPR cache")
        cache_record = _file_record(caches / scene / CACHE_NAME, label=f"{scene}: reliability cache")
        report_record = _file_record(caches / scene / REPORT_NAME, label=f"{scene}: reliability report")
        semantic = _validate_cache_and_report(
            scene=scene,
            cache_record=cache_record,
            report_record=report_record,
            field_record=field_record,
            mpr_record=mpr_record,
        )
        rows[scene] = {
            "field_checkpoint": field_record,
            "mpr_cache": mpr_record,
            "reliability_cache": cache_record,
            "build_report": report_record,
            **semantic,
        }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "complete_query_independent_full8_bound",
        "candidate_scope": "nvos_forward_beta_balanced_residual_v2_input_only",
        "ordered_scenes": list(ORDERED_SCENES),
        "source_root": str(source),
        "cache_root": str(caches),
        "parent_asset_manifest": parent_record,
        "builder_source": builder_record,
        "safety_contract": dict(SAFETY_CONTRACT),
        "target_data_read_by_binder": False,
        "query_data_read_by_binder": False,
        "scenes": rows,
    }
    payload["manifest_payload_sha256"] = canonical_json_sha256(payload)
    validate_manifest_payload(payload, verify_files=False)
    return payload


def validate_manifest_payload(
    payload: Mapping[str, Any], *, verify_files: bool = True
) -> None:
    """Fail closed on cohort, digest, safety, and optionally all disk bytes."""

    _require(
        set(payload)
        == {
            "schema_version", "artifact_type", "status", "candidate_scope",
            "ordered_scenes", "source_root", "cache_root",
            "parent_asset_manifest", "builder_source", "safety_contract",
            "target_data_read_by_binder", "query_data_read_by_binder", "scenes",
            "manifest_payload_sha256",
        },
        "manifest fields differ",
    )
    _require(payload.get("schema_version") == SCHEMA_VERSION, "manifest schema differs")
    _require(payload.get("artifact_type") == ARTIFACT_TYPE, "manifest type differs")
    _require(
        payload.get("status") == "complete_query_independent_full8_bound",
        "manifest status differs",
    )
    _require(
        payload.get("candidate_scope")
        == "nvos_forward_beta_balanced_residual_v2_input_only",
        "candidate scope differs",
    )
    _require(payload.get("ordered_scenes") == list(ORDERED_SCENES), "cohort differs")
    _require(payload.get("safety_contract") == SAFETY_CONTRACT, "safety contract differs")
    _require(
        payload.get("target_data_read_by_binder") is False
        and payload.get("query_data_read_by_binder") is False,
        "binder data-access contract differs",
    )
    digest_payload = dict(payload)
    declared_digest = digest_payload.pop("manifest_payload_sha256", None)
    _require(
        isinstance(declared_digest, str)
        and SHA256_RE.fullmatch(declared_digest) is not None
        and canonical_json_sha256(digest_payload) == declared_digest,
        "manifest payload SHA-256 differs",
    )
    source_root = _absolute(str(payload.get("source_root", ""))).resolve(strict=True)
    cache_root = _absolute(str(payload.get("cache_root", ""))).resolve(strict=True)
    _require(source_root.is_dir(), "manifest source root differs")
    _require(cache_root.is_dir(), "manifest cache root differs")
    rows = _mapping(payload.get("scenes"), "manifest scenes")
    _require(list(rows) == list(ORDERED_SCENES), "manifest scene order differs")
    parent: Mapping[str, Any] | None = None
    if verify_files:
        parent_path = _validate_file_record(
            payload.get("parent_asset_manifest"), label="parent asset manifest"
        )
        parent, _, loaded_parent_path = load_json_object(
            parent_path,
            expected_sha256=str(
                _mapping(payload["parent_asset_manifest"], "parent record")["sha256"]
            ),
            label="parent asset manifest",
        )
        _require(loaded_parent_path == parent_path, "parent manifest identity differs")
        _require(parent.get("scenes") == list(ORDERED_SCENES), "parent cohort differs")
        _require(
            _absolute(str(parent.get("source_root", ""))).resolve(strict=True)
            == source_root,
            "parent source root differs",
        )
        builder_path = _validate_file_record(
            payload.get("builder_source"), label="reliability builder source"
        )
        _require(
            builder_path.name == "build_canonical_reliability_cache.py",
            "reliability builder source path differs",
        )
    for scene in ORDERED_SCENES:
        row = _mapping(rows[scene], f"{scene}: manifest row")
        _require(
            set(row)
            == {
                "field_checkpoint", "mpr_cache", "reliability_cache", "build_report",
                "num_gaussians", "valid_gaussians", "geometry_fingerprint",
                "metadata_canonical_json_sha256",
            },
            f"{scene}: manifest row fields differ",
        )
        _require(
            isinstance(row.get("num_gaussians"), int)
            and not isinstance(row.get("num_gaussians"), bool)
            and isinstance(row.get("valid_gaussians"), int)
            and 0 <= int(row["valid_gaussians"]) <= int(row["num_gaussians"]),
            f"{scene}: manifest row counts differ",
        )
        geometry = _mapping(row.get("geometry_fingerprint"), f"{scene}: geometry")
        _require(
            geometry.get("num_gaussians") == row.get("num_gaussians")
            and isinstance(geometry.get("xyz_sha256"), str)
            and SHA256_RE.fullmatch(str(geometry["xyz_sha256"])) is not None,
            f"{scene}: manifest geometry differs",
        )
        _require(
            isinstance(row.get("metadata_canonical_json_sha256"), str)
            and SHA256_RE.fullmatch(str(row["metadata_canonical_json_sha256"]))
            is not None,
            f"{scene}: metadata digest differs",
        )
        if verify_files:
            field_path = _validate_file_record(row["field_checkpoint"], label=f"{scene}: canonical field")
            mpr_path = _validate_file_record(row["mpr_cache"], label=f"{scene}: MPR cache")
            cache_path = Path(
                str(_mapping(row["reliability_cache"], f"{scene}: cache record")["path"])
            )
            report_path = Path(
                str(_mapping(row["build_report"], f"{scene}: report record")["path"])
            )
            _require(
                field_path == source_root / scene / FIELD_NAME
                and mpr_path == source_root / scene / MPR_NAME
                and cache_path == cache_root / scene / CACHE_NAME
                and report_path == cache_root / scene / REPORT_NAME,
                f"{scene}: fixed source/cache path differs",
            )
            _require(parent is not None, "parent manifest was not loaded")
            parent_field = _parent_field_record(parent, scene=scene)
            field_record = _mapping(row["field_checkpoint"], f"{scene}: field record")
            _require(
                field_record["bytes"] == parent_field.get("bytes")
                and field_record["sha256"] == parent_field.get("sha256")
                and field_path == _absolute(str(parent_field.get("path", ""))),
                f"{scene}: field differs from parent asset authority",
            )
            semantic = _validate_cache_and_report(
                scene=scene,
                cache_record=_mapping(row["reliability_cache"], f"{scene}: cache record"),
                report_record=_mapping(row["build_report"], f"{scene}: report record"),
                field_record=_mapping(row["field_checkpoint"], f"{scene}: field record"),
                mpr_record=_mapping(row["mpr_cache"], f"{scene}: MPR record"),
            )
            _require(field_path.parent.name == scene, f"{scene}: field scene path differs")
            _require(mpr_path.parent.name == scene, f"{scene}: MPR scene path differs")
            _require(
                all(row[key] == semantic[key] for key in semantic),
                f"{scene}: semantic manifest row differs",
            )


def validate_manifest(path: str | Path) -> dict[str, Any]:
    payload, _, _ = load_json_object(path, label="Beta-v2 reliability manifest")
    validate_manifest_payload(payload, verify_files=True)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bind = commands.add_parser("bind")
    bind.add_argument("--source-root", type=Path, required=True)
    bind.add_argument("--cache-root", type=Path, required=True)
    bind.add_argument("--parent-asset-manifest", type=Path, required=True)
    bind.add_argument("--builder-source", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "bind":
            payload = build_manifest(
                source_root=args.source_root,
                cache_root=args.cache_root,
                parent_asset_manifest=args.parent_asset_manifest,
                builder_source=args.builder_source,
            )
            write_frozen_json(args.output, payload)
        else:
            payload = validate_manifest(args.manifest)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
