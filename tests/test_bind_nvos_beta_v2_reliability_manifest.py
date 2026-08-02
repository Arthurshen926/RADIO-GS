import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from radio_gs.scripts.bind_nvos_beta_v2_reliability_manifest import (
    FIELD_NAME,
    FORMULA,
    ORDERED_SCENES,
    SAFETY_CONTRACT,
    build_manifest,
    validate_manifest,
    validate_manifest_payload,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, write_frozen_json


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xyz_sha(values: torch.Tensor) -> str:
    array = values.float().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_root = tmp_path / "source"
    cache_root = tmp_path / "cache"
    builder = tmp_path / "build_canonical_reliability_cache.py"
    builder.write_text("# fixed builder\n", encoding="utf-8")
    source_artifacts = {}
    for index, scene in enumerate(ORDERED_SCENES):
        source_dir = source_root / scene
        cache_dir = cache_root / scene
        source_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        field = source_dir / FIELD_NAME
        mpr = source_dir / "raw_radio.pt"
        field.write_bytes(f"field-{scene}".encode())
        mpr.write_bytes(f"mpr-{scene}".encode())
        xyz = torch.tensor([[float(index), 0.0, 0.0], [float(index), 1.0, 0.0]])
        valid = torch.tensor([True, False])
        confidence = torch.tensor([0.75, 0.0])
        metadata = {
            "schema_version": 1,
            "source": "canonical_primitive_reliability_v1",
            "formula": FORMULA,
            "observation_prior_count": 1,
            "combination": "equal_weight_geometric_mean",
            "field_checkpoint": str(field.resolve()),
            "field_checkpoint_sha256": _sha(field),
            "mpr_cache": str(mpr.resolve()),
            "mpr_cache_sha256": _sha(mpr),
            "mpr_construction": "test_mpr",
            "geometry_fingerprint": {
                "num_gaussians": 2,
                "xyz_sha256": _xyz_sha(xyz),
            },
            **SAFETY_CONTRACT,
            "third_mpr_reliability_channel_used": False,
        }
        cache = cache_dir / "canonical_reliability.pt"
        torch.save(
            {
                "schema_version": 1,
                "xyz": xyz,
                "valid": valid,
                "confidence": confidence,
                "components": {
                    "observation_evidence": confidence,
                    "multiview_agreement": confidence,
                    "reconstruction_fidelity": confidence,
                },
                "metadata": metadata,
            },
            cache,
        )
        report = {
            "output": str(cache.resolve()),
            "num_gaussians": 2,
            "valid_gaussians": 1,
            "metadata": metadata,
        }
        (cache_dir / "canonical_reliability.pt.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        source_artifacts[scene] = {
            FIELD_NAME: {
                "path": str(field.resolve()),
                "bytes": field.stat().st_size,
                "sha256": _sha(field),
            }
        }
    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps(
            {
                "scenes": list(ORDERED_SCENES),
                "source_root": str(source_root.resolve()),
                "source_artifacts": source_artifacts,
            }
        ),
        encoding="utf-8",
    )
    return source_root, cache_root, parent, builder


def _build(tmp_path: Path):
    source, caches, parent, builder = _fixture(tmp_path)
    return build_manifest(
        source_root=source,
        cache_root=caches,
        parent_asset_manifest=parent,
        builder_source=builder,
    )


def _redigest(payload: dict) -> None:
    payload.pop("manifest_payload_sha256", None)
    payload["manifest_payload_sha256"] = canonical_json_sha256(payload)


def test_build_and_full_disk_validation(tmp_path: Path):
    payload = _build(tmp_path)
    assert list(payload["scenes"]) == list(ORDERED_SCENES)
    assert payload["safety_contract"] == SAFETY_CONTRACT
    output = tmp_path / "manifest.json"
    write_frozen_json(output, payload)
    assert validate_manifest(output) == payload


def test_manifest_fails_closed_on_query_safety_or_geometry_drift(tmp_path: Path):
    payload = _build(tmp_path)
    unsafe = copy.deepcopy(payload)
    unsafe["safety_contract"]["uses_query"] = True
    _redigest(unsafe)
    with pytest.raises(ValueError, match="safety contract differs"):
        validate_manifest_payload(unsafe, verify_files=False)

    geometry = copy.deepcopy(payload)
    geometry["scenes"]["fern"]["geometry_fingerprint"]["xyz_sha256"] = "0" * 64
    _redigest(geometry)
    with pytest.raises(ValueError, match="semantic manifest row differs"):
        validate_manifest_payload(geometry, verify_files=True)


def test_manifest_rejects_cache_tamper_and_symlink(tmp_path: Path):
    payload = _build(tmp_path)
    cache = Path(payload["scenes"]["fern"]["reliability_cache"]["path"])
    cache.write_bytes(cache.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="immutable bytes differ"):
        validate_manifest_payload(payload, verify_files=True)

    payload = _build(tmp_path / "second")
    report_record = payload["scenes"]["fern"]["build_report"]
    report = Path(report_record["path"])
    target = report.with_name("real-report.json")
    report.rename(target)
    report.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        validate_manifest_payload(payload, verify_files=True)


def test_frozen_manifest_is_no_clobber(tmp_path: Path):
    payload = _build(tmp_path)
    output = tmp_path / "manifest.json"
    write_frozen_json(output, payload)
    write_frozen_json(output, payload)
    changed = copy.deepcopy(payload)
    changed["status"] = "changed"
    with pytest.raises(ValueError, match="existing frozen artifact differs"):
        write_frozen_json(output, changed)
