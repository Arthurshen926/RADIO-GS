import json
from pathlib import Path

import pytest
import torch

from radio_gs.scripts.run_lerf_official_crop_summary_heldout_gate import (
    AUDIT_MODULE,
    build_gate_authority,
    validate_gate_result,
)
from radio_gs.scripts.seal_lerf_official_crop_summary_bundle import (
    RAMEN_SOURCE_HELDOUT_FRAME_IDS,
    build_reseal_plan,
    file_sha256,
    load_preregistration,
    seal_bundle,
)


def _write_json(path: Path, value: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _file_record(path: Path, payload: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _fixture(tmp_path: Path) -> dict[str, object]:
    selected = [1, 3]
    heldout = list(RAMEN_SOURCE_HELDOUT_FRAME_IDS)
    authority = _write_json(
        tmp_path / "selected.json",
        {
            "frame_indices": selected,
            "metadata": {"selected_frame_indices": selected},
        },
    )
    tensor_dir = tmp_path / "raw" / "ramen"
    tensor_dir.mkdir(parents=True)
    tensor_hashes: dict[int, str] = {}
    for frame_id in sorted(selected + heldout):
        value = torch.zeros(1536, 2, 3, dtype=torch.float16)
        value[frame_id % 1536] = 1
        path = tensor_dir / f"rgb_{frame_id}.pt"
        torch.save(value, path)
        tensor_hashes[frame_id] = file_sha256(path)
    config = _file_record(tmp_path / "config.yaml", b"feature_height: 2\nfeature_width: 3\n")
    geometry = _file_record(tmp_path / "geometry.pth", b"geometry")
    mpr = _file_record(tmp_path / "mpr.pt", b"mpr")
    audit = _file_record(tmp_path / "audit.py", b"# audit\n")
    root = Path(__file__).resolve().parents[1]
    implementations = {
        role: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
        for role, path in {
            "resealer": root
            / "radio_gs/scripts/seal_lerf_official_crop_summary_bundle.py",
            "gate_wrapper": root
            / "radio_gs/scripts/run_lerf_official_crop_summary_heldout_gate.py",
        }.items()
    }
    preregistration = {
        "schema": "radio_gs.lerf_official_crop_summary_source_heldout_preregistration.v1",
        "schema_version": 1,
        "status": "sealed_before_source_gate_execution",
        "implementation": implementations,
        "heldout_audit_implementation": audit,
        "scenes": {
            "ramen": {
                "raw_tensor_dir": str(tensor_dir.resolve()),
                "raw_frame_count": 6,
                "selected_frame_count": 2,
                "selected_view_authority": authority,
                "source_heldout_frame_ids": heldout,
                "forbidden_target_frame_ids": [6],
                "tensor_contract": {
                    "shape": [1536, 2, 3],
                    "dtype": "float16",
                    "maximum_norm_deviation": 0.002,
                },
                "config": config,
                "geometry_checkpoint": geometry,
                "genuine_mpr": mpr,
            }
        },
        "target_data_or_metrics_opened_at_seal": False,
        "target_metric_execution_authorized": False,
    }
    preregistration_record = _write_json(tmp_path / "preregistration.json", preregistration)
    repo_root = tmp_path / "repo"
    launcher = repo_root / "radio_gs/scripts/run_repo_python.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    return {
        "preregistration": preregistration,
        "preregistration_record": preregistration_record,
        "tensor_dir": tensor_dir,
        "tensor_hashes": tensor_hashes,
        "repo_root": repo_root,
    }


def test_reseal_is_content_addressed_without_modifying_legacy_tensors(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    preregistration_record = fixture["preregistration_record"]
    preregistration, record = load_preregistration(
        preregistration_record["path"], preregistration_record["sha256"]
    )
    plan = build_reseal_plan(preregistration, record, scene="ramen")
    assert plan["mode"] == "metadata_only_reseal_plan_no_tensor_bytes_opened"
    assert plan["source_heldout_frame_ids"] == [2, 45, 87, 130]
    assert plan["tensor_content_hashes_computed"] is False

    output = tmp_path / "reseal.json"
    sealed = seal_bundle(
        preregistration_record["path"],
        preregistration_record["sha256"],
        scene="ramen",
        output=output,
    )
    assert sealed["mode"] == "content_addressed_immutable_reseal"
    assert sealed["tensor_content_hashes_computed"] is True
    assert sealed["source_directory_modified"] is False
    assert [record["frame_id"] for record in sealed["frame_records"]] == [
        1,
        2,
        3,
        45,
        87,
        130,
    ]
    assert {
        record["frame_id"]: record["sha256"] for record in sealed["frame_records"]
    } == fixture["tensor_hashes"]
    assert all(
        file_sha256(fixture["tensor_dir"] / f"rgb_{frame_id}.pt") == digest
        for frame_id, digest in fixture["tensor_hashes"].items()
    )
    resumed = seal_bundle(
        preregistration_record["path"],
        preregistration_record["sha256"],
        scene="ramen",
        output=output,
    )
    assert resumed["idempotent_existing_seal"] is True


def test_ramen_rejects_any_implicit_or_mixed_excluded_frame_selection(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    preregistration = dict(fixture["preregistration"])
    preregistration["scenes"] = {
        "ramen": {
            **preregistration["scenes"]["ramen"],
            "source_heldout_frame_ids": [2, 6, 45, 87, 130],
            "raw_frame_count": 7,
        }
    }
    record = _write_json(tmp_path / "mixed-preregistration.json", preregistration)
    with pytest.raises(ValueError, match="exactly explicit frames 2,45,87,130"):
        seal_bundle(
            record["path"], record["sha256"], scene="ramen", output=tmp_path / "bad.json"
        )


def test_gate_plan_always_passes_exact_explicit_ramen_frames(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    preregistration_record = fixture["preregistration_record"]
    reseal_path = tmp_path / "reseal.json"
    reseal = seal_bundle(
        preregistration_record["path"],
        preregistration_record["sha256"],
        scene="ramen",
        output=reseal_path,
    )
    preregistration, verified_preregistration = load_preregistration(
        preregistration_record["path"], preregistration_record["sha256"]
    )
    authority = build_gate_authority(
        preregistration,
        verified_preregistration,
        scene="ramen",
        reseal=reseal,
        reseal_record={
            "path": str(reseal_path.resolve()),
            "sha256": file_sha256(reseal_path),
        },
        result_output=tmp_path / "result.pt",
        device="cuda:1",
        repo_root=fixture["repo_root"],
        verify_large_inputs=True,
    )
    command = authority["command"]
    assert AUDIT_MODULE in command
    frame_index = command.index("--frame-ids")
    assert command[frame_index + 1] == "2,45,87,130"
    assert authority["source_heldout_frame_ids"] == [2, 45, 87, 130]
    assert authority["forbidden_target_frame_ids"] == [6]
    assert authority["protocol"]["historical_mpr_excluded_frame_fallback_allowed"] is False
    assert authority["protocol"]["target_metric_execution_authorized"] is False

    result = {
        "selected_frame_ids": [2, 45, 87, 130],
        "per_view": [
            {
                "frame_id": frame_id,
                "visible_pixels": 10,
                "registered_rows": 3,
                "mean_pixel_cosine": 0.5 + index / 10,
            }
            for index, frame_id in enumerate([2, 45, 87, 130])
        ],
        "protocol": {
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }
    result_path = tmp_path / "result.pt"
    torch.save(result, result_path)
    validation = validate_gate_result(authority, result_path)
    assert validation["passed"] is True
    assert validation["role"] == "diagnostic_readiness_gate_not_target_promotion"


def test_production_preregistration_freezes_two_source_scenes_and_code() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "paper/artifacts/lerf_official_crop_summary_source_heldout_preregistration_20260809.json"
    )
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    assert set(preregistration["scenes"]) == {"ramen", "teatime"}
    assert preregistration["scenes"]["ramen"]["source_heldout_frame_ids"] == [
        2,
        45,
        87,
        130,
    ]
    assert set(preregistration["scenes"]["ramen"]["source_heldout_frame_ids"]).isdisjoint(
        preregistration["scenes"]["ramen"]["forbidden_target_frame_ids"]
    )
    assert preregistration["source_gate_contract"][
        "historical_mpr_excluded_frame_ids_fallback"
    ] == "forbidden"
    assert preregistration["source_gate_contract"]["fixed_cosine_threshold"] is None
    assert preregistration["target_metric_execution_authorized"] is False
    for record in preregistration["implementation"].values():
        assert file_sha256(record["path"]) == record["sha256"]
