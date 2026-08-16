from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from radio_gs.scripts.build_five_benchmark_joint_receipt import build_receipt
from radio_gs.scripts.validate_five_benchmark_joint_receipt import (
    JointReceiptError,
    TASKS,
    validate,
)
from radio_gs.utils.immutable_artifacts import file_record


def test_current_readiness_is_complete_accounting_but_not_joint_eligible() -> None:
    receipt = build_receipt()
    report = validate(receipt)

    assert len(receipt["deployment_fields"]["instances"]) == 29
    assert sum(
        len(rows)
        for rows in receipt["deployment_fields"]["task_references"].values()
    ) == 33
    assert report["field_gates"]["physical_field_accounting"]["status"] == "pass"
    assert report["field_gates"]["task_reference_accounting"]["status"] == "pass"
    assert report["field_gates"]["all_fields_complete"]["status"] == "fail"
    assert report["joint_development_baseline_eligible"] is False
    assert report["paper_row_eligible"] is False


def test_namespaced_live_inventory_populates_nvos_instances(tmp_path: Path) -> None:
    source = Path("paper/artifacts/universal_field_v1_asset_inventory_20260816.json")
    inventory = json.loads(source.read_text(encoding="utf-8"))
    inventory["schema_version"] = 2
    inventory["artifact_type"] = "radio_gs_universal_field_v1_live_asset_inventory"
    for scene in (
        "fern", "flower", "fortress", "horns_center", "horns_left", "leaves",
        "orchids", "trex",
    ):
        inventory["scenes"][f"nvos/{scene}"] = {
            "status": "complete",
            "universal_field": {
                "path": f"/sealed/nvos/{scene}.pth",
                "sha256": "a" * 64,
            },
        }
    path = tmp_path / "live_inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")

    receipt = build_receipt(inventory_path=path)
    rows = {
        row["dataset_instance_id"]: row
        for row in receipt["deployment_fields"]["instances"]
    }
    assert all(rows[f"nvos:{scene}"]["status"] == "complete" for scene in (
        "fern", "flower", "fortress", "horns_center", "horns_left", "leaves",
        "orchids", "trex",
    ))


def test_build_to_file_then_cli_validate_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "joint_readiness.json"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "radio_gs.scripts.build_five_benchmark_joint_receipt",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    built = json.loads(build.stdout)
    assert built["output"] == str(output.resolve())
    assert len(built["sha256"]) == 64

    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "radio_gs.scripts.validate_five_benchmark_joint_receipt",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(checked.stdout)
    assert report["receipt_sha256"] == built["sha256"]
    assert report["joint_development_baseline_eligible"] is False
    assert report["paper_row_eligible"] is False


def _complete_receipt(tmp_path: Path) -> dict:
    receipt = build_receipt()
    instance_index = {
        row["dataset_instance_id"]: row
        for row in receipt["deployment_fields"]["instances"]
    }
    for index, row in enumerate(instance_index.values()):
        row["status"] = "complete"
        row["content_verified"] = True
        if not isinstance(row.get("universal_field"), dict):
            row["universal_field"] = {
                "path": f"/sealed/universal_field_{index}.pth",
                "sha256": f"{index + 1:064x}",
            }

    candidate_sha = receipt["joint_candidate"]["candidate_sha256"]
    refs = receipt["deployment_fields"]["task_references"]
    for task in TASKS:
        task_payload = {
            "schema_version": 1,
            "artifact_type": "radio_gs_five_benchmark_task_evaluation_receipt",
            "task_id": task,
            "status": "complete",
            "candidate_sha256": candidate_sha,
            "readout_id": receipt["joint_candidate"]["identity"][
                "typed_readout_map"
            ][task],
            "evaluation_contract_id": f"frozen-{task}-v1",
            "scene_order": refs[task],
            "field_bindings": {
                instance_id: instance_index[instance_id]["universal_field"]["sha256"]
                for instance_id in refs[task]
            },
            "exact_evaluation_contract": True,
            "complete_frozen_cohort": True,
            "evaluator_content_bound": True,
            "authorized_target_access": True,
            "prediction_barrier_required": task in ("nvos", "spin_nerf_available9"),
            "prediction_barrier_passed": True,
            "target_metrics_used_for_current_run_selection": False,
            "prospectively_preregistered": False,
            "prospectively_blind": False,
            "target_metrics_used_for_candidate_selection": True,
            "stochastic": False,
            "seed_panel": [],
        }
        evidence = tmp_path / f"{task}_result.json"
        evidence.write_text(json.dumps(task_payload), encoding="utf-8")
        receipt["task_results"][task] = {
            **task_payload,
            "evidence": file_record(evidence),
        }
    receipt["eligibility"] = {
        "joint_development_baseline_eligible": True,
        "paper_row_eligible": False,
        "sota_claim_supported": False,
    }
    return receipt


def test_complete_same_candidate_can_be_development_but_not_paper(
    tmp_path: Path,
) -> None:
    report = validate(_complete_receipt(tmp_path))

    assert report["joint_development_baseline_eligible"] is True
    assert report["paper_row_eligible"] is False
    assert all(
        gate["status"] == "fail" for gate in report["task_paper_gates"].values()
    )


def test_paper_row_requires_prospective_evidence_and_seed_panel(tmp_path: Path) -> None:
    receipt = _complete_receipt(tmp_path)
    for row in receipt["task_results"].values():
        row["prospectively_preregistered"] = True
        row["prospectively_blind"] = True
        row["target_metrics_used_for_candidate_selection"] = False
        row["stochastic"] = True
        row["seed_panel"] = [0, 1, 2]
        evidence = Path(row["evidence"]["path"])
        payload = dict(row)
        payload.pop("evidence")
        evidence.write_text(json.dumps(payload), encoding="utf-8")
        row["evidence"] = file_record(evidence)
    receipt["eligibility"]["paper_row_eligible"] = True

    report = validate(receipt)
    assert report["paper_row_eligible"] is True
    assert report["sota_claim_supported"] is False


def test_rejects_eligibility_overclaim(tmp_path: Path) -> None:
    receipt = _complete_receipt(tmp_path)
    receipt["task_results"]["nvos"]["complete_frozen_cohort"] = False

    with pytest.raises(JointReceiptError, match="overclaims"):
        validate(receipt)


def test_rejects_candidate_identity_drift() -> None:
    receipt = build_receipt()
    receipt["joint_candidate"]["identity"]["typed_readout_map"]["lerf2d"] = "other"

    with pytest.raises(JointReceiptError, match="candidate SHA-256 differs"):
        validate(receipt)


def test_nvos_and_spin_dataset_instances_cannot_alias() -> None:
    receipt = build_receipt()
    refs = receipt["deployment_fields"]["task_references"]
    refs["spin_nerf_available9"][0] = refs["nvos"][0]

    report = validate(receipt)
    assert report["field_gates"]["dataset_instance_namespace"]["status"] == "fail"


def test_lerf2d_and_lerf3d_must_share_exact_field_instances() -> None:
    receipt = build_receipt()
    receipt["deployment_fields"]["task_references"]["lerf3d"] = list(
        receipt["deployment_fields"]["task_references"]["scannet_ovs"][:4]
    )

    report = validate(receipt)
    assert report["field_gates"]["lerf_shared_field_identity"]["status"] == "fail"
