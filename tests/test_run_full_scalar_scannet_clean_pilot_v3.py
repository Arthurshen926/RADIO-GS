from __future__ import annotations

from argparse import Namespace
import io
import json
from pathlib import Path
import tarfile

import pytest

from radio_gs.scripts import build_full_scalar_scannet_clean_cohort as cohort_builder
from radio_gs.scripts import run_full_scalar_scannet_clean_pilot as v1
from radio_gs.scripts import run_full_scalar_scannet_clean_pilot_v3 as v3
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def _freeze(path: Path, payload: dict) -> tuple[str, str]:
    write_frozen_json(path, payload)
    return str(path), sha256_file(path)


def _clean_report(scene_id: str) -> dict:
    return {
        "valid": True,
        "uses_instances_or_semantic_labels": False,
        "uses_private_anchor": False,
        "uses_private_depth_pixel": False,
        "scenes": [{"scene_id": scene_id}],
    }


def _authorities(tmp_path: Path) -> tuple[Path, dict, dict]:
    archive = tmp_path / "scans.tar.part-00"
    with tarfile.open(archive, "w") as handle:
        for index in range(40):
            scene = f"scene{index:04d}_00"
            payload = f"sealed-{scene}".encode("ascii")
            info = tarfile.TarInfo(f"scans/{scene}/{scene}.sens")
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))

    agile0 = _freeze(tmp_path / "agile0.json", _clean_report("scene0100_00"))
    agile1 = _freeze(tmp_path / "agile1.json", _clean_report("scene0101_00"))
    pfpr = _freeze(tmp_path / "pfpr.json", _clean_report("scene0100_00"))
    nvos = _freeze(tmp_path / "nvos.json", {"scenes": [{"scene_id": "fern"}]})
    spin = _freeze(tmp_path / "spin.json", {"scenes": [{"scene_id": "lego"}]})
    paths = {
        "benchmark_registry_output": tmp_path / "registry.json",
        "exclusion_manifest_output": tmp_path / "exclusion.json",
        "cohort_authority_output": tmp_path / "cohort.json",
        "inventory_output": tmp_path / "inventory.json",
    }
    result = cohort_builder.build(
        Namespace(
            scan_archive_part=str(archive),
            agile_report=[agile0[0], agile1[0]],
            expected_agile_report_sha256=[agile0[1], agile1[1]],
            pfpr_report=pfpr[0],
            expected_pfpr_report_sha256=pfpr[1],
            nvos_manifest=nvos[0],
            expected_nvos_manifest_sha256=nvos[1],
            spin_manifest=spin[0],
            expected_spin_manifest_sha256=spin[1],
            additional_scannet_benchmark_scene_id=[],
            lerf_scene_id=["figurines"],
            **paths,
        )
    )
    return archive, paths, result


def _args(
    tmp_path: Path,
    archive: Path,
    paths: dict,
    result: dict,
    *,
    root_name: str,
) -> Namespace:
    return Namespace(
        scene_id="scene0001_00",
        scan_archive_part=str(archive),
        cohort_authority=str(paths["cohort_authority_output"]),
        expected_cohort_authority_sha256=result["cohort_authority"]["sha256"],
        exclusion_manifest=str(paths["exclusion_manifest_output"]),
        expected_exclusion_manifest_sha256=result[
            "benchmark_exclusion_manifest"
        ]["sha256"],
        inventory=str(paths["inventory_output"]),
        expected_inventory_sha256=result["inventory"]["sha256"],
        pilot_root=str(tmp_path / root_name),
        gpu=0,
        stop_after="sens_extraction",
    )


def test_v3_starts_at_stage00_binds_itself_and_resumes_exactly(tmp_path: Path) -> None:
    archive, paths, authority = _authorities(tmp_path)
    args = _args(tmp_path, archive, paths, authority, root_name="v3")

    first = v3.run(args)
    receipt_path = Path(args.pilot_root) / "receipts/00_sens_extraction.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    frozen = receipt_path.read_bytes()
    implementation = receipt["immutable_authorities"]["launcher_implementation"]
    assert receipt["schema"] == v3.STAGE_RECEIPT_SCHEMA
    assert implementation["path"] == str(Path(v3.__file__).resolve())
    assert implementation["sha256"] == sha256_file(v3.__file__)
    assert implementation["sha256"] != sha256_file(v1.__file__)
    assert receipt["inputs"][v3._DELEGATED_STAGE_RUNNER_INPUT] == (
        v1._file_record(Path(v1.__file__).resolve())
    )
    assert first["completed_stage"] == "sens_extraction"

    assert v3.run(args) == first
    assert receipt_path.read_bytes() == frozen

    with pytest.raises(ValueError, match="receipt contract differs"):
        v1.run(args)


def test_v3_exact_mpr_and_selected_gpu_contract(tmp_path: Path) -> None:
    args = Namespace(scene_id="scene0001_00", pilot_root=tmp_path, gpu=0)
    command = v3._mpr_command(
        args=args,
        config=Path("config.yaml"),
        checkpoint=Path("geometry.pth"),
        feature_dir=Path("features"),
        feature_bundle_sha256="a" * 64,
        geometry_sha256="b" * 64,
        output=Path("output.pt"),
        feature_space="radio",
        observation_contract="canonical-factorized-radio-v1",
        save_responsibility=Path("responsibility.json"),
        normalize_each_view=False,
    )
    alpha = [
        index for index, value in enumerate(command) if value == "--alpha-threshold"
    ]
    assert len(alpha) == 1
    assert command[alpha[0] : alpha[0] + 2] == ["--alpha-threshold", "0"]
    assert alpha[0] < command.index("--aggregation-mode")

    thermal = v1._thermal_command(args, ["python", "stage.py"])
    assert "CUDA_VISIBLE_DEVICES=0" in thermal
    env = v3._thermal_env(args, tmp_path / "gpu1_telemetry.csv")
    assert env["GPU"] == "0"
    assert env["GPU_TELEMETRY_LOG"].endswith("gpu0_telemetry.csv")
    assert env["GPU_MAX_POWER_LIMIT_W"] == "300.5"
    assert env["GPU_MAX_TEMP_C"] == "88"
    assert env["GPU_POLL_SECONDS"] == "30"
    assert env["GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS"] == "6"
    assert env["GPU_OWNER_PID_NAMESPACE_MODE"] == (
        "exclusive-singleton-after-clear-v1"
    )
    assert env["GPU_SOFT_PAUSE_TEMP_C"] == "0"
    assert env["GPU_PEER_INDEX"] == ""

    authority = v3._gpu_thermal_guard_authority(args)
    assert authority["guard_implementation"] == v1._file_record(
        v3._THERMAL_GUARD_PATH
    )
    assert authority["physical_gpu_index"] == 0
    policy = authority["production_policy"]
    assert policy["nominal_board_power_limit_w"] == 300
    assert policy["nvidia_smi_reported_limit_ceiling_w"] == "300.5"
    assert policy["nominal_overheat_window_seconds"] == 180
    assert authority["guard_environment"] == v3._FIXED_THERMAL_ENV


def test_v3_strictly_reloads_exact_legacy_cpu_stage00(tmp_path: Path) -> None:
    archive, paths, authority = _authorities(tmp_path)
    args = _args(tmp_path, archive, paths, authority, root_name="legacy-stage00")
    first = v3.run(args)
    receipt_path = Path(args.pilot_root) / "receipts/00_sens_extraction.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["immutable_authorities"]["launcher_implementation"] = dict(
        v3._LEGACY_STAGE00_LAUNCHER_IMPLEMENTATION
    )
    receipt["inputs"].pop(v3._DELEGATED_STAGE_RUNNER_INPUT)
    receipt["authority_sha256"] = v1._receipt_content_sha256(receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    legacy_bytes = receipt_path.read_bytes()

    resumed = v3.run(args)
    assert resumed["completed_stage"] == first["completed_stage"]
    assert receipt_path.read_bytes() == legacy_bytes

    changed_common = v3.validate_pilot_authorities(args)
    changed_common[v3._GPU_THERMAL_GUARD_INPUT] = {
        "deliberately": "changed"
    }
    with v3._v3_runtime():
        assert v3._validate_stage_receipt(
            receipt, stage="sens_extraction", common=changed_common
        )["stage"] == "sens_extraction"


def test_v3_fails_closed_on_unreceipted_output_and_receipt_gap(
    tmp_path: Path,
) -> None:
    archive, paths, authority = _authorities(tmp_path)
    unreceipted = _args(
        tmp_path, archive, paths, authority, root_name="unreceipted"
    )
    sens = (
        Path(unreceipted.pilot_root)
        / "source/scans/scene0001_00/scene0001_00.sens"
    )
    sens.parent.mkdir(parents=True)
    sens.write_bytes(b"not sealed")
    with pytest.raises(FileExistsError, match="unreceipted .sens"):
        v3.run(unreceipted)

    gap = _args(tmp_path, archive, paths, authority, root_name="gap")
    v3.run(gap)
    receipt_root = Path(gap.pilot_root) / "receipts"
    (receipt_root / "02_geometry.json").write_bytes(
        (receipt_root / "00_sens_extraction.json").read_bytes()
    )
    with pytest.raises(ValueError, match="not one contiguous prefix"):
        v3.run(gap)


def test_v3_rejects_a_changed_stage_predecessor(tmp_path: Path) -> None:
    archive, paths, authority = _authorities(tmp_path)
    args = _args(tmp_path, archive, paths, authority, root_name="chain")
    v3.run(args)
    common = v3.validate_pilot_authorities(args)
    receipt = Path(args.pilot_root) / "receipts/01_query_free_materialization.json"
    files = {}
    for name in (
        "materialization_report",
        "field_source_contract",
        "stage_log",
    ):
        path = Path(args.pilot_root) / f"synthetic_{name}"
        path.write_text(name, encoding="utf-8")
        files[name] = v1._file_record(path)
    outputs = {
        **files,
        "field_frame_manifest_sha256": "a" * 64,
    }
    with v3._v3_runtime():
        v1._write_stage_receipt(
            receipt,
            stage="query_free_materialization",
            common=common,
            inputs=v1._common_inputs(common, None),
            outputs=outputs,
            command=["python", "synthetic_stage.py"],
        )
    with pytest.raises(ValueError, match="predecessor chain differs"):
        v3.validate_resume_boundary(args.pilot_root, common=common)


def test_v3_final_receipt_and_every_stage_bind_v3_and_runner(tmp_path: Path) -> None:
    archive, paths, authority = _authorities(tmp_path)
    args = _args(tmp_path, archive, paths, authority, root_name="complete")
    args.stop_after = v3.STAGES[-1]
    common = v3.validate_pilot_authorities(args)
    root = Path(args.pilot_root)
    receipt_root = root / "receipts"
    receipt_root.mkdir(parents=True)
    exact_state = (
        root
        / "run/exact_state/scene0001_00/factorized_primitive_state_v2.pt"
    )
    exact_state.parent.mkdir(parents=True)
    exact_state.write_bytes(b"synthetic-factorized-state")

    previous = None
    with v3._v3_runtime():
        for index, stage in enumerate(v3.STAGES):
            outputs = {}
            for key in v3._STAGE_OUTPUT_KEYS[stage]:
                if key == "archive_member":
                    outputs[key] = {
                        "path": common["archive"]["path"],
                        "member": common["archive"]["member"],
                        "member_size_bytes": common["archive"][
                            "member_size_bytes"
                        ],
                        "payload_sha256": "a" * 64,
                    }
                elif key in {
                    "field_frame_manifest_sha256",
                    "feature_output_bundle_sha256",
                }:
                    outputs[key] = "a" * 64
                elif key == "num_frames":
                    outputs[key] = 1
                else:
                    output = (
                        exact_state
                        if key == "factorized_state"
                        else root / "synthetic" / stage / key
                    )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if not output.exists():
                        output.write_text(f"{stage}:{key}", encoding="utf-8")
                    outputs[key] = v1._file_record(output)
            if stage in v3._COMMANDLESS_STAGES:
                command = None
            elif stage in {
                "factorized_radio",
                "exact_raw_reference",
                "exact_dino_v3",
                "exact_sam3",
            }:
                command = ["python", stage, "--alpha-threshold", "0"]
            else:
                command = ["python", stage]
            if stage in v3._GPU_STAGES:
                command = v1._thermal_command(args, command)
            receipt = receipt_root / f"{index:02d}_{stage}.json"
            v1._write_stage_receipt(
                receipt,
                stage=stage,
                common=common,
                inputs=v1._common_inputs(common, previous),
                outputs=outputs,
                command=command,
            )
            previous = receipt

    result = v3.run(args)
    final_path = root / "pilot_receipt.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert result["final_receipt"] == v1._file_record(final_path)
    assert final["schema"] == v3.FINAL_RECEIPT_SCHEMA
    implementation = final["immutable_authorities"]["launcher_implementation"]
    assert implementation["sha256"] == sha256_file(v3.__file__)
    assert implementation["sha256"] != sha256_file(v1.__file__)
    for index, stage in enumerate(v3.STAGES):
        receipt = json.loads(
            (receipt_root / f"{index:02d}_{stage}.json").read_text(
                encoding="utf-8"
            )
        )
        assert receipt["schema"] == v3.STAGE_RECEIPT_SCHEMA
        assert receipt["immutable_authorities"]["launcher_implementation"] == (
            implementation
        )
        assert receipt["inputs"][v3._DELEGATED_STAGE_RUNNER_INPUT] == (
            common[v3._DELEGATED_STAGE_RUNNER_INPUT]
        )
        guard = receipt["inputs"].get(v3._GPU_THERMAL_GUARD_INPUT)
        if stage in v3._GPU_STAGES:
            assert guard == common[v3._GPU_THERMAL_GUARD_INPUT]
            assert receipt["command"][:5] == [
                "bash",
                "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
                "--",
                "env",
                "CUDA_VISIBLE_DEVICES=0",
            ]
        else:
            assert guard is None

    frozen = final_path.read_bytes()
    assert v3.run(args) == result
    assert final_path.read_bytes() == frozen

    old_gpu_receipt_path = receipt_root / "02_geometry.json"
    old_gpu_receipt = json.loads(
        old_gpu_receipt_path.read_text(encoding="utf-8")
    )
    old_gpu_receipt["inputs"].pop(v3._GPU_THERMAL_GUARD_INPUT)
    old_gpu_receipt["authority_sha256"] = v1._receipt_content_sha256(
        old_gpu_receipt
    )
    old_gpu_receipt_path.write_text(
        json.dumps(old_gpu_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="GPU thermal guard authority differs"):
        v3.validate_resume_boundary(args.pilot_root, common=common)


def test_v3_feature_command_is_strict_resumable_and_runner_bound() -> None:
    command = v1._radio_feature_command(
        scene="scene0001_00",
        image_dir=Path("images"),
        feature_dir=Path("features"),
    )
    assert command[command.index("--batch_size") + 1] == "1"
    assert "--skip_pca_stats" in command
    assert "--resume-partial" in command

    args = Namespace(scene_id="scene0001_00", pilot_root="root", gpu=0)
    common = {
        v3._DELEGATED_STAGE_RUNNER_INPUT: v1._file_record(
            Path(v1.__file__).resolve()
        ),
        v3._GPU_THERMAL_GUARD_INPUT: v3._gpu_thermal_guard_authority(args),
    }
    inputs = {}
    captured = {}

    def fake_write(path, *, stage, common, inputs, outputs, command):
        del path, common, outputs, command
        captured["stage"] = stage
        captured["inputs"] = dict(inputs)
        return {"stage": stage, "inputs": dict(inputs)}

    original = v3._V1_WRITE_STAGE_RECEIPT
    try:
        v3._V1_WRITE_STAGE_RECEIPT = fake_write
        v3._write_stage_receipt(
            Path("receipt.json"),
            stage="query_free_materialization",
            common=common,
            inputs=inputs,
            outputs={},
            command=["python", "stage.py"],
        )
    finally:
        v3._V1_WRITE_STAGE_RECEIPT = original
    assert captured["inputs"][v3._DELEGATED_STAGE_RUNNER_INPUT] == (
        common[v3._DELEGATED_STAGE_RUNNER_INPUT]
    )
