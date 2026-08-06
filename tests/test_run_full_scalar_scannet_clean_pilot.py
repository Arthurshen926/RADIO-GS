from __future__ import annotations

from argparse import Namespace
import io
import json
from pathlib import Path
import tarfile

import pytest

from radio_gs.scripts import build_full_scalar_scannet_clean_cohort as cohort_builder
from radio_gs.scripts import run_full_scalar_scannet_clean_pilot as pilot
from radio_gs.scripts import run_full_scalar_scannet_clean_pilot_v2 as pilot_v2
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def _freeze(path, payload: dict) -> tuple[str, str]:
    write_frozen_json(path, payload)
    return str(path), sha256_file(path)


def _report(scene_id: str) -> dict:
    return {
        "valid": True,
        "uses_instances_or_semantic_labels": False,
        "uses_private_anchor": False,
        "uses_private_depth_pixel": False,
        "scenes": [{"scene_id": scene_id}],
    }


def _authorities(tmp_path):
    archive = tmp_path / "scans.tar.part-00"
    with tarfile.open(archive, "w") as handle:
        for index in range(40):
            scene = f"scene{index:04d}_00"
            payload = f"sealed-{scene}".encode("ascii")
            info = tarfile.TarInfo(f"scans/{scene}/{scene}.sens")
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    agile0 = _freeze(tmp_path / "agile0.json", _report("scene0100_00"))
    agile1 = _freeze(tmp_path / "agile1.json", _report("scene0101_00"))
    pfpr = _freeze(tmp_path / "pfpr.json", _report("scene0100_00"))
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


def _args(tmp_path, archive, paths, result, *, root_name="pilot"):
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
        gpu=1,
        stop_after="sens_extraction",
    )


def test_pilot_extracts_exact_sealed_member_and_resumes_without_clobber(
    tmp_path,
) -> None:
    archive, paths, authorities = _authorities(tmp_path)
    args = _args(tmp_path, archive, paths, authorities)

    first = pilot.run(args)
    sens = (
        tmp_path
        / "pilot/source/scans/scene0001_00/scene0001_00.sens"
    )
    receipt_path = tmp_path / "pilot/receipts/00_sens_extraction.json"
    receipt = json.loads(receipt_path.read_text())
    frozen_receipt_bytes = receipt_path.read_bytes()
    assert sens.read_bytes() == b"sealed-scene0001_00"
    assert receipt["physical_space_id"] == "scene0001"
    assert receipt["inputs"]["archive_authority"]["member"] == (
        "scans/scene0001_00/scene0001_00.sens"
    )
    assert receipt["outputs"]["sens_payload"]["sha256"] == sha256_file(sens)
    assert receipt["source_access"] == pilot._source_access()
    assert first["completed_stage"] == "sens_extraction"

    second = pilot.run(args)
    assert second == first
    assert receipt_path.read_bytes() == frozen_receipt_bytes

    sens.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sens_payload file differs"):
        pilot.run(args)


def test_pilot_refuses_unreceipted_output_and_wrong_protocol_sha(tmp_path) -> None:
    archive, paths, authorities = _authorities(tmp_path)
    args = _args(tmp_path, archive, paths, authorities, root_name="unreceipted")
    target = (
        tmp_path
        / "unreceipted/source/scans/scene0001_00/scene0001_00.sens"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"untrusted")
    with pytest.raises(FileExistsError, match="unreceipted .sens"):
        pilot.run(args)

    wrong = _args(tmp_path, archive, paths, authorities, root_name="wrong_sha")
    wrong.expected_inventory_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 differs"):
        pilot.run(wrong)


def _write_v1_prefix(args: Namespace) -> tuple[dict, Path]:
    common = pilot.validate_pilot_authorities(args)
    root = args.pilot_root
    receipt_root = Path(root) / "receipts"
    receipt_root.mkdir(parents=True)
    feature_manifest = (
        Path(root)
        / "run/radio_features/scene0001_00/frame_manifest.json"
    )
    render_config = (
        Path(root)
        / "run/render_contracts/scene0001_00.yaml"
    )
    render_checkpoint = (
        Path(root)
        / "run/render_contracts/scene0001_00.geometry_renderer.pth"
    )
    for path in (feature_manifest, render_config, render_checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("ascii"))

    previous = None
    for index, stage in enumerate(pilot_v2.V1_PREFIX_STAGES):
        inputs = pilot._common_inputs(common, previous)
        outputs = {}
        if stage == "radio_features":
            outputs = {
                "feature_manifest": pilot._file_record(feature_manifest),
                "feature_output_bundle_sha256": "a" * 64,
                "num_frames": 1,
            }
        elif stage == "render_contract":
            outputs = {
                "render_config": pilot._file_record(render_config),
                "render_checkpoint": pilot._file_record(render_checkpoint),
            }
        path = receipt_root / f"{index:02d}_{stage}.json"
        pilot._write_stage_receipt(
            path,
            stage=stage,
            common=common,
            inputs=inputs,
            outputs=outputs,
            command=None,
        )
        previous = path
    args.expected_v1_prefix_receipt_sha256 = sha256_file(previous)
    return common, previous


def test_v2_continuation_binds_prefix_and_fixes_exact_marginal_command(
    tmp_path,
) -> None:
    archive, paths, authorities = _authorities(tmp_path)
    args = _args(tmp_path, archive, paths, authorities)
    args.stop_after = "factorized_radio"
    common, expected_prefix = _write_v1_prefix(args)

    observed_common, prefix = pilot_v2.validate_v1_prefix(args)
    assert observed_common == common
    assert prefix == expected_prefix
    inputs = pilot_v2.continuation_inputs(
        common, prefix, v1_prefix_receipt=prefix
    )
    assert inputs["v1_prefix_receipt"] == pilot._file_record(prefix)
    assert inputs["v2_continuation_launcher"]["sha256"] == sha256_file(
        pilot_v2.__file__
    )

    command = pilot_v2._mpr_command(
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
    alpha = command.index("--alpha-threshold")
    assert command[alpha : alpha + 2] == ["--alpha-threshold", "0"]


def test_v2_continuation_fails_closed_on_prefix_tamper_and_unreceipted_output(
    tmp_path,
) -> None:
    archive, paths, authorities = _authorities(tmp_path)
    args = _args(tmp_path, archive, paths, authorities)
    args.stop_after = "factorized_radio"
    _common, prefix = _write_v1_prefix(args)

    output = (
        Path(args.pilot_root)
        / "run/exact_state/scene0001_00/factorized_raw_radio.pt"
    )
    output.parent.mkdir(parents=True)
    output.write_bytes(b"unreceipted")
    with pytest.raises(FileExistsError, match="unreceipted outputs"):
        pilot_v2.run(args)

    output.unlink()
    prefix.write_text(prefix.read_text() + " ")
    with pytest.raises(ValueError, match="authority|chain|SHA-256|file differs"):
        pilot_v2.validate_v1_prefix(args)
