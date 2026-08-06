from __future__ import annotations

from argparse import Namespace
import io
import tarfile

from radio_gs.scripts import build_full_scalar_scannet_clean_cohort as builder
from radio_gs.scripts import train_surface_region_full_scalar_residual as trainer
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
    write_frozen_json,
)


def _freeze(path, payload: dict) -> tuple[str, str]:
    write_frozen_json(path, payload)
    return str(path), sha256_file(path)


def _materialization_report(scene_id: str) -> dict:
    return {
        "valid": True,
        "uses_instances_or_semantic_labels": False,
        "uses_private_anchor": False,
        "uses_private_depth_pixel": False,
        "scenes": [{"scene_id": scene_id}],
    }


def _write_header_inventory_tar(path) -> None:
    with tarfile.open(path, mode="w") as handle:
        for physical_index in range(41):
            scene_ids = [f"scene{physical_index:04d}_00"]
            if physical_index == 2:
                scene_ids.append(f"scene{physical_index:04d}_01")
            for scene_id in scene_ids:
                payload = scene_id.encode("ascii")
                info = tarfile.TarInfo(f"scans/{scene_id}/{scene_id}.sens")
                info.size = len(payload)
                handle.addfile(info, io.BytesIO(payload))


def test_builder_seals_physical_disjoint_24_8_without_materializing_payloads(
    tmp_path,
) -> None:
    archive = tmp_path / "scans.tar.part-00"
    _write_header_inventory_tar(archive)
    agile0 = _freeze(
        tmp_path / "agile0.json", _materialization_report("scene0100_00")
    )
    agile1 = _freeze(
        tmp_path / "agile1.json", _materialization_report("scene0101_00")
    )
    pfpr = _freeze(
        tmp_path / "pfpr.json", _materialization_report("scene0100_00")
    )
    # NVOS legitimately has more than one prompt task for a single scene.
    nvos = _freeze(
        tmp_path / "nvos.json",
        {
            "scenes": [
                {"base_scene_id": "horns"},
                {"base_scene_id": "horns"},
                {"base_scene_id": "fern"},
            ]
        },
    )
    spin = _freeze(
        tmp_path / "spin.json",
        {"scenes": [{"scene_id": "lego"}, {"scene_id": "truck"}]},
    )
    outputs = {
        "benchmark_registry_output": tmp_path / "registry.json",
        "exclusion_manifest_output": tmp_path / "exclusion.json",
        "cohort_authority_output": tmp_path / "cohort.json",
        "inventory_output": tmp_path / "inventory.json",
    }
    result = builder.build(
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
            additional_scannet_benchmark_scene_id=["scene0000_01"],
            lerf_scene_id=["figurines", "ramen"],
            **outputs,
        )
    )

    registry, _, _ = load_json_object(
        outputs["benchmark_registry_output"],
        expected_sha256=result["benchmark_registry"]["sha256"],
        label="synthetic benchmark registry",
    )
    assert registry["dataset_scene_ids"]["nvos"] == ["fern", "horns"]
    assert "scene0000" in registry["scannet_physical_space_ids"]

    exclusion, _, _ = load_json_object(
        outputs["exclusion_manifest_output"],
        expected_sha256=result["benchmark_exclusion_manifest"]["sha256"],
        label="synthetic benchmark exclusion",
    )
    trainer.validate_benchmark_exclusion_manifest(exclusion)

    cohort, _, _ = load_json_object(
        outputs["cohort_authority_output"],
        expected_sha256=result["cohort_authority"]["sha256"],
        label="synthetic cohort authority",
    )
    trainer.validate_cohort_authority_payload(cohort)
    assert len(cohort["source_train_scene_ids"]) == 24
    assert len(cohort["source_validation_scene_ids"]) == 8
    assert "scene0000_00" not in (
        cohort["source_train_scene_ids"]
        + cohort["source_validation_scene_ids"]
    )
    assert "scene0002_00" in cohort["source_train_scene_ids"]
    assert "scene0002_01" not in (
        cohort["source_train_scene_ids"]
        + cohort["source_validation_scene_ids"]
    )

    inventory, _, _ = load_json_object(
        outputs["inventory_output"],
        expected_sha256=result["inventory"]["sha256"],
        label="synthetic archive inventory",
    )
    assert inventory["archive"]["payload_content_opened"] is False
    assert inventory["archive"]["member_headers_opened"] is True
    assert len(inventory["selected_records"]) == 32
    assert set(inventory["materialization_status"].values()) == {False}
    assert not list(tmp_path.glob("scans/scene*/*.sens"))
