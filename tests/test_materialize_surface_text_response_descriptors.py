from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

import radio_gs.scripts.materialize_surface_text_response_descriptors as module
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _cache(path: Path, *, scene: str, radio_sha: str) -> None:
    record = {
        "region_id": f"region-{scene}",
        "scene": scene,
        "seed": 17,
        "physical_radius_m": 0.25,
        "teacher_views": [0, 1],
        "teacher_target_sha256": "a" * 64,
        "teacher_support_sha256": "b" * 64,
    }
    metadata = {
        "schema_version": 3,
        "split_role": "validation",
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "annotations_opened": False,
        "labels_opened": False,
        "instances_opened": False,
        "masks_opened": False,
        "text_opened": False,
        "physical_space_disjoint": True,
        "complete_scene_regions": True,
        "failed_scenes": [],
        "teacher_regions_saturated": 0,
        "region_records": [record],
        "scene_names": [scene],
        "scene_region_counts": {scene: 1},
        "region_contract_sha256": "c" * 64,
        "region_contract": {"name": "mock-surface-contract"},
        "radio_checkpoint_sha256": radio_sha,
        "split_file_sha256": "d" * 64,
        "teacher_region_semantics": (
            "fixed_core_geodesic_support_without_input_context_v1"
        ),
        "teacher_region_contract": {"name": "fixed-core"},
        "teacher_region_contract_sha256": "e" * 64,
        "teacher_target_source": "exact_cache_replay",
        "teacher_target_protocol_sha256": "f" * 64,
        "excluded_physical_spaces": ["heldout-space"],
        "exclusion_files": [{"path": "/frozen/split.txt", "sha256": "1" * 64}],
    }
    payload = {
        "radio_features": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        "geometry": torch.zeros(1, 2, 14),
        "token_mask": torch.tensor([[True, True]]),
        "reliability": torch.ones(1, 2),
        "official_crop_summaries": torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
        ),
        "teacher_mask": torch.tensor([[True, True]]),
        "anchor_index": torch.tensor([0]),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    radio = tmp_path / "radio.pt"
    radio.write_bytes(b"fixed-radio")
    cache = tmp_path / "validation_shard0.pt"
    _cache(cache, scene="scene-validation", radio_sha=_sha256(radio))
    _, cache_meta = module._load_validation_caches([cache])

    checkpoint = tmp_path / "candidate_text_response_seed0.pt"
    report = checkpoint.with_suffix(".pt.json")
    run_manifest = tmp_path / "run_manifest.json"
    materializer_relative = (
        "radio_gs/scripts/materialize_surface_text_response_descriptors.py"
    )
    run_payload = {
        "schema_version": 1,
        "artifact_type": "surface_region_text_response_distill_run",
        "candidate": "context_c1024_geometric",
        "validation_caches": cache_meta["cache_bindings"],
        "radio_checkpoint": {"path": str(radio), "sha256": _sha256(radio)},
        "outputs": [
            {"seed": 0, "checkpoint": str(checkpoint), "report": str(report)},
            {
                "seed": 1,
                "checkpoint": str(tmp_path / "candidate_text_response_seed1.pt"),
                "report": str(tmp_path / "candidate_text_response_seed1.pt.json"),
            },
            {
                "seed": 2,
                "checkpoint": str(tmp_path / "candidate_text_response_seed2.pt"),
                "report": str(tmp_path / "candidate_text_response_seed2.pt.json"),
            },
        ],
        "implementation_sources": {
            materializer_relative: _sha256(Path(module.__file__).resolve())
        },
    }
    _write_json(run_manifest, run_payload)

    model = SurfaceRegionSummaryReadoutV2(feature_dim=4, hidden_dim=2)
    architecture = model.architecture("c" * 64)
    baseline = {
        "summary_token_cosine": 0.4,
        "mean_descriptor_cosine": 0.4,
        "all_view_descriptor_cosine": 0.4,
    }
    validation = {
        "summary_token_cosine": 0.6,
        "mean_descriptor_cosine": 0.6,
        "all_view_descriptor_cosine": 0.6,
    }
    fit_binding = {"artifact_sha256": "2" * 64, "query_count": 806}
    provenance = {
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "scene_disjoint": True,
        "custom_text_projection": False,
        "train": {"scenes": ["scene-train"]},
        "validation": cache_meta["checkpoint_validation"],
        "region_contract_sha256": "c" * 64,
        "region_contract": {"name": "mock-surface-contract"},
        "random_seed_contract": {
            "seed": 0,
            "model_initialization": True,
            "data_order": True,
            "canonical_noise": True,
        },
        "text_response_distillation": {
            "response_lambda": 0.25,
            "calibration_manifest": str(tmp_path / "calibration.json"),
            "calibration_manifest_sha256": "3" * 64,
            "fit_text_bank": fit_binding,
        },
        "distill_run_manifest": {
            "path": str(run_manifest),
            "sha256": _sha256(run_manifest),
            "candidate": "context_c1024_geometric",
        },
    }
    checkpoint_payload = {
        "schema_version": 3,
        "architecture": architecture,
        "state_dict": model.state_dict(),
        "provenance": provenance,
        "training_config": {"seed": 0},
        "best_epoch": 1,
        "best_selection_score": 0.6,
        "untrained_baseline": baseline,
        "untrained_baseline_score": 0.4,
    }
    torch.save(checkpoint_payload, checkpoint)
    report_payload = {
        "output": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "architecture": architecture,
        "best_epoch": 1,
        "best_selection_score": 0.6,
        "untrained_baseline": baseline,
        "selection_score_delta": 0.6 - 0.4,
        "validation": validation,
        "response_lambda": 0.25,
        "calibration_manifest": str(tmp_path / "calibration.json"),
        "calibration_manifest_sha256": "3" * 64,
        "fit_text_bank_sha256": "2" * 64,
        "fit_query_count": 806,
        "distill_run_manifest": str(run_manifest),
        "distill_run_manifest_sha256": _sha256(run_manifest),
        "validation_caches": cache_meta["cache_bindings"],
        "train_scenes": 1,
        "validation_scenes": 1,
        "scene_overlap": [],
    }
    _write_json(report, report_payload)

    class IdentityHead:
        @classmethod
        def from_radio_checkpoint(cls, _: str) -> torch.nn.Module:
            return torch.nn.Identity()

    monkeypatch.setattr(module, "SigLIP2SummaryHead", IdentityHead)
    return {
        "radio": radio,
        "cache": cache,
        "checkpoint": checkpoint,
        "report": report,
        "run_manifest": run_manifest,
    }


def _args(paths: dict[str, Path], output: Path, cache: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        validation_cache=[str(cache or paths["cache"])],
        readout_checkpoint=str(paths["checkpoint"]),
        readout_binding_manifest=None,
        radio_checkpoint=str(paths["radio"]),
        method_id="distilled-candidate",
        batch_size=2,
        device="cpu",
        output=str(output),
    )


def _rebind_distill_manifest(paths: dict[str, Path]) -> None:
    manifest_sha = _sha256(paths["run_manifest"])
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    checkpoint["provenance"]["distill_run_manifest"]["sha256"] = manifest_sha
    torch.save(checkpoint, paths["checkpoint"])

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report["checkpoint_sha256"] = _sha256(paths["checkpoint"])
    report["distill_run_manifest_sha256"] = manifest_sha
    _write_json(paths["report"], report)


def _upgrade_to_authority_schema_v2(
    paths: dict[str, Path], tmp_path: Path
) -> Path:
    snapshot_root = tmp_path / "frozen-producer-snapshot"
    producer_source = snapshot_root / (
        "radio_gs/scripts/materialize_surface_text_response_descriptors.py"
    )
    producer_source.parent.mkdir(parents=True)
    producer_source.write_bytes(b"frozen producer materializer implementation\n")
    assert _sha256(producer_source) != _sha256(Path(module.__file__).resolve())

    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest.update(
        {
            "surface_promotion": {},
            "train_caches": [],
            "fit_text_bank": {},
            "calibration_manifest": {},
            "training_contract": {},
            "thermal_safety_contract": {},
            "authority_status": "query_free_three_seed_gpu1_run_frozen",
            "calibration_audit": {},
            "initial_gpu_preflight": {},
            "gpu_identity": {},
            "runtime_closure": {},
            "authority_contract": {"source_snapshot_root": str(snapshot_root)},
            "training_command_contract": {},
        }
    )
    extra_output_fields = {
        "training_log",
        "audit_report",
        "guard_command",
        "guard_telemetry",
        "guard_receipt",
        "kernel_journal",
        "gpu_preflight",
        "gpu_postflight",
        "terminal",
    }
    for row in manifest["outputs"]:
        for field in extra_output_fields:
            row[field] = str(tmp_path / f"seed{row['seed']}.{field}")
    relative = "radio_gs/scripts/materialize_surface_text_response_descriptors.py"
    manifest["implementation_sources"] = {relative: _sha256(producer_source)}
    _write_json(paths["run_manifest"], manifest)
    _rebind_distill_manifest(paths)
    return producer_source


def test_materializer_binds_exact_checkpoint_validation_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = module.materialize(_args(paths, tmp_path / "descriptors.pt"))
    payload = torch.load(result["output"], map_location="cpu", weights_only=True)

    provenance = payload["provenance"]
    assert provenance["readout_checkpoint_sha256"] == _sha256(paths["checkpoint"])
    assert provenance["readout_report_sha256"] == _sha256(paths["report"])
    assert provenance["readout_binding_authority"] == {
        "type": "embedded_distill_run_manifest",
        "path": str(paths["run_manifest"]),
        "sha256": _sha256(paths["run_manifest"]),
        "candidate": "context_c1024_geometric",
    }
    assert provenance["validation_caches"][0]["sha256"] == _sha256(paths["cache"])
    assert provenance["validation_scenes"] == ["scene-validation"]
    assert provenance["validation_split_sha256"] == "d" * 64


def test_materializer_accepts_authority_schema_v2_frozen_producer_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    producer_source = _upgrade_to_authority_schema_v2(paths, tmp_path)

    result = module.materialize(_args(paths, tmp_path / "descriptors.pt"))
    payload = torch.load(result["output"], map_location="cpu", weights_only=True)

    assert _sha256(producer_source) != _sha256(Path(module.__file__).resolve())
    assert payload["provenance"]["readout_binding_authority"] == {
        "type": "embedded_distill_run_manifest",
        "path": str(paths["run_manifest"]),
        "sha256": _sha256(paths["run_manifest"]),
        "candidate": "context_c1024_geometric",
    }


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_materializer_rejects_authority_schema_v2_output_field_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _upgrade_to_authority_schema_v2(paths, tmp_path)
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    if mutation == "missing":
        manifest["outputs"][0].pop("guard_receipt")
    else:
        manifest["outputs"][0]["unexpected"] = str(tmp_path / "unexpected")
    _write_json(paths["run_manifest"], manifest)
    _rebind_distill_manifest(paths)

    with pytest.raises(ValueError, match="output index fields differ"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_authority_schema_v2_producer_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _upgrade_to_authority_schema_v2(paths, tmp_path)
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    relative = "radio_gs/scripts/materialize_surface_text_response_descriptors.py"
    manifest["implementation_sources"][relative] = "0" * 64
    _write_json(paths["run_manifest"], manifest)
    _rebind_distill_manifest(paths)

    with pytest.raises(ValueError, match="producer materializer implementation"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_authority_schema_v2_symlinked_snapshot_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    producer_source = _upgrade_to_authority_schema_v2(paths, tmp_path)
    snapshot_link = tmp_path / "snapshot-link"
    snapshot_link.symlink_to(producer_source.parents[2], target_is_directory=True)
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    manifest["authority_contract"]["source_snapshot_root"] = str(snapshot_link)
    _write_json(paths["run_manifest"], manifest)
    _rebind_distill_manifest(paths)

    with pytest.raises(ValueError, match="canonical non-symlink path"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_authority_schema_v2_top_level_field_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _upgrade_to_authority_schema_v2(paths, tmp_path)
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    _write_json(paths["run_manifest"], manifest)
    _rebind_distill_manifest(paths)

    with pytest.raises(ValueError, match="run-manifest fields differ"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_same_contract_different_scene_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    substitute = tmp_path / "substitute_validation.pt"
    _cache(substitute, scene="scene-substitute", radio_sha=_sha256(paths["radio"]))

    with pytest.raises(ValueError, match="provided validation caches differ"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt", substitute))


def test_materializer_rejects_in_place_validation_cache_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload = torch.load(paths["cache"], map_location="cpu", weights_only=True)
    payload["radio_features"][0, 0, 0] = 9.0
    torch.save(payload, paths["cache"])

    with pytest.raises(ValueError, match="provided validation caches differ"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_checkpoint_without_cache_sha_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    checkpoint["provenance"]["validation"].pop("cache_bindings")
    torch.save(checkpoint, paths["checkpoint"])

    with pytest.raises(ValueError, match="provided validation caches differ"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_checkpoint_report_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report["checkpoint_sha256"] = "0" * 64
    _write_json(paths["report"], report)

    with pytest.raises(ValueError, match="checkpoint_sha256 binding differs"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_refuses_to_overwrite_descriptor_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "descriptors.pt"
    module.materialize(_args(paths, output))
    with pytest.raises(FileExistsError, match="already exists"):
        module.materialize(_args(paths, output))


def _make_legacy_bundle(paths: dict[str, Path], tmp_path: Path) -> Path:
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    checkpoint["provenance"]["validation"].pop("cache_bindings")
    checkpoint["provenance"].pop("text_response_distillation")
    checkpoint["provenance"].pop("distill_run_manifest")
    torch.save(checkpoint, paths["checkpoint"])
    baseline = checkpoint["untrained_baseline"]
    report = {
        "output": str(paths["checkpoint"]),
        "checkpoint_sha256": _sha256(paths["checkpoint"]),
        "architecture": checkpoint["architecture"],
        "best_epoch": checkpoint["best_epoch"],
        "best_selection_score": checkpoint["best_selection_score"],
        "untrained_baseline": baseline,
        "selection_score_delta": checkpoint["best_selection_score"]
        - 0.5
        * (
            baseline["mean_descriptor_cosine"]
            + baseline["all_view_descriptor_cosine"]
        ),
        "validation": {
            "summary_token_cosine": 0.6,
            "mean_descriptor_cosine": 0.6,
            "all_view_descriptor_cosine": 0.6,
        },
        "train_scenes": 1,
        "validation_scenes": 1,
        "scene_overlap": [],
    }
    _write_json(paths["report"], report)
    cache_sidecar = paths["cache"].with_suffix(".pt.json")
    _write_json(cache_sidecar, {"output": str(paths["cache"])})
    anchor = paths["run_manifest"]
    anchor_binding = {"path": str(anchor), "sha256": _sha256(anchor)}
    candidate = "context_c1024_geometric"
    readout = {
        "candidate": candidate,
        "seed": 0,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": _sha256(paths["checkpoint"]),
        "sidecar": str(paths["report"]),
        "sidecar_sha256": _sha256(paths["report"]),
    }
    selected_readouts = [
        readout,
        {
            **readout,
            "seed": 1,
            "checkpoint": str(tmp_path / "seed1.pt"),
            "checkpoint_sha256": "4" * 64,
            "sidecar": str(tmp_path / "seed1.pt.json"),
            "sidecar_sha256": "5" * 64,
        },
        {
            **readout,
            "seed": 2,
            "checkpoint": str(tmp_path / "seed2.pt"),
            "checkpoint_sha256": "6" * 64,
            "sidecar": str(tmp_path / "seed2.pt.json"),
            "sidecar_sha256": "7" * 64,
        },
    ]
    bundle = {
        "schema_version": 1,
        "artifact_type": "surface_region_query_free_three_seed_bundle",
        "status": "query_free_three_seed_bundle_frozen_benchmark_gate_closed",
        "selected_candidate": candidate,
        "seed_selection_policy": "all_required_seeds_no_single_seed_selection",
        "required_seeds": [0, 1, 2],
        "selected_readouts": selected_readouts,
        "benchmark_gate": {
            "status": "closed_not_evaluated",
            "main_result_eligible": False,
        },
        "bindings": {
            "finalizer": anchor_binding,
            "run_manifest": anchor_binding,
            "cache_pairing": anchor_binding,
            "query_free_screen": anchor_binding,
            "screen_completion": anchor_binding,
            "all_compared_readouts": selected_readouts,
            "caches": [
                {
                    "candidate": candidate,
                    "role": "validation",
                    "shard": 0,
                    "path": str(paths["cache"]),
                    "sha256": _sha256(paths["cache"]),
                    "sidecar": str(cache_sidecar),
                    "sidecar_sha256": _sha256(cache_sidecar),
                }
            ],
        },
    }
    bundle_path = tmp_path / "query_free_promotion_bundle.json"
    _write_json(bundle_path, bundle)
    completion = {
        "schema_version": 1,
        "artifact_type": "surface_region_query_free_promotion_completion",
        "status": "complete_benchmark_gate_closed",
        "promotion_manifest": str(bundle_path),
        "promotion_manifest_sha256": _sha256(bundle_path),
        "selected_candidate": candidate,
        "required_seeds": [0, 1, 2],
        "benchmark_gate_status": "closed_not_evaluated",
        "main_result_eligible": False,
    }
    _write_json(tmp_path / "query_free_promotion.complete.json", completion)
    return bundle_path


def test_materializer_allows_legacy_selected_seed_only_through_promotion_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    bundle = _make_legacy_bundle(paths, tmp_path)
    args = _args(paths, tmp_path / "legacy-descriptors.pt")
    args.readout_binding_manifest = str(bundle)

    result = module.materialize(args)
    payload = torch.load(result["output"], map_location="cpu", weights_only=True)
    authority = payload["provenance"]["readout_binding_authority"]
    assert authority["type"] == "query_free_promotion_bundle"
    assert authority["sha256"] == _sha256(bundle)
    assert authority["candidate"] == "context_c1024_geometric"


def test_materializer_rejects_legacy_bundle_wrong_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    bundle_path = _make_legacy_bundle(paths, tmp_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["selected_candidate"] = "control_c256_geometric"
    _write_json(bundle_path, bundle)
    completion_path = tmp_path / "query_free_promotion.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["selected_candidate"] = "control_c256_geometric"
    completion["promotion_manifest_sha256"] = _sha256(bundle_path)
    _write_json(completion_path, completion)
    args = _args(paths, tmp_path / "legacy-descriptors.pt")
    args.readout_binding_manifest = str(bundle_path)

    with pytest.raises(ValueError, match="selected candidate/seed"):
        module.materialize(args)


def test_materializer_rejects_legacy_bundle_wrong_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    bundle_path = _make_legacy_bundle(paths, tmp_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["bindings"]["all_compared_readouts"][0]["seed"] = 1
    _write_json(bundle_path, bundle)
    completion_path = tmp_path / "query_free_promotion.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["promotion_manifest_sha256"] = _sha256(bundle_path)
    _write_json(completion_path, completion)
    args = _args(paths, tmp_path / "legacy-descriptors.pt")
    args.readout_binding_manifest = str(bundle_path)

    with pytest.raises(ValueError, match="selected candidate/seed"):
        module.materialize(args)


def test_materializer_rejects_legacy_completion_hash_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    bundle_path = _make_legacy_bundle(paths, tmp_path)
    completion_path = tmp_path / "query_free_promotion.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["promotion_manifest_sha256"] = "0" * 64
    _write_json(completion_path, completion)
    args = _args(paths, tmp_path / "legacy-descriptors.pt")
    args.readout_binding_manifest = str(bundle_path)

    with pytest.raises(ValueError, match="does not bind"):
        module.materialize(args)


def test_materializer_forbids_distilled_checkpoint_legacy_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    args = _args(paths, tmp_path / "descriptors.pt")
    args.readout_binding_manifest = str(paths["run_manifest"])

    with pytest.raises(ValueError, match="cannot use the legacy"):
        module.materialize(args)
