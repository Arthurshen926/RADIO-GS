import argparse
import json
import subprocess
import sys

import pytest
import torch

from radio_gs.v4.carrier import Camera
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.contracts.build_surface_scene_bundle import run as build_bundle
from radio_gs.v4.contracts.surface_scene_bundle import (
    COMPLETION_RECEIPT_SCHEMA,
    ElementTokenObservedEvidence,
    SurfaceCarrierConfiguration,
    SurfaceSceneBundle,
    cold_load_projection_digest,
    load_geometry_binding,
    projection_digest,
)
from radio_gs.v4.object_memory import DenseObjectAssignments


def _geometry_files(tmp_path, *, radius=1, outer_radius=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    surface_path = tmp_path / "surface.pt"
    torch.save({
        "centres": torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        "normals": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        "confidence": torch.tensor([1.0, 0.8]),
        "voxel_size_colmap": 0.04,
    }, surface_path)
    projection = {
        "maximum_splat_radius": outer_radius if outer_radius is not None else radius,
        "surface_band_voxels": 1.5,
        "maximum_contributors_per_pixel": 8,
    }
    receipt = {
        "schema": "radio_gs.surface_object_memory_v4.geometry_receipt.v1",
        "carrier": "calibrated_moge3_sparse_surface",
        "coordinate_convention": "colmap_world_opencv_camera_pixel_centres",
        "inputs": [{
            "role": "surface_carrier",
            "path": str(surface_path),
            "sha256": sha256_file(surface_path),
        }],
        "source_rgb_opened": True,
        "target_rgb_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "model_family": None,
        "model_checkpoint_sha256": None,
        "calibration": {},
        "metadata": {
            "scene_label": "scene",
            "maximum_splat_radius": radius,
            "surface_band_voxels": 1.5,
            "maximum_contributors_per_pixel": 8,
        },
    }
    authority_path = tmp_path / "geometry.json"
    authority_path.write_text(json.dumps({
        "passes_scene_gate": True,
        "geometry_receipt": receipt,
        "projection_configuration": projection,
    }))
    return authority_path, surface_path


def _bundle(tmp_path):
    authority_path, surface_path = _geometry_files(tmp_path)
    binding, payload = load_geometry_binding(authority_path, surface_path)
    observed = DenseObjectAssignments(
        torch.tensor([[0.8, 0.0], [0.2, 0.5]]),
        torch.tensor([0.2, 0.3]),
    )
    evidence = ElementTokenObservedEvidence(
        positive=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        negative=torch.zeros(2, 2),
        unknown=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        view_count=torch.tensor([[2, 0], [1, 1]]),
        quality=torch.tensor([[0.9, 0.0], [0.8, 0.7]]),
    )
    bundle = SurfaceSceneBundle(
        scene_label="scene",
        configuration=binding.configuration,
        centres=payload["centres"],
        normals=payload["normals"],
        confidence=payload["confidence"],
        observed_assignment=observed,
        observed_evidence=evidence,
        local_surface_memory={"observed": torch.tensor([True, True])},
        object_memory={"descriptor": torch.eye(2)},
        source_frames=(1, 2),
        source_input_digests={
            "geometry_authority": binding.authority_sha256,
            "surface_carrier": binding.surface_carrier_sha256,
        },
        geometry_authority_sha256=binding.authority_sha256,
        source_surface_carrier_sha256=binding.surface_carrier_sha256,
        information_policy={
            "target_rgb_opened_during_construction": False,
            "benchmark_labels_opened_during_construction": False,
            "text_queries_opened_during_construction": False,
        },
    )
    return bundle


def _learned_completion_receipt():
    return {
        "schema": COMPLETION_RECEIPT_SCHEMA,
        "method_family": "scene_disjoint_learned_completion_mlp",
        "checkpoint_sha256": "1" * 64,
        "training_report_sha256": "2" * 64,
        "learned_model": True,
        "training_scenes_disjoint_from_evaluation": True,
        "writes_unknown_only": True,
        "observed_known_clamped": True,
        "completion_confidence_cap": 0.8,
    }


def test_geometry_binding_uses_receipt_configuration_and_fails_on_drift(tmp_path):
    authority_path, surface_path = _geometry_files(tmp_path)
    binding, _ = load_geometry_binding(authority_path, surface_path)
    assert binding.configuration.maximum_splat_radius == 1
    assert binding.configuration.surface_band_voxels == 1.5
    assert binding.configuration.maximum_contributors_per_pixel == 8

    bad_authority, bad_surface = _geometry_files(
        tmp_path / "bad", radius=1, outer_radius=3
    )
    try:
        load_geometry_binding(bad_authority, bad_surface)
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("projection configuration drift was accepted")

    gate_payload = json.loads(authority_path.read_text())
    gate_payload["passes_scene_gate"] = False
    authority_path.write_text(json.dumps(gate_payload))
    with pytest.raises(ValueError, match="passes_scene_gate"):
        load_geometry_binding(authority_path, surface_path)


def test_surface_scene_bundle_cold_load_preserves_projection(tmp_path):
    bundle = _bundle(tmp_path)
    path = tmp_path / "scene.pt"
    file_digest = bundle.save(path)
    camera = Camera(
        "source",
        torch.tensor([[20.0, 0.0, 2.0], [0.0, 20.0, 2.0], [0.0, 0.0, 1.0]]),
        torch.eye(4),
        4,
        4,
    )
    expected = projection_digest(bundle.build_carrier().project(camera))
    assert cold_load_projection_digest(
        path,
        camera,
        expected_bundle_sha256=file_digest,
        expected_projection_sha256=expected,
    ) == expected
    loaded = SurfaceSceneBundle.load(path, expected_sha256=file_digest)
    assert torch.equal(
        loaded.observed_assignment.token_probability,
        bundle.observed_assignment.token_probability,
    )


def test_scene_bundle_verifier_runs_in_a_fresh_process(tmp_path):
    bundle = _bundle(tmp_path)
    path = tmp_path / "scene.pt"
    digest = bundle.save(path)
    camera_path = tmp_path / "camera.pt"
    torch.save({
        "key": "source",
        "intrinsic": torch.tensor([
            [20.0, 0.0, 2.0], [0.0, 20.0, 2.0], [0.0, 0.0, 1.0]
        ]),
        "camera_to_world": torch.eye(4),
        "height": 4,
        "width": 4,
    }, camera_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "radio_gs.v4.contracts.verify_surface_scene_bundle",
            "--scene-bundle",
            str(path),
            "--expected-sha256",
            digest,
            "--camera",
            str(camera_path),
            "--expected-projection-sha256",
            projection_digest(bundle.build_carrier().project(Camera(
                "source",
                torch.tensor([
                    [20.0, 0.0, 2.0], [0.0, 20.0, 2.0], [0.0, 0.0, 1.0]
                ]),
                torch.eye(4),
                4,
                4,
            ))),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["projection_sha256"]
    assert report["projection_matches_expected"] is True
    assert report["query_inputs_opened"] is False
    assert report["benchmark_inputs_opened"] is False


def test_scene_bundle_rejects_unreceipted_or_heuristic_completion(tmp_path):
    bundle = _bundle(tmp_path)
    completed = DenseObjectAssignments(
        bundle.observed_assignment.token_probability,
        bundle.observed_assignment.unknown_probability,
    )
    values = dict(bundle.__dict__)
    values["completed_assignment"] = completed
    values["completion_receipt"] = {
        "schema": COMPLETION_RECEIPT_SCHEMA,
        "method_family": "geometry_envelope_heuristic",
        "checkpoint_sha256": "1" * 64,
        "training_report_sha256": "2" * 64,
        "learned_model": True,
        "training_scenes_disjoint_from_evaluation": True,
        "writes_unknown_only": True,
        "observed_known_clamped": True,
        "completion_confidence_cap": 0.8,
    }
    try:
        SurfaceSceneBundle(**values)
    except ValueError as error:
        assert "learned completion" in str(error)
    else:
        raise AssertionError("heuristic completion entered a formal scene bundle")


def test_scene_bundle_allows_completion_only_on_unknown_pairs_and_clamps_known(tmp_path):
    bundle = _bundle(tmp_path)
    values = dict(bundle.__dict__)
    values["completed_assignment"] = DenseObjectAssignments(
        torch.tensor([[0.8, 0.1], [0.2, 0.5]]),
        torch.tensor([0.1, 0.3]),
    )
    values["completion_receipt"] = _learned_completion_receipt()
    completed = SurfaceSceneBundle(**values)
    assert completed.completed_assignment.token_probability[0, 1] == pytest.approx(0.1)

    values["completed_assignment"] = DenseObjectAssignments(
        torch.tensor([[0.7, 0.2], [0.2, 0.5]]),
        torch.tensor([0.1, 0.3]),
    )
    with pytest.raises(ValueError, match="strictly clamp"):
        SurfaceSceneBundle(**values)


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("local_surface_memory", {}, "non-empty"),
        ("local_surface_memory", {"target_label": torch.ones(2)}, "forbidden"),
        ("object_memory", {"descriptor": torch.zeros(3, 2)}, "object_memory.descriptor"),
        ("object_memory", {"target_id": torch.zeros(2)}, "forbidden"),
    ],
)
def test_scene_bundle_rejects_empty_misaligned_or_target_memory(
    tmp_path, field, value, expected
):
    values = dict(_bundle(tmp_path).__dict__)
    values[field] = value
    with pytest.raises(ValueError, match=expected):
        SurfaceSceneBundle(**values)


def test_scene_bundle_full_state_digest_detects_semantic_tampering(tmp_path):
    bundle = _bundle(tmp_path)
    path = tmp_path / "scene.pt"
    digest = bundle.save(path)
    with pytest.raises(TypeError):
        SurfaceSceneBundle.load(path)
    assert SurfaceSceneBundle.load(path, expected_sha256=digest).content_sha256 == bundle.content_sha256

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["object_memory"]["descriptor"][0, 0] = 99
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="full-state content digest"):
        SurfaceSceneBundle.load(tampered, expected_sha256=sha256_file(tampered))


@pytest.mark.parametrize(
    "radius,cap",
    [(1.5, 8), (1, 8.5), (True, 8), (1, False)],
)
def test_surface_configuration_rejects_non_exact_integer_fields(radius, cap):
    with pytest.raises(ValueError, match="integer"):
        SurfaceCarrierConfiguration(
            0.04,
            radius,
            1.5,
            cap,
            "colmap_world_opencv_camera_pixel_centres",
        )


def test_surface_configuration_rejects_unknown_camera_convention():
    with pytest.raises(ValueError, match="allowlist"):
        SurfaceCarrierConfiguration(0.04, 1, 1.5, 8, "opencv-ish")


def test_element_token_evidence_requires_simplex_and_views_for_known_facts():
    with pytest.raises(ValueError, match="simplex"):
        ElementTokenObservedEvidence(
            positive=torch.tensor([[0.8]]),
            negative=torch.tensor([[0.0]]),
            unknown=torch.tensor([[0.1]]),
            view_count=torch.tensor([[1]]),
            quality=torch.tensor([[0.8]]),
        )
    with pytest.raises(ValueError, match="positive view_count"):
        ElementTokenObservedEvidence(
            positive=torch.tensor([[1.0]]),
            negative=torch.tensor([[0.0]]),
            unknown=torch.tensor([[0.0]]),
            view_count=torch.tensor([[0]]),
            quality=torch.tensor([[0.8]]),
        )


def test_query_free_builder_uses_geometry_authority_without_projection_cli_defaults(tmp_path):
    authority_path, surface_path = _geometry_files(tmp_path)
    source_memory_path = tmp_path / "source_memory.pt"
    torch.save({
        "observed_assignment": {
            "token_probability": torch.tensor([[0.8, 0.0], [0.2, 0.5]]),
            "unknown_probability": torch.tensor([0.2, 0.3]),
        },
        "observed_evidence": {
            "positive": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            "negative": torch.zeros(2, 2),
            "unknown": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            "view_count": torch.tensor([[2, 0], [1, 1]]),
            "quality": torch.tensor([[0.9, 0.0], [0.8, 0.7]]),
        },
        "local_surface_memory": {"observed": torch.tensor([True, True])},
        "object_memory": {"descriptor": torch.eye(2)},
        "source_frames": [1, 2],
        "source_input_digests": {"source_rgb": "3" * 64},
        "information_policy": {
            "target_rgb_opened_during_construction": False,
            "benchmark_labels_opened_during_construction": False,
            "text_queries_opened_during_construction": False,
        },
    }, source_memory_path)
    output = tmp_path / "bundle.pt"
    report = build_bundle(argparse.Namespace(
        scene_label="scene",
        geometry_authority=str(authority_path),
        surface_carrier=str(surface_path),
        source_memory=str(source_memory_path),
        sealed_input=None,
        verify_all_geometry_inputs=False,
        output=str(output),
        receipt_output=None,
    ))
    assert report["carrier_configuration"]["maximum_splat_radius"] == 1
    assert report["completion_present"] is False
    assert SurfaceSceneBundle.load(
        output, expected_sha256=report["scene_bundle_sha256"]
    ).configuration.maximum_splat_radius == 1
