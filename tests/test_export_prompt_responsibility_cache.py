from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from radio_gs.interfaces.prompt_responsibility_cache import (
    COMPOSITOR_CONTRACT,
    PromptResponsibilityAuthority,
)
from radio_gs.scripts import export_prompt_responsibility_cache as exporter


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def test_commit_local_artifact_is_complete_and_fail_closed(tmp_path: Path):
    local = tmp_path / "local.pt"
    local.write_bytes((b"exact-W" * 1024) + b"tail")
    remote = tmp_path / "remote" / "cache.pt"
    exporter._commit_local_artifact(local, remote, overwrite=False)
    assert remote.read_bytes() == local.read_bytes()
    with pytest.raises(FileExistsError):
        exporter._commit_local_artifact(local, remote, overwrite=False)


class _HeaderOnlyImage:
    """Any attempt to cross the PIL lazy-header boundary fails the test."""

    def __init__(self, size):
        self.size = size

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def load(self):
        raise AssertionError("mask pixels must not be loaded")

    def convert(self, *_args, **_kwargs):
        raise AssertionError("mask pixels must not be converted")

    def getdata(self):
        raise AssertionError("mask pixels must not be accessed")

    def __array__(self, *_args, **_kwargs):
        raise AssertionError("mask pixels must not be materialized")


def _full_mask_scene(path: Path) -> dict:
    return {
        "scene_id": "lego",
        "prompt_frame_ids": ["0_00001"],
        "prompt": {
            "type": "reference_binary_mask",
            "frame_id": "0_00001",
            "mask_path": str(path),
        },
    }


def _full_mask_spec(tmp_path, monkeypatch):
    path = tmp_path / "0_00001.png"
    payload = b"opaque source file bytes; never decode me"
    path.write_bytes(payload)
    monkeypatch.setattr(exporter.Image, "open", lambda _path: _HeaderOnlyImage((1015, 764)))
    digest = hashlib.sha256(payload).hexdigest()
    spec = exporter._native_prompt_header_authority(
        _full_mask_scene(path),
        expected_prompt_type="reference_binary_mask",
        expected_reference_frame_id="0_00001",
        expected_reference_mask_sha256=digest,
        expected_native_height=764,
        expected_native_width=1015,
    )
    return spec, digest


def _responsibility_authority(spec, source_sha256):
    return PromptResponsibilityAuthority(
        scene_id="lego",
        frame_id=spec.reference_frame_id,
        camera_name="0_00001",
        colmap_camera_name="0_00001",
        geometry_checkpoint_sha256=source_sha256["geometry_checkpoint"],
        geometry_xyz_sha256=SHA_B,
        pose_sha256=SHA_C,
        intrinsics_sha256=SHA_D,
        height=spec.height,
        width=spec.width,
        num_gaussians=3,
        alpha_threshold=0.0,
        compositor_contract=COMPOSITOR_CONTRACT,
        target_rgb_opened=False,
        target_mask_opened=False,
        source_sha256=source_sha256,
    )


def test_reference_mask_is_header_and_raw_hash_only_and_binds_lego_native_shape(
    tmp_path, monkeypatch
):
    spec, digest = _full_mask_spec(tmp_path, monkeypatch)
    assert spec.prompt_type == "reference_binary_mask"
    assert spec.reference_frame_id == "0_00001"
    assert (spec.width, spec.height) == (1015, 764)
    assert set(spec.paths) == {"reference_binary_mask"}
    assert spec.source_sha256 == {"reference_binary_mask": digest}


def test_reference_mask_report_binds_prompt_raster_source_and_no_target_flags(
    tmp_path, monkeypatch
):
    spec, digest = _full_mask_spec(tmp_path, monkeypatch)
    sources = {
        "benchmark_manifest": SHA_A,
        "camera_mapping": SHA_B,
        "contribution_compositor_source": SHA_C,
        "exporter_source": SHA_D,
        "gaussfm_config": SHA_A,
        "geometry_checkpoint": SHA_A,
        "reference_binary_mask": digest,
    }
    authority = _responsibility_authority(spec, sources)
    metadata = exporter._reference_mask_report_metadata(spec, sources, authority)
    assert metadata["prompt_type"] == "reference_binary_mask"
    assert metadata["reference_frame_id"] == "0_00001"
    assert (metadata["native_width"], metadata["native_height"]) == (1015, 764)
    assert metadata["source_sha256_key"] == "reference_binary_mask"
    assert metadata["reference_binary_mask_sha256"] == digest
    for key in (
        "source_mask_pixels_decoded",
        "source_mask_pixels_interpreted",
        "query_or_evidence_constructed",
        "target_rgb_opened",
        "target_mask_opened",
        "target_metric_computed",
    ):
        assert metadata[key] is False
    assert set(metadata["geometry_camera_implementation_bindings"]) == {
        "camera_mapping_sha256",
        "contribution_compositor_source_sha256",
        "exporter_source_sha256",
        "gaussfm_config_sha256",
        "geometry_checkpoint_sha256",
        "geometry_xyz_sha256",
        "intrinsics_sha256",
        "pose_sha256",
    }


def test_legacy_scribble_shape_paths_and_order_are_unchanged(tmp_path, monkeypatch):
    positive = tmp_path / "positive.png"
    negative = tmp_path / "negative.png"
    positive.write_bytes(b"positive")
    negative.write_bytes(b"negative")
    opened = []

    def header(path):
        opened.append(Path(path).name)
        return _HeaderOnlyImage((640, 480))

    monkeypatch.setattr(exporter.Image, "open", header)
    scene = {
        "prompt_frame_ids": ["image001"],
        "prompt": {
            "type": "positive_negative_scribbles",
            "frame_id": "image001",
            "positive_path": str(positive),
            "negative_path": str(negative),
        },
    }
    legacy = exporter._native_prompt_shape(scene)
    current = exporter._native_prompt_header_authority(scene)
    assert (current.height, current.width, current.paths) == legacy
    assert list(current.paths) == ["positive_scribble", "negative_scribble"]
    assert current.source_sha256 is None
    assert opened == ["positive.png", "negative.png"] * 2


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"expected_prompt_type": None}, "expected-prompt-type"),
        ({"expected_prompt_type": "positive_negative_scribbles"}, "prompt type differs"),
        ({"expected_reference_frame_id": "other"}, "frame id differs"),
        ({"expected_reference_mask_sha256": "A" * 64}, "lowercase SHA-256"),
        ({"expected_reference_mask_sha256": "0" * 64}, "file SHA-256 differs"),
        ({"expected_native_height": 763}, "native shape differs"),
        ({"expected_native_width": 1014}, "native shape differs"),
    ],
)
def test_reference_mask_type_frame_shape_and_hash_fail_closed(
    tmp_path, monkeypatch, change, match
):
    path = tmp_path / "0_00001.png"
    payload = b"header authority bytes"
    path.write_bytes(payload)
    monkeypatch.setattr(exporter.Image, "open", lambda _path: _HeaderOnlyImage((1015, 764)))
    kwargs = {
        "expected_prompt_type": "reference_binary_mask",
        "expected_reference_frame_id": "0_00001",
        "expected_reference_mask_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_native_height": 764,
        "expected_native_width": 1015,
    }
    kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        exporter._native_prompt_header_authority(_full_mask_scene(path), **kwargs)


def test_unknown_prompt_type_and_manifest_frame_disagreement_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="unsupported prompt type"):
        exporter._native_prompt_header_authority({"prompt": {"type": "points"}})
    path = tmp_path / "mask.png"
    path.write_bytes(b"x")
    scene = _full_mask_scene(path)
    scene["prompt_frame_ids"] = ["wrong"]
    with pytest.raises(ValueError, match="exactly one matching prompt frame"):
        exporter._native_prompt_header_authority(
            scene,
            expected_prompt_type="reference_binary_mask",
            expected_reference_frame_id="0_00001",
            expected_reference_mask_sha256=hashlib.sha256(b"x").hexdigest(),
            expected_native_height=764,
            expected_native_width=1015,
        )


def test_report_rejects_sha_shape_or_prompt_type_drift(tmp_path, monkeypatch):
    spec, digest = _full_mask_spec(tmp_path, monkeypatch)
    sources = {
        "camera_mapping": SHA_B,
        "contribution_compositor_source": SHA_C,
        "exporter_source": SHA_D,
        "gaussfm_config": SHA_A,
        "geometry_checkpoint": SHA_A,
        "reference_binary_mask": digest,
    }
    authority = _responsibility_authority(spec, sources)
    with pytest.raises(ValueError, match="source SHA binding changed"):
        exporter._reference_mask_report_metadata(
            spec, {**sources, "reference_binary_mask": "f" * 64}, authority
        )
    with pytest.raises(ValueError, match="raster/frame authority changed"):
        exporter._reference_mask_report_metadata(
            spec, sources, replace(authority, width=1014)
        )
    with pytest.raises(ValueError, match="requires reference_binary_mask"):
        exporter._reference_mask_report_metadata(
            replace(spec, prompt_type="positive_negative_scribbles"), sources, authority
        )
