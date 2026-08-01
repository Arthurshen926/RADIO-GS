from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

from PIL import Image
import pytest

import reproductions.ludvig.stage_spin_pinecone_official_undistortion as stage
import reproductions.ludvig.train_nvos_all_view_3dgs as common_training
import reproductions.ludvig.train_spin_pinecone_all_view_3dgs as pinecone_training


def test_pinecone_spec_freezes_pinned_colmap_contract() -> None:
    spec = pinecone_training.SPIN_PINECONE_SPEC

    assert spec.benchmark == "SPIn-NeRF"
    assert spec.scene == spec.geometry_scene == "pinecone"
    assert spec.expected_registered_images == 99
    assert spec.evaluation_render_resolution == (1600, 1199)
    assert spec.converted_source_relative == Path(
        "SPIn-NeRF/protocol_derived/pinecone_colmap_3p6_undistorted_v2"
    )
    assert spec.raw_identity_source_relative == Path(
        "SPIn-NeRF/source_images/nerf_real_360/extracted/pinecone"
    )
    assert spec.source_asset_contract == (
        common_training.NATIVE_SPIN_PINECONE_PINHOLE_CONTRACT
    )
    assert common_training._automatic_training_resolution(4015, 3011) == (
        1600,
        1199,
    )
    common_training._validate_training_spec(spec)
    with pytest.raises(common_training.TrainingProtocolError, match="99-view"):
        common_training._validate_training_spec(
            replace(spec, expected_registered_images=98)
        )
    with pytest.raises(FrozenInstanceError):
        spec.scene = "lego"


def test_plain_annotation_contract_requires_99_binary_masks(tmp_path: Path) -> None:
    for index in range(99):
        mask = Image.new("L", (2, 2), color=0)
        mask.putpixel((0, 0), 1)
        mask.save(tmp_path / f"IMG_{index:04d}.png")
    hashes = stage._plain_annotation_hashes(tmp_path)
    assert len(hashes) == 99
    assert all(len(value) == 64 for value in hashes.values())

    Image.new("L", (2, 2), color=2).save(tmp_path / "IMG_0000.png")
    with pytest.raises(stage.PineconeStagingError, match="not binary"):
        stage._plain_annotation_hashes(tmp_path)


def test_completed_real_staging_receipt_is_fail_closed() -> None:
    manifest_path = (
        Path("output/protocol_audit_20260731/ludvig/spin/released_all_view/")
        / "pinecone/undistortion/attempts/official_colmap_3p6_v2/"
        / "undistortion_manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip("host-local immutable Pinecone receipt is unavailable")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["colmap"]["executable_sha256"] == (
        "f9e01dec8d9edd8a37663a15867de7da18f01e454877a0d09fdc50295f92515c"
    )
    assert payload["raw"]["camera_model"] == "SIMPLE_RADIAL"
    assert payload["raw"]["registered_images"] == 99
    assert payload["raw"]["source_dataset_modified"] is False
    assert payload["annotations"]["rgb_stem_bijection"] is True
    assert payload["annotations"]["scored_targets"] == 98
    assert payload["undistorted"]["camera_model"] in {
        "PINHOLE",
        "SIMPLE_PINHOLE",
    }
    assert payload["undistorted"]["camera_dimensions"] == [4015, 3011]
    assert payload["undistorted"]["effective_original_3dgs_resolution"] == [
        1600,
        1199,
    ]
    assert payload["undistorted"]["max_qvec_delta_vs_raw"] <= 1e-12
    assert payload["undistorted"]["max_tvec_delta_vs_raw"] <= 1e-12
