import pytest

from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput
from radio_gs.v4.contracts.method_receipt import MethodReceipt
from radio_gs.v4.contracts.source_split import SourceSplit
from radio_gs.v4.contracts.static_audit import audit


def test_v4_namespace_passes_static_isolation():
    from radio_gs import v4

    assert audit(__import__("pathlib").Path(v4.__file__).parent) == []


def test_receipts_and_source_split_fail_closed(tmp_path):
    artifact = tmp_path / "geometry.bin"
    artifact.write_bytes(b"sealed")
    sealed = HashedInput.seal("geometry", artifact)
    with pytest.raises(ValueError, match="target RGB"):
        GeometryReceipt(
            carrier="surface",
            coordinate_convention="camera_to_world",
            inputs=(sealed,),
            source_rgb_opened=True,
            target_rgb_opened=True,
            benchmark_images_opened=False,
            benchmark_masks_opened=False,
            benchmark_labels_opened=False,
        )
    with pytest.raises(ValueError, match="downstream"):
        MethodReceipt("geometry_registration", "surface", sealed.sha256, codebook_enabled=True)
    with pytest.raises(ValueError, match="overlap"):
        SourceSplit(frozenset({"same"}), frozenset(), frozenset({"same"}))
