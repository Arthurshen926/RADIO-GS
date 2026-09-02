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


@pytest.mark.parametrize(
    "source, expected",
    [
        ("from radio_gs.v2.method import legacy\n", "historical-method"),
        ("from radio_gs import v3\n", "historical-method"),
        ("from .. import v3\n", "historical-method"),
        ("import importlib\nimportlib.import_module(name)\n", "dynamic code/import"),
        ("from importlib import import_module as load\nload(name)\n", "dynamic code/import"),
        ("module = __import__(name)\n", "dynamic code/import"),
        ("import builtins as b\nb.__import__(name)\n", "dynamic code/import"),
        ("import builtins\nload = builtins.__import__\nload(name)\n", "dynamic code/import"),
        (
            "from radio_gs.v4.evaluation.lerf_development_pipeline import run\n",
            "quarantined development",
        ),
        ("SCENE = 'teatime'\n", "benchmark scene"),
    ],
)
def test_static_audit_rejects_indirect_history_and_scene_specialization(
    tmp_path, source, expected
):
    (tmp_path / "candidate.py").write_text(source)
    failures = audit(tmp_path)
    assert any(expected in failure for failure in failures)
