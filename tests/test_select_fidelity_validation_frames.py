import json

import pytest

from radio_gs.scripts.select_fidelity_validation_frames import build, select_frame_ids


def test_selects_deterministic_even_interior_views() -> None:
    assert select_frame_ids(list(range(0, 2400, 100)), 4) == [400, 900, 1400, 1900]


def test_keeps_one_training_view_and_requires_two_views() -> None:
    assert select_frame_ids([0, 100], 4) == [100]
    with pytest.raises(ValueError, match="at least two"):
        select_frame_ids([0], 1)


def test_build_is_label_free_and_writes_manifest(tmp_path) -> None:
    features = tmp_path / "features"
    features.mkdir()
    (features / "frame_manifest.json").write_text(
        json.dumps({"frames": [{"frame_idx": value} for value in (0, 100, 200, 300)]}),
        encoding="utf-8",
    )
    output = tmp_path / "validation.json"
    payload = build(features, output, 2)
    assert payload["validation_frame_ids"] == [100, 200]
    assert payload["benchmark_labels_opened"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["policy"].endswith("v1")
