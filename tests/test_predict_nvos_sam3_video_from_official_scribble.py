from pathlib import Path

import numpy as np
from PIL import Image

from radio_gs.scripts.predict_nvos_sam3_video_from_official_scribble import (
    load_binary_scribble,
    choose_prompt_proposal,
    registered_rows,
    scribble_adherence,
)


def test_load_binary_scribble_accepts_rgb(tmp_path: Path) -> None:
    path = tmp_path / "scribble.png"
    values = np.zeros((3, 4, 3), dtype=np.uint8)
    values[1, 2, 1] = 255
    Image.fromarray(values).save(path)
    loaded = load_binary_scribble(path)
    assert loaded.shape == (3, 4)
    assert loaded[1, 2] == 1
    assert int(loaded.sum()) == 1


def test_registered_rows_deduplicates_manifest_records(tmp_path: Path) -> None:
    paths = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(path)
    scene = {
        "training_frames": [
            {"frame_id": "b", "rgb_path": str(paths[1]), "rgb_sorted_index": 1}
        ],
        "frames": [
            {"frame_id": "a", "rgb_path": str(paths[0]), "rgb_sorted_index": 0},
            {"frame_id": "b", "rgb_path": str(paths[1]), "rgb_sorted_index": 1},
            {"frame_id": "c", "rgb_path": str(paths[2]), "rgb_sorted_index": 2},
        ],
    }
    assert [row["frame_id"] for row in registered_rows(scene)] == ["a", "b", "c"]


def test_scribble_adherence_measures_both_signs() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    positive = np.zeros((4, 4), dtype=np.float32)
    positive[0, 0] = 1
    positive[3, 3] = 1
    negative = np.zeros((4, 4), dtype=np.float32)
    negative[0, 0] = 1
    negative[3, 3] = 1
    report = scribble_adherence(mask, positive, negative)
    assert report["positive_recall"] == 0.5
    assert report["negative_rejection"] == 0.5
    assert report["minimum_signed_adherence"] == 0.5


def test_choose_prompt_proposal_prefers_signed_adherence() -> None:
    positive = np.zeros((4, 4), dtype=np.float32)
    negative = np.zeros((4, 4), dtype=np.float32)
    positive[0, 0] = 1
    negative[3, 3] = 1
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, :, :] = True
    masks[1, 0, 0] = True
    selected, report = choose_prompt_proposal(
        masks, np.asarray([0.9, 0.1]), positive, negative
    )
    assert int(report["selected_index"]) == 1
    assert bool(selected[0, 0])
    assert not bool(selected[3, 3])
