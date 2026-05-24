from pathlib import Path

from radio_gs.scripts.render_lerf2d_coarse_masks import (
    choose_training_frame_ids,
    coarse_mask_path,
)


def test_choose_training_frame_ids_excludes_label_frames_evenly():
    frames = list(range(10))

    selected = choose_training_frame_ids(frames, excluded={2, 5, 7}, max_frames=4)

    assert selected == [0, 3, 6, 9]


def test_coarse_mask_path_uses_query_safe_filename():
    path = coarse_mask_path(Path("/tmp/coarse"), frame_id=41, category="cup/plate")

    assert path == Path("/tmp/coarse/frame_00041_cup_plate.png")
