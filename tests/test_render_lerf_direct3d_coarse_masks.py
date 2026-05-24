from pathlib import Path

from radio_gs.scripts.render_lerf_direct3d_coarse_masks import (
    choose_training_frame_ids,
    coarse_mask_path,
)


def test_choose_training_frame_ids_excludes_label_frames_evenly():
    frames = list(range(10))
    selected = choose_training_frame_ids(frames, excluded={2, 5, 7}, max_frames=4)

    assert selected == [0, 3, 6, 9]
    assert not ({2, 5, 7} & set(selected))


def test_coarse_mask_path_matches_direct_eval_naming(tmp_path: Path):
    path = coarse_mask_path(tmp_path, frame_id=41, category="red/cup")

    assert path == tmp_path / "frame_00041_red_cup.png"
