from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from radio_gs.scripts import build_lerf_nearest_view_cache_baseline as baseline


def _write_feature_scene(root: Path) -> None:
    scene = root / "figurines"
    (scene / "backbone").mkdir(parents=True)
    frames = []
    c2w = []
    for x in [0.0, 1.0, 2.0, 3.0]:
        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = x
        c2w.append(mat)
    for frame_id in [1, 2, 4]:
        (scene / "backbone" / f"rgb_{frame_id}.pt").write_text("feature", encoding="utf-8")
        frames.append({"frame_idx": frame_id, "saved_stem": f"rgb_{frame_id}"})
    (scene / "frame_manifest.json").write_text(
        json.dumps({"scene": "figurines", "frames": frames}),
        encoding="utf-8",
    )
    np.savetxt(scene / "traj_w_c.txt", np.stack(c2w).reshape(-1, 16))


def test_load_feature_index_uses_camera_centers_and_existing_feature_paths(tmp_path: Path) -> None:
    _write_feature_scene(tmp_path)

    index = baseline.load_feature_index(tmp_path, "figurines")

    assert sorted(index.frames) == [1, 2, 4]
    assert index.frames[4].center.tolist() == [3.0, 0.0, 0.0]
    assert index.frames[4].feature_path.name == "rgb_4.pt"


def test_build_nearest_mapping_excludes_target_frame(tmp_path: Path) -> None:
    _write_feature_scene(tmp_path)
    index = baseline.load_feature_index(tmp_path, "figurines")

    mapping = baseline.build_nearest_mapping(index, target_frame_ids=[1, 2])

    assert mapping[1].source_frame == 2
    assert mapping[1].distance == 1.0
    assert mapping[2].source_frame == 1


def test_summarize_scene_rows_computes_macro_and_weighted_metrics() -> None:
    rows = [
        baseline.SceneResult("Figurines", 0.5, 0.4, 10, 1.0),
        baseline.SceneResult("Ramen", 0.9, 0.8, 30, 2.0),
    ]

    summary = baseline.summarize_rows(rows, protocol={"selection": "nearest"})

    assert summary["macro"]["loc_acc"] == 0.7
    assert summary["macro"]["miou"] == 0.6
    assert summary["weighted"]["loc_acc"] == 0.8
    assert summary["weighted"]["miou"] == 0.7
    assert summary["mean_nearest_distance"] == 1.5


def test_markdown_and_latex_identify_unwarped_nearest_view_baseline() -> None:
    summary = baseline.summarize_rows(
        [baseline.SceneResult("Figurines", 0.5, 0.4, 10, 1.0)],
        protocol={"selection": "nearest_by_camera_center"},
    )

    markdown = baseline.build_markdown(summary)
    latex = baseline.build_latex_table(summary)

    assert "unwarped nearest-view cached teacher" in markdown
    assert "| Figurines | 0.5000 | 0.4000 | 10 | 1.0000 |" in markdown
    assert "\\label{tab:nearest_view_cache_baseline}" in latex
