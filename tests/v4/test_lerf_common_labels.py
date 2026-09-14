import json
import pytest

from radio_gs.v4.evaluation.lerf_common import load_labels


def test_disconnected_polygons_with_different_vertex_counts(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    payload = {"info": {"height": 20, "width": 30}, "objects": [{"category": "cup", "segmentation": [
        [[1, 1], [2, 1], [2, 2]], [[5, 5], [7, 5], [7, 7], [5, 7]],
    ]}]}
    (scene / "frame_00001.json").write_text(json.dumps(payload))
    annotations, _, _, _ = load_labels(tmp_path, "scene")
    assert [len(p) for p in annotations[1][0]["polygons"]] == [3, 4]
    payload["info"]["width"] = 60
    (scene / "frame_00002.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="per-frame scaling"):
        load_labels(tmp_path, "scene")
