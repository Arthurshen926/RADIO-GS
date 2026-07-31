import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from radio_gs.scripts import summarize_langsplatv2_lerf_audit as summarize
from reproductions.langsplatv2 import run_lerf2d_exact_camera as launcher


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_langsplatv2_patch_matches_lock_and_is_camera_only():
    lock_path = ROOT / "reproductions" / "langsplatv2" / "upstream.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    patch_path = lock_path.parent / lock["patch"]
    patch = patch_path.read_bytes()
    text = patch.decode("utf-8")

    assert lock["commit"] == launcher.UPSTREAM_COMMIT
    assert hashlib.sha256(patch).hexdigest() == launcher.PATCH_SHA256
    assert lock["patch_sha256"] == launcher.PATCH_SHA256
    assert "_select_view_for_label_image" in text
    assert "scene.getTrainCameras()" in text
    assert "scene.getTestCameras()" in text
    assert "utils/loss_utils.py" not in text
    assert "utils/vq_utils.py" not in text


def test_launcher_streaming_sha256_matches_hashlib(tmp_path):
    payload = (b"camera-protocol-audit\x00" * 100_003) + b"tail"
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(payload)

    assert launcher._sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_latest_complete_log_ignores_newer_incomplete_run(tmp_path):
    scene_dir = tmp_path / "teatime_0"
    scene_dir.mkdir()
    complete = scene_dir / "20260731_000001.log"
    complete.write_text(
        "iou chosen: 0.7160\nLocalization accuracy: 0.8814\n",
        encoding="utf-8",
    )
    incomplete = scene_dir / "20260731_000002.log"
    incomplete.write_text("checkpoint: 10000\n", encoding="utf-8")

    path, miou, loc_acc = summarize._latest_complete_log(tmp_path, "teatime")

    assert path == complete
    assert miou == pytest.approx(0.716)
    assert loc_acc == pytest.approx(0.8814)


def test_launcher_reads_namespace_cfg_without_executing_it(tmp_path):
    cfg = tmp_path / "cfg_args"
    cfg.write_text(
        "Namespace(source_path='/data/teatime', eval=True, feature_level=2)\n",
        encoding="utf-8",
    )

    assert launcher._read_cfg_args(cfg) == {
        "source_path": "/data/teatime",
        "eval": True,
        "feature_level": 2,
    }

    cfg.write_text("__import__('os').system('false')\n", encoding="utf-8")
    with pytest.raises(launcher.ProtocolError):
        launcher._read_cfg_args(cfg)


def test_published_langsplatv2_overall_uses_mixed_aggregation():
    scene_miou = np.mean(
        [summarize.PAPER_ROWS[scene]["miou"] for scene in summarize.SCENE_ORDER]
    )
    scene_loc = np.mean(
        [summarize.PAPER_ROWS[scene]["loc_acc"] for scene in summarize.SCENE_ORDER]
    )
    query_counts = {
        "figurines": 56,
        "teatime": 59,
        "ramen": 71,
        "waldo_kitchen": 22,
    }
    micro_loc = sum(
        summarize.PAPER_ROWS[scene]["loc_acc"] * query_counts[scene]
        for scene in summarize.SCENE_ORDER
    ) / sum(query_counts.values())

    assert scene_miou == pytest.approx(0.59875)
    assert round(scene_miou, 3) == summarize.PAPER_OVERALL["miou"]
    assert scene_loc == pytest.approx(0.86375)
    assert micro_loc == pytest.approx(0.8413990385)
    assert round(micro_loc, 3) == summarize.PAPER_OVERALL["loc_acc"]
