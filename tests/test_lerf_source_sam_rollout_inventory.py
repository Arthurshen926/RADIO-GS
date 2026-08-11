from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from radio_gs.scripts.build_lerf_source_sam_rollout_inventory import build


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, overlap: bool = False) -> Namespace:
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    for frame in (1, 41):
        (rgb / f"frame_{frame:05d}.jpg").write_bytes(f"rgb-{frame}".encode())
    frame_manifest = tmp_path / "frame_manifest.json"
    frame_manifest.write_text(json.dumps({
        "execution": {"resolved_source_image_dir": str(rgb)},
        "frames": [
            {
                "frame_idx": frame,
                "source_file": f"frame_{frame:05d}.jpg",
                "source_sha256": _sha(rgb / f"frame_{frame:05d}.jpg"),
            }
            for frame in (1, 41)
        ],
    }), encoding="utf-8")
    responsibility = tmp_path / "responsibility.json"
    selected = [1, 41] if overlap else [1]
    responsibility.write_text(json.dumps({
        "schema": "radio_gs.sparse_exact_marginal_responsibility_authority.v1",
        "schema_version": 1,
        "frame_indices": selected,
        "metadata": {
            "selected_frame_indices": selected,
            "excluded_frame_ids": [41],
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }), encoding="utf-8")
    assets = {}
    for name in (
        "control_field", "canonical_radio_cache", "dino_capability_target",
        "sam_adaptor_capability_target", "support_graph", "adjoint_config",
        "geometry_checkpoint",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        assets[name] = str(path)
        assets[f"expected_{name}_sha256"] = _sha(path)
    return Namespace(
        scene_id="figurines",
        frame_manifest=str(frame_manifest),
        expected_frame_manifest_sha256=_sha(frame_manifest),
        responsibility_authority=str(responsibility),
        expected_responsibility_authority_sha256=_sha(responsibility),
        source_rgb_root="",
        rollout_root=str(tmp_path / "rollout"),
        **assets,
    )


def test_lerf_inventory_aliases_only_explicit_non_eval_frames(tmp_path: Path) -> None:
    payload = build(_fixture(tmp_path))
    assert payload["source_frame_ids"] == [1]
    assert payload["excluded_evaluation_frame_ids"] == [41]
    alias = Path(payload["source_frames"][0]["alias_path"])
    assert alias.is_symlink()
    assert alias.name == "1.jpg"
    assert alias.resolve().name == "frame_00001.jpg"
    assert payload["access_audit"]["benchmark_query_gt_metric_opened"] is False


def test_lerf_inventory_rejects_evaluation_frame_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split differs"):
        build(_fixture(tmp_path, overlap=True))
