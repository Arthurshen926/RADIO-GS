from __future__ import annotations

import json

from radio_gs.scripts.render_nvos_synchronous_candidate_marginal import (
    target_frame_id,
)


def test_target_frame_is_resolved_from_bound_manifest(tmp_path) -> None:
    manifest = {
        "scenes": [{"scene_id": "fern", "evaluation_frame_ids": ["target"]}]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert target_frame_id(
        {
            "scene_id": "fern",
            "inputs": {"dataset_manifest": {"path": str(path), "sha256": digest}},
        }
    ) == "target"
