from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from radio_gs.scripts.build_nvos_synchronous_multiview_candidate_plan import (
    expand_to_all_registered_views,
    exchangeable_candidates,
    exclusive_projected_authority,
)


def test_exclusive_authority_keeps_only_transport_dominance() -> None:
    positive = torch.tensor([[0.9, 0.2, 0.0], [0.5, 0.0, 0.8]])
    negative = torch.tensor([[0.1, 0.2, 0.7], [0.0, 0.0, 0.9]])
    visible = torch.tensor([[True, True, True], [False, True, True]])
    pos, neg = exclusive_projected_authority(positive, negative, visible)
    assert torch.equal(
        pos, torch.tensor([[True, False, False], [False, False, False]])
    )
    assert torch.equal(
        neg, torch.tensor([[False, False, True], [False, False, True]])
    )
    assert not bool((pos & neg).any())


def test_exchangeable_candidates_are_unique_equal_logit_same_cohort() -> None:
    views = [{"view_digest": "a" * 64, "frame_id": "source"}]
    first = exchangeable_candidates(
        scene_id="fern", plan_identity={"x": 1}, views=views, count=10
    )
    second = exchangeable_candidates(
        scene_id="fern", plan_identity={"x": 1}, views=views, count=10
    )
    assert first == second
    assert len({row["candidate_digest"] for row in first}) == 10
    assert {row["candidate_logit"] for row in first} == {0.0}
    assert [row["trial_rank"] for row in first] == list(range(10))
    assert all(
        row["views"]
        == [{**views[0], "candidate_trial_rank": row["trial_rank"]}]
        for row in first
    )


def test_exchangeable_candidate_identity_binds_plan() -> None:
    views = [{"view_digest": "b" * 64}]
    left = exchangeable_candidates(
        scene_id="fern", plan_identity={"x": 1}, views=views, count=2
    )
    right = exchangeable_candidates(
        scene_id="fern", plan_identity={"x": 2}, views=views, count=2
    )
    assert [row["candidate_digest"] for row in left] != [
        row["candidate_digest"] for row in right
    ]


def test_expand_registered_views_preserves_protocol_roles(tmp_path: Path) -> None:
    mapping = []
    colmap = {}
    for rank, name in enumerate(("source", "middle", "target")):
        rgb = tmp_path / f"{name}.jpg"
        rgb.write_bytes(bytes([rank + 1]))
        colmap_path = Path("images") / rgb.name
        mapping.append(
            {
                "rgb_camera_name": name,
                "rgb_path": str(rgb),
                "colmap_camera_name": name,
                "colmap_file_path": str(colmap_path),
                "match_rule": "exact_case_sensitive_basename_stem",
            }
        )
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = float(rank)
        colmap[name] = (str(colmap_path), c2w)
    protocol = [
        {"frame_id": "source", "camera_name": "source", "role": "prompt"},
        {"frame_id": "target", "camera_name": "target", "role": "evaluation"},
    ]
    views = expand_to_all_registered_views(protocol, mapping, colmap)
    assert [view["frame_id"] for view in views] == ["source", "middle", "target"]
    assert [view["role"] for view in views] == [
        "prompt",
        "registered_mapping",
        "evaluation",
    ]
    assert np.allclose(views[1]["w2c"][0, 3], -1.0)
