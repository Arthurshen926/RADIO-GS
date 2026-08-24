from __future__ import annotations

import torch

from radio_gs.scripts.build_nvos_native_siglip_view_reliability import (
    appearance_reliability,
    resolve_view_roles,
    select_mapping_views,
)


def test_appearance_reliability_requires_both_protocol_anchors() -> None:
    prompt = torch.tensor([1.0, 0.0, 0.0])
    evaluation = torch.tensor([0.8, 0.6, 0.0])
    descriptors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scores = appearance_reliability(prompt, evaluation, descriptors)
    assert scores.shape == (4,)
    assert scores[2] > scores[3]
    assert torch.isclose(scores[0], torch.tensor(0.8))
    assert torch.isclose(scores[1], torch.tensor(0.8))


def test_topk_keeps_prompt_evaluation_and_best_mapping_views() -> None:
    roles = [
        "prompt",
        "registered_mapping",
        "registered_mapping",
        "evaluation",
        "registered_mapping",
    ]
    scores = torch.tensor([0.9, 0.4, 0.8, 0.9, 0.6])
    assert select_mapping_views(roles, scores, top_k=2) == (0, 2, 3, 4)


def test_roles_recover_from_manifest_and_cross_check_explicit_plan() -> None:
    frames = ["middle", "target", "source"]
    roles = resolve_view_roles(
        frames,
        prompt_frame_ids=["source"],
        evaluation_frame_ids=["target"],
        explicit_roles=["registered_mapping", "evaluation", "prompt"],
    )
    assert roles == ["registered_mapping", "evaluation", "prompt"]
