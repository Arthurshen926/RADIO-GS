from __future__ import annotations

import numpy as np
import pytest
import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.scripts.eval_spin9_exact_compact_unary_ladder import (
    COMPILER,
    RENDERER,
    VARIANTS,
    compile_k4_probability,
    reference_only_threshold,
)


def _signature(name: str, dim: int) -> FeatureSpaceSignature:
    return FeatureSpaceSignature(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256="a" * 64,
        raw_feature_dim=1280,
        token_type="primitive",
        adaptor_name=name,
        adaptor_sha256="b" * 64,
        adaptor_output_dim=dim,
        normalization="l2",
        field_checkpoint_sha256="c" * 64,
        semantic_alignment="none",
        semantic_alignment_sha256="",
    )


def test_ladder_contract_is_graph_free_and_fixed() -> None:
    assert VARIANTS == ("exact_capability", "exact_raw_adapted", "compact_field")
    assert COMPILER["graph"] == "disabled_not_constructed"
    assert COMPILER["prototype_count"] == 4
    assert COMPILER["prototype_strategy"] == "spherical_mean_fps"
    assert COMPILER["registered_seed_unary_weight"] == 0.0
    assert RENDERER["feature_contribution_gamma"] == 1.0


def test_identical_features_are_bitwise_identical_under_same_compiler() -> None:
    generator = torch.Generator().manual_seed(7)
    appearance = torch.nn.functional.normalize(torch.randn(12, 5, generator=generator), dim=-1)
    boundary = torch.nn.functional.normalize(torch.randn(12, 3, generator=generator), dim=-1)
    positive = torch.tensor([1.0, 0.8, 0.4, 0.2] + [0.0] * 8)
    negative = torch.tensor([0.0] * 4 + [0.2, 0.4, 0.8, 1.0] + [0.0] * 4)
    signatures = {
        "appearance": _signature("dino", 5),
        "boundary": _signature("sam", 3),
    }
    first = compile_k4_probability(
        {"appearance": appearance, "boundary": boundary},
        positive,
        negative,
        signatures,
        score_chunk_size=3,
    )
    second = compile_k4_probability(
        {"appearance": appearance.clone(), "boundary": boundary.clone()},
        positive.clone(),
        negative.clone(),
        signatures,
        score_chunk_size=3,
    )
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    assert set(first[2]) == {"appearance", "boundary"}


def test_compiler_requires_both_frozen_capability_heads() -> None:
    values = torch.eye(4)
    positive = torch.tensor([1.0, 0.0, 0.0, 0.0])
    negative = torch.tensor([0.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="exactly appearance and boundary"):
        compile_k4_probability(
            {"appearance": values},
            positive,
            negative,
            {
                "appearance": _signature("dino", 4),
                "boundary": _signature("sam", 4),
            },
        )


def test_reference_threshold_is_deterministic_and_reference_only() -> None:
    score = np.array([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32)
    mask = np.array([[1, 1], [0, 0]], dtype=bool)
    threshold, iou, records = reference_only_threshold(score, mask)
    assert threshold == pytest.approx(0.8)
    assert iou == 1.0
    assert len(records) == 97
    assert set(records[0]) == {"threshold", "reference_iou"}
