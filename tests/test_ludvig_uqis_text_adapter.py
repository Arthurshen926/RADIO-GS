from __future__ import annotations
import numpy as np
import pytest
from radio_gs.benchmarks.scannet_uqis.ludvig_text_adapter import lerf_relevancy
from radio_gs.benchmarks.scannet_uqis.ludvig_text_diffusion import (
    TextDiffusionConfig,
    align_clip_relevancy_to_dino_carrier,
    build_dino_graph,
    diffuse_clip_relevancy,
)

def test_lerf_relevancy_uses_hardest_negative() -> None:
    similarities = np.array([[1.0, 0.0, 0.5, -1.0, 0.25]], dtype=np.float32)
    value = lerf_relevancy(similarities)
    assert value.dtype == np.float32
    assert value[0] == pytest.approx(1 / (1 + np.exp(-5.0)), rel=1e-6)


def test_clip_relevancy_aligns_to_pruned_dino_carrier() -> None:
    clip = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    source = np.array([3, 1], dtype=np.int64)
    np.testing.assert_array_equal(
        align_clip_relevancy_to_dino_carrier(clip, source),
        np.array([0.4, 0.2], dtype=np.float32),
    )


def test_dino_diffusion_spreads_inside_feature_cluster() -> None:
    xyz = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [2, 0, 0], [2.1, 0, 0], [2.2, 0, 0]],
        dtype=np.float32,
    )
    dino = np.array(
        [[1, 0], [0.99, 0.01], [0.98, 0.02], [0, 1], [0.01, 0.99], [0.02, 0.98]],
        dtype=np.float32,
    )
    config = TextDiffusionConfig(neighbors=2, iterations=8, seed_quantile=0.8)
    graph = build_dino_graph(xyz, dino, config)
    raw = np.array([1.0, 0.05, 0.04, 0.03, 0.02, 0.01], dtype=np.float32)
    result = diffuse_clip_relevancy(raw, graph, config)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result[1] > result[3]
    assert result[2] > result[4]
    assert result.min() >= 0 and result.max() <= 1


def test_diffusion_rejects_invalid_source_mapping() -> None:
    with pytest.raises(ValueError, match="unique"):
        align_clip_relevancy_to_dino_carrier(
            np.ones(3, dtype=np.float32), np.array([1, 1], dtype=np.int64)
        )
