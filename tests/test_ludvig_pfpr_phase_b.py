from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_b import (
    LUDVIG_PCA_COMPONENTS,
    LudvigPFPRPhaseBError,
    PhaseBConfig,
    apply_scene_pca_transform,
    audit_model_architecture,
    fit_ludvig_scene_pca,
    load_checkpoint_exact_ludvig_vendored,
    ludvig_sliding_plan,
)


class _TinyVendoredModel(torch.nn.Module):
    patch_size = 14
    embed_dim = 1536

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 3))


def _config(tmp_path: Path) -> PhaseBConfig:
    return PhaseBConfig(
        phase_a_dir=tmp_path / "phase_a",
        expected_phase_a_manifest_sha256="a" * 64,
        dino_checkpoint=tmp_path / "checkpoint.pth",
        ludvig_upstream=tmp_path / "ludvig",
        source_adapter_ledger=tmp_path / "ledger.json",
        dinov2_source=tmp_path / "ludvig",
        output_dir=tmp_path / "phase_b",
    )


def _checkpoint(path: Path, **extra: torch.Tensor) -> None:
    payload = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "register_tokens": torch.zeros(1, 4, 1536),
        **extra,
    }
    torch.save(payload, path)


def test_ludvig_640x480_sliding_plan_is_two_476_crops() -> None:
    plan = ludvig_sliding_plan(480, 640)

    assert plan["aligned_height"] == 476
    assert plan["aligned_width"] == 630
    assert plan["effective_crop_size"] == 476
    assert plan["effective_stride_width"] == 154
    assert plan["indices_yx"] == [[0, 0], [0, 154]]
    assert plan["patch_count"] == 2
    assert plan["token_grid_height"] == 34
    assert plan["token_grid_width"] == 34
    assert plan["tokens_per_view"] == 2312


def test_exact_ludvig_load_allows_only_frozen_register_tokens(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pth"
    _checkpoint(path)
    model = _TinyVendoredModel()

    audit = load_checkpoint_exact_ludvig_vendored(model, path)

    assert audit["missing_keys"] == []
    assert audit["unexpected_keys"] == ["register_tokens"]
    assert audit["ignored_key_shapes"] == {"register_tokens": [1, 4, 1536]}
    assert audit["verification_after_single_key_filter_strict"] is True
    assert torch.equal(model.weight, torch.arange(6).reshape(2, 3))


def test_exact_ludvig_load_rejects_any_additional_unexpected_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pth"
    _checkpoint(path, hidden_extra=torch.ones(1))

    with pytest.raises(LudvigPFPRPhaseBError, match="vendored contract"):
        load_checkpoint_exact_ludvig_vendored(_TinyVendoredModel(), path)


def test_exact_ludvig_load_rejects_register_shape_change(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "weight": torch.zeros(2, 3),
            "register_tokens": torch.zeros(1, 3, 1536),
        },
        path,
    )

    with pytest.raises(LudvigPFPRPhaseBError, match="shape changed"):
        load_checkpoint_exact_ludvig_vendored(_TinyVendoredModel(), path)


def test_vendored_architecture_requires_no_register_support(tmp_path: Path) -> None:
    audit = audit_model_architecture(_TinyVendoredModel(), _config(tmp_path))

    assert audit["embedding_dim"] == 1536
    assert audit["patch_size"] == 14
    assert audit["register_tokens"] == 0
    assert audit["supports_register_tokens"] is False


def test_true_reg4_model_is_not_exact_ludvig_primary(tmp_path: Path) -> None:
    model = _TinyVendoredModel()
    model.register_tokens = torch.nn.Parameter(torch.zeros(1, 4, 1536))
    model.num_register_tokens = 4

    with pytest.raises(LudvigPFPRPhaseBError, match="register-token count"):
        audit_model_architecture(model, _config(tmp_path))


def test_scene_pca_is_deterministic_and_records_unbiased_standardization() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(64, 8)).astype(np.float32)

    first = fit_ludvig_scene_pca(
        raw,
        n_components=4,
        pca_subsample=500_000,
        seed=0,
        statistics_device=torch.device("cpu"),
    )
    second = fit_ludvig_scene_pca(
        raw,
        n_components=4,
        pca_subsample=500_000,
        seed=0,
        statistics_device=torch.device("cpu"),
    )

    np.testing.assert_allclose(first["feature_mean"], raw.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(first["feature_std"], raw.std(axis=0, ddof=1), atol=1e-6)
    for key in ("pca_mean", "pca_components", "pca_singular_values", "projected"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["sampled_indices"].size == 0


def test_query_transform_reuses_frozen_scene_arrays_with_optional_weighting() -> None:
    raw = np.array([[[1.0, 3.0], [5.0, 7.0]]], dtype=np.float32)
    transform = {
        "feature_mean": np.array([1.0, 1.0], dtype=np.float32),
        "feature_std": np.array([2.0, 2.0], dtype=np.float32),
        "pca_mean": np.array([0.5, 0.5], dtype=np.float32),
        "pca_components": np.eye(2, dtype=np.float32),
        "pca_singular_values": np.array([2.0, 3.0], dtype=np.float32),
    }

    unweighted = apply_scene_pca_transform(
        raw, transform, eigval_weighting=False
    )
    weighted = apply_scene_pca_transform(raw, transform, eigval_weighting=True)

    expected = (raw - 1.0) / 2.0 - 0.5
    np.testing.assert_allclose(unweighted, expected)
    np.testing.assert_allclose(weighted, expected * np.array([2.0, 3.0]))
    assert unweighted.shape == raw.shape
    assert LUDVIG_PCA_COMPONENTS == 40
