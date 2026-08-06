from __future__ import annotations

import torch
import pytest

from radio_gs.scripts.audit_factorized_radio_label_free_gate import (
    _validate_mpr_lineage,
    compare_capability_rows,
    evaluate_promotion_gate,
    select_common_valid_rows,
    summarize_norms,
)


class _GaugeSensitiveAdaptor(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = values[..., 0]
        second = values[..., 1]
        norm = torch.linalg.vector_norm(values, dim=-1)
        return torch.stack(
            (first + 0.2 * norm + 5.0, second - 0.1 * norm - 2.0), dim=-1
        )


def test_select_common_rows_is_ascending_and_bounded() -> None:
    rows, total = select_common_valid_rows(
        [
            torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor([1, 1, 1, 0, 1], dtype=torch.bool),
        ],
        maximum_rows=2,
    )
    assert total == 3
    assert rows.tolist() == [0, 2]


def test_norm_summary_reports_preregistered_statistics() -> None:
    summary = summarize_norms(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
    assert summary["rows"] == 2
    assert summary["mean"] == pytest.approx(3.5)
    assert summary["median"] == pytest.approx(3.5)
    assert summary["minimum"] == pytest.approx(2.0)
    assert summary["maximum"] == pytest.approx(5.0)


def test_capability_comparison_uses_official_module_and_centered_rows() -> None:
    factorized = torch.zeros(4, 1280)
    factorized[:, :2] = torch.tensor(
        [[30.0, 2.0], [2.0, 30.0], [-30.0, 1.0], [1.0, -30.0]]
    )
    legacy = torch.nn.functional.normalize(factorized, dim=-1)
    adaptor = _GaugeSensitiveAdaptor()
    with torch.inference_mode():
        exact = torch.nn.functional.normalize(adaptor(factorized[None])[0], dim=-1)
    result = compare_capability_rows(
        adaptor=adaptor,
        legacy_radio=legacy,
        factorized_radio=factorized,
        exact_capability=exact,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert result["factorized"]["cosine_mean"] == pytest.approx(1.0, abs=1e-6)
    assert result["factorized"]["cosine_p05"] == pytest.approx(1.0, abs=1e-6)
    assert result["factorized"]["centered_row_variation_ratio_to_exact"] == pytest.approx(
        1.0, abs=1e-6
    )
    assert result["factorized_minus_legacy"]["cosine_mean"] > 0


def _capability(delta_mean=0.01, delta_p05=0.01, ratio=0.9):
    return {
        "factorized_minus_legacy": {
            "cosine_mean": delta_mean,
            "cosine_p05": delta_p05,
        },
        "factorized": {"centered_row_variation_ratio_to_exact": ratio},
    }


def _thresholds():
    return {
        "factorized_active_median_norm_lower": 28.6447,
        "factorized_active_median_norm_upper": 34.5264,
        "dino_mean_cosine_delta_vs_legacy_minimum": -0.002,
        "dino_p05_cosine_delta_vs_legacy_minimum": -0.005,
        "sam3_mean_cosine_delta_vs_legacy_minimum": -0.002,
        "sam3_p05_cosine_delta_vs_legacy_minimum": -0.005,
        "centered_row_variation_ratio_to_exact_minimum": 0.8,
        "all_cache_and_lineage_invariants": True,
    }


def test_promotion_gate_requires_each_capability_and_lineage() -> None:
    capabilities = {"dino_v3": _capability(), "sam3": _capability()}
    passed = evaluate_promotion_gate(
        factorized_norms={"median": 31.0},
        capabilities=capabilities,
        thresholds=_thresholds(),
        cache_and_lineage_invariants=True,
    )
    assert passed["passed"] is True

    capabilities["sam3"] = _capability(ratio=0.79)
    failed = evaluate_promotion_gate(
        factorized_norms={"median": 31.0},
        capabilities=capabilities,
        thresholds=_thresholds(),
        cache_and_lineage_invariants=True,
    )
    assert failed["passed"] is False
    assert failed["decision"] == "do_not_train_or_open_benchmark_targets"


def test_legacy_lineage_does_not_require_later_bundle_receipt() -> None:
    geometry = {"num_gaussians": 2, "xyz_sha256": "a" * 64}
    cache = {
        "geometry_fingerprint": geometry,
        "metadata": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "registration_responsibility_cache_sha256": "b" * 64,
            "observation_lifting_contract": {
                "name": "canonical-mpr-v1",
                "feature_projection_order": "per_view_before_mpr",
                "query_independent": True,
                "responsibility_sharing": "exact_sidecar_across_feature_spaces",
            },
        },
    }
    _validate_mpr_lineage(
        cache=cache,
        expected_space="radio",
        expected_geometry=geometry,
        expected_responsibility_sha256="b" * 64,
        expected_radio_checkpoint_sha256=None,
    )
    cache["metadata"]["text_queries_opened"] = True
    with pytest.raises(ValueError, match="source safety"):
        _validate_mpr_lineage(
            cache=cache,
            expected_space="radio",
            expected_geometry=geometry,
            expected_responsibility_sha256="b" * 64,
            expected_radio_checkpoint_sha256=None,
        )
