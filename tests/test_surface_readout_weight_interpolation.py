from __future__ import annotations

import copy

import pytest
import torch

from radio_gs.scripts import diagnose_surface_readout_weight_interpolation as module


def _payload(*, seed: int, value: float) -> dict:
    config = {field: 1 for field in module.COMMON_TRAINING_FIELDS}
    config.update(
        {
            "seed": seed,
            "reliability_attention_mode": "log_prior",
            "context_pooling_mode": "joint_attention_v1",
            "canonical_noise_calibration": "",
        }
    )
    return {
        "architecture": {
            "name": "surface_region_summary_readout_v2",
            "hidden_dim": 2,
            "digest": "a" * 64,
        },
        "training_config": config,
        "state_dict": {
            "linear.weight": torch.full((2, 3), value),
            "linear.bias": torch.tensor([value, value + 1.0]),
        },
    }


def _pareto_row(
    alpha: float,
    *,
    surface: float,
    support: float,
    error: float,
    dev_ranking: float,
) -> dict:
    return {
        "alpha": alpha,
        "surface": {
            "summary_token_cosine": surface,
            "mean_descriptor_cosine": surface,
            "all_view_descriptor_cosine": surface,
            "relation_fidelity": surface,
        },
        "fit": {
            "text_support_top1_agreement": support,
            "text_response_smooth_l1": error,
            "descriptor_relation_smooth_l1": error,
        },
        "dev_posthoc": {
            "aggregate_seed_mean": {"ranking_spearman_mean": dev_ranking},
        },
    }


def test_same_seed_same_architecture_state_is_interpolable() -> None:
    control = _payload(seed=1, value=0.0)
    candidate = _payload(seed=1, value=2.0)

    record = module.assert_interpolable(control, candidate)
    midpoint = module.interpolate_state_dict(
        control["state_dict"], candidate["state_dict"], 0.5
    )

    assert record["seed"] == 1
    assert record["tensor_count"] == 2
    assert torch.equal(midpoint["linear.weight"], torch.ones(2, 3))
    assert torch.equal(midpoint["linear.bias"], torch.tensor([1.0, 2.0]))
    assert torch.equal(control["state_dict"]["linear.weight"], torch.zeros(2, 3))


def test_interpolation_rejects_architecture_or_seed_drift() -> None:
    control = _payload(seed=0, value=0.0)
    candidate = _payload(seed=0, value=1.0)
    candidate["architecture"]["digest"] = "b" * 64
    with pytest.raises(ValueError, match="architectures differ"):
        module.assert_interpolable(control, candidate)

    candidate = _payload(seed=2, value=1.0)
    with pytest.raises(ValueError, match="training field seed differs"):
        module.assert_interpolable(control, candidate)


def test_pareto_front_excludes_dev_metrics() -> None:
    rows = [
        _pareto_row(0.0, surface=0.90, support=0.50, error=0.10, dev_ranking=0.10),
        _pareto_row(0.5, surface=0.95, support=0.45, error=0.08, dev_ranking=0.20),
        _pareto_row(1.0, surface=0.80, support=0.40, error=0.20, dev_ranking=0.99),
    ]
    expected = module.pareto_front_alphas(rows)
    changed_dev = copy.deepcopy(rows)
    changed_dev[0]["dev_posthoc"]["aggregate_seed_mean"]["ranking_spearman_mean"] = -1.0
    changed_dev[2]["dev_posthoc"]["aggregate_seed_mean"]["ranking_spearman_mean"] = 1.0

    assert expected == [0.0, 0.5]
    assert module.pareto_front_alphas(changed_dev) == expected
    assert all("dev" not in metric for metric, _ in module.PARETO_OBJECTIVES)


def test_cpu_preflight_requires_explicit_hidden_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        module._cpu_only_preflight()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    module._cpu_only_preflight()
