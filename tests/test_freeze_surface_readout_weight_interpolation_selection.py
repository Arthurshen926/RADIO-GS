from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from radio_gs.scripts import (
    freeze_surface_readout_weight_interpolation_selection as module,
)


def _file_record(path: Path, content: bytes) -> dict[str, str]:
    path.write_bytes(content)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _surface(value: float) -> dict[str, float]:
    return {metric: value for metric in module.SURFACE_METRICS}


def _diagnostic(tmp_path: Path) -> dict:
    per_seed = []
    aggregate = []
    support_by_alpha = {
        0.0: 0.50,
        0.1: 0.51,
        0.25: 0.52,
        0.5: 0.53,
        0.75: 0.54,
        1.0: 0.55,
    }
    error_by_alpha = {
        0.0: 0.10,
        0.1: 0.09,
        0.25: 0.08,
        0.5: 0.07,
        0.75: 0.06,
        1.0: 0.05,
    }
    for seed in module.REQUIRED_SEEDS:
        control = _file_record(tmp_path / f"control{seed}.pt", f"c{seed}".encode())
        candidate = _file_record(tmp_path / f"candidate{seed}.pt", f"t{seed}".encode())
        points = []
        for alpha in module.FIXED_ALPHAS:
            # 0.1 is retained; every larger nonzero alpha violates Surface.
            surface = 0.90 if alpha == 0.0 else (0.899 if alpha == 0.1 else 0.897)
            points.append(
                {
                    "alpha": alpha,
                    "surface": {**_surface(surface), "selection_score": surface},
                    "fit": {
                        "text_support_top1_agreement": support_by_alpha[alpha],
                        "text_response_smooth_l1": error_by_alpha[alpha],
                        "text_support_valid_query_ratio": 1.0,
                        "descriptor_relation_smooth_l1": 0.0,
                    },
                    "dev_posthoc": {
                        "aggregate": {
                            "ranking_spearman_mean": 100.0 * alpha + seed,
                            "smooth_l1": 1.0 - alpha,
                        }
                    },
                }
            )
        per_seed.append(
            {
                "seed": seed,
                "control": {**control, "report": {}, "authority": {}},
                "candidate": {**candidate, "report": {}, "authority": {}},
                "interpolation_compatibility": {"seed": seed, "tensor_count": 24},
                "points": points,
            }
        )
    for alpha in module.FIXED_ALPHAS:
        aggregate.append(
            {
                "alpha": alpha,
                "surface": {"selection_score": 0.0},
                "fit": {
                    "text_support_top1_agreement": support_by_alpha[alpha],
                    "text_response_smooth_l1": error_by_alpha[alpha],
                },
                "dev_posthoc": {"aggregate_seed_mean": {"ranking_spearman_mean": alpha}},
            }
        )
    return {"per_seed": per_seed, "aggregate_seed_mean": aggregate}


def test_formal_rule_uniquely_selects_minimum_positive_alpha(tmp_path: Path) -> None:
    view = module._selection_view(_diagnostic(tmp_path))
    decision = module.select_from_view(view)

    assert decision["feasible_alphas"] == [0.0, 0.1]
    assert decision["positive_feasible_alphas"] == [0.1]
    assert decision["selected_alpha"] == 0.1


def test_dev_values_cannot_change_selection_view_or_decision(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path)
    changed = copy.deepcopy(diagnostic)
    for seed in changed["per_seed"]:
        for point in seed["points"]:
            point["dev_posthoc"] = {
                "malicious_selection_hint": "choose alpha=1",
                "aggregate": {"ranking_spearman_mean": -999999.0},
            }
    for row in changed["aggregate_seed_mean"]:
        row["dev_posthoc"] = {"malicious_selection_hint": row["alpha"]}

    original_view = module._selection_view(diagnostic)
    changed_view = module._selection_view(changed)

    assert changed_view == original_view
    assert module.select_from_view(changed_view) == module.select_from_view(original_view)


def test_surface_rule_is_per_seed_not_only_aggregate(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path)
    # Make alpha=0.1 fail only seed 2; the other two seeds remain feasible.
    failed = next(
        point
        for point in diagnostic["per_seed"][2]["points"]
        if point["alpha"] == 0.1
    )
    failed["surface"]["summary_token_cosine"] = 0.897

    view = module._selection_view(diagnostic)
    with pytest.raises(ValueError, match="no positive alpha"):
        module.select_from_view(view)


def test_fit_support_must_strictly_improve(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path)
    alpha_one_tenth = next(
        row for row in diagnostic["aggregate_seed_mean"] if row["alpha"] == 0.1
    )
    alpha_one_tenth["fit"]["text_support_top1_agreement"] = 0.50
    for seed in diagnostic["per_seed"]:
        point = next(point for point in seed["points"] if point["alpha"] == 0.1)
        point["fit"]["text_support_top1_agreement"] = 0.50

    view = module._selection_view(diagnostic)
    with pytest.raises(ValueError, match="no positive alpha"):
        module.select_from_view(view)


def test_cpu_preflight_requires_hidden_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        module._cpu_only_preflight()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    module._cpu_only_preflight()
