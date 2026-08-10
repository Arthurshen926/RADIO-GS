from __future__ import annotations

from copy import deepcopy

from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21a_rescue as trainer,
)


def _validation(
    *,
    auxiliary: float,
    absolute: tuple[float, float],
    pairwise: float,
    coverage: float = 1.0,
    v1_passed: bool = True,
) -> dict:
    scenes = {}
    for scene, absolute_value in zip(trainer.VALIDATION_SCENES, absolute):
        scenes[scene] = {
            "response_auxiliary_loss": auxiliary,
            "response_absolute_relevance_loss": absolute_value,
            "response_continuous_pairwise_relevance_loss": pairwise,
        }
    return {
        "v1_non_regression": {
            "non_regression_passed": v1_passed,
            "candidate": {
                "mean_all_view_cosine": 0.5,
                "p05_row_mean_all_view_cosine": 0.4,
                "relation_fidelity": 0.3,
            },
        },
        "response_listwise_v21a": {
            "scene_macro_auxiliary_loss": auxiliary,
            "scene_macro_absolute_relevance_loss": sum(absolute) / 2,
            "scene_macro_continuous_pairwise_relevance_loss": pairwise,
            "minimum_active_over_eligible_coverage": coverage,
            "minimum_pair_trainable_endpoint_coverage": coverage,
            "per_scene": scenes,
        },
    }


def _history() -> list[dict]:
    rows = [
        {
            "epoch": 0,
            "validation": _validation(
                auxiliary=1.0, absolute=(1.0, 1.0), pairwise=1.0
            ),
        }
    ]
    for epoch in range(1, 31):
        rows.append(
            {
                "epoch": epoch,
                "validation": _validation(
                    auxiliary=0.9 - 0.01 * epoch,
                    absolute=(0.9, 0.9),
                    pairwise=0.9,
                ),
            }
        )
    return rows


def test_contract_fixes_full_30_steps_without_early_stopping() -> None:
    contract = trainer.training_contract()
    assert contract["optimizer"]["fixed_optimizer_steps"] == 30
    assert contract["optimizer"]["early_stopping"] is False
    assert contract["objective_intervention"]["triplet_training_denominator"] == (
        "authority_pairs_with_trainable_anchor"
    )
    assert trainer.synthetic_dry_run()["benchmark_opened"] is False


def test_selection_filters_by_all_promotion_constraints_then_minimizes_aux() -> None:
    history = _history()
    assert trainer.select_promotion_epoch(history) == 30
    history[30]["validation"]["response_listwise_v21a"][
        "minimum_active_over_eligible_coverage"
    ] = 0.949
    history[30]["validation"]["response_listwise_v21a"][
        "minimum_pair_trainable_endpoint_coverage"
    ] = 0.949
    assert trainer.select_promotion_epoch(history) == 29


def test_no_promotion_epoch_is_distinct_from_best_raw_aux_diagnostic() -> None:
    history = _history()
    for row in history[1:]:
        row["validation"]["v1_non_regression"]["non_regression_passed"] = False
    assert trainer.select_promotion_epoch(history) is None
    assert trainer.select_best_raw_aux_epoch(history) == 30


def test_absolute_relevance_must_not_regress_on_either_scene() -> None:
    history = _history()
    candidate = deepcopy(history[30]["validation"])
    candidate["response_listwise_v21a"]["per_scene"][
        trainer.VALIDATION_SCENES[0]
    ]["response_absolute_relevance_loss"] = 1.01
    candidate["response_listwise_v21a"]["per_scene"][
        trainer.VALIDATION_SCENES[1]
    ]["response_absolute_relevance_loss"] = 0.1
    history[30]["validation"] = candidate
    checks = trainer.promotion_checks(history, 30)
    assert checks["absolute_relevance_macro_strictly_improved"] is True
    assert checks["absolute_relevance_every_scene_non_regression"] is False


def test_gradient_diagnostics_use_fixed_component_axis() -> None:
    import torch
    from radio_gs.models.surface_region_typed_context_residual import (
        SurfaceRegionAcceptedV2TypedContextResidualV1,
    )

    model = SurfaceRegionAcceptedV2TypedContextResidualV1()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    metrics = trainer._component_gradient_norms(model)
    assert set(metrics) == {
        "descriptor_projection",
        "context_projection",
        "scalar_projection",
        "fusion_projection",
        "residual_projection",
        "global",
    }
    assert all(value > 0 for value in metrics.values())
