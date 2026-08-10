from __future__ import annotations

import torch

from radio_gs.scripts.materialize_source_same_axis_o0_external_query_subset_scene0003 import (
    SPLIT as PRODUCER_SPLIT,
)
from radio_gs.scripts.validate_source_multiscene_monotone_missing_core_selector_scene0003 import (
    EXTERNAL_SPLIT,
    REGISTERED_MEMBERSHIP_SPLIT,
    evaluate_selector_gate,
    fixed_validation,
    source_access,
    validate_raw_pre_membership_value,
)


def _gate_population() -> tuple[torch.Tensor, ...]:
    selected_labels = torch.cat((torch.ones(270), torch.zeros(30))).bool()
    heldout_labels = torch.cat((torch.ones(100), torch.zeros(600))).bool()
    labels = torch.cat((selected_labels, heldout_labels))
    selected = torch.zeros(1000, dtype=torch.bool)
    selected[:300] = True
    probability = torch.cat(
        (
            torch.where(selected_labels, torch.full((300,), 0.95), torch.full((300,), 0.85)),
            torch.where(heldout_labels, torch.full((700,), 0.35), torch.full((700,), 0.05)),
        )
    )
    utility = torch.where(labels, torch.ones(1000), -torch.ones(1000))
    o0 = torch.zeros(1000)
    return labels, utility, probability, o0, selected


def test_scene0003_validator_is_external_fixed_and_target_blind() -> None:
    assert EXTERNAL_SPLIT == "source_external_validation_multisource_selector_only"
    assert EXTERNAL_SPLIT == PRODUCER_SPLIT
    assert REGISTERED_MEMBERSHIP_SPLIT == "source_train"
    fixed = fixed_validation()
    assert fixed["threshold_or_model_refit_on_scene0003"] is False
    assert fixed["selector_gate"]["minimum_hard_precision_Wilson95_lower"] == 0.75
    assert fixed["selector_gate"]["require_selected_signed_utility_above_unconditional"]
    assert fixed["selector_gate"]["require_selector_AP_strictly_above_unit_O0_score_AP"]
    access = source_access()
    assert access["scene0003_not_used_for_selector_fit_or_threshold_selection"]
    assert access["scene0004_membership_opened"] is False
    assert access["target_benchmark_opened"] is False
    assert access["benchmark_labels_opened"] is False


def test_scene0003_fixed_gate_requires_wilson_utility_and_ap() -> None:
    labels, utility, probability, o0, selected = _gate_population()
    outcomes, checks = evaluate_selector_gate(
        labels=labels,
        signed_utility=utility,
        probability=probability,
        unit_o0_score=o0,
        selected_mask=selected,
    )
    assert checks["passed"]
    assert outcomes["selected_hard_precision_Wilson95_lower"] >= 0.75
    assert (
        outcomes["selected_signed_utility_mean"]
        > outcomes["unconditional_signed_utility_mean"]
    )
    assert (
        outcomes["selector_average_precision"]
        > outcomes["unit_O0_score_average_precision"]
    )


def test_scene0003_gate_rejects_non_improving_utility() -> None:
    labels, _, probability, o0, selected = _gate_population()
    utility = torch.ones(1000)
    utility[selected] = 0.1
    _, checks = evaluate_selector_gate(
        labels=labels,
        signed_utility=utility,
        probability=probability,
        unit_o0_score=o0,
        selected_mask=selected,
    )
    assert not checks["selected_signed_utility_above_unconditional"]
    assert not checks["passed"]


def test_scene0003_gate_rejects_no_ap_gain() -> None:
    labels, utility, _, _, selected = _gate_population()
    constant = torch.zeros(1000)
    _, checks = evaluate_selector_gate(
        labels=labels,
        signed_utility=utility,
        probability=constant,
        unit_o0_score=constant,
        selected_mask=selected,
    )
    assert not checks["selector_AP_strictly_above_unit_O0_score_AP"]
    assert not checks["passed"]


def test_scene0003_raw_lineage_requires_exact_deferred_membership_state() -> None:
    def record(name: str) -> dict[str, str]:
        return {"path": f"/sealed/{name}", "sha256": name[0] * 64}

    authority = {
        "subset_execution_authority": record("a_subset"),
        "dense_stream_execution_authority": record("b_dense"),
        "combined_text_subset": record("c_subset"),
        "raw_combined_multiscale_scores": record("d_raw"),
        "accepted_v2": record("e_accepted"),
        "source_full_scalar_shard": record("f_shard"),
        "source_capability_descriptor": record("a_capability"),
        "source_membership_authority": record("b_membership"),
    }
    raw = {
        "schema": "radio_gs.source_same_axis_o0_scene0003_raw_pre_membership_authority.v1",
        "status": "sealed_after_scene0003_raw_O0_before_multisource_selector_and_membership_payload_open",
        "scene_id": "scene0003_00",
        "split": EXTERNAL_SPLIT,
        "registered_membership_split": REGISTERED_MEMBERSHIP_SPLIT,
        "subset_authority": authority["subset_execution_authority"],
        "dense_stream_authority": authority["dense_stream_execution_authority"],
        "combined_text_subset": {**authority["combined_text_subset"], "positive_queries": 1},
        "raw_combined_multiscale_scores": {**authority["raw_combined_multiscale_scores"], "shape": [1]},
        "source_assets": {
            "accepted_v2": authority["accepted_v2"],
            "source_full_scalar_shard": authority["source_full_scalar_shard"],
            "source_capability_descriptor": authority["source_capability_descriptor"],
            "source_membership_authority_deferred": {
                **authority["source_membership_authority"],
                "payload_opened": False,
            },
        },
        "source_access": {
            "scene0003_membership_payload_opened": False,
            "multisource_selector_opened": False,
        },
        "benchmark_execution_authorized": False,
    }
    validate_raw_pre_membership_value(raw, authority)
    raw["source_assets"]["source_membership_authority_deferred"][
        "payload_opened"
    ] = True
    try:
        validate_raw_pre_membership_value(raw, authority)
    except ValueError:
        pass
    else:
        raise AssertionError("opened membership payload must fail lineage validation")
