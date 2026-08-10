from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from radio_gs.scripts import materialize_factorized_native_dba_v2_figurines as target


def test_structural_contract_is_metric_closed_and_method_level() -> None:
    contract = target.structural_contract()
    assert contract["candidate_boundary"] == 0.5
    assert contract["o0_strong_boundary"] == 0.6
    assert contract["checks"]["o0_strong_query_recall_at_least"] == 0.8
    assert contract["checks"]["candidate_coverage_gain_over_contrast_v21_at_least"] == 5
    assert contract["metric_execution_authorized"] is False
    assert target.STRUCTURAL_CONTRACT_SHA256 == target.canonical_json_sha256(contract)


def test_access_audit_never_opens_gt_or_metric() -> None:
    before = target.access_audit(query_opened=False)
    after = target.access_audit(query_opened=True)
    assert before["exact_query_protocol_opened"] is False
    assert after["exact_query_protocol_opened"] is True
    for audit in (before, after):
        assert audit["benchmark_masks_opened"] is False
        assert audit["benchmark_labels_opened"] is False
        assert audit["benchmark_metrics_opened"] is False
        assert audit["target_metrics_computed"] is False


def test_descriptor_channel_changes_with_semantic_descriptor() -> None:
    payload = {
        "schema": target.DESCRIPTOR_SCHEMA,
        "scene_id": "figurines",
        "physical_space_id": "space",
        "source_selected_step": 40,
        "region_row_ids": ["a", "b"],
        "region_fingerprints": ["x", "y"],
        "canonical_region_indices": torch.tensor([1, 2]),
        "semantic_descriptor": torch.eye(2),
    }
    first = target._descriptor_channel(payload)
    payload["semantic_descriptor"] = -payload["semantic_descriptor"]
    assert target._descriptor_channel(payload) != first


def _minimal_o0_pair() -> tuple[dict[str, object], dict[str, object], list[str], str]:
    query_ids = ["bag", "chair"]
    count = 12
    renderer = "a" * 64
    common = {
        "version": 4,
        "contract": "radio_gs.ours_lerf_direct3d_multiscale_query_scores_fp32.v4",
        "scale_ids": list(target.EXPECTED_O0_SCALE_IDS),
        "scale_radii_m": list(target.EXPECTED_O0_SCALE_RADII_M),
        "xyz": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
        "valid": torch.ones(count, dtype=torch.bool),
        "geometry_fingerprint": {"xyz_sha256": "b" * 64},
        "field_checkpoint_sha256": "c" * 64,
        "readout_checkpoint_sha256": "d" * 64,
        "renderer_geometry_checkpoint_sha256": renderer,
    }
    positive = {
        **deepcopy(common),
        "query_ids": list(query_ids),
        "query_scores": torch.linspace(-1, 1, count * 3 * 2).reshape(count, 3, 2),
    }
    negative = {
        **deepcopy(common),
        "query_ids": list(target.exact_o0.v2.frozen.NEGATIVE_PROMPTS),
        "query_scores": torch.linspace(1, -1, count * 3 * 4).reshape(count, 3, 4),
    }
    physical_space_id = f"lerf:figurines:geometry-checkpoint-sha256:{renderer}"
    return positive, negative, query_ids, physical_space_id


def test_selected_source_step_other_than_40_is_rejected() -> None:
    steps = list(
        range(
            0,
            target.dba_v2.OPTIMIZER_STEPS + 1,
            target.dba_v2.EVALUATION_INTERVAL,
        )
    )
    history = [{"step": step} for step in steps]
    with pytest.raises(ValueError, match="selected source history"):
        target._validate_selected_source_history(history, 32)


@pytest.mark.parametrize(
    "mutation",
    [
        "xyz",
        "valid",
        "scale_ids",
        "scale_radii_m",
        "renderer",
        "positive_query",
        "negative_query",
        "geometry",
    ],
)
def test_o0_pair_axis_and_lineage_mismatches_are_rejected(mutation: str) -> None:
    positive, negative, query_ids, physical_space_id = _minimal_o0_pair()
    if mutation == "xyz":
        negative["xyz"][0, 0] += 1
    elif mutation == "valid":
        negative["valid"][0] = False
    elif mutation == "scale_ids":
        negative["scale_ids"][-1] = "0.8"
    elif mutation == "scale_radii_m":
        positive["scale_radii_m"][-1] = 0.8
    elif mutation == "renderer":
        negative["renderer_geometry_checkpoint_sha256"] = "e" * 64
    elif mutation == "positive_query":
        positive["query_ids"][0] = "wrong"
    elif mutation == "negative_query":
        negative["query_ids"][0] = "wrong"
    elif mutation == "geometry":
        negative["geometry_fingerprint"] = {"xyz_sha256": "f" * 64}
    with pytest.raises(ValueError, match="exact O0"):
        target._validate_o0_pair(
            positive,
            negative,
            query_ids=query_ids,
            physical_space_id=physical_space_id,
        )


def test_o0_relevance_delegates_to_exact_knn_peak_minmax_not_raw_sigmoid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positive, negative, _, _ = _minimal_o0_pair()
    expected = torch.tensor([[-1.0, 0.25], [0.5, 1.0]], dtype=torch.float32)
    called: dict[str, object] = {}

    def exact(**kwargs: object) -> SimpleNamespace:
        called.update(kwargs)
        return SimpleNamespace(final_scores=expected)

    monkeypatch.setattr(target.exact_o0, "exact_o0_readout", exact)
    actual = target._o0_relevance(positive, negative)
    assert actual is expected
    assert called["chunk_size"] == 65536
    assert called["positive_scores"] is positive["query_scores"]
    assert called["negative_scores"] is negative["query_scores"]
    assert float(actual.min()) < 0.0  # impossible for a raw sigmoid/probability readout


def _legacy_and_descriptor() -> tuple[dict[str, object], dict[str, object], list[str]]:
    query_ids = ["bag", "chair"]
    rows = [f"row-{index}" for index in range(4096)]
    fingerprints = [f"{index:064x}" for index in range(4096)]
    canonical = torch.arange(4096, dtype=torch.int64)
    descriptor = {
        "scene_id": "figurines",
        "physical_space_id": "space",
        "region_row_ids": rows,
        "canonical_region_indices": canonical,
        "region_fingerprints": fingerprints,
    }
    old = {
        **deepcopy(descriptor),
        "query_ids": list(query_ids),
        "region_absolute_relevance": torch.zeros(4096, 2, dtype=torch.float32),
    }
    return old, descriptor, query_ids


@pytest.mark.parametrize(
    "mutation", ["scene", "physical", "row", "canonical", "fingerprint", "query", "dtype", "finite"]
)
def test_legacy_relevance_full_axis_mismatches_are_rejected(mutation: str) -> None:
    old, descriptor, query_ids = _legacy_and_descriptor()
    if mutation == "scene":
        old["scene_id"] = "wrong"
    elif mutation == "physical":
        old["physical_space_id"] = "wrong"
    elif mutation == "row":
        old["region_row_ids"][0] = "wrong"
    elif mutation == "canonical":
        old["canonical_region_indices"][0] = 7
    elif mutation == "fingerprint":
        old["region_fingerprints"][0] = "f" * 64
    elif mutation == "query":
        old["query_ids"][0] = "wrong"
    elif mutation == "dtype":
        old["region_absolute_relevance"] = old["region_absolute_relevance"].double()
    elif mutation == "finite":
        old["region_absolute_relevance"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="legacy relevance full axes"):
        target._validate_legacy_query_axes(old, descriptor, query_ids)


@pytest.mark.parametrize(
    "mutation", ["selected", "source_result", "source_checkpoint", "execution_result", "execution_checkpoint"]
)
def test_descriptor_source_record_mismatches_are_rejected(mutation: str) -> None:
    result_record = {"path": "/result.json", "sha256": "a" * 64}
    checkpoint_record = {"path": "/model.pt", "sha256": "b" * 64}
    source = {
        "result": {"verified_record": result_record},
        "checkpoint": {"verified_record": checkpoint_record},
    }
    descriptor = {
        "source_selected_step": 40,
        "input_authority": {
            "source_result": deepcopy(result_record),
            "source_checkpoint": deepcopy(checkpoint_record),
        },
    }
    execution = {"verified_source": deepcopy(source)}
    if mutation == "selected":
        descriptor["source_selected_step"] = 32
    elif mutation == "source_result":
        descriptor["input_authority"]["source_result"]["sha256"] = "c" * 64
    elif mutation == "source_checkpoint":
        descriptor["input_authority"]["source_checkpoint"]["sha256"] = "c" * 64
    elif mutation == "execution_result":
        execution["verified_source"]["result"]["verified_record"]["sha256"] = "c" * 64
    elif mutation == "execution_checkpoint":
        execution["verified_source"]["checkpoint"]["verified_record"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="descriptor source binding"):
        target._validate_descriptor_source_binding(descriptor, execution, source)


def test_any_false_structural_gate_rejects_and_never_recommends_metric() -> None:
    names = tuple(target.structural_contract()["checks"])
    for false_name in names:
        checks = {name: True for name in names}
        checks[false_name] = False
        status, metric_recommended = target._audit_decision(checks)
        assert status == "REJECT"
        assert metric_recommended is False


@pytest.mark.parametrize("checks", [{}, {"gate": 1}, {"gate": None}])
def test_structural_gate_requires_nonempty_exact_booleans(checks: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="non-empty booleans"):
        target._audit_decision(checks)
