from pathlib import Path

import pytest
import torch

from radio_gs.field.field_signature import FeatureSpaceSignature
from radio_gs.field.primitive_reliability import canonical_primitive_reliability
from radio_gs.interfaces.capability_cache import (
    load_canonical_primitive_reliability,
)
from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.query_spec import (
    PrototypeSet,
    QueryIntent,
    QueryModality,
    QuerySpec,
    RegistrationMode,
    SelectionMode,
    SoftSeedSet,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _signature() -> FeatureSpaceSignature:
    return FeatureSpaceSignature(
        radio_version="test",
        radio_checkpoint_sha256="radio-hash",
        raw_feature_dim=2,
        adaptor_name="test",
        adaptor_sha256="adaptor-hash",
        adaptor_output_dim=2,
        token_type="primitive",
        normalization="l2",
    )


def _empty_graph(count: int) -> PrimitiveSupportGraph:
    return PrimitiveSupportGraph(
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_weight=torch.empty(0),
        raw_affinity=torch.empty(0),
        local_sigma=torch.ones(count),
        num_nodes=count,
    )


def test_canonical_reliability_is_query_free_monotonic_and_invalid_zero():
    result = canonical_primitive_reliability(
        torch.tensor([1, 2, 5, 0]),
        torch.tensor(
            [
                [0.1, 1.0, 1.0],
                [0.2, 1.0, 0.0],
                [0.5, 1.0, 0.5],
                [0.0, 1.0, 1.0],
            ]
        ),
        torch.tensor([0.9, 0.9, 0.9, 1.0]),
        valid=torch.tensor([True, True, True, False]),
    )
    assert result.confidence[0] < result.confidence[1] < result.confidence[2]
    assert result.confidence[3] == 0
    # The ambiguous historical third channel cannot change the result.
    changed_third = canonical_primitive_reliability(
        torch.tensor([1, 2, 5, 0]),
        torch.tensor(
            [
                [0.1, 1.0, 0.0],
                [0.2, 1.0, 1.0],
                [0.5, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        torch.tensor([0.9, 0.9, 0.9, 1.0]),
        valid=torch.tensor([True, True, True, False]),
    )
    torch.testing.assert_close(result.confidence, changed_third.confidence)


def test_reconstruction_and_agreement_reduce_confidence():
    base = canonical_primitive_reliability(
        torch.tensor([4, 4, 4]),
        torch.tensor([[1.0, 1.0], [1.0, 0.5], [1.0, 1.0]]),
        torch.tensor([1.0, 1.0, 0.5]),
    )
    assert base.confidence[1] < base.confidence[0]
    assert base.confidence[2] < base.confidence[0]


def test_query_engine_shrinks_feature_unary_but_not_seed_constraints():
    signature = _signature()
    query = QuerySpec(
        modality=QueryModality.REGISTERED_2D,
        intent=QueryIntent.REGION,
        registration=RegistrationMode.CAMERA,
        appearance_evidence=PrototypeSet(torch.tensor([[1.0, 0.0]]), signature),
        positive_seeds=SoftSeedSet(torch.tensor([1.0, 0.0]), "test"),
        selection_mode=SelectionMode.SEEDED_COMPONENT,
    )
    engine = CanonicalQueryEngine(
        _empty_graph(2), node_reliability=torch.tensor([0.5, 0.2])
    )
    result = engine.execute(
        query,
        {"appearance": torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
        feature_signatures={"appearance": signature},
    )
    torch.testing.assert_close(result.unary, torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(
        result.evidence_components["appearance"], torch.tensor([0.5, 0.0])
    )
    assert result.reliability_applied is True
    assert result.probabilities[0] == 1.0


def test_seed_only_query_keeps_neutral_prior_contract():
    query = QuerySpec(
        modality=QueryModality.WORLD_3D,
        intent=QueryIntent.INSTANCE,
        registration=RegistrationMode.WORLD,
        positive_seeds=SoftSeedSet(torch.tensor([1.0, 0.0]), "test"),
        selection_mode=SelectionMode.SEEDED_COMPONENT,
    )
    result = CanonicalQueryEngine(
        _empty_graph(2), node_reliability=torch.tensor([0.1, 0.9])
    ).execute(query, {})
    torch.testing.assert_close(result.unary, torch.full((2,), -1.0))
    assert result.reliability_applied is False


def _write_reliability(path: Path, *, xyz: torch.Tensor, valid: torch.Tensor) -> None:
    confidence = torch.tensor([0.8, 0.0])
    torch.save(
        {
            "schema_version": 1,
            "xyz": xyz,
            "valid": valid,
            "confidence": confidence,
            "components": {
                "observation_evidence": confidence,
                "multiview_agreement": confidence,
                "reconstruction_fidelity": confidence,
            },
            "metadata": {
                "source": "canonical_primitive_reliability_v1",
                "field_checkpoint_sha256": "field-hash",
                "query_independent": True,
                "uses_query": False,
                "uses_text": False,
                "uses_target_labels": False,
                "uses_target_masks": False,
                "uses_metric_feedback": False,
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
            },
        },
        path,
    )


def test_reliability_loader_fails_closed_on_alignment_and_hash(tmp_path: Path):
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    valid = torch.tensor([True, False])
    path = tmp_path / "reliability.pt"
    _write_reliability(path, xyz=xyz, valid=valid)
    loaded = load_canonical_primitive_reliability(
        path,
        expected_xyz=xyz,
        expected_valid=valid,
        expected_field_checkpoint_sha256="field-hash",
    )
    torch.testing.assert_close(loaded.valid_confidence(), torch.tensor([0.8]))
    with pytest.raises(ValueError, match="geometry does not align"):
        load_canonical_primitive_reliability(path, expected_xyz=xyz + 1)
    with pytest.raises(ValueError, match="field hash mismatch"):
        load_canonical_primitive_reliability(
            path, expected_field_checkpoint_sha256="wrong"
        )
