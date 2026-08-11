from pathlib import Path

import pytest
import torch

from radio_gs.field.field_signature import FeatureSpaceSignature
from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.multiview_region_memory import METHOD, method_contract
from radio_gs.querying.multiview_region_memory_runtime import (
    ARTIFACT_TYPE,
    augment_query_with_region_tokens,
    complete_abstaining_observation,
    load_region_memory,
)
from radio_gs.querying.query_spec import (
    PrimitiveUnaryEvidence,
    PrototypeSet,
    QueryIntent,
    QueryModality,
    QuerySpec,
    RegistrationMode,
    SoftSeedSet,
)
from radio_gs.scripts import materialize_multiview_region_memory as materializer
from radio_gs.utils.immutable_artifacts import file_record, sha256_file


def _signature(dim: int) -> FeatureSpaceSignature:
    return FeatureSpaceSignature(
        radio_version="unit",
        radio_checkpoint_sha256="a",
        raw_feature_dim=dim,
        adaptor_name="unit",
        adaptor_sha256="b",
        adaptor_output_dim=dim,
        token_type="primitive",
    )


def _memory(tmp_path: Path):
    tensors = {
        "valid_rows": torch.tensor([1, 3], dtype=torch.long),
        "membership_probability": torch.tensor([0.9, 0.1]),
        "membership_confidence": torch.tensor([0.4, 0.3]),
        "membership_observed": torch.tensor([True, True]),
        "positive_mass_by_view": torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
        ),
        "proposal_masks_feature": torch.ones((3, 1, 2), dtype=torch.bool),
        "observation_domains_feature": torch.ones((3, 1, 2), dtype=torch.bool),
        "view_reliability": torch.tensor([0.3, 0.6, 0.9]),
    }
    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "primitive_region_memory_sealed_before_target_access",
        "scene_id": "scene",
        "method": METHOD,
        "method_contract": method_contract(),
        "num_gaussians": 4,
        "view_count": 3,
        "views": [{"selection_rank": i} for i in range(3)],
        "capability_cache": {
            "path": str(tmp_path / "capability.pt"),
            "sha256": "c" * 64,
        },
        "tensor_sha256": {
            name: tensor_sha256(value) for name, value in tensors.items()
        },
        "source_access": {
            "source_rgb_opened_by_upstream_sam3": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
            "candidate_selected_with_gt": False,
        },
        "implementation": file_record(Path(materializer.__file__)),
        **tensors,
    }
    path = tmp_path / "memory.pt"
    torch.save(payload, path)
    return load_region_memory(
        path,
        expected_sha256=sha256_file(path),
        scene_id="scene",
        capability_path=tmp_path / "capability.pt",
        capability_sha256="c" * 64,
        global_rows=torch.tensor([1, 3]),
        num_gaussians=4,
    )


def test_runtime_loader_and_completion_preserve_observed_rows_bitwise(tmp_path):
    memory = _memory(tmp_path)
    base = PrimitiveUnaryEvidence(
        torch.tensor([0.7, 0.0]),
        "base",
        confidence=torch.tensor([0.8, 0.0]),
    )
    completed, changed, diagnostics = complete_abstaining_observation(base, memory)
    assert torch.equal(changed, torch.tensor([False, True]))
    assert torch.equal(completed.values[:1], base.values[:1])
    assert torch.equal(completed.confidence[:1], base.confidence[:1])
    assert completed.values[1] == pytest.approx(-0.24)
    assert completed.confidence[1] == pytest.approx(0.3)
    assert diagnostics.completed_rows == 1
    assert diagnostics.observed_values_bitwise_equal is True
    assert diagnostics.observed_confidence_bitwise_equal is True


def test_region_tokens_append_three_reliability_weighted_prototypes(tmp_path):
    memory = _memory(tmp_path)
    signature = _signature(2)
    observation = PrimitiveUnaryEvidence.from_probability(
        torch.tensor([0.8, 0.2]),
        confidence=torch.tensor([0.5, 0.5]),
        source="base",
    )
    query = QuerySpec(
        modality=QueryModality.REGISTERED_2D,
        intent=QueryIntent.REGION,
        registration=RegistrationMode.CAMERA,
        appearance_evidence=PrototypeSet(torch.tensor([[1.0, 0.0]]), signature),
        boundary_evidence=PrototypeSet(torch.tensor([[0.0, 1.0]]), signature),
        positive_seeds=SoftSeedSet(torch.tensor([1.0, 0.0]), "base"),
        primitive_unary_evidence=observation,
    )
    banks = {
        "appearance": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "boundary": torch.tensor([[1.0, 1.0], [1.0, -1.0]]),
    }
    augmented, diagnostics = augment_query_with_region_tokens(
        query, banks, memory, chunk_size=1
    )
    assert augmented.appearance_evidence.features.shape == (4, 2)
    assert augmented.boundary_evidence.features.shape == (4, 2)
    assert augmented.positive_seeds is query.positive_seeds
    assert augmented.primitive_unary_evidence is query.primitive_unary_evidence
    assert float(augmented.appearance_evidence.weights.sum()) == pytest.approx(1.0)
    assert diagnostics["appearance_token_count"] == 3
    assert diagnostics["source_raw_weight_sum"] == pytest.approx(0.6)
