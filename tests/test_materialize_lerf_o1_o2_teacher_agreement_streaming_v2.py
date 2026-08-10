from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from radio_gs.interfaces.lerf_reliability_geodesic_budget import (
    VIEW_AGREEMENT_SCALAR,
    VIEW_AGREEMENT_SHA256_FIELD,
    reliability_conditioned_geodesic_fusion,
)
from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as stream,
)
from radio_gs.scripts import materialize_lerf_o1_o2_streaming as core
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def test_v2_contract_binds_entrypoint_and_untouched_core_and_keeps_ceiling_closed() -> None:
    contract = stream.method_contract()
    assert contract["schema"].endswith(".v2")
    assert contract["schema_version"] == 2
    assert contract["streaming_entrypoint_implementation"] == (
        stream.ENTRYPOINT_IMPLEMENTATION
    )
    assert contract["streaming_core_implementation"] == stream.CORE_IMPLEMENTATION
    assert contract["teacher_payload"]["additional_tensor"] == (
        VIEW_AGREEMENT_SCALAR
    )
    assert contract["teacher_payload"]["additional_tensor_hash"] == (
        VIEW_AGREEMENT_SHA256_FIELD
    )
    assert contract["O1"]["maximum_angle_radians"] == 0.15
    assert contract["O2"] == core.method_contract()["O2"]
    assert contract["execution_device_authority"] == {
        "implemented_physical_gpu": 0,
        "required_cuda_visible_devices": "0",
        "program_device": "cuda:0",
        "other_physical_gpu_authorized": False,
    }
    assert contract["reliability_budget_candidate_materialized"] is False
    assert contract["reliability_budget_maximum_angle_authorized"] is False
    audit = contract["source_only_leave_one_view_out_ceiling_audit"]
    assert audit["candidate_maximum_angles_radians"] == [0.15, 0.3, 0.45, 0.6, 0.75]
    assert audit["target_candidate_authorization"] is False
    assert audit["durable_per_view_descriptors"] is False
    assert stream.METHOD_CONTRACT_SHA256 == canonical_json_sha256(contract)


def _top_view_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    descriptors = torch.zeros(3, 4, 1536, dtype=torch.float16)
    frame_ids = torch.full((3, 4), -1, dtype=torch.int32)
    # Four aligned directions -> one.
    descriptors[0, :, 0] = 1.0
    frame_ids[0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    # Two orthogonal directions -> sqrt(2)/2.
    descriptors[1, 0, 0] = 1.0
    descriptors[1, 1, 1] = 1.0
    frame_ids[1, :2] = torch.tensor([7, 9], dtype=torch.int32)
    # Third row is invalid and must remain exact zero.
    return descriptors, frame_ids


def test_small_tensor_resultant_is_unit_direction_formula_and_invalid_exact_zero() -> None:
    descriptors, frame_ids = _top_view_fixture()
    resultant, count = stream.directional_resultant_from_canonical_top_views(
        descriptors, frame_ids
    )
    torch.testing.assert_close(
        resultant,
        torch.tensor([1.0, 2.0**-0.5, 0.0]),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.equal(count, torch.tensor([4, 2, 0], dtype=torch.uint8))
    assert resultant.dtype == torch.float32
    assert resultant[2].item() == 0.0


def test_fp16_quantization_is_renormalized_for_resultant_but_not_mutated() -> None:
    descriptors = F.normalize(torch.randn(2, 4, 1536), dim=-1).half()
    frame_ids = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    original = descriptors.clone()
    resultant, _ = stream.directional_resultant_from_canonical_top_views(
        descriptors, frame_ids
    )
    expected = torch.linalg.vector_norm(
        F.normalize(descriptors.float(), dim=-1).sum(dim=1), dim=-1
    ) / 4.0
    torch.testing.assert_close(resultant, expected)
    assert torch.equal(descriptors, original)


def test_resultant_row_chunking_does_not_change_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors = F.normalize(torch.randn(5, 4, 1536), dim=-1).half()
    frame_ids = torch.arange(20, dtype=torch.int32).reshape(5, 4)
    monkeypatch.setattr(stream, "AGREEMENT_ROW_CHUNK", 2)
    resultant, count = stream.directional_resultant_from_canonical_top_views(
        descriptors, frame_ids
    )
    expected = torch.linalg.vector_norm(
        F.normalize(descriptors.float(), dim=-1).sum(dim=1), dim=-1
    ) / 4.0
    torch.testing.assert_close(resultant, expected)
    assert torch.equal(count, torch.full((5,), 4, dtype=torch.uint8))


def _teacher_payload() -> dict[str, object]:
    mean = torch.zeros(3, 1536, dtype=torch.float16)
    mean[:2, 0] = 1.0
    payload: dict[str, object] = {
        "schema": stream.MEAN_SCHEMA,
        "schema_version": stream.SCHEMA_VERSION,
        "scene_id": "source_scene",
        "global_rows": torch.tensor([0, 2, 4]),
        "teacher_mean": mean,
        "teacher_valid": torch.tensor([True, True, False]),
        "retained_view_count": torch.tensor([4, 2, 0], dtype=torch.uint8),
        "producer": dict(stream.ENTRYPOINT_IMPLEMENTATION),
        "execution_authority": {"path": "/tmp/a.json", "sha256": "a" * 64},
        "input_authority": {},
        "method_contract_sha256": stream.METHOD_CONTRACT_SHA256,
        "teacher_mean_sha256": core.tensor_sha256_typed(mean),
        "access_audit": core.access_audit(),
    }
    return payload


def _loo_audit() -> dict[str, object]:
    descriptors = torch.zeros(3, 4, 1536, dtype=torch.float16)
    frame_ids = torch.full((3, 4), -1, dtype=torch.int32)
    descriptors[0, :, 1] = 1.0
    frame_ids[0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    descriptors[1, 0, 1] = 1.0
    descriptors[1, 1, 2] = 1.0
    frame_ids[1, :2] = torch.tensor([5, 6], dtype=torch.int32)
    base = torch.zeros(3, 3, 1536, dtype=torch.float16)
    base[..., 0] = 1.0
    return stream.source_only_leave_one_view_out_ceiling_audit(
        descriptors, frame_ids, base
    )


def test_source_only_loo_summary_makes_global_ceiling_gate_executable() -> None:
    audit = _loo_audit()
    stream.validate_source_only_loo_ceiling_audit(audit)
    assert audit["query_independent"] is True
    assert audit["target_candidate_authorized"] is False
    assert audit["rows_with_valid_loo_prediction"] == 2
    assert audit["rows_with_expansion_evidence"] == 1
    assert audit["heldout_predictions"] == 6
    assert audit["heldout_scale_observations"] == 18
    assert audit["candidates"][0]["mean_delta_cosine_vs_o1_0p15"] == 0.0
    assert audit["candidates"][-1]["mean_delta_cosine_vs_o1_0p15"] > 0.0
    assert audit["cross_scene_gate"]["pooled_improvement_required"] is True
    assert audit["cross_scene_gate"][
        "every_source_scene_nonregression_required"
    ] is True


def test_loo_semantic_tamper_fails_even_with_recomputed_summary_hash() -> None:
    result = stream.augment_teacher_payload_v2(
        _teacher_payload(),
        resultant=torch.tensor([1.0, 0.5, 0.0]),
        expected_counts=torch.tensor([4, 2, 0], dtype=torch.uint8),
        loo_audit=_loo_audit(),
    )
    result[stream.LOO_AUDIT_FIELD]["target_candidate_authorized"] = True
    result[stream.LOO_AUDIT_SHA256_FIELD] = canonical_json_sha256(
        result[stream.LOO_AUDIT_FIELD]
    )
    with pytest.raises(ValueError, match="LOO ceiling audit contract"):
        stream.validate_teacher_payload_v2(result)


def test_loo_analytic_audit_matches_deployed_reliability_fusion() -> None:
    generator = torch.Generator().manual_seed(17)
    views = F.normalize(torch.randn(1, 4, 1536, generator=generator), dim=-1).half()
    frame_ids = torch.arange(4, dtype=torch.int32).reshape(1, 4)
    base = F.normalize(torch.randn(1, 3, 1536, generator=generator), dim=-1).half()
    audit = stream.source_only_leave_one_view_out_ceiling_audit(
        views, frame_ids, base
    )
    expected_o1 = 0.0
    expected_max = 0.0
    unit_views = F.normalize(views.float(), dim=-1)
    for heldout_index in range(4):
        kept = torch.arange(4) != heldout_index
        retained = unit_views[:, kept]
        retained_sum = retained.sum(dim=1)
        agreement = torch.linalg.vector_norm(retained_sum, dim=-1) / 3.0
        teacher = F.normalize(retained_sum, dim=-1)
        common = {
            "teacher_valid": torch.tensor([True]),
            "retained_view_count": torch.tensor([3]),
        }
        o1 = reliability_conditioned_geodesic_fusion(
            F.normalize(base.float(), dim=-1), teacher, **common
        ).descriptor
        maximum = reliability_conditioned_geodesic_fusion(
            F.normalize(base.float(), dim=-1),
            teacher,
            teacher_view_directional_resultant=agreement,
            **common,
        ).descriptor
        heldout = unit_views[:, heldout_index, None, :]
        expected_o1 += float((o1.float() * heldout).sum())
        expected_max += float((maximum.float() * heldout).sum())
    assert audit["candidates"][0]["cosine_sum"] == pytest.approx(
        expected_o1, abs=2e-5
    )
    assert audit["candidates"][-1]["cosine_sum"] == pytest.approx(
        expected_max, abs=2e-5
    )


def test_payload_augmentation_adds_typed_hash_and_preserves_o1_o2_inputs() -> None:
    payload = _teacher_payload()
    original_mean = payload["teacher_mean"].clone()
    result = stream.augment_teacher_payload_v2(
        payload,
        resultant=torch.tensor([1.0, 0.5, 0.0]),
        expected_counts=torch.tensor([4, 2, 0], dtype=torch.uint8),
        loo_audit=_loo_audit(),
    )
    stream.validate_teacher_payload_v2(result)
    assert torch.equal(result["teacher_mean"], original_mean)
    assert result[VIEW_AGREEMENT_SHA256_FIELD] == core.tensor_sha256_typed(
        result[VIEW_AGREEMENT_SCALAR]
    )
    assert VIEW_AGREEMENT_SCALAR not in payload


@pytest.mark.parametrize(
    "mutation", ["hash", "invalid", "count", "schema", "loo_hash"]
)
def test_teacher_payload_v2_tampering_fails_closed(mutation: str) -> None:
    result = stream.augment_teacher_payload_v2(
        _teacher_payload(),
        resultant=torch.tensor([1.0, 0.5, 0.0]),
        expected_counts=torch.tensor([4, 2, 0], dtype=torch.uint8),
        loo_audit=_loo_audit(),
    )
    broken = deepcopy(result)
    if mutation == "hash":
        broken[VIEW_AGREEMENT_SHA256_FIELD] = "f" * 64
    elif mutation == "invalid":
        broken[VIEW_AGREEMENT_SCALAR][2] = 0.25
        broken[VIEW_AGREEMENT_SHA256_FIELD] = core.tensor_sha256_typed(
            broken[VIEW_AGREEMENT_SCALAR]
        )
    elif mutation == "count":
        broken["retained_view_count"][2] = 1
    elif mutation == "schema":
        broken["schema_version"] = 1
    elif mutation == "loo_hash":
        broken[stream.LOO_AUDIT_SHA256_FIELD] = "e" * 64
    with pytest.raises(ValueError, match="payload contract"):
        stream.validate_teacher_payload_v2(broken)


def test_unretained_descriptor_and_captured_count_mismatch_fail_closed() -> None:
    descriptors, frame_ids = _top_view_fixture()
    descriptors[2, 0, 0] = 1.0
    with pytest.raises(ValueError, match="exact zero"):
        stream.directional_resultant_from_canonical_top_views(
            descriptors, frame_ids
        )
    with pytest.raises(ValueError, match="counts differ"):
        stream.augment_teacher_payload_v2(
            _teacher_payload(),
            resultant=torch.tensor([1.0, 0.5, 0.0]),
            expected_counts=torch.tensor([4, 1, 0], dtype=torch.uint8),
            loo_audit=_loo_audit(),
        )
