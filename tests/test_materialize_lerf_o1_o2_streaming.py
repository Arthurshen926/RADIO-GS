from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from radio_gs.scripts import materialize_lerf_o1_o2_streaming as stream


def test_contract_is_o1_o2_only_and_metric_closed() -> None:
    contract = stream.method_contract()
    assert contract["O1"]["maximum_angle_radians"] == 0.15
    assert contract["O2"]["repeated_scale_slots"] == 3
    assert contract["top4_descriptors_durable"] is False
    assert contract["O3_materialized"] is False
    assert contract["O4_materialized"] is False
    assert contract["metric_execution_authorized"] is False
    assert stream.METHOD_CONTRACT_SHA256 == stream.canonical_json_sha256(contract)


def test_access_audit_never_opens_gt_mask_or_metric() -> None:
    audit = stream.access_audit()
    assert audit["source_feature_bundle_opened"] is True
    assert audit["source_responsibility_opened"] is True
    for key in (
        "target_images_opened",
        "target_ground_truth_opened",
        "target_masks_opened",
        "target_metrics_opened",
        "target_quality_readout_executed",
    ):
        assert audit[key] is False


def _responsibility_payload() -> dict[str, object]:
    frames = list(range(100, 220))
    formula = {"query_independent": True, "feature_independent": True}
    views = [
        {
            "frame_index": frame,
            "num_hits": 1,
            "relative_path": f"views/view_{index:03d}.pt",
            "sha256": f"{index + 1:064x}",
            "view_index": index,
        }
        for index, frame in enumerate(frames)
    ]
    return {
        "formula_contract": formula,
        "formula_sha256": stream.canonical_json_sha256(formula),
        "frame_indices": frames,
        "metadata": {
            "assignment_mode": "exact_front_to_back_sparse_marginal",
            "registration_weight_mode": "exact_front_to_back_marginal_responsibility",
            "query_independent": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "xyz_sha256": "a" * 64,
            "excluded_frame_ids": [2, 25, 43, 90],
        },
        "num_gaussians": 7,
        "num_pixels": 10,
        "schema": "radio_gs.sparse_exact_marginal_responsibility_authority.v1",
        "schema_version": 1,
        "total_hits": 120,
        "views": views,
    }


def test_responsibility_uses_scene_authority_exclusions_not_figurines_constants() -> None:
    payload = _responsibility_payload()
    validated = stream._validate_responsibility_payload(
        payload,
        descriptor_xyz_sha256="a" * 64,
        feature_frame_ids=set(range(100, 220)),
    )
    assert validated["metadata"]["excluded_frame_ids"] == [2, 25, 43, 90]


@pytest.mark.parametrize("mutation", ["query", "feature", "excluded", "xyz"])
def test_responsibility_lineage_mismatch_rejects(mutation: str) -> None:
    payload = _responsibility_payload()
    features = set(range(100, 220))
    xyz = "a" * 64
    if mutation == "query":
        payload["metadata"]["query_independent"] = False
    elif mutation == "feature":
        features.remove(100)
    elif mutation == "excluded":
        payload["metadata"]["excluded_frame_ids"] = [100]
    elif mutation == "xyz":
        xyz = "b" * 64
    with pytest.raises(ValueError, match="exact responsibility"):
        stream._validate_responsibility_payload(
            payload, descriptor_xyz_sha256=xyz, feature_frame_ids=features
        )


def test_projection_batch_preflight_falls_back_only_on_cuda_oom() -> None:
    calls: list[int] = []

    def runner(batch: int) -> int:
        calls.append(batch)
        if batch == 128:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return 1234

    selected, peak = stream.select_projection_batch((128, 64), runner)
    assert calls == [128, 64]
    assert selected == 64
    assert peak == 1234


def test_projection_batch_preflight_does_not_hide_non_oom() -> None:
    with pytest.raises(ValueError, match="not memory"):
        stream.select_projection_batch((128, 64), lambda _: (_ for _ in ()).throw(ValueError("not memory")))


def test_o1_is_bounded_geodesic_and_o2_is_teacher_mean() -> None:
    base = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        ]
    )
    teacher = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    o1, o2 = stream._score_descriptors(
        base=base, teacher_mean=teacher, teacher_valid=torch.tensor([True, False])
    )
    assert torch.allclose(o2, F.normalize(teacher, dim=-1))
    assert torch.allclose(o1[1], base[1])
    angle = torch.acos((o1[0, 0] * base[0, 0]).sum().clamp(-1, 1))
    assert float(angle) == pytest.approx(0.15, abs=2e-6)


def test_text_bank_rejects_query_order_and_nonfinite() -> None:
    bank = {"queries": ["a", "b"], "embeddings": torch.eye(2, 1536)}
    normalized = stream._validate_text_bank(bank, expected_queries=["a", "b"])
    assert torch.allclose(torch.linalg.vector_norm(normalized, dim=-1), torch.ones(2))
    with pytest.raises(ValueError, match="axis"):
        stream._validate_text_bank(bank, expected_queries=["b", "a"])
    broken = deepcopy(bank)
    broken["embeddings"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="embeddings"):
        stream._validate_text_bank(broken, expected_queries=["a", "b"])


@pytest.mark.parametrize("with_optional", [False, True])
def test_base_descriptor_accepts_only_the_two_frozen_v5_optional_variants(
    monkeypatch: pytest.MonkeyPatch, with_optional: bool
) -> None:
    valid = torch.tensor([True, False, True])
    payload = {
        "xyz": torch.zeros(3, 3),
        "features": torch.zeros(2, 1536, dtype=torch.float16),
        "summary_features": torch.zeros(2, 1536, dtype=torch.float16),
        "global_rows": torch.tensor([0, 2]),
        "features_by_scale": torch.zeros(2, 3, 1536, dtype=torch.float16),
        "valid": valid,
        "metadata": {
            "schema_version": 5,
            "query_set_invariant": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "region_radii_m": [0.25, 0.45, 0.7],
        },
    }
    if with_optional:
        payload["primary_valid"] = valid.clone()
        payload["semantic_confidence"] = torch.ones(3)
    monkeypatch.setattr(
        stream,
        "load_torch_mapping",
        lambda *args, **kwargs: (payload, "a" * 64, args[0]),
    )
    validated, rows = stream._validate_base_descriptor_general(
        stream.Path("/tmp/base.pt"), "a" * 64
    )
    assert torch.equal(rows, torch.tensor([0, 2]))
    assert set(validated) == set(payload)


def test_base_descriptor_rejects_unregistered_extra_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "xyz": torch.zeros(1, 3),
        "features": torch.zeros(1, 1536),
        "summary_features": torch.zeros(1, 1536),
        "global_rows": torch.tensor([0]),
        "features_by_scale": torch.zeros(1, 3, 1536),
        "valid": torch.ones(1, dtype=torch.bool),
        "metadata": {},
        "unregistered": True,
    }
    monkeypatch.setattr(
        stream,
        "load_torch_mapping",
        lambda *args, **kwargs: (payload, "a" * 64, args[0]),
    )
    with pytest.raises(ValueError, match="fields differ"):
        stream._validate_base_descriptor_general(
            stream.Path("/tmp/base.pt"), "a" * 64
        )


def test_stream_hasher_matches_frozen_typed_tensor_hash() -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    digest = stream._typed_stream_hasher(tuple(value.shape), value.dtype)
    for row in value:
        digest.update(row.contiguous().numpy().tobytes(order="C"))
    assert digest.hexdigest() == stream.tensor_sha256_typed(value)


def test_raw_cache_binds_oracle_without_distributional_outputs() -> None:
    template = {
        "version": 4,
        "contract": stream.RAW_CACHE_CONTRACT,
        "query_scores": torch.zeros(2, 3, 1),
        "query_ids": ["q"],
        "authority": {
            "contract": "old",
            "descriptor_axis": {},
            "source_artifacts": {},
            "calibration_constraints": {"benchmark_metrics_opened": False},
        },
    }
    record = {"path": "/tmp/mean.pt", "sha256": "a" * 64}
    payload = stream._raw_cache(
        template,
        torch.ones(2, 3, 1),
        oracle="O1",
        representation=record,
        text_cache={"path": "/tmp/text.pt", "sha256": "b" * 64},
        descriptor_sha256="c" * 64,
    )
    assert payload["authority"]["descriptor_axis"]["oracle"] == "O1"
    assert payload["authority"]["source_artifacts"]["descriptor_cache"] == record
    assert not any("o3" in key.lower() or "o4" in key.lower() for key in payload)
