from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from radio_gs.interfaces import rank256_o0_full_lift_premetric as formal
from radio_gs.scripts import materialize_rank256_o0_full_lift_premetric_cache as script


def _audit_inputs(*, fallback_delta: float = 0.0, candidate_value: float = 0.7):
    primitive_count = 4
    query_count = 21
    valid = torch.ones(primitive_count, dtype=torch.bool)
    candidate = torch.full((primitive_count, query_count), candidate_value)
    o0 = torch.full_like(candidate, 0.7)
    positive = torch.zeros((primitive_count, 3, query_count))
    negative = torch.zeros((primitive_count, 3, 4))
    candidate_positive = positive.clone()
    candidate_positive[0, 0, 0] += fallback_delta
    return {
        "candidate_scores": candidate,
        "o0_scores": o0,
        "valid": valid,
        "query_ids": [f"query-{index}" for index in range(query_count)],
        "candidate_positive_raw_sparse": candidate_positive,
        "candidate_negative_raw_sparse": negative.clone(),
        "o0_positive_raw": positive,
        "o0_negative_raw": negative,
        "primitive_global_rows": torch.arange(primitive_count),
        "fallback_mask": torch.ones((primitive_count, 3), dtype=torch.bool),
        "axes_exact": True,
    }


def test_exact_fp32_cosine_matches_normalized_reference() -> None:
    torch.manual_seed(4)
    descriptor = torch.randn(5, 3, 7, dtype=torch.float16)
    positive = F.normalize(torch.randn(21, 7), dim=-1)
    negative = F.normalize(torch.randn(4, 7), dim=-1)
    result = formal.exact_fp32_cosine_scores(
        descriptor,
        positive_text=positive,
        negative_text=negative,
        chunk_rows=2,
    )
    visual = F.normalize(descriptor.float(), dim=-1)
    assert result.positive.dtype == torch.float32
    assert result.negative.dtype == torch.float32
    assert torch.equal(result.positive, visual @ F.normalize(positive, dim=-1).T)
    assert torch.equal(result.negative, visual @ F.normalize(negative, dim=-1).T)


def test_sparse_scatter_preserves_global_row_axis() -> None:
    sparse = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    result = formal.scatter_sparse_scores(
        sparse, global_rows=torch.tensor([1, 4]), total_rows=6
    )
    assert result.shape == (6, 3, 4)
    assert torch.equal(result[1], sparse[0])
    assert torch.equal(result[4], sparse[1])
    assert int(result[[0, 2, 3, 5]].count_nonzero()) == 0


def test_fixed_premetric_gate_passes_exact_preservation() -> None:
    result = formal.build_premetric_audit(**_audit_inputs())
    assert result["status"] == "PASS"
    assert result["premetric_passed"] is True
    assert all(result["checks"].values())
    assert result["aggregate"]["supported_queries"] == 21
    assert result["aggregate"]["micro_o0_seed_precision"] == 1.0
    assert result["aggregate"]["micro_o0_seed_recall"] == 1.0


def test_fixed_premetric_gate_uses_strict_boundary() -> None:
    result = formal.build_premetric_audit(
        **_audit_inputs(candidate_value=formal.SCORE_BOUNDARY)
    )
    assert result["status"] == "REJECT"
    assert result["checks"]["all_21_queries_supported"] is False
    assert result["aggregate"]["candidate_positive_primitive_query_cells"] == 0


def test_fixed_premetric_gate_rejects_fallback_raw_drift() -> None:
    result = formal.build_premetric_audit(**_audit_inputs(fallback_delta=3e-6))
    assert result["status"] == "REJECT"
    assert result["checks"]["fallback_raw_difference_at_most_2e_6"] is False


def test_external_cache_is_evaluator_compatible_and_hash_bound() -> None:
    scores = torch.full((6, 21), 0.25)
    payload = formal.build_external_query_score_cache(
        query_scores=scores,
        valid=torch.tensor([True, True, False, True, False, True]),
        xyz=torch.randn(6, 3),
        query_ids=[f"query-{index}" for index in range(21)],
        scene_id="figurines",
        physical_space_id="lerf:figurines:geometry-checkpoint-sha256:" + "a" * 64,
        input_authority={"lift": {"path": "/tmp/lift.pt", "sha256": "b" * 64}},
    )
    validated = formal.validate_external_query_score_cache(payload)
    assert validated.keys() == payload.keys()
    assert torch.equal(validated["query_scores"], payload["query_scores"])
    broken = dict(payload)
    broken["query_scores"] = scores.clone()
    broken["query_scores"][0, 0] = 0.5
    with pytest.raises(ValueError, match="hashes"):
        formal.validate_external_query_score_cache(broken)


class _Config:
    def to_dict(self):
        return dict(script.EXPECTED_CONFIGURATION)


def _record(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _parent(tmp_path: Path) -> dict:
    authority = tmp_path / "parent.json"
    authority.write_text("{}\n")
    renderer = tmp_path / "renderer.pt"
    renderer.write_bytes(b"renderer")
    return {
        "scene_id": "figurines",
        "source_variant": "v21b",
        "configuration_object": _Config(),
        "scope": {
            "prefix_order": "o0_global_rows_ascending_storage_order",
            "valid_row_prefix_limit": 5,
        },
        "input_authority": {"renderer_geometry_checkpoint": _record(renderer)},
        "verified_record": _record(authority),
    }


def _builder_args(tmp_path: Path, parent: dict) -> argparse.Namespace:
    files = {}
    for name in ("lift.pt", "query.json", "positive.pt", "negative.pt"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = _record(path)
    return argparse.Namespace(
        full_lift_execution_authority=parent["verified_record"]["path"],
        expected_full_lift_execution_authority_sha256=parent["verified_record"]["sha256"],
        full_lift_descriptor=files["lift.pt"]["path"],
        expected_full_lift_descriptor_sha256=files["lift.pt"]["sha256"],
        exact_query_authority=files["query.json"]["path"],
        expected_exact_query_authority_sha256=files["query.json"]["sha256"],
        positive_o0_scores=files["positive.pt"]["path"],
        expected_positive_o0_scores_sha256=files["positive.pt"]["sha256"],
        negative_o0_scores=files["negative.pt"]["path"],
        expected_negative_o0_scores_sha256=files["negative.pt"]["sha256"],
        output_cache=str((tmp_path / "cache.pt").resolve()),
        output_audit=str((tmp_path / "audit.json").resolve()),
        output_authority=str((tmp_path / "authority.json").resolve()),
    )


def test_authority_builder_validates_query_free_parent_before_query(
    tmp_path: Path, monkeypatch
) -> None:
    events = []

    def reject(*_args, **_kwargs):
        events.append("parent")
        raise ValueError("parent rejected")

    monkeypatch.setattr(script.lift, "validate_authority", reject)
    args = argparse.Namespace(
        full_lift_execution_authority=str((tmp_path / "missing-parent.json").resolve()),
        expected_full_lift_execution_authority_sha256="0" * 64,
        full_lift_descriptor=str((tmp_path / "missing-lift.pt").resolve()),
        expected_full_lift_descriptor_sha256="0" * 64,
        exact_query_authority=str((tmp_path / "missing-query.json").resolve()),
        expected_exact_query_authority_sha256="0" * 64,
        positive_o0_scores=str((tmp_path / "missing-positive.pt").resolve()),
        expected_positive_o0_scores_sha256="0" * 64,
        negative_o0_scores=str((tmp_path / "missing-negative.pt").resolve()),
        expected_negative_o0_scores_sha256="0" * 64,
        output_cache=str((tmp_path / "cache.pt").resolve()),
        output_audit=str((tmp_path / "audit.json").resolve()),
        output_authority=str((tmp_path / "authority.json").resolve()),
    )
    with pytest.raises(ValueError, match="parent rejected"):
        script.build_authority(args)
    assert events == ["parent"]
    assert not Path(args.output_authority).exists()


def test_authority_builder_freezes_gate_and_forbids_metric(
    tmp_path: Path, monkeypatch
) -> None:
    parent = _parent(tmp_path)
    args = _builder_args(tmp_path, parent)
    monkeypatch.setattr(script.lift, "validate_authority", lambda *_a, **_k: parent)
    query_record = {
        "path": args.exact_query_authority,
        "sha256": args.expected_exact_query_authority_sha256,
    }
    monkeypatch.setattr(
        script,
        "_validate_exact_query_protocol_authority",
        lambda *_a, **_k: {"record": query_record},
    )
    result = script.build_authority(args)
    assert result["status"] == "rank256_O0_full_lift_premetric_authority_built"
    raw = __import__("json").loads(Path(args.output_authority).read_text())
    assert raw["premetric_contract"] == formal.premetric_contract()
    assert raw["frozen_lift_configuration"] == script.EXPECTED_CONFIGURATION
    assert raw["full_valid_domain_required"] is True
    assert raw["metric_execution_authorized"] is False
    assert "build-metric-authority" not in script.build_parser()._subparsers._group_actions[0].choices
