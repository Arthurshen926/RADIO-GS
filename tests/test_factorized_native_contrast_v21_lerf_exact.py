from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import factorized_native_contrast_v21_lerf_exact as formal
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    CANONICAL_NEGATIVE_MODEL,
    FrozenCanonicalNegativeBank,
    INFERENCE_LOGIT_SCALE,
)
from radio_gs.querying.unified_query import cosine_relevancy_torch
from radio_gs.querying.v21_absolute_relevance_adapter import (
    OFFICIAL_TEXT_CANONICALIZATION,
    V21PositiveTextBank,
    calibrated_v21_absolute_relevance,
)
from radio_gs.scripts import (
    materialize_factorized_native_contrast_v21_lerf_exact_relevance as script,
)


def _record(name: str, digit: str = "1") -> dict[str, str]:
    return {"path": f"/tmp/{name}", "sha256": digit * 64}


def _positive(embeddings: torch.Tensor) -> V21PositiveTextBank:
    return V21PositiveTextBank(
        query_ids=tuple(f"q{index}" for index in range(embeddings.shape[0])),
        embeddings=embeddings,
        file_sha256="1" * 64,
        embedding_tensor_sha256=tensor_sha256(embeddings),
        model_id=CANONICAL_NEGATIVE_MODEL,
        text_canonicalization=OFFICIAL_TEXT_CANONICALIZATION,
    )


def _negative(embeddings: torch.Tensor) -> FrozenCanonicalNegativeBank:
    return FrozenCanonicalNegativeBank(
        embeddings=embeddings,
        file_sha256="2" * 64,
        embedding_tensor_sha256=tensor_sha256(embeddings),
        model_id=CANONICAL_NEGATIVE_MODEL,
    )


def test_contract_freezes_exact_calibrated_relevance_without_scene_knobs() -> None:
    contract = formal.query_contract()
    assert contract["health_gate"] == "source_student_envelope_health_v4_formal_PASS"
    assert contract["formula"] == "binary_softmax_positive_vs_max_canonical_negative"
    assert contract["logit_scale"] == INFERENCE_LOGIT_SCALE == 10.0
    assert contract["absolute_equal_logit_boundary"] == 0.5
    assert contract["output"] == "float32_region_absolute_relevance_R_by_Q"
    assert contract["threshold_scan"] is False
    assert contract["scene_specific_parameters"] is False
    assert contract["postprocess"] == "none"
    assert contract["query_smoothing"] is False
    assert contract["scene_minmax_remap"] is False
    assert contract["query_ranking_normalization"] is False


def test_exact_adapter_matches_shared_cosine_relevance_formula() -> None:
    generator = torch.Generator().manual_seed(20260807)
    descriptor = F.normalize(torch.randn(7, 1536, generator=generator), dim=-1)
    positive = F.normalize(torch.randn(3, 1536, generator=generator), dim=-1)
    negative = F.normalize(torch.randn(4, 1536, generator=generator), dim=-1)
    actual = calibrated_v21_absolute_relevance(
        descriptor,
        positive_bank=_positive(positive),
        canonical_negative_bank=_negative(negative),
    )
    expected = cosine_relevancy_torch(
        descriptor,
        positive,
        negative,
        logit_scale=INFERENCE_LOGIT_SCALE,
        assume_normalized=True,
    )
    assert actual.shape == (7, 3)
    assert torch.equal(actual, expected)


def test_exact_adapter_keeps_equal_logit_boundary_at_half() -> None:
    descriptor = F.pad(torch.tensor([[1.0, 0.0]], dtype=torch.float32), (0, 1534))
    positive = descriptor.clone()
    negative = F.pad(
        F.normalize(
            torch.tensor(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                dtype=torch.float32,
            ),
            dim=-1,
        ),
        (0, 1534),
    )
    value = calibrated_v21_absolute_relevance(
        descriptor,
        positive_bank=_positive(positive),
        canonical_negative_bank=_negative(negative),
    )
    assert value.item() == pytest.approx(0.5, abs=1e-7)


def test_health_v4_dispatch_is_fail_closed_when_formal_is_absent(monkeypatch) -> None:
    def missing(name: str):
        assert name == formal.HEALTH_V4_MODULE
        error = ModuleNotFoundError(name)
        error.name = name
        raise error

    monkeypatch.setattr(formal.importlib, "import_module", missing)
    with pytest.raises(formal.HealthV4UnavailableError, match="stay closed"):
        formal.resolve_health_v4_dispatch()


def test_health_v4_dispatch_binds_formal_schema_and_materializer() -> None:
    dispatch = formal.resolve_health_v4_dispatch()
    assert dispatch["schema"] == (
        "radio_gs.factorized_native_contrast_v21_target_health_audit.v4"
    )
    assert dispatch["formal_record"]["path"].endswith(
        "factorized_native_contrast_v21_target_health_v4.py"
    )
    assert dispatch["implementation_record"]["path"].endswith(
        "audit_factorized_native_contrast_v21_target_descriptor_health_v4.py"
    )


def test_prequery_gate_validates_v4_pass_before_any_query_input(monkeypatch) -> None:
    events: list[str] = []
    source_record = _record("source.json", "1")
    descriptor_record = _record("descriptor.pt", "2")
    health_record = _record("health-v4.json", "3")
    checkpoint_record = _record("checkpoint.pt", "4")
    descriptor = {
        "scene_id": "figurines",
        "physical_space_id": "space",
        "input_authority": {
            "source_contrast_v21_result": source_record,
            "source_contrast_v21_checkpoint": checkpoint_record,
        },
    }
    view = {"scene_id": "figurines", "semantic_descriptor": torch.ones(2, 3)}

    def source_gate(record):
        events.append("source")
        assert record == source_record
        return {
            "result": {"checkpoint": checkpoint_record},
            "selected_step": 60,
        }

    def descriptor_gate(record, *, source_result_record, source_gate):
        events.append("descriptor")
        assert record == descriptor_record
        return descriptor, {"verified_record": _record("execution.json")}, view

    def prereg(*, source_result_record):
        events.append("query_prereg")
        return {"frozen": True}

    def prereg_file(record, *, label):
        events.append("health_prereg")
        assert record == formal.HEALTH_V4_PREREGISTRATION
        return Path(record["path"])

    def validate_health(raw, *, require_pass):
        events.append("health_pass")
        assert require_pass is True
        return {"status": "pass"}

    def validate_binding(
        health,
        descriptor_record,
        descriptor_value,
        source_result_record,
        source_checkpoint_record,
        health_preregistration_record,
    ):
        events.append("health_binding")
        assert health == {"status": "pass"}
        assert descriptor_value is descriptor
        assert source_checkpoint_record == checkpoint_record
        assert health_preregistration_record == formal.HEALTH_V4_PREREGISTRATION
        return health

    fake_health = SimpleNamespace(
        validate_health_audit=validate_health,
        validate_query_gate_binding=validate_binding,
    )
    monkeypatch.setattr(formal.target, "validate_source_contrast_v21_result", source_gate)
    monkeypatch.setattr(formal, "_load_target_descriptor", descriptor_gate)
    monkeypatch.setattr(formal, "_validate_query_preregistration", prereg)
    monkeypatch.setattr(formal, "validate_file_record", prereg_file)
    monkeypatch.setattr(
        formal,
        "resolve_health_v4_dispatch",
        lambda: {
            "module": fake_health,
            "schema": "health.v4",
            "formal_record": _record("health-formal.py"),
            "implementation_record": _record("health-script.py"),
        },
    )

    def load_health(path, *, expected_sha256, label):
        events.append("health_audit")
        assert path == health_record["path"]
        return {"schema": "health.v4"}, expected_sha256, Path(path)

    monkeypatch.setattr(formal, "load_json_object", load_health)
    gate = formal.validate_prequery_gate(
        source_result_record=source_record,
        target_descriptor_record=descriptor_record,
        health_v4_audit_record=health_record,
    )
    assert events == [
        "source",
        "descriptor",
        "query_prereg",
        "health_prereg",
        "health_audit",
        "health_pass",
        "health_binding",
    ]
    assert gate["access_audit"] == formal.prequery_access_audit()
    assert all(
        gate["access_audit"][name] is False
        for name in (
            "exact_query_manifest_opened",
            "positive_text_cache_opened",
            "all_query_text_cache_opened",
            "canonical_negative_bank_opened",
        )
    )


def test_authority_builder_does_not_open_query_files_when_v4_gate_fails(
    tmp_path: Path, monkeypatch
) -> None:
    args = argparse.Namespace(
        source_result="/tmp/source.json",
        expected_source_result_sha256="1" * 64,
        target_descriptor="/tmp/descriptor.pt",
        expected_target_descriptor_sha256="2" * 64,
        health_v4_audit="/tmp/health-v4.json",
        expected_health_v4_audit_sha256="3" * 64,
        exact_query_manifest="/tmp/manifest.json",
        expected_exact_query_manifest_sha256="4" * 64,
        positive_text_cache="/tmp/positive.pt",
        expected_positive_text_cache_sha256="5" * 64,
        query_relevance_output=str(tmp_path / "relevance.pt"),
        output_authority=str(tmp_path / "authority.json"),
    )
    monkeypatch.setattr(
        formal,
        "validate_prequery_gate",
        lambda **_: (_ for _ in ()).throw(ValueError("health-v4 reject")),
    )
    monkeypatch.setattr(
        script,
        "_validate_exact_text_protocol",
        lambda **_: pytest.fail("query/text input opened before health-v4 PASS"),
    )
    with pytest.raises(ValueError, match="health-v4 reject"):
        script.build_authority(args)
    assert not Path(args.output_authority).exists()
    assert not Path(args.query_relevance_output).exists()


def _payload() -> dict[str, object]:
    relevance = torch.tensor([[0.25, 0.75], [0.5, 0.9]], dtype=torch.float32)
    payload: dict[str, object] = {
        "schema": formal.QUERY_RELEVANCE_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "contract": formal.query_contract(),
        "contract_sha256": formal.QUERY_CONTRACT_SHA256,
        "scene_id": "figurines",
        "physical_space_id": "space",
        "producer": _record("producer.py"),
        "query_execution_authority": _record("authority.json", "2"),
        "input_authority": {
            "source_result": _record("source.json", "1"),
            "target_descriptor": _record("descriptor.pt", "2"),
            "health_v4_audit": _record("health-v4.json", "3"),
            "health_v4_preregistration": formal.HEALTH_V4_PREREGISTRATION,
            "query_preregistration": formal.QUERY_PREREGISTRATION,
            "exact_query_manifest": _record("manifest.json", "4"),
            "positive_text_cache": _record("positive.pt", "5"),
            "all_query_text_cache": formal.FROZEN_ALL_QUERY_CACHE,
            "canonical_negative_bank": formal.FROZEN_CANONICAL_NEGATIVE_BANK,
        },
        "region_row_ids": ["r0", "r1"],
        "canonical_region_indices": torch.tensor([3, 9], dtype=torch.int64),
        "region_fingerprints": ["a" * 64, "b" * 64],
        "query_ids": ["red chair", "waldo"],
        "region_absolute_relevance": relevance,
        "access_audit": formal.query_access_audit(),
    }
    payload["channel_sha256"] = formal.query_channel_sha256(payload)
    return payload


def test_relevance_payload_is_immutable_region_by_query_authority() -> None:
    payload = _payload()
    validated = formal.validate_query_relevance(payload)
    assert validated["region_absolute_relevance"].shape == (2, 2)
    assert validated["region_absolute_relevance"].dtype == torch.float32

    tampered = dict(payload)
    tampered["region_absolute_relevance"] = payload[
        "region_absolute_relevance"
    ].clone()
    tampered["region_absolute_relevance"][0, 0] += 0.01
    with pytest.raises(ValueError, match="tensor differs"):
        formal.validate_query_relevance(tampered)


def test_relevance_payload_rejects_scene_parameters_and_wrong_axis() -> None:
    payload = _payload()
    payload["contract"] = {**formal.query_contract(), "scene_threshold": 0.7}
    with pytest.raises(ValueError, match="header differs"):
        formal.validate_query_relevance(payload)

    payload = _payload()
    payload["region_absolute_relevance"] = torch.ones(2, 3, dtype=torch.float32)
    payload["channel_sha256"] = formal.query_channel_sha256(payload)
    with pytest.raises(ValueError, match="tensor differs"):
        formal.validate_query_relevance(payload)
