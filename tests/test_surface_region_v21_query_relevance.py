from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces import surface_region_v21_query_relevance as formal
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    CANONICAL_NEGATIVE_MODEL,
    INFERENCE_LOGIT_SCALE,
)
from radio_gs.querying.unified_query import cosine_relevancy_torch
from radio_gs.querying.v21_absolute_relevance_adapter import (
    V21PositiveTextBank,
    calibrated_v21_absolute_relevance,
    load_v21_positive_text_bank,
)
from radio_gs.scripts.eval_lerf_grounding import (
    _SIGLIP2_TEXT_CANONICALIZATION,
)
from radio_gs.utils.immutable_artifacts import file_record, sha256_file


def _positive_bank(embeddings: torch.Tensor) -> V21PositiveTextBank:
    return V21PositiveTextBank(
        query_ids=tuple(f"q{index}" for index in range(embeddings.shape[0])),
        embeddings=embeddings,
        file_sha256="1" * 64,
        embedding_tensor_sha256=tensor_sha256(embeddings),
        model_id=CANONICAL_NEGATIVE_MODEL,
        text_canonicalization=_SIGLIP2_TEXT_CANONICALIZATION,
    )


def test_adapter_is_exact_training_inference_formula() -> None:
    from radio_gs.losses.source_global_response_listwise_loss_v21 import (
        FrozenCanonicalNegativeBank,
    )

    generator = torch.Generator().manual_seed(17)
    descriptor = F.normalize(torch.randn(7, 1536, generator=generator), dim=-1)
    positive = F.normalize(torch.randn(3, 1536, generator=generator), dim=-1)
    negative = F.normalize(torch.randn(4, 1536, generator=generator), dim=-1)
    negative_bank = FrozenCanonicalNegativeBank(
        embeddings=negative,
        file_sha256="3" * 64,
        embedding_tensor_sha256=tensor_sha256(negative),
        model_id=CANONICAL_NEGATIVE_MODEL,
    )

    actual = calibrated_v21_absolute_relevance(
        descriptor,
        positive_bank=_positive_bank(positive),
        canonical_negative_bank=negative_bank,
    )
    expected = cosine_relevancy_torch(
        descriptor,
        positive,
        negative,
        logit_scale=INFERENCE_LOGIT_SCALE,
        assume_normalized=True,
    )
    assert torch.equal(actual, expected)


def test_adapter_preserves_equal_logit_half_boundary_without_remap() -> None:
    from radio_gs.losses.source_global_response_listwise_loss_v21 import (
        FrozenCanonicalNegativeBank,
    )

    descriptor_2d = F.normalize(
        torch.tensor([[1.0, 2**0.5 - 1.0], [1.0, -1.0]], dtype=torch.float32),
        dim=-1,
    )
    positive_2d = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    negative_2d = torch.tensor(
        [[1.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
        dtype=torch.float32,
    )
    descriptor = F.pad(descriptor_2d, (0, 1534))
    positive = F.pad(positive_2d, (0, 1534))
    negative = F.pad(F.normalize(negative_2d, dim=-1), (0, 1534))
    negative_bank = FrozenCanonicalNegativeBank(
        embeddings=negative,
        file_sha256="3" * 64,
        embedding_tensor_sha256=tensor_sha256(negative),
        model_id=CANONICAL_NEGATIVE_MODEL,
    )
    relevance = calibrated_v21_absolute_relevance(
        descriptor,
        positive_bank=_positive_bank(positive),
        canonical_negative_bank=negative_bank,
    )
    assert relevance[0, 0].item() == pytest.approx(0.5, abs=1e-7)
    assert relevance[1, 0].item() > 0.5


def test_positive_cache_loader_requires_official_model_and_canonicalization(
    tmp_path: Path,
) -> None:
    payload = {
        "queries": ["Red Chair", "red chair"],
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": CANONICAL_NEGATIVE_MODEL,
        "text_canonicalization": _SIGLIP2_TEXT_CANONICALIZATION,
        "embeddings": F.normalize(torch.randn(2, 1536), dim=-1),
    }
    path = tmp_path / "positive.pt"
    torch.save(payload, path)
    bank = load_v21_positive_text_bank(path, expected_file_sha256=sha256_file(path))
    assert bank.query_ids == ("Red Chair", "red chair")

    legacy = dict(payload)
    legacy.pop("text_canonicalization")
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(ValueError, match="fields"):
        load_v21_positive_text_bank(
            legacy_path, expected_file_sha256=sha256_file(legacy_path)
        )

    payload["text_canonicalization"] = "unknown"
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="canonicalization"):
        load_v21_positive_text_bank(bad, expected_file_sha256=sha256_file(bad))


def _execution(tmp_path: Path) -> Path:
    descriptor = tmp_path / "descriptor.pt"
    positive = tmp_path / "positive.pt"
    negative = tmp_path / "negative.pt"
    for path in (descriptor, positive, negative):
        path.write_bytes(path.stem.encode("utf-8"))
    authority = {
        "schema": formal.QUERY_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_v21_source_promotion_for_calibrated_query_relevance",
        "scene_id": "figurines",
        "physical_space_id": "lerf:figurines:geometry-checkpoint-sha256:" + "9" * 64,
        "source_pilot_result": {"path": "/source/result.json", "sha256": "8" * 64},
        "implementation": file_record(formal.IMPLEMENTATION_PATH),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in formal.IMPLEMENTATION_DEPENDENCIES.items()
        },
        "preregistration": file_record(formal.PREREGISTRATION_PATH),
        "target_descriptor": file_record(descriptor),
        "positive_text_cache": file_record(positive),
        "canonical_negative_bank": file_record(negative),
        "query_relevance_output": str((tmp_path / "relevance.pt").resolve()),
        "query_execution_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": formal.query_relevance_access_audit(),
    }
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(authority), encoding="utf-8")
    return path


def test_source_rejection_occurs_before_descriptor_or_query_cache_open(
    monkeypatch, tmp_path: Path
) -> None:
    path = _execution(tmp_path)
    opened = {"files": 0}

    def reject(*args, **kwargs):
        raise ValueError("source promotion rejected")

    def file_open(*args, **kwargs):
        opened["files"] += 1
        raise AssertionError("target/query record opened before source promotion")

    monkeypatch.setattr(formal, "validate_source_pilot_chain", reject)
    monkeypatch.setattr(formal, "validate_file_record", file_open)
    with pytest.raises(ValueError, match="source promotion rejected"):
        formal.validate_query_execution_authority(
            path, expected_sha256=sha256_file(path)
        )
    assert opened["files"] == 0


def test_query_gate_rejects_negative_bank_not_used_by_promoted_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _execution(tmp_path)
    source_execution = tmp_path / "source_execution.json"
    source_execution.write_text(
        json.dumps(
            {
                "canonical_negative_bank": {
                    "path": "/source/training_negative.pt",
                    "sha256": "7" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        formal,
        "validate_source_pilot_chain",
        lambda *args, **kwargs: {
            "source_promotion_authorized": True,
            "execution_authority": file_record(source_execution),
        },
    )
    monkeypatch.setattr(
        formal,
        "validate_file_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("target/query file opened before negative-bank binding")
        ),
    )
    with pytest.raises(ValueError, match="differs from source training"):
        formal.validate_query_execution_authority(
            path, expected_sha256=sha256_file(path)
        )
