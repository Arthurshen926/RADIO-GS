from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from radio_gs.losses import source_global_response_listwise_loss as v2
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    EXPECTED_COUNTS,
    FrozenTypedTextRelationAuthority,
    SCHEMA,
    load_frozen_typed_text_relation_authority,
    source_typed_text_relation_loss_v21,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, sha256_file


def _component_records() -> dict[str, dict[str, object]]:
    rows = {
        "primary": 806,
        "synonym_relation": 657,
        "lexical_sibling_relation": 167,
        "counterfactual_attributes": 3224,
        "high_precision_part_of": 30,
    }
    return {
        name: {
            "path": f"/{name}.pt",
            "sha256": str(index + 1) * 64,
            "embedding_tensor_sha256": chr(ord("a") + index) * 64,
            "query_rows": count,
        }
        for index, (name, count) in enumerate(rows.items())
    }


def test_typed_relation_authority_loads_exact_bound_indices(tmp_path: Path) -> None:
    identity = {
        "schema": SCHEMA,
        "schema_version": 1,
        "split": "fit",
        "source": {"path": "/source.json", "sha256": "f" * 64},
        "components": _component_records(),
        "counts": EXPECTED_COUNTS,
        "index_semantics": {
            "synonym_left": "primary",
            "synonym_right": "synonym_relation",
            "sibling_left": "primary",
            "sibling_right": "primary",
        },
        "source_access": {
            "benchmark_vocabulary_opened": False,
            "target_metrics_computed": False,
        },
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "synonym_record_ids": [f"synonym-{i}" for i in range(657)],
        "synonym_left_primary_indices": torch.arange(657, dtype=torch.int64),
        "synonym_right_component_indices": torch.arange(657, dtype=torch.int64),
        "sibling_record_ids": [f"sibling-{i}" for i in range(167)],
        "sibling_left_primary_indices": torch.arange(167, dtype=torch.int64),
        "sibling_right_primary_indices": torch.arange(1, 168, dtype=torch.int64),
    }
    path = tmp_path / "relations.pt"
    torch.save(payload, path)
    authority = load_frozen_typed_text_relation_authority(
        path, expected_file_sha256=sha256_file(path)
    )
    assert authority.content_authority_sha256 == canonical_json_sha256(identity)
    assert authority.synonym_left_primary_indices.shape == (657,)
    assert authority.sibling_right_primary_indices.shape == (167,)


def test_typed_relation_loss_has_both_sibling_directions_and_gradient() -> None:
    teacher_views = F.normalize(
        torch.tensor(
            [
                [[1.0, 0.1, 0.0, 0.0], [0.9, 0.2, 0.0, 0.0]],
                [[0.1, 1.0, 0.0, 0.0], [0.2, 0.9, 0.0, 0.0]],
                [[0.8, 0.3, 0.0, 0.0], [0.7, 0.4, 0.0, 0.0]],
                [[0.3, 0.8, 0.0, 0.0], [0.4, 0.7, 0.0, 0.0]],
            ]
        ),
        dim=-1,
    )
    teacher_mask = torch.ones(4, 2, dtype=torch.bool)
    primary = F.normalize(
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    synonym = F.normalize(
        torch.tensor(
            [[0.98, 0.05, 0.0, 0.0], [0.05, 0.98, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    negative = torch.zeros(4, 4)
    negative[:, -1] = 1.0
    teacher_negative = v2._teacher_response_chunk(
        teacher_views, teacher_mask, negative, temperature=0.1
    )
    authority = FrozenTypedTextRelationAuthority(
        file_sha256="a" * 64,
        content_authority_sha256="b" * 64,
        source_sha256="c" * 64,
        components={},
        synonym_left_primary_indices=torch.tensor([0, 1]),
        synonym_right_component_indices=torch.tensor([0, 1]),
        sibling_left_primary_indices=torch.tensor([0]),
        sibling_right_primary_indices=torch.tensor([1]),
    )
    student = torch.flip(teacher_views.mean(dim=1), dims=(0,)).requires_grad_(True)
    loss, metrics = source_typed_text_relation_loss_v21(
        F.normalize(student, dim=-1),
        teacher_views,
        teacher_mask,
        primary,
        synonym,
        negative,
        teacher_negative,
        authority,
        response_temperature=0.1,
        logit_scale=10.0,
        smooth_l1_beta=0.05,
        continuous_gap_sigma=0.05,
        stability_std_scale=0.10,
        pair_chunk_rows=1,
    )
    assert bool(torch.isfinite(loss)) and float(loss) > 0
    assert metrics["sibling_left_dominant_units"] > 0
    assert metrics["sibling_right_dominant_units"] > 0
    loss.backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0


def test_typed_relation_loader_rejects_file_sha_change(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save({}, path)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_frozen_typed_text_relation_authority(path, expected_file_sha256="0" * 64)
