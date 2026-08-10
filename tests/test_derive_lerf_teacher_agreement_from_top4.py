from __future__ import annotations

import copy

import pytest
import torch

from radio_gs.scripts import derive_lerf_teacher_agreement_from_top4 as adapter
from radio_gs.scripts import materialize_lerf_o1_o2_streaming as core
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
)


def _payload(tmp_path) -> dict[str, object]:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable-test-source")
    record = file_record(source)
    rows = torch.tensor([1, 5], dtype=torch.int64)
    mean = torch.zeros(2, 1536, dtype=torch.float16)
    mean[0, 0] = 1
    count = torch.tensor([4, 0], dtype=torch.uint8)
    valid = count > 0
    resultant = torch.tensor([0.75, 0.0], dtype=torch.float32)
    return {
        "schema": adapter.SCHEMA,
        "schema_version": adapter.SCHEMA_VERSION,
        "scene_id": "figurines",
        "global_rows": rows,
        "teacher_mean": mean,
        "teacher_valid": valid,
        "retained_view_count": count,
        "teacher_view_directional_resultant": resultant,
        "producer": record,
        "top4_source": record,
        "base_descriptor": record,
        "contract": adapter.contract(),
        "contract_sha256": adapter.CONTRACT_SHA256,
        "teacher_mean_sha256": core.tensor_sha256_typed(mean),
        "teacher_view_directional_resultant_sha256": core.tensor_sha256_typed(
            resultant
        ),
        "access_audit": {
            "canonical_top4_source_opened": True,
            "accepted_v2_base_descriptor_opened": True,
            "query_embeddings_or_text_opened": False,
            "target_images_labels_masks_metrics_opened": False,
            "target_metric_executed": False,
        },
        "metric_execution_authorized": False,
    }


def test_adapter_contract_is_query_free_target_closed_and_hash_bound() -> None:
    contract = adapter.contract()
    assert contract["required_retention_order"] == (
        "marginal_mass_descending_then_frame_id_ascending"
    )
    assert contract["query_independent"] is True
    assert contract["target_data_or_metric_access"] is False
    assert adapter.CONTRACT_SHA256 == canonical_json_sha256(contract)


def test_derived_payload_validation_is_strict_and_metric_closed(tmp_path) -> None:
    payload = _payload(tmp_path)
    adapter.validate_payload(payload)

    mutated = copy.deepcopy(payload)
    mutated["metric_execution_authorized"] = True
    with pytest.raises(ValueError, match="payload differs"):
        adapter.validate_payload(mutated)

    mutated = copy.deepcopy(payload)
    mutated["teacher_view_directional_resultant"][1] = 0.5
    with pytest.raises(ValueError, match="payload differs"):
        adapter.validate_payload(mutated)

