import json
from pathlib import Path

import pytest
import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.interfaces.capability_cache import (
    _load_memory_mapped_capability_payload,
    load_canonical_capability_bank,
    load_canonical_support_graph,
)


def _signature(name: str, dim: int) -> dict:
    return FeatureSpaceSignature(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256="radio-hash",
        raw_feature_dim=1280,
        adaptor_name=name,
        adaptor_sha256="radio-hash",
        adaptor_output_dim=dim,
        token_type="primitive",
        field_checkpoint_sha256="field-hash",
    ).to_dict()


def _write_bank(path: Path) -> None:
    torch.save(
        {
            "schema_version": 1,
            "xyz": torch.zeros(3, 3),
            "valid": torch.tensor([True, False, True]),
            "appearance_dino_v3": torch.zeros(3, 4),
            "boundary_sam3": torch.zeros(3, 2),
            "metadata": {
                "source": "canonical_radio_field_official_frozen_capability_views",
                "field_checkpoint_sha256": "field-hash",
                "radio_checkpoint_sha256": "radio-hash",
                "custom_adaptor_head": False,
                "query_independent": True,
                "capability_signatures": {
                    "appearance": _signature("dino", 4),
                    "boundary": _signature("sam3", 2),
                },
            },
        },
        path,
    )


def test_capability_cache_and_graph_are_hash_and_row_locked(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.pt"
    _write_bank(bank_path)
    bank = load_canonical_capability_bank(
        bank_path, expected_field_checkpoint_sha256="field-hash"
    )
    graph_path = tmp_path / "graph.pt"
    torch.save(
        {
            "schema_version": 1,
            "global_rows": torch.tensor([0, 2]),
            "num_global_rows": 3,
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
            "edge_weight": torch.ones(2),
            "raw_affinity": torch.ones(2),
            "local_sigma": torch.ones(2),
            "metadata": {
                "capability_metadata": {
                    "field_checkpoint_sha256": "field-hash",
                    "radio_checkpoint_sha256": "radio-hash",
                    "capability_signatures": {
                        "appearance": _signature("dino", 4),
                        "boundary": _signature("sam3", 2),
                    },
                }
            },
        },
        graph_path,
    )
    graph = load_canonical_support_graph(graph_path, bank)
    assert graph.num_nodes == 2
    assert bank.valid_feature_banks()["appearance"].shape == (2, 4)


def test_capability_cache_rejects_missing_signatures(tmp_path: Path) -> None:
    path = tmp_path / "bank.pt"
    _write_bank(path)
    payload = torch.load(path)
    payload["metadata"].pop("capability_signatures")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="signatures"):
        load_canonical_capability_bank(path)


def test_compact_valid_row_capability_cache_preserves_global_alignment(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact.pt"
    valid = torch.tensor([True, False, True])
    appearance = torch.arange(8, dtype=torch.float16).reshape(2, 4)
    boundary = torch.arange(4, dtype=torch.float16).reshape(2, 2)
    metadata = {
        "source": "canonical_radio_field_official_frozen_capability_views",
        "field_checkpoint_sha256": "field-hash",
        "radio_checkpoint_sha256": "radio-hash",
        "custom_adaptor_head": False,
        "query_independent": True,
        "feature_storage": "valid_rows_compact_v1",
        "feature_row_order": "torch_where_valid_ascending",
        "feature_row_count": 2,
        "capability_signatures": {
            "appearance": _signature("dino", 4),
            "boundary": _signature("sam3", 2),
        },
    }
    torch.save(
        {
            "schema_version": 1,
            "xyz": torch.arange(9, dtype=torch.float32).reshape(3, 3),
            "valid": valid,
            "appearance_dino_v3": appearance,
            "boundary_sam3": boundary,
            "metadata": metadata,
        },
        path,
    )
    path.with_suffix(".pt.json").write_text(
        json.dumps({**metadata, "num_gaussians": 3}), encoding="utf-8"
    )

    bank = load_canonical_capability_bank(
        path, expected_field_checkpoint_sha256="field-hash"
    )

    assert bank.features_are_compact is True
    assert bank.global_rows.tolist() == [0, 2]
    torch.testing.assert_close(bank.valid_feature_banks()["appearance"], appearance)
    torch.testing.assert_close(bank.valid_feature_banks()["boundary"], boundary)


@pytest.mark.parametrize("mutation", ["payload_only", "sidecar_only", "row_count"])
def test_compact_capability_cache_fails_closed_on_storage_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "compact.pt"
    metadata = {
        "source": "canonical_radio_field_official_frozen_capability_views",
        "field_checkpoint_sha256": "field-hash",
        "radio_checkpoint_sha256": "radio-hash",
        "custom_adaptor_head": False,
        "query_independent": True,
        "feature_storage": "valid_rows_compact_v1",
        "feature_row_order": "torch_where_valid_ascending",
        "feature_row_count": 2,
        "capability_signatures": {
            "appearance": _signature("dino", 4),
            "boundary": _signature("sam3", 2),
        },
    }
    payload_metadata = dict(metadata)
    sidecar_metadata = dict(metadata)
    if mutation == "payload_only":
        sidecar_metadata.pop("feature_storage")
    elif mutation == "sidecar_only":
        payload_metadata.pop("feature_storage")
    else:
        payload_metadata["feature_row_count"] = 1
        sidecar_metadata["feature_row_count"] = 1
    torch.save(
        {
            "schema_version": 1,
            "xyz": torch.zeros(3, 3),
            "valid": torch.tensor([True, False, True]),
            "appearance_dino_v3": torch.zeros(2, 4),
            "boundary_sam3": torch.zeros(2, 2),
            "metadata": payload_metadata,
        },
        path,
    )
    path.with_suffix(".pt.json").write_text(
        json.dumps(sidecar_metadata), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="compact|row count|align"):
        load_canonical_capability_bank(path)


def test_dense_capability_archive_can_be_read_without_eager_storage_load(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dense.pt"
    xyz = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    valid = torch.tensor([True, False, True])
    appearance = torch.arange(15, dtype=torch.float16).reshape(3, 5)
    boundary = torch.arange(6, dtype=torch.float16).reshape(3, 2)
    torch.save(
        {
            "schema_version": 1,
            "xyz": xyz,
            "valid": valid,
            "appearance_dino_v3": appearance,
            "boundary_sam3": boundary,
            "metadata": {},
        },
        path,
    )
    path.with_suffix(".pt.json").write_text(
        json.dumps(
            {
                "num_gaussians": 3,
                "appearance_dim": 5,
                "boundary_dim": 2,
            }
        ),
        encoding="utf-8",
    )

    payload = _load_memory_mapped_capability_payload(path)

    torch.testing.assert_close(payload["xyz"], xyz)
    torch.testing.assert_close(payload["valid"].bool(), valid)
    torch.testing.assert_close(payload["appearance_dino_v3"], appearance)
    torch.testing.assert_close(payload["boundary_sam3"], boundary)
    assert payload["_memory_mapped"] is True
