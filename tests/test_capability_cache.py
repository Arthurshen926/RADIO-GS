import hashlib
import json
from pathlib import Path

import pytest
import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.field.factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
    FactorizedRadioFieldSignature,
)
from radio_gs.interfaces.capability_cache import (
    _load_memory_mapped_capability_payload,
    load_canonical_capability_bank,
    load_canonical_support_graph,
)
from radio_gs.interfaces.primitive_row_authority import PrimitiveRowAuthority
from radio_gs.scripts.build_canonical_support_graph import (
    load_graph_capability_bank,
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


def _write_factorized_bank(path: Path) -> dict:
    xyz = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    valid = torch.tensor([True, False, True])
    field_sha = "b" * 64
    radio_sha = "a" * 64
    cache_sha = "c" * 64
    bundle_sha = "d" * 64
    geometry = {
        "num_gaussians": 3,
        "xyz_sha256": hashlib.sha256(
            xyz.numpy().astype("<f4", copy=False).tobytes()
        ).hexdigest(),
    }
    base = FeatureSpaceSignature(
        radio_version="test",
        radio_checkpoint_sha256=radio_sha,
        raw_feature_dim=1280,
        token_type="primitive",
        normalization="radio_raw_full",
        crop_policy="training_views_canonical_factorized_radio_v1",
    )
    factorized = FactorizedRadioFieldSignature.create(base)

    def capability_signature(name: str, dim: int) -> dict:
        return FeatureSpaceSignature(
            radio_version=base.radio_version,
            radio_checkpoint_sha256=radio_sha,
            raw_feature_dim=base.raw_feature_dim,
            adaptor_name=name,
            adaptor_sha256=radio_sha,
            adaptor_output_dim=dim,
            token_type="primitive",
            normalization="l2",
            crop_policy=base.crop_policy,
            field_checkpoint_sha256=field_sha,
        ).to_dict()

    metadata = {
        "source": "canonical_radio_field_official_frozen_capability_views",
        "field_checkpoint": "/tmp/factorized-field.pt",
        "field_checkpoint_sha256": field_sha,
        "field_checkpoint_schema_version": 2,
        "field_checkpoint_contract": CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
        "factorized_radio_field_signature": factorized.to_dict(),
        "factorized_radio_field_signature_sha256": factorized.digest,
        "factorized_radio_contract_sha256": (
            factorized.factorized_radio_contract_sha256
        ),
        "factorized_radio_cache_sha256": cache_sha,
        "registration_responsibility_cache_sha256": "e" * 64,
        "mpr_cache_sha256": cache_sha,
        "feature_output_bundle_sha256": bundle_sha,
        "mpr_geometry_fingerprint": geometry,
        "radio_checkpoint_sha256": radio_sha,
        "custom_adaptor_head": False,
        "query_independent": True,
        "feature_storage": "valid_rows_compact_v1",
        "feature_row_order": "torch_where_valid_ascending",
        "feature_row_count": 2,
        "primitive_row_authority": PrimitiveRowAuthority.from_tensors(
            xyz, valid
        ).to_dict(),
        "capability_signatures": {
            "appearance": capability_signature("dino_v3_7b.feature_projection", 4),
            "boundary": capability_signature("sam3.feature_projection", 2),
        },
    }
    torch.save(
        {
            "schema_version": 1,
            "xyz": xyz,
            "valid": valid,
            "appearance_dino_v3": torch.zeros(2, 4, dtype=torch.float16),
            "boundary_sam3": torch.zeros(2, 2, dtype=torch.float16),
            "metadata": metadata,
        },
        path,
    )
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps({**metadata, "num_gaussians": 3}), encoding="utf-8"
    )
    return metadata


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


def test_factorized_capability_lineage_is_conditionally_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "factorized-bank.pt"
    metadata = _write_factorized_bank(path)
    bank = load_canonical_capability_bank(
        path,
        expected_field_checkpoint_sha256=metadata["field_checkpoint_sha256"],
        require_row_authority=True,
    )
    assert bank.metadata["field_checkpoint_schema_version"] == 2

    payload = torch.load(path)
    payload["metadata"] = dict(payload["metadata"])
    payload["metadata"]["factorized_radio_field_signature_sha256"] = "0" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError, match="signature digest"):
        load_canonical_capability_bank(path)


def test_factorized_capability_rejects_geometry_and_graph_lineage_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "factorized-bank.pt"
    _write_factorized_bank(path)
    bank = load_canonical_capability_bank(path, require_row_authority=True)
    graph_path = tmp_path / "factorized-graph.pt"
    graph_metadata = {
        "capability_metadata": dict(bank.metadata),
        "primitive_row_authority": PrimitiveRowAuthority.from_tensors(
            bank.xyz, bank.valid
        ).to_dict(),
    }
    graph_payload = {
        "schema_version": 1,
        "global_rows": bank.global_rows,
        "num_global_rows": bank.num_gaussians,
        "xyz": bank.xyz[bank.global_rows],
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_weight": torch.ones(2),
        "raw_affinity": torch.ones(2),
        "local_sigma": torch.ones(2),
        "metadata": graph_metadata,
    }
    torch.save(graph_payload, graph_path)
    load_canonical_support_graph(graph_path, bank, require_row_authority=True)

    graph_payload["metadata"] = dict(graph_metadata)
    graph_payload["metadata"]["capability_metadata"] = dict(bank.metadata)
    graph_payload["metadata"]["capability_metadata"][
        "factorized_radio_cache_sha256"
    ] = ("0" * 64)
    torch.save(graph_payload, graph_path)
    with pytest.raises(ValueError, match="factorized lineage"):
        load_canonical_support_graph(graph_path, bank, require_row_authority=True)

    _write_factorized_bank(path)
    payload = torch.load(path)
    payload["xyz"] = payload["xyz"].clone()
    payload["xyz"][0, 0] += 1.0
    payload["metadata"]["primitive_row_authority"] = PrimitiveRowAuthority.from_tensors(
        payload["xyz"], payload["valid"]
    ).to_dict()
    sidecar = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    sidecar["primitive_row_authority"] = payload["metadata"]["primitive_row_authority"]
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(sidecar))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="geometry does not match"):
        load_canonical_capability_bank(path, require_row_authority=True)


def test_capability_cache_rejects_missing_signatures(tmp_path: Path) -> None:
    path = tmp_path / "bank.pt"
    _write_bank(path)
    payload = torch.load(path)
    payload["metadata"].pop("capability_signatures")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="signatures"):
        load_canonical_capability_bank(path)


def test_graph_builder_legacy_authority_bootstrap_is_exact_file_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-bank.pt"
    _write_bank(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bank, audit = load_graph_capability_bank(
        path, legacy_capability_cache_sha256=digest
    )
    assert audit is not None
    assert audit["capability_cache_sha256"] == digest
    assert audit["mutates_source_cache"] is False
    PrimitiveRowAuthority.from_mapping(
        audit["derived_primitive_row_authority"]
    ).validate(bank.xyz, bank.valid)

    with pytest.raises(ValueError, match="SHA-256"):
        load_graph_capability_bank(path, legacy_capability_cache_sha256="0" * 64)
    with pytest.raises(ValueError, match="row authority"):
        load_graph_capability_bank(path)


def test_factorized_graph_builder_requires_exact_modern_capability_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "factorized-bank.pt"
    _write_factorized_bank(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bank, audit = load_graph_capability_bank(
        path,
        expected_capability_cache_sha256=digest,
    )
    assert audit is None
    assert bank.metadata["field_checkpoint_schema_version"] == 2

    with pytest.raises(ValueError, match="caller-trusted"):
        load_graph_capability_bank(path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_graph_capability_bank(
            path,
            expected_capability_cache_sha256="0" * 64,
        )


def test_capability_cache_row_authority_rejects_geometry_or_mask_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority-bank.pt"
    _write_bank(path)
    payload = torch.load(path)
    payload["metadata"]["primitive_row_authority"] = PrimitiveRowAuthority.from_tensors(
        payload["xyz"], payload["valid"]
    ).to_dict()
    torch.save(payload, path)

    load_canonical_capability_bank(path, require_row_authority=True)
    tampered = torch.load(path)
    tampered["xyz"] = tampered["xyz"].clone()
    tampered["xyz"][0, 0] = 1.0
    torch.save(tampered, path)
    with pytest.raises(ValueError, match="row authority"):
        load_canonical_capability_bank(path, require_row_authority=True)

    tampered = dict(payload)
    tampered["valid"] = payload["valid"].clone()
    tampered["valid"][1] = True
    torch.save(tampered, path)
    with pytest.raises(ValueError, match="row authority"):
        load_canonical_capability_bank(path, require_row_authority=True)


def test_support_graph_row_authority_rejects_compact_xyz_drift(
    tmp_path: Path,
) -> None:
    bank_path = tmp_path / "authority-bank.pt"
    _write_bank(bank_path)
    bank_payload = torch.load(bank_path)
    bank_payload["metadata"]["primitive_row_authority"] = (
        PrimitiveRowAuthority.from_tensors(
            bank_payload["xyz"], bank_payload["valid"]
        ).to_dict()
    )
    torch.save(bank_payload, bank_path)
    bank = load_canonical_capability_bank(bank_path, require_row_authority=True)
    graph_path = tmp_path / "authority-graph.pt"
    graph_metadata = {
        "capability_metadata": bank.metadata,
        "primitive_row_authority": PrimitiveRowAuthority.from_tensors(
            bank.xyz, bank.valid
        ).to_dict(),
    }
    graph_payload = {
        "schema_version": 1,
        "global_rows": bank.global_rows,
        "num_global_rows": bank.num_gaussians,
        "xyz": bank.xyz[bank.global_rows].clone(),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_weight": torch.ones(2),
        "raw_affinity": torch.ones(2),
        "local_sigma": torch.ones(2),
        "metadata": graph_metadata,
    }
    torch.save(graph_payload, graph_path)
    load_canonical_support_graph(graph_path, bank, require_row_authority=True)

    graph_payload["xyz"] = graph_payload["xyz"].clone()
    graph_payload["xyz"][0, 1] = 2.0
    torch.save(graph_payload, graph_path)
    with pytest.raises(ValueError, match="xyz rows"):
        load_canonical_support_graph(graph_path, bank, require_row_authority=True)


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
