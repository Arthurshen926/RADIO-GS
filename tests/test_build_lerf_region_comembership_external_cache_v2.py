from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radio_gs.scripts import build_lerf_region_comembership_external_cache_v2 as builder
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json


def _authorities() -> tuple[dict, dict, dict[str, str]]:
    execution = {"path": "/execution.json", "sha256": "e" * 64}
    feature_record = {"path": "/feature.pt", "sha256": "f" * 64}
    feature = {
        "scene_id": "synthetic_target",
        "domain": "target",
        "target_execution_authority": execution,
        "source_access": {"target_feature_authorities_opened": True},
        "region_fingerprints": ["a", "b", "c", "d"],
        "region_fingerprints_sha256": "1" * 64,
        "canonical_axis_sha256": "2" * 64,
        "pair_axis_sha256": "3" * 64,
        "canonical_region_indices": torch.arange(4, dtype=torch.int64),
        "pair_indices": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64),
        "region_rows": torch.tensor([[0, 1], [2, 3], [3, 4], [5, 6]]),
        "token_mask": torch.ones(4, 2, dtype=torch.bool),
    }
    inference = {
        "scene_id": feature["scene_id"],
        "domain": "target",
        "target_execution_authority": execution,
        "feature_authority": feature_record,
        "source_access": feature["source_access"],
        "region_fingerprints": feature["region_fingerprints"],
        "region_fingerprints_sha256": feature["region_fingerprints_sha256"],
        "canonical_axis_sha256": feature["canonical_axis_sha256"],
        "pair_axis_sha256": feature["pair_axis_sha256"],
        "canonical_region_indices": feature["canonical_region_indices"],
        "pair_indices": feature["pair_indices"],
        "pair_probabilities": torch.tensor([0.8, 0.9, 0.7]),
        "selected_rule": {
            "method": "maximum_product",
            "maximum_regions": 2,
            "threshold": 0.75,
        },
    }
    return feature, inference, feature_record


def test_bounded_cache_readout_uses_only_inference_selected_rule() -> None:
    feature, inference, feature_record = _authorities()
    rule = builder.validate_v2_authority_binding(
        feature=feature, inference=inference, feature_record=feature_record
    )
    assert rule == inference["selected_rule"]
    result = builder.bounded_readout_from_v2(
        feature=feature,
        inference=inference,
        region_o0_scores=torch.tensor(
            [[0.9, 0.1], [0.2, 0.8], [0.3, 0.4], [0.1, 0.2]]
        ),
        num_primitives=7,
    )
    assert result.seed_region_indices.tolist() == [0, 1]
    assert result.selected_region_masks.sum(dim=0).tolist() == [2, 2]


def test_support_geometry_accepts_distinct_validity_domains_and_masks_output() -> None:
    full_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    global_rows = torch.tensor([0, 1, 2])
    o0_valid = torch.tensor([False, True, True, True])
    audit = builder.validate_support_geometry_binding(
        graph_xyz=full_xyz[global_rows],
        global_rows=global_rows,
        full_xyz=full_xyz,
        o0_valid=o0_valid,
    )
    assert audit == {
        "graph_active_primitives": 3,
        "o0_valid_primitives": 3,
        "intersection_primitives": 2,
        "o0_only_primitives": 1,
        "mpr_only_primitives": 1,
    }
    membership, removed = builder.mask_membership_to_o0_valid(
        torch.tensor([[1.0], [1.0], [0.0], [0.0]]), o0_valid
    )
    assert removed == 1
    assert membership[:, 0].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_support_geometry_still_rejects_coordinate_or_row_drift() -> None:
    full_xyz = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    with pytest.raises(ValueError, match="support geometry"):
        builder.validate_support_geometry_binding(
            graph_xyz=full_xyz[[0, 2]] + 1e-6,
            global_rows=torch.tensor([0, 2]),
            full_xyz=full_xyz,
            o0_valid=torch.ones(4, dtype=torch.bool),
        )


def test_v2_cache_binding_rejects_pair_axis_drift() -> None:
    feature, inference, feature_record = _authorities()
    inference["pair_axis_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="binding"):
        builder.validate_v2_authority_binding(
            feature=feature, inference=inference, feature_record=feature_record
        )


def test_cli_has_no_method_k_or_threshold_override() -> None:
    parser = builder.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {
        "help",
        "execution_authority",
        "expected_execution_authority_sha256",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--execution-authority",
                "authority.json",
                "--expected-execution-authority-sha256",
                "a" * 64,
                "--maximum-regions",
                "8",
            ]
        )


def _query_execution(
    tmp_path: Path,
    *,
    feature_record: dict[str, str],
    inference_record: dict[str, str],
    positive_record: dict[str, str],
    negative_record: dict[str, str],
    access_queries: bool = True,
    dependencies: dict[str, dict[str, str]] | None = None,
) -> Path:
    authority = {
        "schema": builder.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": builder.EXECUTION_AUTHORITY_STATUS,
        "scene_id": "synthetic_target",
        "implementation": file_record(Path(builder.__file__).resolve()),
        "implementation_dependencies": (
            dependencies
            if dependencies is not None
            else {
                name: file_record(path)
                for name, path in builder.IMPLEMENTATION_DEPENDENCIES.items()
            }
        ),
        "feature_authority": feature_record,
        "inference_authority": inference_record,
        "positive_cache": positive_record,
        "negative_cache": negative_record,
        "renderer_geometry_checkpoint_sha256": "a" * 64,
        "knn_chunk_size": 65536,
        "output_cache": str((tmp_path / "external.pt").resolve()),
        "output_report": str((tmp_path / "external.json").resolve()),
        "query_readout_authorized": True,
        "target_metric_authorized": False,
        "access_audit": {
            "benchmark_queries_opened": access_queries,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "target_metrics_computed": False,
        },
    }
    return write_frozen_json(tmp_path / f"query_execution_{access_queries}.json", authority)


def test_query_readout_authority_is_separate_and_explicitly_opens_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_file = tmp_path / "feature.pt"
    inference_file = tmp_path / "inference.pt"
    accepted_file = tmp_path / "accepted.pt"
    positive_file = tmp_path / "positive.pt"
    negative_file = tmp_path / "negative.pt"
    for path in (
        feature_file,
        inference_file,
        accepted_file,
        positive_file,
        negative_file,
    ):
        path.write_bytes(path.name.encode())
    feature_record = file_record(feature_file)
    inference_record = file_record(inference_file)
    positive_record = file_record(positive_file)
    negative_record = file_record(negative_file)
    accepted_record = file_record(accepted_file)
    feature, inference, _ = _authorities()
    feature["input_authority"] = {"accepted_v2": accepted_record}
    inference["feature_authority"] = feature_record
    accepted = {
        "scene_id": feature["scene_id"],
        "physical_space_authority": {
            "geometry_checkpoint_sha256": "a" * 64,
        },
        "region_fingerprints": feature["region_fingerprints"],
        "canonical_region_indices": feature["canonical_region_indices"],
        "region_rows": feature["region_rows"],
        "token_mask": feature["token_mask"],
    }

    def synthetic_load(path, *, expected_sha256, map_location, label):
        source = Path(path).resolve()
        if source == feature_file.resolve():
            return feature, expected_sha256, source
        if source == inference_file.resolve():
            return inference, expected_sha256, source
        if source == accepted_file.resolve():
            return accepted, expected_sha256, source
        raise AssertionError("query caches must only be hash-validated in preflight")

    monkeypatch.setattr(builder, "load_torch_mapping", synthetic_load)
    monkeypatch.setattr(builder, "validate_feature_authority", lambda value: value)
    monkeypatch.setattr(builder, "validate_inference_authority", lambda value: value)
    monkeypatch.setattr(
        builder, "validate_target_accepted_v2_authority", lambda value: value
    )
    authority = _query_execution(
        tmp_path,
        feature_record=feature_record,
        inference_record=inference_record,
        positive_record=positive_record,
        negative_record=negative_record,
    )
    validated = builder.validate_query_readout_execution_authority(
        authority, expected_sha256=file_record(authority)["sha256"]
    )
    assert validated["access_audit"]["benchmark_queries_opened"] is True
    assert validated["selected_rule"] == inference["selected_rule"]

    rejected = _query_execution(
        tmp_path,
        feature_record=feature_record,
        inference_record=inference_record,
        positive_record=positive_record,
        negative_record=negative_record,
        access_queries=False,
    )
    with pytest.raises(ValueError, match="header"):
        builder.validate_query_readout_execution_authority(
            rejected, expected_sha256=file_record(rejected)["sha256"]
        )


def test_renderer_geometry_binding_rejects_geometry_checkpoint_drift() -> None:
    feature, _, _ = _authorities()
    accepted_record = {"path": "/accepted.pt", "sha256": "a" * 64}
    feature["input_authority"] = {"accepted_v2": accepted_record}
    accepted = {
        "scene_id": feature["scene_id"],
        "physical_space_authority": {
            "geometry_checkpoint_sha256": "b" * 64,
        },
        "region_fingerprints": feature["region_fingerprints"],
        "canonical_region_indices": feature["canonical_region_indices"],
        "region_rows": feature["region_rows"],
        "token_mask": feature["token_mask"],
    }
    with pytest.raises(ValueError, match="geometry binding"):
        builder.validate_renderer_geometry_binding(
            feature=feature,
            accepted=accepted,
            accepted_record=accepted_record,
            renderer_geometry_checkpoint_sha256="c" * 64,
        )


def test_query_caches_remain_unopened_when_feature_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_file = tmp_path / "feature.pt"
    inference_file = tmp_path / "inference.pt"
    feature_file.write_bytes(b"feature")
    inference_file.write_bytes(b"inference")
    feature_record = file_record(feature_file)
    inference_record = file_record(inference_file)
    missing = {
        "path": str(tmp_path / "must_not_open.pt"),
        "sha256": "a" * 64,
    }
    authority = _query_execution(
        tmp_path,
        feature_record=feature_record,
        inference_record=inference_record,
        positive_record=missing,
        negative_record=missing,
    )

    def synthetic_load(path, *, expected_sha256, map_location, label):
        return {}, expected_sha256, Path(path).resolve()

    monkeypatch.setattr(builder, "load_torch_mapping", synthetic_load)

    def reject_feature(value):
        raise ValueError("synthetic feature gate rejected")

    monkeypatch.setattr(builder, "validate_feature_authority", reject_feature)
    with pytest.raises(ValueError, match="feature gate rejected"):
        builder.validate_query_readout_execution_authority(
            authority, expected_sha256=file_record(authority)["sha256"]
        )


def test_query_readout_authority_rejects_dependency_substitution(
    tmp_path: Path,
) -> None:
    files = {}
    for name in ("feature", "inference", "positive", "negative"):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        files[name] = file_record(path)
    dependencies = {
        name: file_record(path)
        for name, path in builder.IMPLEMENTATION_DEPENDENCIES.items()
    }
    substituted = tmp_path / "substituted_frozen_o0.py"
    substituted.write_bytes(b"not the frozen O0 implementation")
    dependencies["frozen_o0"] = file_record(substituted)
    authority = _query_execution(
        tmp_path,
        feature_record=files["feature"],
        inference_record=files["inference"],
        positive_record=files["positive"],
        negative_record=files["negative"],
        dependencies=dependencies,
    )
    with pytest.raises(ValueError, match="another dependency: frozen_o0"):
        builder.validate_query_readout_execution_authority(
            authority, expected_sha256=file_record(authority)["sha256"]
        )
