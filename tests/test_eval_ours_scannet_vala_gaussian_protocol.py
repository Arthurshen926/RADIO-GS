from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.scripts.eval_ours_scannet_vala_gaussian_protocol import (
    ARTIFACT_TYPE,
    CLASS_ORDER_SHA256,
    CURRENT_MATERIALIZER_CONTRACT,
    CURRENT_METHOD_FAMILY,
    CANONICAL_MAINLINE_NAME,
    CANONICAL_MAINLINE_SHA256,
    CANONICAL_METHOD_FREEZE_NAME,
    CANONICAL_METHOD_FREEZE_SHA256,
    CANONICAL_READOUT_SHA256,
    CANONICAL_REGION_RADII_M,
    CANONICAL_TOTALITY_CONTRACT,
    EXTERNAL_PROTOCOL_FREEZE_ID,
    EXTERNAL_PROTOCOL_FREEZE_SHA256,
    EXTERNAL_PROTOCOL_FREEZE_TASK,
    EXTERNAL_PROTOCOL_REGISTRY_ROW,
    LEGACY_MATERIALIZER_CONTRACT,
    LEGACY_METHOD_FAMILY,
    OFFICIAL_RADIO_SHA256,
    PAPER_CLASS_IDS,
    PAPER_CLASS_NAMES,
    PREDICTION_DOMAIN,
    PROTOCOL_CONTRACT,
    QUERY_CLASS_ORDER_SHA256,
    QUERY_TEXT_SHA256,
    ROW_ORDER,
    SCHEMA_VERSION,
    SEMANTIC_READOUT,
    SPATIAL_TRANSFER,
    _tensor_sha256,
    load_ours_gaussian_semantic_score_cache,
    predict_frozen_splits,
    validate_ours_gaussian_semantic_score_cache,
)
from radio_gs.utils.immutable_artifacts import file_record, sha256_file


def _inputs(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    geometry_path = tmp_path / "geometry.pth"
    query_path = tmp_path / "queries.pt"
    semantic_path = tmp_path / "semantic.pt"
    geometry_path.write_bytes(b"synthetic geometry authority")
    query_path.write_bytes(b"synthetic fixed ScanNet class query bank")
    semantic_path.write_bytes(b"synthetic query-independent semantic field")
    repo_root = Path(__file__).resolve().parents[1]
    protocol_freeze = repo_root / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml"
    producer_source = (
        repo_root
        / "radio_gs/scripts/materialize_ours_scannet_gaussian_semantic_score_cache.py"
    )
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    scale = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, 0.2, 0.2], [0.3, 0.4, 0.5]],
        dtype=torch.float32,
    )
    quaternion = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * len(xyz), dtype=torch.float32
    )
    opacity = torch.tensor([0.5, 0.75, 1.0], dtype=torch.float32)
    valid = torch.ones(len(xyz), dtype=torch.bool)
    scores = torch.zeros(len(xyz), len(PAPER_CLASS_IDS), dtype=torch.float32)
    scores[0, PAPER_CLASS_IDS.index(1)] = 3.0
    scores[1, PAPER_CLASS_IDS.index(3)] = 4.0
    scores[2, PAPER_CLASS_IDS.index(33)] = 5.0
    tensors = {
        "xyz": xyz,
        "scale": scale,
        "quaternion": quaternion,
        "opacity": opacity,
        "valid": valid,
        "semantic_scores": scores,
    }
    geometry_sha = sha256_file(geometry_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        **tensors,
        "class_ids": list(PAPER_CLASS_IDS),
        "class_names": list(PAPER_CLASS_NAMES),
        "query_ids": list(PAPER_CLASS_NAMES),
        "metadata": {
            "protocol_contract": PROTOCOL_CONTRACT,
            "scene_id": "scene0000_00",
            "prediction_domain": PREDICTION_DOMAIN,
            "row_order": ROW_ORDER,
            "semantic_readout": SEMANTIC_READOUT,
            "spatial_transfer": SPATIAL_TRANSFER,
            "mesh_vertices_used": False,
            "knn_used": False,
            "geometry_checkpoint": file_record(geometry_path),
            "geometry_checkpoint_sha256": geometry_sha,
            "query_source": file_record(query_path),
            "query_source_sha256": sha256_file(query_path),
            "semantic_source": file_record(semantic_path),
            "semantic_source_sha256": sha256_file(semantic_path),
            "query_text_sha256": QUERY_TEXT_SHA256,
            "class_order_sha256": CLASS_ORDER_SHA256,
            "query_class_order_sha256": QUERY_CLASS_ORDER_SHA256,
            "method_family": LEGACY_METHOD_FAMILY,
            "materializer_contract": LEGACY_MATERIALIZER_CONTRACT,
            "diagnostic_only": True,
            "protocol_freeze_id": EXTERNAL_PROTOCOL_FREEZE_ID,
            "protocol_freeze_task": EXTERNAL_PROTOCOL_FREEZE_TASK,
            "protocol_registry_row": EXTERNAL_PROTOCOL_REGISTRY_ROW,
            "protocol_freeze": file_record(protocol_freeze),
            "protocol_freeze_sha256": EXTERNAL_PROTOCOL_FREEZE_SHA256,
            "producer_source": file_record(producer_source),
            "producer_source_sha256": sha256_file(producer_source),
            "row_tensor_sha256": {
                key: _tensor_sha256(tensor) for key, tensor in tensors.items()
            },
        },
    }
    return {
        "payload": payload,
        "geometry_sha": geometry_sha,
        **tensors,
    }


def _validate(inputs: dict[str, object]) -> dict[str, object]:
    return validate_ours_gaussian_semantic_score_cache(
        inputs["payload"],
        expected_scene_id="scene0000_00",
        expected_xyz=inputs["xyz"],
        expected_scale=inputs["scale"],
        expected_quaternion=inputs["quaternion"],
        expected_opacity=inputs["opacity"],
        expected_valid=inputs["valid"],
        expected_geometry_checkpoint_sha256=inputs["geometry_sha"],
        expected_method_family=LEGACY_METHOD_FAMILY,
    )


def test_strict_cache_load_and_frozen_split_argmax(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    cache_path = tmp_path / "score_cache.pt"
    torch.save(inputs["payload"], cache_path)

    cache, digest, source = load_ours_gaussian_semantic_score_cache(
        cache_path,
        expected_scene_id="scene0000_00",
        expected_xyz=inputs["xyz"],
        expected_scale=inputs["scale"],
        expected_quaternion=inputs["quaternion"],
        expected_opacity=inputs["opacity"],
        expected_valid=inputs["valid"],
        expected_geometry_checkpoint_sha256=inputs["geometry_sha"],
        expected_method_family=LEGACY_METHOD_FAMILY,
    )
    prediction = predict_frozen_splits(cache["semantic_scores"])

    assert source == cache_path.resolve()
    assert digest == sha256_file(cache_path)
    assert prediction["19"].tolist() == [1, 3, 33]
    # Cabinet (3) is absent from split10, so its all-zero restricted tie picks wall.
    assert prediction["10"].tolist() == [1, 1, 33]


def test_cache_rejects_geometry_row_reordering_even_if_its_hash_is_updated(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    payload = inputs["payload"]
    payload["xyz"] = payload["xyz"][[1, 0, 2]]
    payload["metadata"]["row_tensor_sha256"]["xyz"] = _tensor_sha256(payload["xyz"])

    with pytest.raises(ValueError, match="xyz/row-order differs"):
        _validate(inputs)


def test_cache_rejects_checkpoint_and_query_source_sha_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["payload"]["metadata"]["geometry_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="geometry checkpoint SHA256 differs"):
        _validate(inputs)

    inputs = _inputs(tmp_path / "second")
    query_path = Path(inputs["payload"]["metadata"]["query_source"]["path"])
    query_path.write_bytes(b"tampered after score materialization")
    with pytest.raises(ValueError, match="query source SHA-256 differs"):
        _validate(inputs)


def test_cache_rejects_class_order_incomplete_rows_and_mesh_knn8(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "class")
    inputs["payload"]["class_ids"] = list(reversed(PAPER_CLASS_IDS))
    with pytest.raises(ValueError, match="class id/order differs"):
        _validate(inputs)

    inputs = _inputs(tmp_path / "valid")
    inputs["payload"]["valid"][1] = False
    inputs["payload"]["metadata"]["row_tensor_sha256"]["valid"] = _tensor_sha256(
        inputs["payload"]["valid"]
    )
    inputs["valid"] = inputs["payload"]["valid"].clone()
    with pytest.raises(ValueError, match="score for every geometry row"):
        _validate(inputs)

    inputs = _inputs(tmp_path / "mesh")
    inputs["payload"]["knn_indices"] = torch.zeros(3, 8, dtype=torch.int64)
    with pytest.raises(ValueError, match="legacy mesh/kNN8"):
        _validate(inputs)


def test_cache_rejects_score_hash_and_shape_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "hash")
    inputs["payload"]["semantic_scores"][0, 0] += 1.0
    with pytest.raises(ValueError, match="semantic_scores SHA256 differs"):
        _validate(inputs)

    inputs = _inputs(tmp_path / "shape")
    inputs["payload"]["semantic_scores"] = torch.zeros(3, 15)
    with pytest.raises(ValueError, match=r"row-aligned \[N,19\]"):
        _validate(inputs)


def test_prediction_requires_frozen_split19_score_columns() -> None:
    with pytest.raises(ValueError, match="frozen split19 columns"):
        predict_frozen_splits(torch.zeros(2, 15))
    prediction = predict_frozen_splits(torch.zeros(0, len(PAPER_CLASS_IDS)))
    assert all(value.dtype == np.int32 for value in prediction.values())


def test_current_mainline_requires_full_canonical_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    payload = inputs["payload"]
    metadata = payload["metadata"]
    authority_paths = {}
    for name in ("mainline", "method_freeze", "field", "mpr", "graph", "readout", "radio"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        authority_paths[name] = path

    def record(path: Path, digest=None) -> dict[str, str]:
        return {"path": str(path.resolve()), "sha256": digest or sha256_file(path)}

    descriptor_record = dict(metadata["semantic_source"])
    metadata.update(
        {
            "method_family": CURRENT_METHOD_FAMILY,
            "materializer_contract": CURRENT_MATERIALIZER_CONTRACT,
            "diagnostic_only": False,
            "canonical_mainline_name": CANONICAL_MAINLINE_NAME,
            "canonical_mainline_sha256": CANONICAL_MAINLINE_SHA256,
            "canonical_method_freeze_name": CANONICAL_METHOD_FREEZE_NAME,
            "canonical_method_freeze_sha256": CANONICAL_METHOD_FREEZE_SHA256,
            "surface_region_readout_sha256": CANONICAL_READOUT_SHA256,
            "official_radio_checkpoint_sha256": OFFICIAL_RADIO_SHA256,
            "region_radii_m": list(CANONICAL_REGION_RADII_M),
            "score_formula": (
                "l2_normalize(canonical_mpr_v3_surface_region_descriptor) @ "
                "l2_normalize(exact_split19_text_embedding).T"
            ),
            "query_set_calibration": False,
            "logit_calibration": "none",
            "logit_smoothing": "none",
            "canonical_field_geometry_row_match": True,
            "region_graph_geometry_row_match": True,
            "region_scale_aggregation": "max_independent_cosine_over_0.20_0.40_0.70",
            "totality_semantics": (
                "graph_observed_surface_region_h128_else_exact_canonical_field_primitive"
            ),
            "totality_contract": CANONICAL_TOTALITY_CONTRACT,
            "no_evidence_fallback": (
                "canonical_field_primitive_official_summary_head_independent_cosine"
            ),
            "canonical_mainline": record(
                authority_paths["mainline"], CANONICAL_MAINLINE_SHA256
            ),
            "canonical_method_freeze": record(
                authority_paths["method_freeze"], CANONICAL_METHOD_FREEZE_SHA256
            ),
            "canonical_field_source": record(authority_paths["field"]),
            "mpr_source": record(authority_paths["mpr"]),
            "support_graph_source": record(authority_paths["graph"]),
            "surface_region_readout_source": record(
                authority_paths["readout"], CANONICAL_READOUT_SHA256
            ),
            "official_radio_source": record(
                authority_paths["radio"], OFFICIAL_RADIO_SHA256
            ),
        }
    )
    metadata["canonical_field_source"] = descriptor_record
    payload["region_observed"] = torch.tensor([True, False, True])
    metadata["region_observed_count"] = 2
    metadata["no_evidence_fallback_count"] = 1
    metadata["row_tensor_sha256"]["region_observed"] = _tensor_sha256(
        payload["region_observed"]
    )
    import radio_gs.scripts.eval_ours_scannet_vala_gaussian_protocol as evaluator

    monkeypatch.setattr(
        evaluator,
        "validate_file_record",
        lambda value, *, label: Path(value["path"]),
    )
    validate_ours_gaussian_semantic_score_cache(
        payload,
        expected_scene_id="scene0000_00",
        expected_xyz=inputs["xyz"],
        expected_scale=inputs["scale"],
        expected_quaternion=inputs["quaternion"],
        expected_opacity=inputs["opacity"],
        expected_valid=inputs["valid"],
        expected_geometry_checkpoint_sha256=inputs["geometry_sha"],
        expected_method_family=CURRENT_METHOD_FAMILY,
    )
    metadata["no_evidence_fallback_count"] = 0
    with pytest.raises(ValueError, match="no_evidence_fallback_count differs"):
        validate_ours_gaussian_semantic_score_cache(
            payload,
            expected_scene_id="scene0000_00",
            expected_xyz=inputs["xyz"],
            expected_scale=inputs["scale"],
            expected_quaternion=inputs["quaternion"],
            expected_opacity=inputs["opacity"],
            expected_valid=inputs["valid"],
            expected_geometry_checkpoint_sha256=inputs["geometry_sha"],
            expected_method_family=CURRENT_METHOD_FAMILY,
        )
