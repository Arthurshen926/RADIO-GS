from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from radio_gs.scripts import eval_prerendered_lerf_features as legacy_features
from radio_gs.scripts.eval_ours_lerf2d_scalar_maps import (
    ARTIFACT_TYPE,
    CANONICAL_TASK_ID,
    EXPECTED_REGISTRY_ROW,
    SCORE_SEMANTICS,
    FrozenLerf2DContract,
    ScalarMapProtocolError,
    _occam_protocol_config,
    canonical_query_id,
    contract_from_validated_freeze,
    evaluate_scalar_map_bundle,
    write_result_no_clobber,
)


SYNTHETIC_SCENES = ("scene_a", "scene_b", "scene_c", "scene_d")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract() -> FrozenLerf2DContract:
    return FrozenLerf2DContract(
        freeze_path="/frozen/protocol.yaml",
        freeze_sha256="a" * 64,
        freeze_id="synthetic_freeze_v1",
        canonical_task_id=CANONICAL_TASK_ID,
        registry_row=EXPECTED_REGISTRY_ROW,
        scenes=SYNTHETIC_SCENES,
        frames_by_scene={scene: ("frame_00001",) for scene in SYNTHETIC_SCENES},
        labelled_frames=4,
        queries=4,
        protocol_config=_occam_protocol_config(),
    )


def _write_annotation(path: Path, *, query: str = "red cup") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "info": {
                    "name": "frame_00001.jpg",
                    "height": 40,
                    "width": 40,
                },
                "objects": [
                    {
                        "category": query,
                        "bbox": [10, 10, 20, 20],
                        "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_bundle(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    label_root = tmp_path / "labels"
    bundle = tmp_path / "bundle"
    maps = bundle / "maps"
    maps.mkdir(parents=True)
    source = bundle / "render_receipt.json"
    source.write_text('{"renderer":"synthetic"}\n', encoding="utf-8")

    scenes: dict[str, object] = {}
    scale_ids = ["fine", "middle", "coarse"]
    for scene_index, scene in enumerate(SYNTHETIC_SCENES):
        annotation = label_root / scene / "frame_00001.json"
        _write_annotation(annotation)
        score = np.full((3, 1, 40, 40), 0.05, dtype=np.float32)
        score[scene_index % 3, 0, 10:21, 10:21] = 0.95
        map_path = maps / f"{scene}.npy"
        np.save(map_path, score, allow_pickle=False)
        query = "red cup"
        scenes[scene] = {
            "frames": {
                "frame_00001": {
                    "annotation_sha256": _sha(annotation),
                    "camera_name": "frame_00001.jpg",
                    "query_texts": [query],
                    "query_ids": [
                        canonical_query_id(scene, "frame_00001", 0, query)
                    ],
                    "map_file": f"maps/{scene}.npy",
                    "map_sha256": _sha(map_path),
                    "map_shape_lqhw": [3, 1, 40, 40],
                    "map_resolution_hw": [40, 40],
                    "scale_ids": scale_ids,
                }
            }
        }

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "method": "RADIO-GS synthetic",
        "score_semantics": SCORE_SEMANTICS,
        "canonical_task_id": CANONICAL_TASK_ID,
        "registry_row": EXPECTED_REGISTRY_ROW,
        "protocol_freeze": {
            "freeze_id": "synthetic_freeze_v1",
            "sha256": "a" * 64,
        },
        "scales": [
            {"id": "fine", "value": 1.0, "unit": "relative_support"},
            {"id": "middle", "value": 2.0, "unit": "relative_support"},
            {"id": "coarse", "value": 4.0, "unit": "relative_support"},
        ],
        "source_artifacts": [
            {
                "role": "renderer_receipt",
                "path": source.name,
                "sha256": _sha(source),
            }
        ],
        "scenes": scenes,
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, label_root, manifest


def _rewrite_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _frame_entry(manifest: dict[str, object], scene: str = "scene_a") -> dict[str, object]:
    scenes = manifest["scenes"]
    assert isinstance(scenes, dict)
    scene_payload = scenes[scene]
    assert isinstance(scene_payload, dict)
    frames = scene_payload["frames"]
    assert isinstance(frames, dict)
    entry = frames["frame_00001"]
    assert isinstance(entry, dict)
    return entry


def test_scalar_adapter_runs_exact_protocol_and_four_scene_macro(tmp_path: Path) -> None:
    manifest_path, label_root, _ = _write_bundle(tmp_path)

    result = evaluate_scalar_map_bundle(
        manifest_path,
        label_root=label_root,
        contract=_contract(),
    )

    assert result["status"] == "complete_exact_frozen_protocol_evaluation"
    assert result["cohort"] == {
        "scenes": list(SYNTHETIC_SCENES),
        "labelled_frames": 4,
        "queries": 4,
    }
    assert result["scene_macro"]["scenes"] == 4
    assert result["scene_macro"]["aggregation"] == "scene_equal_macro"
    assert result["protocol_config"]["mask_thresh"] == pytest.approx(0.5)
    assert result["protocol_config"]["activation_kernel"] == 30
    assert result["protocol_config"]["smooth_kernel"] == 7
    assert result["protocol_constraints"]["text_encoder_invoked_by_adapter"] is False
    assert result["protocol_constraints"]["threshold_selected_or_tuned"] is False
    assert result["protocol_constraints"]["resize_or_resample"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("camera_name", "frame_00001.png", "exact camera name mismatch"),
        ("query_ids", ["wrong-query-id"], "query ID/order mismatch"),
        ("query_texts", ["cup"], "query text/order mismatch"),
        ("map_resolution_hw", [20, 20], "map resolution binding mismatch"),
        ("map_shape_lqhw", [3, 1, 20, 20], "declared map shape mismatch"),
        ("scale_ids", ["coarse", "middle", "fine"], "scale ID/order mismatch"),
    ],
)
def test_scalar_adapter_rejects_camera_query_resolution_and_scale_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest_path, label_root, manifest = _write_bundle(tmp_path)
    _frame_entry(manifest)[field] = value
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(ScalarMapProtocolError, match=message):
        evaluate_scalar_map_bundle(
            manifest_path,
            label_root=label_root,
            contract=_contract(),
        )


def test_scalar_adapter_rejects_map_and_source_sha_drift(tmp_path: Path) -> None:
    manifest_path, label_root, manifest = _write_bundle(tmp_path)
    map_path = manifest_path.parent / str(_frame_entry(manifest)["map_file"])
    map_path.write_bytes(map_path.read_bytes() + b"tamper")

    with pytest.raises(ScalarMapProtocolError, match="score map SHA256 mismatch"):
        evaluate_scalar_map_bundle(
            manifest_path,
            label_root=label_root,
            contract=_contract(),
        )

    manifest_path, label_root, _ = _write_bundle(tmp_path / "source_case")
    (manifest_path.parent / "render_receipt.json").write_text(
        '{"renderer":"tampered"}\n', encoding="utf-8"
    )
    with pytest.raises(ScalarMapProtocolError, match="source artifact SHA256 mismatch"):
        evaluate_scalar_map_bundle(
            manifest_path,
            label_root=label_root,
            contract=_contract(),
        )


def test_scalar_adapter_rejects_incomplete_cohort_and_nonfinite_map(tmp_path: Path) -> None:
    manifest_path, label_root, manifest = _write_bundle(tmp_path)
    scenes = manifest["scenes"]
    assert isinstance(scenes, dict)
    scenes.pop("scene_d")
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ScalarMapProtocolError, match="scene cohort/order mismatch"):
        evaluate_scalar_map_bundle(
            manifest_path,
            label_root=label_root,
            contract=_contract(),
        )

    manifest_path, label_root, manifest = _write_bundle(tmp_path / "nan_case")
    entry = _frame_entry(manifest)
    map_path = manifest_path.parent / str(entry["map_file"])
    score = np.load(map_path, allow_pickle=False)
    score[0, 0, 0, 0] = np.nan
    np.save(map_path, score, allow_pickle=False)
    entry["map_sha256"] = _sha(map_path)
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ScalarMapProtocolError, match="non-finite"):
        evaluate_scalar_map_bundle(
            manifest_path,
            label_root=label_root,
            contract=_contract(),
        )


def test_scalar_adapter_rejects_symlinked_map(tmp_path: Path) -> None:
    manifest_path, label_root, manifest = _write_bundle(tmp_path)
    entry = _frame_entry(manifest)
    original = manifest_path.parent / str(entry["map_file"])
    alias = original.with_name("scene_a_alias.npy")
    alias.symlink_to(original.name)
    entry["map_file"] = str(alias.relative_to(manifest_path.parent))
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(ScalarMapProtocolError, match="may not contain symlinks"):
        evaluate_scalar_map_bundle(
            manifest_path,
            label_root=label_root,
            contract=_contract(),
        )


def test_contract_rejects_frozen_readout_drift() -> None:
    payload = {
        "freeze_id": "evaluation_protocols_20260801_v1",
        "canonical_tasks": {
            CANONICAL_TASK_ID: {
                "registry_row": EXPECTED_REGISTRY_ROW,
                "cohort": {
                    "scenes": ["figurines", "ramen", "teatime", "waldo_kitchen"],
                    "labelled_frames": 22,
                    "queries": 208,
                },
                "frozen_protocol": {
                    "openclip": "ViT-B-16 / laion2b_s34b_b88k",
                    "camera_lookup": "exact annotation name over all registered cameras",
                    "level_selection": "highest raw OpenCLIP relevance peak over three levels",
                    "segmentation_threshold": 0.5,
                    "activation_filter": "30x30 OpenCV filter2D",
                    "smoothing": "legacy 7x7",
                    "metrics": ["mIoU", "localization_accuracy"],
                    "aggregation": "unweighted equal macro over four scenes",
                },
            }
        },
    }
    contract = contract_from_validated_freeze(
        payload,
        freeze_path=Path("/freeze.yaml"),
        freeze_sha256="b" * 64,
    )
    assert contract.labelled_frames == 22
    assert contract.queries == 208

    payload["canonical_tasks"][CANONICAL_TASK_ID]["frozen_protocol"][  # type: ignore[index]
        "activation_filter"
    ] = "29x29"
    with pytest.raises(ScalarMapProtocolError, match="activation_filter"):
        contract_from_validated_freeze(
            payload,
            freeze_path=Path("/freeze.yaml"),
            freeze_sha256="b" * 64,
        )


def test_scalar_adapter_does_not_change_legacy_openclip_cli_default() -> None:
    args = legacy_features.build_arg_parser().parse_args(
        [
            "--label-root",
            "/labels",
            "--feature-dirs",
            "/features/one",
            "--output-json",
            "/result.json",
        ]
    )
    assert args.protocol_profile == "langsplatv2_released"


def test_scalar_adapter_result_publish_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    write_result_no_clobber(output, {"status": "first"})
    with pytest.raises(ScalarMapProtocolError, match="already exists"):
        write_result_no_clobber(output, {"status": "second"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
