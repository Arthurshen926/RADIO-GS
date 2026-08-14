from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from radio_gs.benchmarks.scannet_uqis.ludvig_image_adapter import (
    DEFAULT_SIGMOID_CALIBRATION,
    FrozenSigmoidCalibration,
    load_frozen_sigmoid_calibration,
    run_precomputed_image_adapter,
    score_mesh_probabilities,
    validate_image_method_manifest,
)
from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION,
    UQISProtocolConfig,
    canonical_json_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_manifest(tmp_path: Path) -> tuple[Path, dict]:
    mesh_path = tmp_path / "mesh_xyz.npy"
    np.save(
        mesh_path,
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        allow_pickle=False,
    )
    crop_path = tmp_path / "crop.png"
    Image.new("RGB", (224, 224), color=(10, 20, 30)).save(crop_path)
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "split_role": "pilot",
        "release_tier": "pilot_harness",
        "formal_benchmark_eligible": False,
        "protocol_config": asdict(UQISProtocolConfig()),
        "protocol_config_sha256": canonical_json_sha256(
            asdict(UQISProtocolConfig())
        ),
        "query_id_salt_sha256": "a" * 64,
        "visibility": "method_input",
        "modality": "image",
        "prediction_domain": "official_scannet_mesh_vertex_probability",
        "scene_domains": [
            {
                "scene_id": "scene0001_00",
                "mesh_xyz_path": str(mesh_path.resolve()),
                "mesh_xyz_sha256": _sha256(mesh_path),
                "mesh_vertices": 2,
            }
        ],
        "queries": [
            {
                "query_id": "uq_" + "1" * 32,
                "scene_id": "scene0001_00",
                "modality": "image",
                "crop_rgb_path": str(crop_path.resolve()),
                "crop_rgb_sha256": _sha256(crop_path),
                "available_method_inputs": ["scene_id", "crop_rgb"],
            }
        ],
    }
    path = tmp_path / "query_manifest.image.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_image_adapter_accepts_only_the_exact_public_method_manifest(
    tmp_path: Path,
) -> None:
    path, payload = _image_manifest(tmp_path)

    validated = validate_image_method_manifest(path)

    assert validated["manifest_sha256"] == _sha256(path)
    assert validated["scene_domains"]["scene0001_00"]["mesh_vertices"] == 2
    assert validated["queries"][0]["available_method_inputs"] == [
        "scene_id",
        "crop_rgb",
    ]

    payload["queries"][0]["instance_id"] = 7
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="method-visible fields changed"):
        validate_image_method_manifest(path)


def test_image_adapter_requires_one_query_workspace(tmp_path: Path) -> None:
    path, payload = _image_manifest(tmp_path)
    duplicate = dict(payload["queries"][0])
    duplicate["query_id"] = "uq_" + "2" * 32
    payload["queries"].append(duplicate)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one query"):
        validate_image_method_manifest(path)


@pytest.mark.parametrize(
    ("location", "name"),
    [
        ("top", "target_manifest_path"),
        ("scene", "mesh_instance_ids_path"),
        ("query", "evaluator_target_id"),
    ],
)
def test_image_adapter_rejects_private_or_evaluator_inputs(
    tmp_path: Path, location: str, name: str
) -> None:
    path, payload = _image_manifest(tmp_path)
    if location == "top":
        payload[name] = "/private/target_manifest.evaluator.json"
    elif location == "scene":
        payload["scene_domains"][0][name] = "/private/instance_ids.npy"
    else:
        payload["queries"][0][name] = "target_0001"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="method-visible fields changed"):
        validate_image_method_manifest(path)


def test_frozen_global_sigmoid_is_monotonic_and_cannot_be_fit_on_test(
    tmp_path: Path,
) -> None:
    logits = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    assert DEFAULT_SIGMOID_CALIBRATION.apply(logits) == pytest.approx(
        [0.26894143, 0.5, 0.7310586]
    )

    calibration_path = tmp_path / "calibration.json"
    calibration = {
        "schema_version": "scannet_uqis_ludvig_global_sigmoid_v1",
        "benchmark_version": BENCHMARK_VERSION,
        "modality": "image",
        "scope": "global",
        "fit_split_role": "dev",
        "frozen": True,
        "scale": 2.0,
        "bias": -0.25,
    }
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

    loaded = load_frozen_sigmoid_calibration(calibration_path)

    assert loaded == FrozenSigmoidCalibration(scale=2.0, bias=-0.25)
    assert np.diff(loaded.apply(logits)).min() > 0.0

    calibration["fit_split_role"] = "test"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="test-split fitting is forbidden"):
        load_frozen_sigmoid_calibration(calibration_path)


def test_pure_40d_scorer_reads_gaussian_cosines_on_the_mesh_domain() -> None:
    features = np.zeros((2, 40), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 0] = -1.0
    descriptor = np.zeros(40, dtype=np.float32)
    descriptor[0] = 1.0
    gaussian_xyz = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    covariance = np.repeat((np.eye(3, dtype=np.float32) * 0.01)[None], 2, axis=0)
    opacity = np.ones(2, dtype=np.float32)
    mesh_xyz = gaussian_xyz.copy()

    probabilities, support = score_mesh_probabilities(
        gaussian_features=features,
        query_descriptor=descriptor,
        gaussian_xyz=gaussian_xyz,
        gaussian_covariance=covariance,
        gaussian_opacity=opacity,
        mesh_xyz=mesh_xyz,
        candidate_indices=np.asarray([[0], [1]], dtype=np.int64),
    )

    assert probabilities.dtype == np.float32
    assert probabilities == pytest.approx([0.7310586, 0.26894143])
    assert support == pytest.approx([1.0, 1.0])
    assert np.isfinite(probabilities).all()
    assert ((0.0 <= probabilities) & (probabilities <= 1.0)).all()

    with pytest.raises(ValueError, match="exactly 40-D"):
        score_mesh_probabilities(
            gaussian_features=features[:, :39],
            query_descriptor=descriptor[:39],
            gaussian_xyz=gaussian_xyz,
            gaussian_covariance=covariance,
            gaussian_opacity=opacity,
            mesh_xyz=mesh_xyz,
            candidate_indices=np.asarray([[0], [1]], dtype=np.int64),
        )


def test_mesh_readout_is_log_stable_for_far_official_vertices() -> None:
    features = np.zeros((1, 40), dtype=np.float32)
    features[0, 0] = 1.0
    descriptor = features[0].copy()

    probabilities, support = score_mesh_probabilities(
        gaussian_features=features,
        query_descriptor=descriptor,
        gaussian_xyz=np.zeros((1, 3), dtype=np.float32),
        gaussian_covariance=np.eye(3, dtype=np.float32)[None] * 1e-4,
        gaussian_opacity=np.ones(1, dtype=np.float32),
        mesh_xyz=np.asarray([[100.0, 100.0, 100.0]], dtype=np.float32),
        candidate_indices=np.asarray([[0]], dtype=np.int64),
    )

    assert probabilities == pytest.approx([0.7310586])
    assert np.isfinite(probabilities).all()
    assert support[0] == pytest.approx(np.finfo(np.float32).tiny)


def test_precomputed_adapter_smoke_writes_official_mesh_probabilities(
    tmp_path: Path,
) -> None:
    method_manifest, payload = _image_manifest(tmp_path)
    query_id = payload["queries"][0]["query_id"]
    features = np.zeros((2, 40), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 0] = -1.0
    descriptor = np.zeros(40, dtype=np.float32)
    descriptor[0] = 1.0
    xyz = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    covariance = np.repeat((np.eye(3, dtype=np.float32) * 0.01)[None], 2, axis=0)
    output_dir = tmp_path / "predictions"

    manifest = run_precomputed_image_adapter(
        query_manifest_path=method_manifest,
        scene_id="scene0001_00",
        gaussian_features=features,
        query_descriptors={query_id: descriptor},
        gaussian_xyz=xyz,
        gaussian_covariance=covariance,
        gaussian_opacity=np.ones(2, dtype=np.float32),
        output_dir=output_dir,
        candidate_indices=np.asarray([[0], [1]], dtype=np.int64),
    )

    probability_path = output_dir / f"{query_id}.npy"
    probabilities = np.load(probability_path, allow_pickle=False)
    assert probabilities == pytest.approx([0.7310586, 0.26894143])
    assert probabilities.shape == (2,)
    assert manifest["official_ludvig_reproduction"] is False
    assert manifest["paper_metric_comparable"] is False
    assert manifest["benchmark_local_adapter"] is True
    assert manifest["privacy_boundary"] == {
        "evaluator_manifest_opened": False,
        "private_target_inputs_opened": False,
        "method_visible_inputs_only": True,
    }
    assert manifest["prediction_domain"] == ("official_scannet_mesh_vertex_probability")
    assert manifest["queries"][0]["probability"]["shape"] == [2]
    assert json.loads((output_dir / "run_manifest.json").read_text()) == manifest


def test_exact_cli_exposes_no_evaluator_or_test_fit_inputs(tmp_path: Path) -> None:
    from reproductions.ludvig.run_uqis_image import parser

    options = parser().parse_args(
        [
            "--query-manifest",
            str(tmp_path / "query_manifest.image.json"),
            "--workspace-receipt",
            str(tmp_path / "workspace_receipt.json"),
            "--phase-b-dir",
            str(tmp_path / "phase_b"),
            "--phase-b-manifest-sha256",
            "b" * 64,
            "--phase-c-dir",
            str(tmp_path / "phase_c"),
            "--phase-c-manifest-sha256",
            "c" * 64,
            "--output-dir",
            str(tmp_path / "predictions"),
        ]
    )

    assert options.calibration is None
    assert options.workspace_receipt == tmp_path / "workspace_receipt.json"
    assert not {
        "evaluator_manifest",
        "target_manifest",
        "fit_calibration",
        "sigmoid_scale",
        "sigmoid_bias",
    }.intersection(vars(options))

    with pytest.raises(SystemExit):
        parser().parse_args(["--fit-calibration", "test"])
