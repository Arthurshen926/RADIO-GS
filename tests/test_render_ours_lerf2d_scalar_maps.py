from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from radio_gs.scripts.eval_lerf_direct_3d_selection import xyz_geometry_fingerprint
from radio_gs.scripts.eval_ours_lerf2d_scalar_maps import (
    CANONICAL_TASK_ID,
    EXPECTED_REGISTRY_ROW,
    FrozenLerf2DContract,
    _occam_protocol_config,
    evaluate_scalar_map_bundle,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    DIRECT3D_CONTRACT,
    SHARED_AUTHORITY_CONTRACT,
    _tensor_sha256,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache_fp32 import (
    SHARED_AUTHORITY_CONTRACT as FP32_SHARED_AUTHORITY_CONTRACT,
)
from radio_gs.scripts import render_ours_lerf2d_scalar_maps as renderer


SCENES = ("scene_a", "scene_b", "scene_c", "scene_d")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _contract() -> FrozenLerf2DContract:
    return FrozenLerf2DContract(
        freeze_path="/synthetic/freeze.yaml",
        freeze_sha256="f" * 64,
        freeze_id="synthetic_freeze_v1",
        canonical_task_id=CANONICAL_TASK_ID,
        registry_row=EXPECTED_REGISTRY_ROW,
        scenes=SCENES,
        frames_by_scene={scene: ("frame_00001",) for scene in SCENES},
        labelled_frames=4,
        queries=4,
        protocol_config=_occam_protocol_config(),
    )


def _write_annotation(path: Path) -> None:
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
                        "category": "red cup",
                        "bbox": [10, 10, 20, 20],
                        "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _score_payload(
    xyz: torch.Tensor,
    *,
    geometry_sha: str,
    queries: tuple[str, ...] = ("red cup",),
) -> dict[str, object]:
    scores = torch.zeros(len(xyz), 3, len(queries), dtype=torch.float16)
    scores[0] = 0.95
    valid = torch.zeros(len(xyz), dtype=torch.bool)
    valid[0] = True
    scale_axis = [
        {"id": "0.2", "value": 0.2, "unit": "meter"},
        {"id": "0.4", "value": 0.4, "unit": "meter"},
        {"id": "0.7", "value": 0.7, "unit": "meter"},
    ]
    field_sha = "1" * 64
    readout_sha = "2" * 64
    xyz_sha = xyz_geometry_fingerprint(xyz)["xyz_sha256"]
    return {
        "version": 2,
        "contract": DIRECT3D_CONTRACT,
        "query_scores": scores,
        "query_ids": list(queries),
        "scale_ids": ["0.2", "0.4", "0.7"],
        "scale_radii_m": [0.2, 0.4, 0.7],
        "xyz": xyz,
        "valid": valid,
        "geometry_fingerprint": xyz_geometry_fingerprint(xyz),
        "field_checkpoint_sha256": field_sha,
        "readout_checkpoint_sha256": readout_sha,
        "renderer_geometry_checkpoint_sha256": geometry_sha,
        "authority": {
            "contract": SHARED_AUTHORITY_CONTRACT,
            "score_semantics": "raw_independent_normalized_cosine",
            "score_formula": (
                "l2_normalize(descriptor) @ l2_normalize(text_embedding).T"
            ),
            "probability_route": "",
            "query_scores_sha256": _tensor_sha256(scores),
            "scale_axis": scale_axis,
            "query_axis": {
                "ids": list(queries),
                "order_sha256": renderer.canonical_json_sha256(list(queries)),
            },
            "geometry_axis": {
                "num_gaussians": len(xyz),
                "xyz_sha256": xyz_sha,
                "renderer_xyz_sha256": xyz_sha,
                "field_checkpoint_sha256": field_sha,
                "readout_checkpoint_sha256": readout_sha,
                "renderer_geometry_checkpoint_sha256": geometry_sha,
            },
            "source_artifacts": {
                "field_checkpoint": {"path": "/field.pt", "sha256": field_sha},
                "readout_checkpoint": {"path": "/readout.pt", "sha256": readout_sha},
                "renderer_geometry_checkpoint": {
                    "path": "/geometry.pt",
                    "sha256": geometry_sha,
                },
            },
            "consumer_contracts": {
                "lerf2d_scalar_map_renderer": {
                    "score_semantics": renderer.SCORE_SEMANTICS,
                    "tensor_layout_before_render": "[primitive_row,scale,query]",
                    "scale_ids": ["0.2", "0.4", "0.7"],
                    "query_text_axis": list(queries),
                }
            },
            "calibration_constraints": {
                "softmax_applied": False,
                "temperature_applied": False,
                "peak_normalization_applied": False,
                "threshold_applied": False,
                "scale_reduction_applied": False,
                "benchmark_images_opened": False,
                "benchmark_annotations_opened": False,
                "benchmark_masks_opened": False,
                "benchmark_metrics_opened": False,
            },
        },
    }


def _write_authority_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, object], torch.Tensor]:
    labels = tmp_path / "labels"
    xyz = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    scenes: dict[str, object] = {}
    for scene in SCENES:
        _write_annotation(labels / scene / "frame_00001.json")
        scene_root = tmp_path / "dataset" / scene
        scene_root.mkdir(parents=True)
        transforms = scene_root / "transforms.json"
        transforms.write_text('{"frames": []}\n', encoding="utf-8")
        config = tmp_path / "inputs" / f"{scene}.yaml"
        checkpoint = tmp_path / "inputs" / f"{scene}.checkpoint"
        ply = tmp_path / "inputs" / f"{scene}.ply"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"scene: {scene}\n", encoding="utf-8")
        checkpoint.write_bytes(f"{scene} geometry\n".encode())
        ply.write_bytes(f"{scene} ply\n".encode())
        cache = tmp_path / "inputs" / f"{scene}.scores.pt"
        torch.save(
            _score_payload(xyz, geometry_sha=_sha(checkpoint)),
            cache,
        )
        scenes[scene] = {
            "scene_root": str(scene_root.resolve()),
            "config": _record(config),
            "geometry_checkpoint": _record(checkpoint),
            "geometry_ply": _record(ply),
            "query_score_cache": _record(cache),
            "camera_sources": {"transforms_json": _record(transforms)},
        }
    authority: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": renderer.AUTHORITY_ARTIFACT_TYPE,
        "method": "RADIO-GS synthetic",
        "canonical_task_id": CANONICAL_TASK_ID,
        "registry_row": EXPECTED_REGISTRY_ROW,
        "protocol_freeze": {
            "freeze_id": "synthetic_freeze_v1",
            "sha256": "f" * 64,
        },
        "label_root": str(labels.resolve()),
        "scenes": scenes,
    }
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority, indent=2), encoding="utf-8")
    return path, labels, authority, xyz


class _FakeModel(torch.nn.Module):
    def __init__(self, xyz: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("xyz", xyz.float())

    def get_xyz(self) -> torch.Tensor:
        return self.xyz


class _FakeScalarRenderer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def render_feature_rows(
        self,
        model: object,
        viewmat: torch.Tensor,
        features: torch.Tensor,
        *,
        feature_height: int,
        feature_width: int,
        alpha_normalize: bool,
    ) -> dict[str, torch.Tensor]:
        self.calls.append(
            {
                "features": features.detach().cpu().clone(),
                "alpha_normalize": alpha_normalize,
                "viewmat": viewmat.detach().cpu().clone(),
            }
        )
        values = features.sum(dim=0)[:, None, None]
        return {
            "feature_map": values.expand(-1, feature_height, feature_width).clone()
        }


def test_render_frame_uses_raw_alpha_composited_channel_order() -> None:
    model = _FakeModel(torch.zeros(2, 3))
    fake = _FakeScalarRenderer()
    scores = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[0.5, 0.25], [1.0, 0.5], [1.5, 0.75]],
        ]
    )
    result = renderer.render_frame_score_maps(
        fake,
        model,
        np.eye(4, dtype=np.float32),
        scores,
        [1, 0],
        height=2,
        width=3,
        device=torch.device("cpu"),
    )
    assert result.shape == (3, 2, 2, 3)
    assert result[:, :, 0, 0].tolist() == [
        pytest.approx([2.25, 1.5]),
        pytest.approx([4.5, 4.0]),
        pytest.approx([6.75, 6.5]),
    ]
    assert len(fake.calls) == 3
    assert all(call["alpha_normalize"] is False for call in fake.calls)


def test_authority_binds_exact_camera_sources_and_order(tmp_path: Path) -> None:
    path, labels, authority, _ = _write_authority_inputs(tmp_path)
    method, source, digest, observed_labels, scenes = renderer.load_render_authority(
        path, expected_sha256=_sha(path), contract=_contract()
    )
    assert method == "RADIO-GS synthetic"
    assert source == path.resolve()
    assert digest == _sha(path)
    assert observed_labels == labels.resolve()
    assert tuple(scenes) == SCENES

    raw_scenes = authority["scenes"]
    assert isinstance(raw_scenes, dict)
    first = raw_scenes["scene_a"]
    assert isinstance(first, dict)
    camera_sources = first["camera_sources"]
    assert isinstance(camera_sources, dict)
    camera_sources["transforms_json"]["sha256"] = "0" * 64
    path.write_text(json.dumps(authority), encoding="utf-8")
    with pytest.raises(renderer.ScalarMapRenderError, match="SHA-256 differs"):
        renderer.load_render_authority(
            path, expected_sha256=_sha(path), contract=_contract()
        )


def test_cache_rejects_calibration_and_invalid_row_drift(tmp_path: Path) -> None:
    path, _labels, authority, xyz = _write_authority_inputs(tmp_path)
    _method, _source, _digest, _root, bindings = renderer.load_render_authority(
        path, expected_sha256=_sha(path), contract=_contract()
    )
    model = _FakeModel(xyz)
    scores, queries, scales = renderer.load_scene_query_scores(
        bindings["scene_a"], model=model, expected_query_ids=("red cup",)
    )
    assert tuple(scores.shape) == (2, 3, 1)
    assert queries == ("red cup",)
    assert [scale["id"] for scale in scales] == ["0.2", "0.4", "0.7"]

    raw_scenes = authority["scenes"]
    assert isinstance(raw_scenes, dict)
    scene = raw_scenes["scene_a"]
    assert isinstance(scene, dict)
    cache_record = scene["query_score_cache"]
    assert isinstance(cache_record, dict)
    cache_path = Path(str(cache_record["path"]))
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    payload["query_scores"][1, 0, 0] = 0.25
    payload["authority"]["query_scores_sha256"] = _tensor_sha256(payload["query_scores"])
    torch.save(payload, cache_path)
    cache_record["sha256"] = _sha(cache_path)
    path.write_text(json.dumps(authority), encoding="utf-8")
    _m, _s, _d, _r, bindings = renderer.load_render_authority(
        path, expected_sha256=_sha(path), contract=_contract()
    )
    with pytest.raises(renderer.ScalarMapRenderError, match="exact zero"):
        renderer.load_scene_query_scores(
            bindings["scene_a"], model=model, expected_query_ids=("red cup",)
        )


def test_cache_authority_accepts_calibration_free_fp32_contract() -> None:
    xyz = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    payload = _score_payload(xyz, geometry_sha="a" * 64)
    payload["authority"]["contract"] = FP32_SHARED_AUTHORITY_CONTRACT

    scales = renderer._validate_cache_authority(
        payload,
        expected_query_ids=("red cup",),
        expected_renderer_geometry_sha256="a" * 64,
    )

    assert [scale["id"] for scale in scales] == ["0.2", "0.4", "0.7"]


def test_full_cpu_synthetic_bundle_is_accepted_by_frozen_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_path, labels, _authority, xyz = _write_authority_inputs(tmp_path)
    contract = _contract()
    fake_renderer = _FakeScalarRenderer()

    monkeypatch.setattr(renderer, "load_frozen_contract", lambda *args, **kwargs: contract)
    monkeypatch.setattr(renderer, "build_mask_renderer", lambda *args, **kwargs: fake_renderer)

    def runtime_loader(binding, *, label_root, device):
        dataset = SimpleNamespace(
            file_paths=["images/frame_00001.jpg"],
            poses_w2c=np.eye(4, dtype=np.float32)[None],
        )
        config = SimpleNamespace(image_height=40, image_width=40)
        return renderer.SceneRuntime(
            model=_FakeModel(xyz), config=config, dataset=dataset
        )

    output = tmp_path / "bundle"
    result = renderer.render_bundle(
        authority_path=authority_path,
        authority_sha256=_sha(authority_path),
        protocol_freeze=tmp_path / "unused-freeze.yaml",
        repo_root=tmp_path,
        output_bundle=output,
        device="cpu",
        runtime_loader=runtime_loader,
    )
    assert result["frames"] == 4
    assert result["queries"] == 4
    assert Path(result["manifest"]).is_file()
    receipt = json.loads((output / "renderer_receipt.json").read_text())
    assert receipt["render_semantics"]["alpha_normalized"] is False
    assert receipt["render_semantics"]["text_encoded"] is False
    assert receipt["render_semantics"]["benchmark_masks_opened"] is False

    evaluated = evaluate_scalar_map_bundle(
        output / "manifest.json", label_root=labels, contract=contract
    )
    assert evaluated["status"] == "complete_exact_frozen_protocol_evaluation"
    assert evaluated["cohort"] == {
        "scenes": list(SCENES),
        "labelled_frames": 4,
        "queries": 4,
    }

    with pytest.raises(FileExistsError, match="already exists"):
        renderer.render_bundle(
            authority_path=authority_path,
            authority_sha256=_sha(authority_path),
            protocol_freeze=tmp_path / "unused-freeze.yaml",
            repo_root=tmp_path,
            output_bundle=output,
            device="cpu",
            runtime_loader=runtime_loader,
        )


def test_public_cli_exposes_no_calibration_or_query_knobs() -> None:
    source = Path("radio_gs/scripts/render_ours_lerf2d_scalar_maps.py").read_text()
    for forbidden in (
        "--temperature",
        "--threshold",
        "--query",
        "--scale-selection",
        "--alpha-normalize",
    ):
        assert forbidden not in source
