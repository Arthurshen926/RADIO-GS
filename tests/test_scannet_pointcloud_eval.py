import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    _apply_logit_calibration,
    _apply_opacity_label_filter,
    _apply_query_opacity_label_filter,
    _decode_gaussian_indices_1280,
    _decode_points_1280,
    _empty_query_diagnostics,
    _finalize_query_diagnostics,
    _fixed_rgb_project_features,
    _add_query_mode_args,
    _load_state_or_raise,
    _parse_scene_list,
    _parse_splits,
    _blend_summary_features,
    _project_compact_with_summary_adapter,
    _read_label_ply,
    _resolve_opacity_filter_mode,
    _resolve_class_aliases,
    _save_language_features_npz,
    _subsample_points,
    _update_query_diagnostics,
)


def _write_label_ply(path: Path) -> None:
    arr = np.empty(
        4,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "i4"),
        ],
    )
    arr["x"] = np.arange(4, dtype=np.float32)
    arr["y"] = 0.0
    arr["z"] = 1.0
    arr["red"] = 255
    arr["green"] = 0
    arr["blue"] = 0
    arr["label"] = np.array([0, 1, 2, 33], dtype=np.int32)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def test_read_label_ply_reads_xyz_and_raw_labels(tmp_path):
    path = tmp_path / "labels.ply"
    _write_label_ply(path)

    xyz, labels = _read_label_ply(path)

    assert xyz.shape == (4, 3)
    assert labels.tolist() == [0, 1, 2, 33]


def test_parse_scene_list_supports_vala_subset_and_dedupes():
    scenes = _parse_scene_list(
        "scene0000_00, scene0062_00;scene0000_00,scene0590_00"
    )

    assert scenes == ["scene0000_00", "scene0062_00", "scene0590_00"]
    assert _parse_scene_list(None) is None


def test_gaussian_index_eval_defaults_to_label_point_positions():
    parser = argparse.ArgumentParser()
    _add_query_mode_args(parser)

    args = parser.parse_args([])

    assert args.gaussian_index_position_mode == "label_point"


def test_apply_opacity_label_filter_marks_low_opacity_labels_invalid():
    labels = np.array([1, 2, 33, 34], dtype=np.int32)
    opacities = torch.tensor([[0.05], [0.10], [0.50], [0.09]], dtype=torch.float32)

    filtered, stats = _apply_opacity_label_filter(
        labels,
        opacities,
        threshold=0.1,
        scene="scene_test",
    )

    assert filtered.tolist() == [0, 2, 33, 0]
    assert labels.tolist() == [1, 2, 33, 34]
    assert stats == {
        "enabled": True,
        "mode": "label_index",
        "threshold": 0.1,
        "num_filtered": 2,
        "num_points": 4,
    }


def test_resolve_opacity_filter_auto_uses_query_top1_for_knn():
    mode = _resolve_opacity_filter_mode(
        "auto",
        query_mode="knn",
        label_count=4,
        gaussian_count=4,
    )

    assert mode == "query_top1"


def test_resolve_opacity_filter_auto_uses_label_index_for_gaussian_index():
    mode = _resolve_opacity_filter_mode(
        "auto",
        query_mode="gaussian_index",
        label_count=4,
        gaussian_count=4,
    )

    assert mode == "label_index"


def test_apply_query_opacity_label_filter_uses_top_neighbor_opacity():
    labels = np.array([1, 2, 33], dtype=np.int32)
    opacities = torch.tensor([[0.2], [0.05], [0.9]], dtype=torch.float32)
    query_aux = {
        "gaussian_indices": torch.tensor([[2, 1], [1, 0], [0, 2]], dtype=torch.long),
        "weights": torch.tensor([[0.6, 0.4], [0.9, 0.1], [0.55, 0.45]]),
    }

    filtered, stats = _apply_query_opacity_label_filter(
        labels,
        query_aux,
        opacities,
        threshold=0.1,
        mode="query_top1",
    )

    assert filtered.tolist() == [1, 0, 33]
    assert stats["num_filtered"] == 1
    assert stats["num_points"] == 3


def test_apply_query_opacity_label_filter_can_use_weighted_neighbor_opacity():
    labels = np.array([1, 2], dtype=np.int32)
    opacities = torch.tensor([[0.2], [0.0], [0.9]], dtype=torch.float32)
    query_aux = {
        "gaussian_indices": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "weights": torch.tensor([[0.75, 0.25], [0.9, 0.1]]),
    }

    filtered, stats = _apply_query_opacity_label_filter(
        labels,
        query_aux,
        opacities,
        threshold=0.1,
        mode="query_weighted",
    )

    assert filtered.tolist() == [1, 0]
    assert stats["num_filtered"] == 1


def test_parse_splits_accepts_protocol_splits_only():
    assert _parse_splits("19,15,10") == ["19", "15", "10"]


def test_subsample_points_is_deterministic():
    xyz = np.arange(30, dtype=np.float32).reshape(10, 3)
    labels = np.arange(10, dtype=np.int32)

    xyz_a, labels_a, idx_a = _subsample_points(xyz, labels, max_points=4, seed=7)
    xyz_b, labels_b, idx_b = _subsample_points(xyz, labels, max_points=4, seed=7)

    assert idx_a.tolist() == idx_b.tolist()
    assert np.array_equal(xyz_a, xyz_b)
    assert np.array_equal(labels_a, labels_b)


def test_load_state_or_raise_rejects_missing_required_keys():
    module = torch.nn.Linear(2, 1)
    bad_state = {"bias": module.bias.detach().clone()}

    with pytest.raises(RuntimeError, match="missing_required"):
        _load_state_or_raise(module, bad_state, "linear")


def test_fixed_rgb_project_features_is_deterministic_uint8():
    features = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    colors_a = _fixed_rgb_project_features(features, seed=11)
    colors_b = _fixed_rgb_project_features(torch.from_numpy(features), seed=11)

    assert colors_a.dtype == np.uint8
    assert colors_a.shape == (4, 3)
    assert np.array_equal(colors_a, colors_b)
    assert np.unique(colors_a, axis=0).shape[0] > 1


def test_query_mode_parser_accepts_nearest():
    parser = argparse.ArgumentParser()
    _add_query_mode_args(parser)

    args = parser.parse_args(["--query_mode", "nearest"])

    assert args.query_mode == "nearest"


def test_query_mode_parser_accepts_scene_mean_logit_calibration():
    parser = argparse.ArgumentParser()
    _add_query_mode_args(parser)

    args = parser.parse_args(["--logit_calibration", "scene_mean"])

    assert args.logit_calibration == "scene_mean"


def test_query_mode_parser_accepts_logit_calibration_alpha():
    parser = argparse.ArgumentParser()
    _add_query_mode_args(parser)

    args = parser.parse_args(["--logit_calibration_alpha", "0.5"])

    assert args.logit_calibration_alpha == 0.5


def test_query_mode_parser_accepts_scannet_class_aliases():
    parser = argparse.ArgumentParser()
    _add_query_mode_args(parser)

    args = parser.parse_args(["--class_aliases", "scannet"])

    assert args.class_aliases == "scannet"


def test_apply_logit_calibration_subtracts_class_bias_without_mutating():
    logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 4.0, 2.0]])
    bias = torch.tensor([1.0, 0.5, -1.0])

    calibrated = _apply_logit_calibration(logits, bias)

    assert torch.allclose(
        calibrated,
        torch.tensor([[2.0, 1.5, 2.0], [0.0, 3.5, 3.0]]),
    )
    assert torch.allclose(logits, torch.tensor([[3.0, 2.0, 1.0], [1.0, 4.0, 2.0]]))


def test_apply_logit_calibration_can_scale_class_bias():
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    bias = torch.tensor([1.0, 0.5, -1.0])

    calibrated = _apply_logit_calibration(logits, bias, alpha=0.5)

    assert torch.allclose(calibrated, torch.tensor([[2.5, 1.75, 1.5]]))


def test_resolve_class_aliases_can_expand_scannet_names():
    aliases = _resolve_class_aliases(["wall", "chair", "unknown class"], "scannet")

    assert aliases[0][0] == "wall"
    assert "indoor wall" in aliases[0]
    assert "chair" in aliases[1]
    assert aliases[2] == ["unknown class"]


def test_query_diagnostics_tracks_weight_and_distance_stats():
    diagnostics = _empty_query_diagnostics()
    aux = {
        "weights": torch.tensor([[0.75, 0.25], [1.0, 0.0]], dtype=torch.float32),
        "euclidean_dist": torch.tensor([[0.2, 0.8], [0.1, 0.4]], dtype=torch.float32),
        "mahalanobis_dist2": torch.tensor([[0.3, 1.2], [0.05, 0.9]], dtype=torch.float32),
        "density": torch.tensor([[0.9, 0.4], [1.0, 0.2]], dtype=torch.float32),
    }

    _update_query_diagnostics(diagnostics, aux)
    result = _finalize_query_diagnostics(diagnostics)

    assert result["num_points"] == 2
    assert result["mean_top1_weight"] == pytest.approx(0.875)
    assert result["mean_top1_euclidean_dist"] == pytest.approx(0.15)
    assert result["mean_top1_mahalanobis_dist2"] == pytest.approx(0.175)
    assert result["mean_top1_density"] == pytest.approx(0.95)
    assert result["mean_effective_neighbors"] > 1.0


def test_decode_points_can_select_semantic_compact_branch():
    class FakeModel:
        def query_compact_points(self, points_xyz, k, return_aux=False):
            assert return_aux
            return {
                "features": torch.full((points_xyz.shape[0], 2), 1.0),
                "semantic": torch.full((points_xyz.shape[0], 2), 3.0),
                "weights": torch.ones(points_xyz.shape[0], 1),
            }

    class FakeCodec:
        def decode(self, compact_map):
            return compact_map

    points = torch.zeros(4, 3)

    decoded, aux = _decode_points_1280(
        FakeModel(),
        FakeCodec(),
        points,
        k=8,
        return_aux=True,
        compact_feature_key="semantic",
    )

    assert torch.all(decoded == 3.0)
    assert "semantic" in aux


def test_decode_points_can_forward_candidate_k_to_direct_query():
    captured = {}

    class FakeModel:
        def query_compact_points(self, points_xyz, k, candidate_k=None, return_aux=False):
            captured["k"] = k
            captured["candidate_k"] = candidate_k
            return torch.full((points_xyz.shape[0], 2), 2.0)

    class FakeCodec:
        def decode(self, compact_map):
            return compact_map

    decoded = _decode_points_1280(
        FakeModel(),
        FakeCodec(),
        torch.zeros(3, 3),
        k=8,
        candidate_k=32,
    )

    assert captured == {"k": 8, "candidate_k": 32}
    assert torch.all(decoded == 2.0)


def test_decode_points_prefers_pointwise_codec_api():
    class FakeModel:
        def query_compact_points(self, points_xyz, k, return_aux=False):
            return torch.full((points_xyz.shape[0], 2), 2.0)

    class FakeCodec:
        def decode_points(self, compact):
            return compact + 10.0

        def decode(self, compact_map):
            raise AssertionError("chunk-coupled map decode should not be used")

    decoded = _decode_points_1280(
        FakeModel(),
        FakeCodec(),
        torch.zeros(3, 3),
        k=8,
    )

    assert decoded.shape == (3, 2)
    assert torch.all(decoded == 12.0)


def test_decode_gaussian_indices_can_forward_label_point_positions():
    captured = {}

    class FakeModel:
        def query_gaussian_points(self, gaussian_indices, return_aux=False, points_xyz=None):
            captured["indices"] = gaussian_indices.clone()
            captured["points_xyz"] = points_xyz.clone()
            return {
                "features": torch.full((gaussian_indices.shape[0], 2), 4.0),
                "gaussian_indices": gaussian_indices,
            }

    class FakeCodec:
        def decode_points(self, compact):
            return compact + 1.0

    indices = torch.tensor([2, 4])
    points = torch.tensor([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=torch.float32)
    decoded, aux = _decode_gaussian_indices_1280(
        FakeModel(),
        FakeCodec(),
        indices,
        points_xyz=points,
        return_aux=True,
    )

    assert torch.equal(captured["indices"], indices)
    assert torch.equal(captured["points_xyz"], points)
    assert torch.all(decoded == 5.0)
    assert aux["features"].shape == (2, 2)


def test_project_compact_with_summary_adapter_returns_normalized_text_features():
    adapter = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        adapter.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )

    visual = _project_compact_with_summary_adapter(
        torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        adapter,
    )

    assert visual.shape == (1, 3)
    assert torch.allclose(visual.norm(dim=-1), torch.ones(1), atol=1e-6)


def test_blend_summary_features_interpolates_and_normalizes():
    base = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    adapter = torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float32)

    assert torch.allclose(
        _blend_summary_features(base, adapter, alpha=0.0),
        F.normalize(base, dim=-1),
    )
    assert torch.allclose(
        _blend_summary_features(base, adapter, alpha=1.0),
        F.normalize(adapter, dim=-1),
    )
    blended = _blend_summary_features(base, adapter, alpha=0.25)

    expected = F.normalize(0.75 * F.normalize(base, dim=-1) + 0.25 * F.normalize(adapter, dim=-1), dim=-1)
    assert torch.allclose(blended, expected)
    assert torch.allclose(blended.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_save_language_features_npz_writes_normalized_vertex_aligned_features(tmp_path):
    path = tmp_path / "language_features.npz"
    xyz = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    labels = np.array([1, 2], dtype=np.int32)
    features = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32)

    _save_language_features_npz(path, xyz, labels, features)

    data = np.load(path)
    assert data["xyz"].shape == (2, 3)
    assert data["labels"].tolist() == [1, 2]
    assert data["features"].dtype == np.float16
    norms = np.linalg.norm(data["features"].astype(np.float32), axis=1)
    assert np.allclose(norms, np.ones(2), atol=1e-3)
