from types import SimpleNamespace

import numpy as np
import pytest
import torch

from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts import eval_scannet_pointcloud_radio_teacher as teacher_module
from radio_gs.scripts.eval_scannet_pointcloud_radio_teacher import (
    _accumulate_multiview_targets,
    _compute_teacher_split_metrics,
    _raw_ids_from_pred_indices,
    _save_teacher_language_features_npz,
    _save_teacher_feature_cache,
)


def _write_tiny_feature(path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.full((1, 2, 2), float(value), dtype=torch.float32), path)


def test_accumulate_multiview_targets_is_equivalent_across_view_chunks(tmp_path):
    feature_dir = tmp_path / "features" / "backbone"
    feature_paths = [feature_dir / "rgb_0.pt", feature_dir / "rgb_1.pt"]
    _write_tiny_feature(feature_paths[0], 2.0)
    _write_tiny_feature(feature_paths[1], 6.0)

    poses_c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    pose_file = tmp_path / "traj_w_c.txt"
    np.savetxt(pose_file, poses_c2w.reshape(-1, 4))
    config = SimpleNamespace(
        feature_dir=str(tmp_path / "features"),
        pose_file=str(pose_file),
        image_height=2,
        image_width=2,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [10.0, 10.0, 1.0],
        ],
        dtype=torch.float32,
    )

    targets_one, valid_one, counts_one = _accumulate_multiview_targets(
        config,
        feature_paths,
        points,
        device=torch.device("cpu"),
        split="all",
        view_chunk_size=1,
    )
    targets_all, valid_all, counts_all = _accumulate_multiview_targets(
        config,
        feature_paths,
        points,
        device=torch.device("cpu"),
        split="all",
        view_chunk_size=2,
    )

    assert torch.allclose(targets_one, targets_all)
    assert torch.equal(valid_one, valid_all)
    assert torch.equal(counts_one, counts_all)
    assert targets_one.tolist() == [[4.0], [0.0]]
    assert valid_one.tolist() == [True, False]
    assert counts_one.tolist() == [2, 0]


def test_accumulate_multiview_targets_can_normalize_before_average(monkeypatch):
    feature_paths = [SimpleNamespace(name="view0"), SimpleNamespace(name="view1")]
    config = SimpleNamespace()
    points = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    def fake_load_teacher_batch(config, chunk_paths, device, split):
        view_index = 0 if chunk_paths[0].name == "view0" else 1
        return torch.tensor([view_index], device=device), torch.eye(4).unsqueeze(0), torch.eye(3)

    def fake_sample_multiview_radio_targets(
        points,
        feature_batch,
        poses_w2c,
        K,
        *,
        normalize_sampled_features=False,
    ):
        assert normalize_sampled_features is True
        if int(feature_batch.item()) == 0:
            targets = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        else:
            targets = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        return targets, torch.tensor([True]), torch.tensor([1], dtype=torch.long)

    monkeypatch.setattr(teacher_module, "_load_teacher_batch", fake_load_teacher_batch)
    monkeypatch.setattr(
        teacher_module,
        "sample_multiview_radio_targets",
        fake_sample_multiview_radio_targets,
    )

    targets, valid, counts = _accumulate_multiview_targets(
        config,
        feature_paths,
        points,
        device=torch.device("cpu"),
        split="all",
        view_chunk_size=1,
        normalize_features=True,
    )

    assert torch.allclose(targets, torch.tensor([[0.5, 0.5]]))
    assert valid.tolist() == [True]
    assert counts.tolist() == [2]


def test_compute_teacher_split_metrics_ignores_teacher_invalid_points():
    split_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS["10"]
    labels = np.array([1, 2, 4], dtype=np.int32)
    pred = np.array([1, 2, 1], dtype=np.int32)
    teacher_valid = np.array([True, False, True])

    metrics = _compute_teacher_split_metrics(pred, labels, teacher_valid, split_ids)

    assert metrics["num_valid"] == 2
    assert metrics["teacher_valid_points"] == 2
    assert metrics["teacher_valid_ratio"] == pytest.approx(2.0 / 3.0)
    assert metrics["per_class"]["2"]["gt_count"] == 0


def test_raw_ids_from_pred_indices_maps_to_nyu40_ids():
    split_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS["10"]
    pred_idx = np.array([0, 1, 9], dtype=np.int64)

    raw_ids = _raw_ids_from_pred_indices(pred_idx, split_ids)

    assert raw_ids.tolist() == [1, 2, 33]


def test_save_teacher_feature_cache_round_trips_training_payload(tmp_path):
    cache_path = tmp_path / "scene0000_00_teacher_features.pt"
    xyz = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    labels = np.array([1, 2], dtype=np.int32)
    features = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.25]], dtype=torch.float32)
    valid = torch.tensor([True, False])
    view_counts = torch.tensor([3, 0], dtype=torch.long)

    _save_teacher_feature_cache(
        cache_path,
        xyz=xyz,
        labels=labels,
        features=features,
        valid=valid,
        view_counts=view_counts,
        sample_indices=np.array([4, 7], dtype=np.int64),
        metadata={"scene": "scene0000_00", "teacher_split": "val"},
    )

    payload = torch.load(cache_path, map_location="cpu")
    assert payload["metadata"]["scene"] == "scene0000_00"
    assert payload["metadata"]["teacher_split"] == "val"
    assert torch.allclose(payload["xyz"], torch.from_numpy(xyz))
    assert torch.equal(payload["labels"], torch.tensor([1, 2], dtype=torch.long))
    assert torch.allclose(payload["features"], features.half())
    assert torch.equal(payload["valid"], valid)
    assert torch.equal(payload["view_counts"], view_counts)
    assert torch.equal(payload["sample_indices"], torch.tensor([4, 7], dtype=torch.long))


def test_save_teacher_language_features_npz_masks_invalid_points(tmp_path):
    path = tmp_path / "teacher_language_features.npz"
    xyz = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    labels = np.array([1, 2], dtype=np.int32)
    features = torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=torch.float32)
    valid = torch.tensor([True, False])

    _save_teacher_language_features_npz(
        path,
        xyz=xyz,
        labels=labels,
        features=features,
        valid=valid,
    )

    data = np.load(path)
    assert data["xyz"].shape == (2, 3)
    assert data["labels"].tolist() == [1, 2]
    assert data["valid"].tolist() == [True, False]
    assert np.allclose(
        data["features"][0].astype(np.float32),
        np.array([0.6, 0.8]),
        atol=1e-3,
    )
    assert np.allclose(data["features"][1].astype(np.float32), np.zeros(2))
