import torch
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData, PlyElement
from types import MethodType, SimpleNamespace

from radio_gs.scripts import train_feature_field as train_feature_field_module
from radio_gs.scripts.train_feature_field import (
    RadioGSTrainer,
    read_ply_xyz,
    resolve_scannet_label_ply,
    sample_multiview_radio_targets,
    select_visible_gaussian_indices,
)
from radio_gs.training.feature_supervision_mixin import (
    _direct_point_view_count_weights,
    _direct_point_weight_mask,
)


def test_sample_multiview_radio_targets_averages_valid_projected_features():
    features = torch.stack(
        [
            torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
            torch.arange(100, 109, dtype=torch.float32).reshape(1, 3, 3),
        ],
        dim=0,
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    k = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    targets, valid, counts = sample_multiview_radio_targets(points, features, poses, k)

    assert valid.tolist() == [True, True, False]
    assert counts.tolist() == [2, 2, 0]
    assert torch.allclose(targets[:2, 0], torch.tensor([54.0, 55.0]))


def test_sample_multiview_radio_targets_can_normalize_each_view_before_averaging():
    features = torch.tensor(
        [
            [[[3.0]], [[4.0]]],
            [[[0.0]], [[2.0]]],
        ],
        dtype=torch.float32,
    )
    points = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    k = torch.eye(3, dtype=torch.float32)

    targets, valid, counts = sample_multiview_radio_targets(
        points,
        features,
        poses,
        k,
        normalize_sampled_features=True,
    )

    assert valid.tolist() == [True]
    assert counts.tolist() == [2]
    assert torch.allclose(targets[0], torch.tensor([0.3, 0.9]), atol=1e-6)


def test_sample_multiview_radio_targets_filters_depth_inconsistent_views():
    features = torch.stack(
        [
            torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
            torch.arange(100, 109, dtype=torch.float32).reshape(1, 3, 3),
        ],
        dim=0,
    )
    points = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    k = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    depth = torch.ones(2, 3, 3)
    depth[1] = 2.0

    targets, valid, counts = sample_multiview_radio_targets(
        points,
        features,
        poses,
        k,
        depth_map=depth,
        depth_tolerance=0.05,
        relative_depth_tolerance=0.0,
    )

    assert valid.tolist() == [True]
    assert counts.tolist() == [1]
    assert torch.allclose(targets[:, 0], torch.tensor([4.0]))


def test_select_visible_gaussian_indices_prefers_current_batch_frustum():
    points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    poses = torch.eye(4).unsqueeze(0)
    k = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    indices, visible = select_visible_gaussian_indices(
        points,
        poses,
        k,
        image_height=3,
        image_width=3,
        sample_count=4,
    )

    assert indices.tolist() == [0, 1]
    assert visible.tolist() == [True, True, False, False]


def test_direct_point_view_count_weights_are_normalized_and_monotonic():
    weights = _direct_point_view_count_weights(
        torch.tensor([0.0, 1.0, 10.0, 100.0]),
        mode="log",
        min_weight=0.1,
    )

    assert weights is not None
    assert weights[0].item() == 0.0
    assert torch.allclose(weights[1:].mean(), torch.tensor(1.0), atol=1e-6)
    assert weights[1] < weights[2] < weights[3]


def test_direct_point_weight_mask_matches_pointwise_and_legacy_shapes():
    weights = torch.tensor([1.0, 2.0, 3.0])

    pointwise = _direct_point_weight_mask(torch.zeros(3, 4, 1, 1), weights)
    legacy = _direct_point_weight_mask(torch.zeros(1, 4, 3, 1), weights)

    assert pointwise.shape == (3, 1, 1, 1)
    assert legacy.shape == (1, 1, 3, 1)


def test_read_ply_xyz_and_resolve_scannet_label_ply(tmp_path):
    scene = "scene0000_00"
    scene_root = tmp_path / scene
    scene_root.mkdir()
    label_ply = scene_root / f"{scene}_vh_clean_2.labels.ply"
    vertices = np.array(
        [(1.0, 2.0, 3.0, 4), (5.0, 6.0, 7.0, 8)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("label", "i4")],
    )
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(label_ply))

    assert resolve_scannet_label_ply(scene_root, scene) == label_ply
    xyz = read_ply_xyz(label_ply)

    assert xyz.shape == (2, 3)
    assert torch.allclose(xyz, torch.tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]]))


def test_direct_point_loss_can_align_summary_space(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 1.0
    trainer.direct_point_sample_count = 2
    trainer.direct_point_query_mode = "knn"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=2,
    )
    trainer.siglip_summary_head = object()
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    class _Model:
        def query_compact_points(self, points, k):
            return torch.zeros(points.shape[0], 2)

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def _decode(self, compact_map):
            n = compact_map.shape[2]
            decoded = torch.zeros(1, 2, n, 1)
            decoded[:, 0] = 1.0
            return decoded

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    def project_summary(self, features):
        return F.normalize(features.float(), dim=1)

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.distill_loss_fn = _ZeroDistill()
    trainer._project_summary_head_features = MethodType(project_summary, trainer)

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 2),
    )

    assert torch.allclose(stats["summary"], torch.tensor(0.5))
    assert torch.allclose(stats["loss"], torch.tensor(0.5))


def test_direct_point_loss_can_use_cached_teacher_features_without_view_sampling(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 0.0
    trainer.direct_point_text_loss_weight = 0.0
    trainer.direct_point_adapter_text_loss_weight = 0.0
    trainer.direct_point_text_distill_weight = 0.0
    trainer.direct_point_adapter_text_distill_weight = 0.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 2
    trainer.direct_point_sample_strategy = "uniform"
    trainer.direct_point_query_mode = "gaussian_index"
    trainer.direct_point_gaussian_position_mode = "gaussian_center"
    trainer.direct_point_k = 1
    trainer.direct_point_candidate_k = 0
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = None
    trainer.direct_point_teacher_features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]],
        dtype=torch.float32,
    )
    trainer.direct_point_teacher_valid = torch.tensor([True, True, False])
    trainer.direct_point_teacher_view_counts = torch.tensor([3, 2, 0], dtype=torch.long)
    trainer.direct_point_text_split_ids = []
    trainer.direct_point_text_embeddings = None
    trainer._is_hybrid = True
    trainer.siglip_summary_head = None
    trainer.point_summary_adapter = None

    class _Model:
        num_gaussians = 3

        def __init__(self):
            self.seen_indices = None

        def query_gaussian_points(self, indices):
            self.seen_indices = indices.clone()
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    class _Codec:
        def decode_points(self, compact):
            return compact.float()

    class _MSEDistill:
        def __call__(self, decoded, target):
            return {"total": F.mse_loss(decoded, target)}

    def fail_if_view_sampling(*args, **kwargs):
        raise AssertionError("cached teacher point features should bypass per-view sampling")

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fail_if_view_sampling,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fail_if_view_sampling,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.distill_loss_fn = _MSEDistill()

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=None,
    )

    assert trainer.model.seen_indices.tolist() == [0, 1]
    assert torch.allclose(stats["valid_ratio"], torch.tensor(1.0))
    assert torch.allclose(stats["loss"], torch.tensor(0.0))


def test_direct_point_loss_can_train_selected_compact_branch(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_feature_key = "semantic"
    trainer.direct_point_sample_count = 2
    trainer.direct_point_query_mode = "knn"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=2,
    )
    trainer.siglip_summary_head = None
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    class _Model:
        def query_compact_points(self, points, k, return_aux=False):
            assert return_aux
            return {
                "features": torch.zeros(points.shape[0], 2),
                "semantic": torch.ones(points.shape[0], 2),
            }

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def _decode(self, compact_map):
            return compact_map

    class _SumDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum()}

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.zeros(points.shape[0], 2)
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.distill_loss_fn = _SumDistill()

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 2),
    )

    assert torch.allclose(stats["loss"], torch.tensor(4.0))


def test_direct_point_loss_can_class_balance_visible_label_points(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 0.0
    trainer.direct_point_text_loss_weight = 0.0
    trainer.direct_point_adapter_text_loss_weight = 0.0
    trainer.direct_point_text_distill_weight = 0.0
    trainer.direct_point_adapter_text_distill_weight = 0.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 3
    trainer.direct_point_sample_strategy = "class_balanced"
    trainer.direct_point_query_mode = "knn"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_text_split_ids = [1, 2, 3]
    trainer.direct_point_pool = torch.tensor(
        [[float(i), 0.0, 1.0] for i in range(6)],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.long)
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=2,
    )
    trainer.siglip_summary_head = None
    trainer.point_summary_adapter = None
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    captured = {}

    class _Model:
        def query_compact_points(self, points, k):
            captured["x"] = points[:, 0].long().clone()
            return torch.zeros(points.shape[0], 2)

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def _decode(self, compact_map):
            return compact_map

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.zeros(points.shape[0], 2)
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.distill_loss_fn = _ZeroDistill()

    trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 2),
    )

    sampled_labels = trainer.direct_point_pool_labels[captured["x"]]
    assert sorted(sampled_labels.tolist()) == [1, 2, 3]


def test_direct_point_teacher_balanced_sampling_uses_teacher_pseudo_labels(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_sample_strategy = "teacher_balanced"
    trainer.direct_point_teacher_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = torch.ones(6, dtype=torch.long)
    trainer.direct_point_text_embeddings = torch.eye(3, dtype=torch.float32)
    trainer.direct_point_text_split_ids = [1, 2, 3]
    trainer.direct_point_text_pseudo_ce_banks = []
    trainer.siglip_summary_head = object()
    trainer._project_summary_head_features = MethodType(
        lambda self, features: F.normalize(features.float(), dim=1),
        trainer,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )

    sampled = trainer._subsample_direct_point_indices(torch.arange(6), sample_count=3)

    assert sampled.tolist() == [0, 3, 5]


def test_direct_point_teacher_balanced_sampling_chunks_teacher_projection(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_sample_strategy = "teacher_balanced"
    trainer.direct_point_teacher_pseudo_label_chunk_size = 2
    trainer.direct_point_teacher_features = torch.eye(6, 3, dtype=torch.float32)
    trainer.direct_point_text_embeddings = torch.eye(3, dtype=torch.float32)
    trainer.direct_point_text_split_ids = [1, 2, 3]
    trainer.direct_point_text_pseudo_ce_banks = []
    trainer.siglip_summary_head = object()
    calls = []

    def project_summary(self, features):
        assert features.shape[0] <= 2
        calls.append(features.shape[0])
        return F.normalize(features.float(), dim=1)

    trainer._project_summary_head_features = MethodType(project_summary, trainer)
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )

    trainer._subsample_direct_point_indices(torch.arange(6), sample_count=3)

    assert calls == [2, 2, 2]


def test_direct_point_loss_can_train_compact_to_summary_adapter(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 2.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 2
    trainer.direct_point_query_mode = "knn"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=2,
    )
    trainer.siglip_summary_head = object()
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    class _Model:
        def query_compact_points(self, points, k):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def _decode(self, compact_map):
            return compact_map

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    class _Adapter(torch.nn.Module):
        def forward(self, compact):
            return torch.stack(
                [
                    compact[:, 1],
                    compact[:, 0],
                ],
                dim=1,
            )

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    def project_summary(self, features):
        return F.normalize(features.float(), dim=1)

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.point_summary_adapter = _Adapter()
    trainer.distill_loss_fn = _ZeroDistill()
    trainer._project_summary_head_features = MethodType(project_summary, trainer)

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 2),
    )

    assert torch.allclose(stats["summary_adapter"], torch.tensor(1.0))
    assert torch.allclose(stats["loss"], torch.tensor(2.0))


def test_direct_point_loss_can_apply_text_ce_from_label_ply_labels(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 0.0
    trainer.direct_point_text_loss_weight = 1.0
    trainer.direct_point_adapter_text_loss_weight = 0.0
    trainer.direct_point_text_temperature = 1.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 3
    trainer.direct_point_query_mode = "knn"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = torch.tensor([1, 2, 99], dtype=torch.long)
    trainer.direct_point_text_split_ids = [1, 2]
    trainer.direct_point_text_embeddings = F.normalize(torch.eye(2), dim=-1)
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=3,
    )
    trainer.siglip_summary_head = object()
    trainer.point_summary_adapter = None
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    class _Model:
        def query_compact_points(self, points, k):
            return torch.zeros(points.shape[0], 2)

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def _decode(self, compact_map):
            decoded = torch.zeros(1, 2, compact_map.shape[2], 1)
            decoded[0, 0, 0, 0] = 1.0
            decoded[0, 1, 1, 0] = 1.0
            decoded[0, 0, 2, 0] = 1.0
            return decoded

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.zeros(points.shape[0], 2)
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    def project_summary(self, features):
        return F.normalize(features.float(), dim=1)

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.distill_loss_fn = _ZeroDistill()
    trainer._project_summary_head_features = MethodType(project_summary, trainer)

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 3),
    )

    expected = F.cross_entropy(torch.eye(2), torch.tensor([0, 1]))
    assert stats.get("text") is not None
    assert torch.allclose(stats["text"], expected)
    assert torch.allclose(stats["text_valid_ratio"], torch.tensor(2.0 / 3.0))
    assert torch.allclose(stats["text_acc"], torch.tensor(1.0))
    assert torch.allclose(stats["loss"], expected)


def test_direct_point_loss_can_use_row_aligned_label_ply_with_gaussian_index(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 0.0
    trainer.direct_point_text_loss_weight = 1.0
    trainer.direct_point_adapter_text_loss_weight = 0.0
    trainer.direct_point_text_temperature = 1.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 3
    trainer.direct_point_query_mode = "gaussian_index"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = torch.tensor([1, 2, 99], dtype=torch.long)
    trainer.direct_point_text_split_ids = [1, 2]
    trainer.direct_point_text_embeddings = F.normalize(torch.eye(2), dim=-1)
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=3,
    )
    trainer.siglip_summary_head = object()
    trainer.point_summary_adapter = None
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    class _Model:
        num_gaussians = 3

        def query_gaussian_points(self, indices):
            compact = torch.zeros(indices.shape[0], 2)
            compact[indices == 0, 0] = 1.0
            compact[indices == 1, 1] = 1.0
            compact[indices == 2, 0] = 1.0
            return compact

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def decode_points(self, compact):
            return compact

        def _decode(self, compact_map):
            raise AssertionError("pointwise decode should be used")

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.zeros(points.shape[0], 2)
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    def project_summary(self, features):
        return F.normalize(features.float(), dim=1)

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.distill_loss_fn = _ZeroDistill()
    trainer._project_summary_head_features = MethodType(project_summary, trainer)

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 3),
    )

    expected = F.cross_entropy(torch.eye(2), torch.tensor([0, 1]))
    assert torch.allclose(stats["text"], expected)
    assert torch.allclose(stats["text_valid_ratio"], torch.tensor(2.0 / 3.0))
    assert torch.allclose(stats["text_acc"], torch.tensor(1.0))
    assert torch.allclose(stats["loss"], expected)


def test_direct_point_gaussian_index_can_decode_at_label_point_positions(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 0.0
    trainer.direct_point_text_loss_weight = 0.0
    trainer.direct_point_adapter_text_loss_weight = 0.0
    trainer.direct_point_text_distill_weight = 0.0
    trainer.direct_point_adapter_text_distill_weight = 0.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 2
    trainer.direct_point_query_mode = "gaussian_index"
    trainer.direct_point_gaussian_position_mode = "label_point"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = None
    trainer.direct_point_text_split_ids = []
    trainer.direct_point_text_embeddings = None
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=3,
    )
    trainer.siglip_summary_head = None
    trainer.point_summary_adapter = None
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )
    captured = {}

    class _Model:
        num_gaussians = 2

        def query_gaussian_points(self, indices, points_xyz=None):
            captured["indices"] = indices.clone()
            captured["points_xyz"] = None if points_xyz is None else points_xyz.clone()
            return torch.zeros(indices.shape[0], 2)

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def decode_points(self, compact):
            return compact

        def _decode(self, compact_map):
            raise AssertionError("pointwise decode should be used")

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.zeros(points.shape[0], 2)
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.distill_loss_fn = _ZeroDistill()

    trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 3),
    )

    assert torch.equal(captured["indices"], torch.tensor([0, 1]))
    assert torch.equal(captured["points_xyz"], trainer.direct_point_pool)


def test_direct_point_text_ce_can_inverse_batch_weight_classes():
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_text_temperature = 1.0
    trainer.direct_point_text_split_ids = [1, 2]
    trainer.direct_point_text_embeddings = F.normalize(torch.eye(2), dim=-1)
    trainer.direct_point_text_ce_weighting = "inverse_batch"
    point_summary = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    point_labels = torch.tensor([1, 1, 2], dtype=torch.long)

    loss, valid_ratio, acc = trainer._compute_direct_point_text_ce(
        point_summary,
        point_labels,
    )

    logits = point_summary
    targets = torch.tensor([0, 0, 1], dtype=torch.long)
    weights = torch.tensor([0.75, 1.5], dtype=torch.float32)
    expected = F.cross_entropy(logits, targets, weight=weights)
    assert torch.allclose(loss, expected)
    assert torch.allclose(valid_ratio, torch.tensor(1.0))
    assert torch.allclose(acc, torch.tensor(2.0 / 3.0))


def test_direct_point_text_ce_can_use_sqrt_inverse_pool_capped_weights():
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_text_temperature = 1.0
    trainer.direct_point_text_split_ids = [1, 2]
    trainer.direct_point_text_embeddings = F.normalize(torch.eye(2), dim=-1)
    trainer.direct_point_text_ce_weighting = "sqrt_inverse_pool_capped"
    trainer.direct_point_text_ce_min_weight = 0.8
    trainer.direct_point_text_ce_max_weight = 2.0
    trainer.direct_point_pool_labels = torch.tensor([1] + [2] * 9, dtype=torch.long)
    point_summary = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    point_labels = torch.tensor([1, 1, 2], dtype=torch.long)

    loss, valid_ratio, acc = trainer._compute_direct_point_text_ce(
        point_summary,
        point_labels,
    )

    logits = point_summary
    targets = torch.tensor([0, 0, 1], dtype=torch.long)
    weights = torch.tensor([2.0, 0.8], dtype=torch.float32)
    expected = F.cross_entropy(logits, targets, weight=weights)
    assert torch.allclose(loss, expected)
    assert torch.allclose(valid_ratio, torch.tensor(1.0))
    assert torch.allclose(acc, torch.tensor(2.0 / 3.0))


def test_direct_point_loss_can_distill_teacher_text_logits_without_labels(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 0.0
    trainer.direct_point_text_loss_weight = 0.0
    trainer.direct_point_adapter_text_loss_weight = 0.0
    trainer.direct_point_text_distill_weight = 1.0
    trainer.direct_point_text_distill_temperature = 1.0
    trainer.direct_point_text_distill_confidence_threshold = 0.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 2
    trainer.direct_point_query_mode = "knn"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = None
    trainer.direct_point_text_split_ids = [1, 2]
    trainer.direct_point_text_embeddings = F.normalize(torch.eye(2), dim=-1)
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=2,
    )
    trainer.siglip_summary_head = object()
    trainer.point_summary_adapter = None
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    class _Model:
        def query_compact_points(self, points, k):
            return torch.zeros(points.shape[0], 2)

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def _decode(self, compact_map):
            decoded = torch.zeros(1, 2, compact_map.shape[2], 1)
            decoded[0, 0, 0, 0] = 1.0
            decoded[0, 1, 1, 0] = 1.0
            return decoded

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=torch.float32,
        )
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    def project_summary(self, features):
        return F.normalize(features.float(), dim=1)

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.distill_loss_fn = _ZeroDistill()
    trainer._project_summary_head_features = MethodType(project_summary, trainer)

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 2),
    )

    student_logits = torch.eye(2)
    teacher_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    )
    assert stats.get("text_distill") is not None
    assert torch.allclose(stats["text_distill"], expected)
    assert torch.allclose(stats["text_distill_valid_ratio"], torch.tensor(1.0))
    assert torch.allclose(stats["text_distill_teacher_conf"], torch.tensor(0.7310586))
    assert torch.allclose(stats["text_distill_agreement"], torch.tensor(0.0))
    assert torch.allclose(stats["loss"], expected)


def test_direct_point_loss_can_distill_adapter_text_logits_without_labels(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 0.0
    trainer.direct_point_text_loss_weight = 0.0
    trainer.direct_point_adapter_text_loss_weight = 0.0
    trainer.direct_point_text_distill_weight = 0.0
    trainer.direct_point_adapter_text_distill_weight = 1.0
    trainer.direct_point_text_distill_temperature = 1.0
    trainer.direct_point_text_distill_confidence_threshold = 0.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 2
    trainer.direct_point_query_mode = "knn"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = None
    trainer.direct_point_text_split_ids = [1, 2]
    trainer.direct_point_text_embeddings = F.normalize(torch.eye(2), dim=-1)
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=2,
    )
    trainer.siglip_summary_head = object()
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    class _Model:
        def query_compact_points(self, points, k):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def _decode(self, compact_map):
            return compact_map

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    class _Adapter(torch.nn.Module):
        def forward(self, compact):
            return compact

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=torch.float32,
        )
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    def project_summary(self, features):
        return F.normalize(features.float(), dim=1)

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.point_summary_adapter = _Adapter()
    trainer.distill_loss_fn = _ZeroDistill()
    trainer._project_summary_head_features = MethodType(project_summary, trainer)

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 2),
    )

    student_logits = torch.eye(2)
    teacher_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    )
    assert torch.allclose(stats["adapter_text_distill"], expected)
    assert torch.allclose(stats["adapter_text_distill_valid_ratio"], torch.tensor(1.0))
    assert torch.allclose(stats["adapter_text_distill_agreement"], torch.tensor(0.0))
    assert torch.allclose(stats["loss"], expected)


def test_direct_point_loss_can_train_adapter_with_teacher_pseudo_ce(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 0.0
    trainer.direct_point_text_loss_weight = 0.0
    trainer.direct_point_adapter_text_loss_weight = 0.0
    trainer.direct_point_text_distill_weight = 0.0
    trainer.direct_point_adapter_text_distill_weight = 0.0
    trainer.direct_point_adapter_text_pseudo_ce_weight = 1.0
    trainer.direct_point_adapter_text_pseudo_ce_confidence_threshold = 0.0
    trainer.direct_point_adapter_text_pseudo_ce_logit_scale = 1.0
    trainer.direct_point_adapter_decoder_anchor_weight = 0.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 2
    trainer.direct_point_query_mode = "knn"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = None
    trainer.direct_point_text_split_ids = [1, 2]
    trainer.direct_point_text_embeddings = F.normalize(torch.eye(2), dim=-1)
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=2,
    )
    trainer.siglip_summary_head = object()
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    class _Model:
        def query_compact_points(self, points, k):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def _decode(self, compact_map):
            return compact_map

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    class _Adapter(torch.nn.Module):
        def forward(self, compact):
            return compact

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=torch.float32,
        )
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    def project_summary(self, features):
        return F.normalize(features.float(), dim=1)

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.point_summary_adapter = _Adapter()
    trainer.distill_loss_fn = _ZeroDistill()
    trainer._project_summary_head_features = MethodType(project_summary, trainer)

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 2),
    )

    expected = F.cross_entropy(torch.eye(2), torch.tensor([1, 0]))
    assert torch.allclose(stats["adapter_text_pseudo_ce"], expected)
    assert torch.allclose(stats["adapter_text_pseudo_ce_valid_ratio"], torch.tensor(1.0))
    assert torch.allclose(stats["adapter_text_pseudo_ce_agreement"], torch.tensor(0.0))
    assert torch.allclose(stats["loss"], expected)


def test_direct_point_loss_can_train_decoded_summary_with_teacher_pseudo_ce(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 0.0
    trainer.direct_point_text_loss_weight = 0.0
    trainer.direct_point_adapter_text_loss_weight = 0.0
    trainer.direct_point_text_distill_weight = 0.0
    trainer.direct_point_adapter_text_distill_weight = 0.0
    trainer.direct_point_text_pseudo_ce_weight = 1.0
    trainer.direct_point_text_pseudo_ce_confidence_threshold = 0.0
    trainer.direct_point_text_pseudo_ce_logit_scale = 1.0
    trainer.direct_point_adapter_text_pseudo_ce_weight = 0.0
    trainer.direct_point_adapter_text_pseudo_ce_confidence_threshold = 0.0
    trainer.direct_point_adapter_text_pseudo_ce_logit_scale = 1.0
    trainer.direct_point_adapter_decoder_anchor_weight = 0.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 2
    trainer.direct_point_query_mode = "knn"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = None
    trainer.direct_point_text_split_ids = [1, 2]
    trainer.direct_point_text_embeddings = F.normalize(torch.eye(2), dim=-1)
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=2,
    )
    trainer.siglip_summary_head = object()
    trainer.point_summary_adapter = None
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    class _Model:
        def query_compact_points(self, points, k):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def _decode(self, compact_map):
            return compact_map

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=torch.float32,
        )
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    def project_summary(self, features):
        return F.normalize(features.float(), dim=1)

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.distill_loss_fn = _ZeroDistill()
    trainer._project_summary_head_features = MethodType(project_summary, trainer)

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 2),
    )

    expected = F.cross_entropy(torch.eye(2), torch.tensor([1, 0]))
    assert torch.allclose(stats["text_pseudo_ce"], expected)
    assert torch.allclose(stats["text_pseudo_ce_valid_ratio"], torch.tensor(1.0))
    assert torch.allclose(stats["text_pseudo_ce_agreement"], torch.tensor(0.0))
    assert torch.allclose(stats["loss"], expected)


def test_direct_point_text_pseudo_ce_can_center_teacher_class_bias():
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_text_split_ids = [1, 2]
    trainer.direct_point_text_embeddings = F.normalize(torch.eye(2), dim=-1)

    point_summary = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    teacher_summary = torch.tensor(
        [
            [0.2, 0.9],
            [0.2, 0.8],
        ],
        dtype=torch.float32,
    )

    loss, valid_ratio, mean_conf, agreement = trainer._compute_direct_point_text_pseudo_ce(
        point_summary,
        teacher_summary,
        logit_scale=1.0,
        confidence_threshold=0.0,
        center_logits=True,
    )

    raw_teacher_logits = teacher_summary
    centered_student_logits = point_summary - raw_teacher_logits.mean(dim=0, keepdim=True)
    expected = F.cross_entropy(centered_student_logits, torch.tensor([1, 0]))
    assert torch.allclose(loss, expected)
    assert torch.allclose(valid_ratio, torch.tensor(1.0))
    assert mean_conf > 0.5
    assert torch.allclose(agreement, torch.tensor(0.0))


def test_direct_point_text_pseudo_ce_can_average_multiple_text_splits():
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    point_summary = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    teacher_summary = F.normalize(
        torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    split19_embeddings = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    split10_embeddings = F.normalize(
        torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    trainer.direct_point_text_pseudo_ce_banks = [
        ("19", [1, 2], split19_embeddings),
        ("10", [1, 33], split10_embeddings),
    ]

    trainer.direct_point_text_split_ids = [1, 2]
    trainer.direct_point_text_embeddings = split19_embeddings
    split19 = trainer._compute_direct_point_text_pseudo_ce(
        point_summary,
        teacher_summary,
        logit_scale=2.0,
        confidence_threshold=0.0,
        center_logits=False,
    )
    trainer.direct_point_text_split_ids = [1, 33]
    trainer.direct_point_text_embeddings = split10_embeddings
    split10 = trainer._compute_direct_point_text_pseudo_ce(
        point_summary,
        teacher_summary,
        logit_scale=2.0,
        confidence_threshold=0.0,
        center_logits=False,
    )

    actual = trainer._compute_multi_split_direct_point_text_pseudo_ce(
        point_summary,
        teacher_summary,
        logit_scale=2.0,
        confidence_threshold=0.0,
        center_logits=False,
    )

    for got, first, second in zip(actual, split19, split10):
        assert torch.allclose(got, 0.5 * (first + second), atol=1e-6)


def test_direct_point_loss_can_anchor_adapter_to_decoder_summary(monkeypatch):
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.direct_point_loss_weight = 1.0
    trainer.direct_point_summary_alignment_weight = 0.0
    trainer.direct_point_summary_adapter_weight = 0.0
    trainer.direct_point_text_loss_weight = 0.0
    trainer.direct_point_adapter_text_loss_weight = 0.0
    trainer.direct_point_text_distill_weight = 0.0
    trainer.direct_point_adapter_text_distill_weight = 0.0
    trainer.direct_point_adapter_text_pseudo_ce_weight = 0.0
    trainer.direct_point_adapter_decoder_anchor_weight = 2.0
    trainer.direct_point_feature_key = "features"
    trainer.direct_point_sample_count = 2
    trainer.direct_point_query_mode = "knn"
    trainer.direct_point_k = 1
    trainer.direct_point_depth_tolerance = 0.08
    trainer.direct_point_relative_depth_tolerance = 0.02
    trainer.direct_point_alpha_threshold = 0.02
    trainer.direct_point_pool = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    trainer.direct_point_pool_labels = None
    trainer.direct_point_text_split_ids = []
    trainer.direct_point_text_embeddings = None
    trainer._is_hybrid = True
    trainer.renderer = SimpleNamespace(
        K=torch.eye(3),
        image_height=2,
        image_width=2,
    )
    trainer.siglip_summary_head = object()
    trainer._canonicalize_spatial_map = MethodType(
        lambda self, value, batch_size, spatial_size, add_channel_dim=False: None,
        trainer,
    )

    class _Model:
        def query_compact_points(self, points, k):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    class _Codec:
        def __init__(self):
            self.decoder = self._decode

        def _decode(self, compact_map):
            return compact_map

    class _ZeroDistill:
        def __call__(self, decoded, target):
            return {"total": decoded.sum() * 0.0}

    class _Adapter(torch.nn.Module):
        def forward(self, compact):
            return torch.stack([compact[:, 1], compact[:, 0]], dim=1)

    def fake_select_visible(points, poses, k, **kwargs):
        return torch.arange(points.shape[0]), torch.ones(points.shape[0], dtype=torch.bool)

    def fake_sample_targets(points, features, poses, k, **kwargs):
        targets = torch.zeros(points.shape[0], 2)
        valid = torch.ones(points.shape[0], dtype=torch.bool)
        counts = torch.ones(points.shape[0], dtype=torch.long)
        return targets, valid, counts

    def project_summary(self, features):
        return F.normalize(features.float(), dim=1)

    monkeypatch.setattr(
        train_feature_field_module,
        "select_visible_gaussian_indices",
        fake_select_visible,
    )
    monkeypatch.setattr(
        train_feature_field_module,
        "sample_multiview_radio_targets",
        fake_sample_targets,
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    trainer.model = _Model()
    trainer.codec = _Codec()
    trainer.point_summary_adapter = _Adapter()
    trainer.distill_loss_fn = _ZeroDistill()
    trainer._project_summary_head_features = MethodType(project_summary, trainer)

    stats = trainer._compute_direct_point_loss(
        batch={"pose_w2c": torch.eye(4).unsqueeze(0)},
        render_result={},
        target_features=torch.zeros(1, 2, 2, 2),
    )

    assert torch.allclose(stats["adapter_decoder_anchor"], torch.tensor(1.0))
    assert torch.allclose(stats["loss"], torch.tensor(2.0))
