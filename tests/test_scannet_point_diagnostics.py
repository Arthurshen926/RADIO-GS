import numpy as np
import torch
from types import SimpleNamespace

from radio_gs.scripts.diagnose_scannet_point_logits import (
    _decode_compact_1280,
    _load_w2c_for_feature_paths,
    _query_model_visuals,
    _sample_label_indices,
    _topk_names,
)


def test_sample_label_indices_balances_valid_classes() -> None:
    labels = np.array([1, 1, 1, 2, 2, 33, 40], dtype=np.int32)

    indices = _sample_label_indices(labels, split_ids=[1, 2, 33], max_points=6, seed=7)

    sampled = labels[indices]
    assert set(sampled.tolist()) == {1, 2, 33}
    assert 40 not in sampled
    assert len(indices) <= 6


def test_topk_names_returns_sorted_scores() -> None:
    names = ["wall", "floor", "chair"]
    top = _topk_names(np.array([0.2, 0.7, 0.1], dtype=np.float32), names, k=2)

    assert top == [
        {"name": "floor", "score": 0.7},
        {"name": "wall", "score": 0.2},
    ]


def test_decode_compact_1280_prefers_pointwise_codec_api() -> None:
    class Codec:
        def decode_points(self, compact):
            return compact + 3.0

        def decode(self, compact_map):  # pragma: no cover - should not be called
            raise AssertionError("diagnostics must use pointwise decode when available")

    compact = torch.zeros(4, 3)

    decoded = _decode_compact_1280(Codec(), compact)

    assert decoded.shape == compact.shape
    assert torch.allclose(decoded, torch.full_like(compact, 3.0))


def test_query_model_visuals_supports_gaussian_index_mode() -> None:
    class Model:
        def get_xyz(self):
            return torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                ],
                dtype=torch.float32,
            )

        def query_compact_points(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("gaussian_index diagnostics must not use kNN query")

        def query_gaussian_points(self, indices, return_aux=False):
            assert return_aux is True
            compact = torch.stack(
                [
                    indices.float(),
                    indices.float() + 1.0,
                    indices.float() + 2.0,
                ],
                dim=1,
            )
            return {"features": compact, "gaussian_indices": indices}

    class Codec:
        def decode_points(self, compact):
            return compact

    class Projection(torch.nn.Module):
        def forward(self, features):
            return features

    points = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float32)
    gaussian_indices = torch.tensor([1, 2], dtype=torch.long)

    visual, nearest_indices, nearest_distances = _query_model_visuals(
        Model(),
        Codec(),
        Projection(),
        points,
        k=8,
        chunk_size=1,
        query_mode="gaussian_index",
        gaussian_indices=gaussian_indices,
    )

    assert visual.shape == (2, 3)
    assert nearest_indices.tolist() == [1, 2]
    assert torch.allclose(nearest_distances, torch.zeros(2))


def test_pose_order_fallback_handles_original_frame_ids(tmp_path) -> None:
    feature_dir = tmp_path / "features" / "backbone"
    feature_dir.mkdir(parents=True)
    for frame_id in (0, 20, 40):
        (feature_dir / f"rgb_{frame_id}.pt").touch()

    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    c2w[1, 0, 3] = 2.0
    pose_file = tmp_path / "traj_w_c.txt"
    np.savetxt(pose_file, c2w.reshape(-1, 4))

    cfg = SimpleNamespace(feature_dir=str(tmp_path / "features"), pose_file=str(pose_file))
    selected = [feature_dir / "rgb_20.pt"]

    w2c = _load_w2c_for_feature_paths(cfg, selected, "all")

    assert w2c.shape == (1, 4, 4)
    assert np.isclose(w2c[0, 0, 3], -2.0)
