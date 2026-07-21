import numpy as np
from plyfile import PlyData, PlyElement
import torch

from radio_gs.benchmarks.scannet_pfir.export_query_prediction import export
from radio_gs.benchmarks.scannet_pfir.evaluation.eval_instance_ranking import (
    evaluate_instance_ranking,
)
from radio_gs.benchmarks.scannet_pfir.evaluation.eval_instance_selection import (
    evaluate_instance_selection,
)
from radio_gs.benchmarks.scannet_pfir.evaluation.map_gaussians_to_mesh import (
    gaussian_scores_to_mesh,
)


def _records():
    return [
        {
            "query_id": "q1",
            "scene_id": "s1",
            "instance_id_3d": 1,
            "nyu40_class_id": 5,
            "same_category_distractor_count": 1,
            "candidate_instance_ids_3d": [1, 2],
            "candidate_instance_class_ids": {"1": 5, "2": 5},
        },
        {
            "query_id": "q2",
            "scene_id": "s1",
            "instance_id_3d": 2,
            "nyu40_class_id": 5,
            "same_category_distractor_count": 1,
            "candidate_instance_ids_3d": [1, 2],
            "candidate_instance_class_ids": {"1": 5, "2": 5},
        },
    ]


def test_track_a_ranking_metrics() -> None:
    instances = {"s1": np.array([1, 1, 2, 2])}
    scores = {
        "q1": np.array([0.9, 0.8, 0.1, 0.2]),
        "q2": np.array([0.1, 0.2, 0.9, 0.8]),
    }
    result = evaluate_instance_ranking(_records(), scores, instances)
    assert result["recall_at_1"] == 1.0
    assert result["mrr"] == 1.0
    assert result["same_category_recall_at_1"] == 1.0


def test_track_b_selection_metrics_and_distractors() -> None:
    instances = {"s1": np.array([1, 1, 2, 2])}
    xyz = {
        "s1": np.array([[0, 0, 0], [0, 1, 0], [5, 0, 0], [5, 1, 0]])
    }
    masks = {
        "q1": np.array([1, 1, 0, 0], dtype=bool),
        "q2": np.array([0, 0, 1, 1], dtype=bool),
    }
    result = evaluate_instance_selection(_records(), masks, instances, xyz)
    assert result["query_micro"]["instance_macro_3d_miou"] == 1.0
    assert result["query_micro"]["same_category_distractor_success"] == 1.0
    assert result["scene_macro"]["3d_miou"] == 1.0


def test_gaussian_to_mesh_mapping_is_gt_independent() -> None:
    gaussian_xyz = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    mesh_xyz = np.array([[0.01, 0, 0], [0.99, 0, 0], [5, 0, 0]], dtype=np.float32)
    mapped, valid = gaussian_scores_to_mesh(
        gaussian_xyz, np.array([0.2, 0.8]), mesh_xyz, neighbors=1
    )
    np.testing.assert_allclose(mapped[:2], [0.2, 0.8])
    assert valid.tolist() == [True, True, False]


def test_query_export_separates_unary_ranking_from_solver_mask(tmp_path) -> None:
    cache = tmp_path / "query.pt"
    torch.save(
        {
            "xyz": torch.tensor([[0.0, 0, 0], [1.0, 0, 0]]),
            "valid": torch.tensor([True, True]),
            "unary": torch.tensor([[0.8], [-0.2]]),
            "features": torch.tensor([[0.7], [0.1]]),
        },
        cache,
    )
    vertex = np.empty(
        2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    vertex["x"] = [0.01, 0.99]
    vertex["y"] = vertex["z"] = 0
    mesh = tmp_path / "mesh.ply"
    PlyData([PlyElement.describe(vertex, "vertex")]).write(mesh)

    class Args:
        query_cache = str(cache)
        mesh_ply = str(mesh)
        ranking_output = str(tmp_path / "ranking.npy")
        selection_output = str(tmp_path / "selection.npy")
        neighbors = 1
        maximum_distance_m = 0.10
        support_threshold = 0.50

    export(Args())
    np.testing.assert_allclose(np.load(Args.ranking_output), [0.8, -0.2])
    np.testing.assert_array_equal(np.load(Args.selection_output), [True, False])
