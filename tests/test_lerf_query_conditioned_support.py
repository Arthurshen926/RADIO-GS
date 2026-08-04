from pathlib import Path

import pytest
import torch

from radio_gs.querying.support_solver import (
    SupportGraphConfig,
    build_primitive_support_graph,
)
from radio_gs.scripts.eval_lerf_query_conditioned_support import (
    _load_surface_graph,
    precompute_query_conditioned_membership,
)


def _graph():
    xyz = torch.tensor(
        [
            [0.00, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [1.00, 0.0, 0.0],
            [1.02, 0.0, 0.0],
            [1.04, 0.0, 0.0],
        ]
    )
    appearance = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    boundary = appearance.clone()
    graph = build_primitive_support_graph(
        xyz,
        appearance_features=appearance,
        boundary_features=boundary,
        config=SupportGraphConfig(neighbors=2),
    )
    return xyz, graph


def test_text_support_precompute_is_target_free_finite_and_deterministic():
    xyz, graph = _graph()
    query_scores = torch.zeros(6, 3, 2)
    query_scores[:, :, 0] = torch.tensor(
        [0.95, 0.90, 0.75, 0.10, 0.05, 0.00]
    )[:, None]
    query_scores[:, :, 1] = 1.0 - query_scores[:, :, 0]
    cache = {
        "query_scores": query_scores,
        "xyz": xyz,
        "valid": torch.ones(6, dtype=torch.bool),
    }
    first = precompute_query_conditioned_membership(
        cache, graph, torch.arange(6), device=torch.device("cpu")
    )
    second = precompute_query_conditioned_membership(
        cache, graph, torch.arange(6), device=torch.device("cpu")
    )
    assert first["propagated_scores"].shape == (6, 2)
    assert bool(torch.isfinite(first["propagated_scores"]).all())
    assert first["membership_sha256"] == second["membership_sha256"]
    torch.testing.assert_close(
        first["propagated_scores"], second["propagated_scores"], atol=0, rtol=0
    )


def _write_graph(path: Path, *, field_sha: str = "a" * 64) -> tuple[dict, torch.Tensor]:
    xyz, graph = _graph()
    payload = {
        "schema_version": 1,
        "global_rows": torch.arange(6),
        "num_global_rows": 6,
        "xyz": xyz,
        "edge_index": graph.edge_index,
        "edge_weight": graph.edge_weight,
        "raw_affinity": graph.raw_affinity,
        "edge_channels": dict(graph.edge_channels),
        "local_sigma": graph.local_sigma,
        "metadata": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "capability_metadata": {"field_checkpoint_sha256": field_sha},
        },
    }
    torch.save(payload, path)
    score_cache = {
        "xyz": xyz,
        "valid": torch.ones(6, dtype=torch.bool),
        "field_checkpoint_sha256": field_sha,
    }
    return score_cache, xyz


def test_surface_graph_loader_binds_rows_geometry_field_and_query_independence(tmp_path):
    path = tmp_path / "graph.pt"
    score_cache, _ = _write_graph(path)
    graph, rows, receipt = _load_surface_graph(path, score_cache=score_cache)
    assert graph.num_nodes == 6
    assert torch.equal(rows, torch.arange(6))
    assert receipt["num_edges"] == graph.edge_index.shape[1]
    assert receipt["field_checkpoint_sha256"] == "a" * 64

    with pytest.raises(ValueError, match="different canonical fields"):
        _load_surface_graph(
            path,
            score_cache={**score_cache, "field_checkpoint_sha256": "b" * 64},
        )


def test_surface_graph_loader_rejects_target_tainted_metadata(tmp_path):
    path = tmp_path / "graph.pt"
    score_cache, _ = _write_graph(path)
    payload = torch.load(path, map_location="cpu")
    payload["metadata"]["benchmark_masks_opened"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="not query-independent"):
        _load_surface_graph(path, score_cache=score_cache)
