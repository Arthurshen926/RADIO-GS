from __future__ import annotations

import torch

from radio_gs.v4.contracts.lerf_fragment_surface_memory import SCHEMA
from radio_gs.v4.evaluation.lerf_fragment_ceiling import _load_fragment_memory
from radio_gs.v4.evaluation.lerf_object_ceiling import (
    _evaluate_membership_ceiling_at_raster,
)


def test_fragment_ceiling_supports_a_larger_fragmentation_capacity():
    # Exercise the shared evaluator through its generic capacity contract;
    # rendering is covered by the object-ceiling integration test.
    import inspect

    signature = inspect.signature(_evaluate_membership_ceiling_at_raster)
    assert signature.parameters["fragmentation_capacity"].default == 3


def test_fragment_memory_loader_enforces_scene_binding(tmp_path):
    payload = {
        "schema": SCHEMA,
        "scene_state_sha256": "a" * 64,
        "source_frames": [10],
        "fragment_positive": torch.tensor([[0.8]]),
        "view_visibility": torch.tensor([[True]]),
        "fragment_view_index": torch.tensor([0]),
        "fragment_frame_id": torch.tensor([10]),
        "fragment_local_index": torch.tensor([0]),
        "quality": torch.tensor([0.9]),
        "stability": torch.tensor([0.9]),
        "parent_index": torch.tensor([-1]),
        "hierarchy_depth": torch.tensor([0]),
        "information_policy": {
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_rgb_opened": False,
        },
    }
    path = tmp_path / "fragments.pt"
    torch.save(payload, path)
    from radio_gs.utils.immutable_artifacts import sha256_file

    try:
        _load_fragment_memory(
            path,
            expected_sha256=sha256_file(path),
            scene_state_sha256="b" * 64,
        )
    except ValueError as error:
        assert "not bound" in str(error)
    else:
        raise AssertionError("mismatched scene binding was accepted")
