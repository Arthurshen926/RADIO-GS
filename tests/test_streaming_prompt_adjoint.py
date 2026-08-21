import hashlib
from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    build_prompt_responsibility_cache,
    save_prompt_responsibility_cache,
)
from radio_gs.interfaces.streaming_prompt_adjoint import (
    normalized_adjoint_from_chunks,
    streaming_prompt_adjoint,
    streaming_prompt_cache_metadata,
)


def _chunks(gids, pids, weights, split=2):
    for start in range(0, len(gids), split):
        stop = min(len(gids), start + split)
        yield gids[start:stop], pids[start:stop], weights[start:stop]


def test_normalized_streaming_adjoint_preserves_constants_and_linearity():
    gids = torch.tensor([0, 1, 0, 2, 1], dtype=torch.int64)
    pids = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    weights = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.25], dtype=torch.float32)
    mass = torch.zeros(3, dtype=torch.float64)
    mass.index_add_(0, gids, weights.double())
    one = normalized_adjoint_from_chunks(
        _chunks(gids, pids, weights),
        torch.ones(3),
        num_gaussians=3,
        visible_mass=mass,
        hit_count=5,
        chunk_hits=2,
    )
    assert torch.equal(one.primitive_probability, torch.ones(3, dtype=torch.float64))

    a = torch.tensor([0.1, 0.4, 0.9])
    b = torch.tensor([0.7, 0.2, 0.3])
    left = normalized_adjoint_from_chunks(
        _chunks(gids, pids, weights),
        0.25 * a + 0.75 * b,
        num_gaussians=3,
        visible_mass=mass,
        hit_count=5,
        chunk_hits=2,
    ).weighted_sum
    right_a = normalized_adjoint_from_chunks(
        _chunks(gids, pids, weights), a, num_gaussians=3,
        visible_mass=mass, hit_count=5, chunk_hits=2,
    ).weighted_sum
    right_b = normalized_adjoint_from_chunks(
        _chunks(gids, pids, weights), b, num_gaussians=3,
        visible_mass=mass, hit_count=5, chunk_hits=2,
    ).weighted_sum
    assert torch.allclose(left, 0.25 * right_a + 0.75 * right_b, atol=1e-15)


def test_torch_zip_stream_matches_in_memory_exact_adjoint(tmp_path):
    authority = PromptResponsibilityAuthority(
        scene_id="scene",
        frame_id="frame",
        camera_name="camera",
        colmap_camera_name="camera",
        geometry_checkpoint_sha256="1" * 64,
        geometry_xyz_sha256="2" * 64,
        pose_sha256="3" * 64,
        intrinsics_sha256="4" * 64,
        height=1,
        width=3,
        num_gaussians=3,
        source_sha256={"test": "5" * 64},
    )
    cache = build_prompt_responsibility_cache(
        authority=authority,
        gaussian_ids=torch.tensor([0, 1, 0, 2, 1]),
        pixel_ids=torch.tensor([0, 0, 1, 1, 2]),
        weights=torch.tensor([0.2, 0.3, 0.4, 0.5, 0.25]),
    )
    path = tmp_path / "cache.pt"
    save_prompt_responsibility_cache(cache, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    probability = torch.tensor([0.1, 0.8, 0.4])
    streamed, payload = streaming_prompt_adjoint(
        path, probability, expected_file_sha256=digest, chunk_hits=2
    )
    expected = cache.adjoint(probability)
    assert payload["authority"]["scene_id"] == "scene"
    assert torch.equal(streamed.visible_mass, expected.visible_mass)
    assert torch.equal(streamed.weighted_sum, expected.weighted_sum)
    assert torch.equal(streamed.primitive_probability, expected.primitive_probability)


def test_metadata_reader_rejects_unknown_pickle_global(tmp_path):
    path = tmp_path / "unknown.pt"
    torch.save({"unsupported": Path("value")}, path)
    with pytest.raises(ValueError, match="unsupported global"):
        streaming_prompt_cache_metadata(path)


def test_metadata_reader_rejects_unsupported_storage(tmp_path):
    path = tmp_path / "int_storage.pt"
    torch.save({"tensors": {"value": torch.tensor([1], dtype=torch.int32)}}, path)
    with pytest.raises(ValueError, match="unsupported global"):
        streaming_prompt_cache_metadata(path)
