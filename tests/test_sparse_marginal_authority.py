from __future__ import annotations

import json

import pytest
import torch

from radio_gs.scripts import build_gaussian_multiview_teacher_cache as builder
from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA,
    SparseExactMarginalAuthorityWriter,
    canonicalize_sparse_marginal_view,
    load_sparse_exact_marginal_authority,
)
from radio_gs.utils.immutable_artifacts import sha256_file


def test_compositor_authority_hash_resolves_unwrapped_repo_source() -> None:
    source = builder._unwrapped_implementation_source(
        builder.rasterize_single_view_contributions
    )

    assert source.as_posix().endswith(
        "/radio_gs/rendering/contribution_compositor.py"
    )


def _metadata() -> dict[str, object]:
    return {
        "assignment_mode": "exact_front_to_back_sparse_marginal",
        "geometry_checkpoint_sha256": "a" * 64,
        "gaussian_state_sha256": "b" * 64,
        "pose_sha256": "c" * 64,
        "intrinsics_sha256": "d" * 64,
        "builder_implementation_sha256": "e" * 64,
        "authority_implementation_sha256": "1" * 64,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }


def _write_authority(tmp_path):
    path = tmp_path / "authority.json"
    writer = SparseExactMarginalAuthorityWriter(
        path,
        metadata=_metadata(),
        frame_indices=[3],
        num_gaussians=3,
        num_pixels=2,
    )
    # Deliberately interleave pixels. Stable canonicalization keeps the two
    # front-to-back pixel-zero hits in their original relative order.
    writer.add_view(
        0,
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 1, 0]),
        torch.tensor([0.8, 0.6, 0.2]),
    )
    _path, digest = writer.finalize()
    return path, digest


def test_sparse_marginal_authority_round_trip_derives_target_weights(tmp_path) -> None:
    path, digest = _write_authority(tmp_path)

    assignments, observed, source = load_sparse_exact_marginal_authority(
        path,
        expected_metadata=_metadata(),
        expected_frame_indices=[3],
        num_gaussians=3,
        num_pixels=2,
        expected_sha256=digest,
    )

    assert source == path.resolve()
    assert observed == digest
    assert assignments[0]["gaussian_ids"].tolist() == [0, 1, 0]
    assert assignments[0]["pixel_ids"].tolist() == [0, 0, 1]
    torch.testing.assert_close(
        assignments[0]["base_weights"], torch.tensor([0.8, 0.2, 0.6])
    )
    torch.testing.assert_close(
        assignments[0]["marginal_weights"], torch.tensor([0.64, 0.04, 0.6])
    )
    manifest = json.loads(path.read_text())
    assert manifest["schema"] == SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA
    assert "marginal_weights" not in manifest
    shard = torch.load(path.parent / manifest["views"][0]["relative_path"])
    assert "marginal_weights" not in shard


def test_sparse_marginal_writer_resumes_bound_views(tmp_path) -> None:
    path = tmp_path / "authority.json"
    writer = SparseExactMarginalAuthorityWriter(
        path,
        metadata=_metadata(),
        frame_indices=[3, 5],
        num_gaussians=2,
        num_pixels=2,
    )
    writer.add_view(
        0, torch.tensor([0]), torch.tensor([0]), torch.tensor([0.5])
    )
    writer._release_lock()

    resumed = SparseExactMarginalAuthorityWriter(
        path,
        metadata=_metadata(),
        frame_indices=[3, 5],
        num_gaussians=2,
        num_pixels=2,
    )
    assert resumed.completed_view_indices == frozenset({0})
    resumed.add_view(
        1, torch.tensor([1]), torch.tensor([1]), torch.tensor([0.25])
    )
    _path, digest = resumed.finalize()
    assert sha256_file(path) == digest
    assert not path.with_suffix(".json.partial.json").exists()


def test_sparse_marginal_writer_rejects_concurrent_writer(tmp_path) -> None:
    path = tmp_path / "authority.json"
    writer = SparseExactMarginalAuthorityWriter(
        path,
        metadata=_metadata(),
        frame_indices=[3],
        num_gaussians=2,
        num_pixels=2,
    )
    with pytest.raises(RuntimeError, match="already being written"):
        SparseExactMarginalAuthorityWriter(
            path,
            metadata=_metadata(),
            frame_indices=[3],
            num_gaussians=2,
            num_pixels=2,
        )
    writer._release_lock()


def test_sparse_marginal_writer_recovers_shard_committed_before_progress(
    tmp_path,
) -> None:
    path = tmp_path / "authority.json"
    writer = SparseExactMarginalAuthorityWriter(
        path,
        metadata=_metadata(),
        frame_indices=[3, 5],
        num_gaussians=2,
        num_pixels=2,
    )
    writer.add_view(
        0, torch.tensor([0]), torch.tensor([0]), torch.tensor([0.5])
    )
    progress_path = path.with_suffix(".json.partial.json")
    progress = json.loads(progress_path.read_text())
    progress["views"] = []
    progress_path.write_text(json.dumps(progress))
    writer._release_lock()

    resumed = SparseExactMarginalAuthorityWriter(
        path,
        metadata=_metadata(),
        frame_indices=[3, 5],
        num_gaussians=2,
        num_pixels=2,
    )
    assert resumed.completed_view_indices == frozenset({0})
    rebound = json.loads(progress_path.read_text())
    assert [record["view_index"] for record in rebound["views"]] == [0]
    resumed.add_view(
        1, torch.tensor([1]), torch.tensor([1]), torch.tensor([0.25])
    )
    resumed.finalize()


def test_sparse_marginal_writer_rejects_malformed_unbound_shard(tmp_path) -> None:
    path = tmp_path / "authority.json"
    writer = SparseExactMarginalAuthorityWriter(
        path,
        metadata=_metadata(),
        frame_indices=[3],
        num_gaussians=2,
        num_pixels=2,
    )
    writer._release_lock()
    shard = path.parent / f"{path.name}.views/view_00000.pt"
    shard.parent.mkdir(parents=True)
    torch.save({"gaussian_ids": torch.tensor([0])}, shard)

    with pytest.raises(ValueError, match="contract differs|malformed"):
        SparseExactMarginalAuthorityWriter(
            path,
            metadata=_metadata(),
            frame_indices=[3],
            num_gaussians=2,
            num_pixels=2,
        )


@pytest.mark.parametrize(
    "gids,pids,weights,match",
    [
        ([0, 0], [0, 0], [0.5, 0.4], "repeats"),
        ([0], [2], [0.5], "outside"),
        ([3], [0], [0.5], "outside"),
        ([0], [0], [0.0], "lie in"),
        ([0], [0], [1.1], "lie in"),
        ([0], [0], [float("nan")], "lie in"),
    ],
)
def test_sparse_marginal_view_rejects_invalid_values(
    gids, pids, weights, match
) -> None:
    with pytest.raises(ValueError, match=match):
        canonicalize_sparse_marginal_view(
            torch.tensor(gids),
            torch.tensor(pids),
            torch.tensor(weights),
            num_gaussians=3,
            num_pixels=2,
        )


def test_sparse_marginal_authority_fails_closed_on_metadata_or_sha(tmp_path) -> None:
    path, digest = _write_authority(tmp_path)
    with pytest.raises(ValueError, match="contract differs"):
        load_sparse_exact_marginal_authority(
            path,
            expected_metadata={**_metadata(), "pose_sha256": "f" * 64},
            expected_frame_indices=[3],
            num_gaussians=3,
            num_pixels=2,
            expected_sha256=digest,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        load_sparse_exact_marginal_authority(
            path,
            expected_metadata=_metadata(),
            expected_frame_indices=[3],
            num_gaussians=3,
            num_pixels=2,
            expected_sha256="f" * 64,
        )


def test_sparse_marginal_authority_accepts_builder_lineage_only_drift(tmp_path) -> None:
    path, digest = _write_authority(tmp_path)

    assignments, observed, _source = load_sparse_exact_marginal_authority(
        path,
        expected_metadata={
            **_metadata(),
            "builder_implementation_sha256": "f" * 64,
            "authority_implementation_sha256": "2" * 64,
        },
        expected_frame_indices=[3],
        num_gaussians=3,
        num_pixels=2,
        expected_sha256=digest,
    )

    assert observed == digest
    assert len(assignments) == 1


def test_sparse_marginal_authority_is_no_clobber(tmp_path) -> None:
    path, _digest = _write_authority(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        SparseExactMarginalAuthorityWriter(
            path,
            metadata=_metadata(),
            frame_indices=[3],
            num_gaussians=3,
            num_pixels=2,
        )
