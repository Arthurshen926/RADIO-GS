import hashlib
from pathlib import Path

import pytest
import torch

from radio_gs.scripts.build_field_b_boundary_relation_triplets import (
    build_triplets_from_arrays,
)
from radio_gs.scripts.train_canonical_radio_field import (
    _load_field_b_relation_triplets,
)
from radio_gs.training.canonical_field_losses import (
    hard_boundary_relation_ranking_loss,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_field_b_triplet_builder_is_deterministic_and_never_uses_self() -> None:
    xyz = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.3, 0, 0], [0.7, 0, 0], [1.2, 0, 0]])
    angles = torch.tensor([0.0, 0.05, 0.5, 1.2, 2.0])
    dino = torch.stack([angles.cos(), angles.sin()], dim=1)
    sam = torch.stack([(1.1 * angles).cos(), (1.1 * angles).sin()], dim=1)
    first = build_triplets_from_arrays(
        xyz, torch.ones(5, dtype=torch.bool), dino, sam, neighbors=2, chunk_size=2
    )
    second = build_triplets_from_arrays(
        xyz, torch.ones(5, dtype=torch.bool), dino, sam, neighbors=2, chunk_size=3
    )
    torch.testing.assert_close(first["pair_index"], second["pair_index"])
    torch.testing.assert_close(first["teacher_margin"], second["teacher_margin"])
    pairs = first["pair_index"]
    count = first["teacher_margin"].numel()
    assert count > 0
    assert torch.equal(pairs[0, :count], pairs[0, count:])
    assert not bool((pairs[0] == pairs[1]).any())
    assert bool((first["teacher_margin"] > 0).all())
    assert first["audit"]["candidate_anchors"] == 5
    assert first["audit"]["triplets"] == count
    assert 0.0 < first["audit"]["retained_fraction"] <= 1.0
    assert first["audit"]["teacher_margin"]["min"] > 0.0


def test_field_b_ranking_loss_uses_teacher_gap_without_scalar_margin() -> None:
    pair_index = torch.tensor([[0, 0], [1, 2]])
    teacher_margin = torch.tensor([0.5])
    preserved = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    loss = hard_boundary_relation_ranking_loss(
        preserved, preserved, pair_index, teacher_margin
    )
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)

    reversed_rows = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    reversed_loss = hard_boundary_relation_ranking_loss(
        reversed_rows, reversed_rows, pair_index, teacher_margin
    )
    assert float(reversed_loss) > 0.9


def test_field_b_cache_loader_is_hash_geometry_and_target_locked(tmp_path: Path) -> None:
    path = tmp_path / "triplets.pt"
    geometry = {"num_gaussians": 3, "xyz_sha256": "geometry"}
    metadata = {
        "schema_version": "canonical_field_b_boundary_relation_triplets_v1",
        "construction": "exact_capability_local_hard_boundary_ranking_v1",
        "neighbors": 16,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "fixed_scalar_margin": None,
        "geometry_fingerprint": geometry,
        "dino_mpr": {"sha256": "dino"},
        "sam3_mpr": {"sha256": "sam"},
    }
    torch.save(
        {
            "schema_version": 1,
            "pair_index": torch.tensor([[0, 0], [1, 2]]),
            "teacher_margin": torch.tensor([0.25]),
            "boundary_channel": torch.tensor([0], dtype=torch.uint8),
            "metadata": metadata,
        },
        path,
    )
    cache, provenance = _load_field_b_relation_triplets(
        path,
        expected_sha256=_sha(path),
        num_rows=3,
        capability_valid=torch.ones(3, dtype=torch.bool),
        geometry_fingerprint=geometry,
        expected_dino_sha256="dino",
        expected_sam3_sha256="sam",
    )
    assert cache["teacher_margin"].tolist() == [0.25]
    assert provenance["triplets"] == 1
    with pytest.raises(ValueError, match="targets differ"):
        _load_field_b_relation_triplets(
            path,
            expected_sha256=_sha(path),
            num_rows=3,
            capability_valid=torch.ones(3, dtype=torch.bool),
            geometry_fingerprint=geometry,
            expected_dino_sha256="other",
            expected_sam3_sha256="sam",
        )
