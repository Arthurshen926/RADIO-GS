import dataclasses
import hashlib
import inspect

import pytest
import torch

from radio_gs.querying.source_footprint_fold_authority import (
    FIELD_BASE_ACTION,
    LONG_AXIS_BLOCKS,
    MINIMUM_CLASS_ROWS,
    RUN_SOURCE_OOF_ACTION,
    build_source_raster_dominant_footprint_authority,
    clear_four_source_evidence_for_fold,
    load_source_footprint_fold_authority,
    save_source_footprint_fold_authority,
    source_fold_population_base_decision,
    splitmix64_source_group_folds,
)
from radio_gs.querying.source_conditioned_hierarchy_completion import (
    deterministic_group_folds,
)


TRIPLET_SHA = "a" * 64


def _build(
    pixel_ids: torch.Tensor,
    primitive_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    height: int,
    width: int,
    rows: torch.Tensor,
    domain: str = "global_rows",
):
    return build_source_raster_dominant_footprint_authority(
        pixel_ids,
        primitive_ids,
        weights,
        height=height,
        width=width,
        hierarchy_primitive_rows=rows,
        primitive_id_domain=domain,
        source_triplet_authority_sha256=TRIPLET_SHA,
        expected_source_triplet_authority_sha256=TRIPLET_SHA,
    )


def test_portrait_and_landscape_pixel_centers_use_fixed_long_axis_grid():
    rows = torch.tensor([10, 20, 30, 40, 50])
    landscape = _build(
        torch.tensor([0, 7, 24, 31]),
        torch.tensor([10, 20, 30, 40]),
        torch.ones(4),
        height=4,
        width=8,
        rows=rows,
    )
    assert (landscape.block_rows, landscape.block_cols) == (4, LONG_AXIS_BLOCKS)
    assert landscape.group_ids.tolist() == [0, 7, 24, 31, 32]
    assert landscape.invisible_group_id == 32

    portrait = _build(
        torch.tensor([0, 3, 28, 31]),
        torch.tensor([10, 20, 30, 40]),
        torch.ones(4),
        height=8,
        width=4,
        rows=rows,
    )
    assert (portrait.block_rows, portrait.block_cols) == (LONG_AXIS_BLOCKS, 4)
    assert portrait.group_ids.tolist() == [0, 3, 28, 31, 32]
    assert portrait.invisible_group_id == 32


def test_triplet_permutation_is_canonical_and_exact_mass_tie_uses_smallest_block():
    rows = torch.tensor([10, 20, 30])
    pixel_ids = torch.tensor([4, 1, 63, 0, 9])
    primitive_ids = torch.tensor([10, 10, 20, 20, 20])
    weights = torch.tensor([0.5, 0.5, 0.25, 0.25, 0.5], dtype=torch.float64)
    first = _build(
        pixel_ids,
        primitive_ids,
        weights,
        height=8,
        width=8,
        rows=rows,
    )
    permutation = torch.tensor([3, 0, 4, 2, 1])
    second = _build(
        pixel_ids[permutation],
        primitive_ids[permutation],
        weights[permutation],
        height=8,
        width=8,
        rows=rows,
    )
    assert first.group_ids.tolist() == [1, 9, 64]
    torch.testing.assert_close(
        first.visible_mass, torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        first.dominant_mass, torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        first.purity, torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    )
    assert first.canonical_triplet_sha256 == second.canonical_triplet_sha256
    assert first.tensor_bundle_sha256 == second.tensor_bundle_sha256
    assert first.authority_sha256 == second.authority_sha256
    for name in ("group_ids", "visible_mass", "dominant_mass", "purity"):
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_authority_artifact_roundtrip_and_fail_closed_tamper(tmp_path):
    authority = _build(
        torch.tensor([0, 7, 63]),
        torch.tensor([10, 20, 30]),
        torch.tensor([0.25, 0.5, 0.75]),
        height=8,
        width=8,
        rows=torch.tensor([10, 20, 30, 40]),
    )
    path = tmp_path / "authority.pt"
    artifact = save_source_footprint_fold_authority(authority, path)
    loaded = load_source_footprint_fold_authority(
        path,
        expected_file_sha256=artifact["file_sha256"],
        expected_authority_sha256=authority.authority_sha256,
    )
    assert loaded.authority_sha256 == authority.authority_sha256
    assert loaded.tensor_bundle_sha256 == authority.tensor_bundle_sha256
    assert torch.equal(loaded.group_ids, authority.group_ids)

    with pytest.raises(ValueError, match="file SHA-256"):
        load_source_footprint_fold_authority(
            path,
            expected_file_sha256="b" * 64,
            expected_authority_sha256=authority.authority_sha256,
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["tensors"]["group_ids"][0] += 1
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    tampered_sha = hashlib.sha256(tampered.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="tensor hash"):
        load_source_footprint_fold_authority(
            tampered,
            expected_file_sha256=tampered_sha,
            expected_authority_sha256=authority.authority_sha256,
        )


def test_local_indices_map_to_exact_hierarchy_order_and_empty_triplets_are_invisible():
    rows = torch.tensor([7, 11, 19])
    authority = _build(
        torch.tensor([0, 63]),
        torch.tensor([2, 0]),
        torch.tensor([0.3, 0.7]),
        height=8,
        width=8,
        rows=rows,
        domain="local_indices",
    )
    assert authority.primitive_rows.tolist() == [7, 11, 19]
    assert authority.group_ids.tolist() == [63, 64, 0]
    empty = _build(
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.float32),
        height=8,
        width=8,
        rows=rows,
    )
    assert empty.group_ids.tolist() == [64, 64, 64]
    assert int(torch.count_nonzero(empty.visible_mass)) == 0
    assert int(torch.count_nonzero(empty.dominant_mass)) == 0
    assert int(torch.count_nonzero(empty.purity)) == 0


def test_authority_constructor_has_no_mask_label_query_or_target_api():
    parameters = set(
        inspect.signature(build_source_raster_dominant_footprint_authority).parameters
    )
    forbidden_fragments = ("mask", "label", "query", "target", "probability")
    assert not any(
        fragment in name for name in parameters for fragment in forbidden_fragments
    )
    assert parameters == {
        "pixel_ids",
        "primitive_ids",
        "weights",
        "height",
        "width",
        "hierarchy_primitive_rows",
        "primitive_id_domain",
        "source_triplet_authority_sha256",
        "expected_source_triplet_authority_sha256",
    }


def test_splitmix64_never_splits_one_group():
    groups = torch.tensor([0, 0, 1, 5, 5, 5, 64, 64])
    folds = splitmix64_source_group_folds(groups)
    assert torch.equal(folds, deterministic_group_folds(groups))
    for group in groups.unique():
        assert folds[groups == group].unique().numel() == 1
    with pytest.raises(ValueError, match="exactly three"):
        splitmix64_source_group_folds(groups, num_folds=2)


def _balanced_synthetic_exact_adjoint():
    """One positive and one negative source pixel live inside each fold block."""

    height = width = 16
    block_folds = splitmix64_source_group_folds(torch.arange(64))
    chosen_blocks = [int(torch.where(block_folds == fold)[0][0]) for fold in range(3)]
    pixel_ids = []
    primitive_ids = []
    positive_pixels = torch.zeros(height * width, dtype=torch.bool)
    rows = torch.arange(1000, 1000 + 3 * 2 * MINIMUM_CLASS_ROWS)
    cursor = 0
    for block in chosen_blocks:
        block_y, block_x = divmod(block, 8)
        positive_pixel = (2 * block_y) * width + 2 * block_x
        negative_pixel = positive_pixel + 1
        positive_pixels[positive_pixel] = True
        for _ in range(MINIMUM_CLASS_ROWS):
            pixel_ids.append(positive_pixel)
            primitive_ids.append(int(rows[cursor]))
            cursor += 1
        for _ in range(MINIMUM_CLASS_ROWS):
            pixel_ids.append(negative_pixel)
            primitive_ids.append(int(rows[cursor]))
            cursor += 1
    pixel_ids_tensor = torch.tensor(pixel_ids)
    primitive_ids_tensor = torch.tensor(primitive_ids)
    weights = torch.full(
        (len(pixel_ids),), 1.0 / MINIMUM_CLASS_ROWS, dtype=torch.float64
    )
    authority = _build(
        pixel_ids_tensor,
        primitive_ids_tensor,
        weights,
        height=height,
        width=width,
        rows=rows,
    )
    # Synthetic exact W^T adjoint is deliberately outside authority creation.
    local = torch.searchsorted(rows, primitive_ids_tensor)
    positive = torch.zeros(len(rows), dtype=torch.float64)
    negative = torch.zeros(len(rows), dtype=torch.float64)
    positive.index_add_(0, local, weights * positive_pixels[pixel_ids_tensor])
    negative.index_add_(0, local, weights * ~positive_pixels[pixel_ids_tensor])
    return authority, positive, negative


def test_synthetic_exact_adjoint_authority_is_eligible_and_clears_whole_groups():
    authority, positive, negative = _balanced_synthetic_exact_adjoint()
    decision = source_fold_population_base_decision(
        authority,
        positive,
        negative,
        expected_authority_sha256=authority.authority_sha256,
    )
    assert decision.run_source_oof
    assert decision.selected_action == RUN_SOURCE_OOF_ACTION
    assert decision.reason == "eligible"
    for report in decision.fold_reports:
        assert report["heldout_positive_rows"] == MINIMUM_CLASS_ROWS
        assert report["heldout_negative_rows"] == MINIMUM_CLASS_ROWS
        assert report["training_positive_rows"] == 2 * MINIMUM_CLASS_ROWS
        assert report["training_negative_rows"] == 2 * MINIMUM_CLASS_ROWS

    original = (positive, negative, 2 * positive, 3 * negative)
    for fold in range(3):
        cleared = clear_four_source_evidence_for_fold(
            authority,
            *original,
            heldout_fold=fold,
            expected_authority_sha256=authority.authority_sha256,
        )
        # Group preservation means this row predicate clears every row in
        # every complete raster block assigned to the heldout fold.
        expected = splitmix64_source_group_folds(authority.group_ids) == fold
        assert torch.equal(cleared.heldout_rows, expected)
        for value, reference in zip(
            (
                cleared.training_positive_weight,
                cleared.training_negative_weight,
                cleared.training_raw_positive_mass,
                cleared.training_raw_negative_mass,
            ),
            original,
        ):
            assert bool((value[expected] == 0).all())
            assert torch.equal(value[~expected], reference[~expected])


def test_class_degenerate_structured_fold_returns_base_instead_of_raising():
    authority = _build(
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0, 1, 2, 3]),
        torch.ones(4),
        height=8,
        width=8,
        rows=torch.arange(4),
        domain="local_indices",
    )
    decision = source_fold_population_base_decision(
        authority,
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        expected_authority_sha256=authority.authority_sha256,
    )
    assert not decision.run_source_oof
    assert decision.selected_action == FIELD_BASE_ACTION
    assert "below_32" in decision.reason


@pytest.mark.parametrize(
    ("pixel_ids", "primitive_ids", "weights", "message"),
    [
        (torch.tensor([64]), torch.tensor([0]), torch.ones(1), "outside the source raster"),
        (torch.tensor([0]), torch.tensor([4]), torch.ones(1), "absent from hierarchy"),
        (torch.tensor([0]), torch.tensor([0]), torch.tensor([0.0]), "interval"),
        (torch.tensor([0]), torch.tensor([0]), torch.tensor([float("nan")]), "interval"),
        (torch.tensor([0]), torch.tensor([0]), torch.tensor([1.1]), "interval"),
        (
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            torch.ones(2),
            "repeat pixel/primitive",
        ),
    ],
)
def test_malformed_exact_triplets_fail_closed(pixel_ids, primitive_ids, weights, message):
    with pytest.raises(ValueError, match=message):
        _build(
            pixel_ids,
            primitive_ids,
            weights,
            height=8,
            width=8,
            rows=torch.arange(4),
        )


def test_exact_front_to_back_pixel_mass_cannot_exceed_one():
    with pytest.raises(ValueError, match="mass must not exceed one per pixel"):
        _build(
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            torch.tensor([0.6, 0.6]),
            height=8,
            width=8,
            rows=torch.arange(2),
        )


def test_row_alignment_sha_and_tensor_tamper_fail_closed():
    with pytest.raises(ValueError, match="unknown exact"):
        build_source_raster_dominant_footprint_authority(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.ones(1),
            height=8,
            width=8,
            hierarchy_primitive_rows=torch.arange(2),
            primitive_id_domain="global_rows",
            source_triplet_authority_sha256=TRIPLET_SHA,
            expected_source_triplet_authority_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="unique, and sorted"):
        _build(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.ones(1),
            height=8,
            width=8,
            rows=torch.tensor([1, 0]),
        )

    authority = _build(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.ones(2),
        height=8,
        width=8,
        rows=torch.arange(2),
    )
    changed_groups = authority.group_ids.clone()
    changed_groups[0] = 2
    with pytest.raises(ValueError, match="tensor hash"):
        dataclasses.replace(authority, group_ids=changed_groups)
    with pytest.raises(ValueError, match="unknown source-footprint"):
        clear_four_source_evidence_for_fold(
            authority,
            torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            heldout_fold=0,
            expected_authority_sha256="c" * 64,
        )

    mutable = _build(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.ones(2),
        height=8,
        width=8,
        rows=torch.arange(2),
    )
    mutable.visible_mass[0] = 9.0
    with pytest.raises(ValueError, match="dominant footprint mass|purity differs|tensor hash"):
        mutable.validate(expected_authority_sha256=mutable.authority_sha256)
