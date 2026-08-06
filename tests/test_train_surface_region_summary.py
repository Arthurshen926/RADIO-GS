from argparse import Namespace
import hashlib
from pathlib import Path

import pytest
import torch

from radio_gs.scripts.train_surface_region_summary_readout import (
    FIXED_SPARSE_VALIDATION_EPOCH,
    FIXED_SPARSE_VALIDATION_SEED,
    SPARSE_SUPPORT_AUGMENTATION_DEFAULT,
    _center_target_blind_text_bank,
    _fixed_sparse_validation_report,
    _build_versioned_readout,
    _completion_validation_views,
    _eligibility_completion_training_rows,
    _load,
    _paths,
    _scene_complete_epoch_batches,
    _seed_training,
    _sparsify_inputs,
    _targets,
    _training_epoch_order,
    _validate_sparse_support_inputs,
    _view_set_token_loss,
    inject_tangent_direction_noise,
    train,
)
from radio_gs.interfaces.surface_region_contract import (
    SurfaceRegionContractV2,
    SurfaceRegionContractV3,
    SurfaceRegionContractV4,
)
from radio_gs.interfaces.surface_region_summary import (
    JOINT_CONTEXT_POOLING,
    SURFACE_REGION_V3_GATED_RAW_PRIOR,
    SURFACE_REGION_V3_LEGACY_RAW_BASE,
    SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION,
    SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION,
    SurfaceRegionSummaryReadoutV2,
    SurfaceRegionSummaryReadoutV3,
)
from radio_gs.training.surface_region_eligibility_completion import (
    STRUCTURED_ELIGIBILITY_POLICY,
)


def _cache(path: Path, role: str, scene: str) -> None:
    contract = SurfaceRegionContractV2(minimum_tokens=1, maximum_tokens=4)
    torch.save({
        "radio_features": torch.randn(2, 4, 1280), "geometry": torch.randn(2, 4, 14),
        "token_mask": torch.ones(2, 4, dtype=torch.bool), "reliability": torch.ones(2, 4, 1),
        "anchor_index": torch.tensor([0, 1]),
        "official_summary_tokens": torch.randn(2, 3, 1280),
        "official_crop_summaries": torch.randn(2, 3, 1536),
        "teacher_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        "metadata": {"schema_version": 3, "split_role": role, "scene_names": [scene],
            "split_file_sha256": "abc", "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False, "annotations_opened": False,
            "labels_opened": False, "instances_opened": False, "text_opened": False,
            "masks_opened": False,
            "radio_checkpoint_sha256": "radio",
            "region_contract": contract.to_dict(),
            "region_contract_version": contract.version,
            "region_contract_sha256": contract.digest},
    }, path)


def _cache_v3(path: Path, role: str, scene: str) -> None:
    contract = SurfaceRegionContractV3(minimum_tokens=1, maximum_tokens=4)
    torch.manual_seed(101 if role == "train" else 103)
    token_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    support_fill = torch.tensor(
        [[False, False, True, False], [False, False, True, True]]
    )
    features = torch.nn.functional.normalize(
        torch.randn(2, 4, 1280), dim=-1
    ).masked_fill(~token_mask[..., None], 0.0)
    geometry = torch.zeros(2, 4, 16)
    geometry[..., 6] = token_mask.float()
    geometry[:, 0, 8] = 1.0
    geometry[:, 1, 9] = 1.0
    geometry[..., 14] = support_fill.float()
    # index 15 is log(raw L2 norm); zero reconstructs unit amplitude.
    geometry = geometry.masked_fill(~token_mask[..., None], 0.0)
    reliability = token_mask[..., None].float()
    records = [
        {
            "scene": scene,
            "region_id": f"{scene}-region-{row}",
            "tokens": int(token_mask[row].sum()),
            "support_fill_tokens": int(support_fill[row].sum()),
            "semantic_tokens": int(
                token_mask[row].sum() - support_fill[row].sum()
            ),
            "minimum_satisfied": True,
        }
        for row in range(2)
    ]
    torch.save(
        {
            "radio_features": features,
            "geometry": geometry,
            "token_mask": token_mask,
            "support_fill_mask": support_fill,
            "reliability": reliability,
            "anchor_index": torch.tensor([0, 0]),
            "official_summary_tokens": torch.randn(2, 3, 1280),
            "official_crop_summaries": torch.randn(2, 3, 1536),
            "teacher_mask": torch.tensor(
                [[1, 1, 0], [1, 1, 1]], dtype=torch.bool
            ),
            "metadata": {
                "schema_version": 4,
                "split_role": role,
                "scene_names": [scene],
                "region_records": records,
                "split_file_sha256": "v3-split",
                "uses_benchmark_scenes": False,
                "uses_benchmark_test_vocabulary": False,
                "annotations_opened": False,
                "labels_opened": False,
                "instances_opened": False,
                "masks_opened": False,
                "text_opened": False,
                "radio_checkpoint_sha256": "radio",
                "region_contract": contract.to_dict(),
                "region_contract_version": contract.version,
                "region_contract_sha256": contract.digest,
            },
        },
        path,
    )


def _cache_v3_completion(path: Path, role: str, scene: str) -> None:
    _cache_v3(path, role, scene)
    payload = torch.load(path)
    completion_contract = SurfaceRegionContractV3(
        minimum_tokens=3, maximum_tokens=4
    )
    payload["metadata"]["region_contract"] = completion_contract.to_dict()
    payload["metadata"]["region_contract_version"] = completion_contract.version
    payload["metadata"]["region_contract_sha256"] = completion_contract.digest
    row_index = torch.tensor([0, 0])
    for key in (
        "radio_features",
        "geometry",
        "token_mask",
        "support_fill_mask",
        "reliability",
        "anchor_index",
        "official_summary_tokens",
        "official_crop_summaries",
        "teacher_mask",
    ):
        payload[key] = payload[key][row_index].clone()
    original = payload["metadata"]["region_records"][:1]
    records = []
    for base_index, source in enumerate(original):
        full_id = str(source["region_id"])
        teacher = {
            "seed": base_index,
            "physical_radius_m": 0.25,
            "teacher_support_sha256": f"{base_index + 1:064x}",
            "teacher_target_sha256": f"{base_index + 11:064x}",
        }
        records.append(
            {
                **source,
                **teacher,
                "row_role": "full_support",
                "paired_full_region_id": full_id,
                "eligibility_variants_per_teacher_region": 1,
                "eligibility_variant_index": -1,
                "eligibility_sha256": "",
            }
        )
        records.append(
            {
                **source,
                **teacher,
                "region_id": f"{full_id}-completion-0",
                "row_role": "eligibility_completion",
                "paired_full_region_id": full_id,
                "eligibility_variants_per_teacher_region": 1,
                "eligibility_variant_index": 0,
                "eligibility_sha256": f"{base_index + 21:064x}",
                "eligibility_policy": STRUCTURED_ELIGIBILITY_POLICY,
                "eligibility_semantic_eligible_tokens": 2,
                "eligibility_nominal_semantic_keep_tokens": 2,
                "eligibility_expected_fill_tokens": 1,
                "eligibility_extreme_graph_fallback": False,
                "eligibility_extreme_graph_fallback_reason": "",
            }
        )
    payload["metadata"]["region_records"] = records
    completion_rows = records[1::2]
    payload["metadata"]["eligibility_completion"] = {
        "schema_version": 1,
        "policy": STRUCTURED_ELIGIBILITY_POLICY,
        "variants_per_teacher_region": 1,
        "nominal_semantic_keep_tokens": 2,
        "nominal_support_fill_tokens": 1,
        "full_support_rows": 1,
        "completion_variant_rows": 1,
        "completion_rows_with_fill": 1,
        "extreme_graph_fallback_rows": 0,
        "completion_support_fill_tokens": sum(
            int(record["support_fill_tokens"])
            for record in completion_rows
        ),
        "completion_selected_tokens": sum(
            int(record["tokens"]) for record in completion_rows
        ),
        "validation_checkpoint_selection": "full_support_rows_only",
    }
    torch.save(payload, path)


def test_load_and_multiview_targets(tmp_path: Path) -> None:
    path = tmp_path / "train.pt"; _cache(path, "train", "scene1")
    data, meta = _load([path], "train")
    token, descriptor, views, mask = _targets(data, torch.tensor([0, 1]))
    assert token.shape == (2, 1280) and descriptor.shape == (2, 1536)
    assert views.shape == (2, 3, 1536) and mask.shape == (2, 3)
    assert meta["scenes"] == ["scene1"]
    assert data["scene_ids"] == ["scene1", "scene1"]
    assert "region_ids" not in data


def test_view_set_token_loss_is_permutation_padding_and_view_count_invariant() -> None:
    predicted = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [100.0, -100.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])
    loss = _view_set_token_loss(predicted, teacher, mask)
    # Row losses are 0.5 and 1/3; rows, not valid views, are equally weighted.
    assert torch.allclose(loss, torch.tensor((0.5 + 1.0 / 3.0) / 2.0))

    permutation = torch.tensor([2, 0, 1])
    assert torch.equal(
        loss,
        _view_set_token_loss(
            predicted,
            teacher[:, permutation],
            mask[:, permutation],
        ),
    )

    changed_padding = teacher.clone()
    changed_padding[0, 2] = torch.tensor([-1.0e6, 1.0e6])
    assert torch.equal(
        loss,
        _view_set_token_loss(predicted, changed_padding, mask),
    )


def test_centered_target_blind_bank_is_deterministic_and_unit_norm() -> None:
    source = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    )
    centered, record = _center_target_blind_text_bank(source)
    repeated, repeated_record = _center_target_blind_text_bank(source.clone())
    assert torch.equal(centered, repeated)
    assert record == repeated_record
    assert torch.allclose(
        torch.linalg.vector_norm(centered, dim=-1),
        torch.ones(3),
    )
    expected = torch.nn.functional.normalize(source, dim=-1)
    expected = torch.nn.functional.normalize(
        expected - expected.mean(dim=0, keepdim=True), dim=-1
    )
    assert torch.allclose(centered, expected)
    assert record["gauge"] == (
        "normalize(l2_text_direction_minus_bank_mean)_v1"
    )


def test_scene_complete_batches_never_split_a_scene() -> None:
    scene_ids = ["a", "a", "b", "b", "b", "c", "c"]
    batches = _scene_complete_epoch_batches(
        torch.arange(len(scene_ids)),
        scene_ids,
        target_rows=3,
        generator=torch.Generator().manual_seed(17),
    )
    assert sorted(torch.cat(batches).tolist()) == list(range(len(scene_ids)))
    row_to_batch = {
        int(row): batch_index
        for batch_index, batch in enumerate(batches)
        for row in batch.tolist()
    }
    for scene in set(scene_ids):
        assert len(
            {
                row_to_batch[index]
                for index, value in enumerate(scene_ids)
                if value == scene
            }
        ) == 1


def test_version_aware_load_accepts_strict_v3_cache(tmp_path: Path) -> None:
    path = tmp_path / "v3.pt"
    _cache_v3(path, "train", "v3-scene")

    data, metadata = _load([path], "train", derive_region_ids=True)

    assert metadata["region_contract_version"] == "surface-region-contract-v3"
    assert data["geometry"].shape == (2, 4, 16)
    assert torch.equal(
        data["support_fill_mask"], data["geometry"][..., 14] > 0.5
    )
    assert data["region_ids"] == ["v3-scene-region-0", "v3-scene-region-1"]


def test_version_aware_load_accepts_sha_bound_v4_tensor_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v4.pt"
    _cache_v3(path, "validation", "v4-scene")
    payload = torch.load(path)
    contract = SurfaceRegionContractV4(minimum_tokens=1, maximum_tokens=4)
    payload["metadata"].update(
        {
            "region_contract": contract.to_dict(),
            "region_contract_version": contract.version,
            "region_contract_sha256": contract.digest,
        }
    )
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    data, metadata = _load(
        [path],
        "validation",
        expected_sha256=[digest],
    )

    assert metadata["region_contract_version"] == "surface-region-contract-v4"
    assert metadata["cache_artifacts"] == [
        {"path": str(path.resolve()), "sha256": digest}
    ]
    assert data["geometry"].shape[-1] == 16

    payload["radio_features"][0, 0, 0] += 0.01
    torch.save(payload, path)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        _load([path], "validation", expected_sha256=[digest])


def test_completion_pairs_are_strict_and_validation_authority_is_full_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paired-v3.pt"
    _cache_v3_completion(path, "validation", "v3-scene")
    data, metadata = _load([path], "validation", derive_region_ids=True)
    full, completion = _completion_validation_views(data)

    assert metadata["eligibility_completion"] == {
        "enabled": True,
        "schema_version": 1,
        "policy": STRUCTURED_ELIGIBILITY_POLICY,
        "variants_per_teacher_region": 1,
        "validation_checkpoint_selection": "full_support_rows_only",
    }
    assert len(data["radio_features"]) == 2
    assert len(full["radio_features"]) == 1
    assert completion is not None and len(completion["radio_features"]) == 1
    assert full["row_roles"] == ["full_support"]
    assert completion["row_roles"] == ["eligibility_completion"]
    assert torch.equal(
        full["official_summary_tokens"],
        completion["official_summary_tokens"],
    )

    payload = torch.load(path)
    payload["official_summary_tokens"][1, 0, 0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="exact teacher tensors"):
        _load([path], "validation")


def test_completion_training_weight_zero_selects_only_full_rows_and_discloses_counts(
) -> None:
    data = {
        "radio_features": torch.empty(4, 1, 1),
        "row_roles": [
            "full_support",
            "eligibility_completion",
            "full_support",
            "eligibility_completion",
        ],
    }

    default_rows, default_contract = _eligibility_completion_training_rows(
        data,
        completion_weight=1.0,
    )
    controlled_rows, controlled_contract = (
        _eligibility_completion_training_rows(
            data,
            completion_weight=0.0,
        )
    )

    assert default_rows.tolist() == [0, 1, 2, 3]
    assert controlled_rows.tolist() == [0, 2]
    assert default_contract["completion_rows_sampled_per_epoch"] == 2
    assert controlled_contract == {
        "schema_version": 1,
        "purpose": "query_free_generic_diagnostic",
        "sampling": (
            "uniform_without_replacement_over_positive_weight_rows_v1"
        ),
        "requested_completion_training_weight": 0.0,
        "full_support_sampling_weight": 1.0,
        "completion_sampling_weight": 0.0,
        "full_support_rows_available": 2,
        "completion_rows_available": 2,
        "full_support_rows_sampled_per_epoch": 2,
        "completion_rows_sampled_per_epoch": 0,
        "total_rows_sampled_per_epoch": 2,
        "paired_rows_have_equal_sampling_weight": False,
        "validation_checkpoint_selection": "full_support_rows_only",
        "completion_validation_authority": (
            "diagnostic_robustness_gate_only"
        ),
    }

    # The default path consumes the RNG exactly as the historical direct
    # randperm(total_rows), preserving V2 seeded behavior bit-for-bit.
    observed_generator = torch.Generator().manual_seed(19)
    expected_generator = torch.Generator().manual_seed(19)
    assert torch.equal(
        _training_epoch_order(
            default_rows,
            total_rows=4,
            generator=observed_generator,
        ),
        torch.randperm(4, generator=expected_generator),
    )
    controlled_order = _training_epoch_order(
        controlled_rows,
        total_rows=4,
        generator=torch.Generator().manual_seed(19),
    )
    assert sorted(controlled_order.tolist()) == [0, 2]


def test_loader_treats_missing_or_true_masks_opened_as_contamination(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mask-provenance.pt"
    _cache(path, "train", "scene1")
    payload = torch.load(path)
    payload["metadata"].pop("masks_opened")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="query-free scene-disjoint"):
        _load([path], "train")
    payload["metadata"]["masks_opened"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="query-free scene-disjoint"):
        _load([path], "train")


@pytest.mark.parametrize(
    "violation,match",
    [
        ("schema", "wrong 3-D cache schema"),
        ("geometry_dim", "geometry-16"),
        ("support_fill", "support_fill_mask"),
        ("padding", "padding must be exactly zero"),
        ("gauge", "unit L2 direction gauge"),
        ("minimum", "minimum_satisfied"),
    ],
)
def test_version_aware_load_rejects_invalid_v3_cache(
    tmp_path: Path,
    violation: str,
    match: str,
) -> None:
    path = tmp_path / f"bad-{violation}.pt"
    _cache_v3(path, "train", "v3-scene")
    payload = torch.load(path)
    if violation == "schema":
        payload["metadata"]["schema_version"] = 3
    elif violation == "geometry_dim":
        payload["geometry"] = payload["geometry"][..., :15]
    elif violation == "support_fill":
        payload["support_fill_mask"][0, 2] = False
    elif violation == "padding":
        payload["radio_features"][0, 3, 0] = 1.0
    elif violation == "gauge":
        payload["radio_features"][0, 0] *= 2.0
    else:
        payload["metadata"]["region_records"][0]["minimum_satisfied"] = False
    torch.save(payload, path)

    with pytest.raises(ValueError, match=match):
        _load([path], "train")


def test_training_rejects_mixed_v2_v3_contract_versions(tmp_path: Path) -> None:
    train_cache = tmp_path / "train-v2.pt"
    val_cache = tmp_path / "validation-v3.pt"
    _cache(train_cache, "train", "v2-train-scene")
    _cache_v3(val_cache, "validation", "v3-validation-scene")

    with pytest.raises(ValueError, match="contract versions differ"):
        train(
            Namespace(
                train_caches=str(train_cache),
                validation_caches=str(val_cache),
                sparse_support_augmentation=False,
            )
        )


def test_load_prefers_existing_region_ids(tmp_path: Path) -> None:
    path = tmp_path / "train.pt"
    _cache(path, "train", "scene1")
    payload = torch.load(path)
    payload["metadata"]["region_records"] = [
        {"scene": "scene1", "region_id": "canonical-a"},
        {"scene": "scene1", "region_id": "canonical-b"},
    ]
    torch.save(payload, path)

    data, _meta = _load([path], "train", derive_region_ids=True)

    assert data["region_ids"] == ["canonical-a", "canonical-b"]


def test_derived_region_ids_do_not_depend_on_cache_row_order(
    tmp_path: Path,
) -> None:
    original_path = tmp_path / "original.pt"
    reordered_path = tmp_path / "reordered.pt"
    _cache(original_path, "train", "scene1")
    original_payload = torch.load(original_path)
    reordered_payload = torch.load(original_path)
    tensor_keys = (
        "radio_features",
        "geometry",
        "token_mask",
        "reliability",
        "anchor_index",
        "official_summary_tokens",
        "official_crop_summaries",
        "teacher_mask",
    )
    for key in tensor_keys:
        reordered_payload[key] = reordered_payload[key].flip(0)
    torch.save(reordered_payload, reordered_path)

    original, _ = _load(
        [original_path], "train", derive_region_ids=True
    )
    reordered, _ = _load(
        [reordered_path], "train", derive_region_ids=True
    )

    assert reordered["region_ids"] == list(reversed(original["region_ids"]))
    # The test cache itself remains unchanged; identities derive only from
    # canonical content rather than the file or row containing that content.
    assert torch.equal(
        original_payload["radio_features"],
        torch.load(original_path)["radio_features"],
    )


def test_load_recovers_legacy_multiscene_row_bindings(tmp_path: Path) -> None:
    path = tmp_path / "train.pt"
    _cache(path, "train", "scene1")
    payload = torch.load(path)
    payload["metadata"]["scene_names"] = ["scene1", "scene2"]
    payload["metadata"]["region_records"] = [
        {"scene": "scene2"},
        {"scene": "scene1"},
    ]
    torch.save(payload, path)

    data, meta = _load([path], "train")

    assert meta["scenes"] == ["scene1", "scene2"]
    assert data["scene_ids"] == ["scene2", "scene1"]


@pytest.mark.parametrize(
    "records,match",
    [
        ([{"scene": "scene1"}], "misaligned legacy region records"),
        (
            [{"scene": "scene1"}, {"scene": "unknown"}],
            "invalid row-to-scene bindings",
        ),
    ],
)
def test_load_rejects_invalid_legacy_multiscene_row_bindings(
    tmp_path: Path,
    records: list[dict[str, str]],
    match: str,
) -> None:
    path = tmp_path / "train.pt"
    _cache(path, "train", "scene1")
    payload = torch.load(path)
    payload["metadata"]["scene_names"] = ["scene1", "scene2"]
    payload["metadata"]["region_records"] = records
    torch.save(payload, path)

    with pytest.raises(ValueError, match=match):
        _load([path], "train")


def test_cache_paths_accept_absolute_globs(tmp_path: Path) -> None:
    first = tmp_path / "train_shard0.pt"
    second = tmp_path / "train_shard1.pt"
    first.touch()
    second.touch()

    assert _paths(str(tmp_path / "train_shard*.pt")) == [
        first,
        second,
    ]


def test_training_seed_covers_model_initialization_and_random_stream() -> None:
    assert SPARSE_SUPPORT_AUGMENTATION_DEFAULT is False
    first_generator = _seed_training(17, device="cpu")
    first_model = SurfaceRegionSummaryReadoutV2(
        feature_dim=8,
        hidden_dim=4,
    )
    first_random = torch.randn(5, generator=first_generator)

    second_generator = _seed_training(17, device="cpu")
    second_model = SurfaceRegionSummaryReadoutV2(
        feature_dim=8,
        hidden_dim=4,
    )
    second_random = torch.randn(5, generator=second_generator)

    for name, first_value in first_model.state_dict().items():
        torch.testing.assert_close(
            first_value,
            second_model.state_dict()[name],
        )
    torch.testing.assert_close(first_random, second_random)
    with pytest.raises(ValueError, match="non-negative"):
        _seed_training(-1, device="cpu")


def test_load_rejects_annotation_access(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"; _cache(path, "train", "scene1")
    payload = torch.load(path); payload["metadata"]["labels_opened"] = True; torch.save(payload, path)
    with pytest.raises(ValueError): _load([path], "train")


def test_canonical_noise_stays_on_unit_sphere_and_preserves_padding() -> None:
    values = torch.nn.functional.normalize(torch.randn(2, 4, 16), dim=-1)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    perturbed = inject_tangent_direction_noise(values, mask, angle_degrees=8.0)
    torch.testing.assert_close(
        perturbed[mask].norm(dim=-1), torch.ones(int(mask.sum())), atol=1e-5, rtol=1e-5
    )
    assert torch.equal(perturbed[~mask], torch.zeros_like(perturbed[~mask]))


def test_sparsify_inputs_synchronizes_mask_and_all_token_tensors() -> None:
    data = {
        "radio_features": torch.arange(2 * 8 * 3).reshape(2, 8, 3).float() + 1,
        "geometry": torch.ones(2, 8, 2),
        "reliability": torch.ones(2, 8, 1),
        "token_mask": torch.ones(2, 8, dtype=torch.bool),
        "anchor_index": torch.tensor([0, 3]),
        "region_ids": ["region-a", "region-b"],
    }
    features, geometry, reliability, selection = _sparsify_inputs(
        data,
        torch.tensor([1, 0]),
        torch.device("cpu"),
        minimum_tokens=2,
        seed=5,
        epoch=7,
    )

    assert selection.token_mask[0, 3]
    dropped = ~selection.token_mask
    assert not features[dropped].any()
    assert not geometry[dropped].any()
    assert not reliability[dropped].any()
    assert features[selection.token_mask].all()
    assert geometry[selection.token_mask].all()
    assert reliability[selection.token_mask].all()


def test_sparse_input_validation_rejects_duplicate_ids_or_short_support() -> None:
    valid = {
        "radio_features": torch.ones(2, 4, 3),
        "token_mask": torch.ones(2, 4, dtype=torch.bool),
        "region_ids": ["a", "b"],
    }
    _validate_sparse_support_inputs(valid, minimum_tokens=3, label="test")
    duplicate = {**valid, "region_ids": ["a", "a"]}
    with pytest.raises(ValueError, match="duplicate"):
        _validate_sparse_support_inputs(
            duplicate, minimum_tokens=3, label="test"
        )
    short = {
        **valid,
        "token_mask": torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
    }
    with pytest.raises(ValueError, match="below the sparse minimum"):
        _validate_sparse_support_inputs(short, minimum_tokens=3, label="test")


def test_fixed_sparse_validation_gate_is_parameter_free_and_explicit() -> None:
    baseline_full = {
        "summary_token_cosine": 0.5,
        "mean_descriptor_cosine": 0.6,
        "all_view_descriptor_cosine": 0.4,
    }
    baseline_sparse = {
        "summary_token_cosine": 0.45,
        "mean_descriptor_cosine": 0.55,
        "all_view_descriptor_cosine": 0.35,
    }
    candidate_full = {
        "summary_token_cosine": 0.7,
        "mean_descriptor_cosine": 0.7,
        "all_view_descriptor_cosine": 0.6,
    }
    candidate_sparse = {
        "summary_token_cosine": 0.68,
        "mean_descriptor_cosine": 0.68,
        "all_view_descriptor_cosine": 0.58,
    }

    report = _fixed_sparse_validation_report(
        minimum_tokens=24,
        baseline_full=baseline_full,
        baseline_sparse=baseline_sparse,
        candidate_full=candidate_full,
        candidate_sparse=candidate_sparse,
    )

    assert report["sampling"]["seed"] == FIXED_SPARSE_VALIDATION_SEED
    assert report["sampling"]["epoch"] == FIXED_SPARSE_VALIDATION_EPOCH
    assert report["sampling"]["checkpoint_selection_support"] == "full"
    assert report["gate_passed"] is True
    degraded = {
        **candidate_sparse,
        "mean_descriptor_cosine": 0.2,
        "all_view_descriptor_cosine": 0.2,
    }
    failed = _fixed_sparse_validation_report(
        minimum_tokens=24,
        baseline_full=baseline_full,
        baseline_sparse=baseline_sparse,
        candidate_full=candidate_full,
        candidate_sparse=degraded,
    )
    assert failed["gate_passed"] is False


def test_sparse_training_integration_keeps_full_validation_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    radio = tmp_path / "radio.pt"
    radio.write_bytes(b"frozen-radio-test")
    radio_sha256 = hashlib.sha256(radio.read_bytes()).hexdigest()
    train_cache = tmp_path / "train.pt"
    val_cache = tmp_path / "validation.pt"
    _cache(train_cache, "train", "train-scene")
    _cache(val_cache, "validation", "validation-scene")
    for path in (train_cache, val_cache):
        payload = torch.load(path)
        payload["metadata"]["radio_checkpoint_sha256"] = radio_sha256
        torch.save(payload, path)

    class _Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(1280, 1536, bias=False)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.projection(values)

    monkeypatch.setattr(
        "radio_gs.scripts.train_surface_region_summary_readout."
        "SigLIP2SummaryHead.from_radio_checkpoint",
        lambda _path: _Head(),
    )
    output = tmp_path / "sparse.pt"
    report = train(
        Namespace(
            train_caches=str(train_cache),
            validation_caches=str(val_cache),
            output=str(output),
            hidden_dim=4,
            epochs=1,
            patience=0,
            batch_size=2,
            learning_rate=2e-4,
            weight_decay=1e-4,
            token_weight=0.25,
            relation_weight=0.1,
            reliability_attention_mode="log_prior",
            context_pooling_mode="joint_attention_v1",
            canonical_noise_degrees=0.0,
            canonical_noise_calibration="test",
            sparse_support_augmentation=True,
            seed=0,
            device="cpu",
            radio_checkpoint=str(radio),
        )
    )

    checkpoint = torch.load(output)
    restored_v2, _ = SurfaceRegionSummaryReadoutV2.from_checkpoint(output)
    assert isinstance(restored_v2, SurfaceRegionSummaryReadoutV2)
    assert checkpoint["schema_version"] == 3
    assert (
        checkpoint["provenance"]["training_scope"]
        == "global_cross_scene_3d_surface_v2"
    )
    assert report["sparse_support_augmentation"]["enabled"] is True
    assert (
        report["sparse_validation"]["sampling"][
            "checkpoint_selection_support"
        ]
        == "full"
    )
    assert checkpoint["history"][0]["sparse_support_mean_kept_tokens"] >= 1
    assert checkpoint["sparse_validation"] == report["sparse_validation"]


def test_v3_gated_training_integration_records_schema8_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    radio = tmp_path / "radio.pt"
    radio.write_bytes(b"frozen-radio-v3-test")
    radio_sha256 = hashlib.sha256(radio.read_bytes()).hexdigest()
    train_cache = tmp_path / "train-v3.pt"
    val_cache = tmp_path / "validation-v3.pt"
    _cache_v3_completion(train_cache, "train", "v3-train-scene")
    _cache_v3_completion(
        val_cache,
        "validation",
        "v3-validation-scene",
    )
    for path in (train_cache, val_cache):
        payload = torch.load(path)
        payload["metadata"]["radio_checkpoint_sha256"] = radio_sha256
        torch.save(payload, path)

    class _Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(1280, 1536, bias=False)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.projection(values)

    monkeypatch.setattr(
        "radio_gs.scripts.train_surface_region_summary_readout."
        "SigLIP2SummaryHead.from_radio_checkpoint",
        lambda _path: _Head(),
    )
    # These are V2-facing CLI values.  V3 must not inherit either one.
    args = Namespace(
        train_caches=str(train_cache),
        validation_caches=str(val_cache),
        output=str(tmp_path / "readout-v3.pt"),
        hidden_dim=4,
        epochs=1,
        patience=0,
        batch_size=2,
        learning_rate=2e-4,
        weight_decay=1e-4,
        token_weight=0.25,
        relation_weight=0.1,
        reliability_attention_mode="log_prior",
        context_pooling_mode="core_context_separate_attention_v1",
        canonical_noise_degrees=0.0,
        canonical_noise_calibration="v3-test",
        sparse_support_augmentation=True,
        eligibility_completion_training_weight=0.0,
        v3_base_output_mode=SURFACE_REGION_V3_GATED_RAW_PRIOR,
        seed=0,
        device="cpu",
        radio_checkpoint=str(radio),
    )

    report = train(args)
    checkpoint = torch.load(args.output)
    restored, _ = SurfaceRegionSummaryReadoutV3.from_checkpoint(args.output)

    assert isinstance(restored, SurfaceRegionSummaryReadoutV3)
    assert checkpoint["schema_version"] == (
        SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION
    )
    assert report["checkpoint_schema_version"] == (
        SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION
    )
    assert report["training_scope"] == "global_cross_scene_3d_surface_v3"
    assert checkpoint["architecture"]["geometry_dim"] == 16
    assert (
        checkpoint["architecture"]["reliability_attention_mode"]
        == "input_only"
    )
    assert (
        checkpoint["architecture"]["context_pooling_mode"]
        == JOINT_CONTEXT_POOLING
    )
    assert checkpoint["architecture"]["base_output_mode"] == (
        SURFACE_REGION_V3_GATED_RAW_PRIOR
    )
    assert checkpoint["training_config"]["v3_base_output_mode"] == (
        SURFACE_REGION_V3_GATED_RAW_PRIOR
    )
    assert checkpoint["provenance"]["surface_region_v3"] == {
        "cache_geometry_dim": 16,
        "support_fill_mask_validated": True,
        "padding_exact_zero_validated": [
            "radio_features",
            "geometry",
            "reliability",
        ],
        "all_regions_minimum_satisfied": True,
        "feature_normalization": "l2_direction_plus_log_raw_norm_v1",
        "effective_reliability_attention_mode": "input_only",
        "effective_context_pooling_mode": JOINT_CONTEXT_POOLING,
        "effective_base_output_mode": SURFACE_REGION_V3_GATED_RAW_PRIOR,
    }
    assert checkpoint["sparse_validation"] is not None
    assert checkpoint["provenance"]["eligibility_completion_training"] == {
        "schema_version": 1,
        "purpose": "query_free_generic_diagnostic",
        "sampling": (
            "uniform_without_replacement_over_positive_weight_rows_v1"
        ),
        "requested_completion_training_weight": 0.0,
        "full_support_sampling_weight": 1.0,
        "completion_sampling_weight": 0.0,
        "full_support_rows_available": 1,
        "completion_rows_available": 1,
        "full_support_rows_sampled_per_epoch": 1,
        "completion_rows_sampled_per_epoch": 0,
        "total_rows_sampled_per_epoch": 1,
        "paired_rows_have_equal_sampling_weight": False,
        "validation_checkpoint_selection": "full_support_rows_only",
        "completion_validation_authority": (
            "diagnostic_robustness_gate_only"
        ),
        "enabled": True,
        "full_support_training_rows": 1,
        "completion_training_rows": 0,
        "completion_training_row_fraction": 0.0,
        "completion_training_fill_token_fraction": 0.0,
        "completion_cache_fill_token_fraction": 1.0 / 3.0,
        "epochs_trained": 1,
        "actual_full_support_row_samples": 1,
        "actual_completion_row_samples": 0,
    }
    assert report["eligibility_completion_training"] == checkpoint[
        "provenance"
    ]["eligibility_completion_training"]
    assert report["eligibility_completion_validation"] is not None
    assert report["eligibility_completion_validation"]["rows"] == 1


def test_v3_target_blind_response_training_records_centered_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    radio = tmp_path / "radio.pt"
    radio.write_bytes(b"frozen-radio-v3-response-test")
    radio_sha256 = hashlib.sha256(radio.read_bytes()).hexdigest()
    train_cache = tmp_path / "train-v3.pt"
    val_cache = tmp_path / "validation-v3.pt"
    _cache_v3(train_cache, "train", "response-train-scene")
    _cache_v3(val_cache, "validation", "response-validation-scene")
    for path in (train_cache, val_cache):
        payload = torch.load(path)
        payload["metadata"]["radio_checkpoint_sha256"] = radio_sha256
        torch.save(payload, path)

    class _Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(1280, 1536, bias=False)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.projection(values)

    monkeypatch.setattr(
        "radio_gs.scripts.train_surface_region_summary_readout."
        "SigLIP2SummaryHead.from_radio_checkpoint",
        lambda _path: _Head(),
    )
    generator = torch.Generator().manual_seed(211)
    banks = {
        "fit": torch.randn(4, 1536, generator=generator),
        "dev": torch.randn(3, 1536, generator=generator),
    }

    def _bank(_path, *, expected_sha256, expected_split):
        del expected_sha256
        return (
            torch.nn.functional.normalize(banks[expected_split], dim=-1),
            {"path": str(_path), "sha256": f"{expected_split}-sha"},
            tuple(f"{expected_split}-{index}" for index in range(len(banks[expected_split]))),
        )

    monkeypatch.setattr(
        "radio_gs.scripts.train_surface_region_summary_readout."
        "_load_target_blind_text_bank",
        _bank,
    )
    args = Namespace(
        train_caches=str(train_cache),
        validation_caches=str(val_cache),
        output=str(tmp_path / "response-readout-v3.pt"),
        hidden_dim=4,
        epochs=1,
        patience=0,
        batch_size=2,
        learning_rate=2e-4,
        weight_decay=1e-4,
        token_weight=0.25,
        relation_weight=0.1,
        reliability_attention_mode="log_prior",
        context_pooling_mode=JOINT_CONTEXT_POOLING,
        canonical_noise_degrees=0.0,
        canonical_noise_calibration="response-test",
        sparse_support_augmentation=False,
        eligibility_completion_training_weight=1.0,
        v3_base_output_mode=SURFACE_REGION_V3_LEGACY_RAW_BASE,
        fit_text_bank="fit.pt",
        fit_text_bank_sha256="fit-sha",
        validation_text_bank="dev.pt",
        validation_text_bank_sha256="dev-sha",
        text_response_gradient_ratio=0.25,
        text_response_warmup_epochs=0,
        text_response_selection_floor=0.0,
        text_response_temperature=0.05,
        seed=0,
        device="cpu",
        radio_checkpoint=str(radio),
    )

    report = train(args)
    checkpoint = torch.load(args.output)
    response = checkpoint["provenance"]["target_blind_text_response"]
    assert response["text_direction_gauge"] == (
        "normalize(l2_text_direction_minus_bank_mean)_v1"
    )
    assert response["token_target"] == (
        "equal_region_equal_valid_view_cosine_set_v1"
    )
    assert response["fit_text_bank"]["gauge"] == response[
        "text_direction_gauge"
    ]
    assert response["fit_dev_query_disjoint"] is True
    assert checkpoint["target_blind_generic_gate"] == report[
        "target_blind_generic_gate"
    ]
    assert "response_profile_cosine_p05" in report["validation"]


def test_versioned_readout_factory_preserves_v2_defaults_and_forces_v3() -> None:
    args = Namespace(
        hidden_dim=8,
        reliability_attention_mode="log_prior",
        context_pooling_mode="core_context_separate_attention_v1",
    )
    v2, v2_contract = _build_versioned_readout(
        args,
        contract_version="surface-region-contract-v2",
        device=torch.device("cpu"),
    )
    v3, v3_contract = _build_versioned_readout(
        args,
        contract_version="surface-region-contract-v3",
        device=torch.device("cpu"),
    )
    assert isinstance(v2, SurfaceRegionSummaryReadoutV2)
    assert v2.reliability_attention_mode == "log_prior"
    assert v2.context_pooling_mode == "core_context_separate_attention_v1"
    assert v2_contract["checkpoint_schema_version"] == 3
    assert isinstance(v3, SurfaceRegionSummaryReadoutV3)
    assert v3.reliability_attention_mode == "input_only"
    assert v3.context_pooling_mode == JOINT_CONTEXT_POOLING
    assert v3.base_output_mode == SURFACE_REGION_V3_LEGACY_RAW_BASE
    assert v3_contract["checkpoint_schema_version"] == (
        SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION
    )
    assert "effective_base_output_mode" not in v3_contract

    args.v3_base_output_mode = SURFACE_REGION_V3_GATED_RAW_PRIOR
    gated, gated_contract = _build_versioned_readout(
        args,
        contract_version="surface-region-contract-v3",
        device=torch.device("cpu"),
    )
    assert isinstance(gated, SurfaceRegionSummaryReadoutV3)
    assert gated.base_output_mode == SURFACE_REGION_V3_GATED_RAW_PRIOR
    assert gated_contract["checkpoint_schema_version"] == (
        SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION
    )
    assert gated_contract["effective_base_output_mode"] == (
        SURFACE_REGION_V3_GATED_RAW_PRIOR
    )
