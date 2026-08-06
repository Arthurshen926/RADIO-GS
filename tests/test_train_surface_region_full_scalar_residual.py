from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces.surface_region_full_scalar_contract import (
    apply_full_scalar_normalization,
    build_full_scalar_normalization_authority,
)
from radio_gs.models.surface_region_dual_descriptor import (
    SurfaceRegionAcceptedV2FullScalarResidualV1,
)
from radio_gs.scripts import train_surface_region_full_scalar_residual as trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SOURCE_COHORT_SHA = "a" * 64
COHORT_FILE_SHA = "b" * 64
BENCHMARK_SOURCE_SHA = "e" * 64
TRAIN_SCENES = [f"scene{1000 + index:04d}_00" for index in range(24)]
VALIDATION_SCENES = [f"scene{2000 + index:04d}_00" for index in range(8)]
BENCHMARK_SCENES = [f"scene{3000 + index:04d}_01" for index in range(4)]


def _authority_payload(
    *,
    train_scenes: list[str] | None = None,
    validation_scenes: list[str] | None = None,
    exclusion_authority_sha: str = "e" * 64,
    exclusion_file_sha: str = "f" * 64,
) -> dict:
    payload = {
        "schema": trainer.COHORT_AUTHORITY_SCHEMA,
        "schema_version": trainer.COHORT_AUTHORITY_SCHEMA_VERSION,
        "contract": trainer.cohort_authority_contract(),
        "contract_sha256": trainer.COHORT_AUTHORITY_CONTRACT_SHA256,
        "source_train_scene_ids": list(train_scenes or TRAIN_SCENES),
        "source_validation_scene_ids": list(
            validation_scenes or VALIDATION_SCENES
        ),
        "source_train_physical_space_ids": sorted(
            {
                trainer.canonical_physical_space_id(scene)
                for scene in (train_scenes or TRAIN_SCENES)
            }
        ),
        "source_validation_physical_space_ids": sorted(
            {
                trainer.canonical_physical_space_id(scene)
                for scene in (validation_scenes or VALIDATION_SCENES)
            }
        ),
        "benchmark_exclusion": {
            "manifest_authority_sha256": exclusion_authority_sha,
            "manifest_file_sha256": exclusion_file_sha,
        },
        "source_access": trainer._cohort_authority_access(),
    }
    payload["authority_sha256"] = trainer.cohort_authority_content_sha256(
        payload
    )
    return payload


COHORT_SHA = _authority_payload()["authority_sha256"]


def _payload(
    split: str,
    scene_ids: list[str],
    *,
    views: int = 3,
    scalar_offset: float = 0.0,
    teacher_sha: str = "d" * 64,
    source_manifest_file_sha: str = "c" * 64,
    teacher_manifest_file_sha: str = "9" * 64,
    cohort_sha: str = COHORT_SHA,
    cohort_file_sha: str = COHORT_FILE_SHA,
) -> dict:
    generator = torch.Generator().manual_seed(
        1200 + len(scene_ids) + (0 if split == "source_train" else 100)
    )
    rows = len(scene_ids)
    base = F.normalize(
        torch.randn(rows, trainer.DESCRIPTOR_DIM, generator=generator), dim=-1
    )
    scalars = (
        torch.randn(
            rows,
            trainer.SURFACE_REGION_FULL_SCALAR_DIM,
            generator=generator,
        )
        + float(scalar_offset)
    )
    eligible = torch.ones(rows, dtype=torch.bool)
    teachers = F.normalize(
        torch.randn(
            rows, views, trainer.DESCRIPTOR_DIM, generator=generator
        ),
        dim=-1,
    )
    mask = torch.ones(rows, views, dtype=torch.bool)
    if views > 1:
        mask[::2, -1] = False
        teachers[~mask] = 0.0
    occurrence: dict[str, int] = {}
    region_row_ids: list[str] = []
    for scene in scene_ids:
        index = occurrence.get(scene, 0)
        occurrence[scene] = index + 1
        region_row_ids.append(f"{split}:{scene}:region_{index:06d}")
    teacher_view_ids = [
        [
            f"{scene}:view_{view:04d}" if bool(mask[row, view]) else None
            for view in range(views)
        ]
        for row, scene in enumerate(scene_ids)
    ]
    pair_rows, pair_views = torch.where(mask)
    pair_descriptors = teachers[mask]
    pair_view_ids = [
        str(teacher_view_ids[int(row)][int(view)])
        for row, view in zip(pair_rows, pair_views)
    ]
    selected_by_scale = [rows]
    payload = {
        "schema": trainer.TRAINING_SHARD_SCHEMA,
        "schema_version": trainer.TRAINING_SHARD_SCHEMA_VERSION,
        "contract": trainer.training_shard_contract(),
        "contract_sha256": trainer.TRAINING_SHARD_CONTRACT_SHA256,
        "split": split,
        "accepted_v2_e0": base,
        "raw_full_scalar_summary": scalars,
        "eligible": eligible,
        "official_multiview_siglip2_teacher_pair_region_indices": pair_rows,
        "official_multiview_siglip2_teacher_pair_descriptors": pair_descriptors,
        "scene_ids": list(scene_ids),
        "region_row_ids": region_row_ids,
        "teacher_pair_view_ids": pair_view_ids,
        "sampling_audit": {
            "scene_id": scene_ids[0],
            "sampling_contract_sha256": trainer.SAMPLING_CONTRACT_SHA256,
            "canonical_region_indices_sha256": "a" * 64,
            "accepted_selection_audit": {
                "sampling_contract_sha256": trainer.SAMPLING_CONTRACT_SHA256,
                "canonical_candidate_region_count": rows,
                "exact_overlap_candidate_count": rows,
                "teacher_visible_candidate_count": rows,
                "selected_region_count": rows,
                "selected_count_by_scale": selected_by_scale,
            },
            "selected_region_count": rows,
            "pair_count": int(pair_rows.numel()),
            "maximum_views_per_region": int(mask.sum(dim=1).max()),
        },
        "lineage": {
            "accepted_v2_authority": trainer._accepted_v2_authority(),
            "source_state_cohort_authority_sha256": SOURCE_COHORT_SHA,
            "source_state_manifest_file_sha256": source_manifest_file_sha,
            "cohort_authority_sha256": cohort_sha,
            "cohort_authority_file_sha256": cohort_file_sha,
            "teacher_authority_sha256": teacher_sha,
            "teacher_manifest_file_sha256": teacher_manifest_file_sha,
        },
        "source_access": trainer._source_access(split),
    }
    payload["channel_sha256"] = trainer.training_shard_channel_sha256(payload)
    return payload


def _merged(
    split: str,
    scenes: list[str],
    *,
    rows_per_scene: int = 2,
    cohort_sha: str = COHORT_SHA,
    cohort_file_sha: str = COHORT_FILE_SHA,
):
    payloads = [
        _payload(
            split,
            [scene] * rows_per_scene,
            cohort_sha=cohort_sha,
            cohort_file_sha=cohort_file_sha,
        )
        for scene in scenes
    ]
    return trainer._pad_and_merge_shards(payloads, split=split)


def _freeze_manifest(path, payload: dict, validator):
    payload["authority_sha256"] = trainer._manifest_content_sha256(payload)
    write_frozen_json(path, payload)
    return trainer._load_json_manifest(
        path,
        expected_sha256=sha256_file(path),
        label=path.stem,
        validator=validator,
    )


def _materialized_authorities(tmp_path, train: dict, validation: dict):
    scene_ids = train["scene_ids"] + validation["scene_ids"]
    region_ids = train["region_row_ids"] + validation["region_row_ids"]
    view_ids = (
        trainer._sparse_teacher_view_ids(train)
        + trainer._sparse_teacher_view_ids(validation)
    )
    scenes = sorted(set(scene_ids))
    source_payload = {
        "schema": trainer.SOURCE_STATE_MANIFEST_SCHEMA,
        "schema_version": trainer.SOURCE_MANIFEST_SCHEMA_VERSION,
        "contract": trainer.source_state_manifest_contract(),
        "contract_sha256": trainer.SOURCE_STATE_MANIFEST_CONTRACT_SHA256,
        "scene_records": [
            {
                "scene_id": scene,
                "physical_space_id": trainer.canonical_physical_space_id(
                    scene
                ),
                "artifact_sha256": "1" * 64,
            }
            for scene in scenes
        ],
        "region_records": sorted(
            [
                {"region_row_id": row_id, "scene_id": scene}
                for row_id, scene in zip(region_ids, scene_ids)
            ],
            key=lambda item: item["region_row_id"],
        ),
        "source_access": trainer._source_manifest_access(),
    }
    source, source_file = _freeze_manifest(
        tmp_path / "source.json", source_payload, trainer.validate_source_state_manifest
    )
    teacher_payload = {
        "schema": trainer.TEACHER_MANIFEST_SCHEMA,
        "schema_version": trainer.SOURCE_MANIFEST_SCHEMA_VERSION,
        "contract": trainer.teacher_manifest_contract(),
        "contract_sha256": trainer.TEACHER_MANIFEST_CONTRACT_SHA256,
        "teacher_model_authority_sha256": "2" * 64,
        "region_view_records": sorted(
            [
                {
                    "region_row_id": row_id,
                    "scene_id": scene,
                    "teacher_view_ids": list(views),
                }
                for row_id, scene, views in zip(region_ids, scene_ids, view_ids)
            ],
            key=lambda item: item["region_row_id"],
        ),
        "source_access": trainer._source_manifest_access(),
    }
    teacher, teacher_file = _freeze_manifest(
        tmp_path / "teacher.json", teacher_payload, trainer.validate_teacher_manifest
    )
    exclusion_payload = {
        "schema": trainer.BENCHMARK_EXCLUSION_MANIFEST_SCHEMA,
        "schema_version": trainer.SOURCE_MANIFEST_SCHEMA_VERSION,
        "contract": trainer.benchmark_exclusion_manifest_contract(),
        "contract_sha256": trainer.BENCHMARK_EXCLUSION_MANIFEST_CONTRACT_SHA256,
        "source_identifier": "frozen_unit_test_benchmark_scene_manifest",
        "source_artifact_sha256": BENCHMARK_SOURCE_SHA,
        "scene_ids": list(BENCHMARK_SCENES),
        "scene_ids_sha256": canonical_json_sha256(BENCHMARK_SCENES),
        "physical_space_ids": sorted(
            {
                trainer.canonical_physical_space_id(scene)
                for scene in BENCHMARK_SCENES
            }
        ),
        "physical_space_ids_sha256": canonical_json_sha256(
            sorted(
                {
                    trainer.canonical_physical_space_id(scene)
                    for scene in BENCHMARK_SCENES
                }
            )
        ),
        "source_access": trainer._source_manifest_access(),
    }
    exclusion, exclusion_file = _freeze_manifest(
        tmp_path / "exclusion.json",
        exclusion_payload,
        trainer.validate_benchmark_exclusion_manifest,
    )
    authority_payload = _authority_payload(
        train_scenes=sorted(set(train["scene_ids"])),
        validation_scenes=sorted(set(validation["scene_ids"])),
        exclusion_authority_sha=exclusion["authority_sha256"],
        exclusion_file_sha=exclusion_file["sha256"],
    )
    authority_path = tmp_path / "cohort.json"
    write_frozen_json(authority_path, authority_payload)
    authority, authority_file = trainer.load_cohort_authority(
        authority_path, expected_sha256=sha256_file(authority_path)
    )
    for lineage in train["lineages"] + validation["lineages"]:
        lineage["cohort_authority_sha256"] = authority["authority_sha256"]
        lineage["cohort_authority_file_sha256"] = authority_file["sha256"]
        lineage["source_state_cohort_authority_sha256"] = source[
            "authority_sha256"
        ]
        lineage["source_state_manifest_file_sha256"] = source_file["sha256"]
        lineage["teacher_authority_sha256"] = teacher["authority_sha256"]
        lineage["teacher_manifest_file_sha256"] = teacher_file["sha256"]
    return (
        authority, authority_file, source, source_file, teacher, teacher_file,
        exclusion, exclusion_file,
    )


def _normalization(data: dict) -> dict:
    return build_full_scalar_normalization_authority(
        data["raw_full_scalar_summary"],
        data["eligible"],
        source_state_cohort_sha256=SOURCE_COHORT_SHA,
    )


def _model(normalization: dict):
    return SurfaceRegionAcceptedV2FullScalarResidualV1(
        descriptor_dim=trainer.DESCRIPTOR_DIM,
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
        max_angle_radians=trainer.MAX_ANGLE_RADIANS,
        max_alpha=trainer.MAX_ALPHA,
    )


def test_training_and_shard_contract_are_strictly_source_only() -> None:
    shard = trainer.training_shard_contract()
    contract = trainer.training_contract()

    assert shard["accepted_v2_authority"] == trainer._accepted_v2_authority()
    assert shard["benchmark_queries_opened"] is False
    assert shard["benchmark_labels_opened"] is False
    assert shard["target_heldout_opened"] is False
    assert contract["cohort"] == {
        "source_train_scene_count": 24,
        "source_validation_scene_count": 8,
        "scene_disjoint": True,
        "physical_space_id": "canonical_ScanNet_scene####",
        "one_scan_per_physical_space": True,
        "train_validation_physical_space_disjoint": True,
        "external_caller_sha_bound_authority_json": True,
        "actual_scene_sets_equal_authority": True,
        "benchmark_exclusion_list_verified": True,
        "benchmark_physical_space_exclusion_verified": True,
        "cohort_authority_identical_across_all_shards": True,
        "source_state_cohort_authority_identical_across_all_shards": True,
    }
    assert contract["model"]["trainable_object"] == (
        "content_plus_scalar_full_scalar_residual_only"
    )
    assert contract["normalization"]["validation_contribution"] is False
    assert contract["normalization"]["ood_rows_trainable"] is False
    assert contract["cohort_authority_contract_sha256"] == (
        trainer.COHORT_AUTHORITY_CONTRACT_SHA256
    )
    assert contract["prohibited"][
        "benchmark_query_label_mask_or_target_heldout"
    ] is True


def test_shard_schema_checks_exact_keys_gauges_padding_and_access() -> None:
    payload = _payload("source_train", ["a"] * 4)
    validated = trainer.validate_training_shard_payload(
        payload, expected_split="source_train"
    )
    assert validated["scene_ids"] == ["a"] * 4
    assert not validated["accepted_v2_e0"].requires_grad

    extra = copy.deepcopy(payload)
    extra["target_labels"] = torch.ones(4)
    with pytest.raises(ValueError, match="fields differ"):
        trainer.validate_training_shard_payload(extra)

    contaminated = copy.deepcopy(payload)
    contaminated["source_access"]["benchmark_labels_opened"] = True
    with pytest.raises(ValueError, match="contract differs"):
        trainer.validate_training_shard_payload(contaminated)

    non_unit = copy.deepcopy(payload)
    non_unit["accepted_v2_e0"][0] *= 2.0
    with pytest.raises(ValueError, match="unit L2"):
        trainer.validate_training_shard_payload(non_unit)

    dense_impostor = copy.deepcopy(payload)
    dense_impostor["official_multiview_siglip2_teacher_descriptors"] = (
        torch.zeros(4, 1, trainer.DESCRIPTOR_DIM)
    )
    with pytest.raises(ValueError, match="fields differ"):
        trainer.validate_training_shard_payload(dense_impostor)

    duplicate_view = copy.deepcopy(payload)
    duplicate_view["teacher_pair_view_ids"][1] = duplicate_view[
        "teacher_pair_view_ids"
    ][0]
    duplicate_view["channel_sha256"] = trainer.training_shard_channel_sha256(
        duplicate_view
    )
    with pytest.raises(ValueError, match="unique"):
        trainer.validate_training_shard_payload(duplicate_view)

    ineligible_teacher = copy.deepcopy(payload)
    ineligible_teacher["eligible"][0] = False
    with pytest.raises(ValueError, match="ineligible"):
        trainer.validate_training_shard_payload(ineligible_teacher)

    channel_tamper = copy.deepcopy(payload)
    channel_tamper["raw_full_scalar_summary"][0, 0] += 1.0
    with pytest.raises(ValueError, match="channel SHA-256"):
        trainer.validate_training_shard_payload(channel_tamper)


def test_shard_loader_requires_external_expected_sha256(tmp_path) -> None:
    path = tmp_path / "source_train.pt"
    write_torch_noclobber(
        path, _payload("source_train", ["a"] * 4)
    )
    digest = sha256_file(path)

    shard, record = trainer.load_training_shard(
        path,
        expected_sha256=digest,
        expected_split="source_train",
    )
    assert shard["split"] == "source_train"
    assert record == {"path": str(path.resolve()), "sha256": digest}
    with pytest.raises(ValueError, match="SHA-256 differs"):
        trainer.load_training_shard(
            path,
            expected_sha256="0" * 64,
            expected_split="source_train",
        )


def test_cohort_authority_loader_binds_file_and_content_sha(tmp_path) -> None:
    path = tmp_path / "clean_cohort.json"
    write_frozen_json(path, _authority_payload())
    digest = sha256_file(path)

    authority, record = trainer.load_cohort_authority(
        path, expected_sha256=digest
    )
    assert authority["authority_sha256"] == COHORT_SHA
    assert record == {"path": str(path.resolve()), "sha256": digest}
    with pytest.raises(ValueError, match="SHA-256 differs"):
        trainer.load_cohort_authority(path, expected_sha256="0" * 64)

    tampered = _authority_payload()
    tampered["benchmark_exclusion"]["manifest_file_sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered_cohort.json"
    write_frozen_json(tampered_path, tampered)
    with pytest.raises(ValueError, match="content SHA-256 differs"):
        trainer.load_cohort_authority(
            tampered_path,
            expected_sha256=sha256_file(tampered_path),
        )


def test_merge_keeps_sparse_pairs_and_batch_local_gather() -> None:
    first = _payload("source_train", ["a", "a"], views=2)
    second = _payload("source_train", ["b", "b"], views=4)
    merged = trainer._pad_and_merge_shards(
        [first, second], split="source_train"
    )

    pair_rows = merged[
        "official_multiview_siglip2_teacher_pair_region_indices"
    ]
    pair_descriptors = merged[
        "official_multiview_siglip2_teacher_pair_descriptors"
    ]
    assert pair_rows.numel() == sum(
        item["official_multiview_siglip2_teacher_pair_region_indices"].numel()
        for item in (first, second)
    )
    assert pair_descriptors.shape == (pair_rows.numel(), trainer.DESCRIPTOR_DIM)
    assert int(merged[
        "official_multiview_siglip2_teacher_pair_row_offsets"
    ][-1]) == pair_rows.numel()
    assert "official_multiview_siglip2_teacher_descriptors" not in merged
    gathered, mask = trainer._gather_sparse_teacher_rows(
        merged, torch.tensor([0, 2])
    )
    assert gathered.shape[0] == 2
    assert gathered.shape[1] <= trainer.VIEW_CAP_PER_REGION
    assert torch.equal(mask.sum(dim=1), torch.tensor([1, 3]))


def test_descriptor_metrics_never_gather_more_than_256_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _merged("source_train", ["scene_a"], rows_per_scene=300)
    observed: list[int] = []
    original = trainer._gather_sparse_teacher_rows

    def spy(data, rows):
        observed.append(int(torch.as_tensor(rows).numel()))
        return original(data, rows)

    monkeypatch.setattr(trainer, "_gather_sparse_teacher_rows", spy)
    trainer._descriptor_scene_metrics(
        data["accepted_v2_e0"],
        data,
        torch.arange(300),
        data["scene_ids"],
        torch.ones(300, dtype=torch.bool),
    )
    assert observed and max(observed) <= 256


def test_train_and_evaluate_sparse_gathers_obey_batch_and_pair_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _merged("source_train", ["scene_a"], rows_per_scene=300)
    normalization = _normalization(data)
    model = _model(normalization)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trainer.LEARNING_RATE, weight_decay=0.0
    )
    original = trainer._gather_sparse_teacher_rows
    train_calls: list[tuple[int, int]] = []

    def train_spy(data, rows):
        selected = torch.as_tensor(rows).long().cpu()
        offsets = data[
            "official_multiview_siglip2_teacher_pair_row_offsets"
        ]
        pair_count = int((offsets[selected + 1] - offsets[selected]).sum())
        train_calls.append((int(selected.numel()), pair_count))
        return original(data, rows)

    monkeypatch.setattr(trainer, "_gather_sparse_teacher_rows", train_spy)
    trainer.train_one_epoch(
        model,
        optimizer,
        data,
        normalization,
        torch.device("cpu"),
        epoch=1,
    )
    assert train_calls
    assert max(rows for rows, _pairs in train_calls) <= trainer.BATCH_ROWS == 64
    assert max(pairs for _rows, pairs in train_calls) <= 256

    evaluate_calls: list[int] = []

    def evaluate_spy(data, rows):
        evaluate_calls.append(int(torch.as_tensor(rows).numel()))
        return original(data, rows)

    monkeypatch.setattr(trainer, "_gather_sparse_teacher_rows", evaluate_spy)
    trainer.evaluate(model, data, normalization, torch.device("cpu"))
    assert evaluate_calls and max(evaluate_calls) <= 256


def test_sparse_gather_does_not_surface_unrequested_pair_sentinels() -> None:
    data = _merged("source_train", ["scene_a"], rows_per_scene=3)
    offsets = data["official_multiview_siglip2_teacher_pair_row_offsets"]
    start, stop = int(offsets[1]), int(offsets[2])
    data["official_multiview_siglip2_teacher_pair_descriptors"][start:stop] = (
        float("nan")
    )
    gathered, mask = trainer._gather_sparse_teacher_rows(data, torch.tensor([0, 2]))
    assert bool(torch.isfinite(gathered[mask]).all())


def test_cohort_requires_exact_clean_24_plus_8_scene_split_and_lineage(
    tmp_path,
) -> None:
    train = _merged(
        "source_train", TRAIN_SCENES
    )
    validation = _merged(
        "source_validation",
        VALIDATION_SCENES,
    )
    authorities = _materialized_authorities(tmp_path, train, validation)
    authority, authority_file = authorities[:2]
    authority_file_sha = authority_file["sha256"]

    cohort = trainer.validate_training_cohort(
        train, validation, *authorities
    )
    assert cohort["train_scenes"] == TRAIN_SCENES
    assert cohort["validation_scenes"] == VALIDATION_SCENES
    assert cohort["cohort_authority_sha256"] == authority["authority_sha256"]
    assert cohort["cohort_authority_file"] == authority_file
    assert cohort["source_state_cohort_authority_sha256"] == authorities[2][
        "authority_sha256"
    ]

    overlap = _merged(
        "source_validation",
        [TRAIN_SCENES[0], *VALIDATION_SCENES[1:]],
        cohort_file_sha=authority_file_sha,
    )
    overlap["lineages"] = copy.deepcopy(validation["lineages"])
    with pytest.raises(ValueError, match="overlap"):
        trainer.validate_training_cohort(
            train, overlap, *authorities
        )

    physical_overlap = _merged(
        "source_validation",
        [TRAIN_SCENES[0].replace("_00", "_01"), *VALIDATION_SCENES[1:]],
        cohort_file_sha=authority_file_sha,
    )
    physical_overlap["lineages"] = copy.deepcopy(validation["lineages"])
    with pytest.raises(ValueError, match="physical spaces overlap"):
        trainer.validate_training_cohort(
            train, physical_overlap, *authorities
        )

    wrong_count = _merged(
        "source_validation",
        VALIDATION_SCENES[:-1],
        cohort_file_sha=authority_file_sha,
    )
    wrong_count["lineages"] = copy.deepcopy(validation["lineages"])
    with pytest.raises(ValueError, match="exactly 8"):
        trainer.validate_training_cohort(
            train, wrong_count, *authorities
        )

    wrong_scene = _merged(
        "source_validation",
        [*VALIDATION_SCENES[:-1], "scene2999_00"],
        cohort_file_sha=authority_file_sha,
    )
    wrong_scene["lineages"] = copy.deepcopy(validation["lineages"])
    with pytest.raises(ValueError, match="differ from clean cohort authority"):
        trainer.validate_training_cohort(
            train, wrong_scene, *authorities
        )

    mismatched = _merged(
        "source_validation",
        VALIDATION_SCENES,
        cohort_file_sha=authority_file_sha,
    )
    mismatched["lineages"] = copy.deepcopy(validation["lineages"])
    mismatched["lineages"][0]["cohort_authority_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="different cohort"):
        trainer.validate_training_cohort(
            train, mismatched, *authorities
        )

    wrong_file_lineage = _merged(
        "source_validation",
        VALIDATION_SCENES,
        cohort_file_sha="f" * 64,
    )
    wrong_file_lineage["lineages"] = copy.deepcopy(validation["lineages"])
    wrong_file_lineage["lineages"][0]["cohort_authority_file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different cohort authority files"):
        trainer.validate_training_cohort(
            train, wrong_file_lineage, *authorities
        )


def test_benchmark_exclusion_rejects_same_physical_space_different_scan_suffix(
    tmp_path,
) -> None:
    train_scenes = [BENCHMARK_SCENES[0].replace("_01", "_00"), *TRAIN_SCENES[1:]]
    train = _merged("source_train", train_scenes)
    validation = _merged("source_validation", VALIDATION_SCENES)
    case = tmp_path / "physical_benchmark_leak"
    case.mkdir()
    authorities = _materialized_authorities(case, train, validation)

    with pytest.raises(ValueError, match="benchmark exclusion physical space"):
        trainer.validate_training_cohort(
            train, validation, *authorities
        )


def test_normalization_uses_train_only_and_validation_outlier_is_ood() -> None:
    train = _merged(
        "source_train", [f"train_{index:02d}" for index in range(2)],
        rows_per_scene=3,
    )
    validation = _merged(
        "source_validation", ["validation_00"], rows_per_scene=2
    )
    normalization = _normalization(train)
    before_median = normalization["median"].clone()
    validation["raw_full_scalar_summary"][0] = 1e6

    # The validation tensor is never passed to the authority builder.
    assert torch.equal(normalization["median"], before_median)
    routed = apply_full_scalar_normalization(
        validation["raw_full_scalar_summary"],
        validation["eligible"],
        normalization,
    )
    assert bool(routed.ood_mask[0])
    assert bool(routed.base_fallback_mask[0])
    assert not bool(routed.use_full_scalar_mask[0])


def test_zero_initialization_evaluation_is_exact_non_regressing_base() -> None:
    data = _merged(
        "source_validation", ["validation_a", "validation_b"],
        rows_per_scene=3,
    )
    normalization = _normalization(data)
    model = _model(normalization)

    parity = trainer._zero_init_parity(
        model, [data], normalization, torch.device("cpu")
    )
    metrics = trainer.evaluate(
        model, data, normalization, torch.device("cpu")
    )

    assert parity["passed"] is True
    assert parity["bitwise_equal"] is True
    assert metrics["candidate"] == metrics["base"]
    assert metrics["candidate_minus_base"] == {
        "mean_all_view_cosine": 0.0,
        "p05_row_mean_all_view_cosine": 0.0,
        "relation_fidelity": 0.0,
    }
    assert metrics["non_regression_passed"] is True
    assert metrics["eligible_scene_coverage"]["passed"] is True
    assert metrics["in_domain_scene_coverage"]["passed"] is True
    assert metrics["in_domain"]["vacuous_fallback_only"] is False


def test_validation_rejects_vacuous_all_ood_fallback_gate() -> None:
    data = _merged(
        "source_validation", ["validation_a", "validation_b"], rows_per_scene=2
    )
    normalization = build_full_scalar_normalization_authority(
        torch.zeros(2, 18),
        torch.ones(2, dtype=torch.bool),
        source_state_cohort_sha256=SOURCE_COHORT_SHA,
    )
    metrics = trainer.evaluate(
        _model(normalization), data, normalization, torch.device("cpu")
    )

    assert metrics["ood_fallback_rows"] == 4
    assert metrics["eligible_scene_coverage"]["passed"] is True
    assert metrics["in_domain_scene_coverage"]["passed"] is False
    assert metrics["in_domain_scene_coverage"][
        "missing_or_insufficient_scenes"
    ] == ["validation_a", "validation_b"]
    assert metrics["in_domain"]["vacuous_fallback_only"] is True
    assert metrics["non_regression_passed"] is False


def test_validation_requires_two_in_domain_rows_in_every_scene() -> None:
    data = _merged(
        "source_validation", ["validation_a", "validation_b"], rows_per_scene=3
    )
    normalization = _normalization(data)
    # Leave only one in-envelope row in validation_a; validation_b remains
    # fully covered.  Bitwise base fallback is safe but cannot certify that
    # the residual generalized to every frozen validation scene.
    data["raw_full_scalar_summary"][[0, 1]] = 1e6
    metrics = trainer.evaluate(
        _model(normalization), data, normalization, torch.device("cpu")
    )

    assert metrics["in_domain_scene_coverage"]["per_scene_rows"] == {
        "validation_a": 1,
        "validation_b": 3,
    }
    assert metrics["in_domain_scene_coverage"][
        "missing_or_insufficient_scenes"
    ] == ["validation_a"]
    assert metrics["in_domain_scene_coverage"]["passed"] is False
    assert metrics["non_regression_passed"] is False


def test_training_excludes_ood_and_never_mutates_base_or_teacher() -> None:
    data = _merged(
        "source_train", ["train_a", "train_b"], rows_per_scene=3
    )
    # Fit the envelope on two rows per scene, then make the third row of each
    # scene a strict OOD observation while it remains otherwise eligible.
    authority_mask = torch.tensor([True, True, False, True, True, False])
    normalization = build_full_scalar_normalization_authority(
        data["raw_full_scalar_summary"],
        authority_mask,
        source_state_cohort_sha256=SOURCE_COHORT_SHA,
    )
    data["raw_full_scalar_summary"][[2, 5]] = 1e6
    routed = apply_full_scalar_normalization(
        data["raw_full_scalar_summary"], data["eligible"], normalization
    )
    assert int(routed.ood_mask.sum()) == 2
    model = _model(normalization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer.LEARNING_RATE,
        weight_decay=trainer.WEIGHT_DECAY,
    )
    frozen_base = data["accepted_v2_e0"].clone()
    frozen_teacher = data[
        "official_multiview_siglip2_teacher_pair_descriptors"
    ].clone()

    report = trainer.train_one_epoch(
        model,
        optimizer,
        data,
        normalization,
        torch.device("cpu"),
        epoch=1,
    )

    assert report["trainable_rows"] == 4
    assert report["ood_rows_excluded"] == 2
    assert torch.equal(data["accepted_v2_e0"], frozen_base)
    assert torch.equal(
        data["official_multiview_siglip2_teacher_pair_descriptors"], frozen_teacher
    )
    assert torch.count_nonzero(model.residual_projection.weight) > 0


def test_checkpoint_selection_filters_regressions_and_uses_fixed_ranking() -> None:
    def row(epoch: int, mean: float, p05: float, relation: float, passed: bool):
        return {
            "epoch": epoch,
            "validation": {
                "non_regression_passed": passed,
                "candidate": {
                    "mean_all_view_cosine": mean,
                    "p05_row_mean_all_view_cosine": p05,
                    "relation_fidelity": relation,
                },
            },
        }

    history = [
        row(0, 0.70, 0.40, 0.80, True),
        row(1, 0.75, 0.39, 0.82, False),
        row(2, 0.73, 0.42, 0.81, True),
        row(3, 0.73, 0.42, 0.81, True),
    ]
    assert trainer.select_best_epoch(history) == 2

    history[0]["validation"]["non_regression_passed"] = False
    history[2]["validation"]["non_regression_passed"] = False
    history[3]["validation"]["non_regression_passed"] = False
    with pytest.raises(RuntimeError, match="no source-validation"):
        trainer.select_best_epoch(history)


def test_scene_macro_gate_rejects_small_scene_collapse_hidden_by_many_rows() -> None:
    large_rows, small_rows, dimension = 100, 2, trainer.DESCRIPTOR_DIM
    total = large_rows + small_rows
    teachers = torch.zeros(total, 1, dimension)
    teachers[..., 0] = 1.0
    base = teachers[:, 0].clone()
    base[:large_rows, 0] = 0.99
    base[:large_rows, 1] = (1.0 - 0.99**2) ** 0.5
    base = F.normalize(base, dim=-1)
    candidate = teachers[:, 0].clone()
    candidate[large_rows:, 0] = 0.0
    candidate[large_rows:, 1] = 1.0

    class Fixed(torch.nn.Module):
        def forward(self, base, scalars, *, ood_mask=None):
            return candidate.to(base.device)

    normalization = build_full_scalar_normalization_authority(
        torch.zeros(2, 18),
        torch.ones(2, dtype=torch.bool),
        source_state_cohort_sha256=SOURCE_COHORT_SHA,
    )
    result = trainer.evaluate(
        Fixed(),
        {
            "accepted_v2_e0": base,
            "raw_full_scalar_summary": torch.zeros(total, 18),
            "eligible": torch.ones(total, dtype=torch.bool),
            "official_multiview_siglip2_teacher_pair_region_indices": (
                torch.arange(total)
            ),
            "official_multiview_siglip2_teacher_pair_descriptors": teachers[:, 0],
            "official_multiview_siglip2_teacher_pair_row_offsets": (
                torch.arange(total + 1)
            ),
            "scene_ids": ["large"] * large_rows + ["small"] * small_rows,
        },
        normalization,
        torch.device("cpu"),
    )
    assert result["aggregation"] == "scene_macro"
    assert result["candidate_minus_base"]["mean_all_view_cosine"] < 0
    assert result["paired_scene_deltas"]["mean_all_view_cosine"]["worst"] == pytest.approx(-1.0)
    assert result["non_regression_passed"] is False


def test_scene_balanced_batches_have_one_equal_weight_sample_per_scene() -> None:
    sizes = {"s002": 2, "s063": 63, "s065": 65, "s066": 66}
    scene_ids = [scene for scene, size in sizes.items() for _ in range(size)]
    batches = trainer._training_batches(
        scene_ids, torch.ones(len(scene_ids), dtype=torch.bool), epoch=1
    )
    assert len(batches) == len(sizes)
    observed = {
        scene_ids[int(rows[0])]: int(rows.numel()) for rows in batches
    }
    assert observed == {"s002": 2, "s063": 63, "s065": 64, "s066": 64}


def test_training_and_validation_share_strict_off_diagonal_relation() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    _absolute, training_values = trainer._off_diagonal_relation(student, teacher)
    validation = trainer._relation_metrics(
        student,
        teacher,
        ["scene", "scene"],
        torch.ones(2, dtype=torch.bool),
    )
    assert float(training_values.mean()) == pytest.approx(0.5)
    assert validation["relation_smooth_l1"] == pytest.approx(0.5)
