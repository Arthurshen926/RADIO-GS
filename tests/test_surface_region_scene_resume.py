from __future__ import annotations

import os
from pathlib import Path
import random

import pytest
import torch

from radio_gs.scripts.surface_region_scene_resume import (
    RESUME_CONTRACT_ARTIFACT_TYPE,
    RESUME_SCHEMA_VERSION,
    SCENE_PARTIAL_SUFFIX,
    SCENE_ROW_SCHEMA_V3,
    SCENE_TENSOR_KEYS,
    SCENE_TENSOR_KEYS_V3,
    SceneResumeStateError,
    append_scene_rows,
    commit_scene_partial,
    decode_rng_state,
    encode_rng_state,
    load_scene_partial,
    open_or_create_resume_contract,
    scene_resume_paths,
    validate_resume_inventory,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json
from radio_gs.training.surface_region_eligibility_completion import (
    STRUCTURED_ELIGIBILITY_POLICY,
)


SCENES = ["scene0001_00", "scene0002_00", "scene0003_00"]
ROWS = 2
TOKENS = 4
VIEWS = 3


def _contract(scenes: list[str] | None = None) -> dict:
    selected = list(scenes or SCENES)
    return {
        "artifact_type": RESUME_CONTRACT_ARTIFACT_TYPE,
        "schema_version": RESUME_SCHEMA_VERSION,
        "builder": {
            "entrypoint": {"path": "/source/builder.py", "sha256": "a" * 64},
            "scene_resume_implementation": {
                "path": "/source/resume.py",
                "sha256": "b" * 64,
            },
        },
        "cli": {
            "seed": 17,
            "frames_per_scene": 8,
            "output": "/output/train_shard0.pt",
        },
        "inputs": {
            "split_file": {"path": "/data/split.txt", "sha256": "c" * 64},
            "scenes": selected,
        },
        "selected_scenes": selected,
        "row_contract": {
            "regions_per_scene": ROWS,
            "maximum_tokens": TOKENS,
            "teacher_views": VIEWS,
        },
        "resume_protocol": {
            "partial_suffix": SCENE_PARTIAL_SUFFIX,
            "load": "weights_only_same_fd_external_sha256",
        },
    }


def _scene_rows(scene: str, rng: random.Random) -> dict:
    generator = torch.Generator().manual_seed(rng.randrange(2**31))
    token_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    teacher_mask = torch.tensor(
        [[True, True, False], [True, True, True]]
    )
    features = torch.randn(
        ROWS, TOKENS, 1280, generator=generator, dtype=torch.float16
    )
    geometry = torch.randn(
        ROWS, TOKENS, 14, generator=generator, dtype=torch.float16
    )
    reliability = torch.rand(
        ROWS, TOKENS, 1, generator=generator, dtype=torch.float16
    )
    summaries = torch.randn(
        ROWS, VIEWS, 1280, generator=generator, dtype=torch.float16
    )
    descriptors = torch.randn(
        ROWS, VIEWS, 8, generator=generator, dtype=torch.float16
    )
    features[~token_mask] = 0
    geometry[~token_mask] = 0
    reliability[~token_mask] = 0
    summaries[~teacher_mask] = 0
    descriptors[~teacher_mask] = 0
    anchor = torch.tensor([0, 1], dtype=torch.long)
    records = [
        {
            "scene": scene,
            "region_id": f"{scene}-{index}",
            "tokens": int(token_mask[index].sum()),
            "anchor_local_index": int(anchor[index]),
            "teacher_views": [
                {"frame": f"{view}.jpg", "crop_box_tlbr": [0, 0, 24, 24]}
                for view in range(int(teacher_mask[index].sum()))
            ],
        }
        for index in range(ROWS)
    ]
    return {
        "radio_features": features,
        "geometry": geometry,
        "token_mask": token_mask,
        "reliability": reliability,
        "official_summary_tokens": summaries,
        "official_crop_summaries": descriptors,
        "teacher_mask": teacher_mask,
        "anchor_index": anchor,
        "records": records,
    }


def _scene_rows_v3(scene: str, rng: random.Random) -> dict:
    rows = _scene_rows(scene, rng)
    token_mask = rows["token_mask"]
    generator = torch.Generator().manual_seed(rng.randrange(2**31))
    directions = torch.nn.functional.normalize(
        torch.randn(ROWS, TOKENS, 1280, generator=generator),
        dim=-1,
    ).half()
    directions[~token_mask] = 0
    support_fill = torch.tensor(
        [[False, True, False, False], [False, False, True, False]]
    )
    geometry = torch.randn(
        ROWS, TOKENS, 16, generator=generator, dtype=torch.float16
    )
    geometry[..., 14] = support_fill.to(geometry.dtype)
    geometry[~token_mask] = 0
    rows["radio_features"] = directions
    rows["geometry"] = geometry
    rows["support_fill_mask"] = support_fill
    for index, record in enumerate(rows["records"]):
        fill = int(support_fill[index].sum())
        record.update(
            {
                "support_fill_tokens": fill,
                "semantic_tokens": int(token_mask[index].sum()) - fill,
                "minimum_satisfied": True,
            }
        )
    return rows


def _paired_scene_rows_v3(scene: str, rng: random.Random) -> dict:
    rows = _scene_rows_v3(scene, rng)
    for key in SCENE_TENSOR_KEYS_V3:
        rows[key] = rows[key][torch.tensor([0, 0])].clone()
    full_id = f"{scene}-full"
    shared = {
        **rows["records"][0],
        "region_id": full_id,
        "seed": 7,
        "physical_radius_m": 0.25,
        "teacher_support_sha256": "a" * 64,
        "teacher_target_sha256": "b" * 64,
        "tokens": int(rows["token_mask"][0].sum()),
        "support_fill_tokens": int(rows["support_fill_mask"][0].sum()),
        "semantic_tokens": int(
            rows["token_mask"][0].sum()
            - rows["support_fill_mask"][0].sum()
        ),
        "anchor_local_index": int(rows["anchor_index"][0]),
        "eligibility_variants_per_teacher_region": 1,
        "paired_full_region_id": full_id,
    }
    rows["records"] = [
        {
            **shared,
            "row_role": "full_support",
            "eligibility_variant_index": -1,
            "eligibility_sha256": "",
        },
        {
            **shared,
            "region_id": f"{full_id}-completion",
            "row_role": "eligibility_completion",
            "eligibility_variant_index": 0,
            "eligibility_sha256": "c" * 64,
            "eligibility_policy": STRUCTURED_ELIGIBILITY_POLICY,
            "eligibility_semantic_eligible_tokens": int(
                rows["token_mask"][1].sum()
                - rows["support_fill_mask"][1].sum()
            ),
            "eligibility_nominal_semantic_keep_tokens": int(
                rows["token_mask"][1].sum()
                - rows["support_fill_mask"][1].sum()
            ),
            "eligibility_expected_fill_tokens": int(
                rows["support_fill_mask"][1].sum()
            ),
            "eligibility_extreme_graph_fallback": False,
            "eligibility_extreme_graph_fallback_reason": "",
        },
    ]
    return rows


def _empty_merge() -> tuple[list[dict], dict[str, list[torch.Tensor]]]:
    return [], {key: [] for key in SCENE_TENSOR_KEYS}


def test_v3_scene_resume_round_trip_and_cross_version_fail_closed(
    tmp_path: Path,
) -> None:
    contract = _contract([SCENES[0]])
    contract["row_contract"] = {
        **contract["row_contract"],
        "row_schema_version": SCENE_ROW_SCHEMA_V3,
        "geometry_dimension": 16,
    }
    root, authority, contract_sha = open_or_create_resume_contract(
        tmp_path / "resume-v3", contract
    )
    rng = random.Random(91)
    before = rng.getstate()
    rows = _scene_rows_v3(SCENES[0], rng)
    commit_scene_partial(
        root,
        scene_index=0,
        scene_name=SCENES[0],
        scene_rows=rows,
        rng_state_before=before,
        rng_state_after=rng.getstate(),
        expected_rows=ROWS,
        maximum_tokens=TOKENS,
        teacher_views=VIEWS,
        contract_record=authority,
        contract_payload_sha256=contract_sha,
        row_schema_version=SCENE_ROW_SCHEMA_V3,
    )
    partial = load_scene_partial(
        root,
        scene_index=0,
        scene_name=SCENES[0],
        expected_rows=ROWS,
        maximum_tokens=TOKENS,
        teacher_views=VIEWS,
        contract_record=authority,
        contract_payload_sha256=contract_sha,
        row_schema_version=SCENE_ROW_SCHEMA_V3,
    )
    assert partial is not None
    assert set(partial["rows"]) == set(SCENE_TENSOR_KEYS_V3) | {"records"}
    records: list[dict] = []
    tensors = {key: [] for key in SCENE_TENSOR_KEYS_V3}
    append_scene_rows(partial["rows"], records=records, tensor_rows=tensors)
    assert records == rows["records"]
    for key in SCENE_TENSOR_KEYS_V3:
        assert torch.equal(torch.stack(tensors[key]), rows[key])

    with pytest.raises(SceneResumeStateError, match="cannot be trusted"):
        load_scene_partial(
            root,
            scene_index=0,
            scene_name=SCENES[0],
            expected_rows=ROWS,
            maximum_tokens=TOKENS,
            teacher_views=VIEWS,
            contract_record=authority,
            contract_payload_sha256=contract_sha,
        )


def test_v3_paired_completion_resume_requires_exact_shared_teacher(
    tmp_path: Path,
) -> None:
    contract = _contract([SCENES[0]])
    contract["row_contract"] = {
        **contract["row_contract"],
        "row_schema_version": SCENE_ROW_SCHEMA_V3,
        "geometry_dimension": 16,
        "eligibility_variants_per_teacher_region": 1,
    }
    root, authority, contract_sha = open_or_create_resume_contract(
        tmp_path / "resume-paired-v3", contract
    )
    rng = random.Random(92)
    before = rng.getstate()
    rows = _paired_scene_rows_v3(SCENES[0], rng)
    commit_scene_partial(
        root,
        scene_index=0,
        scene_name=SCENES[0],
        scene_rows=rows,
        rng_state_before=before,
        rng_state_after=rng.getstate(),
        expected_rows=ROWS,
        maximum_tokens=TOKENS,
        teacher_views=VIEWS,
        contract_record=authority,
        contract_payload_sha256=contract_sha,
        row_schema_version=SCENE_ROW_SCHEMA_V3,
        eligibility_variants_per_region=1,
    )
    assert load_scene_partial(
        root,
        scene_index=0,
        scene_name=SCENES[0],
        expected_rows=ROWS,
        maximum_tokens=TOKENS,
        teacher_views=VIEWS,
        contract_record=authority,
        contract_payload_sha256=contract_sha,
        row_schema_version=SCENE_ROW_SCHEMA_V3,
        eligibility_variants_per_region=1,
    ) is not None

    bad = _paired_scene_rows_v3(SCENES[0], random.Random(92))
    bad["official_summary_tokens"][1, 0, 0] += 1
    with pytest.raises(ValueError, match="exact teacher tensors"):
        commit_scene_partial(
            tmp_path / "unused-resume",
            scene_index=0,
            scene_name=SCENES[0],
            scene_rows=bad,
            rng_state_before=before,
            rng_state_after=rng.getstate(),
            expected_rows=ROWS,
            maximum_tokens=TOKENS,
            teacher_views=VIEWS,
            contract_record=authority,
            contract_payload_sha256=contract_sha,
            row_schema_version=SCENE_ROW_SCHEMA_V3,
            eligibility_variants_per_region=1,
        )


def _stack_merge(
    records: list[dict], tensor_rows: dict[str, list[torch.Tensor]]
) -> dict:
    return {
        **{key: torch.stack(values) for key, values in tensor_rows.items()},
        "records": records,
    }


def _commit(
    root: Path,
    authority: dict[str, str],
    contract_sha: str,
    *,
    index: int,
    scene: str,
    rows: dict,
    before: object,
    after: object,
) -> None:
    commit_scene_partial(
        root,
        scene_index=index,
        scene_name=scene,
        scene_rows=rows,
        rng_state_before=before,
        rng_state_after=after,
        expected_rows=ROWS,
        maximum_tokens=TOKENS,
        teacher_views=VIEWS,
        contract_record=authority,
        contract_payload_sha256=contract_sha,
    )


def test_rng_state_is_basic_type_and_exact_round_trip() -> None:
    rng = random.Random(123)
    rng.random()
    encoded = encode_rng_state(rng.getstate())
    assert set(encoded) == {"version", "internal", "gaussian"}
    assert isinstance(encoded["internal"], list)
    assert all(isinstance(value, int) for value in encoded["internal"])
    assert decode_rng_state(encoded) == rng.getstate()


def test_interrupted_resume_matches_continuous_rows_and_records(
    tmp_path: Path,
) -> None:
    continuous_records, continuous_tensors = _empty_merge()
    continuous_rng = random.Random(17)
    continuous_rows: list[dict] = []
    for scene in SCENES:
        rows = _scene_rows(scene, continuous_rng)
        continuous_rows.append(rows)
        append_scene_rows(
            rows,
            records=continuous_records,
            tensor_rows=continuous_tensors,
        )
    continuous = _stack_merge(continuous_records, continuous_tensors)

    resume_root, authority, contract_sha = open_or_create_resume_contract(
        tmp_path / "resume",
        _contract(),
    )
    first_process_rng = random.Random(17)
    for index, scene in enumerate(SCENES[:2]):
        before = first_process_rng.getstate()
        rows = _scene_rows(scene, first_process_rng)
        _commit(
            resume_root,
            authority,
            contract_sha,
            index=index,
            scene=scene,
            rows=rows,
            before=before,
            after=first_process_rng.getstate(),
        )

    resumed_records, resumed_tensors = _empty_merge()
    second_process_rng = random.Random(17)
    for index, scene in enumerate(SCENES):
        partial = load_scene_partial(
            resume_root,
            scene_index=index,
            scene_name=scene,
            expected_rows=ROWS,
            maximum_tokens=TOKENS,
            teacher_views=VIEWS,
            contract_record=authority,
            contract_payload_sha256=contract_sha,
        )
        if partial is None:
            before = second_process_rng.getstate()
            rows = _scene_rows(scene, second_process_rng)
            _commit(
                resume_root,
                authority,
                contract_sha,
                index=index,
                scene=scene,
                rows=rows,
                before=before,
                after=second_process_rng.getstate(),
            )
        else:
            assert partial["rng_state_before"] == encode_rng_state(
                second_process_rng.getstate()
            )
            rows = partial["rows"]
            second_process_rng.setstate(
                decode_rng_state(partial["rng_state_after"])
            )
        append_scene_rows(
            rows,
            records=resumed_records,
            tensor_rows=resumed_tensors,
        )
    resumed = _stack_merge(resumed_records, resumed_tensors)

    assert resumed["records"] == continuous["records"]
    for key in SCENE_TENSOR_KEYS:
        assert torch.equal(resumed[key], continuous[key]), key
    assert resumed["records"] == [
        record for rows in continuous_rows for record in rows["records"]
    ]


def test_partial_tampering_is_rejected_by_external_sha(
    tmp_path: Path,
) -> None:
    root, authority, contract_sha = open_or_create_resume_contract(
        tmp_path / "resume", _contract()
    )
    rng = random.Random(17)
    before = rng.getstate()
    rows = _scene_rows(SCENES[0], rng)
    _commit(
        root,
        authority,
        contract_sha,
        index=0,
        scene=SCENES[0],
        rows=rows,
        before=before,
        after=rng.getstate(),
    )
    partial, _ = scene_resume_paths(root, scene_index=0, scene_name=SCENES[0])
    data = bytearray(partial.read_bytes())
    data[-1] ^= 1
    partial.write_bytes(data)
    with pytest.raises(SceneResumeStateError, match="cannot be trusted"):
        load_scene_partial(
            root,
            scene_index=0,
            scene_name=SCENES[0],
            expected_rows=ROWS,
            maximum_tokens=TOKENS,
            teacher_views=VIEWS,
            contract_record=authority,
            contract_payload_sha256=contract_sha,
        )


def test_contract_drift_and_unknown_inventory_fail_closed(tmp_path: Path) -> None:
    root, _, _ = open_or_create_resume_contract(
        tmp_path / "resume", _contract()
    )
    drifted = _contract()
    drifted["cli"]["seed"] = 18
    with pytest.raises(SceneResumeStateError, match="contract drifted"):
        open_or_create_resume_contract(root, drifted)

    (root / "train_shard0.pt").write_bytes(b"not allowed")
    with pytest.raises(SceneResumeStateError, match="unexpected inventory"):
        validate_resume_inventory(root, SCENES)


def test_repeat_commit_and_interrupted_publication_are_safe(tmp_path: Path) -> None:
    root, authority, contract_sha = open_or_create_resume_contract(
        tmp_path / "resume", _contract()
    )
    rng = random.Random(17)
    before = rng.getstate()
    rows = _scene_rows(SCENES[0], rng)
    kwargs = dict(
        index=0,
        scene=SCENES[0],
        rows=rows,
        before=before,
        after=rng.getstate(),
    )
    _commit(root, authority, contract_sha, **kwargs)
    with pytest.raises(FileExistsError, match="already exists"):
        _commit(root, authority, contract_sha, **kwargs)

    partial, terminal = scene_resume_paths(
        root, scene_index=0, scene_name=SCENES[0]
    )
    terminal.unlink()
    assert partial.exists()
    validate_resume_inventory(root, SCENES)
    assert not partial.exists()
    assert list(root.glob(f".{partial.name}.abandoned-*.tmp"))

    terminal.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SceneResumeStateError, match="half-published"):
        validate_resume_inventory(root, SCENES)


def test_symlinked_contract_and_partial_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "contract.json").symlink_to(target)
    with pytest.raises(SceneResumeStateError, match="cannot be reopened"):
        open_or_create_resume_contract(resume, _contract())

    safe_root, authority, contract_sha = open_or_create_resume_contract(
        tmp_path / "safe", _contract()
    )
    rng = random.Random(17)
    before = rng.getstate()
    rows = _scene_rows(SCENES[0], rng)
    _commit(
        safe_root,
        authority,
        contract_sha,
        index=0,
        scene=SCENES[0],
        rows=rows,
        before=before,
        after=rng.getstate(),
    )
    partial, _ = scene_resume_paths(
        safe_root, scene_index=0, scene_name=SCENES[0]
    )
    copied = tmp_path / "copied.partial"
    copied.write_bytes(partial.read_bytes())
    partial.unlink()
    partial.symlink_to(copied)
    with pytest.raises(SceneResumeStateError, match="cannot be trusted"):
        load_scene_partial(
            safe_root,
            scene_index=0,
            scene_name=SCENES[0],
            expected_rows=ROWS,
            maximum_tokens=TOKENS,
            teacher_views=VIEWS,
            contract_record=authority,
            contract_payload_sha256=contract_sha,
        )


def test_malicious_pickle_is_not_executed(tmp_path: Path) -> None:
    root, authority, contract_sha = open_or_create_resume_contract(
        tmp_path / "resume", _contract()
    )
    marker = tmp_path / "pickle-executed"

    class Evil:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    partial, terminal = scene_resume_paths(
        root, scene_index=0, scene_name=SCENES[0]
    )
    torch.save({"evil": Evil()}, partial)
    terminal_payload = {
        "artifact_type": "surface-region-scene-partial-terminal-v1",
        "schema_version": RESUME_SCHEMA_VERSION,
        "scene_index": 0,
        "scene_name": SCENES[0],
        "rows": ROWS,
        "resume_contract": authority,
        "resume_contract_payload_sha256": contract_sha,
        "partial": file_record(partial),
        "row_bundle_sha256": "d" * 64,
    }
    write_frozen_json(terminal, terminal_payload)
    with pytest.raises(SceneResumeStateError, match="cannot be trusted"):
        load_scene_partial(
            root,
            scene_index=0,
            scene_name=SCENES[0],
            expected_rows=ROWS,
            maximum_tokens=TOKENS,
            teacher_views=VIEWS,
            contract_record=authority,
            contract_payload_sha256=contract_sha,
        )
    assert not marker.exists()
