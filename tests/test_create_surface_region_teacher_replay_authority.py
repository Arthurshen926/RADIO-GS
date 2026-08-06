import hashlib
import json

import pytest
import torch

from radio_gs.scripts.create_surface_region_teacher_replay_authority import (
    create_authority,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_create_authority_binds_exact_cache_and_registration(tmp_path) -> None:
    cache = tmp_path / "train_shard0.pt"
    manifest = tmp_path / "registration.json"
    output = tmp_path / "train_shard0.json"
    manifest.write_text(json.dumps({"registered": True}))
    torch.save(
        {
            "official_summary_tokens": torch.zeros(1, 2, 3),
            "official_crop_summaries": torch.zeros(1, 2, 4),
            "teacher_mask": torch.ones(1, 2, dtype=torch.bool),
            "metadata": {
                "schema_version": 3,
                "split_role": "train",
                "split_file_sha256": _sha("split"),
                "scene_names": ["scene0001_00"],
                "region_records": [{"region_id": "r"}],
                "teacher_region_semantics": (
                    "fixed_core_geodesic_support_without_input_context_v1"
                ),
                "teacher_target_source": "fresh_official_runtime",
                "teacher_regions_saturated": 0,
                "complete_scene_regions": True,
                "failed_scenes": {},
                "builder_script_sha256": _sha("builder"),
                "teacher_region_contract_sha256": _sha("contract"),
                "teacher_target_protocol_sha256": _sha("protocol"),
                "radio_checkpoint_sha256": _sha("radio"),
            },
        },
        cache,
    )

    payload = create_authority(
        cache,
        run_manifest=manifest,
        output=output,
    )

    assert payload["cache"]["path"] == str(cache.resolve())
    assert payload["authorization_scope"] == (
        "exact_historical_cache_fixed_teacher_replay_only"
    )
    assert json.loads(output.read_text()) == payload
    assert create_authority(cache, run_manifest=manifest, output=output) == payload


def test_create_authority_rejects_nonfresh_teacher(tmp_path) -> None:
    cache = tmp_path / "cache.pt"
    manifest = tmp_path / "registration.json"
    manifest.write_text("{}")
    torch.save(
        {
            "official_summary_tokens": torch.zeros(1, 1, 1),
            "official_crop_summaries": torch.zeros(1, 1, 1),
            "teacher_mask": torch.ones(1, 1, dtype=torch.bool),
            "metadata": {
                "schema_version": 3,
                "split_role": "train",
                "teacher_target_source": "exact_cache_replay",
            },
        },
        cache,
    )
    with pytest.raises(ValueError, match="not a complete"):
        create_authority(
            cache,
            run_manifest=manifest,
            output=tmp_path / "authority.json",
        )


def test_create_authority_accepts_strict_paired_schema4_cache(tmp_path) -> None:
    cache = tmp_path / "paired.pt"
    manifest = tmp_path / "registration.json"
    output = tmp_path / "authority.json"
    manifest.write_text("{}")
    full = {
        "region_id": "full",
        "paired_full_region_id": "full",
        "row_role": "full_support",
    }
    completion = {
        "region_id": "completion",
        "paired_full_region_id": "full",
        "row_role": "eligibility_completion",
    }
    torch.save(
        {
            "official_summary_tokens": torch.zeros(2, 2, 3),
            "official_crop_summaries": torch.zeros(2, 2, 4),
            "teacher_mask": torch.ones(2, 2, dtype=torch.bool),
            "metadata": {
                "schema_version": 4,
                "split_role": "validation",
                "split_file_sha256": _sha("split"),
                "scene_names": ["scene0001_00"],
                "region_records": [full, completion],
                "eligibility_completion": {
                    "schema_version": 1,
                    "validation_checkpoint_selection": "full_support_rows_only",
                },
                "teacher_region_semantics": (
                    "fixed_core_geodesic_support_without_input_context_v1"
                ),
                "teacher_target_source": "fresh_official_runtime",
                "teacher_regions_saturated": 0,
                "complete_scene_regions": True,
                "failed_scenes": {},
                "builder_script_sha256": _sha("builder"),
                "teacher_region_contract_sha256": _sha("contract"),
                "teacher_target_protocol_sha256": _sha("protocol"),
                "radio_checkpoint_sha256": _sha("radio"),
            },
        },
        cache,
    )

    payload = create_authority(
        cache,
        run_manifest=manifest,
        output=output,
    )
    assert payload["cache"]["sha256"] == hashlib.sha256(
        cache.read_bytes()
    ).hexdigest()
