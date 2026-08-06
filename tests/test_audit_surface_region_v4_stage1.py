import hashlib
from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV4
from radio_gs.scripts.audit_surface_region_v4_stage1 import audit_v4_stage1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity() -> dict:
    return {
        "scene": "scene0001_00",
        "seed": 7,
        "physical_radius_m": 0.25,
        "teacher_views": [{"frame": "000000.jpg", "crop_box_tlbr": [0, 0, 2, 2]}],
        "teacher_medoid": 0,
        "teacher_region_tokens": 2,
        "teacher_support_sha256": "a" * 64,
        "teacher_target_sha256": "b" * 64,
    }


def _write_pair(tmp_path: Path) -> tuple[Path, str, Path, str]:
    teacher_token = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]], dtype=torch.float16)
    teacher_crop = torch.tensor([[[3.0, 4.0, 5.0], [0.0, 0.0, 0.0]]], dtype=torch.float16)
    teacher_mask = torch.tensor([[True, False]])
    source_record = {"region_id": "source-full", **_identity()}
    common_metadata = {
        "teacher_target_source": "fresh_official_runtime",
        "complete_scene_regions": True,
        "failed_scenes": {},
        "teacher_regions_saturated": 0,
        "teacher_region_contract_sha256": "c" * 64,
        "teacher_target_protocol_sha256": "d" * 64,
        "radio_checkpoint_sha256": "e" * 64,
        "split_role": "validation",
        "split_file_sha256": "f" * 64,
    }
    source = tmp_path / "source.pt"
    torch.save(
        {
            "official_summary_tokens": teacher_token,
            "official_crop_summaries": teacher_crop,
            "teacher_mask": teacher_mask,
            "metadata": {
                **common_metadata,
                "schema_version": 3,
                "region_records": [source_record],
            },
        },
        source,
    )
    source_sha = _sha(source)

    contract = SurfaceRegionContractV4(
        maximum_tokens=4,
        minimum_tokens=2,
        token_candidate_limit=8,
        core_token_fraction=0.5,
    )
    features = torch.zeros(2, 4, 1280, dtype=torch.float16)
    features[0, 0, 0] = 1
    features[0, 1, 1] = 1
    features[0, 2, 0:2] = torch.tensor([2**-0.5, 2**-0.5])
    features[1, 0, 0] = 1
    features[1, 1, 1] = 1
    geometry = torch.zeros(2, 4, 16, dtype=torch.float16)
    geometry[0, :3, 6] = torch.tensor([0.9, 0.8, 0.7])
    geometry[1, :2, 6] = torch.tensor([0.9, 0.4])
    geometry[0, 0, 7] = geometry[1, 0, 7] = 1
    geometry[0, 0:2, 8] = 1
    geometry[0, 2, 9] = 1
    geometry[1, 0, 8] = 1
    geometry[1, 1, 14] = 1
    geometry[0, :3, 15] = torch.log(torch.tensor([2.0, 3.0, 4.0])).half()
    geometry[1, :2, 15] = torch.log(torch.tensor([2.0, 5.0])).half()
    geometry[0, 1, 0] = 0.4
    geometry[0, 2, 0] = 1.1
    geometry[1, 1, 0] = 1.5
    reliability = geometry[..., 6:7].clone()
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    fill = torch.tensor([[False, False, False, False], [False, True, False, False]])
    full_record = {
        "region_id": "v4-full",
        "paired_full_region_id": "v4-full",
        "row_role": "full_support",
        "eligibility_variants_per_teacher_region": 1,
        "eligibility_variant_index": -1,
        "teacher_replay_source_row": 0,
        "teacher_replay_source_region_id": "source-full",
        "tokens": 3,
        "core_tokens": 2,
        "context_tokens": 1,
        "semantic_tokens": 3,
        "support_fill_tokens": 0,
        "minimum_satisfied": True,
        "anchor_local_index": 0,
        **_identity(),
    }
    completion_record = {
        **full_record,
        "region_id": "v4-completion",
        "paired_full_region_id": "v4-full",
        "row_role": "eligibility_completion",
        "eligibility_variant_index": 0,
        "eligibility_sha256": "1" * 64,
        "eligibility_policy": (
            "hash_direction_anchor_connected_shortest_path_tree_with_external_support_v2"
        ),
        "tokens": 2,
        "core_tokens": 1,
        "context_tokens": 0,
        "semantic_tokens": 1,
        "support_fill_tokens": 1,
    }
    cache = tmp_path / "v4.pt"
    torch.save(
        {
            "radio_features": features,
            "geometry": geometry,
            "token_mask": mask,
            "reliability": reliability,
            "anchor_index": torch.tensor([0, 0]),
            "support_fill_mask": fill,
            "official_summary_tokens": teacher_token.expand(2, -1, -1).clone(),
            "official_crop_summaries": teacher_crop.expand(2, -1, -1).clone(),
            "teacher_mask": teacher_mask.expand(2, -1).clone(),
            "metadata": {
                **common_metadata,
                "teacher_target_source": "exact_cache_replay",
                "schema_version": 4,
                "surface_region_row_schema_version": 2,
                "training_scope": "global_cross_scene_3d_surface_v4",
                "region_contract": contract.to_dict(),
                "region_contract_version": contract.version,
                "region_contract_sha256": contract.digest,
                "teacher_replay_cache": {
                    "path": str(source.resolve()),
                    "sha256": source_sha,
                },
                "region_records": [full_record, completion_record],
                "eligibility_completion": {
                    "schema_version": 1,
                    "policy": (
                        "hash_direction_anchor_connected_shortest_path_tree_with_external_support_v2"
                    ),
                    "variants_per_teacher_region": 1,
                    "validation_checkpoint_selection": "full_support_rows_only",
                    "full_support_rows": 1,
                    "completion_variant_rows": 1,
                    "completion_rows_with_fill": 1,
                    "completion_support_fill_tokens": 1,
                    "completion_selected_tokens": 2,
                },
                "semantic_tokens_total": 4,
                "support_fill_tokens_total": 1,
            },
        },
        cache,
    )
    return cache, _sha(cache), source, source_sha


def test_stage1_audit_is_strict_query_free_and_marks_cap_boundary(tmp_path) -> None:
    cache, cache_sha, source, source_sha = _write_pair(tmp_path)
    output = tmp_path / "audit.json"
    report = audit_v4_stage1(
        cache,
        cache_sha256=cache_sha,
        replay_cache_path=source,
        replay_cache_sha256=source_sha,
        output=output,
    )

    assert report["status"] == "passed"
    assert report["strict_checks"]["teacher_replay"]["status"].startswith("bitwise_exact")
    assert report["membership_counts"]["context_zero_rows"] == 1
    assert report["query_free_context_risk_proxies"]["semantic_purity_claim"] is False
    cap = report["candidate_cap_audit_boundary"]
    assert cap["candidate_complete_claim_authorized"] is False
    assert cap["graph_replay_required_row_count"] == 2
    assert {row["row"] for row in cap["graph_replay_required_rows"]} == {0, 1}
    assert output.is_file()


def test_stage1_audit_rejects_teacher_tampering(tmp_path) -> None:
    cache, _cache_sha, source, source_sha = _write_pair(tmp_path)
    value = torch.load(cache, map_location="cpu", weights_only=True)
    value["official_summary_tokens"][1, 0, 0] += 1
    torch.save(value, cache)
    with pytest.raises(ValueError, match="teacher tensor official_summary_tokens differs"):
        audit_v4_stage1(
            cache,
            cache_sha256=_sha(cache),
            replay_cache_path=source,
            replay_cache_sha256=source_sha,
            output=tmp_path / "audit.json",
        )


def test_stage1_audit_rejects_nonzero_student_padding(tmp_path) -> None:
    cache, _cache_sha, source, source_sha = _write_pair(tmp_path)
    value = torch.load(cache, map_location="cpu", weights_only=True)
    value["geometry"][0, 3, 0] = 1
    torch.save(value, cache)
    with pytest.raises(ValueError, match="geometry padding is nonzero"):
        audit_v4_stage1(
            cache,
            cache_sha256=_sha(cache),
            replay_cache_path=source,
            replay_cache_sha256=source_sha,
            output=tmp_path / "audit.json",
        )


def test_stage1_audit_requires_external_sha_binding(tmp_path) -> None:
    cache, _cache_sha, source, source_sha = _write_pair(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        audit_v4_stage1(
            cache,
            cache_sha256="0" * 64,
            replay_cache_path=source,
            replay_cache_sha256=source_sha,
            output=tmp_path / "audit.json",
        )
