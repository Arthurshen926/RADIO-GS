import copy

import pytest

from uqis_benchmark.method_fields import (
    FIELD_INVENTORY_SCHEMA,
    FIELD_INVENTORY_SCHEMA_V2,
    ludvig_modality_field_plan,
    validate_method_field_inventory,
)
from uqis_benchmark.protocol import (
    BENCHMARK_VERSION,
    BENCHMARK_VERSION_V2_CANDIDATE,
    canonical_json_sha256,
)


def _inventory(fields):
    scene = {
        "scene_id": "scene0000_00",
        "fields": fields,
        "field_count": len(fields),
        "persistent_bytes": sum(
            {artifact["sha256"]: artifact["bytes"] for field in fields for artifact in field["artifacts"]}.values()
        ),
    }
    scope = "single_universal_field" if len(fields) == 1 else "modality_specific_multi_field"
    payload = {
        "schema_version": FIELD_INVENTORY_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "complete",
        "method_system_id": "ludvig_uqis_v1",
        "method_identity_sha256": "a" * 64,
        "representation_scope": scope,
        "scene_count": 1,
        "scenes": [scene],
        "totals": {
            "field_count": len(fields),
            "persistent_bytes": scene["persistent_bytes"],
        },
    }
    payload["inventory_sha256"] = canonical_json_sha256(payload)
    return payload


def _field(field_id, modalities, digest, size):
    return {
        "field_id": field_id,
        "field_family": field_id + "_family",
        "modalities": modalities,
        "mapping_receipt_sha256": digest,
        "artifacts": [{"relative_path": "field.bin", "bytes": size, "sha256": digest}],
    }


def test_ludvig_plan_is_two_fields_with_complete_modality_coverage() -> None:
    plan = ludvig_modality_field_plan()
    assert len(plan) == 2
    assert {modality for field in plan for modality in field["modalities"]} == {
        "text", "image", "point_2d", "point_3d"
    }


def test_multi_field_inventory_counts_all_unique_persistent_storage() -> None:
    payload = _inventory(
        [
            _field("clip", ["text"], "b" * 64, 11),
            _field("dino", ["image", "point_2d", "point_3d"], "c" * 64, 17),
        ]
    )
    validated = validate_method_field_inventory(payload, expected_scene_ids=["scene0000_00"])
    assert validated["representation_scope"] == "modality_specific_multi_field"
    assert validated["totals"] == {"field_count": 2, "persistent_bytes": 28}


def test_field_inventory_rejects_double_assignment_and_storage_understatement() -> None:
    payload = _inventory(
        [
            _field("clip", ["text", "image"], "b" * 64, 11),
            _field("dino", ["image", "point_2d", "point_3d"], "c" * 64, 17),
        ]
    )
    with pytest.raises(ValueError, match="exactly one field"):
        validate_method_field_inventory(payload)

    payload = _inventory(
        [
            _field("clip", ["text"], "b" * 64, 11),
            _field("dino", ["image", "point_2d", "point_3d"], "c" * 64, 17),
        ]
    )
    tampered = copy.deepcopy(payload)
    tampered["scenes"][0]["persistent_bytes"] = 1
    with pytest.raises(ValueError, match="persistent storage"):
        validate_method_field_inventory(tampered)


def test_v2_field_dependency_sets_allow_clip_dino_text_diffusion() -> None:
    fields = [
        {
            key: value
            for key, value in _field("clip", [], "b" * 64, 11).items()
            if key != "modalities"
        },
        {
            key: value
            for key, value in _field("dino", [], "c" * 64, 17).items()
            if key != "modalities"
        },
    ]
    scene = {
        "scene_id": "scene0000_00",
        "fields": fields,
        "modality_field_dependencies": {
            "text": ["clip", "dino"],
            "image": ["dino"],
            "point_2d": ["dino"],
            "point_3d": ["dino"],
        },
        "field_count": 2,
        "persistent_bytes": 28,
    }
    body = {
        "schema_version": FIELD_INVENTORY_SCHEMA_V2,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "complete",
        "method_system_id": "ludvig_uqis_v2_diffusion",
        "method_identity_sha256": "a" * 64,
        "representation_scope": "modality_specific_multi_field",
        "scene_count": 1,
        "scenes": [scene],
        "totals": {"field_count": 2, "persistent_bytes": 28},
    }
    payload = {**body, "inventory_sha256": canonical_json_sha256(body)}

    validated = validate_method_field_inventory(payload)

    assert validated["scenes"][0]["modality_field_dependencies"]["text"] == [
        "clip",
        "dino",
    ]
    assert validated["totals"]["persistent_bytes"] == 28


def test_v2_field_inventory_accepts_candidate_release_identity() -> None:
    fields = [
        {
            key: value
            for key, value in _field("shared", [], "b" * 64, 11).items()
            if key != "modalities"
        }
    ]
    scene = {
        "scene_id": "scene0000_00",
        "fields": fields,
        "modality_field_dependencies": {
            modality: ["shared"]
            for modality in ("text", "image", "point_2d", "point_3d")
        },
        "field_count": 1,
        "persistent_bytes": 11,
    }
    body = {
        "schema_version": FIELD_INVENTORY_SCHEMA_V2,
        "benchmark_version": BENCHMARK_VERSION_V2_CANDIDATE,
        "status": "complete",
        "method_system_id": "candidate_universal_field",
        "method_identity_sha256": "a" * 64,
        "representation_scope": "single_universal_field",
        "scene_count": 1,
        "scenes": [scene],
        "totals": {"field_count": 1, "persistent_bytes": 11},
    }
    payload = {**body, "inventory_sha256": canonical_json_sha256(body)}

    validated = validate_method_field_inventory(payload)

    assert validated["benchmark_version"] == BENCHMARK_VERSION_V2_CANDIDATE


def test_v2_field_dependency_sets_reject_undeclared_field() -> None:
    fields = [
        {
            key: value
            for key, value in _field("clip", [], "b" * 64, 11).items()
            if key != "modalities"
        }
    ]
    scene = {
        "scene_id": "scene0000_00",
        "fields": fields,
        "modality_field_dependencies": {
            "text": ["clip", "missing"],
            "image": ["clip"],
            "point_2d": ["clip"],
            "point_3d": ["clip"],
        },
        "field_count": 1,
        "persistent_bytes": 11,
    }
    body = {
        "schema_version": FIELD_INVENTORY_SCHEMA_V2,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "complete",
        "method_system_id": "bad",
        "method_identity_sha256": "a" * 64,
        "representation_scope": "single_universal_field",
        "scene_count": 1,
        "scenes": [scene],
        "totals": {"field_count": 1, "persistent_bytes": 11},
    }
    payload = {**body, "inventory_sha256": canonical_json_sha256(body)}
    with pytest.raises(ValueError, match="undeclared field"):
        validate_method_field_inventory(payload)
