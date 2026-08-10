from pathlib import Path
import json

from radio_gs.evaluation.source_query_response_hard_negatives import (
    validate_negative_authority,
)
from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    validate_adaptive_typed_context_authority,
)
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    load_frozen_canonical_negative_bank,
    load_frozen_compositional_generic_bank,
)
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    load_frozen_typed_text_relation_authority,
)
from radio_gs.scripts import (
    build_surface_region_v21_pilot_execution_authority as authority_builder,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v2 as v2_trainer,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as pilot,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    write_frozen_json,
)


repo = Path("/root/RADIO-GS")
pool = Path("/mnt/pool/sqy/results/RADIO-GS/output")
spec_path = repo / (
    "paper/artifacts/"
    "surface_region_v21_pilot_execution_build_spec_20260807.json"
)
if spec_path.exists() or spec_path.is_symlink():
    raise FileExistsError(f"refusing to clobber build spec: {spec_path}")

cohort_path = repo / (
    "paper/artifacts/"
    "full_scalar_scannet_clean_24train_8validation_cohort_authority_20260805.json"
)
registry_path = pool / (
    "optimization_20260807/"
    "full_scalar_clean_cohort_region_view_registry/"
    "pilot_exact4train_2validation/region_view_registry_v1.json"
)
exclusion_path = repo / (
    "paper/artifacts/"
    "full_scalar_clean_benchmark_exclusion_manifest_v2_20260805.json"
)
fit_path = pool / (
    "optimization_20260731/target_blind_siglip2_text_bank_v1/"
    "target_blind_siglip2_fit_embeddings.pt"
)
canonical_path = repo / (
    "checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt"
)
component_root = pool / (
    "optimization_20260807/"
    "target_blind_compositional_siglip2_v2_fit_gpu1"
)
component_paths = {
    "synonym_relation": component_root / "synonym_relation_fit_embeddings.pt",
    "lexical_sibling_relation": (
        component_root / "lexical_sibling_relation_fit_embeddings.pt"
    ),
    "counterfactual_attributes": (
        component_root / "counterfactual_attributes_fit_embeddings.pt"
    ),
    "high_precision_part_of": (
        component_root / "high_precision_part_of_fit_embeddings.pt"
    ),
}
relation_path = pool / (
    "optimization_20260807/"
    "target_blind_typed_text_relation_authority_v1/"
    "fit_relation_indices.pt"
)
shard_root = pool / (
    "optimization_20260807/"
    "full_scalar_clean_pilot_4train_2validation_v21/shards_v1"
)

run_roots = {
    "scene0001_00": pool / (
        "optimization_20260805/"
        "full_scalar_clean_scannet_pilot_scene0001_v1/run"
    ),
    **{
        scene: pool / (
            "optimization_20260805/"
            f"full_scalar_clean_scannet_cohort_v3/{scene}/run"
        )
        for scene in (
            "scene0002_00",
            "scene0003_00",
            "scene0004_00",
            "scene0005_00",
            "scene0008_00",
        )
    },
}
hard_negative_paths = {
    "scene0001_00": pool / (
        "optimization_20260806/"
        "source_query_response_hard_negatives_v1/scene0001_00/"
        "hard_negative_index_authority_bound_v1.pt"
    ),
    **{
        scene: pool / (
            "optimization_20260807/"
            f"source_query_response_hard_negatives_cohort_v2/{scene}/"
            "hard_negative_index_authority_bound_v1.pt"
        )
        for scene in (
            "scene0002_00",
            "scene0003_00",
            "scene0004_00",
            "scene0005_00",
            "scene0008_00",
        )
    },
}

fit_record = file_record(fit_path)
fit = v2_trainer.load_fit_text_bank(
    fit_path,
    expected_sha256=fit_record["sha256"],
)
if fit.record != fit_record:
    raise ValueError("primary fit bank verified record differs")

canonical_record = file_record(canonical_path)
canonical = load_frozen_canonical_negative_bank(
    canonical_path,
    expected_file_sha256=canonical_record["sha256"],
)
if canonical.file_sha256 != canonical_record["sha256"]:
    raise ValueError("canonical-negative bank verified record differs")

compositional_records = {}
for component_id, weight in pilot.COMPONENT_WEIGHTS.items():
    path = component_paths[component_id]
    record = file_record(path)
    bank = load_frozen_compositional_generic_bank(
        path,
        expected_file_sha256=record["sha256"],
        component_id=component_id,
        loss_weight=weight,
    )
    if bank.file_sha256 != record["sha256"]:
        raise ValueError(f"{component_id} verified record differs")
    compositional_records[component_id] = {
        **record,
        "loss_weight": weight,
    }

relation_record = file_record(relation_path)
relations = load_frozen_typed_text_relation_authority(
    relation_path,
    expected_file_sha256=relation_record["sha256"],
)
if relations.file_sha256 != relation_record["sha256"]:
    raise ValueError("typed relation verified record differs")

expected_relation_components = {
    "primary": fit_record["sha256"],
    **{
        name: compositional_records[name]["sha256"]
        for name in pilot.COMPONENT_WEIGHTS
    },
}
for name, expected_sha in expected_relation_components.items():
    if relations.components[name]["sha256"] != expected_sha:
        raise ValueError(f"typed relation component binding differs: {name}")

adaptive_channel_audit = {}


def scene_record(scene_id):
    shard = file_record(shard_root / scene_id / "training_shard_v1.pt")
    adaptive_path = (
        run_roots[scene_id]
        / "accepted_v2_source_only"
        / scene_id
        / "typed_context_stage_b_adaptive_v2.pt"
    )
    adaptive_raw, adaptive_sha, adaptive_source = load_torch_mapping(
        adaptive_path,
        map_location="cpu",
        label=f"{scene_id} adaptive typed-context authority",
    )
    adaptive = validate_adaptive_typed_context_authority(adaptive_raw)
    if adaptive["scene_id"] != scene_id:
        raise ValueError(f"adaptive typed-context scene differs: {scene_id}")
    adaptive_channel_audit[scene_id] = canonical_json_sha256(
        adaptive["channel_sha256"]
    )

    negative_raw, negative_sha, negative_source = load_torch_mapping(
        hard_negative_paths[scene_id],
        map_location="cpu",
        label=f"{scene_id} hard-negative authority",
    )
    negative = validate_negative_authority(negative_raw)
    if negative["scene_id"] != scene_id:
        raise ValueError(f"hard-negative scene differs: {scene_id}")

    return {
        "scene_id": scene_id,
        "training_shard": shard,
        "adaptive_context": {
            "path": str(adaptive_source),
            "sha256": adaptive_sha,
        },
        "hard_negative_authority": {
            "path": str(negative_source),
            "sha256": negative_sha,
        },
        "hard_negative_content_authority_sha256": negative[
            "content_authority_sha256"
        ],
    }


spec = {
    "schema": authority_builder.BUILD_SPEC_SCHEMA,
    "schema_version": authority_builder.SCHEMA_VERSION,
    "cohort_authority": file_record(cohort_path),
    "pilot_cohort_region_view_registry": file_record(registry_path),
    "benchmark_exclusion_manifest": file_record(exclusion_path),
    "fit_text_bank": fit_record,
    "canonical_negative_bank": canonical_record,
    "compositional_banks": compositional_records,
    "typed_relation_authority": {
        **relation_record,
        "content_authority_sha256": relations.content_authority_sha256,
    },
    "source_train": [scene_record(scene) for scene in pilot.TRAIN_SCENES],
    "source_validation": [
        scene_record(scene) for scene in pilot.VALIDATION_SCENES
    ],
}

authority_builder.validate_build_spec(spec)
written = write_frozen_json(spec_path, spec)

print(
    json.dumps(
        {
            "status": "source-only V2.1 build spec sealed",
            "build_spec": file_record(written),
            "typed_relation_content_authority_sha256": (
                relations.content_authority_sha256
            ),
            "hard_negative_content_authority_sha256": {
                row["scene_id"]: row[
                    "hard_negative_content_authority_sha256"
                ]
                for row in spec["source_train"] + spec["source_validation"]
            },
            "adaptive_channel_manifest_sha256_audit_only": (
                adaptive_channel_audit
            ),
            "benchmark_target_or_query_opened": False,
        },
        indent=2,
    )
)
