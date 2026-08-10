"""Seal the source-only exact4x2 authority for the three gauge/state arms."""

from pathlib import Path
import json

from radio_gs.models.factorized_native_gauge_state_readout import (
    FACTORIZED_NATIVE_READOUT_ARMS,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2 as trainer,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json


repo = Path("/root/RADIO-GS")
pool = Path("/mnt/pool/sqy/results/RADIO-GS/output")
destination = repo / (
    "paper/artifacts/"
    "factorized_native_gauge_state_readout_exact4x2_"
    "execution_authority_20260807.json"
)
if destination.exists() or destination.is_symlink():
    raise FileExistsError(f"refusing to clobber authority: {destination}")

cohort = repo / (
    "paper/artifacts/"
    "full_scalar_scannet_clean_24train_8validation_cohort_authority_20260805.json"
)
registry = pool / (
    "optimization_20260807/"
    "full_scalar_clean_cohort_region_view_registry/"
    "pilot_exact4train_2validation/region_view_registry_v1.json"
)
exclusion = repo / (
    "paper/artifacts/"
    "full_scalar_clean_benchmark_exclusion_manifest_v2_20260805.json"
)
radio = Path("/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
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


def scene_record(scene: str) -> dict:
    run = run_roots[scene]
    return {
        "scene_id": scene,
        "training_shard": file_record(
            shard_root / scene / "training_shard_v1.pt"
        ),
        "accepted_region_authority": file_record(
            run
            / "accepted_v2_source_only"
            / scene
            / "accepted_v2_sparse_v2.pt"
        ),
        "factorized_state": file_record(
            run
            / "exact_state"
            / scene
            / "factorized_primitive_state_v2.pt"
        ),
    }


authority = {
    "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
    "schema_version": 1,
    "status": "authorized_source_only_exact4train_2validation",
    **{
        name: file_record(path)
        for name, path in trainer._expected_code_paths().items()
    },
    "cohort_authority": file_record(cohort),
    "pilot_cohort_region_view_registry": file_record(registry),
    "benchmark_exclusion_manifest": file_record(exclusion),
    "official_radio_checkpoint": file_record(radio),
    "source_train": [scene_record(scene) for scene in trainer.TRAIN_SCENES],
    "source_validation": [
        scene_record(scene) for scene in trainer.VALIDATION_SCENES
    ],
    "authorized_arms": list(FACTORIZED_NATIVE_READOUT_ARMS),
    "training_authorized": True,
    "benchmark_execution_authorized": False,
    "source_access": trainer.source_access(),
}
trainer.validate_execution_authority(authority)
written = write_frozen_json(destination, authority)
record = file_record(written)
trainer.prepare_inputs(written, expected_sha256=record["sha256"])
print(
    json.dumps(
        {
            "status": "factorized-native exact4x2 source authority sealed",
            "execution_authority": record,
            "authorized_arms": list(FACTORIZED_NATIVE_READOUT_ARMS),
            "benchmark_opened": False,
        },
        indent=2,
    )
)
