from __future__ import annotations

import copy

import pytest
import torch

from radio_gs.interfaces.surface_region_target_adaptive_typed_context import (
    validate_target_adaptive_typed_context_authority,
)
from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    ADAPTIVE_WORKING_MEMORY_CEILING_BYTES,
    AdaptiveTypedBudgetSelection,
    TERMINATION_NATURAL,
    adaptive_typed_context_channel_sha256,
)
from radio_gs.scripts.materialize_target_adaptive_typed_context_authority import (
    assemble_target_adaptive_authority_payload,
)


def _sha(character: str) -> str:
    return character * 64


def _selection(
    rows: list[int], core: list[bool], distances: list[float]
) -> AdaptiveTypedBudgetSelection:
    return AdaptiveTypedBudgetSelection(
        rows=torch.tensor(rows),
        core_mask=torch.tensor(core),
        context_mask=~torch.tensor(core),
        semantic_geodesic_distance=torch.tensor(distances),
        termination=TERMINATION_NATURAL,
        final_probe_width=max(1, len(rows)),
        settled_candidate_count=max(1, len(rows)),
        adaptive_round_count=1,
    )


def _payload() -> dict:
    selections = (
        _selection([0, 1], [True, False], [0.0, 0.26]),
        _selection([2], [True], [0.0]),
    )
    direction = torch.zeros(2, 1280, dtype=torch.float16)
    direction[0, 0] = 1.0
    statistics = torch.zeros(2, 12)
    statistics[0, 0] = -1.0
    geometry_sha = _sha("9")
    accepted = {
        "scene_id": "figurines",
        "physical_space_id": (
            "lerf:figurines:geometry-checkpoint-sha256:" + geometry_sha
        ),
        "physical_space_authority": {
            "kind": "target_geometry_checkpoint_v1",
            "dataset_id": "lerf",
            "scene_id": "figurines",
            "geometry_checkpoint_sha256": geometry_sha,
            "physical_space_id": (
                "lerf:figurines:geometry-checkpoint-sha256:" + geometry_sha
            ),
        },
        "region_fingerprints": [_sha("1"), _sha("2")],
        "channel_sha256": {"region_rows": _sha("3")},
        "canonical_region_indices": torch.tensor([4, 9]),
        "scale_indices": torch.tensor([0, 1]),
    }
    return assemble_target_adaptive_authority_payload(
        accepted=accepted,
        accepted_file={"path": "/synthetic/accepted.pt", "sha256": _sha("a")},
        field_file={"path": "/synthetic/field.pt", "sha256": _sha("b")},
        state_file={"path": "/synthetic/state.pt", "sha256": _sha("c")},
        factorized_radio_cache_sha256=_sha("d"),
        graph_file={"path": "/synthetic/graph.pt", "sha256": _sha("e")},
        primitive_row_authority_sha256=_sha("f"),
        anchor_local_rows=torch.tensor([0, 2]),
        anchor_global_rows=torch.tensor([100, 102]),
        selections=selections,
        memory_audit={
            "memory_ceiling_bytes": ADAPTIVE_WORKING_MEMORY_CEILING_BYTES,
            "maximum_estimated_working_bytes": 2 * 1024 * 1024,
            "requested_batch_size": 2,
        },
        pooled={
            "pooled_context_radio_direction": direction,
            "typed_context_statistics": statistics,
            "context_present": torch.tensor([True, False]),
            "typed_context_valid": torch.tensor([True, False]),
            "context_token_count": torch.tensor([1, 0]),
            "context_token_row_offsets": torch.tensor([0, 1, 1]),
            "context_token_local_rows": torch.tensor([1]),
            "context_token_global_rows": torch.tensor([101]),
        },
        producer={"path": "/synthetic/target-adaptive.py", "sha256": _sha("8")},
    )


def test_target_adaptive_accepts_lerf_identity_without_e0_or_query() -> None:
    payload = _payload()
    validated = validate_target_adaptive_typed_context_authority(payload)
    assert validated["scene_id"] == "figurines"
    assert validated["typed_context_valid"].tolist() == [True, False]
    assert "accepted_v2_e0" not in validated
    assert validated["access_audit"]["target_metrics_computed"] is False


def test_target_adaptive_physical_space_binding_fails_closed() -> None:
    payload = _payload()
    payload["physical_space_id"] = "lerf:figurines:geometry-checkpoint-sha256:" + _sha(
        "7"
    )
    with pytest.raises(ValueError, match="physical-space binding"):
        validate_target_adaptive_typed_context_authority(payload)


def test_target_adaptive_preserves_inactive_zero_and_channel_sha() -> None:
    payload = _payload()
    tampered = copy.deepcopy(payload)
    tampered["typed_context_statistics"][1, 0] = 1.0
    tampered["channel_sha256"] = adaptive_typed_context_channel_sha256(tampered)
    with pytest.raises(ValueError, match="inactive"):
        validate_target_adaptive_typed_context_authority(tampered)

    tampered = copy.deepcopy(payload)
    tampered["context_token_global_rows"][0] = 999
    with pytest.raises(ValueError, match="SHA"):
        validate_target_adaptive_typed_context_authority(tampered)
