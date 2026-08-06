import copy
from argparse import Namespace

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    ADAPTIVE_WORKING_MEMORY_CEILING_BYTES,
    AdaptiveTypedBudgetSelection,
    TERMINATION_NATURAL,
    adaptive_typed_context_channel_sha256,
    validate_adaptive_typed_context_authority,
)
from radio_gs.scripts.materialize_accepted_v2_typed_context_adaptive_authority import (
    _pool_adaptive_rows,
    assemble_adaptive_authority_payload,
    materialize,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
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


class _Field(torch.nn.Module):
    def radio_features(self, rows: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(rows.numel(), 1280, device=rows.device)
        result[:, 0] = 1.0
        result[:, 1] = rows.float() / 100.0
        return result


def test_adaptive_pool_uses_only_final_context_rows_and_preserves_mapping() -> None:
    selection = _selection([0, 1], [True, False], [0.0, 0.26])
    pooled = _pool_adaptive_rows(
        [selection],
        radii_m=(0.25, 0.45, 0.70),
        context_ratio=1.2,
        scale_indices=torch.tensor([0]),
        anchor_local_rows=torch.tensor([0]),
        graph_global_rows=torch.tensor([100, 101]),
        reliability=torch.ones(2),
        field=_Field(),
        field_batch_size=2,
        device=torch.device("cpu"),
    )
    assert pooled["typed_context_valid"].tolist() == [True]
    assert pooled["context_token_local_rows"].tolist() == [1]
    assert pooled["context_token_global_rows"].tolist() == [101]
    assert pooled["pooled_context_radio_direction"].shape == (1, 1280)


def _payload() -> dict:
    selections = (
        _selection([0, 1], [True, False], [0.0, 0.26]),
        _selection([2], [True], [0.0]),
    )
    direction = torch.zeros(2, 1280, dtype=torch.float16)
    direction[0, 0] = 1.0
    statistics = torch.zeros(2, 12)
    statistics[0, 0] = -1.0
    accepted = {
        "scene_id": "scene0001_00",
        "physical_space_id": "scene0001",
        "region_fingerprints": [_sha("1"), _sha("2")],
        "channel_sha256": {"region_rows": _sha("3")},
        "canonical_region_indices": torch.tensor([4, 9]),
        "scale_indices": torch.tensor([0, 1]),
    }
    return assemble_adaptive_authority_payload(
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
    )


def test_v2_authority_is_no_e0_sha_bound_and_fail_closed() -> None:
    payload = _payload()
    assert "accepted_v2_e0" not in payload
    assert validate_adaptive_typed_context_authority(payload)[
        "typed_context_valid"
    ].tolist() == [True, False]

    tampered = copy.deepcopy(payload)
    tampered["typed_context_statistics"][1, 0] = 1.0
    tampered["channel_sha256"] = adaptive_typed_context_channel_sha256(tampered)
    with pytest.raises(ValueError, match="inactive"):
        validate_adaptive_typed_context_authority(tampered)

    tampered = copy.deepcopy(payload)
    tampered["context_token_global_rows"][0] = 999
    with pytest.raises(ValueError, match="SHA"):
        validate_adaptive_typed_context_authority(tampered)

    tampered = copy.deepcopy(payload)
    tampered["memory_audit"]["maximum_estimated_working_bytes"] = (
        ADAPTIVE_WORKING_MEMORY_CEILING_BYTES + 1
    )
    with pytest.raises(ValueError, match="memory"):
        validate_adaptive_typed_context_authority(tampered)


def test_materializer_rejects_oversized_candidate_batch_before_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="strict maximum"):
        materialize(
            Namespace(
                output=str(tmp_path / "new.pt"),
                candidate_batch_size=9,
                field_batch_size=1,
            )
        )


def test_v2_carrier_keeps_inactive_and_ood_rows_bitwise_e0() -> None:
    payload = _payload()
    base = F.normalize(torch.randn(2, 1536), dim=-1)
    model = SurfaceRegionAcceptedV2TypedContextResidualV1()
    with torch.no_grad():
        model.residual_projection.bias.copy_(torch.linspace(-0.5, 0.5, 1536))
    full_scalar = torch.randn(2, 18)
    output = model(
        base,
        payload["pooled_context_radio_direction"],
        full_scalar,
        payload["typed_context_statistics"],
        active_mask=payload["typed_context_valid"],
    )
    assert torch.equal(output[1], base[1])
    assert not torch.equal(output[0], base[0])
    ood_output = model(
        base,
        payload["pooled_context_radio_direction"],
        full_scalar,
        payload["typed_context_statistics"],
        active_mask=payload["typed_context_valid"],
        ood_mask=torch.tensor([True, False]),
    )
    assert torch.equal(ood_output, base)
