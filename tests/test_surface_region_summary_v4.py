import copy
import hashlib

import pytest
import torch

from radio_gs.interfaces.surface_region_summary import (
    SURFACE_SUMMARY_READOUT_V4_SCHEMA_VERSION,
    SurfaceRegionSummaryReadoutV2,
    SurfaceRegionSummaryReadoutV4,
    surface_region_geometry_v3,
    surface_region_state_dict_sha256,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(contract_sha256: str) -> dict:
    return {
        "training_scope": "global_cross_scene_3d_surface_v2",
        "frozen": True,
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "scene_disjoint": True,
        "official_summary_head": "c-radio_v4 siglip2-g",
        "custom_text_projection": False,
        "region_contract_sha256": contract_sha256,
        "train": {
            "scenes": ["train_scene"],
            "region_contract_sha256": contract_sha256,
        },
        "validation": {
            "scenes": ["validation_scene"],
            "region_contract_sha256": contract_sha256,
        },
    }


def _write_v2_checkpoint(tmp_path):
    torch.manual_seed(83)
    contract_sha256 = "a" * 64
    base = SurfaceRegionSummaryReadoutV2(feature_dim=8, hidden_dim=6).eval()
    with torch.no_grad():
        base.residual[-1].weight.normal_(std=0.02)
        base.residual[-1].bias.normal_(std=0.02)
    architecture = base.architecture(contract_sha256)
    state = base.state_dict()
    provenance = _provenance(contract_sha256)
    payload = {
        "schema_version": 3,
        "architecture": architecture,
        "state_dict": state,
        "provenance": provenance,
    }
    path = tmp_path / "base-v2.pt"
    torch.save(payload, path)
    authority = {
        "checkpoint": _sha256(path),
        "architecture": architecture["digest"],
        "state": surface_region_state_dict_sha256(state),
        "provenance": canonical_json_sha256(provenance),
        "contract": contract_sha256,
    }
    model, _ = SurfaceRegionSummaryReadoutV4.from_v2_checkpoint(
        path,
        expected_checkpoint_sha256=authority["checkpoint"],
        expected_architecture_sha256=authority["architecture"],
        expected_state_dict_sha256=authority["state"],
        expected_provenance_sha256=authority["provenance"],
        expected_contract_sha256=authority["contract"],
    )
    return path, model, authority


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(89)
    raw = torch.randn(2, 6, 8)
    token_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, False],
        ]
    )
    raw[1, 5] = 0.0
    raw_norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    direction = torch.nn.functional.normalize(raw, dim=-1).masked_fill(
        ~token_mask[..., None], 0.0
    )
    xyz = torch.randn(2, 6, 3)
    scale = torch.rand(2, 6, 3) * 0.04 + 0.01
    reliability = torch.tensor(
        [
            [[0.9], [0.8], [0.7], [0.6], [0.5], [0.4]],
            [[0.9], [0.8], [0.7], [0.6], [0.5], [0.0]],
        ]
    )
    core = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, True, False, False, False, False],
        ]
    )
    context = torch.tensor(
        [
            [False, False, True, True, False, False],
            [False, False, True, False, False, False],
        ]
    )
    support_fill = token_mask & ~core & ~context
    anchor = torch.tensor([0, 1])
    geometry = surface_region_geometry_v3(
        xyz,
        scale,
        reliability,
        torch.tensor([0.3, 0.6]),
        raw_radio_l2_norm=raw_norm,
        anchor_index=anchor,
        core_mask=core,
        context_mask=context,
        support_fill_mask=support_fill,
        token_mask=token_mask,
    )
    return direction, geometry, reliability, token_mask, support_fill, anchor


def test_v4_is_exact_v2_raw_gauge_forward_and_has_no_residual(tmp_path) -> None:
    _, model, _ = _write_v2_checkpoint(tmp_path)
    direction, geometry, reliability, mask, support_fill, anchor = _inputs()
    base_mask = mask & ~support_fill
    raw = (direction.float() * torch.exp(geometry[..., 15:16])).masked_fill(
        ~base_mask[..., None], 0.0
    )
    base_geometry = geometry[..., :14].masked_fill(~base_mask[..., None], 0.0)
    base_reliability = geometry[..., 6:7].masked_fill(
        ~base_mask[..., None], 0.0
    )
    expected, expected_context = model.base_readout.forward_with_context(
        raw,
        base_geometry,
        anchor_index=anchor,
        token_mask=base_mask,
        reliability=base_reliability,
    )
    actual, context = model.forward_with_context(
        direction,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(context, expected_context, rtol=0, atol=0)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    architecture = model.architecture("b" * 64)
    assert architecture["trainable_parameter_count"] == 0
    assert architecture["residual_mode"] == "disabled_exact_fallback_v1"
    assert architecture["ood_gate"] == "not_applicable_residual_disabled"
    model.train()
    assert not model.base_readout.training


def test_v4_support_fill_is_exactly_excluded(tmp_path) -> None:
    _, model, _ = _write_v2_checkpoint(tmp_path)
    direction, geometry, reliability, mask, support_fill, anchor = _inputs()
    expected = model(
        direction,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    changed_direction = direction.clone()
    changed_direction[support_fill] = -changed_direction[support_fill]
    changed_geometry = geometry.clone()
    changed_geometry[..., 0][support_fill] += 100.0
    changed_geometry[..., 15][support_fill] += 3.0
    actual = model(
        changed_direction,
        changed_geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=reliability,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_v4_checkpoint_schema_and_all_base_authorities_fail_closed(tmp_path) -> None:
    base_path, model, authority = _write_v2_checkpoint(tmp_path)
    payload = model.checkpoint_payload("c" * 64)
    checkpoint = tmp_path / "readout-v4.pt"
    torch.save(payload, checkpoint)
    restored, reopened = SurfaceRegionSummaryReadoutV4.from_checkpoint(
        checkpoint,
        expected_checkpoint_sha256=_sha256(checkpoint),
        expected_base_checkpoint_sha256=authority["checkpoint"],
        expected_base_architecture_sha256=authority["architecture"],
        expected_base_state_dict_sha256=authority["state"],
        expected_base_provenance_sha256=authority["provenance"],
        expected_base_contract_sha256=authority["contract"],
    )
    assert reopened["schema_version"] == SURFACE_SUMMARY_READOUT_V4_SCHEMA_VERSION
    assert restored.architecture("c" * 64) == payload["architecture"]

    wrong_schema = tmp_path / "wrong-schema.pt"
    torch.save({**payload, "schema_version": 8}, wrong_schema)
    with pytest.raises(ValueError, match="invalid V4"):
        SurfaceRegionSummaryReadoutV4.from_checkpoint(
            wrong_schema,
            expected_base_checkpoint_sha256=authority["checkpoint"],
            expected_base_architecture_sha256=authority["architecture"],
            expected_base_state_dict_sha256=authority["state"],
            expected_base_provenance_sha256=authority["provenance"],
            expected_base_contract_sha256=authority["contract"],
        )

    tampered = copy.deepcopy(payload)
    first = next(iter(tampered["state_dict"]))
    tampered["state_dict"][first].view(-1)[0] += 1.0
    tampered_path = tmp_path / "tampered-state.pt"
    torch.save(tampered, tampered_path)
    with pytest.raises(ValueError, match="state_dict authority differs"):
        SurfaceRegionSummaryReadoutV4.from_checkpoint(
            tampered_path,
            expected_base_checkpoint_sha256=authority["checkpoint"],
            expected_base_architecture_sha256=authority["architecture"],
            expected_base_state_dict_sha256=authority["state"],
            expected_base_provenance_sha256=authority["provenance"],
            expected_base_contract_sha256=authority["contract"],
        )

    with pytest.raises(ValueError, match="SHA-256 differs"):
        SurfaceRegionSummaryReadoutV4.from_v2_checkpoint(
            base_path,
            expected_checkpoint_sha256="f" * 64,
            expected_architecture_sha256=authority["architecture"],
            expected_state_dict_sha256=authority["state"],
            expected_provenance_sha256=authority["provenance"],
            expected_contract_sha256=authority["contract"],
        )


def test_v4_rejects_fill_anchor_and_conflicting_reliability(tmp_path) -> None:
    _, model, _ = _write_v2_checkpoint(tmp_path)
    direction, geometry, reliability, mask, support_fill, anchor = _inputs()
    fill_anchor = anchor.clone()
    fill_anchor[0] = int(torch.nonzero(support_fill[0], as_tuple=False)[0])
    with pytest.raises(ValueError, match="non-fill core"):
        model(
            direction,
            geometry,
            anchor_index=fill_anchor,
            token_mask=mask,
            reliability=reliability,
        )
    conflicting = reliability.clone()
    conflicting[0, 0] = 0.1
    with pytest.raises(ValueError, match="authoritative geometry index 6"):
        model(
            direction,
            geometry,
            anchor_index=anchor,
            token_mask=mask,
            reliability=conflicting,
        )
