import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import radio_gs.scripts.build_surface_region_semantic_cache as semantic_builder

from radio_gs.interfaces.surface_region_contract import (
    SurfaceRegionContractV2,
    SurfaceRegionContractV3,
)
from radio_gs.interfaces.surface_region_summary import (
    SURFACE_REGION_V3_GATED_RAW_PRIOR,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.build_surface_region_semantic_cache import (
    _adjacency,
    _full_scalar_overlay_arguments,
    _aggregate_prevalidated_full_scalars,
    _validate_full_scalar_overlay_mode,
    _validate_full_scalar_runtime_base_authority,
    apply_accepted_v2_full_scalar_overlay,
    completion_primary_valid,
    expand_surface_region_v3_batch_at_radius,
    load_surface_factorized_state_bundle,
    preserve_primary_region_tokens,
    two_hop_physical_regions,
    validate_query_router_deployment_gauge,
    validate_surface_region_readout_deployment_authority,
    AcceptedV2FullScalarRuntimeCarrier,
)


def test_full_scalar_cli_is_strictly_all_or_none() -> None:
    assert _full_scalar_overlay_arguments(SimpleNamespace()) is None
    complete = {
        "accepted_v2_full_scalar_state": "state.pt",
        "accepted_v2_full_scalar_state_sha256": "a" * 64,
        "full_scalar_normalization_authority": "normalization.pt",
        "full_scalar_normalization_authority_sha256": "b" * 64,
        "full_scalar_residual_checkpoint": "residual.pt",
        "full_scalar_residual_checkpoint_sha256": "c" * 64,
        "full_scalar_training_certificate": "certificate.json",
        "full_scalar_training_certificate_sha256": "d" * 64,
    }
    assert _full_scalar_overlay_arguments(SimpleNamespace(**complete)) == complete
    partial = dict(complete)
    partial["full_scalar_residual_checkpoint_sha256"] = ""
    with pytest.raises(ValueError, match="requires every"):
        _full_scalar_overlay_arguments(SimpleNamespace(**partial))


def test_disabled_full_scalar_path_keeps_legacy_normalize_then_half_bitwise() -> None:
    torch.manual_seed(7)
    official_head_output = torch.randn(11, 1536)
    legacy = F.normalize(official_head_output.float(), dim=-1).half()
    immutable_e0 = F.normalize(official_head_output.float(), dim=-1)
    disabled_overlay = immutable_e0.half()
    assert torch.equal(disabled_overlay, legacy)


def test_prevalidated_full_scalar_carrier_matches_anchor_weighted_aggregation() -> None:
    scalars = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    weights = torch.tensor([1.0, 3.0, 2.0])
    carrier = AcceptedV2FullScalarRuntimeCarrier(
        scalar_source=scalars,
        reliability_source=weights,
        global_to_compact=torch.tensor([0, -1, 1, 2]),
        overlap_mask=torch.tensor([True, False, True, False]),
        base_only_mask=torch.tensor([False, True, False, False]),
        exact_only_mask=torch.tensor([False, False, False, True]),
        neither_mask=torch.zeros(4, dtype=torch.bool),
    )
    result = _aggregate_prevalidated_full_scalars(
        carrier,
        torch.tensor([[0, 2, 0], [1, 3, 0]]),
        torch.tensor([[True, True, False], [True, True, False]]),
        torch.tensor([0, 0]),
    )
    selected = scalars[:2]
    selected_weights = weights[:2]
    mean = (selected * selected_weights[:, None]).sum(0) / selected_weights.sum()
    variance = (
        (selected - mean).square() * selected_weights[:, None]
    ).sum(0) / selected_weights.sum()
    expected = torch.cat((scalars[0], mean, variance.sqrt()))
    torch.testing.assert_close(result.summary[0], expected)
    assert not bool(result.summary[1].any())
    assert torch.equal(result.use_full_scalar_mask, torch.tensor([True, False]))
    assert torch.equal(result.base_fallback_mask, torch.tensor([False, True]))


def test_full_scalar_mode_cannot_replace_any_accepted_v2_base_component() -> None:
    contract = SurfaceRegionContractV2()
    _validate_full_scalar_overlay_mode(
        contract=contract,
        field_schema="canonical-v1",
        canonical_radio_source="field_decode",
        context_field=None,
        query_router_mode=False,
    )
    with pytest.raises(ValueError, match="accepted V2 contract"):
        _validate_full_scalar_overlay_mode(
            contract=SurfaceRegionContractV3(),
            field_schema="canonical-v1",
            canonical_radio_source="field_decode",
            context_field=None,
            query_router_mode=False,
        )
    with pytest.raises(ValueError, match="cannot replace"):
        _validate_full_scalar_overlay_mode(
            contract=contract,
            field_schema="factorized-v2",
            canonical_radio_source="field_decode",
            context_field=None,
            query_router_mode=False,
        )
    with pytest.raises(ValueError, match="field decoding"):
        _validate_full_scalar_overlay_mode(
            contract=contract,
            field_schema="canonical-v1",
            canonical_radio_source="mpr_teacher",
            context_field=None,
            query_router_mode=False,
        )
    with pytest.raises(ValueError, match="context field"):
        _validate_full_scalar_overlay_mode(
            contract=contract,
            field_schema="canonical-v1",
            canonical_radio_source="field_decode",
            context_field=torch.nn.Identity(),
            query_router_mode=False,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_full_scalar_overlay_mode(
            contract=contract,
            field_schema="canonical-v1",
            canonical_radio_source="field_decode",
            context_field=None,
            query_router_mode=True,
        )


def test_full_scalar_overlay_maps_local_rows_and_passes_raw_summary_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    raw_summary = torch.arange(36, dtype=torch.float32).reshape(2, 18)

    def aggregate(state, accepted_valid, region_rows, token_mask, anchor_index):
        captured["accepted_valid"] = accepted_valid.clone()
        captured["region_rows"] = region_rows.clone()
        captured["token_mask"] = token_mask.clone()
        captured["anchor_index"] = anchor_index.clone()
        return SimpleNamespace(
            summary=raw_summary,
            use_full_scalar_mask=torch.tensor([True, False]),
            base_fallback_mask=torch.tensor([False, True]),
            abstain_mask=torch.tensor([False, False]),
        )

    normalized_values = torch.full_like(raw_summary, -17.0)

    def normalize(summary, eligible, authority):
        assert torch.equal(summary, raw_summary)
        assert torch.equal(eligible, torch.tensor([True, False]))
        return SimpleNamespace(
            normalized=normalized_values,
            ood_mask=torch.tensor([False, False]),
            use_full_scalar_mask=torch.tensor([True, False]),
            base_fallback_mask=torch.tensor([False, False]),
        )

    class Residual:
        def forward_with_diagnostics(self, base, scalars, *, ood_mask):
            captured["residual_scalars"] = scalars.detach().cpu()
            captured["ood_mask"] = ood_mask.detach().cpu()
            return SimpleNamespace(
                base_descriptor=base,
                semantic_descriptor=base,
                tangent_update=torch.zeros_like(base),
                ood_fallback=ood_mask,
            )

    monkeypatch.setattr(
        semantic_builder, "aggregate_surface_region_full_scalars", aggregate
    )
    monkeypatch.setattr(
        semantic_builder, "apply_full_scalar_normalization", normalize
    )
    base = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = apply_accepted_v2_full_scalar_overlay(
        base,
        residual=Residual(),
        exact_state=object(),
        normalization_authority={},
        accepted_base_valid=torch.tensor([True, False, True, False]),
        accepted_global_rows=torch.tensor([0, 2]),
        local_region_rows=torch.tensor([[0, 1, 0], [1, 0, 0]]),
        token_mask=torch.tensor([[True, True, False], [True, False, False]]),
        anchor_index=torch.tensor([0, 0]),
    )
    assert torch.equal(
        captured["region_rows"],
        torch.tensor([[0, 2, 0], [2, 0, 0]]),
    )
    assert torch.equal(captured["residual_scalars"], raw_summary)
    assert not torch.equal(captured["residual_scalars"], normalized_values)
    assert torch.equal(captured["ood_mask"], torch.tensor([False, True]))
    assert torch.equal(result.semantic_descriptor, base)
    assert torch.equal(result.overlap_candidate_mask, torch.tensor([True, False]))
    assert torch.equal(result.base_only_fallback_mask, torch.tensor([False, True]))


def test_full_scalar_overlay_rejects_abstain_anchor_before_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        semantic_builder,
        "aggregate_surface_region_full_scalars",
        lambda *args, **kwargs: SimpleNamespace(
            summary=torch.zeros(1, 18),
            use_full_scalar_mask=torch.tensor([False]),
            base_fallback_mask=torch.tensor([False]),
            abstain_mask=torch.tensor([True]),
        ),
    )
    with pytest.raises(RuntimeError, match="exact-only/neither"):
        apply_accepted_v2_full_scalar_overlay(
            torch.tensor([[1.0, 0.0]]),
            residual=object(),
            exact_state=object(),
            normalization_authority={},
            accepted_base_valid=torch.tensor([True]),
            accepted_global_rows=torch.tensor([0]),
            local_region_rows=torch.tensor([[0]]),
            token_mask=torch.tensor([[True]]),
            anchor_index=torch.tensor([0]),
        )


def test_full_scalar_checkpoint_authority_must_equal_runtime_accepted_v2() -> None:
    state = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    architecture = {"digest": "a" * 64}
    provenance = {"frozen": True, "scene_disjoint": True}
    contract_sha = "b" * 64
    readout_sha = "c" * 64
    accepted = {
        "checkpoint_sha256": readout_sha,
        "architecture_sha256": architecture["digest"],
        "state_dict_sha256": semantic_builder.surface_region_state_dict_sha256(
            state
        ),
        "provenance_sha256": semantic_builder.canonical_json_sha256(provenance),
        "contract_sha256": contract_sha,
    }
    readout = {
        "architecture": architecture,
        "state_dict": state,
        "provenance": provenance,
    }
    _validate_full_scalar_runtime_base_authority(
        {"accepted_v2_authority": accepted},
        readout_payload=readout,
        readout_checkpoint_sha256=readout_sha,
        contract_sha256=contract_sha,
    )
    changed = dict(accepted)
    changed["state_dict_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="runtime accepted V2"):
        _validate_full_scalar_runtime_base_authority(
            {"accepted_v2_authority": changed},
            readout_payload=readout,
            readout_checkpoint_sha256=readout_sha,
            contract_sha256=contract_sha,
        )


def test_surface_factorized_bundle_reuses_shared_loader_and_binds_state_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = SimpleNamespace(
        sha256="c" * 64,
        xyz=torch.zeros(3, 3),
        valid=torch.tensor([True, False, True]),
    )
    support = SimpleNamespace(cache=cache)
    state = object()
    calls = []

    def load_support(*args, **kwargs):
        calls.append(("support", args, kwargs))
        return support

    def load_state(*args, **kwargs):
        calls.append(("state", args, kwargs))
        assert kwargs["expected_factorized_radio_cache_sha256"] == cache.sha256
        assert torch.equal(kwargs["expected_xyz"], cache.xyz)
        assert torch.equal(kwargs["expected_valid"], cache.valid)
        return state

    monkeypatch.setattr(semantic_builder, "load_factorized_field_support", load_support)
    monkeypatch.setattr(semantic_builder, "load_factorized_primitive_state", load_state)
    loaded_support, loaded_state = load_surface_factorized_state_bundle(
        tmp_path / "field.pt",
        expected_field_checkpoint_sha256="f" * 64,
        expected_factorized_radio_cache_sha256="c" * 64,
        state_path=tmp_path / "state.pt",
        expected_state_sha256="s" * 64,
    )
    assert loaded_support is support
    assert loaded_state is state
    assert [item[0] for item in calls] == ["support", "state"]
from radio_gs.scripts.eval_scannet_canonical_text_query import (
    load_primitive_multiscale_features,
    load_primitive_semantic_cache,
)


def test_two_hop_regions_are_unique_and_physical_scale_clipped() -> None:
    graph = {"xyz": torch.zeros(4, 3),
             "edge_index": torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
             "raw_affinity": torch.ones(6)}
    adjacency = _adjacency(graph, 2)
    xyz = torch.tensor([[0., 0, 0], [.1, 0, 0], [.2, 0, 0], [.5, 0, 0]])
    rows, mask = two_hop_physical_regions(torch.tensor([0]), adjacency, xyz, 0.25)
    kept = rows[0, mask[0]]
    assert set(kept.tolist()) == {0, 1, 2}
    assert len(kept) == len(torch.unique(kept))


def test_completed_surface_regions_preserve_primary_context() -> None:
    primary = torch.tensor([True, True, False, False])
    rows = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    mask = torch.ones_like(rows, dtype=torch.bool)

    kept = preserve_primary_region_tokens(
        rows,
        mask,
        centers=torch.tensor([0, 2]),
        primary_valid=primary,
    )

    assert torch.equal(kept[0], torch.tensor([True, True, False, False]))
    assert torch.equal(kept[1], torch.tensor([True, True, True, False]))


def test_v3_builder_applies_primary_eligibility_before_graph_expansion() -> None:
    xyz = torch.tensor([[0.0, 0, 0], [0.05, 0, 0], [0.10, 0, 0]])
    edge = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    affinity = torch.ones(2)
    graph = PrimitiveSupportGraph(
        edge_index=edge,
        edge_weight=torch.ones(2),
        raw_affinity=affinity,
        local_sigma=torch.ones(3),
        num_nodes=3,
        edge_channels={"appearance": affinity, "boundary": affinity},
    )
    contract = SurfaceRegionContractV3(
        radii_m=(0.2,),
        minimum_tokens=2,
        maximum_tokens=3,
    )
    prepared = contract.prepare_graph(graph, xyz)
    selections = expand_surface_region_v3_batch_at_radius(
        contract,
        graph,
        xyz,
        torch.tensor([0, 1]),
        0.2,
        prepared_graph=prepared,
        primary_local=torch.tensor([True, False, True]),
    )

    primary, fallback = selections
    assert primary.rows.tolist() == [0, 2, 0]
    assert primary.token_mask.tolist() == [True, True, False]
    assert primary.support_fill_mask.tolist() == [False, True, False]
    assert fallback.rows.tolist() == [1, 0, 2]
    assert fallback.core_mask.tolist() == [True, True, True]
    assert not bool(fallback.support_fill_mask.any())


def test_readout_deployment_authority_binds_contract_and_radio() -> None:
    contract = SurfaceRegionContractV3()
    radio_sha = "a" * 64
    payload = {
        "architecture": {"contract_sha256": contract.digest},
        "provenance": {
            "region_contract_sha256": contract.digest,
            "train": {"radio_checkpoint_sha256": radio_sha},
            "validation": {"radio_checkpoint_sha256": radio_sha},
        },
    }
    validate_surface_region_readout_deployment_authority(
        payload,
        contract=contract,
        radio_checkpoint_sha256=radio_sha,
    )
    bad_architecture = {
        **payload,
        "architecture": {"contract_sha256": "b" * 64},
    }
    with pytest.raises(ValueError, match="architecture/provenance contract"):
        validate_surface_region_readout_deployment_authority(
            bad_architecture,
            contract=contract,
            radio_checkpoint_sha256=radio_sha,
        )
    bad_radio = {
        **payload,
        "provenance": {
            **payload["provenance"],
            "validation": {"radio_checkpoint_sha256": "c" * 64},
        },
    }
    with pytest.raises(ValueError, match="training/current RADIO"):
        validate_surface_region_readout_deployment_authority(
            bad_radio,
            contract=contract,
            radio_checkpoint_sha256=radio_sha,
        )


def test_accepted_v2_legacy_radio_authority_reopens_exact_training_caches(
    tmp_path,
) -> None:
    contract = SurfaceRegionContractV2()
    radio_sha = "a" * 64
    readout_sha = "b" * 64
    radio_version = "c-radio_v4-h"
    split_hashes = {"train": "c" * 64, "validation": "d" * 64}
    provenance = {}
    authority = {
        "schema_version": 1,
        "registration": (
            "surface_region_accepted_v2_legacy_radio_training_authority_v1"
        ),
        "readout_checkpoint_sha256": readout_sha,
        "region_contract_sha256": contract.digest,
        "radio_version": radio_version,
        "radio_checkpoint_sha256": radio_sha,
        "cache_validation": {
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
        },
    }
    for split, role in (("train", "train"), ("validation", "validation")):
        cache_path = tmp_path / f"{split}.pt"
        torch.save(
            {
                "metadata": {
                    "split_role": role,
                    "split_file_sha256": split_hashes[split],
                    "region_contract_sha256": contract.digest,
                    "radio_version": radio_version,
                    "radio_checkpoint_sha256": radio_sha,
                    "uses_benchmark_scenes": False,
                    "uses_benchmark_test_vocabulary": False,
                    "labels_opened": False,
                    "text_opened": False,
                }
            },
            cache_path,
        )
        cache_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        provenance[split] = {
            "cache_paths": [str(cache_path)],
            "split_hashes": [split_hashes[split]],
        }
        authority[split] = {
            "split_file_sha256": split_hashes[split],
            "caches": [{"path": str(cache_path), "sha256": cache_sha}],
        }
    payload = {
        "architecture": {
            "name": "surface_region_summary_readout_v2",
            "contract_sha256": contract.digest,
        },
        "provenance": {
            "region_contract_sha256": contract.digest,
            **provenance,
        },
    }

    validate_surface_region_readout_deployment_authority(
        payload,
        contract=contract,
        radio_checkpoint_sha256=radio_sha,
        readout_checkpoint_sha256=readout_sha,
        legacy_radio_authority=authority,
    )
    with pytest.raises(ValueError, match="legacy SurfaceRegion V2 RADIO authority"):
        validate_surface_region_readout_deployment_authority(
            payload,
            contract=contract,
            radio_checkpoint_sha256=radio_sha,
            readout_checkpoint_sha256=readout_sha,
            legacy_radio_authority=None,
        )
    tampered = {**authority, "readout_checkpoint_sha256": "e" * 64}
    with pytest.raises(ValueError, match="legacy SurfaceRegion V2 RADIO authority"):
        validate_surface_region_readout_deployment_authority(
            payload,
            contract=contract,
            radio_checkpoint_sha256=radio_sha,
            readout_checkpoint_sha256=readout_sha,
            legacy_radio_authority=tampered,
        )


def test_raw_query_router_gauge_requires_narrow_preregistration_authority() -> None:
    registration = {
        "registration": "surface_region_accepted_physical_v2_residual_router_v1"
    }
    authority = {
        "registration": (
            "surface_region_accepted_physical_v2_deployment_gauge_parity_addendum_v1"
        ),
        "evidence_without_benchmark_labels": {
            "accepted_positive_cache_sha256": (
                "3366c96839fe392d6cc1f6d55939691787e095491581c63006f74e44862d4cac"
            ),
            "accepted_negative_cache_sha256": (
                "d778361b2ea860cfb07f586ace82cb1558df5129a55ad462b097e4e6c366ba90"
            ),
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
        },
        "gauge_resolution_rule": {"candidates": ["l2_direction", "legacy_raw"]},
        "attribution_constraint": {"generic_training_gauge": "l2_direction"},
    }
    assert validate_query_router_deployment_gauge(
        normalization="legacy_raw",
        experiment_registration=registration,
        gauge_authority=authority,
    ) == "accepted_v2_legacy_raw_mixed_gauge_sentinel"
    with pytest.raises(ValueError, match="exact gauge authority"):
        validate_query_router_deployment_gauge(
            normalization="legacy_raw",
            experiment_registration=registration,
            gauge_authority=None,
        )
    with pytest.raises(ValueError, match="cannot claim raw-gauge"):
        validate_query_router_deployment_gauge(
            normalization="l2_direction",
            experiment_registration=registration,
            gauge_authority=authority,
        )
def test_gated_readout_deployment_authority_binds_architecture_to_provenance() -> None:
    contract = SurfaceRegionContractV3()
    radio_sha = "a" * 64
    payload = {
        "schema_version": 8,
        "architecture": {
            "contract_sha256": contract.digest,
            "base_output_mode": SURFACE_REGION_V3_GATED_RAW_PRIOR,
        },
        "provenance": {
            "region_contract_sha256": contract.digest,
            "train": {"radio_checkpoint_sha256": radio_sha},
            "validation": {"radio_checkpoint_sha256": radio_sha},
            "surface_region_v3": {
                "effective_base_output_mode": SURFACE_REGION_V3_GATED_RAW_PRIOR,
            },
        },
    }
    validate_surface_region_readout_deployment_authority(
        payload,
        contract=contract,
        radio_checkpoint_sha256=radio_sha,
    )
    bad = {
        **payload,
        "provenance": {
            **payload["provenance"],
            "surface_region_v3": {
                "effective_base_output_mode": "fixed_raw_base_with_anchor_mix_v1",
            },
        },
    }
    with pytest.raises(ValueError, match="architecture/provenance mode"):
        validate_surface_region_readout_deployment_authority(
            bad,
            contract=contract,
            radio_checkpoint_sha256=radio_sha,
        )


def test_completed_mpr_primary_partition_is_fail_closed() -> None:
    valid = torch.tensor([True, True, True, False])
    mpr = {
        "reliability": torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        ),
        "metadata": {
            "construction": "dominant_primary_with_query_free_support_completion",
            "primary_valid_count": 2,
        },
    }

    primary = completion_primary_valid(mpr, valid)

    assert torch.equal(primary, torch.tensor([True, False, True, False]))


def test_sparse_v4_cache_expands_losslessly(tmp_path) -> None:
    path = tmp_path / "semantic.pt"
    xyz = torch.randn(5, 3)
    rows = torch.tensor([1, 4])
    valid = torch.zeros(5, dtype=torch.bool); valid[rows] = True
    sparse = torch.randn(2, 1536).half()
    torch.save({
        "xyz": xyz, "features": sparse, "summary_features": sparse,
        "global_rows": rows, "valid": valid,
        "metadata": {
            "schema_version": 4,
            "source": "canonical_radio_surface_region_readout",
            "official_summary_head": True,
            "custom_text_projection": False,
            "query_set_invariant": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }, path)
    loaded_xyz, loaded_valid, features, _ = load_primitive_semantic_cache(path)
    assert torch.equal(loaded_xyz, xyz)
    assert torch.equal(loaded_valid, valid)
    assert torch.equal(features[rows], sparse)
    assert not bool(features[~valid].any())


def test_sparse_v5_multiscale_cache_expands_losslessly(tmp_path) -> None:
    path = tmp_path / "semantic_multiscale.pt"
    xyz = torch.randn(5, 3)
    rows = torch.tensor([1, 4])
    valid = torch.zeros(5, dtype=torch.bool); valid[rows] = True
    sparse = torch.randn(2, 1536).half()
    scales = torch.randn(2, 3, 1536).half()
    torch.save({
        "xyz": xyz, "features": sparse, "summary_features": sparse,
        "features_by_scale": scales, "global_rows": rows, "valid": valid,
        "metadata": {
            "schema_version": 5,
            "source": "canonical_radio_surface_region_readout",
            "official_summary_head": True,
            "custom_text_projection": False,
            "query_set_invariant": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }, path)
    _, loaded_valid, features, _ = load_primitive_semantic_cache(path)
    loaded_scales = load_primitive_multiscale_features(path, valid=loaded_valid)
    assert torch.equal(features[rows], sparse)
    assert loaded_scales is not None
    assert torch.equal(loaded_scales[rows], scales)
    assert not bool(loaded_scales[~valid].any())
