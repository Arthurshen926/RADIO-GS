from argparse import Namespace
import json

import pytest
import torch

from radio_gs.field.factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
    parse_factorized_radio_payload,
)
from radio_gs.scripts import build_gaussian_multiview_teacher_cache as builder
from radio_gs.training.factorized_radio_cache import (
    validate_factorized_radio_training_payload,
)


def _factorized_args(**updates) -> Namespace:
    values = {
        "observation_contract": CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
        "feature_space": "radio",
        "aggregation_mode": "raster_gaussian_top1",
        "raster_view_fusion": "contribution_mean",
        "capability_map_source": "project_raw",
        "capability_storage": "dense",
        "normalize_each_view": False,
        "responsibility_cache": "responsibility.pt",
        "save_responsibility_cache": "",
        "expected_responsibility_cache_sha256": "a" * 64,
        "expected_feature_output_bundle_sha256": "b" * 64,
        "expected_geometry_checkpoint_sha256": "c" * 64,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "query_names": "",
        "max_views": 120,
        "registration_weight_mode": "alpha_depth",
        "robust_mpr": False,
        "raster_reliability_mode": "legacy_valid",
        "max_estimated_cpu_memory_fraction": 0.85,
        "capability_shard_channels": 256,
    }
    values.update(updates)
    return Namespace(**values)


def _aggregate_two_views(scale: float = 1.0):
    state = builder.initialize_factorized_radio_accumulators(3, 2)
    builder.accumulate_factorized_radio_view(
        scale
        * torch.tensor(
            [
                [[3.0, 0.0, 0.0]],
                [[4.0, 0.0, 2.0]],
            ]
        ),
        {
            "gaussian_ids": torch.tensor([0, 1, 2]),
            "pixel_ids": torch.tensor([0, 1, 2]),
            "weights": torch.tensor([2.0, 7.0, 1.0]),
        },
        state,
        observation_chunk_size=1,
    )
    builder.accumulate_factorized_radio_view(
        scale
        * torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[5.0, 0.0, 2.0]],
            ]
        ),
        {
            "gaussian_ids": torch.tensor([0, 1, 2]),
            "pixel_ids": torch.tensor([0, 1, 2]),
            "weights": torch.tensor([1.0, 3.0, 1.0]),
        },
        state,
        observation_chunk_size=2,
    )
    return builder.finalize_factorized_radio_accumulators(state, row_chunk_size=1)


def test_factorized_builder_preserves_amplitude_zero_policy_and_view_counts() -> None:
    core, view_counts = _aggregate_two_views()
    rows = parse_factorized_radio_payload(core)

    assert rows.canonical_feature.dtype == torch.float16
    assert rows.log_amplitude.dtype == torch.float32
    assert rows.reliability.dtype == torch.float32
    assert rows.valid.tolist() == [True, False, True]
    assert view_counts.tolist() == [2, 0, 2]
    # Gaussian 0 sees norms 5 and 5; direction uses responsibility weights
    # two-to-one across the two pixel observations.
    expected_direction = torch.tensor([1.2, 2.6])
    expected_direction /= expected_direction.norm()
    torch.testing.assert_close(
        rows.canonical_feature[0].float(),
        5.0 * expected_direction,
        atol=3e-3,
        rtol=3e-3,
    )
    # Gaussian 1 is observed only at exactly zero amplitude and stays all-zero.
    assert torch.equal(rows.canonical_feature[1], torch.zeros(2, dtype=torch.float16))
    assert torch.equal(rows.reliability[1], torch.zeros(5))
    torch.testing.assert_close(rows.reliability[[0, 2], 3], torch.full((2,), 2.0 / 3.0))
    # Missing visible mass is not disguised as a measured purity.
    assert torch.equal(rows.reliability[:, 4], torch.zeros(3))
    assert "semantic_direction" not in core


def test_factorized_builder_uniform_raw_scaling_restores_the_gauge_only() -> None:
    base, base_counts = _aggregate_two_views()
    scaled, scaled_counts = _aggregate_two_views(scale=7.0)

    torch.testing.assert_close(
        scaled["log_amplitude"][base["valid"]],
        base["log_amplitude"][base["valid"]] + torch.log(torch.tensor(7.0)),
    )
    torch.testing.assert_close(
        scaled["canonical_feature"].float(),
        7.0 * base["canonical_feature"].float(),
        atol=2e-2,
        rtol=2e-3,
    )
    torch.testing.assert_close(scaled["reliability"], base["reliability"])
    assert torch.equal(scaled_counts, base_counts)


def test_factorized_exact_marginal_uses_shared_weights_and_measured_purity() -> None:
    state = builder.initialize_factorized_radio_accumulators(2, 2)
    feature = torch.tensor([[[3.0, 0.0]], [[4.0, 2.0]]])
    builder.accumulate_factorized_radio_view(
        feature,
        {
            "gaussian_ids": torch.tensor([0, 1, 0]),
            "pixel_ids": torch.tensor([0, 0, 1]),
            "base_weights": torch.tensor([0.8, 0.2, 0.6]),
            "marginal_weights": torch.tensor([0.64, 0.04, 0.6]),
        },
        state,
        observation_chunk_size=1,
    )
    core, counts = builder.finalize_factorized_radio_accumulators(
        state,
        row_chunk_size=1,
        visibility_purity_measured=True,
    )

    assert counts.tolist() == [1, 1]
    expected_log = (0.64 * torch.log(torch.tensor(5.0)) + 0.6 * torch.log(torch.tensor(2.0))) / 1.24
    torch.testing.assert_close(core["log_amplitude"][0], expected_log)
    direction = torch.tensor([0.64 * 0.6, 0.64 * 0.8 + 0.6])
    direction /= direction.norm()
    torch.testing.assert_close(
        core["canonical_feature"][0].float(),
        torch.exp(expected_log) * direction,
        atol=3e-3,
        rtol=3e-3,
    )
    torch.testing.assert_close(
        core["reliability"][:, 4], torch.tensor([1.24 / 1.4, 0.2])
    )


def test_factorized_exact_marginal_zero_amplitude_remains_visible() -> None:
    state = builder.initialize_factorized_radio_accumulators(1, 2)
    feature = torch.tensor([[[0.0, 0.0]], [[0.0, 2.0]]])
    builder.accumulate_factorized_radio_view(
        feature,
        {
            "gaussian_ids": torch.tensor([0, 0]),
            "pixel_ids": torch.tensor([0, 1]),
            "base_weights": torch.tensor([0.8, 0.6]),
            "marginal_weights": torch.tensor([0.64, 0.6]),
        },
        state,
        observation_chunk_size=1,
    )
    core, _counts = builder.finalize_factorized_radio_accumulators(
        state,
        row_chunk_size=1,
        visibility_purity_measured=True,
    )

    torch.testing.assert_close(
        core["reliability"][0, 4], torch.tensor(0.6 / 1.4)
    )


def test_factorized_exact_marginal_geometric_only_row_is_semantically_invalid() -> None:
    state = builder.initialize_factorized_radio_accumulators(1, 2)
    builder.accumulate_factorized_radio_view(
        torch.zeros(2, 1, 1),
        {
            "gaussian_ids": torch.tensor([0]),
            "pixel_ids": torch.tensor([0]),
            "base_weights": torch.tensor([0.8]),
            "marginal_weights": torch.tensor([0.64]),
        },
        state,
        observation_chunk_size=1,
    )
    core, counts = builder.finalize_factorized_radio_accumulators(
        state,
        row_chunk_size=1,
        visibility_purity_measured=True,
    )

    assert counts.tolist() == [0]
    assert core["valid"].tolist() == [False]
    assert bool((core["reliability"] == 0).all())


def test_exact_marginal_channel_purity_uses_geometric_denominator() -> None:
    purity = builder._exact_marginal_visibility_purity(
        [
            {
                "gaussian_ids": torch.tensor([0, 0, 1]),
                "base_weights": torch.tensor([0.8, 0.2, 0.5]),
                "marginal_weights": torch.tensor([0.64, 0.04, 0.25]),
            }
        ],
        num_gaussians=2,
        valid=torch.tensor([True, False]),
    )
    torch.testing.assert_close(purity, torch.tensor([0.68, 0.0]))


def test_exact_marginal_semantic_gate_is_shared_pre_adaptor_raw(
    monkeypatch,
) -> None:
    raw = torch.zeros(1, 1280, 1, 2)
    raw[0, 0, 0, 1] = 2.0
    load_arguments = {}

    def load_raw(**kwargs):
        load_arguments.update(kwargs)
        return raw

    monkeypatch.setattr(builder, "_load_bundle_feature_maps", load_raw)
    geometric = [
        {
            "gaussian_ids": torch.tensor([0, 1]),
            "pixel_ids": torch.tensor([0, 1]),
            "base_weights": torch.tensor([0.8, 0.2]),
            "marginal_weights": torch.tensor([0.64, 0.04]),
            "weights": torch.tensor([0.64, 0.04]),
        }
    ]
    semantic, geometric_counts = (
        builder._gate_exact_marginal_assignments_by_raw_amplitude(
            feature_dir="unused",
            selected_frame_indices=[3],
            tensor_records={},
            feature_size=(1, 2),
            responsibility_assignments=geometric,
            num_gaussians=2,
        )
    )

    assert geometric_counts.tolist() == [1, 1]
    assert load_arguments["output_dtype"] == torch.float16
    assert semantic[0]["gaussian_ids"].tolist() == [1]
    purity = builder._exact_marginal_visibility_purity(
        semantic,
        num_gaussians=2,
        valid=torch.tensor([False, True]),
        geometric_assignments=geometric,
    )
    torch.testing.assert_close(purity, torch.tensor([0.0, 0.2]))


def test_channel_shard_recovers_commit_before_progress(tmp_path, monkeypatch) -> None:
    output = tmp_path / "raw.pt"
    shard = tmp_path / "raw.pt.channels_00000_00002.f16"
    shard.write_bytes(torch.tensor([[1.0, 2.0]]).half().numpy().tobytes())
    monkeypatch.setattr(
        builder,
        "_load_bundle_feature_maps",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("recovered shard must not reopen features")
        ),
    )
    records, valid, counts, _reliability = (
        builder._stream_channel_sharded_contribution_mean(
            output=output,
            feature_space="radio",
            feature_dir=tmp_path,
            feature_tensor_records={},
            selected_frame_indices=[3],
            feature_size=(1, 1),
            responsibility_assignments=[
                {
                    "gaussian_ids": torch.tensor([0]),
                    "pixel_ids": torch.tensor([0]),
                    "weights": torch.tensor([1.0]),
                }
            ],
            num_gaussians=1,
            output_dim=2,
            shard_channels=2,
            inner_channel_chunk_size=2,
            point_chunk_size=1,
            num_views=1,
            normalize_each_view=False,
            reliability_mode="legacy_valid",
            adaptor=None,
            device=torch.device("cpu"),
            resume_contract={"fixture": "orphan-recovery-v1"},
        )
    )
    assert records[0]["relative_path"] == shard.name
    assert valid.tolist() == [True]
    assert counts.tolist() == [1]
    progress = json.loads(
        output.with_suffix(".pt.partial.json").read_text(encoding="utf-8")
    )
    assert progress["shards"] == records


def test_factorized_builder_streams_exactly_one_raw_view_at_a_time(
    monkeypatch,
) -> None:
    calls = []

    def fake_load(**kwargs):
        frame_indices = kwargs["selected_frame_indices"]
        calls.append(list(frame_indices))
        assert len(frame_indices) == 1
        assert kwargs["normalize"] is False
        assert kwargs["output_dtype"] == torch.float32
        feature = torch.zeros(1, 1280, 1, 1)
        feature[0, 0, 0, 0] = 2.0 if frame_indices[0] == 4 else 8.0
        return feature

    monkeypatch.setattr(builder, "_load_bundle_feature_maps", fake_load)
    assignment = {
        "gaussian_ids": torch.tensor([0]),
        "pixel_ids": torch.tensor([0]),
        "weights": torch.tensor([1.0]),
    }

    core, counts = builder._stream_factorized_radio_from_bundle(
        feature_dir="unused",
        selected_frame_indices=[4, 9],
        tensor_records={},
        feature_size=(1, 1),
        responsibility_assignments=[assignment, assignment],
        num_gaussians=1,
        observation_chunk_size=1,
        row_chunk_size=1,
    )

    assert calls == [[4], [9]]
    assert counts.tolist() == [2]
    torch.testing.assert_close(core["log_amplitude"], torch.log(torch.tensor([4.0])))
    torch.testing.assert_close(
        core["canonical_feature"].float()[0, 0], torch.tensor(4.0)
    )


def test_factorized_builder_memory_preflight_accounts_for_dense_field() -> None:
    estimate = builder.estimate_factorized_radio_cpu_bytes(
        num_gaussians=636_148,
        feature_dim=1280,
        feature_height=48,
        feature_width=63,
        observation_chunk_size=4096,
        row_chunk_size=4096,
    )

    assert estimate["weighted_unit_sum_float32"] == 636_148 * 1280 * 4
    assert estimate["canonical_feature_float16"] == 636_148 * 1280 * 2
    assert estimate["estimated_peak_bytes"] == sum(
        value for key, value in estimate.items() if key != "estimated_peak_bytes"
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"feature_space": "sam3"}, "frozen options"),
        ({"aggregation_mode": "center"}, "frozen options"),
        ({"raster_view_fusion": "view_mean"}, "frozen options"),
        ({"normalize_each_view": True}, "raw RADIO amplitudes"),
        ({"max_views": 32}, "--max-views 120"),
        ({"registration_weight_mode": "uniform"}, "frozen options"),
        ({"robust_mpr": True}, "--no-robust-mpr"),
        ({"responsibility_cache": ""}, "requires --responsibility-cache"),
        ({"save_responsibility_cache": "new.pt"}, "forbids live"),
        ({"expected_responsibility_cache_sha256": ""}, "expected SHA-256"),
        ({"query_names": "object"}, "query-free"),
    ],
)
def test_factorized_builder_cli_contract_fails_closed(updates, message) -> None:
    with pytest.raises(ValueError, match=message):
        builder.validate_raster_reliability_policy(_factorized_args(**updates))


def test_factorized_builder_cli_parser_roundtrip(monkeypatch, capsys) -> None:
    captured = {}

    def fake_build(args):
        captured.update(vars(args))
        return {"schema": "test"}

    monkeypatch.setattr(builder, "build_cache", fake_build)
    builder.main(
        [
            "--config",
            "scene.yaml",
            "--checkpoint",
            "geometry.pth",
            "--output",
            "factorized.pt",
            "--observation-contract",
            CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
            "--aggregation-mode",
            "raster_gaussian_top1",
            "--max-views",
            "120",
            "--no-robust-mpr",
            "--responsibility-cache",
            "responsibility.pt",
            "--expected-responsibility-cache-sha256",
            "a" * 64,
            "--expected-feature-output-bundle-sha256",
            "b" * 64,
            "--expected-geometry-checkpoint-sha256",
            "c" * 64,
        ]
    )

    assert captured["observation_contract"] == CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME
    assert captured["feature_space"] == "radio"
    assert captured["aggregation_mode"] == "raster_gaussian_top1"
    assert captured["raster_view_fusion"] == "contribution_mean"
    assert captured["max_views"] == 120
    assert captured["registration_weight_mode"] == "alpha_depth"
    assert captured["robust_mpr"] is False
    json_output = capsys.readouterr().out
    assert json_output
    assert '"schema": "test"' in json_output


def test_cache_output_and_report_are_preflight_no_clobber(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "factorized.pt"
    output.write_bytes(b"existing")
    monkeypatch.setattr(
        builder,
        "_build_cache",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("preflight must precede scene loading")
        ),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        builder.build_cache(Namespace(output=str(output)))

    output.unlink()
    output.with_suffix(".pt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        builder.build_cache(Namespace(output=str(output)))


def test_factorized_exact_marginal_cli_accepts_create_or_bound_reuse() -> None:
    create = _factorized_args(
        aggregation_mode="raster_marginal_responsibility",
        registration_weight_mode="alpha_depth",
        alpha_threshold=0.0,
        responsibility_cache="",
        save_responsibility_cache="authority.json",
        expected_responsibility_cache_sha256="",
    )
    builder.validate_raster_reliability_policy(create)
    assert create.registration_weight_mode == (
        "exact_front_to_back_marginal_responsibility"
    )

    reuse = _factorized_args(
        aggregation_mode="raster_marginal_responsibility",
        registration_weight_mode="alpha_depth",
        alpha_threshold=0.0,
        responsibility_cache="authority.json",
        save_responsibility_cache="",
    )
    builder.validate_raster_reliability_policy(reuse)
    assert reuse.registration_weight_mode == (
        "exact_front_to_back_marginal_responsibility"
    )


def test_legacy_policy_arguments_remain_unchanged() -> None:
    args = Namespace(
        observation_contract="legacy",
        aggregation_mode="center",
        raster_reliability_mode="legacy_valid",
        normalize_each_view=False,
        max_estimated_cpu_memory_fraction=0.85,
        capability_shard_channels=256,
    )
    before = vars(args).copy()

    builder.validate_raster_reliability_policy(args)

    assert vars(args) == before


def test_factorized_builder_envelope_binds_unknown_purity_to_sidecar() -> None:
    state = builder.initialize_factorized_radio_accumulators(1, 1280)
    feature = torch.zeros(1280, 1, 1)
    feature[0, 0, 0] = 2.0
    builder.accumulate_factorized_radio_view(
        feature,
        {
            "gaussian_ids": torch.tensor([0]),
            "pixel_ids": torch.tensor([0]),
            "weights": torch.tensor([0.25]),
        },
        state,
        observation_chunk_size=1,
    )
    core, counts = builder.finalize_factorized_radio_accumulators(
        state, row_chunk_size=1
    )
    sidecar_sha = "d" * 64
    xyz = torch.zeros(1, 3)
    payload = {
        "schema": builder.CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA,
        "schema_version": 1,
        "xyz": xyz,
        "geometry_fingerprint": {
            "num_gaussians": 1,
            "xyz_sha256": builder._sha256_tensor_rows(xyz),
        },
        "factorized_radio": core,
        "view_counts": counts,
        "metadata": {
            "builder_contract": builder.canonical_factorized_radio_builder_contract(),
            "builder_contract_sha256": builder.factorized_radio_builder_contract_sha256(),
            "construction": CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
            "feature_space": "radio",
            "input_feature_space": "radio_raw_full",
            "feature_dim": 1280,
            "geometry_checkpoint_sha256": "a" * 64,
            "feature_frame_manifest_sha256": "b" * 64,
            "feature_output_bundle_sha256": "c" * 64,
            "selected_dataset_indices": [0],
            "selected_frame_indices": [0],
            "num_declared_views": 1,
            "max_views_authority": 120,
            "aggregation_mode": "raster_gaussian_top1",
            "raster_view_fusion": "contribution_mean",
            "registration_weight_mode": "alpha_depth",
            "semantic_direction_storage": "derived_from_canonical_feature_not_persisted",
            "canonical_feature_dtype": "float16",
            "log_amplitude_dtype": "float32",
            "reliability_dtype": "float32",
            "robust_mpr": False,
            "benchmark_masks_opened": False,
            "benchmark_images_opened": False,
            "text_queries_opened": False,
            "query_independent": True,
            "registration_responsibility_cache_sha256": sidecar_sha,
            "visibility_purity_authority": {
                **builder.FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY,
                "registration_responsibility_cache_sha256": sidecar_sha,
            },
        },
    }

    builder.validate_factorized_radio_builder_payload(payload)
    payload["geometry_fingerprint"]["num_gaussians"] = 2
    with pytest.raises(ValueError, match="geometry fingerprint"):
        builder.validate_factorized_radio_builder_payload(payload)
    payload["geometry_fingerprint"]["num_gaussians"] = 1
    original_dispersion = core["reliability"][0, 1].item()
    core["reliability"][0, 1] = 0.5
    with pytest.raises(ValueError, match="resultant/dispersion"):
        builder.validate_factorized_radio_builder_payload(payload)
    core["reliability"][0, 1] = original_dispersion
    payload["metadata"]["visibility_purity_authority"]["measurement_available"] = True
    with pytest.raises(ValueError, match="purity authority"):
        builder.validate_factorized_radio_builder_payload(payload)


def test_factorized_builder_envelope_rejects_contract_tamper() -> None:
    state = builder.initialize_factorized_radio_accumulators(1, 1280)
    feature = torch.zeros(1280, 1, 1)
    feature[0, 0, 0] = 2.0
    builder.accumulate_factorized_radio_view(
        feature,
        {
            "gaussian_ids": torch.tensor([0]),
            "pixel_ids": torch.tensor([0]),
            "weights": torch.tensor([1.0]),
        },
        state,
        observation_chunk_size=1,
    )
    core, counts = builder.finalize_factorized_radio_accumulators(
        state, row_chunk_size=1
    )
    sidecar_sha = "d" * 64
    metadata = {
        "builder_contract": builder.canonical_factorized_radio_builder_contract(),
        "builder_contract_sha256": builder.factorized_radio_builder_contract_sha256(),
        "construction": CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
        "feature_space": "radio",
        "input_feature_space": "radio_raw_full",
        "feature_dim": 1280,
        "geometry_checkpoint_sha256": "a" * 64,
        "feature_frame_manifest_sha256": "b" * 64,
        "feature_output_bundle_sha256": "c" * 64,
        "selected_dataset_indices": [0],
        "selected_frame_indices": [0],
        "num_declared_views": 1,
        "max_views_authority": 120,
        "aggregation_mode": "raster_gaussian_top1",
        "raster_view_fusion": "contribution_mean",
        "registration_weight_mode": "alpha_depth",
        "semantic_direction_storage": "derived_from_canonical_feature_not_persisted",
        "canonical_feature_dtype": "float16",
        "log_amplitude_dtype": "float32",
        "reliability_dtype": "float32",
        "robust_mpr": False,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "query_independent": True,
        "registration_responsibility_cache_sha256": sidecar_sha,
        "visibility_purity_authority": {
            **builder.FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY,
            "registration_responsibility_cache_sha256": sidecar_sha,
        },
    }
    payload = {
        "schema": builder.CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA,
        "schema_version": 1,
        "xyz": torch.zeros(1, 3),
        "geometry_fingerprint": {"num_gaussians": 1, "xyz_sha256": "e" * 64},
        "factorized_radio": core,
        "view_counts": counts,
        "metadata": metadata,
    }
    payload["geometry_fingerprint"]["xyz_sha256"] = builder._sha256_tensor_rows(
        payload["xyz"]
    )
    builder.validate_factorized_radio_builder_payload(payload)
    payload["metadata"]["builder_contract"]["maximum_views"] = 32
    with pytest.raises(ValueError, match="builder contract"):
        builder.validate_factorized_radio_builder_payload(payload)


def test_factorized_builder_v2_envelope_is_training_compatible() -> None:
    state = builder.initialize_factorized_radio_accumulators(2, 1280)
    feature = torch.zeros(1280, 1, 1)
    feature[0, 0, 0] = 2.0
    builder.accumulate_factorized_radio_view(
        feature,
        {
            "gaussian_ids": torch.tensor([0, 1]),
            "pixel_ids": torch.tensor([0, 0]),
            "base_weights": torch.tensor([0.8, 0.2]),
            "marginal_weights": torch.tensor([0.64, 0.04]),
        },
        state,
        observation_chunk_size=1,
    )
    core, counts = builder.finalize_factorized_radio_accumulators(
        state,
        row_chunk_size=1,
        visibility_purity_measured=True,
    )
    authority_sha = "d" * 64
    metadata = {
        "builder_contract": builder.canonical_factorized_radio_builder_contract_v2(),
        "builder_contract_sha256": builder.factorized_radio_builder_contract_v2_sha256(),
        "construction": CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
        "feature_space": "radio",
        "input_feature_space": "radio_raw_full",
        "feature_dim": 1280,
        "geometry_checkpoint_sha256": "a" * 64,
        "feature_frame_manifest_sha256": "b" * 64,
        "feature_output_bundle_sha256": "c" * 64,
        "selected_dataset_indices": [0],
        "selected_frame_indices": [0],
        "num_declared_views": 1,
        "max_views_authority": 120,
        "aggregation_mode": "raster_marginal_responsibility",
        "raster_view_fusion": "contribution_mean",
        "registration_weight_mode": "exact_front_to_back_marginal_responsibility",
        "semantic_direction_storage": "derived_from_canonical_feature_not_persisted",
        "canonical_feature_dtype": "float16",
        "log_amplitude_dtype": "float32",
        "reliability_dtype": "float32",
        "robust_mpr": False,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "query_independent": True,
        "registration_responsibility_cache_sha256": authority_sha,
        "valid_semantics": (
            "positive_raw_radio_amplitude_responsibility_mass_and_"
            "nonzero_direction_resultant"
        ),
        "semantic_assignment_gate": (
            "pre_adaptor_raw_radio_l2_norm_strictly_positive"
        ),
        "view_count_semantics": (
            "views_with_pre_adaptor_raw_radio_l2_norm_strictly_positive"
        ),
        "geometric_visibility_semantics": (
            "independent_exact_base_weight_authority_includes_zero_amplitude_hits"
        ),
        "geometric_view_counts_sha256": builder._sha256_int64_vector(
            torch.tensor([1, 1], dtype=torch.long)
        ),
        "geometric_visible_gaussian_count": 2,
        "semantic_valid_gaussian_count": 2,
        "geometric_visible_semantic_invalid_gaussian_count": 0,
        "invalid_row_purity_policy": (
            "core_v1_requires_zero_for_semantically_invalid_rows"
        ),
        "visibility_purity_authority": {
            **builder.FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
            "registration_responsibility_cache_sha256": authority_sha,
        },
    }
    xyz = torch.zeros(2, 3)
    payload = {
        "schema": builder.CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2,
        "schema_version": 2,
        "xyz": xyz,
        "geometry_fingerprint": {
            "num_gaussians": 2,
            "xyz_sha256": builder._sha256_tensor_rows(xyz),
        },
        "factorized_radio": core,
        "view_counts": counts,
        "metadata": metadata,
    }

    builder.validate_factorized_radio_builder_payload(payload)
    validate_factorized_radio_training_payload(
        payload, expected_feature_output_bundle_sha256="c" * 64
    )
    core["reliability"][0, 4] = 1.1
    with pytest.raises(ValueError, match="purity"):
        builder.validate_factorized_radio_builder_payload(payload)
