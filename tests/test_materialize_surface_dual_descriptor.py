from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import radio_gs.scripts.materialize_surface_dual_descriptor as module
import radio_gs.scripts.build_frozen_scalar_compositor_replay as replay_builder
import radio_gs.scripts.build_surface_dual_descriptor_primitive_input_cache as input_builder
import radio_gs.scripts.finalize_surface_dual_descriptor_seed0_gate as gate_finalizer
import radio_gs.scripts.train_surface_region_dual_descriptor_residual_pilot as dual_pilot
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2
from radio_gs.models.surface_region_dual_descriptor import SurfaceRegionDualDescriptor
from radio_gs.scripts.select_query_free_scalar_compositor import (
    FIXED_SCALAR_OPERATOR_CONTRACT,
    SCREEN_NAME,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Head(nn.Module):
    def __init__(self, feature_dim: int = 8) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_dim, 1536, bias=False)
        with torch.no_grad():
            torch.manual_seed(811)
            self.projection.weight.copy_(torch.randn(1536, feature_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _false_flags() -> dict[str, bool]:
    return {key: False for key in module._QUERY_FREE_FLAGS}


def _args(paths: dict[str, Path], **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "input_cache": [str(paths["cache"])],
        "base_checkpoint": str(paths["base"]),
        "adapter_checkpoint": str(paths["adapter"]),
        "radio_checkpoint": str(paths["radio"]),
        "scalar_compositor_manifest": str(paths["compositor"]),
        "compositor_weights": str(paths["weights"]),
        "output": str(paths["output"]),
        "scalar_output": str(paths["scalar"]),
        "report": str(paths["report"]),
        "batch_size": 3,
        "device": "cpu",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(809)
    radio = tmp_path / "official-radio.pt"
    radio.write_bytes(b"official-head-lineage")
    base = tmp_path / "base-surface.pt"
    base_model = SurfaceRegionSummaryReadoutV2(feature_dim=8, hidden_dim=256)
    base_payload = {
        "schema_version": 3,
        "architecture": base_model.architecture("c" * 64),
        "state_dict": base_model.state_dict(),
        "provenance": {
            **_false_flags(),
            "frozen": True,
            "scene_disjoint": True,
            "custom_text_projection": False,
        },
    }
    torch.save(base_payload, base)
    head = _Head()
    monkeypatch.setattr(
        module.SigLIP2SummaryHead,
        "from_radio_checkpoint",
        classmethod(lambda cls, path: _Head()),
    )

    rows, tokens = 3, 4
    features = torch.randn(rows, tokens, 8)
    geometry = torch.randn(rows, tokens, 14)
    token_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False], [True, True, True, False]]
    )
    reliability = torch.rand(rows, tokens).clamp_min(0.1)
    anchors = torch.tensor([0, 1, 2])
    with torch.inference_mode():
        official_tokens = base_model.eval()(
            features,
            geometry,
            anchor_index=anchors,
            token_mask=token_mask,
            reliability=reliability,
        )
        official_descriptors = F.normalize(
            head(official_tokens[:, None])[:, 0], dim=-1
        )
    cache = tmp_path / "primitive-input.pt"
    production_geometry = tmp_path / "fixture-production-geometry.pt"
    production_geometry.write_bytes(b"fixture-complete-gaussian-geometry")
    cache_payload = {
        "schema_version": 1,
        "artifact_type": module.INPUT_ARTIFACT_TYPE,
        "primitive_ids": ["scene/primitive-0", "scene/primitive-1", "scene/primitive-2"],
        "radio_features": features,
        "geometry": geometry,
        "token_mask": token_mask,
        "reliability": reliability,
        "anchor_index": anchors,
        "official_summary_tokens": official_tokens,
        "official_descriptors": official_descriptors,
        "metadata": {
            **_false_flags(),
            "complete_primitive_rows": True,
            "target_blind": True,
            "benchmark_targets_or_metrics_used": False,
            "primitive_input_builder_implementation": {
                "path": str(Path(input_builder.__file__).resolve()),
                "sha256": _sha(Path(input_builder.__file__).resolve()),
            },
            "production_primitive_row_authority": {
                "contract": (
                    "complete_single_scene_gaussian_checkpoint_row_order_v1"
                ),
                "scene_id": "scene",
                "geometry_checkpoint": {
                    "path": str(production_geometry),
                    "sha256": _sha(production_geometry),
                },
                "geometry_xyz_sha256": "e" * 64,
                "total_geometry_rows": 3,
                "row_order": "zero_based_geometry_checkpoint_row_order",
                "complete_geometry_rows_present": True,
            },
        },
    }
    torch.save(cache_payload, cache)

    frozen_base, _ = SurfaceRegionSummaryReadoutV2.from_checkpoint(base)
    dual = SurfaceRegionDualDescriptor(frozen_base, _Head())
    with torch.no_grad():
        dual.film.bias[:8].fill_(0.025)
        dual.gate.bias.fill_(0.1)
    adapter = tmp_path / "adapter.pt"
    adapter_state = {
        key: dual.state_dict()[key].clone()
        for key in sorted(module._ADAPTER_STATE_KEYS)
    }
    adapter_payload = {
        "schema_version": 1,
        "artifact_type": module.ADAPTER_ARTIFACT_TYPE,
        "training_complete": True,
        "gate_status": "training_complete_pending_point_render_replay",
        "pilot_advance_gate_passed": False,
        "continuation_authorized": False,
        "seed1_executed": False,
        "additional_seed_or_architecture_authorized": False,
        "base_surface_state_dict": {
            key: value.clone() for key, value in frozen_base.state_dict().items()
        },
        "base_surface_state_dict_sha256": module._state_dict_sha256(
            frozen_base.state_dict(), label="base"
        ),
        "dual_descriptor_architecture": dual.architecture(),
        "adapter_state_dict": adapter_state,
        "adapter_state_dict_sha256": module._state_dict_sha256(
            adapter_state, label="adapter"
        ),
        "best_epoch": 1,
        "provenance": {
            "external_benchmarks_opened": False,
            "benchmark_vocabulary_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_targets_opened": False,
            "metric_continuation": False,
            "fit_split_only_for_optimizer": True,
            "dev_split_used_for_selection_only": True,
            "surface_control": {
                "path": str(base),
                "sha256": _sha(base),
                "seed": 0,
            },
            "radio_checkpoint": {"path": str(radio), "sha256": _sha(radio)},
            "train_caches": [{"path": str(cache), "sha256": _sha(cache)}],
            "validation_caches": [{"path": str(cache), "sha256": _sha(cache)}],
        },
    }
    torch.save(adapter_payload, adapter)
    adapter.with_suffix(".pt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": f"{module.ADAPTER_ARTIFACT_TYPE}_report",
                "output": str(adapter),
                "checkpoint_sha256": _sha(adapter),
                "gate_status": adapter_payload["gate_status"],
                "pilot_advance_gate_passed": False,
                "dual_descriptor_architecture": adapter_payload[
                    "dual_descriptor_architecture"
                ],
                "base_surface_state_dict_sha256": adapter_payload[
                    "base_surface_state_dict_sha256"
                ],
                "adapter_state_dict_sha256": adapter_payload[
                    "adapter_state_dict_sha256"
                ],
                "best_epoch": 1,
            }
        ),
        encoding="utf-8",
    )

    compositor = tmp_path / "compositor.json"
    compositor.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "screen": SCREEN_NAME,
                "selection_uses_benchmark_scenes": False,
                "queries_opened": False,
                "masks_opened": False,
                "labels_opened": False,
                "selected_variant": "alpha_mean",
                "scalar_operator_contract": FIXED_SCALAR_OPERATOR_CONTRACT,
            }
        ),
        encoding="utf-8",
    )
    queries = torch.zeros(2, 1536)
    queries[0, 0] = 1.0
    queries[1, 7] = -1.0
    weights = tmp_path / "replay-weights.pt"
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": module.WEIGHTS_ARTIFACT_TYPE,
            "primitive_ids": cache_payload["primitive_ids"],
            "contribution_weights": torch.tensor(
                [[1.0, 0.0, 0.0], [0.25, 0.25, 0.5]]
            ),
            "query_bank": queries,
            "scalar_compositor_manifest": {
                "path": str(compositor),
                "sha256": _sha(compositor),
            },
            "metadata": {
                **_false_flags(),
                "target_blind": True,
                "benchmark_targets_or_metrics_used": False,
                "frozen_before_materialization": True,
                "query_bank_source": "target_blind_replay_only",
                "selected_variant": "alpha_mean",
                "weights_semantics": (
                    "frozen_selected_normalized_contribution_weights"
                ),
            },
        },
        weights,
    )
    return {
        "radio": radio,
        "base": base,
        "cache": cache,
        "geometry": production_geometry,
        "adapter": adapter,
        "compositor": compositor,
        "weights": weights,
        "output": tmp_path / "descriptors.pt",
        "scalar": tmp_path / "scalars.pt",
        "report": tmp_path / "materialization.json",
    }


def test_materializes_unique_normalized_descriptors_and_real_two_path_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = module.materialize(_args(paths))

    descriptor = torch.load(paths["output"], map_location="cpu")
    scalar = torch.load(paths["scalar"], map_location="cpu")
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    semantic = descriptor["semantic_descriptors"]
    assert semantic.shape == (3, 1536)
    torch.testing.assert_close(
        semantic.norm(dim=-1), torch.ones(3), rtol=0.0, atol=1e-6
    )
    assert len(set(descriptor["primitive_ids"])) == 3
    assert descriptor["official_replay"] == {
        "official_token_bitwise_equal": True,
        "official_descriptor_bitwise_equal": True,
        "official_summary_tokens_sha256": tensor_sha256(
            descriptor["official_summary_tokens"]
        ),
        "official_descriptors_sha256": tensor_sha256(
            descriptor["official_descriptors"]
        ),
    }
    assert scalar["contract"]["render_1536d_then_renormalize"] is False
    assert scalar["contract"]["render_then_query_is_audit_only"] is True
    assert scalar["contract"]["shared_between_paths"] == "primitive_scalar_scores_only"
    assert torch.equal(
        scalar["point_then_render_scores"], scalar["render_then_query_scores"]
    )
    assert report["point_render_replay_max_abs_error"] == 0.0
    assert report["point_render_replay_passed"] is True
    assert report["point_render_replay_evidence"] == {
        "schema_version": 1,
        "artifact_type": "dual_descriptor_point_render_replay_evidence",
        "candidate_adapter_state_dict_sha256": torch.load(
            paths["adapter"], map_location="cpu"
        )["adapter_state_dict_sha256"],
        "independent_materializer_replay": True,
        "frozen_scalar_compositor_replay": False,
        "point_render_replay_max_abs_error": 0.0,
    }
    assert report["formal_point_render_replay_evidence_eligible"] is False
    assert report["replay_weights_schema_version"] == 1
    assert report["query_bank_sha256"] == tensor_sha256(scalar["query_bank"])
    assert report["scalar_cache"] == {
        "path": str(paths["scalar"]),
        "sha256": _sha(paths["scalar"]),
    }
    assert report["descriptor_cache"] == {
        "path": str(paths["output"]),
        "sha256": _sha(paths["output"]),
    }
    assert result == report
    candidate = torch.load(paths["adapter"], map_location="cpu")
    with pytest.raises(ValueError, match="target-blind candidate replay"):
        gate_finalizer._load_materializer_report(
            paths["report"],
            candidate_record={
                "path": str(paths["adapter"]),
                "sha256": _sha(paths["adapter"]),
            },
            adapter_state_sha256=candidate["adapter_state_dict_sha256"],
        )


def test_no_clobber_is_fail_closed_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    paths["output"].write_bytes(b"owned")
    with pytest.raises(FileExistsError, match="overwrite"):
        module.materialize(_args(paths))
    assert paths["output"].read_bytes() == b"owned"
    assert not paths["scalar"].exists()


def test_rejects_non_cpu_and_target_read_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="CPU-only"):
        module.materialize(_args(paths, device="cuda:0"))

    payload = torch.load(paths["cache"], map_location="cpu")
    payload["metadata"]["benchmark_targets_opened"] = True
    torch.save(payload, paths["cache"])
    with pytest.raises(ValueError, match="benchmark_targets_opened=false"):
        module.materialize(_args(paths))


def test_rejects_duplicate_primitives_and_malformed_training_cache_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    adapter = torch.load(paths["adapter"], map_location="cpu")
    adapter["provenance"]["train_caches"][0]["sha256"] = "short"
    torch.save(adapter, paths["adapter"])
    with pytest.raises(ValueError, match="train_caches bindings"):
        module.materialize(_args(paths))

    paths = _fixture(tmp_path / "duplicate", monkeypatch)
    cache = torch.load(paths["cache"], map_location="cpu")
    cache["primitive_ids"][1] = cache["primitive_ids"][0]
    torch.save(cache, paths["cache"])
    with pytest.raises(ValueError, match="unique non-empty primitive IDs"):
        module.materialize(_args(paths))


def test_rejects_official_replay_or_missing_real_compositor_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    cache = torch.load(paths["cache"], map_location="cpu")
    cache["official_summary_tokens"][0, 0] += 1e-5
    torch.save(cache, paths["cache"])
    with pytest.raises(ValueError, match="official token/descriptor replay"):
        module.materialize(_args(paths))

    paths = _fixture(tmp_path / "missing", monkeypatch)
    paths["weights"].unlink()
    with pytest.raises(ValueError, match="compositor replay weights is missing"):
        module.materialize(_args(paths))


def test_real_surface_region_input_builder_replays_frozen_official_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    raw = torch.load(paths["cache"], map_location="cpu")
    contract = SurfaceRegionContractV2(
        maximum_tokens=4,
        minimum_tokens=1,
        token_candidate_limit=4,
    )
    mask = raw["token_mask"]
    base_payload = torch.load(paths["base"], map_location="cpu")
    base_payload["architecture"] = SurfaceRegionSummaryReadoutV2(
        feature_dim=8, hidden_dim=256
    ).architecture(contract.digest)
    torch.save(base_payload, paths["base"])
    raw["radio_features"][~mask] = 0
    raw["geometry"][~mask] = 0
    production_geometry = tmp_path / "production-geometry.pt"
    production_geometry.write_bytes(b"complete-gaussian-geometry")
    source = tmp_path / "real-surface-region.pt"
    source_payload = {
        "radio_features": raw["radio_features"],
        "geometry": raw["geometry"],
        "token_mask": mask,
        "reliability": raw["reliability"],
        "anchor_index": raw["anchor_index"],
        "metadata": {
            "schema_version": 3,
            "complete_scene_regions": True,
            "physical_space_disjoint": True,
            "failed_scenes": {},
            "radio_checkpoint_sha256": _sha(paths["radio"]),
            "region_contract": contract.to_dict(),
            "region_contract_version": contract.version,
            "region_contract_sha256": contract.digest,
            "scene_names": ["scene"],
            "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False,
            "annotations_opened": False,
            "labels_opened": False,
            "instances_opened": False,
            "masks_opened": False,
            "text_opened": False,
            "production_primitive_row_authority": {
                "contract": (
                    "complete_single_scene_gaussian_checkpoint_row_order_v1"
                ),
                "scene_id": "scene",
                "geometry_checkpoint": {
                    "path": str(production_geometry),
                    "sha256": _sha(production_geometry),
                },
                "geometry_xyz_sha256": "d" * 64,
                "total_geometry_rows": 3,
                "row_start": 0,
                "row_stop": 3,
                "row_order": "zero_based_geometry_checkpoint_row_order",
            },
            "region_records": [
                {
                    "scene": "scene",
                    "seed": index,
                    "region_id": f"region-{index}",
                    "anchor_local_index": int(raw["anchor_index"][index]),
                    "tokens": int(mask[index].sum()),
                }
                for index in range(3)
            ],
        },
    }
    torch.save(source_payload, source)
    monkeypatch.setattr(input_builder, "_load_official_head", lambda path: _Head())
    output = tmp_path / "built-primitive-input.pt"

    report = input_builder.build(
        argparse.Namespace(
            source_cache=[str(source)],
            base_checkpoint=str(paths["base"]),
            radio_checkpoint=str(paths["radio"]),
            output=str(output),
            batch_size=2,
            device="cpu",
        )
    )

    payload = torch.load(output, map_location="cpu")
    assert payload["artifact_type"] == module.INPUT_ARTIFACT_TYPE
    assert payload["primitive_ids"] == [
        "scene/primitive-0",
        "scene/primitive-1",
        "scene/primitive-2",
    ]
    assert payload["official_descriptors"].shape == (3, 1536)
    torch.testing.assert_close(
        payload["official_descriptors"].norm(dim=-1),
        torch.ones(3),
        rtol=0.0,
        atol=1e-6,
    )
    assert report["output"] == {"path": str(output), "sha256": _sha(output)}

    incomplete = tmp_path / "random-region-sample-without-row-authority.pt"
    incomplete_payload = dict(source_payload)
    incomplete_payload["metadata"] = dict(source_payload["metadata"])
    incomplete_payload["metadata"].pop("production_primitive_row_authority")
    torch.save(incomplete_payload, incomplete)
    with pytest.raises(ValueError, match="full production primitive-row authority"):
        input_builder._validate_source_cache(
            incomplete,
            radio_sha256=_sha(paths["radio"]),
            base_contract_sha256=contract.digest,
        )


def test_exact_contribution_export_rejects_random_seed_row_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload = torch.load(paths["cache"], map_location="cpu")
    payload["primitive_ids"] = [
        "scene/primitive-2",
        "scene/primitive-0",
        "scene/primitive-1",
    ]
    mismatched = tmp_path / "random-seed-input.pt"
    torch.save(payload, mismatched)
    cameras = tmp_path / "mismatch-cameras.json"
    cameras.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="production row order differs"):
        replay_builder.exact_contribution_cache_payload(
            primitive_input_cache=mismatched,
            view_hits=[
                (
                    "view",
                    {
                        "gaussian_ids": torch.tensor([0]),
                        "pixel_ids": torch.tensor([0]),
                        "weights": torch.tensor([1.0]),
                    },
                )
            ],
            geometry_checkpoint=paths["geometry"],
            camera_manifest=cameras,
            target_blind_provenance={
                **_false_flags(),
                "target_blind": True,
                "benchmark_targets_or_metrics_used": False,
            },
        )


def test_sparse_frozen_compositor_builder_and_materializer_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    # Bind the actual repository implementation, as a production export must.
    implementation = Path(module.__file__).resolve().parents[1] / "rendering" / "contribution_compositor.py"
    producer_runner = Path(module.__file__).resolve().with_name(
        "audit_feature_compositing.py"
    )
    geometry = paths["geometry"]
    cameras = tmp_path / "cameras.json"
    cameras.write_text("{}", encoding="utf-8")
    contribution = tmp_path / "contributions.pt"
    primitive_ids = torch.load(paths["cache"], map_location="cpu")["primitive_ids"]
    contribution_payload = replay_builder.exact_contribution_cache_payload(
        primitive_input_cache=paths["cache"],
        view_hits=[
            (
                "view0",
                {
                    "gaussian_ids": torch.tensor([0, 1, 1, 2]),
                    "pixel_ids": torch.tensor([0, 0, 1, 1]),
                    "weights": torch.tensor([0.25, 0.75, 0.5, 0.5]),
                },
            )
        ],
        geometry_checkpoint=geometry,
        camera_manifest=cameras,
        target_blind_provenance={
            **_false_flags(),
            "target_blind": True,
            "benchmark_targets_or_metrics_used": False,
            "production_capture_runner": (
                "query_free_feature_compositing_exact_hit_export_v1"
            ),
            "production_capture_runner_implementation": {
                "path": str(producer_runner),
                "sha256": _sha(producer_runner),
            },
        },
    )
    assert contribution_payload["metadata"][
        "contribution_compositor_implementation"
    ] == {"path": str(implementation), "sha256": _sha(implementation)}
    torch.save(contribution_payload, contribution)
    query_bank = torch.zeros(2, 1536)
    query_bank[0, 0] = 1
    query_bank[1, 1] = 1
    fit_bank = tmp_path / "unused-fit.pt"
    fit_manifest = tmp_path / "unused-fit.json"
    fit_bank.write_bytes(b"target-blind-fit-bank")
    fit_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        replay_builder,
        "_load_replay_queries",
        lambda path, manifest: (
            query_bank,
            {
                "artifact": {"path": str(fit_bank), "sha256": _sha(fit_bank)},
                "manifest": {
                    "path": str(fit_manifest), "sha256": _sha(fit_manifest)
                },
                "selection": "first_8_rows_of_frozen_order_or_all_if_fewer",
                "selected_row_indices": [0, 1],
            },
        ),
    )
    sparse = tmp_path / "sparse-replay.pt"
    result = replay_builder.build(
        argparse.Namespace(
            output=str(sparse),
            primitive_input_cache=str(paths["cache"]),
            scalar_compositor_manifest=str(paths["compositor"]),
            contribution_cache=[str(contribution)],
            fit_text_bank=str(fit_bank),
            fit_text_bank_manifest=str(fit_manifest),
        )
    )
    payload = torch.load(sparse, map_location="cpu")
    sums = torch.zeros(len(payload["render_row_keys"]))
    sums.index_add_(
        0, payload["render_row_index"], payload["contribution_weights"]
    )
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0.0, atol=1e-6)
    assert result["render_rows"] == 2

    sparse_paths = dict(paths)
    sparse_paths["weights"] = sparse
    materialized = module.materialize(_args(sparse_paths))
    assert materialized["point_render_replay_passed"] is True
    assert materialized["point_render_replay_max_abs_error"] <= 1e-6


def test_materializer_report_independently_finalizes_seed0_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    candidate = torch.load(paths["adapter"], map_location="cpu")
    adapter_sha = candidate["adapter_state_dict_sha256"]
    selected = {
        "epoch": 1,
        "adapter_state_dict_sha256": adapter_sha,
        "fit_constraints_feasible": True,
        "fit_constraint_checks": {
            name: True for name in dual_pilot.FIT_CONSTRAINT_NAMES
        },
        "official_token_bitwise_equal": True,
        "official_descriptor_bitwise_equal": True,
        "dev_control_referenced_selector": {
            "normalized_mean_delta": -0.003,
            "normalized_upper_cvar10_delta": 0.004,
            "worst_scene_mean_delta": 0.009,
            "worst_scene_upper_cvar10_delta": 0.009,
        },
        "validation_unary_relative_deltas": {
            "text_response_smooth_l1": -0.01,
            "text_response_mae": -0.01,
        },
        "validation_descriptor_deltas": {
            "summary_token_cosine": 0.0,
            "mean_descriptor_cosine": 0.0,
            "all_view_descriptor_cosine": 0.0,
        },
        "dev": {"surface_selection_score": 0.9},
    }
    control = dict(selected)
    control["epoch"] = 0
    candidate["best_epoch"] = 1
    candidate["history"] = [control, selected]
    torch.save(candidate, paths["adapter"])
    candidate_report_path = paths["adapter"].with_suffix(".pt.json")
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    candidate_report["checkpoint_sha256"] = _sha(paths["adapter"])
    candidate_report["best_epoch"] = 1
    candidate_report["selected_history_record"] = selected
    candidate_report_path.write_text(json.dumps(candidate_report), encoding="utf-8")

    cameras = tmp_path / "finalizer-cameras.json"
    cameras.write_text("{}", encoding="utf-8")
    contribution = tmp_path / "finalizer-contributions.pt"
    torch.save(
        replay_builder.exact_contribution_cache_payload(
            primitive_input_cache=paths["cache"],
            view_hits=[
                (
                    "finalizer-view",
                    {
                        "gaussian_ids": torch.tensor([0, 1, 1, 2]),
                        "pixel_ids": torch.tensor([0, 0, 1, 1]),
                        "weights": torch.tensor([0.25, 0.75, 0.5, 0.5]),
                    },
                )
            ],
            geometry_checkpoint=paths["geometry"],
            camera_manifest=cameras,
            target_blind_provenance={
                **_false_flags(),
                "target_blind": True,
                "benchmark_targets_or_metrics_used": False,
                "production_capture_runner": (
                    "query_free_feature_compositing_exact_hit_export_v1"
                ),
                "production_capture_runner_implementation": {
                    "path": str(
                        Path(module.__file__).resolve().with_name(
                            "audit_feature_compositing.py"
                        )
                    ),
                    "sha256": _sha(
                        Path(module.__file__).resolve().with_name(
                            "audit_feature_compositing.py"
                        )
                    ),
                },
            },
        ),
        contribution,
    )
    query_bank = torch.zeros(2, 1536)
    query_bank[0, 0] = 1
    query_bank[1, 1] = 1
    fit_bank = tmp_path / "finalizer-fit.pt"
    fit_manifest = tmp_path / "finalizer-fit.json"
    fit_bank.write_bytes(b"target-blind-finalizer-fit")
    fit_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        replay_builder,
        "_load_replay_queries",
        lambda path, manifest: (
            query_bank,
            {
                "artifact": {"path": str(fit_bank), "sha256": _sha(fit_bank)},
                "manifest": {
                    "path": str(fit_manifest), "sha256": _sha(fit_manifest)
                },
                "selection": "first_8_rows_of_frozen_order_or_all_if_fewer",
                "selected_row_indices": [0, 1],
            },
        ),
    )
    sparse_weights = tmp_path / "finalizer-sparse-replay.pt"
    replay_builder.build(
        argparse.Namespace(
            output=str(sparse_weights),
            primitive_input_cache=str(paths["cache"]),
            scalar_compositor_manifest=str(paths["compositor"]),
            contribution_cache=[str(contribution)],
            fit_text_bank=str(fit_bank),
            fit_text_bank_manifest=str(fit_manifest),
        )
    )
    paths["weights"] = sparse_weights

    module.materialize(_args(paths))
    decision_path = tmp_path / "seed0-final.json"
    decision = gate_finalizer.finalize(
        argparse.Namespace(
            adapter_checkpoint=str(paths["adapter"]),
            materializer_report=str(paths["report"]),
            output=str(decision_path),
        )
    )

    assert decision["pilot_advance_gate_passed"] is True
    assert decision["continuation_authorized"] is True
    assert decision["decision"] == "advance"
    assert decision["seed0_single_conjunction_gate"]["passed"] is True
    assert json.loads(decision_path.read_text(encoding="utf-8")) == decision

    tampered_scalar_payload = torch.load(paths["scalar"], map_location="cpu")
    tampered_scalar_payload["primitive_scalar_scores"][0, 0] += 0.1
    tampered_scalar = tmp_path / "tampered-scalars.pt"
    torch.save(tampered_scalar_payload, tampered_scalar)
    tampered_descriptor_payload = torch.load(paths["output"], map_location="cpu")
    tampered_descriptor_payload["scalar_cache"] = {
        "path": str(tampered_scalar), "sha256": _sha(tampered_scalar)
    }
    tampered_descriptor = tmp_path / "tampered-descriptors.pt"
    torch.save(tampered_descriptor_payload, tampered_descriptor)
    recompute_report = json.loads(paths["report"].read_text(encoding="utf-8"))
    recompute_report["scalar_cache"] = {
        "path": str(tampered_scalar), "sha256": _sha(tampered_scalar)
    }
    recompute_report["descriptor_cache"] = {
        "path": str(tampered_descriptor), "sha256": _sha(tampered_descriptor)
    }
    with pytest.raises(ValueError, match="recomputed primitive_scalar_scores differs"):
        gate_finalizer._recompute_materializer_replay(
            recompute_report,
            candidate_record=decision["candidate_checkpoint"],
        )

    tampered = json.loads(paths["report"].read_text(encoding="utf-8"))
    tampered["point_render_replay_evidence"][
        "candidate_adapter_state_dict_sha256"
    ] = "f" * 64
    bad_report = tmp_path / "tampered-materializer.json"
    bad_report.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence binding differs"):
        gate_finalizer.finalize(
            argparse.Namespace(
                adapter_checkpoint=str(paths["adapter"]),
                materializer_report=str(bad_report),
                output=str(tmp_path / "bad-final.json"),
            )
        )
