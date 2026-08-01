from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2
from radio_gs.scripts import finalize_surface_region_query_free_promotion as finalizer


class _TinySummaryHead(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(values).float()[..., :3]


def install_surface_finalizer_test_doubles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install lightweight dimensions/head for fixtures reused by other tests."""

    monkeypatch.setattr(finalizer, "SUMMARY_TOKEN_DIM", 4)
    monkeypatch.setattr(finalizer, "SIGLIP2_DESCRIPTOR_DIM", 3)
    monkeypatch.setattr(
        finalizer,
        "_load_summary_head",
        lambda _path: _TinySummaryHead().eval(),
    )


@pytest.fixture(autouse=True)
def _tiny_audit_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    install_surface_finalizer_test_doubles(monkeypatch)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _candidate_contract(candidate: str) -> SurfaceRegionContractV2:
    return SurfaceRegionContractV2(
        radii_m=(0.25, 0.45, 0.70),
        context_ratio=1.0 if candidate == "core_c1024_geometric" else 1.2,
        neighbors=16,
        maximum_tokens=256,
        minimum_tokens=24,
        path_cost_mode="appearance_boundary_geometric",
        path_affinity_floor=1e-4,
        token_subsampling="core_context_radial_stratified_v1",
        token_candidate_limit=256 if candidate == finalizer.CONTROL else 1024,
        core_token_fraction=0.60,
        reliability_semantics=(
            "uniform_valid"
            if candidate == "context_c1024_uniform"
            else "geometric_mean_observation_agreement"
        ),
    )


def _teacher_contract(contract: SurfaceRegionContractV2) -> SurfaceRegionContractV2:
    return SurfaceRegionContractV2(
        **{
            **contract.__dict__,
            "context_ratio": 1.0,
            "maximum_tokens": 4096,
            "minimum_tokens": 1,
            "token_candidate_limit": 4096,
            "token_subsampling": "nearest_geodesic_then_node_index",
            "core_token_fraction": 1.0,
            "reliability_semantics": "uniform_valid",
        }
    )


def _score(candidate: str, seed: int) -> float:
    del seed
    value = torch.tensor(_candidate_feature(candidate), dtype=torch.float32)
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    return float(F.cosine_similarity(value[None], target[None], dim=-1)[0])


def _candidate_feature(candidate: str) -> list[float]:
    return {
        finalizer.CONTROL: [0.5, 0.5, 0.0, 0.0],
        "context_c1024_geometric": [1.0, 0.0, 0.0, 0.0],
        "context_c1024_uniform": [0.0, 1.0, 0.0, 0.0],
        "core_c1024_geometric": [0.8, 0.2, 0.0, 0.0],
    }[candidate]


def _fixture(tmp_path: Path) -> Path:
    repo_root = Path(finalizer.__file__).resolve().parents[2]
    inputs = tmp_path / "inputs"
    output_root = tmp_path / "surface"
    dataset = inputs / "dataset"
    radio_repo = inputs / "radio"
    dataset.mkdir(parents=True)
    radio_repo.mkdir(parents=True)
    train_split = inputs / "train.txt"
    validation_split = inputs / "validation.txt"
    radio_checkpoint = inputs / "radio.pt"
    exclusion_a = inputs / "pfir_dev.txt"
    exclusion_b = inputs / "pfir_test.txt"
    split_scenes = {
        "train": [
            "scene1000_00",
            "scene1001_00",
            "scene1002_00",
            "scene1003_00",
        ],
        "validation": ["scene2000_00", "scene2001_00"],
    }
    for scene in (*split_scenes["train"], *split_scenes["validation"]):
        (dataset / scene).mkdir()
    train_split.write_text(
        "\n".join(split_scenes["train"]) + "\n",
        encoding="utf-8",
    )
    validation_split.write_text(
        "\n".join(split_scenes["validation"]) + "\n",
        encoding="utf-8",
    )
    radio_checkpoint.write_bytes(b"mock-radio-checkpoint")
    exclusion_a.write_text("scene0000_00\n", encoding="utf-8")
    exclusion_b.write_text("scene0001_00\n", encoding="utf-8")
    explicit_excluded_scenes = sorted(finalizer.FORBIDDEN_EVAL_SCENES)

    candidates = {
        finalizer.CONTROL: {
            "context_ratio": 1.20,
            "token_candidate_limit": 256,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "fresh_official_runtime",
        },
        "context_c1024_geometric": {
            "context_ratio": 1.20,
            "token_candidate_limit": 1024,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "exact_cache_replay",
        },
        "context_c1024_uniform": {
            "context_ratio": 1.20,
            "token_candidate_limit": 1024,
            "reliability": "uniform_valid",
            "teacher_source": "exact_cache_replay",
        },
        "core_c1024_geometric": {
            "context_ratio": 1.00,
            "token_candidate_limit": 1024,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "exact_cache_replay",
        },
    }
    implementation_sources = {
        relative: _sha256(repo_root / relative)
        for relative in finalizer.IMPLEMENTATION_SOURCES
    }
    guard = repo_root / "radio_gs/scripts/run_with_gpu_thermal_guard.sh"
    manifest = {
        "schema_version": 1,
        "screen": finalizer.SCREEN_NAME,
        "dataset_root": str(dataset.resolve()),
        "train_split": str(train_split.resolve()),
        "train_split_sha256": _sha256(train_split),
        "validation_split": str(validation_split.resolve()),
        "validation_split_sha256": _sha256(validation_split),
        "radio_repo": str(radio_repo.resolve()),
        "radio_version": "c-radio_v4-h",
        "radio_checkpoint": str(radio_checkpoint.resolve()),
        "radio_checkpoint_sha256": _sha256(radio_checkpoint),
        "exclusion_files": {
            str(exclusion_a.resolve()): _sha256(exclusion_a),
            str(exclusion_b.resolve()): _sha256(exclusion_b),
        },
        "excluded_scene_names": explicit_excluded_scenes,
        "cache_contract": {
            "train_shards": 4,
            "validation_shards": 2,
            "frames_per_scene": 8,
            "regions_per_scene": 12,
            "region_radii_m": [0.25, 0.45, 0.70],
            "maximum_tokens": 256,
            "teacher_region_candidate_limit": 4096,
            "path_cost_mode": "appearance_boundary_geometric",
            "path_affinity_floor": 1e-4,
            "token_subsampling": "core_context_radial_stratified_v1",
            "core_token_fraction": 0.60,
            "teacher_views": 3,
            "adaptor_batch_size": 512,
            "radio_thermal_pacing_seconds_per_image": 2.0,
            "seed": 0,
        },
        "candidates": candidates,
        "readout_contract": {
            "hidden_dim": 256,
            "epochs": 60,
            "patience": 10,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "token_weight": 0.25,
            "relation_weight": 0.1,
            "reliability_attention_mode": "log_prior",
            "seeds": [0, 1, 2],
        },
        "thermal_safety_contract": {
            "guard": str(guard.resolve()),
            "guard_sha256": _sha256(guard),
            "physical_gpu": 1,
        },
        "selection_contract": {
            "minimum_mean_score_gain": 0.001,
            "minimum_seed_wins": 2,
            "maximum_component_drop": 0.002,
            "uses_benchmark_queries": False,
        },
        "runner_sha256": _sha256(
            repo_root / "radio_gs/scripts/run_surface_region_context_recovery_screen.sh"
        ),
        "implementation_sources": implementation_sources,
    }
    manifest_path = output_root / "run_manifest.json"
    _write_json(manifest_path, manifest)

    pairing_rows = []
    cache_payloads = {}
    exclusion_records = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(manifest["exclusion_files"].items())
    ]
    excluded_spaces = sorted(
        {
            finalizer._physical_space(scene)
            for scene in (
                *explicit_excluded_scenes,
                "scene0000_00",
                "scene0001_00",
            )
        }
    )
    for role, count in (("train", 4), ("validation", 2)):
        for shard in range(count):
            scene = split_scenes[role][shard]
            for candidate in finalizer.EXPECTED_CANDIDATES:
                contract = _candidate_contract(candidate)
                teacher = _teacher_contract(contract)
                protocol = {
                    "schema_version": 1,
                    "support_semantics": finalizer.FIXED_CORE_TEACHER_SEMANTICS,
                    "teacher_region_contract_sha256": teacher.digest,
                    "crop_protocol": finalizer.TEACHER_CROP_PROTOCOL,
                    "frame_selection": "sorted_valid_frames_even_spacing_v1",
                    "frames_per_scene": 8,
                    "minimum_visible_support_tokens": 12,
                    "maximum_teacher_views": 3,
                    "crop_resize_resolution": 384,
                    "radio_version": manifest["radio_version"],
                    "radio_checkpoint_sha256": manifest[
                        "radio_checkpoint_sha256"
                    ],
                    "target_padding": "left_aligned_zero_padding_v1",
                    "teacher_medoid": (
                        "official_descriptor_pairwise_consensus_v1"
                    ),
                }
                protocol_sha = finalizer._canonical_json_sha256(protocol)
                cache_path = (
                    output_root / "caches" / candidate / f"{role}_shard{shard}.pt"
                )
                control_path = (
                    output_root
                    / "caches"
                    / finalizer.CONTROL
                    / f"{role}_shard{shard}.pt"
                )
                replay = (
                    {}
                    if candidate == finalizer.CONTROL
                    else {
                        "path": str(control_path),
                        "sha256": _sha256(control_path),
                    }
                )
                row_count = 12
                maximum_tokens = 256
                radio_features = torch.zeros(
                    row_count,
                    maximum_tokens,
                    4,
                    dtype=torch.float16,
                )
                radio_features[:, :2] = torch.tensor(
                    _candidate_feature(candidate), dtype=torch.float16
                )
                geometry = torch.zeros(
                    row_count,
                    maximum_tokens,
                    14,
                    dtype=torch.float16,
                )
                geometry[:, :2] = 0.25
                token_mask = torch.zeros(
                    row_count,
                    maximum_tokens,
                    dtype=torch.bool,
                )
                token_mask[:, :2] = True
                reliability = torch.zeros(
                    row_count,
                    maximum_tokens,
                    1,
                    dtype=torch.float16,
                )
                reliability[:, :2] = 1.0
                anchor_index = torch.zeros(row_count, dtype=torch.int64)
                summary_tokens = torch.zeros(
                    row_count,
                    3,
                    4,
                    dtype=torch.float16,
                )
                summary_tokens[:, :2, 0] = 1.0
                crop_summaries = torch.zeros(
                    row_count,
                    3,
                    3,
                    dtype=torch.float16,
                )
                crop_summaries[:, :2, 0] = 1.0
                teacher_mask = torch.zeros(row_count, 3, dtype=torch.bool)
                teacher_mask[:, :2] = True
                records = []
                for row_index in range(row_count):
                    radius = (0.25, 0.45, 0.70)[row_index % 3]
                    seed_value = shard * 100 + row_index
                    support_sha = hashlib.sha256(
                        f"{role}:{shard}:{row_index}".encode("utf-8")
                    ).hexdigest()
                    target_sha = finalizer._teacher_target_sha256(
                        summary_tokens[row_index],
                        crop_summaries[row_index],
                        teacher_mask[row_index],
                    )
                    records.append(
                        {
                            "region_id": finalizer._surface_region_id(
                                scene,
                                seed_value,
                                radius,
                                teacher.digest,
                                support_sha,
                            ),
                            "scene": scene,
                            "seed": seed_value,
                            "physical_radius_m": radius,
                            "tokens": 2,
                            "teacher_views": [
                                {
                                    "frame": "000000.jpg",
                                    "crop_box_tlbr": (0, 0, 8, 8),
                                },
                                {
                                    "frame": "000001.jpg",
                                    "crop_box_tlbr": (1, 1, 9, 9),
                                },
                            ],
                            "teacher_medoid": 0,
                            "teacher_region_tokens": 2,
                            "teacher_support_sha256": support_sha,
                            "teacher_region_saturated": False,
                            "teacher_target_source": candidates[candidate][
                                "teacher_source"
                            ],
                            "teacher_target_sha256": target_sha,
                            "anchor_local_index": 0,
                            "core_tokens": 2,
                            "below_nominal_minimum": True,
                        }
                    )
                metadata = {
                    "schema_version": 3,
                    "training_scope": "global_cross_scene_3d_surface_v2",
                    "dataset_id": "ScanNet_frames_25k_query_free",
                    "dataset_root": str(dataset.resolve()),
                    "split_role": role,
                    "split_file": str(
                        (train_split if role == "train" else validation_split).resolve()
                    ),
                    "split_file_sha256": manifest[f"{role}_split_sha256"],
                    "builder_script_sha256": implementation_sources[
                        "radio_gs/scripts/build_scannet_surface_region_cache.py"
                    ],
                    "uses_benchmark_scenes": False,
                    "uses_benchmark_test_vocabulary": False,
                    "annotations_opened": False,
                    "labels_opened": False,
                    "instances_opened": False,
                    "masks_opened": False,
                    "text_opened": False,
                    "physical_space_disjoint": True,
                    "failed_scenes": [],
                    "complete_scene_regions": True,
                    "teacher_regions_saturated": 0,
                    "region_construction": "shared_surface_region_contract_v2",
                    "region_contract": contract.to_dict(),
                    "region_contract_version": contract.version,
                    "region_contract_sha256": contract.digest,
                    "teacher_region_semantics": finalizer.FIXED_CORE_TEACHER_SEMANTICS,
                    "teacher_region_contract": teacher.to_dict(),
                    "teacher_region_contract_sha256": teacher.digest,
                    "teacher_crop_protocol": finalizer.TEACHER_CROP_PROTOCOL,
                    "teacher_target_protocol": protocol,
                    "teacher_target_protocol_sha256": protocol_sha,
                    "teacher_target_source": candidates[candidate]["teacher_source"],
                    "teacher_replay_cache": replay,
                    "teacher_target_schema_version": 1,
                    "regions_per_scene_requested": 12,
                    "teacher_views_requested": 3,
                    "execution_radio_thermal_pacing_seconds_per_image": 2.0,
                    "radio_checkpoint_sha256": manifest["radio_checkpoint_sha256"],
                    "radio_version": manifest["radio_version"],
                    "region_records": records,
                    "scene_names": [scene],
                    "scene_region_counts": {scene: row_count},
                    "forbidden_eval_scenes": sorted(
                        finalizer.FORBIDDEN_EVAL_SCENES
                    ),
                    "excluded_physical_spaces": excluded_spaces,
                    "exclusion_files": exclusion_records,
                }
                payload = {
                    "radio_features": radio_features,
                    "geometry": geometry,
                    "token_mask": token_mask,
                    "reliability": reliability,
                    "official_summary_tokens": summary_tokens,
                    "official_crop_summaries": crop_summaries,
                    "teacher_mask": teacher_mask,
                    "anchor_index": anchor_index,
                    "metadata": metadata,
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, cache_path)
                cache_payloads[(candidate, role, shard)] = payload
                sidecar = {
                    "output": str(cache_path),
                    "regions": row_count,
                    "scenes": 1,
                    "failed_scenes": [],
                    "split_role": role,
                    "split_file_sha256": manifest[f"{role}_split_sha256"],
                    "teacher_target_source": candidates[candidate]["teacher_source"],
                    "teacher_replay_cache": replay,
                }
                _write_json(cache_path.with_suffix(".pt.json"), sidecar)
                pairing_rows.append(
                    {
                        "candidate": candidate,
                        "role": role,
                        "shard": shard,
                        "path": str(cache_path),
                        "sha256": _sha256(cache_path),
                        "regions": row_count,
                        "teacher_target_protocol_sha256": protocol_sha,
                    }
                )

    pairing_path = output_root / "cache_pairing.json"
    pairing = {
        "schema_version": 1,
        "status": "exact_teacher_replay_verified",
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "caches": pairing_rows,
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
    }
    _write_json(pairing_path, pairing)

    readouts = {}
    for candidate in finalizer.EXPECTED_CANDIDATES:
        candidate_values = []
        contract = _candidate_contract(candidate)
        teacher = _teacher_contract(contract)
        target_source = candidates[candidate]["teacher_source"]
        for seed in finalizer.REQUIRED_SEEDS:
            checkpoint_path = output_root / "readouts" / f"{candidate}_seed{seed}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model = SurfaceRegionSummaryReadoutV2(feature_dim=4, hidden_dim=256)
            architecture = model.architecture(contract.digest)

            def merged(role: str, count: int) -> dict:
                scenes = sorted(split_scenes[role])
                return {
                    "scenes": scenes,
                    "split_hashes": [manifest[f"{role}_split_sha256"]],
                    "cache_paths": [
                        str(
                            output_root
                            / "caches"
                            / candidate
                            / f"{role}_shard{shard}.pt"
                        )
                        for shard in range(count)
                    ],
                    "region_contract_sha256": contract.digest,
                    "region_contract": contract.to_dict(),
                    "teacher_region": {
                        "semantics": (
                            "fixed_core_geodesic_support_without_input_context_v1"
                        ),
                        "contract": teacher.to_dict(),
                        "contract_sha256": teacher.digest,
                        "target_source": target_source,
                        "target_protocol_sha256": cache_payloads[
                            (candidate, role, 0)
                        ]["metadata"]["teacher_target_protocol_sha256"],
                    },
                    "radio_checkpoint_sha256": manifest["radio_checkpoint_sha256"],
                    "excluded_physical_spaces": excluded_spaces,
                    "exclusion_files": exclusion_records,
                    "physical_space_disjoint": True,
                }

            train_meta = merged("train", 4)
            validation_meta = merged("validation", 2)
            score = _score(candidate, seed)
            baseline = {
                "summary_token_cosine": score,
                "mean_descriptor_cosine": score,
                "all_view_descriptor_cosine": score,
            }
            validation = {
                "summary_token_cosine": score,
                "mean_descriptor_cosine": score,
                "all_view_descriptor_cosine": score,
            }
            history = [
                {
                    "epoch": 1,
                    "loss": 1.0 - score,
                    "selection_score": score,
                    **validation,
                }
            ]
            training_config = finalizer._expected_training_config(
                output_root, manifest, candidate, seed
            )
            checkpoint = {
                "schema_version": 3,
                "architecture": architecture,
                "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "provenance": {
                    "training_scope": "global_cross_scene_3d_surface_v2",
                    "frozen": True,
                    "uses_benchmark_scenes": False,
                    "uses_benchmark_test_vocabulary": False,
                    "train": train_meta,
                    "validation": validation_meta,
                    "scene_disjoint": True,
                    "official_summary_head": "c-radio_v4 siglip2-g",
                    "custom_text_projection": False,
                    "region_contract_sha256": contract.digest,
                    "region_contract": contract.to_dict(),
                    "canonical_direction_noise_degrees": 0.0,
                    "canonical_noise_calibration": "",
                    "random_seed_contract": {
                        "seed": seed,
                        "model_initialization": True,
                        "data_order": True,
                        "canonical_noise": True,
                    },
                },
                "history": history,
                "best_epoch": 1,
                "best_selection_score": score,
                "untrained_baseline": baseline,
                "untrained_baseline_score": score,
                "training_config": training_config,
            }
            torch.save(checkpoint, checkpoint_path)
            checkpoint_sha = _sha256(checkpoint_path)
            sidecar = {
                "output": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "architecture": architecture,
                "best_epoch": 1,
                "best_selection_score": score,
                "untrained_baseline": baseline,
                "selection_score_delta": 0.0,
                "validation": validation,
                "train_scenes": 4,
                "validation_scenes": 2,
                "scene_overlap": [],
            }
            sidecar_path = checkpoint_path.with_suffix(".pt.json")
            _write_json(sidecar_path, sidecar)
            candidate_values.append(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                    "best_epoch": 1,
                    "best_selection_score": score,
                    "selection_score_delta": 0.0,
                    "validation": validation,
                }
            )
        readouts[candidate] = candidate_values

    rows, selected = finalizer._recompute_candidate_rows(
        readouts, manifest["selection_contract"]
    )
    screen_path = output_root / "query_free_screen.json"
    screen = {
        "schema_version": 1,
        "selection_status": "query_free_candidate_selected_benchmark_gate_still_closed",
        "selected_candidate": selected,
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "cache_pairing_report": str(pairing_path),
        "cache_pairing_report_sha256": _sha256(pairing_path),
        "candidates": rows,
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
        "next_gate": (
            "freeze the selected query-free readout, then evaluate benchmarks "
            "without changing graph, unary, score, or connected-selection rules"
        ),
    }
    _write_json(screen_path, screen)
    complete = output_root / "screen.complete"
    complete.write_text("2026-07-31T12:00:00+08:00\n", encoding="utf-8")
    stamp = max(screen_path.stat().st_mtime_ns, complete.stat().st_mtime_ns)
    os.utime(complete, ns=(stamp, stamp))
    return output_root


def _rewrite_cache_binding(
    output_root: Path,
    *,
    candidate: str,
    role: str,
    shard: int,
    payload: dict,
) -> Path:
    path = output_root / "caches" / candidate / f"{role}_shard{shard}.pt"
    torch.save(payload, path)
    pairing_path = output_root / "cache_pairing.json"
    pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
    row = next(
        value
        for value in pairing["caches"]
        if value["candidate"] == candidate
        and value["role"] == role
        and value["shard"] == shard
    )
    row["sha256"] = _sha256(path)
    row["regions"] = len(payload["metadata"]["region_records"])
    _write_json(pairing_path, pairing)
    sidecar_path = path.with_suffix(".pt.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["regions"] = row["regions"]
    _write_json(sidecar_path, sidecar)
    return path


def test_finalizer_freezes_selected_candidate_as_three_seed_bundle(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    result = finalizer.finalize(output_root)

    assert result["selected_candidate"] == "context_c1024_geometric"
    assert result["required_seeds"] == [0, 1, 2]
    assert result["benchmark_gate_status"] == "closed_not_evaluated"
    assert result["main_result_eligible"] is False
    bundle_path = Path(result["promotion_manifest"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["seed_selection_policy"] == "all_required_seeds_no_single_seed_selection"
    assert [row["seed"] for row in bundle["selected_readouts"]] == [0, 1, 2]
    assert len(bundle["bindings"]["all_compared_readouts"]) == 12
    assert len(bundle["bindings"]["caches"]) == 24
    assert bundle["benchmark_gate"] == {
        "status": "closed_not_evaluated",
        "text_response_gate": "required_before_benchmark_evaluation",
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
        "main_result_eligible": False,
    }
    completion = json.loads(Path(result["completion"]).read_text(encoding="utf-8"))
    assert completion["promotion_manifest_sha256"] == _sha256(bundle_path)

    Path(result["completion"]).unlink()
    resumed = finalizer.finalize(output_root)
    assert resumed["promotion_manifest_sha256"] == result["promotion_manifest_sha256"]
    assert Path(resumed["completion"]).is_file()


def test_finalizer_rejects_cache_changed_after_pairing(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    cache = output_root / "caches" / finalizer.CONTROL / "train_shard0.pt"
    with cache.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="cache SHA256 mismatch"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_checkpoint_sidecar_hash_drift(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    sidecar_path = (
        output_root
        / "readouts"
        / "context_c1024_geometric_seed0.pt.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint_sha256"] = "0" * 64
    _write_json(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="checkpoint SHA mismatch"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_checkpoint_seed_configuration_drift(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    checkpoint_path = output_root / "readouts" / f"{finalizer.CONTROL}_seed0.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["training_config"]["seed"] = 99
    torch.save(checkpoint, checkpoint_path)
    sidecar_path = checkpoint_path.with_suffix(".pt.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint_sha256"] = _sha256(checkpoint_path)
    _write_json(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="training configuration differs"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_checkpoint_architecture_drift(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    checkpoint_path = output_root / "readouts" / f"{finalizer.CONTROL}_seed0.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["architecture"]["name"] = "forged_readout"
    torch.save(checkpoint, checkpoint_path)
    sidecar_path = checkpoint_path.with_suffix(".pt.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint_sha256"] = _sha256(checkpoint_path)
    sidecar["architecture"] = checkpoint["architecture"]
    _write_json(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="architecture differs"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_tampered_query_free_selection(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    screen_path = output_root / "query_free_screen.json"
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    screen["selected_candidate"] = finalizer.CONTROL
    _write_json(screen_path, screen)
    complete = output_root / "screen.complete"
    stamp = max(screen_path.stat().st_mtime_ns, complete.stat().st_mtime_ns)
    os.utime(complete, ns=(stamp, stamp))
    with pytest.raises(ValueError, match="strict recomputation"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_partial_promotion_output(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    completion = output_root / "query_free_promotion.complete.json"
    _write_json(completion, {"status": "forged"})
    with pytest.raises(ValueError, match="completion exists without"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_cache_scene_set_not_derived_from_split(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    candidate = "context_c1024_geometric"
    path = output_root / "caches" / candidate / "train_shard0.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"]["scene_names"] = ["scene9999_00"]
    _rewrite_cache_binding(
        output_root,
        candidate=candidate,
        role="train",
        shard=0,
        payload=payload,
    )
    with pytest.raises(ValueError, match="manifest-bound split shard"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_cache_exclusions_not_recomputed_from_inputs(
    tmp_path: Path,
) -> None:
    output_root = _fixture(tmp_path)
    candidate = "context_c1024_geometric"
    path = output_root / "caches" / candidate / "train_shard0.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"]["excluded_physical_spaces"] = []
    _rewrite_cache_binding(
        output_root,
        candidate=candidate,
        role="train",
        shard=0,
        payload=payload,
    )
    with pytest.raises(ValueError, match="exclusion semantics"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_incomplete_actual_scene_region_count(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    candidate = "context_c1024_geometric"
    path = output_root / "caches" / candidate / "train_shard0.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in finalizer.EVALUATION_KEYS:
        payload[key] = payload[key][:-1]
    records = payload["metadata"]["region_records"][:-1]
    scene = payload["metadata"]["scene_names"][0]
    payload["metadata"]["region_records"] = records
    payload["metadata"]["scene_region_counts"] = {scene: len(records)}
    _rewrite_cache_binding(
        output_root,
        candidate=candidate,
        role="train",
        shard=0,
        payload=payload,
    )
    with pytest.raises(ValueError, match="scene metadata is inconsistent"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_teacher_tensor_digest_tamper(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    candidate = "context_c1024_geometric"
    path = output_root / "caches" / candidate / "train_shard0.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["official_summary_tokens"][0, 0, 0] = 0.5
    _rewrite_cache_binding(
        output_root,
        candidate=candidate,
        role="train",
        shard=0,
        payload=payload,
    )
    with pytest.raises(ValueError, match="teacher target digest"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_non_left_aligned_teacher_mask(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    candidate = "context_c1024_geometric"
    path = output_root / "caches" / candidate / "train_shard0.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["teacher_mask"][0] = torch.tensor([True, False, True])
    _rewrite_cache_binding(
        output_root,
        candidate=candidate,
        role="train",
        shard=0,
        payload=payload,
    )
    with pytest.raises(ValueError, match="teacher mask/padding"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_forged_region_identity(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    candidate = "context_c1024_geometric"
    path = output_root / "caches" / candidate / "train_shard0.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"]["region_records"][0]["region_id"] = "0" * 64
    _rewrite_cache_binding(
        output_root,
        candidate=candidate,
        role="train",
        shard=0,
        payload=payload,
    )
    with pytest.raises(ValueError, match="invalid region ID"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_invalid_teacher_support_hash(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    candidate = "context_c1024_geometric"
    path = output_root / "caches" / candidate / "train_shard0.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"]["region_records"][0]["teacher_support_sha256"] = (
        "not-a-sha256"
    )
    _rewrite_cache_binding(
        output_root,
        candidate=candidate,
        role="train",
        shard=0,
        payload=payload,
    )
    with pytest.raises(ValueError, match="invalid support SHA256"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_teacher_tensor_dtype_drift(tmp_path: Path) -> None:
    output_root = _fixture(tmp_path)
    candidate = "context_c1024_geometric"
    path = output_root / "caches" / candidate / "train_shard0.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["official_summary_tokens"] = payload[
        "official_summary_tokens"
    ].float()
    _rewrite_cache_binding(
        output_root,
        candidate=candidate,
        role="train",
        shard=0,
        payload=payload,
    )
    with pytest.raises(ValueError, match="malformed teacher target tensors"):
        finalizer.finalize(output_root)


def test_finalizer_rejects_checkpoint_metrics_not_reproduced_on_cpu(
    tmp_path: Path,
) -> None:
    output_root = _fixture(tmp_path)
    checkpoint_path = output_root / "readouts" / f"{finalizer.CONTROL}_seed0.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["state_dict"]["residual.3.bias"][1] = 5.0
    torch.save(checkpoint, checkpoint_path)
    sidecar_path = checkpoint_path.with_suffix(".pt.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint_sha256"] = _sha256(checkpoint_path)
    _write_json(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="differs from CPU recomputation"):
        finalizer.finalize(output_root)
