import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import radio_gs.evaluation.text_response_fidelity as fidelity_module
import radio_gs.scripts.eval_text_response_fidelity_gate as fidelity_gate_module

from radio_gs.evaluation.text_response_fidelity import (
    REPORT_ARTIFACT_TYPE,
    REPORT_SCHEMA_VERSION,
    aggregate_paired_seed_gate,
    evaluate_response_fidelity,
    row_identity_sha256,
    tensor_sha256,
)
from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_independent_normalized_cosine_response_smooth_l1_loss,
)
from radio_gs.scripts.build_target_blind_siglip2_embedding_artifact import (
    MODEL_REVISION,
    OUTPUT_DIMENSION,
    build_embedding_artifact,
)
from radio_gs.scripts.eval_text_response_fidelity_gate import (
    _split_sha256,
    evaluate_artifacts,
    load_descriptor_pair,
    load_text_embedding_bank,
)
from radio_gs.scripts.materialize_surface_text_response_descriptors import (
    _legacy_region_id,
    _teacher_descriptor,
)


def _descriptors():
    teacher = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.2],
                [0.8, 0.6, 0.1],
                [0.1, 1.0, 0.3],
                [-0.5, 0.8, 0.4],
                [0.9, -0.2, 0.1],
                [0.4, 0.8, 0.2],
                [-0.2, 0.9, 0.5],
                [-0.7, 0.3, 0.6],
            ]
        ),
        dim=-1,
    )
    text = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.3, 0.1, 1.0],
            ]
        ),
        dim=-1,
    )
    scenes = ["scene_a"] * 4 + ["scene_b"] * 4
    regions = [f"region_{index}" for index in range(8)]
    queries = ["q0", "q1", "q2"]
    return teacher, text, scenes, regions, queries


def test_matching_descriptors_have_exact_response_fidelity():
    teacher, text, scenes, regions, queries = _descriptors()
    result = evaluate_response_fidelity(
        teacher,
        teacher,
        text,
        scene_ids=scenes,
        region_ids=regions,
        query_ids=queries,
    )

    aggregate = result["aggregate"]
    assert aggregate["smooth_l1"] == 0.0
    assert aggregate["mae"] == 0.0
    assert aggregate["response_profile_cosine_mean"] == pytest.approx(1.0)
    assert aggregate["response_profile_cosine_p05"] == pytest.approx(1.0)
    assert aggregate["ranking_spearman_mean"] == pytest.approx(1.0)
    assert aggregate["ranking_spearman_p05"] == pytest.approx(1.0)
    assert aggregate["top_decile_overlap_mean"] == 1.0
    assert aggregate["top_decile_overlap_p05"] == 1.0
    assert result["counts"]["ranking_valid_scene_queries"] == 6


def test_response_metrics_detect_scene_local_ranking_and_top_decile_damage():
    teacher, text, scenes, regions, queries = _descriptors()
    student = teacher.clone()
    student[:4] = student[torch.tensor([3, 2, 1, 0])]
    result = evaluate_response_fidelity(
        student,
        teacher,
        text,
        scene_ids=scenes,
        region_ids=regions,
        query_ids=queries,
    )

    aggregate = result["aggregate"]
    assert aggregate["smooth_l1"] > 0.0
    assert aggregate["mae"] > 0.0
    assert aggregate["ranking_spearman_mean"] < 1.0
    assert aggregate["top_decile_overlap_mean"] < 1.0
    assert aggregate["ranking_spearman_p05"] < aggregate["ranking_spearman_mean"]


def test_reported_smooth_l1_exactly_matches_training_loss_contract():
    teacher, text, scenes, regions, queries = _descriptors()
    student = teacher.roll(1, dims=0)
    result = evaluate_response_fidelity(
        student,
        teacher,
        text,
        scene_ids=scenes,
        region_ids=regions,
        query_ids=queries,
    )
    training_loss = compute_independent_normalized_cosine_response_smooth_l1_loss(
        student,
        teacher,
        text,
    )
    assert result["aggregate"]["smooth_l1"] == pytest.approx(training_loss.item())


def test_zero_norm_text_embedding_is_rejected():
    teacher = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), dim=-1)
    text = torch.tensor([[0.0, 0.0]])
    with pytest.raises(ValueError, match="zero-norm"):
        evaluate_response_fidelity(
            teacher,
            teacher,
            text,
            scene_ids=["s"] * 3,
            region_ids=["a", "b", "c"],
        )


def test_constant_teacher_query_is_excluded_only_from_ranking():
    teacher = F.normalize(
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        dim=-1,
    )
    result = evaluate_response_fidelity(
        teacher,
        teacher,
        torch.tensor([[0.0, 0.0, 1.0]]),
        scene_ids=["s"] * 3,
        region_ids=["a", "b", "c"],
    )
    assert result["counts"]["ranking_valid_scene_queries"] == 0
    assert result["unit_metrics"][0]["ranking_spearman"] is None
    assert result["aggregate"]["top_decile_overlap_mean"] == 1.0


def _report(method, seed, student, teacher, text, scenes, regions, queries):
    metrics = evaluate_response_fidelity(
        student,
        teacher,
        text,
        scene_ids=scenes,
        region_ids=regions,
        query_ids=queries,
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "method_id": method,
        "seed": seed,
        "split_role": "query_free_validation",
        "query_split": "dev",
        "selection_contract": {
            "benchmark_vocabulary_opened": False,
            "uses_benchmark_vocabulary_for_construction": False,
            "queries": "target_blind_imagenet1k_primary_text_bank_v1",
            "query_axis": "heldout_generic_only",
            "device": "cpu",
        },
        "descriptor_artifact": {
            "path": f"/tmp/{method}_seed{seed}.descriptors.pt",
            "sha256": "0" * 64,
        },
        "descriptor_rows_sha256": row_identity_sha256(scenes, regions),
        "teacher_descriptors_sha256": tensor_sha256(teacher),
        "query_bank": {
            "path": "/tmp/dev_bank.pt",
            "sha256": "3" * 64,
            "manifest_path": "/tmp/dev_bank.manifest.json",
            "manifest_sha256": "4" * 64,
            "vocabulary_sha256": "5" * 64,
            "query_split": "dev",
            "selected_queries": len(queries),
            "selected_records_sha256": "1" * 64,
            "ordered_records_sha256": "6" * 64,
            "embedding_tensor_sha256": "2" * 64,
            "embedding_semantic_sha256": "7" * 64,
            "text_encoder": {"unit_test": True},
        },
        "metrics": metrics,
    }


def _unit_gate_kwargs(scenes):
    return {
        "phase": "dev",
        "_test_expected_scene_ids": tuple(sorted(set(scenes))),
        "_test_report_recomputer": lambda report, phase: report,
    }


def test_selection_contract_family_mapping_is_exact_and_fail_closed():
    legacy_family = fidelity_module.IMAGENET1K_PRIMARY_BANK_FAMILY
    holdout_family = fidelity_module.IMAGENET12K_HOLDOUT_BANK_FAMILY

    legacy = fidelity_module.selection_contract_for_bank_family(legacy_family)
    holdout = fidelity_module.selection_contract_for_bank_family(holdout_family)
    assert legacy == {
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "queries": "target_blind_imagenet1k_primary_text_bank_v1",
        "query_axis": "heldout_generic_only",
        "device": "cpu",
    }
    assert holdout == {
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "queries": "target_blind_imagenet12k_minus_imagenet1k_holdout_v1",
        "query_axis": "heldout_generic_only",
        "device": "cpu",
    }

    # Callers receive a copy and cannot mutate the frozen module authority.
    holdout["device"] = "cuda"
    assert fidelity_module.selection_contract_for_bank_family(holdout_family)[
        "device"
    ] == "cpu"
    with pytest.raises(ValueError, match="unknown text query-bank family"):
        fidelity_module.selection_contract_for_bank_family("unregistered_family")


def test_formal_text_bank_pair_classification_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifact = tmp_path / "dev.pt"
    manifest = tmp_path / "dev.manifest.json"
    artifact.write_bytes(b"registered-dev-artifact")
    manifest.write_bytes(b"registered-dev-sidecar")
    registered = dict(fidelity_gate_module.FORMAL_HISTORICAL_TEXT_BANKS)
    registered["dev"] = {
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(
        fidelity_gate_module, "FORMAL_HISTORICAL_TEXT_BANKS", registered
    )

    assert fidelity_gate_module.classify_formal_text_bank_pair(
        artifact, manifest, "dev"
    ) == fidelity_module.IMAGENET1K_PRIMARY_BANK_FAMILY

    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="unregistered or changed"):
        fidelity_gate_module.classify_formal_text_bank_pair(
            artifact, manifest, "dev"
        )


def test_report_validation_accepts_only_exact_registered_family_contracts():
    teacher, text, scenes, regions, queries = _descriptors()
    report = _report(
        "candidate",
        0,
        teacher,
        teacher,
        text,
        scenes,
        regions,
        queries,
    )
    holdout_family = fidelity_module.IMAGENET12K_HOLDOUT_BANK_FAMILY
    report["selection_contract"] = (
        fidelity_module.selection_contract_for_bank_family(holdout_family)
    )
    fidelity_module._validate_report(report)

    drifted = deepcopy(report)
    drifted["selection_contract"]["unregistered_field"] = True
    with pytest.raises(ValueError, match="selection_contract differs"):
        fidelity_module._validate_report(drifted)

    hybrid = deepcopy(report)
    hybrid["selection_contract"]["queries"] = (
        "target_blind_imagenet1k_primary_text_bank_v1"
    )
    hybrid["selection_contract"]["query_axis"] = "changed_axis"
    with pytest.raises(ValueError, match="selection_contract differs"):
        fidelity_module._validate_report(hybrid)


def test_paired_multi_seed_scene_bootstrap_promotes_consistent_improvement():
    teacher, text, scenes, regions, queries = _descriptors()
    degraded = teacher.clone()
    degraded[:4] = degraded[torch.tensor([3, 2, 1, 0])]
    degraded[4:] = degraded[torch.tensor([7, 6, 5, 4])]
    controls = [
        _report("control", seed, degraded, teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]
    candidates = [
        _report("candidate", seed, teacher, teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]

    gate = aggregate_paired_seed_gate(
        controls,
        candidates,
        bootstrap_samples=100,
        bootstrap_seed=9,
        **_unit_gate_kwargs(scenes),
    )

    assert gate["decision"] == "promote"
    assert gate["improved_seed_counts"] == {"smooth_l1": 3, "mae": 3}
    assert gate["scene_bootstrap_ci95"]["smooth_l1"]["low"] > 0.0
    assert all(gate["checks"].values())


def test_paired_gate_rejects_an_unpaired_teacher_artifact():
    teacher, text, scenes, regions, queries = _descriptors()
    controls = [
        _report("control", seed, teacher, teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]
    candidates = [
        _report("candidate", seed, teacher, teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]
    candidates[1]["teacher_descriptors_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="not paired"):
        aggregate_paired_seed_gate(
            controls,
            candidates,
            bootstrap_samples=100,
            **_unit_gate_kwargs(scenes),
        )


def test_paired_gate_rejects_tampered_aggregate_metrics():
    teacher, text, scenes, regions, queries = _descriptors()
    controls = [
        _report("control", seed, teacher, teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]
    candidates = deepcopy(controls)
    for report in candidates:
        report["method_id"] = "candidate"
    candidates[0]["metrics"]["aggregate"]["top_decile_overlap_mean"] = 0.5

    with pytest.raises(ValueError, match="aggregate top_decile_overlap_mean"):
        aggregate_paired_seed_gate(
            controls,
            candidates,
            bootstrap_samples=100,
            **_unit_gate_kwargs(scenes),
        )


@pytest.mark.parametrize("metric", ["smooth_l1", "mae"])
def test_paired_gate_rejects_tampered_error_aggregate(metric):
    teacher, text, scenes, regions, queries = _descriptors()
    controls = [
        _report("control", seed, teacher, teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]
    candidates = deepcopy(controls)
    for report in candidates:
        report["method_id"] = "candidate"
    candidates[0]["metrics"]["aggregate"][metric] += 0.01

    with pytest.raises(ValueError, match=f"aggregate {metric}"):
        aggregate_paired_seed_gate(
            controls,
            candidates,
            bootstrap_samples=100,
            **_unit_gate_kwargs(scenes),
        )


def test_paired_gate_rejects_tampered_scene_error_detail():
    teacher, text, scenes, regions, queries = _descriptors()
    controls = [
        _report("control", seed, teacher, teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]
    candidates = deepcopy(controls)
    for report in candidates:
        report["method_id"] = "candidate"
    candidates[0]["metrics"]["scene_metrics"][0]["smooth_l1"] += 0.01

    with pytest.raises(ValueError, match="scene scene_a smooth_l1"):
        aggregate_paired_seed_gate(
            controls,
            candidates,
            bootstrap_samples=100,
            **_unit_gate_kwargs(scenes),
        )


def test_paired_gate_rejects_cross_split_reports_before_bootstrap():
    teacher, text, scenes, regions, queries = _descriptors()
    controls = [
        _report("control", seed, teacher.roll(1, 0), teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]
    candidates = [
        _report("candidate", seed, teacher, teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]
    for report in candidates:
        report["query_split"] = "audit"
        report["query_bank"]["query_split"] = "audit"

    with pytest.raises(ValueError, match="query_split differs from phase dev"):
        aggregate_paired_seed_gate(
            controls,
            candidates,
            phase="dev",
            bootstrap_samples=100,
            _test_expected_scene_ids=("scene_a", "scene_b"),
            _test_report_recomputer=lambda report, phase: report,
        )


def test_paired_gate_rejects_single_scene_contract():
    with pytest.raises(ValueError, match="at least two unique scenes"):
        aggregate_paired_seed_gate(
            [],
            [],
            phase="dev",
            bootstrap_samples=100,
            _test_expected_scene_ids=("only_scene",),
            _test_report_recomputer=lambda report, phase: report,
        )


def test_paired_gate_rejects_preregistered_scene_subset():
    teacher, text, scenes, regions, queries = _descriptors()
    controls = [
        _report("control", seed, teacher.roll(1, 0), teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]
    candidates = [
        _report("candidate", seed, teacher, teacher, text, scenes, regions, queries)
        for seed in (0, 1, 2)
    ]
    with pytest.raises(ValueError, match="complete preregistered scene set"):
        aggregate_paired_seed_gate(
            controls,
            candidates,
            phase="dev",
            bootstrap_samples=100,
            _test_expected_scene_ids=("scene_a", "scene_b", "scene_c"),
            _test_report_recomputer=lambda report, phase: report,
        )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_TEST_SOURCE_SHA256 = {"mock_imagenet_source.txt": "a" * 64}


def _test_contracts(vocabulary_path: Path, records: list[dict], snapshot: Path) -> dict:
    split_hashes = {
        split: _split_sha256(records, split) for split in ("fit", "dev", "audit")
    }
    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    snapshot_names = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        *(str(value) for value in index["weight_map"].values()),
    }
    return {
        "_test_vocabulary_contract": {
            "canonical_vocabulary_sha256": _file_sha(vocabulary_path),
            "counts": {
                "source_synsets": len(records),
                "deduplicated_queries": len(records),
                **{
                    split: sum(record["split"] == split for record in records)
                    for split in ("fit", "dev", "audit")
                },
            },
            "source_sha256": _TEST_SOURCE_SHA256,
            "split_sha256": split_hashes,
        },
        "_test_snapshot_files_sha256": {
            name: _file_sha(snapshot / name) for name in snapshot_names
        },
    }


def _write_strict_artifacts(tmp_path: Path, *, split: str = "dev"):
    records = [
        {"synset": "n1", "query": "alpha object", "split": "fit"},
        {"synset": "n2", "query": "beta object", "split": "dev"},
        {"synset": "n3", "query": "gamma object", "split": "dev"},
        {"synset": "n4", "query": "delta object", "split": "audit"},
    ]
    vocabulary = {
        "schema_version": 1,
        "artifact_type": "target_blind_imagenet1k_primary_text_bank",
        "algorithm_version": "imagenet1k-primary-v1",
        "prompt_templates": ["{query}"],
        "benchmark_vocabulary_opened": False,
        "records": records,
    }
    vocabulary_path = tmp_path / "vocabulary.json"
    vocabulary_path.write_text(
        json.dumps(vocabulary, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "target_blind_imagenet1k_primary_text_bank_manifest",
        "algorithm_version": "imagenet1k-primary-v1",
        "benchmark_vocabulary_opened": False,
        "counts": {
            "source_synsets": len(records),
            "deduplicated_queries": len(records),
            **{
                name: sum(record["split"] == name for record in records)
                for name in ("fit", "dev", "audit")
            },
        },
        "sources": {
            name: {"path": f"/mock/{name}", "sha256": digest}
            for name, digest in _TEST_SOURCE_SHA256.items()
        },
        "canonical_json": {
            "path": str(vocabulary_path),
            "sha256": _file_sha(vocabulary_path),
        },
        "split_synset_tab_query_lf_sha256": {
            split: _split_sha256(records, split) for split in ("fit", "dev", "audit")
        },
    }
    manifest_path = tmp_path / "vocabulary.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    snapshot = tmp_path / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "siglip",
                "text_config": {
                    "hidden_size": 8,
                    "projection_size": OUTPUT_DIMENSION,
                },
            }
        ),
        encoding="utf-8",
    )
    for name, content in {
        "tokenizer.json": b"tokenizer-json",
        "tokenizer.model": b"tokenizer-model",
        "tokenizer_config.json": b"{}",
        "special_tokens_map.json": b"{}",
        "preprocessor_config.json": b"{}",
    }.items():
        (snapshot / name).write_bytes(content)
    shard_name = "model-00001-of-00001.safetensors"
    (snapshot / shard_name).write_bytes(b"mock-local-weight-shard")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "text_model.head.weight": shard_name,
                    "text_model.head.bias": shard_name,
                    "text_model.embeddings.token_embedding.weight": shard_name,
                }
            }
        ),
        encoding="utf-8",
    )
    contracts = _test_contracts(vocabulary_path, records, snapshot)

    query_dimension = {
        "beta object": 0,
        "gamma object": 1,
        "delta object": 2,
    }

    def encoder(queries, unused_snapshot):
        del unused_snapshot
        result = torch.zeros(len(queries), OUTPUT_DIMENSION, dtype=torch.float32)
        for row, query in enumerate(queries):
            result[row, query_dimension[query]] = 1.0
        return result

    bank_path = tmp_path / f"{split}_embeddings.pt"
    sidecar_path = tmp_path / f"{split}_embeddings.manifest.json"
    build_embedding_artifact(
        vocabulary=vocabulary_path,
        vocabulary_manifest=manifest_path,
        split=split,
        snapshot=snapshot,
        output=bank_path,
        sidecar_output=sidecar_path,
        batch_size=1,
        batch_encoder=encoder,
        **contracts,
    )

    teacher, _, scenes, regions, _ = _descriptors()
    teacher_1536 = torch.zeros(len(teacher), OUTPUT_DIMENSION)
    teacher_1536[:, : teacher.shape[1]] = teacher
    teacher_1536 = F.normalize(teacher_1536, dim=-1)
    readout_checkpoint = tmp_path / "readout.pt"
    readout_checkpoint.write_bytes(b"mock-readout")
    readout_report = tmp_path / "readout.pt.json"
    readout_report.write_text("{}\n", encoding="utf-8")
    radio_checkpoint = tmp_path / "radio.pt"
    radio_checkpoint.write_bytes(b"mock-radio")
    authority_path = tmp_path / "distill_run_manifest.json"
    authority_path.write_text("{}\n", encoding="utf-8")
    split_sha = "a" * 64
    region_contract_sha = "b" * 64
    teacher_protocol_sha = "c" * 64
    teacher_region = {
        "semantics": "unit-test",
        "contract": {"unit_test": True},
        "contract_sha256": "d" * 64,
        "target_source": "unit-test",
        "target_protocol_sha256": teacher_protocol_sha,
    }
    cache_path = tmp_path / "validation_cache.pt"
    cache_records = [
        {"scene": scene, "region_id": region}
        for scene, region in zip(scenes, regions)
    ]
    torch.save(
        {
            "metadata": {
                "schema_version": 3,
                "split_role": "validation",
                "uses_benchmark_scenes": False,
                "uses_benchmark_test_vocabulary": False,
                "annotations_opened": False,
                "labels_opened": False,
                "instances_opened": False,
                "masks_opened": False,
                "text_opened": False,
                "region_records": cache_records,
                "scene_names": sorted(set(scenes)),
                "scene_region_counts": {
                    scene: scenes.count(scene) for scene in sorted(set(scenes))
                },
                "split_file_sha256": split_sha,
                "region_contract_sha256": region_contract_sha,
                "radio_checkpoint_sha256": _file_sha(radio_checkpoint),
                "teacher_region_semantics": teacher_region["semantics"],
                "teacher_region_contract": teacher_region["contract"],
                "teacher_region_contract_sha256": teacher_region["contract_sha256"],
                "teacher_target_source": teacher_region["target_source"],
                "teacher_target_protocol_sha256": teacher_protocol_sha,
            }
        },
        cache_path,
    )
    descriptor = {
        "schema_version": 1,
        "artifact_type": "surface_text_response_descriptor_pair",
        "method_id": "unit-test",
        "seed": 0,
        "split_role": "validation",
        "student_descriptors": teacher_1536,
        "teacher_descriptors": teacher_1536,
        "scene_ids": scenes,
        "region_ids": regions,
        "student_descriptors_sha256": tensor_sha256(teacher_1536),
        "teacher_descriptors_sha256": tensor_sha256(teacher_1536),
        "descriptor_rows_sha256": row_identity_sha256(scenes, regions),
        "descriptor_space": {
            "name": "official_siglip2_g_summary",
            "dimension": OUTPUT_DIMENSION,
            "normalization": "l2",
            "official_summary_head": "c-radio_v4 _heads.siglip2-g",
        },
        "provenance": {
            "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False,
            "annotations_opened": False,
            "labels_opened": False,
            "instances_opened": False,
            "masks_opened": False,
            "text_opened": False,
            "device": "cpu",
            "readout_checkpoint": str(readout_checkpoint.resolve()),
            "readout_checkpoint_sha256": _file_sha(readout_checkpoint),
            "readout_report": str(readout_report.resolve()),
            "readout_report_sha256": _file_sha(readout_report),
            "readout_binding_authority": {
                "type": "embedded_distill_run_manifest",
                "path": str(authority_path.resolve()),
                "sha256": _file_sha(authority_path),
                "candidate": "unit-test",
            },
            "radio_checkpoint": str(radio_checkpoint.resolve()),
            "radio_checkpoint_sha256": _file_sha(radio_checkpoint),
            "region_contract_sha256": region_contract_sha,
            "validation_split_sha256": split_sha,
            "validation_scenes": sorted(set(scenes)),
            "teacher_region": teacher_region,
            "validation_caches": [
                {
                    "path": str(cache_path.resolve()),
                    "sha256": _file_sha(cache_path),
                    "rows": len(cache_records),
                    "split_file_sha256": split_sha,
                    "region_contract_sha256": region_contract_sha,
                    "radio_checkpoint_sha256": _file_sha(radio_checkpoint),
                    "teacher_target_protocol_sha256": teacher_protocol_sha,
                }
            ],
        },
    }
    descriptor_path = tmp_path / "descriptors.pt"
    torch.save(descriptor, descriptor_path)
    return bank_path, sidecar_path, descriptor_path, snapshot, contracts


def test_strict_artifact_evaluation_selects_only_dev_queries(tmp_path):
    bank_path, sidecar_path, descriptor_path, _, contracts = _write_strict_artifacts(tmp_path)

    report = evaluate_artifacts(
        descriptor_path,
        bank_path,
        sidecar_path,
        query_split="dev",
        **contracts,
    )

    assert report["query_bank"]["selected_queries"] == 2
    assert report["metrics"]["counts"]["queries"] == 2
    assert report["metrics"]["aggregate"]["smooth_l1"] == 0.0
    assert report["selection_contract"]["benchmark_vocabulary_opened"] is False
    assert report["query_bank"]["manifest_path"] == str(sidecar_path.resolve())
    assert report["query_bank"]["embedding_semantic_sha256"]
    assert not torch.cuda.is_initialized()


def test_evaluate_many_reuses_one_hash_and_bank_cache(monkeypatch, tmp_path):
    observed = []

    def fake_evaluate(
        descriptor_path,
        text_bank_path,
        text_bank_manifest_path,
        *,
        query_split,
        _hash_cache,
        _descriptor_cache,
        _bank_cache,
    ):
        observed.append(
            (id(_hash_cache), id(_descriptor_cache), id(_bank_cache), query_split)
        )
        return {"descriptor": str(descriptor_path)}

    monkeypatch.setattr(fidelity_gate_module, "evaluate_artifacts", fake_evaluate)
    descriptors = [tmp_path / "a.pt", tmp_path / "b.pt"]
    reports = fidelity_gate_module.evaluate_many_artifacts(
        descriptors,
        tmp_path / "bank.pt",
        tmp_path / "bank.json",
        query_split="dev",
    )

    assert reports == [{"descriptor": str(path)} for path in descriptors]
    assert len(set(observed)) == 1


def test_text_bank_loader_refuses_fit_as_a_promotion_split(tmp_path):
    bank_path, sidecar_path, _, _, contracts = _write_strict_artifacts(tmp_path)
    with pytest.raises(ValueError, match="held-out"):
        load_text_embedding_bank(bank_path, sidecar_path, "fit", **contracts)


def test_text_bank_loader_rejects_requested_split_mismatch(tmp_path):
    bank_path, sidecar_path, _, _, contracts = _write_strict_artifacts(tmp_path)
    with pytest.raises(ValueError, match="differs from the frozen embedding split"):
        load_text_embedding_bank(bank_path, sidecar_path, "audit", **contracts)


def test_production_text_bank_loader_rejects_self_signed_mock_bank(tmp_path):
    bank_path, sidecar_path, _, _, _ = _write_strict_artifacts(tmp_path)
    with pytest.raises(ValueError, match="not the frozen target-blind bank"):
        load_text_embedding_bank(bank_path, sidecar_path, "dev")


def test_text_bank_loader_accepts_frozen_audit_split(tmp_path):
    bank_path, sidecar_path, _, _, contracts = _write_strict_artifacts(
        tmp_path,
        split="audit",
    )
    bank = load_text_embedding_bank(bank_path, sidecar_path, "audit", **contracts)
    assert bank["query_split"] == "audit"
    assert bank["query_ids"] == ["n4"]
    assert bank["embeddings"].shape == (1, OUTPUT_DIMENSION)


def test_historical_builder_compatibility_is_exactly_bound_to_formal_dev_bank():
    formal = fidelity_gate_module.FORMAL_HISTORICAL_TEXT_BANKS["dev"]
    fidelity_gate_module._validate_embedding_builder_provenance(
        builder_path=Path(
            fidelity_gate_module.FORMAL_HISTORICAL_BUILDER["path"]
        ),
        builder_sha256=fidelity_gate_module.FORMAL_HISTORICAL_BUILDER["sha256"],
        artifact_path=Path(formal["artifact_path"]),
        artifact_sha256=formal["artifact_sha256"],
        manifest_path=Path(formal["manifest_path"]),
        manifest_sha256=formal["manifest_sha256"],
        query_split="dev",
        algorithm_version=fidelity_gate_module.TEXT_BANK_ALGORITHM_VERSION,
        hash_cache={},
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("artifact_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
    ],
)
def test_historical_builder_compatibility_rejects_formal_bank_drift(
    field, replacement
):
    formal = dict(fidelity_gate_module.FORMAL_HISTORICAL_TEXT_BANKS["dev"])
    formal[field] = replacement
    with pytest.raises(ValueError, match="unexpected or changed builder"):
        fidelity_gate_module._validate_embedding_builder_provenance(
            builder_path=Path(
                fidelity_gate_module.FORMAL_HISTORICAL_BUILDER["path"]
            ),
            builder_sha256=fidelity_gate_module.FORMAL_HISTORICAL_BUILDER["sha256"],
            artifact_path=Path(formal["artifact_path"]),
            artifact_sha256=formal["artifact_sha256"],
            manifest_path=Path(formal["manifest_path"]),
            manifest_sha256=formal["manifest_sha256"],
            query_split="dev",
            algorithm_version=fidelity_gate_module.TEXT_BANK_ALGORITHM_VERSION,
            hash_cache={},
        )


def test_text_bank_loader_rejects_tampered_artifact_file_hash(tmp_path):
    bank_path, sidecar_path, _, _, contracts = _write_strict_artifacts(tmp_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["artifact"]["sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="embedding artifact SHA256 mismatch"):
        load_text_embedding_bank(bank_path, sidecar_path, "dev", **contracts)


def test_text_bank_loader_rejects_tampered_embedding_semantic_hash(tmp_path):
    bank_path, sidecar_path, _, _, contracts = _write_strict_artifacts(tmp_path)
    payload = torch.load(bank_path, map_location="cpu")
    payload["embedding_semantic_sha256"] = "0" * 64
    torch.save(payload, bank_path)
    with pytest.raises(ValueError, match="embedding semantic tensor SHA256 mismatch"):
        load_text_embedding_bank(bank_path, sidecar_path, "dev", **contracts)


def test_text_bank_loader_rejects_tampered_encoder_snapshot_file(tmp_path):
    bank_path, sidecar_path, _, snapshot, contracts = _write_strict_artifacts(tmp_path)
    (snapshot / "tokenizer.json").write_bytes(b"tampered-tokenizer")
    with pytest.raises(ValueError, match="text encoder file tokenizer.json"):
        load_text_embedding_bank(bank_path, sidecar_path, "dev", **contracts)


def test_text_gate_snapshot_reader_reuses_strict_huggingface_blob_resolution(
    tmp_path,
):
    model_root = tmp_path / "models--google--siglip2-giant-opt-patch16-384"
    snapshot = model_root / "snapshots" / MODEL_REVISION
    blob_root = model_root / "blobs"
    snapshot.mkdir(parents=True)
    blob_root.mkdir()
    blob = blob_root / ("a" * 64)
    blob.write_bytes(b"fixed-content")
    (snapshot / "config.json").symlink_to(Path("../../blobs") / blob.name)

    assert fidelity_gate_module._safe_snapshot_file(snapshot, "config.json") == blob

    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to("../../../outside")
    with pytest.raises(ValueError, match="not a Hugging Face blob"):
        fidelity_gate_module._safe_snapshot_file(snapshot, "config.json")


def test_descriptor_loader_reopens_bound_provenance_files(tmp_path):
    _, _, descriptor_path, _, _ = _write_strict_artifacts(tmp_path)
    descriptor = torch.load(descriptor_path, map_location="cpu")
    readout = Path(descriptor["provenance"]["readout_checkpoint"])
    readout.write_bytes(readout.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="readout checkpoint SHA256 mismatch"):
        load_descriptor_pair(descriptor_path)


def test_descriptor_authority_accepts_hash_bound_attention_postcache_screen(
    tmp_path,
):
    screen = tmp_path / "attention_screen.json"
    completion = tmp_path / "screen.complete"
    screen.write_text("{}\n", encoding="utf-8")
    completion.write_text("2026-08-01T00:00:00+08:00\n", encoding="utf-8")
    authority = {
        "type": "attention_postcache_screen",
        "path": str(screen.resolve()),
        "sha256": _file_sha(screen),
        "completion": str(completion.resolve()),
        "completion_sha256": _file_sha(completion),
        "candidate": "context_c1024_geometric",
    }

    assert fidelity_gate_module._validate_binding_authority(
        authority,
        relative_to=tmp_path,
        hash_cache={},
    ) == authority

    authority["completion_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="completion SHA256 mismatch"):
        fidelity_gate_module._validate_binding_authority(
            authority,
            relative_to=tmp_path,
            hash_cache={},
        )


def test_gate_rejects_coherently_rewritten_metrics_by_reopening_sources(
    tmp_path,
    monkeypatch,
):
    bank_path, sidecar_path, descriptor_path, _, contracts = _write_strict_artifacts(
        tmp_path
    )
    report = evaluate_artifacts(
        descriptor_path,
        bank_path,
        sidecar_path,
        query_split="dev",
        **contracts,
    )
    descriptor = load_descriptor_pair(descriptor_path)
    bank = load_text_embedding_bank(bank_path, sidecar_path, "dev", **contracts)
    # Replace every detailed and aggregate metric coherently.  Internal JSON
    # consistency alone cannot detect this; source-artifact recomputation must.
    report["metrics"] = evaluate_response_fidelity(
        descriptor["student"].roll(1, dims=0),
        descriptor["teacher"],
        bank["embeddings"],
        scene_ids=descriptor["scene_ids"],
        region_ids=descriptor["region_ids"],
        query_ids=bank["query_ids"],
    )
    monkeypatch.setattr(
        fidelity_module,
        "_load_frozen_validation_scenes",
        lambda: tuple(sorted(set(descriptor["scene_ids"]))),
    )
    monkeypatch.setattr(
        fidelity_gate_module,
        "FROZEN_VOCABULARY_CONTRACT",
        contracts["_test_vocabulary_contract"],
    )
    monkeypatch.setattr(
        fidelity_gate_module,
        "FROZEN_SNAPSHOT_FILES_SHA256",
        contracts["_test_snapshot_files_sha256"],
    )

    with pytest.raises(ValueError, match="strict source-artifact recomputation"):
        aggregate_paired_seed_gate(
            [report],
            [report],
            required_seeds=(0,),
            minimum_improved_seeds=1,
            bootstrap_samples=100,
            phase="dev",
        )


def test_teacher_descriptor_matches_normalized_valid_view_mean():
    views = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]]])
    mask = torch.tensor([[True, True, False]])
    descriptor = _teacher_descriptor(views, mask)
    expected = F.normalize(torch.tensor([[0.5, 0.5]]), dim=-1)
    torch.testing.assert_close(descriptor, expected)


def test_legacy_region_id_ignores_student_context_size():
    record = {
        "scene": "scene0001_00",
        "seed": 17,
        "physical_radius_m": 0.45,
        "teacher_views": [{"frame": "0001.jpg", "crop_box_tlbr": [1, 2, 3, 4]}],
        "tokens": 128,
    }
    changed_context = {**record, "tokens": 1024}
    assert _legacy_region_id(record) == _legacy_region_id(changed_context)
