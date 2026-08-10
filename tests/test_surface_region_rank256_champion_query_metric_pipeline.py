from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.interfaces import surface_region_rank256_champion as formal
from radio_gs.scripts import run_surface_region_rank256_champion_lerf_pipeline as pipeline
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, file_record


SHA = "0" * 64


def _fake_gate(tmp_path: Path) -> dict:
    return {
        "source_result": {"path": str((tmp_path / "source.json").resolve()), "sha256": SHA},
        "checkpoint": {"path": str((tmp_path / "checkpoint.pt").resolve()), "sha256": SHA},
        "normalization_authority": {"path": str((tmp_path / "normalization.pt").resolve()), "sha256": SHA},
        "source_promotion_authorized": True,
        "benchmark_opened": False,
    }


def test_exact_query_subset_is_source_first_and_exact_order(tmp_path: Path, monkeypatch) -> None:
    gate = _fake_gate(tmp_path)
    monkeypatch.setattr(pipeline, "_source", lambda _args: gate)
    queries = ["other", "query-b", "query-a"]
    embeddings = torch.eye(1536, dtype=torch.float32)[: len(queries)]
    all_cache = tmp_path / "all.pt"
    torch.save(
        {
            "queries": queries,
            "prompt_templates": ["{query}"],
            "text_encoder": "siglip2",
            "model_name": pipeline.CANONICAL_NEGATIVE_MODEL,
            "embeddings": embeddings,
        },
        all_cache,
    )
    evaluator = Path(pipeline.frozen_evaluator.__file__).resolve()
    manifest = {
        "schema": formal.EXACT_QUERY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "scene_id": "figurines",
        "query_ids": ["query-a", "query-b"],
        "query_ids_sha256": canonical_json_sha256(["query-a", "query-b"]),
        "frozen_all_query_cache": file_record(all_cache),
        "frozen_evaluator": file_record(evaluator),
        "frozen_before_champion_query_execution": True,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "exact.pt"
    receipt = tmp_path / "exact.receipt.json"
    result = pipeline.materialize_exact_query_subset(
        argparse.Namespace(
            source_variant="v21b", source_result=gate["source_result"]["path"],
            expected_source_result_sha256=SHA, scene_id="figurines",
            query_manifest=str(manifest_path.resolve()),
            expected_query_manifest_sha256=file_record(manifest_path)["sha256"],
            all_query_cache=str(all_cache.resolve()),
            expected_all_query_cache_sha256=file_record(all_cache)["sha256"],
            output=str(output.resolve()), output_receipt=str(receipt.resolve()),
        )
    )
    assert result["status"] == "exact_query_subset_complete"
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["queries"] == ["query-a", "query-b"]
    assert payload["text_canonicalization"] == pipeline.OFFICIAL_TEXT_CANONICALIZATION
    assert torch.equal(payload["embeddings"], embeddings[[2, 1]])
    frozen_receipt = formal.validate_exact_query_receipt(json.loads(receipt.read_text()))
    assert frozen_receipt["query_ids"] == ["query-a", "query-b"]


def test_exact_query_subset_rejects_before_query_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "_source", lambda _args: (_ for _ in ()).throw(ValueError("source failed")))
    with pytest.raises(ValueError, match="source failed"):
        pipeline.materialize_exact_query_subset(
            argparse.Namespace(
                source_variant="v21b", source_result="/missing/source.json",
                expected_source_result_sha256=SHA, scene_id="figurines",
                query_manifest="/missing/query.json", expected_query_manifest_sha256=SHA,
                all_query_cache="/missing/all.pt", expected_all_query_cache_sha256=SHA,
                output=str((tmp_path / "out.pt").resolve()),
                output_receipt=str((tmp_path / "receipt.json").resolve()),
            )
        )
    assert not (tmp_path / "out.pt").exists()


def test_exact_query_subset_rejects_another_evaluator(tmp_path: Path, monkeypatch) -> None:
    gate = _fake_gate(tmp_path)
    monkeypatch.setattr(pipeline, "_source", lambda _args: gate)
    all_cache = tmp_path / "all.pt"
    torch.save(
        {
            "queries": ["query-a"],
            "prompt_templates": ["{query}"],
            "text_encoder": "siglip2",
            "model_name": pipeline.CANONICAL_NEGATIVE_MODEL,
            "embeddings": torch.nn.functional.normalize(
                torch.ones(1, 1536, dtype=torch.float32), dim=-1
            ),
        },
        all_cache,
    )
    other_evaluator = tmp_path / "other_evaluator.py"
    other_evaluator.write_text("# not the frozen evaluator\n", encoding="utf-8")
    manifest = {
        "schema": formal.EXACT_QUERY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "scene_id": "figurines",
        "query_ids": ["query-a"],
        "query_ids_sha256": canonical_json_sha256(["query-a"]),
        "frozen_all_query_cache": file_record(all_cache),
        "frozen_evaluator": file_record(other_evaluator),
        "frozen_before_champion_query_execution": True,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest evaluator differs"):
        pipeline.materialize_exact_query_subset(
            argparse.Namespace(
                source_variant="v21b", source_result=gate["source_result"]["path"],
                expected_source_result_sha256=SHA, scene_id="figurines",
                query_manifest=str(manifest_path.resolve()),
                expected_query_manifest_sha256=file_record(manifest_path)["sha256"],
                all_query_cache=str(all_cache.resolve()),
                expected_all_query_cache_sha256=file_record(all_cache)["sha256"],
                output=str((tmp_path / "out.pt").resolve()),
                output_receipt=str((tmp_path / "receipt.json").resolve()),
            )
        )


def test_descriptor_binding_requires_frozen_producer_and_exact_inputs() -> None:
    producer = {"path": "/producer.py", "sha256": "1" * 64}
    execution_record = {"path": "/authority.json", "sha256": "2" * 64}
    inputs = {"champion_checkpoint": {"path": "/model.pt", "sha256": "3" * 64}}
    descriptor = {
        "producer": producer,
        "target_execution_authority": execution_record,
        "input_authority": inputs,
        "scene_id": "figurines",
        "physical_space_id": "physical",
    }
    execution = {
        "implementation": producer,
        "verified_record": execution_record,
        "target_inputs": inputs,
        "scene_id": "figurines",
        "physical_space_id": "physical",
    }
    pipeline._validate_frozen_target_binding(descriptor, execution)
    descriptor["producer"] = {"path": "/other.py", "sha256": "4" * 64}
    with pytest.raises(ValueError, match="descriptor/frozen-target binding differs"):
        pipeline._validate_frozen_target_binding(descriptor, execution)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("extra_field", "contract differs"),
        ("wrong_model", "contract differs"),
        ("non_unit", "not unit L2"),
    ],
)
def test_legacy_all_query_cache_is_strict(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    payload = {
        "queries": ["query-a", "query-b"],
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": pipeline.CANONICAL_NEGATIVE_MODEL,
        "embeddings": torch.nn.functional.normalize(
            torch.ones(2, 1536, dtype=torch.float32), dim=-1
        ).contiguous(),
    }
    if tamper == "extra_field":
        payload["text_canonicalization"] = pipeline.OFFICIAL_TEXT_CANONICALIZATION
    elif tamper == "wrong_model":
        payload["model_name"] = "wrong-model"
    else:
        payload["embeddings"][0].mul_(2.0)
    path = tmp_path / f"{tamper}.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        pipeline._load_legacy_frozen_all_query_cache(
            path, expected_sha256=file_record(path)["sha256"]
        )


def test_external_materializer_emits_frozen_evaluator_cache(tmp_path: Path, monkeypatch) -> None:
    feature = {
        "scene_id": "figurines",
        "region_fingerprints": ["1" * 64, "2" * 64],
        "canonical_region_indices": torch.tensor([0, 1]),
        "region_rows": torch.tensor([[0, 1], [2, 3]]),
        "token_mask": torch.ones(2, 2, dtype=torch.bool),
        "pair_indices": torch.tensor([[0, 1]], dtype=torch.long),
    }
    inference = {
        "selected_rule": {"method": "dual_path_widest", "maximum_regions": 8, "threshold": 0.9},
        "pair_probabilities": torch.tensor([0.95]),
    }
    relevance = {
        "query_ids": ["a", "b"],
        "scene_id": "figurines",
        "region_fingerprints": feature["region_fingerprints"],
        "canonical_region_indices": feature["canonical_region_indices"],
        "region_absolute_relevance": torch.tensor([[0.9, 0.1], [0.1, 0.8]]),
    }
    state = SimpleNamespace(valid=torch.ones(4, dtype=torch.bool), xyz=torch.arange(12, dtype=torch.float32).view(4, 3))
    output = tmp_path / "external.pt"
    report = tmp_path / "external.json"
    monkeypatch.setattr(
        pipeline,
        "validate_external_authority",
        lambda *_args, **_kwargs: {
            "output_cache": str(output.resolve()), "output_report": str(report.resolve()),
            "verified_state": state, "verified_feature": feature,
            "verified_inference": inference, "verified_relevance": relevance,
            "verified_record": {"path": str((tmp_path / "authority.json").resolve()), "sha256": SHA},
        },
    )
    result = pipeline.materialize_external(
        argparse.Namespace(execution_authority="unused", expected_execution_authority_sha256=SHA)
    )
    assert result["status"] == "rank256_champion_external_cache_complete"
    cache = torch.load(output, map_location="cpu", weights_only=False)
    assert cache["schema"] == formal.EXTERNAL_CACHE_SCHEMA
    assert cache["metadata"]["query_names"] == ["a", "b"]
    assert cache["query_scores"].shape == (4, 2)


def test_metric_wrapper_is_no_clobber_and_uses_only_frozen_argv(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "evaluation"
    evaluator = tmp_path / "evaluator.py"
    for path in (evaluator, tmp_path / "config.yaml", tmp_path / "renderer.pth", tmp_path / "head.pth", tmp_path / "queries.pt", tmp_path / "negatives.pt", tmp_path / "external.pt"):
        path.write_bytes(b"x")
    authority = {
        "output_dir": str(output.resolve()), "scene_id": "figurines",
        "frozen_evaluator": file_record(evaluator),
        "external_cache": file_record(tmp_path / "external.pt"),
        "frozen_inputs": {
            "config": file_record(tmp_path / "config.yaml"),
            "renderer_geometry_checkpoint": file_record(tmp_path / "renderer.pth"),
            "summary_head": file_record(tmp_path / "head.pth"),
            "all_query_text_cache": file_record(tmp_path / "queries.pt"),
            "canonical_negative_text_cache": file_record(tmp_path / "negatives.pt"),
        },
        "label_root": str(tmp_path.resolve()),
        "verified_record": {"path": str((tmp_path / "authority.json").resolve()), "sha256": SHA},
    }
    monkeypatch.setattr(pipeline, "validate_metric_authority", lambda *_args, **_kwargs: authority)
    seen: list[list[str]] = []

    def fake_run(command, check):
        assert check is True
        seen.append(command)
        result = output / "figurines" / "lerf_direct_3d_selection_results.json"
        result.parent.mkdir(parents=True)
        result.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    result = pipeline.run_metric(argparse.Namespace(execution_authority="unused", expected_execution_authority_sha256=SHA, gpu=0))
    assert result["status"] == "rank256_champion_one_shot_metric_complete"
    command = seen[0]
    assert command[command.index("--protocol_preset") + 1] == "vala_paper_3d"
    assert "--external_query_score_cache" in command
    with pytest.raises(FileExistsError):
        pipeline.run_metric(argparse.Namespace(execution_authority="unused", expected_execution_authority_sha256=SHA, gpu=0))
