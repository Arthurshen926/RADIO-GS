from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import threading
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from radio_gs.evaluation import text_response_fidelity as fidelity
from radio_gs.scripts import confirm_surface_readout_weight_interpolation_audit as module


FORMAL_SELECTION = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/"
    "surface_readout_weight_interpolation_selection_alpha01_joint_c1024_"
    "src50c48dfab98e.json"
)
FORMAL_SELECTION_SHA256 = (
    "428b92d18dab62a5747ea0602fb7ce36251430f712ce9e5346c18d9f2aa9dbf8"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: bytes) -> dict[str, str]:
    path.write_bytes(content)
    return {"path": str(path), "sha256": _sha(path)}


def _teacher_fixture():
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
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.1, 1.0]]
        ),
        dim=-1,
    )
    scenes = ["scene_a"] * 4 + ["scene_b"] * 4
    regions = [f"region_{index}" for index in range(8)]
    queries = ["q0", "q1", "q2"]
    degraded = teacher.clone()
    degraded[:4] = degraded[torch.tensor([3, 2, 1, 0])]
    degraded[4:] = degraded[torch.tensor([7, 6, 5, 4])]
    return teacher, degraded, text, scenes, regions, queries


def _query_bank_record(artifact: dict[str, str], manifest: dict[str, str]) -> dict:
    return {
        "path": artifact["path"],
        "sha256": artifact["sha256"],
        "manifest_path": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "vocabulary_sha256": "1" * 64,
        "query_split": "audit",
        "selected_queries": 3,
        "selected_records_sha256": "2" * 64,
        "ordered_records_sha256": "3" * 64,
        "embedding_tensor_sha256": "4" * 64,
        "embedding_semantic_sha256": "5" * 64,
        "text_encoder": {"unit_test": True},
    }


def _embedded_record(
    method: str,
    seed: int,
    student: torch.Tensor,
    teacher: torch.Tensor,
    text: torch.Tensor,
    scenes: list[str],
    regions: list[str],
    queries: list[str],
    query_bank: dict,
) -> dict:
    return {
        "method_id": method,
        "seed": seed,
        "query_split": "audit",
        "descriptor_rows_sha256": fidelity.row_identity_sha256(scenes, regions),
        "teacher_descriptors_sha256": fidelity.tensor_sha256(teacher),
        "query_bank": query_bank,
        "metrics": fidelity.evaluate_response_fidelity(
            student,
            teacher,
            text,
            scene_ids=scenes,
            region_ids=regions,
            query_ids=queries,
        ),
    }


def test_same_process_production_gate_binds_real_sources_without_test_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher, degraded, text, scenes, regions, queries = _teacher_fixture()
    scene_file = tmp_path / "scenes.txt"
    scene_file.write_text("scene_a\nscene_b\n", encoding="utf-8")
    monkeypatch.setattr(fidelity, "FROZEN_VALIDATION_SCENE_FILE", scene_file)
    monkeypatch.setattr(fidelity, "FROZEN_VALIDATION_SCENE_FILE_SHA256", _sha(scene_file))
    monkeypatch.setattr(fidelity, "FROZEN_VALIDATION_SCENES", ("scene_a", "scene_b"))

    selection = _write(tmp_path / "selection.json", b"selection")
    diagnostic = _write(tmp_path / "diagnostic.json", b"diagnostic")
    cache = _write(tmp_path / "cache.pt", b"cache")
    bank_artifact = _write(tmp_path / "audit.pt", b"audit")
    bank_manifest = _write(tmp_path / "audit.json", b"manifest")
    endpoints = []
    for seed in (0, 1, 2):
        endpoints.append(
            {
                "seed": seed,
                "control": _write(tmp_path / f"control_{seed}.pt", f"c{seed}".encode()),
                "candidate": _write(
                    tmp_path / f"candidate_{seed}.pt", f"x{seed}".encode()
                ),
            }
        )
    authority = {
        "schema_version": 1,
        "authority_type": "same_process_interpolation_audit_response_metrics_v1",
        "selection": selection,
        "diagnostic": diagnostic,
        "validation_caches": [cache],
        "endpoints": endpoints,
        "audit_bank": {
            "artifact": bank_artifact,
            "manifest": bank_manifest,
            "split": "audit",
            "query_count": 3,
        },
        "construction": (
            "evaluate_response_fidelity_from_cpu_descriptors_"
            "recomputed_after_selection_validation"
        ),
    }
    query_bank = _query_bank_record(bank_artifact, bank_manifest)
    controls = [
        _embedded_record(
            "control", seed, degraded, teacher, text, scenes, regions, queries, query_bank
        )
        for seed in (0, 1, 2)
    ]
    candidates = [
        _embedded_record(
            "candidate", seed, teacher, teacher, text, scenes, regions, queries, query_bank
        )
        for seed in (0, 1, 2)
    ]

    gate = fidelity.aggregate_paired_seed_gate_from_same_process_metrics(
        controls,
        candidates,
        source_authority=authority,
        phase="audit",
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert gate["decision"] == "promote"
    assert gate["protocol"]["preregistered_scene_file_sha256"] == _sha(scene_file)
    assert gate["protocol"]["embedded_metrics_source_authority"] == authority
    assert all("descriptor_artifact" not in record for record in controls + candidates)


def test_same_process_gate_rejects_source_sha_drift(
    tmp_path: Path,
) -> None:
    authority = {
        "schema_version": 1,
        "authority_type": "same_process_interpolation_audit_response_metrics_v1",
        "selection": _write(tmp_path / "selection", b"selection"),
        "diagnostic": _write(tmp_path / "diagnostic", b"diagnostic"),
        "validation_caches": [_write(tmp_path / "cache", b"cache")],
        "endpoints": [
            {
                "seed": seed,
                "control": _write(tmp_path / f"c{seed}", b"c"),
                "candidate": _write(tmp_path / f"x{seed}", b"x"),
            }
            for seed in (0, 1, 2)
        ],
        "audit_bank": {
            "artifact": _write(tmp_path / "bank", b"bank"),
            "manifest": _write(tmp_path / "manifest", b"manifest"),
            "split": "audit",
            "query_count": 3,
        },
        "construction": (
            "evaluate_response_fidelity_from_cpu_descriptors_"
            "recomputed_after_selection_validation"
        ),
    }
    authority["endpoints"][1]["candidate"]["sha256"] = "f" * 64

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fidelity.aggregate_paired_seed_gate_from_same_process_metrics(
            [], [], source_authority=authority, phase="audit"
        )


def test_formal_selection_recomputes_without_opening_audit() -> None:
    selection, diagnostic = module.validate_frozen_selection(
        FORMAL_SELECTION,
        FORMAL_SELECTION_SHA256,
        module.selector.FORMAL_DIAGNOSTIC_PATH,
        module.selector.FORMAL_DIAGNOSTIC_SHA256,
    )

    assert selection["selected_alpha"] == 0.1
    assert selection["audit"] == {"opened": False, "status": "unopened", "artifact": None}
    assert diagnostic["selection_contract"]["audit_opened"] is False


def _fake_runtime(tmp_path: Path) -> dict:
    teacher = F.normalize(
        torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]]),
        dim=-1,
    )
    control = teacher[torch.tensor([1, 0, 3, 2])]
    surface = {metric: 0.9 for metric in module.SURFACE_METRICS}
    cache = _write(tmp_path / "cache.pt", b"cache")
    return {
        "per_seed": [
            {
                "seed": seed,
                "control_checkpoint": {"path": f"/control/{seed}", "sha256": "c" * 64},
                "candidate_checkpoint": {"path": f"/candidate/{seed}", "sha256": "d" * 64},
                "control": {"student": control, "teacher": teacher, "surface": surface},
                "interpolated": {"student": teacher, "teacher": teacher, "surface": surface},
            }
            for seed in (0, 1, 2)
        ],
        "scene_ids": ["a", "a", "b", "b"],
        "region_ids": ["r0", "r1", "r2", "r3"],
        "scenes": ["a", "b"],
        "rows_sha256": fidelity.row_identity_sha256(
            ["a", "a", "b", "b"], ["r0", "r1", "r2", "r3"]
        ),
        "input_bindings": {"validation_caches": [cache]},
    }


def _fake_bank(tmp_path: Path) -> dict:
    artifact = tmp_path / "audit.pt"
    manifest = tmp_path / "audit.json"
    artifact.write_bytes(b"audit")
    manifest.write_bytes(b"manifest")
    text = torch.eye(2)
    return {
        "path": artifact,
        "file_sha256": _sha(artifact),
        "manifest_path": manifest,
        "manifest_sha256": _sha(manifest),
        "embeddings": text,
        "query_ids": ["q0", "q1"],
        "selected_records": [{"query": "q0"}, {"query": "q1"}],
        "selected_records_sha256": "1" * 64,
        "ordered_records_sha256": "2" * 64,
        "embedding_tensor_sha256": fidelity.tensor_sha256(text),
        "embedding_semantic_sha256": "3" * 64,
        "vocabulary_sha256": "4" * 64,
        "text_encoder": {"unit_test": True},
    }


def test_confirmation_publishes_receipt_before_single_bank_load_and_refuses_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    selection_path = tmp_path / "selection.json"
    diagnostic_path = tmp_path / "diagnostic.json"
    selection_path.write_text("{}", encoding="utf-8")
    diagnostic_path.write_text("{}", encoding="utf-8")
    runtime = _fake_runtime(tmp_path)
    bank = _fake_bank(tmp_path)
    receipt = tmp_path / "opened.json"
    output = tmp_path / "confirmation.json"
    calls = []

    monkeypatch.setattr(
        module,
        "validate_frozen_selection",
        lambda *args: ({"selected_alpha": 0.1}, {}),
    )
    monkeypatch.setattr(module, "_load_query_free_runtime", lambda *args, **kwargs: runtime)

    def load_after_receipt(path):
        assert path == receipt and receipt.is_file()
        calls.append(path)
        return bank

    monkeypatch.setattr(module, "_load_audit_bank_after_receipt", load_after_receipt)
    monkeypatch.setattr(
        module,
        "_paired_gate_in_memory",
        lambda *args, **kwargs: {"decision": "promote", "protocol": {"unit_test": True}},
    )
    args = argparse.Namespace(
        selection=selection_path,
        selection_sha256="a" * 64,
        diagnostic=diagnostic_path,
        diagnostic_sha256="b" * 64,
        opening_receipt=receipt,
        output=output,
        batch_size=2,
    )

    payload = module.confirm(args)

    assert len(calls) == 1
    assert receipt.is_file() and output.is_file()
    assert payload["audit"]["opening_count"] == 1
    assert payload["provenance"]["implementation"]["sha256"] == _sha(
        Path(module.__file__).resolve()
    )
    assert payload["provenance"]["evaluation_implementation"]["sha256"] == _sha(
        Path(fidelity.__file__).resolve()
    )
    opening = json.loads(receipt.read_text(encoding="utf-8"))
    opening_closure = opening["implementation_closure"]
    final_closure = payload["provenance"]["implementation_closure"]
    assert opening_closure == final_closure
    assert [record["role"] for record in final_closure] == [
        role for role, _ in module.IMPLEMENTATION_CLOSURE
    ]
    for record in final_closure:
        assert record["sha256"] == _sha(Path(record["path"]))

    output.unlink()
    with pytest.raises(ValueError, match="opening receipt already exists"):
        module.confirm(args)
    assert len(calls) == 1


def test_concurrent_confirmation_claims_receipt_exclusively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    selection_path = tmp_path / "selection.json"
    diagnostic_path = tmp_path / "diagnostic.json"
    selection_path.write_text("{}", encoding="utf-8")
    diagnostic_path.write_text("{}", encoding="utf-8")
    runtime = _fake_runtime(tmp_path)
    bank = _fake_bank(tmp_path)
    receipt = tmp_path / "opened.json"
    output = tmp_path / "confirmation.json"
    replay_barrier = threading.Barrier(2)
    calls = []
    calls_lock = threading.Lock()

    monkeypatch.setattr(
        module,
        "validate_frozen_selection",
        lambda *args: ({"selected_alpha": 0.1}, {}),
    )

    def replay(*args, **kwargs):
        replay_barrier.wait(timeout=10)
        return runtime

    monkeypatch.setattr(module, "_load_query_free_runtime", replay)

    def load_after_receipt(path):
        assert path == receipt and receipt.is_file()
        with calls_lock:
            calls.append(path)
        return bank

    monkeypatch.setattr(module, "_load_audit_bank_after_receipt", load_after_receipt)
    monkeypatch.setattr(
        module,
        "_paired_gate_in_memory",
        lambda *args, **kwargs: {"decision": "promote", "protocol": {"unit_test": True}},
    )
    args = argparse.Namespace(
        selection=selection_path,
        selection_sha256="a" * 64,
        diagnostic=diagnostic_path,
        diagnostic_sha256="b" * 64,
        opening_receipt=receipt,
        output=output,
        batch_size=2,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(module.confirm, args) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=30))
            except Exception as exc:  # noqa: BLE001 - assert exact race loser below.
                outcomes.append(exc)

    assert sum(isinstance(value, dict) for value in outcomes) == 1
    losers = [value for value in outcomes if isinstance(value, Exception)]
    assert len(losers) == 1 and isinstance(losers[0], FileExistsError)
    assert len(calls) == 1
    assert receipt.is_file() and output.is_file()


def test_confirmation_canonicalizes_outputs_below_a_symlinked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    selection_path = tmp_path / "selection.json"
    diagnostic_path = tmp_path / "diagnostic.json"
    selection_path.write_text("{}", encoding="utf-8")
    diagnostic_path.write_text("{}", encoding="utf-8")
    runtime = _fake_runtime(tmp_path)
    bank = _fake_bank(tmp_path)
    real_parent = tmp_path / "pool_family"
    real_parent.mkdir()
    lexical_parent = tmp_path / "output_alias"
    lexical_parent.symlink_to(real_parent, target_is_directory=True)
    lexical_receipt = lexical_parent / "opened.json"
    lexical_output = lexical_parent / "confirmation.json"
    canonical_receipt = real_parent / "opened.json"
    canonical_output = real_parent / "confirmation.json"
    calls = []

    monkeypatch.setattr(
        module,
        "validate_frozen_selection",
        lambda *args: ({"selected_alpha": 0.1}, {}),
    )
    monkeypatch.setattr(module, "_load_query_free_runtime", lambda *args, **kwargs: runtime)

    def load_after_receipt(path):
        assert path == canonical_receipt and canonical_receipt.is_file()
        calls.append(path)
        return bank

    monkeypatch.setattr(module, "_load_audit_bank_after_receipt", load_after_receipt)
    monkeypatch.setattr(
        module,
        "_paired_gate_in_memory",
        lambda *args, **kwargs: {"decision": "promote", "protocol": {"unit_test": True}},
    )
    args = argparse.Namespace(
        selection=selection_path,
        selection_sha256="a" * 64,
        diagnostic=diagnostic_path,
        diagnostic_sha256="b" * 64,
        opening_receipt=lexical_receipt,
        output=lexical_output,
        batch_size=2,
    )

    payload = module.confirm(args)

    opening = json.loads(canonical_receipt.read_text(encoding="utf-8"))
    assert calls == [canonical_receipt]
    assert opening["intended_confirmation_output"] == str(canonical_output)
    assert payload["audit"]["opening_receipt"]["path"] == str(canonical_receipt)
    assert canonical_output.is_file()


def test_invalid_selection_cannot_reach_audit_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    selection = tmp_path / "selection"
    diagnostic = tmp_path / "diagnostic"
    selection.write_bytes(b"selection")
    diagnostic.write_bytes(b"diagnostic")
    calls = []
    monkeypatch.setattr(
        module,
        "validate_frozen_selection",
        lambda *args: (_ for _ in ()).throw(ValueError("selection invalid")),
    )
    monkeypatch.setattr(
        module,
        "_load_audit_bank_after_receipt",
        lambda *args: calls.append(True),
    )
    args = argparse.Namespace(
        selection=selection,
        selection_sha256="a" * 64,
        diagnostic=diagnostic,
        diagnostic_sha256="b" * 64,
        opening_receipt=tmp_path / "opened.json",
        output=tmp_path / "confirmation.json",
        batch_size=2,
    )

    with pytest.raises(ValueError, match="selection invalid"):
        module.confirm(args)
    assert calls == []
    assert not args.opening_receipt.exists()
