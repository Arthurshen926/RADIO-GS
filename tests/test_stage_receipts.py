from __future__ import annotations

import json
from pathlib import Path

import pytest

from radio_gs.candidate_authority import (
    build_candidate_authority,
    reference_candidate_authority_inputs,
)
from radio_gs.stage_receipts import (
    STAGE_ORDER,
    StageReceiptChain,
    StageReceiptError,
    canonical_manifest,
    directory_merkle,
    load_stage_receipt,
    opaque_file,
    prediction_inventory,
    tensor_container,
    validate_receipt_chain,
    write_stage_receipt,
)


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _stage_context(tmp_path: Path, stage: str, index: int) -> dict:
    code_root = tmp_path / "code"
    _write(code_root / "module.py", b"pass\n")
    source = _write(tmp_path / "inputs" / f"{stage}.bin", stage.encode())
    output = _write(tmp_path / "outputs" / f"{stage}.bin", stage.encode())
    return {
        "stage": stage,
        "stage_contract": {
            "identity": f"{stage}-contract-v1",
            "stage_index": index,
            "scope": "synthetic-candidate-lifecycle",
        },
        "inputs": {"source": opaque_file(source)},
        "outputs": {"result": opaque_file(output)},
        "code_identity": {
            "repository": "Arthurshen926/RADIO-GS",
            "commit": "fixture-commit",
            "code_tree": directory_merkle(code_root),
            "dirty_patch_sha256": "0" * 64,
        },
        "configuration": {"stage": stage, "index": index},
        "command": ["radio-gs-fixture", stage],
        "dependency_container": {
            "container": "radio-gs-ci-fixture-v1",
            "lock_sha256": "1" * 64,
        },
        "seeds": {"stochastic": [0], "deterministic": "not_applicable"},
        "determinism": {"policy": "fixture-deterministic-v1", "verified": True},
        "environment": {"runtime": "python-fixture", "device": "cpu"},
        "runtime_trace": {"trace_id": f"trace-{stage}", "complete": True},
        "private_evidence": {"targets_opened": False, "metrics_computed": False},
    }


def _seal_positive_chain(tmp_path: Path) -> tuple[StageReceiptChain, list[Path]]:
    chain = StageReceiptChain(
        build_candidate_authority(**reference_candidate_authority_inputs())
    )
    paths: list[Path] = []
    for index, stage in enumerate(STAGE_ORDER):
        context = _stage_context(tmp_path, stage, index)
        if index > 0 and stage != "evaluation":
            context["inputs"] = {
                "predecessor_result": chain.receipts[-1]["outputs"]["result"]
            }
        if stage == "query_prediction_sealing":
            prediction_root = tmp_path / "predictions"
            _write(prediction_root / "query-0.json", b"{\"score\": 0.5}\n")
            _write(prediction_root / "query-1.json", b"{\"score\": 0.75}\n")
            inventory = prediction_inventory(
                prediction_root, ["query-0.json", "query-1.json"]
            )
            context["outputs"] = {"prediction_inventory": inventory}
        elif stage == "evaluation":
            target = _write(tmp_path / "private" / "target.bin", b"private-target")
            context["inputs"] = {
                "prediction_inventory": chain.receipts[-1]["prediction_inventory"],
                "target": opaque_file(target),
            }
            context["outputs"] = {
                "metrics": canonical_manifest({"foreground_iou": 0.75})
            }
            context["private_evidence"] = {
                "targets_opened": True,
                "metrics_computed": True,
            }
        receipt = chain.seal_stage(**context)
        path = tmp_path / "receipts" / f"{index:02d}-{stage}.json"
        write_stage_receipt(path, receipt)
        paths.append(path)
    return chain, paths


def test_positive_fixture_seals_all_five_stages_and_prediction_root(
    tmp_path: Path,
) -> None:
    chain, paths = _seal_positive_chain(tmp_path)

    assert [receipt["stage"] for receipt in chain.receipts] == list(STAGE_ORDER)
    query = chain.receipts[3]
    assert query["prediction_inventory"]["complete"] is True
    assert len(query["prediction_inventory"]["prediction_ids"]) == 2
    assert query["private_evidence"] == {
        "targets_opened": False,
        "metrics_computed": False,
    }
    authority = build_candidate_authority(**reference_candidate_authority_inputs())
    assert validate_receipt_chain(paths, authority) == [
        receipt.as_dict() for receipt in chain.receipts
    ]
    assert load_stage_receipt(paths[0])["stage"] == "mapping_training"


def test_opaque_directory_manifest_and_tensor_identities_are_content_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "directory"
    _write(root / "b.bin", b"b")
    _write(root / "a.bin", b"a")
    directory = directory_merkle(root)
    assert [entry["path"] for entry in directory["entries"]] == ["a.bin", "b.bin"]

    tensor_path = _write(tmp_path / "tensor.pt", b"container")
    tensor = tensor_container(
        tensor_path,
        {
            "feature": {
                "dtype": "float32",
                "shape": [2, 2],
                "sha256": "2" * 64,
            }
        },
    )
    assert tensor["container"]["size_bytes"] == len(b"container")
    assert tensor["members"][0]["name"] == "feature"
    assert canonical_manifest({"b": 2, "a": 1})["sha256"] == canonical_manifest(
        {"a": 1, "b": 2}
    )["sha256"]


def test_missing_predecessor_and_mixed_stage_output_fail_closed(tmp_path: Path) -> None:
    chain = StageReceiptChain(
        build_candidate_authority(**reference_candidate_authority_inputs())
    )
    context = _stage_context(tmp_path, "deployment_sealing", 1)

    with pytest.raises(StageReceiptError, match="predecessor"):
        chain.seal_stage(**context)

    first = _stage_context(tmp_path, "mapping_training", 0)
    chain.seal_stage(**first)
    second = _stage_context(tmp_path, "deployment_sealing", 1)
    with pytest.raises(StageReceiptError, match="sealed by the predecessor"):
        chain.seal_stage(**second)

    second = _stage_context(tmp_path, "deployment_sealing", 1)
    wrong_stage = dict(chain.receipts[0]["outputs"]["result"])
    wrong_stage["producer_stage"] = "warm_cache_compilation"
    second["inputs"] = {"wrong_stage": wrong_stage}
    with pytest.raises(StageReceiptError, match="producer stage"):
        chain.seal_stage(**second)


def test_input_output_and_environment_drift_invalidates_receipt_and_descendants(
    tmp_path: Path,
) -> None:
    _chain, paths = _seal_positive_chain(tmp_path)
    original = json.loads(paths[0].read_text(encoding="utf-8"))

    _write(tmp_path / "inputs" / "mapping_training.bin", b"changed")
    with pytest.raises(StageReceiptError, match="artifact|digest|size"):
        validate_receipt_chain(
            paths,
            build_candidate_authority(**reference_candidate_authority_inputs()),
        )

    paths[0].write_text(json.dumps({**original, "stage_order": ["wrong"]}), encoding="utf-8")
    with pytest.raises(StageReceiptError, match="receipt|stage order"):
        load_stage_receipt(paths[0])


def test_prediction_inventory_and_environment_drift_fail_closed(
    tmp_path: Path,
) -> None:
    _chain, paths = _seal_positive_chain(tmp_path)

    _write(tmp_path / "predictions" / "query-2.json", b"late-prediction")
    with pytest.raises(StageReceiptError, match="Merkle|prediction|digest"):
        validate_receipt_chain(
            paths,
            build_candidate_authority(**reference_candidate_authority_inputs()),
        )

    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["execution"]["environment"]["value"]["device"] = "gpu"
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StageReceiptError, match="environment|digest|receipt"):
        load_stage_receipt(paths[0])


def test_late_prediction_generation_and_premature_target_access_are_rejected(
    tmp_path: Path,
) -> None:
    chain = StageReceiptChain(
        build_candidate_authority(**reference_candidate_authority_inputs())
    )
    for index, stage in enumerate(STAGE_ORDER[:3]):
        context = _stage_context(tmp_path, stage, index)
        if index > 0:
            context["inputs"] = {
                "predecessor_result": chain.receipts[-1]["outputs"]["result"]
            }
        chain.seal_stage(**context)

    query = _stage_context(tmp_path, "query_prediction_sealing", 3)
    query["inputs"] = {
        "predecessor_result": chain.receipts[-1]["outputs"]["result"]
    }
    query["outputs"] = {"result": opaque_file(_write(tmp_path / "late.bin", b"x"))}
    with pytest.raises(StageReceiptError, match="prediction inventory"):
        chain.seal_stage(**query)

    query = _stage_context(tmp_path, "query_prediction_sealing", 3)
    query["inputs"] = {
        "predecessor_result": chain.receipts[-1]["outputs"]["result"]
    }
    prediction_root = tmp_path / "predictions"
    _write(prediction_root / "query.json", b"prediction")
    query["outputs"] = {
        "prediction_inventory": prediction_inventory(prediction_root, ["query.json"])
    }
    query["private_evidence"] = {"targets_opened": True, "metrics_computed": False}
    with pytest.raises(StageReceiptError, match="private evidence"):
        chain.seal_stage(**query)

    query["private_evidence"] = {"targets_opened": False, "metrics_computed": False}
    chain.seal_stage(**query)
    evaluation = _stage_context(tmp_path, "evaluation", 4)
    evaluation["inputs"] = {"prediction_inventory": chain.receipts[-1]["prediction_inventory"]}
    evaluation["outputs"] = {"metrics": canonical_manifest({"metric": 1.0})}
    evaluation["private_evidence"] = {"targets_opened": True, "metrics_computed": True}
    chain.seal_stage(**evaluation)

    assert len(chain.receipts) == len(STAGE_ORDER)
