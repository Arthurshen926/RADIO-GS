from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from radio_gs.candidate_authority import (
    build_candidate_authority,
    reference_candidate_authority_inputs,
)
from radio_gs.runtime_compliance import (
    RUNTIME_COMPLIANCE_AUDIT_SCHEMA,
    RuntimeComplianceVerifier,
    activity_record,
    evidence_node,
    lineage_edge,
    load_runtime_compliance_proof,
    row_identity,
    validate_runtime_compliance_proof,
    write_runtime_compliance_proof,
)
from radio_gs.stage_receipts import (
    STAGE_ORDER,
    StageReceiptChain,
    canonical_manifest,
    directory_merkle,
    opaque_file,
    prediction_inventory,
    write_stage_receipt,
)


_ENVIRONMENT = {
    "runtime": "python3.9",
    "container_or_environment": "radio-gs-ci",
    "dependency_lock_sha256": "1" * 64,
    "kernel": "fixture-kernel-v1",
    "gpu": "cpu-fixture",
    "driver_cuda": "not_applicable",
    "deterministic_flags": "fixture-deterministic-v1",
    "thread_settings": "single-threaded",
    "locale": "C.UTF-8",
    "timezone": "UTC",
    "environment_variable_allowlist": "fixture-env-v1",
}


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stage_context(tmp_path: Path, stage: str, index: int) -> dict:
    code_root = tmp_path / "code"
    _write(code_root / "fixture.py", b"pass\n")
    source = _write(tmp_path / "inputs" / f"{stage}.bin", stage.encode())
    output = _write(tmp_path / "outputs" / f"{stage}.bin", stage.encode())
    return {
        "stage": stage,
        "stage_contract": {
            "identity": f"{stage}-contract-v1",
            "stage_index": index,
            "scope": "synthetic-runtime-compliance-row",
        },
        "inputs": {"source": opaque_file(source)},
        "outputs": {"result": opaque_file(output)},
        "code_identity": {
            "repository": "Arthurshen926/RADIO-GS",
            "commit": "candidate-authority-fixture",
            "code_tree": directory_merkle(code_root),
            "dirty_patch_sha256": "0" * 64,
        },
        "configuration": {"stage": stage, "index": index},
        "command": ["radio-gs-runtime-fixture", stage],
        "dependency_container": {
            "container": "radio-gs-ci",
            "lock_sha256": "1" * 64,
        },
        "seeds": {"stochastic": [0], "deterministic": "not_applicable"},
        "determinism": {"policy": "fixture-deterministic-v1", "verified": True},
        "environment": _ENVIRONMENT,
        "runtime_trace": {
            "trace_schema": "runtime-compliance-trace-v1",
            "trace_id": f"trace-{stage}",
            "complete": True,
        },
        "private_evidence": {"targets_opened": False, "metrics_computed": False},
    }


def _seal_receipts(tmp_path: Path, candidate) -> list[dict]:
    chain = StageReceiptChain(candidate)
    receipts: list[dict] = []
    for index, stage in enumerate(STAGE_ORDER):
        context = _stage_context(tmp_path, stage, index)
        if index > 0 and stage != "evaluation":
            context["inputs"] = {
                "predecessor_result": chain.receipts[-1]["outputs"]["result"]
            }
        if stage == "warm_cache_compilation":
            context["outputs"] = {
                "result": opaque_file(
                    _write(tmp_path / "outputs" / "warm.bin", b"warm")
                )
            }
        elif stage == "query_prediction_sealing":
            query_workspace = _write(
                tmp_path / "outputs" / "query-workspace.bin", b"query"
            )
            prediction_root = tmp_path / "predictions"
            _write(prediction_root / "scene-0__query-0.json", b'{"score":0.5}\n')
            inventory = prediction_inventory(prediction_root, ["scene-0__query-0.json"])
            context["inputs"] = {
                "predecessor_result": chain.receipts[-1]["outputs"]["result"]
            }
            context["outputs"] = {
                "query_workspace": opaque_file(query_workspace),
                "prediction_inventory": inventory,
            }
        elif stage == "evaluation":
            target = _write(tmp_path / "private" / "ground-truth.bin", b"target")
            metrics_payload = {"foreground_iou": 0.5}
            metrics_file = _write(
                tmp_path / "outputs" / "metrics.bin",
                json.dumps(
                    metrics_payload, sort_keys=True, separators=(",", ":")
                ).encode(),
            )
            context["inputs"] = {
                "prediction_inventory": chain.receipts[-1]["prediction_inventory"],
                "target": opaque_file(target),
            }
            context["outputs"] = {
                "metrics": canonical_manifest(metrics_payload),
                "metrics_payload": opaque_file(metrics_file),
            }
            context["private_evidence"] = {
                "targets_opened": True,
                "metrics_computed": True,
            }
        receipt = chain.seal_stage(**context)
        path = tmp_path / "receipts" / f"{index:02d}-{stage}.json"
        write_stage_receipt(path, receipt)
        receipts.append(receipt.as_dict())
    return receipts


def _node(
    tmp_path: Path,
    name: str,
    *,
    content_type: str,
    lifecycle_class: str,
    stage: str,
    row_id: str,
    contract_id: str,
    information_grant: str,
    query_id: str | None = None,
    producer_receipt: dict | None = None,
    members: list[dict] | None = None,
    content_sha256: str | None = None,
    content_size_bytes: int | None = None,
    content_kind: str = "opaque_file",
    locator: str | None = None,
) -> dict:
    path = (
        Path(locator)
        if locator is not None
        else _write(tmp_path / "assets" / name, name.encode())
    )
    if content_sha256 is None:
        content_sha256 = _digest(path.read_bytes())
    if content_size_bytes is None:
        content_size_bytes = path.stat().st_size
    return evidence_node(
        content_type=content_type,
        lifecycle_class=lifecycle_class,
        content_sha256=content_sha256,
        size_bytes=content_size_bytes,
        schema_identity=f"fixture/{content_type}/v1",
        locator=str(path),
        information_grant=information_grant,
        stage=stage,
        row_id=row_id,
        contract_id=contract_id,
        query_id=query_id,
        producer_receipt=producer_receipt,
        logical_bytes=sum(int(member["logical_bytes"]) for member in (members or [])),
        members=members or [],
        content_kind=content_kind,
    )


def _build_audit(tmp_path: Path) -> tuple[dict, object]:
    candidate = build_candidate_authority(**reference_candidate_authority_inputs())
    contract_id = "lerf2d-field-only-four-scene-v1"
    execution_id = "scene-0/query-0"
    row_id = row_identity(contract_id, [execution_id])
    receipts = _seal_receipts(tmp_path, candidate)
    (
        mapping_receipt,
        deployment_receipt,
        warm_receipt,
        query_receipt,
        evaluation_receipt,
    ) = receipts

    member_digest = _digest(b"canonical-capability-feature")
    members = [
        {
            "name": "canonical_capability_feature",
            "content_type": "canonical_capability_feature",
            "dtype": "float16",
            "shape": [100_000, 512],
            "logical_bytes": 100_000 * 512 * 2,
            "sha256": member_digest,
        },
        {
            "name": "conventional_geometry",
            "content_type": "conventional_rendering_state",
            "dtype": "float32",
            "shape": [100_000, 3],
            "logical_bytes": 100_000 * 3 * 4,
            "sha256": _digest(b"conventional-geometry"),
        },
    ]

    nodes: list[dict] = []
    method = _node(
        tmp_path,
        "method-contract.bin",
        content_type="method_contract",
        lifecycle_class="method_contract",
        stage="mapping_training",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="method_contract",
    )
    evaluation_contract = _node(
        tmp_path,
        "evaluation-contract.bin",
        content_type="evaluation_contract",
        lifecycle_class="evaluation_contract",
        stage="evaluation",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="evaluation_contract",
    )
    mapping_observation = _node(
        tmp_path,
        "mapping-observation.bin",
        content_type="mapping_observation",
        lifecycle_class="mapping_observation",
        stage="mapping_training",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="mapping_observation",
    )
    global_parameters = _node(
        tmp_path,
        "global-parameters.bin",
        content_type="global_method_parameters",
        lifecycle_class="global_method_parameters",
        stage="mapping_training",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="global_method_parameters",
    )
    deployment = _node(
        tmp_path,
        "deployment-state.bin",
        content_type="deployment_scene_state",
        lifecycle_class="deployment_scene_state",
        stage="deployment_sealing",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="deployment_scene_state",
        producer_receipt={
            "stage": "deployment_sealing",
            "receipt_id": deployment_receipt["receipt_id"],
        },
        members=members,
    )
    warm = _node(
        tmp_path,
        "warm-cache.bin",
        content_type="warm_cache",
        lifecycle_class="rebuildable_warm_cache",
        stage="warm_cache_compilation",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="warm_cache",
        producer_receipt={
            "stage": "warm_cache_compilation",
            "receipt_id": warm_receipt["receipt_id"],
        },
    )
    query_input = _node(
        tmp_path,
        "authorized-query.bin",
        content_type="authorized_query_input",
        lifecycle_class="query_workspace_input",
        stage="query_prediction_sealing",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="authorized_query_input",
        query_id="query-0",
    )
    output_metadata = _node(
        tmp_path,
        "output-request-metadata.bin",
        content_type="output_request_metadata",
        lifecycle_class="query_workspace_input",
        stage="query_prediction_sealing",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="output_request_metadata",
        query_id="query-0",
    )
    query_workspace = _node(
        tmp_path,
        "query-workspace.bin",
        content_type="query_workspace",
        lifecycle_class="query_workspace",
        stage="query_prediction_sealing",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="query_workspace",
        query_id="query-0",
        producer_receipt={
            "stage": "query_prediction_sealing",
            "receipt_id": query_receipt["receipt_id"],
        },
    )
    prediction = _node(
        tmp_path,
        "prediction-inventory.bin",
        content_type="prediction_inventory",
        lifecycle_class="prediction",
        stage="query_prediction_sealing",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="prediction",
        query_id="query-0",
        producer_receipt={
            "stage": "query_prediction_sealing",
            "receipt_id": query_receipt["receipt_id"],
        },
        content_sha256=query_receipt["prediction_inventory"]["merkle_root_sha256"],
        content_size_bytes=sum(
            entry["size_bytes"]
            for entry in query_receipt["prediction_inventory"]["directory"]["entries"]
        ),
        content_kind="directory_merkle",
        locator=str(tmp_path / "predictions"),
    )
    target = _node(
        tmp_path,
        "ground-truth.bin",
        content_type="ground_truth",
        lifecycle_class="evaluator_private_target",
        stage="evaluation",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="evaluator_private_target",
        query_id="query-0",
    )
    evaluator_result = _node(
        tmp_path,
        "evaluator-result.bin",
        content_type="evaluator_result",
        lifecycle_class="evaluator_result",
        stage="evaluation",
        row_id=row_id,
        contract_id=contract_id,
        information_grant="evaluator_result",
        query_id="query-0",
        producer_receipt={
            "stage": "evaluation",
            "receipt_id": evaluation_receipt["receipt_id"],
        },
        content_sha256=_digest((tmp_path / "outputs" / "metrics.bin").read_bytes()),
        content_size_bytes=(tmp_path / "outputs" / "metrics.bin").stat().st_size,
        locator=str(tmp_path / "outputs" / "metrics.bin"),
    )
    nodes.extend(
        [
            method,
            evaluation_contract,
            mapping_observation,
            global_parameters,
            deployment,
            warm,
            query_input,
            output_metadata,
            query_workspace,
            prediction,
            target,
            evaluator_result,
        ]
    )

    edges = [
        lineage_edge(
            source=method["node_id"],
            target=deployment["node_id"],
            edge_type="derive",
            stage="deployment_sealing",
            purpose="bind method contract to deployment schema",
            information_grant="method_contract",
        ),
        lineage_edge(
            source=mapping_observation["node_id"],
            target=deployment["node_id"],
            edge_type="derive",
            stage="deployment_sealing",
            purpose="construct deployment scene state",
            information_grant="mapping_observation",
        ),
        lineage_edge(
            source=global_parameters["node_id"],
            target=deployment["node_id"],
            edge_type="read",
            stage="deployment_sealing",
            purpose="read frozen global parameters",
            information_grant="global_method_parameters",
        ),
        lineage_edge(
            source=deployment["node_id"],
            target=warm["node_id"],
            edge_type="derive",
            stage="warm_cache_compilation",
            purpose="compile field-derived warm cache",
            information_grant="deployment_scene_state",
        ),
        lineage_edge(
            source=warm["node_id"],
            target=query_workspace["node_id"],
            edge_type="read",
            stage="query_prediction_sealing",
            purpose="read rebuilt warm cache",
            information_grant="warm_cache",
        ),
        lineage_edge(
            source=query_input["node_id"],
            target=query_workspace["node_id"],
            edge_type="read",
            stage="query_prediction_sealing",
            purpose="read authorized query input",
            information_grant="authorized_query_input",
        ),
        lineage_edge(
            source=output_metadata["node_id"],
            target=query_workspace["node_id"],
            edge_type="read",
            stage="query_prediction_sealing",
            purpose="read output placement metadata",
            information_grant="output_request_metadata",
        ),
        lineage_edge(
            source=query_workspace["node_id"],
            target=prediction["node_id"],
            edge_type="emit",
            stage="query_prediction_sealing",
            purpose="seal complete prediction inventory",
            information_grant="query_workspace",
        ),
        lineage_edge(
            source=evaluation_contract["node_id"],
            target=evaluator_result["node_id"],
            edge_type="evaluate",
            stage="evaluation",
            purpose="apply inert evaluation adapter",
            information_grant="evaluation_contract",
        ),
        lineage_edge(
            source=prediction["node_id"],
            target=evaluator_result["node_id"],
            edge_type="evaluate",
            stage="evaluation",
            purpose="evaluate sealed predictions",
            information_grant="prediction",
        ),
        lineage_edge(
            source=target["node_id"],
            target=evaluator_result["node_id"],
            edge_type="evaluate",
            stage="evaluation",
            purpose="read evaluator-private target after seal",
            information_grant="evaluator_private_target",
        ),
    ]

    root_ids = [node["node_id"] for node in nodes if node["root"]]
    process_ids = {stage: f"process-{stage}" for stage in STAGE_ORDER}
    process_tree = [
        {
            "process_id": process_ids[stage],
            "parent_id": None,
            "stage": stage,
            "entrypoint_node_id": method["node_id"],
            "children_complete": True,
        }
        for stage in STAGE_ORDER
    ]
    activities: list[dict] = []
    for stage in STAGE_ORDER:
        activities.append(
            activity_record(
                kind="executable",
                stage=stage,
                process_id=process_ids[stage],
                operation="execute",
                node_id=method["node_id"],
                identity_sha256=method["content_identity"]["sha256"],
            )
        )
    for node in nodes:
        activity_stage = node["stage"]
        process_id = process_ids[activity_stage]
        activities.append(
            activity_record(
                kind="file",
                stage=activity_stage,
                process_id=process_id,
                operation="write" if not node["root"] else "read",
                node_id=node["node_id"],
                identity_sha256=node["content_identity"]["sha256"],
            )
        )
    activities = sorted(activities, key=lambda item: json.dumps(item, sort_keys=True))

    row = {
        "row_id": row_id,
        "contract_id": contract_id,
        "required_execution_ids": [execution_id],
        "execution_children": [
            {
                "execution_id": execution_id,
                "scene_id": "scene-0",
                "query_id": "query-0",
                "status": "succeeded",
                "complete": True,
                "stage_receipt_id": query_receipt["receipt_id"],
                "prediction_node_ids": [prediction["node_id"]],
                "evaluator_result_node_id": evaluator_result["node_id"],
            }
        ],
        "prediction_node_id": prediction["node_id"],
        "evaluator_result_node_id": evaluator_result["node_id"],
    }
    storage = {
        "schema": "radio_gs.storage_assertion.v1",
        "field_family": "sidecar_free_d512_l512_single_feature",
        "local_code_dimension": 512,
        "persistent_semantic_fields": 1,
        "deployment_support_state": {"validity_bits": 1, "quality_scalars": 5},
        "scene_gaussian_count": 100_000,
        "persistent_scene_storage_increment_bytes": 100_000_000,
        "scene_soft_target_bytes": 2048 * 100_000 + 8 * 1024 * 1024,
        "scene_hard_limit_bytes": min(
            2304 * 100_000 + 16 * 1024 * 1024,
            2560 * 100_000 - 1,
        ),
        "method_specific_global_bytes": 1024,
        "method_specific_global_soft_target_bytes": 8 * 1024 * 1024,
        "method_specific_global_hard_limit_bytes": 128 * 1024 * 1024,
        "serialization_overhead_bytes": 0,
        "forbidden_member_types": [],
        "cold_start_query_executed": True,
        "warm_cache_rebuilds_bitwise_identical": True,
    }
    mounts = [
        {"path": f"/fixture/mount/{index}", "mode": "ro", "node_id": node_id}
        for index, node_id in enumerate(root_ids)
    ]
    audit = {
        "schema": RUNTIME_COMPLIANCE_AUDIT_SCHEMA,
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "contract_id": contract_id,
        "producer_identity": "synthetic-lifecycle-producer-v1",
        "row": row,
        "stage_receipts": receipts,
        "lineage": {"nodes": nodes, "edges": edges},
        "execution": {
            "schema": "radio_gs.runtime_observation.v1",
            "schema_version": 1,
            "auditor_started_before_first_instruction": True,
            "process_tree_complete": True,
            "network_disabled": True,
            "network_attempts": [],
            "inherited_descriptors_cleared": True,
            "shared_memory_cleared": True,
            "trace_channels": {
                "file": True,
                "metadata": True,
                "mmap": True,
                "executable": True,
                "library": True,
                "model": True,
                "descendant": True,
                "ipc": True,
                "shared_memory": True,
                "inherited_descriptor": True,
                "network": True,
            },
            "unknown_activity": [],
            "allowlist_mounts": mounts,
            "stage_workspaces": [
                {
                    "stage": stage,
                    "path": f"/fixture/workspace/{stage}",
                    "empty_at_start": True,
                    "empty_at_end": True,
                    "caches_empty": True,
                }
                for stage in STAGE_ORDER
            ],
            "process_tree": process_tree,
            "stage_executions": [
                {
                    "stage": stage,
                    "receipt_id": receipts[index]["receipt_id"],
                    "process_ids": [process_ids[stage]],
                    "complete": True,
                }
                for index, stage in enumerate(STAGE_ORDER)
            ],
            "declared_activity": activities,
            "observed_activity": copy.deepcopy(activities),
        },
        "environment": {
            "declared": _ENVIRONMENT,
            "observed": copy.deepcopy(_ENVIRONMENT),
            "identity_sha256": _digest(
                json.dumps(_ENVIRONMENT, sort_keys=True, separators=(",", ":")).encode()
            ),
        },
        "storage": storage,
    }
    return audit, candidate


def test_positive_row_returns_an_immutable_runtime_compliance_proof(
    tmp_path: Path,
) -> None:
    audit, candidate = _build_audit(tmp_path)

    report = RuntimeComplianceVerifier().verify(audit, candidate)

    assert report["status"] == "PASS"
    proof = report["proof"]
    assert proof["schema"] == "runtime-compliance-proof-v1"
    assert proof["candidate_id"] == candidate["candidate_id"]
    assert proof["row_id"] == audit["row"]["row_id"]
    assert len(proof["stage_receipt_ids"]) == 5
    assert len(proof["proof_id"]) == 64
    with pytest.raises(TypeError):
        proof["status"] = "FAIL"


def test_runtime_compliance_proof_is_persisted_without_clobbering(
    tmp_path: Path,
) -> None:
    audit, candidate = _build_audit(tmp_path)
    report = RuntimeComplianceVerifier().verify(audit, candidate)
    proof = report["proof"]
    path = tmp_path / "proof.json"

    assert write_runtime_compliance_proof(path, proof) == path
    assert load_runtime_compliance_proof(path).as_dict() == proof.as_dict()
    assert validate_runtime_compliance_proof(proof).as_dict() == proof.as_dict()

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["contract_id"] = "scannet-ovs-paper8-v1"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="content identity|existing frozen"):
        load_runtime_compliance_proof(path)


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        (
            lambda audit: audit["row"].update({"execution_children": []}),
            "INCOMPLETE",
        ),
        (
            lambda audit: audit["execution"]["observed_activity"].append(
                activity_record(
                    kind="ipc",
                    stage="query_prediction_sealing",
                    process_id="process-query_prediction_sealing",
                    operation="send",
                    node_id=audit["lineage"]["nodes"][0]["node_id"],
                    identity_sha256=audit["lineage"]["nodes"][0]["content_identity"][
                        "sha256"
                    ],
                )
            ),
            "FAIL",
        ),
        (
            lambda audit: audit["execution"].update(
                {"network_attempts": [{"socket": 1}]}
            ),
            "FAIL",
        ),
        (
            lambda audit: audit["environment"]["observed"].update(
                {"kernel": "drifted-kernel"}
            ),
            "FAIL",
        ),
        (
            lambda audit: audit["execution"]["trace_channels"].update(
                {"metadata": False}
            ),
            "INCOMPLETE",
        ),
        (
            lambda audit: audit["lineage"]["nodes"][4]["members"].append(
                {
                    "name": "renamed-capability-bank",
                    "content_type": "capability_bank",
                    "dtype": "float16",
                    "shape": [2, 2],
                    "logical_bytes": 8,
                    "sha256": "2" * 64,
                }
            ),
            "FAIL",
        ),
        (
            lambda audit: audit["execution"].update(
                {"auditor_started_before_first_instruction": False}
            ),
            "FAIL",
        ),
    ],
)
def test_runtime_compliance_fail_closed_mutations(
    tmp_path: Path, mutation, expected_status: str
) -> None:
    audit, candidate = _build_audit(tmp_path)
    mutation(audit)

    report = RuntimeComplianceVerifier().verify(audit, candidate)

    assert report["status"] == expected_status
    assert report["failures"]


def test_renamed_ground_truth_before_prediction_seal_is_rejected(
    tmp_path: Path,
) -> None:
    audit, candidate = _build_audit(tmp_path)
    target = next(
        node
        for node in audit["lineage"]["nodes"]
        if node["content_type"] == "ground_truth"
    )
    target["stage"] = "query_prediction_sealing"
    target["locator"] = str(tmp_path / "assets" / "weights.bin")

    report = RuntimeComplianceVerifier().verify(audit, candidate)

    assert report["status"] == "FAIL"
    assert any(
        "ground truth" in failure or "target" in failure
        for failure in report["failures"]
    )


@pytest.mark.parametrize(
    "node_index",
    [4, 5, 8],
)
def test_forbidden_lineage_artifacts_are_rejected_by_type_not_filename(
    tmp_path: Path, node_index: int
) -> None:
    audit, candidate = _build_audit(tmp_path)
    forbidden_type = {
        4: "teacher_mpr_artifact",
        5: "query_score_cache",
        8: "capability_bank",
    }[node_index]
    audit["lineage"]["nodes"][node_index]["content_type"] = forbidden_type

    report = RuntimeComplianceVerifier().verify(audit, candidate)

    assert report["status"] == "FAIL"
    assert report["failures"]
