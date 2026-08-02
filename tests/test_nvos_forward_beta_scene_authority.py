from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from radio_gs.scripts import nvos_forward_beta_scene_authority as scene_authority
from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    build_authority,
    canonical_json_sha256,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _candidate_method_manifest_contract,
)
from radio_gs.scripts.nvos_forward_beta_scene_authority import (
    CANDIDATE_ID,
    CUDA_ATTESTATION_MECHANISM,
    EXPECTED_BLOCKERS,
    EXPECTED_SCORING,
    GPU_OWNER_PID_NAMESPACE_MODE,
    HOST_MEMORY_POLICY,
    MAXIMUM_CONCURRENT_SCENE_EVALUATORS,
    OWNER_AUDIT_COLUMNS,
    SCENE_GPU_ASSIGNMENT,
    SERIAL_SCENE_GPU_PLAN,
    TELEMETRY_COLUMNS,
    _derived_argv_contract,
    _gpu_identity,
    _validate_beta_cuda_attestation,
    _validate_result_report,
    finalize_scene_receipt,
    main,
    prepare_scene_command,
    validate_run_manifest,
    validate_scene_argv,
    validate_scene_receipt,
    write_forward_beta_cuda_child_attestation,
    write_scene_postcheck,
)
from radio_gs.utils.immutable_artifacts import (
    write_frozen_json,
)


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-0eac2c76-4004-49eb-bc0c-a9a30aec041a"
GPU_BUS = "00000000:82:00.0"
SCENE = "fern"


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        support_mode="canonical_support",
        region_space="sam3",
        prompt_registration_mode="raster_adjoint",
        prompt_registration_scale=1.0,
        alpha_threshold=0.0,
        depth_tolerance=0.08,
        relative_depth_tolerance=0.02,
        registered_seed_construction="joint_signed",
        registered_observation_fusion="probability_mixture",
        registered_seed_unary_weight=0.0,
        registered_observation_confidence="relative_joint_max",
        registered_observation_mass_scale=1.0,
        support_threshold=0.0,
        prototype_count=1,
        prototype_strategy="spherical_mean_fps",
        appearance_weight=1.0,
        boundary_weight=0.35,
        prototype_temperature=0.07,
        feature_calibration="none",
        background_centroids=0,
        score_calibration="none",
        negative_spatial_mode="none",
        registered_selection_mode="seeded_component",
        registered_readout_stage="propagated",
        registered_forward_unary="beta_coverage_v1",
        graph_policy="legacy",
        component_graph_policy="same",
        graph_legacy_residual=0.0,
        channel_confidence_mode="none",
        score_render_resolution="scaled_renderer",
        score_render_scale=1.0,
        valid_support_normalization=True,
        valid_support_coverage_power=0.0,
        feature_contribution_gamma=1.0,
        score_chunk_size=1024,
        solver_support_threshold=0.5,
        solver_type="diffusion",
        solver_iterations=2,
        solver_residual=0.3,
        solver_unary_temperature=0.5,
        laplacian_weight=1.0,
        cg_iterations=8,
        cg_tolerance=1e-5,
        hard_seed_threshold=0.2,
        hard_seed_conflict_policy="positive_priority",
        hard_seed_conflict_margin=0.0,
        component_edge_threshold=1e-5,
        seeded_component_min_weight=0.2,
        canonical_reliability_cache="",
        diagnostic_graph_affinity_override="",
        require_asset_hashes=True,
    )


def _manifest_payload(method: dict | None = None) -> dict:
    method = method or _candidate_method_manifest_contract(_args())
    method_sha = canonical_json_sha256(method)
    authority = build_authority(
        candidate_method_sha256=method_sha,
        scoring_contract=EXPECTED_SCORING,
        repo_root=ROOT,
    )
    return {
        "candidate": CANDIDATE_ID,
        "scenes": [
            *SCENE_GPU_ASSIGNMENT["gpu0"],
            *SCENE_GPU_ASSIGNMENT["gpu1"],
        ],
        "scene_gpu_assignment": deepcopy(SCENE_GPU_ASSIGNMENT),
        "maximum_concurrent_scene_evaluators": (
            MAXIMUM_CONCURRENT_SCENE_EVALUATORS
        ),
        "host_memory_policy": HOST_MEMORY_POLICY,
        "serial_scene_gpu_plan": deepcopy(SERIAL_SCENE_GPU_PLAN),
        "method_contract": method,
        "method_contract_sha256": method_sha,
        "registered_forward_protocol_authority": authority,
        "registered_forward_protocol_authority_sha256": (
            canonical_json_sha256(authority)
        ),
    }


def _write_manifest(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(payload or _manifest_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _argv(
    manifest_path: Path,
    result: Path,
    attestation: Path,
    *,
    physical_index: int = 1,
    gpu_uuid: str = GPU_UUID,
    gpu_bus: str = GPU_BUS,
    scene: str = SCENE,
) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = ["python", str(ROOT / "radio_gs/scripts/eval_nvos_gaussian_first.py")]
    for flag, expected in _derived_argv_contract(manifest["method_contract"]).items():
        values.extend([flag, str(expected)])
    values.extend(
        [
            "--candidate-id", CANDIDATE_ID,
            "--scene-id", scene,
            "--device", "cuda:0",
            "--expected-gpu-physical-index", str(physical_index),
            "--run-manifest", str(manifest_path),
            "--gpu-attestation-output", str(attestation),
            "--expected-gpu-uuid", gpu_uuid,
            "--expected-gpu-bus-id", gpu_bus,
            "--output-dir", str(result.parent),
            "--valid-support-normalization",
            "--require-asset-hashes",
        ]
    )
    return values


def _write_cuda_attestation(
    path: Path,
    *,
    physical_index: int,
    gpu_uuid: str,
    gpu_bus: str,
    scene: str = SCENE,
) -> None:
    write_frozen_json(
        path,
        {
            "schema_version": 1,
            "artifact_type": "nvos-forward-beta-cuda-child-attestation-v1",
            "status": "torch_cuda0_live_owner_matches_physical_gpu_uuid_and_pci",
            "scene": scene,
            "observed_epoch": 1,
            "hostname": "test-host",
            "environment": {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
                "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
            },
            "expected_gpu": {
                "physical_index": physical_index,
                "uuid": gpu_uuid,
                "pci_bus_id": gpu_bus,
            },
            "torch_cuda": {
                "visible_device_count": 1,
                "current_device": 0,
                "device": "cuda:0",
                "device_name": "RTX 3090",
                "compute_capability": [8, 6],
                "total_memory": 24 * 1024**3,
                "torch_version": "test",
                "torch_cuda_build": "11.8",
            },
            "process_namespace_pids": [777],
            "nvidia_inventory_row": [
                str(physical_index), gpu_uuid, gpu_bus, "RTX 3090"
            ],
            "nvidia_preallocation_owner_rows": [],
            "nvidia_compute_owner_rows": [[gpu_uuid, "777", "python"]],
            "owner_pid_binding": "process_namespace_pid",
            "attestation_mechanism": CUDA_ATTESTATION_MECHANISM,
        },
    )


def _write_owner_audit(path: Path, *, gpu_uuid: str) -> None:
    path.write_text(
        ",".join(OWNER_AUDIT_COLUMNS)
        + "\n"
        + f"t0,{gpu_uuid},42,,,,prelaunch_owner_clear\n"
        + f"t1,{gpu_uuid},42,777,777,,runtime_owner_audit\n"
        + f"t2,{gpu_uuid},42,,,,postexit_owner_clear\n",
        encoding="utf-8",
    )


def _write_telemetry(
    path: Path,
    *,
    physical_index: int,
    gpu_bus: str,
) -> None:
    path.write_text(
        ",".join(TELEMETRY_COLUMNS)
        + "\n"
        + (
            f"t0,{physical_index},{gpu_bus},70,250,300,90,12000,P2,"
            "prelaunch_owner_clear\n"
        )
        + (
            f"t1,{physical_index},{gpu_bus},68,10,300,0,0,P8,"
            "cuda_release_verified_no_compute_owner\n"
        ),
        encoding="utf-8",
    )


def _report_payload(
    manifest: dict,
    method_sha: str,
    authority_sha: str,
    *,
    scene: str = SCENE,
) -> dict:
    authority = manifest["registered_forward_protocol_authority"]
    return {
        "scene_id": scene,
        "registered_forward_protocol_authority": authority,
        "registered_forward_protocol_authority_sha256": authority_sha,
        "method_contract": {
            "candidate_id": CANDIDATE_ID,
            "candidate_method_contract_sha256": method_sha,
            "registered_forward_protocol_authority": authority,
            "registered_forward_protocol_authority_sha256": authority_sha,
            "shared_solver": {
                "registered_readout_stage": "propagated",
                "registered_observation_fusion": "probability_mixture",
                "registered_forward_unary": {"mode": "beta_coverage_v1"},
            },
        },
        "safety": {
            "main_result_eligible": False,
            "frozen_diagnostic_eligible": False,
            "strict_unseen_protocol_exact_match": False,
            "registered_forward_protocol_authority_sha256": authority_sha,
        },
    }


def test_manifest_binds_candidate_method_and_nonexact_protocol(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)

    validated = validate_run_manifest(path, scene=SCENE)

    assert validated["payload"]["candidate"] == CANDIDATE_ID
    assert validated["method_contract_sha256"] == validated["payload"][
        "method_contract_sha256"
    ]
    assert validated["protocol_authority"][
        "strict_unseen_protocol_exact_match"
    ] is False
    assert validated["payload"]["maximum_concurrent_scene_evaluators"] == 1
    assert validated["payload"]["host_memory_policy"] == HOST_MEMORY_POLICY
    assert validated["protocol_authority"][
        "strict_unseen_exact_match_blockers"
    ] == EXPECTED_BLOCKERS


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "maximum_concurrent_scene_evaluators",
            2,
            "scene-evaluator concurrency must be exactly one",
        ),
        ("host_memory_policy", "dual_resident", "host-memory policy differs"),
        ("serial_scene_gpu_plan", [], "serial scene GPU plan differs"),
    ],
)
def test_manifest_rejects_relaxed_single_resident_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _manifest_payload()
    payload[field] = value
    path = _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError, match=message):
        validate_run_manifest(path, scene=SCENE)


@pytest.mark.parametrize("mutation", ("swap", "missing_scene"))
def test_manifest_rejects_mutated_fixed_scene_gpu_assignment(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _manifest_payload()
    assignment = payload["scene_gpu_assignment"]
    if mutation == "swap":
        assignment["gpu0"][0], assignment["gpu1"][0] = (
            assignment["gpu1"][0],
            assignment["gpu0"][0],
        )
    else:
        assignment["gpu1"].pop()
    manifest = _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match="fixed scene GPU assignment"):
        validate_run_manifest(manifest, scene=SCENE)


@pytest.mark.parametrize(
    ("scene", "wrong_physical_index"),
    (("fern", 1), ("horns_left", 0)),
)
def test_prepare_rejects_scene_execution_on_swapped_physical_gpu(
    tmp_path: Path,
    scene: str,
    wrong_physical_index: int,
) -> None:
    manifest = _write_manifest(tmp_path)
    result = tmp_path / "result" / f"{scene}_evaluation.json"
    attestation = tmp_path / "attestation.json"
    guard = tmp_path / "guard.sh"
    guard.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=f"scene is assigned to physical GPU{1 - wrong_physical_index}",
    ):
        prepare_scene_command(
            output=tmp_path / "command.json",
            run_manifest=manifest,
            scene=scene,
            result=result,
            telemetry=tmp_path / "telemetry.csv",
            owner_audit=tmp_path / "owner.csv",
            attestation=attestation,
            postcheck=tmp_path / "postcheck.json",
            receipt=tmp_path / "receipt.json",
            evaluator_log=tmp_path / "evaluator.log",
            guard=guard,
            physical_index=wrong_physical_index,
            gpu_uuid=GPU_UUID,
            gpu_bus_id=GPU_BUS,
            command=_argv(
                manifest,
                result,
                attestation,
                physical_index=wrong_physical_index,
                scene=scene,
            ),
        )


@pytest.mark.parametrize(
    ("physical_index", "gpu_uuid", "gpu_bus"),
    [
        (-1, GPU_UUID, GPU_BUS),
        (2, GPU_UUID, GPU_BUS),
        (0, "not-a-uuid", GPU_BUS),
        (1, GPU_UUID, "not-a-bus"),
    ],
)
def test_gpu_identity_rejects_non_gpu0_gpu1_or_malformed_identity(
    physical_index: int,
    gpu_uuid: str,
    gpu_bus: str,
) -> None:
    with pytest.raises(ValueError):
        _gpu_identity(physical_index, gpu_uuid, gpu_bus)


def test_cuda_attestation_rejects_index_uuid_bus_mismatch(tmp_path: Path) -> None:
    attestation = tmp_path / "attestation.json"
    _write_cuda_attestation(
        attestation,
        physical_index=0,
        gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
        gpu_bus="00000000:01:00.0",
    )

    with pytest.raises(ValueError):
        _validate_beta_cuda_attestation(
            attestation,
            scene=SCENE,
            physical_index=1,
            gpu_uuid="GPU-11111111-1111-1111-1111-111111111111",
            gpu_bus_id="00000000:01:00.0",
        )


@pytest.mark.parametrize(
    ("physical_index", "gpu_uuid", "gpu_bus"),
    [
        (0, "GPU-11111111-1111-1111-1111-111111111111", "00000000:01:00.0"),
        (1, GPU_UUID, GPU_BUS),
    ],
)
def test_cuda_attestation_writer_supports_gpu0_and_gpu1_with_mocked_cuda(
    tmp_path: Path,
    monkeypatch,
    physical_index: int,
    gpu_uuid: str,
    gpu_bus: str,
) -> None:
    class Probe:
        def zero_(self):
            return self

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def set_device(_index):
            return None

        @staticmethod
        def synchronize(_index):
            return None

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(_index):
            return SimpleNamespace(
                name="RTX 3090",
                major=8,
                minor=6,
                total_memory=24 * 1024**3,
            )

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        empty=lambda *_args, **_kwargs: Probe(),
        uint8=object(),
        __version__="test",
        version=SimpleNamespace(cuda="11.8"),
    )
    compute_queries = 0

    def fake_nvidia_query(arguments):
        nonlocal compute_queries
        if arguments[0].startswith("--query-gpu"):
            return [[str(physical_index), gpu_uuid, gpu_bus, "RTX 3090"]]
        compute_queries += 1
        return [] if compute_queries == 1 else [[gpu_uuid, "777", "python"]]

    for key, value in {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
        "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(scene_authority, "_nvidia_query", fake_nvidia_query)
    monkeypatch.setattr(scene_authority, "_namespace_process_ids", lambda: [777])
    output = tmp_path / f"gpu{physical_index}_attestation.json"

    payload = write_forward_beta_cuda_child_attestation(
        output=output,
        scene=SCENE,
        physical_index=physical_index,
        expected_uuid=gpu_uuid,
        expected_bus_id=gpu_bus,
    )
    validated = _validate_beta_cuda_attestation(
        output,
        scene=SCENE,
        physical_index=physical_index,
        gpu_uuid=gpu_uuid,
        gpu_bus_id=gpu_bus,
    )

    assert payload["expected_gpu"]["physical_index"] == physical_index
    assert validated["payload"] == payload


@pytest.mark.parametrize(
    "mutation",
    ("candidate", "method_sha", "authority_sha", "exact", "hard_seed"),
)
def test_manifest_tampering_and_hard_seed_fusion_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    payload = _manifest_payload()
    if mutation == "candidate":
        payload["candidate"] = "registered-region-v3"
    elif mutation == "method_sha":
        payload["method_contract_sha256"] = "b" * 64
    elif mutation == "authority_sha":
        payload["registered_forward_protocol_authority_sha256"] = "b" * 64
    elif mutation == "exact":
        payload["nested_metadata"] = {"caller_exact_override": True}
    else:
        method = deepcopy(payload["method_contract"])
        method["observation_fusion"] = "hard_seed_anchored_probability"
        method["strong_unary"] = {"forged": True}
        payload = _manifest_payload(method)
    path = _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError):
        validate_run_manifest(path, scene=SCENE)


def test_actual_argv_requires_every_fixed_beta_parameter(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = tmp_path / "results" / f"{SCENE}_evaluation.json"
    attestation = tmp_path / "attestation.json"
    argv = _argv(manifest_path, result, attestation)

    assert validate_scene_argv(
        argv,
        manifest=manifest,
        run_manifest_path=manifest_path,
        scene=SCENE,
        result=result,
        attestation=attestation,
        physical_index=1,
        gpu_uuid=GPU_UUID,
        gpu_bus_id=GPU_BUS,
    ) == argv

    for flag, replacement in (
        ("--registered-forward-unary", "none"),
        ("--registered-observation-fusion", "hard_seed_anchored_probability"),
        ("--registered-readout-stage", "connected"),
        ("--alpha-threshold", "0.02"),
    ):
        forged = list(argv)
        forged[forged.index(flag) + 1] = replacement
        with pytest.raises(ValueError, match=flag):
            validate_scene_argv(
                forged,
                manifest=manifest,
                run_manifest_path=manifest_path,
                scene=SCENE,
                result=result,
                attestation=attestation,
                physical_index=1,
                gpu_uuid=GPU_UUID,
                gpu_bus_id=GPU_BUS,
            )

    with pytest.raises(ValueError, match="authority/exact override"):
        validate_scene_argv(
            [*argv, "--strict-unseen-protocol-exact-match"],
            manifest=manifest,
            run_manifest_path=manifest_path,
            scene=SCENE,
            result=result,
            attestation=attestation,
            physical_index=1,
            gpu_uuid=GPU_UUID,
            gpu_bus_id=GPU_BUS,
        )


def test_result_must_remain_nonmain_and_nonfrozen(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    validated = validate_run_manifest(manifest_path, scene=SCENE)
    report = tmp_path / "report.json"
    command = {
        "scene": SCENE,
        "method_contract_sha256": validated["method_contract_sha256"],
        "protocol_authority_sha256": validated["protocol_authority_sha256"],
    }
    payload = _report_payload(
        validated["payload"],
        validated["method_contract_sha256"],
        validated["protocol_authority_sha256"],
    )
    report.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _validate_result_report(report, command=command, manifest=validated["payload"])

    for field in ("main_result_eligible", "frozen_diagnostic_eligible"):
        forged = deepcopy(payload)
        forged["safety"][field] = True
        forged_path = tmp_path / f"forged_{field}.json"
        forged_path.write_text(json.dumps(forged) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="promotion safety"):
            _validate_result_report(
                forged_path, command=command, manifest=validated["payload"]
            )


@pytest.mark.parametrize(
    ("scene", "physical_index", "gpu_uuid", "gpu_bus"),
    [
        (
            "fern", 0,
            "GPU-11111111-1111-1111-1111-111111111111",
            "00000000:01:00.0",
        ),
        ("horns_left", 1, GPU_UUID, GPU_BUS),
    ],
)
def test_scene_receipt_binds_gpu_owner_report_and_postclear(
    tmp_path: Path,
    monkeypatch,
    scene: str,
    physical_index: int,
    gpu_uuid: str,
    gpu_bus: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    validated = validate_run_manifest(manifest_path, scene=scene)
    result = tmp_path / "result" / f"{scene}_evaluation.json"
    result.parent.mkdir()
    result.write_text(
        json.dumps(
            _report_payload(
                validated["payload"],
                validated["method_contract_sha256"],
                validated["protocol_authority_sha256"],
                scene=scene,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    attestation = tmp_path / "attestation.json"
    telemetry = tmp_path / "telemetry.csv"
    owner_audit = tmp_path / "owner.csv"
    postcheck = tmp_path / "postcheck.json"
    command = tmp_path / "command.json"
    receipt = tmp_path / "receipt.json"
    evaluator_log = tmp_path / "evaluator.log"
    guard = tmp_path / "thermal_guard.sh"
    _write_cuda_attestation(
        attestation,
        physical_index=physical_index,
        gpu_uuid=gpu_uuid,
        gpu_bus=gpu_bus,
        scene=scene,
    )
    _write_telemetry(
        telemetry,
        physical_index=physical_index,
        gpu_bus=gpu_bus,
    )
    _write_owner_audit(owner_audit, gpu_uuid=gpu_uuid)
    evaluator_log.write_text("evaluator exited zero\n", encoding="utf-8")
    guard.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    evaluator_argv = _argv(
        manifest_path,
        result,
        attestation,
        physical_index=physical_index,
        gpu_uuid=gpu_uuid,
        gpu_bus=gpu_bus,
        scene=scene,
    )
    assert main(
        [
            "prepare-scene",
            "--output", str(command),
            "--run-manifest", str(manifest_path),
            "--scene", scene,
            "--result", str(result),
            "--telemetry", str(telemetry),
            "--owner-audit", str(owner_audit),
            "--attestation", str(attestation),
            "--postcheck", str(postcheck),
            "--receipt", str(receipt),
            "--evaluator-log", str(evaluator_log),
            "--guard", str(guard),
            "--physical-index", str(physical_index),
            "--gpu-uuid", gpu_uuid,
            "--gpu-bus-id", gpu_bus,
            "--",
            *evaluator_argv,
        ]
    ) == 0
    monkeypatch.setattr(
        scene_authority,
        "_nvidia_query",
        lambda arguments: (
            [[str(physical_index), gpu_uuid, gpu_bus]]
            if arguments[0].startswith("--query-gpu")
            else []
        ),
    )
    assert main(
        [
            "postcheck-scene",
            "--output", str(postcheck),
            "--command-record", str(command),
        ]
    ) == 0
    assert main(
        [
            "finalize-scene",
            "--output", str(receipt),
            "--command-record", str(command),
            "--postcheck", str(postcheck),
        ]
    ) == 0
    assert main(
        [
            "validate-scene",
            "--receipt", str(receipt),
            "--run-manifest", str(manifest_path),
            "--scene", scene,
            "--result", str(result),
        ]
    ) == 0

    finalized = validate_scene_receipt(
        receipt,
        run_manifest=manifest_path,
        scene=scene,
        result=result,
    )
    validated_receipt = validate_scene_receipt(
        receipt,
        run_manifest=manifest_path,
        scene=scene,
        result=result,
    )

    assert finalized["payload"]["promotion"] == {
        "main_result_eligible": False,
        "frozen_diagnostic_eligible": False,
        "strict_unseen_protocol_exact_match": False,
    }
    assert validated_receipt["payload"]["gpu_identity"] == {
        "physical_index": physical_index,
        "uuid": gpu_uuid,
        "pci_bus_id": gpu_bus,
    }
    assert validated_receipt["payload"]["owner_audit"][
        "child_owner_pids"
    ] == ["777"]
    assert validated_receipt["payload"]["telemetry"]["sample_count"] == 2
    assert validated_receipt["payload"]["guard"]["path"] == str(guard.resolve())
    assert validated_receipt["payload"]["evaluator_log"]["path"] == str(
        evaluator_log.resolve()
    )

    result.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_scene_receipt(
            receipt,
            run_manifest=manifest_path,
            scene=scene,
            result=result,
        )
