from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import textwrap

import pytest
import torch
import torch.nn.functional as F

from radio_gs.scripts import materialize_lerf_o1_o2_streaming as o1o2
from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as agreement_v2,
)
from radio_gs.scripts import (
    materialize_lerf_reliability_conditioned_candidate_scores as candidate,
)
from radio_gs.scripts import (
    select_lerf_source_only_global_reliability_ceiling as selector,
)
from radio_gs.scripts.materialize_lerf_teacher_view_oracle_matrix import (
    geodesic_project,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    write_frozen_json,
)


def _preregistration(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "selector_preregistration.json"
    write_frozen_json(
        path,
        {
            "schema": selector.PREREGISTRATION_SCHEMA,
            "schema_version": 1,
            "status": (
                "preregistered_before_teacher_agreement_v2_result_inspection"
            ),
            "selector_implementation": file_record(Path(selector.__file__).resolve()),
            "method_contract": selector.method_contract(),
            "method_contract_sha256": selector.METHOD_CONTRACT_SHA256,
            "registration_scope": (
                "two_or_more_distinct_source_scenes_one_global_ceiling"
            ),
            "actual_teacher_agreement_results_opened": False,
            "target_data_or_metrics_opened": False,
            "metric_execution_authorized": False,
            "next_gate": (
                "source_only_candidate_execution_authority_before_frozen_target_metric"
            ),
        },
    )
    return file_record(path)


def _selector_authority(
    tmp_path: Path,
    *,
    last_scene_delta: float = -0.1,
) -> dict[str, str]:
    scene_ids = ["source_a", "source_b"]
    rows = []
    for index, angle in enumerate(selector.CANDIDATE_GRID_RADIANS):
        first = 0.0 if index == 0 else float(index)
        second = 0.0 if index == 0 else 0.25
        if index == len(selector.CANDIDATE_GRID_RADIANS) - 1:
            second = last_scene_delta
        per_scene = []
        for scene_id, delta in zip(scene_ids, (first, second)):
            per_scene.append(
                {
                    "scene_id": scene_id,
                    "heldout_scale_observations": 30,
                    "delta_cosine_sum_vs_o1_0p15": delta,
                    "mean_delta_cosine_vs_o1_0p15": delta / 30,
                    "nonregression": delta >= 0.0,
                }
            )
        pooled_delta = first + second
        pooled_mean = pooled_delta / 60
        every = first >= 0.0 and second >= 0.0
        improved = pooled_mean > 0.0
        rows.append(
            {
                "maximum_angle_radians": angle,
                "pooled_delta_cosine_sum_vs_o1_0p15": pooled_delta,
                "pooled_heldout_scale_observations": 60,
                "pooled_mean_delta_cosine_vs_o1_0p15": pooled_mean,
                "pooled_improvement_strict": improved,
                "every_source_scene_nonregression": every,
                "eligible": improved and every,
                "per_scene": per_scene,
            }
        )
    eligible = [
        row["maximum_angle_radians"] for row in rows if row["eligible"]
    ]
    selected = max(eligible, default=0.15)
    inputs = [
        {
            "scene_id": scene_id,
            "source_format": "compact_source_summary_v1",
            "source": {"path": f"/tmp/{scene_id}.json", "sha256": "a" * 64},
            "teacher_payload": {
                "path": f"/tmp/{scene_id}.pt",
                "sha256": "b" * 64,
            },
            "execution_authority": {
                "path": f"/tmp/{scene_id}_authority.json",
                "sha256": "c" * 64,
            },
            "source_only_loo_audit_sha256": "d" * 64,
        }
        for scene_id in scene_ids
    ]
    path = tmp_path / "selector_candidate.json"
    write_frozen_json(
        path,
        {
            "schema": selector.OUTPUT_SCHEMA,
            "schema_version": 1,
            "status": "source_only_candidate_selected_metric_not_authorized",
            "selector_implementation": file_record(Path(selector.__file__).resolve()),
            "preregistration": _preregistration(tmp_path),
            "method_contract": selector.method_contract(),
            "method_contract_sha256": selector.METHOD_CONTRACT_SHA256,
            "source_scene_ids": scene_ids,
            "source_count": 2,
            "inputs": inputs,
            "candidate_grid": rows,
            "selection": {
                "global_maximum_angle_radians": selected,
                "selection_rule": "largest_eligible_angle_else_0.15",
                "baseline_fallback_used": not eligible,
                "one_global_ceiling": True,
                "per_scene_or_per_query_override_authorized": False,
            },
            "access_audit": {
                "teacher_agreement_payload_or_summary_opened": True,
                "source_only_loo_summary_opened": True,
                "target_images_opened": False,
                "target_labels_or_masks_opened": False,
                "target_metrics_opened": False,
            },
            "query_independent": True,
            "metric_execution_authorized": False,
            "metric_executed": False,
            "candidate_role": "source_only_global_ceiling_candidate_authority",
            "next_gate": (
                "source_only_candidate_execution_authority_before_frozen_target_metric"
            ),
        },
    )
    return file_record(path)


def _score_fixture() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1701)
    accepted = 7
    dimension = 11
    global_count = 12
    positive_count = 4
    negative_count = 3
    base = F.normalize(
        torch.randn(accepted, 3, dimension, generator=generator), dim=-1
    ).half()
    teacher = F.normalize(
        torch.randn(accepted, dimension, generator=generator), dim=-1
    ).half()
    count = torch.tensor([4, 3, 2, 1, 0, 4, 3], dtype=torch.uint8)
    valid = count > 0
    teacher[~valid] = 0
    agreement = torch.tensor([1.0, 0.8, 0.6, 1.0, 0.0, 0.9, 0.7])
    positive_text = F.normalize(
        torch.randn(positive_count, dimension, generator=generator), dim=-1
    )
    negative_text = F.normalize(
        torch.randn(negative_count, dimension, generator=generator), dim=-1
    )
    return {
        "base_features_by_scale": base,
        "global_rows": torch.tensor([0, 2, 3, 5, 7, 9, 11]),
        "teacher_mean": teacher,
        "teacher_valid": valid,
        "retained_view_count": count,
        "directional_resultant": agreement,
        "positive_embeddings": positive_text,
        "negative_embeddings": negative_text,
        "o0_positive_scores": torch.randn(
            global_count, 3, positive_count, generator=generator
        ),
        "o0_negative_scores": torch.randn(
            global_count, 3, negative_count, generator=generator
        ),
    }


def test_small_tensor_chunked_scores_are_exact_dense_equivalent() -> None:
    fixture = _score_fixture()
    dense = candidate.materialize_scores_lowmem(
        **fixture,
        global_ceiling_radians=0.6,
        device=torch.device("cpu"),
        row_batch_size=fixture["global_rows"].numel(),
    )
    chunked = candidate.materialize_scores_lowmem(
        **fixture,
        global_ceiling_radians=0.6,
        device=torch.device("cpu"),
        row_batch_size=2,
    )
    assert torch.equal(chunked["positive_scores"], dense["positive_scores"])
    assert torch.equal(chunked["negative_scores"], dense["negative_scores"])
    assert chunked["descriptor_sha256"] == dense["descriptor_sha256"]
    assert chunked["rows_with_score_replacement"] == dense[
        "rows_with_score_replacement"
    ]
    assert chunked["maximum_batch_rows_observed"] == 2


def test_global_0p15_is_exact_existing_o1_for_nonfallback_scales() -> None:
    generator = torch.Generator().manual_seed(8)
    base = F.normalize(torch.randn(5, 3, 13, generator=generator), dim=-1)
    teacher = F.normalize(torch.randn(5, 13, generator=generator), dim=-1)
    counts = torch.tensor([1, 2, 3, 4, 1], dtype=torch.uint8)
    descriptor, replace, audit = candidate.reliability_candidate_descriptor_batch(
        base,
        teacher,
        torch.ones(5, dtype=torch.bool),
        counts,
        torch.ones(5),
        global_ceiling_radians=0.15,
    )
    expected = torch.stack(
        [geodesic_project(F.normalize(base.float(), dim=-1)[:, scale], teacher, 0.15)
         for scale in range(3)],
        dim=1,
    ).contiguous().float()
    assert bool(replace.all())
    assert torch.equal(descriptor, expected)
    torch.testing.assert_close(
        audit["angular_budget_radians"], torch.full((5,), 0.15)
    )
    assert audit["expanded_budget"][[0, 4]].tolist() == [False, False]


def test_global_0p15_scores_are_bitwise_existing_o1_where_authorized() -> None:
    fixture = _score_fixture()
    actual = candidate.materialize_scores_lowmem(
        **fixture,
        global_ceiling_radians=0.15,
        device=torch.device("cpu"),
        row_batch_size=fixture["global_rows"].numel(),
    )
    base = fixture["base_features_by_scale"]
    teacher = fixture["teacher_mean"]
    valid = fixture["teacher_valid"]
    o1, _ = o1o2._score_descriptors(
        base=base, teacher_mean=teacher, teacher_valid=valid
    )
    positive_text = F.normalize(fixture["positive_embeddings"].float(), dim=-1)
    negative_text = F.normalize(fixture["negative_embeddings"].float(), dim=-1)
    positive_batch = torch.zeros(base.shape[0], 3, positive_text.shape[0])
    negative_batch = torch.zeros(base.shape[0], 3, negative_text.shape[0])
    positive_batch[valid] = torch.einsum(
        "bsd,qd->bsq", o1[valid], positive_text
    )
    negative_batch[valid] = torch.einsum(
        "bsd,qd->bsq", o1[valid], negative_text
    )
    _, replace, _ = candidate.reliability_candidate_descriptor_batch(
        base,
        teacher,
        valid,
        fixture["retained_view_count"],
        fixture["directional_resultant"],
        global_ceiling_radians=0.15,
    )
    rows = fixture["global_rows"]
    expected_positive = fixture["o0_positive_scores"].clone()
    expected_negative = fixture["o0_negative_scores"].clone()
    expected_positive[rows] = torch.where(
        replace[..., None], positive_batch, expected_positive[rows]
    )
    expected_negative[rows] = torch.where(
        replace[..., None], negative_batch, expected_negative[rows]
    )
    assert torch.equal(actual["positive_scores"], expected_positive)
    assert torch.equal(actual["negative_scores"], expected_negative)


def test_invalid_and_antipodal_are_o0_but_single_view_is_exact_o1() -> None:
    base = torch.zeros(3, 3, 5)
    base[..., 0] = 1.0
    teacher = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0],
         [-1.0, 0.0, 0.0, 0.0, 0.0]]
    )
    valid = torch.tensor([False, True, True])
    count = torch.tensor([0, 1, 4], dtype=torch.uint8)
    agreement = torch.tensor([0.0, 1.0, 1.0])
    descriptor, replace, audit = candidate.reliability_candidate_descriptor_batch(
        base, teacher, valid, count, agreement, global_ceiling_radians=0.75
    )
    expected_single = torch.stack(
        [geodesic_project(base[1:2, scale], teacher[1:2], 0.15) for scale in range(3)],
        dim=1,
    )[0]
    assert torch.equal(descriptor[0], base[0])
    assert torch.equal(descriptor[1], expected_single)
    assert torch.equal(descriptor[2], base[2])
    assert not bool(replace[0].any())
    assert bool(replace[1].all())
    assert not bool(replace[2].any())
    assert float(audit["angular_budget_radians"][1]) == pytest.approx(0.15)
    assert audit["expanded_budget"][1].item() is False

    generator = torch.Generator().manual_seed(2)
    positive_o0 = torch.randn(3, 3, 2, generator=generator)
    negative_o0 = torch.randn(3, 3, 2, generator=generator)
    scores = candidate.materialize_scores_lowmem(
        base_features_by_scale=base,
        global_rows=torch.arange(3),
        teacher_mean=teacher,
        teacher_valid=valid,
        retained_view_count=count,
        directional_resultant=agreement,
        positive_embeddings=F.normalize(torch.randn(2, 5, generator=generator), dim=-1),
        negative_embeddings=F.normalize(torch.randn(2, 5, generator=generator), dim=-1),
        o0_positive_scores=positive_o0,
        o0_negative_scores=negative_o0,
        global_ceiling_radians=0.75,
        device=torch.device("cpu"),
        row_batch_size=2,
    )
    assert torch.equal(scores["positive_scores"][[0, 2]], positive_o0[[0, 2]])
    assert torch.equal(scores["negative_scores"][[0, 2]], negative_o0[[0, 2]])
    assert not torch.equal(scores["positive_scores"][1], positive_o0[1])
    assert not torch.equal(scores["negative_scores"][1], negative_o0[1])
    assert scores["rows_with_score_replacement"] == 1
    assert scores["rows_with_expanded_budget"] == 0


def test_selected_ceiling_increases_budget_monotonically() -> None:
    base = torch.zeros(1, 3, 7)
    base[..., 0] = 1.0
    teacher = torch.zeros(1, 7)
    teacher[..., 1] = 1.0
    angles = []
    for ceiling in selector.CANDIDATE_GRID_RADIANS:
        descriptor, replace, _ = candidate.reliability_candidate_descriptor_batch(
            base,
            teacher,
            torch.tensor([True]),
            torch.tensor([4], dtype=torch.uint8),
            torch.tensor([1.0]),
            global_ceiling_radians=ceiling,
        )
        assert bool(replace.all())
        angle = torch.acos((descriptor * base).sum(dim=-1).clamp(-1.0, 1.0))
        angles.append(float(angle[0, 0]))
    assert angles == sorted(angles)
    assert angles == pytest.approx(selector.CANDIDATE_GRID_RADIANS, abs=2e-6)


def test_selector_authority_is_global_two_scene_and_fail_closed(tmp_path: Path) -> None:
    record = _selector_authority(tmp_path)
    payload, verified, ceiling = candidate._validate_selector_candidate(
        record["path"], record["sha256"]
    )
    assert verified == record
    assert payload["source_count"] == 2
    assert ceiling == 0.6
    with pytest.raises(ValueError, match="SHA-256 differs"):
        candidate._validate_selector_candidate(record["path"], "0" * 64)

    forged = json.loads(Path(record["path"]).read_text())
    forged["source_count"] = 1
    forged_path = tmp_path / "forged_selector.json"
    write_frozen_json(forged_path, forged)
    forged_record = file_record(forged_path)
    with pytest.raises(ValueError, match="scene set differs"):
        candidate._validate_selector_candidate(
            forged_record["path"], forged_record["sha256"]
        )


def test_selector_cannot_forge_a_scene_or_query_override(tmp_path: Path) -> None:
    record = _selector_authority(tmp_path)
    forged = json.loads(Path(record["path"]).read_text())
    forged["selection"]["per_scene_or_per_query_override_authorized"] = True
    path = tmp_path / "forged_override_selector.json"
    write_frozen_json(path, forged)
    changed = file_record(path)
    with pytest.raises(ValueError, match="global selection differs"):
        candidate._validate_selector_candidate(changed["path"], changed["sha256"])


def test_formal_lowmem_compatibility_selector_is_accepted_and_recomputed() -> None:
    path = Path(
        "/root/RADIO-GS/paper/artifacts/"
        "lerf_source_only_global_ceiling_lowmem_lineage_compatibility_result_20260807.json"
    )
    record = file_record(path)
    payload, verified, ceiling = candidate._validate_selector_candidate(
        record["path"], record["sha256"]
    )
    assert verified == record
    assert payload["created_after_source_results_for_lineage_compatibility_only"] is True
    assert payload["selection_rule_preregistered_before_results"] is True
    assert payload["source_count"] == 2
    assert ceiling == 0.75


def test_teacher_base_lineage_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path = tmp_path / "teacher_execution.json"
    producer = dict(agreement_v2.ENTRYPOINT_IMPLEMENTATION)
    write_frozen_json(
        authority_path,
        {
            "schema": agreement_v2.AUTHORITY_SCHEMA,
            "schema_version": agreement_v2.SCHEMA_VERSION,
            "scene_id": "target",
            "implementation": producer,
            "method_contract_sha256": agreement_v2.METHOD_CONTRACT_SHA256,
            "query_free_materialization_authorized": True,
            "metric_execution_authorized": False,
            "access_audit": o1o2.access_audit(),
        },
    )
    base_record = {"path": "/tmp/base.pt", "sha256": "b" * 64}
    payload = {
        "producer": producer,
        "scene_id": "target",
        "method_contract_sha256": agreement_v2.METHOD_CONTRACT_SHA256,
        "input_authority": {"base_descriptor": base_record},
        "execution_authority": file_record(authority_path),
        "access_audit": o1o2.access_audit(),
    }
    monkeypatch.setattr(
        candidate,
        "load_torch_mapping",
        lambda *args, **kwargs: (payload, "e" * 64, Path("/tmp/teacher.pt")),
    )
    monkeypatch.setattr(
        candidate._agreement_v2, "validate_teacher_payload_v2", lambda value: None
    )
    loaded, record = candidate._validate_teacher_payload(
        "/tmp/teacher.pt", "e" * 64,
        scene_id="target", base_descriptor_record=base_record,
    )
    assert loaded is payload
    assert record["sha256"] == "e" * 64
    with pytest.raises(ValueError, match="target lineage differs"):
        candidate._validate_teacher_payload(
            "/tmp/teacher.pt", "e" * 64,
            scene_id="target",
            base_descriptor_record={"path": "/tmp/other.pt", "sha256": "f" * 64},
        )


def test_runtime_device_authority_is_fail_closed() -> None:
    execution = {
        "physical_gpu": 1,
        "cuda_visible_devices": "1",
        "program_device": "cuda:0",
    }
    assert candidate.validate_runtime_device(
        execution,
        environ={"CUDA_VISIBLE_DEVICES": "1"},
        cuda_available=True,
    ) == torch.device("cuda:0")
    with pytest.raises(RuntimeError, match="device authority differs"):
        candidate.validate_runtime_device(
            execution,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            cuda_available=True,
        )
    with pytest.raises(RuntimeError, match="device authority differs"):
        candidate.validate_runtime_device(
            execution,
            environ={"CUDA_VISIBLE_DEVICES": "1"},
            cuda_available=False,
        )


def test_memory_ast_has_no_full_candidate_descriptor_allocation() -> None:
    source = textwrap.dedent(inspect.getsource(candidate.materialize_scores_lowmem))
    tree = ast.parse(source)
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"empty", "zeros", "ones", "full"}:
            continue
        names = {
            child.id for child in ast.walk(node) if isinstance(child, ast.Name)
        }
        if "n_rows" in names and "SCALE_COUNT" in names:
            forbidden.append(node)
    assert forbidden == []
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cat"
        for node in ast.walk(tree)
    )
    assert "base[start:stop]" in source
    assert candidate.method_contract()[
        "full_n_by_3_by_d_candidate_descriptor_allocated"
    ] is False


def test_contract_is_scene_general_metric_closed_and_hash_bound() -> None:
    contract = candidate.method_contract()
    assert contract["candidate_grid_radians"] == [0.15, 0.3, 0.45, 0.6, 0.75]
    assert contract["one_global_ceiling"] is True
    assert contract["scene_or_query_specific_parameters"] is False
    assert contract["scale_axis"] == "three_O0_scales_preserved_independently"
    assert contract["fallback"]["single_retained_view"] == (
        "exact_conservative_O1_no_expansion"
    )
    assert contract["target_data_or_metric_access"] is False
    assert contract["metric_execution_authorized"] is False
    assert candidate.METHOD_CONTRACT_SHA256 == canonical_json_sha256(contract)
