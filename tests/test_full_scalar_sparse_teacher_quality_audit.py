from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import hashlib
from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.full_scalar_sparse_teacher_quality_audit import (
    build_quality_audit,
    validate_quality_audit,
)
from radio_gs.interfaces.full_scalar_sparse_teacher_selection import (
    region_fingerprint,
)
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts import (
    audit_full_scalar_sparse_teacher_quality as audit_script,
)
from radio_gs.scripts import materialize_accepted_v2_canonical_region_authority as accepted_producer
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts import materialize_official_multiview_siglip2_teacher_authority as teacher_producer
from radio_gs.scripts import train_surface_region_full_scalar_residual as trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    sha256_file,
    write_torch_noclobber,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _accepted_input(*, graph_sha: str, geometry_count: int) -> dict:
    geometry = {
        "num_gaussians": geometry_count,
        "xyz_sha256": _digest("xyz"),
    }
    return {
        "geometry_authority": {
            "kind": "factorized_primitive_state_v2",
            "factorized_primitive_state_file_sha256": _digest("state"),
            "factorized_primitive_state_contract_sha256": (
                shard.FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
            ),
            "factorized_field_checkpoint_file_sha256": _digest("field"),
            "factorized_radio_cache_file_sha256": _digest("cache"),
            "primitive_row_authority_sha256": _digest("rows"),
            "geometry_fingerprint": geometry,
        },
        "support_graph_authority": {
            "kind": "canonical_query_free_support_graph_v1",
            "support_graph_file_sha256": graph_sha,
            "primitive_row_authority_sha256": _digest("rows"),
        },
        "selection_authority": {
            "kind": "exact_marginal_anchor_visibility_sparse_selection_v1",
            "exact_marginal_responsibility_authority_file_sha256": _digest(
                "responsibility"
            ),
            "exact_marginal_formula_sha256": _digest("formula"),
            "responsibility_view_records_sha256": _digest("view-records"),
            "sampling_contract_sha256": shard.SAMPLING_CONTRACT_SHA256,
        },
        "accepted_v2_checkpoint_authority": trainer._accepted_v2_authority(),
        "official_summary_head_authority": (
            shard.accepted_region_official_head_authority()
        ),
    }


def _accepted_four_regions(*, graph_sha: str) -> dict:
    scene = "scene0001_00"
    count = 30
    active_rows = [
        list(range(0, 24)),
        list(range(1, 25)),
        list(range(2, 26)),
        list(range(3, 27)),
    ]
    scales = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    rows = torch.tensor(active_rows, dtype=torch.long)
    anchors = torch.zeros(4, dtype=torch.long)
    fingerprints = [
        region_fingerprint(
            scene_id=scene,
            scale_index=int(scales[index]),
            anchor_global_row=active_rows[index][0],
            active_global_rows=active_rows[index],
        )
        for index in range(4)
    ]
    e0 = torch.zeros(4, trainer.DESCRIPTOR_DIM, dtype=torch.float32)
    e0[:, 0] = 1.0
    return accepted_producer.build_authority_payload(
        scene_id=scene,
        geometry_fingerprint={"num_gaussians": count, "xyz_sha256": _digest("xyz")},
        accepted_base_valid=torch.ones(count, dtype=torch.bool),
        canonical_region_indices=torch.tensor([0, 1, 31, 62], dtype=torch.long),
        region_fingerprints=fingerprints,
        selection_audit={
            "sampling_contract_sha256": shard.SAMPLING_CONTRACT_SHA256,
            "canonical_candidate_region_count": 90,
            "exact_overlap_candidate_count": 90,
            "teacher_visible_candidate_count": 4,
            "selected_region_count": 4,
            "selected_count_by_scale": [2, 1, 1],
        },
        region_rows=rows,
        token_mask=torch.ones_like(rows, dtype=torch.bool),
        anchor_index=anchors,
        scale_indices=scales,
        accepted_v2_e0=e0,
        input_authority=_accepted_input(graph_sha=graph_sha, geometry_count=count),
    )


def _views(count: int = 4) -> list[dict]:
    return [
        {
            "frame_id": f"{index:05d}",
            "source_relative_path": f"color/{index:05d}.jpg",
            "source_image_sha256": _digest(f"rgb-{index}"),
            "field_frame_authority_sha256": _digest(f"frame-{index}"),
            "source_image_height": 100,
            "source_image_width": 200,
            "feature_grid_height": 10,
            "feature_grid_width": 20,
            "responsibility_view_index": index,
            "responsibility_view_file_sha256": _digest(f"responsibility-{index}"),
        }
        for index in range(count)
    ]


def _teacher(
    accepted: dict,
    *,
    accepted_file_sha: str,
    row_view_counts: list[int],
) -> dict:
    pair_rows = torch.repeat_interleave(
        torch.arange(len(row_view_counts), dtype=torch.long),
        torch.tensor(row_view_counts, dtype=torch.long),
    )
    pair_views = torch.cat(
        [torch.arange(count, dtype=torch.long) for count in row_view_counts]
    )
    pair_count = int(pair_rows.numel())
    descriptors = torch.zeros(
        pair_count, trainer.DESCRIPTOR_DIM, dtype=torch.float32
    )
    descriptors[:, 0] = 1.0
    boxes = torch.tensor(
        [[index, 2 * index, index + 10, 2 * index + 20] for index in range(pair_count)],
        dtype=torch.long,
    )
    input_authority = {
        "source_rgb_scene_authority_file_sha256": _digest("source-rgb-file"),
        "source_rgb_scene_authority_content_sha256": _digest("source-rgb-content"),
        "factorized_primitive_state_file_sha256": _digest("state"),
        "accepted_region_authority_file_sha256": accepted_file_sha,
        "accepted_region_channel_sha256": canonical_json_sha256(
            accepted["channel_sha256"]
        ),
        "accepted_region_fingerprints_sha256": canonical_json_sha256(
            accepted["region_fingerprints"]
        ),
        "exact_marginal_responsibility_authority_file_sha256": _digest(
            "responsibility"
        ),
        "official_radio_checkpoint_file_sha256": shard.OFFICIAL_RADIO_CHECKPOINT_SHA256,
        "descriptor_definition": shard.official_teacher_descriptor_definition(),
    }
    return teacher_producer.build_teacher_payload(
        scene_id=accepted["scene_id"],
        source_rgb_scene_authority_sha256=_digest("source-rgb-content"),
        canonical_region_indices=accepted["canonical_region_indices"],
        region_fingerprints=accepted["region_fingerprints"],
        view_records=_views(max(row_view_counts)),
        pair_region_indices=pair_rows,
        pair_view_indices=pair_views,
        pair_descriptors=descriptors,
        pair_crop_boxes_tlbr=boxes,
        pair_support_hit_counts=torch.arange(1, pair_count + 1, dtype=torch.long),
        pair_visible_primitive_counts=(pair_views + 1).long(),
        selection_audit={
            "accepted_selection_audit": accepted["selection_audit"],
            "pair_count": pair_count,
            "maximum_views_per_region": max(row_view_counts),
        },
        input_authority=input_authority,
    )


def test_source_only_audit_reports_sparse_teacher_quality_and_no_clobber(
    tmp_path: Path,
) -> None:
    graph_sha = _digest("unused-graph")
    accepted = _accepted_four_regions(graph_sha=graph_sha)
    accepted_path = tmp_path / "accepted.pt"
    write_torch_noclobber(accepted_path, accepted)
    accepted_sha = sha256_file(accepted_path)
    teacher = _teacher(
        accepted,
        accepted_file_sha=accepted_sha,
        row_view_counts=[1, 2, 3, 4],
    )
    teacher_path = tmp_path / "teacher.pt"
    write_torch_noclobber(teacher_path, teacher)
    teacher_sha = sha256_file(teacher_path)
    output = tmp_path / "audit.json"
    args = Namespace(
        accepted_region_authority=str(accepted_path),
        expected_accepted_region_authority_sha256=accepted_sha,
        teacher_observation_authority=str(teacher_path),
        expected_teacher_observation_authority_sha256=teacher_sha,
        support_graph=None,
        expected_support_graph_sha256=None,
        output=str(output),
    )
    receipt = audit_script.materialize(args)
    assert receipt["outputs_written"] is True
    payload, _, _ = load_json_object(output, label="test quality audit")
    validate_quality_audit(payload)
    statistics = payload["statistics"]
    assert statistics["tokens"]["below_minimum_token_region_fraction"] == 0.0
    assert [row["region_count"] for row in statistics["views_per_region"]["histogram"]] == [1, 1, 1, 1]
    assert statistics["core_context"]["availability"] == "unavailable"
    assert statistics["viewpoint_geometry"]["availability"] == "unavailable"
    assert statistics["descriptor_consistency"]["multiview_region_count"] == 3
    assert statistics["descriptor_consistency"][
        "within_region_unordered_pair_cosine"
    ]["mean"] == pytest.approx(1.0)
    assert payload["conclusion_gate"]["minimum_24_token_diagnostic"] == "pass"
    assert payload["conclusion_gate"]["overall"] == "review_required"
    assert payload["source_access"]["benchmark_queries_opened"] is False
    with pytest.raises(FileExistsError, match="refuses to clobber"):
        audit_script.materialize(args)


def _chain_graph(global_count: int, global_rows: torch.Tensor) -> dict:
    rows = torch.as_tensor(global_rows).long().cpu().contiguous()
    local_count = int(rows.numel())
    xyz = torch.zeros(local_count, 3, dtype=torch.float32)
    xyz[:, 0] = torch.arange(local_count, dtype=torch.float32) * 0.01
    forward = torch.arange(local_count - 1, dtype=torch.long)
    edge_index = torch.stack(
        [
            torch.cat((forward, forward + 1)),
            torch.cat((forward + 1, forward)),
        ]
    )
    edge_count = edge_index.shape[1]
    contract = SurfaceRegionContractV2()
    return {
        "schema_version": 1,
        "global_rows": rows,
        "num_global_rows": global_count,
        "xyz": xyz,
        "edge_index": edge_index,
        "edge_weight": torch.ones(edge_count, dtype=torch.float32),
        "raw_affinity": torch.ones(edge_count, dtype=torch.float32),
        "edge_channels": {
            "geometry": torch.ones(edge_count, dtype=torch.float32),
            "appearance": torch.ones(edge_count, dtype=torch.float32),
            "boundary": torch.ones(edge_count, dtype=torch.float32),
        },
        "local_sigma": torch.ones(local_count, dtype=torch.float32),
        "metadata": {
            "graph_config": asdict(contract.graph_config()),
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }


def test_optional_sha_bound_graph_reconstructs_exact_core_context(
    tmp_path: Path,
) -> None:
    # Match the real graph schema: ``num_global_rows`` is the complete global
    # primitive domain while graph tensors contain only an active, possibly
    # non-contiguous subset of those rows.
    graph_rows = torch.tensor(
        [value for value in range(30) if value not in {10, 21, 29}],
        dtype=torch.long,
    )
    graph = _chain_graph(30, graph_rows)
    graph_path = tmp_path / "graph.pt"
    write_torch_noclobber(graph_path, graph)
    graph_sha = sha256_file(graph_path)
    contract = SurfaceRegionContractV2()
    support = PrimitiveSupportGraph(
        edge_index=graph["edge_index"],
        edge_weight=graph["edge_weight"],
        raw_affinity=graph["raw_affinity"],
        local_sigma=graph["local_sigma"],
        num_nodes=int(graph_rows.numel()),
        edge_channels=graph["edge_channels"],
    )
    selected, core, _distance = contract.expand(
        support, graph["xyz"], 0, contract.radii_m[0]
    )
    active = graph_rows[selected].tolist()
    accepted_valid = torch.zeros(30, dtype=torch.bool)
    accepted_valid[graph_rows] = True
    accepted = accepted_producer.build_authority_payload(
        scene_id="scene0001_00",
        geometry_fingerprint={"num_gaussians": 30, "xyz_sha256": _digest("xyz")},
        accepted_base_valid=accepted_valid,
        canonical_region_indices=torch.tensor([0], dtype=torch.long),
        region_fingerprints=[
            region_fingerprint(
                scene_id="scene0001_00",
                scale_index=0,
                anchor_global_row=0,
                active_global_rows=active,
            )
        ],
        selection_audit={
            "sampling_contract_sha256": shard.SAMPLING_CONTRACT_SHA256,
            "canonical_candidate_region_count": 90,
            "exact_overlap_candidate_count": 90,
            "teacher_visible_candidate_count": 1,
            "selected_region_count": 1,
            "selected_count_by_scale": [1],
        },
        region_rows=graph_rows[selected][None].long(),
        token_mask=torch.ones(1, selected.numel(), dtype=torch.bool),
        anchor_index=torch.tensor([0], dtype=torch.long),
        scale_indices=torch.tensor([0], dtype=torch.long),
        accepted_v2_e0=torch.nn.functional.one_hot(
            torch.tensor([0]), num_classes=trainer.DESCRIPTOR_DIM
        ).float(),
        input_authority=_accepted_input(graph_sha=graph_sha, geometry_count=30),
    )
    accepted_path = tmp_path / "accepted.pt"
    write_torch_noclobber(accepted_path, accepted)
    accepted_sha = sha256_file(accepted_path)
    teacher = _teacher(
        accepted, accepted_file_sha=accepted_sha, row_view_counts=[1]
    )
    teacher_path = tmp_path / "teacher.pt"
    write_torch_noclobber(teacher_path, teacher)
    teacher_sha = sha256_file(teacher_path)
    payload = build_quality_audit(
        accepted_value=accepted,
        accepted_file_sha256=accepted_sha,
        teacher_value=teacher,
        teacher_file_sha256=teacher_sha,
        support_graph_value=graph,
        support_graph_file_sha256=graph_sha,
    )
    stats = payload["statistics"]["core_context"]
    assert stats["availability"] == "available"
    assert stats["core_token_count"]["mean"] == pytest.approx(float(core.sum()))
    assert stats["context_token_count"]["mean"] == pytest.approx(
        float((~core).sum())
    )
    assert payload["source_access"]["support_graph_opened"] is True


def test_audit_fails_closed_on_sha_and_teacher_alignment(tmp_path: Path) -> None:
    accepted = _accepted_four_regions(graph_sha=_digest("graph"))
    accepted_path = tmp_path / "accepted.pt"
    write_torch_noclobber(accepted_path, accepted)
    accepted_sha = sha256_file(accepted_path)
    teacher = _teacher(
        accepted,
        accepted_file_sha=accepted_sha,
        row_view_counts=[1, 2, 3, 4],
    )
    with pytest.raises(ValueError, match="teacher caller binding"):
        build_quality_audit(
            accepted_value=accepted,
            accepted_file_sha256=_digest("wrong-accepted-file"),
            teacher_value=teacher,
            teacher_file_sha256=_digest("teacher"),
        )
    drift = dict(teacher)
    drift["canonical_region_indices"] = teacher["canonical_region_indices"].clone()
    drift["canonical_region_indices"][0] = 5
    drift["channel_sha256"] = shard.teacher_observation_channel_sha256(drift)
    with pytest.raises(ValueError):
        build_quality_audit(
            accepted_value=accepted,
            accepted_file_sha256=accepted_sha,
            teacher_value=drift,
            teacher_file_sha256=_digest("teacher"),
        )
