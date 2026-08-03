from dataclasses import asdict

import cv2
import numpy as np
import torch

from radio_gs.scripts.eval_nvos_gaussian_first import (
    _resize_nvos_score_for_evaluation,
)
from radio_gs.scripts.eval_frozen_nvos_primitive_unary import (
    _readout_contract,
)
from radio_gs.scripts.fuse_nvos_dense_exact_anchors import (
    METHOD_CONTRACT as EXACT_ANCHOR_METHOD_CONTRACT,
    exact_exclusive_anchor_fusion,
)
from radio_gs.scripts.propagate_nvos_dense_selector_fixed_graph import (
    METHOD_CONTRACT as GRAPH_METHOD_CONTRACT,
    SOLVER_CONFIG,
)
from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    solve_primitive_support,
)
def test_ordinary_nvos_score_resize_is_exact_cv2_linear_and_not_beta_nearest():
    score = np.asarray([[0.0, 0.3, 1.0], [1.0, 0.6, 0.0]], dtype=np.float32)
    target_shape = (5, 7)
    ordinary = _resize_nvos_score_for_evaluation(
        score, target_shape, registered_forward_unary="none"
    )
    expected = cv2.resize(score, (7, 5), interpolation=cv2.INTER_LINEAR)
    beta = _resize_nvos_score_for_evaluation(
        score, target_shape, registered_forward_unary="beta_coverage_v1"
    )
    assert np.array_equal(ordinary, expected)
    assert not np.array_equal(ordinary, beta)


def test_graph_selector_readout_contract_binds_graph_and_upstream_contract():
    contract = _readout_contract(
        {
            "artifact_type": "nvos_dino_dense_prompt_fixed_graph_selector",
            "method_contract_sha256": "a" * 64,
            "support_graph_sha256": "b" * 64,
        }
    )
    assert contract["graph"] == "frozen_canonical_mpr_v3_shared_support_graph_k16"
    assert contract["selector_method_contract_sha256"] == "a" * 64
    assert contract["support_graph_sha256"] == "b" * 64


def test_graph_method_contract_contains_exact_solver_config():
    assert GRAPH_METHOD_CONTRACT["solver_config"] == asdict(SOLVER_CONFIG)
    assert GRAPH_METHOD_CONTRACT["solver_config"] == {
        "iterations": 12,
        "residual": 0.30,
        "unary_temperature": 0.10,
        "support_threshold": 0.50,
        "component_edge_threshold": 1e-5,
        "seeded_component_min_weight": 0.20,
        "top_k_components": 3,
        "solver_type": "confidence_random_walker",
        "laplacian_weight": 1.0,
        "cg_iterations": 64,
        "cg_tolerance": 1e-5,
        "hard_seed_threshold": 0.20,
        "hard_seed_conflict_policy": "exclusive_relative",
        "hard_seed_conflict_margin": 0.0,
        "unary_edge_contrast": 0.0,
    }


def test_confidence_random_walker_ignores_edge_weight_and_uses_raw_affinity():
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long
    )

    def graph(edge_weight, raw_affinity):
        return PrimitiveSupportGraph(
            edge_index=edge_index,
            edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
            raw_affinity=torch.tensor(raw_affinity, dtype=torch.float32),
            local_sigma=torch.ones(3),
            num_nodes=3,
        )

    unary = 0.1 * torch.logit(torch.tensor([0.9, 0.2, 0.6]))
    raw = [1.0, 1.0, 0.2, 0.2, 0.05, 0.05]
    first = solve_primitive_support(
        graph([0.1, 0.1, 0.2, 0.2, 0.7, 0.7], raw),
        unary,
        config=SOLVER_CONFIG,
    )
    changed_edge_weight = solve_primitive_support(
        graph([0.9, 0.9, 0.8, 0.8, 0.3, 0.3], raw),
        unary,
        config=SOLVER_CONFIG,
    )
    changed_raw_affinity = solve_primitive_support(
        graph(
            [0.1, 0.1, 0.2, 0.2, 0.7, 0.7],
            [0.05, 0.05, 1.0, 1.0, 0.2, 0.2],
        ),
        unary,
        config=SOLVER_CONFIG,
    )
    assert torch.equal(first, changed_edge_weight)
    assert not torch.allclose(first, changed_raw_affinity, atol=1e-4, rtol=0.0)


def test_exact_anchor_fusion_clamps_only_exclusive_observations():
    source = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
    positive = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    negative = torch.tensor([0.0, 2.0, 3.0, 0.0, 0.0], dtype=torch.float64)
    fused, pos_only, neg_only, conflict = exact_exclusive_anchor_fusion(
        source, positive, negative
    )
    assert torch.equal(fused, torch.tensor([1.0, 0.0, 0.4, 0.5, 0.6]))
    assert torch.equal(pos_only, torch.tensor([True, False, False, False, False]))
    assert torch.equal(neg_only, torch.tensor([False, True, False, False, False]))
    assert torch.equal(conflict, torch.tensor([False, False, True, False, False]))
    assert torch.equal(fused[~(pos_only | neg_only)], source[~(pos_only | neg_only)])


def test_exact_anchor_fusion_is_sign_symmetric_without_a_mass_threshold():
    source = torch.tensor([0.13, 0.87, 0.41], dtype=torch.float32)
    positive = torch.tensor([1e-300, 0.0, 2.0], dtype=torch.float64)
    negative = torch.tensor([0.0, 1e-300, 3.0], dtype=torch.float64)
    fused, *_ = exact_exclusive_anchor_fusion(source, positive, negative)
    swapped, *_ = exact_exclusive_anchor_fusion(
        1.0 - source, negative, positive
    )
    assert torch.equal(swapped, 1.0 - fused)
    assert EXACT_ANCHOR_METHOD_CONTRACT["anchor_threshold"] == (
        "none_exact_positive_mass_support"
    )


def test_exact_anchor_readout_contract_stays_graph_free_and_binds_upstream():
    contract = _readout_contract(
        {
            "artifact_type": "nvos_dino_dense_exact_exclusive_anchor_selector",
            "method_contract_sha256": "c" * 64,
            "source_completion_sha256": "d" * 64,
        }
    )
    assert contract["graph"] == "none"
    assert contract["connected_selection"] == "none"
    assert contract["selector_method_contract_sha256"] == "c" * 64
    assert contract["selector_source_completion_sha256"] == "d" * 64
