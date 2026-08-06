"""Source-only hierarchical completion over a frozen surface graph.

This module implements the CPU reference contract for
``source_conditioned_hierarchy_completion_v1``.  It deliberately has no
dataset, renderer, target-image, or benchmark-evaluator dependency.  A fixed
maximum-spanning forest turns every Kruskal merge into an ancestor proposal;
source responsibility is then pooled on those proposals without another
round of graph message passing or connected-component postselection.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .query_specific_propagation_cv import (
    responsibility_balanced_log_loss,
    responsibility_weighted_auc,
)


METHOD_CONTRACT = "source_conditioned_hierarchy_completion_v1"
ACTION_FIELD_BASE = "field_base"
ACTION_HIERARCHY = "hierarchy_completion_v1"
REGISTERED_ACTIONS = (ACTION_FIELD_BASE, ACTION_HIERARCHY)
OOF_FOLDS = 3
METRIC_ROUND_DECIMALS = 12
PROBABILITY_EPSILON = 1e-7
ECE_BINS = 10

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: str, name: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _cpu_tensor(value: object, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must reside on CPU")
    return tensor.detach().contiguous()


def _probability_vector(value: object, name: str, length: int) -> torch.Tensor:
    tensor = _cpu_tensor(value, name).double().reshape(-1)
    if tensor.shape != (int(length),):
        raise ValueError(f"{name} must align with the hierarchy leaves")
    if not bool(torch.isfinite(tensor).all()) or bool(
        ((tensor < 0) | (tensor > 1)).any()
    ):
        raise ValueError(f"{name} must be finite and in [0,1]")
    return tensor


def _mass_vector(value: object, name: str, length: int) -> torch.Tensor:
    tensor = _cpu_tensor(value, name).double().reshape(-1)
    if tensor.shape != (int(length),):
        raise ValueError(f"{name} must align with the hierarchy leaves")
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
        raise ValueError(f"{name} must be finite and non-negative")
    return tensor


def _hash_tensors(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in named_tensors:
        tensor = _cpu_tensor(value, name).contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class MaximumSpanningForest:
    """Deterministic binary merge hierarchy over canonical primitive rows."""

    primitive_rows: torch.Tensor
    parent: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    merge_affinity: torch.Tensor
    roots: torch.Tensor
    support_graph_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        rows = _cpu_tensor(self.primitive_rows, "primitive_rows").long().reshape(-1)
        parent = _cpu_tensor(self.parent, "parent").long().reshape(-1)
        left = _cpu_tensor(self.left, "left").long().reshape(-1)
        right = _cpu_tensor(self.right, "right").long().reshape(-1)
        affinity = _cpu_tensor(
            self.merge_affinity, "merge_affinity"
        ).double().reshape(-1)
        roots = _cpu_tensor(self.roots, "roots").long().reshape(-1)
        graph_sha = _require_sha256(
            self.support_graph_sha256, "support_graph_sha256"
        )
        content_sha = _require_sha256(self.content_sha256, "content_sha256")
        leaf_count = int(rows.numel())
        node_count = int(parent.numel())
        if leaf_count <= 0 or node_count < leaf_count or node_count > 2 * leaf_count - 1:
            raise ValueError("maximum-spanning forest has an invalid node domain")
        if not (
            left.shape == parent.shape
            and right.shape == parent.shape
            and affinity.shape == parent.shape
        ):
            raise ValueError("maximum-spanning forest tensors do not align")
        if bool((rows < 0).any()) or (
            leaf_count > 1 and not bool((rows[1:] > rows[:-1]).all())
        ):
            raise ValueError("primitive_rows must be unique and strictly increasing")
        if not bool((left[:leaf_count] == -1).all()) or not bool(
            (right[:leaf_count] == -1).all()
        ):
            raise ValueError("hierarchy leaves cannot have children")
        if not bool(torch.isnan(affinity[:leaf_count]).all()):
            raise ValueError("hierarchy leaf merge affinity must be NaN")
        if node_count > leaf_count:
            internal = torch.arange(leaf_count, node_count, dtype=torch.long)
            if (
                bool((left[leaf_count:] < 0).any())
                or bool((right[leaf_count:] < 0).any())
                or bool((left[leaf_count:] >= internal).any())
                or bool((right[leaf_count:] >= internal).any())
                or bool((left[leaf_count:] == right[leaf_count:]).any())
                or not bool(torch.isfinite(affinity[leaf_count:]).all())
                or bool((affinity[leaf_count:] < 0).any())
            ):
                raise ValueError("hierarchy internal node is malformed")
        expected_parent = torch.full((node_count,), -1, dtype=torch.long)
        for node in range(leaf_count, node_count):
            for child in (int(left[node]), int(right[node])):
                if int(expected_parent[child]) >= 0:
                    raise ValueError("hierarchy node has multiple parents")
                expected_parent[child] = node
        if not torch.equal(parent, expected_parent):
            raise ValueError("hierarchy parent links differ from child links")
        expected_roots = torch.where(parent < 0)[0]
        if not torch.equal(roots, expected_roots) or roots.numel() == 0:
            raise ValueError("hierarchy roots differ from parent authority")
        expected_content = _hash_tensors(
            [
                ("primitive_rows", rows),
                ("parent", parent),
                ("left", left),
                ("right", right),
                ("merge_affinity", affinity),
                ("roots", roots),
            ]
        )
        if content_sha != expected_content:
            raise ValueError("hierarchy content digest differs")
        object.__setattr__(self, "primitive_rows", rows)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "merge_affinity", affinity)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "support_graph_sha256", graph_sha)
        object.__setattr__(self, "content_sha256", content_sha)

    @property
    def leaf_count(self) -> int:
        return int(self.primitive_rows.numel())

    @property
    def node_count(self) -> int:
        return int(self.parent.numel())


def build_maximum_spanning_forest(
    edge_index: torch.Tensor,
    edge_affinity: torch.Tensor,
    primitive_rows: torch.Tensor,
    *,
    support_graph_sha256: str,
    expected_support_graph_sha256: str,
) -> MaximumSpanningForest:
    """Build canonical Kruskal merges from an immutable undirected graph.

    Reciprocal directed storage is accepted, but every repeated undirected edge
    must carry exactly the same affinity.  Equal-affinity edges are ordered by
    the pair of canonical primitive-row identifiers, never input position.
    """

    graph_sha = _require_sha256(support_graph_sha256, "support_graph_sha256")
    expected_sha = _require_sha256(
        expected_support_graph_sha256, "expected_support_graph_sha256"
    )
    if graph_sha != expected_sha:
        raise ValueError("unknown support-graph authority")
    edges = _cpu_tensor(edge_index, "edge_index").long()
    affinity = _cpu_tensor(edge_affinity, "edge_affinity").double().reshape(-1)
    rows = _cpu_tensor(primitive_rows, "primitive_rows").long().reshape(-1)
    if edges.ndim != 2 or edges.shape[0] != 2 or affinity.shape != (edges.shape[1],):
        raise ValueError("edge_index [2,E] and edge_affinity [E] must align")
    if rows.numel() == 0 or bool((rows < 0).any()) or (
        rows.numel() > 1 and not bool((rows[1:] > rows[:-1]).all())
    ):
        raise ValueError("primitive_rows must be non-empty, unique, and sorted")
    if not bool(torch.isfinite(affinity).all()) or bool((affinity < 0).any()):
        raise ValueError("surface affinities must be finite and non-negative")
    leaf_count = int(rows.numel())
    if edges.numel() and (
        int(edges.min()) < 0 or int(edges.max()) >= leaf_count
    ):
        raise ValueError("surface edge is outside the primitive-row domain")

    # Canonicalize reciprocal/single-sided undirected storage without a Python
    # tuple/dict per edge.  The real graphs contain millions of directed
    # edges, so encode local endpoint pairs into one int64 key, sort once, and
    # check adjacent duplicate affinities exactly.  Strictly increasing
    # primitive_rows make local (u,v) order identical to canonical-row order.
    if leaf_count > np.iinfo(np.int64).max // leaf_count:
        raise ValueError("primitive-row domain is too large for canonical edge keys")
    edges_numpy = edges.numpy()
    affinity_numpy = affinity.numpy()
    first_all = edges_numpy[0]
    second_all = edges_numpy[1]
    nonself = first_all != second_all
    if bool(nonself.all()):
        first = first_all
        second = second_all
        nonself_affinity = affinity_numpy
    else:
        first = first_all[nonself]
        second = second_all[nonself]
        nonself_affinity = affinity_numpy[nonself]
    lower = np.minimum(first, second)
    upper = np.maximum(first, second)
    lower *= np.int64(leaf_count)
    lower += upper
    canonical_key = lower
    del lower, upper, first, second
    pair_order = np.argsort(canonical_key, kind="stable")
    ordered_key = canonical_key[pair_order]
    ordered_affinity = nonself_affinity[pair_order]
    del canonical_key, pair_order, nonself_affinity
    duplicate = np.zeros(ordered_key.shape, dtype=np.bool_)
    if ordered_key.size > 1:
        duplicate[1:] = ordered_key[1:] == ordered_key[:-1]
        if bool(np.any(duplicate[1:] & (ordered_affinity[1:] != ordered_affinity[:-1]))):
            raise ValueError("reciprocal surface edge affinity differs")
    unique = ~duplicate
    unique_key = ordered_key[unique]
    unique_affinity = ordered_affinity[unique]
    del ordered_key, ordered_affinity, duplicate, unique
    unique_first = unique_key // np.int64(leaf_count)
    unique_second = unique_key % np.int64(leaf_count)
    del unique_key
    kruskal_order = np.lexsort(
        (unique_second, unique_first, -unique_affinity)
    )

    maximum_nodes = 2 * leaf_count - 1
    parent_numpy = np.full(maximum_nodes, -1, dtype=np.int64)
    left_numpy = np.full(maximum_nodes, -1, dtype=np.int64)
    right_numpy = np.full(maximum_nodes, -1, dtype=np.int64)
    merge_affinity_numpy = np.full(maximum_nodes, np.nan, dtype=np.float64)
    dsu = np.arange(leaf_count, dtype=np.int64)
    component_node = np.arange(leaf_count, dtype=np.int64)
    component_min_row = rows.numpy().copy()

    def find(value: int) -> int:
        root = value
        while int(dsu[root]) != root:
            root = int(dsu[root])
        while int(dsu[value]) != value:
            following = int(dsu[value])
            dsu[value] = root
            value = following
        return root

    next_node = leaf_count
    successful_unions = 0
    for edge_position in kruskal_order:
        index = int(edge_position)
        root_a = find(int(unique_first[index]))
        root_b = find(int(unique_second[index]))
        if root_a == root_b:
            continue
        if component_min_row[root_b] < component_min_row[root_a]:
            root_a, root_b = root_b, root_a
        child_a = int(component_node[root_a])
        child_b = int(component_node[root_b])
        node = next_node
        next_node += 1
        left_numpy[node] = child_a
        right_numpy[node] = child_b
        parent_numpy[child_a] = node
        parent_numpy[child_b] = node
        merge_affinity_numpy[node] = unique_affinity[index]
        dsu[root_b] = root_a
        component_node[root_a] = node
        component_min_row[root_a] = min(
            component_min_row[root_a], component_min_row[root_b]
        )
        successful_unions += 1
        # A connected graph cannot contribute another successful union.  This
        # avoids scanning the long tail of dense non-tree edges.
        if successful_unions == leaf_count - 1:
            break

    parent = torch.from_numpy(parent_numpy[:next_node])
    left = torch.from_numpy(left_numpy[:next_node])
    right = torch.from_numpy(right_numpy[:next_node])
    merge_affinity = torch.from_numpy(merge_affinity_numpy[:next_node])
    roots = torch.from_numpy(np.flatnonzero(parent_numpy[:next_node] < 0))
    content = _hash_tensors(
        [
            ("primitive_rows", rows),
            ("parent", parent),
            ("left", left),
            ("right", right),
            ("merge_affinity", merge_affinity),
            ("roots", roots),
        ]
    )
    return MaximumSpanningForest(
        primitive_rows=rows,
        parent=parent,
        left=left,
        right=right,
        merge_affinity=merge_affinity,
        roots=roots,
        support_graph_sha256=graph_sha,
        content_sha256=content,
    )


@dataclass(frozen=True)
class SourceObservation:
    """Source responsibility sufficient statistics and sealed identity."""

    primitive_rows: torch.Tensor
    positive_weight: torch.Tensor
    negative_weight: torch.Tensor
    q_obs: torch.Tensor
    c_obs: torch.Tensor
    authority_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        rows = _cpu_tensor(self.primitive_rows, "primitive_rows").long().reshape(-1)
        if rows.numel() == 0 or bool((rows < 0).any()) or (
            rows.numel() > 1 and not bool((rows[1:] > rows[:-1]).all())
        ):
            raise ValueError("source primitive_rows must be unique and sorted")
        count = int(rows.numel())
        positive = _mass_vector(self.positive_weight, "positive_weight", count)
        negative = _mass_vector(self.negative_weight, "negative_weight", count)
        q_obs = _probability_vector(self.q_obs, "q_obs", count)
        c_obs = _probability_vector(self.c_obs, "c_obs", count)
        mass = positive + negative
        observed = mass > 0
        expected_q = torch.full((count,), 0.5, dtype=torch.float64)
        expected_q[observed] = positive[observed] / mass[observed]
        if not torch.allclose(q_obs, expected_q, rtol=1e-6, atol=1e-7):
            raise ValueError("q_obs differs from positive responsibility fraction")
        if bool((c_obs[~observed] != 0).any()) or bool((c_obs[observed] <= 0).any()):
            raise ValueError("c_obs does not identify exactly the observed source rows")
        authority = _require_sha256(self.authority_sha256, "authority_sha256")
        content = _require_sha256(self.content_sha256, "content_sha256")
        expected_content = source_observation_content_sha256(
            rows, positive, negative, q_obs, c_obs
        )
        if content != expected_content:
            raise ValueError("source-observation content digest differs")
        object.__setattr__(self, "primitive_rows", rows)
        object.__setattr__(self, "positive_weight", positive)
        object.__setattr__(self, "negative_weight", negative)
        object.__setattr__(self, "q_obs", q_obs)
        object.__setattr__(self, "c_obs", c_obs)
        object.__setattr__(self, "authority_sha256", authority)
        object.__setattr__(self, "content_sha256", content)


def source_observation_content_sha256(
    primitive_rows: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    q_obs: torch.Tensor,
    c_obs: torch.Tensor,
) -> str:
    return _hash_tensors(
        [
            ("primitive_rows", _cpu_tensor(primitive_rows, "primitive_rows").long()),
            ("positive_weight", _cpu_tensor(positive_weight, "positive_weight").double()),
            ("negative_weight", _cpu_tensor(negative_weight, "negative_weight").double()),
            ("q_obs", _cpu_tensor(q_obs, "q_obs").double()),
            ("c_obs", _cpu_tensor(c_obs, "c_obs").double()),
        ]
    )


def make_source_observation(
    primitive_rows: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    q_obs: torch.Tensor,
    c_obs: torch.Tensor,
    *,
    authority_sha256: str,
) -> SourceObservation:
    rows = _cpu_tensor(primitive_rows, "primitive_rows").long().reshape(-1)
    positive = _cpu_tensor(positive_weight, "positive_weight").double().reshape(-1)
    negative = _cpu_tensor(negative_weight, "negative_weight").double().reshape(-1)
    q = _cpu_tensor(q_obs, "q_obs").double().reshape(-1)
    confidence = _cpu_tensor(c_obs, "c_obs").double().reshape(-1)
    return SourceObservation(
        primitive_rows=rows,
        positive_weight=positive,
        negative_weight=negative,
        q_obs=q,
        c_obs=confidence,
        authority_sha256=authority_sha256,
        content_sha256=source_observation_content_sha256(
            rows, positive, negative, q, confidence
        ),
    )


def _validate_authorities(
    hierarchy: MaximumSpanningForest,
    observation: SourceObservation,
    *,
    expected_support_graph_sha256: str,
    expected_source_authority_sha256: str,
) -> None:
    expected_graph = _require_sha256(
        expected_support_graph_sha256, "expected_support_graph_sha256"
    )
    expected_source = _require_sha256(
        expected_source_authority_sha256, "expected_source_authority_sha256"
    )
    if hierarchy.support_graph_sha256 != expected_graph:
        raise ValueError("unknown hierarchy support-graph authority")
    if observation.authority_sha256 != expected_source:
        raise ValueError("unknown source-observation authority")
    if not torch.equal(hierarchy.primitive_rows, observation.primitive_rows):
        raise ValueError("hierarchy and source primitive-row authorities differ")


@dataclass(frozen=True)
class HierarchyProbability:
    hierarchy_probability: torch.Tensor
    proposal_probability: torch.Tensor
    candidate_nodes: torch.Tensor
    positive_seed_leaves: torch.Tensor
    promoted_rows: torch.Tensor
    proposal_support_size: int
    positive_branch_components: int


def hierarchy_probability_from_source(
    hierarchy: MaximumSpanningForest,
    p_field: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
) -> HierarchyProbability:
    """Pool source responsibility over every positive leaf's ancestor chain.

    For an ancestor ``A``, ``r_A=sum_A(m+)/sum_A(m+ + m-)``.  Only ancestors
    containing a source-positive leaf (``m+ > m-`` at that leaf) are proposals.
    Each leaf receives the maximum ``r_A`` among proposals containing it, which
    is a bounded, idempotent multi-branch union.  The immutable field is a
    floor: ``p_H=max(p_field, max_A r_A 1[leaf in A])``.  Thus completion can
    add missing support but cannot erase field support.
    """

    count = hierarchy.leaf_count
    field = _probability_vector(p_field, "p_field", count)
    positive = _mass_vector(positive_weight, "positive_weight", count)
    negative = _mass_vector(negative_weight, "negative_weight", count)
    node_count = hierarchy.node_count
    aggregate_positive = torch.zeros(node_count, dtype=torch.float64)
    aggregate_negative = torch.zeros(node_count, dtype=torch.float64)
    seed_count = torch.zeros(node_count, dtype=torch.long)
    aggregate_positive[:count] = positive
    aggregate_negative[:count] = negative
    positive_seed = (positive + negative > 0) & (positive > negative)
    seed_count[:count] = positive_seed.long()
    for node in range(count, node_count):
        left = int(hierarchy.left[node])
        right = int(hierarchy.right[node])
        aggregate_positive[node] = aggregate_positive[left] + aggregate_positive[right]
        aggregate_negative[node] = aggregate_negative[left] + aggregate_negative[right]
        seed_count[node] = seed_count[left] + seed_count[right]
    total = aggregate_positive + aggregate_negative
    candidate = (seed_count > 0) & (total > 0)
    node_probability = torch.zeros(node_count, dtype=torch.float64)
    node_probability[candidate] = aggregate_positive[candidate] / total[candidate]

    proposal = torch.zeros(count, dtype=torch.float64)
    stack: list[tuple[int, float]] = [
        (int(root), 0.0) for root in reversed(hierarchy.roots.tolist())
    ]
    while stack:
        node, inherited = stack.pop()
        current = max(inherited, float(node_probability[node]))
        if node < count:
            proposal[node] = current
        else:
            stack.append((int(hierarchy.right[node]), current))
            stack.append((int(hierarchy.left[node]), current))
    probability = torch.maximum(field, proposal)
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0) | (probability > 1)).any()
    ):
        raise RuntimeError("hierarchy completion produced an invalid probability")
    promoted = proposal > field
    seed_roots: set[int] = set()
    for leaf in torch.where(positive_seed)[0].tolist():
        node = int(leaf)
        while int(hierarchy.parent[node]) >= 0:
            node = int(hierarchy.parent[node])
        seed_roots.add(node)
    return HierarchyProbability(
        hierarchy_probability=probability,
        proposal_probability=proposal,
        candidate_nodes=torch.where(candidate)[0],
        positive_seed_leaves=torch.where(positive_seed)[0],
        promoted_rows=torch.where(promoted)[0],
        proposal_support_size=int((probability >= 0.5).sum()),
        positive_branch_components=len(seed_roots),
    )


def fuse_source_observation(
    p_hierarchy: torch.Tensor,
    q_obs: torch.Tensor,
    c_obs: torch.Tensor,
) -> torch.Tensor:
    count = int(_cpu_tensor(p_hierarchy, "p_hierarchy").numel())
    hierarchy = _probability_vector(p_hierarchy, "p_hierarchy", count)
    source = _probability_vector(q_obs, "q_obs", count)
    confidence = _probability_vector(c_obs, "c_obs", count)
    result = (1.0 - confidence) * hierarchy + confidence * source
    if not bool(torch.isfinite(result).all()) or bool(
        ((result < 0) | (result > 1)).any()
    ):
        raise RuntimeError("source fusion produced an invalid probability")
    return result


def deterministic_group_folds(
    group_ids: torch.Tensor,
    *,
    num_folds: int = OOF_FOLDS,
) -> torch.Tensor:
    """Assign complete source-raster/subtree groups with SplitMix64."""

    if int(num_folds) != OOF_FOLDS:
        raise ValueError("hierarchy completion v1 requires exactly three folds")
    groups = _cpu_tensor(group_ids, "group_ids").long().reshape(-1)
    if groups.numel() == 0 or bool((groups < 0).any()):
        raise ValueError("fold group ids must be non-negative")
    values = groups.numpy().astype(np.uint64, copy=True)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    values ^= values >> np.uint64(31)
    return torch.from_numpy((values % np.uint64(OOF_FOLDS)).astype(np.int64))


def _weighted_brier(
    labels: torch.Tensor, probability: torch.Tensor, weights: torch.Tensor
) -> float:
    target = labels.double()
    return float((weights * (probability.double() - target).square()).sum() / weights.sum())


def _weighted_ece(
    labels: torch.Tensor, probability: torch.Tensor, weights: torch.Tensor
) -> float:
    target = labels.double()
    score = probability.double()
    total = weights.sum()
    result = torch.zeros((), dtype=torch.float64)
    # Right endpoint 1.0 belongs to the last of ten fixed equal-width bins.
    bins = torch.clamp((score * ECE_BINS).long(), max=ECE_BINS - 1)
    for index in range(ECE_BINS):
        keep = bins == index
        if not bool(keep.any()):
            continue
        mass = weights[keep].sum()
        confidence = (weights[keep] * score[keep]).sum() / mass
        accuracy = (weights[keep] * target[keep]).sum() / mass
        result += mass / total * (confidence - accuracy).abs()
    return float(result)


@dataclass(frozen=True)
class HierarchyOOFResult:
    selected_action: str
    metrics: dict[str, dict[str, float]]
    fold_reports: list[dict[str, object]]
    fold_ids: torch.Tensor
    observed: torch.Tensor
    oof_predictions: dict[str, torch.Tensor]
    support_graph_sha256: str
    hierarchy_content_sha256: str
    source_authority_sha256: str
    source_content_sha256: str
    fold_unit_authority_sha256: str


@torch.inference_mode()
def run_source_only_hierarchy_oof(
    hierarchy: MaximumSpanningForest,
    observation: SourceObservation,
    fold_group_ids: torch.Tensor,
    field_predictor: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    *,
    fold_unit_authority_sha256: str,
    expected_fold_unit_authority_sha256: str,
    expected_support_graph_sha256: str,
    expected_source_authority_sha256: str,
    minimum_class_rows: int = 1,
) -> HierarchyOOFResult:
    """Select the hierarchy arm using only held-out source responsibility."""

    _validate_authorities(
        hierarchy,
        observation,
        expected_support_graph_sha256=expected_support_graph_sha256,
        expected_source_authority_sha256=expected_source_authority_sha256,
    )
    fold_authority = _require_sha256(
        fold_unit_authority_sha256, "fold_unit_authority_sha256"
    )
    expected_fold_authority = _require_sha256(
        expected_fold_unit_authority_sha256,
        "expected_fold_unit_authority_sha256",
    )
    if fold_authority != expected_fold_authority:
        raise ValueError("unknown OOF fold-unit authority")
    groups = _cpu_tensor(fold_group_ids, "fold_group_ids").long().reshape(-1)
    if groups.shape != (hierarchy.leaf_count,):
        raise ValueError("fold groups must align with hierarchy leaves")
    fold_ids = deterministic_group_folds(groups)
    # This audit makes spatial leakage visible: one declared raster/subtree
    # group can never straddle folds.
    for group in groups.unique().tolist():
        if fold_ids[groups == int(group)].unique().numel() != 1:
            raise RuntimeError("one fold group was split across held-out folds")

    signed = observation.positive_weight - observation.negative_weight
    reference_weight = observation.positive_weight + observation.negative_weight
    observed = (reference_weight > 0) & (signed != 0)
    labels = signed > 0
    minimum = int(minimum_class_rows)
    if minimum <= 0:
        raise ValueError("minimum_class_rows must be positive")
    fold_reports: list[dict[str, object]] = []
    oof = {
        action: torch.full((hierarchy.leaf_count,), float("nan"), dtype=torch.float64)
        for action in REGISTERED_ACTIONS
    }
    total_support = {action: 0 for action in REGISTERED_ACTIONS}
    for fold in range(OOF_FOLDS):
        heldout_group = fold_ids == fold
        heldout = observed & heldout_group
        training = observed & ~heldout_group
        report: dict[str, object] = {
            "fold": fold,
            "heldout_groups": int(groups[heldout_group].unique().numel()),
        }
        for population_name, mask in (("heldout", heldout), ("training", training)):
            positive_rows = mask & labels
            negative_rows = mask & ~labels
            report[f"{population_name}_positive_rows"] = int(positive_rows.sum())
            report[f"{population_name}_negative_rows"] = int(negative_rows.sum())
            report[f"{population_name}_positive_weight"] = float(
                reference_weight[positive_rows].sum()
            )
            report[f"{population_name}_negative_weight"] = float(
                reference_weight[negative_rows].sum()
            )
            if (
                int(positive_rows.sum()) < minimum
                or int(negative_rows.sum()) < minimum
                or float(reference_weight[positive_rows].sum()) <= 0
                or float(reference_weight[negative_rows].sum()) <= 0
            ):
                raise ValueError(
                    f"fold {fold} {population_name} lacks signed source population"
                )
        training_positive = observation.positive_weight.clone()
        training_negative = observation.negative_weight.clone()
        training_positive[heldout_group] = 0
        training_negative[heldout_group] = 0
        if bool((training_positive[heldout_group] != 0).any()) or bool(
            (training_negative[heldout_group] != 0).any()
        ):
            raise RuntimeError("held-out source evidence survived fold clearing")
        field = _probability_vector(
            field_predictor(training_positive.clone(), training_negative.clone(), fold),
            f"fold_{fold}_p_field",
            hierarchy.leaf_count,
        )
        completion = hierarchy_probability_from_source(
            hierarchy, field, training_positive, training_negative
        )
        oof[ACTION_FIELD_BASE][heldout] = field[heldout]
        oof[ACTION_HIERARCHY][heldout] = completion.hierarchy_probability[heldout]
        total_support[ACTION_FIELD_BASE] += int((field >= 0.5).sum())
        total_support[ACTION_HIERARCHY] += completion.proposal_support_size
        report["field_base_support_size"] = int((field >= 0.5).sum())
        report["hierarchy_support_size"] = completion.proposal_support_size
        report["ancestor_candidate_nodes"] = int(completion.candidate_nodes.numel())
        report["positive_branch_components"] = completion.positive_branch_components
        fold_reports.append(report)

    if not bool(observed.any()) or any(
        not bool(torch.isfinite(oof[action][observed]).all())
        for action in REGISTERED_ACTIONS
    ):
        raise RuntimeError("OOF hierarchy gate lacks held-out source predictions")
    observed_labels = labels[observed]
    observed_weight = reference_weight[observed]
    metrics: dict[str, dict[str, float]] = {}
    for action in REGISTERED_ACTIONS:
        probability = oof[action][observed]
        metrics[action] = {
            "responsibility_balanced_log_loss": responsibility_balanced_log_loss(
                observed_labels,
                probability,
                observed_weight,
                probability_epsilon=PROBABILITY_EPSILON,
            ),
            "responsibility_weighted_auc": responsibility_weighted_auc(
                observed_labels, probability, observed_weight
            ),
            "responsibility_weighted_brier": _weighted_brier(
                observed_labels, probability, observed_weight
            ),
            "responsibility_weighted_ece10": _weighted_ece(
                observed_labels, probability, observed_weight
            ),
            "proposal_support_size": float(total_support[action]),
        }

    def rank(action: str) -> tuple[float, float, int, int]:
        values = metrics[action]
        return (
            round(values["responsibility_balanced_log_loss"], METRIC_ROUND_DECIMALS),
            -round(values["responsibility_weighted_auc"], METRIC_ROUND_DECIMALS),
            int(values["proposal_support_size"]),
            0 if action == ACTION_FIELD_BASE else 1,
        )

    selected = min(REGISTERED_ACTIONS, key=rank)
    return HierarchyOOFResult(
        selected_action=selected,
        metrics=metrics,
        fold_reports=fold_reports,
        fold_ids=fold_ids,
        observed=observed,
        oof_predictions=oof,
        support_graph_sha256=hierarchy.support_graph_sha256,
        hierarchy_content_sha256=hierarchy.content_sha256,
        source_authority_sha256=observation.authority_sha256,
        source_content_sha256=observation.content_sha256,
        fold_unit_authority_sha256=fold_authority,
    )


@dataclass(frozen=True)
class SourceConditionedCompletion:
    selected_action: str
    p_hierarchy: torch.Tensor
    p_final: torch.Tensor
    proposal: HierarchyProbability | None


def apply_source_conditioned_completion(
    hierarchy: MaximumSpanningForest,
    observation: SourceObservation,
    p_field: torch.Tensor,
    oof_result: HierarchyOOFResult,
    *,
    expected_support_graph_sha256: str,
    expected_source_authority_sha256: str,
    expected_fold_unit_authority_sha256: str,
) -> SourceConditionedCompletion:
    """Apply the sealed OOF action and exact probability-preserving fusion."""

    _validate_authorities(
        hierarchy,
        observation,
        expected_support_graph_sha256=expected_support_graph_sha256,
        expected_source_authority_sha256=expected_source_authority_sha256,
    )
    expected_fold_authority = _require_sha256(
        expected_fold_unit_authority_sha256,
        "expected_fold_unit_authority_sha256",
    )
    if (
        oof_result.support_graph_sha256 != hierarchy.support_graph_sha256
        or oof_result.hierarchy_content_sha256 != hierarchy.content_sha256
        or oof_result.source_authority_sha256 != observation.authority_sha256
        or oof_result.source_content_sha256 != observation.content_sha256
        or oof_result.fold_unit_authority_sha256 != expected_fold_authority
    ):
        raise ValueError("OOF decision provenance differs from completion inputs")
    field = _probability_vector(p_field, "p_field", hierarchy.leaf_count)
    if oof_result.selected_action == ACTION_FIELD_BASE:
        p_hierarchy = field
        proposal = None
    elif oof_result.selected_action == ACTION_HIERARCHY:
        proposal = hierarchy_probability_from_source(
            hierarchy,
            field,
            observation.positive_weight,
            observation.negative_weight,
        )
        p_hierarchy = proposal.hierarchy_probability
    else:
        raise ValueError("OOF decision selected an unknown action")
    final = fuse_source_observation(
        p_hierarchy, observation.q_obs, observation.c_obs
    )
    return SourceConditionedCompletion(
        selected_action=oof_result.selected_action,
        p_hierarchy=p_hierarchy,
        p_final=final,
        proposal=proposal,
    )
