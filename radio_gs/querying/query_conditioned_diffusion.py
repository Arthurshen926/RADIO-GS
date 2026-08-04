"""Query-conditioned graph diffusion with an explicit LUDVIG compatibility mode.

The released LUDVIG implementation and the mathematically conventional
symmetric-normalized graph use different normalizations.  Keeping both paths
in this module prevents a clean reimplementation from being mistaken for an
exact execution-semantic match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


DiffusionKernel = Literal[
    "ludvig_release_compat", "symmetric_normalized", "continuous_convex_v2"
]
LogisticFitPopulation = Literal[
    "auto_release", "signed_nonzero", "all_nodes_positive_only"
]


@dataclass(frozen=True)
class QueryConditionedDiffusionConfig:
    """Frozen parameters for one query-conditioned diffusion execution."""

    kernel: DiffusionKernel = "ludvig_release_compat"
    feature_bandwidth: float = 1.0
    regularizer_bandwidth: float = 1.0
    logistic_c: float = 0.01
    logistic_fit_population: LogisticFitPopulation = "auto_release"
    iterations: int = 100
    edge_binarize_threshold: float | None = 1e-5
    distance_chunk_size: int = 32
    laplacian_weight: float = 1.0
    cg_iterations: int = 64
    cg_tolerance: float = 1e-5
    unobserved_fidelity: float = 0.01
    hard_observation_confidence: float = 0.99
    hard_positive_probability: float = 0.90
    hard_negative_probability: float = 0.10
    solver_row_chunk_size: int = 32768
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.kernel not in {
            "ludvig_release_compat",
            "symmetric_normalized",
            "continuous_convex_v2",
        }:
            raise ValueError(f"unsupported diffusion kernel: {self.kernel}")
        if self.logistic_fit_population not in {
            "auto_release",
            "signed_nonzero",
            "all_nodes_positive_only",
        }:
            raise ValueError("unsupported logistic_fit_population")
        if min(
            float(self.feature_bandwidth),
            float(self.regularizer_bandwidth),
            float(self.logistic_c),
            float(self.eps),
        ) <= 0:
            raise ValueError("bandwidths, logistic_c, and eps must be positive")
        if int(self.iterations) <= 0:
            raise ValueError("iterations must be positive")
        if int(self.distance_chunk_size) <= 0:
            raise ValueError("distance_chunk_size must be positive")
        if float(self.laplacian_weight) < 0 or int(self.cg_iterations) <= 0:
            raise ValueError("continuous solver weight/iterations are invalid")
        if float(self.cg_tolerance) <= 0 or int(self.solver_row_chunk_size) <= 0:
            raise ValueError("continuous solver tolerance/chunk are invalid")
        if not 0 <= float(self.unobserved_fidelity) <= 1:
            raise ValueError("unobserved_fidelity must be in [0,1]")
        if not 0 <= float(self.hard_observation_confidence) <= 1:
            raise ValueError("hard_observation_confidence must be in [0,1]")
        if not (
            0 <= float(self.hard_negative_probability)
            < float(self.hard_positive_probability) <= 1
        ):
            raise ValueError("hard reference probability thresholds are invalid")
        if (
            self.edge_binarize_threshold is not None
            and float(self.edge_binarize_threshold) < 0
        ):
            raise ValueError("edge_binarize_threshold cannot be negative")
        if (
            self.kernel != "ludvig_release_compat"
            and self.edge_binarize_threshold is not None
        ):
            raise ValueError(
                "RADIO-GS clean diffusion kernels require continuous affinities; "
                "set edge_binarize_threshold=None explicitly"
            )


def normalize_node_features(features: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Match LUDVIG's per-node feature normalization."""

    rows = torch.as_tensor(features).float()
    if rows.ndim != 2 or not bool(torch.isfinite(rows).all()):
        raise ValueError("features must be a finite [num_nodes, dimension] matrix")
    return rows / (rows.norm(dim=-1, keepdim=True) + float(eps))


def cap_positive_reference_evidence(
    evidence: torch.Tensor, *, max_positive_fraction: float
) -> torch.Tensor:
    """Match LUDVIG's argsort-based ``maxpos`` truncation exactly."""

    values = torch.as_tensor(evidence).detach().float().clone().reshape(-1)
    fraction = float(max_positive_fraction)
    if not 0 <= fraction <= 1:
        raise ValueError("max_positive_fraction must be in [0,1]")
    if bool((values < 0).any()) or not bool(torch.isfinite(values).all()):
        raise ValueError("positive reference evidence must be finite and non-negative")
    positive_count = int((values > 0).sum())
    if fraction <= 0 or positive_count == 0:
        return values
    retained = int(fraction * positive_count)
    # The release uses a[:-int(...)].  When int(...) is zero, ``[:-0]`` is
    # empty and therefore (perhaps surprisingly) nothing is removed.
    if retained > 0:
        order = torch.argsort(values)
        values[order[:-retained]] = 0
    return values


def weighted_logistic_query_compatibility(
    normalized_features: torch.Tensor,
    signed_reference_evidence: torch.Tensor,
    reference_weight: torch.Tensor,
    *,
    logistic_c: float = 0.01,
    regularizer_bandwidth: float = 1.0,
    fit_population: LogisticFitPopulation = "auto_release",
) -> torch.Tensor:
    """Fit LUDVIG's reference-only balanced logistic query regularizer.

    Only nodes with non-zero signed evidence enter the fit.  This function has
    no scene labels or target-view inputs; ``reference_weight`` is the exact
    inverse-rendering responsibility accumulated for the reference prompt.
    """

    from sklearn.linear_model import LogisticRegression

    features = torch.as_tensor(normalized_features).detach().float().cpu()
    evidence = torch.as_tensor(signed_reference_evidence).detach().float().cpu().reshape(-1)
    weights = torch.as_tensor(reference_weight).detach().float().cpu().reshape(-1)
    count = int(features.shape[0]) if features.ndim == 2 else -1
    if evidence.shape != (count,) or weights.shape != (count,):
        raise ValueError("reference evidence and weight must align with feature rows")
    if not bool(torch.isfinite(evidence).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("reference evidence and weight must be finite")
    if bool((weights < 0).any()):
        raise ValueError("reference weights cannot be negative")
    mode = str(fit_population)
    if mode == "auto_release":
        mode = "signed_nonzero" if bool((evidence < 0).any()) else "all_nodes_positive_only"
    if mode == "signed_nonzero":
        fit_mask = evidence != 0
    elif mode == "all_nodes_positive_only":
        if bool((evidence < 0).any()):
            raise ValueError("all_nodes_positive_only does not accept negative evidence")
        # The SPIn release passes every node to sklearn, including invisible
        # rows whose inverse-rendering responsibility (sample weight) is zero.
        fit_mask = torch.ones_like(evidence, dtype=torch.bool)
    else:
        raise ValueError(f"unsupported fit_population: {fit_population}")
    labels = evidence[fit_mask] > 0
    if int(fit_mask.sum()) < 2 or int(labels.unique().numel()) != 2:
        raise ValueError("reference-only logistic fit requires both signed classes")
    fit_weights = weights[fit_mask]
    if mode == "signed_nonzero" and not bool((fit_weights > 0).all()):
        raise ValueError("every observed reference node needs positive responsibility")
    if not bool((fit_weights >= 0).all()) or float(fit_weights.sum()) <= 0:
        raise ValueError("reference logistic fit needs non-negative, non-empty weight")
    classifier = LogisticRegression(
        C=float(logistic_c),
        class_weight="balanced",
    ).fit(
        features[fit_mask].numpy(),
        labels.numpy().astype(np.int64, copy=False),
        sample_weight=fit_weights.numpy(),
    )
    probability = torch.from_numpy(
        classifier.predict_proba(features.numpy())[:, 1]
    ).float()
    return probability.pow(1.0 / float(regularizer_bandwidth)).clamp(0.0, 1.0)


def knn_feature_distances(
    normalized_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    distance_chunk_size: int = 32,
) -> torch.Tensor:
    """Compute reusable node-to-kNN distances without materializing N x K x D."""

    features = torch.as_tensor(normalized_features).float()
    neighbors = torch.as_tensor(neighbor_indices, device=features.device).long()
    if neighbors.ndim != 2 or neighbors.shape[0] != features.shape[0]:
        raise ValueError("neighbor_indices must be [num_nodes, num_neighbors]")
    if neighbors.numel() and (
        int(neighbors.min()) < 0 or int(neighbors.max()) >= features.shape[0]
    ):
        raise IndexError("neighbor index is outside the feature bank")
    if int(distance_chunk_size) <= 0:
        raise ValueError("distance_chunk_size must be positive")
    # Never materialize [N,K,D].  Full canonical DINO rows can be 4096-D,
    # making that seemingly innocent broadcast terabyte-scale on a real
    # scene.  N-by-K distances are the only persistent edge tensor.
    distances = torch.empty(
        neighbors.shape, device=features.device, dtype=torch.float32
    )
    for start in range(0, features.shape[0], int(distance_chunk_size)):
        stop = min(start + int(distance_chunk_size), features.shape[0])
        distances[start:stop] = torch.linalg.vector_norm(
            features[start:stop, None, :] - features[neighbors[start:stop]],
            dim=-1,
        )
    return distances


def rbf_similarity_from_distances(
    distances: torch.Tensor,
    *,
    feature_bandwidth: float,
    positive_reference_mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply LUDVIG's reference-conditioned RBF scale to cached distances."""

    distances = torch.as_tensor(distances).float()
    if distances.ndim != 2 or not bool(torch.isfinite(distances).all()):
        raise ValueError("distances must be a finite [num_nodes,K] matrix")
    if bool((distances < 0).any()) or float(feature_bandwidth) <= 0:
        raise ValueError("distances and feature bandwidth must be non-negative/positive")
    if positive_reference_mask is None:
        median = distances.median()
    else:
        mask = torch.as_tensor(
            positive_reference_mask, device=distances.device
        ).bool().reshape(-1)
        if mask.shape != (distances.shape[0],) or not bool(mask.any()):
            raise ValueError("positive_reference_mask must select at least one node")
        # This deliberately matches ``x[mask].median()`` in LUDVIG rather than
        # selecting edges whose two endpoints are both positive.
        median = distances[mask].median()
    scale = float(feature_bandwidth) * median.square().clamp_min(float(eps))
    return torch.exp(-distances.square() / scale)


def rbf_knn_feature_similarity(
    normalized_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    feature_bandwidth: float,
    positive_reference_mask: torch.Tensor | None = None,
    eps: float = 1e-12,
    distance_chunk_size: int = 32,
) -> torch.Tensor:
    """Compute the released LUDVIG RBF on node-to-kNN feature distances."""

    distances = knn_feature_distances(
        normalized_features,
        neighbor_indices,
        distance_chunk_size=distance_chunk_size,
    )
    return rbf_similarity_from_distances(
        distances,
        feature_bandwidth=feature_bandwidth,
        positive_reference_mask=positive_reference_mask,
        eps=eps,
    )


def gate_knn_similarity(
    similarities: torch.Tensor,
    neighbor_indices: torch.Tensor,
    query_compatibility: torch.Tensor,
) -> torch.Tensor:
    """Apply the released ``sqrt(P_i P_j)`` query gate."""

    values = torch.as_tensor(similarities).float()
    neighbors = torch.as_tensor(neighbor_indices, device=values.device).long()
    compatibility = torch.as_tensor(
        query_compatibility, device=values.device
    ).float().reshape(-1)
    if values.shape != neighbors.shape or compatibility.shape != (values.shape[0],):
        raise ValueError("similarities, neighbors, and compatibility do not align")
    if not bool(torch.isfinite(compatibility).all()) or bool(
        ((compatibility < 0) | (compatibility > 1)).any()
    ):
        raise ValueError("query compatibility must be finite and in [0,1]")
    return values * torch.sqrt(
        compatibility[:, None] * compatibility[neighbors]
    )


def ludvig_release_position_normalize(
    similarities: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """Match LUDVIG's N-by-K row/neighbor-position normalization exactly.

    ``Dright`` is the sum of each *kNN column position*, not graph-node degree.
    The released path performs no later node-degree normalization when the
    unary term is absent.
    """

    values = torch.as_tensor(similarities).float()
    if values.ndim != 2 or not bool(torch.isfinite(values).all()):
        raise ValueError("similarities must be a finite [num_nodes, K] matrix")
    dleft = values.sum(dim=1, keepdim=True) + float(eps)
    dright = values.sum(dim=0, keepdim=True) + float(eps)
    return values / (torch.sqrt(dleft) * torch.sqrt(dright))


def _symmetrized_sparse_adjacency(
    neighbor_indices: torch.Tensor,
    similarities: torch.Tensor,
    *,
    edge_binarize_threshold: float | None,
    symmetric_degree_normalize: bool,
    eps: float,
) -> torch.Tensor:
    values = torch.as_tensor(similarities).float()
    neighbors = torch.as_tensor(neighbor_indices, device=values.device).long()
    if values.shape != neighbors.shape:
        raise ValueError("similarities and neighbor_indices must align")
    if edge_binarize_threshold is not None:
        values = (values > float(edge_binarize_threshold)).to(values.dtype)
    node_count, neighbor_count = neighbors.shape
    rows = torch.arange(node_count, device=values.device).repeat_interleave(
        neighbor_count
    )
    cols = neighbors.reshape(-1)
    flat = values.reshape(-1)
    indices = torch.stack(
        [torch.cat([rows, cols]), torch.cat([cols, rows])], dim=0
    )
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.cat([flat, flat]),
        (node_count, node_count),
        device=values.device,
        dtype=values.dtype,
    ).coalesce()
    if symmetric_degree_normalize:
        degree = torch.zeros(node_count, device=values.device, dtype=values.dtype)
        degree.index_add_(0, adjacency.indices()[0], adjacency.values())
        inverse = degree.clamp_min(float(eps)).rsqrt()
        row, col = adjacency.indices()
        adjacency = torch.sparse_coo_tensor(
            adjacency.indices(),
            adjacency.values() * inverse[row] * inverse[col],
            adjacency.shape,
            device=values.device,
            dtype=values.dtype,
        ).coalesce()
    return adjacency


def solve_continuous_query_support(
    observation_probability: torch.Tensor,
    observation_confidence: torch.Tensor,
    neighbor_indices: torch.Tensor,
    feature_similarities: torch.Tensor,
    query_compatibility: torch.Tensor,
    *,
    config: QueryConditionedDiffusionConfig,
) -> torch.Tensor:
    """Minimize the continuous Evidence-to-Support v2 convex energy.

    The unary retains exact reference-mask probability and confidence.  Rows
    without a reference observation receive only a fixed weak fidelity to the
    query classifier, so the graph fills them without making the system
    singular.  High-purity, fully observed foreground/background rows are
    eliminated from the linear system and remain exactly one/zero.

    Each cached directed kNN relation contributes one non-negative undirected
    energy term.  We apply ``sqrt(P_i P_j)`` only to this pairwise conductance,
    then symmetrically endpoint-degree normalize it.  The Laplacian product is
    evaluated from the implicit N-by-K cache, avoiding a doubled sparse COO
    graph on large scenes while preserving a symmetric positive-semidefinite
    operator.  A Jacobi-preconditioned CG solve operates only on free rows.
    """

    if config.kernel != "continuous_convex_v2":
        raise ValueError("continuous query support requires continuous_convex_v2")
    probability = torch.as_tensor(observation_probability).float().reshape(-1)
    confidence = torch.as_tensor(
        observation_confidence, device=probability.device
    ).float().reshape(-1)
    neighbors = torch.as_tensor(
        neighbor_indices, device=probability.device
    ).long()
    similarities = torch.as_tensor(
        feature_similarities, device=probability.device
    ).float()
    compatibility = torch.as_tensor(
        query_compatibility, device=probability.device
    ).float().reshape(-1)
    node_count = probability.numel()
    if (
        confidence.shape != (node_count,)
        or compatibility.shape != (node_count,)
        or neighbors.ndim != 2
        or neighbors.shape[0] != node_count
        or similarities.shape != neighbors.shape
    ):
        raise ValueError("continuous query-support inputs do not align")
    for name, values in (
        ("observation_probability", probability),
        ("observation_confidence", confidence),
        ("feature_similarities", similarities),
        ("query_compatibility", compatibility),
    ):
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"{name} must be finite")
    if bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("observation_probability must be in [0,1]")
    if bool(((confidence < 0) | (confidence > 1)).any()):
        raise ValueError("observation_confidence must be in [0,1]")
    if bool((similarities < 0).any()):
        raise ValueError("feature_similarities must be non-negative")
    if bool(((compatibility < 0) | (compatibility > 1)).any()):
        raise ValueError("query_compatibility must be in [0,1]")
    if neighbors.numel() and (
        int(neighbors.min()) < 0 or int(neighbors.max()) >= node_count
    ):
        raise IndexError("neighbor index is outside the query-support rows")

    observed = confidence > 0
    unary_target = torch.where(observed, probability, compatibility)
    unary_confidence = confidence + (1.0 - confidence) * float(
        config.unobserved_fidelity
    )
    hard_reliable = confidence >= float(config.hard_observation_confidence)
    hard_positive = hard_reliable & (
        probability >= float(config.hard_positive_probability)
    )
    hard_negative = hard_reliable & (
        probability <= float(config.hard_negative_probability)
    )
    fixed = hard_positive | hard_negative
    free = ~fixed
    fixed_values = hard_positive.to(probability.dtype)
    if not bool(free.any()):
        return fixed_values

    rows = torch.arange(node_count, device=probability.device)[:, None]
    nonself = neighbors != rows
    conductance = gate_knn_similarity(
        similarities, neighbors, compatibility
    )
    conductance.mul_(nonself)
    # Treat each cached directed relation as one undirected convex-energy
    # term.  Endpoint degree therefore receives its source and destination
    # mass; duplicate reciprocal kNN relations are harmless duplicate terms.
    degree = conductance.sum(dim=1)
    degree.index_add_(0, neighbors.reshape(-1), conductance.reshape(-1))
    conductance.div_(
        torch.sqrt(
            degree[:, None].clamp_min(float(config.eps))
            * degree[neighbors].clamp_min(float(config.eps))
        )
    )
    edge_diagonal = conductance.sum(dim=1)
    edge_diagonal.index_add_(
        0, neighbors.reshape(-1), conductance.reshape(-1)
    )
    row_chunk = int(config.solver_row_chunk_size)

    def laplacian(vector: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(vector)
        for start in range(0, node_count, row_chunk):
            stop = min(start + row_chunk, node_count)
            chunk_neighbors = neighbors[start:stop]
            chunk_weights = conductance[start:stop]
            delta = vector[start:stop, None] - vector[chunk_neighbors]
            weighted_delta = chunk_weights * delta
            output[start:stop].add_(weighted_delta.sum(dim=1))
            output.index_add_(
                0,
                chunk_neighbors.reshape(-1),
                -weighted_delta.reshape(-1),
            )
        return output

    laplacian_weight = float(config.laplacian_weight)

    def operator(vector: torch.Tensor) -> torch.Tensor:
        masked = vector * free
        return (
            unary_confidence * masked
            + laplacian_weight * laplacian(masked)
        ) * free

    right = (
        unary_confidence * unary_target
        - laplacian_weight * laplacian(fixed_values)
    ) * free
    solution = unary_target * free
    residual = right - operator(solution)
    initial_norm = torch.linalg.vector_norm(residual).clamp_min(float(config.eps))
    inverse_diagonal = (
        unary_confidence + laplacian_weight * edge_diagonal
    ).clamp_min(float(config.eps)).reciprocal()
    preconditioned = inverse_diagonal * residual
    direction = preconditioned.clone()
    residual_product = torch.dot(residual, preconditioned)
    for _ in range(int(config.cg_iterations)):
        product = operator(direction)
        denominator = torch.dot(direction, product).clamp_min(float(config.eps))
        step = residual_product / denominator
        solution.add_(step * direction)
        residual.sub_(step * product)
        if float(torch.linalg.vector_norm(residual) / initial_norm) <= float(
            config.cg_tolerance
        ):
            break
        next_preconditioned = inverse_diagonal * residual
        next_product = torch.dot(residual, next_preconditioned)
        direction.mul_(next_product / residual_product.clamp_min(float(config.eps)))
        direction.add_(next_preconditioned)
        preconditioned = next_preconditioned
        residual_product = next_product
    return (solution * free + fixed_values).clamp(0.0, 1.0)


def _undirected_boolean_propagation(
    initial_active: torch.Tensor,
    neighbor_indices: torch.Tensor,
    directed_edge_mask: torch.Tensor,
    *,
    iterations: int,
    row_chunk_size: int = 32768,
) -> torch.Tensor:
    """Propagate support on the implicit symmetrized kNN graph.

    This is a memory-only execution optimization for the non-negative release
    path.  It computes the same positivity support as repeated multiplication
    by ``D + D.T`` without materializing the doubled COO edge list.  Edge
    weights and per-iteration column normalization cannot change positivity
    when both the initial values and the retained edges are non-negative.
    """

    active = torch.as_tensor(initial_active).bool()
    neighbors = torch.as_tensor(neighbor_indices, device=active.device).long()
    edge_mask = torch.as_tensor(directed_edge_mask, device=active.device).bool()
    if active.ndim == 1:
        active = active[:, None]
    if active.ndim != 2 or neighbors.shape != edge_mask.shape:
        raise ValueError("active rows and directed kNN edges must align")
    if neighbors.shape[0] != active.shape[0]:
        raise ValueError("active rows and directed kNN edges must align")
    if int(iterations) <= 0 or int(row_chunk_size) <= 0:
        raise ValueError("iterations and row_chunk_size must be positive")

    node_count, column_count = active.shape
    for _ in range(int(iterations)):
        next_active = torch.zeros_like(active)
        for start in range(0, node_count, int(row_chunk_size)):
            stop = min(start + int(row_chunk_size), node_count)
            chunk_neighbors = neighbors[start:stop]
            chunk_edges = edge_mask[start:stop]
            # D @ active: a directed i->j kNN edge contributes j to row i.
            for column in range(column_count):
                next_active[start:stop, column] = (
                    chunk_edges & active[chunk_neighbors, column]
                ).any(dim=1)
                # D.T @ active: the same directed edge also contributes i to
                # row j after the release symmetrization.  Chunking bounds the
                # temporary repeated destination list on million-node scenes.
                reverse_mask = chunk_edges & active[start:stop, column, None]
                reverse_destinations = chunk_neighbors[reverse_mask]
                if reverse_destinations.numel():
                    next_active[:, column].index_fill_(
                        0, reverse_destinations, True
                    )
        if torch.equal(next_active, active):
            active = next_active
            break
        active = next_active
    return active


def run_query_conditioned_diffusion(
    initial_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    feature_similarities: torch.Tensor,
    query_compatibility: torch.Tensor,
    *,
    config: QueryConditionedDiffusionConfig,
) -> torch.Tensor:
    """Gate, normalize, symmetrize, and diffuse one reference query."""

    initial = torch.as_tensor(initial_features).float()
    if initial.ndim == 1:
        initial = initial[:, None]
    compatibility = torch.as_tensor(
        query_compatibility, device=initial.device
    ).float().reshape(-1)
    if initial.ndim != 2 or compatibility.shape != (initial.shape[0],):
        raise ValueError("initial features and compatibility do not align")
    gated = gate_knn_similarity(
        torch.as_tensor(feature_similarities, device=initial.device),
        torch.as_tensor(neighbor_indices, device=initial.device),
        compatibility,
    )
    if config.kernel == "ludvig_release_compat":
        # Normalize in-place here to avoid retaining a second N-by-K float
        # tensor.  The public helper remains side-effect free.
        dleft = gated.sum(dim=1, keepdim=True) + float(config.eps)
        dright = gated.sum(dim=0, keepdim=True) + float(config.eps)
        gated.div_(torch.sqrt(dleft))
        gated.div_(torch.sqrt(dright))
        degree_normalize = False
    else:
        degree_normalize = True
    # The registered SPIn release path starts from positive-only evidence and
    # ends with only ``diffused > 0``.  On non-negative edges its output is
    # exactly graph reachability.  Keeping the graph implicit avoids the 2NK
    # COO materialization that otherwise exceeds 24 GiB on fern (1.315M
    # primitives).  Signed inputs and the clean kernel retain the numeric COO
    # execution below.
    if config.kernel == "ludvig_release_compat" and bool((initial >= 0).all()):
        if config.edge_binarize_threshold is None:
            directed_edge_mask = gated > 0
        else:
            directed_edge_mask = gated > float(config.edge_binarize_threshold)
        active = _undirected_boolean_propagation(
            initial > 0,
            torch.as_tensor(neighbor_indices, device=initial.device),
            directed_edge_mask,
            iterations=int(config.iterations),
        )
        return active.to(torch.float32) * compatibility[:, None]
    adjacency = _symmetrized_sparse_adjacency(
        torch.as_tensor(neighbor_indices, device=initial.device),
        gated,
        edge_binarize_threshold=config.edge_binarize_threshold,
        symmetric_degree_normalize=degree_normalize,
        eps=config.eps,
    )
    propagated = initial
    for _ in range(int(config.iterations)):
        # Released LUDVIG normalizes each feature column over graph nodes.
        propagated = propagated / (
            propagated.norm(dim=0, keepdim=True) + float(config.eps)
        )
        propagated = torch.sparse.mm(adjacency, propagated)
    if config.kernel == "ludvig_release_compat":
        # This is intentionally a reachability mask followed by P, matching
        # ``(diffused_features > 0) * reg_similarities`` in the release.
        propagated = (propagated > 0).to(torch.float32) * compatibility[:, None]
    return propagated


def compute_query_conditioned_support(
    features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    signed_reference_evidence: torch.Tensor,
    reference_weight: torch.Tensor,
    *,
    config: QueryConditionedDiffusionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """End-to-end reference-only support and its query compatibility P."""

    normalized = normalize_node_features(features, eps=config.eps)
    compatibility_cpu = weighted_logistic_query_compatibility(
        normalized,
        signed_reference_evidence,
        reference_weight,
        logistic_c=config.logistic_c,
        regularizer_bandwidth=config.regularizer_bandwidth,
        fit_population=config.logistic_fit_population,
    )
    compatibility = compatibility_cpu.to(normalized.device)
    evidence = torch.as_tensor(
        signed_reference_evidence, device=normalized.device
    ).float().reshape(-1)
    similarities = rbf_knn_feature_similarity(
        normalized,
        neighbor_indices,
        feature_bandwidth=config.feature_bandwidth,
        positive_reference_mask=evidence > 0,
        eps=config.eps,
        distance_chunk_size=config.distance_chunk_size,
    )
    initial = evidence[:, None]
    support = run_query_conditioned_diffusion(
        initial,
        neighbor_indices,
        similarities,
        compatibility,
        config=config,
    )
    return support.squeeze(1), compatibility
