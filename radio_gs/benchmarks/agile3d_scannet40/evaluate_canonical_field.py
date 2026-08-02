#!/usr/bin/env python3
"""Direct canonical-field AGILE3D prediction without an observation-domain lift."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.field.field_signature import FeatureSpaceSignature
from radio_gs.field.observation_lifting_contract import (
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES,
    validate_observation_contract_metadata,
)
from radio_gs.interfaces.capability_cache import (
    load_canonical_capability_bank,
    load_canonical_primitive_reliability,
    load_canonical_support_graph,
)
from radio_gs.querying.evidence_scorer import EvidenceScoringConfig
from radio_gs.querying.query_compilers import (
    compile_world_3d_query,
    continuous_gaussian_readout,
)
from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.query_spec import SelectionMode
from radio_gs.querying.support_solver import PrimitiveSupportGraph, SupportSolverConfig
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _build_hybrid_model

from .frozen_full312_contract import (
    bind_frozen_method_contract,
    source_contract_bindings_sha256,
)
from .protocol import (
    Click,
    aggregate_official_metrics,
    evaluate_interactive_predictions,
    interaction_health_metrics,
    load_official_object_list,
    quantize_scannet_points,
)


# A unit-opacity Gaussian at three standard deviations contributes
# exp(-9/2) ~= 0.011.  The full-observation support gate therefore does not
# count an official point as reconstructed merely because of a five-sigma
# Gaussian tail.  This is geometry-only and fixed before evaluator labels are
# opened; it is not a class/object-specific threshold.
FULL_OBSERVATION_MIN_MEANINGFUL_SUPPORT = 1e-2
FULL_OBSERVATION_DIAGNOSTIC_CONTRACT = "scannet_full_observation_diagnostic_v1"
FULL_OBSERVATION_SUPPORT_CONTRACTS = frozenset(
    {
        "scannet_full_observation_pilot",
        "scannet_full_observation_v1",
        "scannet_full_observation_pfpr_queryheldout_v1",
        FULL_OBSERVATION_DIAGNOSTIC_CONTRACT,
    }
)


def validate_observation_contract(
    contract: str,
    source_observation_root: str,
    *,
    require_support_gate: bool,
    allow_ungated_diagnostic: bool = False,
    declared_source_contract: str = "",
    field_source_contract_sha256: str = "",
    field_source_contract_version: str = "",
) -> None:
    """Reject a sparse convenience frame source from the full-observation mode."""

    contract = str(contract)
    source = str(source_observation_root)
    full_observation_contracts = {
        "scannet_full_observation_pilot",
        "scannet_full_observation_v1",
        FULL_OBSERVATION_DIAGNOSTIC_CONTRACT,
    }
    if contract not in {
        "dense_overlap_pilot",
        "dense_pfpr_queryheldout_pilot",
        "dense_agile_all_observations_pilot",
        *full_observation_contracts,
    }:
        raise ValueError("unsupported AGILE3D observation contract")
    if contract in full_observation_contracts:
        if contract != FULL_OBSERVATION_DIAGNOSTIC_CONTRACT and not bool(require_support_gate):
            raise ValueError(f"{contract} requires the support gate")
        if contract == FULL_OBSERVATION_DIAGNOSTIC_CONTRACT:
            if bool(require_support_gate) or not bool(allow_ungated_diagnostic):
                raise ValueError(
                    "the ungated full-observation diagnostic requires the explicit "
                    "diagnostic opt-in and cannot be a formal result"
                )
        if "scannet_frames_25k" in source:
            raise ValueError(
                f"{contract} cannot use scannet_frames_25k"
            )
        if str(declared_source_contract) != "scannet_full_observation_v1":
            raise ValueError(
                f"{contract} requires fields that explicitly declare "
                "scannet_full_observation_v1 provenance"
            )
        if not str(field_source_contract_sha256):
            raise ValueError(
                f"{contract} requires an auditable field-source "
                "contract digest"
            )
        if str(field_source_contract_version) != "scannet_full_observation_v1":
            raise ValueError(
                f"{contract} requires matching source-contract version"
            )
    # Named pilot protocols must likewise fail closed on an accidentally
    # reused field.  The legacy dense-overlap pilot predates explicit source
    # declarations and remains readable only under its historic name.
    expected_declaration = {
        "dense_pfpr_queryheldout_pilot": "dense_pfpr_queryheldout_v1",
        "dense_agile_all_observations_pilot": "dense_agile_all_observations_pilot",
    }.get(contract)
    if expected_declaration is not None and str(declared_source_contract) != expected_declaration:
        raise ValueError(
            f"{contract} requires fields that explicitly declare "
            f"{expected_declaration!r} provenance"
        )
    if expected_declaration is not None and not str(field_source_contract_sha256):
        raise ValueError(
            f"{contract} requires an auditable field-source contract digest"
        )
    expected_source_version = {
        "dense_pfpr_queryheldout_pilot": "scannet-pfpr-query-heldout-field-v1",
        "dense_agile_all_observations_pilot": "scannet-agile-dense-observation-field-v1",
    }.get(contract)
    if expected_source_version is not None and str(field_source_contract_version) != expected_source_version:
        raise ValueError(
            f"{contract} requires matching source-contract version "
            f"{expected_source_version!r}"
        )


def validate_continuous_support_threshold(
    observation_contract: str,
    support_threshold: float,
) -> None:
    """Require a non-tail support definition for full ScanNet observation fields."""

    threshold = float(support_threshold)
    if threshold < 0:
        raise ValueError("continuous support threshold must be non-negative")
    if (
        str(observation_contract) in FULL_OBSERVATION_SUPPORT_CONTRACTS
        and threshold < FULL_OBSERVATION_MIN_MEANINGFUL_SUPPORT
    ):
        raise ValueError(
            "full-observation AGILE requires meaningful continuous support "
            f">={FULL_OBSERVATION_MIN_MEANINGFUL_SUPPORT:g}; got {threshold:g}"
        )


def validate_full_observation_mpr_contract(
    observation_contract: str,
    mpr_metadata: Mapping[str, object],
    *,
    expected_source_contract_sha256: str,
    expected_source_contract_version: str,
) -> None:
    """Require dense canonical evidence, not merely dense Gaussian geometry.

    A full-.sens render contract by itself is insufficient: an old 120-view
    temporal MPR cache can be paired with it after geometry has been rebuilt.
    The full-observation AGILE protocol therefore accepts a field only if its
    raw teacher cache declares the matching coverage-ranked MPR contract and
    was selected from exactly the same label-free source manifest.
    """

    if str(observation_contract) not in {
        "scannet_full_observation_pilot",
        "scannet_full_observation_v1",
        FULL_OBSERVATION_DIAGNOSTIC_CONTRACT,
    }:
        return
    metadata = dict(mpr_metadata)
    declared = metadata.get("observation_lifting_contract")
    declared_name = (
        str(declared.get("name", "")) if isinstance(declared, Mapping) else ""
    )
    if declared_name not in CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES:
        accepted = ", ".join(sorted(CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES))
        raise ValueError(
            "full-observation AGILE requires one of the coverage-ranked canonical "
            f"full MPR contracts ({accepted}), got {declared_name or '<missing>'!r}"
        )
    try:
        validate_observation_contract_metadata(
            metadata,
            require_declaration=True,
            contract_name=declared_name,
        )
    except ValueError as error:
        raise ValueError(
            "full-observation AGILE requires "
            f"{declared_name}: {error}"
        ) from error
    if metadata.get("full_observation_coverage_order_applied") is not True:
        raise ValueError(
            "full-observation AGILE field lacks the coverage-ranked MPR audit"
        )
    if str(metadata.get("full_observation_source_contract_sha256", "")) != str(
        expected_source_contract_sha256
    ):
        raise ValueError(
            "full-observation MPR source-contract digest differs from the "
            "geometry source"
        )
    if str(metadata.get("full_observation_source_contract_version", "")) != str(
        expected_source_contract_version
    ):
        raise ValueError(
            "full-observation MPR source-contract version differs from the "
            "geometry source"
        )


def validate_capability_teacher_fidelity(
    metadata: Mapping[str, object],
    *,
    require_official_extracted: bool,
) -> dict[str, object]:
    """Optionally require a field trained against native official adaptor maps.

    This is an artifact-provenance gate, not an AGILE evaluator setting: it
    leaves released clicks, 5 cm points, prediction readout, and metrics
    untouched.  It prevents a high-spatial-fidelity field label from silently
    resolving to the historic ``project_raw`` teacher path.
    """

    sources = metadata.get("capability_training_mpr_sources", {})
    render_source = str(metadata.get("render_capability_teacher_source", ""))
    if not isinstance(sources, Mapping):
        raise ValueError("canonical capability cache has invalid teacher provenance")
    normalized_sources: dict[str, str] = {}
    for name in ("appearance", "boundary"):
        item = sources.get(name, {})
        if not isinstance(item, Mapping):
            raise ValueError(
                f"canonical capability cache has invalid {name} teacher provenance"
            )
        source = str(item.get("capability_map_source", ""))
        normalized_sources[name] = source
        if source == "official_extracted":
            if (
                str(item.get("capability_adaptor_execution", ""))
                != "official_c_radio_runtime_adaptor_output"
            ):
                raise ValueError(
                    f"canonical {name} teacher is labeled official_extracted "
                    "without official runtime adaptor provenance"
                )
    if require_official_extracted:
        invalid = {
            name: source
            for name, source in normalized_sources.items()
            if source != "official_extracted"
        }
        if invalid or render_source != "official_extracted":
            raise ValueError(
                "this evaluation requires native official C-RADIO capability "
                f"teachers; MPR={invalid or normalized_sources}, render={render_source!r}"
            )
    return {
        "capability_training_mpr_sources": normalized_sources,
        "render_capability_teacher_source": render_source,
        "requires_official_extracted_capability_teachers": bool(
            require_official_extracted
        ),
    }


def observation_source_from_render_contract(config_path: Path) -> dict[str, str]:
    """Read source provenance from the field render contract, not its path.

    Historic field caches predate the explicit full-observation manifest.  They
    remain valid only for a named pilot.  A future full ScanNet field must add
    ``observation_contract: scannet_full_observation_v1`` to this render
    contract, which makes the formal evaluator fail closed if provenance is
    absent or ambiguous.
    """

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"field render contract is missing: {path}")
    config = load_config(str(path))
    source_root = str(config.scene_root)
    if not source_root:
        raise ValueError(f"render contract has no scene_root: {path}")
    # ``load_config`` intentionally drops unknown metadata.  Read this one
    # explicit audit field from YAML without changing the training schema.
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    declared = str(raw.get("observation_contract", "field_only_dense_rgbd_v1"))
    return {
        "source_observation_root": str(Path(source_root).resolve()),
        "declared_source_contract": declared,
        "field_source_contract_sha256": str(
            raw.get("field_source_contract_sha256", "")
        ),
        "field_source_contract_version": str(
            raw.get("field_source_contract_version", "")
        ),
        "field_source_frame_manifest_sha256": str(
            raw.get("field_source_frame_manifest_sha256", "")
        ),
        "field_source_excluded_query_frame_count": str(
            raw.get("field_source_excluded_query_frame_count", "")
        ),
        "field_source_excluded_query_frame_ids_sha256": str(
            raw.get("field_source_excluded_query_frame_ids_sha256", "")
        ),
    }


def stable_support_record(record: Mapping[str, object]) -> dict[str, object]:
    """Return the label-free field-support identity independent of cache reuse.

    Geometry cache reuse is an execution detail: the first preflight naturally
    materializes a cache and the evaluator pass reuses it.  It must not turn a
    byte-identical field into an apparent support-contract change.
    """

    return {
        str(key): value
        for key, value in dict(record).items()
        if str(key) != "geometry_cache_reused"
    }


def select_object_shard(
    objects: Sequence[object],
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[object]:
    """Select a deterministic disjoint AGILE object shard by released order.

    Object trajectories are conditionally independent once the field and the
    released click simulator are frozen.  This helper is therefore only an
    execution partition: it neither changes clicks, the official 5 cm point
    domain, solver settings, nor any metric.  A shard result is explicitly
    marked partial and can be promoted only through the strict merge routine,
    which verifies that every released object appears exactly once.
    """

    count = int(shard_count)
    index = int(shard_index)
    if count <= 0 or index < 0 or index >= count:
        raise ValueError("object shard index/count are invalid")
    return [item for position, item in enumerate(objects) if position % count == index]


def _nearest_candidate_indices(
    source_xyz: torch.Tensor,
    target_xyz: np.ndarray,
    *,
    count: int,
) -> torch.Tensor:
    source = torch.as_tensor(source_xyz).float().cpu().numpy()
    target = np.asarray(target_xyz, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3 or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("source_xyz and target_xyz must be finite [N,3] arrays")
    if count <= 0:
        raise ValueError("readout candidate count must be positive")
    _distance, indices = cKDTree(source).query(target, k=min(int(count), len(source)))
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices[:, None]
    return torch.from_numpy(np.ascontiguousarray(indices))


def constrain_released_click_scores(
    scores: torch.Tensor,
    *,
    positive_indices: Sequence[int],
    negative_indices: Sequence[int],
    mode: str = "none",
) -> torch.Tensor:
    """Apply an optional, label-free constraint at released click locations.

    This is method-side readout, before the AGILE evaluator's mandatory
    overwrite.  It consumes only the released interaction events and leaves
    every non-clicked official point unchanged.  The continuous readout still
    uses its frozen candidate indices, Gaussian responsibilities and opacity;
    this final constraint preserves the defining semantics of an interactive
    prompt at the exact sampled point.

    ``none`` is the backwards-compatible default.  ``click_score_clamp`` is a
    named method variant and must be reported explicitly.
    """

    value = torch.as_tensor(scores)
    if value.ndim != 1:
        raise ValueError("official point scores must be one-dimensional")
    if mode not in {"none", "click_score_clamp"}:
        raise ValueError("point readout constraint must be none or click_score_clamp")
    if mode == "none":
        return value
    positive = torch.as_tensor(positive_indices, dtype=torch.long, device=value.device)
    negative = torch.as_tensor(negative_indices, dtype=torch.long, device=value.device)
    all_indices = torch.cat((positive, negative))
    if all_indices.numel() and (
        bool((all_indices < 0).any()) or bool((all_indices >= value.numel()).any())
    ):
        raise IndexError("released click index is outside the official point domain")
    if positive.numel() and negative.numel():
        if bool(torch.isin(positive, negative).any()):
            raise ValueError("one released point cannot have conflicting click signs")
    constrained = value.clone()
    constrained[positive] = 1.0
    constrained[negative] = 0.0
    return constrained


class CanonicalFieldPointPredictor:
    """AGILE callback backed by the shared world-query compiler and field graph."""

    def __init__(
        self,
        *,
        gaussian_xyz: torch.Tensor,
        gaussian_covariance: torch.Tensor,
        gaussian_precision: torch.Tensor,
        gaussian_opacity: torch.Tensor,
        appearance_features: torch.Tensor,
        boundary_features: torch.Tensor,
        appearance_signature: FeatureSpaceSignature,
        boundary_signature: FeatureSpaceSignature,
        graph: PrimitiveSupportGraph,
        official_xyz: np.ndarray,
        device: str,
        solver_config: SupportSolverConfig,
        readout_candidate_k: int = 64,
        readout_support_threshold: float = 1e-6,
        evaluation_voxel_size_m: float = 0.0,
        click_seed_kernel: str = "native_gaussian",
        seed_candidate_k: int = 64,
        seed_topk: int = 0,
        seed_temperature: float = 1.0,
        prototype_count: int = 4,
        prototype_strategy: str = "weighted_fps",
        world_point_prototype_mode: str = "per_click_local",
        world_point_max_prototypes: int = 0,
        world_point_prototype_weighting: str = "support_mass",
        appearance_unary_weight: float = 1.0,
        boundary_unary_weight: float = 0.35,
        feature_calibration: str = "none",
        background_centroids: int = 0,
        background_negative_policy: str = "pooled_mean",
        calibration_sample_size: int = 8192,
        centroid_iterations: int = 4,
        score_calibration: str = "none",
        score_chunk_size: int = 8192,
        graph_policy: str = "typed",
        channel_confidence_mode: str = "none",
        negative_spatial_mode: str = "none",
        negative_spatial_steps: int = 4,
        negative_spatial_decay: float = 0.8,
        node_reliability: torch.Tensor | None = None,
        point_readout_constraint: str = "none",
        selection_mode: SelectionMode | str = SelectionMode.SEEDED_COMPONENT,
    ) -> None:
        self.device = torch.device(device)
        self.gaussian_xyz = torch.as_tensor(gaussian_xyz, device=self.device).float()
        self.gaussian_covariance = torch.as_tensor(
            gaussian_covariance, device=self.device
        ).float()
        self.gaussian_precision = torch.as_tensor(
            gaussian_precision, device=self.device
        ).float()
        self.gaussian_opacity = torch.as_tensor(
            gaussian_opacity, device=self.device
        ).float().reshape(-1)
        count = int(self.gaussian_xyz.shape[0])
        if (
            self.gaussian_xyz.shape != (count, 3)
            or self.gaussian_covariance.shape != (count, 3, 3)
            or self.gaussian_precision.shape != (count, 3, 3)
            or self.gaussian_opacity.shape != (count,)
        ):
            raise ValueError("Gaussian geometry tensors must align as [N,3]/[N,3,3]")
        if not bool(torch.isfinite(self.gaussian_xyz).all()) or not bool(
            torch.isfinite(self.gaussian_covariance).all()
        ):
            raise ValueError("Gaussian geometry must be finite")
        if bool((self.gaussian_opacity < 0).any()):
            raise ValueError("Gaussian opacity must be non-negative")
        self.evaluation_voxel_size_m = float(evaluation_voxel_size_m)
        if self.evaluation_voxel_size_m < 0:
            raise ValueError("evaluation_voxel_size_m must be non-negative")
        self.click_seed_kernel = str(click_seed_kernel)
        if self.click_seed_kernel not in {
            "native_gaussian",
            "evaluator_voxel_convolved",
        }:
            raise ValueError(
                "click_seed_kernel must be native_gaussian or "
                "evaluator_voxel_convolved"
            )
        # The released AGILE domain is a 5 cm voxel cell, not an infinitesimal
        # mathematical point.  Read the field convolved with that fixed cell;
        # for a uniform cell, each coordinate has variance h^2 / 12.  This
        # replaces a hand-set 10 cm observation bridge with evaluator-domain
        # geometry and is identical for every query/object in a run.
        self.voxel_cell_variance_m2 = self.evaluation_voxel_size_m**2 / 12.0
        if self.voxel_cell_variance_m2 > 0:
            identity = torch.eye(3, device=self.device, dtype=self.gaussian_covariance.dtype)
            self.readout_precision = torch.linalg.pinv(
                self.gaussian_covariance + self.voxel_cell_variance_m2 * identity
            )
        else:
            self.readout_precision = self.gaussian_precision
        # A released AGILE callback coordinate is an exact retained point from
        # the source PLY.  The evaluator's 5 cm cell only describes the
        # *output* aggregation domain.  Keep the world-click responsibility
        # native by default so an interaction does not acquire an artificial
        # half-cell blur; retain the convolved option only as a reproducible
        # geometric ablation.
        self.seed_precision = (
            self.gaussian_precision
            if self.click_seed_kernel == "native_gaussian"
            else self.readout_precision
        )

        appearance = torch.as_tensor(appearance_features, device=self.device).float()
        boundary = torch.as_tensor(boundary_features, device=self.device).float()
        if appearance.ndim != 2 or boundary.ndim != 2 or appearance.shape[0] != count or boundary.shape[0] != count:
            raise ValueError("capability features must align with Gaussian geometry")
        if graph.num_nodes != count:
            raise ValueError("support graph must align with Gaussian geometry")
        self.feature_banks = {
            "appearance": F.normalize(appearance, dim=-1, eps=1e-8),
            "boundary": F.normalize(boundary, dim=-1, eps=1e-8),
        }
        self.feature_signatures = {
            "appearance": appearance_signature,
            "boundary": boundary_signature,
        }
        self.solver_config = solver_config
        self.seed_candidate_k = int(seed_candidate_k)
        self.seed_topk = int(seed_topk)
        self.seed_temperature = float(seed_temperature)
        if self.seed_topk < 0:
            raise ValueError("seed_topk cannot be negative")
        if self.seed_temperature <= 0:
            raise ValueError("seed_temperature must be positive")
        self.prototype_count = int(prototype_count)
        self.prototype_strategy = str(prototype_strategy)
        self.world_point_prototype_mode = str(world_point_prototype_mode)
        self.world_point_max_prototypes = int(world_point_max_prototypes)
        self.world_point_prototype_weighting = str(world_point_prototype_weighting)
        if self.world_point_prototype_mode not in {"aggregate_fps", "per_click_local"}:
            raise ValueError("unsupported world point prototype mode")
        if self.world_point_max_prototypes < 0:
            raise ValueError("world_point_max_prototypes cannot be negative")
        if self.world_point_prototype_weighting not in {"support_mass", "equal_click"}:
            raise ValueError(
                "world_point_prototype_weighting must be support_mass or equal_click"
            )
        self.engine = CanonicalQueryEngine(
            graph.to(self.device),
            scoring_config=EvidenceScoringConfig(
                semantic_weight=0.0,
                appearance_weight=float(appearance_unary_weight),
                boundary_weight=float(boundary_unary_weight),
                feature_calibration=str(feature_calibration),
                background_centroids=int(background_centroids),
                background_negative_policy=str(background_negative_policy),
                calibration_sample_size=int(calibration_sample_size),
                centroid_iterations=int(centroid_iterations),
                score_calibration=str(score_calibration),
                score_chunk_size=int(score_chunk_size),
                negative_spatial_mode=str(negative_spatial_mode),
                negative_spatial_steps=int(negative_spatial_steps),
                negative_spatial_decay=float(negative_spatial_decay),
            ),
            solver_config=solver_config,
            graph_policy=str(graph_policy),
            component_graph_policy="same",
            channel_confidence_mode=str(channel_confidence_mode),
            node_reliability=node_reliability,
        )
        self.primitive_reliability_applied = node_reliability is not None
        self.official_xyz = np.asarray(official_xyz, dtype=np.float32)
        if self.official_xyz.ndim != 2 or self.official_xyz.shape[1] != 3:
            raise ValueError("official_xyz must be [P,3]")
        self.readout_candidate_indices = _nearest_candidate_indices(
            self.gaussian_xyz,
            self.official_xyz,
            count=int(readout_candidate_k),
        ).to(self.device)
        probe = torch.ones(count, dtype=torch.float32, device=self.device)
        _values, support = continuous_gaussian_readout(
            self.gaussian_xyz,
            self.gaussian_covariance,
            probe,
            torch.as_tensor(self.official_xyz, device=self.device),
            gaussian_precision=self.readout_precision,
            opacity=self.gaussian_opacity,
            candidate_k=int(readout_candidate_k),
            candidate_indices=self.readout_candidate_indices,
        )
        self.readout_support = support
        self.readout_support_threshold = float(readout_support_threshold)
        if self.readout_support_threshold < 0:
            raise ValueError("readout_support_threshold must be non-negative")
        self.readout_valid = support >= self.readout_support_threshold
        self.point_readout_constraint = str(point_readout_constraint)
        if self.point_readout_constraint not in {"none", "click_score_clamp"}:
            raise ValueError(
                "point_readout_constraint must be none or click_score_clamp"
            )
        self.selection_mode = SelectionMode(selection_mode)
        # Populated after each callback for diagnostics only.  The evaluator
        # never feeds these values back into click selection or prediction.
        self.last_seed_satisfaction_stages: dict[
            str, dict[str, float | None]
        ] = {}

    @staticmethod
    def _summarize_click_matches(
        positive_matches: Sequence[bool],
        negative_matches: Sequence[bool],
    ) -> dict[str, float | None]:
        positive = [bool(value) for value in positive_matches]
        negative = [bool(value) for value in negative_matches]
        combined = positive + negative
        return {
            "positive": float(np.mean(positive)) if positive else None,
            "negative": float(np.mean(negative)) if negative else None,
            "all": float(np.mean(combined)) if combined else None,
        }

    def protocol_report(self) -> dict[str, object]:
        support_quantiles = torch.quantile(
            self.readout_support.float(),
            torch.tensor(
                [0.0, 0.01, 0.05, 0.10, 0.50, 0.90, 1.0],
                device=self.readout_support.device,
            ),
        )
        return {
            "world_query_compiler": "compile_world_3d_query",
            "point_seed_lifting": "gaussian_mahalanobis",
            "official_point_readout": "continuous_opacity_weighted_gaussian",
            "point_readout_constraint": self.point_readout_constraint,
            "observation_lift": "none",
            "selection_mode": self.selection_mode.value,
            "world_point_prototype_mode": self.world_point_prototype_mode,
            "world_point_max_prototypes": int(self.world_point_max_prototypes),
            "world_point_prototype_weighting": self.world_point_prototype_weighting,
            "support_graph_edge_channels": sorted(self.engine.graph.edge_channels),
            "support_graph_policy": self.engine.graph_policy,
            "channel_confidence_mode": self.engine.channel_confidence_mode,
            "negative_spatial_mode": self.engine.scoring_config.negative_spatial_mode,
            "negative_spatial_steps": int(
                self.engine.scoring_config.negative_spatial_steps
            ),
            "negative_spatial_decay": float(
                self.engine.scoring_config.negative_spatial_decay
            ),
            "spatial_log_weight": float(
                self.engine.scoring_config.spatial_log_weight
            ),
            "spatial_floor": float(self.engine.scoring_config.spatial_floor),
            "readout_candidate_count": int(self.readout_candidate_indices.shape[1]),
            "readout_support_threshold": float(self.readout_support_threshold),
            "continuous_support_quantiles": {
                name: float(value)
                for name, value in zip(
                    ("p00", "p01", "p05", "p10", "p50", "p90", "p100"),
                    support_quantiles.detach().cpu().tolist(),
                )
            },
            "readout_kernel": (
                "gaussian_convolved_with_evaluator_voxel_cell"
                if self.voxel_cell_variance_m2 > 0
                else "native_gaussian"
            ),
            "evaluation_voxel_size_m": float(self.evaluation_voxel_size_m),
            "voxel_cell_variance_m2": float(self.voxel_cell_variance_m2),
            "click_seed_kernel": self.click_seed_kernel,
            "hard_seed_topk": int(self.seed_topk),
            "seed_temperature": float(self.seed_temperature),
            "hard_seed_threshold": float(self.solver_config.hard_seed_threshold),
            "hard_seed_conflict_policy": str(
                self.solver_config.hard_seed_conflict_policy
            ),
            "hard_seed_conflict_margin": float(
                self.solver_config.hard_seed_conflict_margin
            ),
            "feature_calibration": self.engine.scoring_config.feature_calibration,
            "background_centroids": int(self.engine.scoring_config.background_centroids),
            "background_negative_policy": self.engine.scoring_config.background_negative_policy,
            "calibration_sample_size": int(
                self.engine.scoring_config.calibration_sample_size
            ),
            "centroid_iterations": int(self.engine.scoring_config.centroid_iterations),
            "score_calibration": self.engine.scoring_config.score_calibration,
            "score_chunk_size": int(self.engine.scoring_config.score_chunk_size),
            "primitive_reliability_applied": bool(
                self.primitive_reliability_applied
            ),
            "continuous_support_fraction": float(self.readout_valid.float().mean()),
            "labels_opened": False,
        }

    @torch.inference_mode()
    def __call__(
        self,
        coordinates: np.ndarray,
        _previous: np.ndarray,
        clicks: Sequence[Click],
    ) -> np.ndarray:
        points = np.asarray(coordinates, dtype=np.float32)
        if points.shape != self.official_xyz.shape or not np.allclose(
            points, self.official_xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError("AGILE callback coordinates do not match its fixed official domain")
        positive_indices = [int(item.point_index) for item in clicks if item.is_positive]
        negative_indices = [int(item.point_index) for item in clicks if not item.is_positive]
        if not positive_indices:
            return np.zeros(len(points), dtype=bool)
        if min(positive_indices + negative_indices) < 0 or max(positive_indices + negative_indices) >= len(points):
            raise IndexError("AGILE click is outside the official point domain")
        positive_index_tensor = torch.as_tensor(
            positive_indices, dtype=torch.long, device=self.device
        )
        negative_index_tensor = torch.as_tensor(
            negative_indices, dtype=torch.long, device=self.device
        )
        point_tensor = torch.as_tensor(points, device=self.device)
        positive_points = point_tensor[positive_index_tensor]
        negative_points = (
            point_tensor[negative_index_tensor]
            if negative_indices
            else None
        )
        query = compile_world_3d_query(
            self.gaussian_xyz,
            self.gaussian_covariance,
            positive_points,
            appearance_features=self.feature_banks["appearance"],
            boundary_features=self.feature_banks["boundary"],
            appearance_signature=self.feature_signatures["appearance"],
            boundary_signature=self.feature_signatures["boundary"],
            negative_points=negative_points,
            prototype_count=self.prototype_count,
            prototype_strategy=self.prototype_strategy,
            world_point_prototype_mode=self.world_point_prototype_mode,
            world_point_max_prototypes=self.world_point_max_prototypes,
            world_point_prototype_weighting=self.world_point_prototype_weighting,
            scene_mean_negative=True,
            gaussian_precision=self.seed_precision,
            euclidean_candidate_k=self.seed_candidate_k,
            seed_topk=self.seed_topk,
            seed_temperature=self.seed_temperature,
            selection_mode=self.selection_mode,
        )
        result = self.engine.execute(
            query, self.feature_banks, feature_signatures=self.feature_signatures
        )
        probability, _probability_support = continuous_gaussian_readout(
            self.gaussian_xyz,
            self.gaussian_covariance,
            result.probabilities,
            point_tensor,
            gaussian_precision=self.readout_precision,
            opacity=self.gaussian_opacity,
            candidate_k=int(self.readout_candidate_indices.shape[1]),
            candidate_indices=self.readout_candidate_indices,
        )
        component, _component_support = continuous_gaussian_readout(
            self.gaussian_xyz,
            self.gaussian_covariance,
            result.selected_support.float(),
            point_tensor,
            gaussian_precision=self.readout_precision,
            opacity=self.gaussian_opacity,
            candidate_k=int(self.readout_candidate_indices.shape[1]),
            candidate_indices=self.readout_candidate_indices,
        )
        probability = constrain_released_click_scores(
            probability,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            mode=self.point_readout_constraint,
        )
        component = constrain_released_click_scores(
            component,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            mode=self.point_readout_constraint,
        )
        prediction = (
            (probability >= float(self.solver_config.support_threshold))
            & (component >= float(self.solver_config.support_threshold))
            & self.readout_valid
        )
        if self.point_readout_constraint == "click_score_clamp":
            # A click is itself an observation in the fixed official domain;
            # do not let a query-free support gate erase that exact evidence.
            prediction[positive_index_tensor] = True
            prediction[negative_index_tensor] = False
        threshold = float(self.solver_config.support_threshold)

        def primitive_group_matches(groups, *, positive: bool) -> list[bool]:
            if groups is None:
                return []
            weights = groups.weights.to(
                device=result.probabilities.device,
                dtype=result.probabilities.dtype,
            )
            mass = weights.sum(dim=0).clamp_min(1e-12)
            scores = (weights * result.probabilities[:, None]).sum(dim=0) / mass
            labels = scores >= threshold
            return (
                labels.detach().cpu().tolist()
                if positive
                else (~labels).detach().cpu().tolist()
            )

        primitive_positive = primitive_group_matches(
            query.positive_seed_groups, positive=True
        )
        primitive_negative = primitive_group_matches(
            query.negative_seed_groups, positive=False
        )
        probability_labels = probability >= threshold
        # This is the actual method output returned to the evaluator, including
        # the named method-side click constraint when enabled.
        selected_labels = prediction
        official_positive = probability_labels[
            positive_index_tensor
        ].detach().cpu().tolist()
        official_negative = (
            (~probability_labels[
                negative_index_tensor
            ]).detach().cpu().tolist()
            if negative_indices
            else []
        )
        selected_positive = selected_labels[
            positive_index_tensor
        ].detach().cpu().tolist()
        selected_negative = (
            (~selected_labels[
                negative_index_tensor
            ]).detach().cpu().tolist()
            if negative_indices
            else []
        )
        self.last_seed_satisfaction_stages = {
            "primitive_solver": self._summarize_click_matches(
                primitive_positive, primitive_negative
            ),
            "official_continuous_readout": self._summarize_click_matches(
                official_positive, official_negative
            ),
            "post_selection_pre_overwrite": self._summarize_click_matches(
                selected_positive, selected_negative
            ),
        }
        return prediction.cpu().numpy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _binary_ply_vertex_layout(path: Path) -> tuple[int, int, np.dtype]:
    """Return a selective binary-PLY dtype for AGILE geometry/RGB fields.

    The released AGILE PLY packs evaluator-only labels alongside vertex
    geometry.  ``PlyData.read`` eagerly materializes every property, including
    ``label``.  The support preflight must instead parse only the header and
    memory-map the six allowed scalar fields with their original record
    stride.  This is deliberately a narrow reader for the released binary PLY
    contract; the evaluator's label reader remains separate below.
    """

    path = Path(path)
    required = ("x", "y", "z", "R", "G", "B")
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, np.dtype, int]] = []
    record_size = 0
    current_element = ""
    byte_order = ""
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="strict").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"{path} ends before PLY end_header")
            line = raw.decode("ascii", errors="strict").strip()
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            parts = line.split()
            if parts[0] == "format":
                if len(parts) != 3 or parts[2] != "1.0":
                    raise ValueError(f"{path} has an unsupported PLY format declaration")
                formats = {
                    "binary_little_endian": "<",
                    "binary_big_endian": ">",
                }
                if parts[1] not in formats:
                    raise ValueError(
                        f"{path} must use a binary PLY for label-free geometry preflight"
                    )
                byte_order = formats[parts[1]]
                continue
            if parts[0] == "element":
                if len(parts) != 3:
                    raise ValueError(f"{path} has an invalid PLY element declaration")
                current_element = parts[1]
                if current_element == "vertex":
                    try:
                        vertex_count = int(parts[2])
                    except ValueError as error:
                        raise ValueError(f"{path} has an invalid vertex count") from error
                    if vertex_count < 0:
                        raise ValueError(f"{path} has a negative vertex count")
                continue
            if parts[0] == "property" and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError(
                        f"{path} has unsupported list/invalid vertex properties"
                    )
                scalar = _PLY_SCALAR_DTYPES.get(parts[1])
                if scalar is None:
                    raise ValueError(
                        f"{path} has unsupported vertex scalar type {parts[1]!r}"
                    )
                dtype = np.dtype(scalar).newbyteorder(byte_order or "=")
                vertex_properties.append((parts[2], dtype, record_size))
                record_size += int(dtype.itemsize)
                continue
            if parts[0] == "end_header":
                header_offset = int(handle.tell())
                break
    if not byte_order or vertex_count is None or record_size <= 0:
        raise ValueError(f"{path} lacks a valid binary vertex layout")
    property_by_name = {name: (dtype, offset) for name, dtype, offset in vertex_properties}
    missing = [name for name in required if name not in property_by_name]
    if missing:
        raise ValueError(f"{path} lacks official AGILE3D geometry/color properties: {missing}")
    if len(property_by_name) != len(vertex_properties):
        raise ValueError(f"{path} has duplicate vertex property names")
    dtype = np.dtype(
        {
            "names": list(required),
            "formats": [property_by_name[name][0] for name in required],
            "offsets": [property_by_name[name][1] for name in required],
            "itemsize": record_size,
        }
    )
    return header_offset, int(vertex_count), dtype


def _read_official_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only released geometry/RGB without materializing evaluator labels."""

    path = Path(path)
    header_offset, count, dtype = _binary_ply_vertex_layout(path)
    vertex = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=int(header_offset),
        shape=(int(count),),
    )
    try:
        xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
            np.float32
        )
        rgb = np.column_stack([vertex[name] for name in ("R", "G", "B")]).astype(
            np.float32
        ) / 255.0
    finally:
        # Explicitly drop the selective memmap before the label evaluator
        # opens its independent PLY reader below.
        del vertex
    return xyz, rgb


def _read_official_labels(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    if "label" not in names:
        raise ValueError(f"{path} lacks evaluator-only AGILE3D labels")
    return np.asarray(vertex["label"], dtype=np.int32)


def _quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first Gaussian quaternions without architecture coupling."""

    quat = F.normalize(torch.as_tensor(quaternion).float(), dim=-1, eps=1e-8)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError("Gaussian rotations must be [N,4] scalar-first quaternions")
    r, x, y, z = quat.unbind(dim=-1)
    rows = (
        1 - 2 * (y.square() + z.square()),
        2 * (x * y - r * z),
        2 * (x * z + r * y),
        2 * (x * y + r * z),
        1 - 2 * (x.square() + z.square()),
        2 * (y * z - r * x),
        2 * (x * z - r * y),
        2 * (y * z + r * x),
        1 - 2 * (x.square() + y.square()),
    )
    return torch.stack(rows, dim=-1).reshape(-1, 3, 3)


def _gaussian_covariances(model) -> torch.Tensor:
    rotation = _quaternion_to_rotation_matrix(model.get_rotation().float())
    scale = model.get_scaling().float().clamp_min(1e-6)
    if scale.shape != (rotation.shape[0], 3):
        raise ValueError("continuous 3-D readout requires three Gaussian scale axes")
    return rotation @ torch.diag_embed(scale.square()) @ rotation.transpose(1, 2)


def _load_geometry_model(config, checkpoint_path: str, device: torch.device):
    """Load frozen Gaussian geometry for either supported field architecture."""

    architecture = str(getattr(config, "architecture", "explicit"))
    if architecture == "hybrid":
        model, _codec = _build_hybrid_model(config, checkpoint_path, device)
        return model
    if architecture != "explicit":
        raise ValueError(f"unsupported canonical field architecture: {architecture}")

    from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian

    ply_path = str(getattr(config, "ply_path", ""))
    if not ply_path:
        raise ValueError("explicit geometry render contract has no ply_path")
    model = ExplicitFeatureGaussian(
        latent_dim=int(getattr(config, "latent_dim", 64)),
        train_sh=bool(getattr(config, "train_sh", False)),
    )
    model.load_from_ply(ply_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("explicit geometry checkpoint lacks model_state_dict")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"incompatible explicit geometry checkpoint: unexpected={result.unexpected_keys}"
        )
    missing_geometry = [
        key
        for key in result.missing_keys
        if key in {"_xyz", "_rotation", "_scaling", "_opacity"}
    ]
    if missing_geometry:
        raise RuntimeError(
            f"explicit geometry checkpoint misses frozen geometry: {missing_geometry}"
        )
    return model.to(device).eval()


def _field_source_metadata(scene_dir: Path) -> dict[str, object]:
    report_path = scene_dir / "raw_radio_mpr.pt.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"canonical field lacks MPR metadata: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{report_path} lacks MPR metadata mapping")
    return dict(metadata)


def _load_scene_geometry(
    scene_dir: Path,
    *,
    bank_xyz: torch.Tensor,
    valid_rows: torch.Tensor,
    expected_field_sha256: str,
    cache_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Load or build fixed geometry needed by the continuous canonical reader."""

    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu")
        if (
            int(payload.get("schema_version", -1)) != 1
            or str(payload.get("field_checkpoint_sha256", ""))
            != str(expected_field_sha256)
        ):
            raise ValueError(f"canonical geometry cache contract mismatch: {cache_path}")
        xyz = torch.as_tensor(payload.get("xyz")).float()
        covariance = torch.as_tensor(payload.get("covariance")).float()
        precision = torch.as_tensor(payload.get("precision")).float()
        opacity = torch.as_tensor(payload.get("opacity")).float().reshape(-1)
        expected_xyz = bank_xyz[valid_rows].float().cpu()
        if (
            xyz.shape != expected_xyz.shape
            or not torch.allclose(xyz, expected_xyz, atol=1e-6, rtol=0.0)
            or covariance.shape != (len(xyz), 3, 3)
            or precision.shape != covariance.shape
            or opacity.shape != (len(xyz),)
        ):
            raise ValueError(f"canonical geometry cache tensors do not align: {cache_path}")
        return (
            xyz.to(device),
            covariance.to(device),
            precision.to(device),
            opacity.to(device),
            True,
        )

    source = _field_source_metadata(scene_dir)
    config_path = Path(str(source.get("config", "")))
    checkpoint_path = Path(str(source.get("checkpoint", "")))
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "canonical continuous reader needs the field's recorded geometry config/checkpoint"
        )
    config = load_config(str(config_path))
    model = _load_geometry_model(config, str(checkpoint_path), device)
    try:
        full_xyz = model.get_xyz().detach().float().cpu()
        expected_full = bank_xyz.float().cpu()
        if full_xyz.shape != expected_full.shape or not torch.allclose(
            full_xyz, expected_full, atol=1e-6, rtol=0.0
        ):
            raise ValueError("canonical geometry and capability-bank rows differ")
        covariance = _gaussian_covariances(model).detach()[valid_rows.to(device)]
        identity = torch.eye(3, device=device, dtype=covariance.dtype)
        precision = torch.linalg.pinv(covariance + 1e-6 * identity)
        opacity = model.get_opacity().detach().float().reshape(-1)[valid_rows.to(device)]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.tmp")
        torch.save(
            {
                "schema_version": 1,
                "field_checkpoint_sha256": str(expected_field_sha256),
                "xyz": expected_full[valid_rows].clone(),
                "covariance": covariance.detach().float().cpu(),
                "precision": precision.detach().float().cpu(),
                "opacity": opacity.detach().float().cpu(),
                "source": "fixed_canonical_gaussian_geometry",
                "query_or_labels_used": False,
            },
            temporary,
        )
        temporary.replace(cache_path)
        return (
            expected_full[valid_rows].to(device),
            covariance,
            precision,
            opacity,
            False,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _scene_artifact(scene_dir: Path, filename: str, *, role: str) -> Path:
    """Resolve a named field artifact without allowing a cross-scene escape."""

    name = str(filename).strip()
    if not name or Path(name).name != name:
        raise ValueError(f"{role} must be a basename inside the scene field directory")
    return scene_dir / name


def _load_scene_predictor(
    scene_dir: Path,
    *,
    official_xyz: np.ndarray,
    geometry_cache_root: Path,
    device: torch.device,
    observation_contract: str,
    solver_config: SupportSolverConfig,
    readout_candidate_k: int,
    readout_support_threshold: float,
    evaluation_voxel_size_m: float,
    click_seed_kernel: str,
    seed_candidate_k: int,
    seed_topk: int,
    seed_temperature: float,
    world_point_prototype_mode: str,
    world_point_max_prototypes: int,
    world_point_prototype_weighting: str,
    feature_calibration: str,
    background_centroids: int,
    background_negative_policy: str,
    calibration_sample_size: int,
    centroid_iterations: int,
    score_calibration: str,
    score_chunk_size: int,
    channel_confidence_mode: str,
    negative_spatial_mode: str,
    negative_spatial_steps: int,
    negative_spatial_decay: float,
    require_official_extracted_capability_teachers: bool,
    point_readout_constraint: str = "none",
    selection_mode: SelectionMode | str = SelectionMode.SEEDED_COMPONENT,
    field_checkpoint_name: str = "canonical_mpr_v2.pt",
    capability_cache_name: str = "official_dino_sam3_views.pt",
    support_graph_name: str = "shared_support_graph_k16.pt",
    reliability_cache_name: str = "",
) -> tuple[CanonicalFieldPointPredictor, dict[str, object]]:
    field_path = _scene_artifact(
        scene_dir, field_checkpoint_name, role="field checkpoint name"
    )
    capability_path = _scene_artifact(
        scene_dir, capability_cache_name, role="capability cache name"
    )
    graph_path = _scene_artifact(
        scene_dir, support_graph_name, role="support graph name"
    )
    for path in (field_path, capability_path, graph_path):
        if not path.is_file():
            raise FileNotFoundError(f"canonical AGILE scene input is missing: {path}")
    reliability_path = (
        _scene_artifact(
            scene_dir, reliability_cache_name, role="reliability cache name"
        )
        if str(reliability_cache_name).strip()
        else None
    )
    if reliability_path is not None and not reliability_path.is_file():
        raise FileNotFoundError(
            f"canonical AGILE reliability cache is missing: {reliability_path}"
        )
    mpr_metadata = _field_source_metadata(scene_dir)
    observation_source = observation_source_from_render_contract(
        Path(str(mpr_metadata.get("config", "")))
    )
    validate_full_observation_mpr_contract(
        observation_contract,
        mpr_metadata,
        expected_source_contract_sha256=str(
            observation_source["field_source_contract_sha256"]
        ),
        expected_source_contract_version=str(
            observation_source["field_source_contract_version"]
        ),
    )
    field_hash = _sha256(field_path)
    bank = load_canonical_capability_bank(
        capability_path, expected_field_checkpoint_sha256=field_hash
    )
    teacher_fidelity = validate_capability_teacher_fidelity(
        bank.metadata,
        require_official_extracted=bool(
            require_official_extracted_capability_teachers
        ),
    )
    graph = load_canonical_support_graph(graph_path, bank)
    graph_payload = torch.load(graph_path, map_location="cpu")
    graph_metadata = dict(graph_payload.get("metadata", {}))
    primitive_reliability = (
        load_canonical_primitive_reliability(
            reliability_path,
            expected_xyz=bank.xyz,
            expected_valid=bank.valid,
            expected_field_checkpoint_sha256=field_hash,
        )
        if reliability_path is not None
        else None
    )
    gaussian_xyz, covariance, precision, opacity, cache_reused = _load_scene_geometry(
        scene_dir,
        bank_xyz=bank.xyz,
        valid_rows=bank.global_rows,
        expected_field_sha256=field_hash,
        cache_path=geometry_cache_root / f"{scene_dir.name}.pt",
        device=device,
    )
    feature_banks = bank.valid_feature_banks()
    predictor = CanonicalFieldPointPredictor(
        gaussian_xyz=gaussian_xyz,
        gaussian_covariance=covariance,
        gaussian_precision=precision,
        gaussian_opacity=opacity,
        appearance_features=feature_banks["appearance"],
        boundary_features=feature_banks["boundary"],
        appearance_signature=bank.signatures["appearance"],
        boundary_signature=bank.signatures["boundary"],
        graph=graph,
        official_xyz=official_xyz,
        device=str(device),
        solver_config=solver_config,
        readout_candidate_k=int(readout_candidate_k),
        readout_support_threshold=float(readout_support_threshold),
        evaluation_voxel_size_m=float(evaluation_voxel_size_m),
        click_seed_kernel=str(click_seed_kernel),
        seed_candidate_k=int(seed_candidate_k),
        seed_topk=int(seed_topk),
        seed_temperature=float(seed_temperature),
        world_point_prototype_mode=str(world_point_prototype_mode),
        world_point_max_prototypes=int(world_point_max_prototypes),
        world_point_prototype_weighting=str(world_point_prototype_weighting),
        feature_calibration=str(feature_calibration),
        background_centroids=int(background_centroids),
        background_negative_policy=str(background_negative_policy),
        calibration_sample_size=int(calibration_sample_size),
        centroid_iterations=int(centroid_iterations),
        score_calibration=str(score_calibration),
        score_chunk_size=int(score_chunk_size),
        channel_confidence_mode=str(channel_confidence_mode),
        negative_spatial_mode=str(negative_spatial_mode),
        negative_spatial_steps=int(negative_spatial_steps),
        negative_spatial_decay=float(negative_spatial_decay),
        point_readout_constraint=str(point_readout_constraint),
        selection_mode=selection_mode,
        node_reliability=(
            primitive_reliability.valid_confidence()
            if primitive_reliability is not None
            else None
        ),
    )
    return predictor, {
        "field_checkpoint": str(field_path.resolve()),
        "field_checkpoint_sha256": field_hash,
        "capability_cache": str(capability_path.resolve()),
        "support_graph": str(graph_path.resolve()),
        "support_graph_sha256": _sha256(graph_path),
        "support_graph_config": dict(graph_metadata.get("graph_config", {})),
        "support_graph_surface_relation": dict(
            graph_metadata.get("surface_relation") or {}
        ),
        "support_graph_covisibility_relation": dict(
            graph_metadata.get("covisibility_relation") or {}
        ),
        "mpr_observation_contract": str(
            dict(mpr_metadata.get("observation_lifting_contract", {})).get("name", "")
        ),
        "mpr_observation_contract_sha256": str(
            mpr_metadata.get("observation_lifting_contract_sha256", "")
        ),
        "mpr_declared_views": int(mpr_metadata.get("num_declared_views", 0)),
        "mpr_full_observation_coverage_order_applied": bool(
            mpr_metadata.get("full_observation_coverage_order_applied", False)
        ),
        "mpr_full_observation_source_view_count": int(
            mpr_metadata.get("full_observation_source_view_count", 0)
        ),
        "primitive_reliability_cache": (
            str(reliability_path.resolve()) if reliability_path is not None else ""
        ),
        "primitive_reliability_cache_sha256": (
            _sha256(reliability_path) if reliability_path is not None else ""
        ),
        "primitive_reliability_formula": (
            str(primitive_reliability.metadata.get("formula", ""))
            if primitive_reliability is not None
            else ""
        ),
        **teacher_fidelity,
        **observation_source,
        "geometry_cache": str((geometry_cache_root / f"{scene_dir.name}.pt").resolve()),
        "geometry_cache_reused": bool(cache_reused),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Run the released interaction on direct canonical-field predictions."""

    benchmark_root = Path(args.benchmark_root)
    field_root = Path(args.field_root)
    geometry_cache_root = Path(args.geometry_cache_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source_contract = str(args.observation_contract)
    require_support_gate = bool(args.require_support_gate)
    diagnostic_no_support_gate = bool(
        getattr(args, "diagnostic_no_support_gate", False)
    )
    if source_contract in {
        "scannet_full_observation_pilot",
        "scannet_full_observation_v1",
    } and not require_support_gate:
        raise ValueError("scannet_full_observation_v1 requires --require-support-gate")
    if source_contract == FULL_OBSERVATION_DIAGNOSTIC_CONTRACT:
        if require_support_gate or not diagnostic_no_support_gate:
            raise ValueError(
                "scannet_full_observation_diagnostic_v1 requires "
                "--diagnostic-no-support-gate and no --require-support-gate"
            )
    validate_continuous_support_threshold(
        source_contract, float(args.readout_support_threshold)
    )
    geometry_scenes = {
        path.stem for path in (benchmark_root / "scans").glob("scene*.ply")
    }
    field_scenes = {
        path.name
        for path in (field_root / "canonical_fields").iterdir()
        if path.is_dir()
    }
    if str(args.scene_names).strip():
        requested = set(str(args.scene_names).replace(",", " ").split())
        unknown = requested - geometry_scenes
        if unknown:
            raise ValueError(f"unknown AGILE3D geometry scenes: {sorted(unknown)}")
        missing_fields = requested - field_scenes
        if missing_fields:
            raise FileNotFoundError(
                f"direct-canonical fields are missing for requested scenes: {sorted(missing_fields)}"
            )
        selected_scenes = sorted(requested)
    else:
        selected_scenes = sorted(field_scenes & geometry_scenes)
    if not selected_scenes:
        raise ValueError("no AGILE3D scenes selected")
    # ``getattr`` preserves the programmatic evaluator API used by historic
    # support-only callers; CLI invocations always receive the explicit flags.
    object_shard_count = int(getattr(args, "object_shard_count", 1))
    object_shard_index = int(getattr(args, "object_shard_index", 0))
    # Validate the scheduling-only partition before any evaluator object list
    # or label property is opened.  The same values are persisted in every
    # shard report and never enter field/query inference.
    select_object_shard(
        (),
        shard_index=object_shard_index,
        shard_count=object_shard_count,
    )
    if source_contract == "scannet_full_observation_v1" and set(selected_scenes) != geometry_scenes:
        missing = sorted(geometry_scenes - set(selected_scenes))
        raise ValueError(
            "scannet_full_observation_v1 requires a field for every official "
            f"AGILE3D geometry scene; missing={missing[:8]}"
        )
    solver_config = SupportSolverConfig(
        solver_type=str(args.solver_type),
        laplacian_weight=float(args.laplacian_weight),
        cg_iterations=int(args.cg_iterations),
        support_threshold=float(args.support_threshold),
        hard_seed_threshold=float(args.hard_seed_threshold),
        hard_seed_conflict_policy=str(args.hard_seed_conflict_policy),
        hard_seed_conflict_margin=float(args.hard_seed_conflict_margin),
        unary_edge_contrast=float(args.unary_edge_contrast),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def load_label_free_scene(
        scene_id: str,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        object,
        np.ndarray,
        CanonicalFieldPointPredictor,
        dict[str, object],
    ]:
        """Load field support for one scene before any evaluator label is used."""

        ply_path = benchmark_root / "scans" / f"{scene_id}.ply"
        xyz, colors = _read_official_geometry(ply_path)
        quantized_geometry = quantize_scannet_points(
            xyz,
            colors,
            np.zeros(len(xyz), dtype=np.int32),
            voxel_size=float(args.voxel_size),
        )
        scene_origin = xyz.min(axis=0, keepdims=True)
        official_world_xyz = quantized_geometry.raw_coordinates + scene_origin
        scene_dir = field_root / "canonical_fields" / scene_id
        predictor, source = _load_scene_predictor(
            scene_dir,
            official_xyz=official_world_xyz,
            geometry_cache_root=geometry_cache_root,
            device=device,
            observation_contract=source_contract,
            solver_config=solver_config,
            readout_candidate_k=int(args.readout_candidate_k),
            readout_support_threshold=float(args.readout_support_threshold),
            evaluation_voxel_size_m=float(args.evaluation_voxel_size_m),
            click_seed_kernel=str(args.click_seed_kernel),
            seed_candidate_k=int(args.seed_candidate_k),
            seed_topk=int(args.seed_topk),
            seed_temperature=float(args.seed_temperature),
            world_point_prototype_mode=str(args.world_point_prototype_mode),
            world_point_max_prototypes=int(args.world_point_max_prototypes),
            world_point_prototype_weighting=str(args.world_point_prototype_weighting),
            feature_calibration=str(args.feature_calibration),
            background_centroids=int(args.background_centroids),
            background_negative_policy=str(args.background_negative_policy),
            calibration_sample_size=int(args.calibration_sample_size),
            centroid_iterations=int(args.centroid_iterations),
            score_calibration=str(args.score_calibration),
            score_chunk_size=int(getattr(args, "score_chunk_size", 8192)),
            channel_confidence_mode=str(
                getattr(args, "channel_confidence_mode", "none")
            ),
            negative_spatial_mode=str(
                getattr(args, "negative_spatial_mode", "none")
            ),
            negative_spatial_steps=int(
                getattr(args, "negative_spatial_steps", 4)
            ),
            negative_spatial_decay=float(
                getattr(args, "negative_spatial_decay", 0.8)
            ),
            point_readout_constraint=str(
                getattr(args, "point_readout_constraint", "none")
            ),
            selection_mode=str(
                getattr(args, "selection_mode", SelectionMode.SEEDED_COMPONENT.value)
            ),
            require_official_extracted_capability_teachers=bool(
                args.require_official_extracted_capability_teachers
            ),
            field_checkpoint_name=str(args.field_checkpoint_name),
            capability_cache_name=str(args.capability_cache_name),
            support_graph_name=str(args.support_graph_name),
            reliability_cache_name=str(args.reliability_cache_name),
        )
        validate_observation_contract(
            source_contract,
            str(source["source_observation_root"]),
            require_support_gate=require_support_gate,
            allow_ungated_diagnostic=diagnostic_no_support_gate,
            declared_source_contract=str(source["declared_source_contract"]),
            field_source_contract_sha256=str(source["field_source_contract_sha256"]),
            field_source_contract_version=str(source["field_source_contract_version"]),
        )
        support = predictor.protocol_report()
        support_row: dict[str, object] = {
            "scene_id": scene_id,
            "quantized_points": int(len(quantized_geometry.raw_coordinates)),
            "official_coordinate_contract": (
                "released_shifted_5cm_callback_plus_label_free_scene_origin_to_scannet_world"
            ),
            "quantization_scene_origin_m": [
                float(value) for value in np.asarray(scene_origin).reshape(-1)
            ],
            **source,
            **support,
        }
        support_row["support_gate_passed"] = bool(
            not require_support_gate
            or float(support["continuous_support_fraction"])
            >= float(args.minimum_support_fraction)
        )
        return xyz, colors, quantized_geometry, scene_origin, predictor, support_row

    # This field-quality preflight intentionally happens before the object
    # list or per-vertex label property is requested.  Thus a support-gate
    # rejection cannot be influenced by evaluator targets or hidden under a
    # coverage stratum after the interactive score is known.
    preflight_support: dict[str, dict[str, object]] = {}
    for scene_id in selected_scenes:
        _xyz, _colors, _geometry, _origin, predictor, support_row = load_label_free_scene(
            scene_id
        )
        preflight_support[scene_id] = support_row
        del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    failed_support = [
        record
        for record in preflight_support.values()
        if not bool(record["support_gate_passed"])
    ]
    if bool(args.support_only):
        # Persist an auditable, label-free failure record before any released
        # object list or PLY label property can be requested.  A field-source
        # ladder can use this to increase only its query-free view budget;
        # it is not an evaluator result and contains no object metric.
        report: dict[str, object] = {
            "benchmark": "AGILE3D ScanNet40 single-object",
            "mode": "label_free_field_support_preflight",
            "protocol": {
                "observation_contract": source_contract,
                "support_gate_required": require_support_gate,
                "minimum_support_fraction": float(args.minimum_support_fraction),
                "readout_support_threshold": float(args.readout_support_threshold),
                "evaluation_voxel_size_m": float(args.evaluation_voxel_size_m),
                "official_coordinate_contract": (
                    "released_shifted_5cm_callback_plus_label_free_scene_origin_to_scannet_world"
                ),
                "labels_opened": False,
                "object_list_opened": False,
                "test_set_calibration": False,
            },
            "scene_support": [preflight_support[scene] for scene in selected_scenes],
            "support_gate_passed": not failed_support,
        }
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    if failed_support:
        first = failed_support[0]
        raise ValueError(
            f"{first['scene_id']} fails continuous support gate: "
            f"{float(first['continuous_support_fraction']):.6f} < "
            f"{float(args.minimum_support_fraction):.6f}"
        )

    # Only after every selected field is accepted do we open the evaluator's
    # instance list and label values to run the released interaction protocol.
    all_objects = load_official_object_list(benchmark_root)
    full_selected_objects = [
        item for item in all_objects if item.scene_id in preflight_support
    ]
    objects = select_object_shard(
        full_selected_objects,
        shard_index=object_shard_index,
        shard_count=object_shard_count,
    )
    if not objects:
        raise ValueError(
            "selected AGILE object shard is empty; choose a valid shard for "
            "the fixed scene list"
        )
    by_scene: dict[str, list] = defaultdict(list)
    for item in objects:
        by_scene[item.scene_id].append(item)
    missing_object_scenes = [scene for scene in selected_scenes if scene not in by_scene]
    if object_shard_count == 1 and missing_object_scenes:
        raise ValueError(
            "selected AGILE3D scenes have no released evaluator objects: "
            f"{missing_object_scenes}"
        )

    rows: list[dict[str, object]] = []
    scene_support: list[dict[str, object]] = []
    for scene_index, scene_id in enumerate(selected_scenes):
        xyz, colors, quantized_geometry, scene_origin, predictor, support_row = load_label_free_scene(
            scene_id
        )
        if stable_support_record(support_row) != stable_support_record(
            preflight_support[scene_id]
        ):
            raise AssertionError("field support changed between preflight and evaluator")
        scene_support.append(support_row)
        ply_path = benchmark_root / "scans" / f"{scene_id}.ply"

        labels_full = _read_official_labels(ply_path)
        quantized = quantize_scannet_points(
            xyz, colors, labels_full, voxel_size=float(args.voxel_size)
        )
        if not np.array_equal(quantized.unique_map, quantized_geometry.unique_map):
            raise AssertionError("label-free and evaluator quantization maps differ")

        def callback(
            coordinates: np.ndarray, previous: np.ndarray, clicks: Sequence[Click]
        ) -> np.ndarray:
            world = np.asarray(coordinates, dtype=np.float32) + scene_origin
            prediction = predictor(world, previous, clicks)
            callback.last_seed_satisfaction_stages = (  # type: ignore[attr-defined]
                predictor.last_seed_satisfaction_stages
            )
            return prediction

        for item in by_scene[scene_id]:
            target_full = labels_full == int(item.object_id)
            target_quantized = quantized.labels == int(item.object_id)
            result = evaluate_interactive_predictions(
                quantized.raw_coordinates,
                target_quantized,
                target_full,
                quantized.inverse_map,
                callback,
                max_clicks=int(args.max_clicks),
                click_workers=int(args.click_workers),
            )
            rows.append(
                {
                    "scene_id": scene_id,
                    "object_id": int(item.object_id),
                    "semantic_class": str(item.semantic_class),
                    **result,
                }
            )
        partial = {
            "benchmark": "AGILE3D ScanNet40 single-object",
            "completed_scenes": scene_index + 1,
            "total_scenes": len(by_scene),
            "completed_objects": len(rows),
            "scene_support": scene_support,
            "rows": rows,
            "shard": {
                "object_shard_index": object_shard_index,
                "object_shard_count": object_shard_count,
                "object_assignment": "released_object_list_position_modulo",
                "metrics_are_partial": object_shard_count > 1,
            },
        }
        output.write_text(json.dumps(partial), encoding="utf-8")
        del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    canonical_mpr_contracts = {
        str(record.get("mpr_observation_contract", ""))
        for record in scene_support
    }
    if len(canonical_mpr_contracts) != 1:
        raise ValueError(
            "direct-canonical AGILE scenes do not share one MPR observation contract"
        )
    canonical_mpr_contract = next(iter(canonical_mpr_contracts))
    canonical_mpr_coverage_ranked = all(
        bool(record.get("mpr_full_observation_coverage_order_applied", False))
        for record in scene_support
    )
    if source_contract in {
        "scannet_full_observation_pilot",
        "scannet_full_observation_v1",
        FULL_OBSERVATION_DIAGNOSTIC_CONTRACT,
    } and (
        canonical_mpr_contract not in CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES
        or not canonical_mpr_coverage_ranked
    ):
        raise AssertionError(
            "accepted full-observation AGILE field lacks the required full MPR contract"
        )

    report: dict[str, object] = {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "protocol": {
            "official_preprocessed_data": str(benchmark_root.resolve()),
            "field_root": str(field_root.resolve()),
            "field_checkpoint_name": str(args.field_checkpoint_name),
            "capability_cache_name": str(args.capability_cache_name),
            "support_graph_name": str(args.support_graph_name),
            "reliability_cache_name": str(args.reliability_cache_name),
            "canonical_mpr_contract": canonical_mpr_contract,
            "canonical_mpr_coverage_ranked": canonical_mpr_coverage_ranked,
            "observation_contract": source_contract,
            "result_status": (
                "diagnostic_only" if diagnostic_no_support_gate else "formal"
            ),
            "formal_comparable": not diagnostic_no_support_gate,
            "diagnostic_no_support_gate": diagnostic_no_support_gate,
            "voxel_size_m": float(args.voxel_size),
            "objects": len(rows),
            "scenes": len(by_scene),
            "max_clicks": int(args.max_clicks),
            "click_search_workers": int(args.click_workers),
            "click_policy": "center_of_largest_FP_or_FN_error_by_inradius",
            "clicked_labels_forced": True,
            "test_set_calibration": False,
            "world_query": "compile_world_3d_query",
            "selection_mode": str(
                getattr(args, "selection_mode", SelectionMode.SEEDED_COMPONENT.value)
            ),
            "official_coordinate_contract": (
                "released_shifted_5cm_callback_plus_label_free_scene_origin_to_scannet_world"
            ),
            "observation_lift": "none",
            "official_point_readout": "continuous_opacity_weighted_gaussian",
            "readout_candidate_k": int(args.readout_candidate_k),
            "readout_support_threshold": float(args.readout_support_threshold),
            "evaluation_voxel_size_m": float(args.evaluation_voxel_size_m),
            "voxel_cell_variance_m2": float(args.evaluation_voxel_size_m) ** 2 / 12.0,
            "click_seed_kernel": str(args.click_seed_kernel),
            "seed_candidate_k": int(args.seed_candidate_k),
            "hard_seed_topk": int(args.seed_topk),
            "seed_temperature": float(args.seed_temperature),
            "hard_seed_threshold": float(args.hard_seed_threshold),
            "hard_seed_conflict_policy": str(args.hard_seed_conflict_policy),
            "hard_seed_conflict_margin": float(args.hard_seed_conflict_margin),
            "prototype_count": 4,
            "prototype_strategy": "weighted_fps",
            "support_gate_required": require_support_gate,
            "minimum_support_fraction": float(args.minimum_support_fraction),
            "solver_type": str(args.solver_type),
            "laplacian_weight": float(args.laplacian_weight),
            "cg_iterations": int(args.cg_iterations),
            "support_threshold": float(args.support_threshold),
            "unary_edge_contrast": float(args.unary_edge_contrast),
            "world_point_prototype_mode": str(args.world_point_prototype_mode),
            "world_point_max_prototypes": int(args.world_point_max_prototypes),
            "world_point_prototype_weighting": str(args.world_point_prototype_weighting),
            "appearance_unary_weight": 1.0,
            "boundary_unary_weight": 0.35,
            "feature_calibration": str(args.feature_calibration),
            "background_centroids": int(args.background_centroids),
            "background_negative_policy": str(args.background_negative_policy),
            "calibration_sample_size": int(args.calibration_sample_size),
            "centroid_iterations": int(args.centroid_iterations),
            "score_calibration": str(args.score_calibration),
            "score_chunk_size": int(getattr(args, "score_chunk_size", 8192)),
            "channel_confidence_mode": str(
                getattr(args, "channel_confidence_mode", "none")
            ),
            "negative_spatial_mode": str(
                getattr(args, "negative_spatial_mode", "none")
            ),
            "negative_spatial_steps": int(
                getattr(args, "negative_spatial_steps", 4)
            ),
            "negative_spatial_decay": float(
                getattr(args, "negative_spatial_decay", 0.8)
            ),
            "spatial_log_weight": 0.25,
            "spatial_floor": 0.01,
            "point_readout_constraint": str(args.point_readout_constraint),
            "requires_official_extracted_capability_teachers": bool(
                args.require_official_extracted_capability_teachers
            ),
            "labels_opened_during_field_or_support_audit": False,
        },
        "scene_support": scene_support,
        "metrics": aggregate_official_metrics(
            [row["trajectory"] for row in rows], max_clicks=int(args.max_clicks)
        ),
        "interaction_health": interaction_health_metrics(
            [row["trajectory"] for row in rows],
            seed_satisfaction=[row["seed_satisfaction"] for row in rows],
            max_clicks=int(args.max_clicks),
        ),
        "rows": rows,
        "shard": {
            "object_shard_index": object_shard_index,
            "object_shard_count": object_shard_count,
            "object_assignment": "released_object_list_position_modulo",
            "official_objects_in_selected_scenes": len(full_selected_objects),
            "objects_in_this_shard": len(rows),
            "metrics_are_partial": object_shard_count > 1,
        },
    }
    if (
        source_contract == "scannet_full_observation_v1"
        and int(args.max_clicks) == 10
        and not diagnostic_no_support_gate
    ):
        # Bind future paper-facing shards before they leave the evaluator.
        # Per-scene 240/480/960 source identities are committed separately so
        # the label-free source ladder does not alter the method contract.
        method_contract, method_contract_sha256 = bind_frozen_method_contract(
            report["protocol"]  # type: ignore[arg-type]
        )
        source_bindings, source_bindings_sha256 = (
            source_contract_bindings_sha256(scene_support)
        )
        report["method_contract"] = method_contract
        report["method_contract_sha256"] = method_contract_sha256
        report["source_contract_bindings"] = source_bindings
        report["source_contract_bindings_sha256"] = source_bindings_sha256
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--field-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-cache-root", required=True)
    parser.add_argument("--field-checkpoint-name", default="canonical_mpr_v2.pt")
    parser.add_argument("--capability-cache-name", default="official_dino_sam3_views.pt")
    parser.add_argument(
        "--require-official-extracted-capability-teachers",
        action="store_true",
        help=(
            "Artifact-provenance gate for the native official C-RADIO adaptor "
            "teacher variant; does not alter AGILE inputs or metrics."
        ),
    )
    parser.add_argument("--support-graph-name", default="shared_support_graph_k16.pt")
    parser.add_argument(
        "--reliability-cache-name",
        default="",
        help=(
            "optional query-independent canonical reliability sidecar; "
            "its feature-evidence shrinkage never changes hard click seeds"
        ),
    )
    parser.add_argument("--scene-names", default="")
    parser.add_argument(
        "--object-shard-index",
        type=int,
        default=0,
        help=(
            "zero-based execution shard over released object-list order; does "
            "not change the interaction protocol"
        ),
    )
    parser.add_argument(
        "--object-shard-count",
        type=int,
        default=1,
        help=(
            "number of disjoint execution shards; merge all shards before "
            "reporting a benchmark metric"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument(
        "--evaluation-voxel-size-m",
        type=float,
        default=0.05,
        help="fixed released evaluator voxel edge used for cell-convolved field readout",
    )
    parser.add_argument(
        "--click-seed-kernel",
        choices=("native_gaussian", "evaluator_voxel_convolved"),
        default="native_gaussian",
        help="native Gaussian world-click kernel; convolved mode is an ablation only",
    )
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--click-workers", type=int, default=2)
    parser.add_argument(
        "--observation-contract",
        choices=(
            "dense_overlap_pilot",
            "dense_pfpr_queryheldout_pilot",
            "dense_agile_all_observations_pilot",
            "scannet_full_observation_pilot",
            "scannet_full_observation_v1",
            FULL_OBSERVATION_DIAGNOSTIC_CONTRACT,
        ),
        required=True,
    )
    parser.add_argument("--require-support-gate", action="store_true")
    parser.add_argument(
        "--diagnostic-no-support-gate",
        action="store_true",
        help=(
            "explicitly mark an incomplete/ungated full-observation run as "
            "diagnostic only; it is never a formal comparable result"
        ),
    )
    parser.add_argument(
        "--support-only",
        action="store_true",
        help=(
            "write the label-free continuous-field support audit and stop "
            "before opening AGILE objects or PLY labels"
        ),
    )
    parser.add_argument("--minimum-support-fraction", type=float, default=0.95)
    parser.add_argument("--readout-candidate-k", type=int, default=64)
    parser.add_argument("--readout-support-threshold", type=float, default=1e-6)
    parser.add_argument(
        "--point-readout-constraint",
        choices=("none", "click_score_clamp"),
        default="none",
        help=(
            "optional label-free method-side constraint at exact released click "
            "points; default none preserves all previous results"
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=(
            SelectionMode.SEEDED_COMPONENT.value,
            SelectionMode.MIN_SEED_COVER.value,
        ),
        default=SelectionMode.SEEDED_COMPONENT.value,
        help=(
            "component readout variant; seeded_component is the frozen protocol, "
            "min_seed_cover is an explicit no-GT multi-positive ablation"
        ),
    )
    parser.add_argument("--seed-candidate-k", type=int, default=64)
    parser.add_argument(
        "--seed-topk",
        type=int,
        default=0,
        help=(
            "optional count of most-responsible Gaussian rows retained as "
            "hard click constraints; zero preserves covariance-soft seeds"
        ),
    )
    parser.add_argument(
        "--seed-temperature",
        type=float,
        default=1.0,
        help="label-free exponent for relative covariance seed responsibilities",
    )
    parser.add_argument(
        "--world-point-prototype-mode",
        choices=("aggregate_fps", "per_click_local"),
        default="per_click_local",
    )
    parser.add_argument("--world-point-max-prototypes", type=int, default=0)
    parser.add_argument(
        "--world-point-prototype-weighting",
        choices=("support_mass", "equal_click"),
        default="support_mass",
        help=(
            "support-mass weighting is the frozen promotion default; "
            "equal_click is a named ablation"
        ),
    )
    parser.add_argument(
        "--solver-type",
        choices=("random_walker", "confidence_random_walker"),
        default="confidence_random_walker",
    )
    parser.add_argument("--laplacian-weight", type=float, default=1.0)
    parser.add_argument("--cg-iterations", type=int, default=64)
    parser.add_argument("--support-threshold", type=float, default=0.50)
    parser.add_argument(
        "--hard-seed-threshold",
        type=float,
        default=0.20,
        help="fixed solver threshold that converts Gaussian seed mass into hard constraints",
    )
    parser.add_argument(
        "--hard-seed-conflict-policy",
        choices=("positive_priority", "exclusive_relative"),
        default="positive_priority",
        help=(
            "positive_priority preserves the frozen baseline; "
            "exclusive_relative leaves opposite-sign Gaussian-overlap ties soft"
        ),
    )
    parser.add_argument(
        "--hard-seed-conflict-margin",
        type=float,
        default=0.0,
        help="minimum relative Gaussian seed-mass advantage for an exclusive hard seed",
    )
    parser.add_argument(
        "--feature-calibration",
        choices=("none", "diagonal_robust"),
        default="none",
        help="query-independent scene-space normalization; default preserves the frozen baseline",
    )
    parser.add_argument(
        "--background-centroids",
        type=int,
        default=0,
        help="query-independent spherical scene modes used as additional background evidence",
    )
    parser.add_argument(
        "--background-negative-policy",
        choices=("pooled_mean", "explicit_hard_max"),
        default="pooled_mean",
        help=(
            "how scene-background modes combine with explicit negative clicks; "
            "pooled_mean preserves the frozen baseline"
        ),
    )
    parser.add_argument("--calibration-sample-size", type=int, default=8192)
    parser.add_argument("--centroid-iterations", type=int, default=4)
    parser.add_argument(
        "--score-calibration",
        choices=("none", "robust_tanh", "robust_tanh_centered", "robust_tanh_zero"),
        default="none",
        help="frozen query-score mapping; never fitted from AGILE targets",
    )
    parser.add_argument(
        "--score-chunk-size",
        type=int,
        default=8192,
        help="memory-bounded primitive rows per mathematically equivalent score block",
    )
    parser.add_argument(
        "--unary-edge-contrast",
        type=float,
        default=0.0,
        help="fixed query-unary boundary gating of the shared random-walker graph",
    )
    parser.add_argument(
        "--channel-confidence-mode",
        choices=("none", "affinity_mass", "max_affinity"),
        default="none",
        help=(
            "optional label-free capability abstention; confidence modes add "
            "self-loop conductance when every neighbour relation is weak"
        ),
    )
    parser.add_argument(
        "--negative-spatial-mode",
        choices=("none", "truncated_graph_decay", "signed_geodesic"),
        default="none",
        help=(
            "couple click descriptors to their fixed-geometry neighborhoods; "
            "scene-background evidence remains global"
        ),
    )
    parser.add_argument("--negative-spatial-steps", type=int, default=4)
    parser.add_argument("--negative-spatial-decay", type=float, default=0.8)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
