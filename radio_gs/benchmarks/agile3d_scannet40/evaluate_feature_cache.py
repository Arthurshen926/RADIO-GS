#!/usr/bin/env python3
"""Evaluate mesh-aligned canonical RADIO caches with the AGILE3D protocol."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from radio_gs.querying.query_spec import SelectionMode, SoftSeedSet
from radio_gs.querying.support_solver import (
    SupportGraphConfig,
    SupportSolverConfig,
    build_primitive_support_graph,
    graph_for_query_intent,
    normalized_laplacian_affinity,
    select_support_components,
    solve_primitive_support,
)
from radio_gs.querying.query_spec import QueryIntent

from .protocol import (
    Click,
    aggregate_official_metrics,
    evaluate_interactive_predictions,
    load_official_object_list,
    quantize_scannet_points,
)


def _read_official_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "R", "G", "B", "label"}
    if not required.issubset(names):
        raise ValueError(f"{path} lacks official AGILE3D PLY properties")
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float32)
    colors = np.column_stack([vertex[name] for name in ("R", "G", "B")]).astype(
        np.float32
    ) / 255.0
    labels = np.asarray(vertex["label"], dtype=np.int32)
    return xyz, colors, labels


class _ObservationDomainLift:
    """Fixed geometric map between official points and observed field rows.

    Sparse registered RGB-D observations leave some official 5 cm points
    without an authoritative canonical descriptor.  Those rows must not enter
    a feature graph as zero vectors or become a zero-valued click prototype.
    This map instead solves on observed field rows and uses only query-free
    local geometry to lift click seeds into, and support back out of, that
    observation domain.  It never reads labels, object IDs, or trajectories.
    """

    def __init__(
        self,
        full_xyz: np.ndarray,
        solver_rows: np.ndarray,
        *,
        neighbors: int,
        maximum_distance_m: float,
        device: torch.device,
    ) -> None:
        points = np.asarray(full_xyz, dtype=np.float32)
        rows = np.asarray(solver_rows, dtype=np.int64).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("full_xyz must be [N,3]")
        if rows.size == 0 or np.any(rows < 0) or np.any(rows >= len(points)):
            raise ValueError("solver_rows must be non-empty indices into full_xyz")
        if int(neighbors) <= 0 or float(maximum_distance_m) <= 0:
            raise ValueError("observation lift neighbors and radius must be positive")

        count = min(int(neighbors), int(rows.size))
        distance, local_index = cKDTree(points[rows]).query(
            points, k=count, workers=-1
        )
        distance = np.asarray(distance, dtype=np.float32)
        local_index = np.asarray(local_index, dtype=np.int64)
        if distance.ndim == 1:
            distance = distance[:, None]
            local_index = local_index[:, None]
        keep = distance <= float(maximum_distance_m)
        weight = np.where(keep, 1.0 / np.maximum(distance, 1e-4), 0.0)
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-8)

        # Observed rows are exact identity correspondences, even when another
        # observed row happens to be closer due to a duplicated mesh vertex.
        global_to_local = np.full(len(points), -1, dtype=np.int64)
        global_to_local[rows] = np.arange(rows.size, dtype=np.int64)
        direct = global_to_local >= 0
        local_index[direct] = 0
        local_index[direct, 0] = global_to_local[direct]
        weight[direct] = 0.0
        weight[direct, 0] = 1.0

        self.indices = torch.as_tensor(local_index, dtype=torch.long, device=device)
        self.weights = torch.as_tensor(weight, dtype=torch.float32, device=device)
        self.full_count = int(len(points))
        self.solver_count = int(rows.size)
        self.liftable = self.weights.sum(dim=1) > 0
        self.neighbors = int(count)
        self.maximum_distance_m = float(maximum_distance_m)

    def lift_clicks(self, point_indices: Sequence[int]) -> torch.Tensor:
        result = torch.zeros(self.solver_count, dtype=torch.float32, device=self.weights.device)
        if not point_indices:
            return result
        points = torch.as_tensor(point_indices, dtype=torch.long, device=self.weights.device)
        if bool((points < 0).any()) or bool((points >= self.full_count).any()):
            raise IndexError("click point is outside the official quantized scene")
        indices = self.indices[points]
        weights = self.weights[points]
        nonzero = weights > 0
        if bool(nonzero.any()):
            result.index_add_(0, indices[nonzero], weights[nonzero])
        return result.clamp(0.0, 1.0)

    def project(self, values: torch.Tensor) -> torch.Tensor:
        source = torch.as_tensor(values, device=self.weights.device).float().reshape(-1)
        if source.shape != (self.solver_count,):
            raise ValueError("solver values do not align with observation-domain lift")
        return (source[self.indices] * self.weights).sum(dim=1)

    def report(self) -> dict[str, int | float]:
        return {
            "solver_nodes": self.solver_count,
            "projectable_points": int(self.liftable.sum()),
            "projectable_fraction": float(self.liftable.float().mean()),
            "neighbors": self.neighbors,
            "maximum_distance_m": self.maximum_distance_m,
        }


class CanonicalPointPredictor:
    """Shared capability unary plus hard-seed support on observed primitives."""

    def __init__(
        self,
        xyz: np.ndarray,
        features: np.ndarray,
        *,
        appearance_features: np.ndarray | None,
        boundary_features: np.ndarray | None,
        observation_valid: np.ndarray | None,
        device: str,
        graph_config: SupportGraphConfig,
        solver_config: SupportSolverConfig,
        selection_mode: SelectionMode | str = SelectionMode.SEEDED_COMPONENT,
        unary_mode: str = "shared_capability",
        appearance_unary_weight: float = 1.0,
        boundary_unary_weight: float = 0.35,
        observation_lift_mode: str = "observed_domain",
        observation_lift_neighbors: int = 3,
        observation_lift_maximum_distance_m: float = 0.10,
    ) -> None:
        self.device = torch.device(device)
        full_xyz = np.asarray(xyz, dtype=np.float32)
        full_features = np.asarray(features)
        if full_xyz.shape != (full_features.shape[0], 3):
            raise ValueError("xyz and radio features must align")
        appearance = features if appearance_features is None else appearance_features
        boundary = features if boundary_features is None else boundary_features
        appearance = np.asarray(appearance)
        boundary = np.asarray(boundary)
        if appearance.shape[0] != len(full_xyz) or boundary.shape[0] != len(full_xyz):
            raise ValueError("capability feature banks must align with xyz")
        valid = (
            torch.ones(len(full_xyz), dtype=torch.bool, device=self.device)
            if observation_valid is None
            else torch.as_tensor(observation_valid, device=self.device)
            .bool()
            .reshape(-1)
        )
        if valid.shape != (len(full_xyz),) or not bool(valid.any()):
            raise ValueError("observation_valid must contain at least one aligned row")
        self.raw_observation_valid = valid
        self.observation_lift_mode = str(observation_lift_mode)
        if self.observation_lift_mode == "observed_domain":
            solver_rows = torch.where(valid)[0].cpu().numpy()
            solver_valid = torch.ones(len(solver_rows), dtype=torch.bool, device=self.device)
        elif self.observation_lift_mode == "full_cache_legacy":
            solver_rows = np.arange(len(full_xyz), dtype=np.int64)
            solver_valid = valid
        else:
            raise ValueError(
                "observation_lift_mode must be observed_domain or full_cache_legacy"
            )
        self.observation_lift = _ObservationDomainLift(
            full_xyz,
            solver_rows,
            neighbors=int(observation_lift_neighbors),
            maximum_distance_m=float(observation_lift_maximum_distance_m),
            device=self.device,
        )
        self.features = F.normalize(
            torch.as_tensor(full_features[solver_rows], device=self.device).float(),
            dim=-1,
            eps=1e-8,
        )
        self.appearance_features = F.normalize(
            torch.as_tensor(appearance[solver_rows], device=self.device).float(),
            dim=-1,
            eps=1e-8,
        )
        self.boundary_features = F.normalize(
            torch.as_tensor(boundary[solver_rows], device=self.device).float(),
            dim=-1,
            eps=1e-8,
        )
        self.observation_valid = solver_valid
        graph = build_primitive_support_graph(
            torch.as_tensor(full_xyz[solver_rows]),
            appearance_features=torch.as_tensor(appearance[solver_rows]),
            boundary_features=torch.as_tensor(boundary[solver_rows]),
            config=graph_config,
        )
        self.graph = graph_for_query_intent(
            graph,
            QueryIntent.INSTANCE,
            policy="typed",
        ).to(self.device)
        # The normalized graph Laplacian is scene-only state.  AGILE3D invokes
        # this predictor for every click of every object in a scene, so keeping
        # this exact tensor avoids rebuilding it thousands of times without
        # changing any query evidence, hard seed, or output.
        self.normalized_affinity = normalized_laplacian_affinity(self.graph)
        self.solver_config = solver_config
        self.selection_mode = SelectionMode(selection_mode)
        self.unary_mode = str(unary_mode)
        if self.unary_mode not in {"legacy_radio", "shared_capability"}:
            raise ValueError("unary_mode must be legacy_radio or shared_capability")
        self.appearance_unary_weight = float(appearance_unary_weight)
        self.boundary_unary_weight = float(boundary_unary_weight)
        if min(self.appearance_unary_weight, self.boundary_unary_weight) < 0:
            raise ValueError("capability unary weights cannot be negative")
        if self.appearance_unary_weight + self.boundary_unary_weight <= 0:
            raise ValueError("at least one capability unary weight must be positive")

    def observation_lift_report(self) -> dict[str, int | float | str]:
        return {
            "mode": self.observation_lift_mode,
            **self.observation_lift.report(),
        }

    def _shared_capability_unary(
        self,
        positive: list[int],
        negative: list[int],
    ) -> torch.Tensor:
        """Use the same official capability banks as the canonical 3-D route.

        With no negative click, the unlabeled scene mean is the fixed
        background prototype already used by ``compile_world_3d_query``.  It
        is calculated over observed rows only, so an unobserved zero feature
        is never silently treated as semantic background.
        """

        def margin(bank: torch.Tensor) -> torch.Tensor:
            foreground = (bank @ bank[positive].T).amax(dim=1)
            if negative:
                background = (bank @ bank[negative].T).amax(dim=1)
            else:
                scene_mean = F.normalize(
                    bank[self.observation_valid].mean(dim=0, keepdim=True),
                    dim=-1,
                    eps=1e-8,
                )
                background = (bank @ scene_mean.T).amax(dim=1)
            return foreground - background

        appearance = margin(self.appearance_features)
        boundary = margin(self.boundary_features)
        total = self.appearance_unary_weight + self.boundary_unary_weight
        return (
            self.appearance_unary_weight * appearance
            + self.boundary_unary_weight * boundary
        ) / total

    @torch.inference_mode()
    def __call__(
        self,
        _coordinates: np.ndarray,
        _previous: np.ndarray,
        clicks: Sequence[Click],
    ) -> np.ndarray:
        positive_points = [click.point_index for click in clicks if click.is_positive]
        negative_points = [click.point_index for click in clicks if not click.is_positive]
        if not positive_points:
            return np.zeros(self.observation_lift.full_count, dtype=bool)
        positive_weights = self.observation_lift.lift_clicks(positive_points)
        negative_weights = self.observation_lift.lift_clicks(negative_points)
        if not bool((positive_weights >= self.solver_config.hard_seed_threshold).any()):
            # An unobserved click that has no local observed primitive cannot
            # safely seed a semantic support graph.  The released evaluator
            # still forces the clicked point itself; we deliberately avoid
            # hallucinating a remote instance from no observation evidence.
            return np.zeros(self.observation_lift.full_count, dtype=bool)
        positive = torch.where(positive_weights > 0)[0].tolist()
        negative = torch.where(negative_weights > 0)[0].tolist()
        if self.unary_mode == "shared_capability":
            unary = self._shared_capability_unary(positive, negative)
        else:
            positive_similarity = (
                self.features @ self.features[positive].T
            ).amax(dim=1)
            if negative:
                negative_similarity = (
                    self.features @ self.features[negative].T
                ).amax(dim=1)
                unary = positive_similarity - negative_similarity
            else:
                # Kept solely as a diagnostic for the original AGILE reader.
                unary = positive_similarity - positive_similarity.median()
        probabilities = solve_primitive_support(
            self.graph,
            unary,
            positive_seeds=SoftSeedSet(positive_weights, "registered_3d_positive_click"),
            negative_seeds=(
                SoftSeedSet(negative_weights, "registered_3d_negative_click")
                if negative
                else None
            ),
            config=self.solver_config,
            normalized_affinity=self.normalized_affinity,
        )
        active = probabilities >= self.solver_config.support_threshold
        if self.selection_mode is SelectionMode.SEEDED_COMPONENT:
            active &= select_support_components(
                self.graph,
                probabilities,
                self.selection_mode,
                positive_seeds=SoftSeedSet(
                    positive_weights, "registered_3d_positive_click"
                ),
                config=self.solver_config,
            )
        elif self.selection_mode is not SelectionMode.ALL_COMPONENTS:
            raise ValueError(
                "AGILE3D point prediction supports seeded_component or all_components"
            )
        projected_probability = self.observation_lift.project(probabilities)
        projected_component = self.observation_lift.project(active.float())
        return (
            (projected_probability >= self.solver_config.support_threshold)
            & (projected_component >= self.solver_config.support_threshold)
        ).cpu().numpy()


def _load_feature_cache(
    path: Path,
    xyz: np.ndarray,
    *,
    quantized_unique_map: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        result = {key: np.asarray(payload[key]) for key in payload.files}
    if "radio_features" not in result:
        raise ValueError(f"{path} lacks radio_features")
    rows = int(result["radio_features"].shape[0])
    quantized_rows = int(len(quantized_unique_map))
    if result["radio_features"].ndim != 2 or rows not in {len(xyz), quantized_rows}:
        raise ValueError(
            f"{path} radio_features align with neither full nor quantized PLY"
        )
    is_quantized = rows == quantized_rows and rows != len(xyz)
    expected_xyz = xyz[quantized_unique_map] if is_quantized else xyz
    if "xyz" in result and (
        result["xyz"].shape != expected_xyz.shape
        or not np.allclose(result["xyz"], expected_xyz, atol=1e-5, rtol=0.0)
    ):
        raise ValueError(f"{path} xyz do not match its declared PLY domain")
    if is_quantized:
        if "unique_map" not in result or not np.array_equal(
            np.asarray(result["unique_map"], dtype=np.int64),
            np.asarray(quantized_unique_map, dtype=np.int64),
        ):
            raise ValueError(f"{path} quantized feature rows lack the exact unique map")
    for key in ("appearance_features", "boundary_features"):
        if key in result and result[key].shape[0] != rows:
            raise ValueError(f"{path} {key} do not align with the official PLY")
    if "valid" in result and np.asarray(result["valid"]).reshape(-1).shape != (rows,):
        raise ValueError(f"{path} valid does not align with the feature rows")
    result["_is_quantized"] = np.asarray(is_quantized)
    return result


def evaluate(args: argparse.Namespace) -> dict:
    root = Path(args.benchmark_root)
    feature_root = Path(args.feature_root)
    objects = load_official_object_list(root)
    by_scene: dict[str, list] = defaultdict(list)
    for item in objects:
        by_scene[item.scene_id].append(item)
    if args.scene_names:
        requested = set(str(args.scene_names).replace(",", " ").split())
        unknown = requested - set(by_scene)
        if unknown:
            raise ValueError(f"unknown official validation scenes: {sorted(unknown)}")
        by_scene = {name: by_scene[name] for name in sorted(requested)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    scene_coverage = []
    for scene_index, (scene_id, scene_objects) in enumerate(sorted(by_scene.items())):
        xyz, colors, labels_full = _read_official_ply(root / "scans" / f"{scene_id}.ply")
        quantized = quantize_scannet_points(
            xyz,
            colors,
            labels_full,
            voxel_size=float(args.voxel_size),
        )
        cache = _load_feature_cache(
            feature_root / f"{scene_id}.npz",
            xyz,
            quantized_unique_map=quantized.unique_map,
        )
        cache_rows = (
            slice(None)
            if bool(cache["_is_quantized"])
            else quantized.unique_map
        )
        cache_valid = np.asarray(
            cache.get("valid", np.ones(cache["radio_features"].shape[0], dtype=bool)),
            dtype=bool,
        ).reshape(-1)[cache_rows]
        if cache_valid.shape != (len(quantized.raw_coordinates),):
            raise ValueError(f"{scene_id} feature validity does not match quantized rows")
        coverage_row = {
            "scene_id": scene_id,
            "quantized_points": int(cache_valid.size),
            "valid_feature_points": int(cache_valid.sum()),
            "feature_coverage": float(cache_valid.mean()),
        }
        predictor = CanonicalPointPredictor(
            quantized.raw_coordinates,
            cache["radio_features"][cache_rows],
            appearance_features=(
                cache.get("appearance_features", None)[cache_rows]
                if "appearance_features" in cache
                else None
            ),
            boundary_features=(
                cache.get("boundary_features", None)[cache_rows]
                if "boundary_features" in cache
                else None
            ),
            observation_valid=cache_valid,
            device=args.device,
            graph_config=SupportGraphConfig(
                neighbors=int(args.graph_neighbors),
                topology_mode=args.topology_mode,
            ),
            solver_config=SupportSolverConfig(
                solver_type=args.solver_type,
                laplacian_weight=float(args.laplacian_weight),
                cg_iterations=int(args.cg_iterations),
                support_threshold=float(args.support_threshold),
                hard_seed_threshold=0.20,
            ),
            selection_mode=args.selection_mode,
            unary_mode=args.unary_mode,
            appearance_unary_weight=float(args.appearance_unary_weight),
            boundary_unary_weight=float(args.boundary_unary_weight),
            observation_lift_mode=args.observation_lift_mode,
            observation_lift_neighbors=int(args.observation_lift_neighbors),
            observation_lift_maximum_distance_m=float(
                args.observation_lift_maximum_distance_m
            ),
        )
        coverage_row["observation_lift"] = predictor.observation_lift_report()
        scene_coverage.append(coverage_row)
        for item in scene_objects:
            target_full = labels_full == int(item.object_id)
            target_quantized = quantized.labels == int(item.object_id)
            result = evaluate_interactive_predictions(
                quantized.raw_coordinates,
                target_quantized,
                target_full,
                quantized.inverse_map,
                predictor,
                max_clicks=int(args.max_clicks),
                click_workers=int(args.click_workers),
            )
            rows.append(
                {
                    "scene_id": scene_id,
                    "object_id": item.object_id,
                    "semantic_class": item.semantic_class,
                    **result,
                }
            )
        partial = {
            "benchmark": "AGILE3D ScanNet40 single-object",
            "completed_scenes": scene_index + 1,
            "total_scenes": len(by_scene),
            "completed_objects": len(rows),
            "scene_coverage": scene_coverage,
            "rows": rows,
        }
        output.write_text(json.dumps(partial))
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "protocol": {
            "official_preprocessed_data": str(root.resolve()),
            "voxel_size_m": float(args.voxel_size),
            "objects": len(rows),
            "scenes": len(by_scene),
            "max_clicks": int(args.max_clicks),
            "click_search_workers": int(args.click_workers),
            "click_policy": "center_of_largest_FP_or_FN_error_by_inradius",
            "clicked_labels_forced": True,
            "test_set_calibration": False,
            "selection_mode": str(args.selection_mode),
            "unary_mode": str(args.unary_mode),
            "appearance_unary_weight": float(args.appearance_unary_weight),
            "boundary_unary_weight": float(args.boundary_unary_weight),
            "observation_lift_mode": str(args.observation_lift_mode),
            "observation_lift_neighbors": int(args.observation_lift_neighbors),
            "observation_lift_maximum_distance_m": float(
                args.observation_lift_maximum_distance_m
            ),
        },
        "scene_coverage": scene_coverage,
        "coverage_summary": {
            "mean_feature_coverage": float(
                np.mean([row["feature_coverage"] for row in scene_coverage])
            ),
            "minimum_feature_coverage": float(
                np.min([row["feature_coverage"] for row in scene_coverage])
            ),
            "mean_projectable_fraction": float(
                np.mean(
                    [
                        row["observation_lift"]["projectable_fraction"]
                        for row in scene_coverage
                    ]
                )
            ),
            "minimum_projectable_fraction": float(
                np.min(
                    [
                        row["observation_lift"]["projectable_fraction"]
                        for row in scene_coverage
                    ]
                )
            ),
        },
        "metrics": aggregate_official_metrics(
            [row["trajectory"] for row in rows],
            max_clicks=int(args.max_clicks),
        ),
        "rows": rows,
    }
    output.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-names", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument(
        "--click-workers",
        type=int,
        default=-1,
        help=(
            "SciPy KD-tree query workers used only to schedule the exact "
            "official click search (-1 retains SciPy's all-core default)."
        ),
    )
    parser.add_argument("--graph-neighbors", type=int, default=16)
    parser.add_argument(
        "--topology-mode",
        choices=("symmetric_union", "mutual_knn"),
        default="symmetric_union",
    )
    parser.add_argument(
        "--solver-type",
        choices=("random_walker", "confidence_random_walker"),
        default="confidence_random_walker",
    )
    parser.add_argument("--laplacian-weight", type=float, default=1.0)
    parser.add_argument("--cg-iterations", type=int, default=64)
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument(
        "--unary-mode",
        choices=("shared_capability", "legacy_radio"),
        default="shared_capability",
        help=(
            "shared_capability aligns world-point unary evidence with the "
            "canonical DINO/SAM query compiler; legacy_radio is diagnostic only."
        ),
    )
    parser.add_argument("--appearance-unary-weight", type=float, default=1.0)
    parser.add_argument("--boundary-unary-weight", type=float, default=0.35)
    parser.add_argument(
        "--observation-lift-mode",
        choices=("observed_domain", "full_cache_legacy"),
        default="observed_domain",
        help=(
            "Solve only on observed canonical primitive rows and use fixed local "
            "geometry to lift clicks/support, or retain the zero-row legacy reader."
        ),
    )
    parser.add_argument("--observation-lift-neighbors", type=int, default=3)
    parser.add_argument(
        "--observation-lift-maximum-distance-m", type=float, default=0.10
    )
    parser.add_argument(
        "--selection-mode",
        choices=(
            SelectionMode.SEEDED_COMPONENT.value,
            SelectionMode.ALL_COMPONENTS.value,
        ),
        default=SelectionMode.SEEDED_COMPONENT.value,
        help=(
            "Use the shared canonical 3-D support selection; all_components is "
            "retained only as a diagnostic of the legacy global threshold."
        ),
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
