"""Build sealed, scene-disjoint AGILE source-train likelihood examples.

The formal 312-scene AGILE cohort is an evaluation-only inventory.  This
builder derives fit/development partitions solely from the released ScanNet
training list and refuses to open a scene PLY unless the sealed split marks it
as source-train.  Output shards align directly with ``QueryLikelihoodInputs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
import torch

from radio_gs.querying.query_compilers import world_point_soft_seed_matrix
from radio_gs.querying.query_likelihood_head import QueryLikelihoodInputs

from .evaluate_canonical_field import _read_official_geometry, _read_official_labels
from .protocol import Click, quantize_scannet_points, select_next_click


SPLIT_SCHEMA = "agile3d-likelihood-scene-split-v1"
SHARD_SCHEMA = "agile3d-query-likelihood-training-shard-v1"
DATASET_SCHEMA = "agile3d-query-likelihood-training-dataset-v1"
SPLIT_SALT = "radio-gs-agile3d-likelihood-split-v1"
SOURCE_KEY = re.compile(r"^(scene\d{4}_\d{2})_obj_(\d+)$")
HEAD_FEATURES = (
    "positive_affinity",
    "negative_affinity",
    "prior_probability",
    "coverage",
    "reliability",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_no_clobber(path: str | Path, value: Mapping[str, object]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError:
        if output.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different artifact: {output}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_torch_no_clobber(path: str | Path, value: object) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if output.exists():
        raise ValueError(f"refusing to replace training shard: {output}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _source_objects(path: str | Path) -> dict[str, int]:
    raw = _load_json(path)
    result: dict[str, int] = {}
    for key in raw:
        match = SOURCE_KEY.fullmatch(str(key))
        if match is None:
            raise ValueError(f"invalid released source-train key: {key!r}")
        scene, object_id = match.groups()
        if scene in result:
            raise ValueError(f"source-train list repeats scene: {scene}")
        result[scene] = int(object_id)
    return result


def _test_scenes(path: str | Path) -> list[str]:
    raw = _load_json(path)
    result: list[str] = []
    for key in raw:
        match = SOURCE_KEY.fullmatch(str(key))
        if match is None:
            raise ValueError(f"invalid released test key: {key!r}")
        result.append(match.group(1))
    if len(set(result)) != len(result):
        raise ValueError("released test list repeats scenes")
    return sorted(result)


def build_scene_split(
    *,
    source_train_list: str | Path,
    test_list: str | Path,
    development_count: int = 120,
) -> dict[str, object]:
    """Create a stable holdout within official train; open no PLY assets."""

    source_train_path = Path(source_train_list).expanduser().resolve()
    test_path = Path(test_list).expanduser().resolve()
    source_objects = _source_objects(source_train_path)
    test_scenes = _test_scenes(test_path)
    source_scenes = set(source_objects)
    if source_scenes & set(test_scenes):
        raise ValueError("official source-train and test scenes overlap")
    count = int(development_count)
    if count <= 0 or count >= len(source_scenes):
        raise ValueError("development count must leave non-empty fit/validation sets")
    ranked = sorted(
        source_scenes,
        key=lambda scene: (
            hashlib.sha256(f"{SPLIT_SALT}\0{scene}".encode()).hexdigest(),
            scene,
        ),
    )
    development = sorted(ranked[:count])
    fit = sorted(source_scenes - set(development))
    return {
        "schema_version": 1,
        "artifact_type": SPLIT_SCHEMA,
        "status": "sealed_before_any_scene_ply_label_open",
        "split_salt": SPLIT_SALT,
        "source_train_inventory": {
            "path": str(source_train_path),
            "sha256": sha256_file(source_train_path),
        },
        "test_inventory": {"path": str(test_path), "sha256": sha256_file(test_path)},
        "partitions": {
            "fit": fit,
            "development_validation": development,
            "test": test_scenes,
        },
        "source_object_ids": {
            scene: source_objects[scene] for scene in sorted(source_objects)
        },
        "counts": {
            "official_source_train": len(source_scenes),
            "fit": len(fit),
            "development_validation": len(development),
            "test": len(test_scenes),
        },
        "safety": {
            "scene_disjoint": True,
            "test_membership_read": True,
            "test_ply_geometry_opened": False,
            "test_ply_labels_opened": False,
            "development_labels_are_official_source_train_only": True,
        },
    }


def validate_scene_split(value: Mapping[str, object]) -> dict[str, object]:
    split = dict(value)
    if split.get("artifact_type") != SPLIT_SCHEMA:
        raise ValueError("unexpected AGILE likelihood split schema")
    partitions = split.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("split partitions are missing")
    fit = set(partitions.get("fit", ()))
    development = set(partitions.get("development_validation", ()))
    test = set(partitions.get("test", ()))
    if not fit or not development or not test:
        raise ValueError("fit/development/test partitions must be non-empty")
    if fit & development or fit & test or development & test:
        raise ValueError("AGILE likelihood partitions are not scene-disjoint")
    source_ids = split.get("source_object_ids")
    if not isinstance(source_ids, Mapping) or set(source_ids) != fit | development:
        raise ValueError("source object ids do not match source-train partitions")
    safety = split.get("safety")
    if not isinstance(safety, Mapping) or safety.get("test_ply_labels_opened") is not False:
        raise ValueError("split does not seal the test-label boundary")
    return split


def verify_scene_split_sources(value: Mapping[str, object]) -> dict[str, object]:
    """Rebind the sealed list files without opening any scene PLY."""

    split = validate_scene_split(value)
    for name in ("source_train_inventory", "test_inventory"):
        record = split.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"split lacks {name}")
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"sealed split source changed: {name}")
    return split


def _authorized_source_scene(
    split: Mapping[str, object], scene_id: str, partition: str
) -> int:
    validated = validate_scene_split(split)
    if partition not in {"fit", "development_validation"}:
        raise ValueError("training shards may use only fit/development_validation")
    allowed = validated["partitions"][partition]
    if scene_id not in allowed:
        if scene_id in validated["partitions"]["test"]:
            raise PermissionError(f"test scene labels are forbidden: {scene_id}")
        raise ValueError(f"scene is not in requested source partition: {scene_id}")
    return int(validated["source_object_ids"][scene_id])


def _nearest_signed_prediction(
    points: np.ndarray, clicks: Sequence[Click]
) -> np.ndarray:
    """Label-free rollout model used only to place the next source click."""

    xyz = np.asarray(points, dtype=np.float32)
    positive = [int(click.point_index) for click in clicks if click.is_positive]
    negative = [int(click.point_index) for click in clicks if not click.is_positive]
    if not positive:
        return np.zeros(len(xyz), dtype=bool)
    if not negative:
        return np.ones(len(xyz), dtype=bool)
    positive_distance = cKDTree(xyz[positive]).query(xyz, k=1, workers=1)[0]
    negative_distance = cKDTree(xyz[negative]).query(xyz, k=1, workers=1)[0]
    prediction = positive_distance <= negative_distance
    prediction[np.asarray(positive, dtype=np.int64)] = True
    prediction[np.asarray(negative, dtype=np.int64)] = False
    return prediction


def synthesize_click_trajectory(
    point_xyz: np.ndarray,
    point_target: np.ndarray,
    *,
    max_clicks: int,
    click_workers: int = 1,
) -> list[Click]:
    xyz = np.asarray(point_xyz, dtype=np.float32)
    target = np.asarray(point_target, dtype=bool).reshape(-1)
    if xyz.shape != (target.size, 3) or not bool(target.any()) or bool(target.all()):
        raise ValueError("source trajectory requires aligned nontrivial point target")
    prediction = np.zeros_like(target)
    clicks: list[Click] = []
    for order in range(int(max_clicks)):
        click = select_next_click(
            xyz, prediction, target, order=order, workers=int(click_workers)
        )
        if click is None:
            break
        clicks.append(click)
        prediction = _nearest_signed_prediction(xyz, clicks)
    if not clicks or not clicks[0].is_positive:
        raise AssertionError("released empty-prediction trajectory must start positive")
    return clicks


def build_training_payload(
    *,
    scene_id: str,
    object_id: int,
    partition: str,
    primitive_xyz: torch.Tensor,
    primitive_covariance: torch.Tensor,
    prior_probability: torch.Tensor,
    coverage: torch.Tensor,
    reliability: torch.Tensor,
    primitive_to_point_index: torch.Tensor,
    point_xyz: np.ndarray,
    point_target: np.ndarray,
    max_clicks: int,
    affinity_candidate_k: int,
    click_workers: int = 1,
    adapter: str,
) -> dict[str, object]:
    points = np.asarray(point_xyz, dtype=np.float32)
    target = torch.as_tensor(point_target).bool().reshape(-1)
    centers = torch.as_tensor(primitive_xyz).float()
    covariance = torch.as_tensor(primitive_covariance).float()
    rows = int(centers.shape[0])
    if centers.shape != (rows, 3) or covariance.shape != (rows, 3, 3):
        raise ValueError("primitive geometry must be [N,3]/[N,3,3]")
    mapping = torch.as_tensor(primitive_to_point_index).long().reshape(-1)
    if mapping.shape != (rows,) or bool((mapping < 0).any()) or bool(
        (mapping >= len(points)).any()
    ):
        raise ValueError("primitive_to_point_index must map every primitive to a point")
    inputs = QueryLikelihoodInputs(
        positive_affinity=torch.zeros((rows, 0)),
        negative_affinity=torch.zeros((rows, 0)),
        prior_probability=prior_probability,
        coverage=coverage,
        reliability=reliability,
    ).validated()
    clicks = synthesize_click_trajectory(
        points, target.numpy(), max_clicks=max_clicks, click_workers=click_workers
    )
    click_points = torch.from_numpy(
        np.ascontiguousarray(points[[click.point_index for click in clicks]])
    )
    affinity = world_point_soft_seed_matrix(
        centers,
        covariance,
        click_points,
        euclidean_candidate_k=int(affinity_candidate_k),
    ).float()
    if affinity.shape != (rows, len(clicks)) or bool(((affinity < 0) | (affinity > 1)).any()):
        raise AssertionError("registered click affinity differs from head contract")
    steps = []
    for click_count in range(1, len(clicks) + 1):
        prefix = clicks[:click_count]
        steps.append(
            {
                "click_count": click_count,
                "positive_columns": [
                    index for index, click in enumerate(prefix) if click.is_positive
                ],
                "negative_columns": [
                    index for index, click in enumerate(prefix) if not click.is_positive
                ],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": SHARD_SCHEMA,
        "head_schema_version": "monotone-query-likelihood-v1",
        "head_features": list(HEAD_FEATURES),
        "scene_id": str(scene_id),
        "object_id": int(object_id),
        "partition": str(partition),
        "adapter": str(adapter),
        "primitive_count": rows,
        "point_count": int(len(points)),
        "clicks": [
            {
                "point_index": int(click.point_index),
                "is_positive": bool(click.is_positive),
                "order": int(click.order),
            }
            for click in clicks
        ],
        "steps": steps,
        "click_affinity": affinity.cpu(),
        "prior_probability": inputs.prior_probability.cpu(),
        "coverage": inputs.coverage.cpu(),
        "reliability": inputs.reliability.cpu(),
        "primitive_to_point_index": mapping.cpu(),
        "primitive_target": target.index_select(0, mapping).float().cpu(),
        "point_target": target.float().cpu(),
        "safety": {
            "scene_authorized_before_label_open": True,
            "labels_opened": True,
            "label_scope": "official_source_train_scene_only",
            "source_train_labels_opened": True,
            "labels_from_official_source_train_scene": True,
            "test_labels_opened": False,
            "target_used_by_rollout_click_selector_only": True,
            "target_used_by_rollout_predictor": False,
            "target_used_by_affinity_registration": False,
        },
    }


def iter_head_training_examples(
    payload: Mapping[str, object],
) -> Iterator[tuple[QueryLikelihoodInputs, torch.Tensor, Mapping[str, object]]]:
    if payload.get("artifact_type") != SHARD_SCHEMA:
        raise ValueError("unexpected likelihood training shard")
    affinity = torch.as_tensor(payload["click_affinity"]).float()
    target = torch.as_tensor(payload["primitive_target"]).float().reshape(-1)
    for step in payload["steps"]:
        positive_columns = torch.as_tensor(step["positive_columns"], dtype=torch.long)
        negative_columns = torch.as_tensor(step["negative_columns"], dtype=torch.long)
        observations = QueryLikelihoodInputs(
            positive_affinity=affinity.index_select(1, positive_columns),
            negative_affinity=affinity.index_select(1, negative_columns),
            prior_probability=payload["prior_probability"],
            coverage=payload["coverage"],
            reliability=payload["reliability"],
        ).validated()
        if target.shape != (affinity.shape[0],):
            raise ValueError("primitive target does not align with head inputs")
        yield observations, target, step


def _load_primitive_bundle(
    path: Path,
    *,
    point_count: int,
    point_xyz_world: np.ndarray,
) -> tuple[dict[str, torch.Tensor], np.ndarray, str]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict):
        raise ValueError("primitive bundle must be a torch dictionary")
    required = {
        "primitive_xyz",
        "primitive_covariance",
        "prior_probability",
        "coverage",
        "reliability",
        "primitive_to_point_index",
    }
    if not required <= set(bundle):
        raise ValueError(f"primitive bundle misses {sorted(required - set(bundle))}")
    mapping = torch.as_tensor(bundle["primitive_to_point_index"]).long().reshape(-1)
    if mapping.numel() == 0 or bool((mapping < 0).any()) or bool((mapping >= point_count).any()):
        raise ValueError("primitive bundle mapping is outside official point domain")
    return {key: torch.as_tensor(bundle[key]) for key in required}, point_xyz_world, "canonical_primitive_bundle_v1"


def _point_primitive_adapter(
    point_xyz: np.ndarray, *, voxel_size: float
) -> tuple[dict[str, torch.Tensor], np.ndarray, str]:
    points = torch.from_numpy(np.ascontiguousarray(point_xyz)).float()
    count = len(points)
    covariance = (
        torch.eye(3).reshape(1, 3, 3).repeat(count, 1, 1)
        * float(voxel_size) ** 2
    )
    return {
        "primitive_xyz": points,
        "primitive_covariance": covariance,
        "prior_probability": torch.full((count,), 0.5),
        "coverage": torch.ones(count),
        "reliability": torch.ones(count),
        "primitive_to_point_index": torch.arange(count),
    }, np.asarray(point_xyz, dtype=np.float32), "released_5cm_point_identity_smoke_v1"


def materialize_scene(
    *,
    benchmark_root: str | Path,
    split_manifest: str | Path,
    contract: str | Path,
    scene_id: str,
    partition: str,
    output_root: str | Path,
    primitive_bundle: str | Path | None,
    max_clicks: int,
    affinity_candidate_k: int,
    voxel_size: float = 0.05,
    click_workers: int = 1,
) -> tuple[Path, dict[str, object]]:
    split_path = Path(split_manifest).expanduser().resolve()
    split = verify_scene_split_sources(_load_json(split_path))
    object_id = _authorized_source_scene(split, scene_id, partition)
    contract_path = Path(contract).expanduser().resolve()
    contract_payload = _load_json(contract_path)
    if contract_payload.get("artifact_type") != (
        "agile3d_monotone_query_likelihood_training_data_contract_v1"
    ):
        raise ValueError("unexpected learned-head training data contract")
    root = Path(benchmark_root).expanduser().resolve()
    ply_path = root / "scans" / f"{scene_id}.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"source-train PLY is missing: {ply_path}")

    # This is the first label open, after split and contract authorization.
    xyz, colors = _read_official_geometry(ply_path)
    labels = _read_official_labels(ply_path)
    quantized = quantize_scannet_points(xyz, colors, labels, voxel_size=voxel_size)
    point_target = quantized.labels == int(object_id)
    point_xyz_shifted = quantized.raw_coordinates
    scene_origin = xyz.min(axis=0, keepdims=True)
    point_xyz_world = point_xyz_shifted + scene_origin
    if primitive_bundle is None:
        bundle, affinity_points, adapter = _point_primitive_adapter(
            point_xyz_shifted, voxel_size=voxel_size
        )
        bundle_record = None
    else:
        bundle_path = Path(primitive_bundle).expanduser().resolve()
        bundle, affinity_points, adapter = _load_primitive_bundle(
            bundle_path,
            point_count=len(point_xyz_world),
            point_xyz_world=point_xyz_world,
        )
        bundle_record = {"path": str(bundle_path), "sha256": sha256_file(bundle_path)}
    payload = build_training_payload(
        scene_id=scene_id,
        object_id=object_id,
        partition=partition,
        primitive_xyz=bundle["primitive_xyz"],
        primitive_covariance=bundle["primitive_covariance"],
        prior_probability=bundle["prior_probability"],
        coverage=bundle["coverage"],
        reliability=bundle["reliability"],
        primitive_to_point_index=bundle["primitive_to_point_index"],
        point_xyz=affinity_points,
        point_target=point_target,
        max_clicks=max_clicks,
        affinity_candidate_k=affinity_candidate_k,
        click_workers=click_workers,
        adapter=adapter,
    )
    payload["source_authority"] = {
        "ply_path": str(ply_path),
        "ply_sha256": sha256_file(ply_path),
        "split_manifest": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "primitive_bundle": bundle_record,
    }
    output = Path(output_root).expanduser().resolve()
    shard_path = _write_torch_no_clobber(output / "shards" / f"{scene_id}.pt", payload)
    positive_count = sum(bool(item["is_positive"]) for item in payload["clicks"])
    receipt = {
        "schema_version": 1,
        "artifact_type": "agile3d-query-likelihood-training-shard-receipt-v1",
        "scene_id": scene_id,
        "object_id": object_id,
        "partition": partition,
        "adapter": adapter,
        "shard": {"path": str(shard_path), "sha256": sha256_file(shard_path)},
        "primitive_count": payload["primitive_count"],
        "point_count": payload["point_count"],
        "step_count": len(payload["steps"]),
        "positive_click_count": positive_count,
        "negative_click_count": len(payload["clicks"]) - positive_count,
        "head_features": list(HEAD_FEATURES),
        "labels_opened": True,
        "label_scope": "official_source_train_scene_only",
        "source_train_labels_opened": True,
        "test_labels_opened": False,
    }
    receipt_path = _write_json_no_clobber(
        output / "receipts" / f"{scene_id}.json", receipt
    )
    return receipt_path, receipt


def seal_dataset(
    *,
    output_root: str | Path,
    split_manifest: str | Path,
    contract: str | Path,
    receipts: Sequence[Mapping[str, object]],
) -> Path:
    output = Path(output_root).expanduser().resolve()
    split_path = Path(split_manifest).expanduser().resolve()
    contract_path = Path(contract).expanduser().resolve()
    split = validate_scene_split(_load_json(split_path))
    records = []
    seen: set[str] = set()
    for receipt in receipts:
        scene = str(receipt["scene_id"])
        if scene in seen:
            raise ValueError(f"dataset repeats scene: {scene}")
        seen.add(scene)
        _authorized_source_scene(split, scene, str(receipt["partition"]))
        shard = receipt["shard"]
        if sha256_file(shard["path"]) != shard["sha256"]:
            raise ValueError(f"training shard changed before sealing: {scene}")
        if receipt.get("test_labels_opened") is not False:
            raise ValueError("training receipt crosses the test-label boundary")
        if (
            receipt.get("labels_opened") is not True
            or receipt.get("label_scope") != "official_source_train_scene_only"
            or receipt.get("source_train_labels_opened") is not True
        ):
            raise ValueError("training receipt does not disclose its source labels")
        records.append(dict(receipt))
    if not records:
        raise ValueError("cannot seal an empty training dataset")
    manifest = {
        "schema_version": 1,
        "artifact_type": DATASET_SCHEMA,
        "status": "sealed_ready_for_monotone_query_likelihood_training",
        "head_schema_version": "monotone-query-likelihood-v1",
        "head_features": list(HEAD_FEATURES),
        "split_manifest": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "scene_count": len(records),
        "records": sorted(records, key=lambda row: row["scene_id"]),
        "safety": {
            "labels_opened": True,
            "label_scope": "official_source_train_scene_only",
            "source_train_labels_opened": True,
            "all_labels_from_official_source_train_scenes": True,
            "test_scene_intersection": [],
            "test_labels_opened": False,
            "existing_full312_trajectories_used": False,
            "easy3d_predictions_or_metrics_used": False,
            "full312_evaluation_authorized": False,
        },
    }
    return _write_json_no_clobber(output / "dataset_manifest.json", manifest)


def _create_split_command(args: argparse.Namespace) -> None:
    root = Path(args.benchmark_root).expanduser().resolve()
    split = build_scene_split(
        source_train_list=root / "train_list.json",
        test_list=root / "val_list.json",
        development_count=args.development_count,
    )
    output = _write_json_no_clobber(args.output, split)
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **split["counts"]}))


def _build_command(args: argparse.Namespace) -> None:
    receipts = []
    for scene in args.scene:
        bundle = (
            Path(args.primitive_bundle_root) / f"{scene}.pt"
            if str(args.primitive_bundle_root).strip()
            else None
        )
        _path, receipt = materialize_scene(
            benchmark_root=args.benchmark_root,
            split_manifest=args.split_manifest,
            contract=args.contract,
            scene_id=scene,
            partition=args.partition,
            output_root=args.output_root,
            primitive_bundle=bundle,
            max_clicks=args.max_clicks,
            affinity_candidate_k=args.affinity_candidate_k,
            voxel_size=args.voxel_size,
            click_workers=args.click_workers,
        )
        receipts.append(receipt)
    manifest = seal_dataset(
        output_root=args.output_root,
        split_manifest=args.split_manifest,
        contract=args.contract,
        receipts=receipts,
    )
    print(json.dumps({"manifest": str(manifest), "sha256": sha256_file(manifest), "scenes": list(args.scene)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("create-split")
    split.add_argument("--benchmark-root", required=True)
    split.add_argument("--development-count", type=int, default=120)
    split.add_argument("--output", required=True)
    split.set_defaults(func=_create_split_command)
    build = commands.add_parser("build")
    build.add_argument("--benchmark-root", required=True)
    build.add_argument("--split-manifest", required=True)
    build.add_argument("--contract", required=True)
    build.add_argument("--partition", choices=("fit", "development_validation"), required=True)
    build.add_argument("--scene", action="append", required=True)
    build.add_argument("--primitive-bundle-root", default="")
    build.add_argument("--output-root", required=True)
    build.add_argument("--max-clicks", type=int, default=10)
    build.add_argument("--affinity-candidate-k", type=int, default=64)
    build.add_argument("--voxel-size", type=float, default=0.05)
    build.add_argument("--click-workers", type=int, default=1)
    build.set_defaults(func=_build_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
