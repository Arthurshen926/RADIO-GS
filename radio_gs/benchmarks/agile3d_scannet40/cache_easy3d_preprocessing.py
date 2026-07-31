#!/usr/bin/env python3
"""Cache Easy3D preprocessing from its real four-worker DataLoader path."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluate_easy3d import (
    EASY3D_AUDITED_COMMIT,
    OFFICIAL_OBJECT_IDS_SHA256,
    OFFICIAL_VOXEL_DATASET_SHA256,
    OFFICIAL_WORKER_CACHE_SCHEMA,
    assignment_quantization_diagnostics,
)
from .protocol import load_official_object_list


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: str | Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(path)), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        values = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


class OfficialEasy3DWorkerDataset:
    """Call the untouched VoxelDataset, then expose its full instance labels."""

    def __init__(
        self,
        data_root: str | Path,
        easy3d_repo: str | Path,
        scene_names: Sequence[str],
    ) -> None:
        repo = str(Path(easy3d_repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        voxel_dataset = importlib.import_module(
            "easy3d.dataset.voxel_dataset"
        ).VoxelDataset
        # Query masks are discarded. One query minimizes worker IPC/memory and
        # does not affect the preceding voxelization assignments.
        self.base = voxel_dataset(
            str(Path(data_root).resolve()),
            "val",
            False,
            1,
            0.05,
            40.0,
        )
        indices = {scene: index for index, scene in enumerate(self.base.scenes)}
        unknown = set(scene_names) - set(indices)
        if unknown:
            raise ValueError(f"official Easy3D val split lacks scenes: {unknown}")
        self.scene_names = list(scene_names)
        self.base_indices = [indices[scene] for scene in self.scene_names]
        self.data_root = Path(data_root).resolve()

    def __len__(self) -> int:
        return len(self.scene_names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from plyfile import PlyData

        scene_id = self.scene_names[index]
        item = self.base[self.base_indices[index]]
        vertex = PlyData.read(
            str(self.data_root / "scans" / f"{scene_id}.ply")
        )["vertex"]
        colors = (
            np.column_stack((vertex["R"], vertex["G"], vertex["B"]))
            .astype(np.float32)
            / 255.0
        )
        point_labels_numpy = np.asarray(vertex["label"], dtype=np.int32)
        point_labels = torch.from_numpy(point_labels_numpy.copy()).int()
        voxel_labels = torch.full(
            (len(item["voxel_coords"]),), -1, dtype=torch.int
        )
        # This is the exact released indexed assignment, executed in the same
        # DataLoader worker (one CPU thread) as its RGB/valid assignments.
        voxel_labels[item["point_voxel_id"]] = point_labels
        if not torch.equal(item["voxel_valid"], voxel_labels != -1):
            raise RuntimeError(
                f"{scene_id}: official RGB/label indexed writes disagree"
            )
        assigned_colors = (
            (item["voxel_features"][:, 3:] + 1.0) / 2.0
        ).numpy()
        diagnostics = assignment_quantization_diagnostics(
            colors,
            point_labels_numpy,
            item["point_voxel_id"].numpy(),
            assigned_colors,
            voxel_labels.numpy(),
        )
        worker = torch.utils.data.get_worker_info()
        return {
            "scene_id": scene_id,
            "worker_id": -1 if worker is None else int(worker.id),
            "torch_num_threads": int(torch.get_num_threads()),
            "coordinates": item["voxel_coords"],
            "features": item["voxel_features"],
            "voxel_labels": voxel_labels,
            "voxel_valid": item["voxel_valid"],
            "point_labels": point_labels,
            "inverse_map": item["point_voxel_id"].to(torch.int64),
            "quantization_diagnostics": diagnostics,
        }


def _validate_existing_cache(
    npz_path: Path,
    metadata_path: Path,
    expected: Mapping[str, Any],
) -> bool:
    if not npz_path.exists() and not metadata_path.exists():
        return False
    if not npz_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"incomplete Easy3D preprocessing cache; use a new root: {npz_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "existing Easy3D preprocessing cache has incompatible provenance; "
            f"use a new root: {mismatches}"
        )
    if _sha256(npz_path) != metadata.get("npz_sha256"):
        raise ValueError(f"Easy3D preprocessing cache hash failed: {npz_path}")
    return True


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if int(args.num_workers) != 4:
        raise ValueError(
            "the audited Easy3D preprocessing contract requires exactly four "
            "DataLoader workers"
        )
    data_root = Path(args.data_root).resolve()
    easy3d_repo = Path(args.easy3d_repo).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    easy3d_commit = _git_commit(easy3d_repo)
    object_ids_sha = _sha256(data_root / "single" / "object_ids.npy")
    voxel_dataset_path = (
        easy3d_repo / "easy3d" / "dataset" / "voxel_dataset.py"
    )
    voxel_dataset_sha = _sha256(voxel_dataset_path)
    if easy3d_commit != EASY3D_AUDITED_COMMIT:
        raise ValueError("preprocessing cache requires the audited Easy3D commit")
    if voxel_dataset_sha != OFFICIAL_VOXEL_DATASET_SHA256:
        raise ValueError("official Easy3D VoxelDataset source hash changed")
    objects = load_official_object_list(data_root)
    all_scenes = sorted({item.scene_id for item in objects})
    requested = {
        value
        for value in str(args.scene_names).replace(",", " ").split()
        if value
    }
    if requested - set(all_scenes):
        raise ValueError(f"unknown requested scenes: {requested - set(all_scenes)}")
    scenes = (
        sorted(requested)
        if requested
        else all_scenes
    )
    if not requested and (
        len(scenes) != 312 or object_ids_sha != OFFICIAL_OBJECT_IDS_SHA256
    ):
        raise ValueError("formal preprocessing cache requires canonical 312 scenes")
    expected_common = {
        "cache_schema": OFFICIAL_WORKER_CACHE_SCHEMA,
        "easy3d_commit": easy3d_commit,
        "voxel_dataset_sha256": voxel_dataset_sha,
        "object_ids_sha256": object_ids_sha,
        "data_root": str(data_root),
        "torch_version": torch.__version__,
        "torch_built_cuda": torch.version.cuda,
        "configured_num_workers": int(args.num_workers),
    }
    pending = []
    for scene_id in scenes:
        npz_path = output_root / "scenes" / f"{scene_id}.npz"
        metadata_path = output_root / "scenes" / f"{scene_id}.json"
        if not _validate_existing_cache(
            npz_path,
            metadata_path,
            {**expected_common, "scene_id": scene_id},
        ):
            pending.append(scene_id)
    if pending:
        dataset = OfficialEasy3DWorkerDataset(
            data_root, easy3d_repo, pending
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=int(args.num_workers),
            shuffle=False,
            persistent_workers=int(args.num_workers) > 0,
            prefetch_factor=(1 if int(args.num_workers) > 0 else None),
        )
        for processed, sample in enumerate(loader, start=1):
            scene_id = str(sample["scene_id"])
            if (
                int(sample["worker_id"]) < 0
                or int(sample["torch_num_threads"]) != 1
            ):
                raise RuntimeError(
                    "official Easy3D preprocessing must run in a one-thread "
                    "DataLoader worker"
                )
            arrays = {
                name: np.ascontiguousarray(sample[name].numpy())
                for name in (
                    "coordinates",
                    "features",
                    "voxel_labels",
                    "voxel_valid",
                    "point_labels",
                    "inverse_map",
                )
            }
            npz_path = output_root / "scenes" / f"{scene_id}.npz"
            metadata_path = output_root / "scenes" / f"{scene_id}.json"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_npz = npz_path.with_suffix(".tmp.npz")
            temporary_json = metadata_path.with_suffix(".tmp.json")
            if temporary_npz.exists() or temporary_json.exists():
                raise FileExistsError(
                    f"stale cache temporary exists; inspect manually: {scene_id}"
                )
            np.savez(temporary_npz, **arrays)
            metadata = {
                **expected_common,
                "scene_id": scene_id,
                "worker_id": int(sample["worker_id"]),
                "worker_torch_num_threads": int(
                    sample["torch_num_threads"]
                ),
                "array_content_sha256": _array_digest(arrays),
                "npz_sha256": _sha256(temporary_npz),
                "quantization_diagnostics": dict(
                    sample["quantization_diagnostics"]
                ),
            }
            temporary_json.write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            temporary_npz.replace(npz_path)
            temporary_json.replace(metadata_path)
            print(
                f"[{processed}/{len(pending)}] cached {scene_id} "
                f"worker={metadata['worker_id']}",
                flush=True,
            )
    manifest_rows = []
    for scene_id in scenes:
        metadata_path = output_root / "scenes" / f"{scene_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest_rows.append(
            {
                "scene_id": scene_id,
                "npz_sha256": metadata["npz_sha256"],
                "array_content_sha256": metadata["array_content_sha256"],
                "worker_id": metadata["worker_id"],
                "worker_torch_num_threads": metadata[
                    "worker_torch_num_threads"
                ],
            }
        )
    manifest = {
        **expected_common,
        "scene_count": len(scenes),
        "selection": "formal_312" if not requested else "declared_pilot",
        "scenes": manifest_rows,
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                "existing cache manifest differs; use a new output root"
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--easy3d-repo", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-names", default="")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    manifest = build_cache(args)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(Path(args.output_root).resolve()),
                "scene_count": manifest["scene_count"],
                "selection": manifest["selection"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
