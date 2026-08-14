"""Stage one fail-closed method workspace for one UQIS query.

The benchmark authority contains evaluator-private pairing by necessity, but a
method must never receive that authority directory.  This module copies only a
single authorized query, its public mesh domain, and its declared query asset
into a fresh directory.  Launchers should mount that directory read-only and
discard it after the query so state cannot cross query identities.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from PIL import Image

from .protocol import (
    QUERY_MANIFEST_NAMES,
    QueryModality,
    audit_release,
    sha256_file,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def stage_query_workspace(
    release_root: str | Path,
    *,
    modality: str | QueryModality,
    query_id: str,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    """Materialize exactly one method-visible query in a fresh workspace."""

    root = Path(release_root).resolve()
    report = audit_release(root, check_files=True)
    if not report.get("valid"):
        raise ValueError("release audit failed before workspace staging")
    try:
        query_modality = QueryModality(str(getattr(modality, "value", modality)))
    except ValueError as error:
        raise ValueError(f"unknown UQIS modality: {modality!r}") from error
    source_manifest = root / QUERY_MANIFEST_NAMES[query_modality.value]
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    matches = [row for row in payload["queries"] if row["query_id"] == str(query_id)]
    if len(matches) != 1:
        raise ValueError("query_id is missing or duplicated in its modality manifest")
    query = dict(matches[0])
    scene_id = str(query["scene_id"])
    domains = [row for row in payload["scene_domains"] if row["scene_id"] == scene_id]
    if len(domains) != 1:
        raise ValueError("query scene domain is missing or duplicated")

    workspace = Path(workspace_dir).resolve()
    if workspace.exists():
        raise FileExistsError(f"refusing to overwrite query workspace: {workspace}")
    assets = workspace / "assets"
    assets.mkdir(parents=True)
    try:
        source_mesh = Path(str(domains[0]["mesh_xyz_path"])).resolve()
        mesh_path = assets / "mesh_xyz.npy"
        shutil.copyfile(source_mesh, mesh_path)
        domain = {
            **dict(domains[0]),
            "mesh_xyz_path": str(mesh_path),
            "mesh_xyz_sha256": sha256_file(mesh_path),
        }
        if query_modality is QueryModality.IMAGE:
            crop_path = assets / "query.png"
            with Image.open(query["crop_rgb_path"]) as image:
                image.convert("RGB").save(
                    crop_path,
                    format="PNG",
                    optimize=False,
                    compress_level=9,
                )
            query["crop_rgb_path"] = str(crop_path)
            query["crop_rgb_sha256"] = sha256_file(crop_path)

        method_manifest = {
            **{key: value for key, value in payload.items() if key not in {"scene_domains", "queries"}},
            "scene_domains": [domain],
            "queries": [query],
        }
        manifest_path = workspace / "query_manifest.json"
        _write_json(manifest_path, method_manifest)
        receipt = {
            "schema_version": "scannet_uqis_query_workspace_v1",
            "status": "staged",
            "formal_benchmark_eligible": False,
            "benchmark_version": payload["benchmark_version"],
            "release_manifest_sha256": sha256_file(root / "release.json"),
            "source_query_manifest_sha256": sha256_file(source_manifest),
            "workspace_query_manifest_sha256": sha256_file(manifest_path),
            "query_id": str(query_id),
            "scene_id": scene_id,
            "modality": query_modality.value,
            "query_count": 1,
            "independent_workspace_required": True,
            "mount_policy": "workspace_only_read_only",
            "evaluator_private_files_staged": False,
        }
        _write_json(workspace / "workspace_receipt.json", receipt)
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    return receipt

