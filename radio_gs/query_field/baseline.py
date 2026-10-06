"""Freeze saved v10 results and their declared primary asset dependencies.

This is an evidence inventory, not a baseline rerun or a fully portable archive.
Construction caches and query packages listed as directories remain explicit
unfrozen dependencies; they cannot silently count as a complete export.
"""
import json
from pathlib import Path

from .storage import sha256


SUMMARY = Path("output/query_conditioned_membership_20261006/image_convergence/system/summary.json")


def audit(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "baseline_a.json"
    if target.exists():
        raise FileExistsError("baseline receipt already exists; use another audit directory")
    receipts = {}

    def record(path):
        p = Path(path)
        resolved = str(p.resolve())
        if resolved not in receipts:
            if p.is_file():
                receipts[resolved] = dict(path=str(p), resolved=resolved, bytes=p.stat().st_size, sha256=sha256(p))
            else:
                receipts[resolved] = dict(path=str(p), resolved=resolved,
                    status="directory_not_content_frozen" if p.is_dir() else "missing")
        return resolved

    summary = json.loads(SUMMARY.read_text())
    record(SUMMARY)
    record(summary["recipe"])
    rows = []
    for row in summary["rows"]:
        job = row["job"]
        refs = [record(job[key]) for key in ["scene", "queries", "evaluation", "reference_tracks", "source_evidence", "package"] if key in job]
        scene_path = Path(job["scene"])
        scene = json.loads(scene_path.read_text())
        if "map_files" in scene:
            for name in scene["map_files"]:
                refs.append(record(scene_path.parent / name))
            for key in ["direct_anchor_membership", "inferred_anchor_membership"]:
                # Record every declared existing file in these model contracts.
                def visit(value):
                    if isinstance(value, dict):
                        for child in value.values():
                            visit(child)
                    elif isinstance(value, str) and len(value) < 2048 and Path(value).is_file():
                        refs.append(record(value))
                visit(scene.get(key, {}))
        else:
            for key in ["geometry", "native_field", "camera_report", "source_rgb_authority"]:
                value = scene.get(key)
                if isinstance(value, dict):
                    value = value.get("path")
                if isinstance(value, str):
                    refs.append(record(value))
            for value in scene.get("native_region_directories", []):
                refs.append(record(value))
        rows.append(dict(job=job, metrics=row["metrics"], asset_receipts=sorted(set(refs))))
    report = dict(schema="radio_gs.baseline_a_receipt.v1", source_snapshot_commit="0f40545c2b93fa390e37038af8315c24b5bc57b5",
        saved_results_only=True, rerun=False, macro=summary["macro"], jobs=rows,
        independent_test=summary.get("independent_test", False),
        official_sota_equivalence_verified=summary.get("official_sota_equivalence_verified", False),
        contracts={
            "lerf": "Source-assisted text union; selected_projection_iou is actual Gaussian subset 2D projection, not independent 3D instance truth.",
            "nvos": "Source-assisted retained positive-derived reference SAM and cached source tracks; not full signed official prompts.",
            "scannet": "Provided unlabelled mesh; complete category menu; source-assisted dual-native prior includes MaskAdapter in existing configs.",
            "uqis": "Map-only official mesh/RGBD condition; 9 already-opened development rooms. Group member model used additional 22-room labelled 3D training; text and image/point readout differ."
        }, receipts=list(receipts.values()), full_external_dependency_archive=False,
        pending="Hash/copy transitive directory contents, shared query encoders and runtime assets before claiming portable baseline reproduction.",
        new_candidate_supervision="Native DINO/SAM/SigLIP mapping observations only; do not import extra v10 3D labels implicitly.")
    target.write_text(json.dumps(report, indent=2)+"\n")
    return dict(path=str(target), sha256=sha256(target), jobs=len(rows),
                hashed_files=sum("sha256" in r for r in receipts.values()),
                unresolved_dependencies=sum("sha256" not in r for r in receipts.values()),
                full_external_dependency_archive=False, macro=summary["macro"])
