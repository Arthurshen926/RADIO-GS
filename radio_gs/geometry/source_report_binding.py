"""Resolve a source geometry report by authority, never by scene name alone."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def load_source_geometry_report(root, scene, report_path=None):
    root = Path(root)
    authority_path = root/'responsibility.json'
    info = json.loads((root/'teachers.json').read_text())
    if sha256_file(authority_path) != info['authority_sha256']:
        raise ValueError('source contribution authority changed')
    metadata = json.loads(authority_path.read_text())['metadata']
    path = Path(report_path) if report_path is not None else Path('output/geometry_only_repair_final_20260914')/scene/'report.json'
    if sha256_file(path) != metadata['frozen_geometry_report_sha256']:
        raise ValueError('geometry report differs from source authority; pass its bound --geometry-report')
    report = json.loads(path.read_text())
    if report['results']['independent_scale']['geometry_sha256'] != metadata['frozen_geometry_sha256']:
        raise ValueError('geometry bundle differs from source authority')
    return path, report
