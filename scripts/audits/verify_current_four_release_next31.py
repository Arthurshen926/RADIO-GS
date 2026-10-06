"""Verify active four-task evidence and archive the exact historical launcher.

This is a new, explicitly narrower evidence inventory, not a replacement for the
seven-task canonical freeze validator. Unavailable SPIn/PFPR assets stay recorded
as missing by audit_publication_readiness_next31.py. A pass is not SOTA, unified
method, or independent-test certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from scripts.audits.audit_publication_readiness_next31 import ROOT, check, sha256


ARCHIVE_COMMIT = '4a668744707b93aa5394585810677e0a3f3e80ba'
TASKS = ('concept_lerf2d_occamlgs', 'concept_lerf3d_vala',
         'concept_scannet_ovs_vala_paper8', 'spatial_nvos_ludvig')
SEALS = ('frozen_four_benchmark_complete_20260919',
         'full_field_rebuild_next19_complete',
         'source_admission_adaptive_coverage_next29',
         'nvos_reference_completion_next19_runtime',
         'nvos_reference_completion_next19_eight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    freeze_path = ROOT / 'paper/artifacts/evaluation_protocol_freeze_20260801.yaml'
    freeze = yaml.safe_load(freeze_path.read_text())
    protocol_path = ROOT / 'paper/artifacts/promptable_nvs_protocol_registry.yaml'
    protocol = yaml.safe_load(protocol_path.read_text())
    reproduction = protocol['protocols']['ludvig_official_online_multiview_v1']['local_reproduction']
    launcher_path = reproduction['launcher']
    blob = subprocess.check_output(['git', 'show', f'{ARCHIVE_COMMIT}:{launcher_path}'], cwd=ROOT)
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    if blob_sha256 != reproduction['launcher_sha256']:
        raise RuntimeError('Archived launcher does not match sealed historical code')
    archive = args.output_root / 'historical_launcher.py'
    archive.write_bytes(blob)
    task_checks = {task: [check(row['path'], row['sha256'])
                         for row in freeze['canonical_tasks'][task]['authoritative_artifacts']]
                   for task in TASKS}
    manifest_path = ROOT / 'output/frozen_cross_benchmark_method_20260918/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_errors = [r for p, h in manifest.items() if (r := check(p, h))['status'] != 'match']
    seal_checks = {}
    checked = {}
    for name in SEALS:
        seal_path = ROOT / 'paper/artifacts' / (name + '.json')
        payload = json.loads(seal_path.read_text())
        pairs = {}
        for key in ('artifacts', 'bindings', 'implementations', 'sources'):
            if isinstance(payload.get(key), dict):
                pairs.update({p: h for p, h in payload[key].items()
                              if isinstance(h, str) and len(h) == 64})
        if not pairs:
            raise RuntimeError(f'No directly bound artifacts: {seal_path}')
        errors = []
        for path, expected in pairs.items():
            identity = (path, expected)
            if identity not in checked:
                checked[identity] = check(path, expected)
            if checked[identity]['status'] != 'match':
                errors.append(checked[identity])
        seal_checks[name] = dict(seal_sha256=sha256(seal_path),
                                direct_bindings_checked=len(pairs), errors=errors)
        print(json.dumps(dict(seal=name, checked=len(pairs), errors=len(errors))), flush=True)
    active_artifacts_pass = all(r['status'] == 'match' for rows in task_checks.values() for r in rows)
    passed = active_artifacts_pass and not source_errors and all(not r['errors'] for r in seal_checks.values())
    report = dict(schema='active_four_release_evidence.next31.v1', evidence_inventory_pass=passed,
                  unified_method_certified=False, sota_certified=False, independent_test_certified=False,
                  active_external_tasks=task_checks, retained_candidate_seals=seal_checks,
                  unique_direct_bindings_checked=len(checked), frozen_source_count=len(manifest),
                  frozen_source_errors=source_errors,
                  archived_launcher=dict(git_commit=ARCHIVE_COMMIT, git_path=launcher_path,
                      archived_path=str(archive), sha256=blob_sha256,
                      equals_frozen_registry=True, current_script_modified=False,
                      note='Code identity only; external checkout, dependencies and full rerun not certified'),
                  current_launcher=check(launcher_path, reproduction['launcher_sha256']),
                  quarantined_historical_tasks=['spatial_spin9_ludvig', 'correspondence_pfpr_ludvig_adapter'],
                  historical_global_freeze_replaced=False,
                  direct_bindings_only=True,
                  inputs={str(p.relative_to(ROOT)): sha256(p)
                          for p in (freeze_path, protocol_path, manifest_path, Path(__file__))})
    with (args.output_root / 'report.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(evidence_inventory_pass=passed, unique_direct_bindings_checked=len(checked))), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
