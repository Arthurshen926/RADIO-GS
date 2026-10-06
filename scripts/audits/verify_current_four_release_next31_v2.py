"""Active-four evidence preflight with explicit, exact historical code archives.

Never overwrites historical hashes. A current source mismatch can be resolved
only by a named archive with the *original* bound hash. This is evidence identity,
not a common-method, SOTA, dependency-installation, or independent-test check.
"""
import argparse
import json
from pathlib import Path

import yaml

from scripts.audits.audit_publication_readiness_next31 import ROOT, check, sha256
from scripts.audits.verify_current_four_release_next31 import SEALS, TASKS


ARCHIVES = {
    'scripts/audits/evaluate_full_field_rebuild_next19.py':
        'scripts/audits/frozen_sources_next31/evaluate_full_field_rebuild_next19.py',
    'reproductions/ludvig/run_ludvig_sam.py':
        'scripts/audits/frozen_sources_next31/ludvig_run_sam_20260801.py',
}


def resolve_bound_artifact(path, expected, *, root=ROOT, archives=ARCHIVES):
    """No fuzzy equivalence, new hash, or arbitrary archival path is accepted."""
    original = root / path
    actual = sha256(original) if original.is_file() else None
    row = dict(path=path, expected=expected, actual=actual,
               status='match' if actual == expected else 'missing' if actual is None else 'mismatch')
    if row['status'] == 'match' or path not in archives:
        return row
    archive = root / archives[path]
    archive_sha = sha256(archive) if archive.is_file() else None
    row.update(archive_path=archives[path], archive_sha256=archive_sha)
    if archive_sha == expected:
        row['original_status'] = row['status']
        row['status'] = 'archived_match'
    return row


def success(row):
    return row['status'] in ('match', 'archived_match')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    authority_path = ROOT / 'paper/artifacts/frozen_cross_benchmark_method_20260918.json'
    authority = json.loads(authority_path.read_text())
    contracts = {p: check(p, h) for p, h in authority['evaluation_protocols'].items()}
    if not all(success(row) for row in contracts.values()):
        raise RuntimeError('Frozen evaluation contract identity differs')
    freeze = yaml.safe_load((ROOT / 'paper/artifacts/evaluation_protocol_freeze_20260801.yaml').read_text())
    promptable = yaml.safe_load((ROOT / 'paper/artifacts/promptable_nvs_protocol_registry.yaml').read_text())
    local = promptable['protocols']['ludvig_official_online_multiview_v1']['local_reproduction']
    launcher = resolve_bound_artifact(local['launcher'], local['launcher_sha256'])
    task_checks = {task: [check(row['path'], row['sha256'])
                         for row in freeze['canonical_tasks'][task]['authoritative_artifacts']]
                   for task in TASKS}
    manifest_path = ROOT / authority['source_manifest']
    if sha256(manifest_path) != authority['source_manifest_sha256']:
        raise RuntimeError('Frozen source manifest identity differs')
    sources = json.loads(manifest_path.read_text())
    source_errors = [r for p, h in sources.items() if (r := check(p, h))['status'] != 'match']
    seal_checks = {}
    checked = {}
    for name in SEALS:
        path = ROOT / 'paper/artifacts' / (name + '.json')
        payload = json.loads(path.read_text())
        pairs = {}
        for key in ('artifacts', 'bindings', 'implementations', 'sources'):
            if isinstance(payload.get(key), dict):
                pairs.update({p: h for p, h in payload[key].items()
                              if isinstance(h, str) and len(h) == 64})
        if not pairs:
            raise RuntimeError(f'No directly bound artifacts: {path}')
        errors, resolved = [], []
        for p, h in pairs.items():
            if (p, h) not in checked:
                checked[p, h] = resolve_bound_artifact(p, h)
            row = checked[p, h]
            if not success(row):
                errors.append(row)
            elif row['status'] == 'archived_match':
                resolved.append(row)
        seal_checks[name] = dict(seal_sha256=sha256(path), direct_bindings_checked=len(pairs),
                                errors=errors, exact_archive_resolutions=resolved)
        print(json.dumps(dict(seal=name, checked=len(pairs), errors=len(errors))), flush=True)
    passed = (not source_errors and success(launcher)
              and all(success(r) for rows in task_checks.values() for r in rows)
              and all(not r['errors'] for r in seal_checks.values()))
    report = dict(schema='active_four_release_evidence.next31.v2', evidence_inventory_pass=passed,
                  unified_method_certified=False, sota_certified=False, independent_test_certified=False,
                  historical_global_freeze_replaced=False, direct_bindings_only=True,
                  contract_identity=contracts, active_external_tasks=task_checks,
                  retained_candidate_seals=seal_checks, unique_direct_bindings_checked=len(checked),
                  frozen_source_count=len(sources), frozen_source_errors=source_errors,
                  ludvig_launcher=launcher,
                  archives={p: dict(path=a, sha256=sha256(ROOT / a)) for p, a in ARCHIVES.items()},
                  quarantined_historical_tasks={
                      'spatial_spin9_ludvig': 'one required historical aggregate missing; not certified',
                      'correspondence_pfpr_ludvig_adapter': 'five external historical manifests missing; not certified'},
                  limitations=['Exact code archives are not execution dependency bundles',
                      'Retained controls are not a jointly promoted four-query method',
                      'Repeatedly examined benchmark results remain development evidence'],
                  inputs={str(p.relative_to(ROOT)): sha256(p)
                          for p in (authority_path, manifest_path, Path(__file__))})
    with (args.output_root / 'report.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(evidence_inventory_pass=passed, unique_direct_bindings_checked=len(checked))), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
