"""Read-only publication evidence inventory; never repairs or blesses stale seals.

This checks artifact availability/identity, not scientific validity or SOTA.
Missing historical tasks remain failures even when outside the active cohort.
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def check(path, expected):
    target = ROOT / path
    if not target.is_file():
        return dict(path=path, status='missing', expected=expected)
    actual = sha256(target)
    return dict(path=path, status='match' if actual == expected else 'mismatch',
                expected=expected, actual=actual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze_path = ROOT / 'paper/artifacts/evaluation_protocol_freeze_20260801.yaml'
    freeze = yaml.safe_load(freeze_path.read_text())
    task_artifacts = {
        task: [check(row['path'], row['sha256']) for row in payload['authoritative_artifacts']]
        for task, payload in freeze['canonical_tasks'].items()
    }
    source_manifest = ROOT / 'output/frozen_cross_benchmark_method_20260918/source_manifest.json'
    sources = json.loads(source_manifest.read_text())
    source_errors = [row for path, expected in sources.items()
                     if (row := check(path, expected))['status'] != 'match']
    promptable_path = ROOT / 'paper/artifacts/promptable_nvs_protocol_registry.yaml'
    local = yaml.safe_load(promptable_path.read_text())['protocols'][
        'ludvig_official_online_multiview_v1']['local_reproduction']
    launcher = check(local['launcher'], local['launcher_sha256'])
    report = dict(
        schema='publication_evidence_inventory.next31.v1',
        artifact_availability_only=True,
        declares_sota=False,
        declares_independent_test=False,
        canonical_task_artifacts=task_artifacts,
        frozen_source_count=len(sources),
        frozen_source_errors=source_errors,
        ludvig_launcher=launcher,
        documented_ludvig_checkout_exists=Path('/root/baselines/LUDVIG').is_dir(),
        global_artifact_identity_pass=(not source_errors and launcher['status'] == 'match'
            and all(row['status'] == 'match' for rows in task_artifacts.values() for row in rows)),
        inputs={str(p.relative_to(ROOT)): sha256(p)
                for p in (freeze_path, source_manifest, promptable_path, Path(__file__))},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    errors = [row for rows in task_artifacts.values() for row in rows if row['status'] != 'match']
    print(json.dumps(dict(output=str(args.output), source_errors=len(source_errors),
                          historical_artifact_errors=len(errors), launcher_status=launcher['status'],
                          global_artifact_identity_pass=report['global_artifact_identity_pass'])))


if __name__ == '__main__':
    main()
