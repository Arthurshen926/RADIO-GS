"""Bind the complete fixed cohort, including failed gates and post-study audit."""
import json
from pathlib import Path

from radio_gs.query_field.storage import sha256
from scripts.research.prepare_competitive_evidence import ROOT, CONFIG, write
from scripts.research.run_competitive_evidence import verify_launch


def main():
    verify_launch()
    study = json.loads((CONFIG/'competitive_evidence_study.json').read_text())
    nvos = json.loads((ROOT/'nvos/evaluation.json').read_text())
    nvos_rows = {r['job']: r for r in nvos['results']}
    parent = nvos_rows['product_signed']
    prompts = json.loads((ROOT/'prompt_prescore.json').read_text())
    cold = json.loads((ROOT/'cold_replay.json').read_text())
    if len(cold['comparisons']) != 6 or not cold['original_open_rejected']:
        raise ValueError('incomplete cold replay')
    if not all(all(row['exact'].values()) for row in cold['comparisons']):
        raise ValueError('cold replay mismatch')
    evidence_paths = [ROOT/'launch.json', ROOT/'nvos/evaluation.json', ROOT/'prompt_prescore.json', ROOT/'cold_replay.json']
    configuration = json.loads((CONFIG/'competitive_evidence_nvos.json').read_text())
    runtimes = []
    for job in configuration['jobs']:
        path = ROOT/'nvos'/job['name']/'prediction.json'
        record = json.loads(path.read_text())
        if record['configuration'] != configuration or sha256(path.parent/'predictions.pt') != record['predictions_sha256']:
            raise ValueError('sealed prediction cohort changed')
        evidence_paths.append(path)
        runtimes.append(dict(job=job['name'], timing_seconds=record['timing_seconds'],
                             peak_cuda_allocated_bytes=record['peak_cuda_allocated_bytes'],
                             selected=record['selected']))
    models = []
    for job in study['models']:
        root = Path(job['output'])
        package = json.loads((root/'package.json').read_text())
        smoke = json.loads((root/'runtime_smoke.json').read_text())
        if not package['all_state_tensors_exact'] or not all(smoke['exact_replay'].values()) or smoke['forbidden_open_attempts']:
            raise ValueError('field verification incomplete')
        if sha256(Path(job['field'])/'field.pt') != package['field_sha256']:
            raise ValueError('frozen field changed')
        models.append(dict(**job, package=package, four_input_replay=smoke['exact_replay']))
        evidence_paths += [root/'package.json', root/'runtime_smoke.json', Path(job['field'])/'manifest.json']
    teacher, association, unsigned = {}, {}, {}
    for job in study['diagnostics']:
        scene = job['scene']
        path = ROOT/f'{scene}_signed_teacher.json'
        report = json.loads(path.read_text())
        if not report['unsigned_membership_exact']:
            raise ValueError('unsigned teacher response changed')
        teacher[scene] = report['summary']
        audit_path = ROOT/f'{scene}_identity_extent_audit.json'
        audit = json.loads(audit_path.read_text())
        if not audit['all_parent_teacher_ious_exact'] or audit['public_target_labels_opened']:
            raise ValueError('identity/extent diagnostic is not bound to the parent')
        association[scene] = audit['summary']
        evidence_paths += [path, audit_path]
    for job in study['unsigned_lerf']:
        path = ROOT/f"{job['scene']}_unsigned_lerf.json"
        report = json.loads(path.read_text())
        if not all(all(r['exact'].values()) for r in report['rows']):
            raise ValueError('unsigned LERF changed')
        unsigned[job['scene']] = dict(queries=len(report['rows'])//3, policies=3, all_exact=True,
                                     reference_prediction_sha256=report['reference_prediction_sha256'])
        evidence_paths.append(path)
    gates = {}
    for rule in ['competitive_pooled', 'competitive_rays']:
        row = nvos_rows[rule+'_signed']
        conditions = dict(prompt=all(prompts['gates'][rule].values()),
            teacher_scenes={scene: values[rule]['actual_subset_iou'] >= values['product']['actual_subset_iou']
                            for scene, values in teacher.items()},
            fern_response_improved=row['retained_response']['foreground_iou'] > parent['retained_response']['foreground_iou'],
            fern_ap_not_worse=row['retained_response']['average_precision'] >= parent['retained_response']['average_precision'],
            fern_subset_not_worse=row['foreground_iou'] >= parent['foreground_iou'], unsigned_exact=True)
        passed = all(value if isinstance(value, bool) else all(value.values()) for value in conditions.values())
        gates[rule] = dict(conditions=conditions, passed=passed)
    if any(row['passed'] for row in gates.values()):
        raise RuntimeError('passed candidate requires the predeclared second-task continuation')
    files = list(Path('radio_gs/query_field').glob('*.py'))
    files += list(Path('scripts/research').glob('*competitive_evidence*.py'))
    files += [Path('scripts/research/audit_query_field_identity_extent.py'),
              Path('tests/test_query_field_negative_evidence.py'), Path('tests/test_query_field_negative_inferred.py')]
    evidence = dict(schema='radio_gs.competitive_evidence_results.v1', parent_commit='a1a3b8e',
        models=models, signed_teacher=teacher, nvos=nvos, prompt_gates=prompts['gates'], gates=gates,
        source_only_identity_extent_audit=association, unsigned_lerf=unsigned, runtimes=runtimes,
        tests=dict(passed=63, batches=[59, 4],
            command='pytest -q tests/test_query_field*.py tests/test_query_preserving*.py tests/test_query_encoding_fixture.py (59 before supplemental file); pytest -q tests/test_query_field_negative_inferred.py (4)',
            note='Supplemental inferred text/image cases added after launch; core launch files unchanged.'),
        promotion=False, retained_candidate='single-support fitted physical member field; product negative rule',
        second_nvos_task_triggered=False, fits=0, parameter_sweep=False,
        caveats=['One historically opened NVOS case of eight; different v10 input/source/resource contract.',
                 'All current fields are high-capacity exports, not compact models.',
                 'Same-device inference replay is exact; prior cross-GPU renderer and training repeatability limits remain.',
                 'Small scene-mean regressions are fixed-gate failures, not statistical significance claims.',
                 'Post-study teacher-anchor audit is per-view mask association, not complete 3D object identity.'],
        code=[dict(path=str(p), sha256=sha256(p)) for p in sorted(set(files))],
        evidence=[dict(path=str(p), sha256=sha256(p)) for p in evidence_paths])
    write(Path('docs/experiments/2026-10-07-competitive-evidence-results.json'), evidence)
    print(json.dumps(dict(gates=gates, unsigned_lerf=unsigned, models=len(models)), indent=2))


if __name__ == '__main__':
    main()
