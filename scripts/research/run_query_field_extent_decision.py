"""Complete-cohort finite decisions on field-owned geometry and fixed scopes."""
import argparse
import copy
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.extent_decision import ExtentDecision
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes
from scripts.research.run_query_field_canonical_subset_reference import ROOT as PARENT
from scripts.research.compile_query_field_extent_factors import ROOT as FIELDS, PROTOCOL

ROOT = Path('output/query_field_20261011/extent_decision_v1')
ARMS = ['seed_addition', 'dense_joint', 'source_initial', 'extent_discrete']
CODE = [__file__, 'scripts/research/compile_query_field_extent_factors.py',
        'radio_gs/query_field/extent_factors.py', 'radio_gs/query_field/extent_decision.py',
        'radio_gs/query_field/coverage_factors.py', 'radio_gs/query_field/field.py',
        'radio_gs/query_field/storage.py', 'radio_gs/pipeline/subset.py']


def prepare():
    assert read(PARENT/'summary.json')['complete']
    ROOT.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(read(PARENT/'prediction_config.json'))
    cfg.update(semantic_controls=ARMS, status='fixed privileged scopes; field-owned geometry; finite binary decisions')
    records = []
    for j in cfg['jobs']:
        p = PARENT/'predictions'/j['name']
        records += [dict(scene=j['scene'], **receipt(x)) for x in [p/'predictions.pt', p/'prediction.json',
                    p/'admitted_optimization_state.pt', p/'identity_evidence/inputs.json',
                    Path(j['field'])/'manifest.json', Path(j['construction'])]]
        inp = read(p/'identity_evidence/inputs.json')
        records += [dict(scene=j['scene'], **receipt(x)) for x in [Path(inp['scene']), Path(inp['response']), *Path(inp['detections']).glob('identity_*.pt')]]
    write(ROOT/'prediction_config.json', cfg)
    write(ROOT/'design.json', dict(configuration=receipt(ROOT/'prediction_config.json'), protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in CODE], inputs=records, primary='extent_discrete', max_passes=4,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        adaptive_development=True, privileged_scopes=True, query_parameter_fitting=False, default_promoted=False))


def design(scene=None):
    d = read(ROOT/'design.json')
    for r in d['code'] + [d['configuration'], d['protocol']]: check(r)
    for r in d['inputs']:
        if scene is None or r['scene'] == scene: check(r)
    return d


def predict(index):
    cfg = read(ROOT/'prediction_config.json'); j = cfg['jobs'][index]; scene = j['scene']; d = design(scene)
    out = ROOT/'predictions'/j['name']; out.mkdir(parents=True, exist_ok=False)
    parent = PARENT/'predictions'/j['name']
    execute(['-m', 'radio_gs.pipeline.subset', '--inputs', str(parent/'identity_evidence/inputs.json'),
        '--key', scene, '--output-base', str(out/'scope_export'), '--mode', 'admission', '--scope-initialization', 'source',
        '--pixel-loss', 'balanced', '--support-prior', 'footprint', '--raster-scale', '2', '--steps', '240',
        '--scope-state-out', str(out/'privileged_scope.pt'), '--scope-state-only'], out/'scope_export.log')
    evidence = torch.load(out/'privileged_scope.pt', weights_only=True)
    state = torch.load(parent/'admitted_optimization_state.pt', weights_only=True)
    old = torch.load(parent/'predictions.pt', weights_only=True)
    field, manifest = load_field(FIELDS/scene/'field')
    for k, v in old['geometry'].items(): assert torch.equal(getattr(field, k), v)
    assert torch.equal(field.observed | field.semantic_measure.supported, old['observed'])
    assert evidence['texts'] == old['texts'] and evidence['source_raster'] == [384, 512]
    assert evidence['source_frames'] == [r['frame'] for r in read(j['construction'])['source_records']]
    decoder = ExtentDecision(field.extent_factors, len(field.xyz))
    members = dict(seed_addition=old['membership_by_control']['seed_addition'], dense_joint=old['membership_by_control']['admitted_joint'],
                   source_initial=old['membership_by_control']['admitted_fusion'])
    result = members['source_initial'].clone(); details = []
    for qi, text in enumerate(old['texts']):
        began = time.monotonic(); initial = members['source_initial'][:, qi] > .5
        if qi not in state:
            assert not any(bool(v['confidence'][qi] > 0 and v['positive'][:, qi].any()) for v in evidence['views'])
            assert torch.equal(result[:, qi], old['membership_by_control']['semantic_initial'][:, qi])
            details.append(dict(query=text, fallback=True, seconds=time.monotonic()-began)); continue
        fixed = state[qi]; confidence = np.zeros(18)
        active = [vi for vi, v in enumerate(evidence['views']) if bool(v['confidence'][qi] > 0 and v['positive'][:, qi].any())]
        assert active == fixed['view_indices']
        for vi, c, n in zip(active, fixed['confidence'], fixed['negative']):
            v = evidence['views'][vi]
            assert float(v['confidence'][qi]) == c and torch.equal(v['negative'][:, qi], n)
            confidence[vi] = c
        assert torch.equal(fixed['start_membership'], members['source_initial'][:, qi])
        target = torch.cat([v['target'][:, qi] for v in evidence['views']]).numpy()
        positive = torch.cat([v['positive'][:, qi] for v in evidence['views']]).numpy()
        negative = torch.cat([v['negative'][:, qi] for v in evidence['views']]).numpy()
        # This call performs no IO. Inputs retain the research-only privilege.
        selected, report = decoder.solve(initial.numpy(), target, positive, negative, confidence,
                                          fixed['support_cost'].numpy(), max_passes=d['max_passes'])
        assert not selected[~old['observed'].numpy()].any()
        result[:, qi] = torch.from_numpy(selected).float()
        details.append(dict(query=text, seconds=time.monotonic()-began, **report))
        print(scene, qi, text, report['energy_trace'], details[-1]['seconds'], flush=True)
    members['extent_discrete'] = result
    sets = {a: v > .5 for a, v in members.items()}
    old.update(scores={a: old['scores']['semantic_initial'] for a in ARMS}, membership_by_control=members,
               selected_by_control=sets, selected=sets['extent_discrete'])
    torch.save(old, out/'predictions.pt')
    write(out/'prediction.json', dict(configuration=cfg, job=j, queries=len(old['texts']),
        predictions_sha256=receipt(out/'predictions.pt')['sha256'], diagnostics=details,
        field_manifest=receipt(FIELDS/scene/'field/manifest.json'), scope=receipt(out/'privileged_scope.pt'),
        scopes_initial_cost_confidence_exact=True, source_access_in_scope_preparation=True,
        numeric_decoder_source_access=False, transient_query_fitting=False, standard_deployment=False,
        target_labels_opened=False, default_promoted=False))


def run(gpus):
    d = design(); cfg = read(d['configuration']['path'])
    for i, j in enumerate(cfg['jobs']):
        if not (FIELDS/j['scene']/'compilation.json').exists():
            execute(['-m', 'scripts.research.compile_query_field_extent_factors', '--index', str(i)], ROOT/'logs'/f'compile_{i}.log', gpus[0])
    def pred(i, gpu):
        execute(['-m', 'scripts.research.run_query_field_extent_decision', 'predict', '--index', str(i)], ROOT/'logs'/f'predict_{i}.log', gpu)
    lanes(list(range(4)), gpus, pred)
    records = []
    for j in cfg['jobs']:
        path = ROOT/'predictions'/j['name']/'prediction.json'; r = read(path)
        check(dict(path=str(path.with_name('predictions.pt')), sha256=r['predictions_sha256']))
        records.append(receipt(path))
    write(ROOT/'predictions_complete.json', dict(records=records, queries=68, controls=4))
    def score(j, gpu):
        execute(['-m', 'radio_gs.query_field.public_lerf', 'score', '--config', str(ROOT/'prediction_config.json'), '--job', j['name'],
                 '--output', str(ROOT/'predictions')], ROOT/'logs'/('score_'+j['scene']+'.log'), gpu)
    lanes(cfg['jobs'], gpus, score)
    per = {a: {} for a in ARMS}
    for j in cfg['jobs']:
        metrics = read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        previous = read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a, m in metrics.items():
            for k in ['iou', 'ap', 'localization', 'observations']: assert m[k] == previous['semantic_initial'][k]
            if a != 'extent_discrete':
                name = dict(dense_joint='admitted_joint', source_initial='admitted_fusion').get(a, a)
                for k in ['miou', 'count', 'missing']: assert m['selected'][k] == previous[name]['selected'][k]
            per[a][j['scene']] = m['selected']['miou']
    report = dict(complete=True, per_scene=per, macro={a: sum(v.values())/4 for a, v in per.items()},
        privileged_scope=True, transient_query_fitting=False, standard_deployment=False,
        predictions_before_scoring=True, default_promoted=False, whole_five_task=False)
    write(ROOT/'summary.json', report); print(report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'predict', 'run']); p.add_argument('--index', type=int); p.add_argument('--gpus', default='0')
    a = p.parse_args(); torch.set_num_threads(4)
    if a.stage == 'prepare': prepare()
    elif a.stage == 'predict': predict(a.index)
    else: run(list(map(int, a.gpus.split(','))))
