"""Cross-task evidence/decision audit; no fitted policy or benchmark thresholds.

Compile label-free source agreement for 22 jobs, seal every decision, then join
already-scored intact candidate metrics. No new dense prediction files are made.
Historical pool oracles are post-hoc diagnostics, never deployed predictions.
"""
import argparse
import gc
from pathlib import Path
import time

import numpy as np
import torch

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.response_study_adapters import SYSTEM, load, read, readout_path, source_observations
from radio_gs.response_study.decision_evidence import CompleteResponseAgreement, whole_response_metrics
from scripts.research.replay_tmm_response import STUDY, POLICY, sha

CONFIG = Path('configs/research/source_decision_audit.json')


def source_paths(job, data):
    root = SYSTEM/job['benchmark']/job['name']/'query'
    scene = read(job['scene'])
    paths = [Path(job['scene']), Path(job['queries']), readout_path(job), Path(data['semantic_geometry']['path'])]
    if job['benchmark'] == 'lerf':
        inputs = root/'text/identity_evidence/inputs.json'
        spec = read(inputs); scene = read(spec['scene'])
        paths += [inputs, Path(spec['scene'])]
        paths += [Path(spec['detections'])/f'identity_{f}.pt' for f in scene['source_frames']]
    elif job['benchmark'] == 'nvos':
        paths += [root/'reference/0/evidence'/f'identity_{f}.pt' for f in scene['source_frames']]
    else:
        paths += [Path(v[k]) for v in scene['source_evidence']['views'] for k in ('identity', 'contributions')]
    if 'camera_report' in scene:
        paths += [Path(scene['camera_report'])]
    return sorted(set(paths))


@torch.no_grad()
def predict(job, config):
    start = time.monotonic()
    source = STUDY/f"{config['candidate_source']}_{job['benchmark']}_{job['name']}"
    receipt = read(source/'prediction.json')
    if receipt['job'] != job or receipt['source_response_sha256'] != sha(readout_path(job)):
        raise ValueError('candidate and incumbent lineage mismatch')
    hashes = {str(source/p): sha(source/p) for p in receipt['prediction_files']}
    if any(hashes[str(source/p)] != h for p, h in receipt['prediction_files'].items()):
        raise ValueError('candidate no longer equals sealed E02 response')
    data = load(readout_path(job))
    prior = (data['category_logits'].sigmoid() if job['benchmark'] == 'scannet' else data['scores']).cuda()
    alternative = torch.from_numpy(np.stack([np.load(source/f'observable_{i:03d}.npy')
        for i in range(prior.shape[1])], 1)).cuda()
    candidates = torch.stack([prior, alternative])
    full = CompleteResponseAgreement(candidates, whole_menu=job['benchmark'] == 'scannet')
    control = CompleteResponseAgreement(candidates, whole_menu=job['benchmark'] == 'scannet')
    hashes.update({str(p): sha(p) for p in source_paths(job, data)})
    hashes[str(source/'prediction.json')] = sha(source/'prediction.json')
    rng = np.random.default_rng(20261016)
    for item, _ in source_observations(job, data, prior):
        full.add(item)
        # Only visible operator rows are shuffled, jointly across the whole menu.
        visible = torch.sparse.sum(item.operator, dim=1).to_dense().cpu().numpy() > 0
        ids = np.flatnonzero(visible); permutation = np.arange(len(visible))
        permutation[ids] = rng.permutation(ids)
        control.add(item, permutation=permutation)
    return dict(job=job, columns=data['texts'], full=full.finish(), permuted=control.finish(),
        input_hashes=hashes, ground_truth_read=False, seconds=time.monotonic()-start,
        complete_candidates_exact=True, new_dense_predictions=False,
        evidence_independent_of_candidate_construction=False)


def candidate_tables(job, configs):
    """Retained metrics, with one baseline shared across all historical arms."""
    names, tables, paths = [], [], []
    for run_name in configs['historical_capacity_runs']:
        folder = STUDY/f"{run_name}_{job['benchmark']}_{job['name']}"
        path = folder/('query_results.json' if job['benchmark'] == 'lerf' else 'evaluation.json')
        data = read(path)
        groups = data if job['benchmark'] == 'lerf' else data['metrics']
        if not tables:
            names.append('baseline'); tables.append(groups['baseline'])
        elif job['benchmark'] == 'lerf':
            a = {(r['frame'], r['query']): r['iou'] for r in tables[0]}
            b = {(r['frame'], r['query']): r['iou'] for r in groups['baseline']}
            if a != b:
                raise ValueError('historical baseline query records differ')
        elif any(abs(tables[0][k]-groups['baseline'][k]) > 1e-12 for k in tables[0]):
            raise ValueError('historical baseline metrics differ')
        for arm, values in groups.items():
            if arm != 'baseline':
                names.append(run_name+'/'+arm); tables.append(values)
        paths.append(path)
    return names, tables, paths


def score_decisions(report, config):
    job = report['job']; names, tables, paths = candidate_tables(job, config)
    candidate = names.index(config['candidate_source']+'/observable')
    key = {'lerf': 'semantic_iou', 'nvos': 'iou', 'scannet': 'miou_19'}[job['benchmark']]
    if job['benchmark'] == 'lerf':
        queries = sorted({r['query'] for r in tables[0]})
        utility = {q: [float(np.mean([r['iou'] for r in t if r['query'] == q])) for t in tables] for q in queries}
        choices = {'baseline': {q: 0 for q in queries}, 'observable': {q: candidate for q in queries}}
        for arm in ('full', 'permuted'):
            choices[arm] = {q: candidate if report[arm]['choices'][report['columns'].index(q)] else 0 for q in queries}
        choices['two_candidate_oracle'] = {q: candidate if utility[q][candidate] > utility[q][0] else 0 for q in queries}
        choices['historical_pool_oracle'] = {q: int(np.argmax(utility[q])) for q in queries}
        metrics = {arm: whole_response_metrics(tables, selected, tables[0]) for arm, selected in choices.items()}
        metrics = {arm: {'semantic_iou': m['iou'], 'ap': m['ap']} for arm, m in metrics.items()}
        units = [dict(query=q, baseline=utility[q][0], alternative=utility[q][candidate],
            evidence_gain=report['full']['gains'][1][report['columns'].index(q)],
            choices={a: v[q] for a, v in choices.items()}, candidate_utility=utility[q]) for q in queries]
    else:
        choices = {'baseline': 0, 'observable': candidate,
                   **{a: candidate if report[a]['choices'][0] else 0 for a in ('full', 'permuted')},
                   'two_candidate_oracle': candidate if tables[candidate][key] > tables[0][key] else 0,
                   'historical_pool_oracle': int(np.argmax([t[key] for t in tables]))}
        metrics = {a: tables[k] for a, k in choices.items()}
        units = [dict(query='whole_menu' if job['benchmark'] == 'scannet' else job['name'],
            baseline=tables[0][key], alternative=tables[candidate][key],
            evidence_gain=report['full']['gains'][1][0], choices=choices,
            candidate_utility=[t[key] for t in tables])]
    return dict(job=job, candidate_names=names, metrics=metrics, units=units,
                primary_metric=key, cached_score_hashes={str(p): sha(p) for p in paths},
                metric_status='exact whole-response choice from previously scored frozen predictions; no fresh target rendering',
                historical_pool_dense_replay=False)


def summarize(run, reports, config):
    scored = [score_decisions(report, config) for report in reports]
    macro, decisions = {}, {}
    for family in ('lerf', 'nvos', 'scannet'):
        local = [r for r in scored if r['job']['benchmark'] == family]
        macro[family] = {a: {k: float(np.mean([r['metrics'][a][k] for r in local])) for k in local[0]['metrics'][a]}
                         for a in local[0]['metrics']}
        units = [u for r in local for u in r['units']]
        decisions[family] = dict(units=len(units), available_improvements=sum(u['alternative'] > u['baseline']+1e-12 for u in units))
        for arm in ('full', 'permuted'):
            selected = [u for u in units if u['choices'][arm] != 0]
            decisions[family][arm] = dict(replacements=len(selected),
                improved=sum(u['alternative'] > u['baseline']+1e-12 for u in selected),
                harmed=sum(u['alternative'] < u['baseline']-1e-12 for u in selected))
    uqis_path = STUDY/'response_replacement_capacity_v1/summary.json'
    uqis = read(uqis_path)
    summary = dict(complete=True, diagnostic_only=True, default_changed=False,
        source_jobs=22, rows=scored, macro=macro, decisions=decisions,
        uqis_capacity_reference=dict(path=str(uqis_path), sha256=sha(uqis_path), macro=uqis['macro'],
            fresh_inference=False, pools_exactly_verified_by='A07 and L06'),
        lerf_selected_projection=dict(status='one frozen independent selection; no alternative supplied by E01-E04',
            iou=read(SYSTEM/'summary.json')['macro']['lerf']['selected_projection_iou'],
            finite_pool_gain=0., fresh_scoring=False),
        no_five_task_new_method_claim=True, evidence_held_out=False,
        dense_prediction_bytes=0, study_bytes=tree_bytes(STUDY))
    run.json('summary.json', summary)
    print('SUMMARY', macro, decisions, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='source_decision_audit_v1')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(4)
    config = read(CONFIG)
    jobs = [j for j in read('configs/pipeline/system_v10.json')['jobs'] if j['benchmark'] != 'uqis']
    jobs.sort(key=lambda j: (j['benchmark'], j['name']))
    run = BoundedExperiment(STUDY, args.run, POLICY, resume=args.resume)
    run.preflight(4*2**20)
    sources = [str(CONFIG), __file__, 'radio_gs/response_study/decision_evidence.py',
        'radio_gs/querying/response_study_adapters.py', 'radio_gs/querying/response_permissions.py',
        'radio_gs/querying/typed_evidence.py', 'scripts/research/fit_consolidated_response.py']
    manifest = dict(config=config, jobs=jobs, source_hashes={p: sha(p) for p in sources},
        candidates_seal_sha256=sha(STUDY/config['candidate_source']/'predictions_complete.json'))
    if args.resume:
        if read(run.root/'manifest.json') != manifest:
            raise ValueError('resume changed frozen protocol or implementation')
    else:
        run.json('manifest.json', manifest)
        run.json('source_snapshot.json', {p: dict(text=Path(p).read_text(), sha256=sha(p)) for p in sources})
    reports, receipts = [], {}
    for job in jobs:
        name = job['benchmark']+'_'+job['name']+'.json'
        if (run.root/name).exists():
            report = read(run.root/name)
            if any(sha(p) != h for p, h in report['input_hashes'].items()):
                raise ValueError('source inputs changed on resume')
        else:
            report = predict(job, config); run.json(name, report)
        reports.append(report); receipts[name] = sha(run.root/name)
        print('compiled', job['benchmark'], job['name'], report['full']['choices'], round(report['seconds'], 1), flush=True)
        gc.collect(); torch.cuda.empty_cache()
    if not (run.root/'decisions_complete.json').exists():
        run.json('decisions_complete.json', dict(jobs=len(reports), report_hashes=receipts,
            ground_truth_read=False, dense_candidates_exact=True, no_threshold_fit=True))
    if not (run.root/'summary.json').exists():
        summarize(run, reports, config)


if __name__ == '__main__':
    main()
