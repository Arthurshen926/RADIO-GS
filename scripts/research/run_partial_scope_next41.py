"""Fixed Task A pilot: one fresh fit per scene, two source-only selection arms."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/partial_scope_next41')
KEYS=['lerf_figurines','lerf_ramen','scannet_scene0000_00']
ARMS=['legacy_visible_iou','partial_known_brier']


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def input_bindings(parent):
    """Hash fixed fit inputs before and after the real scene jobs."""
    import torch
    payload=torch.load(parent,map_location='cpu',weights_only=False)
    root=Path(payload['root'])
    paths={parent,Path(payload['base_field']),Path(payload['geometry_report'])}
    paths.update(root/name for name in ['teachers.json','responsibility.json','caption_region_evidence.pt'])
    auth=json.loads((root/'responsibility.json').read_text())
    info=json.loads((root/'teachers.json').read_text())
    for frame in info['train_frames']:
        paths.add(root/auth['views'][info['frames'].index(frame)]['relative_path'])
    for extra in [root/'bounded_fallback_withheld/report.json',Path(payload['base_field']).parent/'report.json']:
        if extra.exists():paths.add(extra)
    return [{'path':str(p),'sha256':digest(p)} for p in sorted(paths)]


def main():
    ROOT.mkdir(exist_ok=True)
    run=ROOT/time.strftime('run_%Y%m%d_%H%M%S');run.mkdir()
    prior=json.loads((ROOT/'report.json').read_text()) if (ROOT/'report.json').exists() else None
    if prior is not None:
        (run/'prior_report.json').write_text(json.dumps(prior,indent=2)+'\n')
        (run/'prior_notes.md').write_text((ROOT/'notes.md').read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONPATH='.')
    report=dict(task='A',run=str(run),development=True,promotion=False,scenes={},commands=[],
                selection_contract=dict(ridges=[.1,1,10],primary='class-balanced known Brier; query-view equal',tie_tolerance=1e-6,tie='larger ridge',unknown='diagnostic only'),
                residual_policy='fresh single 60-step pipeline/odds fit per scene, shared by both arms',
                prior_run=dict(run=prior['run'],status='superseded for correctness: double-odds saturation',archived_report=str(run/'prior_report.json')) if prior else None,
                correctness_fix='pre-readout state retained; deployed odds forward; straight-through identity gradient approximation',
                correctness_passed=False,effect_passed=False,complete=False)
    def save():
        (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (run/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def call(args,log):
        command=[PYTHON]+args
        report['commands'].append(dict(argv=command,environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','PYTHONPATH']},log=str(log)))
        save()
        start=time.monotonic()
        with log.open('w') as f:rc=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        report['commands'][-1].update(exit_code=rc,seconds=time.monotonic()-start)
        save()
        if rc:raise RuntimeError(f'command failed ({rc}): {shlex.join(command)}; see {log}')
    try:
        gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        report['gpu0_preflight']=gpu or 'no compute processes'
        if gpu:raise RuntimeError('GPU0 occupied; no external process will be killed: '+gpu)
        call(['-m','pytest','-q','tests/test_partial_scope_metrics_next41.py'],run/'tests.log')
        for key in KEYS:
            parent=Path('output/taskwise_scope_composition_next38')/key/'field.pt'
            calibration=Path('output/scope_query_modes_next34')/key/'calibration.pt'
            row=dict(parent=dict(path=str(parent),sha256=digest(parent),calibration_path=str(calibration),calibration_sha256=digest(calibration)),arms={})
            row['input_bindings_before']=input_bindings(parent)
            report['scenes'][key]=row;save()
            call(['scripts/research/fit_source_kernel_extent.py','--key',key,'--baseline','pipeline','--readout','odds','--steps','60','--save-training','--dual-selection','--selection-rule','partial_known_brier','--output',str(run)],run/(key+'_fit.log'))
            training=run/key/'training.json';row['training']=str(training);row['source']=json.loads(training.read_text())
            import torch
            cache=torch.load(run/key/'source_training.pt',map_location='cpu',weights_only=False)
            row['shared_residual_sha256']=hashlib.sha256(cache['residual'].contiguous().numpy().tobytes()).hexdigest()
            for arm in ARMS:
                directory=run/arm/key
                payload=torch.load(directory/'field.pt',map_location='cpu',weights_only=False)
                assert torch.equal(payload['source_kernel_extent']['residual'],cache['residual'])
                row['arms'][arm]=dict(selection=json.loads((directory/'selection.json').read_text()),field=str(directory/'field.pt'),residual_identical=True)
                del payload
                call(['scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(run/arm),'--readout','anchored','--pipeline'],run/(key+'_'+arm+'_evaluation.log'))
                evaluation=directory/'anchored/pipeline/evaluation.json'
                row['arms'][arm].update(evaluation=str(evaluation),benchmark=json.loads(evaluation.read_text())['metrics']['hierarchical_pipeline'])
            assert digest(parent)==row['parent']['sha256'] and digest(calibration)==row['parent']['calibration_sha256']
            row['input_bindings_after']=input_bindings(parent)
            assert row['input_bindings_before']==row['input_bindings_after']
            row['inputs_unchanged']=True
            del cache
            save()
        report['correctness_passed']=True
        report['complete']=True
        # Strict pilot non-regression in every requested metric; no promotion follows.
        deltas=[]
        for key,row in report['scenes'].items():
            old=row['arms'][ARMS[0]]['benchmark'];new=row['arms'][ARMS[1]]['benchmark']
            if key.startswith('lerf'):
                d=dict(iou=new['iou']-old['iou'],ap=new['ap']-old['ap'],selected=new['selected']['miou']-old['selected']['miou'])
            else:d={s:new[s]['miou']-old[s]['miou'] for s in ['19','15','10']}
            row['selection_rule_benchmark_delta']=d;deltas.extend(d.values())
        report['effect_passed']=all(d>=-1e-8 for d in deltas) and any(d>1e-8 for d in deltas)
    except Exception as exc:
        report['error']=str(exc)
        raise
    finally:
        save()
        notes=['# Task A partial supervision pilot', '', 'One fresh 60-step pipeline/odds fit per scene. Both arms share the exact serialized residual. Source selection never reads benchmark GT.',
               'P=V&K&target; N=V&K&~target; U=V&~K. Missing support exports null; single-class episodes average only the supported class. Unknown prediction/change is diagnostic, never TP/FP.',
               'Metric semantics repair is shown by known_iou versus legacy_visible_iou on the SAME candidate in source.kernel_choices. Selection-rule change is shown by the two arms and benchmark deltas. These are separate effects.',
               'Legacy reproduces the historical visible-prediction IoU objective (including its historical target/empty-union behavior); it is not the corrected supervised metric. Unmarked deployment payload interpretation is unchanged.',
               'Fresh fit avoids unverifiable historical training bindings. Fixed seed 0. Serialized half residual/anchors are used during source candidate selection. Existing native map has seen held source views: not independent test. Source masks and negatives are pseudo labels. No B/C, default promotion, SOTA claim, or full-cohort conclusion.',
               'Outer review confirmed a double-odds saturation bug: the old fit overwrote pre-readout state with deployed odds probability, then serialized its difference and applied odds again. Prior run is superseded for correctness, preserved in place; prior root summaries are archived in this run. The corrected fit retains pre_state and serializes pre_state-baseline, while residuals and measures use the actual single-odds forward. Updates use a straight-through identity approximation, NOT exact chain rule. Regularization acts on the same pre-readout correction. Half residual serialization still quantizes values; the saturated regression uses absolute tolerance 3e-4 and an identity kernel to separate readout correctness from deliberate ridge smoothing.',
               'P/N/U, query-view equal weighting, selection metrics, ridge={0.1,1,10}, and tie rules are unchanged. Selected ridge and measured scores can change because the shared residual is refitted. This is a correctness repair, not a new selection objective or promotion.',
               'Unknown diagnostics do not override the declared 1e-6/larger-ridge tie rule.', '',f'Complete: {report["complete"]}; correctness: {report["correctness_passed"]}; strict pilot effect: {report["effect_passed"]}.',f'Error/uncompleted: {report.get("error","none for the three-scene Task A pilot; full cohorts and multi-seed acceptance remain outside this pilot")}.','', 'Raw training, per-query source statistics, benchmark scores/masks, and logs are linked in report.json. Exact commands:']
        notes += ['```\n'+ 'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=. '+shlex.join(c['argv'])+'\n```' for c in report['commands']]
        (ROOT/'notes.md').write_text('\n\n'.join(notes)+'\n')


def enrich_report():
    """Finish provenance and concise scene tables from completed raw artifacts."""
    import torch
    report=json.loads((ROOT/'report.json').read_text())
    report['metric_names']={'negative_selection':'qualified-negative FPR on N',
                            'known_brier':'class-balanced P/N probability squared error',
                            'unknown_selection_change':'fraction of U whose binary selection differs from parent'}
    report['test_log']=str(Path(report['run'])/'tests.log')
    report['acceptance_scope']='Task A three-scene correctness/development pilot only; outer assistant independent review pending'
    report['limitations']=['source pseudo-labels', 'base map saw held source views', 'single seed', 'not full cohorts or independent test', 'no promotion', 'peak GPU memory and isolated query latency not instrumented']
    report['input_bindings']={}
    report['costs']=dict(offline_seconds=sum(c.get('seconds',0) for c in report['commands']),peak_gpu_memory_bytes=None,per_query_latency_seconds=None,limitation='peak GPU memory and isolated query latency were not instrumented; command wall times and serialized file sizes are reported')
    table=[]
    for key,row in report['scenes'].items():
        payload=torch.load(row['parent']['path'],map_location='cpu',weights_only=False)
        paths=[Path(payload['base_field']),Path(payload['geometry_report'])]
        root=Path(payload['root'])
        paths += [root/name for name in ['teachers.json','responsibility.json','caption_region_evidence.pt']]
        report['input_bindings'][key]=[{ 'path':str(p), 'sha256':digest(p)} for p in paths]
        row['historical_comparators']={}
        for name,base,suffix in [('next38','output/taskwise_scope_composition_next38','pipeline_calibrated'),('next40','output/odds_kernel_extent_next40','pipeline')]:
            path=Path(base)/key/'anchored'/suffix/'evaluation.json'
            if path.exists():row['historical_comparators'][name]=dict(path=str(path),sha256=digest(path),metrics=json.loads(path.read_text())['metrics']['hierarchical_pipeline'])
        row['metric_semantics_same_candidate']=[{k:c[k] for k in ['ridge','legacy_visible_iou','known_iou','known_brier','positive_recall','negative_selection','unknown_selection','unknown_absolute_change','unknown_selection_change','valid_counts','support','episode_classes']} for c in row['source']['kernel_choices']]
        for arm,a in row['arms'].items():
            a['serialized_field_bytes']=Path(a['field']).stat().st_size
            a['shared_training_bytes']=(Path(report['run'])/key/'source_training.pt').stat().st_size
            m=a['benchmark']; ridge=a['selection']['selected']['ridge']
            metrics=({k:m[k] for k in ['iou','ap']}|{'selected':m['selected']['miou']} if key.startswith('lerf') else {k:m[k]['miou'] for k in ['19','15','10']})
            table.append(f"{key} / {arm}: ridge={ridge}; {json.dumps(metrics)}")
    report['correctness_evidence']=dict(direct_tests=9,all_commands_exit_zero=all(c.get('exit_code')==0 for c in report['commands']),all_residuals_identical=all(a['residual_identical'] for row in report['scenes'].values() for a in row['arms'].values()))
    report['source_files']={str(p):digest(p) for p in map(Path,['scripts/research/fit_source_kernel_extent.py','scripts/research/run_partial_scope_next41.py','radio_gs/training/partial_scope_metrics.py','radio_gs/querying/partial_extent_supervision.py','tests/test_partial_scope_metrics_next41.py'])}
    (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (Path(report['run'])/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (ROOT/'notes.md').open('a') as f:
        f.write('\nScene results (fractions, existing scorers):\n\n'+'\n\n'.join(table)+'\n')
        f.write('\nAdditional verification: `PYTHONPATH=. '+PYTHON+' -m pytest -q tests/test_partial_scope_metrics_next41.py` (9 passed).\n')
        f.write('\nThe prior task A implementation and its eight tests were already present at the start of this repair. This continuation changed only task A fit/runner/tests and output/partial_scope_next41; existing unrelated workspace changes were preserved.\n')

    (Path(report['run'])/'notes.md').write_text((ROOT/'notes.md').read_text())


if __name__=='__main__':
    main()
    enrich_report()
