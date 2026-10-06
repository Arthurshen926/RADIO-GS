"""Execute the fixed Stage2 fit/evaluation recipe in its prepared isolated run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--evaluation-only',action='store_true');a=p.parse_args();out=Path(a.output)
    assert (out/'recipe.json').is_file()
    stages=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONPATH='.')
    for stage in (['evaluate'] if a.evaluation_only else ['fit','evaluate']):
        if stage=='fit' and (out/'B1/head.pt').exists():raise FileExistsError('Preserve previous fits: prepare a new run')
        if stage=='evaluate' and (out/'execution.json').exists():raise FileExistsError('Preserve previous evaluations: use a new run')
        argv=[PYTHON,f'scripts/research/{stage}_visual_scope_membership_next42.py','--output',str(out)]
        start=time.monotonic()
        with (out/(stage+'.log')).open('w') as f:rc=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        stages.append(dict(stage=stage,argv=argv,seconds=time.monotonic()-start,exit_code=rc));(out/'stages.json').write_text(json.dumps(stages,indent=2))
        if rc:raise RuntimeError(stage+' failed; inspect log')
if __name__=='__main__':main()


def finalize(out):
    """Seal unchanged inputs, scoped edits, raw six-job outputs and honest gates."""
    import difflib
    import re
    from radio_gs.querying.query_region_evidence import digest,load
    out=Path(out);recipe=json.loads((out/'recipe.json').read_text());stage1=Path(recipe['stage1'])
    report=json.loads((out/'report.json').read_text());assert report['complete']
    files=['radio_gs/querying/visual_scope_membership.py','radio_gs/querying/query_region_evidence.py',
           'scripts/research/fit_visual_scope_membership_next42.py','scripts/research/evaluate_visual_scope_membership_next42.py',
           'scripts/research/run_visual_scope_membership_next42.py','tests/test_visual_scope_membership_next42.py']
    initial=json.loads((out/'source_before.json').read_text());changes=[];patch=[]
    for name in files:
        previous=out/'before'/name;old=previous.read_text() if previous.exists() else ''
        assert (name not in initial) if not old else digest(previous)==initial[name]
        new=Path(name).read_text();patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        changes.append(dict(path=name,before_sha256=initial.get(name),after_sha256=digest(name),before_bytes=len(old.encode()),after_bytes=len(new.encode())))
    (out/'scoped.patch').write_text(''.join(patch));(out/'changes.json').write_text(json.dumps(changes,indent=2))
    other=[p for p,h in initial.items() if p not in files and (not Path(p).is_file() or digest(p)!=h)]
    index_unchanged=subprocess.check_output(['git','diff','--cached','--binary'])==(out/'index_before.patch').read_bytes()
    old_outputs=json.loads((out/'stage1_before.json').read_text());altered=[p for p,h in old_outputs.items() if not Path(p).is_file() or digest(p)!=h]
    assert not other and not altered and index_unchanged
    report['workspace']=dict(changes=changes,unrelated_initial_files_changed=other,staged_diff_unchanged=index_unchanged,stage1_files_verified=len(old_outputs),stage1_changed=altered,no_commit_push_publish=True,scorer_unchanged=digest('scripts/research/evaluate_scope_dictionary.py')==initial['scripts/research/evaluate_scope_dictionary.py'])
    report['input_integrity']={};report['banks']={};source={};report['coverage']={}
    for key in recipe['scenes']:
        bindings=json.loads((stage1/key/'inputs_before.json').read_text());after=[dict(path=b['path'],sha256=digest(b['path'])) for b in bindings];assert after==bindings
        (out/key/'inputs_final.json').write_text(json.dumps(after,indent=2))
        report['input_integrity'][key]=dict(unchanged=True,before=str(stage1/key/'inputs_before.json'),after=str(out/key/'inputs_final.json'),source_split_matches_stage1=(out/key/'source_split.json').read_bytes()==(stage1/key/'source_split.json').read_bytes())
        assert report['input_integrity'][key]['source_split_matches_stage1']
        bank=load(out/key/'bank.pt');report['banks'][key]=dict(path=str(out/key/'bank.pt'),reused_from=str(stage1/key/'bank.pt'),sha256=digest(out/key/'bank.pt'),bytes=(out/key/'bank.pt').stat().st_size,regions=bank['shape'][1],nnz=len(bank['weights']),evidence_reencoded=False)
        rows=json.loads((out/key/'coverage.json').read_text());report['coverage'][key]={}
        for group in ['train','held_view','held_phrase']:
            selected=[r for r in rows if r['group']==group];summary={n:sum(r[n] for r in selected) for n in ['valid','positive','negative','weak_area','weak_known']};summary.update(query_views=len(selected),single_class=sum(r['classes']==1 for r in selected),unsupported=sum(r['classes']==0 for r in selected),total_pairs=len(selected)*bank['shape'][1],discarded_pairs=len(selected)*bank['shape'][1]-summary['valid'])
            report['coverage'][key][group]=summary
        for arm in ['B1','B2']:
            row=report['scenes'][key][arm];directory=out/arm/key/'anchored'
            row.update(evaluation_sha256=digest(row['evaluation']),predictions_sha256=digest(row['predictions']),source_split=str(out/key/'source_split.json'),training=str(out/arm/'source_diagnostics.json'),head=str(out/arm/'head.pt'),query_runtime=json.loads((directory/'next42_runtime.json').read_text()),replay=json.loads((directory/'next42_parent_replay.json').read_text()))
    for arm in ['B1','B2']:source[arm]=json.loads((out/arm/'source_diagnostics.json').read_text())
    report['source_diagnostics']=source
    assert [(x['scene'],x['view'],x['queries']) for x in source['B1']['trace']]==[(x['scene'],x['view'],x['queries']) for x in source['B2']['trace']]
    report['matched_training']=dict(shared_scene_head=True,steps_per_arm=200,seed=0,optimizer='Adam .01',parameter_count=sum(v.numel() for k,v in load(out/'B1/head.pt')['state_dict'].items() if k.startswith('net.')),same_data_support_split=True,recorded_training_episode_sequence_matches=True,standardization_pairs_equal=source['B1']['fit_standardization_pairs']==source['B2']['fit_standardization_pairs'])
    executions=json.loads((out/'execution.json').read_text());assert len(executions)==6 and all(e['exit_code']==0 for e in executions)
    tests=(out/'tests.log').read_text();passed=int(re.search(r'(\d+) passed',tests).group(1));assert passed==23
    report['tests']=dict(passed=passed,new=7,stage1_preserved=7,taskA_preserved=9,log=str(out/'tests.log'))
    report['costs']=dict(evidence_reused=True,evidence_encoding_seconds=0,evidence_reused_bytes=sum(b['bytes'] for b in report['banks'].values()),head_bytes={a:(out/a/'head.pt').stat().st_size for a in ['B1','B2']},fit_and_source_diagnostic_seconds={a:s['seconds'] for a,s in source.items()},preparation_seconds=source['B1']['preparation_seconds'],mean_query_seconds={a:s['mean_query_seconds'] for a,s in source.items()},target_evaluation_seconds=sum(e['seconds'] for e in executions),source_net_added_bytes=sum(c['after_bytes']-c['before_bytes'] for c in changes),archived_failed_attempt=dict(path=str(out/'pre_sparse_precision_fix'),extra_B1_steps=200,extra_B2_steps=0,reason='postfit exact cold assertion exposed CUDA float32 sparse atomic drift; both final arms rerun using float64 accumulation; no target scoring preceded repair'))
    report['paths']=dict(recipe=str(out/'recipe.json'),execution=str(out/'execution.json'),source_gate=str(out/'source_gate.json'),changes=str(out/'changes.json'),patch=str(out/'scoped.patch'),notes=str(out/'notes.md'))
    report['limitations']=['Source projected P/N/U membership is geometry-conditioned pseudo-supervision, not human GT or physical whole-object evidence.',
      'Held views were seen by the frozen base map; this is held-out supervision only for the added module.',
      'Soft target class balancing uses y>=0.5 versus y<0.5 per query-view; single-class and unsupported counts remain explicit.',
      'Effective area is peak-normalized R M on visible pixels; this intentionally rescales transport mass only for coverage measurement, not for deployed transport.',
      'B2 region Brier improves meaningfully, but rendered known Brier fails the fixed 1e-4 source gate.',
      'B2 still admits zero source-untagged regions in these source episodes; removing the architectural shortcut does not prove new untagged recall.',
      'Qualified negatives can still have zero transport when the parent already assigns lower probability; direct classification gradients remain active independently.',
      'Matched benchmark prior reuses exact next38 scores to remove float32 GPU drift; separate source cold runtime recomputes the parent without benchmark assets.',
      'Sparse transport accumulates in float64 then converts once to float32 for exact repeated/cold label-invariance checks; batch tolerance 2e-6.',
      'Stage1 known_loss aggregation is superseded only in this independent implementation; old Stage1 numbers remain unchanged.',
      'Single seed and three development scenes; no full benchmark rollout, default promotion, or SOTA claim.']
    report['structural_findings']=dict(label_admission_shortcut_removed=True,bank_labels_erased_runtime_exact=True,negative_classifier_dead_gradient_removed=True,weak_known_containing_positive_excluded=True,B2_unlabelled_admissions=sum(g['unlabelled_positive'] for groups in source['B2']['activation'].values() for g in groups.values()),B2_positive_active=sum(g['positive_active'] for groups in source['B2']['activation'].values() for g in groups.values()),B2_negative_active=sum(g['negative_active'] for groups in source['B2']['activation'].values() for g in groups.values()),B2_saturated=sum(g['saturated'] for groups in source['B2']['activation'].values() for g in groups.values()))
    report['promotion']=False;report['outer_acceptance_pending']=True
    report['conclusion']='Stage2 implemented and six real diagnostics completed. Visual region discrimination improves but source rendered-risk gate and target screen fail; no promotion or tuning.'
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# next42 Stage2 visual membership','',report['conclusion'],'','B0 is reused from hash-bound Stage1 real evaluations. B1/B2 are six new evaluator executions. All values in the table are percentages.','', '| Scene | Arm | IoU / Scan19 | AP / Scan15 | localization / Scan10 | selected IoU |','|---|---|---:|---:|---:|---:|']
    for key in recipe['scenes']:
        for arm in ['B0','B1','B2']:
            m=report['scenes'][key][arm]['metrics'];values=[m['iou'],m['ap'],m['localization'],m['selected']['miou']] if key.startswith('lerf') else [m[s]['miou'] for s in ['19','15','10']]+[None]
            lines.append('| '+key+' | '+arm+' | '+' | '.join('—' if v is None else f'{100*v:.6f}' for v in values)+' |')
    lines+=['','Target deltas, percentage points: `'+json.dumps(report['target_screen'])+'`.','','Source gate (frozen before target scoring): `'+json.dumps(report['source_gate'])+'`.','','Actual changes: no source text/association/scene/parent identity input; direct soft membership BCE provides negative gradients even when every logit starts positive; weak-known containing regions are excluded. Source B2 negative corrections are active in figurines and ScanNet, but none in ramen; B2 has zero untagged-region admissions. Do not equate structural repair with complete identity/recall recovery.','','Cold checks: fresh parent_model source queries, no benchmark scores; exact serialized and label-erased outputs on all three scenes. Batch maximum error <=9.54e-7 for both arms. No exactness claim for independently recomputed benchmark float32 parent; measured drift is in each scene replay receipt.','','Coverage: '+json.dumps(report['coverage']),'','Costs: '+json.dumps(report['costs']),'','Limits:']
    lines+=['- '+v for v in report['limitations']]
    lines+=['','23 tests pass (7 new, 7 Stage1, 9 TaskA). All Stage1 files and unrelated initial source files hash unchanged; index unchanged; no commit/push/publish. Six evaluation.json/scores.pt paths and hashes are listed in report.json. Recipe, source splits/coverage, each-arm heads/training diagnostics, execution.json and scoped.patch accompany this report.']
    (out/'notes.md').write_text('\n'.join(lines)+'\n')
    doc=dict(stages=[dict(stage='fit',exit_code=0,gpu=0,steps_per_arm=200,log=str(out/'fit.log'),preparation_seconds=source['B1']['preparation_seconds'],fit_and_diagnostic_seconds={a:s['seconds'] for a,s in source.items()})],evaluations=executions,reused_B0={k:report['scenes'][k]['B0'] for k in recipe['scenes']},all_six_evaluations_exit_zero=True,archived_failed_attempts=['pre_sparse_precision_fix'])
    (out/'execution.json').write_text(json.dumps(doc,indent=2))
    report['costs']['run_added_logical_bytes_excluding_reused_bank_symlinks']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
    report['costs']['total_added_bytes']=report['costs']['run_added_logical_bytes_excluding_reused_bank_symlinks']+report['costs']['source_net_added_bytes']
    report['costs']['byte_accounting']='Logical newly written files and net code bytes, including archived failed attempt and snapshots; original bank bytes excluded as reused symlinks. Final report size affects count only within its own metadata serialization.'
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(target_screen=report['target_screen'],source_gate=report['source_gate'],costs=report['costs']),indent=2))
    return report
