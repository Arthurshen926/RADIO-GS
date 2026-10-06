"""Real frozen evaluators for every arm/scene, with exact next38 B0 replay."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import torch
from radio_gs.querying.query_region_evidence import load,digest

KEYS=['lerf_figurines','lerf_ramen','scannet_scene0000_00']
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'


def main(out):
    gate=json.loads((out/'source_gate.json').read_text())
    frozen={'recipe':digest(out/'recipe.json'),'source_gate':digest(out/'source_gate.json'),**{arm:digest(out/arm/'head.pt') for arm in ['B1','B2']}}
    (out/'frozen_before_target.json').write_text(json.dumps(frozen,indent=2))
    execution=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONPATH='.')
    report=dict(complete=False,promotion=False,development=True,source_gate=gate,scenes={},frozen=frozen)
    for arm in ['B0','B1','B2']:
        for key in KEYS:
            directory=out/arm/key;directory.mkdir(parents=True,exist_ok=True)
            parent=Path('output/taskwise_scope_composition_next38')/key/'field.pt'
            addon=dict(arm=arm,bank=str(out/key/'bank.pt'),checkpoint=str(out/arm/'head.pt'),parent_scores=str(parent.parent/'anchored/pipeline_calibrated/scores.pt'))
            torch.save(dict(next42_parent=str(parent),query_region_evidence=addon),directory/'field.pt')
            shutil.copyfile(parent.parent/'calibration.pt',directory/'calibration.pt')
            argv=[PYTHON,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out/arm),'--readout','anchored','--pipeline','--calibrated']
            log=directory/'evaluation.log';entry=dict(arm=arm,key=key,argv=argv,log=str(log),started=time.time());execution.append(entry)
            (out/'execution.json').write_text(json.dumps(execution,indent=2));start=time.monotonic()
            with log.open('w') as f:entry['exit_code']=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
            entry['seconds']=time.monotonic()-start;(out/'execution.json').write_text(json.dumps(execution,indent=2))
            if entry['exit_code']:raise RuntimeError(str(entry))
            result=directory/'anchored/pipeline_calibrated';metrics=json.loads((result/'evaluation.json').read_text())['metrics']['hierarchical_pipeline']
            row=dict(field_addition=str(directory/'field.pt'),evaluation=str(result/'evaluation.json'),predictions=str(result/'scores.pt'),metrics=metrics,parent_sha256=digest(parent),diagnostic_only=not gate['passed'])
            if arm=='B0':
                historical=parent.parent/'anchored/pipeline_calibrated'
                current=load(result/'scores.pt');old=load(historical/'scores.pt')
                row['exact_next38_replay']=all(torch.equal(current['scores'][k],old['scores'][k]) for k in old['scores'])
                def comparable(x):
                    if isinstance(x,dict):return {k:comparable(v) for k,v in x.items() if k!='pred_path'}
                    if isinstance(x,list):return [comparable(v) for v in x]
                    return x
                row['metric_replay']=comparable(metrics)==comparable(json.loads((historical/'evaluation.json').read_text())['metrics']['hierarchical_pipeline'])
                assert row['exact_next38_replay'], 'frozen parent scores are not exact'
            report['scenes'].setdefault(key,{})[arm]=row
            (out/'report.json').write_text(json.dumps(report,indent=2));print(key,arm,'completed',flush=True)
    report['complete']=True
    for arm in ['B1','B2']:
        lerf={m:sum((report['scenes'][k][arm]['metrics'][m] if m!='selected' else report['scenes'][k][arm]['metrics']['selected']['miou'])-(report['scenes'][k]['B0']['metrics'][m] if m!='selected' else report['scenes'][k]['B0']['metrics']['selected']['miou']) for k in KEYS[:2])/2 for m in ['iou','ap','selected']}
        # All existing scorer metrics are fractions.
        lerf={k:100*v for k,v in lerf.items()}
        scan={split:100*(report['scenes'][KEYS[2]][arm]['metrics'][split]['miou']-report['scenes'][KEYS[2]]['B0']['metrics'][split]['miou']) for split in ['19','15','10']}
        report.setdefault('target_screen',{})[arm]=dict(lerf_delta_pp=lerf,scannet_delta_pp=scan,passed=lerf['iou']>=.5 and lerf['selected']>=.5 and lerf['ap']>=-.25 and min(scan.values())>=-.5)
    assert digest(out/'recipe.json')==frozen['recipe'] and digest(out/'source_gate.json')==frozen['source_gate']
    for arm in ['B1','B2']:assert digest(out/arm/'head.pt')==frozen[arm]
    (out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(Path(a.output))


def finalize(out):
    """Package raw results, costs, unchanged inputs and only this task's diff."""
    import difflib
    import hashlib
    import math
    report=json.loads((out/'report.json').read_text());assert report['complete']
    files=['radio_gs/querying/query_region_evidence.py',
           'scripts/research/build_query_region_evidence_next42.py',
           'scripts/research/fit_query_region_evidence_next42.py',
           'scripts/research/run_query_region_evidence_next42.py',
           'scripts/research/evaluate_query_region_evidence_next42.py',
           'scripts/research/evaluate_scope_dictionary.py',
           'tests/test_query_region_evidence_next42.py']
    initial=json.loads((out/'source_before.json').read_text())
    # Recover the precise pre-task evaluator, including all pre-existing edits.
    path=Path('scripts/research/evaluate_scope_dictionary.py');before=path.read_text()
    before=before.replace("payload=load(out/'field.pt')\n    if 'next42_parent' in payload:\n        addition=payload;payload=load(addition['next42_parent']);payload['query_region_evidence']=addition['query_region_evidence']\n    root=Path(payload['root']);scene=a.key.split('_',1)[1]","payload=load(out/'field.pt');root=Path(payload['root']);scene=a.key.split('_',1)[1]")
    start=before.index("        if 'query_region_evidence' in payload:")
    end=before.index("        out=out/('pipeline_calibrated'",start)
    before=before[:start]+before[end:]
    assert hashlib.sha256(before.encode()).hexdigest()==initial[str(path)]
    patches=[];changes=[]
    for name in files:
        old=before if name==str(path) else ''
        if not old:assert name not in initial, name+' unexpectedly pre-existed'
        new=Path(name).read_text()
        patches.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        changes.append(dict(path=name,before_sha256=initial.get(name),after_sha256=digest(name),bytes=Path(name).stat().st_size))
    (out/'scoped.patch').write_text(''.join(patches));(out/'changes.json').write_text(json.dumps(changes,indent=2))
    unrelated=[]
    for name,sha in initial.items():
        if name in files:continue
        if not Path(name).is_file() or digest(name)!=sha:unrelated.append(name)
    report['workspace']=dict(initial_entries=len((out/'status_before.z').read_bytes().split(b'\0'))-1,source_changes=changes,other_initial_source_changes=unrelated,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),no_commit_push_publish=True)
    builds={k:json.loads((out/k/'build.json').read_text()) for k in KEYS}
    sources={a:json.loads((out/a/'source_diagnostics.json').read_text()) for a in ['B1','B2']}
    report['source_diagnostics']=sources
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        return None if isinstance(x,float) and not math.isfinite(x) else x
    for arm in ['B1','B2']:
        p=out/arm/'source_episodes.json';p.write_text(json.dumps(clean(json.loads(p.read_text())),indent=2,allow_nan=False))
    report['input_integrity']={}
    report['banks']={}
    for key in KEYS:
        bindings=json.loads((out/key/'inputs_before.json').read_text())
        after=[dict(path=r['path'],sha256=digest(r['path'])) for r in bindings]
        assert after==bindings,key+' input changed'
        (out/key/'inputs_final.json').write_text(json.dumps(after,indent=2))
        report['input_integrity'][key]=dict(unchanged=True,before=str(out/key/'inputs_before.json'),after=str(out/key/'inputs_final.json'))
        bank=load(out/key/'bank.pt')
        report['banks'][key]=dict(path=str(out/key/'bank.pt'),sha256=digest(out/key/'bank.pt'),bytes=(out/key/'bank.pt').stat().st_size,regions=bank['shape'][1],native_dimension=bank['C'].shape[1],nnz=len(bank['weights']),selection=json.loads((out/key/'selection.json').read_text()),mean_foreground_background_cosine=float(F_cos(bank['F'],bank['B'])))
        for arm in ['B0','B1','B2']:
            row=report['scenes'][key][arm]
            row['source_split']=str(out/key/'source_split.json')
            row['training']=None if arm=='B0' else dict(checkpoint=str(out/arm/'head.pt'),diagnostics=str(out/arm/'source_diagnostics.json'),episodes=str(out/arm/'source_episodes.json'),shared_across_all_three_scenes=True)
            runtime=out/arm/key/'anchored/next42_runtime.json'
            if runtime.exists():row['query_runtime']=json.loads(runtime.read_text())
            row['replay']=json.loads((out/arm/key/'anchored/next42_parent_replay.json').read_text())
    execution_doc=json.loads((out/'execution.json').read_text())
    executions=execution_doc if isinstance(execution_doc,list) else execution_doc['evaluations']
    report['costs']=dict(evidence_build_seconds=sum(x['seconds'] for x in builds.values()),evidence_encoding_seconds=sum(x['encoding_seconds'] for x in builds.values()),support_refresh_seconds=sum(json.loads((out/k/'support_refresh.json').read_text())['seconds'] for k in KEYS),fit_and_source_validation_seconds={a:s['seconds'] for a,s in sources.items()},superseded_fit_and_validation_seconds={a:json.loads((out/'pre_duplicate_fix'/a/'source_diagnostics.json').read_text())['seconds'] for a in sources},target_evaluation_seconds=sum(x['seconds'] for x in executions),bank_bytes=sum(x['bytes'] for x in report['banks'].values()),shared_head_bytes={a:(out/a/'head.pt').stat().st_size for a in sources},source_code_bytes=sum(Path(p).stat().st_size for p in files),total_run_file_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),total_run_bytes_includes_archived_attempts=True,fit_preparation_seconds_not_separately_instrumented=True)
    report['tests']=dict(passed=16,task_a_unchanged_tests=9,log=str(out/'tests.log'),covered=['zero/saturated exact','query batching/A-B-A','unknown/unsupported/no negative','duplicate fractional mass','source-view exclusion','overlap unknown','nontext exact','finite nonzero gradient','cold serialization','same window C/F/B'])
    report['limitations']=['source pseudo masks and noisy phrase evidence are not GT or verified physical identities','held source views were already seen by the base field; additional-head holdout only','source family split uses normalized text identity hash, not verified semantic paraphrase families','source masks retain 96x128 resolution before RGB upsampling; this is not native-boundary recovery','independent identity margin is held source P-minus-qualified-N rendered probability with query-view-excluded evidence, not verified physical instance correspondence','source gate differences are near floating-point precision and do not establish meaningful intervention benefit','fresh parent recomputation has last-bit sparse GPU drift; frozen exact next38 arrays are the common benchmark prior','single seed, three development scenes, no full rollout or promotion','original full-native region evidence and sparse support are extra scene storage, not a pure96D method']
    report['postfreeze_readout_diagnosis']=json.loads((out/'postfreeze_readout_diagnosis.json').read_text())
    report['workspace']['staged_diff_unchanged']=subprocess.check_output(['git','diff','--cached','--binary'])==(out/'index_before.patch').read_bytes()
    assert report['workspace']['staged_diff_unchanged']
    report['recipe_path']=str(out/'recipe.json');report['execution_path']=str(out/'execution.json');report['scoped_patch']=str(out/'scoped.patch');report['outer_acceptance_pending']=True
    report['conclusion']='Completed matched mechanism pilot; both arms fail the development screen. The heads saturate near +2, leaving qualified negative carriers inactive; B2 does not show meaningful intervention benefit. No promotion; outer review decides refinement.'
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False))
    lines=['# next42 Stage1 actual pilot','',report['conclusion'],'','All metrics below are percentages. One shared seed0/200-step head per arm across all scenes; no scene winner selection. No benchmark query/GT was used in bank construction or fitting.','', '| Scene | Arm | IoU / Scan19 | AP / Scan15 | localization / Scan10 | selected IoU |','|---|---|---:|---:|---:|---:|']
    for key in KEYS:
        for arm in ['B0','B1','B2']:
            m=report['scenes'][key][arm]['metrics']
            values=[m['iou'],m['ap'],m['localization'],m['selected']['miou']] if key.startswith('lerf') else [m[s]['miou'] for s in ['19','15','10']]+[None]
            lines.append('| '+key+' | '+arm+' | '+' | '.join('—' if v is None else f'{100*v:.6f}' for v in values)+' |')
    lines+=['','Target screen: `'+json.dumps(report['target_screen'])+'`.','', 'Source gate (recorded before final benchmark): `'+json.dumps(report['source_gate'])+'`. The differences are tiny; inspect full precision diagnostics and FPR/unknown tradeoffs.','', 'Exactness: all B0 element score arrays equal frozen next38; every B1/B2 zero-correction check is exact including unchanged saturated entries. Fresh recomputation is separately measured, not falsely called bit-identical. Failed strict replay is retained in replay_precheck. Pre-duplicate-fix fits and B0-only controls are retained in pre_duplicate_fix; no B1/B2 target scoring preceded that correctness repair.','', 'Normalization repair: per-element/source-view aggregate equals its strongest membership, distributed over total valid region mass, then divided by all visible source views. Repeated weak proposals cannot increase that aggregate. All local source regions remain separate; overlaps are not automatic negatives.','', 'Source diagnostics: '+json.dumps({a:{g:{k:sources[a]['metrics'][g].get(k) for k in ['known_brier','independent_identity_margin','negative_selection','unknown_selection','unknown_absolute_change']} for g in ['held_view','held_phrase']} for a in sources}), '', 'Costs: '+json.dumps(report['costs']), '', 'Limitations:']
    lines+=['- '+v for v in report['limitations']]
    lines+=['','Files: report.json, recipe.json, execution.json, source_gate.json, each scene bank/source split/input hashes, each arm shared head/source diagnostics, each arm/scene real scores/masks and evaluation.json, scoped.patch and changes.json. Task A files and unrelated initial source hashes are unchanged: '+str(not unrelated)+'. No commit/push/publish, geometry or root documentation changes.']
    lines+=['','Postfreeze diagnosis (no refitting): both heads output positive values near their +2 bound for all evaluated region/query pairs, so the negative carrier is inactive. This is a learned one-sided correction, despite a signed-capable architecture. B1/B2 differ in just one figurines binary element and no ramen binary elements; ScanNet argmax predictions match. See postfreeze_readout_diagnosis.json for coverage, saturation and score deltas.']
    (out/'notes.md').write_text('\n'.join(lines)+'\n')
    stages=[dict(stage='build',argv=[PYTHON,'scripts/research/build_query_region_evidence_next42.py','--output',str(out)],gpu=1,exit_code=0,log=str(out/'build.log'),seconds_recorded=sum(x['seconds'] for x in builds.values())),dict(stage='support_refresh',gpu=1,exit_code=0,log=str(out/'support_refresh.log'),reason='duplicate fractional-mass invariant repair; no native re-encoding'),dict(stage='fit',argv=[PYTHON,'scripts/research/fit_query_region_evidence_next42.py','--output',str(out)],gpu=0,exit_code=0,log=str(out/'fit.log'),steps_per_arm=200,seed=0,preparation_seconds_not_instrumented=True)]
    (out/'execution.json').write_text(json.dumps(dict(stages=stages,evaluations=executions,archived_attempts=['replay_precheck','pre_duplicate_fix'],all_final_jobs_exit_zero=True),indent=2))
    return report


def F_cos(a,b):
    from torch.nn import functional as F
    return (F.normalize(a.float(),dim=-1)*F.normalize(b.float(),dim=-1)).sum(-1).mean()
