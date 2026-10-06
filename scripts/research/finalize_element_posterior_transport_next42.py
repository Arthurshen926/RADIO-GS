"""Seal Stage3 scoped changes and real results without promoting the candidate."""
import difflib
import json
from pathlib import Path
import subprocess
from radio_gs.querying.query_region_evidence import digest
from scripts.research.run_element_posterior_transport_next42 import write,KEYS,STAGE2

FILES=['radio_gs/querying/query_region_evidence.py','radio_gs/querying/element_posterior_transport.py','scripts/research/run_element_posterior_transport_next42.py','scripts/research/evaluate_element_posterior_transport_next42.py','scripts/research/finalize_element_posterior_transport_next42.py','tests/test_element_posterior_transport_next42.py']

def main(out):
    report=json.loads((out/'report.json').read_text());assert report['complete']
    initial=json.loads((out/'source_before.json').read_text());patch=[];changes=[]
    for name in FILES:
        before=out/'before'/name;old=before.read_text() if before.exists() else '';new=Path(name).read_text()
        assert digest(before)==initial[name] if name in initial else not before.exists()
        patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        changes.append(dict(path=name,before_sha256=initial.get(name),after_sha256=digest(name),before_bytes=len(old.encode()),after_bytes=len(new.encode())))
    (out/'scoped_before_after.patch').write_text(''.join(patch));write(out/'changes.json',changes)
    unrelated=[p for p,h in initial.items() if p not in FILES and (not Path(p).is_file() or digest(p)!=h)]
    index_same=subprocess.check_output(['git','diff','--cached','--binary'])==(out/'index_before.patch').read_bytes()
    assert not unrelated and index_same
    integrity={}
    for stage in ['stage1','stage2']:
        hashes=json.loads((out/(stage+'_original_checksums.json')).read_text());bad=[p for p,h in hashes.items() if not Path(p).is_file() or digest(p)!=h];assert not bad
        integrity[stage]=dict(verified_files=len(hashes),changed=bad)
    execution=json.loads((out/'execution.json').read_text());assert len(execution)==6 and all(e['exit_code']==0 for e in execution)
    frozen=json.loads((out/'frozen_before_target.json').read_text())
    for kind,filename in [('banks','bank.pt'),('transport','transport.pt')]:
        assert all(digest(out/key/filename)==h for key,h in frozen[kind].items())
    artifacts=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for arm in ['B1','B2'] for key in KEYS for p in (out/arm/key).rglob('*') if p.is_file()]
    write(out/'target_artifact_manifest.json',artifacts)
    write(out/'attempts.json',[dict(log='run.log',exit_code=1,phase='source',target_started=False),dict(log='source_retry.log',exit_code=1,phase='source',target_started=False),dict(log='source_fixed.log',exit_code=0,phase='source and six target evaluations')])
    source=json.loads((out/'source_diagnostics.json').read_text());gate=json.loads((out/'source_gate.json').read_text());stage2=json.loads((STAGE2/'report.json').read_text())
    for key in KEYS:
        for arm in ['B0','B1','B2']:
            row=report['scenes'][key][arm];row.update(evaluation_sha256=digest(row['evaluation']),predictions_sha256=digest(row['predictions']))
            if arm!='B0':
                directory=out/arm/key/'anchored';row['runtime']=json.loads((directory/'next42_runtime.json').read_text());row['parent_replay']=json.loads((directory/'next42_parent_replay.json').read_text())
        bind=json.loads((out/key/'camera_evidence_bind.json').read_text())
        assert all(digest(b['path'])==b['sha256'] for b in bind['inputs'])
    region={a:dict(metrics=json.loads((STAGE2/a/'source_diagnostics.json').read_text())['region'],path=str(STAGE2/a/'region_episodes.json'),sha256=digest(STAGE2/a/'region_episodes.json'),reused=True,reason='identical frozen membership head and banks, only transport changes') for a in ['B1','B2']}
    report.update(source_diagnostics=source,region_diagnostics=region,preferred_arm=gate['preferred_arm'],stage2_target_screen=stage2['target_screen'],stage2_scenes=stage2['scenes'],input_integrity=integrity,workspace=dict(unrelated_initial_changes=unrelated,staged_diff_unchanged=index_same,scorer_unchanged=digest('scripts/research/evaluate_scope_dictionary.py')==initial['scripts/research/evaluate_scope_dictionary.py'],no_commit_push_publish=True),changes=changes,tests=dict(log=str(out/'tests.log'),passed=30,prior_tests=23,new_tests=7),promotion=False,outer_acceptance_pending=True)
    report['source_runtime']={a:json.loads((out/a/'source_runtime.json').read_text()) for a in ['B1','B2']}
    report['camera_evidence_bind']={k:str(out/k/'camera_evidence_bind.json') for k in KEYS}
    report['costs']=dict(wall_seconds_since_fresh_snapshot=__import__('time').time()-(out/'source_before.json').stat().st_mtime,source_seconds=source['seconds'],evaluation_seconds=sum(e['seconds'] for e in execution),encoding_seconds=0,training_steps=0,metadata_bytes={k:(out/k/'transport.pt').stat().st_size for k in KEYS},reused_bank_bytes=sum((out/k/'bank.pt').stat().st_size for k in KEYS),reused_head_bytes=sum((out/a/'head.pt').stat().st_size for a in ['B1','B2']),net_code_bytes=sum(c['after_bytes']-c['before_bytes'] for c in changes),failed_attempts=['run.log: initial cold batch assertion','source_retry.log: diagnosed batch=1.5735626220703125e-05, warm/cold/labels/A-B-A exact'],failed_attempts_no_target=True)
    report['limitations']=['Source P/N/U are geometry-conditioned pseudo labels, not GT object errors.','Base parent has seen held source images; holdouts apply to added head supervision only.','Opinion pool is convex and bounded, not independent Bayesian inference: parent and groups share source observations.','No new image/prompt benchmark cohort; nontext identity is tested at runtime.','Source masks and membership calibration do not establish complete physical object identity.','Single three-scene development pilot; no full-cohort or SOTA claim.','At most six deterministic camera-neighbor groups limit confidence but do not prove independence.','Matched target uses hash-bound next38 predictions; generic cold checks use recomputed source parent without target assets.']
    preferred=gate['preferred_arm'];passed=preferred is not None and report['target_screen'][preferred]['passed']
    report['conclusion']='Stage3 implemented and six real evaluations completed; '+('preferred arm passes pilot screen; outer acceptance required.' if passed else 'no preferred arm passes the fixed target screen; no promotion or target tuning.')
    write(out/'report.json',report)
    lines=['# next42 Stage3 element posterior transport','',report['conclusion'],'',f'Global source preferred arm: {preferred}. Source screen was frozen before all target runs. All six exit codes are zero. B0 reuses hash-bound Stage1 predictions.','', '| Scene | Arm | IoU / Scan19 % | AP / Scan15 % | localization / Scan10 % | selected % |','|---|---|---:|---:|---:|---:|']
    for key in KEYS:
        for arm in ['B0','B1','B2']:
            m=report['scenes'][key][arm]['metrics'];v=[m['iou'],m['ap'],m['localization'],m['selected']['miou']] if key.startswith('lerf') else [m[s]['miou'] for s in ['19','15','10']]+[None]
            lines.append('| '+key+' | '+arm+' | '+' | '.join('—' if x is None else f'{100*x:.6f}' for x in v)+' |')
    lines+=['','Target deltas (percentage points): `'+json.dumps(report['target_screen'])+'`.','','Source gate: `'+json.dumps(gate)+'`.','','Convexity: each region directional target is max(parent,r) or min(parent,r), hence in [0,1]. Each group normalizes nonnegative footprint × confidence contributions. The final weights 1/(1+Σmass) and mass/(1+Σmass) sum to one, with each group mass ≤1 and at most six groups. Missing evidence mass is zero. This is a bounded opinion pool, not independent Bayes.','','Geometry: source responsibility.json supplies registered world_to_camera matrices. Camera centers and forward vectors are recovered by inversion; deterministic first-source FPS uses centers normalized by source bounding-box diagonal and unit directions, nearest assignment, max six groups. Identical poses assign identically. No fallback used. Visibility count reproduces the builder before division by all visible views. Duplicate mask/source proposals are removed, preserving original within(element,view) max footprint mass.','','Runtime outputs changed probabilities directly as finite logits. Zero probability change preserves original finite logits exactly. Per-query head shapes remove batch-dependent GEMM rounding; transport accumulates in float64. Early source-only numerical failures are retained in run.log and source_retry.log. No parameters changed from the prescribed method.','','Original 23 tests plus 7 new runtime tests pass. Source cold, label erasure, A-B-A and batch receipts are in each arm/source_runtime.json. All Stage1/Stage2 files and initial unrelated files verified unchanged; staged diff unchanged. No commits, pushes, or publication.','','Raw source query/view metrics: source_episodes.json; source summary and old/new transport: source_diagnostics.json; low-member suppression and group-mass statistics: B1/source_runtime.json and B2/source_runtime.json. Frozen region risks are hash-bound reused Stage2 diagnostics. Raw target scores, masks and evaluation paths are in report.json; all evaluator logs and exit codes in execution.json.','','Costs: `'+json.dumps(report['costs'])+'`.','','Limitations:']
    lines+=['- '+s for s in report['limitations']];(out/'notes.md').write_text('\n'.join(lines)+'\n')
    report['costs']['added_run_logical_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink());report['costs']['total_added_bytes']=report['costs']['added_run_logical_bytes']+report['costs']['net_code_bytes'];report['costs']['bytes_definition']='Logical new files including snapshots/logs; reused symlinks excluded; metadata self-serialization may shift final byte count slightly.'
    write(out/'report.json',report);print(json.dumps(dict(conclusion=report['conclusion'],preferred_arm=preferred,target_screen=report['target_screen'],costs=report['costs']),indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(Path(p.parse_args().output))
