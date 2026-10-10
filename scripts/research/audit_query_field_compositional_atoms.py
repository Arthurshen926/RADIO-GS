"""All-scene post-score audit of composition intervention and input bindings."""
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_compositional_atoms import ROOT,REFERENCE,ARMS,read,write,receipt,check,design
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();barrier=read(ROOT/'predictions_complete.json');scored=read(ROOT/'scoring_complete.json')
    for r in barrier['records']+scored['records']:check(r)
    cfg=read(d['configuration']['path']);per=read(ROOT/'summary.json')['per_scene'];details={};labels={}
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json');check(r['compilation']);c=read(r['compilation']['path']);check(c['artifact'])
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));check(r['teacher_selection'])
        data=torch.load(p/'predictions.pt',weights_only=True)
        before=torch.load(REFERENCE/'predictions'/j['name']/'predictions.pt',weights_only=True)
        for arm in ARMS[:3]:
            assert torch.equal(data['scores'][arm],before['scores'][arm])
            assert torch.equal(data['selected_by_control'][arm],before['selected_by_control'][arm])
        assert r['source_assisted_diagnostic'] and not r['target_labels_opened'] and not r['integrated_deployed_field']
        assert c['query_independent'] and not c['teacher_acceptance_read'] and not c['target_labels_opened']
        for v in c['source_records']:assert max(v['single_region_max_errors'].values())<=c['single_region_tolerance']
        delta=data['scores']['atom_protected']-data['scores']['atom_parts'];assert bool((delta>=0).all())
        for q in r['diagnostics']:
            for k in ['positive_mass','negative_mass_before','admitted_regions','confidence_sum']:
                assert q['controls']['atom_parts'][k]==q['controls']['atom_protected'][k]
        e=read(p/'evaluation/evaluation.json');old=read(REFERENCE/'predictions'/j['name']/'evaluation/evaluation.json')
        assert e['evaluation_scene_sha256']==old['evaluation_scene_sha256']
        for arm in ARMS:
            m=e['metrics'][arm];precision=[];recall=[]
            for x in m['selected']['objects']:
                assert not x['missing']
                target=load_binary_mask(Path(x['gt_path']),threshold=10);pred=load_binary_mask(Path(x['pred_path']),threshold=10,grayscale=True)
                tp=int((target&pred).sum());assert abs(tp/max(int((target|pred).sum()),1)-x['iou'])<1e-12
                precision.append(tp/max(int(pred.sum()),1));recall.append(tp/max(int(target.sum()),1))
                if x['gt_path'] not in labels:labels[x['gt_path']]=receipt(x['gt_path'])
            per[arm][j['scene']].update(precision=float(np.mean(precision)),recall=float(np.mean(recall)))
        changes=dict(true_positive_added=0,false_positive_added=0,true_positive_removed=0,false_positive_removed=0)
        for x,y in zip(e['metrics']['teacher_union_protected']['selected']['objects'],e['metrics']['atom_protected']['selected']['objects']):
            assert (x['frame'],x['query'],x['gt_path'])==(y['frame'],y['query'],y['gt_path'])
            target=load_binary_mask(Path(x['gt_path']),threshold=10)
            left=load_binary_mask(Path(x['pred_path']),threshold=10,grayscale=True)
            right=load_binary_mask(Path(y['pred_path']),threshold=10,grayscale=True)
            changes['true_positive_added']+=int((~left&right&target).sum())
            changes['false_positive_added']+=int((~left&right&~target).sum())
            changes['true_positive_removed']+=int((left&~right&target).sum())
            changes['false_positive_removed']+=int((left&~right&~target).sum())
        details[j['scene']]=dict(queries=len(data['texts']),atom_state_bytes=c['tensor_bytes'],atoms=c['atom_count'],
            regions=c['regions'],single_region_max_error=max(v for row in c['source_records'] for v in row['single_region_max_errors'].values()),
            monotonicity_exact=True,extra_state_integrated_into_field=False,
            added_member_query_pairs_vs_marginal_union=int((data['selected_by_control']['atom_protected']&~data['selected_by_control']['teacher_union_protected']).sum()),
            removed_member_query_pairs_vs_marginal_union=int((~data['selected_by_control']['atom_protected']&data['selected_by_control']['teacher_union_protected']).sum()),
            actual_pixel_error_changes_vs_marginal_union=changes)
    macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS}
    comparisons={a+'__'+b:{k:paired({s:v[k] for s,v in per[a].items()},{s:v[k] for s,v in per[b].items()},'LERF '+k) for k in macro[a]} for a,b in combinations(ARMS,2)}
    report=dict(passed=True,source_assisted_diagnostic=True,deployment_claim=False,whole_five_task=False,
        macro=macro,per_scene=per,comparisons=comparisons,details=details,total_atom_state_bytes=sum(x['atom_state_bytes'] for x in details.values()),
        all_predictions_before_scoring=True,all_cached_control_payloads_exact=True,gt_hashes_current_audit_snapshot=True,
        gt_files=list(labels.values()),code=receipt(__file__),design=receipt(ROOT/'design.json'))
    write(ROOT/'validation/audit.json',report);write('docs/experiments/2026-10-10-compositional-atoms-audit.json',report)
    print(macro,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
