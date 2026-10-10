"""Recompute all actual projections; audit independent semantic/member roles."""
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_scope_exclusion import ROOT,REFERENCE,FINITE,SEMANTIC,ARMS,read,write,receipt,check,design
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();cfg=read(d['configuration']['path']);per=read(ROOT/'summary.json')['per_scene'];details={};labels={}
    for stage in ['predictions','scoring']:
        for r in read(ROOT/(stage+'_complete.json'))['records']:check(r)
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));check(r['teacher_selection']);check(r['semantic_parent'])
        data=torch.load(p/'predictions.pt',weights_only=True);old=torch.load(REFERENCE/'predictions'/j['name']/'predictions.pt',weights_only=True)
        finite=torch.load(FINITE/'predictions'/j['name']/'predictions.pt',weights_only=True)
        native=torch.load(SEMANTIC/'predictions'/j['name']/'predictions.pt',weights_only=True)
        assert torch.equal(data['scores']['partial_member'],old['scores']['joint_finite_initializer'])
        for a in ['partial_typed','scope_typed']:assert torch.equal(data['scores'][a],native['scores']['measure_prior'])
        for a,b in [('partial_member','partial_typed'),('scope_member','scope_typed')]:
            assert torch.equal(data['selected_by_control'][a],data['selected_by_control'][b])
            assert torch.equal(data['selected_by_control'][a],data['scores'][a]>.5)
        assert torch.equal(data['selected'],data['selected_by_control']['scope_typed'])
        assert torch.equal(data['observed'],old['observed']) and data['texts']==old['texts']
        for k,v in data['geometry'].items():assert torch.equal(v,old['geometry'][k])
        assert r['transient_query_fitting'] and not r['persistent_field_updated'] and not r['target_labels_opened']
        fitting=dict(active_queries=0,steps=0,seconds=0.,changed_queries=0,changed_source_pairs=0,added_negative_pixels=0,old_negative_pixels=0,new_negative_pixels=0)
        for qi,q in enumerate(r['diagnostics']):
            v=q['fitting'];assert v['steps']==(d['steps'] if v['active_views'] else 0)
            assert v['active_views']==read(REFERENCE/'predictions'/j['name']/'prediction.json')['diagnostics'][qi]['controls']['joint_finite_initializer']['active_views']
            fitting['active_queries']+=int(v['active_views']>0);fitting['steps']+=v['steps'];fitting['seconds']+=v.get('seconds',0.)
            fitting['changed_queries']+=int(any(x['added_negative_pixels'] for x in q['permissions']))
            for x in q['permissions']:
                assert x['new_negative_pixels']==x['old_negative_pixels']+x['added_negative_pixels']
                fitting['changed_source_pairs']+=int(x['added_negative_pixels']>0)
                for k in ['added_negative_pixels','old_negative_pixels','new_negative_pixels']:fitting[k]+=x[k]
            if not v['active_views']:assert torch.equal(data['scores']['scope_member'][:,qi],finite['scores']['atom_parts'][:,qi])
        e=read(p/'evaluation/evaluation.json');before=read(REFERENCE/'predictions'/j['name']/'evaluation/evaluation.json')
        assert e['evaluation_scene_sha256']==before['evaluation_scene_sha256']
        for a in ARMS:
            precision=[];recall=[]
            for x in e['metrics'][a]['selected']['objects']:
                assert not x['missing'];gt=load_binary_mask(Path(x['gt_path']),threshold=10)
                pred=load_binary_mask(Path(x['pred_path']),threshold=10,grayscale=True);tp=int((gt&pred).sum())
                assert abs(tp/max(int((gt|pred).sum()),1)-x['iou'])<1e-12
                precision.append(tp/max(int(pred.sum()),1));recall.append(tp/max(int(gt.sum()),1))
                if x['gt_path'] not in labels:labels[x['gt_path']]=receipt(x['gt_path'])
            per[a][j['scene']].update(precision=float(np.mean(precision)),recall=float(np.mean(recall)))
        added=(data['selected_by_control']['scope_typed']&~data['selected_by_control']['partial_typed']).sum()
        removed=(~data['selected_by_control']['scope_typed']&data['selected_by_control']['partial_typed']).sum()
        details[j['scene']]=dict(queries=len(data['texts']),fitting=fitting,geometry_exact=True,semantic_exact=True,
            added_member_query_pairs=int(added),removed_member_query_pairs=int(removed))
    macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS}
    comparisons={k:paired({s:v[k] for s,v in per['partial_typed'].items()},{s:v[k] for s,v in per['scope_typed'].items()},'LERF '+k) for k in macro['scope_typed']}
    report=dict(passed=True,source_assisted_diagnostic=True,transient_query_fitting=True,deployment_claim=False,whole_five_task=False,
        macro=macro,per_scene=per,comparisons=comparisons,details=details,all_predictions_before_scoring=True,
        gt_hashes_current_audit_snapshot=True,gt_files=list(labels.values()),code=receipt(__file__),design=receipt(ROOT/'design.json'))
    write(ROOT/'validation/audit.json',report);write('docs/experiments/2026-10-10-scope-exclusion-audit.json',report);print(macro,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
