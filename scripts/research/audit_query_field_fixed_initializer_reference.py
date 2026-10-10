"""Post-score audit for the privileged joint-solver capacity reference."""
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_fixed_initializer_reference import ROOT,REFERENCE,FINITE,ARMS,read,write,receipt,check,design
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();cfg=read(d['configuration']['path']);per=read(ROOT/'summary.json')['per_scene'];details={};labels={}
    for stage in ['predictions','scoring']:
        for r in read(ROOT/(stage+'_complete.json'))['records']:check(r)
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));check(r['teacher_selection'])
        data=torch.load(p/'predictions.pt',weights_only=True);old=torch.load(REFERENCE/'predictions'/j['name']/'predictions.pt',weights_only=True)
        finite=torch.load(FINITE/'predictions'/j['name']/'predictions.pt',weights_only=True)
        assert torch.equal(data['scores']['finite_parts'],finite['scores']['atom_parts'])
        assert torch.equal(data['scores']['joint_source_initializer'],old['scores']['joint_footprint'])
        assert torch.equal(data['observed'],old['observed']) and data['texts']==old['texts']
        for k,v in data['geometry'].items():assert torch.equal(v,old['geometry'][k])
        assert r['transient_query_fitting'] and not r['persistent_field_updated'] and not r['target_labels_opened']
        fitting={a:dict(active_queries=0,steps=0,seconds=0.) for a in ARMS[2:]}
        for qi,q in enumerate(r['diagnostics']):
            for a in ARMS[2:]:
                v=q['controls'][a];assert v['steps']==(d['steps'] if v['active_views'] else 0)
                fitting[a]['active_queries']+=int(v['active_views']>0);fitting[a]['steps']+=v['steps'];fitting[a]['seconds']+=v.get('seconds',0.)
                if not v['active_views']:
                    assert torch.equal(data['scores'][a][:,qi],data['scores']['finite_parts'][:,qi])
                    assert torch.equal(data['scores'][a][:,qi],old['scores']['source_parts'][:,qi])
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
        details[j['scene']]=dict(queries=len(data['texts']),fitting=fitting,geometry_exact=True,initializer_exact=True)
    macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS}
    comparisons={a+'__'+b:{k:paired({s:v[k] for s,v in per[a].items()},{s:v[k] for s,v in per[b].items()},'LERF '+k) for k in macro[a]} for a,b in combinations(ARMS,2)}
    report=dict(passed=True,source_assisted_diagnostic=True,transient_query_fitting=True,deployment_claim=False,whole_five_task=False,
        macro=macro,per_scene=per,comparisons=comparisons,details=details,all_predictions_before_scoring=True,
        gt_hashes_current_audit_snapshot=True,gt_files=list(labels.values()),code=receipt(__file__),design=receipt(ROOT/'design.json'))
    write(ROOT/'validation/audit.json',report);write('docs/experiments/2026-10-10-fixed-initializer-reference-audit.json',report);print(macro,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
