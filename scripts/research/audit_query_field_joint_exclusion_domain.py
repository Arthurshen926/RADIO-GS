"""Audit complete joint member exclusion controls and sampled constraints."""
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_joint_exclusion_domain import ROOT,REFERENCE,ARMS,OLD_JOINT,ADMISSION,read,write,receipt,check,design
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();cfg=read(d['configuration']['path']);per=read(ROOT/'summary.json')['per_scene'];details={}
    for stage in ['predictions','scoring']:
        for r in read(ROOT/(stage+'_complete.json'))['records']:check(r)
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json');check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
        data=torch.load(p/'predictions.pt',weights_only=True);old=torch.load(REFERENCE/'predictions'/j['name']/'predictions.pt',weights_only=True)
        assert not any(r[k] for k in ['target_labels_opened','teacher_acceptance_used','transient_query_fitting','persistent_field_updated','runtime_source_raster_access'])
        assert data['texts']==old['texts'] and torch.equal(data['observed'],old['observed'])
        for k,v in data['geometry'].items():assert torch.equal(v,old['geometry'][k])
        for a in ARMS:
            assert torch.equal(data['scores'][a],old['scores']['confidence_scope'])
            assert torch.equal(data['selected_by_control'][a],data['membership_by_control'][a]>.5)
        admission=torch.load(ADMISSION/'predictions'/j['name']/'predictions.pt',weights_only=True)
        previous=torch.load(OLD_JOINT/'predictions'/j['name']/'predictions.pt',weights_only=True)
        assert torch.equal(data['selected_by_control']['semantic_initial'],admission['selected'])
        assert torch.equal(data['selected_by_control']['observed_joint'],previous['selected_by_control']['joint_coverage'])
        assert torch.equal(data['selected_by_control']['observed_fixed'],previous['selected_by_control']['frozen_visibility'])
        assert torch.equal(data['selected'],data['selected_by_control']['joint_coverage'])
        for arm in ['frozen_visibility','joint_coverage']:
            assert not bool((data['selected_by_control'][arm]&~data['selected_by_control']['semantic_initial']).any())
            for item in r['diagnostics']:
                x=item['controls'][arm]
                assert x['positive_coverage_losses']==0 and x['negative_covered_after']<=x['negative_covered_before']
        validation=read(ROOT/'validation'/(j['scene']+'.json'))
        assert validation['passed'] and validation['conditions']==16 and validation['all_factors_checked']
        e=read(p/'evaluation/evaluation.json')
        for a in ARMS:
            precision=[];recall=[]
            for x in e['metrics'][a]['selected']['objects']:
                assert not x['missing'];gt=load_binary_mask(Path(x['gt_path']),threshold=10)
                pred=load_binary_mask(Path(x['pred_path']),threshold=10,grayscale=True);tp=int((gt&pred).sum())
                assert abs(tp/max(int((gt|pred).sum()),1)-x['iou'])<1e-12
                precision.append(tp/max(int(pred.sum()),1));recall.append(tp/max(int(gt.sum()),1))
            per[a][j['scene']].update(precision=float(np.mean(precision)),recall=float(np.mean(recall)))
        details[j['scene']]=dict(queries=len(data['texts']),geometry_exact=True,semantic_exact=True,
            controls={a:{k:sum(x['controls'][a][k] for x in r['diagnostics']) for k in
                ['removed','initial_selected','sampled_positive','sampled_negative','positive_covered_before','positive_coverage_losses','negative_covered_before','negative_covered_after']}
                for a in ['frozen_visibility','joint_coverage']})
    macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS}
    comparisons={a+'__'+b:{k:paired({s:v[k] for s,v in per[a].items()},{s:v[k] for s,v in per[b].items()},'LERF '+k) for k in macro[a]} for a,b in [('semantic_initial','frozen_visibility'),('semantic_initial','joint_coverage'),('observed_joint','joint_coverage'),('observed_fixed','frozen_visibility'),('frozen_visibility','joint_coverage')]}
    report=dict(passed=True,macro=macro,per_scene=per,comparisons=comparisons,details=details,all_predictions_before_scoring=True,
        model_promotion=False,whole_five_task=False,teacher_acceptance_used=False,query_fitting=False,code=receipt(__file__),design=receipt(ROOT/'design.json'))
    write(ROOT/'validation/audit.json',report);write('docs/experiments/2026-10-11-joint-exclusion-domain-audit.json',report);print(macro,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
