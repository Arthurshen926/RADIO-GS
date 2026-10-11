"""Independent saved-mask metric and matched member-function audit."""
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_member_witness import ROOT,PARENT,ARMS,read,write,receipt,check,design
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();cfg=read(d['configuration']['path']);per=read(ROOT/'summary.json')['per_scene'];details={}
    for stage in ['predictions','scoring']:
        for r in read(ROOT/(stage+'_complete.json'))['records']:check(r)
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
        data=torch.load(p/'predictions.pt',weights_only=True)
        old=torch.load(PARENT/'predictions'/j['name']/'predictions.pt',weights_only=True)
        assert not any(r[k] for k in ['target_labels_opened','teacher_acceptance_used','transient_query_fitting','persistent_field_updated','runtime_source_raster_access'])
        assert data['texts']==old['texts'] and torch.equal(data['observed'],old['observed'])
        for k,v in data['geometry'].items():assert torch.equal(v,old['geometry'][k])
        for a in ARMS:
            assert torch.equal(data['scores'][a],old['scores']['semantic_initial'])
            assert torch.equal(data['selected_by_control'][a],data['membership_by_control'][a]>.5)
        for a in ARMS[:3]:assert torch.equal(data['selected_by_control'][a],old['selected_by_control'][a])
        assert torch.equal(data['selected'],data['selected_by_control']['witnessed_energy'])
        for qi,item in enumerate(r['diagnostics']):
            for a in ARMS[3:]:
                x=item['controls'][a];assert np.max(np.diff(x['energy_trace']))<=1e-10
                initial=data['selected_by_control']['semantic_initial'][:,qi];selected=data['selected_by_control'][a][:,qi]
                assert x['added']==int((selected&~initial).sum()) and x['removed']==int((initial&~selected).sum())
                assert x['passes']<=8 and not x['query_parameter_fitting']
        e=read(p/'evaluation/evaluation.json')
        for a in ARMS:
            precision=[];recall=[]
            for x in e['metrics'][a]['selected']['objects']:
                assert not x['missing'];gt=load_binary_mask(Path(x['gt_path']),threshold=10)
                pred=load_binary_mask(Path(x['pred_path']),threshold=10,grayscale=True);tp=int((gt&pred).sum())
                assert abs(tp/max(int((gt|pred).sum()),1)-x['iou'])<1e-12
                precision.append(tp/max(int(pred.sum()),1));recall.append(tp/max(int(gt.sum()),1))
            per[a][j['scene']].update(precision=float(np.mean(precision)),recall=float(np.mean(recall)))
        added_priors={}
        for a in ARMS[3:]:
            mask=data['selected_by_control'][a]&~data['selected_by_control']['semantic_initial']
            values=data['membership_by_control']['semantic_initial'][mask]
            added_priors[a]=dict(count=len(values),below_01=int((values<.1).sum()),below_001=int((values<.01).sum()),
                median=float(values.median()) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)
        details[j['scene']]=dict(queries=len(data['texts']),added_native_priors=added_priors,controls={a:{k:sum(x['controls'][a][k] for x in r['diagnostics'])
            for k in ['added','removed','exchanges','sampled_positive_losses','sampled_positive_gains']} for a in ARMS[3:]})
    macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS}
    comparisons={a+'__witnessed_energy':{k:paired({s:v[k] for s,v in per[a].items()},
        {s:v[k] for s,v in per['witnessed_energy'].items()},'LERF '+k) for k in macro[a]} for a in ARMS if a!='witnessed_energy'}
    report=dict(passed=True,macro=macro,per_scene=per,comparisons=comparisons,details=details,
        all_predictions_before_scoring=True,model_promotion=False,whole_five_task=False,code=receipt(__file__),design=receipt(ROOT/'design.json'))
    write(ROOT/'audit.json',report);write('docs/experiments/2026-10-11-member-witness-audit.json',report);print(macro,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
