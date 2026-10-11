"""Independent mask replay and marginal extension audit for region scope."""
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_region_scope import ROOT,PARENT,ARMS,read,write,receipt,check,design
from scripts.research.audit_query_field_extension_quality import compare_masks
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();cfg=read(d['configuration']['path']);summary=read(ROOT/'summary.json');scenes={}
    for r in read(ROOT/'predictions_complete.json')['records']:check(r)
    for j in cfg['jobs']:
        out=ROOT/'predictions'/j['name'];r=read(out/'prediction.json');check(dict(path=str(out/'predictions.pt'),sha256=r['predictions_sha256']))
        data=torch.load(out/'predictions.pt',weights_only=True)
        previous=torch.load(PARENT/'predictions'/j['name']/'predictions.pt',weights_only=True)
        for a in ARMS:
            assert torch.equal(data['selected_by_control'][a],data['membership_by_control'][a]>.5)
            assert torch.equal(data['scores'][a],previous['scores']['semantic_initial'])
        for a in ARMS[:2]:assert torch.equal(data['selected_by_control'][a],previous['selected_by_control'][a])
        initial=data['selected_by_control']['frozen_visibility']
        assert not bool((initial&~data['selected_by_control']['conditional_addition']).any())
        assert not r['target_labels_opened'] and not r['source_access']
        e=read(out/'evaluation/evaluation.json');metrics={};objects={}
        for a in ARMS:
            rows=e['metrics'][a]['selected']['objects'];objects[a]={(x['frame'],x['query']):x for x in rows}
            assert len(objects[a])==len(rows)
            ious=[];precision=[];recall=[]
            for x in rows:
                assert not x['missing']
                gt=load_binary_mask(Path(x['gt_path']),threshold=10)
                pred=load_binary_mask(Path(x['pred_path']),threshold=10,grayscale=True)
                tp=int((gt&pred).sum());iou=tp/max(int((gt|pred).sum()),1)
                assert abs(iou-x['iou'])<1e-12
                ious.append(iou);precision.append(tp/max(int(pred.sum()),1));recall.append(tp/max(int(gt.sum()),1))
            assert abs(float(np.mean(ious))-summary['per_scene'][a][j['scene']])<1e-12
            metrics[a]=dict(iou=float(np.mean(ious)),precision=float(np.mean(precision)),recall=float(np.mean(recall)))
        extension={}
        for a in ['seed_addition','conditional_addition']:
            rows=[]
            for key,x in objects['frozen_visibility'].items():
                y=objects[a][key];assert x['gt_path']==y['gt_path']
                gt=load_binary_mask(Path(x['gt_path']),threshold=10)
                before=load_binary_mask(Path(x['pred_path']),threshold=10,grayscale=True)
                after=load_binary_mask(Path(y['pred_path']),threshold=10,grayscale=True)
                rows.append(dict(frame=key[0],query=key[1],**compare_masks(before,after,gt)))
            tp=sum(x['added_tp'] for x in rows);fp=sum(x['added_fp'] for x in rows)
            extension[a]=dict(added_tp=tp,added_fp=fp,added_precision=tp/max(tp+fp,1),
                removed_pixels=sum(x['removed_pixels'] for x in rows),beneficial_cases=sum(x['beneficial'] for x in rows),harmful_cases=sum(x['harmful'] for x in rows),rows=rows)
        scenes[j['scene']]=dict(metrics=metrics,extension=extension,queries=len(r['diagnostics']),
            applied=sum(x['applied'] for x in r['diagnostics']),
            added_members={a:int((data['selected_by_control'][a]&~initial).sum()) for a in ARMS[2:]},
            removed_members={a:int((initial&~data['selected_by_control'][a]).sum()) for a in ARMS[2:]})
    macro={a:{k:sum(s['metrics'][a][k] for s in scenes.values())/4 for k in ['iou','precision','recall']} for a in ARMS}
    compare={a:paired({s:v['metrics']['frozen_visibility']['iou'] for s,v in scenes.items()},
                     {s:v['metrics'][a]['iou'] for s,v in scenes.items()},'actual selected-set IoU') for a in ARMS[2:]}
    report=dict(passed=True,macro=macro,scenes=scenes,comparisons=compare,design=receipt(ROOT/'design.json'),
        code=receipt(__file__),posthoc_extension_diagnosis=True,all_ious_replayed=True,default_promoted=False,whole_five_task=False)
    write(ROOT/'audit.json',report);write('docs/experiments/2026-10-11-binary-region-scope-audit.json',report)
    print(macro,flush=True);print(compare,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
