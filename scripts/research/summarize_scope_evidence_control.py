"""Summarize frozen benchmark scoring and separately labelled teacher agreement."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask


def mean(values):return float(np.mean(values)) if values else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    results={}
    for arm in ['fusion','fit','abstention','identity','identity_veto']:
        scenes=[];queries=[]
        for scene in ['figurines','ramen','teatime','waldo_kitchen']:
            dest=root/arm/scene
            evaluation=json.loads((dest/'evaluation/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
            fit=json.loads((dest/'sets'/f'lerf_{scene}'/'fit.json').read_text())
            held={r['query']:r for r in fit['heldout_teacher_agreement']};byquery=defaultdict(list)
            for obj in evaluation['selected']['objects']:
                truth=load_binary_mask(Path(obj['gt_path']));prediction=load_binary_mask(Path(obj['pred_path']),grayscale=True)
                tp=int((truth&prediction).sum());fp=int((~truth&prediction).sum());fn=int((truth&~prediction).sum())
                byquery[obj['query']].append(dict(iou=obj['iou'],precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1)))
            rows=[]
            for query,objects in byquery.items():
                row=dict(scene=scene,query=query,iou=mean([r['iou'] for r in objects]),precision=mean([r['precision'] for r in objects]),recall=mean([r['recall'] for r in objects]),heldout_teacher=held[query])
                rows.append(row);queries.append(row)
            hs=[h for h in held.values() if h['views']]
            scenes.append(dict(scene=scene,selected_iou=evaluation['selected']['miou'],semantic_iou=evaluation['iou'],ap=evaluation['ap'],mean_query_precision=mean([r['precision'] for r in rows]),mean_query_recall=mean([r['recall'] for r in rows]),heldout_teacher_iou=mean([h['mean_view_iou'] for h in hs]),heldout_teacher_recall=mean([h['positive_recall'] for h in hs]),heldout_teacher_fpr=mean([h['complement_false_positive_rate'] for h in hs]),heldout_views=len(fit['heldout_source_frames']),removed_negative_pixels=sum(fit['conflict_negative_pixels_removed']),fit_seconds=fit['seconds']))
        results[arm]=dict(scenes=scenes,macro={key:mean([s[key] for s in scenes]) for key in ['selected_iou','semantic_iou','ap','mean_query_precision','mean_query_recall','heldout_teacher_iou','heldout_teacher_recall','heldout_teacher_fpr']},queries=queries)
    base={(r['scene'],r['query']):r for r in results['fit']['queries']}
    deltas=[]
    for r in results['abstention']['queries']:
        b=base[r['scene'],r['query']]
        deltas.append(dict(scene=r['scene'],query=r['query'],iou_delta=r['iou']-b['iou'],precision_delta=r['precision']-b['precision'],recall_delta=r['recall']-b['recall']))
    candidate_deltas={}
    for arm in ['abstention','identity','identity_veto']:
        candidate_deltas[arm]=sorted([dict(scene=r['scene'],query=r['query'],iou_delta=r['iou']-base[r['scene'],r['query']]['iou'],precision_delta=r['precision']-base[r['scene'],r['query']]['precision'],recall_delta=r['recall']-base[r['scene'],r['query']]['recall']) for r in results[arm]['queries']],key=lambda r:r['iou_delta'])
    gaps=sorted([dict(scene=r['scene'],query=r['query'],actual_iou=r['iou'],teacher_iou=r['heldout_teacher']['mean_view_iou'],teacher_views=r['heldout_teacher']['views']) for r in results['fit']['queries'] if r['heldout_teacher']['views']],key=lambda r:r['teacher_iou']-r['actual_iou'],reverse=True)
    out=dict(arms=results,candidate_deltas=candidate_deltas,abstention_deltas=sorted(deltas,key=lambda r:r['iou_delta']),teacher_benchmark_gaps=gaps,protocol='frozen development scoring; held-out per-view detections are pseudo labels, not independent GT; frozen map still uses all declared sources',decision='Conflict abstention rejected; identity candidate reported separately; default recipe remains fixed pending stage conclusions')
    (root/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    for arm,data in results.items():print(arm,data['macro'])
    print('largest teacher/benchmark gaps',gaps[:8])
    print('largest regressions',sorted(deltas,key=lambda r:r['iou_delta'])[:6])


if __name__=='__main__':main()
