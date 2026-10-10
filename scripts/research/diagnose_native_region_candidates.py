"""Source teacher agreement diagnostic; never deployment or target-GT scoring."""
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from scripts.research.run_query_field_native_region import ROOT,SOURCE,read,write,receipt
from radio_gs.query_field.part_conditioning import native_region_probability


@torch.no_grad()
def greedy(masks,target):
    union=torch.zeros_like(target);best=0.;chosen=[]
    # Achievable greedy composition, NOT a global optimum or upper bound.
    for _ in range(len(masks)):
        candidates=masks|union[None]
        iou=(candidates&target).sum(1)/(candidates|target).sum(1).clamp_min(1)
        value,index=iou.max(0)
        if float(value)<=best:break
        best=float(value);chosen.append(int(index));union=candidates[index]
    return best,chosen


def main():
    read(ROOT/'scoring_complete.json')
    cfg=read(ROOT/'prediction_config.json');legacy={j['name']:j for j in read('configs/pipeline/system_v10.json')['jobs'] if j['benchmark']=='lerf'}
    out=ROOT/'validation/source_candidates';out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(target_gt_used=False,source_detections_diagnostic_only=True,
        matched_frames='existing partial-region construction only',query_threshold=.5,
        detector_threshold=.5,raster='existing observation raster; bilinear detector probability then >.5',
        greedy='strict IoU improvement, lowest-index tie, achieved feasible composition, not global optimum',
        mixture='source-pixel diagnostic: native-margin weighted mean vs Boolean union on identical accepted raw masks; not Gaussian predictions',
        code=receipt(__file__),driver_design=receipt(ROOT/'design.json')))
    scenes={}
    for job in cfg['jobs']:
        data=torch.load(SOURCE/'predictions'/job['name']/'predictions.pt',weights_only=True)
        texts=data['texts'];vectors=data['native_query_vectors'];negatives=data['generic_negative_vectors'];del data
        construction=read(job['construction']);rows=[];inputs=[]
        for record in construction['source_records']:
            frame=record['frame'];path=Path(job['observations'])/f'view_{frame:05d}.pt'
            detection=Path(legacy[job['scene']]['source_evidence'])/f'identity_{frame}.pt'
            inputs.extend([receipt(path),receipt(detection)])
            raw=torch.load(path,weights_only=True);teacher=torch.load(detection,weights_only=True)
            assert teacher['frame']==frame and all(t in teacher['queries'] for t in texts)
            ids=record['regions'];masks=raw['masks'][ids].bool();height,width=masks.shape[-2:];masks=masks.flatten(1)
            evidence=SimpleNamespace(region_semantic=raw['semantic_descriptors'][ids]);del raw
            for qi,text in enumerate(texts):
                entry=teacher['instances'][teacher['queries'].index(text)]
                target=(F.interpolate(entry['masks'].float().amax(0)[None,None],size=(height,width),mode='bilinear',align_corners=False)[0,0]>.5).flatten() if len(entry['scores']) else torch.zeros(height*width,dtype=torch.bool)
                p=native_region_probability(evidence,vectors[qi],negatives);accepted=p>.5
                pred=masks[accepted].any(0) if accepted.any() else torch.zeros_like(target)
                preference=(2*p-1).clamp_min(0)
                mixture=(masks.float().T@preference/preference.sum().clamp_min(1e-12))>.5
                mixed_tp=int((mixture&target).sum())
                tp=int((pred&target).sum());positive=bool(target.any());row=dict(frame=frame,text=text,teacher_present=positive,
                    accepted_regions=int(accepted.sum()),predicted_pixels=int(pred.sum()),teacher_pixels=int(target.sum()),
                    native_iou=tp/max(int((pred|target).sum()),1),
                    native_mixture_iou=mixed_tp/max(int((mixture|target).sum()),1),
                    native_mixture_recall=mixed_tp/max(int(target.sum()),1),
                    positive_covered_but_mixture_rejected=int((target&pred&~mixture).sum()),native_precision=tp/max(int(pred.sum()),1),native_recall=tp/max(int(target.sum()),1))
                if positive:
                    single=(masks&target).sum(1)/(masks|target).sum(1).clamp_min(1)
                    feasible,chosen=greedy(masks,target)
                    row.update(best_single=float(single.max()),greedy_feasible_iou=feasible,greedy_regions=len(chosen),
                        all_candidate_coverage=float((masks.any(0)&target).sum()/target.sum()))
                rows.append(row)
            print('source diagnostic',job['scene'],frame,flush=True)
        positive=[r for r in rows if r['teacher_present']];absent=[r for r in rows if not r['teacher_present']]
        keys=['native_iou','native_mixture_iou','native_mixture_recall','native_precision','native_recall','best_single','greedy_feasible_iou','all_candidate_coverage']
        scenes[job['scene']]=dict(positive_pairs=len(positive),absent_pairs=len(absent),
            positive_mean={k:sum(r[k] for r in positive)/len(positive) for k in keys},
            absent_with_any_native_region=sum(r['accepted_regions']>0 for r in absent),
            positive_with_no_native_region=sum(r['accepted_regions']==0 for r in positive),rows=rows,inputs=inputs)
    report=dict(scenes=scenes,macro_positive={k:sum(s['positive_mean'][k] for s in scenes.values())/4 for k in keys},
        target_gt_used=False,not_model_accuracy=True,not_global_oracle=True,teacher_may_be_wrong=True,
        no_model_or_threshold_changes=True,code=receipt(__file__))
    write(out/'report.json',report);write('docs/experiments/2026-10-10-native-region-source-diagnostic.json',report)
    print(report['macro_positive'],flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
