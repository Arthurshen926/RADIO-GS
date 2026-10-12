"""Post-score pixel changes; target labels never enter field or query decisions."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from scripts.research.run_query_field_reference_edit import ROOT,PREVIOUS,design
from scripts.research.run_query_field_relations import read,write,receipt,check
from scripts.research.audit_query_field_reference_extent import mask_metrics


def main():
    d=design();assert read(ROOT/'audit.json')['passed']
    summaries={'composed':read(PREVIOUS/'summary.json'),
        'free':read(d['grouped_summary']['path']),'edit':read(ROOT/'summary.json'),'none':read(ROOT/'summary.json')}
    roots={'composed':PREVIOUS,'free':Path(d['grouped_summary']['path']).parent,'edit':ROOT,'none':ROOT}
    cases={};evidence=[]
    for j in d['jobs']:
        name=j['name'];scoring=read(j['scoring']['path']);evaluation=read(ROOT/'predictions'/name/'evaluation.json')
        check(dict(path=scoring['gt_mask'],sha256=evaluation['gt_sha256']))
        raw=np.asarray(Image.open(scoring['gt_mask']));truth=np.any(raw>127,axis=2) if raw.ndim==3 else raw>127
        masks={};members={}
        for arm,root in roots.items():
            path=root/'predictions'/name/(name+'__'+arm);record=read(path/'prediction.json')
            check(dict(path=str(path/'predictions.pt'),sha256=record['predictions_sha256']))
            values=torch.load(path/'predictions.pt',weights_only=True,map_location='cpu')
            masks[arm]=cv2.resize(values['response'].numpy(),truth.shape[::-1],interpolation=cv2.INTER_LINEAR)>.5
            members[arm]=values['selected'].numpy();metrics=mask_metrics(masks[arm],truth)
            expected=summaries[arm]['per_case'][arm][name]
            for k,m in [('iou','selection_iou'),('precision','precision'),('recall','recall')]:assert abs(metrics[k]-expected[m])<1e-12
            evidence.append(receipt(path/'prediction.json'))
        comparisons={}
        for source,candidate in [('composed','free'),('composed','none'),('composed','edit'),('free','edit'),('none','edit')]:
            before,after=masks[source],masks[candidate];added=after&~before;removed=before&~after
            comparisons[source+'__'+candidate]=dict(added_tp=int((added&truth).sum()),added_fp=int((added&~truth).sum()),
                removed_tp=int((removed&truth).sum()),removed_fp=int((removed&~truth).sum()),
                added_gaussians=int((members[candidate]&~members[source]).sum()),
                removed_gaussians=int((members[source]&~members[candidate]).sum()))
        cases[name]=dict(target_hw=list(truth.shape),foreground_pixels=int(truth.sum()),comparisons=comparisons)
    result=dict(cases=cases,posthoc_target_label_diagnosis=True,labels_used_for_predictions=False,
        rendered_pixel_changes_not_gaussian_identity_truth=True,different_image_sizes_not_normalized=True,
        no_prediction_or_configuration_changes=True,evidence=evidence,code=receipt(__file__),audit=receipt(ROOT/'audit.json'))
    write(ROOT/'diagnosis.json',result);write('docs/experiments/2026-10-12-reference-edit-diagnosis.json',result)
    print({name:c['comparisons']['composed__edit'] for name,c in cases.items()},flush=True)


if __name__=='__main__':main()
