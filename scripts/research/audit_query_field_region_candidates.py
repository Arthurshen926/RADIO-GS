"""Validate exhaustive candidate receipts, nesting and unchanged GT bindings."""
import numpy as np
import torch
from torch.nn import functional as F
from pathlib import Path
from scripts.research.run_query_field_relations import read,write,receipt,check
from scripts.research.diagnose_query_field_region_candidates import ROOT,PREVIOUS,design

def main():
    d=design();barrier=read(ROOT/'predictions_complete.json');summary=read(ROOT/'summary.json');records={}
    for r in barrier['records']:check(r)
    for r in read(ROOT/'helper_receipts.json')['files']:check(r)
    for j in d['jobs']:
        name=j['name'];root=ROOT/'predictions'/name
        for r in read(root/'receipt.json')['files']:check(r)
        meta=read(root/'metadata.json');alpha=np.load(root/'alpha.npy',mmap_mode='r')
        assert alpha.shape==(meta['regions'],*meta['hw'])
        for start in range(0,len(alpha),64):
            chunk=alpha[start:start+64];assert np.isfinite(chunk).all() and chunk.min()>=0 and chunk.max()<=1+1e-6
        c=np.asarray(meta['coefficient']);compatible=np.asarray(meta['compatible']);assert np.all(compatible[c>0])
        old=read(PREVIOUS/'predictions'/name/'stages.json')
        assert meta['regions']==old['regions'] and int(compatible.sum())==old['compatible'] and int((c>0).sum())==old['admitted']
        result=summary['per_case'][name];choices=result['choices']
        scores=[choices[k]['iou'] if choices[k] else 0 for k in ['all','compatible','admitted','coefficient_top']]
        assert all(x+1e-12>=y for x,y in zip(scores,scores[1:]))
        assert choices['coefficient_top'] is None or choices['coefficient_top']['region']==int(c.argmax())
        gt=result['gt'];check(gt);assert gt['sha256']==read(PREVIOUS/'predictions'/name/'evaluation.json')['gt_sha256']
        assert not meta['target_labels_opened']
        errors={}
        cfg=read(j['configuration']['path'])
        for arm,a in np.load(root/'controls.npz').items():
            previous=torch.load(PREVIOUS/'predictions'/name/(name+'__'+arm)/'predictions.pt',map_location='cpu',weights_only=True)
            restored=F.interpolate(torch.from_numpy(a)[None,None],cfg['output_hw'],mode='bilinear',align_corners=False)[0,0]
            errors[arm]=float((restored-previous['response']).abs().max())
            assert errors[arm]<=1e-6
        records[name]=dict(control_response_max_errors=errors,regions=len(alpha),pool_nesting=True,unchanged_gate_counts=True,unchanged_gt=True,finite_alpha=True)
    result=dict(passed=True,cases=records,regions=sum(x['regions'] for x in records.values()),predictions_before_scoring=True,barrier=receipt(ROOT/'predictions_complete.json'),summary=receipt(ROOT/'summary.json'),scope='Receipt and numeric checks, not OS isolation; diagnostic oracle never promoted.')
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-09-region-candidates-audit.json',result)
    print(result,flush=True)

if __name__=='__main__':main()
