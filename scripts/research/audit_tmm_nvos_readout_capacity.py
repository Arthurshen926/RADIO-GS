"""Diagnostic GT-oracle threshold capacity of retained v10, never an inference rule."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from scripts.research.audit_tmm_protocol_correctness import REPO,SYSTEM,read,load,binary_counts
from scripts.research.replay_tmm_response import STUDY,POLICY,sha


def curve_capacity(scores,truth):
    # Inclusive >= decisions, with whole equal-score groups processed together.
    scores=scores.ravel();truth=truth.ravel();order=np.argsort(scores,kind='stable')[::-1]
    sorted_score=scores[order];g=truth[order];ends=np.r_[np.flatnonzero(sorted_score[:-1]!=sorted_score[1:]),len(scores)-1]
    tp=np.cumsum(g,dtype=np.int64)[ends];fp=ends+1-tp;positive=int(truth.sum())
    iou=tp/(positive+fp);best=float(iou.max());ap=float(np.sum(np.diff(np.r_[0,tp])/positive*tp/(tp+fp)))
    # No GT-selected threshold is exported, installed, or used to produce a method output.
    return dict(oracle_iou=best,average_precision=ap,unique_decisions=len(ends),positive_pixels=positive)


def main():
    os.chdir(REPO);jobs=[j for j in read('configs/pipeline/system_v10.json')['jobs'] if j['benchmark']=='nvos']
    run=BoundedExperiment(STUDY,'nvos_readout_capacity_v1',POLICY);run.preflight(128*1024)
    source=Path(__file__).read_text();run.json('manifest.json',dict(jobs=jobs,source_text=source,source_sha256=sha(__file__),
        historical_predictions_only=True,gt_oracle_diagnostic_only=True,deployable=False,thresholds_exported=False,default_changed=False))
    rows={};hashes={}
    for j in jobs:
        n=j['name'];spec=read(j['evaluation']);path=SYSTEM/'nvos'/n/'evaluation/response.pt';score=load(path)['response'].numpy()
        truth=np.asarray(Image.open(spec['gt_mask']));score=cv2.resize(score,(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR)
        assert np.isfinite(score).all();rows[n]={}
        for track,g in dict(retained_gt_gt127=truth>127,ludvig_gt_maximum=(truth//truth.max()).astype(bool)).items():
            r=curve_capacity(score,g);r['fixed_half']=binary_counts(score>=.5,g);assert r['oracle_iou']>=r['fixed_half']['iou']-1e-12
            if track=='retained_gt_gt127':assert abs(r['fixed_half']['iou']-read(SYSTEM/'nvos'/n/'result.json')['metrics']['iou'])<1e-7
            rows[n][track]=r
        for p in (j['evaluation'],str(path),spec['gt_mask']):hashes[p]=sha(p)
        run.json(n+'.json',rows[n]);print(n,rows[n]['retained_gt_gt127']['oracle_iou'],flush=True)
    run.json('summary.json',dict(complete=True,rows=rows,input_hashes=hashes,gt_oracle_diagnostic_only=True,
        average_precision_measured=True,thresholds_exported=False,deployable=False,per_task_winners=False,
        macro={t:dict(oracle_iou=float(np.mean([r[t]['oracle_iou'] for r in rows.values()])),
            fixed_half_iou=float(np.mean([r[t]['fixed_half']['iou'] for r in rows.values()])),
            average_precision=float(np.mean([r[t]['average_precision'] for r in rows.values()]))) for t in rows[jobs[0]['name']]},
        durable_dense_prediction_bytes=0,default_changed=False))


if __name__=='__main__':main()
