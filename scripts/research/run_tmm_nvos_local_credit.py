"""N03: sealed all-eight source-verified local response edits and matched controls."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch
from gsplat import rasterization

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.querying.coverage_evidence import coverage_evidence
from radio_gs.querying.local_response_credit import LocalResponseCredit, matched_uniform_edit, huber
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from scripts.research.audit_tmm_protocol_correctness import REPO, SYSTEM, binary_counts, read, load
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.run_tmm_nvos_source_readout import solve
from scripts.research.tmm_ram_predictions import RamPredictionBank

CONFIG='configs/research/nvos_local_credit.json'


def edit_report(old, value):
    delta=(value-old).double()
    return dict(positive_edit_mass=float(delta.clamp_min(0).sum()),
                negative_edit_mass=float((-delta).clamp_min(0).sum()),
                changed_elements=int((delta!=0).sum()))


@torch.inference_mode()
def predict(job, run, bank, config):
    name=job['name'];started=time.monotonic()
    fit_path=SYSTEM/'nvos'/name/'query/reference/0/fit.json';fit=read(fit_path)
    scene=read(fit['scene']);spec=read(job['evaluation']);tracks=load(fit['tracks'])
    assert spec['target_frame'] not in scene['source_frames']
    assert fit['negative_policy']=='original' and not fit['explicit_evidence']
    readout_path=SYSTEM/'nvos'/name/'query/typed_readout.pt';data=load(readout_path)
    old=data['scores'].float();old_observed=data['semantic_observed'].bool()
    binding=data['semantic_geometry']
    geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
    gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256'])
    n=len(geo.xyz);h,w=config['source_hw'];terms=[];frames=[]
    observed=torch.zeros(n,dtype=torch.bool);per_view_ids={}
    eligible=sorted(f for f in scene['source_frames']
                    if f in tracks['frames'] and tracks['frames'][f]['object_logit']>0)
    heldout=eligible[config['source_holdout_phase']::config['source_holdout_stride']]
    assert heldout and set(eligible)-set(heldout)
    inputs=[fit_path,fit['scene'],fit['tracks'],job['queries'],job['evaluation'],
            readout_path,binding['path'],scene['camera_report']]
    for frame in scene['source_frames']:
        cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
        renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,
                                 near_plane=.001*gp.scale,far_plane=100*gp.scale)
        hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),
                                               height=h,width=w,include_depths=False)
        pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['weights'].float()
        ids=gid[weight>1e-8].unique().cpu();observed[ids]=True;per_view_ids[frame]=ids
        if frame not in eligible:
            del hit,pid,gid,weight
            continue
        total=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight)
        op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/total[pid].clamp_min(1e-8),(h*w,n)).coalesce()
        record=tracks['frames'][frame];logits=record['logits'].float().cuda()
        y,mass=coverage_evidence(logits,(h,w));reliability=float(record.get('negative_reliability',1.))
        mass=mass*(y+(1-y)*reliability)*(total[:,None]>1e-8)
        terms.append((op.cpu(),y.cpu(),mass.cpu(),1.));frames.append(frame)
        del hit,op,logits,pid,gid,weight,total,y,mass
    print('operators',name,len(frames),'views',flush=True)
    full=solve(terms,observed,config)
    split_observed=torch.zeros(n,dtype=torch.bool)
    for frame,ids in per_view_ids.items():
        if frame not in heldout:split_observed[ids]=True
    split=solve([t for f,t in zip(frames,terms) if f not in heldout],split_observed,config)
    # Existing anchor support is kept for local outputs. Unknown support never
    # gains edit permission merely from a prior or from changed normalization.
    domain=old_observed&observed
    candidate=torch.where(domain[:,None],full,old)
    common=domain&split_observed
    split_candidate=torch.where(common[:,None],split,old)
    credits=dict(fit_local=LocalResponseCredit(old,candidate,config['huber_clip']),
                 held_local=LocalResponseCredit(old,split_candidate,config['huber_clip']))
    source_rows=[]
    for frame,(operator,y,known,_) in zip(frames,terms):
        if frame not in heldout:continue
        indices=operator.indices();vals=operator.values()*common[indices[1]]
        mass=torch.zeros(h*w).scatter_add_(0,indices[0],vals)
        op=torch.sparse_coo_tensor(indices,vals/mass[indices[0]].clamp_min(1e-8),operator.shape).coalesce().to_sparse_csr()
        weight=known*(mass[:,None]>.5)
        for credit in credits.values():credit.add(op,y,weight)
        source_rows.append(dict(frame=frame,common_pixels=int((weight>0).sum()),
                                common_weight=float(weight.sum()),available_weight=float(known.sum())))
    gates={a:c.gate(candidate) for a,c in credits.items()}
    values=dict(v10=old,coverage=full,coverage_anchor_domain=candidate,
                **{a:old+g*(candidate-old) for a,g in gates.items()})
    values['held_uniform'],matching=matched_uniform_edit(old,candidate,gates['held_local'])
    credit_reports={a:c.report() for a,c in credits.items()}
    for a,row in credit_reports.items():
        assert abs(row['conservation_error'])<1e-6*max(1,abs(row['source_huber_gain']))
        row.update(gate_mean=float(gates[a].mean()),nonzero_gates=int((gates[a]>0).sum()),
                   full_gates=int((gates[a]>=1-1e-6).sum()),
                   opposite_or_zero_edit=int((((candidate-old)*credits[a].delta)<=0).sum()))
    edits={a:edit_report(old,v) for a,v in values.items()}
    for sign in ('positive_edit_mass','negative_edit_mass'):
        assert abs(edits['held_local'][sign]-edits['held_uniform'][sign])<1e-5*max(1,edits['held_local'][sign])
    source_objectives={a:0. for a in values}
    for frame,(operator,y,known,_) in zip(frames,terms):
        if frame not in heldout:continue
        indices=operator.indices();vals=operator.values()*common[indices[1]]
        mass=torch.zeros(h*w).scatter_add_(0,indices[0],vals)
        op=torch.sparse_coo_tensor(indices,vals/mass[indices[0]].clamp_min(1e-8),operator.shape).coalesce().to_sparse_csr()
        weight=known*(mass[:,None]>.5)
        for arm,value in values.items():
            source_objectives[arm]+=float((weight*huber(y-torch.sparse.mm(op,value),config['huber_clip'])).sum(dtype=torch.float64))
    del split,terms,per_view_ids,operator,op,credits,gates
    gc.collect();torch.cuda.empty_cache()
    cam=gp.cameras[spec['target_frame']];width,height=spec['raster_wh']
    k=cam.intrinsic.cuda().clone();k[0]*=width/cam.width;k[1]*=height/cam.height
    predictions={}
    for arm in config['arms']:
        support=observed if arm=='coverage' else old_observed
        value=values[arm].clone();value[~support]=0
        image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,
            opacities=geo.opacities.flatten(),colors=torch.cat([value.cuda(),support.float().cuda()[:,None]],1),
            viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=width,height=height,
            near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
        mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
        response=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0).cpu()
        predictions[arm]=bank.add(run,name+'_'+arm,response.reshape(-1,1))
    report=dict(job=job,predictions=predictions,credit=credit_reports,edits=edits,matching=matching,
        source_huber_objectives=source_objectives,
        calibration_frames=heldout,split_fit_frames=[f for f in frames if f not in heldout],
        all_fit_frames=frames,source_rows=source_rows,input_hashes={str(p):sha(p) for p in inputs},
        target_gt_read=False,target_rgb_used=False,calibration_pixels_identical=True,
        support_elements=dict(v10=int(old_observed.sum()),coverage=int(observed.sum()),
                              common_credit=int(common.sum()),anchor_candidate=int(domain.sum())),
        seconds=time.monotonic()-started)
    run.json(name+'_prediction.json',report)
    print('sealed',name,'six arms','seconds',round(time.monotonic()-started),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',help='new unique run name; completed runs are never overwritten')
    args=parser.parse_args()
    os.chdir(REPO);torch.set_num_threads(4);config=read(CONFIG)
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    run=BoundedExperiment(STUDY,args.run or config['run'],POLICY);run.preflight(config['maximum_new_durable_bytes'])
    bank=RamPredictionBank(limit=config['maximum_volatile_prediction_bytes'])
    jobs=read('configs/pipeline/system_v10.json')['jobs'];nvos=[j for j in jobs if j['benchmark']=='nvos']
    sources=[__file__,CONFIG,'radio_gs/querying/coverage_evidence.py','radio_gs/querying/local_response_credit.py',
             'radio_gs/querying/response_reconstruction.py','scripts/research/run_tmm_nvos_source_readout.py',
             'scripts/research/tmm_ram_predictions.py','scripts/research/audit_tmm_protocol_correctness.py',
             'radio_gs/training/retrieval_metrics.py','docs/adr/0026-local-response-edit-credit.md']
    parent=STUDY/config['parent'];parent_summary=read(parent/'summary.json')
    run.json('manifest.json',dict(config=config,jobs=nvos,source_hashes={p:sha(p) for p in sources},
        parent_summary_sha256=sha(parent/'summary.json'),torch_version=torch.__version__,
        volatile_prediction_directory=str(bank.root),other_benchmarks='unchanged v10 references'))
    run.json('source_snapshot.json',{p:dict(sha256=sha(p),text=Path(p).read_text()) for p in sources})
    reports=[]
    try:
        for job in nvos:
            reports.append(predict(job,run,bank,config));gc.collect();torch.cuda.empty_cache()
        run.json('predictions_complete.json',dict(jobs=8,images=48,target_gt_read=False,
            report_hashes={r['job']['name']:sha(run.root/(r['job']['name']+'_prediction.json')) for r in reports},
            predictions={r['job']['name']:r['predictions'] for r in reports},threshold=.5))
        print('all 48 predictions sealed; opening GT for scoring',flush=True)
        rows={};gt_hashes={}
        for job,report in zip(nvos,reports):
            name=job['name'];spec=read(job['evaluation']);gt=np.asarray(Image.open(spec['gt_mask']))
            gt_hashes[spec['gt_mask']]=sha(spec['gt_mask'])
            truths=dict(retained_gt_gt127=gt>127,ludvig_gt_maximum=(gt//gt.max()).astype(bool))
            rows[name]={}
            for arm in config['arms']:
                response=bank.read(run,name+'_'+arm,1).reshape(spec['raster_wh'][1],spec['raster_wh'][0]).numpy()
                response=cv2.resize(response,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)
                rows[name][arm]={t:dict(**binary_counts(response>=.5,g),
                    average_precision=binary_ranking_metrics(torch.from_numpy(response).flatten(),torch.from_numpy(g).flatten())['average_precision'])
                    for t,g in truths.items()}
            for t in truths:
                for metric in ('iou','average_precision'):
                    assert abs(rows[name]['coverage'][t][metric]-parent_summary['rows'][name]['fixed_half'][t][metric])<1e-5
            retained=read(SYSTEM/'nvos'/name/'result.json')['metrics']
            assert abs(rows[name]['v10']['retained_gt_gt127']['iou']-retained['iou'])<1e-5
            assert abs(rows[name]['v10']['retained_gt_gt127']['average_precision']-retained['ap'])<1e-5
            run.json(name+'_evaluation.json',rows[name]);print('scored',name,flush=True)
        macro={t:{a:{m:float(np.mean([r[a][t][m] for r in rows.values()]))
                      for m in ('iou','precision','recall','average_precision')} for a in config['arms']}
               for t in config['gt_tracks']}
        other={}
        for job in jobs:
            if job['benchmark']=='nvos':continue
            p=SYSTEM/job['benchmark']/job['name']/'result.json'
            other[job['benchmark']+'/'+job['name']]=dict(path=str(p),sha256=sha(p),intervention=False)
        run.json('summary.json',dict(complete=True,rows=rows,macro=macro,gt_hashes=gt_hashes,
            other_benchmarks=other,other_benchmarks_freshly_evaluated=False,
            prediction_seal_sha256=sha(run.root/'predictions_complete.json'),
            target_gt_used_for_prediction=False,target_rgb_used=False,default_changed=False,
            independent_test=False,per_task_winners=False,durable_dense_prediction_bytes=0))
        print(json.dumps(macro,indent=2),flush=True)
    except Exception as error:
        run.json('failure.json',dict(type=type(error).__name__,message=str(error),
            completed_predictions=len(reports),target_gt_opened=(run.root/'predictions_complete.json').exists()))
        raise
    finally:bank.close()


if __name__=='__main__':main()
