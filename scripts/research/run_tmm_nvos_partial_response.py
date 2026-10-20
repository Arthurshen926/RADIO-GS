"""N05: seven fixed response-permission arms, all-eight IoU with two GT protocols."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.querying.coverage_evidence import coverage_evidence
from radio_gs.querying.image_conditioned_response import lift_image_response
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from scripts.research.audit_tmm_protocol_correctness import REPO,SYSTEM,binary_counts,read,load
from scripts.research.replay_tmm_response import STUDY,POLICY,sha
from scripts.research.tmm_ram_predictions import RamPredictionBank
from scripts.research.run_tmm_nvos_image_response import camera_args,render_responses,MANIFEST,DECODER_SOURCES

CONFIG='configs/research/nvos_partial_image_response.json'
WORKER_SOURCES=[p for p in DECODER_SOURCES if 'decode_tmm_nvos_image_response.py' not in p]+[
    'scripts/research/decode_tmm_nvos_partial_response.py','radio_gs/querying/partial_response_prompt.py']
BRANCHES=['signed_core','positive_core','sparse_core']


@torch.inference_mode()
def predict(job,run,bank,config,dataset,access):
    start=time.monotonic();name=job['name'];scene=read(job['scene']);spec=read(job['evaluation'])
    path=SYSTEM/'nvos'/name/'query/typed_readout.pt';data=load(path)
    old,observed=data['scores'].float(),data['semantic_observed'].bool();binding=data['semantic_geometry']
    geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
    gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);target=spec['target_frame']
    assert target not in scene['source_frames'] and target not in scene['tracking_frames']
    fit_path=Path(binding['path']).parent/'fit.json';fit=read(fit_path)
    assert target not in fit['source_frames'] and target not in fit['validation_frames']
    args=camera_args(geo,gp,target,spec['raster_wh'])
    response=render_responses(args,[old],observed)[...,0]
    old_path=SYSTEM/'nvos'/name/'evaluation/response.pt';retained=load(old_path)['response']
    error=float((response-retained).abs().max());assert error<1e-4 and torch.equal(response>=.5,retained>=.5)
    ds=next(s for s in dataset['scenes'] if s['scene_id']==name)
    frame=next(f for f in ds['frames'] if int(f['canonical_index'])==target)
    assert frame['frame_id'] in ds['evaluation_frame_ids'] and Path(frame['gt_mask_path']).resolve()==Path(spec['gt_mask']).resolve()
    values={'v10':response};logits={}
    with tempfile.TemporaryDirectory(prefix='radiogs_partial_',dir='/dev/shm') as tmp:
        root=Path(tmp);np.save(root/'prior.npy',response.numpy(),allow_pickle=False)
        access['rgb']=True
        Image.open(frame['rgb_path']).convert('RGB').resize(tuple(spec['raster_wh']),Image.Resampling.BILINEAR).save(root/'rgb.png')
        access['rgb']=False
        (root/'request.json').write_text(json.dumps(dict(checkpoint=config['image_checkpoint'],
            gpu_memory_fraction=config['gpu_memory_fraction'],source_paths=WORKER_SOURCES,point_budget=config['point_budget'])))
        gc.collect();torch.cuda.empty_cache()
        subprocess.run(['/root/miniconda3/envs/sam3/bin/python','scripts/research/decode_tmm_nvos_partial_response.py','--request',str(root)],check=True)
        worker=read(root/'result.json');registered=read(run.root/'manifest.json')['source_hashes']
        assert not worker['target_gt_available'] and all(worker['source_hashes'][p]==registered[p]==sha(p) for p in WORKER_SOURCES)
        for a in BRANCHES:
            logits[a]=torch.from_numpy(np.load(root/(a+'.npy'),allow_pickle=False))
            values[a]=response.clone() if worker['decoder'][a]['fallback_empty_prompt'] else logits[a].sigmoid()
        assert sum(p.stat().st_size for p in root.iterdir())<32*2**20
    h,w=config['source_hw'];low_args=camera_args(geo,gp,target,(w,h))
    renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:low_args['Ks'][0],near_plane=low_args['near_plane'],far_plane=low_args['far_plane'])
    hit=rasterize_single_view_contributions(geo,renderer,low_args['viewmats'][0],height=h,width=w,include_depths=False)
    pid,gid,weight=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['weights'].float()
    alpha_error=float((hit['accumulated_alpha']-hit['rendered_alpha']).abs().max());assert alpha_error<1e-4
    weight=weight*observed.cuda()[gid];mass=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight)
    valid=(mass/hit['rendered_alpha'].flatten().clamp_min(1e-8)>.5)&(mass>1e-8)
    op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,len(old))).coalesce().cpu()
    projected=torch.sparse.mm(op,old).reshape(h,w);projected[~valid.cpu().reshape(h,w)]=0
    op_error=float((projected-render_responses(low_args,[old],observed)[...,0]).abs().max());assert op_error<1e-4
    fields={};lifting={}
    for a in BRANCHES:
        y,c=coverage_evidence(logits[a],(h,w));c*=valid.cpu()[:,None]
        fields[a+'_lift'],lifting[a+'_lift']=lift_image_response(old,op,y,c,config['solver_steps'])
        assert lifting[a+'_lift']['unsupported_changed']==0 and not torch.count_nonzero(fields[a+'_lift'][~observed])
        values[a+'_lift']=render_responses(args,[fields[a+'_lift']],observed)[...,0]
    del hit,op,pid,gid,weight
    tracks=load(job['reference_tracks'])['frames'];source=[];field_arms=['v10',*fields]
    for f in scene['source_frames']:
        r=tracks.get(f)
        if r is None or r['object_logit']<=0:continue
        y,c=coverage_evidence(r['logits'].float(),(h,w))
        image=render_responses(camera_args(geo,gp,f,(w,h)),[old,*fields.values()],observed).reshape(-1,len(field_arms))
        e=(image-y).abs();loss=torch.where(e<=.25,.5*e.square(),.25*(e-.125))
        source.append(dict(frame=f,confidence_weighted_mean_huber={a:float((loss[:,i:i+1]*c).sum()/c.sum().clamp_min(1e-8)) for i,a in enumerate(field_arms)}))
    seals={a:bank.add(run,name+'_'+a,values[a].reshape(-1,1)) for a in config['arms']}
    paths=[job['scene'],job['queries'],job['evaluation'],path,old_path,fit_path,binding['path'],scene['camera_report'],job['reference_tracks'],frame['rgb_path']]
    access['rgb']=True;hashes={str(p):sha(p) for p in paths};access['rgb']=False
    report=dict(job=job,predictions=seals,input_hashes=hashes,decoder=worker,lifting=lifting,source_teacher_checks=source,
        target_rgb_used=True,target_gt_read=False,seconds=time.monotonic()-start,
        implementation_checks=dict(v10_render_max_error=error,v10_binary_exact=True,compositor_alpha_max_error=alpha_error,normalized_operator_max_error=op_error))
    run.json(name+'_prediction.json',report);print('sealed',name,'seven arms',round(report['seconds'],1),'seconds',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run');a=p.parse_args()
    os.chdir(REPO);torch.set_num_threads(4);config=read(CONFIG);torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    dataset=read(MANIFEST);jobs=read('configs/pipeline/system_v10.json')['jobs'];nvos=[j for j in jobs if j['benchmark']=='nvos']
    gt={str(Path(read(j['evaluation'])['gt_mask']).resolve()) for j in nvos}
    images={str(Path(f['rgb_path']).resolve()) for s in dataset['scenes'] for f in s['frames'] if f['frame_id'] in s['evaluation_frame_ids']}
    access=dict(gt=False,rgb=False)
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=str(Path(os.fsdecode(args[0])).resolve())
            if path in gt and not access['gt']:raise RuntimeError('GT before complete seal')
            if path in images and not access['rgb']:raise RuntimeError('undeclared target RGB access')
    sys.addaudithook(audit)
    run=BoundedExperiment(STUDY,a.run or config['run'],POLICY);run.preflight(config['maximum_new_durable_bytes'])
    bank=RamPredictionBank(limit=config['maximum_volatile_prediction_bytes'])
    sources=list(dict.fromkeys([__file__,CONFIG,*WORKER_SOURCES,'scripts/research/run_tmm_nvos_image_response.py',
        'radio_gs/querying/coverage_evidence.py','radio_gs/querying/response_reconstruction.py',
        'radio_gs/rendering/contribution_compositor.py','scripts/research/tmm_ram_predictions.py',
        'scripts/research/audit_tmm_protocol_correctness.py','docs/adr/0028-partial-image-response-permissions.md']))
    run.json('manifest.json',dict(config=config,jobs=nvos,source_hashes={str(p):sha(p) for p in sources},
        checkpoint_sha256=sha(config['image_checkpoint']),dataset_manifest_sha256=sha(MANIFEST),
        volatile_prediction_directory=str(bank.root),torch_version=torch.__version__))
    run.json('source_snapshot.json',{str(p):dict(sha256=sha(p),text=Path(p).read_text()) for p in sources})
    reports=[]
    try:
        for job in nvos:
            reports.append(predict(job,run,bank,config,dataset,access));gc.collect();torch.cuda.empty_cache()
        run.json('predictions_complete.json',dict(jobs=8,images=56,target_gt_read=False,threshold=.5,
            report_hashes={r['job']['name']:sha(run.root/(r['job']['name']+'_prediction.json')) for r in reports},
            predictions={r['job']['name']:r['predictions'] for r in reports}))
        access['gt']=True;rows={};gt_hashes={};print('all56 sealed; opening GT',flush=True)
        for job in nvos:
            name=job['name'];spec=read(job['evaluation']);truth=np.asarray(Image.open(spec['gt_mask']))
            assert truth.ndim==2 and truth.max()>0
            truths=dict(retained_gt_gt127=truth>127,ludvig_gt_maximum=(truth//truth.max()).astype(bool));gt_hashes[spec['gt_mask']]=sha(spec['gt_mask'])
            rows[name]={}
            for arm in config['arms']:
                x=bank.read(run,name+'_'+arm,1).reshape(spec['raster_wh'][1],spec['raster_wh'][0]).numpy()
                x=cv2.resize(x,(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR);assert np.isfinite(x).all() and x.min()>=0 and x.max()<=1+1e-6
                rows[name][arm]={}
                for track,g in truths.items():
                    b=x>=.5;r=binary_counts(b,g)
                    tn,fp,fn,tp=np.bincount((2*g.astype(np.uint8)+b.astype(np.uint8)).ravel(),minlength=4).tolist()
                    assert (tp,fp,fn)==(r['tp'],r['fp'],r['fn'])
                    rows[name][arm][track]=dict(**r,tn=tn,accuracy=(tp+tn)/g.size,independent_confusion_exact=True)
            assert abs(rows[name]['v10']['retained_gt_gt127']['iou']-read(SYSTEM/'nvos'/name/'result.json')['metrics']['iou'])<1e-7
            run.json(name+'_evaluation.json',rows[name]);print('scored',name,flush=True)
        macro={t:{a:{m:float(np.mean([r[a][t][m] for r in rows.values()])) for m in config['metrics']} for a in config['arms']} for t in config['gt_tracks']}
        other={}
        for job in jobs:
            if job['benchmark']=='nvos':continue
            path=SYSTEM/job['benchmark']/job['name']/'result.json';other[job['benchmark']+'/'+job['name']]=dict(path=str(path),sha256=sha(path),intervention=False)
        run.json('summary.json',dict(complete=True,rows=rows,macro=macro,gt_hashes=gt_hashes,primary_arm=config['primary_arm'],
            prediction_seal_sha256=sha(run.root/'predictions_complete.json'),target_rgb_used=True,target_gt_used_for_prediction=False,
            other_benchmarks=other,other_benchmarks_freshly_evaluated=False,default_changed=False,independent_test=False,
            per_task_winners=False,average_precision_measured=False,durable_dense_prediction_bytes=0))
        print(json.dumps(macro,indent=2),flush=True)
    except Exception as error:
        run.json('failure.json',dict(type=type(error).__name__,message=str(error),completed_predictions=len(reports),target_gt_opened=access['gt']));raise
    finally:bank.close()


if __name__=='__main__':main()
