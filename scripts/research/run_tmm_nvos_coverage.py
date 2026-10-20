"""Fixed 2x2 NVOS evidence/resolution study; all predictions precede scoring."""
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
from torch.nn import functional as F
from gsplat import rasterization

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.querying.coverage_evidence import coverage_evidence
from radio_gs.querying.response_reconstruction import AdmittedResponseSystem
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from scripts.research.audit_tmm_protocol_correctness import REPO, SYSTEM, binary_counts, read, load
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.tmm_ram_predictions import RamPredictionBank

CONFIG='configs/research/nvos_coverage_evidence.json'


@torch.inference_mode()
def predict(job, run, bank, config):
    name=job['name'];started=time.monotonic()
    fit_path=SYSTEM/'nvos'/name/'query/reference/0/fit.json';fit=read(fit_path)
    scene=read(fit['scene']);spec=read(job['evaluation']);tracks=load(fit['tracks'])
    assert spec['target_frame'] not in scene['source_frames']
    assert fit['negative_policy']=='original' and not fit['explicit_evidence']
    data=load(SYSTEM/'nvos'/name/'query/typed_readout.pt');binding=data['semantic_geometry']
    geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
    gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256'])
    n=len(geo.xyz);reports={};arrays={};seals={}
    inputs=[fit_path,fit['scene'],fit['tracks'],job['queries'],job['evaluation'],binding['path'],scene['camera_report']]
    for scale in (1,2):
        h,w=[x*scale for x in scene['source_fit_hw']];radius=scene['boundary_radius']*scale
        terms={'legacy':[],'coverage':[]};observed=torch.zeros(n,dtype=torch.bool,device='cuda');statistics=[]
        for frame in scene['source_frames']:
            cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
            renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
            hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False)
            pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['weights'].float()
            observed[gid[weight>1e-8]]=True
            total=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight)
            record=tracks['frames'].get(frame)
            if record is None or record['object_logit']<=0:continue
            op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/total[pid].clamp_min(1e-8),(h*w,n)).coalesce()
            logits=record['logits'].float().cuda()
            target=F.interpolate(logits[None,None],size=(h,w),mode='bilinear',align_corners=False).sigmoid()
            binary=(target>.5).float()
            pos=(-F.max_pool2d(-binary,2*radius+1,1,radius)>.5).flatten()[:,None]
            neg=(F.max_pool2d(binary,2*radius+1,1,radius)<.5).flatten()[:,None]
            reliability=float(record.get('negative_reliability',1.))
            known=(pos.float()+neg.float()*reliability)*(total[:,None]>1e-8)
            # CPU sparse operators bound GPU memory without changing the objective.
            cpu_op=op.cpu()
            terms['legacy'].append((cpu_op,target.flatten()[:,None].cpu(),known.cpu(),1.))
            y,mass=coverage_evidence(logits,(h,w))
            # Retain any explicit negative reliability already attached to a track.
            mass=mass*(y+(1-y)*reliability)*(total[:,None]>1e-8)
            terms['coverage'].append((cpu_op,y.cpu(),mass.cpu(),1.))
            statistics.append(dict(frame=frame,legacy_positive_pixels=int(pos.sum()),
                hard_foreground_pixels=int(binary.sum()),confidence_positive_mass=float((y*mass).sum()),
                confidence_negative_mass=float(((1-y)*mass).sum()),sparse_hits=op._nnz()))
            del hit,logits
        for mode in ('legacy','coverage'):
            arm=mode+('_low' if scale==1 else '_high')
            system=AdmittedResponseSystem(torch.full((n,1),.5),terms[mode])
            system.terms=[(a.to_sparse_csr(),at.to_sparse_csr(),y,k)
                          for a,at,y,k in system.terms]
            value=system.rendered(steps=config['solver']['steps'],residual_clip=config['solver']['huber_clip'])
            value[(~observed.cpu())[:,None]|(system.mass<=1e-8)]=0
            row=dict(source_huber_objective=system.objective(value,.25),observed_elements=int(observed.sum()),
                     source_hw=[h,w],views=len(terms[mode]),source_evidence_statistics=statistics)
            if arm=='legacy_low':
                assert torch.equal(observed.cpu(),data['semantic_observed'])
                error=(value.cpu()-data['scores']).abs()
                row['compatibility']=dict(maximum=float(error.max()),mean=float(error.mean()))
                assert error.max()<1e-4 and error.mean()<1e-6
            value=value.cuda()
            # Target camera/size are public metadata; target RGB and GT remain unopened.
            cam=gp.cameras[spec['target_frame']];width,height=spec['raster_wh']
            k=cam.intrinsic.cuda().clone();k[0]*=width/cam.width;k[1]*=height/cam.height
            image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,
                opacities=geo.opacities.flatten(),colors=torch.cat([value,observed.float()[:,None]],1),
                viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=width,height=height,
                near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
            mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
            response=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0).cpu()
            seals[name+'_'+arm]=bank.add(run,name+'_'+arm,response.reshape(-1,1))
            reports[arm]=row
            del system,value,image,alpha,response
            print('predicted',name,arm,'sec',round(time.monotonic()-started),flush=True)
        del terms,op
        gc.collect();torch.cuda.empty_cache()
    return dict(job=job,arms=reports,predictions=seals,input_hashes={str(p):sha(p) for p in inputs},
                target_rgb_used=False,target_gt_read=False,seconds=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run');args=parser.parse_args()
    os.chdir(REPO);torch.set_num_threads(4);config=read(CONFIG)
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    run=BoundedExperiment(STUDY,args.run or config['run'],POLICY);run.preflight(config['maximum_new_durable_bytes'])
    bank=RamPredictionBank(limit=config['maximum_volatile_prediction_bytes'])
    jobs=read('configs/pipeline/system_v10.json')['jobs'];nvos=[j for j in jobs if j['benchmark']=='nvos']
    sources=[__file__,CONFIG,'radio_gs/querying/coverage_evidence.py','radio_gs/querying/response_reconstruction.py']
    run.json('manifest.json',dict(config=config,jobs=nvos,source_hashes={p:sha(p) for p in sources},
        volatile_prediction_directory=str(bank.root),other_benchmarks='unchanged v10 reference, no new inference'))
    run.json('source_snapshot.json',{p:dict(sha256=sha(p),text=Path(p).read_text()) for p in sources})
    reports=[]
    try:
        for job in nvos:
            reports.append(predict(job,run,bank,config));run.json(job['name']+'_prediction.json',reports[-1])
            gc.collect();torch.cuda.empty_cache()
        run.json('predictions_complete.json',dict(complete=True,jobs=8,arms=list(config['arms']),
            predictions={k:v for r in reports for k,v in r['predictions'].items()},target_gt_read=False))
        rows={}
        for job in nvos:
            name=job['name'];spec=read(job['evaluation']);gt=np.asarray(Image.open(spec['gt_mask']))
            truths={'retained_gt_gt127':gt>127,'ludvig_gt_maximum':(gt//gt.max()).astype(bool)}
            rows[name]={}
            for arm in config['arms']:
                response=bank.read(run,name+'_'+arm,1).reshape(spec['raster_wh'][1],spec['raster_wh'][0]).numpy()
                response=cv2.resize(response,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)
                rows[name][arm]={track:binary_counts(response>=.5,truth) for track,truth in truths.items()}
            old=read(SYSTEM/'nvos'/name/'result.json')['metrics']['iou']
            assert abs(rows[name]['legacy_low']['retained_gt_gt127']['iou']-old)<1e-5
            run.json(name+'_evaluation.json',rows[name])
        macro={track:{arm:float(np.mean([r[arm][track]['iou'] for r in rows.values()]))
                       for arm in config['arms']} for track in config['gt_tracks']}
        other={}
        for job in jobs:
            if job['benchmark']=='nvos':continue
            p=SYSTEM/job['benchmark']/job['name']/'result.json'
            other[job['benchmark']+'/'+job['name']]=dict(path=str(p),sha256=sha(p),intervention=False)
        summary=dict(complete=True,macro=macro,rows=rows,other_benchmarks=other,
            all_eight_completed=True,other_benchmarks_freshly_evaluated=False,default_changed=False,
            independent_test=False,target_rgb_used=False,durable_dense_prediction_bytes=0,
            prediction_seal_sha256=sha(run.root/'predictions_complete.json'))
        run.json('summary.json',summary);print(json.dumps(macro,indent=2),flush=True)
    except Exception as error:
        run.json('failure.json',dict(type=type(error).__name__,message=str(error),completed_predictions=len(reports),
                 target_gt_opened=(run.root/'predictions_complete.json').exists(),default_changed=False))
        raise
    finally:bank.close()


if __name__=='__main__':main()
