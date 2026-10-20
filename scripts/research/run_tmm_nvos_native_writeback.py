"""N08: exact sealed image evidence, resolution-consistent 3D writeback."""
import gc
import json
import os
from pathlib import Path
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
from scripts.research.run_tmm_support_agreement import array_sha
from scripts.research.run_tmm_nvos_image_response import camera_args,render_responses

CONFIG='configs/research/nvos_native_writeback.json'
PROVIDERS=['static_image','pair_memory','sequence_memory']


def collect_parent(parent,manifest,temp,limit):
    source=Path(manifest['volatile_prediction_directory']);links={}
    expected={j['name']+'_'+a for j in manifest['jobs'] for a in manifest['config']['arms']};start=time.monotonic()
    while len(links)<len(expected) or not (parent/'predictions_complete.json').exists():
        if (source/'manifest.json').exists():
            for key,entry in read(source/'manifest.json').items():
                name=key.rsplit('::',1)[1]
                if name not in expected or name in links:continue
                if sum(p.stat().st_size for p in temp.iterdir())+entry['bytes']+4096>limit:raise ValueError('parent RAM budget')
                target=temp/entry['file'];os.link(source/entry['file'],target);links[name]=dict(entry,path=str(target))
        if (parent/'failure.json').exists() or time.monotonic()-start>7200:raise RuntimeError('parent failed/timed out before N08 GT')
        time.sleep(.1)
    seal=read(parent/'predictions_complete.json');assert not seal['target_gt_read'] and seal['jobs']==8 and seal['images']==56
    for j in manifest['jobs']:
        for a in manifest['config']['arms']:
            e=links[j['name']+'_'+a];x=np.load(e['path'],mmap_mode='r',allow_pickle=False)
            assert array_sha(x)==e['sha256']==seal['predictions'][j['name']][a]['sha256']
    return links,seal


@torch.inference_mode()
def predict(job,run,bank,config,links,parent_report):
    start=time.monotonic();name=job['name'];scene=read(job['scene']);spec=read(job['evaluation'])
    data=load(SYSTEM/'nvos'/name/'query/typed_readout.pt');old=data['scores'].float();observed=data['semantic_observed'].bool();binding=data['semantic_geometry']
    geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
    gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);target=spec['target_frame']
    assert target not in scene['source_frames'] and tuple(spec['raster_wh'])==tuple(config['native_wh'])
    w,h=spec['raster_wh'];args=camera_args(geo,gp,target,(w,h))
    values={a:torch.from_numpy(np.load(links[name+'_'+a]['path'],allow_pickle=False)).reshape(h,w) for a in ['v10',*[p+'_lift' for p in PROVIDERS]]}
    fresh=render_responses(args,[old],observed)[...,0];error=float((fresh-values['v10']).abs().max())
    assert error<1e-4 and torch.equal(fresh>=.5,values['v10']>=.5)
    renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:args['Ks'][0],near_plane=args['near_plane'],far_plane=args['far_plane'])
    hit=rasterize_single_view_contributions(geo,renderer,args['viewmats'][0],height=h,width=w,include_depths=False,footprint_backend=config['footprint_backend'])
    pid,gid,weight=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['weights'].float()
    alpha_error=float((hit['accumulated_alpha']-hit['rendered_alpha']).abs().max());assert alpha_error<1e-4
    weight=weight*observed.cuda()[gid];mass=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight)
    valid=(mass/hit['rendered_alpha'].flatten().clamp_min(1e-8)>.5)&(mass>1e-8)
    op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,len(old))).coalesce().cpu()
    projected=torch.sparse.mm(op,old).reshape(h,w);projected[~valid.cpu().reshape(h,w)]=0
    op_error=float((projected-fresh).abs().max());assert op_error<1e-4
    valid=valid.cpu()[:,None];del hit,pid,gid,weight,mass;torch.cuda.empty_cache()
    fields={};lifting={}
    for provider in PROVIDERS:
        arm=provider+'_native';y=torch.from_numpy(np.load(links[name+'_'+provider]['path'],allow_pickle=False)).reshape(-1,1)
        assert torch.isfinite(y).all() and y.min()>=0 and y.max()<=1
        c=(2*y-1).abs()*valid
        fields[arm],lifting[arm]=lift_image_response(old,op,y,c,config['solver_steps'])
        assert lifting[arm]['unsupported_changed']==0 and not torch.count_nonzero(fields[arm][~observed])
        values[arm]=render_responses(args,[fields[arm]],observed)[...,0]
    del op
    source=[];sh,sw=config['source_hw'];field_arms=['v10',*fields];tracks=load(job['reference_tracks'])['frames']
    for f in scene['source_frames']:
        record=tracks.get(f)
        if record is None or record['object_logit']<=0:continue
        y,c=coverage_evidence(record['logits'].float(),(sh,sw))
        image=render_responses(camera_args(geo,gp,f,(sw,sh)),[old,*fields.values()],observed).reshape(-1,len(field_arms))
        e=(image-y).abs();loss=torch.where(e<=.25,.5*e.square(),.25*(e-.125))
        source.append(dict(frame=f,confidence_weighted_mean_huber={a:float((loss[:,i:i+1]*c).sum()/c.sum().clamp_min(1e-8)) for i,a in enumerate(field_arms)}))
    seals={a:bank.add(run,name+'_'+a,values[a].reshape(-1,1)) for a in config['arms']}
    report=dict(job=job,predictions=seals,parent_report_sha256=sha(parent_report),lifting=lifting,source_teacher_checks=source,
        target_gt_read=False,seconds=time.monotonic()-start,implementation_checks=dict(v10_binary_exact=True,v10_render_max_error=error,
        compositor_alpha_max_error=alpha_error,normalized_operator_max_error=op_error,native_grid_wh=[w,h],probability_evidence_exact=True))
    run.json(name+'_prediction.json',report);print('sealed',name,round(report['seconds'],1),'seconds',flush=True);return report


def main():
    os.chdir(REPO);config=read(CONFIG);parent=STUDY/config['parent_run'];pm=read(parent/'manifest.json');jobs=pm['jobs'];torch.set_num_threads(4)
    gt={str(Path(read(j['evaluation'])['gt_mask']).resolve()) for j in jobs};access=dict(gt=False)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and not access['gt'] and str(Path(os.fsdecode(args[0])).resolve()) in gt:raise RuntimeError('N08 GT before all predictions')
    sys.addaudithook(guard)
    run=BoundedExperiment(STUDY,config['run'],POLICY);run.preflight(config['maximum_new_durable_bytes'])
    bank=RamPredictionBank(limit=config['maximum_volatile_prediction_bytes']);temp=Path(tempfile.mkdtemp(prefix='radiogs_native_parent_',dir='/dev/shm'))
    sources=[__file__,CONFIG,'docs/adr/0031-native-grid-response-writeback.md','radio_gs/querying/image_conditioned_response.py',
        'radio_gs/querying/coverage_evidence.py','radio_gs/querying/response_reconstruction.py','scripts/research/run_tmm_nvos_image_response.py',
        'radio_gs/rendering/contribution_compositor.py','radio_gs/rendering/gsplat_footprint_cuda.py',
        '/root/miniconda3/envs/cybersim_agent/lib/python3.9/site-packages/gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu',
        '/root/miniconda3/envs/cybersim_agent/lib/python3.9/site-packages/gsplat/cuda/csrc/rasterize_to_indices_in_range.cu',
        'scripts/research/tmm_ram_predictions.py','scripts/research/audit_tmm_protocol_correctness.py']
    run.json('manifest.json',dict(config=config,jobs=jobs,parent_manifest_sha256=sha(parent/'manifest.json'),source_hashes={p:sha(p) for p in sources},
        volatile_prediction_directory=str(bank.root),parent_link_directory=str(temp),torch_version=torch.__version__))
    run.json('source_snapshot.json',{p:dict(text=Path(p).read_text(),sha256=sha(p)) for p in sources})
    reports=[]
    try:
        links,parent_seal=collect_parent(parent,pm,temp,config['maximum_volatile_prediction_bytes'])
        run.json('parent_binding.json',dict(parent_seal_sha256=sha(parent/'predictions_complete.json'),images=56,target_gt_read=False,
            exact_array_digests={k:v['sha256'] for k,v in links.items()}))
        print('parent fully sealed; native prediction begins',flush=True)
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
        for job in jobs:
            rp=parent/(job['name']+'_prediction.json');assert sha(rp)==parent_seal['report_hashes'][job['name']]
            for p,hsh in read(rp)['input_hashes'].items():assert sha(p)==hsh
            reports.append(predict(job,run,bank,config,links,rp));gc.collect();torch.cuda.empty_cache()
        run.json('predictions_complete.json',dict(jobs=8,images=56,target_gt_read=False,threshold=.5,
            report_hashes={r['job']['name']:sha(run.root/(r['job']['name']+'_prediction.json')) for r in reports},predictions={r['job']['name']:r['predictions'] for r in reports}))
        access['gt']=True;rows={};gt_hashes={};previous=read(parent/'summary.json')
        for job in jobs:
            name=job['name'];spec=read(job['evaluation']);truth=np.asarray(Image.open(spec['gt_mask']));assert truth.ndim==2 and truth.max()>0
            truths=dict(retained_gt_gt127=truth>127,ludvig_gt_maximum=(truth//truth.max()).astype(bool));gt_hashes[spec['gt_mask']]=sha(spec['gt_mask']);rows[name]={}
            for arm in config['arms']:
                x=bank.read(run,name+'_'+arm,1).reshape(spec['raster_wh'][1],spec['raster_wh'][0]).numpy();x=cv2.resize(x,(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR)
                rows[name][arm]={}
                for track,g in truths.items():
                    b=x>=.5;r=binary_counts(b,g);tn,fp,fn,tp=np.bincount((2*g.astype(np.uint8)+b.astype(np.uint8)).ravel(),minlength=4).tolist()
                    assert (tp,fp,fn)==(r['tp'],r['fp'],r['fn']);value=dict(**r,tn=tn,accuracy=(tp+tn)/g.size,independent_confusion_exact=True)
                    rows[name][arm][track]=value
                    if arm in previous['rows'][name]:assert value==previous['rows'][name][arm][track]
            run.json(name+'_evaluation.json',rows[name])
        macro={t:{a:{m:float(np.mean([r[a][t][m] for r in rows.values()])) for m in config['metrics']} for a in config['arms']} for t in config['gt_tracks']}
        run.json('summary.json',dict(complete=True,rows=rows,macro=macro,gt_hashes=gt_hashes,primary_arm=config['primary_arm'],prediction_seal_sha256=sha(run.root/'predictions_complete.json'),
            parent_summary_sha256=sha(parent/'summary.json'),parent_controls_exact=True,target_rgb_used=True,target_gt_used_for_prediction=False,default_changed=False,
            per_task_winners=False,independent_test=False,average_precision_measured=False,durable_dense_prediction_bytes=0,other_benchmarks=previous['other_benchmarks'],other_benchmarks_freshly_evaluated=False))
        print(json.dumps(macro,indent=2),flush=True)
    except Exception as error:
        run.json('failure.json',dict(type=type(error).__name__,message=str(error),completed_predictions=len(reports),target_gt_opened=access['gt']));raise
    finally:
        bank.close()
        for p in temp.iterdir():p.unlink()
        temp.rmdir()


if __name__=='__main__':main()
