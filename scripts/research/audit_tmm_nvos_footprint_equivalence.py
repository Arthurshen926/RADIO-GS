"""All-eight matched coarse-grid control for the CUDA footprint correction."""
import gc,json,os,tempfile,hashlib
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.coverage_evidence import coverage_evidence
from radio_gs.querying.image_conditioned_response import lift_image_response
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from scripts.research.run_tmm_nvos_native_writeback import collect_parent,PROVIDERS
from scripts.research.run_tmm_nvos_image_response import camera_args,render_responses
from scripts.research.audit_tmm_protocol_correctness import REPO,read,load,SYSTEM,binary_counts
from scripts.research.replay_tmm_response import STUDY,POLICY,sha
from scripts.research.tmm_ram_predictions import RamPredictionBank


@torch.inference_mode()
def main():
    os.chdir(REPO);torch.set_num_threads(4);parent=STUDY/'nvos_memory_response_v2';pm=read(parent/'manifest.json')
    run=BoundedExperiment(STUDY,'nvos_coarse_backend_v1',POLICY);run.preflight(512*1024)
    tmp=Path(tempfile.mkdtemp(prefix='radiogs_coarse_control_',dir='/dev/shm'));bank=RamPredictionBank(limit=128*2**20)
    sources=[__file__,'radio_gs/rendering/gsplat_footprint_cuda.py','radio_gs/rendering/contribution_compositor.py',
        'scripts/research/run_tmm_nvos_native_writeback.py','radio_gs/querying/image_conditioned_response.py',
        'radio_gs/querying/response_reconstruction.py','radio_gs/querying/coverage_evidence.py']
    run.json('manifest.json',dict(parent_manifest_sha256=sha(parent/'manifest.json'),source_hashes={p:sha(p) for p in sources},
        volatile_prediction_directory=str(bank.root),parent_link_directory=str(tmp),diagnostic_only=True,gt_selection=False))
    run.json('source_snapshot.json',{p:dict(text=Path(p).read_text(),sha256=sha(p)) for p in sources})
    reports={}
    try:
        links,pseal=collect_parent(parent,pm,tmp,256*2**20);torch.cuda.set_per_process_memory_fraction(.25)
        for j in pm['jobs']:
            n=j['name'];scene=read(j['scene']);spec=read(j['evaluation']);d=load(SYSTEM/'nvos'/n/'query/typed_readout.pt');old=d['scores'].float();obs=d['semantic_observed'].bool();binding=d['semantic_geometry']
            geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);target=spec['target_frame'];h,w=384,512
            args=camera_args(geo,gp,target,(w,h));renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:args['Ks'][0],near_plane=args['near_plane'],far_plane=args['far_plane'])
            hit=rasterize_single_view_contributions(geo,renderer,args['viewmats'][0],height=h,width=w,include_depths=False,footprint_backend='gsplat_cuda')
            ae=float((hit['accumulated_alpha']-hit['rendered_alpha']).abs().max());assert ae<1e-4
            pid,gid,weight=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['weights'].float();weight*=obs.cuda()[gid]
            mass=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight);valid=(mass/hit['rendered_alpha'].flatten().clamp_min(1e-8)>.5)&(mass>1e-8)
            op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,len(old))).coalesce().cpu();valid=valid.cpu()[:,None]
            reports[n]=dict(alpha_max_error=ae,predictions={},native_binary_changed={},maximum_absolute_error={})
            for a in PROVIDERS:
                prob=torch.from_numpy(np.load(links[n+'_'+a]['path'],allow_pickle=False)).reshape(756,1008)
                # Avoid logit inversion at saturated probabilities. This is exactly
                # N07 confidence-weighted area pooling, applied to its sealed p.
                p=prob[None,None];c=(2*p-1).abs();cm=torch.nn.functional.interpolate(c,size=(h,w),mode='area')
                y=torch.nn.functional.interpolate(c*p,size=(h,w),mode='area');y=torch.where(cm>1e-8,y/cm.clamp_min(1e-8),.5).flatten()[:,None]
                field,_=lift_image_response(old,op,y,cm.flatten()[:,None]*valid,120)
                x=render_responses(camera_args(geo,gp,target,spec['raster_wh']),[field],obs)[...,0]
                original=torch.from_numpy(np.load(links[n+'_'+a+'_lift']['path'],allow_pickle=False)).reshape(756,1008)
                reports[n]['maximum_absolute_error'][a]=float((x-original).abs().max());reports[n]['native_binary_changed'][a]=int(((x>=.5)!=(original>=.5)).sum())
                reports[n]['predictions'][a]=bank.add(run,n+'_'+a,x.reshape(-1,1))
            del geo,hit,op,pid,gid,weight;gc.collect();torch.cuda.empty_cache();print('sealed',n,flush=True)
        run.json('predictions_complete.json',dict(jobs=8,images=24,target_gt_read=False,reports=reports,parent_seal_sha256=sha(parent/'predictions_complete.json')))
        parent_summary=read(parent/'summary.json');rows={}
        for j in pm['jobs']:
            n=j['name'];spec=read(j['evaluation']);gt=np.asarray(Image.open(spec['gt_mask']));rows[n]={}
            for a in PROVIDERS:
                x=bank.read(run,n+'_'+a,1).reshape(756,1008).numpy();old=np.load(links[n+'_'+a+'_lift']['path'],allow_pickle=False).reshape(756,1008)
                x=cv2.resize(x,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR);old=cv2.resize(old,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)
                row=dict(scoring_binary_changed=int(((x>=.5)!=(old>=.5)).sum()),metrics={})
                for t,g in dict(retained_gt_gt127=gt>127,ludvig_gt_maximum=(gt//gt.max()).astype(bool)).items():
                    counts=binary_counts(x>=.5,g);before=binary_counts(old>=.5,g);assert all(before[k]==parent_summary['rows'][n][a+'_lift'][t][k] for k in before)
                    row['metrics'][t]=dict(**counts,original_iou=before['iou'],delta_pp=100*(counts['iou']-before['iou']))
                rows[n][a]=row
        run.json('summary.json',dict(complete=True,rows=rows,implementation=reports,diagnostic_only=True,gt_selection=False,
            prediction_seal_sha256=sha(run.root/'predictions_complete.json'),parent_summary_sha256=sha(parent/'summary.json'),
            all_scoring_masks_identical=all(v['scoring_binary_changed']==0 for r in rows.values() for v in r.values()),
            maximum_cohort_iou_delta_pp=max(abs(np.mean([r[a]['metrics'][t]['delta_pp'] for r in rows.values()])) for a in PROVIDERS for t in ['retained_gt_gt127','ludvig_gt_maximum']),durable_dense_prediction_bytes=0))
    except Exception as e:run.json('failure.json',dict(error=str(e),type=type(e).__name__,completed_tasks=len(reports)));raise
    finally:
        bank.close()
        for p in tmp.iterdir():p.unlink()
        tmp.rmdir()


if __name__=='__main__':main()
