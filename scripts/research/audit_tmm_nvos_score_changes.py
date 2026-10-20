"""Post-seal score-space attribution; GT hybrids are diagnostics, never outputs."""
import gc
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image
import torch
from gsplat import rasterization

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from scripts.research.audit_tmm_protocol_correctness import REPO, SYSTEM, binary_counts, read, load
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.tmm_ram_predictions import RamPredictionBank
from scripts.research.run_tmm_support_agreement import array_sha


@torch.inference_mode()
def predict_old(job,run,bank):
    path=SYSTEM/'nvos'/job['name']/'query/typed_readout.pt';d=load(path)
    scene=read(job['scene']);spec=read(job['evaluation']);binding=d['semantic_geometry']
    assert spec['target_frame'] not in scene['source_frames']
    geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
    gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256'])
    cam=gp.cameras[spec['target_frame']];w,h=spec['raster_wh']
    k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
    image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,
        opacities=geo.opacities.flatten(),colors=torch.cat([d['scores'].cuda(),d['semantic_observed'].float().cuda()[:,None]],1),
        viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,
        near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
    mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
    response=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0).cpu()
    inputs=[path,job['scene'],job['evaluation'],binding['path'],scene['camera_report']]
    return dict(prediction=bank.add(run,job['name'],response.reshape(-1,1)),
                input_hashes={str(p):sha(p) for p in inputs},target_gt_read=False,target_rgb_used=False)


def main():
    os.chdir(REPO);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.14)
    parent=STUDY/'nvos_source_readout_v1';pm=read(parent/'manifest.json')
    source_bank=Path(pm['volatile_prediction_directory'])
    run=BoundedExperiment(STUDY,'nvos_score_changes_v1',POLICY);run.preflight(4*2**20)
    bank=RamPredictionBank(limit=64*2**20)
    temp=Path(tempfile.mkdtemp(prefix='radiogs_nvos_attribution_',dir='/dev/shm'))
    sources=[__file__,'radio_gs/training/retrieval_metrics.py','scripts/research/tmm_ram_predictions.py']
    run.json('manifest.json',dict(source_hashes={p:sha(p) for p in sources},parent='nvos_source_readout_v1',
        parent_manifest_sha256=sha(parent/'manifest.json'),prediction_directory=str(bank.root),
        linked_prediction_directory=str(temp),linked_prediction_limit_bytes=64*2**20,
        target_gt_use='post-seal diagnostics only; never for response selection or threshold fitting'))
    run.json('source_snapshot.json',{p:dict(text=Path(p).read_text(),sha256=sha(p)) for p in sources})
    jobs=[j for j in read('configs/pipeline/system_v10.json')['jobs'] if j['benchmark']=='nvos']
    reports={};links={}
    try:
        for job in jobs:
            reports[job['name']]=predict_old(job,run,bank);gc.collect();torch.cuda.empty_cache()
        run.json('old_predictions_complete.json',dict(jobs=8,rows=reports,target_gt_read=False))
        print('all eight original v10 responses sealed; collecting exact N02 responses',flush=True)
        started=time.monotonic()
        while len(links)<8 or not (parent/'summary.json').exists():
            if (source_bank/'manifest.json').exists():
                for key,entry in read(source_bank/'manifest.json').items():
                    name=key.rsplit('::',1)[1]
                    if name in links:continue
                    target=temp/entry['file']
                    if sum(p.stat().st_size for p in temp.iterdir())+entry['bytes']+4096>64*2**20:
                        raise ValueError('attribution link budget exceeded')
                    os.link(source_bank/entry['file'],target);links[name]=dict(entry,path=str(target))
            if (parent/'failure.json').exists() or time.monotonic()-started>7200:
                raise RuntimeError('N02 failed or timed out; no attribution GT opened')
            time.sleep(.02)
        source_seal=read(parent/'predictions_complete.json');parent_summary=read(parent/'summary.json')
        for name,entry in links.items():
            array=np.load(entry['path'],mmap_mode='r',allow_pickle=False)
            assert array_sha(array)==source_seal['predictions'][name]['sha256']==entry['sha256']
        run.json('joint_prediction_seal.json',dict(old_seal_sha256=sha(run.root/'old_predictions_complete.json'),
            parent_seal_sha256=sha(parent/'predictions_complete.json'),parent_summary_sha256=sha(parent/'summary.json'),
            exact_new_response_digests={n:r['sha256'] for n,r in links.items()},target_gt_read=False))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(8,4,figsize=(12,17))
        rows={};gt_hashes={}
        for row_index,job in enumerate(jobs):
            name=job['name'];spec=read(job['evaluation']);gt=np.asarray(Image.open(spec['gt_mask']))>127
            gt_hashes[spec['gt_mask']]=sha(spec['gt_mask']);h,w=spec['raster_wh'][1],spec['raster_wh'][0]
            old=bank.read(run,name,1).reshape(h,w).numpy()
            new=np.load(links[name]['path'],allow_pickle=False).reshape(h,w)
            old,new=[cv2.resize(a,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR) for a in (old,new)]
            truth=torch.from_numpy(gt).flatten()
            def ap(value):return binary_ranking_metrics(torch.from_numpy(value).flatten(),truth)['average_precision']
            old_ap=ap(old);new_ap=ap(new)
            positive_only=ap(np.where(gt,new,old));negative_only=ap(np.where(gt,old,new))
            ap_foreground=.5*((positive_only-old_ap)+(new_ap-negative_only))
            ap_background=.5*((negative_only-old_ap)+(new_ap-positive_only))
            old_binary=old>=.5;new_binary=new>=.5
            threshold=source_seal['thresholds'][name]['source_holdout'];calibrated=new>=threshold
            counts={a:binary_counts(b,gt) for a,b in [('v10',old_binary),('coverage',new_binary),('calibrated',calibrated)]}
            retained=read(SYSTEM/'nvos'/name/'result.json')['metrics']
            assert abs(counts['v10']['iou']-retained['iou'])<1e-5 and abs(old_ap-retained['ap'])<1e-5
            for arm,key in [('coverage','fixed_half'),('calibrated','source_holdout')]:
                assert counts[arm]=={k:parent_summary['rows'][name][key]['retained_gt_gt127'][k] for k in counts[arm]}
            assert abs(new_ap-parent_summary['rows'][name]['fixed_half']['retained_gt_gt127']['average_precision'])<1e-12
            transitions={a:dict(recovered_foreground=int((~old_binary&b&gt).sum()),
                lost_foreground=int((old_binary&~b&gt).sum()),removed_false_positive=int((old_binary&~b&~gt).sum()),
                added_false_positive=int((~old_binary&b&~gt).sum())) for a,b in [('coverage',new_binary),('calibrated',calibrated)]}
            bins=np.array([0,.01,.05,.1,.25,.5,.75,.9,1.00001]);histograms={}
            for label,mask in [('foreground',gt),('background',~gt)]:
                histograms[label]={a:np.histogram(value[mask],bins)[0].tolist() for a,value in [('v10',old),('coverage',new)]}
            row=dict(metrics=counts,transitions=transitions,score_histogram_edges=bins.tolist(),score_histograms=histograms,
                average_precision=dict(v10=old_ap,coverage=new_ap,gt_hybrid_positive_only=positive_only,
                    gt_hybrid_negative_only=negative_only,foreground_change_pp=100*ap_foreground,
                    background_change_pp=100*ap_background,additive_residual=ap_foreground+ap_background-(new_ap-old_ap)),
                gt_hybrids_are_deployable=False,calibrated_threshold=threshold)
            rows[name]=row;run.json(name+'.json',row)
            for column,(label,prediction) in enumerate([('GT',gt),('v10',old_binary),('Coverage',new_binary),('Source readout',calibrated)]):
                small_gt=cv2.resize(gt.astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool)
                small=cv2.resize(prediction.astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool)
                rgb=np.full((*small.shape,3),245,dtype=np.uint8)
                rgb[small&small_gt]=[40,143,91];rgb[small&~small_gt]=[204,66,62];rgb[~small&small_gt]=[56,111,193]
                axes[row_index,column].imshow(rgb);axes[row_index,column].axis('off')
                axes[row_index,column].set_title(name.replace('_',' ')+' / '+label,fontsize=9)
            print('attributed',name,flush=True)
        fig.suptitle('All eight tasks: green = TP, red = FP, blue = FN (retained GT)',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.98]);figure=temp/'errors.png';fig.savefig(figure,dpi=120);plt.close(fig)
        figure_bytes=figure.read_bytes()
        run._write('errors.png',lambda f:f.write(figure_bytes),limit=len(figure_bytes))
        run.json('summary.json',dict(complete=True,rows=rows,gt_hashes=gt_hashes,
            joint_prediction_seal_sha256=sha(run.root/'joint_prediction_seal.json'),
            figure_sha256=sha(run.root/'errors.png'),
            gt_hybrids_are_deployable=False,score_space_attribution_not_upstream_causality=True,
            new_method_gain=False,default_changed=False,durable_dense_prediction_bytes=0))
    except Exception as error:
        run.json('failure.json',dict(type=type(error).__name__,message=str(error)));raise
    finally:
        bank.close()
        for p in temp.iterdir():p.unlink()
        temp.rmdir()


if __name__=='__main__':main()
