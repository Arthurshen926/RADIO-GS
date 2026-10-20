"""Post-seal all-eight spatial audit of N03's exact transient predictions."""
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from radio_gs.query_field.bounded_experiment import BoundedExperiment
from scripts.research.audit_tmm_protocol_correctness import REPO, binary_counts, read
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.run_tmm_support_agreement import array_sha

ARMS=['v10','coverage','held_local','held_uniform']


def main():
    os.chdir(REPO);parent=STUDY/'nvos_local_credit_v1';pm=read(parent/'manifest.json')
    source_bank=Path(pm['volatile_prediction_directory'])
    run=BoundedExperiment(STUDY,'nvos_local_credit_spatial_v1',POLICY);run.preflight(2*2**20)
    temp=Path(tempfile.mkdtemp(prefix='radiogs_nvos_local_audit_',dir='/dev/shm'))
    jobs=pm['jobs'];expected={j['name']+'_'+a for j in jobs for a in ARMS}
    sources=[__file__,'scripts/research/audit_tmm_protocol_correctness.py']
    run.json('manifest.json',dict(parent='nvos_local_credit_v1',parent_manifest_sha256=sha(parent/'manifest.json'),
        source_hashes={p:sha(p) for p in sources},linked_prediction_directory=str(temp),
        linked_prediction_limit_bytes=128*2**20,target_gt_use='post-seal spatial diagnosis only'))
    run.json('source_snapshot.json',{p:dict(text=Path(p).read_text(),sha256=sha(p)) for p in sources})
    links={};started=time.monotonic()
    try:
        while len(links)<len(expected) or not (parent/'summary.json').exists():
            if (source_bank/'manifest.json').exists():
                for key,entry in read(source_bank/'manifest.json').items():
                    name=key.rsplit('::',1)[1]
                    if name not in expected or name in links:continue
                    target=temp/entry['file']
                    if sum(p.stat().st_size for p in temp.iterdir())+entry['bytes']+4096>128*2**20:
                        raise ValueError('private audit link budget exceeded')
                    os.link(source_bank/entry['file'],target);links[name]=dict(entry,path=str(target))
            if (parent/'failure.json').exists() or time.monotonic()-started>7200:
                raise RuntimeError('N03 failed/timed out; audit has not opened GT')
            time.sleep(.1)
        seal=read(parent/'predictions_complete.json');summary=read(parent/'summary.json')
        for job in jobs:
            for arm in ARMS:
                entry=links[job['name']+'_'+arm];array=np.load(entry['path'],mmap_mode='r',allow_pickle=False)
                assert array_sha(array)==seal['predictions'][job['name']][arm]['sha256']==entry['sha256']
        run.json('prediction_binding.json',dict(jobs=8,images=32,all_parent_images=48,target_gt_read=False,
            parent_seal_sha256=sha(parent/'predictions_complete.json'),parent_summary_sha256=sha(parent/'summary.json'),
            exact_prediction_digests={k:r['sha256'] for k,r in links.items()}))
        print('all parent predictions and scores sealed; starting spatial audit',flush=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(8,5,figsize=(14.5,17))
        rows={};gt_hashes={}
        for i,job in enumerate(jobs):
            name=job['name'];spec=read(job['evaluation']);gt=np.asarray(Image.open(spec['gt_mask']))>127
            gt_hashes[spec['gt_mask']]=sha(spec['gt_mask']);w,h=spec['raster_wh']
            arrays={a:cv2.resize(np.load(links[name+'_'+a]['path'],allow_pickle=False).reshape(h,w),
                                (gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR) for a in ARMS}
            masks={a:v>=.5 for a,v in arrays.items()}
            metrics={a:binary_counts(m,gt) for a,m in masks.items()}
            for a,m in metrics.items():
                assert m=={k:summary['rows'][name][a]['retained_gt_gt127'][k] for k in m}
            transitions={};score_changes={}
            for a,base in [('held_local','v10'),('held_local','coverage'),('held_local','held_uniform')]:
                key=a+'_vs_'+base;before,after=masks[base],masks[a]
                transitions[key]=dict(recovered_foreground=int((~before&after&gt).sum()),
                    lost_foreground=int((before&~after&gt).sum()),removed_false_positive=int((before&~after&~gt).sum()),
                    added_false_positive=int((~before&after&~gt).sum()))
                delta=arrays[a]-arrays[base]
                score_changes[key]={label:dict(mean_delta=float(delta[mask].mean(dtype=np.float64)),
                    absolute_delta=float(np.abs(delta[mask]).mean(dtype=np.float64)),
                    raised_pixels=int((delta[mask]>1e-6).sum()),lowered_pixels=int((delta[mask]<-1e-6).sum()))
                    for label,mask in [('foreground',gt),('background',~gt)]}
            edges=np.array([0,.01,.05,.1,.25,.5,.75,.9,1.00001])
            histograms={label:{a:np.histogram(v[mask],edges)[0].tolist() for a,v in arrays.items()}
                        for label,mask in [('foreground',gt),('background',~gt)]}
            row=dict(metrics=metrics,transitions=transitions,score_changes=score_changes,
                score_histogram_edges=edges.tolist(),score_histograms=histograms)
            rows[name]=row;run.json(name+'.json',row)
            for column,(label,prediction) in enumerate([('GT',gt)]+[(a,masks[a]) for a in ARMS]):
                small_gt=cv2.resize(gt.astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool)
                small=cv2.resize(prediction.astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool)
                rgb=np.full((*small.shape,3),245,dtype=np.uint8)
                rgb[small&small_gt]=[40,143,91];rgb[small&~small_gt]=[204,66,62];rgb[~small&small_gt]=[56,111,193]
                axes[i,column].imshow(rgb);axes[i,column].axis('off')
                axes[i,column].set_title(name.replace('_',' ')+' / '+label.replace('_',' '),fontsize=9)
        fig.suptitle('N03, all eight tasks: green = TP, red = FP, blue = FN (retained GT)',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.98]);path=temp/'errors.png';fig.savefig(path,dpi=120);plt.close(fig)
        data=path.read_bytes();run._write('errors.png',lambda f:f.write(data),limit=len(data))
        run.json('summary.json',dict(complete=True,rows=rows,gt_hashes=gt_hashes,diagnostic_only=True,
            prediction_binding_sha256=sha(run.root/'prediction_binding.json'),figure_sha256=sha(run.root/'errors.png'),
            default_changed=False,durable_dense_prediction_bytes=0))
        print('spatial audit complete',flush=True)
    except Exception as error:
        run.json('failure.json',dict(type=type(error).__name__,message=str(error)));raise
    finally:
        for p in temp.iterdir():p.unlink()
        temp.rmdir()


if __name__=='__main__':main()
