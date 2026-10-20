"""Inspect reference-scope preservation using authorized source inputs, never GT."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from scripts.research.audit_tmm_protocol_correctness import REPO, SYSTEM, read, load, binary_counts
from scripts.research.replay_tmm_response import STUDY, POLICY, sha


def main():
    os.chdir(REPO);torch.set_num_threads(1)
    run=BoundedExperiment(STUDY,'nvos_reference_scope_v1',POLICY);run.preflight(2*2**20)
    sources=[__file__,'scripts/research/audit_tmm_protocol_correctness.py']
    jobs=[j for j in read('configs/pipeline/system_v10.json')['jobs'] if j['benchmark']=='nvos']
    run.json('manifest.json',dict(jobs=jobs,source_hashes={p:sha(p) for p in sources},
        target_gt_read=False,target_rgb_read=False,purpose='source-to-source agreement, not GT accuracy'))
    run.json('source_snapshot.json',{p:dict(text=Path(p).read_text(),sha256=sha(p)) for p in sources})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(8,3,figsize=(10.5,17))
    rows={};inputs={}
    for index,job in enumerate(jobs):
        name=job['name'];query=read(job['queries']);spec=read(job['evaluation'])
        fit_path=SYSTEM/'nvos'/name/'query/reference/0/fit.json';fit=read(fit_path);scene=read(fit['scene'])
        frame=query['reference_frame'];assert frame in scene['source_frames'] and frame!=spec['target_frame']
        assert Path(query['reference_rgb']).resolve()==Path(scene['source_rgb'][str(frame)]).resolve()
        tracks=load(fit['tracks']);record=tracks['frames'][frame]
        logits=record['logits'].float().numpy();tracked=logits>0
        seed=np.asarray(Image.open(query['reference_mask']))
        assert seed.ndim==2
        seed=cv2.resize((seed>127).astype(np.uint8),(tracked.shape[1],tracked.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
        counts=binary_counts(tracked,seed)
        # Reference SAM mask is itself a prediction, never a ground-truth mask.
        row=dict(reference_frame=frame,target_frame=spec['target_frame'],track_hw=list(tracked.shape),
            agreement=counts,seed_pixels=int(seed.sum()),track_pixels=int(tracked.sum()),
            seed_only_pixels=int((seed&~tracked).sum()),track_only_pixels=int((tracked&~seed).sum()),
            object_logit=float(record['object_logit']),reference_mask_is_gt=False,target_gt_read=False,target_rgb_read=False)
        rows[name]=row;run.json(name+'.json',row)
        for path in [job['queries'],job['evaluation'],fit_path,fit['scene'],fit['tracks'],query['reference_rgb'],query['reference_mask']]:
            inputs[str(path)]=sha(path)
        rgb=np.asarray(Image.open(query['reference_rgb']).convert('RGB'))
        rgb=cv2.resize(rgb,(384,288),interpolation=cv2.INTER_AREA)
        sm=[cv2.resize(m.astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool) for m in (seed,tracked)]
        for col,(label,mask) in enumerate(zip(('Reference SAM seed','SAM3 reference track'),sm)):
            overlay=rgb.copy();overlay[mask]=(.55*overlay[mask]+.45*np.array([40,190,95])).astype(np.uint8)
            axes[index,col].imshow(overlay);axes[index,col].axis('off');axes[index,col].set_title(name+' / '+label,fontsize=8)
        difference=np.full((288,384,3),245,dtype=np.uint8)
        difference[sm[0]&sm[1]]=[40,143,91];difference[sm[0]&~sm[1]]=[56,111,193];difference[~sm[0]&sm[1]]=[204,66,62]
        axes[index,2].imshow(difference);axes[index,2].axis('off');axes[index,2].set_title('Source agreement '+f"{100*counts['iou']:.2f}%",fontsize=8)
        del tracks
    fig.suptitle('Authorized reference views only; green = overlap, blue = seed only, red = track only',fontsize=10)
    fig.tight_layout(rect=[0,0,1,.985])
    import io
    stream=io.BytesIO();fig.savefig(stream,format='jpg',dpi=110,pil_kwargs={'quality':88});plt.close(fig)
    data=stream.getvalue();run._write('reference_scope.jpg',lambda f:f.write(data),limit=len(data))
    run.json('summary.json',dict(complete=True,rows=rows,input_hashes=inputs,figure_sha256=sha(run.root/'reference_scope.jpg'),
        reference_mask_is_gt=False,target_gt_read=False,target_rgb_read=False,new_method_gain=False,
        unchanged_inference=True,durable_dense_prediction_bytes=0))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
