"""Small post-seal complete-cohort N09 spatial diagnosis; no inference decisions."""
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scripts.research.audit_tmm_protocol_correctness import read,binary_counts
from scripts.research.replay_tmm_response import sha


def plot(run,bank,jobs,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seal=read(run.root/'predictions_complete.json');assert seal['images']==72 and seal['jobs']==8 and not seal['target_gt_read']
    arms=['v10','static_native','signed_native','pseudo_native'];fig,axes=plt.subplots(8,5,figsize=(14,16))
    changes={}
    for i,j in enumerate(jobs):
        name=j['name'];spec=read(j['evaluation']);gt=np.asarray(Image.open(spec['gt_mask']))>127;w,h=spec['raster_wh'];preds={}
        for a in arms:
            value=bank.read(run,name+'_'+a,1).reshape(h,w).numpy();preds[a]=cv2.resize(value,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)>=.5
            counts=binary_counts(preds[a],gt);assert counts=={k:rows[name][a]['retained_gt_gt127'][k] for k in counts}
        before,after=preds['static_native'],preds['signed_native']
        changes[name]=dict(recovered_foreground=int((~before&after&gt).sum()),lost_foreground=int((before&~after&gt).sum()),
            removed_false_positive=int((before&~after&~gt).sum()),added_false_positive=int((~before&after&~gt).sum()))
        for c,(label,pred) in enumerate([('GT',gt),*preds.items()]):
            truth=cv2.resize(gt.astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool)
            small=cv2.resize(pred.astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool)
            rgb=np.full((*small.shape,3),245,dtype=np.uint8);rgb[small&truth]=[40,143,91];rgb[small&~truth]=[204,66,62];rgb[~small&truth]=[56,111,193]
            axes[i,c].imshow(rgb);axes[i,c].axis('off');axes[i,c].set_title(name+' / '+label,fontsize=8)
    fig.suptitle('N09 all eight tasks: green TP / red FP / blue FN');fig.tight_layout(rect=[0,0,1,.98])
    with tempfile.TemporaryDirectory(prefix='radiogs_affinity_figure_',dir='/dev/shm') as name:
        p=Path(name)/'errors.png';fig.savefig(p,dpi=95);plt.close(fig);data=p.read_bytes();run._write('errors.png',lambda f:f.write(data),limit=len(data))
    run.json('spatial_diagnosis.json',dict(complete=True,diagnostic_only=True,transitions=changes,figure_sha256=sha(run.root/'errors.png'),
        prediction_seal_sha256=sha(run.root/'predictions_complete.json'),summary_sha256=sha(run.root/'summary.json')))
