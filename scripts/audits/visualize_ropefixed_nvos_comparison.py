"""Post-prediction GT/error comparison; target RGB is never read."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--scene',choices=['fern','orchids'],required=True);p.add_argument('--run',default='image_supervised_native_residual');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
base=Path('/mnt/pool/sqy/results/RADIO-GS/output');old=base/f'nvos_{a.scene}_query_ropefix_control_20260917';new=base/'nvos_ropefix_pipeline_20260917'/a.scene/'evaluation'/f'{a.scene}_signed'
reports=[json.loads((d/'report.json').read_text()) for d in [old,new]]
assert reports[0]['query_receipt_sha256']==reports[1]['query_receipt_sha256']
predictions=[]
for d,r in zip([old,new],reports):
    path=d/(a.run+'.pt');assert sha256_file(path)==r['results'][a.run]['prediction_sha256']
    x=torch.load(path,weights_only=False,map_location='cpu');predictions.append(x['response'].numpy())
manifest=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json');m=json.loads(manifest.read_text());s=next(s for s in m['scenes'] if s['scene_id']==a.scene);f=next(f for f in s['frames'] if f['frame_id']==reports[1]['target_frame']);gtpath=Path(f['gt_mask_path']);gt=np.array(Image.open(gtpath));gt=np.any(gt>127,axis=2) if gt.ndim==3 else gt>127
assert m['protocol']['resize']=='nearest'
predictions=[cv2.resize(x,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST)>=.5 for x in predictions]
assert gt.shape==predictions[0].shape==predictions[1].shape
for pred,r in zip(predictions,reports):assert abs(np.count_nonzero(pred&gt)/max(1,np.count_nonzero(pred|gt))-r['results'][a.run]['iou'])<1e-6
errors=np.zeros((*gt.shape,3));pred=predictions[1];errors[pred&gt]=[.2,.7,.3];errors[gt&~pred]=[.9,.2,.2];errors[pred&~gt]=[.2,.4,1.]
fig,axes=plt.subplots(1,4,figsize=(14,4))
for ax,data,title in zip(axes,[gt,*predictions,errors],['GT','Legacy source map %.2f%%'%(100*reports[0]['results'][a.run]['iou']),'Corrected source map %.2f%%'%(100*reports[1]['results'][a.run]['iou']),'Green TP / Red FN / Blue FP']):ax.imshow(data,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
fig.suptitle(a.scene+' | Same corrected reference query and adaptive solver | '+a.run,fontsize=10);fig.tight_layout();fig.savefig(a.output,dpi=150);plt.close(fig)
a.output.with_suffix('.json').write_text(json.dumps(dict(target_rgb_opened=False,gt_read_after_predictions=True,gt_sha256=sha256_file(gtpath),reports={str(d/'report.json'):sha256_file(d/'report.json') for d in [old,new]},implementation_sha256=sha256_file(__file__),plot_sha256=sha256_file(a.output)),indent=2)+'\n')
