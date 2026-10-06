"""Post-score diagnostic only: fixed NVOS mask examples, no parameter fitting."""
from pathlib import Path
import json
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('output/prompt_transfer_20260919')
manifest=json.loads(Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json').read_text())
tasks=['flower','fortress','leaves']
fig,axes=plt.subplots(len(tasks),4,figsize=(14,9))
for row,task in enumerate(tasks):
 report=json.loads((root/task/'report.json').read_text())
 scene=next(s for s in manifest['scenes'] if s['scene_id']==task)
 target=next(f for f in scene['frames'] if f['frame_id'] in scene['evaluation_frame_ids'])
 gt=np.asarray(Image.open(target['gt_mask_path']));gt=np.any(gt>127,axis=2) if gt.ndim==3 else gt>127
 axes[row,0].imshow(gt,cmap='gray');axes[row,0].set_title(task+' / GT')
 for col,arm in enumerate(['baseline','prompt_only','prompt_sam'],1):
  score=torch.load(root/task/(arm+'.pt'),map_location='cpu',weights_only=False)['response'].numpy()
  score=cv2.resize(score,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR);pred=score>=.5
  image=np.full((*gt.shape,3),.12);image[pred&gt]=[.1,.8,.3];image[pred&~gt]=[.95,.2,.2];image[~pred&gt]=[.2,.4,1.]
  axes[row,col].imshow(image);axes[row,col].set_title(f"{arm}: {report['results'][arm]['iou']*100:.2f}%")
 for ax in axes[row]:ax.axis('off')
fig.suptitle('Fixed post-score examples: green=TP, red=FP, blue=FN. No target RGB used.')
fig.tight_layout();path=Path('output/parallel_method_transfer_20260919/prompt_masks.png');fig.savefig(path,dpi=140);plt.close(fig);print(path)
