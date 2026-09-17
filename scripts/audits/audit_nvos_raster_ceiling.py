"""Exact binary IoU ceiling of the declared nearest-neighbor scoring grid.

Target labels are used for diagnosis only; no field or query is optimized.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
e=json.loads(a.evaluation.read_text());mp=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json');m=json.loads(mp.read_text());assert sha256_file(mp)==e['manifest_sha256'] and m['protocol']['resize']=='nearest'
s=next(s for s in m['scenes'] if s['scene_id']==a.scene);t=next(f for f in s['frames'] if f['frame_id']==e['target_frame']);p=Path(t['gt_mask_path']);assert sha256_file(p)==t['ground_truth_sha256'];gt=np.array(Image.open(p));gt=np.any(gt>127,axis=2) if gt.ndim==3 else gt>127;w,h=e['grid_wh'];assert h*w<2**24
index=cv2.resize(np.arange(h*w,dtype=np.float32).reshape(h,w),(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST).astype(np.int64).flatten()
pos=np.bincount(index,weights=gt.flatten(),minlength=h*w);count=np.bincount(index,minlength=h*w)
# A selected cell contributes p TP and (n-p) FP. At the optimum rho,
# selection obeys p-rho*(n-p)>0: a threshold on p/n. Scan all prefixes.
order=np.argsort(-(pos/count.clip(1)),kind='stable');tp=np.cumsum(pos[order]);fp=np.cumsum(count[order]-pos[order]);ious=tp/(gt.sum()+fp).clip(1);best=int(np.argmax(ious))
r=dict(schema='nearest_scoring_grid_binary_iou_ceiling.v1',scene=a.scene,target_gt_used=True,method_score=False,geometry_assumptions=False,render_grid_wh=[w,h],gt_grid_wh=[gt.shape[1],gt.shape[0]],exact_binary_raster_iou_ceiling=float(ious[best]),selected_pixels=best+1,evaluation_sha256=sha256_file(a.evaluation),gt_sha256=sha256_file(p),implementation_sha256=sha256_file(__file__))
a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['exact_binary_raster_iou_ceiling'])
