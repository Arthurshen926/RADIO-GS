"""Capture pinned MoGe-3 encoder/refiner teachers on source RGB only.

Run in the existing sam3 environment. Frozen retrieval geometry is not loaded or
modified; source image coordinates are the only correspondence asserted here.
"""
import argparse,hashlib,json,inspect
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from moge.model.v3 import MoGeModel


def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
 base=Path('/mnt/pool/sqy/results/RADIO-GS/output');src=base/'preserved_source_inventory_20260916'/a.scene;out=base/'moge_feature_teacher_20260917'/a.scene;out.mkdir(parents=True,exist_ok=False)
 info=json.loads((src/'teachers.json').read_text());sources=json.loads((src/'source_frames.json').read_text());lookup={r['frame_idx']:r for r in sources['frames']}
 snapshot=Path('/root/.cache/huggingface/hub/models--Ruicheng--moge-3-vitl/snapshots/a96e58bad16a94c9a3c193a5d4cd75b4b6906c94/model.pt')
 model=MoGeModel.from_pretrained(str(snapshot)).to(a.device).eval().requires_grad_(False)
 captured={}
 def encoder_hook(module,inputs,output):captured['encoder']=output[0].detach()
 def refiner_input(module,inputs):captured['coords']=inputs[1].detach();captured['shape']=inputs[2]
 def feature_hook(module,inputs):captured['refiner']=inputs[0].detach()
 handles=[model.encoder.register_forward_hook(encoder_hook),model.refiner.register_forward_pre_hook(refiner_input),model.refiner.out_proj.register_forward_pre_hook(feature_hook)]
 records=[]
 for frame in info['frames']:
  r=lookup[frame];path=Path(sources['image_dir'])/r['source_file'];assert digest(path)==r['source_sha256']
  image=Image.open(path).convert('RGB');width=640;height=round(image.height*width/image.width);image=image.resize((width,height),Image.Resampling.BICUBIC)
  rgb=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float().to(a.device)/255
  with torch.inference_mode():prediction=model.infer(rgb,resolution_level=9,refine_steps=3,use_fp16=True)
  coords=captured['coords'].long();shape=captured['shape'];features=captured['refiner'];b,h,w=map(int,shape[:3]);assert b==1 and len(coords)==h*w
  linear=coords[:,1]*w+coords[:,2];assert len(linear.unique())==h*w and int(linear.min())==0 and int(linear.max())==h*w-1
  dense=torch.empty(h*w,features.shape[-1],device=features.device,dtype=features.dtype);dense[linear]=features
  refiner=F.interpolate(dense.T.reshape(1,-1,h,w).float(),size=(96,128),mode='bilinear',align_corners=False)[0]
  encoder=F.interpolate(captured['encoder'].float(),size=(96,128),mode='bilinear',align_corners=False)[0]
  target=out/f'frame_{frame:05d}.pt'
  torch.save(dict(encoder=encoder.cpu(),refiner=refiner.cpu(),source_frame=frame,image_sha256=r['source_sha256'],refiner_grid=[h,w],encoder_grid=list(captured['encoder'].shape[-2:]),input_wh=[width,height],feature_grid=[96,128],mask=prediction['mask'].cpu(),normals=prediction.get('normals',torch.empty(0)).cpu(),points=prediction['points'].cpu(),coordinates_verified=True),target)
  records.append(dict(frame=frame,path=str(target),sha256=digest(target),encoder_channels=len(encoder),refiner_channels=len(refiner),refiner_grid=[h,w]));print(a.scene,records[-1],flush=True)
 for handle in handles:handle.remove()
 receipt=dict(schema='moge_source_features.v1',scene=a.scene,source_only=True,benchmark_data_used=False,geometry_modified=False,train_frames=info['train_frames'],validation_frames=info['validation_frames'],checkpoint_sha256=digest(snapshot),revision=snapshot.parent.name,model_source_sha256=digest(inspect.getfile(MoGeModel)),implementation_sha256=digest(__file__),records=records,feature_semantics='encoder image grid and final refiner pre-output features; not text aligned; cross-view invariance unverified',refinement_steps=3,resolution_level=9,input_width=640)
 (out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
