"""Separate geometry alpha from semantic observed-domain rejection on NVOS fern."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file

torch.set_num_threads(4);device='cuda:0'
base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'nvos_frozen_authority_20260916/fern'
gp=base/'nvos_moge_bound_20260916/fern/frozen_geometry/report.json';gr=json.loads(gp.read_text());rec=gr['results']['independent_scale']
geo=FrozenGaussianGeometry.load(rec['geometry'],expected_sha256=rec['geometry_sha256'],device=device);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
p=root/'nvos96_protected_imagecal_balanced/field.pt';payload=torch.load(p,weights_only=False);geo.assert_binding(payload['geometry_binding']);obs=payload['state_dict']['observed'].to(device)
m=json.load(open('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json'));s=next(x for x in m['scenes'] if x['scene_id']=='fern');rows=[]
for role,index in [('reference',12),('target',1)]:
 camera=prior.cameras[index];h,w=756,1008;k=camera.intrinsic.to(device).clone();k[0]*=w/camera.width;k[1]*=h/camera.height
 with torch.no_grad():
  image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=obs.float()[:,None],viewmats=camera.world_to_camera.to(device)[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
  geometry=(alpha[0,...,0]>1e-8).cpu().numpy();strong=(alpha[0,...,0]>.1).cpu().numpy();semantic=((image[0,...,0]/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)).cpu().numpy()
 if role=='reference':path=Path(s['prompt']['positive_path'])
 else:path=Path(next(f for f in s['frames'] if f['canonical_index']==index)['gt_mask_path'])
 mask=np.asarray(Image.open(path));mask=np.any(mask>127,axis=-1) if mask.ndim==3 else mask>127
 size=(mask.shape[1],mask.shape[0]);geometry,strong,semantic=[cv2.resize(x.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST).astype(bool) for x in (geometry,strong,semantic)]
 rows.append(dict(role=role,positive_pixels=int(mask.sum()),geometry_coverage=float(geometry[mask].mean()),alpha_above_point_one_coverage=float(strong[mask].mean()),semantic_coverage=float(semantic[mask].mean()),geometric_holes=int((mask&~geometry).sum()),geometry_present_semantically_rejected=int((mask&geometry&~semantic).sum()),reference_sha256=sha256_file(path)))
r=dict(schema='nvos_geometry_semantic_support.v1',scene='fern',diagnostic_only=True,gt_used_for_scoring_only=True,field_sha256=sha256_file(p),geometry_sha256=rec['geometry_sha256'],observed_elements=int(obs.sum()),geometry_elements=len(obs),rows=rows,implementation_sha256=sha256_file(__file__))
out=Path('paper/artifacts/nvos_fern_geometry_semantic_support_20260917.json');assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(rows)
