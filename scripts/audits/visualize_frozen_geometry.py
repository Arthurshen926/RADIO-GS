"""Matched source-heldout depth renders and world-coordinate geometry views."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.repair_moge_geometry import load_views
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);device='cuda:0'
root=Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914')
receipts=[]
for scene in ('figurines','ramen','teatime','waldo_kitchen'):
 report=json.loads((root/scene/'report.json').read_text())
 prior=MogeGeometryPrior(report['source_report'],report['source_report_sha256'])
 selected=report['heldout_frames'][:2]
 views=[v for v in load_views(prior,480,device,report.get('allow_global_scale_fallback',False)) if v['frame'] in selected]
 views=sorted(views,key=lambda v:selected.index(v['frame']))
 inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']}
 meta=json.loads(Path(inputs['source_authority']['path']).read_text())['metadata']
 fig,axes=plt.subplots(len(views),7,figsize=(21,3.5*len(views)),squeeze=False)
 clouds=[]
 for arm in ('historical_gaussians','independent_scale'):
  if arm=='historical_gaussians':
   model,*rest=load_render_pipeline(meta['config'],meta['checkpoint'],torch.device(device),strict_checkpoint_contract=True,load_ply_rgb_features=False);del rest
  else:
   g=report['results'][arm];model=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=device)
  xyz=model.get_xyz().detach().cpu().numpy()/prior.scale
  ids=np.linspace(0,len(xyz)-1,min(len(xyz),30000)).astype(int)
  clouds.append(xyz[ids])
  for row,v in enumerate(views):
   h,w=v['depth'].shape
   image,alpha,_=rasterization(means=model.get_xyz(),quats=model.get_rotation(),scales=model.get_scaling(),
    opacities=model.get_opacity().reshape(-1),colors=torch.zeros(len(xyz),1,device=device),
    viewmats=torch.linalg.inv(v['pose'])[None],Ks=v['k'][None],width=w,height=h,
    near_plane=.001*prior.scale,far_plane=100*prior.scale,render_mode='RGB+ED',packed=False)
   depth=image[0,...,-1].detach().cpu().numpy()/prior.scale
   al=alpha[0,...,0].detach().cpu().numpy()
   ref=v['depth'].cpu().numpy()/prior.scale;valid=v['valid'].cpu().numpy().astype(bool)
   vmax=float(np.quantile(ref[valid],.98));covered=valid&(al>.1)&np.isfinite(depth)&(depth>0)
   err=np.where(covered,np.abs(depth-ref)/np.maximum(ref,1e-8),np.nan)
   offset=1 if arm=='historical_gaussians' else 4
   if offset==1:axes[row,0].imshow(np.where(valid,ref,np.nan),vmin=0,vmax=vmax,cmap='viridis');axes[row,0].set_title(f'MoGe reference | frame {v["frame"]}')
   axes[row,offset].imshow(np.where(al>.1,depth,np.nan),vmin=0,vmax=vmax,cmap='viridis')
   axes[row,offset].set_title(('Historical' if offset==1 else 'Rebuilt')+f' depth [0,{vmax:.1f}] m')
   axes[row,offset+1].imshow(al,vmin=0,vmax=1,cmap='gray');axes[row,offset+1].set_title('Alpha [0,1]')
   axes[row,offset+2].imshow(err,vmin=0,vmax=.5,cmap='magma');axes[row,offset+2].set_title('AbsRel [0,0.5]; white=missing')
   for ax in axes[row]:ax.set_facecolor('#bdbdbd');ax.axis('off')
  del model;torch.cuda.empty_cache()
 fig.suptitle(scene+' | Same cameras, clipping planes and depth scales; MoGe is not ground truth',fontsize=14)
 fig.tight_layout();fig.savefig(a.output/(scene+'_2d.png'),dpi=140);plt.close(fig)
 fig=plt.figure(figsize=(14,6))
 # Common robust box derived solely from rebuilt geometry. Record excluded counts.
 lo,hi=np.quantile(clouds[1],[.01,.99],axis=0);pad=(hi-lo)*.1;lo-=pad;hi+=pad
 cropped=[]
 for i,xyz in enumerate(clouds):
  ax=fig.add_subplot(1,2,i+1,projection='3d')
  keep=((xyz>=lo)&(xyz<=hi)).all(1);cropped.append(int((~keep).sum()))
  pts=xyz[keep]
  ax.scatter(*pts.T,c=pts[:,2],s=.3,cmap='viridis',vmin=lo[2],vmax=hi[2],alpha=.6,rasterized=True)
  cams=np.stack([prior.cameras[f].camera_to_world[:3,3].numpy()/prior.scale for f in report['mapping_frames']])
  ax.scatter(*cams.T,c='red',s=5,label='mapping cameras')
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
  ax.set_box_aspect(hi-lo);ax.view_init(elev=25,azim=-60)
  ax.set_title(('Historical' if i==0 else 'Rebuilt')+f' centres | {cropped[-1]} sampled points outside box')
 fig.suptitle(scene+' | Gaussian centres, not a mesh; common cropped box; red=cameras')
 fig.tight_layout();fig.savefig(a.output/(scene+'_3d.png'),dpi=160);plt.close(fig)
 # Standalone interactive scatter retains the complete sampled point set.
 try:
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots
  fig=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],subplot_titles=['Historical','Rebuilt'])
  for i,xyz in enumerate(clouds):
   fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',marker=dict(size=1,color=xyz[:,2],colorscale='Viridis'),name=['Historical','Rebuilt'][i]),row=1,col=i+1)
  fig.update_layout(title=scene+' | Full sampled centres; independent interactive views; units=m',scene_aspectmode='data',scene2_aspectmode='data')
  fig.write_html(str(a.output/(scene+'_3d.html')),include_plotlyjs=True)
 except ImportError:pass
 receipts.append({'scene':scene,'frames':[v['frame'] for v in views],'geometry_sha256':report['results']['independent_scale']['geometry_sha256'],
 'sampled_points':[len(c) for c in clouds],'outside_static_box':cropped,'static_box_metres':[lo.tolist(),hi.tolist()],
 'historical_checkpoint':meta['checkpoint'],'note':'centres are not Gaussian footprints; no RGB textures or semantic overlays'})
 print(scene+' complete',flush=True)
(a.output/'manifest.json').write_text(json.dumps(receipts,indent=2)+'\n')
