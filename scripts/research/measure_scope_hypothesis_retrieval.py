"""Separate native scope selection from single-hypothesis range capacity.

GT-selected capacity is diagnostic only, never a deployable query result.
"""
import argparse,json,math
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--base',default='output/scope_crossfit_anchored_next36');a=p.parse_args();torch.set_num_threads(4)
 out=Path(a.base)/('lerf_'+a.scene);data=load(out/'field.pt');h=data['scope_hypergraph'];root=Path(data['root']);query=load(root.parent.parent/'evaluation/text_online.encodings.pt')[0];q=F.normalize(query['encoding'].half().float().cuda(),dim=1)
 native=F.normalize(h['prototypes'].float().cuda(),dim=1)@q.T;choice=native.argmax(0)
 info=json.loads((root/'teachers.json').read_text());descriptors=load(root/info['teacher_files']['siglip']['path'])['descriptors'].float()
 extra=Path('output/scope_hierarchy_full_next33')/('lerf_'+a.scene)/'small_teacher.pt'
 if not extra.exists():extra=Path('output/small_region_restoration_next32')/('lerf_'+a.scene)/'small_teacher.pt'
 if extra.exists():descriptors=torch.cat([descriptors,load(extra)['descriptors'].float()])
 view_scores=F.normalize(descriptors.cuda(),dim=1)@q.T;records=json.loads((out/'construction.json').read_text())['records']
 maximum=torch.stack([view_scores[r['regions']].max(0).values for r in records])
 mixture=torch.stack([(torch.logsumexp(20*view_scores[r['regions']],dim=0)-math.log(len(r['regions'])))/20 for r in records])
 choices={'native_selected_iou':choice,'view_max_iou':maximum.argmax(0),'view_mixture_iou':mixture.argmax(0)}
 if (out/'scope_quality.pt').exists():
  from radio_gs.models.element_retrieval_field import ElementRetrievalField
  from radio_gs.querying.scope_quality import ScopeQuality,scope_features
  parent=load(data['base_field']);model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.eval();fit=load(out/'scope_quality.pt');head=ScopeQuality(fit['mean'].cuda(),fit['std'].cuda()).cuda();head.load_state_dict(fit['state']);head.eval()
  features=scope_features(model,h,q,'text');quality=head(features);choices['learned_quality_iou']=quality.argmax(1);torch.save(dict(scores=quality.cpu(),texts=query['texts']),out/'learned_hypothesis_query.pt')
 torch.save(dict(scores=native.cpu(),chosen_hypotheses={k:v.cpu() for k,v in choices.items()},texts=query['texts']),out/'native_hypothesis_query.pt')
 g_report=json.loads(Path(data['geometry_report']).read_text());gr=g_report['results']['independent_scale'];geo=FrozenGaussianGeometry.load(gr['geometry'],expected_sha256=gr['geometry_sha256'],device='cuda');prior=MogeGeometryPrior(g_report['source_report'],g_report['source_report_sha256']);observed=data['observed'].cuda();indices=h['indices'].cuda();weight=h['weights'].cuda();n,scopes=h['shape']
 frames=load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene,frames=SCENE_GT_FRAMES[a.scene]);rows=[]
 for frame,objects in frames.items():
  height,width=objects[0].mask.shape;camera=prior.cameras[int(frame.split('_')[-1])];k=camera.intrinsic.cuda().clone();k[0]*=width/camera.width;k[1]*=height/camera.height
  truth=torch.stack([torch.from_numpy(o.mask).cuda().flatten() for o in objects]).float();area=truth.sum(1);iou=torch.zeros(scopes,len(objects),device='cuda')
  for start in range(0,scopes,8):
   end=min(scopes,start+8);selected=(indices[1]>=start)&(indices[1]<end);colors=torch.zeros(n,end-start+1,device='cuda');colors[indices[0,selected],indices[1,selected]-start]=weight[selected];colors[:,:-1]*=observed[:,None];colors[:,-1]=observed
   image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=camera.world_to_camera.cuda()[None],Ks=k[None],width=width,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
   mass=image[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);response=image[0,...,:-1]/mass[...,None].clamp_min(1e-8);pred=((response>.5)&valid[...,None]).flatten(0,1).T.float();intersection=pred@truth.T;iou[start:end]=intersection/(pred.sum(1)[:,None]+area[None]-intersection).clamp_min(1)
  for oi,obj in enumerate(objects):
   qi=query['texts'].index(obj.query);best=int(iou[:,oi].argmax());rank=int((native[:,qi]>native[best,qi]).sum())+1
   rows.append(dict(frame=frame,query=obj.query,**{name:float(iou[selection[qi],oi]) for name,selection in choices.items()},single_hypothesis_oracle_iou=float(iou[best,oi]),oracle_native_rank=rank,hypotheses=scopes))
  print(a.scene,frame,'done',flush=True)
 metrics={k:sum(r[k] for r in rows)/len(rows) for k in list(choices)+['single_hypothesis_oracle_iou','oracle_native_rank']};metrics['oracle_rank_at_5']=sum(r['oracle_native_rank']<=5 for r in rows)/len(rows)
 (out/'hypothesis_retrieval.json').write_text(json.dumps(dict(scene=a.scene,metrics=metrics,rows=rows,oracle_is_diagnostic_only=True,single_scope_is_not_union_ceiling=True),indent=2)+'\n');print(metrics,flush=True)


if __name__=='__main__':main()
