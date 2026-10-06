"""Higher source resolution, same next44 typed solver and frozen map."""
import argparse,json,time,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.querying.typed_evidence import visible_operator,view_evidence,solve_set,SetType

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
    base=Path('output/taskwise_scope_composition_next38')/a.key;data=load(base/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());state=load(data['base_field'])['state_dict'];observed=state['observed'].cuda();previous=load(base/'anchored/pipeline_calibrated/scores.pt');prior=previous['scores']['hierarchical_pipeline'].cuda()
    gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda');geometry=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    source=Path('output/subset_readout_next45/highres_source')/a.key;cache={f:load(source/f'identity_{f}.pt') for f in info['train_frames']};square=torch.zeros(prior.shape[1]);count=torch.zeros_like(square)
    for d in cache.values():
        for j,r in enumerate(d['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6);views=[];h,w=192,256
    for frame,d in cache.items():
        camera=geometry.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*geometry.scale,far_plane=100*geometry.scale)
        hit=rasterize_single_view_contributions(geo,renderer,camera.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights'];op,evidence,visible=visible_operator(hit,observed,h*w,False)
        y=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in d['instances']],-1).reshape(h*w,-1).cuda();detected=torch.tensor([len(r['scores'])>0 for r in d['instances']],device='cuda');confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
        views.append(view_evidence(op,evidence,visible,y,detected,frame,h,w,confidence=confidence));del hit
    value,stats=solve_set(prior,views,SetType.TEXT_UNION,corroborate=False,consistency=True)
    out=Path('output/subset_readout_next45/highres')/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=previous['texts'],scores=value.cpu()),out/'elements.pt');shutil.copyfile(base/'field.pt',out/'field.pt');shutil.copyfile(base/'calibration.pt',out/'calibration.pt');stats.update(seconds=time.monotonic()-start,resolution=[h,w],geometry_changed=False);(out/'fit.json').write_text(json.dumps(stats,indent=2)+'\n');print(a.key,stats,flush=True)
if __name__=='__main__':main()
