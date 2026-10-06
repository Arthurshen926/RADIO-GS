"""MoGe-anchored centre and anisotropic footprint refinement using source RGB."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior


def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--steps',type=int,default=1500);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);start=time.monotonic()
    field=load(Path('output/taskwise_scope_composition_next38')/('lerf_'+a.scene)/'field.pt')
    report=json.loads(Path(field['geometry_report']).read_text());base=load(report['results']['independent_scale']['geometry'])
    state={k:v.cuda() for k,v in base['state_dict'].items()};info=json.loads((Path(field['root'])/'teachers.json').read_text())
    cameras=MogeGeometryPrior(report['source_report'],report['source_report_sha256']);views={}
    for f in info['train_frames']+info['validation_frames']:
        im=Image.open(f'/mnt/pool/sqy/3d_understanding/lerf_ovs/{a.scene}/images/frame_{f:05d}.jpg').convert('RGB')
        width=512;height=round(im.height*width/im.width);im=im.resize((width,height),Image.Resampling.LANCZOS)
        cam=cameras.cameras[f];k=cam.intrinsic.cuda().clone();k[0]*=width/cam.width;k[1]*=height/cam.height
        views[f]=(torch.from_numpy(np.asarray(im).copy()).cuda().float()/255,cam.world_to_camera.cuda()[None],k[None])
    color=torch.nn.Parameter(torch.zeros_like(state['xyz']));opacity=torch.nn.Parameter(torch.logit(state['opacities'].flatten()))
    log_size=torch.nn.Parameter(torch.zeros_like(state['scales']))
    offset=torch.nn.Parameter(torch.zeros_like(state['xyz']))
    radius=state['scales'].amax(1,keepdim=True).detach()
    optimizer=torch.optim.Adam([dict(params=[color],lr=.03),dict(params=[opacity],lr=.01),dict(params=[log_size],lr=.003),dict(params=[offset],lr=.003)])
    out=Path('output/carrier_refine_next51')/a.scene;out.mkdir(parents=True,exist_ok=True)
    def render(f):
        target,pose,k=views[f];h,w=target.shape[:2]
        rgb,alpha,_=rasterization(means=state['xyz']+radius*offset.tanh(),quats=state['quaternions'],scales=state['scales']*log_size.exp(),opacities=opacity.sigmoid(),colors=color.sigmoid(),viewmats=pose,Ks=k,width=w,height=h,near_plane=.001*cameras.scale,far_plane=100*cameras.scale,packed=False)
        return rgb[0],target
    @torch.no_grad()
    def evaluate(tag):
        result={}
        for split,frames in [('train',info['train_frames']),('validation',info['validation_frames'])]:
            mse=[]
            for f in frames:
                rgb,target=render(f);mse.append(float((rgb-target).square().mean()))
                if split=='validation':Image.fromarray((rgb.clamp(0,1)*255).byte().cpu().numpy()).save(out/f'{tag}_{f}.png')
            result[split]=dict(mse=float(np.mean(mse)),psnr=float(np.mean([-10*np.log10(x) for x in mse])))
        return result
    before=evaluate('before');history=[]
    # Fixed source ordering, no semantic labels and no benchmark selection.
    frames=info['train_frames'];order=torch.randperm(len(frames)).tolist()
    for step in range(a.steps):
        if step%len(frames)==0:order=torch.randperm(len(frames)).tolist()
        f=frames[order[step%len(frames)]];rgb,target=render(f)
        photometric=(rgb-target).abs().mean();loss=photometric+.001*log_size.square().mean()+.001*offset.tanh().square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (step+1)%100==0:
            record=dict(step=step+1,l1=float(photometric),median_size=float(log_size.exp().median()),median_opacity=float(opacity.sigmoid().median()))
            history.append(record);print(record,flush=True)
    after=evaluate('after')
    result={k:v.detach().cpu() for k,v in state.items()};result['xyz']=(state['xyz']+radius*offset.tanh()).detach().cpu();result['scales']=(state['scales']*log_size.exp()).detach().cpu();result['opacities']=opacity.sigmoid().detach().cpu()[:,None]
    payload=dict(base);payload['state_dict']=result;payload['source_rgb_fit_next51']=dict(frames=frames,steps=a.steps,centres_and_normals_fixed=False,rotations_fixed=True,centre_displacement_bound="one maximum parent sigma per axis",query_or_label_access=False)
    torch.save(payload,out/'geometry.pt');torch.save(color.sigmoid().detach().cpu(),out/'rgb.pt')
    meta=dict(scene=a.scene,steps=a.steps,seconds=time.monotonic()-start,before=before,after=after,history=history,source_frames=frames,validation_frames=info['validation_frames'],centres_fixed=False,normals_fixed=True,geometry_frozen_before_semantic_rebuild=True,benchmark_masks_used=False)
    (out/'fit.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta,flush=True)

if __name__=='__main__':main()
