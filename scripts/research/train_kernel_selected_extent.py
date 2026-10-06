"""Train the continuous text extent field through both deployed render outputs."""
import argparse
import json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from scripts.audits.train_rendered_extent import balanced_mask_loss


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--steps',type=int,default=400);p.add_argument('--source',default='output/pipeline_kernel_extent_next40');p.add_argument('--output',default='output/selected_kernel_extent_next40');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    src=Path(a.source)/a.key;data=load(src/'field.pt');cache=load(src/'source_training.pt');state=data['source_kernel_extent']
    root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    report=json.loads(Path(data['geometry_report']).read_text());g=report['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda')
    parent=load(data['base_field']);observed=parent['state_dict']['observed'].cuda();n=len(observed)
    q=torch.nn.functional.normalize(state['anchors'].float().cuda(),dim=1)
    coefficients=state['inverse'].cuda()@torch.exp(state['concentration']*(q@q.T-1))
    initial=state['residual'].float().cuda();parameter=torch.nn.Parameter(initial.clone());baseline=cache['baseline'].float().cuda()
    examples=[{k:(v.cuda() if torch.is_tensor(v) else v) for k,v in e.items()} for e in cache['examples']]
    tasks=[];validation=[]
    h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    for index,e in enumerate(examples):
        camera=auth['metadata']['cameras'][str(info['frames'][e['view']])]
        hit=load(root/auth['views'][e['view']]['relative_path'])
        pid,gid=hit['pixel_ids'].long().cuda(),hit['gaussian_ids'].long().cuda()
        e['mass']=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,hit['base_weights'].float().cuda()*observed[gid])
        e['camera']=torch.tensor(camera['world_to_camera'],device='cuda');e['intrinsic']=torch.tensor(camera['intrinsic'],device='cuda')
        available=[(index,j) for j in range(len(e['columns']))]
        if e['view'] in cache['held']:
            ids=torch.linspace(0,len(available)-1,min(16,len(available))).long().tolist();validation.extend(available[j] for j in ids)
        else:tasks.extend(available)
    def predict(e,j):
        qi=e['columns'][j]
        return (baseline[:,qi]+parameter@coefficients[:,qi]).clamp(1e-6,1-1e-6)*observed
    def alpha(e,gate):
        _,result,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten()*gate,colors=torch.ones(n,1,device='cuda'),viewmats=e['camera'][None],Ks=e['intrinsic'][None],width=w,height=h,near_plane=auth['metadata']['near_plane'],far_plane=auth['metadata']['far_plane'],packed=False)
        return result[0,...,0].flatten()
    @torch.no_grad()
    def measure(step):
        values=[]
        for vi,j in validation:
            e=examples[vi];s=predict(e,j);truth=e['target'][:,j].bool()
            probability=(torch.sparse.mm(e['op'],s[:,None])[:,0]>.5)&e['visible']
            selected=(alpha(e,(s>.5).float())*255+.5).floor()>10
            def iou(v):return float((v&truth).sum()/(v|truth).sum().clamp_min(1))
            values.append((iou(probability),iou(selected)))
        row=dict(step=step,source_held_probability_iou=sum(x[0] for x in values)/len(values),source_held_selected_iou=sum(x[1] for x in values)/len(values),query_views=len(values));print(a.key,row,flush=True);return row
    history=[measure(0)];optimizer=torch.optim.Adam([parameter],lr=.003)
    order=torch.randperm(len(tasks)).tolist()
    for step in range(a.steps):
        vi,j=tasks[order[step%len(order)]];e=examples[vi];s=predict(e,j);truth=e['target'][:,j];known=e['known'][:,j].bool()
        projected=torch.sparse.mm(e['op'],s[:,None])[:,0]
        full_loss=balanced_mask_loss(projected[:,None],truth[:,None],known&e['visible'])
        with torch.no_grad():hard_alpha=alpha(e,(s>.5).float())
        # Hard forward matches selected-only deployment. A separate soft render
        # supplies gradients also to currently unselected Gaussian opacities.
        soft_alpha=alpha(e,torch.sigmoid((s-.5)/.1)*observed)
        silhouette=hard_alpha+(soft_alpha-soft_alpha.detach())
        selected=silhouette/(silhouette+10.5/255)
        # Hidden geometry revealed by removing occluders is not a known negative.
        visible_contribution=torch.sparse.mm(e['op'],(s.detach()>.5).float()[:,None])[:,0]
        visible_fraction=(visible_contribution*e['mass'])/hard_alpha.clamp_min(1e-8)
        selected_known=truth.bool()|(known&e['visible']&(visible_fraction>=.5))
        selected_loss=balanced_mask_loss(selected[:,None],truth[:,None],selected_known)
        loss=.5*(full_loss+selected_loss)+.01*(parameter-initial).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%100==0:
            row=measure(step+1);row['loss']=float(loss.detach());history.append(row)
    out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True)
    state['residual']=parameter.detach().half().cpu();torch.save(data,out/'field.pt')
    (out/'training.json').write_text(json.dumps(dict(key=a.key,steps=a.steps,history=history,source_training_query_views=len(tasks),geometry_frozen=True,native_coordinates_frozen=True,benchmark_query_or_gt_used=False,selected_forward='hard selected opacity',selected_backward='separate sigmoid-gated opacity render',source_only=True),indent=2)+'\n')


if __name__=='__main__':main()
