"""Learn native mixture coefficients from qualified source masks and both outputs."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.models.native_scope_dictionary import NativeScopeDictionary,attach_scope_text_readout
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from scripts.audits.train_rendered_extent import balanced_mask_loss


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--source',default='output/scope_query_modes_next34');p.add_argument('--output',required=True);p.add_argument('--mode',choices=['full','joint','hard'],default='hard');p.add_argument('--steps',type=int,default=400);p.add_argument('--train-scope',action='store_true');p.add_argument('--image-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);source=Path(a.source)/a.key;data=load(source/'field.pt');root=Path(data['root']);parent=load(data['base_field'])
    model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.requires_grad_(False);model.eval()
    data['native_modalities']=['image'] if a.image_only else ['text','image']
    field=NativeScopeDictionary(data);attach_scope_text_readout(model,field)
    if not a.image_only:
        calibration=load(source/'calibration.pt');original=model.native_query_thresholds['text'].forward
        def threshold(q):
            delta=F.normalize(q,dim=-1)@calibration['weight'].cuda()+calibration['bias'].cuda()
            return original(q)+delta.clamp(-calibration['max_delta'],calibration['max_delta'])
        model.native_query_thresholds['text'].forward=threshold
    initial=field.weights.detach().clone();allowed=initial>0;logits=torch.nn.Parameter(initial.clamp_min(1e-8).log());parameters=[dict(params=[logits],lr=.03)]
    if a.train_scope:
        model.grounded_transfer.decoder.requires_grad_(True)
        if model.grounded_transfer.residual is not None:model.grounded_transfer.residual.requires_grad_(True)
        parameters.append(dict(params=[p for p in model.parameters() if p.requires_grad],lr=1e-4))
    opt=torch.optim.Adam(parameters)
    gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda')
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());ep=load(root/'episodes.pt');teacher=load(root/info['teacher_files']['siglip']['path']);raw=load(root/'source_grounded_text.pt')
    descriptors=F.normalize(teacher['descriptors'].float(),dim=1);masks=ep['masks'];proposal_views=ep['proposal_views'];text=ep['queries']['text'].float().cuda()
    small_path=Path('output/scope_hierarchy_full_next33')/a.key/'small_teacher.pt'
    if not small_path.exists():small_path=Path('output/small_region_restoration_next32')/a.key/'small_teacher.pt'
    if small_path.exists():
        small=load(small_path);descriptors=torch.cat([descriptors,F.normalize(small['descriptors'].float(),dim=1)]);masks=torch.cat([masks,small['masks']]);proposal_views=torch.cat([proposal_views,small['proposal_views']])
    h,w=raw['probability'].shape[-2:];pixels=h*w;n=len(model.latent);views=[];train_tasks=[[],[]];validation=[]
    training_frames=info['train_frames'][:-2]
    validation_frames=info['validation_frames']+info['train_frames'][-2:]
    for frame in info['frames']:
        vi=info['frames'].index(frame);hit=load(root/auth['views'][vi]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weights=hit['base_weights'].float().cuda()
        total=torch.zeros(pixels,device='cuda').scatter_add_(0,pid,weights);ow=weights*model.observed[gid];mass=torch.zeros_like(total).scatter_add_(0,pid,ow)
        mapping=torch.sparse_coo_tensor(torch.stack([pid,gid]),ow/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce();valid=(mass/total.clamp_min(1e-8)>.5)&(total>1e-8)
        ids=torch.where(proposal_views==vi)[0];camera=auth['metadata']['cameras'][str(frame)]
        view=dict(frame=frame,mapping=mapping,mass=mass,valid=valid,k=torch.tensor(camera['intrinsic'],device='cuda'),camera=torch.tensor(camera['world_to_camera'],device='cuda'),masks=masks[ids].flatten(1).bool().cuda(),descriptors=descriptors[ids].cuda())
        view['mapping_image']=mapping;view['mass_image']=mass;view['valid_image']=valid
        if model.image_observed is not None:
            image_weight=weights*model.image_observed[gid];image_mass=torch.zeros_like(total).scatter_add_(0,pid,image_weight)
            view['mapping_image']=torch.sparse_coo_tensor(torch.stack([pid,gid]),image_weight/image_mass[pid].clamp_min(1e-8),(pixels,n)).coalesce()
            view['mass_image']=image_mass;view['valid_image']=(image_mass/total.clamp_min(1e-8)>.5)&(total>1e-8)
        views.append(view);tasks=[[],[]]
        if not a.image_only and frame in raw['frames']:
            j=raw['frames'].index(frame)
            for qi in range(raw['probability'].shape[1]):
                target=raw['probability'][j,qi].flatten()>.5
                if raw['scores'][j,qi]>=.7 and target.sum()>=20:tasks[0].append((len(views)-1,text[qi:qi+1],target.cuda(), 'text'))
        for ri in range(len(ids)):
            target=view['masks'][ri]
            if target.sum()>=8:tasks[1].append((len(views)-1,view['descriptors'][ri:ri+1],target,'region'))
        if frame in training_frames:
            for kind in range(2):train_tasks[kind].extend(tasks[kind])
        elif frame in validation_frames:
            for kind in range(2):
                stride=max(1,len(tasks[kind])//4);validation.extend(tasks[kind][::stride][:4])
    validation=validation[:32]
    def install():field.weights=logits.masked_fill(~allowed,-30).softmax(2)*allowed.any(2,keepdim=True)
    if a.key.startswith('scannet_'):
        from radio_gs.querying.source_accepted_category import source_accepted_category_logits
        scope_receipt=json.loads((root/'bounded_fallback_withheld/report.json').read_text())
        admission_receipt=json.loads((Path(data['base_field']).parent/'report.json').read_text())
    def probability(q,kind):
        if kind=='text' and a.key.startswith('scannet_'):
            native=model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(q)-model.native_query_thresholds['text'](q))
            value=source_accepted_category_logits(native,lambda:model.logits(q,'text'),scope_receipt=scope_receipt,admission_receipt=admission_receipt,max_correction=1.)[0]
            return value[:,0].sigmoid()*model.observed
        return model.retrieve(q,'text' if kind=='text' else 'image')['scores'][:,0]
    def observation(view,kind):
        suffix='' if kind=='text' else '_image'
        return view['mapping'+suffix],view['mass'+suffix],view['valid'+suffix]
    def alpha(view,gate):
        _,image,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten()*gate,colors=torch.ones(n,1,device='cuda'),viewmats=view['camera'][None],Ks=view['k'][None],width=w,height=h,near_plane=auth['metadata']['near_plane'],far_plane=auth['metadata']['far_plane'],packed=False)
        return image[0,...,0].flatten()
    def qualified(view,q,target,visible):
        similarity=(view['descriptors']@F.normalize(q,dim=1).T)[:,0]
        response=similarity[:,None].expand_as(view['masks']).masked_fill(~view['masks'],-1).max(0).values
        floor=torch.quantile(response[target],.25)-.03
        negative=view['masks'].any(0)&(response<floor)&~target
        return target|(negative&visible),negative
    @torch.no_grad()
    def evaluate():
        install();rows=[]
        for vi,q,target,kind in validation:
            view=views[vi];mapping,mass,visible=observation(view,kind);s=probability(q,kind);full=(torch.sparse.mm(mapping,s[:,None])[:,0]>.5)&visible;selected=(alpha(view,(s>.5).float())*255+.5).floor()>10
            def iou(p):return float((p&target).sum()/(p|target).sum().clamp_min(1))
            rows.append((iou(full),iou(selected),kind))
        by_kind={k:dict(full_iou=sum(r[0] for r in rows if r[2]==k)/sum(r[2]==k for r in rows),selected_iou=sum(r[1] for r in rows if r[2]==k)/sum(r[2]==k for r in rows)) for k in sorted(set(r[2] for r in rows))}
        return dict(full_iou=sum(r[0] for r in rows)/len(rows),selected_iou=sum(r[1] for r in rows)/len(rows),count=len(rows),by_kind=by_kind)
    history=[dict(step=0,source_validation=evaluate())];print(a.key,history[-1],flush=True)
    for step in range(a.steps):
        kind=1 if a.image_only else step%2;vi,q,target,query_kind= train_tasks[kind][int(torch.randint(len(train_tasks[kind]),(1,)))];view=views[vi];mapping,mass,visible=observation(view,query_kind);install();s=probability(q,query_kind);full=torch.sparse.mm(mapping,s[:,None])[:,0];valid,negative=qualified(view,q,target,visible)
        full_loss=balanced_mask_loss(full[:,None],target[:,None].float(),valid&visible)
        loss=full_loss
        if a.mode in ('joint','hard'):
            gate=(((s>.5).float()+(s-s.detach())) if a.mode=='hard' else ((s-.25)/.5).clamp(0,1))
            silhouette=alpha(view,gate);selected= silhouette/(silhouette+10.5/255)
            visible_mass=torch.sparse.mm(mapping,gate[:,None])[:,0]*mass;visibility=(visible_mass.detach()/silhouette.detach().clamp_min(1e-8)).clamp(0,1)
            selected_valid=target|(negative&visible&(visibility>=.5))
            loss=.5*(full_loss+balanced_mask_loss(selected[:,None],target[:,None].float(),selected_valid))
        active=torch.unique(mapping.indices()[1]);weights_now=field.weights[active];prior=initial[active]
        kl=(weights_now*(weights_now.clamp_min(1e-8).log()-prior.clamp_min(1e-8).log())).sum(2).mean();loss=loss+.01*kl
        opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:
            row=dict(step=step+1,loss=float(loss.detach()),source_validation=evaluate());history.append(row);print(a.key,row,flush=True)
    with torch.no_grad():install();data['weights']=field.weights.detach().cpu().half()
    if a.train_scope:data['scope_state']={k:v.detach().cpu() for k,v in model.grounded_transfer.state_dict().items()}
    out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt')
    if not a.image_only:torch.save(calibration,out/'calibration.pt')
    report=dict(key=a.key,mode=a.mode,scope_trained=a.train_scope,steps=a.steps,text_tasks=len(train_tasks[0]),region_tasks=len(train_tasks[1]),train_frames=training_frames,validation_frames=validation_frames,geometry_frozen=True,native_codebook_frozen=True,original_anchor_frozen=True,history=history,benchmark_supervision=False)
    (out/'training.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
