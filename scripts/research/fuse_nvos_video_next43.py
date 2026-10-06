"""Fuse independent temporal masks through the frozen source renderer."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
    key='nvos_'+('horns_center' if a.scene=='horns_left' else a.scene);base=Path('output/taskwise_scope_composition_next38')/key
    data=torch.load(base/'field.pt',map_location='cpu',weights_only=False);root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    state=torch.load(data['base_field'],map_location='cpu',weights_only=False)['state_dict']
    observed=state.get('image_observed',state['observed']).cuda()
    tracks=torch.load(Path('output/source_video_next43')/a.scene/'tracks.pt',map_location='cpu',weights_only=False)
    prior=torch.load(base/('evaluation_'+a.scene)/'trained.pt',map_location='cpu',weights_only=False)['elements'].float().cuda();n=len(prior);h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    examples=[];records=[]
    for frame,record in tracks['frames'].items():
        if frame==tracks['reference_frame'] or frame not in info['train_frames'] or record['object_logit']<=0:continue
        hit=torch.load(root/auth['views'][info['frames'].index(frame)]['relative_path'],map_location='cpu',weights_only=False);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weight=hit['base_weights'].float().cuda()
        total=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight);weight*=observed[gid]
        mass=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight);op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,n)).coalesce()
        probability=F.interpolate(record['logits'].float()[None,None].cuda(),size=(h,w),mode='bilinear',align_corners=False).sigmoid()
        binary=(probability>.5).float();positive=(-F.max_pool2d(-binary,3,1,1)>.5).flatten()[:,None];negative=(F.max_pool2d(binary,3,1,1)<.5).flatten()[:,None]
        visible=((mass>1e-8)&(mass/total.clamp_min(1e-8)>.5))[:,None];examples.append((op,op.T.coalesce(),probability.flatten()[:,None],positive,negative,visible));records.append(dict(frame=frame,object_logit=record['object_logit'],positive_pixels=int(positive.sum())))
    for mode in ['positive','signed']:
        den=torch.zeros_like(prior);vis=torch.zeros_like(prior)
        for op,adj,y,pos,neg,v in examples:
            known=(pos if mode=='positive' else pos|neg)&v;den+=torch.sparse.mm(adj,known.float());vis+=torch.sparse.mm(adj,v.float())
        value=prior.clone()
        for _ in range(60):
            gradient=torch.zeros_like(value)
            for op,adj,y,pos,neg,v in examples:
                known=(pos if mode=='positive' else pos|neg)&v;residual=(y-torch.sparse.mm(op,value)).clamp(-.25,.25)*known;gradient+=torch.sparse.mm(adj,residual)
            value.add_(.95*(gradient-.1*vis*(value-prior))/(den+.1*vis).clamp_min(1e-8)).clamp_(0,1)
        out=Path('output/source_video_next43')/mode/a.scene;out.mkdir(parents=True,exist_ok=True);torch.save(dict(elements=value.cpu()),out/'elements.pt')
        (out/'fusion.json').write_text(json.dumps(dict(scene=a.scene,mode=mode,source_records=records,steps=60,prior_weight=.1,target_rgb_used=False,negative_prompt_used=False,background_policy='unknown' if mode=='positive' else 'eroded tracker background only when object presence logit positive; pseudo-negative, not verified truth'),indent=2)+'\n')


if __name__=='__main__':main()
