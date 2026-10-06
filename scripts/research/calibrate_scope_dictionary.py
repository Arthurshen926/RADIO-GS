"""Refit native query thresholds to the new field, using source masks only."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.models.native_scope_dictionary import NativeScopeDictionary
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_threshold import fit_threshold_ridge


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--base',default='output/scope_hierarchy_full_next33');a=p.parse_args();torch.set_num_threads(4)
    out=Path(a.base)/a.key;data=load(out/'field.pt');root=Path(data['root']);parent=load(data['base_field'])
    model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.eval()
    field=NativeScopeDictionary(data)
    old=json.loads((root/'full_source_query_threshold_control/report.json').read_text());episodes=load(root/'episodes.pt')
    raw=load(root/'source_grounded_text.pt');authority=json.loads((root/'responsibility.json').read_text());info=json.loads((root/'teachers.json').read_text())
    ids=[r['query_index'] for r in old['source_query_fits']];query=episodes['queries']['text'][ids].float().cuda();count=len(ids)
    if 'region_native_features' in data:
        from radio_gs.querying.region_native_field import attach_region_native_field
        attach_region_native_field(model,data)
        cosine=model.native_cosine(query)
    else:
        cosine=field.scores(query,pool='anchored',anchor=(model.latent[:,:64],model.native_decoder.weight))
    previous=model.native_query_thresholds['text'](query);best=previous.clone();scale=model.readout_log_scale['text'].clamp(-4,6).exp()
    # Refit only a local correction to an already source-calibrated readout.
    width=(cosine[model.observed].amax(0)-cosine[model.observed].amin(0))/8
    history=[]
    for iteration in range(2):
        grid=best[:,None]+width[:,None]*torch.linspace(-1,1,17,device='cuda')[None]
        response=torch.sigmoid(scale*(cosine[:,:,None]-grid[None]))*model.observed[:,None,None]
        colors=response.flatten(1)
        hits=torch.zeros(count,17,device='cuda');unions=torch.zeros_like(hits)
        for vi,frame in enumerate(raw['frames']):
            source_index=info['frames'].index(frame);hit=load(root/authority['views'][source_index]['relative_path'])
            pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda()
            h,width_pixels=raw['probability'].shape[-2:];pixels=h*width_pixels
            observed_weight=w*model.observed[gid]
            mass=torch.zeros(pixels,device='cuda').scatter_add_(0,pid,observed_weight)
            total=torch.zeros_like(mass).scatter_add_(0,pid,w)
            operator=torch.sparse_coo_tensor(torch.stack([pid,gid]),observed_weight/mass[pid].clamp_min(1e-8),(pixels,len(model.observed))).coalesce()
            pred=torch.sparse.mm(operator,colors).reshape(pixels,count,17)>.5
            valid=(mass/total.clamp_min(1e-8)>.5)&(total>1e-8)
            pred&=valid[:,None,None]
            target=(raw['probability'][vi,ids].cuda()>.5).flatten(1).T
            eligible=(raw['scores'][vi,ids].cuda()>=.7)&(target.sum(0)>=20)
            hits+=(pred&target[...,None]).sum(0)*eligible[:,None]
            unions+=(pred|target[...,None]).sum(0)*eligible[:,None]
        iou=hits/unions.clamp_min(1)
        best=grid[torch.arange(count,device='cuda'),iou.argmax(1)]
        history.append(dict(iteration=iteration,source_mask_iou=float(iou.max(1).values.mean())))
        width/=8
        print(a.key,history[-1],flush=True)
    weight,bias=fit_threshold_ridge(query,best-previous,1.)
    torch.save(dict(weight=weight.cpu(),bias=bias.cpu(),max_delta=float((best-previous).abs().max())),out/'calibration.pt')
    (out/'calibration.json').write_text(json.dumps(dict(source_queries=count,source_frames=len(raw['frames']),history=history,source_only=True),indent=2)+'\n')


if __name__=='__main__':main()
