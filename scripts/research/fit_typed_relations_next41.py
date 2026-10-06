"""Joint source-instance extent reconstruction with reviewed typed relations.

This is an opt-in small-source experiment, not benchmark inference. Relations
come from an existing assistant visual review; masks remain teacher estimates.
"""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.partial_extent_supervision import extent_measures


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    review=json.loads(Path('docs/reviews/2026-09-18-typed-multiview-scope-review.json').read_text());rows=[]
    for scene in ['figurines','ramen']:
        tracks=[t for t in review['tracks'] if t['scene']==scene];names=[t['id'] for t in tracks]
        data=load(f'output/taskwise_scope_composition_next38/lerf_{scene}/field.pt');root=Path(data['root']);parent=load(data['base_field'])
        model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.eval()
        evidence=load(tracks[0]['evidence_path']);masks=evidence['masks'].flatten(1).cuda().bool();desc=F.normalize(evidence['native_descriptors'].float().cuda(),dim=1)
        info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());n=len(model.observed);pixels=masks.shape[1];views={}
        frames=sorted({o['frame'] for t in tracks for o in t['observations']})
        for frame in frames:
            hit=load(root/auth['views'][info['frames'].index(frame)]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda()
            total=torch.zeros(pixels,device='cuda').scatter_add_(0,pid,w);w*=model.observed_for('image')[gid];mass=torch.zeros_like(total).scatter_add_(0,pid,w)
            op=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce();views[frame]=(op,op.T.coalesce(),(mass/total.clamp_min(1e-8)>.5)&(total>1e-8))
        relations=[]
        for r in review['relations']:
            if 'parent' in r and r['parent'] in names:relations.append(('contains',names.index(r['parent']),names.index(r['child'])))
            elif 'left' in r and r['left'] in names:relations.append(('exclusive',names.index(r['left']),names.index(r['right'])))
        def labels(j,frame):
            own=[o for o in tracks[j]['observations'] if o['frame']==frame]
            if not own:return None
            pos=masks[own[0]['region']];neg=torch.zeros_like(pos)
            for typ,l,r in relations:
                if typ=='exclusive' and j in [l,r]:
                    other=r if j==l else l
                    for o in tracks[other]['observations']:
                        if o['frame']==frame:neg|=masks[o['region']]
            if names[j]=='front_chopsticks' and frame==76:neg|=masks[528]
            return pos,((pos|neg)&views[frame][2])
        for held in frames:
            refs=[next(o for o in t['observations'] if o['frame']!=held) for t in tracks]
            q=desc[[o['region'] for o in refs]];prior=model.retrieve(q,'image')['scores'];den=torch.zeros_like(prior);examples=[]
            for frame in frames:
                if frame==held:continue
                for j in range(len(tracks)):
                    labels_j=labels(j,frame)
                    if labels_j is None:continue
                    y,k=labels_j;op,adj,_=views[frame];den[:,j]+=torch.sparse.mm(adj,k.float()[:,None])[:,0];examples.append((j,op,adj,y.float(),k))
            active=den>0;results={}
            for mode in ['independent','typed_relations']:
                value=prior.clone()
                for _ in range(100):
                    grad=torch.zeros_like(value)
                    for j,op,adj,y,k in examples:grad[:,j]+=torch.sparse.mm(adj,((y-torch.sparse.mm(op,value[:,j:j+1])[:,0])*k)[:,None])[:,0]
                    value.add_(.95*(grad-.1*den*(value-prior))/(1.1*den).clamp_min(1e-8)).clamp_(0,1)
                    if mode=='typed_relations':
                        for _ in range(4):
                            for typ,l,r in relations:
                                supported=active[:,l]|active[:,r]
                                if typ=='contains':
                                    change=((value[:,r]-value[:,l]).clamp_min(0)/2)*supported
                                    value[:,l]+=change;value[:,r]-=change
                                else:
                                    excess=(value[:,l]+value[:,r]-1).clamp_min(0)/2*supported
                                    value[:,l]-=excess;value[:,r]-=excess
                results[mode]=value
            for j,t in enumerate(tracks):
                lab=labels(j,held)
                if lab is None:continue
                pos,k=lab;op,_,vis=views[held];e=dict(target=pos[:,None],known=k[:,None],visible=vis);record=dict(scene=scene,track=t['id'],held_frame=held,reference_frame=refs[j]['frame'],results={})
                for mode,value in results.items():
                    prob=torch.sparse.mm(op,value[:,j:j+1]);m={k:float(v.mean()) for k,v in extent_measures(prob,e).items()}
                    hard=torch.sparse.mm(op,(value[:,j:j+1]>.5).float());m['visible_selected_mask_iou']=float(extent_measures(hard,e)['closed_mask_iou'].mean());record['results'][mode]=m
                rows.append(record)
            print(scene,held,'done',flush=True)
    result=dict(source_only=True,benchmark_data_used=False,independent_human_ground_truth=False,mask_boundaries_certified=False,
                original_map_saw_held_views=True,relations_source='docs/reviews/2026-09-18-typed-multiview-scope-review.json',observations=rows)
    result['macro']={mode:{k:sum(r['results'][mode][k] for r in rows)/len(rows) for k in rows[0]['results'][mode]} for mode in ['independent','typed_relations']}
    Path('output/partial_extent_next41/typed_relations.json').write_text(json.dumps(result,indent=2)+'\n');print(result['macro'],flush=True)


if __name__=='__main__':main()
