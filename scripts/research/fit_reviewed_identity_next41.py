"""Paired identity-supervision experiment on the current frozen geometry.

Reuses assistant-reviewed source referents, not human-certified mask boundaries.
Each held source view is absent from fitting and automatic correspondence.
The original map has seen these views. No benchmark labels or query list used.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.partial_extent_supervision import extent_measures


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--reviewed-enclosures',action='store_true');args=p.parse_args()
    enclosures=json.loads(Path('docs/reviews/2026-09-28-source-instance-enclosures.json').read_text()) if args.reviewed_enclosures else None
    torch.set_num_threads(4)
    review=json.loads(Path('docs/reviews/2026-09-18-typed-multiview-scope-review.json').read_text())
    output=[]
    for scene in ['figurines','ramen']:
        tracks=[t for t in review['tracks'] if t['scene']==scene]
        data=load(f'output/taskwise_scope_composition_next38/lerf_{scene}/field.pt');root=Path(data['root'])
        parent=load(data['base_field']);model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.eval()
        evidence=load(tracks[0]['evidence_path']);masks=evidence['masks'].flatten(1).cuda().bool();mh,mw=evidence['masks'].shape[-2:]
        yy,xx=torch.meshgrid((torch.arange(mh,device='cuda')+.5)/mh,(torch.arange(mw,device='cuda')+.5)/mw,indexing='ij')
        desc=F.normalize(evidence['native_descriptors'].float().cuda(),dim=1)
        info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
        views={};n=len(model.observed);pixels=masks.shape[1]
        for frame in sorted({o['frame'] for t in tracks for o in t['observations']}):
            index=info['frames'].index(frame);hit=load(root/auth['views'][index]['relative_path'])
            pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda()
            total=torch.zeros(pixels,device='cuda').scatter_add_(0,pid,w);w*=model.observed_for('image')[gid]
            mass=torch.zeros_like(total).scatter_add_(0,pid,w)
            op=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce()
            candidates=[i for i,r in enumerate(evidence['records']) if r['frame']==frame and r['confidence']>=.7 and int(masks[i].sum())>=10]
            views[frame]=dict(op=op,adjoint=op.T.coalesce(),visible=(mass/total.clamp_min(1e-8)>.5)&(total>1e-8),candidates=candidates)
        def negative_mask(track,frame,use_enclosure=False):
            negative=torch.zeros(pixels,dtype=torch.bool,device='cuda')
            # Only explicitly reviewed distinct instances yield negative identity
            # evidence. Whole/part relationships never create exclusion labels.
            partners=[]
            for rel in review['relations']:
                if rel['relation']=='distinct_same_category_instances' and track['id'] in [rel['left'],rel['right']]:
                    partners.append(rel['right'] if rel['left']==track['id'] else rel['left'])
            for t in tracks:
                if t['id'] in partners:
                    for o in t['observations']:
                        if o['frame']==frame:negative|=masks[o['region']]
            if track['id']=='front_chopsticks' and frame==76:negative|=masks[528]
            if use_enclosure:
                x0,y0,x1,y1=enclosures['boxes'][track['id']][str(frame)]
                negative|=((xx<x0)|(xx>x1)|(yy<y0)|(yy>y1)).flatten()
            return negative
        for track in tracks:
            for target in track['observations']:
                sources=[o for o in track['observations'] if o['frame']!=target['frame']]
                ref=sources[0];q=desc[ref['region']:ref['region']+1];prior=model.retrieve(q,'image')['scores'][:,0]
                tv=views[target['frame']];positive=masks[target['region']];negative=negative_mask(track,target['frame'],args.reviewed_enclosures)&~positive
                example=dict(target=positive[:,None],known=((positive|negative)&tv['visible'])[:,None],visible=tv['visible'])
                row=dict(scene=scene,track=track['id'],granularity=track['granularity'],held_frame=target['frame'],reference_frame=ref['frame'],results={},automatic_regions=[])
                def measure(value):
                    prob=torch.sparse.mm(tv['op'],value[:,None])
                    result={k:float(v.mean()) for k,v in extent_measures(prob,example).items()}
                    hard=torch.sparse.mm(tv['op'],(value>.5).float()[:,None])
                    result['visible_selected_mask_iou']=float(extent_measures(hard,example)['closed_mask_iou'].mean())
                    return result
                row['results']['prior']=measure(prior)
                for mode in (['automatic','reviewed','reviewed_visibility_prior','reviewed_bounds','reviewed_bounds_visibility'] if args.reviewed_enclosures else ['automatic','reviewed','reviewed_visibility_prior']):
                    samples=[];den=torch.zeros(n,device='cuda');numerator=torch.zeros_like(den);visible_mass=torch.zeros_like(den)
                    for obs in sources:
                        v=views[obs['frame']];region=obs['region']
                        if mode=='automatic' and obs['frame']!=ref['frame']:
                            ids=v['candidates'];region=ids[int((desc[ids]@q[0]).argmax())]
                        if mode=='automatic':row['automatic_regions'].append(dict(frame=obs['frame'],region=region,reviewed=obs['region']))
                        pos=masks[region];neg=negative_mask(track,obs['frame'],mode.startswith('reviewed_bounds'))&~pos
                        known=(pos|neg)&v['visible'];y=pos.float()
                        mass=torch.sparse.mm(v['adjoint'],known.float()[:,None])[:,0]
                        visible_mass+=torch.sparse.mm(v['adjoint'],v['visible'].float()[:,None])[:,0]
                        numerator+=torch.sparse.mm(v['adjoint'],(known*y)[:,None])[:,0];den+=mass
                        samples.append((v,known,y))
                    value=prior.clone();regularizer=visible_mass if mode in ['reviewed_visibility_prior','reviewed_bounds_visibility'] else den
                    for _ in range(100):
                        grad=torch.zeros_like(value)
                        for v,known,y in samples:
                            residual=(y-torch.sparse.mm(v['op'],value[:,None])[:,0])*known
                            grad+=torch.sparse.mm(v['adjoint'],residual[:,None])[:,0]
                        value.add_(.95*(grad-.1*regularizer*(value-prior))/(den+.1*regularizer).clamp_min(1e-8)).clamp_(0,1)
                    row['results'][mode]=measure(value)
                output.append(row);print(scene,track['id'],target['frame'],row['results'],flush=True)
    result=dict(source_only=True,benchmark_data_used=False,independent_human_ground_truth=False,
                new_experiment_training_use=True,original_review_unchanged=True,
                prior_task='image category relevance; fitting conditions it on an instance reference',
                mask_boundaries='teacher-derived, not certified complete',steps=100,prior_weight=.1,
                views_held_for='correspondence and extent fit, not original map',observations=output)
    result['reviewed_enclosures']=args.reviewed_enclosures
    result['macro']={mode:{metric:sum(r['results'][mode][metric] for r in output)/len(output) for metric in output[0]['results'][mode]} for mode in output[0]['results']}
    out=Path('output/partial_extent_next41');out.mkdir(parents=True,exist_ok=True);(out/('reviewed_identity_enclosures.json' if args.reviewed_enclosures else 'reviewed_identity_visibility.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
