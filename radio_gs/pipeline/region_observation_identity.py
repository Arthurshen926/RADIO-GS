"""Encode individual source observations before identity/extent aggregation.

Only exactly identical binary footprints coalesce within a view. Cross-view
links use reciprocal, co-visible geometry, without semantic labels or features.
They transport identity evidence once; they never union or grow object masks.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from radio_gs.models.pretrained_mask_adapter import PretrainedMaskAdapter
from radio_gs.pipeline.category_union_adapter import CHECKPOINT
from radio_gs.querying.typed_evidence import visible_operator


def region_nodes(record,union=False):
    h,w=record['raster_hw'];c=len(record['queries']);nodes=[];keys={}
    for qi,row in enumerate(record['instances']):
        masks=row['masks']>.5;scores=row['scores']
        if union and len(masks):masks=masks.any(0,keepdim=True);scores=scores.max()[None]
        for mask,score in zip(masks,scores):
            if not mask.any():continue
            key=np.packbits(mask.numpy().reshape(-1)).tobytes()
            if key not in keys:
                keys[key]=len(nodes);nodes.append(dict(mask=mask,emitting=torch.zeros(c)))
            item=nodes[keys[key]];item['emitting'][qi]=max(float(item['emitting'][qi]),float(score))
    if not nodes:return dict(masks=torch.zeros(0,h,w,dtype=torch.bool),emitting=torch.zeros(0,c),confidence=torch.zeros(0))
    score=torch.stack([r['emitting'] for r in nodes])
    return dict(masks=torch.stack([r['mask'] for r in nodes]),emitting=score/score.sum(1,keepdim=True),confidence=score.max(1).values)


@torch.inference_mode()
def build_links(records,root,count_elements):
    edges=[];weights=[];total=0;offsets=[];previous=None;reports=[]
    for view in records:
        row=torch.load(root/f"frame_{view['frame']:05d}.pt",map_location='cpu');n=len(row['instance']['masks'])
        offsets.append(total)
        membership=torch.zeros(count_elements,n,device='cuda')
        membership[row['element_rows'].cuda().long(),row['region_columns'].cuda().long()]=1
        visible=torch.zeros(count_elements,device='cuda',dtype=torch.bool);visible[row['visible_element_ids'].cuda().long()]=True
        if previous is not None and n and previous['membership'].shape[1]:
            old=previous['membership'];intersection=old.T@membership
            size_old=old[visible].sum(0);size_new=membership[previous['visible']].sum(0)
            iou=intersection/(size_old[:,None]+size_new[None]-intersection).clamp_min(1)
            shared=(size_old/old.sum(0).clamp_min(1)>=.5)[:,None]&(size_new/membership.sum(0).clamp_min(1)>=.5)[None]
            iou*=shared;best_new=iou.argmax(1);best_old=iou.argmax(0);oi=torch.arange(len(best_new),device='cuda')
            admitted=(best_old[best_new]==oi)&(iou[oi,best_new]>=.75)
            oi=oi[admitted];ni=best_new[admitted]
            edges.append(torch.stack([oi+previous['offset'],ni+total],1).cpu());weights.append(iou[oi,ni].cpu())
            reports.append(dict(first=previous['frame'],second=view['frame'],reciprocal_matches=len(oi)))
        previous=dict(membership=membership,visible=visible,offset=total,frame=view['frame']);total+=n
    graph=dict(edges=torch.cat(edges) if edges else torch.zeros(0,2,dtype=torch.long),
        weights=torch.cat(weights) if weights else torch.zeros(0),offsets=offsets,node_count=total,
        source_frames=[v['frame'] for v in records],matching='reciprocal adjacent-view co-visible IoU >= .75; each region >= .5 shared visibility',
        semantic_features_used_for_matching=False,transitive_merging=False,ground_truth_read=False,records=reports)
    torch.save(graph,root/'graph.pt');return graph


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);started=time.monotonic();scene=json.loads(Path(a.scene).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    field=torch.load(scene['native_observation_field'],map_location='cpu');observed=field['supported'].cuda();del field
    teacher=PretrainedMaskAdapter(CHECKPOINT);records=sorted(scene['source_evidence']['views'],key=lambda v:v['frame']);count_instance=count_union=0
    for view in records:
        data=torch.load(view['identity'],map_location='cpu');instance=region_nodes(data);union=region_nodes(data,True)
        ni,nu=len(instance['masks']),len(union['masks']);h,w=data['raster_hw']
        if ni+nu:
            rgb=torch.from_numpy(np.array(Image.open(view['source_rgb']).convert('RGB'),copy=True)).permute(2,0,1).float()/255
            descriptor=teacher.encode(rgb,torch.cat([instance['masks'],union['masks']]))['mask_adapter'].half().cpu()
        else:descriptor=torch.zeros(0,768,dtype=torch.float16)
        instance['descriptors']=descriptor[:ni];union['descriptors']=descriptor[ni:]
        op,_,visible=visible_operator(torch.load(view['contributions'],map_location='cpu'),observed,h*w,False)
        visibility=torch.sparse.mm(op.T,visible.float())[:,0]
        if ni:
            membership=torch.sparse.mm(op.T,instance['masks'].flatten(1).T.float().cuda()*visible)/visibility[:,None].clamp_min(1e-8)
            element,region=torch.where(membership>.5)
        else:element=region=torch.zeros(0,device='cuda',dtype=torch.long)
        torch.save(dict(instance=instance,union=union,queries=data['queries'],frame=view['frame'],boundary_radius=data.get('boundary_radius',1),
            element_rows=element.cpu().int(),region_columns=region.cpu().int(),visible_element_ids=torch.where(visibility>1e-8)[0].cpu().int(),
            geometry=scene['geometry'],query_conditioned_source_asset=True,ground_truth_read=False),out/f"frame_{view['frame']:05d}.pt")
        count_instance+=ni;count_union+=nu;print(scene['scene'],view['frame'],ni,nu,flush=True)
    del teacher;torch.cuda.empty_cache();graph=build_links(records,out,len(observed))
    (out/'build.json').write_text(json.dumps(dict(scene=a.scene,views=len(records),instances=count_instance,unions=count_union,
        correspondence_edges=len(graph['edges']),correspondence_uses_semantics=False,source_geometry_frozen=True,
        source_rgb_allowed=True,ground_truth_read=False,new_detector_calls=0,checkpoint=CHECKPOINT,
        matched_encodings_share_backbone=True,seconds=time.monotonic()-started),indent=2)+'\n')
    print('COMPLETE',scene['scene'],count_instance,'regions',len(graph['edges']),'links',flush=True)

if __name__=='__main__':main()
