"""Containment ambiguity becomes unknown, never an added foreground label.

Two fixed controls: nearest structural parent; the same parent only when its
native query similarity is no lower than the matched child's similarity.
"""
import argparse,json,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--scope-assignment',action='store_true',help='Separate control: query-preferred containing parent supplies positive scope');a=p.parse_args();torch.set_num_threads(4)
    inputs=json.loads(Path(a.inputs).read_text());scene=json.loads(Path(inputs['scene']).read_text());response=load(inputs['response']);queries=F.normalize(response['encodings'].float(),dim=-1);root=Path(a.output);arms=['assignment'] if a.scope_assignment else ['structural','semantic'];counts={arm:torch.zeros(len(queries),dtype=torch.long) for arm in arms};records=[]
    for arm in arms:(root/arm).mkdir(parents=True,exist_ok=True)
    for frame in scene['source_frames']:
        data=load(Path(inputs['detections'])/f'identity_{frame}.pt');maskparts=[];fullparts=[];desc=[]
        for directory in scene['native_region_directories']:
            native=load(Path(directory)/f'frame_{frame:05d}.pt')
            if not len(native['descriptors']):continue
            sam=load(native['sam_path']);unpacked=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);chosen=unpacked[native['native_proposal_indices'].numpy()];maskparts.append(F.interpolate(torch.from_numpy(chosen).float()[:,None],size=(96,128),mode='nearest')[:,0].flatten(1).bool());desc.append(F.normalize(native['descriptors'].float(),dim=-1))
            if a.scope_assignment:fullparts.append(F.interpolate(torch.from_numpy(chosen).float()[:,None],size=(192,256),mode='bilinear',align_corners=False)[:,0])
        masks=torch.cat(maskparts);scores=torch.cat(desc)@queries.T;areas=masks.sum(1);variants={arm:copy.deepcopy(data) for arm in arms}
        if a.scope_assignment:fullmasks=torch.cat(fullparts)
        for qi,r in enumerate(data['instances']):
            if not len(r['scores']):continue
            proposals=F.interpolate(r['masks'].float()[:,None],size=(96,128),mode='bilinear',align_corners=False)[:,0].flatten(1)>.5;chosen={arm:[] for arm in arms}
            for mi,mask in enumerate(proposals):
                area=int(mask.sum())
                if not area:continue
                intersection=(masks&mask).sum(1);iou=intersection/(areas+area-intersection).clamp_min(1);child=int(iou.argmax())
                if iou[child]<.5:continue
                candidates=(intersection/area>=.9)&(areas>=1.25*area)
                if not candidates.any():continue
                index=int(torch.where(candidates,areas,torch.iinfo(torch.int64).max).argmin());semantic=bool(scores[index,qi]>=scores[child,qi])
                if a.scope_assignment:
                    if semantic:
                        parent=F.interpolate(fullmasks[index][None,None],size=r['masks'].shape[-2:],mode='bilinear',align_corners=False)[0,0].to(r['masks'].dtype)
                        variants['assignment']['instances'][qi]['masks'][mi]=torch.maximum(r['masks'][mi],parent)
                        chosen['assignment'].append(index)
                else:
                    chosen['structural'].append(index)
                    if semantic:chosen['semantic'].append(index)
                records.append(dict(frame=frame,query=qi,instance=mi,parent_area_ratio=float(areas[index]/area),containment=float(intersection[index]/area),child_match_iou=float(iou[child]),parent_score=float(scores[index,qi]),child_score=float(scores[child,qi]),semantic_admitted=semantic))
            for arm,indices in chosen.items():
                if not indices:continue
                if a.scope_assignment:
                    counts[arm][qi]+=1
                    continue
                unknown=masks[indices].any(0).reshape(1,1,96,128).float();unknown=F.interpolate(unknown,size=r['masks'].shape[-2:],mode='nearest')[0];old=r.get('unknown_masks');variants[arm]['instances'][qi]['unknown_masks']=torch.cat([old,unknown.to(r['masks'].dtype)],0) if old is not None and len(old) else unknown.to(r['masks'].dtype);counts[arm][qi]+=1
        for arm in arms:torch.save(variants[arm],root/arm/f'identity_{frame}.pt')
    for arm in arms:
        contract='identity-admitted instance augmented by nearest containing native region only if query cosine >= matched child; original unknown identity regions retained' if a.scope_assignment else f'identity-admitted positives unchanged; nearest {arm} parent is unknown, not added positive'
        manifest=dict(inputs,detections=str(root/arm),evidence_contract=contract)
        (root/arm/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'containment.json').write_text(json.dumps(dict(inputs=a.inputs,policy='child-match IoU>=.5;parent contains>=.9 detector mask;area>=1.25x;choose smallest;semantic requires parent cosine>=child cosine',source_only=True,new_vfm_calls=0,positive_masks_changed=a.scope_assignment,counts={a:v.tolist() for a,v in counts.items()},queries=response['texts'],records=records),indent=2)+'\n');print(scene['scene'],{a:int(v.sum()) for a,v in counts.items()},flush=True)
if __name__=='__main__':main()
