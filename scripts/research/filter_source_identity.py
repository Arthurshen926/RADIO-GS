"""Bounded native identity admission before detector-instance union.

Use the upper 5% of train-source native region scores for each literal query.
A detector mask needs a region IoU >= .5 and a native score above that threshold.
Rejected masks become unknown, not complement negatives. No target labels,
competing query text, new encoders, or historical predictions enter this rule.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--heldout-stride',type=int,default=5);p.add_argument('--record-only',action='store_true',help='Extract native mask correspondences without applying the historical global quantile filter');a=p.parse_args()
    torch.set_num_threads(2);inputs=json.loads(Path(a.inputs).read_text());scene=json.loads(Path(inputs['scene']).read_text())
    response=load(inputs['response']);queries=F.normalize(response['encodings'].float(),dim=-1)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);score_banks=[];matches={}
    for i,frame in enumerate(scene['source_frames']):
        if a.heldout_stride and i%a.heldout_stride==0:continue
        d=load(Path(inputs['detections'])/f'identity_{frame}.pt');desc=[];masks=[]
        for directory in scene['native_region_directories']:
            native=load(Path(directory)/f'frame_{frame:05d}.pt')
            if not len(native['descriptors']):continue
            sam=load(native['sam_path']);unpacked=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])
            chosen=unpacked[native['native_proposal_indices'].numpy()]
            masks.append(F.interpolate(torch.from_numpy(chosen).float()[:,None],size=(96,128),mode='nearest')[:,0].flatten(1).bool())
            desc.append(F.normalize(native['descriptors'].float(),dim=-1))
        masks=torch.cat(masks);scores=torch.cat(desc)@queries.T;score_banks.append(scores)
        records=[]
        for qi,r in enumerate(d['instances']):
            if not len(r['scores']):records.append([]);continue
            proposals=F.interpolate(r['masks'].float()[:,None],size=(96,128),mode='bilinear',align_corners=False)[:,0].flatten(1)>.5
            row=[]
            for mask in proposals:
                iou=(masks&mask).sum(1)/(masks|mask).sum(1).clamp_min(1);index=int(iou.argmax())
                row.append(dict(region_iou=float(iou[index]),native_score=float(scores[index,qi])))
            records.append(row)
        matches[frame]=records
    threshold=torch.quantile(torch.cat(score_banks),.95,dim=0);counts=torch.zeros(len(queries),3,dtype=torch.long)
    record=dict(quantile=.95,minimum_region_iou=.5,heldout_stride=a.heldout_stride,training_only_threshold=True,queries=[dict(query=t,threshold=float(threshold[q])) for q,t in enumerate(response['texts'])],matches=matches)
    if a.record_only:
        (out/'identity.json').write_text(json.dumps(record,indent=2)+'\n')
        print(scene['scene'],'native identity correspondences ready',flush=True)
        return
    for i,frame in enumerate(scene['source_frames']):
        d=load(Path(inputs['detections'])/f'identity_{frame}.pt')
        if not a.heldout_stride or i%a.heldout_stride:
            for qi,r in enumerate(d['instances']):
                match=matches[frame][qi]
                if not match:continue
                keep=torch.tensor([m['region_iou']>=.5 and m['native_score']>=float(threshold[qi]) for m in match])
                # Save original rejected extent at the detector's stored resolution.
                r['unknown_masks']=r['masks'][~keep]
                for key in ['scores','masks']:
                    r[key]=r[key][keep]
                counts[qi]+=torch.tensor([len(keep),int(keep.sum()),int((~keep).sum())])
        torch.save(d,out/f'identity_{frame}.pt')
    updated=dict(inputs,detections=str(out),evidence_contract='native identity admitted detector positives; rejected extents unknown; every fifth view unchanged and diagnostic only')
    (out/'inputs.json').write_text(json.dumps(updated,indent=2)+'\n')
    (out/'identity.json').write_text(json.dumps(dict(quantile=.95,minimum_region_iou=.5,heldout_stride=a.heldout_stride,training_only_threshold=True,queries=[dict(query=t,threshold=float(threshold[q]),original=int(counts[q,0]),admitted=int(counts[q,1]),unknown=int(counts[q,2])) for q,t in enumerate(response['texts'])],matches=matches),indent=2)+'\n')
    print(scene['scene'],'identity evidence ready',flush=True)


if __name__=='__main__':main()
