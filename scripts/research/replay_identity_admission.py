"""Revise identity admission without repeating feature extraction.

No matching native region means no identity contradiction; preserve the original
SAM3 evidence in that case. Matched low-native instances alone become unknown.
"""
import argparse,json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--identity-record',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);inputs=json.loads(Path(a.inputs).read_text());config=json.loads(Path(inputs['scene']).read_text())
    record=json.loads(Path(a.identity_record).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    counts=torch.zeros(len(record['queries']),3,dtype=torch.long)
    for i,frame in enumerate(config['source_frames']):
        d=torch.load(Path(inputs['detections'])/f'identity_{frame}.pt',map_location='cpu',weights_only=False)
        if i%record['heldout_stride']:
            for qi,r in enumerate(d['instances']):
                match=record['matches'][str(frame)][qi]
                if not match:continue
                keep=torch.tensor([m['region_iou']<record['minimum_region_iou'] or m['native_score']>=record['queries'][qi]['threshold'] for m in match])
                r['unknown_masks']=r['masks'][~keep]
                for key in ['scores','masks']:r[key]=r[key][keep]
                counts[qi]+=torch.tensor([len(keep),int(keep.sum()),int((~keep).sum())])
        torch.save(d,out/f'identity_{frame}.pt')
    updated=dict(inputs,detections=str(out),evidence_contract='only matched low-native identity veto; unmatched source instances preserve original detector evidence; rejected extents unknown')
    (out/'inputs.json').write_text(json.dumps(updated,indent=2)+'\n')
    (out/'admission.json').write_text(json.dumps(dict(identity_record=a.identity_record,unmatched_policy='retain original detector evidence',queries=[dict(query=q['query'],original=int(counts[i,0]),admitted=int(counts[i,1]),unknown=int(counts[i,2])) for i,q in enumerate(record['queries'])]),indent=2)+'\n')


if __name__=='__main__':main()
