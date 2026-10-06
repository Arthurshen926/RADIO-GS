"""Apply detector-conditioned identity evidence to explicit source observations."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.querying.source_identity import native_identity_posterior


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--identity-record',required=True);p.add_argument('--output',required=True);p.add_argument('--force-two-groups',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);inputs=json.loads(Path(a.inputs).read_text());scene=json.loads(Path(inputs['scene']).read_text())
    record=json.loads(Path(a.identity_record).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    reliability={};models=[]
    for qi,q in enumerate(record['queries']):
        entries=[(frame,mi,m['native_score']) for frame,rows in record['matches'].items() for mi,m in enumerate(rows[qi]) if (m['identity_score_valid'] if 'identity_score_valid' in m else m['region_iou']>=record['minimum_region_iou'])]
        posterior,info=native_identity_posterior([s for _,_,s in entries],compare_unimodal=not a.force_two_groups)
        for (frame,mi,_),r in zip(entries,posterior):reliability[(int(frame),qi,mi)]=float(r)
        models.append(dict(query=q['query'],**info))
    counts=torch.zeros(len(record['queries']),3,dtype=torch.long)
    for i,frame in enumerate(scene['source_frames']):
        d=torch.load(Path(inputs['detections'])/f'identity_{frame}.pt',map_location='cpu',weights_only=False)
        if not record['heldout_stride'] or i%record['heldout_stride']:
            for qi,r in enumerate(d['instances']):
                if not len(r['scores']):continue
                confidence=torch.tensor([reliability.get((frame,qi,mi),1.) for mi in range(len(r['scores']))]);keep=confidence>=.5
                r['unknown_masks']=r['masks'][~keep]
                for key in ['scores','masks','viewports']:
                    if key in r:r[key]=r[key][keep]
                counts[qi]+=torch.tensor([len(keep),int(keep.sum()),int((~keep).sum())])
        torch.save(d,out/f'identity_{frame}.pt')
    updated=dict(inputs,detections=str(out),evidence_contract='detector-conditioned native density evidence; unimodal/unmatched not vetoed; rejected instances unknown')
    (out/'inputs.json').write_text(json.dumps(updated,indent=2)+'\n')
    (out/'admission.json').write_text(json.dumps(dict(identity_record=a.identity_record,force_two_groups=a.force_two_groups,heldout_stride=record['heldout_stride'],posterior_is_calibrated_correctness=False,models=models,queries=[dict(query=q['query'],original=int(counts[i,0]),admitted=int(counts[i,1]),unknown=int(counts[i,2])) for i,q in enumerate(record['queries'])]),indent=2)+'\n')


if __name__=='__main__':main()
