"""Explicit semantic contradiction removes positives, preserving unknown extent."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='output/source_referent_verification')
    p.add_argument('--policies',nargs='+',choices=['veto','affirmative','contradiction'],default=['veto','affirmative']);a=p.parse_args()
    torch.set_num_threads(2);root=Path(a.root);manifest=json.loads((root/'manifest.json').read_text())
    reports=[json.loads(p.read_text()) for p in sorted((root/'judgments').glob('shard_*.json'))]
    results={r['key']:r for report in reports for r in report['results']}
    # All manifest keys are dereferenced below: every source candidate must have
    # a model result. Empty masks carry no image evidence and remain unknown.
    count={}
    for arm in a.policies:
        for cohort_scene,mapping in manifest['mappings'].items():
            cohort,scene=cohort_scene.split(':');inputs=json.loads(Path(mapping['inputs']).read_text())
            dest=root/arm/cohort/scene/'evidence';dest.mkdir(parents=True,exist_ok=True)
            decisions=[]
            for frame,rows in mapping['instances'].items():
                data=torch.load(Path(inputs['detections'])/f'identity_{frame}.pt',weights_only=False,map_location='cpu')
                for qi,(keys,record) in enumerate(zip(rows,data['instances'])):
                    keep=torch.tensor([(not results[k]['balanced_veto'] if arm=='veto' else
                                       max(range(3),key=lambda c:results[k]['class_support'][c])!=1 if arm=='contradiction' else
                                       results[k]['affirmative_admission'])
                                       if k is not None else True for k in keys],dtype=torch.bool)
                    excluded=record['masks'][~keep]
                    old=record.get('unknown_masks')
                    record['unknown_masks']=torch.cat([old,excluded]) if old is not None else excluded
                    for name in ['masks','scores','viewports']:
                        if name in record:record[name]=record[name][keep]
                    decisions.append(dict(frame=int(frame),query=data['queries'][qi],before=len(keep),kept=int(keep.sum())))
                torch.save(data,dest/f'identity_{frame}.pt')
            (dest/'inputs.json').write_text(json.dumps(dict(inputs,detections=str(dest),
                evidence_contract=f'native-admitted masks plus referent {arm}; removed extents are unknown, not new negatives'),indent=2)+'\n')
            (dest/'admission.json').write_text(json.dumps(dict(arm=arm,parent=mapping['inputs'],
                decisions=decisions,scope_replacement=False,unknown_is_negative=False),indent=2)+'\n')
            count[f'{arm}:{cohort_scene}']=dict(before=sum(d['before'] for d in decisions),kept=sum(d['kept'] for d in decisions))
    # Include already completed policies when extending the same fixed judgments.
    for path in root.glob('*/*/*/evidence/admission.json'):
        arm,cohort,scene=path.relative_to(root).parts[:3]
        rows=json.loads(path.read_text())['decisions']
        count[f'{arm}:{cohort}:{scene}']=dict(before=sum(d['before'] for d in rows),kept=sum(d['kept'] for d in rows))
    (root/'verification.json').write_text(json.dumps(dict(candidates=len(results),image_forwards=sum(d['image_forwards'] for d in reports),
        seconds=sum(d['seconds'] for d in reports),label_sensitive=sum(r['label_sensitive'] for r in results.values()),
        counts=count,new_scope_generated=False,independent_ground_truth=False),indent=2)+'\n')
    print(json.dumps(count,indent=2))


if __name__=='__main__':main()
