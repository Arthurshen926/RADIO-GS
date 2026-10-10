"""Post-prediction full source-image diagnosis, never an inference input."""
import argparse
from pathlib import Path
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.relation_training import SourceRelations
from radio_gs.query_field.scope_exclusion import union_exclusion
from radio_gs.query_field.part_conditioning import native_region_probability
from radio_gs.query_field.geometry import selected_alpha
from radio_gs.query_field.joint_exclusion import CUTOFF
from scripts.research.run_query_field_joint_exclusion import ROOT,ARMS,read,write


def run(index):
    read(ROOT/'predictions_complete.json');job=read(ROOT/'prediction_config.json')['jobs'][index]
    field,manifest=load_field(job['field'],'cuda')
    data=torch.load(ROOT/'predictions'/job['name']/'predictions.pt',weights_only=True)
    source=SourceRelations(job['observations'],field.observed);construction=read(job['construction'])
    assert len(source.views)==len(field.scope_constraints.views)
    available=field.observed|field.semantic_measure.supported
    mass=torch.sparse.mm(field.evidence.matrix('positive').transpose(0,1),available.float()[:,None]).flatten()
    for v,r in zip(source.views,construction['source_records']):
        assert v['frame']==r['frame'] and v['original_regions']==r['regions']
        raw=torch.load(Path(job['observations'])/f"view_{v['frame']:05d}.pt",weights_only=True)
        v['masks']=raw['masks'].bool();v['all_scales']=raw['scales_m'];v['height'],v['width']=v['masks'].shape[-2:]
        v['visible']=raw['raw_mass'].reshape(v['height'],v['width'])>0
        h,w=v['height'],v['width'];ys=((torch.arange(16)+.5)*h/16).long();xs=((torch.arange(16)+.5)*w/16).long()
        sampled=torch.zeros(h*w,dtype=torch.bool);sampled[(ys[:,None]*w+xs[None,:]).flatten()]=True;v['sampled']=sampled.cuda()
    records=[]
    for qi,text in enumerate(data['texts']):
        prob=native_region_probability(field.evidence,data['native_query_vectors'][qi].cuda(),data['generic_negative_vectors'].cuda())
        accepted=(prob>.5)&(mass>0);offset=0
        for vi,v in enumerate(source.views):
            chosen=accepted[offset:offset+len(v['positive'])].cpu();offset+=len(chosen)
            if not bool(chosen.any()):continue
            positive=v['positive'][chosen].any(0);old=v['negative'][chosen].all(0)
            full=torch.zeros(len(v['masks']),dtype=torch.bool);full[torch.tensor(v['original_regions'])[chosen]]=True
            negative,_=union_exclusion(v['masks'],v['all_scales'],full,v['visible'],old.reshape(v['height'],v['width']))
            positive=positive.cuda();negative=negative.flatten().cuda();pred={}
            for arm in ARMS:
                pred[arm]=selected_alpha(field,data['selected_by_control'][arm][:,qi].cuda(),v['camera'],v['height'],v['width'],field.world_units_per_metre)>CUTOFF
            item=dict(query=qi,text=text,frame=v['frame'],positive=int(positive.sum()),negative=int(negative.sum()),arms={})
            for arm in ARMS:
                lost=positive&pred['finite']&~pred[arm]
                item['arms'][arm]=dict(positive_covered=int((positive&pred[arm]).sum()),negative_covered=int((negative&pred[arm]).sum()),
                    positive_coverage_lost=int(lost.sum()),sampled_positive_lost=int((lost&v['sampled']).sum()),
                    unsampled_positive_lost=int((lost&~v['sampled']).sum()))
            records.append(item)
        print('source_diagnosed',job['scene'],qi,flush=True)
    total={a:{k:sum(r['arms'][a][k] for r in records) for k in ['positive_covered','negative_covered','positive_coverage_lost','sampled_positive_lost','unsampled_positive_lost']} for a in ARMS}
    report=dict(scene=job['scene'],query_view_pairs=len(records),positive=sum(x['positive'] for x in records),negative=sum(x['negative'] for x in records),totals=total,records=records,
        diagnostic_source_access=True,inference_inputs_changed=False,target_labels_used=False,
        source_masks_are_partial_pseudo_labels=True,scores_or_thresholds_tuned=False)
    write(ROOT/'diagnostics'/('source_'+job['scene']+'.json'),report);print(total,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
