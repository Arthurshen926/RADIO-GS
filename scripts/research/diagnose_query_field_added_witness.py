"""Post-prediction visibility witness audit; never changes a selected set."""
import argparse
import torch
from scripts.research.run_query_field_member_energy import ROOT,read,write
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.part_conditioning import native_region_probability


def run(index):
    j=read(ROOT/'prediction_config.json')['jobs'][index]
    field,_=load_field(j['field'])
    data=torch.load(ROOT/'predictions'/j['name']/'predictions.pt',weights_only=True)
    positive=field.evidence.matrix('positive')
    available=field.observed|field.semantic_measure.supported
    mass=torch.sparse.mm(positive.transpose(0,1),available.float()[:,None]).flatten()
    rows=[]
    for qi,text in enumerate(data['texts']):
        probability=native_region_probability(field.evidence,data['native_query_vectors'][qi],data['generic_negative_vectors'])
        accepted=(probability>.5)&(mass>0)
        witness=torch.sparse.mm(positive,accepted.float()[:,None]).flatten()>0
        additions=data['selected_by_control']['member_energy'][:,qi]&~data['selected_by_control']['semantic_initial'][:,qi]
        rows.append(dict(text=text,added=int(additions.sum()),without_positive_witness=int((additions&~witness).sum())))
    report=dict(scene=j['scene'],rows=rows,total_added=sum(x['added'] for x in rows),
        total_without_positive_witness=sum(x['without_positive_witness'] for x in rows),
        post_hoc=True,acceptance_recomputed_on_cpu=True,inference_inputs_changed=False)
    write(ROOT/'diagnostics'/('witness_'+j['scene']+'.json'),report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
