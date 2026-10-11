"""Post-hoc source/GT-free diagnosis of reference observation vs set coverage.

Reads sealed current-query factors only, never changes any prediction or selects
a model. A_obs normalizes available-domain mass, not the legacy observed-only
target response; both are reported under their own contracts.
"""
import numpy as np
from scipy import sparse
import torch
from scripts.research.run_query_field_reference_extent import ROOT,ARMS,design
from scripts.research.run_query_field_relations import read,write,receipt,check


def agreement(response,target,valid):
    selected=(response>.5)&valid;intersection=float(target[selected].sum())
    return dict(soft_target_binary_iou=intersection/max(float(selected.sum()+target[valid].sum()-intersection),1e-12),
        mean_foreground_response=float((response[valid]*target[valid]).sum()/target[valid].sum()),
        mean_background_response=float((response[valid]*(1-target[valid])).sum()/(1-target[valid]).sum()))


def main():
    d=design();rows={};records=[]
    for job in d['jobs']:
        name=job['name'];rows[name]={}
        for arm in ARMS[1:]:
            path=ROOT/'predictions'/name/(name+'__'+arm)/'prediction.json';r=read(path)
            check(r['extent_state']);state=torch.load(r['extent_state']['path'],weights_only=True)
            records.append(receipt(path));lift=state['lift'];scope=state['scope']
            op=sparse.coo_matrix((lift['weight'].double().numpy(),lift['index'].numpy()),
                shape=(len(lift['valid']),len(state['prior']))).tocsr()
            prior=state['prior'].numpy()>.5;final=state['selected'].numpy()
            y=scope['target'].double().numpy();valid=lift['valid'].numpy()
            removed=prior&~final;added=final&~prior
            removed_response=op@removed;added_response=op@added
            known=np.asarray(op.T@valid.astype(float)).ravel()
            rows[name][arm]=dict(prior=agreement(op@prior,y,valid),final=agreement(op@final,y,valid),
                prior_members=int(prior.sum()),final_members=int(final.sum()),
                removed_members=int(removed.sum()),added_members=int(added.sum()),
                removed_with_no_observation=int((removed&(known<=1e-8)).sum()),
                added_with_no_observation=int((added&(known<=1e-8)).sum()),
                removed_foreground_observation=float((removed_response*y*valid).sum()),
                added_foreground_observation=float((added_response*y*valid).sum()))
        print('observation diagnosis',name,flush=True)
    result=dict(post_hoc=True,predictions_changed=False,target_labels_used=False,
        historical_sources_read=False,operator='full-scene available-domain normalized A_obs',
        legacy_observed_only_response_equivalence_claimed=False,cases=rows,records=records,code=receipt(__file__))
    write(ROOT/'observation_diagnosis.json',result)
    write('docs/experiments/2026-10-12-reference-observation-diagnosis.json',result)


if __name__=='__main__':torch.set_num_threads(4);main()
