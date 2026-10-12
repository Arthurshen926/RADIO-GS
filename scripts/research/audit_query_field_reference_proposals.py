"""Independent sparse candidate-domain reconstruction and physical set audit."""
import argparse
import numpy as np
from scipy import sparse
import torch
from radio_gs.query_field.storage import load_field
from scripts.research.run_query_field_reference_proposals import ROOT,PREVIOUS,ARMS,design
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes


def audit(index):
    d=design()
    for r in read(ROOT/'audit_design.json')['code']:check(r)
    j=d['jobs'][index];name=j['name'];cfg=read(j['configuration']['path'])
    field,_=load_field(cfg['jobs'][0]['field'],'cpu');n=len(field.xyz)
    ev=field.evidence;matrix=sparse.csr_matrix((np.ones(len(ev.positive_index)),ev.positive_index.numpy(),ev.positive_pointer.numpy()),shape=(n,len(ev.views)))
    old=read(PREVIOUS/'predictions'/name/(name+'__edit')/'prediction.json')
    chosen=old['diagnostics']['selected_regions']
    candidate=(np.asarray(matrix[:,chosen].sum(axis=1)).ravel()>0)&field.member_available.numpy()
    records={}
    for arm,job in zip(ARMS,cfg['jobs']):
        r=read(ROOT/'predictions'/name/job['name']/'prediction.json');check(r['extent_state'])
        assert r['diagnostics']['selected_regions']==chosen
        state=torch.load(r['extent_state']['path'],weights_only=True);initial=state['initial'].numpy()>.5
        final=state['selected'].numpy();available=state['domain'].numpy();assert not (initial&~candidate).any()
        if arm=='free':
            assert 'proposal_domain' not in state
            records[arm]=dict(added_outside_candidates=int((final&~initial&~candidate).sum()),
                added_inside_candidates=int((final&~initial&candidate).sum()))
            continue
        lift=state['lift'];valid=lift['valid'].numpy();target=state['scope']['target'].numpy()
        op=sparse.coo_matrix((lift['weight'].double().numpy(),lift['index'].numpy()),shape=(len(target),n)).tocsc()
        mass=np.asarray(op.T@valid.astype(float)).ravel();positive=np.asarray(op.T@(valid*target).astype(float)).ravel()
        strata=np.zeros(n,np.int64);known=mass>1e-8;strata[known]=1
        strata[known&(positive>.5*mass)]=2;strata[known&(positive<.5*mass)]=3
        assert np.array_equal(strata,state['proposal_strata'].numpy()),(name,arm,'strata')
        assert np.array_equal(candidate,state['candidate_domain'].numpy())
        allowed=initial.copy() if arm=='prioronly' else candidate.copy()
        if arm=='matched':
            rng=np.random.default_rng(d['proposal_shuffle_seed'])
            for group in np.unique(strata):
                ids=np.flatnonzero(available&~initial&(strata==group));allowed[ids]=candidate[rng.permutation(ids)]
        assert np.array_equal(allowed,state['proposal_domain'].numpy())
        assert not final[~allowed].any() and allowed[initial].all()
        assert state['proposal_policy']==d['proposal_policies'][arm]
        info=r['diagnostics']['extent_proposals'];assert info['allowed_new_members']==int((allowed&~initial).sum())
        assert not info['initial_members_locked'] and not info['outside_support_is_negative']
        records[arm]=dict(allowed=int(allowed.sum()),allowed_additions=int((allowed&~initial).sum()),
            added=int((final&~initial).sum()),removed=int((initial&~final).sum()),
            added_outside_candidates=int((final&~initial&~candidate).sum()),
            strata_addition_counts={str(int(s)):int((allowed&~initial&(strata==s)).sum()) for s in np.unique(strata)})
    assert records['scope']['strata_addition_counts']==records['matched']['strata_addition_counts']
    write(ROOT/'proposal_audits'/(name+'.json'),dict(passed=True,case=name,conditions=records,code=receipt(__file__)))
    del field,ev,matrix,state,op
    import scripts.research.audit_query_field_reference_edit as physical
    from scripts.research.run_query_field_reference_extent import ROOT as original
    physical.ROOT=ROOT;physical.PREVIOUS=original;physical.ARMS=ARMS;physical.design=design
    physical.audit(index)


def run(gpus):
    d=design()
    for r in read(ROOT/'audit_design.json')['code']:check(r)
    for r in read(ROOT/'predictions_complete.json')['records']+read(ROOT/'scoring_complete.json')['records']:check(r)
    lanes(list(range(8)),gpus,lambda i,g:execute(['-m','scripts.research.audit_query_field_reference_proposals','case','--index',str(i)],ROOT/'logs'/('audit_'+d['jobs'][i]['name']+'.log'),g))
    cases={j['name']:read(ROOT/'audits'/(j['name']+'.json')) for j in d['jobs']}
    proposals={j['name']:read(ROOT/'proposal_audits'/(j['name']+'.json')) for j in d['jobs']}
    assert all(r['passed'] for r in cases.values()) and all(r['passed'] for r in proposals.values())
    result=dict(passed=True,cases=cases,proposals=proposals,predictions=32,endpoints=64,
        audit_design=receipt(ROOT/'audit_design.json'),code=receipt(__file__),design=receipt(ROOT/'design.json'),
        previous_float64_footprint_failure_not_repaired=True,default_promoted=False,whole_five_task=False)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-reference-proposals-audit.json',result)
    print('PASS all domains, 32 metrics and 64 physical endpoints',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['case','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='2,3,4')
    a=p.parse_args();torch.set_num_threads(4)
    audit(a.index) if a.stage=='case' else run(list(map(int,a.gpus.split(','))))
