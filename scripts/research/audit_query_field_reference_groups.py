"""Independent tuple-based partition reconstruction plus physical/metric audit."""
import argparse
import numpy as np
import torch
from radio_gs.query_field.storage import load_field
from scripts.research.run_query_field_reference_groups import ROOT,PREVIOUS,ARMS,design
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes


def independent_partition(pointer,index,domain,initial):
    groups=np.full(len(domain),-1,np.int64);seen={};next_group=0
    for member in np.flatnonzero(domain):
        start,stop=pointer[member:member+2]
        if start==stop:groups[member]=next_group;next_group+=1;continue
        key=(bool(initial[member]),tuple(int(i) for i in index[start:stop]))
        if key not in seen:seen[key]=next_group;next_group+=1
        groups[member]=seen[key]
    return groups


def audit(index):
    for r in read(ROOT/'audit_design.json')['code']:check(r)
    d=design();j=d['jobs'][index];cfg=read(j['configuration']['path']);name=j['name']
    field,_=load_field(cfg['jobs'][0]['field'],'cpu');domain=field.member_available.numpy()
    pointer=field.evidence.positive_pointer.numpy();indices=field.evidence.positive_index.numpy()
    states={};records={}
    for arm,job in zip(ARMS,cfg['jobs']):
        record=read(ROOT/'predictions'/name/job['name']/'prediction.json');check(record['extent_state'])
        states[arm]=torch.load(record['extent_state']['path'],weights_only=True);records[arm]=record
    initial=states['free']['initial'].numpy()>.5
    expected=independent_partition(pointer,indices,domain,initial)
    assert np.array_equal(expected,states['parts']['groups'].numpy()),name
    sizes=np.bincount(expected[domain]);covered=np.diff(pointer)>0
    shuffled=expected.copy();rng=np.random.default_rng(d['group_shuffle_seed'])
    for bit in [False,True]:
        ids=np.flatnonzero(domain&covered&(initial==bit));shuffled[ids]=expected[rng.permutation(ids)]
    assert np.array_equal(shuffled,states['shuffled']['groups'].numpy()),name
    assert np.array_equal(sizes,np.bincount(shuffled[domain]))
    info={}
    for arm,groups in [('parts',expected),('shuffled',shuffled)]:
        state=states[arm];final=state['selected'].numpy();n=len(sizes)
        initial_mass=np.bincount(groups[domain],weights=initial[domain],minlength=n)
        final_mass=np.bincount(groups[domain],weights=final[domain],minlength=n)
        assert np.all((initial_mass==0)|(initial_mass==sizes)) and np.all((final_mass==0)|(final_mass==sizes))
        assert np.all(sizes[groups[domain&~covered]]==1)
        for k in ['prior','initial','cost','reference','domain']:assert torch.equal(state[k],states['free'][k]),(name,arm,k)
        report=records[arm]['diagnostics']['extent_grouping']
        assert report['groups']==n and report['max_group_size']==int(sizes.max())
        info[arm]=dict(groups=n,grouped_members=int(sizes[sizes>1].sum()),singleton_groups=int((sizes==1).sum()),
            max_group_size=int(sizes.max()),added_groups=int(((initial_mass==0)&(final_mass>0)).sum()),
            removed_groups=int(((initial_mass>0)&(final_mass==0)).sum()),added_members=int((~initial&final).sum()),
            removed_members=int((initial&~final).sum()),total_modified_members=int((initial!=final).sum()),
            unknown_singletons=int((domain&~covered).sum()),all_groups_finally_homogeneous=True)
    write(ROOT/'partition_audits'/(name+'.json'),dict(passed=True,case=name,conditions=info,
        every_group_size_matched=True,initial_unchanged=True,tuple_reconstruction_exact=True,
        identity_truth_used=False,spatial_locality_controlled=False,code=receipt(__file__)))
    del field,states
    # Reuse the already-independent raw Gaussian/renderer/metric audit, with the
    # current study's declared arm names and constant weight bound explicitly.
    import scripts.research.audit_query_field_reference_coupled as physical
    from scripts.research.run_query_field_reference_extent import ROOT as original_root
    physical.ROOT=ROOT;physical.PREVIOUS=original_root;physical.ARMS=ARMS
    def bound_design():
        current=design();assert current['observation_weight']==.5
        return dict(current,observation_weight={arm:current['observation_weight'] for arm in ARMS})
    physical.design=bound_design
    physical.audit(index)


def run(gpus):
    d=design()
    for r in read(ROOT/'audit_design.json')['code']:check(r)
    for r in read(ROOT/'predictions_complete.json')['records']+read(ROOT/'scoring_complete.json')['records']:check(r)
    lanes(list(range(8)),gpus,lambda i,g:execute(['-m','scripts.research.audit_query_field_reference_groups','case','--index',str(i)],ROOT/'logs'/('audit_'+d['jobs'][i]['name']+'.log'),g))
    cases={j['name']:read(ROOT/'audits'/(j['name']+'.json')) for j in d['jobs']}
    partitions={j['name']:read(ROOT/'partition_audits'/(j['name']+'.json')) for j in d['jobs']}
    assert all(c['passed'] for c in cases.values()) and all(c['passed'] for c in partitions.values())
    result=dict(passed=True,cases=cases,partitions=partitions,predictions=24,endpoints=48,
        design=receipt(ROOT/'design.json'),summary=receipt(ROOT/'summary.json'),code=receipt(__file__),
        audit_design=receipt(ROOT/'audit_design.json'),
        physical_auditor=receipt('scripts/research/audit_query_field_reference_coupled.py'),
        previous_float64_footprint_failure_not_repaired=True,default_promoted=False,whole_five_task=False)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-reference-groups-audit.json',result)
    print('PASS all partitions, 24 target metrics and 48 physical endpoints',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['case','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,3,4')
    a=p.parse_args();torch.set_num_threads(4)
    audit(a.index) if a.stage=='case' else run(list(map(int,a.gpus.split(','))))
