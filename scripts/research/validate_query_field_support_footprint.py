"""Cold self-contained field reload and exact parent tensor preservation."""
import argparse
import builtins
import json
from unittest.mock import patch
import torch
from radio_gs.query_field.storage import load_field
from scripts.research.compile_query_field_support_footprint import ROOT, PARENT, CONFIG


def run(index):
    job=json.loads(CONFIG.read_text())['jobs'][index];out=ROOT/job['scene']
    field,_=load_field(out/'field')
    parent,_=load_field(PARENT/job['scene']/'field')
    current=field.state_dict();old=parent.state_dict()
    for key,value in old.items():
        assert torch.equal(current[key],value),key
    assert set(current)-set(old)=={'support_footprint.cost'}
    cost=field.support_footprint.cost
    def deny(*args,**kwargs):raise AssertionError('unexpected source file access')
    with patch.object(builtins,'open',deny):
        empty=torch.zeros(len(cost),dtype=torch.bool)
        assert torch.equal(field.support_footprint(empty),torch.zeros(cost.shape[1]))
        for vi in range(len(cost)):
            active=empty.clone();active[vi]=True
            assert torch.equal(field.support_footprint(active),cost[vi])
        assert torch.equal(field.support_footprint(~empty),cost.sum(0))
    report=dict(complete=True,scene=job['scene'],parent_tensors_exact=True,
        views=len(cost),members=cost.shape[1],function_conditions=len(cost)+2,
        file_reads_denied_during_footprint=True,benchmark_evaluated=False)
    (out/'cold_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4);run(a.index)
