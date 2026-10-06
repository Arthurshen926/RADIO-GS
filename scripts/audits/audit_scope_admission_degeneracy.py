"""Audit sealed validation policies without fitting or changing a checkpoint.

For historical reports without per-query decisions, infer exact decisions only
when the recorded count proves all-allow or all-reject. Never infer selectivity.
"""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.training.extent_admission_diagnostics import admission_diagnostics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    results=[]
    for path in a.reports:
        r=json.loads(path.read_text());held=set(r['heldout_phrases'])
        grouped={}
        for o in r['observations']:
            if o['phrase'] in held:grouped.setdefault(o['query_index'],[]).append(o)
        total=len(grouped);allowed=r['selected']['validation_allowed']
        item=dict(path=str(path),sha256=sha256_file(path),validation_queries=total,allowed=allowed)
        if allowed not in (0,total):
            item['diagnostics_unavailable']='Per-query decisions absent; cannot recover selective policy from count'
        else:
            for kind in ['all','routed']:
                rows=[v for v in grouped.values() if kind=='all' or v[0]['kind']=='routed']
                native=torch.tensor([sum(o['native_iou'] for o in v)/len(v) for v in rows])
                extent=torch.tensor([sum(o['extent_iou'] for o in v)/len(v) for v in rows])
                item[kind]=admission_diagnostics(native,extent,torch.full((len(rows),),allowed==total,dtype=torch.bool))
        results.append(item)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(schema='admission_degeneracy_audit.v1',checkpoint_changed=False,independent_validation=False,reports=results,implementation_sha256=sha256_file(__file__)),indent=2)+'\n')

if __name__=='__main__':main()
