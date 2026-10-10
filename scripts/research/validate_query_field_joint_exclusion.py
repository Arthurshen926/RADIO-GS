"""Independent construction conditions, no benchmark query or score inputs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder,CUTOFF
from scripts.research.compile_query_field_coverage_factors import ROOT as FIELDS
ROOT=Path('output/query_field_20261010/joint_exclusion_v1')
SCENES=['figurines','ramen','teatime','waldo_kitchen']


def run(index):
    scene=SCENES[index];field,manifest=load_field(FIELDS/scene/'field','cuda');decoder=JointExclusionDecoder(field)
    rng=np.random.default_rng(20261010);reports=[];n=len(field.xyz);views=field.scope_permissions.accepted_views.cpu().numpy()
    for i in range(16):
        if i==0:accepted=np.zeros(len(views),bool)
        elif i==1:accepted=np.ones(len(views),bool)
        elif i==2:
            accepted=np.zeros(len(views),bool)
            for v in np.unique(views):accepted[rng.choice(np.flatnonzero(views==v))]=True
        else:accepted=rng.random(len(views))<[.005,.02,.1][i%3]
        accepted=torch.from_numpy(accepted).cuda();initial=torch.from_numpy(rng.random(n).astype(np.float32)).cuda()
        p,neg=decoder.labels(accepted);p=torch.from_numpy(p).cuda();neg=torch.from_numpy(neg).cuda()
        def opacity(member):return torch.cat([v((member>.5).float()) for v in field.coverage_factors.views])
        before=opacity(initial)>CUTOFF;arms={}
        for joint,name in [(False,'frozen_visibility'),(True,'joint_coverage')]:
            member,detail=decoder.decode(initial,accepted,joint=joint);after=opacity(member)>CUTOFF
            assert not bool((p&before&~after).any()) and not bool((neg&~before&after).any())
            assert not bool(((member>.5)&(initial<=.5)).any())
            assert torch.equal(member[member!=0],initial[member!=0])
            if i==0:assert torch.equal(member,initial)
            arms[name]=detail
        reports.append(dict(condition=i,arms=arms));print(scene,i,arms,flush=True)
    out=ROOT/'validation';out.mkdir(parents=True,exist_ok=True)
    report=dict(passed=True,scene=scene,conditions=16,arms=2,all_factors_checked=True,
        target_labels_used=False,benchmark_queries_used=False,field_sha256=manifest['files']['field.pt']['sha256'],records=reports)
    (out/(scene+'.json')).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True,type=int);a=p.parse_args();torch.set_num_threads(4);run(a.index)
