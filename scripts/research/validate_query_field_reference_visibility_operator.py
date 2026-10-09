"""Compare contribution-space reference observation to retained color rendering."""
from types import SimpleNamespace
import torch
from scripts.research.run_query_field_reference_visibility import ROOT,design,read,write,receipt,check
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.geometry import projection
from radio_gs.query_field.public_nvos import retained_response
from radio_gs.query_field.reference_scope import visible_region_scores

@torch.no_grad()
def main():
    torch.set_num_threads(4);d=design();j=d['jobs'][0]
    for r in [j['configuration'],j['field'],j['manifest']]+j['inputs']:check(r)
    cfg=read(j['configuration']['path']);job=cfg['jobs'][0]
    field,_=load_field(job['field'],'cuda');fixture=torch.load(job['camera_fixture'],weights_only=True)
    camera=SimpleNamespace(**fixture['reference']);world=fixture['world_units_per_metre'];h,w=cfg['projection_hw']
    geom=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
    _,op,_,_=projection(geom,camera,h,w,world)
    region=field.evidence.matrix('positive').transpose(0,1).to_sparse_csr();ids=region.col_indices()[:int(region.crow_indices()[1])]
    first=torch.zeros(len(field.xyz),device='cuda');first[ids]=1;first*=field.member_available
    inputs=dict(zero=torch.zeros_like(first),observed=field.observed.float(),periodic=(torch.arange(len(first),device='cuda')%3==0).float()*field.observed,first_region=first)
    records={}
    for name,member in inputs.items():
        a=visible_region_scores(op,field.observed,member[:,None]).reshape(h,w)
        b=retained_response(field,member,camera,h,w,world)
        error=float((a-b).abs().max());differences=int(((a>.5)!=(b>.5)).sum())
        records[name]=dict(maximum_error=error,threshold_disagreements=differences,pixels=h*w)
        if error>1e-5:raise ValueError(records)
    report=dict(passed=True,case=j['name'],patterns=records,scope='One field, four fixed non-GT membership patterns; numerical operator parity, not all-scene equivalence.',code=receipt(__file__))
    write(ROOT/'validation/visibility_operator.json',report);write('docs/experiments/2026-10-09-reference-visibility-operator.json',report);print(report,flush=True)

if __name__=='__main__':main()
