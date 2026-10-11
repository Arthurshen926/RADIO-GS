"""Query-free migration of the source reference analytic footprint into one field."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.query_field.storage import load_field, save_field, sha256
from radio_gs.query_field.support_footprint import SupportFootprint, compile_support_footprint
from scripts.research.conditional_subset_reference import unoccluded_footprint
from scripts.research.compile_query_field_scope_permissions import CONFIG

PARENT=Path('output/query_field_20261011/compiled_available_coverage_v1')
ROOT=Path('output/query_field_20261011/compiled_support_footprint_v1')


def run(index):
    job=json.loads(CONFIG.read_text())['jobs'][index]
    out=ROOT/job['scene'];out.mkdir(parents=True,exist_ok=False)
    field,parent=load_field(PARENT/job['scene']/'field','cuda')
    source=Path(job['observations'])
    manifest=json.loads((source/'observations.json').read_text())
    construction=json.loads(Path(job['construction']).read_text())
    assert len(construction['source_records'])==len(field.scope_constraints.views)
    domain=field.observed|field.semantic_measure.supported
    costs=[];checks=[]
    for r in construction['source_records']:
        frame=r['frame'];assert frame in manifest['splits']['train_frames']
        assert frame not in manifest['splits']['development_frames']
        record=next(x for x in manifest['frames'] if x['frame']==frame)
        path=source/f'view_{frame:05d}.pt';digest=sha256(path)
        assert digest==record['compiled']['sha256']
        raw=torch.load(path,map_location='cpu',weights_only=True)
        h,w=raw['masks'].shape[-2:]
        camera=SimpleNamespace(intrinsic=raw['intrinsic'],world_to_camera=raw['world_to_camera'],
            height=raw['original_hw'][0],width=raw['original_hw'][1])
        legacy=unoccluded_footprint(field,camera,h,w,field.world_units_per_metre)
        observed=compile_support_footprint(field,camera,h,w)
        assert torch.equal(legacy,observed)
        cost=compile_support_footprint(field,camera,h,w,member_domain=domain)
        assert torch.equal(cost[field.observed],legacy[field.observed])
        assert not bool(cost[~domain].any())
        costs.append(cost.cpu())
        checks.append(dict(frame=frame,source_sha256=digest,observed_reference_exact=True,
            semantic_only_nonzero=int(((cost>0)&~field.observed).sum())))
        print('compiled',job['scene'],frame,flush=True)
    field.support_footprint=SupportFootprint({'cost':torch.stack(costs).to(field.xyz.device)},len(field.xyz))
    exported=save_field(field,out/'field',dict(parent_field_sha256=parent['files']['field.pt']['sha256'],
        records=checks,benchmark_queries_used=False,target_labels_used=False,default_decoder_changed=False,
        measure='analytic infinite-plane projected support, not clipped full-image coverage'))
    added=sum(x.numel()*x.element_size() for x in costs)
    assert exported['tensor_bytes']==parent['tensor_bytes']+added
    (out/'compilation.json').write_text(json.dumps(dict(complete=True,scene=job['scene'],
        records=checks,additional_tensor_bytes=added,field_sha256=exported['files']['field.pt']['sha256'],
        cold_reload_validated=False,benchmark_evaluated=False),indent=2)+'\n')
    print('complete',job['scene'],added,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60)
    run(a.index)
