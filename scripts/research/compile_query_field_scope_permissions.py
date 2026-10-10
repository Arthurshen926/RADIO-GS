"""Compile query-free regional permission graphs into the existing scene field."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.query_field.storage import load_field,save_field,sha256
from radio_gs.query_field.scope_permissions import ScopePermissions,compile_view,concatenate
from radio_gs.query_field.scope_exclusion import union_exclusion

ROOT=Path('output/query_field_20261010/compiled_scope_permissions_v1')
CONFIG=Path('output/query_field_20261010/scope_exclusion_v1/prediction_config.json')


def run(index):
    job=json.loads(CONFIG.read_text())['jobs'][index];out=ROOT/job['scene'];out.mkdir(parents=True,exist_ok=False)
    field,parent=load_field(job['field']);source=Path(job['observations'])
    manifest=json.loads((source/'observations.json').read_text());construction=json.loads(Path(job['construction']).read_text())
    assert field.scope_permissions is None and manifest['world_units_per_metre']==field.world_units_per_metre
    records={v['frame']:v for v in manifest['frames']};states=[];checks=[];fixtures=[];offset=0
    generator=torch.Generator().manual_seed(20261010)
    for vi,r in enumerate(construction['source_records']):
        frame=r['frame'];assert frame in manifest['splits']['train_frames'] and frame not in manifest['splits']['development_frames']
        path=source/f'view_{frame:05d}.pt';digest=sha256(path);assert digest==records[frame]['compiled']['sha256']
        raw=torch.load(path,map_location='cpu',weights_only=True);ids=torch.tensor(r['regions'],dtype=torch.long)
        n=len(ids);assert torch.equal(field.evidence.views[offset:offset+n],torch.full((n,),vi))
        assert torch.equal(field.evidence.scales[offset:offset+n],raw['scales_m'][ids])
        masks=raw['masks'].bool();state=compile_view(masks,raw['scales_m'],ids,vi);graph=ScopePermissions(state)
        conditions=torch.cat([torch.zeros(1,n,dtype=torch.bool),torch.eye(n,dtype=torch.bool),torch.rand(64,n,generator=generator)<.15,torch.ones(1,n,dtype=torch.bool)])
        outputs=[];empty=torch.zeros_like(masks[0]);visible=torch.ones_like(empty)
        for accepted in conditions:
            full=torch.zeros(len(masks),dtype=torch.bool);full[ids]=accepted
            _,expected=union_exclusion(masks,raw['scales_m'],full,visible,empty)
            actual=graph(accepted);assert torch.equal(actual,expected),'scope permission changed'
            outputs.append(expected)
        fixtures.append(dict(accepted=conditions,expected=torch.stack(outputs)))
        checks.append(dict(frame=frame,sha256=digest,accepted_regions=n,candidate_regions=len(masks),
            tested_conditions=len(conditions),exact=True))
        states.append(state);offset+=n
        print('compiled',job['scene'],vi,checks[-1],flush=True)
    assert offset==len(field.evidence.views)
    field.scope_permissions=ScopePermissions(concatenate(states));added=sum(v.numel()*v.element_size() for v in field.scope_permissions.state_dict().values())
    exported=save_field(field,out/'field',dict(parent_field_sha256=parent['files']['field.pt']['sha256'],
        observations_sha256=sha256(source/'observations.json'),construction_sha256=sha256(job['construction']),source_records=checks,
        benchmark_queries_used=False,source_teacher_used=False,target_labels_used=False,training_steps=0,
        scope_output='candidate permissions only; not Gaussian negatives or final members',default_decoder_changed=False))
    assert exported['tensor_bytes']==parent['tensor_bytes']+added
    torch.save(fixtures,out/'validation_conditions.pt')
    report=dict(scene=job['scene'],complete=True,additional_tensor_bytes=added,source_views=len(checks),
        regions=offset,candidates=sum(x['candidate_regions'] for x in checks),tested_conditions=sum(x['tested_conditions'] for x in checks),
        records=checks,compiled_without_queries=True,not_member_function_parity=True,model_promotion=False,
        field_sha256=exported['files']['field.pt']['sha256'])
    (out/'compilation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
