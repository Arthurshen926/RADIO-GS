"""Seal compression localization and completion readout controls."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4);base=Path('/mnt/pool/sqy/results/RADIO-GS/output');archive=base/'rank_completion_sources_20260916';archive.mkdir(exist_ok=True)
    sources=['radio_gs/models/grounded_semantic_transfer.py','radio_gs/models/element_retrieval_field.py','tests/test_rank_preserving_completion.py']
    sources+=['scripts/audits/'+s+'.py' for s in ['diagnose_current_native_compression','fit_completion_admission','export_rank_preserving_completion','evaluate_frozen_field_lerf_labels','evaluate_real_reference_queries','seal_rank_completion']]
    result=dict(schema='rank_completion_seal.v1',implementations={},artifacts={},scenes={})
    for name in sources:
        p=Path(name);digest=sha256_file(p);snapshot=archive/(digest+'.py')
        if not snapshot.exists():snapshot.write_bytes(p.read_bytes())
        assert sha256_file(snapshot)==digest;result['implementations'][name]=digest
    for scene in ['figurines','ramen']:
        root=base/'preserved_source_inventory_20260916'/scene;parent=torch.load(root/'hard_source_scope_gate96/field.pt',weights_only=False)
        for name in ['source_completion_admission96','rank_preserving_completion96','native_within_extent96']:
            p=root/name/'field.pt';r=json.loads((p.parent/'report.json').read_text())
            assert sha256_file(p)==r['field_sha256'] and sha256_file(root/'hard_source_scope_gate96/field.pt')==r['parent_field_sha256']
            for key in ['implementation_sha256','readout_implementation_sha256']:assert sha256_file(archive/(r[key]+'.py'))==r[key]
            new=torch.load(p,weights_only=False)
            assert new['geometry_binding']==parent['geometry_binding'] and new['query_encoder']==parent['query_encoder']
            assert new['state_dict'].keys()==parent['state_dict'].keys()
            for key,value in parent['state_dict'].items():assert torch.equal(value,new['state_dict'][key]),key
            del new;result['artifacts'][str(p.parent/'report.json')]=sha256_file(p.parent/'report.json')
        del parent
        metrics={}
        for tag in ['rank_preserving_completion_text','native_within_extent_text','rank_completion_reference']:
            p=Path(f'paper/artifacts/{scene}_{tag}_20260916.json');d=json.loads(p.read_text());assert sha256_file(p.with_suffix('.protocol.json'))==d['protocol_sha256']
            for name,run in d['runs'].items():
                assert sha256_file(root/name/'field.pt')==run['field_sha256']
                if 'observation_miou' in run:metrics[name]={k:run[k] for k in ['observation_miou','average_precision_macro']}
            if tag=='rank_preserving_completion_text':
                old=json.loads(Path(f'paper/artifacts/{scene}_hard_source_scope_gate_text_20260916.json').read_text())['runs']['hard_source_scope_gate96']
                assert old['observations']==d['runs']['hard_source_scope_gate96']['observations']
            if tag.endswith('reference'):
                old=json.loads(Path(f'paper/artifacts/{scene}_source_scope_gate_reference_20260916.json').read_text())['runs']['hard_source_scope_gate96']
                for run in d['runs'].values():
                    assert len(run['observations'])==len(old['observations'])
                    for a,b in zip(run['observations'],old['observations']):
                        for k in ['reference','target_frame','category','image','prompt']:assert a[k]==b[k]
            result['artifacts'][str(p)]=sha256_file(p);result['artifacts'][str(p.with_suffix('.protocol.json'))]=sha256_file(p.with_suffix('.protocol.json'))
        p=Path(f'paper/artifacts/{scene}_current_native_compression_20260916.json');d=json.loads(p.read_text())
        assert d['projection_max_error']<2e-5 and d['field_sha256']==sha256_file(root/'hard_source_scope_gate96/field.pt')
        assert d['episode_sha256']==sha256_file(root/'episodes.pt')
        result['artifacts'][str(p)]=sha256_file(p);result['scenes'][scene]=dict(text=metrics,parent_tensors_exact=True,image_prompt_exact=True,parent_text_reproduced=True,compression_projection_max_error=d['projection_max_error'])
    out=Path('paper/artifacts/rank_completion_seal_20260916.json')
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');print(out)


if __name__=='__main__':main()
