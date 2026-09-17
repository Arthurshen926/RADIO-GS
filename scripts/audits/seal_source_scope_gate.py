"""Seal source scope supervision controls, failed coverage and real-query results."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4);base=Path('/mnt/pool/sqy/results/RADIO-GS/output')
    archive=base/'source_scope_gate_sources_20260916';archive.mkdir(exist_ok=True)
    sources=['radio_gs/models/grounded_semantic_transfer.py','radio_gs/models/element_retrieval_field.py','tests/test_source_scope_gate.py']
    sources+=['scripts/audits/'+s+'.py' for s in ['fit_source_scope_gate','trace_source_scope_gate','evaluate_frozen_field_lerf_labels','evaluate_real_reference_queries','seal_source_scope_gate']]
    result=dict(schema='source_scope_gate_seal.v1',implementations={},artifacts={},scenes={})
    for name in sources:
        p=Path(name);digest=sha256_file(p);snapshot=archive/(digest+'.py')
        if not snapshot.exists():snapshot.write_bytes(p.read_bytes())
        assert sha256_file(snapshot)==digest;result['implementations'][name]=digest
    for scene in ['figurines','ramen']:
        root=base/'preserved_source_inventory_20260916'/scene;parent=torch.load(root/'native_overlap_fallback96/field.pt',weights_only=False)
        fields={}
        for name in ['source_scope_gate96','hard_source_scope_gate96']+(['selected_source_scope_gate96'] if scene=='figurines' else []):
            p=root/name/'field.pt';r=json.loads((p.parent/'report.json').read_text())
            assert sha256_file(p)==r['field_sha256'] and sha256_file(root/'native_overlap_fallback96/field.pt')==r['parent_field_sha256']
            assert r['source_only'] and not r['benchmark_data_used']
            for k in ['implementation_sha256','readout_implementation_sha256']:assert sha256_file(archive/(r[k]+'.py'))==r[k]
            payload=torch.load(p,weights_only=False)
            assert payload['geometry_binding']==parent['geometry_binding'] and payload['query_encoder']==parent['query_encoder']
            for k,v in parent['state_dict'].items():assert torch.equal(v,payload['state_dict'][k]),k
            fields[name]=r['field_sha256'];del payload
            result['artifacts'][str(p.parent/'report.json')]=sha256_file(p.parent/'report.json')
        del parent
        metrics={}
        for tag in ['source_scope_gate_text','hard_source_scope_gate_text','source_scope_gate_reference']:
            p=Path(f'paper/artifacts/{scene}_{tag}_20260916.json');d=json.loads(p.read_text())
            assert sha256_file(p.with_suffix('.protocol.json'))==d['protocol_sha256']
            for name,run in d['runs'].items():
                assert sha256_file(root/name/'field.pt')==run['field_sha256']
                if 'observation_miou' in run:metrics[name]={k:run[k] for k in ['observation_miou','average_precision_macro']}
            if tag=='source_scope_gate_text':
                old=json.loads(Path(f'paper/artifacts/{scene}_native_overlap_fallback_text_20260916.json').read_text())['runs']['native_overlap_fallback96']
                assert old['observations']==d['runs']['native_overlap_fallback96']['observations']
            if tag.endswith('reference'):
                old=json.loads(Path(f'paper/artifacts/{scene}_native_overlap_fallback_reference_20260916.json').read_text())['runs']['native_overlap_fallback96']
                for run in d['runs'].values():
                    assert len(old['observations'])==len(run['observations'])
                    for a,b in zip(old['observations'],run['observations']):
                        for k in ['reference','target_frame','category','image','prompt']:assert a[k]==b[k]
            result['artifacts'][str(p)]=sha256_file(p);result['artifacts'][str(p.with_suffix('.protocol.json'))]=sha256_file(p.with_suffix('.protocol.json'))
        p=Path(f'paper/artifacts/{scene}_hard_source_scope_gate96_trace_20260916.json');result['artifacts'][str(p)]=sha256_file(p)
        result['scenes'][scene]=dict(fields=fields,text=metrics,parent_tensors_exact=True,image_prompt_observations_exact=True,parent_text_reproduced_exactly=True)
    p=Path('paper/artifacts/ramen_source_scope_gate_insufficient_20260916.json');result['artifacts'][str(p)]=sha256_file(p)
    out=Path('paper/artifacts/source_scope_gate_seal_20260916.json')
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');print(out)


if __name__=='__main__':main()
