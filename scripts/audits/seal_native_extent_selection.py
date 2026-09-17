"""Seal identity/extent selection controls and candidate without hiding AP regressions."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4);base=Path('/mnt/pool/sqy/results/RADIO-GS/output')
    archive=base/'native_identity_sources_20260916';archive.mkdir(exist_ok=True)
    names=['radio_gs/models/grounded_semantic_transfer.py','radio_gs/models/element_retrieval_field.py','tests/test_grounded_semantic_transfer.py']
    names+=['scripts/audits/'+x+'.py' for x in ['export_native_identity_extent','export_native_support_overlap',
        'diagnose_native_extent_selection','evaluate_frozen_field_lerf_labels','evaluate_real_reference_queries','seal_native_extent_selection']]
    result=dict(schema='native_extent_selection_seal.v1',implementations={},artifacts={},scenes={},
        decision='native_overlap_fallback96 is an IoU-improving development candidate; retain previous baseline due figurines AP regression and incomplete benchmark coverage')
    for name in names:
        p=Path(name);digest=sha256_file(p);snapshot=archive/(digest+'.py')
        if not snapshot.exists():snapshot.write_bytes(p.read_bytes())
        assert sha256_file(snapshot)==digest;result['implementations'][name]=digest
    for scene in ['figurines','ramen']:
        root=base/'preserved_source_inventory_20260916'/scene
        parent=torch.load(root/'source_hierarchical_candidates96/field.pt',weights_only=False)
        fields={};metrics={}
        for name in ['native_identity_restricted96','native_identity_global96','native_overlap_restricted96','native_overlap_global96','native_overlap_fallback96']:
            p=root/name/'field.pt';r=json.loads((p.parent/'report.json').read_text())
            assert sha256_file(p)==r['field_sha256'] and sha256_file(root/'source_hierarchical_candidates96/field.pt')==r['parent_field_sha256']
            assert r['source_only'] and not r['benchmark_data_used']
            for k in ['implementation_sha256','readout_implementation_sha256']:
                assert sha256_file(archive/(r[k]+'.py'))==r[k]
            new=torch.load(p,weights_only=False)
            assert new['geometry_binding']==parent['geometry_binding'] and new['query_encoder']==parent['query_encoder']
            for k,v in parent['state_dict'].items():assert torch.equal(v,new['state_dict'][k]),k
            fields[name]=r['field_sha256'];del new
            result['artifacts'][str(p.parent/'report.json')]=sha256_file(p.parent/'report.json')
        del parent
        for tag in ['native_identity_extent_text','native_overlap_extent_text','native_overlap_fallback_text','native_overlap_fallback_reference']:
            p=Path(f'paper/artifacts/{scene}_{tag}_20260916.json');d=json.loads(p.read_text())
            assert sha256_file(p.with_suffix('.protocol.json'))==d['protocol_sha256']
            for name,run in d['runs'].items():
                assert sha256_file(root/name/'field.pt')==run['field_sha256']
                if 'observation_miou' in run:metrics[name]={k:run[k] for k in ['observation_miou','average_precision_macro']}
            if tag.endswith('reference'):
                baseline=d['runs']['source_extent_rendered_residual96'];candidate=d['runs']['native_overlap_fallback96']
                assert len(baseline['observations'])==len(candidate['observations'])
                for a,b in zip(baseline['observations'],candidate['observations']):
                    for k in ['reference','target_frame','category','image','prompt']:assert a[k]==b[k]
            result['artifacts'][str(p)]=sha256_file(p);result['artifacts'][str(p.with_suffix('.protocol.json'))]=sha256_file(p.with_suffix('.protocol.json'))
        # Reproducing the incumbent verifies old behavior after readout API changes.
        old=json.loads(Path(f'paper/artifacts/{scene}_source_extent_residual_text_20260916.json').read_text())['runs']['source_extent_rendered_residual96']
        new=json.loads(Path(f'paper/artifacts/{scene}_native_identity_extent_text_20260916.json').read_text())['runs']['source_extent_rendered_residual96']
        assert old['observations']==new['observations']
        trace=Path(f'paper/artifacts/{scene}_native_extent_selection_trace_20260916.json')
        result['artifacts'][str(trace)]=sha256_file(trace)
        result['scenes'][scene]=dict(fields=fields,text=metrics,parent_tensors_exact=True,image_prompt_observations_exact=True,incumbent_reproduced_exactly=True)
    p=Path('paper/artifacts/native_extent_selection_seal_20260916.json')
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(result,indent=2)+'\n');print(p)


if __name__=='__main__':main()
