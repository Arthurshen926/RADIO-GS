"""Seal bounded candidate-union controls and unchanged capability checks."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4);base=Path('/mnt/pool/sqy/results/RADIO-GS/output');archive=base/'native_union_sources_20260916';archive.mkdir(exist_ok=True)
    sources=['radio_gs/models/grounded_semantic_transfer.py','radio_gs/models/element_retrieval_field.py','tests/test_native_support_union.py']
    sources+=['scripts/audits/'+s+'.py' for s in ['export_native_support_union','trace_native_support_union','fit_exact_render_query_threshold','evaluate_frozen_field_lerf_labels','evaluate_real_reference_queries','seal_native_union']]
    sources+=['radio_gs/models/native_query_threshold.py']
    result=dict(schema='native_union_seal.v1',implementations={},artifacts={},scenes={})
    for name in sources:
        p=Path(name);digest=sha256_file(p);snapshot=archive/(digest+'.py')
        if not snapshot.exists():snapshot.write_bytes(p.read_bytes())
        assert sha256_file(snapshot)==digest;result['implementations'][name]=digest
    for scene in ['figurines','ramen']:
        root=base/'preserved_source_inventory_20260916'/scene;parent=torch.load(root/'native_within_extent96/field.pt',weights_only=False)
        for name in ['native_support_union96','native_precision_union96','exact_render_query_threshold96']:
            p=root/name/'field.pt';r=json.loads((p.parent/'report.json').read_text())
            assert sha256_file(p)==r['field_sha256'] and sha256_file(root/'native_within_extent96/field.pt')==r['parent_field_sha256']
            for key in ['implementation_sha256','readout_implementation_sha256']:assert sha256_file(archive/(r[key]+'.py'))==r[key]
            new=torch.load(p,weights_only=False)
            assert new['geometry_binding']==parent['geometry_binding'] and new['query_encoder']==parent['query_encoder']
            assert new['state_dict'].keys()==parent['state_dict'].keys()
            for key,value in parent['state_dict'].items():
                if name=='exact_render_query_threshold96' and key.startswith('native_query_thresholds.text.'):continue
                assert torch.equal(value,new['state_dict'][key]),key
            if name=='exact_render_query_threshold96':
                assert all(new>=old-1e-7 for new,old in zip(r['history'][-1]['source_teacher_iou'],r['source_initial_deployed_iou']))
            del new;result['artifacts'][str(p.parent/'report.json')]=sha256_file(p.parent/'report.json')
        del parent
        metrics={}
        for tag in ['native_support_union_text','native_precision_union_text','native_union_reference','exact_render_query_threshold_text','exact_render_query_threshold_reference']:
            p=Path(f'paper/artifacts/{scene}_{tag}_20260916.json');d=json.loads(p.read_text());assert sha256_file(p.with_suffix('.protocol.json'))==d['protocol_sha256']
            for name,run in d['runs'].items():
                assert sha256_file(root/name/'field.pt')==run['field_sha256']
                if 'observation_miou' in run:metrics[name]={k:run[k] for k in ['observation_miou','average_precision_macro']}
            if tag=='native_support_union_text':
                old=json.loads(Path(f'paper/artifacts/{scene}_native_within_extent_text_20260916.json').read_text())['runs']['native_within_extent96']
                assert old['observations']==d['runs']['native_within_extent96']['observations']
            if tag.endswith('reference'):
                old=json.loads(Path(f'paper/artifacts/{scene}_rank_completion_reference_20260916.json').read_text())['runs']['native_within_extent96']
                for run in d['runs'].values():
                    assert len(run['observations'])==len(old['observations'])
                    for a,b in zip(run['observations'],old['observations']):
                        for k in ['reference','target_frame','category','image','prompt']:assert a[k]==b[k]
            result['artifacts'][str(p)]=sha256_file(p);result['artifacts'][str(p.with_suffix('.protocol.json'))]=sha256_file(p.with_suffix('.protocol.json'))
        p=Path(f'paper/artifacts/{scene}_native_support_union_trace_20260916.json');result['artifacts'][str(p)]=sha256_file(p)
        result['scenes'][scene]=dict(text=metrics,union_parent_tensors_exact=True,calibration_only_text_threshold_tensors_changed=True,image_prompt_exact=True,parent_text_reproduced=True)
    out=Path('paper/artifacts/native_union_seal_20260916.json')
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');print(out)


if __name__=='__main__':main()
