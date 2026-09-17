"""Seal source-supervised selection controls and historical scoring alignment."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4);base=Path('/mnt/pool/sqy/results/RADIO-GS/output')
    archive=base/'cross_view_ranker_sources_20260916';archive.mkdir(exist_ok=True)
    sources=['radio_gs/models/grounded_semantic_transfer.py','radio_gs/models/element_retrieval_field.py',
        'tests/test_grounded_semantic_transfer.py','tests/test_cross_view_region_ranker.py']
    sources+=['scripts/audits/'+s+'.py' for s in ['fit_cross_view_region_ranker','export_hierarchical_extent_candidates',
        'compare_retained_predictions_current_score','evaluate_frozen_field_lerf_labels','evaluate_real_reference_queries','seal_cross_view_selection']]
    implementation={}
    for name in sources:
        path=Path(name);digest=sha256_file(path);snapshot=archive/(digest+'.py')
        if not snapshot.exists():snapshot.write_bytes(path.read_bytes())
        assert sha256_file(snapshot)==digest;implementation[name]=digest
    result=dict(schema='cross_view_selection_seal.v1',implementations=implementation,artifacts={},scenes={},
        decision='retain source_extent_rendered_residual96; cross-view selector controls not jointly promoted')
    for scene in ['figurines','ramen']:
        root=base/'preserved_source_inventory_20260916'/scene
        parents={'source_cross_view_ranker96':'source_region_extent96','source_hierarchical_candidates96':'source_region_extent96',
                 'source_hierarchical_ranker96':'source_hierarchical_candidates96'}
        fields={}
        for name,parent in parents.items():
            path=root/name/'field.pt';report=json.loads((path.parent/'report.json').read_text())
            assert sha256_file(path)==report['field_sha256'] and sha256_file(root/parent/'field.pt')==report['parent_field_sha256']
            assert report['source_only'] and not report['benchmark_data_used']
            for key in ['implementation_sha256','readout_implementation_sha256']:
                snapshot=archive/(report[key]+'.py');assert snapshot.exists() and sha256_file(snapshot)==report[key]
            new=torch.load(path,weights_only=False);old=torch.load(root/parent/'field.pt',weights_only=False)
            assert new['geometry_binding']==old['geometry_binding'] and new['query_encoder']==old['query_encoder']
            for k,v in old['state_dict'].items():
                if name=='source_hierarchical_candidates96' and k.startswith('grounded_transfer.'):continue
                assert torch.equal(v,new['state_dict'][k]),k
            fields[name]=report['field_sha256'];del new,old
            result['artifacts'][str(path.parent/'report.json')]=sha256_file(path.parent/'report.json')
        metrics={}
        for tag in ['cross_view_ranker_text','hierarchical_ranker_text','cross_view_selection_reference']:
            path=Path(f'paper/artifacts/{scene}_{tag}_20260916.json');data=json.loads(path.read_text())
            assert sha256_file(path.with_suffix('.protocol.json'))==data['protocol_sha256']
            assert sha256_file(archive/(data['protocol']['implementation_sha256']+'.py'))==data['protocol']['implementation_sha256']
            for name,run in data['runs'].items():
                assert run['field_sha256']==fields[name]
                if 'observation_miou' in run:metrics[name]={k:run[k] for k in ['observation_miou','average_precision_macro']}
            if tag.endswith('reference'):
                previous=json.loads(Path(f'paper/artifacts/{scene}_extent_repair_reference_20260916.json').read_text())['runs']['source_extent_rendered_residual96']
                for run in data['runs'].values():
                    assert len(run['observations'])==len(previous['observations'])
                    for a,b in zip(run['observations'],previous['observations']):
                        for k in ['reference','target_frame','category','image','prompt']:assert a[k]==b[k]
            result['artifacts'][str(path)]=sha256_file(path);result['artifacts'][str(path.with_suffix('.protocol.json'))]=sha256_file(path.with_suffix('.protocol.json'))
        result['scenes'][scene]=dict(fields=fields,text=metrics,all_parent_field_tensors_exact=True,image_prompt_observations_exact=True)
    historical=Path('paper/artifacts/historical_current_score_alignment_20260916.json');data=json.loads(historical.read_text())
    for scene,values in data['scenes'].items():
        assert sha256_file(f'paper/artifacts/{scene}_source_extent_residual_text_20260916.json')==values['current_result_sha256']
        for row in values['observations']:assert sha256_file(row['prediction'])==row['prediction_sha256']
    result['artifacts'][str(historical)]=sha256_file(historical)
    path=Path('paper/artifacts/cross_view_selection_seal_20260916.json')
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n');print(path)


if __name__=='__main__':main()
